"""Scripted pick-and-place policy for the bar.

Adapted from submodules/ogbench/ogbench/manipspace/oracles/markov/{markov_oracle,cube_markov}.py (ogbench a977253,
MIT; see src/ogbar/LICENSE_OGBENCH). Phase numbers 1-9 are OGBench's. Changes:

- Grasp point instead of bar centre. Every *pickup attempt* draws a grasp offset s along the bar (local x). The
  policy tracks the grasp point  g = bar_pos + s * bar_axis  where the cube oracle tracked the cube centre, and
  carries it to  target_pos + s * target_axis, so the bar's *centre* lands on the target.
- Pickup attempts. A new attempt (fresh s) starts at each sub-task, after a slip (bar lost from the closed gripper
  for `slip_patience` steps while carrying) and after a miss (gripper closed on nothing, or no two-pad grasp within
  `grasp_patience` steps).
  The gripper must reopen fully before a new grasp is allowed.
- Yaw symmetry n=2 (the bar looks the same after 180 deg, not 90 deg), and yaw targets are chosen so the effector
  never needs to leave [-pi, pi] (where ManipSpaceEnv clips it).
- Own RNG (no global np.random); OGBench's reset() bug that hard-codes max_step=200 is not copied.
"""
import numpy as np

PHASE_NAMES = {
    1: 'move above bar', 2: 'move to bar', 3: 'grasp', 4: 'lift', 5: 'move above target',
    6: 'move to target', 7: 'release', 8: 'move up', 9: 'move to final pose',
    10: 'topple standing bar',   # ogbar: a dropped bar can land on its heavy end and stay upright
    11: 'nudge bar free',        # ogbar: roll a bar lodged where it cannot be grasped (e.g. against the board)
    12: 'set down in front of slot',  # slot v2 (slide-in): on the floor at the staging pose (levels and lines it up)
    13: 'wait until level',      # slot v2: at groove height
    14: 'slide into slot',       # slot v2
    15: 'open to regrip',        # slot v2
    16: 'regrip at COM',         # slot v2
    17: 'lift to groove height', # slot v2
    18: 'slide out of slot',     # slot v2/v3: a seated bar is slid back out along +y before it is carried elsewhere
}
OUTCOMES = ('placed', 'slipped', 'missed', 'timeout')

ABOVE_OFFSET = 0.18        # OGBench block_above_offset z
ABOVE_THRESHOLD = 0.16     # OGBench above_threshold (absolute; ogbar measures retreat heights from the target)
Z_MAX = 0.35               # ManipSpaceEnv workspace ceiling for the grip point
XY_TOL = 0.04              # OGBench xy_aligned / target_xy_aligned
POS_TOL = 0.02             # OGBench pos_aligned / target_pos_aligned
YAW_MARGIN = 0.15          # keep effector yaw this far inside [-pi, pi]


def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class BarMarkovOracle:
    def __init__(self, env, cfg, rng: np.random.Generator):
        self._env = env
        self._cfg = cfg
        self._o = cfg['oracle']
        self._g = cfg['grasp']
        self._rng = rng
        self._arm_bounds = env.unwrapped._arm_sampling_bounds
        self.pickups = []          # one dict per pickup attempt (whole episode)
        self._done = True

    # ------------------------------------------------------------------ helpers

    def shape_diff(self, diff):
        n = np.linalg.norm(diff)
        return diff if n >= self._o['min_norm'] else diff / (n + 1e-6) * self._o['min_norm']

    def sample_s(self):
        s_max = self._g['s_max']
        if self._g['distribution'] == 'uniform':
            return float(self._rng.uniform(-s_max, s_max))
        if self._g['distribution'] == 'com_offset':
            # Diagnostic only: grasp a chosen distance from the (hidden) centre of mass, |s - c| ~ U[lo, hi], on a
            # random side that keeps the grasp on the bar. Uses privileged info.
            c = self._env.unwrapped.mass.com_offset
            lo, hi = self._g['com_offset_range']
            sides = [sd for sd in (-1, 1) if abs(c + sd * lo) <= s_max] or [1 if c < 0 else -1]
            sd = sides[self._rng.integers(len(sides))]
            return float(np.clip(c + sd * self._rng.uniform(lo, hi), -s_max, s_max))
        if self._g['distribution'] == 'at_com':
            # Diagnostic only: grasp at the (hidden) centre of mass, plus optional jitter. Uses privileged info.
            c = self._env.unwrapped.mass.com_offset
            return float(np.clip(c + self._rng.normal(0.0, self._g.get('com_jitter', 0.0)), -s_max, s_max))
        if self._g['distribution'] == 'normal':
            return float(np.clip(self._rng.normal(0.0, self._g['sigma']), -s_max, s_max))
        raise ValueError(f"unknown grasp.distribution: {self._g['distribution']}")

    @staticmethod
    def feasible_yaw(eff_yaw, yaw, period):
        """Copy of `yaw` (mod period) the effector can reach fastest without leaving [-pi, pi]."""
        best = None
        for k in range(-4, 5):
            cand = yaw + k * period
            if abs(cand) <= np.pi - YAW_MARGIN and (best is None or abs(cand - eff_yaw) < abs(best - eff_yaw)):
                best = cand
        return best

    def _topple_action(self, eff_pos, bar_pos, axis, opening):
        """Push the top of a standing bar over: from outside the workspace centre, inwards, gripper closed."""
        o = self._o
        top = bar_pos + axis * self._cfg['bar']['half_length'] * (1.0 if axis[2] >= 0 else -1.0)
        push_z = max(top[2] - o['topple_below_top'], o['topple_min_z'])
        if self._topple_stage == 0 or self._topple_out is None:
            self._topple_out = self._topple_direction(top, getattr(self, '_topple_turn', 0))
        out = self._topple_out
        start = np.array([top[0], top[1], push_z]) + out * o['topple_standoff']
        end = np.array([top[0], top[1], push_z]) - out * o['topple_push']
        if self._topple_stage == 0:                             # go above the start point
            dest = start + np.array([0, 0, 0.06])
            if np.linalg.norm(dest - eff_pos) < XY_TOL:
                self._topple_stage = 1
        elif self._topple_stage == 1:                           # come down beside the top
            dest = start
            if np.linalg.norm(dest - eff_pos) < POS_TOL:
                self._topple_stage = 2
        else:                                                   # push through
            dest = end
        a = np.zeros(5)
        a[:3] = self.shape_diff(self._route(eff_pos, dest, self._mode == 'carry', transit=self._topple_stage == 0) - eff_pos) * o['gain_pos']
        a[4] = 1.0
        return np.clip(a, -1, 1)

    def _topple_direction(self, top, turn=0):
        """Horizontal unit vector pointing from the bar's top to the push start point.

        Prefer pushing towards the workspace centre (start outside, push inwards), but only from a start point the
        effector can reach (ManipSpaceEnv clips targets to the workspace bounds).
        """
        o = self._o
        lo, hi = self._env.unwrapped._workspace_bounds[:, :2] + np.array([[0.01, 0.01], [-0.01, -0.01]])
        inward = np.array([0.425 - top[0], -top[1]])
        inward = inward / (np.linalg.norm(inward) + 1e-9)
        best, best_score = None, -np.inf
        for ang in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            out = np.array([np.cos(ang), np.sin(ang)])
            start = top[:2] + out * o['topple_standoff']
            if np.any(start < lo) or np.any(start > hi):
                continue
            score = -out @ inward                               # pushing along -out should point inwards
            if score > best_score:
                best, best_score = out, score
        if best is None:
            best = -inward
        c, s_ = np.cos(turn * np.pi / 2), np.sin(turn * np.pi / 2)     # retries rotate the push by 90 deg
        best = np.array([c * best[0] - s_ * best[1], s_ * best[0] + c * best[1]])
        return np.array([best[0], best[1], 0.0])

    def _nudge_action(self, t, eff_pos, bar_pos, axis, eff_yaw, bar_yaw):
        """Roll a lodged bar into the open: fingertips (closed) on top of it, dragged sideways.

        The gripper first turns so its closed fingers lie along the bar: spread across it (the grasp orientation)
        the wall-side finger hits a tall board the bar is lying next to, and the gripper cannot get down.
        """
        o = self._o
        nudge_yaw = self.feasible_yaw(eff_yaw, bar_yaw + np.pi / 2 + self._cfg['grasp']['yaw_offset'], np.pi)
        yaw_ok = abs(nudge_yaw - eff_yaw) < o['yaw_tol_descend']
        r = self._cfg['bar']['half_width']
        if self._nudge_stage == 0:
            perp = np.array([-axis[1], axis[0]])
            perp = perp / (np.linalg.norm(perp) + 1e-9)
            region_y = 0.0
            plat = self._cfg.get('platform') or {}
            if plat.get('enabled'):
                region_y = 0.185 if bar_pos[1] > plat['y_range'][1] else -0.185
            towards = np.array([0.425, region_y]) - bar_pos[:2]
            self._nudge_dir = perp if perp @ towards >= 0 else -perp
            self._nudge_from = bar_pos[:2].copy()
            self._nudge_z = bar_pos[2] + r - o['nudge_press']
            self._nudge_stage = 1
        self._nudge_steps += 1
        if self._nudge_stage == 1:                               # above the bar's centre, fingers along the bar
            dest = np.array([*self._nudge_from, min(self._nudge_z + 0.06, Z_MAX)])
            if np.linalg.norm(dest - eff_pos) < XY_TOL and yaw_ok:
                self._nudge_stage = 2
        elif self._nudge_stage == 2:                             # down onto its top
            dest = np.array([*self._nudge_from, self._nudge_z])
            if eff_pos[2] < self._nudge_z + POS_TOL or self._nudge_steps > 40:
                self._nudge_stage = 3
        else:                                                    # drag sideways: it rolls away from the obstacle
            dest = np.array([*(self._nudge_from + self._nudge_dir * o['nudge_dist']), self._nudge_z])
        # Give up after 90 steps in ANY stage: a bar whose middle is out of the arm's reach never lets stage 1 finish.
        if (self._nudge_stage == 3 and np.linalg.norm(dest[:2] - eff_pos[:2]) < XY_TOL) or self._nudge_steps > 90:
            self._end_attempt(t, 'missed')
            self._mode = 'approach'
            self._start_attempt(t)
        a = np.zeros(5)
        a[:3] = self.shape_diff(self._route(eff_pos, dest, self._mode == 'carry', transit=self._nudge_stage == 1) - eff_pos) * o['gain_pos']
        a[3] = (nudge_yaw - eff_yaw) * o['gain_yaw']
        a[4] = 1.0
        return np.clip(a, -1, 1)

    def _route(self, eff_pos, dest, carrying, transit=False):
        """Travel heights.

        Empty gripper, travel moves (`transit`: approach, retreat, getting to a nudge/topple start): straight up to the
        ceiling, across at the ceiling, then straight down once over `dest`. Nothing in the scene is taller than the
        ceiling, so the empty arm never sweeps through the fence or cradle. Hysteresis: it goes up once more than
        2 * XY_TOL off in xy and comes down once within XY_TOL.

        Carrying, fence enabled: crossing until the whole bar (from above) is on the target's side goes up first too.
        """
        o = self._o
        if not transit:
            self._aloft = False
        if transit and not carrying and o.get('empty_transit_ceiling', True):
            err = np.linalg.norm(np.asarray(dest[:2]) - eff_pos[:2])
            if err > 2 * XY_TOL:
                self._aloft = True
            elif err <= XY_TOL:
                self._aloft = False
            if not getattr(self, '_aloft', False):
                return dest
            return self._over_the_top(eff_pos, dest, o['fence_z_tol_empty'])
        f = self._cfg.get('fence') or {}
        if not (carrying and f.get('enabled')):
            return dest
        tgt_side = 1 if self._info['privileged/target_block_pos'][1] > f['y'] else -1
        if int(self._info['privileged/bar_side'][0]) == tgt_side:
            return dest
        return self._over_the_top(eff_pos, dest, o['fence_z_tol'])

    def _over_the_top(self, eff_pos, dest, tol):
        # Rise until at the ceiling, or until the gripper stops rising (the highest the arm reaches here).
        hist = self._z_hist = (getattr(self, '_z_hist', []) + [float(eff_pos[2])])[-6:]
        stalled = len(hist) == 6 and hist[-1] - hist[0] < 0.001
        if eff_pos[2] < Z_MAX - tol and not stalled:
            return np.array([eff_pos[0], eff_pos[1], Z_MAX])       # straight up first
        return np.array([dest[0], dest[1], Z_MAX])                 # then across at the ceiling

    @staticmethod
    def _carry_turn(grip_yaw, bar_yaw, tgt_yaw):
        """Smallest feasible effector turn that brings the held bar to the target yaw (bar symmetric under 180 deg)."""
        d = wrap_pi(2 * (tgt_yaw - bar_yaw)) / 2
        turns = [d + k * np.pi for k in (-1, 0, 1) if abs(grip_yaw + d + k * np.pi) <= np.pi - YAW_MARGIN]
        return min((abs(x) for x in turns), default=np.inf)

    def _planned_grip_yaw(self, eff_yaw, grip_yaw, bar_yaw, tgt_yaw):
        """Of the two equivalent grasp yaws (two-finger gripper, 180 deg apart), take the one that keeps the later
        carry turn short and inside the effector's yaw limits: a grasp near a limit otherwise forces a ~180 deg turn
        the long way round WITH the bar in hand. Any extra turning is then done empty, before grasping. The choice is
        made once per pickup attempt and then tracked, so it does not flip while approaching."""
        cands = [grip_yaw + k * np.pi for k in (-1, 0, 1) if abs(grip_yaw + k * np.pi) <= np.pi - YAW_MARGIN]
        plan = getattr(self, '_grip_plan', None)
        if plan is None:
            best = min(cands, key=lambda c: self._carry_turn(c, bar_yaw, tgt_yaw) + 0.25 * abs(c - eff_yaw))
            self._grip_plan = best
            return best
        best = min(cands, key=lambda c: abs(c - plan))
        self._grip_plan = best
        return best

    def _to_slot(self, tgt_pos):
        """The current target is the slot (not a floor pose)."""
        sl = self._cfg.get('slot') or {}
        return bool(sl.get('enabled')) and np.hypot(tgt_pos[0] - sl['x'], tgt_pos[1] - sl['y']) < 0.01

    def _slide_out_action(self, a, bar_pos, bar_yaw, grasped, g, info):
        """Slide a seated bar straight back out of the slot along +y, at pocket height and lined up with the slot,
        to the staging pose in front of it; then the normal carry takes over. While its tips are in the pockets the
        pockets keep it level, so this works wherever along the bar it was grasped. Steers the bar's true pose."""
        o, sl = self._o, self._cfg['slot']
        eff_yaw = info['proprio/effector_yaw'][0]
        dest = np.array([sl['x'], sl['y'] + o['slot_stage_dy'], sl['height'] + self._cfg['bar']['half_width']])
        diff = dest - bar_pos
        diff[1] = min(diff[1], o['slot_slide_step'])                        # slowly along +y
        d_out = float(bar_pos[1] - sl['y'])
        if d_out > self._extract_best + 0.0005:
            self._extract_best, self._extract_stall = d_out, 0
        else:
            self._extract_stall += 1
        if d_out >= o['slot_stage_dy'] - 0.005 or self._extract_stall >= o['slot_stall_steps']:
            self._extract = False                                           # out (or stuck): normal carry from here
        # keep the bar along the slot's axis (yaw 0 or pi, whichever is nearer)
        rot = self.feasible_yaw(eff_yaw, eff_yaw + wrap_pi(2 * (0.0 - bar_yaw)) / 2, np.pi) - eff_yaw
        if getattr(self, '_slide_pi_phase', None) != 18:
            self._slide_int, self._slide_pi_phase = np.zeros(3), 18
        self._slide_int = np.clip(self._slide_int + diff, -0.05, 0.05)
        a[:3] = np.clip(diff * o['slot_gain'] + self._slide_int * o.get('slot_ki', 2.0), -o['slot_max_action'], o['slot_max_action'])
        a[3], a[4] = rot * o['gain_yaw'], 1.0
        self.phase, self.grasped, self.grasp_point = 18, grasped, g
        return a

    def _slide_mode(self):
        return (self._cfg.get('slot') or {}).get('insert') == 'slide'

    def _slide_action(self, t, a, bar_pos, bar_yaw, tgt_pos, tgt_yaw, rot, grasped, g, info):
        """Slot v2 (perfect-information placer): slide the bar in sideways.

        0 over a staging pose `slot_stage_dy` in front (+y) of the slot, lined up -> 1 set it down flat on the floor
        there (the floor levels it) -> 2 open -> 3 re-centre on the COM precisely and grip again -> 4 lift just to
        groove height and wait until level (re-do 1-3 if it tilts) -> 5 slide along -y into both grooves until home
        against the back stops (or jammed) -> let go. Steers the bar's TRUE pose (privileged).
        """
        if self._o.get('slot_smash_grip'):
            return self._smash_grip_action(a, bar_pos, bar_yaw, tgt_pos, tgt_yaw, rot, grasped, g, info)
        o, hw = self._o, self._cfg['bar']['half_width']
        eff_pos = info['proprio/effector_pos']
        opening = info['proprio/gripper_opening'][0]
        axis = info['privileged/bar_axis']
        yaw_cmd, grip = rot * o['gain_yaw'], 1.0
        yaw_e = abs(wrap_pi(2 * (tgt_yaw - bar_yaw)) / 2)
        yaw_ok = yaw_e < np.radians(o['slot_yaw_tol_deg'])
        tilt = abs(float(info['privileged/bar_tilt'][0]))
        stage = tgt_pos + np.array([0.0, o['slot_stage_dy'], 0.0])          # groove height, in front of the slot
        floor = np.array([stage[0], stage[1], hw])
        clear_z = tgt_pos[2] + o['slot_clear_z']
        e_st = np.linalg.norm(stage[:2] - bar_pos[:2])
        cap, eff_mode = o['slot_max_action'], False
        st = self._slide_stage
        if st == 0:
            # lift: judged at the grasp point when not setting down (the bar's middle lags when it hangs tilted)
            zl = g[2] if not o.get('slot_setdown', True) else bar_pos[2]
            if zl < clear_z - 0.03 and e_st > 0.02:
                phase, diff, cap = 4, np.array([0.0, 0.0, clear_z - zl]), o['slot_lift_action']
            else:
                phase, diff = 5, np.array([stage[0], stage[1], clear_z]) - bar_pos
                if e_st < o['slot_xy_tol'] and yaw_ok and bar_pos[2] > clear_z - 0.03:
                    if o.get('slot_setdown', True):
                        self._slide_stage = 1
                    else:
                        # No set-down / regrip (user, 2026-10-14): keep the grasp, go straight down to groove height
                        # in front of the slot, wait briefly for it to hang level, then slide in.
                        self._slide_stage, self._slide_level, self._slide_wait = 4, 0, 0
        elif st == 1 and o.get('slot_no_level'):
            # 'Smash' variant (play data): go straight to pocket height in front of the slot and slide in whatever the
            # tilt -- no set-down, regrip or waiting until level. Jams count as attempts (released after a stall).
            phase, diff = 17, stage - bar_pos
            diff[2] = max(diff[2], -o['slot_descend_step'])
            if e_st > 2 * o['slot_xy_tol']:
                self._slide_stage = 0
            elif abs(bar_pos[2] - stage[2]) < 0.004:
                self._slide_stage = 5
        elif st == 1:                                                         # set down flat in front of the slot
            phase, diff = 12, floor - bar_pos
            diff[2] = max(diff[2], -o['slot_descend_step'])
            if e_st > 2 * o['slot_xy_tol']:
                self._slide_stage = 0
            elif info['privileged/bar_floor_contact'][0] and bar_pos[2] < hw + 0.002:
                self._slide_stage, self._slide_hold = 2, 0
        elif st == 2:                                                         # let go (gently)
            phase, diff, yaw_cmd, grip = 15, np.zeros(3), 0.0, -o['release_rate']
            if opening < 0.1:
                self._slide_stage, self._slide_hold, self._slide_regrip_steps = 3, 0, 0
        elif st == 3:                                                         # re-centre on the COM, grip again
            phase, eff_mode, yaw_cmd = 16, True, 0.0
            # Aim a few mm above the bar's centre line: open fingertips dragging on the floor stall the last mm.
            diff = g + np.array([0.0, 0.0, o.get('slot_regrip_lift', 0.004)]) - eff_pos
            along = abs(float((eff_pos - g) @ axis))
            self._slide_regrip_steps = getattr(self, '_slide_regrip_steps', 0) + 1
            tol = o.get('precise_grasp_axis_tol', 0.0015) * (1.0 if self._slide_regrip_steps < 30 else 1.7)
            if along <= tol and np.linalg.norm(diff) <= POS_TOL:
                grip = 1.0
                both = bool(info['privileged/pad_contact_left'][0] and info['privileged/pad_contact_right'][0])
                self._slide_hold = self._slide_hold + 1 if both else 0
                if self._slide_hold >= o['grasp_confirm']:
                    self._slide_stage, self._slide_level = 4, 0
            else:
                grip, self._slide_hold = -1.0, 0
        elif st == 4:                                                         # lift to groove height, wait level
            # Height is that of the grasp point (what the gripper holds). The bar's middle drops by |s| sin(tilt) when
            # gripped off-middle, so checking the middle coupled height to levelness: a lopsided bar gripped at its
            # COM ~11 cm out never counted as 'at height' above ~1.6 deg, crept past 3 deg and was set down again.
            hz = g[2]
            htol = o.get('slot_height_tol', 0.003)
            phase, diff = (17 if abs(hz - stage[2]) > htol else 13), stage - bar_pos
            diff[2] = min(stage[2] - hz, o['slot_descend_step'])
            setdown = o.get('slot_setdown', True)
            if setdown and tilt > np.radians(3.0) and self._slide_tries < 3:  # tilted again: set it down, regrip
                self._slide_tries += 1
                self._slide_stage = 1
            at_h = abs(hz - stage[2]) < htol and yaw_ok
            ok = tilt < np.radians(o['slot_level_deg']) and at_h
            self._slide_level = self._slide_level + 1 if ok else 0
            self._slide_wait = getattr(self, '_slide_wait', 0) + (1 if abs(hz - stage[2]) < 0.01 else 0)
            # without set-down: slide anyway after `slot_level_wait_max` steps within 1 cm of groove height (the
            # 'at height' test flickers for a swinging bar; a tilted bar just jams)
            if self._slide_level >= o['slot_settle_steps'] or (not setdown and self._slide_wait >= o.get('slot_level_wait_max', 20)):
                self._slide_stage = 5
        else:                                                                 # slide in along -y
            phase, diff = 14, tgt_pos - bar_pos
            diff[1] = max(diff[1], -o['slot_slide_step'])
            dist = abs(bar_pos[1] - tgt_pos[1])
            if dist < self._slide_best - 0.0005:
                self._slide_best, self._slide_stall = dist, 0
            else:
                self._slide_stall += 1
            if dist < 0.003 or self._slide_stall >= o['slot_stall_steps']:
                self._slide_done = True
        # PI control: the arm sags under small relative commands, so pure P stalls a few mm short.
        if phase != getattr(self, '_slide_pi_phase', None):
            self._slide_int, self._slide_pi_phase = np.zeros(3), phase
        self._slide_int = np.clip(self._slide_int + diff, -0.05, 0.05)
        a[:3] = np.clip(diff * o['slot_gain'] + self._slide_int * o.get('slot_ki', 2.0), -cap, cap)
        a[3], a[4] = yaw_cmd, grip
        self.phase, self.grasped, self.grasp_point = phase, grasped, g
        return a

    def _smash_grip_action(self, a, bar_pos, bar_yaw, tgt_pos, tgt_yaw, rot, grasped, g, info):
        """'Grip-height smash' (play data): wherever the bar was grasped, bring the GRIPPER to pocket height in front of
        the slot, at the spot it would be in if the bar were seated (slot centre + s along the slot axis), and push
        straight in along -y. Steers the gripper, not the bar; no levelling, set-down or regrip: a tilted bar just
        hits whatever is there. Released after reaching the seat line or stalling."""
        o = self._o
        eff = info['proprio/effector_pos']
        axis = info['privileged/bar_axis']
        sgn = 1.0 if axis[0] >= 0 else -1.0                         # bar ends up along +x or -x, whichever is nearer
        seat = tgt_pos + np.array([sgn * self._s, 0.0, 0.0])         # gripper position if the bar were seated
        stage = seat + np.array([0.0, o['slot_stage_dy'], 0.0])
        clear_z = tgt_pos[2] + o['slot_clear_z']
        yaw_cmd = rot * o['gain_yaw']
        yaw_ok = abs(rot) < np.radians(o['slot_yaw_tol_deg'])
        e_st = np.linalg.norm(stage[:2] - eff[:2])
        cap = o['slot_max_action']
        st = self._slide_stage
        if st == 0:                                                  # up, over the staging spot, lined up
            if eff[2] < clear_z - 0.03 and e_st > 0.02:
                phase, diff, cap = 4, np.array([0.0, 0.0, clear_z - eff[2]]), o['slot_lift_action']
            else:
                phase, diff = 5, np.array([stage[0], stage[1], clear_z]) - eff
                if e_st < o['slot_xy_tol'] and yaw_ok:
                    self._slide_stage = 1
        elif st == 1:                                                # down to pocket height (gripper)
            phase, diff = 17, stage - eff
            diff[2] = max(diff[2], -o['slot_descend_step'])
            if e_st > 2 * o['slot_xy_tol']:
                self._slide_stage = 0
            elif abs(eff[2] - stage[2]) < 0.004:
                self._slide_stage = 5
        else:                                                        # straight in along -y
            phase, diff = 14, seat - eff
            diff[1] = max(diff[1], -o['slot_slide_step'])
            dist = abs(eff[1] - seat[1])
            if dist < self._slide_best - 0.0005:
                self._slide_best, self._slide_stall = dist, 0
            else:
                self._slide_stall += 1
            if dist < 0.003 or self._slide_stall >= o['slot_stall_steps']:
                self._slide_done = True
        if phase != getattr(self, '_slide_pi_phase', None):
            self._slide_int, self._slide_pi_phase = np.zeros(3), phase
        self._slide_int = np.clip(self._slide_int + diff, -0.05, 0.05)
        a[:3] = np.clip(diff * o['slot_gain'] + self._slide_int * o.get('slot_ki', 2.0), -cap, cap)
        a[3], a[4] = yaw_cmd, 1.0
        self.phase, self.grasped, self.grasp_point = phase, grasped, g
        return a

    def _slot_on(self):
        # `oracle.slot_oracle: false` keeps OGBench cube-single's plain pick-and-place phases even with the slot.
        return bool((self._cfg.get('slot') or {}).get('enabled')) and self._o.get('slot_oracle', True)

    def _bar_axis(self, yaw):
        return np.array([np.cos(yaw), np.sin(yaw), 0.0])

    # ------------------------------------------------------------------ attempts

    def _sample_reachable_s(self):
        """Draw s until the grasp point is inside the arm's workspace (only bars near a wall are affected)."""
        bar_pos, axis = self._seen
        lo, hi = self._env.unwrapped._workspace_bounds[:, :2] + np.array([[0.01, 0.01], [-0.01, -0.01]])
        for _ in range(self._o['s_reach_tries']):
            s = self.sample_s()
            g = bar_pos[:2] + s * axis[:2]
            if np.all(g >= lo) and np.all(g <= hi):
                return s
        return 0.0

    def _start_attempt(self, t):
        self._s = self._sample_reachable_s()
        sl = self._cfg.get('slot') or {}
        if sl.get('enabled') and sl.get('insert') == 'slide' and self._env.unwrapped.bar_in_slot():
            # Seated bar: grasps within ~2 cm of a tip put the fingers against the end blocks (they jam and slip),
            # so keep the grasp on the open part of the bar.
            lim = self._o.get('slot_seated_grasp_s_max', 0.14)
            self._s = float(np.clip(self._s, -lim, lim))
        self._approach_steps = 0
        self._topple_steps = 0
        self._nudge_stage = 0
        self._nudge_steps = 0
        self._armed = False        # gripper must reopen fully before grasping
        self._lost = 0
        self._miss = 0
        self._hold = 0
        self._on_table = 0
        self._grip_plan = None     # grasp yaw chosen for this attempt (see _planned_grip_yaw)
        self._extract = False      # sliding a seated bar out of the slot (see _slide_out_action)
        self._slide_stage = 0      # slot v2: 0 over staging, 1 lower, 2 level, 3 slide
        self._slide_done = False
        self._slide_level = 0
        self._slide_best = np.inf
        self._slide_stall = 0
        self._slide_tries = 0
        self._slide_hold = 0
        self._topple_stage = 0
        self._toppling = False
        self._topple_out = None
        self.pickups.append(dict(subtask_idx=self.subtask_idx, t_start=t, t_grasp=-1, t_end=-1,
                                 s=self._s, outcome='', max_tilt=0.0))

    def _end_attempt(self, t, outcome):
        p = self.pickups[-1]
        if p['outcome'] == '':
            p['t_end'] = t
            p['outcome'] = outcome

    @property
    def pickup_idx(self):
        return len(self.pickups) - 1

    def set_grasp_point(self, s):
        """Override the current attempt's grasp point (scripted play). Keeps the pickup log consistent."""
        self._s = float(s)
        if self.pickups:
            self.pickups[-1]['s'] = float(s)

    @property
    def s(self):
        return self._s

    @property
    def done(self):
        return self._done

    def reset(self, ob, info, subtask_idx, t):
        """Start a new sub-task at env step t (the current target is already set in the env)."""
        if self.pickups:
            self._end_attempt(t, 'timeout')
        self.subtask_idx = subtask_idx
        self._done = False
        self._step = 0
        self._mode = 'approach'
        self._released_above = False
        self._final_pos = self._rng.uniform(*self._arm_bounds)
        self._final_yaw = self._rng.uniform(-np.pi, np.pi)
        self._seen = (info['privileged/block_0_pos'].copy(), info['privileged/bar_axis'].copy())
        self._start_attempt(t)

    def finish_episode(self, t):
        if self.pickups:
            self._end_attempt(t, 'timeout')

    # ------------------------------------------------------------------ policy

    def select_action(self, ob, info, t):
        o = self._o
        self._info = info
        eff_pos = info['proprio/effector_pos']
        eff_yaw = info['proprio/effector_yaw'][0]
        opening = info['proprio/gripper_opening'][0]
        contact = info['proprio/gripper_contact'][0]
        grasped = bool(info['privileged/pad_contact_left'][0] and info['privileged/pad_contact_right'][0])

        bar_pos = info['privileged/block_0_pos']
        bar_yaw = info['privileged/block_0_yaw'][0]
        axis = info['privileged/bar_axis']
        self._seen = (bar_pos.copy(), axis.copy())
        g = bar_pos + self._s * axis                                          # grasp point (world)
        tgt_pos = info['privileged/target_block_pos']
        tgt_yaw = info['privileged/target_block_yaw'][0]

        # Yaw the effector must hold for the fingers to close across the bar's narrow side.
        grip_yaw = self.feasible_yaw(eff_yaw, bar_yaw + self._cfg['grasp']['yaw_offset'], np.pi)
        if self._mode == 'approach' and o.get('plan_grasp_yaw', True):
            grip_yaw = self._planned_grip_yaw(eff_yaw, grip_yaw, bar_yaw, tgt_yaw)
        # Of the two equivalent target yaws (bar symmetric under 180 deg), the one reachable by a feasible
        # effector rotation; the target grasp point follows that choice.
        rot = self.feasible_yaw(eff_yaw, eff_yaw + wrap_pi(2 * (tgt_yaw - bar_yaw)) / 2, np.pi) - eff_yaw
        tgt_yaw_eff = bar_yaw + rot
        tg = tgt_pos + self._s * self._bar_axis(tgt_yaw_eff)                 # target grasp point (world)
        if (self._cfg.get('cradle') or {}).get('enabled'):
            # Short cradle: set the bar down at the point it is held, i.e. the grasp point onto the cradle. The policy
            # cannot see the centre of mass, so this balances only when it grasped near it.
            tg = tgt_pos.copy()
            if o.get('place_point') == 'com_exact':
                # DIAGNOSTIC ONLY, perfect privileged information: every step, put the grasp point where it must be for
                # the centre of mass's ACTUAL 3D position (tilt included: (s-c)cos(theta) out, (s-c)sin(theta) down) to
                # sit on the support. Closed loop, so as the support levers the bar flat the gripper follows.
                tg = tgt_pos - (info['privileged/bar_com_world'] - g)
            elif o.get('place_point') == 'com':
                # DIAGNOSTIC ONLY (reads the hidden centre of mass): put the centre of mass, not the grasp point, over
                # the support -- the best a policy that "sees the tilt and compensates" could do.
                tg = tgt_pos + (self._s - self._env.unwrapped.mass.com_offset) * self._bar_axis(tgt_yaw_eff)

        above = eff_pos[2] > ABOVE_THRESHOLD
        xy_aligned = np.linalg.norm(g[:2] - eff_pos[:2]) <= XY_TOL
        pos_aligned = np.linalg.norm(g - eff_pos) <= POS_TOL
        if o.get('precise_grasp_axis_tol'):
            # Perfect placer: close only once the grip is this close to the grasp point ALONG the bar (held at the
            # COM, a few mm of offset is enough for the weak grip to pivot the bar slowly over).
            pos_aligned = pos_aligned and abs(float((eff_pos - g) @ axis)) <= o['precise_grasp_axis_tol']
        tgt_xy_aligned = np.linalg.norm(tg[:2] - g[:2]) <= XY_TOL
        tgt_pos_aligned = np.linalg.norm(tg - g) <= POS_TOL
        gripper_open = contact < 0.1
        yaw_err = abs(grip_yaw - eff_yaw)
        p = self.pickups[-1]

        # ---- mode transitions
        if self._mode == 'approach':
            if opening < 0.1:
                self._armed = True
            # Both pads on the bar (privileged contacts). OGBench's force threshold (gripper_contact > 0.5, i.e.
            # 25 N on the right pad) is not required: a weaker grip never reaches it.
            self._hold = self._hold + 1 if (self._armed and pos_aligned and grasped) else 0
            if self._hold >= o['grasp_confirm']:
                self._mode = 'carry'
                p['t_grasp'] = t
                # Gripped a bar that is seated in a slide-in slot and has to go elsewhere: slide it out first.
                self._extract = bool(self._slot_on() and self._slide_mode() and not self._to_slot(tgt_pos)
                                     and info['privileged/bar_slot_contact'][0])
                self._extract_best, self._extract_stall = -np.inf, 0
            elif self._armed and pos_aligned and yaw_err <= o['yaw_tol_grasp']:
                # Grasping. A miss is either the gripper closed on nothing, or no stable two-pad grasp in time.
                self._miss += 1
                closed_on_nothing = opening > o['miss_opening'] and not grasped
                if closed_on_nothing or self._miss >= o['grasp_patience']:
                    self._end_attempt(t, 'missed')
                    self._start_attempt(t)
        elif self._mode == 'carry':
            regrip = self._slot_on() and self._slide_mode() and self._to_slot(tgt_pos) and self._slide_stage in (2, 3)
            self._lost = 0 if (grasped or regrip) else self._lost + 1
            if self._lost >= o['slip_patience'] and self._slot_on() and self._to_slot(tgt_pos) \
                    and self._env.unwrapped.bar_in_slot(ignore_fingers=True):
                # Out of the fingers (or held by one fingertip only) but seated (a lopsided bar can pivot off the fingertips right at the end of the
                # slide): done. Re-grasping it would pull it back out.
                self._mode = 'release'
                self._end_attempt(t, 'placed')
            elif self._lost >= o['slip_patience']:
                self._end_attempt(t, 'slipped')
                self._start_attempt(t)
                self._mode = 'approach'
                g = bar_pos + self._s * axis
                pos_aligned = xy_aligned = False
            elif getattr(self, '_extract', False):
                pass                                                  # sliding out: no release
            elif self._slot_on() and self._slide_mode() and self._to_slot(tgt_pos):
                if self._slide_done:                                  # slid home (or jammed): let go
                    self._mode = 'release'
                    self._end_attempt(t, 'placed')
            elif self._slot_on() and self._to_slot(tgt_pos):
                # Slot: let go only once the bar is seated -- centre over the slot, down at seat height, level and lined
                # up -- or after `place_stall_steps` touching the slot (closed loop on the bar's true pose).
                e = tgt_pos - bar_pos
                yaw_e = abs(wrap_pi(2 * (tgt_yaw - bar_yaw)) / 2)
                tilt = abs(float(info['privileged/bar_tilt'][0]))
                touching = bool(info['privileged/bar_slot_contact'][0])
                self._on_table = self._on_table + 1 if touching else 0
                seated = (np.linalg.norm(e[:2]) < o['slot_xy_tol'] and e[2] > -o['slot_z_tol']
                          and tilt < np.radians(o['slot_release_tilt_deg']) and yaw_e < np.radians(o['slot_yaw_tol_deg']))
                if (touching and seated) or self._on_table >= o['place_stall_steps']:
                    self._mode = 'release'
                    self._end_attempt(t, 'placed')
            else:
                # A bar hanging tilted touches the table with its low end before the grasp point reaches placing
                # height. Keep lowering (it pivots flatter in the grasp) and let go once it is nearly level, or
                # after `place_stall_steps` of table contact. Dropping it steeply makes it land on its heavy end,
                # where it stays standing.
                # The gripper at its lowest reachable height counts too: a bar that slid up in the grip rests on the
                # fingertips above the floor and would otherwise be held there until the sub-task times out.
                at_floor = eff_pos[2] < self._env.unwrapped._workspace_bounds[0, 2] + 0.005
                on_table = tgt_xy_aligned and bool(info['privileged/bar_floor_contact'][0]
                                                   or info['privileged/bar_platform_contact'][0]
                                                   or info['privileged/bar_cradle_contact'][0] or at_floor)
                self._on_table = self._on_table + 1 if on_table else 0
                max_tilt = o['place_level_deg'] if o.get('place_require_level') else o['place_max_tilt_deg']
                level = abs(float(info['privileged/bar_tilt'][0])) < np.radians(max_tilt)
                if o.get('place_require_level'):
                    # Short cradle: hold the bar on it until it is level, then let go (any tilt left at release
                    # makes it slide off along its axis).
                    release = (tgt_pos_aligned or on_table) and level
                else:
                    release = tgt_pos_aligned or (on_table and level)
                if release or self._on_table >= o['place_stall_steps']:
                    self._mode = 'release'
                    self._end_attempt(t, 'placed')
        if self._mode == 'carry' or (self._mode == 'approach' and grasped):
            p['max_tilt'] = max(p['max_tilt'], abs(float(info['privileged/bar_tilt'][0])))

        # ---- a bar the approach cannot reach (lodged against the board or a wall): roll it free, then retry
        if self._mode == 'approach' and not grasped and not self._toppling:
            # Travel at the ceiling doesn't count: only time spent at the bar without getting a grasp means it is lodged.
            if not getattr(self, '_aloft', False):
                self._approach_steps += 1
            if self._approach_steps > o['approach_patience']:
                self._mode = 'nudge'
        if self._mode == 'nudge':
            a = self._nudge_action(t, eff_pos, bar_pos, axis, eff_yaw, bar_yaw)
            self.phase, self.grasped, self.grasp_point = 11, grasped, g
            self._step += 1
            if self._step >= o['max_subtask_steps']:
                self._done = True
            return a

        # ---- a bar standing on end cannot be grasped from above: push its top over first
        # Hysteresis: start at upright_deg, keep pushing until below topple_done_deg. A bar with its mass at the
        # bottom rights itself (like a roly-poly toy) unless pushed well past its tipping angle.
        tilt = abs(float(info['privileged/bar_tilt'][0]))
        if tilt > np.radians(o['upright_deg']):
            self._toppling = True
        elif tilt < np.radians(o['topple_done_deg']):
            self._toppling = False
        if self._mode == 'approach' and self._toppling and not grasped:
            self._topple_steps += 1
            if self._topple_steps > o['topple_patience']:       # not tipping over: push from another side
                self._topple_steps = 0
                self._topple_stage = 0
                self._topple_out = None
                self._topple_turn = getattr(self, '_topple_turn', 0) + 1
            a = self._topple_action(eff_pos, bar_pos, axis, opening)
            self.phase, self.grasped, self.grasp_point = 10, grasped, g
            self._step += 1
            if self._step >= o['max_subtask_steps']:
                self._done = True
            return a
        self._topple_stage = 0

        # ---- phase actions (OGBench numbering)
        a = np.zeros(5)
        if self._mode == 'approach':
            yaw_cmd = (grip_yaw - eff_yaw) * o['gain_yaw']
            # Descend only once roughly aligned in yaw, close only once tightly aligned: closing while still turning
            # sweeps the fingers along the bar and knocks it away.
            if not xy_aligned or yaw_err > o['yaw_tol_descend']:
                phase, dest, grip = 1, g + np.array([0, 0, ABOVE_OFFSET]), -1
            elif not pos_aligned or not self._armed or yaw_err > o['yaw_tol_grasp']:
                phase, dest, grip = 2, g, -1
            else:
                phase, dest, grip = 3, g, 1
        elif self._mode == 'carry' and getattr(self, '_extract', False):
            a = self._slide_out_action(a, bar_pos, bar_yaw, grasped, g, info)
            self._step += 1
            if self._step >= o['max_subtask_steps']:
                self._done = True
            return np.clip(a, -1, 1)
        elif self._mode == 'carry' and self._slot_on() and self._slide_mode() and self._to_slot(tgt_pos):
            a = self._slide_action(t, a, bar_pos, bar_yaw, tgt_pos, tgt_yaw, rot, grasped, g, info)
            self._step += 1
            if self._step >= o['max_subtask_steps']:
                self._done = True
            return np.clip(a, -1, 1)
        elif self._mode == 'carry' and self._slot_on() and self._to_slot(tgt_pos):
            # Slot (perfect-information placer): steer the bar's TRUE centre (privileged pose), not the grasp point.
            # Lift clear of the slot's end walls, go over the slot lined up with it, then lower slowly into it.
            yaw_cmd = rot * o['gain_yaw']
            grip = 1
            e_xy = np.linalg.norm(tgt_pos[:2] - bar_pos[:2])
            yaw_e = abs(wrap_pi(2 * (tgt_yaw - bar_yaw)) / 2)
            clear_z = tgt_pos[2] + o['slot_clear_z']
            if e_xy > 2 * o['slot_xy_tol'] or yaw_e > 2 * np.radians(o['slot_yaw_tol_deg']):
                self._slot_lowering = False
            elif e_xy < o['slot_xy_tol'] and yaw_e < np.radians(o['slot_yaw_tol_deg']) and bar_pos[2] > clear_z - 0.03:
                self._slot_lowering = True
            if getattr(self, '_slot_lowering', False):
                phase, bar_dest = 6, tgt_pos.copy()
            elif bar_pos[2] < clear_z - 0.03 and e_xy > 0.02:
                # Straight up first, holding the GRIPPER's xy (chasing a swinging bar slides it through the fingers).
                phase = 4
                diff = np.array([0.0, 0.0, clear_z - bar_pos[2]])
                a[:3] = np.clip(diff * o['slot_gain'], -o['slot_lift_action'], o['slot_lift_action'])
                a[3], a[4] = yaw_cmd, grip
                self.phase, self.grasped, self.grasp_point = phase, grasped, g
                self._step += 1
                if self._step >= o['max_subtask_steps']:
                    self._done = True
                return np.clip(a, -1, 1)
            else:
                phase, bar_dest = 5, np.array([tgt_pos[0], tgt_pos[1], clear_z])      # over the slot, lined up
            diff = bar_dest - bar_pos
            if phase == 6:
                diff[2] = max(diff[2], -o['slot_descend_step'])                       # lower slowly
            self.phase, self.grasped, self.grasp_point = phase, grasped, g
            a[:3] = np.clip(diff * o['slot_gain'], -o['slot_max_action'], o['slot_max_action'])   # gentle, no min-norm
            a[3], a[4] = yaw_cmd, grip
            self._step += 1
            if self._step >= o['max_subtask_steps']:
                self._done = True
            return np.clip(a, -1, 1)
        elif self._mode == 'carry':
            yaw_cmd = rot * o['gain_yaw']
            if not above and not tgt_xy_aligned:
                phase, dest = 4, np.array([g[0], g[1], ABOVE_OFFSET * 2])
            elif above and not tgt_xy_aligned:
                phase, dest = 5, tg + np.array([0, 0, ABOVE_OFFSET])
            else:
                phase, dest = 6, tg
            grip = 1
            dest = eff_pos + (dest - g)       # move the grasp point, not the effector, to dest
        else:  # release / retreat
            # Heights are measured from the target, not the floor: on a raised board OGBench's absolute 0.16 m
            # 'above' test passes right after release, and a low final pose sweeps the fingers across the bar.
            base = tgt_pos[2] - self._cfg['bar']['half_width']           # support surface under the target
            above_rel = eff_pos[2] > min(base + ABOVE_THRESHOLD, Z_MAX - 0.02)
            final = self._final_pos.copy()
            final[2] = min(max(final[2], base + o['retreat_clearance']), Z_MAX)
            if not gripper_open:
                # Open gently: snapping the soft fingertips open flicks the capsule into a roll.
                phase, dest, grip, yaw_cmd = 7, eff_pos, -o['release_rate'], 0.0
                self._released_above = False
            elif not (above_rel or self._released_above):
                phase, dest, grip, yaw_cmd = 8, np.array([eff_pos[0], eff_pos[1], min(base + 2 * ABOVE_THRESHOLD, Z_MAX)]), -1, 0.0
            else:
                self._released_above = True
                phase, dest, grip = 9, final, -1
                yaw_cmd = (self._final_yaw - eff_yaw) * o['gain_yaw']
                if np.linalg.norm(final - eff_pos) <= XY_TOL:
                    self._done = True

        a[:3] = self.shape_diff(self._route(eff_pos, dest, self._mode == 'carry', transit=phase in (1, 9)) - eff_pos) * o['gain_pos']
        if o.get('carry_max_action') and phase in (4, 5, 6):
            # Capped proportional carry (play 'free' moves, user 2026-10-14: as slow as the probes' lifts). OGBench's
            # min-norm shaping above commands full speed (~2 cm/step) until the waypoint is reached.
            cap = o['carry_max_action']
            v = (self._route(eff_pos, dest, True, transit=False) - eff_pos) * 15.0
            a[:3] = v * min(1.0, cap / max(np.linalg.norm(v), 1e-9))       # cap the speed in any direction
        if o.get('precise_grasp_axis_tol') and phase in (2, 3):
            # Fine proportional approach (no min-norm steps, which overshoot by millimetres).
            a[:3] = np.clip((dest - eff_pos) * o['slot_gain'], -o['slot_max_action'], o['slot_max_action'])
        a[3] = yaw_cmd
        a[4] = grip
        self.phase = phase
        self.grasped = grasped
        self.grasp_point = g

        self._step += 1
        if self._step >= o['max_subtask_steps']:
            self._done = True
        return np.clip(a, -1, 1)
