"""Play episodes for the OGBar slot task (v3): a random sequence of scripted behaviour segments per episode.

Segment types:
  probe      grasp at a new point, lift 10-25 cm, hold, set it back down where it was, let go. The held tilt is
             recorded; tilt ~ k * (s - c), so a line through (s, tilt) estimates the COM c from what the camera sees.
  slide_in   grasp at the COM and slide the bar into the slot. Always followed directly by slide_out.
             COM = the probe estimate if >= 2 probes came earlier in the episode, else the privileged (true) COM.
  slide_out  push the seated bar back out of the slot with the closed gripper (no grasp), from behind its middle along
             +y; it drops off the blocks onto the floor in front.
  smash      grasp at a random point, bring the gripper to pocket height in front of the slot and push straight in
             (no levelling). If the bar ends up in or against the slot, a slide_out follows automatically.
  free       pick up at a random point and put it down at a random pose on the free floor.
  clear      non-prehensile: with the gripper closed, go behind the bar (-y side) at bar height and sweep it out of the
             slot area along +y onto the free floor. Runs after any segment that leaves the bar (not seated) in or
             right in front of the slot, e.g. a failed smash, where grasping it jams against the pockets.

Each plan samples 0-3 probes, an in/out pair (p 0.6), a smash (p 0.5) and 0-3 free moves, shuffled (at least one
segment). `PlayEpisode.run(..., plan=[...])` takes an explicit plan of units instead ('probe', 'pair', 'smash', 'free').
Episodes have a fixed length of `max_steps` (1000): a plan that finishes early is followed by another sampled plan, and
whatever is still running at `max_steps` is cut off. The episode ends early only if the bar leaves the workspace.
Only the COM choice for slide_in may use privileged information, and only when no probe estimate exists.
"""
import copy

import numpy as np

from ogbar.envs.bar_env import _segment_rect_distance
from ogbar.oracle import BarMarkovOracle

SEGMENTS = ('probe', 'slide_in', 'slide_out', 'smash', 'free', 'clear')
# Stages of the scripted (non-oracle) parts: probe (lift .. up) and clear (raise .. push, up). '' = oracle-driven.
PROBE_STAGES = ('', 'lift', 'hold', 'lower', 'open', 'up', 'raise', 'over', 'down', 'push')


def _quat_z(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


class PlayEpisode:
    def __init__(self, env, cfg, rng, noise_xi=0.0, max_steps=1000, on_step=None, before_step=None, noise_rng=None,
                 n_probe_max=3, p_pair=0.6, p_smash=0.5, n_free_max=3, lift_range=(0.10, 0.25), refill=True,
                 p_free=0.5, free_max_action=None, probe_s=None, probe_lift=None, smash_s=None):
        self.env, self.cfg, self.rng = env, cfg, rng
        self.probe_s = None if probe_s is None else list(probe_s)   # eval: the k-th probe grasps at probe_s[k] (no redraw)
        self.probe_lift = probe_lift                                    # eval: fixed lift height (m)
        self.smash_s = None if smash_s is None else list(smash_s)   # showcase/eval: the k-th smash grasps at smash_s[k]
        self.noise_rng = noise_rng if noise_rng is not None else rng
        self.xi, self.max_steps, self.refill = noise_xi, max_steps, refill
        self.n_probe_max, self.p_pair, self.p_smash, self.n_free_max, self.lift_range = n_probe_max, p_pair, p_smash, n_free_max, lift_range
        self.p_free, self.free_max_action = p_free, free_max_action
        self.plan = None
        self.com_source = {}          # segment index -> 'estimate' / 'privileged'
        self.on_step = on_step            # on_step(ob, info, a, label) after every env step
        self.before_step = before_step    # before_step(a, a_clean, label) before it (self.ob / self.info = frame t)
        self.oracles = []                 # every oracle used, in order (their pickup logs)
        self.out_of_bounds = False
        self._lifted = False              # the bar has been picked up and lifted at least once (probe / free / smash)
        self.unreachable = False          # the whole bar ended up out of the arm's reach: episode ended early
        self.t = 0
        self.log = []                 # one dict per segment
        self.c_est = None
        self._probes = []             # (s, held tilt in deg)
        self._sub = 0

    # ------------------------------------------------------------------ plumbing
    def _set_target(self, pos, yaw):
        e = self.env.unwrapped
        e._data.mocap_pos[e._target_mocap_id] = pos
        e._data.mocap_quat[e._target_mocap_id] = _quat_z(yaw)

    def _slot_target(self):
        sl, hw = self.cfg['slot'], self.cfg['bar']['half_width']
        return np.array([sl['x'], sl['y'], sl['height'] + hw]), 0.0

    def _floor_target(self):
        return self.env.unwrapped._sample_side_pose(1)

    def _step(self, a, label):
        a_clean = np.clip(np.asarray(a, dtype=np.float64), -1.0, 1.0)
        if self.xi > 0:
            a = a + self.noise_rng.normal(0.0, self.xi * np.asarray(self.cfg['noise']['scale']))
        a = np.clip(a, -1.0, 1.0)
        if self.before_step:
            self.before_step(a, a_clean, label)
        ob, _, _, _, info = self.env.step(a)
        self.t += 1
        if self.on_step:
            self.on_step(ob, info, a, label)
        self.ob, self.info = ob, info
        if self.env.unwrapped.bar_out_of_bounds():
            self.out_of_bounds = True               # bar left the workspace: stop here (no budget left)
            self.max_steps = self.t
        return ob, info

    def _oracle(self, overrides=None):
        cfg = copy.deepcopy(self.cfg)
        for k, v in (overrides or {}).items():
            sec, key = k.split('.')
            cfg[sec][key] = v
        orc = BarMarkovOracle(self.env, cfg, np.random.default_rng(self.rng.integers(1 << 31)))
        orc.reset(self.ob, self.info, len(self.log), self.t)   # pickup rows carry this segment's index
        self._sub += 1
        self.oracles.append(orc)
        return orc

    def _label(self, seg, orc=None, **kw):
        d = dict(segment=seg, segment_idx=len(self.log), oracle_phase=getattr(orc, 'phase', 0) if orc else 0,
                 grasp_s=float(orc.s) if orc is not None else np.nan, c_est=np.nan if self.c_est is None else self.c_est,
                 pickup_idx=self._global_pickup_idx(orc))
        d.update(kw)
        return d

    def _global_pickup_idx(self, orc):
        """Index of orc's current pickup attempt among all attempts of the episode (-1: none)."""
        if orc is None or not orc.pickups:
            return -1
        k = next(i for i, o in enumerate(self.oracles) if o is orc)
        return sum(len(o.pickups) for o in self.oracles[:k]) + orc.pickup_idx

    def _budget_left(self):
        return self.max_steps - self.t

    def _grasp_rule(self, orc, rule, s=None):
        """Set orc's grasp point by the segment's rule (or to s, already drawn by it) and remember the rule:
        ('fixed', s) or ('range', lo, hi)."""
        self._rule, self._rule_attempt = rule, orc.pickup_idx
        if s is None:
            s = rule[1] if rule[0] == 'fixed' else self._draw_s(rule[1], rule[2])
        orc.set_grasp_point(float(s))

    def _keep_rule(self, orc):
        """Before each oracle step while approaching: a new attempt (after a slip or miss) gets its grasp point from the
        segment's rule again (not the oracle's own draw), and a 'range' grasp point that is (or became, e.g. the bar
        was still rolling when it was drawn) out of the arm's reach is redrawn."""
        rule = getattr(self, '_rule', None)
        if rule is None or orc._mode != 'approach':
            return
        if orc.pickup_idx != self._rule_attempt:
            self._grasp_rule(orc, rule)
        elif rule[0] == 'range' and not self._reachable(orc.s):
            orc.set_grasp_point(self._draw_s(rule[1], rule[2]))

    def _run_oracle(self, seg, orc, budget, stop=None):
        t0 = self.t
        while not orc.done and self.t - t0 < budget and self._budget_left() > 0:
            if stop is not None and stop(orc):
                break
            self._keep_rule(orc)
            a = orc.select_action(self.ob, self.info, self.t)
            self._step(a, self._label(seg, orc))
        self._rule = None
        if seg in self.LIFTS and any(p['t_grasp'] >= 0 for p in orc.pickups):
            self._lifted = True
        return orc

    # ------------------------------------------------------------------ segments
    def probe(self, s, lift=0.10, s_range=None):
        """Pick up at s, lift by `lift` m, hold, put back down where it was, let go. Returns the held tilt (deg)."""
        e = self.env.unwrapped
        pos, yaw = self.info['privileged/block_0_pos'].copy(), float(self.info['privileged/block_0_yaw'][0])
        self._set_target(pos, yaw)                       # 'put it back where it was'
        orc = self._oracle({'oracle.slot_oracle': False})
        self._grasp_rule(orc, ('fixed', float(s)) if s_range is None else ('range', *s_range), s=s)
        t0 = self.t
        while orc._mode != 'carry' and not orc.done and self.t - t0 < 300 and self._budget_left() > 0:
            self._keep_rule(orc)
            self._step(orc.select_action(self.ob, self.info, self.t), self._label('probe', orc))
        self._rule = None
        s = orc.s                                        # the grasp point actually used (may have been redrawn)
        out = dict(segment='probe', s=float(s), lift=float(lift), t_start=t0, grasped=orc._mode == 'carry')
        if orc._mode != 'carry':
            out['t_end'] = self.t
            return out
        z0 = float(self.info['proprio/effector_pos'][2])
        e0 = self.info['proprio/effector_pos'][:2].copy()
        integ, ixy, tilts = 0.0, np.zeros(2), []
        o = orc._o
        stages = [('lift', 100), ('hold', 40), ('lower', 140), ('open', 30), ('up', 20)]
        lost, slipped = 0, False
        for stage, n in stages:
            if slipped:
                break
            for k in range(n):
                if self._budget_left() <= 0:
                    break
                ef = self.info['proprio/effector_pos']
                a = np.zeros(5)
                exy = e0 - ef[:2]
                ixy = np.clip(ixy + exy, -0.05, 0.05)
                a[:2] = np.clip(exy * 15 + ixy * 2, -0.25, 0.25)
                if stage in ('lift', 'hold'):
                    err = z0 + lift - ef[2]
                    integ = np.clip(integ + err, -0.05, 0.05)
                    a[2], a[4] = np.clip(err * 15 + integ * 2, -0.25, 0.25), 1.0
                    held = bool(self.info['privileged/pad_contact_left'][0] and self.info['privileged/pad_contact_right'][0])
                    lost = 0 if held else lost + 1
                    if lost >= o['slip_patience']:      # bar fell out of the hand: no tilt reading from this probe
                        slipped = True
                        break
                    if stage == 'hold' and k >= n - 10 and held:
                        tilts.append(float(np.degrees(self.info['privileged/bar_tilt'][0])))
                elif stage == 'lower':
                    err = z0 + 0.002 - ef[2]
                    a[2], a[4] = np.clip(err * 15, -0.15, 0.15), 1.0
                elif stage == 'open':
                    a[4] = -o['release_rate']
                else:
                    a[2], a[4] = 0.3, -1.0
                self._step(a, self._label('probe', orc, probe_stage=stage))
                if stage == 'lift' and abs(z0 + lift - self.info['proprio/effector_pos'][2]) < 0.005:
                    break
                if stage == 'lower' and self.info['privileged/bar_floor_contact'][0] and abs(err) < 0.006:
                    break
                if stage == 'open' and self.info['proprio/gripper_opening'][0] < 0.1:
                    break
        orc._end_attempt(self.t, 'slipped' if slipped else 'placed' if not self.info['proprio/gripper_contact'][0] else 'timeout')
        out.update(t_end=self.t, slipped=slipped, tilt=np.nan if slipped or not tilts else float(np.mean(tilts)))
        self._lifted = True
        return out

    def _slot_blocks(self):
        """Footprints (from above) of the two slot end blocks: [(lo, hi), (lo, hi)] in xy."""
        sl, b = self.cfg['slot'], self.cfg['bar']
        hl, hw = b['half_length'], b['half_width']
        rout = hw + sl['end_clearance'] + sl['wedge_thickness']
        y_lo = sl['y'] - hw - sl['end_clearance'] - sl['stop_thickness']
        y_hi = sl['y'] + hw + sl['entry_extend']
        out = []
        for sx in (-1, 1):
            cxs = sl['x'] + sx * (hl - hw)
            xs = sorted([cxs, cxs + sx * rout])
            out.append((np.array([xs[0], y_lo]), np.array([xs[1], y_hi])))
        return out

    def _slot_zone(self):
        """The bar is not seated but touches the slot, or overlaps a slot block seen from above (it lies in or under a
        pocket): grasping it there jams against the pockets. A bar merely lying next to a block is graspable."""
        e, sl = self.env.unwrapped, self.cfg.get('slot') or {}
        if not sl.get('enabled') or e.bar_in_slot():
            return False
        if self.info['privileged/bar_slot_contact'][0]:
            return True
        p0, p1, r = e.bar_footprint()
        return any(_segment_rect_distance(p0, p1, lo, hi) < r for lo, hi in self._slot_blocks())

    def _push_plan(self):
        """Where to put the open gripper and which way to push (unit xy vector, always with a +y part).

        The gripper is open and turned so its two fingertips (~10 cm apart) sit side by side across the push direction:
        two contacts keep the bar from spinning while it is pushed. A bar within 60 deg of the x axis is pushed sideways,
        along its normal, from its -y side, at the reachable spot nearest its middle; a bar closer to the y axis is
        pushed along its length from behind its -y end. If the preferred way has no valid spot, the other is tried. The
        gripper and both fingertips must be clear of the bar and the slot blocks, and inside the arm's workspace.
        None if there is no such spot."""
        e, hw = self.env.unwrapped, self.cfg['bar']['half_width']
        ws_lo, ws_hi = e._workspace_bounds[:, :2] + np.array([[0.01, 0.01], [-0.01, -0.01]])
        q, ax = self.info['privileged/block_0_pos'], e.bar_axis_world()
        p0, p1, r = e.bar_footprint()
        a2 = ax[:2] / max(np.linalg.norm(ax[:2]), 1e-9)
        n = np.array([-a2[1], a2[0]])
        n = n if n[1] >= 0 else -n                                              # bar normal pointing +y
        half = 0.5 * np.linalg.norm(p1 - p0)
        along = a2 if a2[1] > 0 else -a2
        sideways = (n, [q + sv * ax for sv in sorted(np.linspace(-half, half, 37), key=abs)], r + 0.025)
        lengthwise = (along, [min((q - half * ax, q + half * ax), key=lambda p: p[1])], r + hw + 0.025)
        for d, cands, dist in ((sideways, lengthwise) if n[1] >= 0.5 else (lengthwise, sideways)):
            f = np.array([-d[1], d[0]])                                         # fingertip line, across the push
            for p in cands:
                b = p[:2] - d * dist
                pts = (b, b + 0.05 * f, b - 0.05 * f)                           # gripper centre and both fingertips
                if any(_segment_rect_distance(p0, p1, x, x) < r + 0.012 for x in pts):
                    continue                                                    # would land on the bar
                if any(_segment_rect_distance(x, x, lo - 0.015, hi + 0.015) == 0 for x in pts for lo, hi in self._slot_blocks()):
                    continue                                                    # on / against a slot block
                if np.all(b >= ws_lo) and np.all(b <= ws_hi):
                    # fingertips sit ~14 mm below the gripper reference point: push_dz 0.012 puts them at the bar's axis
                    # height; lower pushes lower on the bar
                    return np.array([b[0], b[1], max(p[2] + self._push_dz(), 0.03)]), d
        return None

    def _push_dz(self):
        """Gripper reference height above the bar's axis while pushing (play.push_dz; fingertips ~14 mm lower)."""
        return float((self.cfg.get('play') or {}).get('push_dz', 0.012))

    def _push_yaw(self, d, eff_yaw):
        """Effector yaw that puts the fingertip line across the push direction d (finger line = yaw + 90 deg)."""
        base = float(np.arctan2(d[1], d[0]))
        cands = [base + k * np.pi for k in (-2, -1, 0, 1, 2)]
        cands = [c for c in cands if -np.pi + 0.15 <= c <= np.pi - 0.15] or [base]
        return min(cands, key=lambda c: abs(c - eff_yaw))

    def _stuck(self):
        return self.env.unwrapped.bar_in_slot() or self._slot_zone()

    def clear(self, budget=400, margin=0.01, max_travel=0.30, passes=3, seg='clear', push_speed=0.3):
        """Non-prehensile sweep (no grasp, see _push_plan): up above the pockets while turning the open gripper, over the
        push spot, down to bar height, push until the bar is `margin` clear of the slot blocks and not touching the slot
        (or it stops moving, or after `max_travel`), then up. If the bar is still seated or stuck (e.g. one end dropped
        off a block and the push passed over it), plan again from its new pose: up to `passes` pushes.
        Also the slide-out (seg='slide_out'): a seated bar is pushed straight out of the pockets along +y from behind
        its middle, between the two blocks; it drops off the blocks onto the floor in front."""
        e, sl, hw = self.env.unwrapped, self.cfg['slot'], self.cfg['bar']['half_width']
        z_clear = sl['height'] + 2 * hw + sl['wedge_thickness'] + 0.06          # above the pocket roofs
        t0 = self.t
        for _ in range(passes):
            plan = self._push_plan()
            if plan is None or self._budget_left() <= 0 or self.t - t0 >= budget:
                break                                                           # nowhere to push from: leave it
            tgt, d = plan
            stage, integ, best, stall, start, over = 'raise', np.zeros(3), -np.inf, 0, None, False
            while self._budget_left() > 0 and self.t - t0 < budget:
                ef, eyaw = self.info['proprio/effector_pos'], float(self.info['proprio/effector_yaw'][0])
                a = np.zeros(5)
                if stage == 'raise':
                    dest = np.array([ef[0], ef[1], z_clear])
                    nxt = 'over' if ef[2] > z_clear - 0.01 else stage
                elif stage == 'over':
                    tgt, d = self._push_plan() or (tgt, d)                      # the bar may still be settling
                    dest = np.array([tgt[0], tgt[1], z_clear])
                    yaw_ok = abs(self._push_yaw(d, eyaw) - eyaw) < np.radians(20)       # finish turning on the way down
                    nxt = 'down' if np.linalg.norm(dest[:2] - ef[:2]) < 0.01 and yaw_ok else stage
                elif stage == 'down':
                    dest = tgt
                    yaw_ok = abs(self._push_yaw(d, eyaw) - eyaw) < np.radians(5)
                    nxt = 'push' if abs(ef[2] - tgt[2]) < 0.006 and yaw_ok else stage
                elif stage == 'push':
                    start = ef[:2].copy() if start is None else start
                    p0, p1, r = e.bar_footprint()
                    prog = float((0.5 * (p0 + p1)) @ d)                          # bar progress along the push
                    best, stall = (prog, 0) if prog > best + 0.001 else (best, stall + 1)
                    # The target follows the bar: the point of its axis nearest the gripper (it may have dropped or
                    # turned), a few mm into its back surface, fingertips at its axis height there. The gripper can
                    # then never get ahead of the bar; if it ends up over it (the bar dropped below the fingertips),
                    # it lifts, steps back behind it and comes down again.
                    q, ax = self.info['privileged/block_0_pos'], e.bar_axis_world()
                    sv = float(np.clip((ef[:2] - q[:2]) @ ax[:2] / max(ax[:2] @ ax[:2], 1e-9), -0.16, 0.16))
                    bp = q + sv * ax
                    back = float(bp[:2] @ d) - r                                 # bar's back surface along d
                    line = start + (float(start @ d) * -1) * d                   # push line through start
                    over = float(ef[:2] @ d) > back - 0.004 and ef[2] - 0.014 > bp[2] + 0.004
                    if over:                                                    # above the bar: lift, step back
                        dest = np.array([*(line + (back - 0.03) * d), bp[2] + r + 0.04])
                    else:                                                       # hold the line, push at constant speed
                        dest = np.array([*(line + float(ef[:2] @ d) * d), max(bp[2] + self._push_dz(), 0.03)])
                    clear_of = all(_segment_rect_distance(p0, p1, lo, hi) > r + margin for lo, hi in self._slot_blocks())
                    done = clear_of and not self.info['privileged/bar_slot_contact'][0]
                    travel = float((ef[:2] - start) @ d)
                    nxt = 'up' if done or stall >= 40 or travel > max_travel else stage
                else:
                    # up: just clear of the bar if it is out (the next segment starts with its own move), else all the
                    # way above the pockets for another pass
                    z_up = z_clear if self._stuck() else min(z_clear, self.info['privileged/block_0_pos'][2] + hw + 0.06)
                    dest = np.array([ef[0], ef[1], z_up])
                    nxt = None if ef[2] > z_up - 0.01 else stage
                err = dest - ef
                integ = np.clip(integ + err, -0.05, 0.05)
                a[:3] = np.clip(err * 15 + integ * 2, -0.3, 0.3)
                if stage == 'push' and not over:
                    a[:2] += push_speed * d                                     # commanded push speed along d
                a[3] = np.clip((self._push_yaw(d, eyaw) - eyaw) * 3.0, -0.8, 0.8)
                a[4] = -1.0                                                     # open: two fingertips side by side, no grasp
                self._step(a, self._label(seg, None, probe_stage=stage))
                if nxt != stage:
                    integ = np.zeros(3)
                    if nxt is None:
                        break
                    stage = nxt
            if not self._stuck():
                break
        return dict(segment=seg, t_start=t0, t_end=self.t, out_of_slot=not self._stuck())

    def _tidy(self):
        """Leave the bar on the free floor: push a seated bar out (slide_out), sweep a stuck one out (clear)."""
        if self._budget_left() <= 0:
            return
        if self.env.unwrapped.bar_in_slot():
            self._slide_out()
        elif self._slot_zone():
            self.log.append(self.clear())

    def estimate_com(self):
        """COM from the probes: tilt ~ k (s - c)  =>  c = -intercept / slope (visible quantities only)."""
        pts = [(s, tl) for s, tl in self._probes if np.isfinite(tl)]
        if len(pts) >= 2 and len({round(s, 3) for s, _ in pts}) >= 2:
            k, b = np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)
            if k > 1e-3:
                return float(np.clip(-b / k, -0.14, 0.14))
        if pts:                                          # fallback: the probe that stayed most level
            return float(min(pts, key=lambda p: abs(p[1]))[0])
        return None                                      # no probe gave a reading (all slipped)

    LIFTS = ('probe', 'free', 'smash')

    def sample_plan(self):
        """Rule (user): a slide-in (pair) only comes after the bar has been lifted at least once in the episode. A pair
        drawn before the first lifting unit moves to just after it; with no lifting unit (and no lift yet) it is
        dropped."""
        r = self.rng
        units = ['probe'] * int(r.integers(0, self.n_probe_max + 1)) + ['pair'] * int(r.random() < self.p_pair) \
            + ['smash'] * int(r.random() < self.p_smash) + ['free'] * int(r.binomial(self.n_free_max, self.p_free))
        if not units:
            units = [str(r.choice(['probe', 'pair', 'smash', 'free']))]
        units = [str(u) for u in r.permutation(units)]
        if 'pair' in units and not self._lifted:
            units.remove('pair')
            first = next((i for i, u in enumerate(units) if u in self.LIFTS), None)
            if first is not None:
                units.insert(first + 1, 'pair')
        return units or [str(r.choice(self.LIFTS))]

    def _com_for_grasp(self):
        """COM to aim at: the probe estimate if >= 2 probes so far, else the privileged (true) COM."""
        est = self.estimate_com() if len(self._probes) >= 2 else None
        if est is not None:
            self.c_est = est
            return est, 'estimate'
        return float(self.env.unwrapped.mass.com_offset), 'privileged'

    def _reachable(self, s):
        """Grasp point s (along the bar, from its middle) lies inside the arm's workspace (as the oracle's own draws)."""
        e = self.env.unwrapped
        lo, hi = e._workspace_bounds[:, :2] + np.array([[0.01, 0.01], [-0.01, -0.01]])
        # Near the arm's base the gripper cannot get down to the floor (measured: at y = -0.15 not below x ~0.31; at
        # x 0.27 with the wrist turned it stalled 10 cm up and every grasp missed).
        lo = np.array([max(lo[0], 0.31), lo[1]])
        g = self.info['privileged/block_0_pos'][:2] + s * e.bar_axis_world()[:2]
        return bool(np.all(g >= lo) and np.all(g <= hi))

    def _draw_s(self, lo, hi, tries=50):
        """Uniform grasp point on [lo, hi], redrawn while it is out of the arm's reach. If none of the tries is
        reachable, a uniform draw from the reachable points of [lo, hi] (fine grid), then of the straight part of the
        bar (+-0.16; grasps on the rounded tips do not hold). If none can be reached, the episode ends (`unreachable`,
        like out of bounds)."""
        for _ in range(tries):
            s = float(self.rng.uniform(lo, hi))
            if self._reachable(s):
                return s
        for a, b in ((lo, hi), (-0.16, 0.16)):
            ok = [float(v) for v in np.linspace(a, b, 81) if self._reachable(v)]
            if ok:
                return float(self.rng.choice(ok))
        self.unreachable = True
        self.max_steps = self.t                        # no budget left: run() stops before the next segment
        return 0.0

    def _lim(self):
        return self.cfg['oracle'].get('slot_seated_grasp_s_max', 0.14)

    def _slide_out(self):
        """Seated bar out of the slot without grasping it (user): the closed gripper pushes it out along +y from behind
        its middle, between the two blocks, at pocket height; it drops off the blocks onto the floor in front."""
        self.log.append(self.clear(seg='slide_out'))

    def _units(self, plan):
        """The given plan, or sampled plans one after another until the step budget is used up."""
        if plan is not None:
            yield from plan
            return
        while True:
            units = self.sample_plan()
            self.plans.append(units)
            yield from units
            if not self.refill or self._budget_left() <= 0:
                return

    def run(self, ob, info, plan=None):
        self.ob, self.info = ob, info
        self.plans = [list(plan)] if plan is not None else []
        self.plan = []                                  # units actually started, in order
        for unit in self._units(plan):
            if self._budget_left() <= 0:
                break
            self.plan.append(str(unit))
            if self._slot_zone():                       # left at the slot (not seated): grasping there jams, sweep first
                self.log.append(self.clear())
                if self._budget_left() <= 0:
                    break
            if unit == 'probe':
                k = sum(1 for g in self.log if g['segment'] == 'probe')
                lift = float(self.probe_lift) if self.probe_lift is not None else float(self.rng.uniform(*self.lift_range))
                if self.probe_s is not None and k < len(self.probe_s):
                    r = self.probe(float(self.probe_s[k]), lift=lift)             # fixed grasp point, retries too
                else:
                    r = self.probe(self._draw_s(-0.12, 0.12), lift=lift, s_range=(-0.12, 0.12))
                self.log.append(r)
                if r.get('grasped'):
                    self._probes.append((r['s'], r.get('tilt', np.nan)))
            elif unit == 'pair':
                if not self._lifted:                        # rule: never slide in before the bar was lifted once
                    self.plan[-1] = 'pair(skipped: no lift yet)'
                    continue
                c, src = self._com_for_grasp()
                self._set_target(*self._slot_target())
                orc = self._oracle()
                self._grasp_rule(orc, ('fixed', c))           # every attempt (retries too) at the COM
                t0 = self.t
                self._run_oracle('slide_in', orc, 700)
                self.log.append(dict(segment='slide_in', s=c, s_last=orc.s, com_source=src, t_start=t0, t_end=self.t,
                                     seated=bool(self.env.unwrapped.bar_in_slot())))
                self._tidy()                                # seated: slide it out; stuck at the slot: sweep it out
            elif unit == 'smash':
                # Grip-height smash: grasp at a random point, gripper to pocket height, straight in.
                src = 'random'
                self._set_target(*self._slot_target())
                orc = self._oracle({'oracle.slot_smash_grip': True})
                k = sum(1 for g in self.log if g['segment'] == 'smash')
                if self.smash_s is not None and k < len(self.smash_s):
                    self._grasp_rule(orc, ('fixed', float(self.smash_s[k])))     # fixed grasp point, retries too
                else:
                    self._grasp_rule(orc, ('range', -0.16, 0.16))
                s = orc.s
                t0 = self.t
                pushed = {}

                def one_push(o):
                    # One push per smash: once the push (phase 14) has happened, stop when a new attempt would start
                    # (the oracle would grab the jammed bar and smash again).
                    if getattr(o, 'phase', 0) == 14 and 'idx' not in pushed:
                        pushed['idx'] = o.pickup_idx
                    return 'idx' in pushed and o.pickup_idx > pushed['idx']
                self._run_oracle('smash', orc, 600, stop=one_push)
                self.log.append(dict(segment='smash', s=float(s), s_last=orc.s, com_source=src, t_start=t0, t_end=self.t,
                                     seated=bool(self.env.unwrapped.bar_in_slot()),
                                     touching_slot=bool(self.info['privileged/bar_slot_contact'][0])))
                self._tidy()                                # leave the bar on the free floor for what comes next
            elif unit == 'free':
                self._set_target(*self._floor_target())
                orc = self._oracle({'oracle.carry_max_action': self.free_max_action} if self.free_max_action else None)
                on_slot = bool(self.info['privileged/bar_slot_contact'][0])
                lim = self._lim() if on_slot else 0.16
                self._grasp_rule(orc, ('range', -lim, lim))
                t0 = self.t
                self._run_oracle('free', orc, 500)
                self.log.append(dict(segment='free', s=orc.s, t_start=t0, t_end=self.t,
                                     placed=orc.pickups[-1]['outcome'] == 'placed'))
        for orc in self.oracles:                        # close attempts cut off by the step budget
            if orc.pickups and orc.pickups[-1]['outcome'] == '':
                orc._end_attempt(self.t, 'timeout')
        if len(self._probes) >= 2:
            self.c_est = self.estimate_com()
        return self.log
