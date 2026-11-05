"""OGBar data generation: closed-loop noisy pick-and-place of the hidden-mass bar, all privileged state recorded.

One episode = `episode_length` env steps (20 Hz) with one hidden mass split (W_L, W_R), chaining pick-and-place
sub-tasks to random targets. Every pickup attempt grasps at a fresh offset s along the bar (ogbar.oracle).
Action noise follows OGBench's 'noisy' datasets (generate_manipspace.py): per-episode xi ~ U(0, noise_max) and
N(0, xi * noise.scale) per step (OGBench: scale [1, 1, 1, 3, 10]), plus optional uniform random actions.

Frame t holds the observation *before* action t (OGBench convention).
"""
import json
import os
import subprocess
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from ogbar.envs.bar_env import BarEnv
from ogbar.oracle import OUTCOMES, PHASE_NAMES, BarMarkovOracle
from ogbar.play import PROBE_STAGES, SEGMENTS, PlayEpisode
from ogbar.recorder import OgbarRecorder

REPO = Path(__file__).resolve().parents[2]
_ENV = {}   # one env per worker process, keyed by config


def get_env(cfg):
    key = json.dumps(cfg, sort_keys=True)
    if key not in _ENV:
        _ENV.clear()
        _ENV[key] = BarEnv(
            cfg, ob_type='pixels', width=cfg['resolution'], height=cfg['resolution'], visualize_info=False,
            terminate_at_goal=False, pixel_transparent_arm=cfg['env']['pixel_transparent_arm'],
            assets_dir=cfg['env']['assets_dir'],
        )
    return _ENV[key]


def close_envs():
    for env in _ENV.values():
        env.close()
    _ENV.clear()


def episode_seed(master_seed, index, retry=0):
    return int(np.random.SeedSequence([int(master_seed), int(index), int(retry)]).generate_state(1)[0] >> 1)


# ---------------------------------------------------------------------- debug overlay

def compose_debug_frame(dbg, ob, lines):
    """Debug render (left), the clean observation upscaled (right), text overlay on the left panel."""
    res = dbg.shape[0]
    obs_big = np.asarray(Image.fromarray(ob).resize((res, res), Image.NEAREST))
    left = Image.fromarray(dbg)
    draw = ImageDraw.Draw(left)
    draw.rectangle([0, 0, res, 12 * len(lines) + 4], fill=(0, 0, 0))
    for i, line in enumerate(lines):
        draw.text((4, 2 + 12 * i), line, fill=(255, 255, 255))
    right = Image.fromarray(obs_big)
    ImageDraw.Draw(right).text((4, 2), 'model input (64x64)', fill=(255, 255, 255))
    return np.concatenate([np.asarray(left), np.asarray(right)], axis=1)


# ---------------------------------------------------------------------- one episode

class Rejected(Exception):
    pass


def run_episode(cfg, seed, w_right=None, debug=False):
    env = get_env(cfg)
    T = cfg['episode_length']
    hl = cfg['health']
    nz = cfg['noise']
    options = None if w_right is None else {'w_right': float(w_right)}
    ob, info = env.reset(seed=seed, options=options)
    oracle_rng = np.random.default_rng([seed, 1])
    noise_rng = np.random.default_rng([seed, 2])
    xi = float(noise_rng.uniform(0.0, nz['noise_max']))
    mass = env.mass

    orc = BarMarkovOracle(env, cfg, oracle_rng)
    subtask = 0
    orc.reset(ob, info, subtask, 0)
    sub_rows = []
    sub_start = 0

    F = {k: [] for k in (
        'rgb', 'state', 'action', 'action_clean', 'random_action', 'qpos', 'qvel', 'ctrl',
        'bar_pos', 'bar_quat', 'bar_com_world', 'bar_linvel', 'bar_com_linvel', 'bar_angvel', 'bar_tilt', 'bar_yaw_mod_pi',
        'bar_in_hand_pose', 'effector_pos', 'effector_yaw', 'gripper_opening', 'gripper_contact',
        'pad_contact_l', 'pad_contact_r', 'bar_floor_contact', 'bar_wall_contact', 'bar_platform_contact', 'bar_ramp_contact',
        'bar_fence_contact', 'bar_cradle_contact', 'bar_in_cradle', 'bar_side', 'target_side',
        'bar_on_platform', 'target_on_platform', 'grasped', 'oracle_phase',
        'subtask_idx', 'pickup_idx', 'grasp_offset_s', 'grasp_point_world', 'target_pos', 'target_yaw', 'success')}
    debug_frames = []
    upright = 0
    out_of_bounds = False

    def close_subtask(t_end):
        sub_rows.append(dict(subtask_idx=subtask, t_start=sub_start, t_end=t_end,
                             target_pos=info['privileged/target_block_pos'].copy(),
                             target_yaw=float(info['privileged/target_block_yaw'][0]),
                             n_pickups=sum(1 for p in orc.pickups if p['subtask_idx'] == subtask),
                             target_on_platform=bool(info['privileged/target_on_platform'][0]),
                             target_side=int(info['privileged/target_side'][0]),
                             success=bool(env._compute_success())))

    for t in range(T):
        a_clean = orc.select_action(ob, info, t)
        if noise_rng.random() < nz['p_random_action']:
            a, rand = noise_rng.uniform(-1, 1, 5), True
        else:
            a, rand = a_clean + noise_rng.normal(0.0, xi * np.asarray(nz['scale'])), False
        a = np.clip(a, -1, 1)

        # ---- frame t (state before action t)
        F['rgb'].append(ob)
        F['state'].append(env.compute_state_observation())
        F['action'].append(a)
        F['action_clean'].append(a_clean)
        F['random_action'].append(rand)
        F['qpos'].append(env.data.qpos.copy())
        F['qvel'].append(env.data.qvel.copy())
        F['bar_pos'].append(info['privileged/block_0_pos'])
        F['bar_quat'].append(info['privileged/block_0_quat'])
        for k in ('bar_com_world', 'bar_linvel', 'bar_com_linvel', 'bar_angvel', 'bar_in_hand_pose'):
            F[k].append(info[f'privileged/{k}'])
        F['bar_tilt'].append(info['privileged/bar_tilt'][0])
        F['bar_yaw_mod_pi'].append(info['privileged/bar_yaw_mod_pi'][0])
        F['effector_pos'].append(info['proprio/effector_pos'])
        F['effector_yaw'].append(info['proprio/effector_yaw'][0])
        F['gripper_opening'].append(info['proprio/gripper_opening'][0])
        F['gripper_contact'].append(info['proprio/gripper_contact'][0])
        F['pad_contact_l'].append(info['privileged/pad_contact_left'][0])
        F['pad_contact_r'].append(info['privileged/pad_contact_right'][0])
        F['bar_floor_contact'].append(info['privileged/bar_floor_contact'][0])
        F['bar_wall_contact'].append(info['privileged/bar_wall_contact'][0])
        F['bar_platform_contact'].append(info['privileged/bar_platform_contact'][0])
        F['bar_ramp_contact'].append(info['privileged/bar_ramp_contact'][0])
        F['bar_fence_contact'].append(info['privileged/bar_fence_contact'][0])
        F['bar_cradle_contact'].append(info['privileged/bar_cradle_contact'][0])
        F['bar_in_cradle'].append(info['privileged/bar_in_cradle'][0])
        F['bar_side'].append(info['privileged/bar_side'][0])
        F['target_side'].append(info['privileged/target_side'][0])
        F['bar_on_platform'].append(info['privileged/bar_on_platform'][0])
        F['target_on_platform'].append(info['privileged/target_on_platform'][0])
        F['grasped'].append(orc.grasped)
        F['oracle_phase'].append(orc.phase)
        F['subtask_idx'].append(subtask)
        F['pickup_idx'].append(orc.pickup_idx)
        F['grasp_offset_s'].append(orc.s)
        F['grasp_point_world'].append(orc.grasp_point)
        F['target_pos'].append(info['privileged/target_block_pos'])
        F['target_yaw'].append(info['privileged/target_block_yaw'][0])
        F['success'].append(env._compute_success())

        if debug:
            env.set_debug_grasp(orc.s)
            last = orc.pickups[-1]
            tilt = np.degrees(info['privileged/bar_tilt'][0])
            lines = [
                f'W_L={mass.w_left*1000:.0f}g  W_R={mass.w_right*1000:.0f}g  c={mass.com_offset*100:+.1f}cm',
                f's={orc.s*100:+.1f}cm  s-c={(orc.s-mass.com_offset)*100:+.1f}cm  tilt={tilt:+.1f}deg',
                f'phase {orc.phase}: {PHASE_NAMES[orc.phase]}',
                f'subtask {subtask}  pickup {orc.pickup_idx} {last["outcome"] or "(in progress)"}',
                f't={t}',
            ]
            debug_frames.append(compose_debug_frame(env.render_debug(), ob, lines))

        # ---- health checks on the current state
        speed = np.linalg.norm(info['privileged/bar_com_linvel'])
        if not np.isfinite(speed) or speed > hl['max_speed']:
            raise Rejected(f'bar centre-of-mass speed {speed:.2f} m/s at t={t}')
        if info['privileged/block_0_pos'][2] < hl['min_z']:
            raise Rejected(f'bar below table at t={t}')
        upright = upright + 1 if (abs(info['privileged/bar_tilt'][0]) > np.radians(60) and not orc.grasped) else 0
        if upright > hl['max_upright_steps']:
            raise Rejected(f'bar standing on end at t={t}')

        # ---- step
        ob, _, _, _, info = env.step(a)
        F['ctrl'].append(info['control'])
        if env.bar_out_of_bounds():
            # The bar left the workspace (there are no walls): end the episode here. Frame t is its last frame.
            out_of_bounds = True
            T = t + 1
            break
        if orc.done and t + 1 < T:
            close_subtask(t + 1)
            subtask += 1
            sub_start = t + 1
            ob, info = env.set_new_target()
            orc.reset(ob, info, subtask, t + 1)
    close_subtask(T)
    orc.finish_episode(T)

    frames = {
        'rgb': np.asarray(F['rgb'], dtype=np.uint8),
        **{k: np.asarray(F[k], dtype=bool) for k in (
            'random_action', 'pad_contact_l', 'pad_contact_r', 'bar_floor_contact', 'bar_wall_contact', 'grasped',
            'success', 'bar_platform_contact', 'bar_ramp_contact', 'bar_on_platform', 'target_on_platform',
            'bar_fence_contact', 'bar_cradle_contact', 'bar_in_cradle')},
        'bar_side': np.asarray(F['bar_side'], dtype=np.int8),        # +1 / -1: whole bar on that side; 0: straddling
        'target_side': np.asarray(F['target_side'], dtype=np.int8),
        'oracle_phase': np.asarray(F['oracle_phase'], dtype=np.uint8),
        'subtask_idx': np.asarray(F['subtask_idx'], dtype=np.int32),
        'pickup_idx': np.asarray(F['pickup_idx'], dtype=np.int32),
    }
    for k, v in F.items():
        if k not in frames:
            frames[k] = np.asarray(v, dtype=np.float32)
    frames['terminal'] = np.zeros(T, dtype=bool)
    frames['terminal'][-1] = out_of_bounds       # OGBench-style terminal: True only when the bar left the workspace
    outcome_code = {o: i for i, o in enumerate(OUTCOMES)}
    pickups = {
        'subtask_idx': [p['subtask_idx'] for p in orc.pickups],
        't_start': [p['t_start'] for p in orc.pickups],
        't_grasp': [p['t_grasp'] for p in orc.pickups],
        't_end': [p['t_end'] for p in orc.pickups],
        's': np.asarray([p['s'] for p in orc.pickups], dtype=np.float32),
        'outcome': np.asarray([outcome_code[p['outcome']] for p in orc.pickups], dtype=np.uint8),
        'max_tilt': np.asarray([p['max_tilt'] for p in orc.pickups], dtype=np.float32),
    }
    subtasks = {k: [r[k] for r in sub_rows] for k in sub_rows[0]}
    subtasks['target_pos'] = np.asarray(subtasks['target_pos'], dtype=np.float32)
    episode = {
        'w_left': np.float32(mass.w_left), 'w_right': np.float32(mass.w_right),
        'com_offset_c': np.float32(mass.com_offset), 'inertia_diag': mass.inertia_diag.astype(np.float32),
        'noise_xi': np.float32(xi), 'n_subtasks': np.int32(len(sub_rows)),
        'n_pickups': np.int32(len(orc.pickups)),
        'n_drops': np.int32(sum(1 for p in orc.pickups if p['outcome'] == 'slipped')),
        'n_placed': np.int32(sum(1 for p in orc.pickups if p['outcome'] == 'placed')),
        'out_of_bounds': np.bool_(out_of_bounds),     # episode ended early: the bar left the workspace
        'length': np.int32(T),
    }
    summary = (f"subtasks={len(sub_rows)} pickups={int(episode['n_pickups'])} placed={int(episode['n_placed'])} "
               f"slipped={int(episode['n_drops'])}")
    return dict(frames=frames, episode=episode, tables={'subtask': subtasks, 'pickup': pickups},
                debug_frames=debug_frames, summary=summary)


# ---------------------------------------------------------------------- one play episode (ogbar.play)

def compose_play_debug_frame(dbg_front, dbg_side, rgb, rgb_front, lines):
    """Top: debug renders (front | side, masses drawn). Bottom: the two recorded 64 px views, upscaled. Header text."""
    res = dbg_front.shape[0]
    up = lambda im: np.asarray(Image.fromarray(im).resize((res, res), Image.NEAREST))
    img = np.concatenate([np.concatenate([dbg_front, dbg_side], 1), np.concatenate([up(rgb_front), up(rgb)], 1)], 0)
    head = Image.new('RGB', (2 * res, 12 * len(lines) + 4))
    draw = ImageDraw.Draw(head)
    for i, line in enumerate(lines):
        draw.text((4, 2 + 12 * i), line, fill=(255, 255, 0) if i == 0 else (255, 255, 255))
    img = Image.fromarray(np.concatenate([np.asarray(head), img], 0))
    draw = ImageDraw.Draw(img)
    y0 = 12 * len(lines) + 4
    draw.text((4, y0 + 2), 'debug: front (hidden masses drawn)', fill=(255, 255, 255))
    draw.text((res + 4, y0 + 2), 'debug: side', fill=(255, 255, 255))
    draw.text((4, y0 + res + 2), f'front view ({rgb_front.shape[0]}px)', fill=(255, 255, 255))
    draw.text((res + 4, y0 + res + 2), f'recorded rgb = world-model input ({rgb.shape[0]}px)', fill=(255, 255, 255))
    return np.asarray(img)


def run_play_episode(cfg, seed, w_right=None, debug=False, plan=None, bar_pose=None, noise_xi=None, max_steps=None,
                     probe_s=None, probe_lift=None, smash_s=None, restore_state=None):
    """One play episode (ogbar.play.PlayEpisode): fixed length cfg['episode_length'] (cut off mid-segment), shorter
    only if the bar leaves the workspace. `rgb` comes from env.pixel_camera (side view); `rgb_front` (front_pixels) only
    with env.record_front (user 2026-10-14: dropped by default, the world model uses the side view only).
    Eval episodes (scripts/generate_ogbar_eval.py) fix the plan, start pose (x, y, yaw), noise level, length cap and the
    probes' grasp points / lift; with an explicit plan the episode ends when the plan does."""
    env = get_env(cfg)
    T = cfg['episode_length'] if max_steps is None else int(max_steps)
    hl, nz, pc = cfg['health'], cfg['noise'], cfg['play']
    options = {k: v for k, v in (('w_right', None if w_right is None else float(w_right)), ('bar_pose', bar_pose))
               if v is not None} or None
    ob, info = env.reset(seed=seed, options=options)
    if restore_state is not None:
        # Branch from a recorded moment (counterfactual eval): simulator state copied from a stored frame. The arm
        # controller works from the current state plus the action; its small verticality integral is not recorded and
        # restarts at zero.
        u = env.unwrapped
        u._data.qpos[:] = restore_state['qpos']
        u._data.qvel[:] = restore_state['qvel']
        u._data.ctrl[:] = restore_state['ctrl']
        u._ik_tilt_corr = np.zeros(3)
        mujoco.mj_forward(u._model, u._data)
        u.pre_step()
        u.post_step()
        ob, info = u.compute_observation(), u.get_reset_info()
    noise_rng = np.random.default_rng([seed, 2])
    xi = float(noise_rng.uniform(0.0, nz['noise_max']))
    if noise_xi is not None:
        xi = float(noise_xi)
    mass = env.mass
    seg_code = {k: i for i, k in enumerate(SEGMENTS)}
    stage_code = {k: i for i, k in enumerate(PROBE_STAGES)}
    record_front = bool(cfg['env'].get('record_front', False))
    keys = ('rgb', 'rgb_front', 'state', 'action', 'action_clean', 'qpos', 'qvel', 'ctrl',
            'bar_pos', 'bar_quat', 'bar_com_world', 'bar_linvel', 'bar_com_linvel', 'bar_angvel', 'bar_tilt',
            'bar_yaw_mod_pi', 'bar_in_hand_pose', 'effector_pos', 'effector_yaw', 'gripper_opening', 'gripper_contact',
            'pad_contact_l', 'pad_contact_r', 'bar_floor_contact', 'bar_slot_contact', 'bar_in_slot', 'grasped',
            'segment_type', 'segment_idx', 'oracle_phase', 'probe_stage', 'pickup_idx', 'grasp_offset_s', 'com_estimate',
            'target_pos', 'target_yaw', 'success')
    F = {k: [] for k in keys}
    debug_frames = []
    play = None

    def before_step(a, a_clean, lab):
        """Frame t = the state before action t (play.ob / play.info)."""
        o, inf = play.ob, play.info
        F['rgb'].append(o)
        if record_front or debug:
            F['rgb_front'].append(env.render(camera='front_pixels'))
        F['state'].append(env.compute_state_observation())
        F['action'].append(a)
        F['action_clean'].append(a_clean)
        F['qpos'].append(env.data.qpos.copy())
        F['qvel'].append(env.data.qvel.copy())
        F['bar_pos'].append(inf['privileged/block_0_pos'])
        F['bar_quat'].append(inf['privileged/block_0_quat'])
        for k in ('bar_com_world', 'bar_linvel', 'bar_com_linvel', 'bar_angvel', 'bar_in_hand_pose'):
            F[k].append(inf[f'privileged/{k}'])
        F['bar_tilt'].append(inf['privileged/bar_tilt'][0])
        F['bar_yaw_mod_pi'].append(inf['privileged/bar_yaw_mod_pi'][0])
        F['effector_pos'].append(inf['proprio/effector_pos'])
        F['effector_yaw'].append(inf['proprio/effector_yaw'][0])
        F['gripper_opening'].append(inf['proprio/gripper_opening'][0])
        F['gripper_contact'].append(inf['proprio/gripper_contact'][0])
        F['pad_contact_l'].append(inf['privileged/pad_contact_left'][0])
        F['pad_contact_r'].append(inf['privileged/pad_contact_right'][0])
        F['bar_floor_contact'].append(inf['privileged/bar_floor_contact'][0])
        F['bar_slot_contact'].append(inf['privileged/bar_slot_contact'][0])
        F['bar_in_slot'].append(inf['privileged/bar_in_slot'][0])
        F['grasped'].append(bool(inf['privileged/pad_contact_left'][0] and inf['privileged/pad_contact_right'][0]))
        F['segment_type'].append(seg_code[lab['segment']])
        F['segment_idx'].append(lab['segment_idx'])
        F['oracle_phase'].append(lab['oracle_phase'])
        F['probe_stage'].append(stage_code[lab.get('probe_stage', '')])
        F['pickup_idx'].append(lab['pickup_idx'])
        F['grasp_offset_s'].append(lab['grasp_s'])
        F['com_estimate'].append(lab['c_est'])
        F['target_pos'].append(inf['privileged/target_block_pos'])
        F['target_yaw'].append(inf['privileged/target_block_yaw'][0])
        F['success'].append(inf['privileged/bar_in_slot'][0])

        if debug:
            t = len(F['rgb']) - 1
            env.set_debug_grasp(lab['grasp_s'] if np.isfinite(lab['grasp_s']) else None)
            ce = '' if not np.isfinite(lab['c_est']) else f"   COM estimate {lab['c_est']*100:+.1f} cm"
            stage = f" / {lab['probe_stage']}" if lab.get('probe_stage') else ''
            gs = f"{lab['grasp_s']*100:+.1f} cm" if np.isfinite(lab['grasp_s']) else '-'
            lines = [
                f"t={t:4d}  segment {lab['segment_idx'] + 1}: {lab['segment'].upper()}{stage}   "
                f"phase {lab['oracle_phase']}: {PHASE_NAMES.get(lab['oracle_phase'], '')}",
                f"W_L={mass.w_left*1000:.0f} g  W_R={mass.w_right*1000:.0f} g  true COM {mass.com_offset*100:+.1f} cm"
                f"   grasp s {gs}{ce}   noise xi {xi:.3f}",
                f"tilt {np.degrees(inf['privileged/bar_tilt'][0]):+5.1f} deg   seated {bool(inf['privileged/bar_in_slot'][0])}"
                f"   touching slot {bool(inf['privileged/bar_slot_contact'][0])}   plan so far: {' '.join(play.plan)}",
            ]
            debug_frames.append(compose_play_debug_frame(env.render_debug(), env.render_debug(camera=env._pixel_camera),
                                                         o, F['rgb_front'][-1], lines))

        speed = np.linalg.norm(inf['privileged/bar_com_linvel'])
        if not np.isfinite(speed) or speed > hl['max_speed']:
            raise Rejected(f'bar centre-of-mass speed {speed:.2f} m/s at t={len(F["rgb"]) - 1}')
        if inf['privileged/block_0_pos'][2] < hl['min_z']:
            raise Rejected(f'bar below table at t={len(F["rgb"]) - 1}')

    def on_step(o, inf, a, lab):
        F['ctrl'].append(inf['control'])

    play = PlayEpisode(env, cfg, np.random.default_rng([seed, 1]), noise_xi=xi, max_steps=T, on_step=on_step,
                       before_step=before_step, noise_rng=noise_rng, n_probe_max=pc['n_probe_max'],
                       p_pair=pc['p_pair'], p_smash=pc['p_smash'], n_free_max=pc['n_free_max'],
                       lift_range=tuple(pc['lift_range']), p_free=pc.get('p_free', 0.5),
                       free_max_action=pc.get('free_max_action'), probe_s=probe_s, probe_lift=probe_lift, smash_s=smash_s)
    log = play.run(ob, info, plan=plan)
    T = len(F['rgb'])
    if T == 0:
        # No point of the bar was reachable at the start (e.g. it starts right next to the arm's base): nothing to record.
        raise Rejected('no reachable point on the bar at the start')

    frames = {
        'rgb': np.asarray(F['rgb'], dtype=np.uint8),
        **{k: np.asarray(F[k], dtype=bool) for k in (
            'pad_contact_l', 'pad_contact_r', 'bar_floor_contact', 'bar_slot_contact', 'bar_in_slot', 'grasped',
            'success')},
        'segment_type': np.asarray(F['segment_type'], dtype=np.uint8),
        'probe_stage': np.asarray(F['probe_stage'], dtype=np.uint8),
        'oracle_phase': np.asarray(F['oracle_phase'], dtype=np.uint8),
        'segment_idx': np.asarray(F['segment_idx'], dtype=np.int32),
        'pickup_idx': np.asarray(F['pickup_idx'], dtype=np.int32),
    }
    if record_front:
        frames['rgb_front'] = np.asarray(F['rgb_front'], dtype=np.uint8)
    for k, v in F.items():
        if k not in frames and k != 'rgb_front':
            frames[k] = np.asarray(v, dtype=np.float32)
    frames['terminal'] = np.zeros(T, dtype=bool)
    frames['terminal'][-1] = play.out_of_bounds or play.unreachable

    outcome_code = {o: i for i, o in enumerate(OUTCOMES)}
    pk = [p for orc in play.oracles for p in orc.pickups]
    pickups = {
        'segment_idx': [p['subtask_idx'] for p in pk],     # the play segment (row of segment/) the attempt ran in
        't_start': [p['t_start'] for p in pk], 't_grasp': [p['t_grasp'] for p in pk], 't_end': [p['t_end'] for p in pk],
        's': np.asarray([p['s'] for p in pk], dtype=np.float32),
        'outcome': np.asarray([outcome_code[p['outcome'] or 'timeout'] for p in pk], dtype=np.uint8),
        'max_tilt': np.asarray([p['max_tilt'] for p in pk], dtype=np.float32),
    }
    nan = float('nan')
    segments = {
        'type': np.asarray([seg_code[g['segment']] for g in log], dtype=np.uint8),
        't_start': [g['t_start'] for g in log], 't_end': [g['t_end'] for g in log],
        's_planned': np.asarray([g.get('s', nan) for g in log], dtype=np.float32),      # first attempt's grasp point
        's_last': np.asarray([g.get('s_last', g.get('s', nan)) for g in log], dtype=np.float32),  # after slip retries
        'com_from_estimate': np.asarray([g.get('com_source') == 'estimate' for g in log], dtype=bool),
        'probe_lift': np.asarray([g.get('lift', nan) for g in log], dtype=np.float32),
        'probe_tilt_deg': np.asarray([g.get('tilt', nan) for g in log], dtype=np.float32),
        'seated': np.asarray([bool(g.get('seated', False)) for g in log], dtype=bool),
        'out_of_slot': np.asarray([bool(g.get('out_of_slot', False)) for g in log], dtype=bool),
        'placed': np.asarray([bool(g.get('placed', False)) for g in log], dtype=bool),
        'probe_slipped': np.asarray([bool(g.get('slipped', False)) for g in log], dtype=bool),
        'complete': np.asarray([g['t_end'] < T or (i + 1 < len(log)) for i, g in enumerate(log)], dtype=bool),
    }
    n_by = lambda typ, key: int(sum(1 for g in log if g['segment'] == typ and g.get(key)))
    episode = {
        'w_left': np.float32(mass.w_left), 'w_right': np.float32(mass.w_right),
        'com_offset_c': np.float32(mass.com_offset), 'inertia_diag': mass.inertia_diag.astype(np.float32),
        'noise_xi': np.float32(xi), 'n_segments': np.int32(len(log)),
        'n_probes': np.int32(n_by('probe', 'grasped')),
        'com_estimate': np.float32(nan if play.c_est is None else play.c_est),
        'n_slide_in_seated': np.int32(n_by('slide_in', 'seated')),
        'n_smash_seated': np.int32(n_by('smash', 'seated')),
        'n_pickups': np.int32(len(pk)),
        'out_of_bounds': np.bool_(play.out_of_bounds),
        'unreachable': np.bool_(play.unreachable),     # ended early: no point of the bar within the arm's reach
        'length': np.int32(T),
    }
    counts = {k: sum(1 for g in log if g['segment'] == k) for k in SEGMENTS}
    summary = (' '.join(f'{k}={v}' for k, v in counts.items() if v)
               + f" | slide_in seated {int(episode['n_slide_in_seated'])}/{counts['slide_in']}"
               + f" smash seated {int(episode['n_smash_seated'])}/{counts['smash']}"
               + ('' if play.c_est is None else f" | COM est {play.c_est*100:+.1f}cm") + f" xi={xi:.3f} T={T}"
               + (' UNREACHABLE' if play.unreachable else ''))
    return dict(frames=frames, episode=episode, tables={'segment': segments, 'pickup': pickups},
                debug_frames=debug_frames, summary=summary)


def run_episode_with_retries(job):
    """Worker entry. job = (cfg, master_seed, index, w_right, debug). Returns (seed, result, n_rejected, reasons)."""
    cfg, master_seed, index, w_right, debug = job
    reasons = []
    run = run_play_episode if (cfg.get('play') or {}).get('enabled') else run_episode
    for retry in range(cfg['health']['max_retries'] + 1):
        seed = episode_seed(master_seed, index, retry)
        try:
            return seed, run(cfg, seed, w_right=w_right, debug=debug), retry, reasons
        except Rejected as e:
            reasons.append(str(e))
    raise RuntimeError(f'episode {index}: {len(reasons)} rejections in a row: {reasons[-3:]}')


def _worker_init():
    os.environ.setdefault('MUJOCO_GL', 'egl')


def ogbench_commit():
    try:
        return subprocess.check_output(['git', '-C', str(REPO / 'submodules' / 'ogbench'), 'rev-parse', 'HEAD'],
                                       text=True).strip()
    except Exception:
        return 'unknown'


def generate(cfg, n, zarr_path, video_dir, append=False, w_right_values=None, n_debug=0, workers=1,
             video_prefix='debug', log=print):
    """Generate n episodes into zarr_path. w_right_values (optional) fixes each episode's split, in order."""
    import multiprocessing as mp

    rec = OgbarRecorder(zarr_path, cfg['resolution'], append=append)
    first = rec.n_episodes
    master_seed = cfg['seed']
    jobs = [(cfg, master_seed, first + i, None if w_right_values is None else w_right_values[i], i < n_debug)
            for i in range(n)]
    os.makedirs(video_dir, exist_ok=True)
    n_rejected, reasons_all = 0, []
    ctx = mp.get_context('spawn')
    pool = ctx.Pool(workers, initializer=_worker_init) if workers > 1 else None
    it = pool.imap(run_episode_with_retries, jobs) if pool else map(run_episode_with_retries, jobs)
    try:
        for i, (seed, res, retries, reasons) in enumerate(it):
            n_rejected += retries
            reasons_all += reasons
            rec.add_episode(res['frames'], res['episode'], res['tables'], seed)
            ep = res['episode']
            if res['debug_frames']:
                wl, wr = float(ep['w_left']) * 1000, float(ep['w_right']) * 1000
                name = f'{video_prefix}_{first + i}_WL{wl:.0f}_WR{wr:.0f}.mp4'
                imageio.mimwrite(os.path.join(video_dir, name), res['debug_frames'], fps=cfg['fps'], quality=8,
                                 macro_block_size=8)
            log(f"ep {first + i}: W_L={float(ep['w_left'])*1000:.0f}g W_R={float(ep['w_right'])*1000:.0f}g "
                f"c={float(ep['com_offset_c'])*100:+.1f}cm {res['summary']} "
                f"retries={retries}" + (f" OUT OF BOUNDS at step {int(ep['length'])}" if bool(ep['out_of_bounds']) else ''))
    finally:
        if pool:
            pool.close()
            pool.join()
        close_envs()
        if rec.n_episodes:
            finalize_store(rec, cfg, master_seed)                   # also after a crash: keep the store usable
    log(f'wrote {n} episodes to {zarr_path} ({rec.n_episodes} total); rejected+retried {n_rejected}')
    for r in reasons_all[:10]:
        log(f'  rejected: {r}')
    return rec


def finalize_store(rec, cfg, master_seed):
    """Write the store attributes (config, privileged key list, name lists, cameras)."""
    play_attrs = {}
    if (cfg.get('play') or {}).get('enabled'):
        play_attrs = {'play_segments': list(SEGMENTS), 'probe_stages': list(PROBE_STAGES),
                      'cameras': {'rgb': cfg['env'].get('pixel_camera', 'front_pixels'),
                                  **({'rgb_front': 'front_pixels'} if cfg['env'].get('record_front') else {})}}
    rec.finalize(cfg, master_seed, extra_attrs={
        'ogbench_source_commit': ogbench_commit(),
        'pickup_outcomes': list(OUTCOMES),
        'oracle_phases': {str(k): v for k, v in PHASE_NAMES.items()},
        **play_attrs,
    })


def check_debug_does_not_leak(cfg, seed, w_right):
    """The clean pixels must be byte-identical with and without debug rendering."""
    run = run_play_episode if (cfg.get('play') or {}).get('enabled') else run_episode
    a = run(cfg, seed, w_right=w_right, debug=True)['frames']
    close_envs()
    b = run(cfg, seed, w_right=w_right, debug=False)['frames']
    close_envs()
    return {k: bool(np.array_equal(a[k], b[k])) for k in ('rgb', 'qpos', 'action')}
