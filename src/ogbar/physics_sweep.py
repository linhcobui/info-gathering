"""Phase 0 physics sweep: how much does the bar tilt in the hand, as a function of the hidden mass split?

For each contact regime, W_R split and grasp offset s, the bar is placed flat at the workspace centre (yaw 0), the
oracle grasps it at s, lifts it to `lift_z` and holds it still. Steady tilt is the mean of the last `settle_steps`
of the hold; jitter is their std; slip is how far the grasp point moved along the bar; dropped means the pads lost
the bar. Gate (original design): tilt monotonic in the gravity torque M*g*(s - c), no drops, no jitter.
"""
import copy
import json
import os

import imageio.v2 as imageio
import mujoco
import numpy as np

from ogbar import lie
from ogbar.envs.bar_env import BarEnv
from ogbar.mass import sample_w_right
from ogbar.oracle import BarMarkovOracle

G = 9.81
SEQ_BLUE = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281']   # sequential ramp steps 250..650 (W_R)
CRITICAL = '#d03b3b'
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def regime_cfg(base, overrides):
    cfg = copy.deepcopy(base)
    cfg['physics'].update(overrides)
    cfg['noise']['noise_max'] = 0.0
    return cfg


def grasp_point_on_bar(info):
    """Where along the bar (local x) the pinch point currently is."""
    rel = info['proprio/effector_pos'] - info['privileged/block_0_pos']
    return float(rel @ info['privileged/bar_axis'])


def run_one(env, cfg, sw, w_right, s, seed, video=False):
    ob, info = env.reset(seed=seed, options={'w_right': w_right})
    env.data.joint('object_joint_0').qpos[:3] = (0.425, 0.0, cfg['bar']['half_width'])
    env.data.joint('object_joint_0').qpos[3:] = lie.SO3.from_z_radians(0.0).wxyz
    mujoco.mj_forward(env.model, env.data)
    info = env.compute_ob_info()

    orc = BarMarkovOracle(env, cfg, np.random.default_rng(seed))
    orc.reset(ob, info, 0, 0)
    orc._s = s                                     # force the grasp offset for this cell
    orc.pickups[-1]['s'] = s
    frames, tilts, grasp_pts, grasped = [], [], [], []
    t = 0
    while orc._mode != 'carry' and t < sw['max_grasp_steps']:
        a = orc.select_action(ob, info, t)
        env.set_debug_grasp(s)
        if video:
            frames.append(env.render_debug())
        ob, _, _, _, info = env.step(a)
        t += 1
    if orc._mode != 'carry':
        return dict(w_right=w_right, s=s, grasped=False)
    s_at_grasp = grasp_point_on_bar(info)
    lift_target = None
    for k in range(sw['lift_steps'] + sw['hold_steps']):
        eff = info['proprio/effector_pos']
        if lift_target is None:
            lift_target = np.array([eff[0], eff[1], sw['lift_z']])
        a = np.zeros(5)
        a[:3] = np.clip((lift_target - eff) * cfg['oracle']['gain_pos'], -1, 1)
        a[4] = 1.0
        if video:
            frames.append(env.render_debug())
        ob, _, _, _, info = env.step(a)
        if k >= sw['lift_steps']:
            tilts.append(float(info['privileged/bar_tilt'][0]))
            grasp_pts.append(grasp_point_on_bar(info))
            grasped.append(bool(info['privileged/pad_contact_left'][0] and info['privileged/pad_contact_right'][0]))
    last = np.array(tilts[-sw['settle_steps']:])
    return dict(
        w_right=w_right, s=s, c=env.mass.com_offset, grasped=True,
        torque=env.mass.total * G * (s - env.mass.com_offset),
        tilt=float(last.mean()), jitter=float(last.std()),
        slip=float(grasp_pts[-1] - s_at_grasp), dropped=not grasped[-1],
        frames=frames,
    )


def plot(results, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = list(results)
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4.0), sharey=True, facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    w_values = sorted({r['w_right'] for rs in results.values() for r in rs})
    for ax, name in zip(axes, names):
        ax.set_facecolor(SURFACE)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        for spine in ('left', 'bottom'):
            ax.spines[spine].set_color(INK2)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.axhline(0, color=INK2, linewidth=0.8)
        rs = [r for r in results[name] if r['grasped']]
        for i, w in enumerate(w_values):
            sel = sorted([r for r in rs if r['w_right'] == w and not r['dropped']], key=lambda r: r['torque'])
            if sel:
                ax.plot([r['torque'] for r in sel], [np.degrees(r['tilt']) for r in sel], '-o', color=SEQ_BLUE[i % 5],
                        linewidth=2, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5,
                        label=f'W_R = {w * 1000:.0f} g')
        drops = [r for r in rs if r['dropped']]
        if drops:
            ax.plot([r['torque'] for r in drops], [np.degrees(r['tilt']) for r in drops], 'x', color=CRITICAL,
                    markersize=9, markeredgewidth=2, label='dropped')
        n_fail = sum(1 for r in results[name] if not r['grasped'])
        title = name + (f'  ({n_fail} grasp failures)' if n_fail else '')
        ax.set_title(title, color=INK, fontsize=11, loc='left')
        ax.set_xlabel('gravity torque  M·g·(s − c)  [N·m]', color=INK2)
        ax.tick_params(colors=INK2)
    axes[0].set_ylabel('steady tilt in hand  [deg]\n(+ = W_R end up)', color=INK2)
    axes[-1].legend(frameon=False, labelcolor=INK2, fontsize=9, loc='best')
    fig.suptitle('OGBar physics sweep: tilt vs hidden-mass torque, by pad contact regime', color=INK, x=0.01,
                 ha='left', fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def run_sweep(cfg, sweep_cfg, out_json, out_fig, video_dir):
    results = {}
    for name, overrides in sweep_cfg['regimes'].items():
        rcfg = regime_cfg(cfg, overrides)
        env = BarEnv(rcfg, ob_type='states', visualize_info=False, terminate_at_goal=False)
        rs = []
        for i, w in enumerate(sweep_cfg['w_right_values']):
            for j, s in enumerate(sweep_cfg['s_values']):
                video = (w, s) == tuple(sweep_cfg['video_cell'])
                r = run_one(env, rcfg, sweep_cfg, w, s, seed=1000 * i + j, video=video)
                frames = r.pop('frames', [])
                if video and frames:
                    os.makedirs(video_dir, exist_ok=True)
                    imageio.mimwrite(os.path.join(video_dir, f'sweep_{name}.mp4'), frames, fps=20, quality=8)
                rs.append(r)
                if r['grasped']:
                    print(f"{name:12s} W_R={w*1000:3.0f}g s={s:+.2f} torque={r['torque']:+.3f} "
                          f"tilt={np.degrees(r['tilt']):+6.2f}deg jitter={np.degrees(r['jitter']):.3f} "
                          f"slip={r['slip']*100:+.2f}cm dropped={r['dropped']}", flush=True)
                else:
                    print(f'{name:12s} W_R={w*1000:3.0f}g s={s:+.2f} GRASP FAILED', flush=True)
        env.close()
        results[name] = rs
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=1)
    plot(results, out_fig)
    return results


# ---------------------------------------------------------------------- natural drop curves

def drop_trial(args):
    """One real pick-and-place sub-task (oracle, no action noise) with a forced first grasp offset.

    Returns the lever arm s - c and how the first pickup attempt ended (placed / slipped / missed / timeout).
    """
    cfg, seed = args
    rng = np.random.default_rng(seed)
    env = _drop_env(cfg)
    w_right = sample_w_right(cfg['mass'], rng)
    ob, info = env.reset(seed=int(seed), options={'w_right': w_right})
    s = float(rng.uniform(-cfg['grasp']['s_max'], cfg['grasp']['s_max']))
    orc = BarMarkovOracle(env, cfg, np.random.default_rng(seed + 1))
    orc.reset(ob, info, 0, 0)
    orc._s = s
    orc.pickups[-1]['s'] = s
    for t in range(cfg['drop']['max_steps']):
        ob, _, _, _, info = env.step(orc.select_action(ob, info, t))
        if orc.pickups[0]['outcome']:
            break
    p = orc.pickups[0]
    return dict(w_right=w_right, s=s, c=env.mass.com_offset, lever=s - env.mass.com_offset,
                outcome=p['outcome'] or 'timeout', grasped=p['t_grasp'] >= 0, max_tilt=p['max_tilt'])


_DROP_ENV = {}


def _drop_env(cfg):
    key = json.dumps(cfg['physics'], sort_keys=True)
    if key not in _DROP_ENV:
        _DROP_ENV.clear()
        _DROP_ENV[key] = BarEnv(cfg, ob_type='states', visualize_info=False, terminate_at_goal=False)
    return _DROP_ENV[key]


def run_drop_curves(cfg, drop_cfg, workers, log=print):
    """For each physics setting, P(first pickup drops | |s - c|) from n_trials random real pick-and-places."""
    import multiprocessing as mp

    jobs, names = [], []
    for name, overrides in drop_cfg['settings'].items():
        rcfg = regime_cfg(cfg, overrides)
        rcfg['drop'] = drop_cfg
        for i in range(drop_cfg['n_trials']):
            jobs.append((rcfg, 10_000 * i + 7))
            names.append(name)
    with mp.get_context('spawn').Pool(workers) as pool:
        res = pool.map(drop_trial, jobs, chunksize=4)
    out = {}
    for name, r in zip(names, res):
        out.setdefault(name, []).append(r)
    edges = np.asarray(drop_cfg['lever_bins_cm']) / 100
    log('P(drop | grasped) by |s - c| bin (cm): ' + '  '.join(f'{a*100:.0f}-{b*100:.0f}' for a, b in zip(edges[:-1], edges[1:]))
        + '   | grasped  placed  dropped  missed')
    for name, rs in out.items():
        g = [r for r in rs if r['grasped']]
        cells = []
        for a, b in zip(edges[:-1], edges[1:]):
            sel = [r for r in g if a <= abs(r['lever']) < b]
            cells.append(f"{np.mean([r['outcome'] == 'slipped' for r in sel]):.2f}({len(sel):2d})" if sel else '   -    ')
        n = {k: sum(r['outcome'] == k for r in rs) for k in ('placed', 'slipped', 'missed')}
        log(f'{name:28s} ' + ' '.join(cells) + f"   | {len(g):3d}/{len(rs)}  {n['placed']:3d}  {n['slipped']:3d}  {n['missed']:3d}")
    return out


def plot_drop_curves(results, bins_cm, path, highlight=None):
    """P(first pickup drops | grasped) against |s - c|, one line per physics setting."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cat = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
    edges = np.asarray(bins_cm, dtype=float)
    mids = (edges[:-1] + edges[1:]) / 2
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_color(INK2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for i, (name, rs) in enumerate(results.items()):
        g = [r for r in rs if r['grasped']]
        ys = []
        for a, b in zip(edges[:-1] / 100, edges[1:] / 100):
            sel = [r for r in g if a <= abs(r['lever']) < b]
            ys.append(np.mean([r['outcome'] == 'slipped' for r in sel]) if sel else np.nan)
        lw = 2.5 if name == highlight else 2
        ax.plot(mids, ys, '-o', color=cat[i % len(cat)], linewidth=lw, markersize=7, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=name + ('  (default)' if name == highlight else ''))
    ax.set_xlabel('distance from grasp to centre of mass  |s − c|  [cm]', color=INK2)
    ax.set_ylabel('P(bar drops | grasped)', color=INK2)
    ax.set_ylim(-0.02, 1.02)
    ax.tick_params(colors=INK2)
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9, loc='upper left')
    ax.set_title('OGBar: natural drops from a weak soft-fingertip grip (real pick-and-place trials)',
                 color=INK, fontsize=11, loc='left')
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------- floor -> platform task curves

def platform_trial(args):
    """One floor -> platform sub-task with a forced first grasp offset; optional OGBench-style action noise.

    success: the first pickup attempt is placed and, once the policy has let go, the bar rests on the platform
    within the success tolerance of the target.
    """
    cfg, seed, noisy = args
    rng = np.random.default_rng(seed)
    env = _drop_env(cfg)
    w_right = sample_w_right(cfg['mass'], rng)
    ob, info = env.reset(seed=int(seed), options={'w_right': w_right})
    s = float(rng.uniform(-cfg['grasp']['s_max'], cfg['grasp']['s_max']))
    orc = BarMarkovOracle(env, cfg, np.random.default_rng(seed + 1))
    orc.reset(ob, info, 0, 0)
    orc._s = s
    orc.pickups[-1]['s'] = s
    xi = float(rng.uniform(0, cfg['noise']['noise_max'])) if noisy else 0.0
    scale = np.asarray(cfg['noise']['scale'])
    for t in range(cfg['drop']['max_steps']):
        a = orc.select_action(ob, info, t)
        a = np.clip(a + rng.normal(0.0, xi * scale), -1, 1)
        ob, _, _, _, info = env.step(a)
        if orc.pickups[0]['outcome'] and orc.pickups[0]['outcome'] != 'placed':
            break
        if orc.pickups[0]['outcome'] == 'placed' and (orc.done or orc.phase == 9):
            break
    p = orc.pickups[0]
    success_release = bool(p['outcome'] == 'placed' and env._compute_success())
    # Keep the policy running (it retreats) for `settle_steps` more steps: a released capsule can roll away.
    for k in range(cfg['drop']['settle_steps'] if p['outcome'] == 'placed' else 0):
        ob, _, _, _, info = env.step(orc.select_action(ob, info, t + 1 + k) if not orc.done else np.zeros(5))
    return dict(w_right=w_right, s=s, c=env.mass.com_offset, lever=s - env.mass.com_offset, noisy=noisy,
                outcome=p['outcome'] or 'timeout', grasped=p['t_grasp'] >= 0,
                success_release=success_release,
                success=bool(p['outcome'] == 'placed' and env._compute_success()), on_platform=env.bar_on_platform())


def run_platform_curves(cfg, n_trials, bins_cm, workers, log=print):
    import multiprocessing as mp

    jobs = [(cfg, 10_000 * i + 11, noisy) for noisy in (False, True) for i in range(n_trials)]
    with mp.get_context('spawn').Pool(workers) as pool:
        res = pool.map(platform_trial, jobs, chunksize=4)
    edges = np.asarray(bins_cm) / 100
    log('first pickup, start -> target (over the fence / onto the board), by |s - c| bin (cm): '
        + '  '.join(f'{a*100:.0f}-{b*100:.0f}' for a, b in zip(edges[:-1], edges[1:])))
    for noisy in (False, True):
        rs = [r for r in res if r['noisy'] == noisy and r['grasped']]
        for label, key in (('P(at release)', 'success_release'), ('P(settled)', 'success'), ('P(dropped)', 'slipped')):
            cells = []
            for a, b in zip(edges[:-1], edges[1:]):
                sel = [r for r in rs if a <= abs(r['lever']) < b]
                v = np.mean([r[key] if key.startswith('success') else r['outcome'] == key for r in sel]) if sel else np.nan
                cells.append(f'{v:.2f}({len(sel):3d})')
            log(f"{'noisy' if noisy else 'clean':5s} {label:11s} " + ' '.join(cells))
        n_all = sum(1 for r in res if r['noisy'] == noisy)
        log(f"      grasped {len(rs)}/{n_all}; overall success {np.mean([r['success'] for r in rs]):.2f}")
    return res
