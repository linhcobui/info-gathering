"""MPC with CEM on the privileged-aligned OGBar-SMT, in the real simulator: reach a seated bar from an eval start.

Goal: the eval episode's start state with the bar teleported into the slot (seated pose, level, arm where it started),
settled with zero actions and checked with BarEnv.bar_in_slot(); its side-view frame is DINOv2-encoded and turned
into a belief z_goal by the teacher on a window holding only that frame.

Loop (receding horizon), from the eval episode's frame 0 (no pickups before):
  z_t      = teacher on every real frame so far (DINOv2 tokens of the simulator's side view) and the actions taken
  CEM      plans H model steps; one model step = one 5-D action repeated for its 5 raw steps (25-D, as trained).
           Each candidate is imagined with the prior, z_{k+1} = P(z_k, a_k), and frames with the decoder's first
           output, x_{k+1} = readout(z_k, x_k, a_k) (compare_from_cut.py). Cost of a candidate (--cost):
             belief   mean over the last 5 imagined steps of mean (z_k - z_goal)^2
             frame    mean over the last 5 imagined steps of mean (x_k - x_goal)^2 over all 256 x 384 patch values
  execute  the first --replan model steps of the elite mean in the simulator (5 raw steps each), then replan.
Writes <out>/<stem>.mp4 (sim side view | goal | imagined frame at the plan's end, DINO-WM pixel decoder) and a log.

    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 pixi run python scripts/mpc_cem_ogbar.py --cost belief
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import zarr  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from cache_dinov2_ogbar import encode, load_dinov2  # noqa: E402
from dino_wm_videos import decode_all  # noqa: E402
from dino_wm_watch_imagine import panel  # noqa: E402
from ogbar.generate import get_env  # noqa: E402
from train_dino_wm import build_model  # noqa: E402
from world_model.models.smt_ogbar import SmtOgbar  # noqa: E402
from world_model.utils.checkpoint import CheckpointManager  # noqa: E402

AMP = dict(device_type="cuda", dtype=torch.bfloat16)
K_RAW = 5


def restore(env, qpos, qvel, ctrl):
    """generate.run_play_episode's restore block."""
    u = env.unwrapped
    u._data.qpos[:] = qpos
    u._data.qvel[:] = qvel
    u._data.ctrl[:] = ctrl
    u._ik_tilt_corr = np.zeros(3)
    mujoco.mj_forward(u._model, u._data)
    u.pre_step()
    u.post_step()
    return u.compute_observation()


def bar_state(env):
    u = env.unwrapped
    p = u._data.joint("object_joint_0").qpos[:3].copy()
    tilt = np.degrees(np.arcsin(np.clip(u.bar_axis_world()[2], -1, 1)))
    c = u.contacts()
    return p, tilt, bool(c["left"] or c["right"]), bool(u.bar_in_slot())


def make_goal(env, start, cfg):
    """Seated bar (slot centre, pocket height, level, along x), arm at its start pose; settle with zero actions."""
    u = env.unwrapped
    qpos = start["qpos"].copy()
    adr = u._bar_qpos_adr
    sl = cfg["slot"]
    qpos[adr:adr + 3] = [sl["x"], sl["y"], 0.084]
    qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
    qvel = np.zeros_like(start["qvel"])
    restore(env, qpos, qvel, start["ctrl"])
    ob = None
    for _ in range(40):
        ob, *_ = env.step(np.zeros(5))
    p, tilt, held, seated = bar_state(env)
    print(f"goal: bar at {np.round(p, 3)}, tilt {tilt:+.1f} deg, seated {seated}", flush=True)
    if not seated:
        raise RuntimeError("the generated goal does not count as seated (bar_in_slot is False)")
    return ob


@torch.no_grad()
def batch_readout(m, z, x, a):
    """compare_from_cut.readout for N candidates: z (N, K, d), x (N, 256, 384), a (N, 25) model-normalised."""
    tok = m.dec_step_in(torch.cat([m.pool(x), a.to(x.dtype)], -1))
    return m.head(m.decoder(torch.cat([z, tok[:, None]], 1))[:, m.K])


@torch.no_grad()
def imagine(m, z0, x0, acts, need_frames, keep=5):
    """acts (N, H, 25) model-normalised. Returns the last `keep` imagined beliefs (N, keep, K, d) and frames
    (N, keep, 256, 384) or None. P reads (z_k, a_k) only; the frame read-out also reads x_k."""
    N, H = acts.shape[:2]
    z, x = z0.expand(N, -1, -1).contiguous(), x0.expand(N, -1, -1).contiguous()
    zs, xs = [], []
    with torch.autocast(**AMP):
        for k in range(H):
            if need_frames:
                x = batch_readout(m, z, x, acts[:, k])
                if k >= H - keep:
                    xs.append(x)
            z, _ = m.prior(z, None, acts[:, k])
            if k >= H - keep:
                zs.append(z)
    return torch.stack(zs, 1), (torch.stack(xs, 1) if need_frames else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smt", default="logdir/world_model/smt_ogbar/2026-10-30_17-37-45_privalign_push/checkpoints/wm_latest.pt")
    ap.add_argument("--dino-wm", default="logdir/world_model/dino_wm_ogbar/v3_400_proprio/checkpoints/wm_latest.pt")
    ap.add_argument("--zarr", default="data/ogbar_v3_eval5.zarr")
    ap.add_argument("--episode", type=int, default=2)
    ap.add_argument("--cost", choices=["belief", "frame"], default="belief")
    ap.add_argument("--goal-raw", type=int, default=None,
                    help="goal = the recorded side-view frame at this raw step of the same episode (e.g. 562 = ep 2's "
                         "probe-3 hold: gripped at the COM, 18 cm up, level) instead of the generated seated bar")
    ap.add_argument("--horizon", type=int, default=30, help="model steps planned (x5 raw steps)")
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--elites", type=int, default=64)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--replan", type=int, default=2, help="model steps executed per plan")
    ap.add_argument("--steps", type=int, default=120, help="model steps executed in all (x5 raw)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device(args.device)

    st = CheckpointManager.load(args.smt)
    m = SmtOgbar(action_dim=25, **st["config"]["model"]).to(dev).eval()
    m.load_state_dict(st["model"])
    dst = CheckpointManager.load(args.dino_wm)
    dw = build_model(OmegaConf.create(dst["config"]), dst["stats"], 25, 19).to(dev).eval()
    dw.load_state_dict(dst["model"])
    dino = load_dinov2(dev)

    z = zarr.open_group(str(ROOT / args.zarr), "r")
    cfg = z.attrs["config"]
    cfg = json.loads(cfg) if isinstance(cfg, str) else dict(cfg)
    s0 = 0 if args.episode == 0 else int(z["episode_ends"][args.episode - 1])
    start = {k: z[k][s0] for k in ("qpos", "qvel", "ctrl")}
    w_right = float(z["episode/w_right"][args.episode])
    c_cm = float(z["episode/com_offset_c"][args.episode]) * 100
    env = get_env(cfg)
    env.reset(seed=int(z["episode_seeds"][args.episode]), options={"w_right": w_right})

    goal = make_goal(env, start, cfg) if args.goal_raw is None else z["rgb"][s0 + args.goal_raw]
    if args.goal_raw is not None:
        print(f"goal: recorded frame raw {args.goal_raw}: bar z {float(z['bar_pos'][s0 + args.goal_raw][2]):.3f} m, "
              f"grasped {int(z['grasped'][s0 + args.goal_raw])}", flush=True)
    ob = restore(env, start["qpos"], start["qvel"], start["ctrl"])
    print(f"start frame vs recorded rgb[0]: mean |diff| {np.abs(ob.astype(float) - z['rgb'][s0]).mean():.2f} (0-255)", flush=True)

    enc = lambda ims: encode(dino, np.stack(ims), dev)[0]
    x_goal = enc([goal])                                                          # (1, 256, 384)
    with torch.no_grad(), torch.autocast(**AMP):
        z_goal = m.teacher_windows(m.pool(x_goal[None].to(torch.bfloat16)), torch.zeros(1, 1, 25, device=dev),
                                   torch.zeros(1, 1, dtype=torch.long, device=dev))[0, 0]

    a_mean = m.action_mean.view(K_RAW, 5)[0].cpu().numpy()                       # training stats of one raw action
    a_std = m.action_std.view(K_RAW, 5)[0].cpu().numpy()
    H, N, E = args.horizon, args.samples, args.elites
    mu = np.tile(a_mean, (H, 1))
    frames, patches, acts = [ob], [enc([ob])[0]], []
    stem = f"mpc_{args.cost}_ep{args.episode}_H{H}_N{N}" + ("" if args.goal_raw is None else f"_goalraw{args.goal_raw}")
    out_dir = Path(args.smt).parents[1] / "mpc"
    out_dir.mkdir(parents=True, exist_ok=True)
    video, log = [], []
    goal_panel = panel(goal, "goal (seated, sim)" if args.goal_raw is None else f"goal (ep{args.episode} raw {args.goal_raw})")
    seated_ever, t_model, lifted_ever, z_max = False, 0, False, 0.0
    t_start = time.time()
    while t_model < args.steps:
        P = torch.stack(patches).to(dev).to(torch.bfloat16)[None]                 # (1, T, 256, 384)
        A = torch.zeros(1, len(patches), 25, device=dev)
        if acts:
            A[0, :len(acts)] = torch.as_tensor(np.stack(acts), device=dev, dtype=torch.float32)
        with torch.no_grad(), torch.autocast(**AMP):
            z_t = m.teacher_windows(m.pool(P), m.norm_action(A),
                                    torch.full((1, 1), len(patches) - 1, dtype=torch.long, device=dev))[0, 0]
        x_t = P[0, -1]
        sd = np.tile(a_std, (H, 1))
        mu = np.concatenate([mu[args.replan:], np.tile(a_mean, (args.replan, 1))])     # warm start: shift
        for it in range(args.iters):
            cand = np.clip(mu[None] + sd[None] * rng.standard_normal((N, H, 5)), -1, 1)
            cand[0] = np.clip(mu, -1, 1)                                         # keep the current mean
            raw = torch.as_tensor(np.repeat(cand, K_RAW, axis=1).reshape(N, H, 25), device=dev, dtype=torch.float32)
            zs, xs = imagine(m, z_t[None], x_t[None], m.norm_action(raw), need_frames=args.cost == "frame")
            if args.cost == "belief":
                cost = (zs.float() - z_goal.float()).pow(2).mean((1, 2, 3))           # (N, 5, K, d) vs (K, d)
            else:
                cost = (xs.float() - x_goal.float()).pow(2).mean((1, 2, 3))           # (N, 5, 256, 384) vs (1, 256, 384)
            cost = cost.cpu().numpy()
            el = np.argsort(cost)[:E]
            mu, sd = cand[el].mean(0), cand[el].std(0) + 1e-3
        best = cost[el[0]]
        # what the plan imagines at its end (frames for the video, either cost)
        raw = torch.as_tensor(np.repeat(mu[None], K_RAW, axis=1).reshape(1, H, 25), device=dev, dtype=torch.float32)
        _, xs = imagine(m, z_t[None], x_t[None], m.norm_action(raw), need_frames=True, keep=1)
        imagined_end = decode_all(dw, xs[0, -1:].float())[0]
        for k in range(args.replan):
            a = np.clip(mu[k], -1, 1)
            for _ in range(K_RAW):
                ob, *_ = env.step(a)
                bp, _, bh, bs = bar_state(env)
                seated_ever |= bs
                z_max = max(z_max, float(bp[2]))
                lifted_ever |= bh and bp[2] > 0.07                                   # held and 5 cm above its rest height
            acts.append(np.tile(a, K_RAW))
            frames.append(ob)
            patches.append(enc([ob])[0])
            t_model += 1
            p, tilt, held, seated = bar_state(env)
            video.append(np.concatenate([panel(ob, f"sim ep{args.episode} c {c_cm:+.1f}  t {t_model}", f"{'HELD ' if held else ''}{'SEATED ' if seated else ''}"),
                                         goal_panel, panel(imagined_end, f"imagined +{H} ({args.cost} cost)")], 1))
        line = (f"t {t_model:3d}  cost {best:.4f}  bar {np.round(p, 3)}  tilt {tilt:+5.1f}  held {int(held)}  "
                f"seated {int(seated)} (ever {int(seated_ever)})  lifted-ever {int(lifted_ever)}  max z {z_max:.3f}  dist-to-slot {np.hypot(p[0] - cfg['slot']['x'], p[1] - cfg['slot']['y']):.3f} m  "
                f"{time.time() - t_start:.0f}s")
        print(line, flush=True)
        log.append(line)
    imageio.mimwrite(out_dir / f"{stem}.mp4", video, fps=8, macro_block_size=8)
    imageio.imwrite(out_dir / f"{stem}_goal.png", goal)
    (out_dir / f"{stem}.log").write_text("\n".join(log) + "\n")
    print(f"video {out_dir / (stem + '.mp4')}\nseated at any raw step: {seated_ever}; lifted (held, > 5 cm up): {lifted_ever}; "
          f"max bar z {z_max:.3f} m")


if __name__ == "__main__":
    main()
