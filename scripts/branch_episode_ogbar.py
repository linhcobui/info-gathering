"""Counterfactual branches of one recorded OGBar episode: identical history, a different grasp for the next probe.

The source episode's frames up to raw step T are copied unchanged. At T the simulator is restored from the recorded
joint positions, velocities and control targets, and one probe is run with a chosen grasp point s (same lift height as
the source's probe, no action noise). Each branch is stored as a full episode: copied history + new continuation, with
every privileged key, so the same eval scripts run on it. A branch with the source's own grasp point checks the
restore (it should reproduce the recorded outcome).

    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 pixi run python scripts/branch_episode_ogbar.py --episode 33 --raw-t 690 \\
        --s -0.105 -0.093 -0.056 0.0 0.04 0.09 --lift 0.223
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402
import zarr  # noqa: E402

from ogbar.generate import finalize_store, run_play_episode  # noqa: E402
from ogbar.recorder import OgbarRecorder  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/ogbar_slot_v3_flat_noslip.yaml")
    ap.add_argument("--src", default="data/ogbar_v3_400.zarr")
    ap.add_argument("--episode", type=int, default=33)
    ap.add_argument("--raw-t", type=int, default=690, help="branch point (raw step); a multiple of 5 keeps model steps aligned")
    ap.add_argument("--s", type=float, nargs="+", required=True, help="grasp points along the bar (m from its middle)")
    ap.add_argument("--lift", type=float, default=0.223)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--out", default=None)
    ap.add_argument("--append", action="store_true", help="add branches to an existing store")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / args.config))
    src = zarr.open_group(str(ROOT / args.src), "r")
    e, T = args.episode, args.raw_t
    s0 = int(np.r_[0, src["episode_ends"][:][:-1]][e])
    seed, w_right = int(src["episode_seeds"][e]), float(src["episode/w_right"][e])
    restore = dict(qpos=src["qpos"][s0 + T], qvel=src["qvel"][s0 + T], ctrl=src["ctrl"][s0 + T - 1])
    out = ROOT / (args.out or f"data/ogbar_v3_ep{e:03d}_branches.zarr")
    if out.exists() and not args.append:
        raise SystemExit(f"{out} exists; remove it first (or --append)")
    rec = OgbarRecorder(str(out), cfg["resolution"], append=args.append)
    c = float(src["episode/com_offset_c"][e])
    print(f"source episode {e} (c {c * 100:+.1f} cm), branch at raw {T} (model step {T // 5})")
    for s in args.s:
        res = run_play_episode(cfg, seed, w_right=w_right, plan=["probe"], probe_s=[s], probe_lift=args.lift, noise_xi=0.0,
                               max_steps=args.max_steps, restore_state=restore)
        F = res["frames"]
        frames = {k: np.concatenate([src[k][s0:s0 + T].astype(v.dtype), v]) for k, v in F.items()}
        tab = res["tables"]
        for name in ("segment", "pickup"):
            for key in ("t_start", "t_end", "t_grasp"):
                if key in tab[name]:
                    a = np.asarray(tab[name][key])
                    tab[name][key] = np.where(a >= 0, a + T, a)
        ep = dict(res["episode"], length=np.int32(len(frames["rgb"])))
        rec.add_episode(frames, ep, tab, seed)
        seg = res["tables"]["segment"]
        print(f"  s {s * 100:+6.1f} cm (from COM {(s - c) * 100:+5.1f}): branch {len(F['rgb'])} steps, probe tilt "
              f"{float(seg['probe_tilt_deg'][0]):+6.1f} deg, slipped {bool(seg['probe_slipped'][0])}")
    prev = list(zarr.open_group(str(out), "r").attrs.get("branch_s", [])) if args.append else []
    cfg["branch"] = dict(source=args.src, episode=e, raw_t=T, s=prev + list(args.s), lift=args.lift, noise_xi=0.0)
    finalize_store(rec, cfg, seed)
    zarr.open_group(str(out), "r+").attrs["branch_s"] = cfg["branch"]["s"]
    print(out)


if __name__ == "__main__":
    main()
