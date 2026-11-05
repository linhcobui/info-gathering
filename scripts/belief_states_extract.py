"""Chained-posterior memory at MOTIONLESS hold frames of the shown-start AND closed-start test sets -> npz for t-SNE.

    pixi run python scripts/belief_states_extract.py --out <npz> NAME=<run dir> ...

Per state: the ball's true slot now, the honest belief (class as belief_labels_phase1.SETS: a known slot when the
start was shown; the forward-filtered belief when it was not), shown/closed, episode, swap count. Same frames for
every run. Plot with scripts/belief_states_tsne.py (system python3: needs scikit-learn).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import zarr  # noqa: E402
from torch.utils.data import default_collate  # noqa: E402

from belief_labels_phase1 import CLASS, truth_labels  # noqa: E402
from info_gathering.dataset_memtest1 import memtest1_datasets  # noqa: E402
from info_gathering.swap_sequences import sequences  # noqa: E402
from train_mcl_prior_phase1 import load_model  # noqa: E402
from world_model.training.loop import to_device  # noqa: E402

STORES = (("shown", "data/memtest4_eval.zarr"), ("closed", "data/memtest4_evalu.zarr"))


def frames_and_labels():
    """(store, episode, t) of every motionless hold frame, with the true slot and the honest belief class."""
    rows = []
    for kind, path in STORES:
        z = zarr.open_group(str(ROOT / path), mode="r")
        ends = z["episode_ends"][:]
        st = np.r_[0, ends[:-1]]
        rgb_free = z["embed_r2d_memtest4_wm15000"]
        ph, slot, sw = z["phase"][:], z["ball_slot"][:], z["episode_swapped"][:]
        if kind == "closed":
            seqs = sequences(ROOT / path)
            cfg = json.loads(z.attrs["config"])
            peek_len = 2 * int(cfg["move_frames"]) + int(cfg.get("peek_up_frames", 2))
            rest = float(z["cup_z_max"][:].min())
        for e in range(len(ends)):
            s, n = int(st[e]), int(ends[e] - st[e])
            E = rgb_free[s:s + n]
            still = np.r_[False, np.abs(E[1:] - E[:-1]).max(-1) < 1e-6] & (ph[s:s + n] == 1)
            if kind == "closed":
                cls = truth_labels(z, e, s, n, seqs, rest, peek_len)
            for t in np.flatnonzero(still):
                b = cls[t] if kind == "closed" else CLASS[frozenset({int(slot[s + t])})]
                rows.append((kind, e, int(t), int(slot[s + t]), int(b), int(sw[e])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="NAME=<stage-1 run dir>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    rows = frames_and_labels()
    want = {}
    for i, (kind, e, t, *_rest) in enumerate(rows):
        want.setdefault((kind, e), []).append((t, i))
    names, Z = [], []
    for spec in args.runs:
        name, run = spec.split("=", 1)
        model = load_model(Path(run) / "checkpoints" / "wm_latest.pt", args.device)
        model.observe_mode = "recurrent"
        out = np.zeros((len(rows), model.n_registers * 256 if hasattr(model, "n_registers") else 2560), np.float32)
        for kind, path in STORES:
            ds, _ = memtest1_datasets(str(ROOT / path), image_size=64, holdout_episodes=0,
                                      meta_keys=("ball_under_target", "ball_slot", "phase"))
            rl = ds.arrays.steps - ds.arrays.meta["pad"].sum(1)
            for h in np.unique(rl):
                eps = [int(e) for e in np.flatnonzero(rl == h) if (kind, int(e)) in want]
                if not eps:
                    continue
                batch = to_device(default_collate([ds[e] for e in eps]), args.device)
                with torch.no_grad():
                    zz = model.observe(batch)["z"].float().flatten(2).cpu().numpy()      # (B, T, K*d)
                for b, e in enumerate(eps):
                    for t, i in want[(kind, e)]:
                        out[i] = zz[b, t]
        names.append(name); Z.append(out)
        print(f"{name}: {len(rows)} states")
    r = np.array([(0 if k == "shown" else 1, e, t, s, b, w) for k, e, t, s, b, w in rows])
    np.savez(ROOT / args.out, names=np.array(names), z=np.stack(Z), closed=r[:, 0], episode=r[:, 1], t=r[:, 2],
             slot=r[:, 3], belief=r[:, 4], swaps=r[:, 5])
    print(f"-> {args.out}: shown {int((r[:, 0] == 0).sum())}, closed {int((r[:, 0] == 1).sum())} states")


if __name__ == "__main__":
    main()
