"""WINDOW60 mechanism check: does the state carry WHERE THE BALL IS NOW, mid-hold, before the swaps
are over -- and does it still carry the history?

    pixi run midhold-probe-phase1 NAME=<stage-1 run dir> [NAME=<dir> ...] [--zarr data/memtest4_eval.zarr]

For every episode of the test store, at every hold frame from "cups down" (hold frame 3) to the
frame before the lift, excluding frames in the middle of a swap (the slot is ambiguous there),
it takes two states:
    teacher    the teacher's encoding of the window ending at t
    posterior  warm up on frame 0, chain the posterior to t (the eval's own path)
and fits LINEAR probes (L2 logistic regression) on even-numbered episodes, tested on odd ones:

    ball slot now        by swaps DONE (0-4), and mid-hold (>=1 done, >=1 still to come)
                         -- 0 done is just the start slot, visible at the reveal; the rest need tracking
    start slot           vs what the current slot alone predicts   -- above it = history kept
    swaps done so far    vs what the current slot alone predicts

Labels are read only to evaluate the trained states; nothing here trains the model.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import zarr  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from torch.utils.data import default_collate  # noqa: E402

from info_gathering.dataset_memtest1 import memtest1_datasets  # noqa: E402
from info_gathering.dataset_smt_prelim import dense_ball_slot  # noqa: E402
from world_model.training.loop import to_device  # noqa: E402
from world_model.utils.checkpoint import CheckpointManager  # noqa: E402

HOLD, MOVE, SWAP = 1, 4, 6


def collect(model, ds, n_swaps_per_episode, device):
    A = ds.arrays
    real_len = A.steps - A.meta["pad"].sum(1)
    dense = dense_ball_slot(A.meta["ball_slot"], real_len)
    out = {"teacher": [], "posterior": [], "slot": [], "done": [], "left": [], "start": [], "ep": []}
    for e in range(A.num_episodes):
        n = int(real_len[e])
        phase = A.meta["phase"][e, :n]
        h0, onset = int(np.flatnonzero(phase == HOLD)[0]), int(np.flatnonzero(phase == 2)[0])
        n_swaps = int(n_swaps_per_episode[e])               # the generator's per-episode record
        first = h0 + MOVE - 1                                  # cups just down, before swap 1
        ts, done = [], []
        for t in range(first, onset):
            k = t - (h0 + MOVE)                                # frames into the swap block
            if 0 <= k < SWAP * n_swaps and k % SWAP != SWAP - 1:
                continue                                       # mid-swap: slot ambiguous
            ts.append(t)
            done.append(min(n_swaps, 0 if k < 0 else (k + 1) // SWAP))
        batch = to_device(default_collate([ds[e]]), device)
        batch = {q: v[:, :n] if v.dim() > 1 else v for q, v in batch.items()}
        model.observe_mode = "recurrent"
        post = model.observe(batch)["z"][0]
        model.observe_mode = "window"
        teach = model.observe(batch)["z"][0]
        model.observe_mode = "recurrent"
        idx = torch.as_tensor(ts, device=post.device)
        out["teacher"].append(teach[idx].flatten(1).float().cpu())
        out["posterior"].append(post[idx].flatten(1).float().cpu())
        out["slot"] += [int(dense[e, t]) for t in ts]
        out["done"] += done
        out["left"] += [n_swaps - d for d in done]
        out["start"] += [int(A.meta["ball_slot"][e, 0])] * len(ts)
        out["ep"] += [e] * len(ts)
    for k in ("teacher", "posterior"):
        out[k] = torch.cat(out[k])
    for k in ("slot", "done", "left", "start", "ep"):
        out[k] = np.array(out[k])
    return out


def fit_probe(X, y, n_cls, device, wd=1e-2, steps=400):
    """L2-regularised multinomial logistic regression, full batch. Returns a predict function."""
    mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True).clamp(min=1e-4)
    Xs = ((X - mu) / sd).to(device)
    yt = torch.as_tensor(y, device=device)
    lin = torch.nn.Linear(X.shape[1], n_cls).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=wd)
    for _ in range(steps):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(lin(Xs), yt).backward()
        opt.step()
    return lambda Z: lin(((Z - mu) / sd).to(device)).argmax(-1).cpu().numpy()


def slot_only_baseline(slot_tr, y_tr, slot_te, y_te):
    """Accuracy of predicting y from the CURRENT SLOT alone (majority y within each slot)."""
    pred = {s: np.bincount(y_tr[slot_tr == s]).argmax() for s in np.unique(slot_tr)}
    return float(np.mean([pred.get(s, 0) == y for s, y in zip(slot_te, y_te)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="NAME=<stage-1 run dir>")
    ap.add_argument("--zarr", default="data/memtest4_eval.zarr")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    n_swaps = zarr.open_group(str(ROOT / args.zarr), mode="r")["episode_swapped"][:]
    ds, _ = memtest1_datasets(str(ROOT / args.zarr), image_size=64, holdout_episodes=0,
                              meta_keys=("ball_under_target", "ball_slot", "phase"))
    results = {}
    for spec in args.runs:
        name, run = spec.split("=", 1)
        state = CheckpointManager.load(Path(run) / "checkpoints" / "wm_latest.pt")
        cfg = OmegaConf.create(state["config"])
        train_set, _ = instantiate(cfg.data)
        model = instantiate(cfg.model, action_dim=train_set.action_dim, device=args.device, _recursive_=False)
        model.load_train_state_dict(state["model"])
        model.requires_grad_(False).eval()
        d = collect(model, ds, n_swaps, args.device)
        tr, te = d["ep"] % 2 == 0, d["ep"] % 2 == 1
        res = {}
        for src in ("teacher", "posterior"):
            X = d[src]
            p_slot = fit_probe(X[tr], d["slot"][tr], 3, args.device)(X[te])
            ok = p_slot == d["slot"][te]
            done, left = d["done"][te], d["left"][te]
            acc = lambda m: (float(ok[m].mean()), int(m.sum())) if m.any() else None
            res[src] = {"slot_by_done": {int(k): acc(done == k) for k in range(5) if (done == k).any()},
                        "slot_midhold": acc((done >= 1) & (left >= 1)),
                        "slot_after_last": acc((done >= 1) & (left == 0)),
                        "slot_all": float(ok.mean())}
            for key, n_cls in (("start", 3), ("done", 5)):
                p = fit_probe(X[tr], d[key][tr], n_cls, args.device)(X[te])
                res[src][key] = {"probe": float((p == d[key][te]).mean()),
                                 "slot_only": slot_only_baseline(d["slot"][tr], d[key][tr], d["slot"][te], d[key][te])}
        results[name] = res
        del model
        torch.cuda.empty_cache()

    print("\nBall slot NOW, linear probe accuracy (chance ~0.33). n in brackets.")
    print("By swaps DONE: 0 done = the start slot, read off the opening reveal; every later column needs tracking.")
    print("mid-hold = >= 1 swap done AND >= 1 still to come: the states only a long decoder window asks about.")
    print(f"{'run':<16}{'state':<11}" + "".join(f"{f'{k} done':>13}" for k in range(5)) + f"{'mid-hold':>14}{'after last':>14}")
    for name, res in results.items():
        for src in ("teacher", "posterior"):
            r = res[src]
            fmt = lambda v: f"{v[0]:>7.3f} ({v[1]:>3})" if v else f"{'-':>13}"
            print(f"{name:<16}{src:<11}" + "".join(fmt(r["slot_by_done"].get(k)) for k in range(5))
                  + f"{fmt(r['slot_midhold']):>14}{fmt(r['slot_after_last']):>14}")
    print("\nHistory: probe accuracy vs what the CURRENT SLOT alone predicts. Probe above it = history is kept.")
    print(f"{'run':<16}{'state':<11}{'start slot':>22}{'swaps so far':>22}")
    for name, res in results.items():
        for src in ("teacher", "posterior"):
            s, w = res[src]["start"], res[src]["done"]
            print(f"{name:<16}{src:<11}{s['probe']:>10.3f} (slot-only {s['slot_only']:.2f}){w['probe']:>9.3f} (slot-only {w['slot_only']:.2f})")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
