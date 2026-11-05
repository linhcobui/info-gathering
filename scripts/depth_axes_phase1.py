"""Are there DEPTH axes in the posterior state (number of swaps done, time in the hold)? And does
removing them help the onset answer on deep episodes?

    pixi run python scripts/depth_axes_phase1.py --run <stage-1-format checkpoint dir> [--device cuda:0]

1. Posterior states at every hold frame (cups down .. lift onset) of the stage-1 TRAINING episodes.
   PCA on them; for each top component: variance explained, R^2 with swaps done and with frames since
   the hold began, and eta^2 with the ball's current slot and its start slot.
2. Depth subspace U: the ridge-regression directions for swaps done and for time, orthonormalised.
   Both labels are observable (count of swaps seen, frame count) -- nothing hidden.
3. Normalise: z <- z - ((z - m) U) U^T  (m = training mean), then re-rms-normalise each token.
     onset   : only the state the decoder reads at the lift onset
     every   : after EVERY posterior step, so depth never accumulates in the chain
   Score: the frozen reward decoder's memory-only lift read-out, on test seen 0-4 / unseen 3 / unseen 4
   / 5 swaps, against the unmodified chain.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from torch.amp import autocast  # noqa: E402
from torch.utils.data import default_collate  # noqa: E402

from info_gathering.dataset_memtest1 import memtest1_datasets  # noqa: E402
from info_gathering.swap_sequences import sequences  # noqa: E402
from world_model.models.smt_prelim.blocks import normalize_mem  # noqa: E402
from world_model.training.loop import to_device  # noqa: E402
from world_model.utils.checkpoint import CheckpointManager  # noqa: E402

HOLD, SHOW, MOVE, SWAP = 1, 2, 4, 6


def load_model(run, device):
    st = CheckpointManager.load(Path(run) / "checkpoints" / "wm_latest.pt")
    cfg = OmegaConf.create(st["config"])
    tr, _ = instantiate(cfg.data)
    m = instantiate(cfg.model, action_dim=tr.action_dim, device=device, _recursive_=False)
    m.load_train_state_dict(st["model"])
    m.requires_grad_(False).eval()
    return m, cfg


def episode(ds, e, device):
    A = ds.arrays
    n = int(A.steps - A.meta["pad"][e].sum())
    ph = A.meta["phase"][e, :n]
    b = to_device(default_collate([ds[e]]), device)
    b = {q: v[:, :n] if v.dim() > 1 else v for q, v in b.items()}
    return b, n, int(np.flatnonzero(ph == HOLD)[0]), int(np.flatnonzero(ph == SHOW)[0])


@torch.no_grad()
def rollout(model, b, normalise=None):
    """The eval's posterior chain (warm-up on frame 0), optionally normalising after every step."""
    emb = model._embed(b["image"])
    pa = b["prev_action"].float()
    T = emb.shape[1]
    z = model.encode_windows(emb[:, :1], pa[:, :1], b["is_first"][:, :1])[:, -1]
    zs = [z]
    for t in range(1, T):
        with autocast(emb.device.type, dtype=model.amp_dtype):
            z = model.posterior.step(z, emb[:, t], pa[:, t]).float()
        if normalise is not None:
            z = normalise(z)
        zs.append(z)
    return torch.stack(zs, 1)[0], emb[0]


def hold_states(model, ds, seqs, device, episodes):
    X, swaps, time, slot, start = [], [], [], [], []
    for e in episodes:
        b, n, h0, on = episode(ds, e, device)
        Z, _ = rollout(model, b)
        k = len(seqs[e][1])
        for t in range(h0 + MOVE - 1, on + 1):
            j = t - (h0 + MOVE)
            done = 0 if j < 0 else min(k, (j + 1) // SWAP)
            X.append(Z[t].flatten().cpu())
            swaps.append(done)
            time.append(t - h0)
            slot.append(int(ds.arrays.meta["ball_slot"][e, t]))
            start.append(seqs[e][0])
    return torch.stack(X).double(), *(np.array(v) for v in (swaps, time, slot, start))


def r2(x, y):
    c = np.corrcoef(x, y)[0, 1]
    return c * c


def eta2(x, lab):
    tot = x.var() + 1e-12
    return sum((lab == c).mean() * (x[lab == c].mean() - x.mean()) ** 2 for c in np.unique(lab)) / tot


@torch.no_grad()
def onset_answer(model, ds, e, device, normalise=None, at="none"):
    b, n, h0, on = episode(ds, e, device)
    Z, emb = rollout(model, b, normalise if at == "every" else None)
    z = Z[on:on + 1]
    if at == "onset":
        z = normalise(z)
    F = int(model.fut_len)
    idx = np.clip(np.arange(on, on + F), 0, n - 1)
    with autocast(emb.device.type, dtype=model.amp_dtype):
        lg = model.reward_decoder.ball_logit(model.reward_decoder(
            z, emb[idx][None], b["action"][:, idx].float())).float()[0]
    return bool(lg[1] > 0), bool(ds.arrays.meta["ball_under_target"][e, n - 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ridge", type=float, default=10.0)
    args = ap.parse_args()
    dev = args.device
    model, cfg = load_model(args.run, dev)
    keep = ("ball_under_target", "ball_slot", "phase")
    dtr, _ = memtest1_datasets("data/memtest4.zarr", image_size=64, holdout_episodes=0, meta_keys=keep)
    seq_tr = sequences(ROOT / "data/memtest4.zarr")
    print(f"== {args.name or args.run}")

    # ---- 1. PCA on training-episode hold states
    X, swaps, time, slot, start = hold_states(model, dtr, seq_tr, dev, range(225))
    m = X.mean(0, keepdim=True)
    Xc = X - m
    U_, S_, Vt = torch.linalg.svd(Xc, full_matrices=False)
    var = (S_ ** 2) / (S_ ** 2).sum()
    P = (Xc @ Vt[:10].T).numpy()
    print(f"PCA on {len(X)} posterior hold states from the 225 training episodes")
    print(f"{'PC':>4}{'var':>8}{'R2 swaps done':>15}{'R2 time':>10}{'eta2 slot now':>15}{'eta2 start slot':>17}")
    for i in range(10):
        print(f"{i + 1:>4}{float(var[i]):>8.3f}{r2(P[:, i], swaps):>15.3f}{r2(P[:, i], time):>10.3f}"
              f"{eta2(P[:, i], slot):>15.3f}{eta2(P[:, i], start):>17.3f}")

    # ---- 2. depth subspace: ridge directions for swaps done and time
    G = Xc.T @ Xc + args.ridge * torch.eye(X.shape[1], dtype=X.dtype)
    dirs = []
    for y in (swaps, time):
        yc = torch.from_numpy(y - y.mean()).double()
        w = torch.linalg.solve(G, Xc.T @ yc)
        fit = (Xc @ w).numpy()
        dirs.append(w / w.norm())
        print(f"ridge direction for {'swaps done' if y is swaps else 'time in hold'}: in-sample R2 {r2(fit, y):.3f}")
    U, _ = torch.linalg.qr(torch.stack(dirs, 1))
    proj = (Xc @ U).numpy()
    print(f"share of state variance in this 2-D depth subspace: {float((proj ** 2).sum() / (Xc.numpy() ** 2).sum()):.3f}")
    print(f"slot information inside it: eta2 slot now {eta2(proj[:, 0], slot):.3f} / {eta2(proj[:, 1], slot):.3f}")
    Uf, mf = U.float().to(dev), m.float().to(dev)

    def normalise(z):
        flat = z.reshape(z.shape[0], -1).float()
        flat = flat - ((flat - mf) @ Uf) @ Uf.T
        return normalize_mem(flat.view_as(z))

    # ---- 3. onset answer with / without normalisation
    seen = set(sequences(ROOT / cfg.data_anchors.zarr_path)[:-int(cfg.data_anchors.holdout_episodes)])
    dev_, _ = memtest1_datasets("data/memtest4_eval.zarr", image_size=64, holdout_episodes=0, meta_keys=keep)
    ds5, _ = memtest1_datasets("data/memtest4_eval5.zarr", image_size=64, holdout_episodes=0, meta_keys=keep)
    ev, ev5 = sequences(ROOT / "data/memtest4_eval.zarr"), sequences(ROOT / "data/memtest4_eval5.zarr")
    buckets = {"seen 0-4": (dev_, [e for e in range(len(ev)) if ev[e] in seen]),
               "unseen 3": (dev_, [e for e in range(len(ev)) if len(ev[e][1]) == 3 and ev[e] not in seen]),
               "unseen 4": (dev_, [e for e in range(len(ev)) if len(ev[e][1]) == 4 and ev[e] not in seen]),
               "5 swaps": (ds5, list(range(len(ev5))))}
    print("\nonset answer (memory-only read-out), accuracy:")
    print(f"{'bucket':<12}{'n':>4}{'as is':>9}{'normalised at onset':>22}{'normalised every step':>24}")
    for name, (d, eps) in buckets.items():
        acc = {}
        for at in ("none", "onset", "every"):
            res = [onset_answer(model, d, e, dev, normalise, at) for e in eps]
            acc[at] = np.mean([p == y for p, y in res])
        print(f"{name:<12}{len(eps):>4}{acc['none']:>9.2f}{acc['onset']:>22.2f}{acc['every']:>24.2f}")


if __name__ == "__main__":
    main()
