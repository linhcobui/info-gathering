"""Self-labels for alignment: a trained model's OWN answer to "is the ball in slot 0 right now?".

    pixi run python scripts/build_self_slot0_labels.py --run <stage-1 run> --zarr data/memtest4.zarr --out <npz>

For every AT-REST hold frame t of every episode -- cups just down, the end of each swap, idle frames,
the lift onset; mid-swap frames are skipped, a "no swap" future cannot be spliced onto a moving cup --
the frozen model's TEACHER state z_t is read by its reward decoder with a spliced future:

    (x_t, hold) , 3 x (rest frame, hold) , (rest frame, lift) , lift frames ...

and the memory-only read-out of that lift (its first frame) is the label. With the cups down the frames
do not depend on where the ball is (verified: identical embeddings across episodes), so the spliced
future is a real-looking one, and its answer can only come from the state.

The true "ball in slot 0 now" is saved alongside, for the agreement check ONLY; training reads `label`.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import zarr  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from torch.amp import autocast  # noqa: E402

from info_gathering.swap_sequences import sequences  # noqa: E402
from world_model.utils.checkpoint import CheckpointManager  # noqa: E402

HOLD, SHOW, MOVE, SWAP, IDLE = 1, 2, 4, 6, 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="logdir/world_model/smt_phase1_memtest4/window60_alignpos_phase1")
    ap.add_argument("--zarr", default="data/memtest4.zarr")
    ap.add_argument("--out", default="data/labels_slot0_self_w60alignpos_memtest4.npz")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device
    st = CheckpointManager.load(Path(args.run) / "checkpoints" / "wm_latest.pt")
    cfg = OmegaConf.create(st["config"])
    tr, _ = instantiate(cfg.data)
    model = instantiate(cfg.model, action_dim=tr.action_dim, device=dev, _recursive_=False)
    model.load_train_state_dict(st["model"])
    model.requires_grad_(False).eval()
    F, dec = int(model.fut_len), model.reward_decoder

    z = zarr.open_group(str(ROOT / args.zarr), mode="r")
    raw, act, ph, slot = z[model.embed_key][:], z["action"][:].astype(np.float32), z["phase"][:], z["ball_slot"][:]
    ends = z["episode_ends"][:]
    starts = np.concatenate([[0], ends[:-1]])
    seqs = sequences(ROOT / args.zarr)
    norm = lambda v: (torch.from_numpy(np.asarray(v, np.float32)).to(dev) - model.embed_mean) / model.embed_std
    # library: the at-rest frame (cups down) and an empty lift, from the first episode without the ball
    e0 = next(e for e in range(len(ends)) if not z["ball_under_target"][ends[e] - 1])
    on0 = int(np.flatnonzero(ph[starts[e0]:ends[e0]] == SHOW)[0])
    REST, LIFT = norm(raw[starts[e0] + on0]), norm(raw[starts[e0] + on0 + 1:ends[e0]])
    A_HOLD, A_LIFT = torch.tensor([1., 0.], device=dev), torch.tensor([0., 1.], device=dev)
    tail_x = [REST] * IDLE + [REST] + list(LIFT)
    tail_a = [A_HOLD] * IDLE + [A_LIFT] + [A_LIFT] * len(LIFT)
    while len(tail_x) < F - 1:
        tail_x.append(LIFT[-1]); tail_a.append(A_LIFT)
    TX, TA = torch.stack(tail_x[:F - 1]), torch.stack(tail_a[:F - 1])
    READ = 1 + IDLE + 1                                   # read-out index of the spliced lift's first frame

    out = {k: [] for k in ("episode", "t", "label", "logit", "truth", "swaps_done")}
    for e in range(len(ends)):
        s, n = int(starts[e]), int(ends[e] - starts[e])
        p = ph[s:s + n]
        h0, on, k = int(np.flatnonzero(p == HOLD)[0]), int(np.flatnonzero(p == SHOW)[0]), len(seqs[e][1])
        E = norm(raw[s:s + n])[None]
        PA = torch.zeros(1, n, act.shape[1], device=dev)
        PA[0, 1:] = torch.from_numpy(act[s:s + n - 1]).to(dev)
        with torch.no_grad():
            Z = model.encode_windows(E, PA, torch.ones(1, n, dtype=torch.bool, device=dev) & (torch.arange(n, device=dev) == 0))[0]
            ts = [t for t in range(h0 + MOVE - 1, on + 1)
                  if not (0 <= t - (h0 + MOVE) < SWAP * k and (t - (h0 + MOVE)) % SWAP != SWAP - 1)]
            idx = torch.as_tensor(ts, device=dev)
            X = torch.cat([E[0, idx][:, None], TX[None].expand(len(ts), -1, -1)], 1)
            Ac = torch.cat([A_HOLD.expand(len(ts), 1, -1), TA[None].expand(len(ts), -1, -1)], 1)
            with autocast(dev.split(":")[0], dtype=model.amp_dtype):
                lg = dec.ball_logit(dec(Z[idx], X, Ac)).float()[:, READ].cpu().numpy()
        for t, l in zip(ts, lg):
            j = t - (h0 + MOVE)
            out["episode"].append(e); out["t"].append(t); out["logit"].append(float(l)); out["label"].append(bool(l > 0))
            out["truth"].append(bool(slot[s + t] == 0)); out["swaps_done"].append(0 if j < 0 else min(k, (j + 1) // SWAP))
    out = {k: np.array(v) for k, v in out.items()}
    np.savez(ROOT / args.out, **out, note=np.array("label = model's own answer; truth is for diagnostics only"))
    ok = out["label"] == out["truth"]
    print(f"{len(ok)} at-rest hold frames from {len(ends)} episodes -> {args.out}")
    print(f"labelled 'ball in slot 0 now': {out['label'].mean():.3f} (true rate {out['truth'].mean():.3f})")
    print(f"agreement with the true 'ball in slot 0 now': {ok.mean():.3f}")
    for d in range(5):
        m = out["swaps_done"] == d
        if m.any():
            print(f"   swaps done {d}: agreement {ok[m].mean():.3f} (n={m.sum()})")


if __name__ == "__main__":
    main()
