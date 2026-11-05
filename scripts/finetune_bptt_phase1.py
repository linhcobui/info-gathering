"""ABLATION: finetune ONLY the posterior with full BPTT, against the frozen reward decoder.

    pixi run python scripts/finetune_bptt_phase1.py [--lr 1e-6 --max_steps 2000]

Per training step, all training episodes together (padded, masked):

    z_0      = teacher(window ending at frame 0)                     frozen, no gradient
    z_t      = posterior(z_{t-1}, x_t, a_{t-1})    t = 1 .. onset    gradient flows through EVERY step
    r        = reward_decoder(z_onset, (x_onset,a_onset) .. )          frozen (paired layout)
    loss     = BCE( r at lift offset 1 -- the memory-only read-out, ball revealed or not )

Training data: the test set's 4-swap episodes whose swap order is NOT in the stage-1 training set.
Evaluated every --eval_every steps (posterior at the onset, memory-only read-out) on: seen 0-4,
unseen 3 (held out, trained depth), unseen 4 (trained on), 5 swaps (never trained on).
Data comes straight from the embedding cache (no DataLoader workers) to keep RAM low.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import zarr  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from torch.amp import autocast  # noqa: E402

from finetune_onestep_phase1 import Episodes  # noqa: E402
from info_gathering.swap_sequences import sequences  # noqa: E402
from world_model.utils.checkpoint import CheckpointManager  # noqa: E402


def onset_logits(model, data, idx, F_len, amp, grad):
    """Posterior chain z_0 .. z_onset (BPTT when grad=True), then the decoder at each episode's onset.

    Returns the memory-only read-out logit (lift offset 1) per episode, (B,).
    """
    dev = data.E.device
    idx = np.asarray(idx)
    on = torch.from_numpy(data.onset[idx]).to(dev)
    last = int(data.onset[idx].max())
    E, A, PA = data.E[idx], data.A[idx], data.PA[idx]
    B = len(idx)
    z = model.encode_windows(E[:, :1], PA[:, :1], torch.ones(B, 1, dtype=torch.bool, device=dev))[:, -1]
    rows = torch.arange(B, device=dev)
    z_on = z.clone() if (on == 0).any() else None
    with torch.set_grad_enabled(grad):
        states = [z]
        for t in range(1, last + 1):
            with autocast(dev.type, dtype=amp):
                z = model.posterior.step(z, E[:, t], PA[:, t]).float()       # no detach: full BPTT
            states.append(z)
        z_on = torch.stack(states, 1)[rows, on]                                # (B, K, d)
        fidx = torch.clamp(on[:, None] + torch.arange(F_len, device=dev)[None], max=data.L - 1)
        with autocast(dev.type, dtype=amp):
            lg = model.reward_decoder.ball_logit(
                model.reward_decoder(z_on, E[rows[:, None], fidx], A[rows[:, None], fidx])).float()
    return lg[:, 1]                                                            # r_{onset+1}: memory only


@torch.no_grad()
def scores(model, data, F_len, amp):
    lg = torch.cat([onset_logits(model, data, np.arange(i, min(i + 64, data.n)), F_len, amp, grad=False)
                    for i in range(0, data.n, 64)])
    p = (lg > 0).cpu().numpy()
    y = data.label.astype(bool)
    tp = (p & y).sum()
    f1 = 2 * tp / max(2 * tp + (p & ~y).sum() + (~p & y).sum(), 1)
    return {"acc": float((p == y).mean()), "f1": float(f1), "says_ball": float(p.mean()),
            "true_rate": float(y.mean()), "n": int(data.n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="logdir/world_model/smt_phase1_memtest4/window60_phase1")
    ap.add_argument("--out", default="logdir/world_model/smt_phase1_memtest4/window60_phase1/bptt_ft")
    ap.add_argument("--train_set", default="unseen4", choices=["unseen4", "seen04+unseen4"],
                    help="unseen4: the 55 unseen-order 4-swap test episodes. seen04+unseen4: those plus every "
                         "seen-order 0-4 test episode (260); unseen 3 and 5 swaps stay held out")
    ap.add_argument("--minibatch", type=int, default=64,
                    help="BPTT runs per minibatch; gradients accumulate to ONE optimiser step per full pass")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--save_every", type=int, default=100)
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()
    torch.manual_seed(0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    state = CheckpointManager.load(Path(args.run) / "checkpoints" / "wm_latest.pt")
    cfg = OmegaConf.create(state["config"])
    train_set, _ = instantiate(cfg.data)
    model = instantiate(cfg.model, action_dim=train_set.action_dim, device=args.device, _recursive_=False)
    del train_set
    model.load_train_state_dict(state["model"])
    model.requires_grad_(False).eval()
    model.posterior.requires_grad_(True)
    F_len, amp = int(model.fut_len), model.amp_dtype

    seen = set(sequences(ROOT / cfg.data_anchors.zarr_path)[:-int(cfg.data_anchors.holdout_episodes)])
    ev = sequences(ROOT / "data/memtest4_eval.zarr")
    d = lambda e: len(ev[e][1])
    E_ = lambda keep: Episodes("data/memtest4_eval.zarr", model, keep, args.device)
    if args.train_set == "unseen4":
        train = E_(lambda e: d(e) == 4 and ev[e] not in seen)
    else:
        train = E_(lambda e: ev[e] in seen or (d(e) == 4 and ev[e] not in seen))
    in_train = args.train_set != "unseen4"
    sets = {f"seen 0-4{' (TRAINED ON)' if in_train else ''}": E_(lambda e: ev[e] in seen),
            "unseen 3 (held out)": E_(lambda e: d(e) == 3 and ev[e] not in seen),
            "unseen 4 (TRAINED ON)": E_(lambda e: d(e) == 4 and ev[e] not in seen),
            "5 swaps (never trained)": Episodes("data/memtest4_eval5.zarr", model, lambda e: True, args.device)}

    def report(step):
        row = {k: scores(model, v, F_len, amp) for k, v in sets.items()}
        print(f"[{step:5d}] " + " | ".join(f"{k}: acc {v['acc']:.2f} F1 {v['f1']:.2f} ball {v['says_ball']:.2f}"
                                          for k, v in row.items()), flush=True)
        return row

    params = list(model.posterior.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    y = torch.from_numpy(train.label).to(args.device)
    print(f"BPTT finetune of the posterior on {train.n} test episodes ({args.train_set}) | lr {args.lr} | "
          f"minibatch {args.minibatch}, one optimiser step per full pass")
    print("columns: posterior at the onset, memory-only lift read-out. acc / F1 score / how often it says 'ball'")
    log = [{"step": 0, "eval": report(0)}]
    t0 = time.time()
    for step in range(1, args.max_steps + 1):
        opt.zero_grad(set_to_none=True)
        loss = 0.0
        for i in range(0, train.n, args.minibatch):            # accumulate = one full-batch gradient
            idx = np.arange(i, min(i + args.minibatch, train.n))
            lg = onset_logits(model, train, idx, F_len, amp, grad=True)
            part = F.binary_cross_entropy_with_logits(lg, y[idx], reduction="sum") / train.n
            part.backward()
            loss += float(part.detach())
        if step == 1:
            stray = [n for n, p in model.named_parameters() if p.grad is not None and not n.startswith("posterior.")]
            assert not stray, f"gradients reached frozen modules: {stray[:4]}"
            print("ok  only the posterior receives gradients", flush=True)
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        opt.step()
        if step % args.eval_every == 0:
            row = {"step": step, "loss": loss, "grad_norm": gn, "sec": time.time() - t0, "eval": report(step)}
            print(f"        train loss {loss:.4f} | grad-norm {gn:.1f} (clipped to 1) | {time.time() - t0:.0f}s", flush=True)
            log.append(row)
            (out / "log.jsonl").open("a").write(json.dumps(row) + "\n")
        if step % args.save_every == 0:
            torch.save(model.posterior.state_dict(), out / f"posterior_step{step}.pt")


if __name__ == "__main__":
    main()
