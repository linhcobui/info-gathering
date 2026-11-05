"""Branching prior that COMMITS once per 6-frame block (a swap's length), on a frozen stage-1 run.

    pixi run python scripts/train_mcl_block_prior_phase1.py [--run <ckpt>] [--out <dir>]

The one-step version (train_mcl_prior_phase1.py) scored each code one frame ahead. The first frame of 0<->1
and of 0<->2 look alike (the left cup starts to move either way), so one code won both and learned their
average: imagined swaps came out as a blend of the two. Here a code is held fixed for a whole block:

    from the teacher state at a block start (an at-rest hold frame), run the prior 6 steps with code c,
    on its OWN predictions; compare the 6 imagined states with the real 6 that followed; train the code
    whose whole path fits best (losers get a small share). p(c | z, a) learns which block outcome happens.

Over 6 frames 0<->1 and 0<->2 end in different places, so one code can no longer cover both cheaply.
Training through its own 6-step rollouts also targets the drift seen in long imagination.

Block starts: the at-rest frame before each swap, every idle frame after the swaps, and the lift onset.
Reported on memtest4_eval: 6-step error of the deterministic prior vs best-of-K vs sampled, and which code
wins for each kind of block (ground truth used for that table only).
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import zarr  # noqa: E402
from torch.amp import autocast  # noqa: E402

from info_gathering.swap_sequences import sequences  # noqa: E402
from train_mcl_prior_phase1 import load_model  # noqa: E402
from world_model.models.smt_phase1.mcl_prior import MCLPrior  # noqa: E402

HOLD_PH, LIFT_PH, MOVE, SWAP = 1, 2, 4, 6


@torch.no_grad()
def blocks(model, zarr_path, episodes, dev, L=6):
    """(z at block start, the L actions, the L real next teacher states, kind of block)."""
    z = zarr.open_group(str(ROOT / zarr_path), mode="r")
    raw, act, ph = z[model.embed_key], z["action"][:].astype(np.float32), z["phase"][:]
    ends = z["episode_ends"][:]
    starts = np.concatenate([[0], ends[:-1]])
    seqs = sequences(ROOT / zarr_path)
    Z0, A, Y, kind = [], [], [], []
    for e in episodes:
        s, n = int(starts[e]), int(ends[e] - starts[e])
        E = ((torch.from_numpy(raw[s:s + n].astype(np.float32)).to(dev) - model.embed_mean) / model.embed_std)[None]
        PA = torch.zeros(1, n, act.shape[1], device=dev)
        PA[0, 1:] = torch.from_numpy(act[s:s + n - 1]).to(dev)
        first = torch.zeros(1, n, dtype=torch.bool, device=dev); first[0, 0] = True
        Zt = model.encode_windows(E, PA, first)[0]
        p = ph[s:s + n]
        h0, on, k = int(np.flatnonzero(p == HOLD_PH)[0]), int(np.flatnonzero(p == LIFT_PH)[0]), len(seqs[e][1])
        b0 = h0 + MOVE
        ts = [(b0 + SWAP * j, f"swap {seqs[e][1][j][0]}<->{seqs[e][1][j][1]}") for j in range(k)]
        ts += [(t, "idle") for t in range(b0 + SWAP * k, on - 1)]
        ts += [(on - 1, "lift")]
        for t, kd in ts:
            if t + L < n:
                Z0.append(Zt[t]); A.append(PA[0, t + 1:t + 1 + L]); Y.append(Zt[t + 1:t + 1 + L]); kind.append(kd)
    return torch.stack(Z0), torch.stack(A), torch.stack(Y), kind


def rollout(mcl, z, acts, code):
    """Hold `code` fixed for len(acts) steps on the prior's own predictions: (B, L, K, d)."""
    out = []
    for i in range(acts.shape[1]):
        z = mcl.step(z, acts[:, i], code)
        out.append(z)
    return torch.stack(out, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="logdir/world_model/smt_phase1_memtest4/window60_phase1/readout/main/checkpoints/wm_latest.pt")
    ap.add_argument("--train_zarr", default="data/memtest4.zarr")
    ap.add_argument("--n_train", type=int, default=225)
    ap.add_argument("--test_zarr", default="data/memtest4_eval.zarr")
    ap.add_argument("--out", default="logdir/world_model/smt_phase1_memtest4/window60_phase1/mcl_block_prior")
    ap.add_argument("--codes", type=int, default=8)
    ap.add_argument("--block", type=int, default=6)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--loser_share", type=float, default=0.05)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev, L, C = args.device, args.block, args.codes
    torch.manual_seed(0)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train.log", "a")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n"); log.flush()

    model = load_model(args.run, dev)
    amp = dict(device_type=dev.split(":")[0], dtype=model.amp_dtype)
    Ztr, Atr, Ytr, ktr = blocks(model, args.train_zarr, range(args.n_train), dev, L)
    n_test = len(zarr.open_group(str(ROOT / args.test_zarr), mode="r")["episode_ends"])
    Zte, Ate, Yte, kind = blocks(model, args.test_zarr, range(n_test), dev, L)
    say(f"{len(Ztr)} training blocks ({dict(Counter(k.split(' ')[0] for k in ktr))}), {len(Zte)} test blocks")
    var = (Yte.float() - Yte.float().mean(0)).pow(2).mean()

    mcl = MCLPrior(model.prior, n_codes=C).to(dev)
    opt = torch.optim.AdamW(mcl.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-2)

    def path_err(z, a, y):
        """(B, C) mean squared error of each code's whole L-step path."""
        b = z.shape[0]
        codes = torch.arange(C, device=dev).repeat(b)
        with autocast(**amp):
            paths = rollout(mcl, z.repeat_interleave(C, 0), a.repeat_interleave(C, 0), codes)
        return (paths.float() - y.repeat_interleave(C, 0).float()).pow(2).flatten(1).mean(1).view(b, C)

    @torch.no_grad()
    def evaluate(step):
        mcl.eval()
        det, best, samp, wins = [], [], [], []
        g = torch.Generator(device=dev).manual_seed(0)
        for i in range(0, len(Zte), 256):
            z, a, y = Zte[i:i + 256], Ate[i:i + 256], Yte[i:i + 256]
            with autocast(**amp):
                zd, dp = z, []
                for j in range(L):
                    zd = model.prior.step(zd, a[:, j]); dp.append(zd)
                probs = torch.softmax(mcl.code_logits(z, a[:, 0]).float(), -1)
                c = torch.multinomial(probs, 1, generator=g).squeeze(-1)
                sp = rollout(mcl, z, a, c)
            e = path_err(z, a, y)
            det.append((torch.stack(dp, 1).float() - y.float()).pow(2).flatten(1).mean(1))
            best.append(e.min(1).values); wins.append(e.argmin(1))
            samp.append((sp.float() - y.float()).pow(2).flatten(1).mean(1))
        det, best, samp, wins = (torch.cat(v) for v in (det, best, samp, wins))
        use = torch.bincount(wins, minlength=C).float() / len(wins)
        say(f"step {step:5d} | test {L}-step path error (fraction of variance): det {float(det.mean() / var):.4f}  "
            f"best-of-K {float(best.mean() / var):.4f}  sampled {float(samp.mean() / var):.4f} | code use "
            + " ".join(f"{u:.2f}" for u in use.tolist()))
        mcl.train()
        return wins.cpu().numpy(), det, best, samp

    evaluate(0)
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, len(Ztr), (args.batch,), device=dev)
        z, a, y = Ztr[idx], Atr[idx], Ytr[idx]
        e = path_err(z, a, y)
        win = e.argmin(1)
        loss_dyn = (1 - args.loser_share) * e.gather(1, win[:, None]).mean() + args.loser_share * e.mean()
        with autocast(**amp):
            logits = mcl.code_logits(z, a[:, 0])
        loss_choose = torch.nn.functional.cross_entropy(logits.float(), win)
        opt.zero_grad(set_to_none=True)
        (loss_dyn + loss_choose).backward()            # disjoint parameters: the head only reads frozen inputs
        torch.nn.utils.clip_grad_norm_(mcl.parameters(), 1.0)
        opt.step()
        if step % 250 == 0:
            say(f"step {step:5d} | train best-of-K {float(e.min(1).values.mean()):.5f}  choose CE {float(loss_choose):.3f}")
        if step % args.eval_every == 0 or step == args.steps:
            wins, det, best, samp = evaluate(step)
    torch.save({"mcl": mcl.state_dict(), "n_codes": C, "block": L, "run": args.run, "steps": args.steps},
               out / "mcl_prior.pt")

    kinds = np.array(kind)
    say(f"\nwinning code by kind of block (test episodes), share of that kind's blocks:")
    say(f"{'kind':14s} {'n':>5s} " + " ".join(f"c{c:<4d}" for c in range(C)) + "  det-err  best-err  sampled-err")
    table = {}
    for k in sorted(set(kind)):
        m = kinds == k
        cnt = Counter(wins[m].tolist())
        row = [cnt.get(c, 0) / m.sum() for c in range(C)]
        table[k] = row
        mt = torch.as_tensor(m, device=dev)
        say(f"{k:14s} {m.sum():5d} " + " ".join(f"{r:5.2f}" for r in row)
            + f"  {float(det[mt].mean() / var):.4f}   {float(best[mt].mean() / var):.4f}   {float(samp[mt].mean() / var):.4f}")
    json.dump(table, open(out / "code_by_kind.json", "w"), indent=1)
    say(f"-> {out / 'mcl_prior.pt'}")


if __name__ == "__main__":
    main()
