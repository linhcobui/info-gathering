"""Train a branching prior (MCLPrior) on a FROZEN stage-1 run. Nothing else in the model is touched.

    pixi run python scripts/train_mcl_prior_phase1.py --run <ckpt with the stage-1 weights> --out <dir>

Pairs are exactly what the phase-1 prior is trained on: (teacher state at t, a_t) -> teacher state at t+1,
over the run's own training episodes. Reported on the test episodes (memtest4_eval):

    det        the run's own deterministic prior (the smear)
    best-of-K  the closest of the K branches   (how well the branches cover what really happens)
    sampled    c ~ p(c | z, a)                 (what imagination gets)

and at the end, which code wins at each kind of step -- in particular the first frame of each swap, by
the swap's slot pair (ground truth used for this table only).
"""
import argparse
import json
import sys
from collections import Counter
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
from world_model.models.smt_phase1.mcl_prior import MCLPrior  # noqa: E402
from world_model.utils.checkpoint import CheckpointManager  # noqa: E402

HOLD_PH, LIFT_PH, MOVE, SWAP = 1, 2, 4, 6


def load_model(ckpt, dev):
    st = CheckpointManager.load(ROOT / ckpt)
    cfg = OmegaConf.create(st["config"])
    tr, _ = instantiate(cfg.data)
    model = instantiate(cfg.model, action_dim=tr.action_dim, device=dev, _recursive_=False)
    model.load_train_state_dict(st["model"])
    return model.requires_grad_(False).eval()


@torch.no_grad()
def pairs(model, zarr_path, episodes, dev, with_kind=False):
    """Teacher states for every step of every episode -> (z_t, a_t, z_{t+1}) and a step kind."""
    z = zarr.open_group(str(ROOT / zarr_path), mode="r")
    raw, act, ph = z[model.embed_key], z["action"][:].astype(np.float32), z["phase"][:]
    ends = z["episode_ends"][:]
    starts = np.concatenate([[0], ends[:-1]])
    seqs = sequences(ROOT / zarr_path) if with_kind else None
    Z0, A, Z1, kind = [], [], [], []
    for e in episodes:
        s, n = int(starts[e]), int(ends[e] - starts[e])
        E = ((torch.from_numpy(raw[s:s + n].astype(np.float32)).to(dev) - model.embed_mean) / model.embed_std)[None]
        PA = torch.zeros(1, n, act.shape[1], device=dev)
        PA[0, 1:] = torch.from_numpy(act[s:s + n - 1]).to(dev)
        first = torch.zeros(1, n, dtype=torch.bool, device=dev); first[0, 0] = True
        Zt = model.encode_windows(E, PA, first)[0]
        Z0.append(Zt[:-1]); Z1.append(Zt[1:]); A.append(PA[0, 1:])          # prev_action[t+1] == a_t
        if with_kind:
            p = ph[s:s + n]
            h0, on, k = int(np.flatnonzero(p == HOLD_PH)[0]), int(np.flatnonzero(p == LIFT_PH)[0]), len(seqs[e][1])
            for t in range(n - 1):
                u = t + 1 - (h0 + MOVE)
                if t + 1 >= on:
                    kind.append("lift")
                elif t + 1 < h0 + MOVE:
                    kind.append("before swaps")
                elif u < SWAP * k:
                    j, off = divmod(u, SWAP)
                    pair = seqs[e][1][j]
                    kind.append(f"swap {pair[0]}<->{pair[1]} frame 1" if off == 1 else
                                ("between swaps (rest)" if off == 0 else "swap frames 2-5"))
                else:
                    kind.append("idle after swaps")
    return torch.cat(Z0), torch.cat(A), torch.cat(Z1), kind


def err(pred, tgt):
    return (pred.float() - tgt.float()).pow(2).flatten(-2).mean(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="logdir/world_model/smt_phase1_memtest4/window60_phase1/readout/main/checkpoints/wm_latest.pt")
    ap.add_argument("--train_zarr", default="data/memtest4.zarr")
    ap.add_argument("--n_train", type=int, default=225)
    ap.add_argument("--test_zarr", default="data/memtest4_eval.zarr")
    ap.add_argument("--out", default="logdir/world_model/smt_phase1_memtest4/window60_phase1/mcl_prior")
    ap.add_argument("--codes", type=int, default=8)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--loser_share", type=float, default=0.05)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device
    torch.manual_seed(0)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train.log", "a")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n"); log.flush()

    model = load_model(args.run, dev)
    amp = dict(device_type=dev.split(":")[0], dtype=model.amp_dtype)
    Ztr, Atr, Ytr, _ = pairs(model, args.train_zarr, range(args.n_train), dev)
    n_test = len(zarr.open_group(str(ROOT / args.test_zarr), mode="r")["episode_ends"])
    Zte, Ate, Yte, kind = pairs(model, args.test_zarr, range(n_test), dev, with_kind=True)
    say(f"{len(Ztr)} training pairs ({args.n_train} episodes), {len(Zte)} test pairs ({n_test} episodes)")

    mcl = MCLPrior(model.prior, n_codes=args.codes).to(dev)
    opt = torch.optim.AdamW(mcl.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-2)

    @torch.no_grad()
    def evaluate(step):
        mcl.eval()
        det, best, samp, wins = [], [], [], []
        g = torch.Generator(device=dev).manual_seed(0)
        for i in range(0, len(Zte), 1024):
            z, a, y = Zte[i:i + 1024], Ate[i:i + 1024], Yte[i:i + 1024]
            with autocast(**amp):
                d = model.prior.step(z, a)
                allc = mcl.all_codes(z, a)
                s, _ = mcl.sample_step(z, a, generator=g)
            e_all = err(allc, y[:, None])
            det.append(err(d, y)); best.append(e_all.min(1).values); samp.append(err(s, y)); wins.append(e_all.argmin(1))
        det, best, samp, wins = (torch.cat(v) for v in (det, best, samp, wins))
        use = torch.bincount(wins, minlength=args.codes).float() / len(wins)
        var = (Yte.float() - Yte.float().mean(0)).pow(2).flatten(1).mean(1).mean()
        say(f"step {step:5d} | test error (fraction of state variance): det {float(det.mean() / var):.4f}  "
            f"best-of-K {float(best.mean() / var):.4f}  sampled {float(samp.mean() / var):.4f} | "
            f"code use " + " ".join(f"{u:.2f}" for u in use.tolist()))
        mcl.train()
        return wins.cpu().numpy(), det, best, samp, var

    evaluate(0)
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, len(Ztr), (args.batch,), device=dev)
        z, a, y = Ztr[idx], Atr[idx], Ytr[idx]
        with autocast(**amp):
            allc = mcl.all_codes(z, a)
            logits = mcl.code_logits(z, a)
        e_all = err(allc, y[:, None])                                        # (B, C)
        win = e_all.argmin(1)
        loss_dyn = (1 - args.loser_share) * e_all.gather(1, win[:, None]).mean() + args.loser_share * e_all.mean()
        loss_choose = torch.nn.functional.cross_entropy(logits.float(), win)
        opt.zero_grad(set_to_none=True)
        (loss_dyn + loss_choose).backward()         # disjoint parameters: the head only reads frozen inputs
        torch.nn.utils.clip_grad_norm_(mcl.parameters(), 1.0)
        opt.step()
        if step % 250 == 0:
            say(f"step {step:5d} | train best-of-K {float(e_all.min(1).values.mean()):.5f}  choose CE {float(loss_choose):.3f}")
        if step % args.eval_every == 0 or step == args.steps:
            wins, det, best, samp, var = evaluate(step)
    torch.save({"mcl": mcl.state_dict(), "n_codes": args.codes, "run": args.run, "steps": args.steps}, out / "mcl_prior.pt")

    # which code wins at which kind of step (test episodes; ground truth for this table only)
    kinds = np.array(kind)
    say("\nwinning code by kind of step (test episodes), share of that kind's steps:")
    say(f"{'kind':32s} {'n':>5s} " + " ".join(f"c{c:<4d}" for c in range(args.codes)) + "  det-err  best-err (fraction of variance)")
    table = {}
    for k in sorted(set(kind)):
        m = kinds == k
        cnt = Counter(wins[m].tolist())
        row = [cnt.get(c, 0) / m.sum() for c in range(args.codes)]
        mt = torch.as_tensor(m, device=dev)
        table[k] = row
        say(f"{k:32s} {m.sum():5d} " + " ".join(f"{r:5.2f}" for r in row)
            + f"  {float(det[mt].mean() / var):.4f}   {float(best[mt].mean() / var):.4f}")
    json.dump(table, open(out / "code_by_kind.json", "w"), indent=1)
    say(f"-> {out / 'mcl_prior.pt'}")


if __name__ == "__main__":
    main()
