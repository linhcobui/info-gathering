"""Latent actions with segments cut at STILL frames (no generator timing), on a FROZEN stage-1 run.

    pixi run python scripts/train_latent_action_still_phase1.py --run <ckpt> --train_zarr <zarr> --n_train N --out <dir>

train_latent_action_prior_phase1.py used hand-made blocks: 6 frames long (the generator's swap length), started
at "hold start + 4 + 6 j" (the swap schedule, from each episode's known swap count). Here a segment is read off
the frames alone:

    still frame t   = frame t identical to frame t-1 (as the alignment loss's eligibility)
    segment         = from a still frame t to the NEXT still frame t' (> t)
                      t' = t + 1  -> nothing moved in between ("nothing")
                      t' > t + 1  -> something moved, and has come to rest again (a swap, whatever its length)
    kept only while the action flag is "hold" at t-1 .. t'-1 (the flag is an input the model always gets), so
    the cups-down motion right after the reveal and anything reaching the lift are left out.

The inverse model reads e_t .. e_t' (padded with the resting last frame to the longest segment); the prior
plays the code for exactly t' - t steps (loss masked beyond). How long each latent action lasts is also learned
from the data: the median length of the training segments it is read off, saved as `durations` and used when a
code is pressed in imagination. Ground truth (swap schedule and identity) is used ONLY to name segments in the
report.
"""
import argparse
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
from world_model.models.smt_phase1.latent_action_prior import LatentActionPrior  # noqa: E402

HOLD_PH, MOVE, SWAP = 1, 4, 6          # used ONLY to name segments in the report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="logdir/world_model/smt_phase1_all3/naive/checkpoints/wm_latest.pt")
    ap.add_argument("--train_zarr", default="data/memtest4_all3.zarr")
    ap.add_argument("--n_train", type=int, default=384)
    ap.add_argument("--zarr", default="data/memtest4_eval.zarr")
    ap.add_argument("--out", default="logdir/world_model/smt_phase1_all3/naive/latent_action_still")
    ap.add_argument("--codes", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=12, help="segments longer than this are dropped")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tau", type=float, nargs=2, default=[1.0, 0.3])
    ap.add_argument("--balance", action="store_true", help="sample segments where something moved and segments "
                    "where nothing did equally often (length is read off the frames: still a label-free choice)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev, K = args.device, args.codes
    torch.manual_seed(0)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train.log", "a")

    def say(msg=""):
        print(msg, flush=True)
        log.write(msg + "\n"); log.flush()

    model = load_model(args.run, dev)
    amp = dict(device_type=dev.split(":")[0], dtype=model.amp_dtype)
    key = model.embed_key
    norm = lambda v: (torch.from_numpy(np.asarray(v, np.float32)).to(dev) - model.embed_mean) / model.embed_std

    def segments(zarr_path, episodes):
        zg = zarr.open_group(str(ROOT / zarr_path), mode="r")
        en = zg["episode_ends"][:]
        st = np.concatenate([[0], en[:-1]])
        act = zg["action"][:].astype(np.float32)
        seqs, ph = sequences(ROOT / zarr_path), zg["phase"][:]
        raw = []
        for e in episodes:
            s, n = int(st[e]), int(en[e] - st[e])
            E = norm(zg[key][s:s + n])
            PA = torch.zeros(1, n, 2, device=dev)
            PA[0, 1:] = torch.from_numpy(act[s:s + n - 1]).to(dev)
            first = torch.zeros(1, n, dtype=torch.bool, device=dev); first[0, 0] = True
            with torch.no_grad():
                Zt = model.encode_windows(E[None], PA, first)[0]
            still = np.r_[False, (E[1:] - E[:-1]).abs().amax(-1).cpu().numpy() < 1e-6]
            hold = act[s:s + n, 0] > 0.5                                   # the action flag the model is given
            # for naming only: the swap schedule
            h0, k = int(np.flatnonzero(ph[s:s + n] == HOLD_PH)[0]), len(seqs[e][1])
            starts = {h0 + MOVE + SWAP * j: seqs[e][1][j] for j in range(k)}
            stills = np.flatnonzero(still)
            for a_, b_ in zip(stills[:-1], stills[1:]):
                if a_ < 1 or not hold[a_ - 1:b_].all() or b_ - a_ > args.max_len:
                    continue
                L = int(b_ - a_)
                name = (f"swap {starts[a_][0]}<->{starts[a_][1]}" if a_ in starts and L > 1 else
                        "nothing" if L == 1 else f"other (len {L})")
                raw.append((Zt[a_], E[a_:b_ + 1], PA[0, a_ + 1:b_ + 1], Zt[a_ + 1:b_ + 1], L, name))
        Lmax = max(r[4] for r in raw)
        return raw, Lmax

    def pack(raw, Lmax):
        n = len(raw)
        Z0 = torch.stack([r[0] for r in raw])
        FR = torch.stack([torch.cat([r[1], r[1][-1:].expand(Lmax + 1 - len(r[1]), -1)]) for r in raw])
        A = torch.stack([torch.cat([r[2], r[2][-1:].expand(Lmax - len(r[2]), -1)]) for r in raw])
        Y = torch.stack([torch.cat([r[3], r[3][-1:].expand(Lmax - len(r[3]), -1, -1)]) for r in raw])
        lens = torch.tensor([r[4] for r in raw], device=dev)
        mask = (torch.arange(Lmax, device=dev)[None] < lens[:, None]).float()
        return Z0, FR, A, Y, lens, mask, [r[5] for r in raw]

    raw_tr, Lmax_tr = segments(args.train_zarr, range(args.n_train))
    n_test = len(zarr.open_group(str(ROOT / args.zarr), mode="r")["episode_ends"])
    raw_te, Lmax_te = segments(args.zarr, range(n_test))
    Lmax = max(Lmax_tr, Lmax_te)
    Ztr, FRtr, Atr, Ytr, Ltr, Mtr, ktr = pack(raw_tr, Lmax)
    Zte, FRte, Ate, Yte, Lte, Mte, kte = pack(raw_te, Lmax)
    say(f"segments cut at still frames: {len(Ztr)} training, {len(Zte)} test; lengths "
        f"{dict(sorted(Counter(Ltr.tolist()).items()))} (train)")
    say(f"  named by the hidden schedule (report only): {dict(Counter(k.split(' ')[0] for k in ktr))}")
    var = (Yte.float() - Yte.float().mean(0)).pow(2).mean()

    moved = (Ltr > 1).float()
    w = moved / moved.sum() + (1 - moved) / (1 - moved).sum() if args.balance else torch.ones_like(moved)
    say(f"  sampling: {'balanced moved / still' if args.balance else 'uniform over segments'}")
    lap = LatentActionPrior(model.prior, model.embed_dim, block=Lmax, n_codes=K).to(dev)
    opt = torch.optim.AdamW(lap.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-2)

    def masked_err(path, y, m):
        return ((path.float() - y.float()).pow(2).mean((-1, -2)) * m).sum(1) / m.sum(1)

    @torch.no_grad()
    def evaluate(step, Z0, FR, A, Y, M):
        lap.eval()
        codes, err, det = [], [], []
        for i in range(0, len(Z0), 256):
            z, fr, a, y, m = Z0[i:i + 256], FR[i:i + 256], A[i:i + 256], Y[i:i + 256], M[i:i + 256]
            with autocast(**amp):
                u, _ = lap.infer(fr)
                path = lap.rollout(z, a, u)
                zd, dp = z, []
                for j in range(Lmax):
                    zd = model.prior.step(zd, a[:, j]); dp.append(zd)
            codes.append(u.argmax(-1)); err.append(masked_err(path, y, m)); det.append(masked_err(torch.stack(dp, 1), y, m))
        lap.train()
        return torch.cat(codes).cpu().numpy(), torch.cat(err), torch.cat(det)

    for step in range(0, args.steps + 1):
        if step % 500 == 0:
            c, e_, d_ = evaluate(step, Zte, FRte, Ate, Yte, Mte)
            share = np.bincount(c, minlength=K) / len(c)
            say(f"step {step:5d} | test path error (fraction of variance): det {float(d_.mean() / var):.4f}  "
                f"with latent action {float(e_.mean() / var):.4f} | codes used (>1%): {int((share > 0.01).sum())}")
        if step == args.steps:
            break
        tau = args.tau[0] * (args.tau[1] / args.tau[0]) ** ((step + 1) / args.steps)
        idx = torch.multinomial(w, args.batch, replacement=True)
        z, fr, a, y, m = Ztr[idx], FRtr[idx], Atr[idx], Ytr[idx], Mtr[idx]
        with autocast(**amp):
            u, logits = lap.infer(fr, tau=tau)
            path = lap.rollout(z, a, u)
            choose = lap.code_logits(z, a[:, 0])
        loss_dyn = masked_err(path, y, m).mean()
        loss_choose = torch.nn.functional.cross_entropy(choose.float(), logits.argmax(-1).detach())
        opt.zero_grad(set_to_none=True)
        (loss_dyn + loss_choose).backward()
        torch.nn.utils.clip_grad_norm_(lap.parameters(), 1.0)
        opt.step()

    # how long each latent action lasts: the median length of the training segments it is read off (label-free)
    ctr, _, _ = evaluate(-1, Ztr, FRtr, Atr, Ytr, Mtr)
    durations = [int(np.median(Ltr.cpu().numpy()[ctr == c])) if (ctr == c).any() else 1 for c in range(K)]
    codes, err, det = evaluate(-1, Zte, FRte, Ate, Yte, Mte)
    share = np.bincount(codes, minlength=K) / len(codes)
    meaning = {}
    kinds = np.array(kte)
    order = ["swap 0<->1", "swap 0<->2", "swap 1<->2", "nothing"]
    used = [c for c in np.argsort(-share) if share[c] > 0.01]
    say(f"\nA. LATENT ACTIONS USED: {len(used)} of {K} (share of test segments > 1%)")
    say(f"  {'action':7s} {'share':>6s} {'lasts':>6s}  " + "  ".join(f"{k:>11s}" for k in order) + "   (where its segments come from)")
    for c in used:
        m = codes == c
        row = [np.mean(kinds[m] == k) for k in order]
        meaning[int(c)] = order[int(np.argmax(row))] if max(row) > 0 else "other"
        say(f"  u{c:<6d} {share[c]:6.3f} {durations[c]:6d}  " + "  ".join(f"{r:11.2f}" for r in row))
    say("  purity (each kind -> its most common action):")
    for k in order + sorted({x for x in kte if x.startswith("other")}):
        m = kinds == k
        if m.any():
            say(f"    {k:14s} n {m.sum():4d}: " + ", ".join(f"u{c} {v / m.sum():.2f}" for c, v in Counter(codes[m].tolist()).most_common(3)))
    say("\nB. path error by kind (fraction of variance): deterministic prior vs with the latent action")
    for k in order:
        mt = torch.as_tensor(kinds == k, device=dev)
        if mt.any():
            say(f"    {k:11s} det {float(det[mt].mean() / var):.4f}   latent action {float(err[mt].mean() / var):.4f}")
    torch.save({"lap": lap.state_dict(), "n_codes": K, "block": Lmax, "durations": durations, "run": args.run,
                "steps": args.steps, "segments": "still"}, out / "latent_action_prior.pt")
    import json
    json.dump({"share": share.tolist(), "meaning": {k: v.replace("nothing", "idle") for k, v in meaning.items()},
               "durations": durations}, open(out / "eval.json", "w"), indent=1)
    say(f"-> {out}")


if __name__ == "__main__":
    main()
