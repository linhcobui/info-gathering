"""How often is imagination with the block-branching prior right?

    pixi run python scripts/eval_mcl_imagination_phase1.py [--mcl <mcl_prior.pt>] [--out <json>]

Start points: every test episode (memtest4_eval), at the at-rest frame before each swap and at the first idle
frame after the swaps; the state there is the POSTERIOR (memory chained over the real frames). From each:

  1. one imagined block (6 steps, one code held), with the code the prior SAMPLES, and with each code forced;
  2. the block is NAMED by matching its imagined frame embeddings (readout) to real swap clips of each slot
     pair, or to "no swap"; a block whose best match is worse than real blocks ever are is "unclear";
  3. the decoder's "ball if we lift now?" on the imagined state at the block end is scored against
     where the ball should be = the true slot at the start, moved by the named swap.

Chains (drift): from the start before the first swap, imagine 1..5 sampled blocks in a row, and score the
decoder after each block (only chains whose blocks are all clear).
Ground truth is used only for scoring.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import zarr  # noqa: E402
from torch.amp import autocast  # noqa: E402

from info_gathering.dataset_smt_prelim import dense_ball_slot  # noqa: E402
from info_gathering.swap_sequences import sequences  # noqa: E402
from train_mcl_prior_phase1 import load_model  # noqa: E402
from world_model.models.smt_phase1.mcl_prior import MCLPrior  # noqa: E402

HOLD_PH, LIFT_PH, MOVE, SWAP, IDLE = 1, 2, 4, 6, 3
PAIRS = ((0, 1), (0, 2), (1, 2))


def name(p):
    return "unclear" if p == "unclear" else ("no swap" if p is None else f"{p[0]}<->{p[1]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="logdir/world_model/smt_phase1_memtest4/window60_phase1/readout/main/checkpoints/wm_latest.pt")
    ap.add_argument("--mcl", default="logdir/world_model/smt_phase1_memtest4/window60_phase1/mcl_block_prior/mcl_prior.pt")
    ap.add_argument("--train_zarr", default="data/memtest4.zarr")
    ap.add_argument("--n_train", type=int, default=225)
    ap.add_argument("--zarr", default="data/memtest4_eval.zarr")
    ap.add_argument("--chain", type=int, default=5)
    ap.add_argument("--chain_samples", type=int, default=4)
    ap.add_argument("--out", default="logdir/world_model/smt_phase1_memtest4/window60_phase1/mcl_block_prior/imagination_eval.json")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device
    model = load_model(args.run, dev)
    amp = dict(device_type=dev.split(":")[0], dtype=model.amp_dtype)
    ck = torch.load(ROOT / args.mcl, map_location=dev)
    mcl = MCLPrior(model.prior, n_codes=ck["n_codes"]).to(dev).eval()
    mcl.load_state_dict(ck["mcl"])
    C, L = ck["n_codes"], int(ck.get("block", 6))
    key = model.embed_key
    norm = lambda v: (torch.from_numpy(np.asarray(v, np.float32)).to(dev) - model.embed_mean) / model.embed_std
    A_HOLD, A_LIFT = torch.tensor([1., 0.], device=dev), torch.tensor([0., 1.], device=dev)

    def ep_bounds(zg):
        ends = zg["episode_ends"][:]
        return np.concatenate([[0], ends[:-1]]), ends

    @torch.no_grad()
    def posterior(zg, s, n):
        E = norm(zg[key][s:s + n])[None]
        PA = torch.zeros(1, n, 2, device=dev)
        PA[0, 1:] = torch.from_numpy(zg["action"][s:s + n - 1].astype(np.float32)).to(dev)
        first = torch.zeros(1, n, dtype=torch.bool, device=dev); first[0, 0] = True
        warm = max(1, min(model.eval_warmup, n))
        zw = model.encode_windows(E[:, :warm], PA[:, :warm], first[:, :warm])
        zs, z = [zw], zw[:, -1]
        with autocast(**amp):
            for t in range(warm, n):
                z = model.posterior.step(z, E[:, t], PA[:, t]).float()
                zs.append(z.unsqueeze(1))
        return torch.cat(zs, 1)[0].float()

    # ---- templates: real swap clips by slot pair (frames at offsets 0..6 of a block), and "no swap"
    ztr = zarr.open_group(str(ROOT / args.train_zarr), mode="r")
    st_tr, en_tr = ep_bounds(ztr)
    tr_seqs, tr_ph = sequences(ROOT / args.train_zarr), ztr["phase"][:]
    acc = defaultdict(list)
    for e in range(args.n_train):
        s, n = int(st_tr[e]), int(en_tr[e] - st_tr[e])
        h0 = int(np.flatnonzero(tr_ph[s:s + n] == HOLD_PH)[0])
        for j, pr in enumerate(tr_seqs[e][1]):
            b = s + h0 + MOVE + SWAP * j
            acc[pr].append(ztr[key][b:b + 7])
    templates = {pr: norm(np.mean(v, 0)) for pr, v in acc.items()}
    e0 = next(e for e in range(len(en_tr)) if not ztr["ball_under_target"][en_tr[e] - 1])
    on0 = int(np.flatnonzero(tr_ph[st_tr[e0]:en_tr[e0]] == LIFT_PH)[0])
    REST = norm(ztr[key][st_tr[e0] + on0])
    LIFT = norm(ztr[key][st_tr[e0] + on0 + 1:en_tr[e0]])
    templates[None] = REST[None].expand(7, -1)

    def classify(seg):
        """seg: (B, 5, E) the 5 moving frames of a block -> (name list, best distance)."""
        names, best = list(templates), []
        d = []
        for pr in names:
            T = templates[pr]
            d.append(torch.stack([(seg - T[1 - lag:6 - lag]).pow(2).mean((1, 2)) for lag in (0, 1)]).min(0).values)
        d = torch.stack(d, 1)                                   # (B, n_templates)
        i = d.argmin(1)
        return [names[k] for k in i.tolist()], d.min(1).values

    # ---- the decoder's "ball if we lift now?" (as build_self_slot0_labels.py)
    F, dec = int(model.fut_len), model.reward_decoder
    tx = [REST] * (IDLE + 1) + list(LIFT)
    ta = [A_HOLD] * IDLE + [A_LIFT] * (1 + len(LIFT))
    while len(tx) < F - 1:
        tx.append(LIFT[-1]); ta.append(A_LIFT)
    TX, TA, READ = torch.stack(tx[:F - 1]), torch.stack(ta[:F - 1]), 1 + IDLE + 1

    @torch.no_grad()
    def lift_now(z):
        b = len(z)
        X = torch.cat([REST.expand(b, 1, -1), TX[None].expand(b, -1, -1)], 1)
        A = torch.cat([A_HOLD.expand(b, 1, -1), TA[None].expand(b, -1, -1)], 1)
        with autocast(**amp):
            return torch.sigmoid(dec.ball_logit(dec(z, X, A)).float()[:, READ])

    @torch.no_grad()
    def imagine_block(z, code):
        out = []
        with autocast(**amp):
            for _ in range(L):
                z = mcl.step(z, A_HOLD.expand(len(z), -1), code).float()
                out.append(z)
        zs = torch.stack(out, 1)
        with autocast(**amp):
            emb = model.readout(zs).float()                  # (B, L, E)
        return z, emb

    # ---- calibrate "clear": how far REAL test blocks (posterior -> readout) are from their own template
    zev = zarr.open_group(str(ROOT / args.zarr), mode="r")
    st_ev, en_ev = ep_bounds(zev)
    ev_seqs, ev_ph, ev_slot = sequences(ROOT / args.zarr), zev["phase"][:], zev["ball_slot"][:]
    Z0, truth, kinds, real_d, real_ok, chain_starts = [], [], [], [], [], []
    for e in range(len(en_ev)):
        s, n = int(st_ev[e]), int(en_ev[e] - st_ev[e])
        p = ev_ph[s:s + n]
        h0, on, k = int(np.flatnonzero(p == HOLD_PH)[0]), int(np.flatnonzero(p == LIFT_PH)[0]), len(ev_seqs[e][1])
        dense = dense_ball_slot(ev_slot[s:s + n][None].astype(np.int64), np.array([n]))[0]
        Z = posterior(zev, s, n)
        b0 = h0 + MOVE
        starts = [(b0 + SWAP * j, f"before swap {j + 1}") for j in range(k)] + ([(b0 + SWAP * k, "first idle")] if b0 + SWAP * k < on - 1 else [])
        for t, kd in starts:
            Z0.append(Z[t]); truth.append(int(dense[t])); kinds.append(kd)
        chain_starts.append((Z[b0], int(dense[b0])))
        if k:                                                   # real blocks, for the "clear" threshold
            with torch.no_grad(), autocast(**amp):
                emb = model.readout(Z[None, b0 + 1:b0 + 1 + SWAP * k]).float()[0]
            for j, pr in enumerate(ev_seqs[e][1]):
                nm, dist = classify(emb[SWAP * j:SWAP * j + 5][None])
                real_d.append(float(dist[0])); real_ok.append(nm[0] == pr)
    real_d = np.array(real_d)
    thr = float(np.quantile(real_d, 0.99))
    print(f"real test swap blocks named correctly by the template match: {np.mean(real_ok):.3f} (n={len(real_ok)}); "
          f"'clear' threshold = 99th percentile of their distance = {thr:.4f}")
    Z0 = torch.stack(Z0)
    truth = np.array(truth)
    with torch.no_grad():
        p_start = lift_now(Z0).cpu().numpy()
    print(f"{len(Z0)} start points; the posterior's own call at the start is right {np.mean((p_start > 0.5) == (truth == 0)):.3f}")

    def score(z_end, emb, start_slot):
        nm, dist = classify(emb[:, :5])
        p = lift_now(z_end).cpu().numpy()
        rows = []
        for i in range(len(nm)):
            pr = "unclear" if float(dist[i]) > thr else nm[i]
            sl = start_slot[i]
            if pr not in ("unclear", None) and sl in pr:
                sl = pr[1] if sl == pr[0] else pr[0]
            rows.append((pr, sl, bool(p[i] > 0.5), (p[i] > 0.5) == (sl == 0)))
        return rows

    report = {"threshold": thr, "real_named_correctly": float(np.mean(real_ok))}

    def table(rows, title):
        print(f"\n{title}")
        print(f"  {'imagined block':12s} {'n':>5s} {'share':>6s} {'decoder right':>14s}   (should be ball / says ball)")
        by = defaultdict(list)
        for r in rows:
            by[name(r[0])].append(r)
        out = {}
        for nm_ in ("0<->1", "0<->2", "1<->2", "no swap", "unclear"):
            g = by.get(nm_, [])
            if not g:
                continue
            right = np.mean([r[3] for r in g]) if nm_ != "unclear" else float("nan")
            sb = np.mean([r[1] == 0 for r in g]); sa = np.mean([r[2] for r in g])
            print(f"  {nm_:12s} {len(g):5d} {len(g) / len(rows):6.2f} {right:14.3f}   ({sb:.2f} / {sa:.2f})")
            out[nm_] = {"n": len(g), "share": len(g) / len(rows), "right": None if nm_ == "unclear" else float(right)}
        clear = [r for r in rows if r[0] != "unclear"]
        tot = np.mean([r[3] for r in clear]) if clear else float("nan")
        print(f"  {'ALL CLEAR':12s} {len(clear):5d} {len(clear) / len(rows):6.2f} {tot:14.3f}")
        out["all clear"] = {"n": len(clear), "share": len(clear) / len(rows), "right": float(tot)}
        return out

    # 1. sampled code, one block
    g = torch.Generator(device=dev).manual_seed(0)
    rows = []
    with torch.no_grad():
        for i in range(0, len(Z0), 256):
            z = Z0[i:i + 256]
            with autocast(**amp):
                pr = torch.softmax(mcl.code_logits(z, A_HOLD.expand(len(z), -1)).float(), -1)
            c = torch.multinomial(pr, 1, generator=g).squeeze(-1)
            z_end, emb = imagine_block(z, c)
            rows += score(z_end, emb, truth[i:i + 256])
    report["sampled"] = table(rows, "ONE IMAGINED BLOCK, code sampled by the prior (all start points)")
    start_ok = (p_start > 0.5) == (truth == 0)
    for lab, m in (("start call RIGHT", start_ok), ("start call WRONG", ~start_ok)):
        clear = [rows[i] for i in np.flatnonzero(m) if rows[i][0] != "unclear"]
        nos = [r for r in clear if r[0] is None]
        sw = [r for r in clear if r[0] is not None]
        print(f"    {lab}: n {int(m.sum())}, decoder right after the block {np.mean([r[3] for r in clear]):.3f} "
              f"(after an imagined swap {np.mean([r[3] for r in sw]) if sw else float('nan'):.3f}, n {len(sw)}; "
              f"after no swap {np.mean([r[3] for r in nos]) if nos else float('nan'):.3f}, n {len(nos)})")
    for kd in sorted(set(kinds)):
        m = [i for i, k in enumerate(kinds) if k == kd]
        clear = [rows[i] for i in m if rows[i][0] != "unclear"]
        print(f"    from '{kd}': n {len(m)}, clear {len(clear) / len(m):.2f}, decoder right on clear "
              f"{np.mean([r[3] for r in clear]) if clear else float('nan'):.3f}")

    # 2. each code forced
    print("\nEACH CODE FORCED, one block (all start points):")
    print(f"  {'code':5s} " + " ".join(f"{n_:>8s}" for n_ in ("0<->1", "0<->2", "1<->2", "no swap", "unclear")) + "   decoder right (clear)")
    report["forced"] = {}
    for c_ in range(C):
        rows_c = []
        with torch.no_grad():
            for i in range(0, len(Z0), 256):
                z = Z0[i:i + 256]
                z_end, emb = imagine_block(z, torch.full((len(z),), c_, device=dev, dtype=torch.long))
                rows_c += score(z_end, emb, truth[i:i + 256])
        cnt = Counter(name(r[0]) for r in rows_c)
        clear = [r for r in rows_c if r[0] != "unclear"]
        right = np.mean([r[3] for r in clear]) if clear else float("nan")
        print(f"  c{c_:<4d} " + " ".join(f"{cnt.get(n_, 0) / len(rows_c):8.2f}" for n_ in ("0<->1", "0<->2", "1<->2", "no swap", "unclear"))
              + f"   {right:.3f}")
        report["forced"][f"c{c_}"] = {"mix": {k: v / len(rows_c) for k, v in cnt.items()}, "right": float(right)}

    # 3. chains: drift over several sampled blocks
    print(f"\nCHAINS: {args.chain} sampled blocks in a row from the start before the first swap "
          f"({len(chain_starts)} episodes x {args.chain_samples} samples); decoder right after block b, "
          f"chains whose blocks are all clear so far")
    zc = torch.stack([z for z, _ in chain_starts]).repeat(args.chain_samples, 1, 1)
    slot = np.tile([s_ for _, s_ in chain_starts], args.chain_samples)
    alive = np.ones(len(zc), bool)
    report["chain"] = []
    g = torch.Generator(device=dev).manual_seed(1)
    for b in range(1, args.chain + 1):
        rows_b = []
        zn = []
        with torch.no_grad():
            for i in range(0, len(zc), 256):
                z = zc[i:i + 256]
                with autocast(**amp):
                    pr = torch.softmax(mcl.code_logits(z, A_HOLD.expand(len(z), -1)).float(), -1)
                c = torch.multinomial(pr, 1, generator=g).squeeze(-1)
                z_end, emb = imagine_block(z, c)
                rows_b += score(z_end, emb, slot[i:i + 256])
                zn.append(z_end)
        zc = torch.cat(zn)
        alive &= np.array([r[0] != "unclear" for r in rows_b])
        slot = np.array([r[1] for r in rows_b])
        ok = np.array([r[3] for r in rows_b])
        swaps = Counter(name(r[0]) for r in rows_b)
        print(f"  after block {b}: clear chains {alive.mean():.2f}, decoder right {ok[alive].mean():.3f} "
              f"| this block: " + ", ".join(f"{k} {v / len(rows_b):.2f}" for k, v in sorted(swaps.items())))
        report["chain"].append({"block": b, "clear": float(alive.mean()), "right": float(ok[alive].mean())})
    json.dump(report, open(ROOT / args.out, "w"), indent=1)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
