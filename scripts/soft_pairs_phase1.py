"""SOFT pairs: how alike are two memories, judged by the decoder on K real futures they are both run through.

    pixi run python scripts/soft_pairs_phase1.py [--K 8] [--n_pairs 40000] [--out data/pairs_soft_all3u.npz]

Only frozen stage-1 models (Naive-U and its seed-1 twin), frame embeddings and action flags are used:

  pair      x, y = frames of two different training episodes whose last w+1 embeddings and actions are equal
  futures   K real continuations from that same-looking moment: y's own, plus K-1 from other episodes c with a frame
            u matching the same window; each runs from u to just before c's next non-hold action (no lift, so
            nothing about the ball is revealed), at most --max_len frames
  tests     run the posterior from x's memory and from y's memory through each future's real frames, and ask the
            decoder "lift on the next frame?" at every REST frame along it (the frame u itself included); the
            answer d is the mean of the two models (pairs where the models differ by > --twin_tol are dropped)
  weight    s = min over all tests of (1 - |d_x - d_y|): high only if x and y agree on every future, low as soon as
            one future tells them apart

Training (align_label: soft) pulls each pair together with weight s and pushes it apart (uniformity) with weight 1-s.
The hidden slot and the privileged belief labels are used ONLY to score s.
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import zarr  # noqa: E402
from torch.amp import autocast  # noqa: E402

from train_mcl_prior_phase1 import load_model  # noqa: E402

HOLD, LIFT = np.array([1., 0.], np.float32), np.array([0., 1.], np.float32)
CERTAIN = (0, 1, 2)


def main():
    ap = argparse.ArgumentParser()
    A = "logdir/world_model/smt_phase1_all3u"
    ap.add_argument("--runs", nargs="+", default=[f"{A}/naive/checkpoints/wm_latest.pt",
                                                  f"{A}/naive_twin/checkpoints/wm_latest.pt"])
    ap.add_argument("--zarr", default="data/memtest4_all3u.zarr")
    ap.add_argument("--n_train", type=int, default=768)
    ap.add_argument("--n_shown", type=int, default=384, help="scoring only")
    ap.add_argument("--n_pairs", type=int, default=40000)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--w", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=30)
    ap.add_argument("--twin_tol", type=float, default=0.3)
    ap.add_argument("--quant", type=float, default=1e-3)
    ap.add_argument("--priv", default="data/labels_belief_all3u_privileged.npz")
    ap.add_argument("--out", default="data/pairs_soft_all3u.npz")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--dump", default=None, help="also save every test (both models, both members) for analysis")
    ap.add_argument("--rule", choices=["soft", "js", "jsweight"], default="soft",
                    help="js: POSITIVE if max Jensen-Shannon over all tests < --pos_js, NEGATIVE if > --neg_js")
    ap.add_argument("--pos_js", type=float, default=0.05)
    ap.add_argument("--neg_js", type=float, default=0.5)
    ap.add_argument("--tests", choices=["rest", "lifts"], default="rest",
                    help="lifts: no phase/rest rules -- x, y any frames; futures run through whatever really happens "
                         "(swaps, peeks, lifts); one test at every real lift onset, read before the cup rises")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device
    rng = np.random.default_rng(0)
    models = [load_model(r, dev) for r in args.runs]
    m0 = models[0]
    amp = dict(device_type=dev.split(":")[0], dtype=m0.amp_dtype)

    zg = zarr.open_group(str(ROOT / args.zarr), mode="r")
    ends = zg["episode_ends"][:]; starts = np.r_[0, ends[:-1]]
    n_ep = args.n_train; G = int(ends[n_ep - 1])
    act = zg["action"][:G].astype(np.float32)
    raw = np.asarray(zg[m0.embed_key][:G], np.float32)
    slot = zg["ball_slot"][:G].astype(int)                                       # scoring only
    E = (torch.from_numpy(raw).to(dev) - m0.embed_mean) / m0.embed_std
    ep_of = np.repeat(np.arange(n_ep), np.diff(np.r_[0, ends[:n_ep]])); t_of = np.arange(G) - starts[ep_of]
    PA = np.zeros_like(act)
    for e in range(n_ep):
        PA[starts[e] + 1:ends[e]] = act[starts[e]:ends[e] - 1]
    PAt = torch.from_numpy(PA).to(dev)
    hold_in = (PA == HOLD).all(1)
    onset = next(t for t in range(1, G) if (act[t] == LIFT).all() and (act[t - 1] == HOLD).all())
    REST = E[onset]
    rest_like = ((E - REST).abs().amax(-1) < args.quant).cpu().numpy() & hold_in   # "lift now?" is valid here

    Zs = []
    for m in models:
        Z = torch.empty(G, m.posterior.n_registers, m.posterior.d_model, device=dev)
        with torch.no_grad():
            for e in range(n_ep):
                s, en = int(starts[e]), int(ends[e])
                f = torch.zeros(1, en - s, dtype=torch.bool, device=dev); f[0, 0] = True
                Z[s:en] = m.encode_windows(E[None, s:en], PAt[None, s:en], f)[0].float()
        Zs.append(Z)
    F_ = int(m0.fut_len)
    Aq = torch.from_numpy(np.stack([HOLD] + [LIFT] * (F_ - 1))).to(dev)                 # lift on the next frame

    FQ = 3          # the decoder is causal: read-out 2 sees only future tokens 0-1, so 3 tokens give the same answer

    @torch.no_grad()
    def ask(m, z, frame_idx, fq=FQ):
        out = []
        for i in range(0, len(z), 2048):
            zz, fi = z[i:i + 2048], frame_idx[i:i + 2048]
            X = E[fi][:, None].expand(-1, fq, -1)
            # fp32: in bf16 the answer moves by ~1e-2 with sequence length (rounding), in fp32 truncation is exact
            out.append(torch.sigmoid(m.reward_decoder.ball_logit(m.reward_decoder(
                zz.float(), X.float(), Aq[None, :fq].expand(len(zz), -1, -1))).float()[:, 2]))
        return torch.cat(out)

    At = torch.from_numpy(act).to(dev)

    @torch.no_grad()
    def ask_real(m, z, f):
        """A REAL lift: memory z has absorbed frames up to f, the lift starts at f+1. Tokens are the real
        (e_f, a_f), (e_{f+1}, a_{f+1} = lift); read-out 2 = the first rising frame, which it has not seen."""
        out = []
        for i in range(0, len(z), 2048):
            zz, fi = z[i:i + 2048], f[i:i + 2048]
            idx = torch.stack([fi, fi + 1, fi + 1], 1)                    # token 2 is never seen by read-out 2
            out.append(torch.sigmoid(m.reward_decoder.ball_logit(m.reward_decoder(
                zz.float(), E[idx].float(), At[idx].float())).float()[:, 2]))
        return torch.cat(out)

    probe = torch.randint(0, G, (256,), device=dev)
    gap0 = float((ask(m0, Zs[0][probe], probe) - ask(m0, Zs[0][probe], probe, fq=F_)).abs().max())
    print(f"truncated query vs full 60-step query (fp32): max difference {gap0:.1e}")
    assert gap0 < 1e-4, gap0

    # future length from every frame: frames u+1 .. u+L all reached by hold actions, inside the episode
    flen = np.zeros(G, int)
    for e in range(n_ep):
        s, en = int(starts[e]), int(ends[e])
        for u in range(en - 1, s - 1, -1):
            flen[u] = 0 if u + 1 >= en or not hold_in[u + 1] else min(args.max_len, flen[u + 1] + 1)
    if args.tests == "lifts":
        # test frames f: the lift starts at f+1 (from the actions alone). From u the chain runs to the LAST such f
        # within max_len frames (-1: no lift ahead, no test); every such f on the way is a test
        pre_onset = np.zeros(G, bool)
        for e in range(n_ep):
            s, en = int(starts[e]), int(ends[e])
            for f in range(s, en - 1):
                pre_onset[f] = (act[f + 1] == LIFT).all() and not (act[f] == LIFT).all()
        rest_like = pre_onset
        flen = np.full(G, -1)
        for e in range(n_ep):
            s, en = int(starts[e]), int(ends[e])
            for u in range(s, en):
                fs = np.flatnonzero(pre_onset[u:min(en, u + args.max_len + 1)])
                flen[u] = fs[-1] if len(fs) else -1

    qE = np.round(((raw - m0.embed_mean.cpu().numpy()) / m0.embed_std.cpu().numpy()) / args.quant).astype(np.int32)
    pool = defaultdict(list)
    for t in range(G):
        if (hold_in[t] or args.tests == "lifts") and t_of[t] >= args.w:
            pool[qE[t - args.w:t + 1].tobytes() + PA[t - args.w:t + 1].tobytes()].append(t)
    pool = {k: np.array(v) for k, v in pool.items() if len(v) >= 3}
    keys = list(pool)
    sizes = np.array([len(pool[k]) for k in keys], float)
    print(f"{G} frames, {len(keys)} distinct windows with >= 3 frames; rest frames {rest_like.sum()}")

    # ---- sample pairs and their K futures
    pairs, futs = [], []
    for _ in range(args.n_pairs * 3):
        if len(pairs) >= args.n_pairs:
            break
        k = keys[rng.choice(len(keys), p=sizes / sizes.sum())]                        # x uniform over frames
        P = pool[k]
        x = rng.choice(P)
        ys = P[ep_of[P] != ep_of[x]]
        if not len(ys):
            continue
        y = rng.choice(ys)
        cs = P[(ep_of[P] != ep_of[x]) & (ep_of[P] != ep_of[y])]
        us = [y] + list(rng.choice(cs, min(args.K - 1, len(cs)), replace=False))
        if args.tests == "lifts":
            us = [u for u in us if flen[u] >= 0]
            if not us:
                continue
        elif not any(rest_like[u:u + flen[u] + 1].any() for u in us):
            continue
        pairs.append((x, y)); futs.append(us)
    pairs = np.array(pairs)
    print(f"{len(pairs)} pairs, {np.mean([len(f) for f in futs]):.1f} futures each")

    # ---- jobs: (start memory frame, future start u, length), both members of each pair through each future
    J_start, J_u, J_len, J_pair, J_side, J_fut = [], [], [], [], [], []
    for i, ((x, y), us) in enumerate(zip(pairs, futs)):
        for j, u in enumerate(us):
            for side, g in ((0, x), (1, y)):
                J_start.append(g); J_u.append(u); J_len.append(flen[u]); J_pair.append(i); J_side.append(side); J_fut.append(j)
    J_start, J_u, J_len = np.array(J_start), np.array(J_u), np.array(J_len)
    J_pair, J_side, J_fut = np.array(J_pair), np.array(J_side), np.array(J_fut)
    Lmax = int(J_len.max())
    print(f"{len(J_start)} chains, future length mean {J_len.mean():.1f}, max {Lmax}")

    P_all = np.full((len(models), len(J_start), Lmax + 1), np.nan, np.float32)        # p at each step, NaN if not rest
    for mi, (m, Zm) in enumerate(zip(models, Zs)):
        for L in np.unique(J_len):
            idx = np.flatnonzero(J_len == L)
            for b0 in range(0, len(idx), args.batch):
                b = idx[b0:b0 + args.batch]
                z = Zm[torch.from_numpy(J_start[b]).to(dev)]
                u = torch.from_numpy(J_u[b]).to(dev)
                with torch.no_grad():
                    for s in range(0, int(L) + 1):
                        if s:
                            with autocast(**amp):
                                z = m.posterior.step(z, E[u + s], PAt[u + s]).float()
                        ok = rest_like[J_u[b] + s]
                        if ok.any():
                            okd = torch.from_numpy(ok).to(dev)
                            q = ask_real if args.tests == "lifts" else ask
                            P_all[mi, b[ok], s] = q(m, z[okd], u[okd] + s).cpu().numpy()
        print(f"model {mi}: chains done", flush=True)

    D = np.nanmean(P_all, 0)                                                            # mean over models
    if args.rule in ("js", "jsweight"):
        # every test is a pair of coins (ball with probability d_x, d_y); JS in bits: 0 identical, 1 for 0 vs 1
        def Hb(q):
            q = np.clip(q, 1e-12, 1 - 1e-12)
            return -(q * np.log2(q) + (1 - q) * np.log2(1 - q))
        rx = np.flatnonzero(J_side == 0); ry = rx + 1
        assert (J_pair[ry] == J_pair[rx]).all()
        dx, dy = D[rx], D[ry]
        okt = ~np.isnan(dx) & ~np.isnan(dy)
        js = np.where(okt, Hb((dx + dy) / 2) - (Hb(dx) + Hb(dy)) / 2, -1.0).max(1)
        mx = np.full(len(pairs), -1.0); np.maximum.at(mx, J_pair[rx], js)
        nt = np.bincount(J_pair[rx], weights=okt.sum(1), minlength=len(pairs))
        has = nt > 0
        pos, neg = has & (mx < args.pos_js), has & (mx > args.neg_js)
        x_, y_ = pairs[:, 0], pairs[:, 1]
        same = slot[x_] == slot[y_]; shown = (ep_of[x_] < args.n_shown) & (ep_of[y_] < args.n_shown)
        if args.dump:                                                                   # every test, for analysis
            pr = np.load(ROOT / args.priv)
            cl = {int(starts[int(e)] + int(t)): int(c) for e, t, c in zip(pr["episode"], pr["t"], pr["label"])}
            phase = zg["phase"][:G].astype(int); Enp = E.cpu().numpy()
            moving = np.r_[False, (np.abs(Enp[1:] - Enp[:-1]).max(1) > 1e-6)]
            np.savez(ROOT / args.dump, P_all=P_all, J_pair=J_pair, J_side=J_side, J_fut=J_fut, J_u=J_u, J_len=J_len,
                     x=x_, y=y_, slot_x=slot[x_], slot_y=slot[y_], shown_x=ep_of[x_] < args.n_shown,
                     shown_y=ep_of[y_] < args.n_shown, phase_x=phase[x_], moving_x=moving[x_],
                     cls_x=np.array([cl.get(int(v), -1) for v in x_]), cls_y=np.array([cl.get(int(v), -1) for v in y_]))
            print(f"dump -> {args.dump}")
        print(f"JS rule: {has.sum()} pairs with tests (median {np.median(nt[has]):.0f} tests); POSITIVE (max JS < {args.pos_js}) "
              f"{pos.sum()}, NEGATIVE (max JS > {args.neg_js}) {neg.sum()}, ignored {(has & ~pos & ~neg).sum()}")
        print(f"  scoring only: positives truly same cup {same[pos].mean():.4f} (shown starts {same[pos & shown].mean():.4f}); "
              f"negatives truly different cup {(~same[neg]).mean():.4f} (shown starts {(~same[neg & shown]).mean():.4f})")
        if args.rule == "jsweight":
            # no positives / negatives: EVERY pair with a test is pulled with weight w = 1 - max JS (bits, in [0, 1])
            w = np.clip(1.0 - mx[has], 0.0, 1.0); hx, hy = x_[has], y_[has]
            for nm, m in (("same cup (shown starts)", same[has] & shown[has]), ("different cup (shown starts)", ~same[has] & shown[has])):
                print(f"  scoring only: {nm:30s} n {m.sum():6d}  mean w {w[m].mean():.3f}  w > 0.95: {np.mean(w[m] > .95):.3f}  w < 0.05: {np.mean(w[m] < .05):.3f}")
            e0 = np.zeros(0, np.int64)
            np.savez(ROOT / args.out, x_episode=ep_of[hx], x_t=t_of[hx], y_episode=ep_of[hy], y_t=t_of[hy], w=w,
                     neg_x_episode=e0, neg_x_t=e0, neg_y_episode=e0, neg_y_t=e0, neg_w=np.zeros(0),
                     max_js=mx[has], same_slot=same[has], shown=shown[has])
            print(f"-> {args.out}: {has.sum()} pairs, every one pulled with weight w = 1 - max JS")
            return
        np.savez(ROOT / args.out, x_episode=ep_of[x_[pos]], x_t=t_of[x_[pos]], y_episode=ep_of[y_[pos]], y_t=t_of[y_[pos]],
                 neg_x_episode=ep_of[x_[neg]], neg_x_t=t_of[x_[neg]], neg_y_episode=ep_of[y_[neg]], neg_y_t=t_of[y_[neg]],
                 max_js=mx[pos], neg_max_js=mx[neg], slot_right=same[pos], neg_right=~same[neg])
        print(f"-> {args.out}")
        return
    gap = np.nanmax(np.abs(P_all[0] - P_all[-1]), axis=1) if len(models) > 1 else np.zeros(len(J_start))
    s_pair = np.ones(len(pairs)); n_test = np.zeros(len(pairs), int); twin_bad = np.zeros(len(pairs), bool)
    by = defaultdict(dict)
    for r in range(len(J_start)):
        by[J_pair[r]][(J_fut[r], J_side[r])] = r
    for i in range(len(pairs)):
        sims = []
        for j in range(len(futs[i])):
            rx, ry = by[i].get((j, 0)), by[i].get((j, 1))
            dx, dy = D[rx], D[ry]
            ok = ~np.isnan(dx) & ~np.isnan(dy)
            sims.extend(1 - np.abs(dx[ok] - dy[ok]))
            twin_bad[i] |= (np.nan_to_num(gap[rx]) > args.twin_tol) or (np.nan_to_num(gap[ry]) > args.twin_tol)
        n_test[i] = len(sims)
        s_pair[i] = min(sims) if sims else np.nan
    if args.dump:                                                                       # every test, for analysis
        pr = np.load(ROOT / args.priv)
        cl = {int(starts[int(e)] + int(t)): int(c) for e, t, c in zip(pr["episode"], pr["t"], pr["label"])}
        np.savez(ROOT / args.dump, P_all=P_all, J_pair=J_pair, J_side=J_side, J_fut=J_fut, J_u=J_u, J_len=J_len,
                 x=pairs[:, 0], y=pairs[:, 1], slot_x=slot[pairs[:, 0]], slot_y=slot[pairs[:, 1]],
                 shown_x=ep_of[pairs[:, 0]] < args.n_shown, shown_y=ep_of[pairs[:, 1]] < args.n_shown,
                 cls_x=np.array([cl.get(int(v), -1) for v in pairs[:, 0]]), cls_y=np.array([cl.get(int(v), -1) for v in pairs[:, 1]]),
                 s=s_pair, n_test=n_test, twin_bad=twin_bad, twin_tol=args.twin_tol)
        print(f"dump -> {args.dump}")
    keep = (n_test > 0) & ~twin_bad
    x, y, s = pairs[keep, 0], pairs[keep, 1], s_pair[keep]
    print(f"kept {keep.sum()} of {len(pairs)} pairs ({twin_bad.sum()} dropped: the two models disagree > {args.twin_tol}); "
          f"tests per pair median {np.median(n_test[keep]):.0f}")

    # ---- score s against the truth (privileged, report only)
    priv = np.load(ROOT / args.priv)
    cls = {int(starts[int(e)] + int(t)): int(c) for e, t, c in zip(priv["episode"], priv["t"], priv["label"])}
    same_slot = slot[x] == slot[y]
    shown = (ep_of[x] < args.n_shown) & (ep_of[y] < args.n_shown)
    print("\nweight s against the truth (scoring only):")
    for nm, m in (("both shown-start, same cup", shown & same_slot), ("both shown-start, different cup", shown & ~same_slot)):
        print(f"  {nm:34s} n {m.sum():6d}  mean s {s[m].mean():.3f}  s > 0.5: {np.mean(s[m] > 0.5):.3f}")
    bx = np.array([cls.get(int(a), -1) for a in x]); byy = np.array([cls.get(int(b), -1) for b in y])
    lab = (bx >= 0) & (byy >= 0)
    for nm, m in (("same belief, both certain", lab & (bx == byy) & np.isin(bx, CERTAIN)),
                  ("same belief, both uncertain", lab & (bx == byy) & ~np.isin(bx, CERTAIN)),
                  ("different belief", lab & (bx != byy)),
                  ("  ...certain vs certain, different cup", lab & (bx != byy) & np.isin(bx, CERTAIN) & np.isin(byy, CERTAIN)),
                  ("  ...one certain, one uncertain", lab & (bx != byy) & (np.isin(bx, CERTAIN) != np.isin(byy, CERTAIN)))):
        if m.any():
            print(f"  {nm:40s} n {m.sum():6d}  mean s {s[m].mean():.3f}  s > 0.5: {np.mean(s[m] > 0.5):.3f}")

    hi = s >= 0.5
    np.savez(ROOT / args.out, x_episode=ep_of[x[hi]], x_t=t_of[x[hi]], y_episode=ep_of[y[hi]], y_t=t_of[y[hi]], w=s[hi],
             neg_x_episode=ep_of[x[~hi]], neg_x_t=t_of[x[~hi]], neg_y_episode=ep_of[y[~hi]], neg_y_t=t_of[y[~hi]],
             neg_w=s[~hi], same_slot=same_slot[hi], neg_same_slot=same_slot[~hi])
    print(f"-> {args.out}: {hi.sum()} pairs with s >= 0.5 (sampled as 'positives'), {(~hi).sum()} with s < 0.5; "
          f"every pair is pulled with weight s and pushed with weight 1 - s")


if __name__ == "__main__":
    main()
