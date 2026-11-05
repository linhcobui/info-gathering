"""Positive pairs by splicing real futures: walk BACK from a leaf in episode b, walk FORWARD from x on b's frames.

    pixi run python scripts/splice_pairs_phase1.py [--run <stage-1 ckpt>] [--zarr <store>] [--windows 0 1 2]

Only the frozen stage-1 model (posterior updater + reward decoder), frame embeddings and action flags are used:

  leaf      a frame l of episode b that looks like the query's rest frame (the query is itself a splice) and whose
            TEACHER memory makes the decoder say "ball if we lift now?" > p_thr
  walk back y = any earlier frame of b (mid-swap included) with only HOLD actions on (y, l]  -- no lift in
            between, so nothing about the ball is revealed between y and l
  match     x = a frame of ANOTHER episode whose last w+1 frame embeddings and actions equal y's
  walk fwd  start from x's teacher memory, run the posterior through b's real frames e_{y+1} .. e_l
  judge     decoder "lift now?" on that memory > p_thr  ->  (x, y) is a positive pair
  self      the same chain from y's own memory must also say yes, else the test is dropped (the model cannot
            track that future even in its own episode)

"lift now?" = the decoder read-out at lift offset 1 of a canned future (the rest frame, then the lift action on
the next frame: --wait 1).
The decoder is causal, so that read-out sees only frames up to the lift onset (cups down): the rest frame is
taken at a lift onset found by the action flag, and the template's later frames provably cannot matter
(checked at start-up).

The hidden ball slot (and the privileged belief labels) are used ONLY to score. Compared against the
hand-crafted and privileged belief labels of ALL3-U (pairs = two frames with the same label).
"""
import argparse
import json
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

CERTAIN = (0, 1, 2)            # belief classes {0} {a} {b}; 3..6 are the uncertain sets


def main():
    ap = argparse.ArgumentParser()
    N = "logdir/world_model/smt_phase1_all3u/naive"
    ap.add_argument("--run", default=f"{N}/checkpoints/wm_latest.pt")
    ap.add_argument("--zarr", default="data/memtest4_all3u.zarr")
    ap.add_argument("--n_train", type=int, default=768)
    ap.add_argument("--n_shown", type=int, default=384, help="episodes < this have a shown start (scoring only)")
    ap.add_argument("--windows", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--x_per_test", type=int, default=32)
    ap.add_argument("--p_thr", type=float, default=0.9)
    ap.add_argument("--quant", type=float, default=1e-3, help="embedding match tolerance (normalised units)")
    ap.add_argument("--hand", default="data/labels_belief_all3u_handcrafted.npz")
    ap.add_argument("--priv", default="data/labels_belief_all3u_privileged.npz")
    ap.add_argument("--out", default=f"{N}/splice_pairs")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--sure_start", action="store_true",
                    help="use only episodes where the decoder is already sure (p<0.1 or >0.9) at the first still hold frame")
    ap.add_argument("--twin", default=None, help="a second stage-1 checkpoint: every decoder answer must agree")
    ap.add_argument("--neg_thr", type=float, default=0.01,
                    help="NEGATIVE pair: every model and lift timing says p < this after the splice (same events, no ball)")
    ap.add_argument("--wait", type=int, nargs="+", default=[1, 4],
                    help="lift action this many frames after the query frame; several = every one must say yes")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    log = open(out / "splice_pairs.log", "w")

    def say(msg=""):
        print(msg, flush=True)
        log.write(msg + "\n"); log.flush()

    models = [load_model(args.run, dev)] + ([load_model(args.twin, dev)] if args.twin else [])
    model = models[0]
    amp = dict(device_type=dev.split(":")[0], dtype=model.amp_dtype)
    zg = zarr.open_group(str(ROOT / args.zarr), mode="r")
    ends = zg["episode_ends"][:]
    starts = np.concatenate([[0], ends[:-1]])
    n_ep = min(args.n_train, len(ends))
    G = int(ends[n_ep - 1])                                        # frames of the training episodes
    act = zg["action"][:G].astype(np.float32)
    raw = np.asarray(zg[model.embed_key][:G], np.float32)
    slot = zg["ball_slot"][:G].astype(int)                         # scoring only
    E = (torch.from_numpy(raw).to(dev) - model.embed_mean) / model.embed_std
    ep_of = np.repeat(np.arange(n_ep), np.diff(np.r_[0, ends[:n_ep]]))
    t_of = np.arange(G) - starts[ep_of]
    HOLD, LIFT = np.array([1., 0.], np.float32), np.array([0., 1.], np.float32)
    PA = np.zeros_like(act)                                        # action that PRODUCED frame t (model input)
    for e in range(n_ep):
        s, en = int(starts[e]), int(ends[e])
        PA[s + 1:en] = act[s:en - 1]
    PAt = torch.from_numpy(PA).to(dev)
    hold_in = (PA == HOLD).all(1)                                  # frame t was reached by a hold action

    # ---- teacher memory at every frame, for every model
    for m in models[1:]:
        assert torch.allclose(m.embed_mean, model.embed_mean) and torch.allclose(m.embed_std, model.embed_std)
    Zs = []
    for m in models:
        Z = torch.empty(G, m.posterior.n_registers, m.posterior.d_model, device=dev)
        with torch.no_grad():
            for e in range(n_ep):
                s, en = int(starts[e]), int(ends[e])
                first = torch.zeros(1, en - s, dtype=torch.bool, device=dev); first[0, 0] = True
                Z[s:en] = m.encode_windows(E[None, s:en], PAt[None, s:en], first)[0].float()
        Zs.append(Z)
    Z = Zs[0]

    # ---- "lift now?" from the decoder: rest frame at a lift onset found by the action flag
    onset = next(t for t in range(1, G) if (act[t] == LIFT).all() and (act[t - 1] == HOLD).all())
    REST = E[onset]
    # "lift now?": the lift action `wait` frames after the query frame (wait 1 = the next frame). Waiting longer
    # than episodes ever do is out of distribution near the end of the hold (the old IDLE = 3 query, wait 4).
    F_ = int(model.fut_len)
    A_qs = {w: torch.from_numpy(np.stack([HOLD] * w + [LIFT] * (F_ - w))).to(dev) for w in args.wait}
    X_q = REST.expand(F_, -1)

    @torch.no_grad()
    def lift_now(z, X=None, m=None, waits=None, reduce=torch.minimum):
        """min over the requested waits ("yes" only if every version says yes); reduce=torch.maximum for "no"."""
        dec = (m or model).reward_decoder
        X = X_q[None].expand(len(z), -1, -1) if X is None else X
        p = None
        for w in (waits or args.wait):
            with autocast(**amp):
                q = torch.sigmoid(dec.ball_logit(dec(z, X, A_qs[w][None].expand(len(z), -1, -1))).float()[:, w + 1])
            p = q if p is None else reduce(p, q)
        return p

    zz = Z[torch.randint(0, G, (64,), device=dev)]
    for w in args.wait:
        Xr = X_q[None].repeat(64, 1, 1)
        Xr[:, w + 1:] = E[torch.randint(0, G, (64, F_ - w - 1), device=dev)]
        gap = float((lift_now(zz, waits=[w]) - lift_now(zz, Xr, waits=[w])).abs().max())
        say(f"causality check (wait {w}): template frames after the onset changed the read-out by {gap:.1e}")
        assert gap < 1e-4

    Ps = [torch.cat([lift_now(Zm[i:i + args.batch], m=m) for i in range(0, G, args.batch)]).cpu().numpy()
          for m, Zm in zip(models, Zs)]
    P = np.min(Ps, 0)                                              # "yes" only if every model says yes
    if args.twin:
        say(f"twin: models agree (both > 0.5 or both < 0.5) on {np.mean((Ps[0] > 0.5) == (Ps[1] > 0.5)):.3f} of frames")
    # the query is itself a splice (current frame, then rest, rest, rest, lift): valid only where the current frame
    # IS the template's rest frame -- the same embedding-match rule as every other splice
    rest_like = ((E - REST).abs().amax(-1) < args.quant).cpu().numpy()
    leaf = hold_in & rest_like & (P > args.p_thr)
    say(f"{n_ep} training episodes, {G} frames; leaves (hold, frame == rest frame, decoder p > {args.p_thr}): {leaf.sum()}")

    still = np.zeros(G, bool)
    still[1:] = (E[1:] - E[:-1]).abs().amax(-1).cpu().numpy() < 1e-6
    still[starts[:n_ep]] = False

    sure = np.ones(n_ep, bool)
    if args.sure_start:                                            # decoder-only "was the start shown?" (as hand-crafted)
        for e in range(n_ep):
            f = [t for t in range(int(starts[e]), int(ends[e])) if still[t] and hold_in[t]]
            if not f:
                sure[e] = False; continue
            qs = [float(lift_now(Zm[f[0]][None], m=m, waits=[w])[0]) for m, Zm in zip(models, Zs) for w in args.wait]
            sure[e] = all(q < 0.1 for q in qs) or all(q > 0.9 for q in qs)   # every model and question equally sure
        leaf &= sure[ep_of]
        say(f"--sure_start: {sure.sum()} of {n_ep} episodes kept (scoring only: {sure[:args.n_shown].sum()} of the "
            f"{args.n_shown} shown starts, {sure[args.n_shown:].sum()} of the {n_ep - args.n_shown} closed starts)")

    # ---- tests (y, l) in every episode: nearest leaf at or after y with only hold actions on (y, l]
    tests = []                                                     # (y_global, l_global)
    for e in range(n_ep):
        s, en = int(starts[e]), int(ends[e])
        for y in range(s, en):
            if not hold_in[y] or not sure[e]:
                continue
            for l in range(y, en):
                if l > y and not hold_in[l]:
                    break
                if leaf[l]:
                    tests.append((y, l)); break
    tests = np.array(tests)
    say(f"tests (y, leaf l): {len(tests)}; frames from y to l: median {np.median(tests[:, 1] - tests[:, 0]):.0f}, "
        f"max {np.max(tests[:, 1] - tests[:, 0])}")

    qE = np.round(((raw - model.embed_mean.cpu().numpy()) / model.embed_std.cpu().numpy()) / args.quant).astype(np.int32)

    def key(t, w):
        if t_of[t] < w:
            return None
        return qE[t - w:t + 1].tobytes() + PA[t - w:t + 1].tobytes()

    @torch.no_grad()
    def run_chains(xg, yg, L):
        """Memory of x, posterior through frames y+1..y+L, then lift_now. Grouped by L."""
        p = np.full(len(xg), np.inf, np.float32)
        pmax = np.full(len(xg), -np.inf, np.float32)
        for m, Zm in zip(models, Zs):                              # every model runs its own chain
            for d in np.unique(L):
                idx = np.flatnonzero(L == d)
                for i in range(0, len(idx), args.batch):
                    b = idx[i:i + args.batch]
                    z = Zm[torch.from_numpy(xg[b]).to(dev)]
                    y = torch.from_numpy(yg[b]).to(dev)
                    with autocast(**amp):
                        for s in range(1, int(d) + 1):
                            z = m.posterior.step(z, E[y + s], PAt[y + s]).float()
                    p[b] = np.minimum(p[b], lift_now(z, m=m).cpu().numpy())
                    pmax[b] = np.maximum(pmax[b], lift_now(z, m=m, reduce=torch.maximum).cpu().numpy())
        return p, pmax

    hand, priv = np.load(ROOT / args.hand), np.load(ROOT / args.priv)
    table_g = np.array([int(starts[int(e)] + int(t)) for e, t in zip(priv["episode"], priv["t"])])
    cls_g = {int(g): int(c) for g, c in zip(table_g, priv["label"])}

    results = {}
    for w in args.windows:
        pool = defaultdict(list)
        for t in range(G):
            if hold_in[t]:
                k = key(t, w)
                if k is not None:
                    pool[k].append(t)
        pool = {k: np.array(v) for k, v in pool.items()}
        X, Y, L, self_ok = [], [], [], []
        for y, l in tests:
            k = key(y, w)
            if k is None:
                continue
            cand = pool[k][(ep_of[pool[k]] != ep_of[y]) & sure[ep_of[pool[k]]]]
            if len(cand) == 0:
                continue
            take = cand if len(cand) <= args.x_per_test else rng.choice(cand, args.x_per_test, replace=False)
            X.append(np.r_[y, take]); Y.append(np.full(len(take) + 1, y)); L.append(np.full(len(take) + 1, l - y))
        X, Y, L = np.concatenate(X), np.concatenate(Y), np.concatenate(L)
        p, pmax = run_chains(X, Y, L)
        is_self = X == Y
        ok_test = {int(y): bool(pp > args.p_thr) for y, pp in zip(Y[is_self], p[is_self])}
        keep = ~is_self & np.array([ok_test[int(y)] for y in Y])
        pos = keep & (p > args.p_thr)
        xs, ys, ls = X[pos], Y[pos], L[pos]
        neg = keep & (pmax < args.neg_thr)                         # same real events, confidently no ball -> different
        nx, ny = X[neg], Y[neg]
        neg_right = slot[nx] != slot[ny]                           # scoring only
        say(f"  NEGATIVE pairs (every model and timing p < {args.neg_thr}): {neg.sum()}, precision (hidden cups differ) "
            f"{neg_right.mean():.4f}; both still {np.mean(still[nx] & still[ny]):.2f}")
        right = slot[xs] == slot[ys]
        leaf_ok = slot[ys + ls] == 0                               # scoring only
        both_shown = (ep_of[xs] < args.n_shown) & (ep_of[ys] < args.n_shown)
        n_tests = len(ok_test)
        moves = np.array([int(np.sum(~still[y + 1:y + d + 1] & still[y:y + d])) for y, d in zip(ys, ls)])  # still->moving starts
        shown_x = ep_of[xs] < args.n_shown
        both_tab = np.array([(int(a) in cls_g) and (int(b) in cls_g) for a, b in zip(xs, ys)])
        bel_right = np.array([cls_g[int(a)] == cls_g[int(b)] for a, b in zip(xs[both_tab], ys[both_tab])])
        bel_cert = np.array([cls_g[int(a)] in CERTAIN and cls_g[int(b)] in CERTAIN for a, b in zip(xs[both_tab], ys[both_tab])])
        cov = set(xs.tolist()) | set(ys.tolist())
        cov_tab = np.mean([g in cov for g in table_g])
        cov_cert = np.mean([g in cov for g, c in zip(table_g, priv["label"]) if c in CERTAIN])
        r = dict(window=w, tests=n_tests, tests_self_ok=int(sum(ok_test.values())), chains=int(keep.sum()),
                 positives=int(pos.sum()), slot_precision=float(right.mean()),
                 belief_precision_on_still=float(bel_right.mean()) if both_tab.any() else None,
                 both_certain_on_still=float(bel_cert.mean()) if both_tab.any() else None, n_on_still=int(both_tab.sum()),
                 coverage_still=float(cov_tab), coverage_still_certain=float(cov_cert),
                 precision_shown_x=float(right[shown_x].mean()) if shown_x.any() else None,
                 precision_closed_x=float(right[~shown_x].mean()) if (~shown_x).any() else None,
                 n_shown_x=int(shown_x.sum()), n_closed_x=int((~shown_x).sum()),
                 by_moves={int(m): [int((moves == m).sum()), float(right[moves == m].mean())] for m in np.unique(moves)},
                 by_slot={int(s_): [int((slot[ys] == s_).sum()), float(right[slot[ys] == s_].mean())] for s_ in range(3)})
        r.update(leaf_right=float(leaf_ok.mean()), precision_both_shown=float(right[both_shown].mean()),
                 n_both_shown=int(both_shown.sum()))
        results[w] = r
        np.savez(out / f"pairs_w{w}.npz", x=xs, y=ys, depth=ls, p=p[pos], x_episode=ep_of[xs], x_t=t_of[xs],
                 y_episode=ep_of[ys], y_t=t_of[ys], slot_right=right, moves=moves,
                 neg_x=nx, neg_y=ny, neg_x_episode=ep_of[nx], neg_x_t=t_of[nx], neg_y_episode=ep_of[ny],
                 neg_y_t=t_of[ny], neg_right=neg_right, neg_depth=L[neg])
        say(f"\nwindow w={w}: {n_tests} tests, {r['tests_self_ok']} pass the self-check; {r['chains']} x-chains -> "
            f"{r['positives']} positive pairs")
        say(f"  slot precision {r['slot_precision']:.4f} | x shown start {r['precision_shown_x']} (n {r['n_shown_x']}), "
            f"x closed start {r['precision_closed_x']} (n {r['n_closed_x']})")
        say(f"  leaf really holds the ball {r['leaf_right']:.4f} | both x and y shown start: precision "
            f"{r['precision_both_shown']:.4f} (n {r['n_both_shown']})")
        say(f"  on still frames (n {r['n_on_still']}): belief precision {r['belief_precision_on_still']}, "
            f"both certain {r['both_certain_on_still']}")
        say(f"  coverage of the {len(table_g)} still frames: {cov_tab:.3f}; of those with a certain true belief: {cov_cert:.3f}")
        say("  by swaps between y and the leaf: " + ", ".join(f"{m}: n {v[0]} prec {v[1]:.3f}" for m, v in r["by_moves"].items()))
        say("  by true slot at y: " + ", ".join(f"slot {s_}: n {v[0]} prec {v[1]:.3f}" for s_, v in r["by_slot"].items()))

    # ---- label baselines: pairs = two still frames with the same label
    def label_pairs(lab, truth):
        lab, truth = np.asarray(lab), np.asarray(truth)
        res = {}
        for name, classes in (("certain classes {0} {a} {b}", CERTAIN), ("all 7 classes", tuple(range(7)))):
            n_pairs = n_right = 0
            for c in classes:
                tt = truth[lab == c]
                n = len(tt)
                n_pairs += n * (n - 1) // 2
                n_right += sum(k * (k - 1) // 2 for k in np.bincount(tt, minlength=7))
            cov = np.isin(lab, classes).mean()
            res[name] = dict(pairs=int(n_pairs), precision=n_right / max(n_pairs, 1), coverage=float(cov))
        return res

    base = {"hand-crafted": label_pairs(hand["label"], hand["truth"]), "privileged": label_pairs(priv["label"], priv["label"])}
    say("\nlabel baselines (pairs = two still frames with the same label; precision vs the true belief class):")
    for k, v in base.items():
        for kk, vv in v.items():
            say(f"  {k:13s} {kk:28s} pairs {vv['pairs']:>10d}  precision {vv['precision']:.4f}  coverage {vv['coverage']:.3f}")
    json.dump({"splice": {str(k): v for k, v in results.items()}, "labels": base, "args": vars(args)},
              open(out / "splice_pairs.json", "w"), indent=1)
    say(f"-> {out}")


if __name__ == "__main__":
    main()
