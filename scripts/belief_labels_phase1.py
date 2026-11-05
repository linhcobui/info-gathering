"""ALL3-U alignment labels: the BELIEF at every still hold frame, i.e. the set of slots the ball can be in given
what has been seen SO FAR. Labels never use the future: hindsight is only allowed where the start was shown.

    pixi run python scripts/belief_labels_phase1.py --mode privileged
    pixi run python scripts/belief_labels_phase1.py --mode handcrafted --run <naive>/checkpoints/wm_latest.pt \
        --lap <naive>/latent_action_prior/latent_action_prior.pt

Seven classes, slots named 0 (the lifted cup's slot), alpha, beta:
    0 {0}   1 {alpha}   2 {beta}   3 {0, alpha, beta}   4 {alpha, beta}   5 {0, alpha}   6 {0, beta}
A swap permutes the set; a peek at slot 0 that shows the ball collapses it to {0}; one that shows an empty cup
removes 0 from it. A still frame is one identical to the previous frame while the hold flag is on (as ALL3).

privileged   true slots (alpha = 1, beta = 2), true start for shown episodes, true swaps and peek outcomes.

handcrafted  no hidden labels. From the frozen Naive run:
  - "ball if we lift now?" (decoder on the teacher memory, as backtrack_labels_phase1.py), kept as a probability;
  - swap blocks named by the 4-code latent-action inverse model; A / B / C picked by flip rate, as ALL3, measured
    on episodes whose start was shown;
  - the start was SHOWN iff the decoder is sure (p < 0.1 or > 0.9) at the first still frame. A closed start
    gives ~1/3 there if the Naive model is calibrated;
  - start shown: the ALL3 backtracking walk from the sure "ball here" anchors, both directions (the start was
    seen, so earlier frames really knew);
  - start closed: FORWARD ONLY from {0, alpha, beta}. Swap codes permute the set; a peek uses the reward the
    environment gives at the peek's lift frames (ball revealed or not: the model's own training signal);
    an unnamed segment stops the walk.
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
import zarr  # noqa: E402

from info_gathering.swap_sequences import sequences  # noqa: E402

SETS = [frozenset({0}), frozenset({1}), frozenset({2}), frozenset({0, 1, 2}),
        frozenset({1, 2}), frozenset({0, 1}), frozenset({0, 2})]
CLASS = {s: i for i, s in enumerate(SETS)}
NAMES = ["{0}", "{a}", "{b}", "{0,a,b}", "{a,b}", "{0,a}", "{0,b}"]
HOLD_PH, LIFT_PH = 1, 2
SWAP_PERM = {(0, 1): {0: 1, 1: 0, 2: 2}, (0, 2): {0: 2, 2: 0, 1: 1}, (1, 2): {0: 0, 1: 2, 2: 1}}


def permute(s, perm):
    return frozenset(perm[k] for k in s)


def episode_layout(z):
    """Per-episode frame ranges, still frames, and per-frame arrays used by both modes."""
    ends = z["episode_ends"][:]
    st = np.r_[0, ends[:-1]]
    return st, ends


def still_frames(E, act):
    """t >= 1 with frame t identical to frame t-1 while the hold flag was on at t-1."""
    same = np.r_[False, np.abs(E[1:] - E[:-1]).max(-1) < 1e-6]
    return [t for t in np.flatnonzero(same) if t >= 1 and act[t - 1, 0] > 0.5]


def truth_labels(z, e, s, n, seqs, cup_rest, peek_len):
    """Privileged belief class at every frame of episode e (frame-indexed list)."""
    slot, ph = z["ball_slot"][s:s + n], z["phase"][s:s + n]
    cz = z["cup_z_max"][s:s + n]
    closed, peek = int(z["episode_closed"][e]), int(z["episode_peek"][e])
    start, pairs = seqs[e]
    h0 = int(np.flatnonzero(ph == HOLD_PH)[0])
    lifts = np.flatnonzero(ph == LIFT_PH)
    final_on = int(lifts[np.r_[True, np.diff(lifts) > 1]][-1])     # first frame of the last lift run
    swap_end = [h0 + 4 + 6 * (j + 1) - 1 + (peek_len if 0 <= peek <= j else 0) for j in range(len(pairs))]
    seen = [t for t in range(n) if ph[t] == LIFT_PH and t < final_on and cz[t] > cup_rest + 1e-3]
    peek_t = seen[0] if seen else None
    S = frozenset({0, 1, 2}) if closed else frozenset({int(start)})
    out = []
    for t in range(n):
        if peek_t is not None and t == peek_t:
            S = frozenset({0}) if slot[t] == 0 else S - {0}
        for j, te in enumerate(swap_end):
            if t == te:
                S = permute(S, SWAP_PERM[tuple(pairs[j])])
        out.append(CLASS[S])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["privileged", "handcrafted"], required=True)
    ap.add_argument("--train_zarr", default="data/memtest4_all3u.zarr")
    ap.add_argument("--n_train", type=int, default=768)
    ap.add_argument("--run", default="logdir/world_model/smt_phase1_all3u/naive/checkpoints/wm_latest.pt")
    ap.add_argument("--lap", default="logdir/world_model/smt_phase1_all3u/naive/latent_action_prior/latent_action_prior.pt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    out = args.out or f"data/labels_belief_all3u_{args.mode}.npz"

    z = zarr.open_group(str(ROOT / args.train_zarr), mode="r")
    st, en = episode_layout(z)
    seqs = sequences(ROOT / args.train_zarr)
    act_all = z["action"][:].astype(np.float32)
    cz_all = z["cup_z_max"][:]
    cup_rest = float(cz_all.min())
    key = "embed_r2d_memtest4_wm15000"
    cfg = json.loads(z.attrs["config"])
    peek_len = 2 * int(cfg["move_frames"]) + int(cfg.get("peek_up_frames", 2))

    rows = []                                                        # (episode, t, label, truth)
    if args.mode == "privileged":
        for e in range(args.n_train):
            s, n = int(st[e]), int(en[e] - st[e])
            E = z[key][s:s + n]
            tl = truth_labels(z, e, s, n, seqs, cup_rest, peek_len)
            for t in still_frames(E, act_all[s:s + n]):
                rows.append((e, t, tl[t], tl[t]))
        report(rows, z, out, {})
        return

    # ---------------------------------------------------------------- handcrafted
    import torch
    from torch.amp import autocast
    from train_mcl_prior_phase1 import load_model
    from world_model.models.smt_phase1.latent_action_prior import LatentActionPrior
    dev = args.device
    model = load_model(args.run, dev)
    amp = dict(device_type=dev.split(":")[0], dtype=model.amp_dtype)
    ck = torch.load(ROOT / args.lap, map_location=dev)
    lap = LatentActionPrior(model.prior, model.embed_dim, block=ck["block"], n_codes=ck["n_codes"]).to(dev).eval()
    lap.load_state_dict(ck["lap"])
    B = int(ck["block"])
    norm = lambda v: (torch.from_numpy(np.asarray(v, np.float32)).to(dev) - model.embed_mean) / model.embed_std
    A_HOLD, A_LIFT = torch.tensor([1., 0.], device=dev), torch.tensor([0., 1.], device=dev)
    ph_all, but_all = z["phase"][:], z["ball_under_target"][:]

    # "ball if we lift now?": rest frame + idle + the lift of a shown, EMPTY-cup episode; read the offset-1 reward
    e0 = next(e for e in range(384) if not but_all[en[e] - 1])
    ph0 = ph_all[st[e0]:en[e0]]
    on0 = int(np.flatnonzero(ph0 == LIFT_PH)[0])
    REST, LIFT = norm(z[key][st[e0] + on0]), norm(z[key][st[e0] + on0 + 1:en[e0]])
    IDLE, F_, dec = 3, int(model.fut_len), model.reward_decoder
    tx = [REST] * (IDLE + 1) + list(LIFT)
    ta = [A_HOLD] * IDLE + [A_LIFT] * (1 + len(LIFT))
    while len(tx) < F_ - 1:
        tx.append(LIFT[-1]); ta.append(A_LIFT)
    TX, TA, READ = torch.stack(tx[:F_ - 1]), torch.stack(ta[:F_ - 1]), 1 + IDLE + 1

    @torch.no_grad()
    def lift_now(zz):
        X = torch.cat([REST.expand(len(zz), 1, -1), TX[None].expand(len(zz), -1, -1)], 1)
        A = torch.cat([A_HOLD.expand(len(zz), 1, -1), TA[None].expand(len(zz), -1, -1)], 1)
        with autocast(**amp):
            return torch.sigmoid(dec.ball_logit(dec(zz, X, A)).float()[:, READ]).cpu().numpy()

    eps = []
    for e in range(args.n_train):
        s, n = int(st[e]), int(en[e] - st[e])
        Eraw = z[key][s:s + n]
        act, ph = act_all[s:s + n], ph_all[s:s + n]
        E = norm(Eraw)
        PA = torch.zeros(1, n, 2, device=dev)
        PA[0, 1:] = torch.from_numpy(act[:n - 1]).to(dev)
        first = torch.zeros(1, n, dtype=torch.bool, device=dev); first[0, 0] = True
        with torch.no_grad():
            Z = model.encode_windows(E[None], PA, first)[0]
        frames = still_frames(Eraw, act)
        p = dict(zip(frames, lift_now(Z[frames]))) if frames else {}
        lifts = np.flatnonzero(ph == LIFT_PH)
        final_on = int(lifts[np.r_[True, np.diff(lifts) > 1]][-1])
        segs = []                                                   # (a, b, kind): kind = "nothing" | code | ("peek", ball)
        for a_, b_ in zip(frames[:-1], frames[1:]):
            L = b_ - a_
            mid = act[a_:b_]
            if (mid[:, 1] > 0.5).any():                              # a peek happened in between
                lift_t = [t for t in range(a_, b_) if ph[t] == LIFT_PH and cz_all[s + t] > cup_rest + 1e-3]
                if lift_t and a_ < final_on:
                    segs.append((a_, b_, ("peek", bool(but_all[s + lift_t[0]]))))   # the reward at the peek
                continue
            if not (mid[:, 0] > 0.5).all():
                continue
            segs.append((a_, b_, "nothing" if L == 1 else (None if L != B else "swap")))
        sw = [i for i, sg in enumerate(segs) if sg[2] == "swap"]
        if sw:
            fr = torch.stack([E[segs[i][0]:segs[i][0] + B + 1] for i in sw])
            with torch.no_grad(), autocast(**amp):
                codes = lap.infer(fr)[0].argmax(-1).tolist()
            for i, c in zip(sw, codes):
                segs[i] = (segs[i][0], segs[i][1], int(c))
        first_p = p[frames[0]] if frames else np.nan
        shown = bool(frames) and (first_p < 0.1 or first_p > 0.9)
        eps.append(dict(e=e, s=s, n=n, frames=frames, p=p, segs=segs, shown=shown))

    # A / B / C by flip rate of the decoder's sure answer, on shown episodes only
    flips, counts = defaultdict(int), Counter()
    for ep in eps:
        if not ep["shown"]:
            continue
        for a_, b_, c in ep["segs"]:
            if isinstance(c, int):
                counts[c] += 1
                flips[c] += int((ep["p"][a_] > 0.5) != (ep["p"][b_] > 0.5))
    rate = {c: flips[c] / counts[c] for c in counts}
    ranked = sorted(rate, key=rate.get, reverse=True)
    if len(ranked) < 3:
        sys.exit(f"need 3 swap codes on moving segments, found {ranked}")
    cA, cB, cC = ranked[:3]
    print("flip rate per code (shown episodes): " + ", ".join(f"u{c} {rate[c]:.2f} (n {counts[c]})" for c in ranked))
    PERM = {cA: SWAP_PERM[(0, 1)], cB: SWAP_PERM[(0, 2)], cC: SWAP_PERM[(1, 2)], "nothing": {0: 0, 1: 1, 2: 2}}

    stats = Counter()
    for ep in eps:
        e, s, n, frames, p, segs = ep["e"], ep["s"], ep["n"], ep["frames"], ep["p"], ep["segs"]
        tl = truth_labels(z, e, s, n, seqs, cup_rest, peek_len)
        label = {t: -1 for t in frames}
        nxt = {a_: (b_, c) for a_, b_, c in segs}
        if ep["shown"]:
            stats["start shown (decoder sure at first still frame)"] += 1
            prv = {b_: (a_, c) for a_, b_, c in segs}
            anchors = [t for t in frames if p[t] > 0.9]
            ok = bool(anchors)
            for t0 in anchors:
                walk, t, k = {t0: 0}, t0, 0
                while t in nxt and nxt[t][1] in PERM:
                    t2, c = nxt[t]; k = PERM[c][k]; walk[t2] = k; t = t2
                t, k = t0, 0
                while t in prv and prv[t][1] in PERM:
                    t2, c = prv[t]; k = PERM[c][k]; walk[t2] = k; t = t2
                for t, k in walk.items():
                    if label[t] not in (-1, k) or (k == 0) != (p[t] > 0.5):
                        ok = False
                    label[t] = k
            if not ok:
                stats["  shown: no anchor or walk inconsistent -> unlabelled"] += 1
                label = {t: -1 for t in frames}
        else:
            stats["start closed (decoder unsure at first still frame)"] += 1
            S, t = frozenset({0, 1, 2}), frames[0] if frames else None
            while t is not None:
                label[t] = CLASS[S]
                if t not in nxt:
                    break
                t2, c = nxt[t]
                if isinstance(c, tuple):                            # peek: the reward says ball / empty
                    S = frozenset({0}) if c[1] else S - {0}
                elif c in PERM:
                    S = permute(S, PERM[c])
                else:
                    stats["  closed: walk stopped at an unnamed segment"] += 1
                    break
                t = t2
        for t in frames:
            rows.append((e, t, label[t], tl[t]))
    report(rows, z, out, stats, codes=(cA, cB, cC))


def report(rows, z, out, stats, codes=(-1, -1, -1)):
    rows = np.array(rows)
    lab, tru = rows[:, 2], rows[:, 3]
    # alpha / beta are named by flip-rate order (arbitrary between the two slot-0 swaps): score under the
    # better of the two namings, applied consistently to the whole table
    swap_ab = np.array([0, 2, 1, 3, 4, 6, 5])
    has0 = lab >= 0
    if np.mean(swap_ab[lab[has0]] == tru[has0]) > np.mean(lab[has0] == tru[has0]):
        print("(alpha/beta named the other way round from the true slots; scoring with them swapped)")
        tru = swap_ab[tru]
    closed = z["episode_closed"][:][rows[:, 0]]
    np.savez(ROOT / out, episode=rows[:, 0], t=rows[:, 1], label=lab, truth=tru, codes=np.array(codes),
             note=np.array("label: 0 {0} 1 {a} 2 {b} 3 {0,a,b} 4 {a,b} 5 {0,a} 6 {0,b}; -1 none. truth = privileged belief"))
    print(f"\n{len(rows)} still frames from {len(set(rows[:, 0].tolist()))} episodes -> {out}")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    has = lab >= 0
    print(f"labelled {has.mean():.3f}; labelled right (vs privileged belief) {np.mean(lab[has] == tru[has]):.4f}")
    print(f"{'class':9s} {'labelled':>8s} {'right':>7s} {'truth n':>8s} {'covered':>8s}")
    for k, nm in enumerate(NAMES):
        m = lab == k
        mt = tru == k
        print(f"{nm:9s} {m.sum():8d} {np.mean(tru[m] == k) if m.any() else float('nan'):7.3f} {mt.sum():8d} "
              f"{np.mean(lab[mt] == k) if mt.any() else float('nan'):8.3f}")
    for c, nm in ((0, "start shown"), (1, "closed start")):
        m = closed == c
        print(f"  {nm}: frames {m.sum()}, labelled {has[m].mean():.3f}, right {np.mean(lab[m & has] == tru[m & has]):.4f}")


if __name__ == "__main__":
    main()
