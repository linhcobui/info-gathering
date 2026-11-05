"""Backtrack labels: where the ball is at every still frame, from the decoder's sure moments + real swaps.

    pixi run python scripts/backtrack_labels_phase1.py [--run <stage-1 ckpt>] [--lap <latent actions>] [--out <npz>]

No imagination. In each REAL training episode:

  1. still frames = frame identical to the previous one, while the hold flag is on
  2. segments between consecutive still frames: length 1 = nothing moved; length 6 = a swap, named by the
     latent-action inverse model (frames only); any other length = unknown (breaks the chain)
  3. which code is which swap, without labels: a real swap that involves slot 0 FLIPS the decoder's own
     "ball if we lift now?" across the segment (ball here -> gone, or gone -> here); the swap of the other two
     cups never does. The two codes that flip it are A = (0 alpha) and B = (0 beta); the third is C = (alpha beta).
  4. anchors = still frames where the frozen decoder says "ball here" (teacher memory)
  5. walk from the anchors backwards and forwards through the episode's real swaps: every still frame gets
     slot 0, alpha or beta. An episode is dropped if its anchors disagree, or if a frame where the decoder says
     "no ball" would be labelled slot 0. Episodes where the ball never sits in slot 0 get no labels.

Output (episode, t, label): 0 = slot 0, 1 = alpha, 2 = beta, -1 = none. `truth` is saved for diagnostics only.
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

from info_gathering.swap_sequences import sequences  # noqa: E402
from train_mcl_prior_phase1 import load_model  # noqa: E402
from world_model.models.smt_phase1.latent_action_prior import LatentActionPrior  # noqa: E402

LIFT_PH, IDLE = 2, 3


def main():
    ap = argparse.ArgumentParser()
    N = "logdir/world_model/smt_phase1_all3/naive"
    ap.add_argument("--run", default=f"{N}/checkpoints/wm_latest.pt")
    ap.add_argument("--lap", default=f"{N}/latent_action_prior/latent_action_prior.pt")
    ap.add_argument("--train_zarr", default="data/memtest4_all3.zarr")
    ap.add_argument("--n_train", type=int, default=384)
    ap.add_argument("--out", default="data/labels_backtrack_all3naive.npz")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device
    model = load_model(args.run, dev)
    amp = dict(device_type=dev.split(":")[0], dtype=model.amp_dtype)
    ck = torch.load(ROOT / args.lap, map_location=dev)
    lap = LatentActionPrior(model.prior, model.embed_dim, block=ck["block"], n_codes=ck["n_codes"]).to(dev).eval()
    lap.load_state_dict(ck["lap"])
    B = int(ck["block"])
    key = model.embed_key
    norm = lambda v: (torch.from_numpy(np.asarray(v, np.float32)).to(dev) - model.embed_mean) / model.embed_std
    A_HOLD, A_LIFT = torch.tensor([1., 0.], device=dev), torch.tensor([0., 1.], device=dev)

    zg = zarr.open_group(str(ROOT / args.train_zarr), mode="r")
    en = zg["episode_ends"][:]
    st = np.concatenate([[0], en[:-1]])
    act, ph, slot = zg["action"][:].astype(np.float32), zg["phase"][:], zg["ball_slot"][:]

    # the "ball if we lift now?" query (as build_self_slot0_labels.py)
    e0 = next(e for e in range(len(en)) if not zg["ball_under_target"][en[e] - 1])
    on0 = int(np.flatnonzero(ph[st[e0]:en[e0]] == LIFT_PH)[0])
    REST, LIFT = norm(zg[key][st[e0] + on0]), norm(zg[key][st[e0] + on0 + 1:en[e0]])
    F_, dec = int(model.fut_len), model.reward_decoder
    tx = [REST] * (IDLE + 1) + list(LIFT)
    ta = [A_HOLD] * IDLE + [A_LIFT] * (1 + len(LIFT))
    while len(tx) < F_ - 1:
        tx.append(LIFT[-1]); ta.append(A_LIFT)
    TX, TA, READ = torch.stack(tx[:F_ - 1]), torch.stack(ta[:F_ - 1]), 1 + IDLE + 1

    @torch.no_grad()
    def lift_now(z):
        X = torch.cat([REST.expand(len(z), 1, -1), TX[None].expand(len(z), -1, -1)], 1)
        A = torch.cat([A_HOLD.expand(len(z), 1, -1), TA[None].expand(len(z), -1, -1)], 1)
        with autocast(**amp):
            return (torch.sigmoid(dec.ball_logit(dec(z, X, A)).float()[:, READ]) > 0.5).cpu().numpy()

    # ---- 1-2. still frames, segments, decoder answer at every still frame, code of every moving segment
    eps = []
    for e in range(args.n_train):
        s, n = int(st[e]), int(en[e] - st[e])
        E = norm(zg[key][s:s + n])
        PA = torch.zeros(1, n, 2, device=dev)
        PA[0, 1:] = torch.from_numpy(act[s:s + n - 1]).to(dev)
        first = torch.zeros(1, n, dtype=torch.bool, device=dev); first[0, 0] = True
        with torch.no_grad():
            Z = model.encode_windows(E[None], PA, first)[0]
        still = np.r_[False, (E[1:] - E[:-1]).abs().amax(-1).cpu().numpy() < 1e-6]
        hold = act[s:s + n, 0] > 0.5
        frames = [t for t in np.flatnonzero(still) if t >= 1 and hold[t - 1]]
        here = dict(zip(frames, lift_now(Z[frames]))) if frames else {}
        segs = []                                                   # (t, t2, code or "nothing" or None)
        for a_, b_ in zip(frames[:-1], frames[1:]):
            L = b_ - a_
            if not hold[a_:b_].all():
                continue
            segs.append((a_, b_, "nothing" if L == 1 else None, L))
        moving = [i for i, sg in enumerate(segs) if sg[3] == B]
        if moving:
            fr = torch.stack([E[segs[i][0]:segs[i][0] + B + 1] for i in moving])
            with torch.no_grad(), autocast(**amp):
                codes = lap.infer(fr)[0].argmax(-1).tolist()
            for i, c in zip(moving, codes):
                segs[i] = (segs[i][0], segs[i][1], int(c), segs[i][3])
        eps.append(dict(e=e, frames=frames, here=here, segs=segs, slot=slot[s:s + n]))

    # ---- 3. which code is which swap: does it flip the decoder's own answer across real segments?
    flips, counts = defaultdict(int), Counter()
    for ep in eps:
        for a_, b_, c, _ in ep["segs"]:
            if isinstance(c, int):
                counts[c] += 1
                flips[c] += int(ep["here"][a_] != ep["here"][b_])
    rate = {c: flips[c] / counts[c] for c in counts}
    ranked = sorted(rate, key=rate.get, reverse=True)
    if len(ranked) < 3:
        sys.exit(f"need 3 swap codes on moving segments, found {ranked}")
    cA, cB, cC = ranked[:3]
    print("decoder flip rate across real moving segments, per code: "
          + ", ".join(f"u{c} {rate[c]:.2f} (n {counts[c]})" for c in ranked))
    print(f"  -> A = u{cA} (slot 0 <-> alpha), B = u{cB} (slot 0 <-> beta), C = u{cC} (alpha <-> beta)"
          + (f"; other codes on moving segments: {ranked[3:]}" if len(ranked) > 3 else ""))
    PERM = {cA: {0: 1, 1: 0, 2: 2}, cB: {0: 2, 2: 0, 1: 1}, cC: {1: 2, 2: 1, 0: 0}, "nothing": {0: 0, 1: 1, 2: 2}}

    # ---- 4-5. anchors, walk both ways, consistency
    rows, stats = [], Counter()
    for ep in eps:
        frames, here, segs = ep["frames"], ep["here"], ep["segs"]
        label = {t: -1 for t in frames}
        anchors = [t for t in frames if here[t]]
        if not anchors:
            stats["no anchor (ball never under the lifted cup)"] += 1
        else:
            nxt = {a_: (b_, c) for a_, b_, c, _ in segs}
            prv = {b_: (a_, c) for a_, b_, c, _ in segs}
            ok = True
            for t0 in anchors:
                walk = {t0: 0}
                t, k = t0, 0
                while t in nxt and nxt[t][1] in PERM:            # forward
                    t2, c = nxt[t]; k = PERM[c][k]; walk[t2] = k; t = t2
                t, k = t0, 0
                while t in prv and prv[t][1] in PERM:            # backward (a swap undoes itself)
                    t2, c = prv[t]; k = PERM[c][k]; walk[t2] = k; t = t2
                for t, k in walk.items():
                    if label[t] not in (-1, k) or (k == 0) != bool(here[t]):
                        ok = False
                    label[t] = k
            if not ok:
                stats["dropped: anchors disagree / decoder contradicts"] += 1
                label = {t: -1 for t in frames}
            else:
                stats["labelled"] += 1
        for t in frames:
            rows.append((ep["e"], t, label[t], int(ep["slot"][t])))
    rows = np.array(rows)
    seqs = sequences(ROOT / args.train_zarr)
    # swaps done, for the diagnostics only (from the hidden schedule)
    done = []
    for e, t, _, _ in rows:
        h0 = int(np.flatnonzero(ph[st[e]:en[e]] == 1)[0])
        done.append(min(len(seqs[e][1]), max(0, (t - (h0 + 4) + 1) // 6)))
    done = np.array(done)
    lab, tru = rows[:, 2], rows[:, 3]
    np.savez(ROOT / args.out, episode=rows[:, 0], t=rows[:, 1], label=lab, truth=tru, swaps_done=done,
             codes=np.array([cA, cB, cC]),
             note=np.array("label: 0 slot 0, 1 alpha (A brings it to 0), 2 beta, -1 none; truth for diagnostics only"))

    print(f"\n{len(rows)} still frames from {len(eps)} episodes -> {args.out}")
    print("episodes: " + ", ".join(f"{k} {v}" for k, v in stats.items()))
    has = lab >= 0
    print(f"frames labelled: {has.mean():.3f}")
    for k, nm in ((0, "slot 0"), (1, "alpha"), (2, "beta"), (-1, "none")):
        m = lab == k
        print(f"  {nm:7s} {m.sum():5d}  true slot 0/1/2 (diagnostics): " + " / ".join(f"{np.mean(tru[m] == j):.2f}" for j in range(3)))
    mp = {0: 0}
    for k in (1, 2):
        m = lab == k
        if m.any():
            mp[k] = int(np.bincount(tru[m], minlength=3).argmax())
    right = np.array([mp.get(l, -9) == y for l, y in zip(lab[has], tru[has])])
    print(f"labelled frames right (alpha/beta matched to their majority slot {mp}): {right.mean():.3f}")
    print("by swaps done:  " + "  ".join(
        f"{d}: labelled {has[done == d].mean():.2f} right {np.mean([mp.get(l, -9) == y for l, y in zip(lab[(done == d) & has], tru[(done == d) & has])]):.3f}"
        for d in range(4) if ((done == d) & has).any()))
    for s_ in range(3):
        m = tru == s_
        print(f"  ball truly in slot {s_}: labelled {has[m].mean():.2f}")


if __name__ == "__main__":
    main()
