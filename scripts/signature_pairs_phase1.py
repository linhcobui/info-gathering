"""Pair test for SELF-SIGNATURE alignment: are frame-matched memory pairs that both predict "ball" on some shared imagined
future really pairs with the ball in the same place?

    pixi run python scripts/signature_pairs_phase1.py [--episodes 120] [--pairs 3000] [--k 16]

Pairs: frames from DIFFERENT training episodes whose frozen embeddings are identical. Memories: the frozen labeller's
teacher memory at each frame (Naive-U by default). Futures: k trajectories per pair, SHARED by both memories (common
random numbers; the forward model never reads memory, so both see identical futures). At a frame that is not moving,
draw a real motion onset (real action, latent action u) from the 64 real onset frames nearest to it; while moving,
keep the action with u = 0. No snapping, no resting-frame requirement.
Read: whenever a lift starts in a trajectory, the decoder's memory-only answer for that lift (before any generated
lifted frame is visible), for both memories.
  any rule:  positive if on some read both say ball (p > threshold); stop at the first one.
  all rule (ablation): positive if every read agrees within 0.1 (and there was at least one read).
Truth (scoring only): the ball is under the same cup in both frames.
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

from belief_labels_phase1 import truth_labels  # noqa: E402
from info_gathering.swap_sequences import sequences  # noqa: E402
from latent_action_idm import FDM, IDM, transitions  # noqa: E402
from train_mcl_prior_phase1 import load_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="logdir/world_model/smt_phase1_all3u/naive")
    ap.add_argument("--idm", default="logdir/latent_actions/smoke_b10/idm_fdm.pt")
    ap.add_argument("--zarr", default="data/memtest4_all3u.zarr")
    ap.add_argument("--episodes", type=int, default=120, help="training episodes to draw frames from (random)")
    ap.add_argument("--pairs", type=int, default=3000)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--max_read_step", type=int, default=None, help="only read lifts that start at step <= this")
    ap.add_argument("--first_lift_only", action="store_true", help="read only the first lift of each trajectory")
    ap.add_argument("--snap", action="store_true", help="diagnostic: snap generated frames to the nearest real frame")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="logdir/world_model/smt_phase1_all3u/naive/signature_pairs.json")
    args = ap.parse_args()
    dev = args.device
    rng = np.random.default_rng(0)
    g = torch.Generator(device=dev).manual_seed(0)

    # ---- frozen labeller and latent-action model
    model = load_model(Path(args.run) / "checkpoints" / "wm_latest.pt", dev)
    amp = dict(device_type=dev.split(":")[0], dtype=model.amp_dtype)
    dec, F_ = model.reward_decoder, int(model.fut_len)
    ck = torch.load(ROOT / args.idm, map_location=dev, weights_only=False)
    imean, istd, d = ck["mean"], ck["std"], ck["args"]["d"]
    idm, fdm = IDM(1024, 2, d).to(dev), FDM(1024, 2, d).to(dev)
    idm.load_state_dict(ck["idm"]); fdm.load_state_dict(ck["fdm"]); idm.eval(); fdm.eval()
    imean_t, istd_t = torch.from_numpy(imean).float().to(dev), torch.from_numpy(istd).float().to(dev)
    to_model = lambda x: (x * istd_t + imean_t - model.embed_mean) / model.embed_std   # IDM space -> model space

    # ---- onset bank (all training episodes): frame resting, next frame moves
    tr = transitions(args.zarr, range(768))
    T = lambda x: torch.from_numpy(((x - imean) / istd).astype(np.float32)).to(dev)
    E0b, E1b, Ab = T(tr["e0"]), T(tr["e1"]), torch.from_numpy(tr["a"]).to(dev)
    with torch.no_grad():
        MU, _ = idm(E0b, E1b, Ab)
    moved = np.abs(tr["e1"] - tr["e0"]).max(-1) > 1e-6
    onset = torch.from_numpy(np.flatnonzero(tr["resting"] & moved)).to(dev)
    on_e0, on_a, on_u = E0b[onset], Ab[onset], MU[onset]
    step_scale = float((E1b[moved] - E0b[moved]).pow(2).sum(-1).median())

    # ---- frames of the chosen episodes: raw embedding, teacher memory, true slot, true belief class
    z = zarr.open_group(str(ROOT / args.zarr), mode="r")
    ends = z["episode_ends"][:]; st = np.r_[0, ends[:-1]]
    cfg = json.loads(z.attrs["config"])
    peek_len = 2 * int(cfg["move_frames"]) + int(cfg.get("peek_up_frames", 2))
    seqs = sequences(ROOT / args.zarr); rest_z = float(z["cup_z_max"][:].min())
    key = model.embed_key
    eps = rng.choice(768, args.episodes, replace=False)
    frames = []                                                       # (episode, t, raw embedding row, slot, class)
    Zmem = {}
    for e in eps:
        s, n = int(st[e]), int(ends[e] - st[e])
        raw = z[key][s:s + n]
        E = (torch.from_numpy(raw).float().to(dev) - model.embed_mean) / model.embed_std
        act = z["action"][s:s + n].astype(np.float32)
        PA = torch.zeros(1, n, 2, device=dev); PA[0, 1:] = torch.from_numpy(act[:n - 1]).to(dev)
        first = torch.zeros(1, n, dtype=torch.bool, device=dev); first[0, 0] = True
        with torch.no_grad():
            Zmem[int(e)] = model.encode_windows(E[None], PA, first)[0].float()
        cls = truth_labels(z, int(e), s, n, seqs, rest_z, peek_len)
        slot = z["ball_slot"][s:s + n]
        for t in range(n - 10):                                       # not inside the final lift
            frames.append((int(e), t, raw[t], int(slot[t]), int(cls[t])))
    groups = defaultdict(list)
    for i, f in enumerate(frames):
        groups[f[2].astype(np.float16).tobytes()].append(i)
    cand = []
    for idx in groups.values():
        if len(idx) < 2:
            continue
        for _ in range(min(50, len(idx))):
            a, b = rng.choice(idx, 2, replace=False)
            if frames[a][0] != frames[b][0]:
                cand.append((a, b))
    rng.shuffle(cand)
    pairs = cand[:args.pairs]
    print(f"{len(frames)} frames from {args.episodes} episodes, {len(groups)} distinct embeddings, "
          f"{len(cand)} candidate pairs -> {len(pairs)} used; true same-cup share {np.mean([frames[a][3] == frames[b][3] for a, b in pairs]):.2f}")

    # ---- rollouts: k shared trajectories per pair, all pairs batched
    P, K, H = len(pairs), args.k, args.horizon
    e = T(np.stack([frames[a][2] for a, _ in pairs])).repeat_interleave(K, 0)     # (P*K, E), IDM space
    prev = None
    a_cur = torch.zeros(P * K, 2, device=dev)
    X, ACT = [], []
    with torch.no_grad():
        for k in range(H):
            moving = torch.zeros(P * K, dtype=torch.bool, device=dev) if prev is None else \
                (e - prev).pow(2).sum(-1) >= 0.05 * step_scale
            u = torch.zeros(P * K, d, device=dev)
            rest = (~moving).nonzero().squeeze(-1)
            for c in rest.split(4096):
                near = torch.cdist(e[c], on_e0).topk(64, largest=False).indices          # 64 nearest real onsets
                pick = near[torch.arange(len(c), device=dev), torch.randint(0, 64, (len(c),), device=dev, generator=g)]
                a_cur[c], u[c] = on_a[pick], on_u[pick]
            X.append(e.clone()); ACT.append(a_cur.clone())
            nxt = fdm(e, a_cur, u)
            if args.snap:
                allE = torch.cat([E0b, E1b])
                nxt = torch.cat([allE[torch.cdist(c_, allE).argmin(-1)] for c_ in nxt.split(4096)])
            prev, e = e, nxt
    X, ACT = torch.stack(X, 1), torch.stack(ACT, 1)                                    # (P*K, H, ...)
    lift_start = (ACT[:, :, 1] > .5) & torch.cat([torch.ones_like(ACT[:, :1, 1], dtype=torch.bool),
                                                 ACT[:, :-1, 1] < .5], 1)               # (P*K, H)
    print(f"trajectories with at least one lift: {lift_start.any(1).float().mean():.2f}; "
          f"lifts per trajectory {lift_start.sum(1).float().mean():.2f}")

    # ---- decoder reads, memory-only, at every lift start, for both memories
    Xm = to_model(X)
    pad = F_ - H
    Xm = torch.cat([Xm, Xm[:, -1:].expand(-1, pad, -1)], 1)
    Am = torch.cat([ACT, ACT[:, -1:].expand(-1, pad, -1)], 1)
    za = torch.stack([Zmem[frames[a][0]][frames[a][1]] for a, _ in pairs]).repeat_interleave(K, 0)
    zb = torch.stack([Zmem[frames[b][0]][frames[b][1]] for _, b in pairs]).repeat_interleave(K, 0)
    pa, pb = [], []
    with torch.no_grad(), autocast(**amp):
        for c in torch.arange(P * K, device=dev).split(2048):
            pa.append(torch.sigmoid(dec.ball_logit(dec(za[c], Xm[c], Am[c])).float()[:, 1:H + 1]))
            pb.append(torch.sigmoid(dec.ball_logit(dec(zb[c], Xm[c], Am[c])).float()[:, 1:H + 1]))
    pa, pb = torch.cat(pa).view(P, K, H), torch.cat(pb).view(P, K, H)   # index k = read for a lift starting at step k
    ls = lift_start.view(P, K, H)
    if args.first_lift_only:                     # read only before any imagined reveal is visible
        ls = ls & (ls.cumsum(-1) == 1)
    if args.max_read_step is not None:
        ls = ls & (torch.arange(H, device=dev) <= args.max_read_step)
    thr = args.threshold
    both_ball = (ls & (pa > thr) & (pb > thr)).flatten(1).any(1).cpu().numpy()
    n_reads = ls.flatten(1).sum(1).cpu().numpy()
    agree_all = ((~ls) | ((pa - pb).abs() <= 0.1)).flatten(1).all(1).cpu().numpy() & (n_reads > 0)
    # first trajectory where the any rule fires (early stop)
    fires = (ls & (pa > thr) & (pb > thr)).any(2).cpu().numpy()                       # (P, K)
    first = np.where(fires.any(1), fires.argmax(1) + 1, -1)

    same = np.array([frames[a][3] == frames[b][3] for a, b in pairs])
    known = np.array([frames[a][4] <= 2 and frames[b][4] <= 2 for a, b in pairs])     # both beliefs a single cup
    same_belief = np.array([frames[a][4] == frames[b][4] for a, b in pairs])

    def score(pred, m):
        tp, fp = int((pred & same & m).sum()), int((pred & ~same & m).sum())
        fn = int((~pred & same & m).sum())
        return dict(n=int(m.sum()), predicted_pos=int((pred & m).sum()), precision=tp / max(tp + fp, 1),
                    recall=tp / max(tp + fn, 1), true_same_share=float(same[m].mean()) if m.any() else float("nan"))

    report = {"args": vars(args)}
    print(f"\n{'rule':12s} {'pairs':24s} {'n':>6s} {'pred +':>7s} {'precision':>10s} {'recall':>7s} {'truly same':>10s}")
    for rule, pred in (("any (both 1)", both_ball), ("all agree", agree_all)):
        for name, m in (("all pairs", np.ones(P, bool)), ("both know the cup", known), ("at least one unsure", ~known)):
            r = score(pred, m)
            report[f"{rule} | {name}"] = r
            print(f"{rule:12s} {name:24s} {r['n']:6d} {r['predicted_pos']:7d} {r['precision']:10.3f} {r['recall']:7.3f} "
                  f"{r['true_same_share']:10.2f}")
    fp = both_ball & ~same
    print(f"\nany-rule false positives: {int(fp.sum())}; of them with the SAME true belief (e.g. both 'any of three'): "
          f"{int((fp & same_belief).sum())}")
    print(f"trajectories needed before the any rule fired (positives): median {np.median(first[first > 0]) if (first > 0).any() else 'n/a'}")
    Path(ROOT / args.out).write_text(json.dumps(report, indent=1))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
