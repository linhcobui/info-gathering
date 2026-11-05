"""Pair test, memory-space version (docs/SELF_SIGNATURE.md): imagine with the latent-action PRIOR, ask "lift now?" at every step.

    pixi run python scripts/signature_pairs_prior_phase1.py [--horizon 10] [--k 16] [--threshold 0.9]

Pairs: frames from different training episodes with identical embeddings (as signature_pairs_phase1.py).
Futures: k trajectories per pair, shared by both memories. At a frame that is not moving, draw a real motion onset
(hold action, latent action u) from the 64 nearest real hold-action onsets; while moving, keep hold with u = 0.
No lifts are drawn, so nothing is ever revealed inside a future. The frame comes from the forward model; the MEMORY is
rolled by the latent-action prior P(z, [a, u]) from train_latent_prior_phase1.py.
Read at every step k = 0..H (k = 0: the real memories at the real frame): p = decoder "ball if we lift now?" from
(z_k, frame_k, lift action).
  any rule: positive if at some step of some trajectory both p > threshold.   all rule: every read agrees within 0.1.
Truth (scoring only): the ball is under the same cup in both frames; also reported against the true BELIEF.
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
from world_model.models.smt_phase1.updater import PriorUpdater  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="logdir/world_model/smt_phase1_all3u/naive")
    ap.add_argument("--idm", default="logdir/latent_actions/smoke_b10/idm_fdm.pt")
    ap.add_argument("--zarr", default="data/memtest4_all3u.zarr")
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--pairs", type=int, default=3000)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--futures", choices=["frames", "segments", "noise"], default="noise",
                    help="frames: forward-model frames + bank onsets; segments: replay real motion segments' latent actions")
    ap.add_argument("--segments", type=int, default=3, help="segments per trajectory (futures=segments)")
    ap.add_argument("--noise_scale", type=float, default=1.0, help="futures=noise: u = scale * eps, eps ~ N(0, I)")
    ap.add_argument("--reads", choices=["still", "all", "liftable"], default="liftable",
                    help="read 'lift now?' only where the imagined frame has stopped moving (the decoder was only asked at rest)")
    args = ap.parse_args()
    dev = args.device
    rng = np.random.default_rng(0)
    g = torch.Generator(device=dev).manual_seed(0)

    model = load_model(Path(args.run) / "checkpoints" / "wm_latest.pt", dev)
    amp = dict(device_type=dev.split(":")[0], dtype=model.amp_dtype)
    dec, F_ = model.reward_decoder, int(model.fut_len)
    ck = torch.load(ROOT / args.idm, map_location=dev, weights_only=False)
    imean, istd, d = ck["mean"], ck["std"], ck["args"]["d"]
    idm, fdm = IDM(1024, 2, d).to(dev), FDM(1024, 2, d).to(dev)
    idm.load_state_dict(ck["idm"]); fdm.load_state_dict(ck["fdm"]); idm.eval(); fdm.eval()
    lp = torch.load(ROOT / args.run / "latent_prior" / "latent_prior.pt", map_location=dev, weights_only=False)
    P = PriorUpdater(act_dim=2 + d, d_model=model.prior.d_model, n_registers=model.prior.n_registers).to(dev)
    P.load_state_dict(lp["P"]); P.eval()
    im_t, is_t = torch.from_numpy(imean).float().to(dev), torch.from_numpy(istd).float().to(dev)
    to_model = lambda x: (x * is_t + im_t - model.embed_mean) / model.embed_std

    tr = transitions(args.zarr, range(768))
    T = lambda x: torch.from_numpy(((x - imean) / istd).astype(np.float32)).to(dev)
    E0b, E1b, Ab = T(tr["e0"]), T(tr["e1"]), torch.from_numpy(tr["a"]).to(dev)
    with torch.no_grad():
        MU, _ = idm(E0b, E1b, Ab)
    moved = np.abs(tr["e1"] - tr["e0"]).max(-1) > 1e-6
    hold = (tr["a"][:, 0] > .5) & (tr["a"][:, 1] < .5)
    onset = torch.from_numpy(np.flatnonzero(tr["resting"] & moved & hold)).to(dev)
    on_e0, on_a, on_u = E0b[onset], Ab[onset], MU[onset]
    lift_on = np.flatnonzero(tr["resting"] & moved & (tr["a"][:, 1] > .5))   # frames where a real lift started
    lift_frames = E0b[torch.from_numpy(lift_on).to(dev)].unique(dim=0)
    on_ev = torch.from_numpy(tr["event"][onset.cpu().numpy()]).to(dev)       # scoring only: which pair each onset swaps
    PERM = torch.tensor([[0, 1, 2], [1, 0, 2], [2, 1, 0], [0, 2, 1], [0, 1, 2], [0, 1, 2]], device=dev)  # event -> slot map
    step_scale = float((E1b[moved] - E0b[moved]).pow(2).sum(-1).median())

    z = zarr.open_group(str(ROOT / args.zarr), mode="r")
    ends = z["episode_ends"][:]; st = np.r_[0, ends[:-1]]
    cfg = json.loads(z.attrs["config"])
    peek_len = 2 * int(cfg["move_frames"]) + int(cfg.get("peek_up_frames", 2))
    seqs = sequences(ROOT / args.zarr); rest_z = float(z["cup_z_max"][:].min())
    key = model.embed_key
    eps = rng.choice(768, args.episodes, replace=False)
    frames, Zmem = [], {}
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
        for t in range(n - 10):
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
    P_, K, H = len(pairs), args.k, args.horizon
    print(f"{len(pairs)} pairs; true same-cup share {np.mean([frames[a][3] == frames[b][3] for a, b in pairs]):.2f}")

    A_LIFT = torch.tensor([0., 1.], device=dev)


    @torch.no_grad()
    def lift_now(zz, frame_model):
        X = frame_model[:, None].expand(-1, F_, -1)
        A = A_LIFT.expand(len(zz), F_, -1)
        with autocast(**amp):
            return torch.sigmoid(dec.ball_logit(dec(zz, X, A)).float()[:, 1])

    if args.futures == "noise":
        # pure noise latent actions: u = noise_scale * eps, eps ~ N(0, I), the SAME eps for both memories; hold action;
        # memory rolled by P; read "lift now?" at every step with the pair's start frame as the read token
        cur = T(np.stack([frames[a][2] for a, _ in pairs])).repeat_interleave(K, 0)
        fm = to_model(cur)
        za = torch.stack([Zmem[frames[a][0]][frames[a][1]] for a, _ in pairs]).repeat_interleave(K, 0)
        zb = torch.stack([Zmem[frames[b][0]][frames[b][1]] for _, b in pairs]).repeat_interleave(K, 0)
        lift_frames_n = E0b[torch.from_numpy(np.flatnonzero(tr["resting"] & moved & (tr["a"][:, 1] > .5))).to(dev)].unique(dim=0)
        liftable = torch.cat([torch.cdist(cur[c], lift_frames_n).min(-1).values ** 2
                              for c in torch.arange(P_ * K, device=dev).split(4096)]) < 0.05 * step_scale
        hold_a = torch.tensor([1., 0.], device=dev).expand(P_ * K, -1)
        H = args.horizon
        pa, pb = [], []
        with torch.no_grad():
            for k in range(H + 1):
                pa.append(torch.cat([lift_now(za[c], fm[c]) for c in torch.arange(P_ * K, device=dev).split(4096)]))
                pb.append(torch.cat([lift_now(zb[c], fm[c]) for c in torch.arange(P_ * K, device=dev).split(4096)]))
                if k == H:
                    break
                eps = torch.randn(P_ * K, d, device=dev, generator=g) * args.noise_scale
                au = torch.cat([hold_a, eps], -1)
                with autocast(**amp):
                    za = torch.cat([P.step(za[c], au[c]).float() for c in torch.arange(P_ * K, device=dev).split(4096)])
                    zb = torch.cat([P.step(zb[c], au[c]).float() for c in torch.arange(P_ * K, device=dev).split(4096)])
        pa, pb = torch.stack(pa, -1).view(P_, K, H + 1), torch.stack(pb, -1).view(P_, K, H + 1)
        flips = ((pa > .5) != (pa[:, :, :1] > .5)).any(-1).float().mean()
        print(f"noise scale {args.noise_scale}: trajectories where memory A's answer ever changes side of 0.5: {flips:.3f}; "
              f"mean |change| of p over the horizon {(pa - pa[:, :, :1]).abs().mean():.3f}")
        if args.reads == "all":
            still = torch.ones_like(pa, dtype=torch.bool)
        else:
            still = liftable.view(P_, K)[:, :, None].expand(-1, -1, H + 1)
            args.reads = "liftable"

    if args.futures == "segments":
        # real motion segments: from a frame whose next frame moves, until the frame stops changing; their latent
        # actions (IDM means), real actions, real end frame, true event (scoring only). No forward model.
        segs = []
        ep_id = tr["episode"]
        i = 0
        N = len(tr["event"])
        while i < N:
            if moved[i] and tr["resting"][i]:
                j = i
                while j + 1 < N and ep_id[j + 1] == ep_id[i] and moved[j + 1]:
                    j += 1
                segs.append((i, j))                       # transitions i..j: motion; frame after j is the end frame
                i = j + 1
            else:
                i += 1
        seg_start = E0b[torch.tensor([a_ for a_, _ in segs], device=dev)]
        seg_end = E1b[torch.tensor([b_ for _, b_ in segs], device=dev)]
        seg_ev = torch.tensor([int(tr["event"][a_]) for a_, _ in segs], device=dev)
        hold_seg = torch.tensor([bool(tr["a"][a_, 0] > .5 and tr["a"][a_, 1] < .5) for a_, _ in segs], device=dev)
        print(f"{len(segs)} real motion segments ({int(hold_seg.sum())} under the hold action)")
        lift_frames_s = E0b[torch.from_numpy(np.flatnonzero(tr["resting"] & moved & (tr["a"][:, 1] > .5))).to(dev)].unique(dim=0)
        cur = T(np.stack([frames[a][2] for a, _ in pairs])).repeat_interleave(K, 0)
        za = torch.stack([Zmem[frames[a][0]][frames[a][1]] for a, _ in pairs]).repeat_interleave(K, 0)
        zb = torch.stack([Zmem[frames[b][0]][frames[b][1]] for _, b in pairs]).repeat_interleave(K, 0)
        slot_a = torch.tensor([frames[a][3] for a, _ in pairs], device=dev).repeat_interleave(K, 0)
        PERM = torch.tensor([[0, 1, 2], [1, 0, 2], [2, 1, 0], [0, 2, 1], [0, 1, 2], [0, 1, 2]], device=dev)
        pa, pb, valid, ta, ns = [], [], [], [], []
        nsw = torch.zeros(P_ * K, dtype=torch.long, device=dev)
        alive = torch.ones(P_ * K, dtype=torch.bool, device=dev)
        with torch.no_grad():
            for step in range(args.segments + 1):
                fm = to_model(cur)
                pa.append(torch.cat([lift_now(za[c], fm[c]) for c in torch.arange(P_ * K, device=dev).split(4096)]))
                pb.append(torch.cat([lift_now(zb[c], fm[c]) for c in torch.arange(P_ * K, device=dev).split(4096)]))
                dmin = torch.cat([torch.cdist(cur[c], lift_frames_s).min(-1).values ** 2 for c in torch.arange(P_ * K, device=dev).split(4096)])
                valid.append(alive & (dmin < 0.05 * step_scale)); ta.append(slot_a == 0); ns.append(nsw.clone())
                if step == args.segments:
                    break
                # pick a real hold-action segment whose start frame matches the current frame
                d_ = torch.cat([torch.cdist(cur[c], seg_start) for c in torch.arange(P_ * K, device=dev).split(4096)]) ** 2
                ok = (d_ < 0.05 * step_scale) & hold_seg[None]
                alive &= ok.any(1)
                w = ok.float() + 1e-12
                pick = torch.multinomial(w, 1, generator=g).squeeze(1)
                for kk in range(6):                                  # feed up to 6 latent actions of the segment
                    idx = torch.tensor([a_ for a_, _ in segs], device=dev)[pick] + kk
                    inside = idx <= torch.tensor([b_ for _, b_ in segs], device=dev)[pick]
                    au = torch.cat([Ab[idx], MU[idx]], -1)
                    with autocast(**amp):
                        na = torch.cat([P.step(za[c], au[c]).float() for c in torch.arange(P_ * K, device=dev).split(4096)])
                        nb = torch.cat([P.step(zb[c], au[c]).float() for c in torch.arange(P_ * K, device=dev).split(4096)])
                    za = torch.where(inside[:, None, None], na, za); zb = torch.where(inside[:, None, None], nb, zb)
                ev = seg_ev[pick]
                slot_a = torch.where(alive, PERM[ev, slot_a], slot_a)
                nsw += (alive & (ev >= 1) & (ev <= 3)).long()
                cur = torch.where(alive[:, None], seg_end[pick], cur)
        H = args.segments
        pa, pb = torch.stack(pa, -1).view(P_, K, H + 1), torch.stack(pb, -1).view(P_, K, H + 1)
        still = torch.stack(valid, -1).view(P_, K, H + 1)
        ta_, ns_ = torch.stack(ta, -1).view(P_, K, H + 1), torch.stack(ns, -1).view(P_, K, H + 1)
        knownA = torch.tensor([frames[a][4] <= 2 for a, _ in pairs], device=dev)[:, None, None]
        print("memory A knows the cup: 'lift now?' vs the TRUE ball after the replayed segments (valid reads):")
        for k_ in range(4):
            m = still & knownA & ((ns_ == k_) if k_ < 3 else (ns_ >= 3))
            if m.any():
                print(f"   after {k_ if k_ < 3 else '3+'} swaps: n {int(m.sum()):6d}  correct {((pa > .5) == ta_)[m].float().mean():.3f}")
        args.reads = "liftable"

    if args.futures == "frames":
        e = T(np.stack([frames[a][2] for a, _ in pairs])).repeat_interleave(K, 0)
        za = torch.stack([Zmem[frames[a][0]][frames[a][1]] for a, _ in pairs]).repeat_interleave(K, 0)
        zb = torch.stack([Zmem[frames[b][0]][frames[b][1]] for _, b in pairs]).repeat_interleave(K, 0)
        prev = None
        pa, pb, still = [], [], []
        slot_a = torch.tensor([frames[a][3] for a, _ in pairs], device=dev).repeat_interleave(K, 0)   # scoring only
        true_a, nsw, sw_count = [], [], torch.zeros(P_ * K, dtype=torch.long, device=dev)
        n_swap_starts = torch.zeros(P_ * K, device=dev)
        with torch.no_grad():
            for k in range(H + 1):
                fm = to_model(e)
                pa.append(torch.cat([lift_now(za[c], fm[c]) for c in torch.arange(P_ * K, device=dev).split(4096)]))
                pb.append(torch.cat([lift_now(zb[c], fm[c]) for c in torch.arange(P_ * K, device=dev).split(4096)]))
                true_a.append(slot_a == 0); nsw.append(sw_count.clone())
                if args.reads == "liftable":     # ask "lift now?" only where real lifts start (frame within tolerance)
                    dmin = torch.cat([torch.cdist(e[c], lift_frames).min(-1).values ** 2 for c in torch.arange(P_ * K, device=dev).split(4096)])
                    still.append(dmin < 0.05 * step_scale)
                else:
                    still.append(torch.ones(P_ * K, dtype=torch.bool, device=dev) if prev is None else
                                 (e - prev).pow(2).sum(-1) < 0.05 * step_scale)      # the imagined frame has stopped moving
                if k == H:
                    break
                moving = torch.zeros(P_ * K, dtype=torch.bool, device=dev) if prev is None else \
                    (e - prev).pow(2).sum(-1) >= 0.05 * step_scale
                a_cur = torch.tensor([1., 0.], device=dev).expand(P_ * K, -1).clone()
                u = torch.zeros(P_ * K, d, device=dev)
                rest = (~moving).nonzero().squeeze(-1)
                for c in rest.split(4096):
                    near = torch.cdist(e[c], on_e0).topk(64, largest=False).indices
                    pick = near[torch.arange(len(c), device=dev), torch.randint(0, 64, (len(c),), device=dev, generator=g)]
                    a_cur[c], u[c] = on_a[pick], on_u[pick]
                    ev_c = on_ev[pick]
                    slot_a[c] = PERM[ev_c, slot_a[c]]                                   # follow the true ball
                    sw_count[c] += ((ev_c >= 1) & (ev_c <= 3)).long()
                n_swap_starts += (u.norm(dim=-1) > 1).float()
                au = torch.cat([a_cur, u], -1)
                with autocast(**amp):
                    za = torch.cat([P.step(za[c], au[c]).float() for c in torch.arange(P_ * K, device=dev).split(4096)])
                    zb = torch.cat([P.step(zb[c], au[c]).float() for c in torch.arange(P_ * K, device=dev).split(4096)])
                prev, e = e, fdm(e, a_cur, u)
        pa, pb = torch.stack(pa, -1).view(P_, K, H + 1), torch.stack(pb, -1).view(P_, K, H + 1)
        print(f"imagined swap starts per trajectory: {n_swap_starts.mean():.2f}")
        still = torch.stack(still, -1).view(P_, K, H + 1)
        print(f"reads at still imagined frames: {still.float().mean():.2f} of all steps")
        # does the imagined memory follow the swaps? memory A, pairs whose frame A knows the cup, still reads only
        ta, ns = torch.stack(true_a, -1).view(P_, K, H + 1), torch.stack(nsw, -1).view(P_, K, H + 1)
        knownA = torch.tensor([frames[a][4] <= 2 for a, _ in pairs], device=dev)[:, None, None]
        m0 = still & knownA
        print("memory A knows the cup: imagined 'lift now?' vs the TRUE ball after the drawn swaps (still reads):")
        for k_ in range(4):
            m = m0 & (ns == k_) if k_ < 3 else m0 & (ns >= 3)
            if m.any():
                acc = ((pa > .5) == ta)[m].float().mean()
                print(f"   after {k_ if k_ < 3 else '3+'} imagined swaps: n {int(m.sum()):6d}  correct {acc:.3f}")
    thr = args.threshold
    fire = (pa > thr) & (pb > thr)
    if args.reads in ("still", "liftable"):
        fire = fire & still
    print(f"valid reads: {still.float().mean():.2f} of all steps")
    any_pos = fire.flatten(1).any(1).cpu().numpy()
    any_pos_now = fire[:, :, 0].any(1).cpu().numpy()                          # step 0 only: no imagination needed
    valid = still if args.reads in ("still", "liftable") else torch.ones_like(still)
    all_pos = (((pa - pb).abs() <= 0.1) | ~valid).flatten(1).all(1).cpu().numpy() & valid.flatten(1).any(1).cpu().numpy()

    same = np.array([frames[a][3] == frames[b][3] for a, b in pairs])
    known = np.array([frames[a][4] <= 2 and frames[b][4] <= 2 for a, b in pairs])
    same_belief = np.array([frames[a][4] == frames[b][4] for a, b in pairs])
    report = {"args": vars(args)}
    print(f"\n{'rule':26s} {'pairs':22s} {'n':>5s} {'pred +':>7s} {'precision':>10s} {'recall':>7s} {'base':>6s} "
          f"{'prec. vs belief':>15s}")
    for rule, pred in (("any, step 0 only (no imag.)", any_pos_now), ("any, imagined steps 0-H", any_pos),
                       ("all agree (ablation)", all_pos)):
        for name, m in (("both know the cup", known), ("at least one unsure", ~known)):
            tp, fp, fn = int((pred & same & m).sum()), int((pred & ~same & m).sum()), int((~pred & same & m).sum())
            pb_ = float((same_belief[pred & m]).mean()) if (pred & m).any() else float("nan")
            r = dict(n=int(m.sum()), pos=int((pred & m).sum()), precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                     base=float(same[m].mean()), precision_vs_belief=pb_)
            report[f"{rule} | {name}"] = r
            print(f"{rule:26s} {name:22s} {r['n']:5d} {r['pos']:7d} {r['precision']:10.3f} {r['recall']:7.3f} {r['base']:6.2f} {pb_:15.3f}")
    out = args.out or str(Path(args.run) / "signature_pairs_prior.json")
    Path(ROOT / out).write_text(json.dumps(report, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
