"""Latent actions on a FROZEN stage-1 run: learn "what happened" from frames, then play it forward.

    pixi run python scripts/train_latent_action_prior_phase1.py [--codes 16] [--steps 3000] [--out <dir>]

Trains world_model.models.smt_phase1.latent_action_prior.LatentActionPrior (inverse model on frames only,
prior with the latent action, chooser p(u | memory)) on hold-phase blocks of the run's training episodes:
the at-rest frame before each swap and every idle frame (6-frame blocks; lift blocks left out, their frames
show the ball). Nothing else in the model is touched.

Then, on the test episodes (memtest4_eval):
  A. how many latent actions get used, and what each one is (true block kind; ground truth for naming only)
  B. 6-step path error: deterministic prior vs prior with the latent action read off the real frames
  C. buttons: force each latent action from every start point (posterior memory); which swap comes out
     (named by matching imagined frames to real clips) and is the decoder right afterwards
  D. the labelling tree played in imagination: "lift now?"; if no, press the 0<->1 action and ask again;
     if still no, press the 0<->2 action -- scored against the hidden slot
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

HOLD_PH, LIFT_PH, MOVE, SWAP, IDLE = 1, 2, 4, 6, 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="logdir/world_model/smt_phase1_memtest4/window60_phase1/readout/main/checkpoints/wm_latest.pt")
    ap.add_argument("--train_zarr", default="data/memtest4.zarr")
    ap.add_argument("--n_train", type=int, default=225)
    ap.add_argument("--zarr", default="data/memtest4_eval.zarr")
    ap.add_argument("--out", default="logdir/world_model/smt_phase1_memtest4/window60_phase1/latent_action_prior")
    ap.add_argument("--codes", type=int, default=16)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tau", type=float, nargs=2, default=[1.0, 0.3], help="Gumbel temperature, start -> end")
    ap.add_argument("--save_at", type=int, nargs="*", default=[], help="also save the model at these steps (0 = init)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev, L, K = args.device, SWAP, args.codes
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
    A_HOLD, A_LIFT = torch.tensor([1., 0.], device=dev), torch.tensor([0., 1.], device=dev)

    def bounds(zg):
        ends = zg["episode_ends"][:]
        return np.concatenate([[0], ends[:-1]]), ends

    @torch.no_grad()
    def episode_states(zg, s, n, teacher):
        E = norm(zg[key][s:s + n])[None]
        PA = torch.zeros(1, n, 2, device=dev)
        PA[0, 1:] = torch.from_numpy(zg["action"][s:s + n - 1].astype(np.float32)).to(dev)
        first = torch.zeros(1, n, dtype=torch.bool, device=dev); first[0, 0] = True
        if teacher:
            return model.encode_windows(E, PA, first)[0], E[0], PA[0]
        warm = max(1, min(model.eval_warmup, n))
        zw = model.encode_windows(E[:, :warm], PA[:, :warm], first[:, :warm])
        zs, z = [zw], zw[:, -1]
        with autocast(**amp):
            for t in range(warm, n):
                z = model.posterior.step(z, E[:, t], PA[:, t]).float()
                zs.append(z.unsqueeze(1))
        return torch.cat(zs, 1)[0].float(), E[0], PA[0]

    def blocks(zarr_path, episodes, teacher=True):
        """Hold-phase blocks: (z_t, frames e_t..e_{t+L}, actions, next L states, kind, true slot at t)."""
        zg = zarr.open_group(str(ROOT / zarr_path), mode="r")
        st, en = bounds(zg)
        seqs, ph, slot = sequences(ROOT / zarr_path), zg["phase"][:], zg["ball_slot"][:]
        Z0, FR, A, Y, kind, sl = [], [], [], [], [], []
        for e in episodes:
            s, n = int(st[e]), int(en[e] - st[e])
            Zt, E, PA = episode_states(zg, s, n, teacher)
            p = ph[s:s + n]
            h0, on, k = int(np.flatnonzero(p == HOLD_PH)[0]), int(np.flatnonzero(p == LIFT_PH)[0]), len(seqs[e][1])
            dense = slot[s:s + n]                  # at-rest frames only: the stored slot is exact there
            b0 = h0 + MOVE
            ts = [(b0 + SWAP * j, f"swap {seqs[e][1][j][0]}<->{seqs[e][1][j][1]}") for j in range(k)]
            ts += [(t, "idle") for t in range(b0 + SWAP * k, on - L)]            # whole block before the lift
            for t, kd in ts:
                Z0.append(Zt[t]); FR.append(E[t:t + L + 1]); A.append(PA[t + 1:t + 1 + L]); Y.append(Zt[t + 1:t + 1 + L])
                kind.append(kd); sl.append(int(dense[t]))
        return torch.stack(Z0), torch.stack(FR), torch.stack(A), torch.stack(Y), kind, np.array(sl)

    Ztr, FRtr, Atr, Ytr, ktr, _ = blocks(args.train_zarr, range(args.n_train))
    n_test = len(zarr.open_group(str(ROOT / args.zarr), mode="r")["episode_ends"])
    Zte, FRte, Ate, Yte, kte, _ = blocks(args.zarr, range(n_test))
    say(f"{len(Ztr)} training blocks {dict(Counter(k.split(' ')[0] for k in ktr))}, {len(Zte)} test blocks")
    var = (Yte.float() - Yte.float().mean(0)).pow(2).mean()

    lap = LatentActionPrior(model.prior, model.embed_dim, block=L, n_codes=K).to(dev)
    opt = torch.optim.AdamW(lap.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-2)

    @torch.no_grad()
    def usage(step):
        lap.eval()
        codes, err, det = [], [], []
        for i in range(0, len(Zte), 256):
            z, fr, a, y = Zte[i:i + 256], FRte[i:i + 256], Ate[i:i + 256], Yte[i:i + 256]
            with autocast(**amp):
                u, _ = lap.infer(fr)
                path = lap.rollout(z, a, u)
                zd, dp = z, []
                for j in range(L):
                    zd = model.prior.step(zd, a[:, j]); dp.append(zd)
            codes.append(u.argmax(-1)); err.append((path.float() - y.float()).pow(2).flatten(1).mean(1))
            det.append((torch.stack(dp, 1).float() - y.float()).pow(2).flatten(1).mean(1))
        codes, err, det = torch.cat(codes).cpu().numpy(), torch.cat(err), torch.cat(det)
        share = np.bincount(codes, minlength=K) / len(codes)
        say(f"step {step:5d} | test 6-step path error (fraction of variance): det {float(det.mean() / var):.4f}  "
            f"with latent action {float(err.mean() / var):.4f} | codes used (>1%): {int((share > 0.01).sum())}")
        lap.train()
        return codes, share, err, det

    def snapshot(step):
        if step in args.save_at:
            torch.save({"lap": lap.state_dict(), "n_codes": K, "block": L, "run": args.run, "steps": step},
                       out / f"latent_action_prior_step{step}.pt")

    usage(0)
    snapshot(0)
    for step in range(1, args.steps + 1):
        tau = args.tau[0] * (args.tau[1] / args.tau[0]) ** (step / args.steps)
        idx = torch.randint(0, len(Ztr), (args.batch,), device=dev)
        z, fr, a, y = Ztr[idx], FRtr[idx], Atr[idx], Ytr[idx]
        with autocast(**amp):
            u, logits = lap.infer(fr, tau=tau)
            path = lap.rollout(z, a, u)
            choose = lap.code_logits(z, a[:, 0])
        loss_dyn = (path.float() - y.float()).pow(2).mean()
        loss_choose = torch.nn.functional.cross_entropy(choose.float(), logits.argmax(-1).detach())
        opt.zero_grad(set_to_none=True)
        (loss_dyn + loss_choose).backward()
        torch.nn.utils.clip_grad_norm_(lap.parameters(), 1.0)
        opt.step()
        snapshot(step)
        if step % 250 == 0:
            say(f"step {step:5d} | train path {float(loss_dyn):.5f}  choose CE {float(loss_choose):.3f}  tau {tau:.2f}")
        if step % 500 == 0:
            usage(step)
    torch.save({"lap": lap.state_dict(), "n_codes": K, "block": L, "run": args.run, "steps": args.steps},
               out / "latent_action_prior.pt")
    codes, share, err, det = usage(args.steps)

    # ---- A. what each used latent action is
    kinds = np.array(kte)
    order = ["swap 0<->1", "swap 0<->2", "swap 1<->2", "idle"]
    used = [c for c in np.argsort(-share) if share[c] > 0.01]
    say(f"\nA. LATENT ACTIONS USED: {len(used)} of {K} (share of test blocks > 1%)")
    say(f"  {'action':7s} {'share':>6s}  " + "  ".join(f"{k:>11s}" for k in order) + "   (where its blocks come from)")
    meaning = {}
    for c in used:
        m = codes == c
        row = [np.mean(kinds[m] == k) for k in order]
        meaning[int(c)] = order[int(np.argmax(row))]
        say(f"  u{c:<6d} {share[c]:6.3f}  " + "  ".join(f"{r:11.2f}" for r in row))
    say(f"  purity (each block kind -> its most common action):")
    for k in order:
        m = kinds == k
        cnt = Counter(codes[m].tolist())
        top = cnt.most_common(3)
        say(f"    {k:11s} n {m.sum():4d}: " + ", ".join(f"u{c} {v / m.sum():.2f}" for c, v in top))
    say(f"\nB. 6-step path error by kind (fraction of variance): det vs with the latent action")
    for k in order:
        mt = torch.as_tensor(kinds == k, device=dev)
        say(f"    {k:11s} det {float(det[mt].mean() / var):.4f}   latent action {float(err[mt].mean() / var):.4f}")

    # ---- C/D need: start points (posterior), templates to name imagined blocks, the "lift now?" query
    zev = zarr.open_group(str(ROOT / args.zarr), mode="r")
    st_ev, en_ev = bounds(zev)
    ztr = zarr.open_group(str(ROOT / args.train_zarr), mode="r")
    st_tr, en_tr = bounds(ztr)
    tr_seqs, tr_ph = sequences(ROOT / args.train_zarr), ztr["phase"][:]
    acc = defaultdict(list)
    for e in range(args.n_train):
        s, n = int(st_tr[e]), int(en_tr[e] - st_tr[e])
        h0 = int(np.flatnonzero(tr_ph[s:s + n] == HOLD_PH)[0])
        for j, pr in enumerate(tr_seqs[e][1]):
            acc[pr].append(ztr[key][s + h0 + MOVE + SWAP * j:s + h0 + MOVE + SWAP * j + 7])
    templates = {pr: norm(np.mean(v, 0)) for pr, v in acc.items()}
    e0 = next(e for e in range(len(en_tr)) if not ztr["ball_under_target"][en_tr[e] - 1])
    on0 = int(np.flatnonzero(tr_ph[st_tr[e0]:en_tr[e0]] == LIFT_PH)[0])
    REST, LIFT = norm(ztr[key][st_tr[e0] + on0]), norm(ztr[key][st_tr[e0] + on0 + 1:en_tr[e0]])
    templates[None] = REST[None].expand(7, -1)
    F_, dec = int(model.fut_len), model.reward_decoder
    tx = [REST] * (IDLE + 1) + list(LIFT)
    ta = [A_HOLD] * IDLE + [A_LIFT] * (1 + len(LIFT))
    while len(tx) < F_ - 1:
        tx.append(LIFT[-1]); ta.append(A_LIFT)
    TX, TA, READ = torch.stack(tx[:F_ - 1]), torch.stack(ta[:F_ - 1]), 1 + IDLE + 1

    @torch.no_grad()
    def lift_now(z):
        b = len(z)
        X = torch.cat([REST.expand(b, 1, -1), TX[None].expand(b, -1, -1)], 1)
        A = torch.cat([A_HOLD.expand(b, 1, -1), TA[None].expand(b, -1, -1)], 1)
        with autocast(**amp):
            return torch.sigmoid(dec.ball_logit(dec(z, X, A)).float()[:, READ]).cpu().numpy()

    names = list(templates)

    @torch.no_grad()
    def play(z, code):
        """One block with `code` -> (end state, named swap per row, distance)."""
        with autocast(**amp):
            path = lap.rollout(z, A_HOLD.expand(len(z), L, -1), code)
            emb = model.readout(path).float()[:, :5]
        d = torch.stack([torch.stack([(emb - templates[p][1 - lag:6 - lag]).pow(2).mean((1, 2)) for lag in (0, 1)]).min(0).values
                         for p in names], 1)
        return path[:, -1].float(), [names[i] for i in d.argmin(1).tolist()], d.min(1).values.cpu().numpy()

    # start points: posterior memory at the at-rest frame before each swap and the first idle frame
    Zs, truth = [], []
    real_d = []
    ev_seqs, ev_ph = sequences(ROOT / args.zarr), zev["phase"][:]
    for e in range(n_test):
        s, n = int(st_ev[e]), int(en_ev[e] - st_ev[e])
        Zp, _, _ = episode_states(zev, s, n, teacher=False)
        p = ev_ph[s:s + n]
        h0, on, k = int(np.flatnonzero(p == HOLD_PH)[0]), int(np.flatnonzero(p == LIFT_PH)[0]), len(ev_seqs[e][1])
        dense = zev["ball_slot"][s:s + n]          # at-rest frames only: the stored slot is exact there
        b0 = h0 + MOVE
        for t in [b0 + SWAP * j for j in range(k)] + ([b0 + SWAP * k] if b0 + SWAP * k < on - 1 else []):
            Zs.append(Zp[t]); truth.append(int(dense[t]))
        if k:
            with torch.no_grad(), autocast(**amp):
                emb = model.readout(Zp[None, b0 + 1:b0 + 1 + SWAP * k]).float()[0]
            for j in range(k):
                seg = emb[SWAP * j:SWAP * j + 5]
                real_d.append(min(float(torch.stack([(seg - templates[pp][1 - lag:6 - lag]).pow(2).mean() for lag in (0, 1)]).min())
                                  for pp in names))
    thr = float(np.quantile(real_d, 0.99))
    Zs, truth = torch.stack(Zs), np.array(truth)
    p0 = lift_now(Zs)
    say(f"\n{len(Zs)} start points (posterior); its own 'lift now?' call is right {np.mean((p0 > 0.5) == (truth == 0)):.3f}")

    # ---- C. buttons
    say(f"\nC. PRESS EACH LATENT ACTION from every start point: which swap comes out, decoder right after")
    say(f"  {'action':7s} {'means':11s} " + " ".join(f"{n_:>8s}" for n_ in ("0<->1", "0<->2", "1<->2", "no swap", "unclear"))
        + "   decoder right (clear)")
    report = {"used": len(used), "share": share.tolist(), "meaning": meaning, "buttons": {}}
    for c in used:
        z_end, nm, dist = play(Zs, torch.full((len(Zs),), int(c), device=dev, dtype=torch.long))
        nm = ["unclear" if dist[i] > thr else nm[i] for i in range(len(nm))]
        sl = truth.copy()
        for i, pr in enumerate(nm):
            if pr not in ("unclear", None) and sl[i] in pr:
                sl[i] = pr[1] if sl[i] == pr[0] else pr[0]
        pe = lift_now(z_end)
        clear = np.array([x != "unclear" for x in nm])
        right = np.mean(((pe > 0.5) == (sl == 0))[clear]) if clear.any() else float("nan")
        cnt = Counter("no swap" if x is None else (x if x == "unclear" else f"{x[0]}<->{x[1]}") for x in nm)
        mix = {n_: cnt.get(n_, 0) / len(nm) for n_ in ("0<->1", "0<->2", "1<->2", "no swap", "unclear")}
        say(f"  u{c:<6d} {meaning[int(c)]:11s} " + " ".join(f"{mix[n_]:8.2f}" for n_ in mix) + f"   {right:.3f}")
        report["buttons"][f"u{c}"] = {"means": meaning[int(c)], "mix": mix, "right": float(right)}

    # ---- D. labelling tree in imagination
    btn = {}
    for want in ("swap 0<->1", "swap 0<->2"):
        m = kinds == want
        btn[want] = Counter(codes[m].tolist()).most_common(1)[0][0]
    say(f"\nD. LABELLING TREE in imagination (0<->1 action = u{btn['swap 0<->1']}, 0<->2 action = u{btn['swap 0<->2']}):")
    say("   'lift now?' -> yes: slot 0 | no: press 0<->1, ask -> yes: slot 1 | no: press 0<->2, ask -> yes: slot 2 | no: unresolved")
    zA, _, _ = play(Zs, torch.full((len(Zs),), int(btn["swap 0<->1"]), device=dev, dtype=torch.long))
    zB, _, _ = play(Zs, torch.full((len(Zs),), int(btn["swap 0<->2"]), device=dev, dtype=torch.long))
    pA, pB = lift_now(zA), lift_now(zB)
    label = np.where(p0 > 0.5, 0, np.where(pA > 0.5, 1, np.where(pB > 0.5, 2, -1)))
    both = (p0 <= 0.5) & (pA > 0.5) & (pB > 0.5)
    say(f"  label right {np.mean(label == truth):.3f}  | unresolved {np.mean(label == -1):.3f} | "
        f"'no' branch where BOTH presses say ball (inconsistent) {np.mean(both):.3f}")
    for s_ in range(3):
        m = truth == s_
        say(f"    ball really in slot {s_}: n {m.sum():4d}, label right {np.mean(label[m] == s_):.3f}, "
            f"labels given " + ", ".join(f"{l}:{np.mean(label[m] == l):.2f}" for l in (0, 1, 2, -1)))
    report["tree"] = {"right": float(np.mean(label == truth)), "unresolved": float(np.mean(label == -1))}
    json.dump(report, open(out / "eval.json", "w"), indent=1)
    say(f"-> {out}")


if __name__ == "__main__":
    main()
