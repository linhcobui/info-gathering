"""Continuous latent actions from frozen frame embeddings: IDM + forward model (docs/SELF_SIGNATURE.md part 1).

    pixi run python scripts/latent_action_idm.py --beta 1.0 [--out logdir/latent_actions/idm_b1.0]

IDM: (e_t, e_{t+1} - e_t, a_t) -> diagonal Gaussian over u (d numbers): mean mu, log-spread; u = mu + sigma * eps.
FDM: (e_t, a_t, u) -> e_{t+1} - e_t. Losses: next-frame squared error (summed over the 1024 dims) + beta * KL(q(u) ||
N(0, I)) + transfer: the same u applied to a resting frame of another sample must be read back as the same u.
Per step: no blocks, no codes. Trained on the ALL3-U training episodes, scored on its 24 held-out episodes.

The true event of each transition (nothing / swap L-M, L-R, M-R / lift / lower) is used ONLY for scoring.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import zarr  # noqa: E402

from info_gathering.swap_sequences import sequences  # noqa: E402

KEY = "embed_r2d_memtest4_wm15000"
EVENTS = ["nothing", "swap L-M", "swap L-R", "swap M-R", "lift", "lower"]
PAIR = {(0, 1): 1, (0, 2): 2, (1, 2): 3}


def transitions(zarr_path, episodes):
    """Per transition t -> t+1: e_t, e_{t+1}, a_t, true event (scoring only), swap phase (0-4, or -1), episode,
    and whether frame t is resting (identical to frame t-1)."""
    z = zarr.open_group(str(ROOT / zarr_path), mode="r")
    ends = z["episode_ends"][:]
    st = np.r_[0, ends[:-1]]
    cfg = json.loads(z.attrs["config"])
    peek_len = 2 * int(cfg["move_frames"]) + int(cfg.get("peek_up_frames", 2))
    seqs = sequences(ROOT / zarr_path)
    E_all, act, ph, cz = z[KEY], z["action"][:].astype(np.float32), z["phase"][:], z["cup_z_max"][:]
    peek = z["episode_peek"][:] if "episode_peek" in z else np.full(len(ends), -1)
    out = {k: [] for k in ("e0", "e1", "a", "event", "phase", "episode", "resting")}
    for e in episodes:
        s, n = int(st[e]), int(ends[e] - st[e])
        E = E_all[s:s + n]
        p = ph[s:s + n]
        h0 = int(np.flatnonzero(p == 1)[0])
        pairs = seqs[e][1]
        t0 = [h0 + 4 + 6 * j + (peek_len if 0 <= int(peek[e]) <= j else 0) for j in range(len(pairs))]
        same = np.r_[False, np.abs(E[1:] - E[:-1]).max(-1) < 1e-6]
        for t in range(n - 1):
            ev, phase = None, -1
            if same[t + 1]:
                ev = 0
            else:
                for j, a0 in enumerate(t0):
                    if a0 <= t <= a0 + 4:
                        ev, phase = PAIR[tuple(pairs[j])], t - a0
                if ev is None:
                    dz = cz[s + t + 1] - cz[s + t]
                    ev = 4 if dz > 1e-6 else (5 if dz < -1e-6 else -1)
            out["e0"].append(E[t]); out["e1"].append(E[t + 1]); out["a"].append(act[s + t])
            out["event"].append(ev); out["phase"].append(phase); out["episode"].append(e); out["resting"].append(same[t])
    return {k: np.asarray(v) for k, v in out.items()}


def mlp(i, o, h=512, depth=3):
    layers, d = [], i
    for _ in range(depth - 1):
        layers += [nn.Linear(d, h), nn.SiLU()]
        d = h
    return nn.Sequential(*layers, nn.Linear(d, o))


class IDM(nn.Module):
    def __init__(self, E, A, d):
        super().__init__()
        self.net = mlp(2 * E + A, 2 * d)

    def forward(self, e0, e1, a):
        mu, log_sigma = self.net(torch.cat([e0, e1 - e0, a], -1)).chunk(2, -1)
        return mu, log_sigma.clamp(-6, 2)


class FDM(nn.Module):
    def __init__(self, E, A, d):
        super().__init__()
        self.net = mlp(E + A + d, E)

    def forward(self, e0, a, u):
        return e0 + self.net(torch.cat([e0, a, u], -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", default="data/memtest4_all3u.zarr")
    ap.add_argument("--n_train", type=int, default=768)
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--transfer", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = ROOT / (args.out or f"logdir/latent_actions/idm_d{args.d}_b{args.beta:g}")
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    dev = args.device

    n_ep = len(zarr.open_group(str(ROOT / args.zarr), mode="r")["episode_ends"])
    tr = transitions(args.zarr, range(args.n_train))
    te = transitions(args.zarr, range(args.n_train, n_ep))
    mean, std = tr["e0"].mean(0), tr["e0"].std(0) + 1e-3                  # per-dim normalisation from training frames
    T = lambda x: torch.from_numpy(((x - mean) / std).astype(np.float32)).to(dev)
    TR = dict(e0=T(tr["e0"]), e1=T(tr["e1"]), a=torch.from_numpy(tr["a"]).to(dev))
    TE = dict(e0=T(te["e0"]), e1=T(te["e1"]), a=torch.from_numpy(te["a"]).to(dev))
    # resting frames grouped by the real action taken at them: a latent action is only transferred to (and replayed
    # from) a resting frame under the SAME real action -- generic, nothing about cups or swaps
    act_id = lambda a: (a[:, 0] > .5).astype(int) + 2 * (a[:, 1] > .5).astype(int)
    tr_act = act_id(tr["a"])
    rest_by_act = {k: torch.from_numpy(np.flatnonzero(tr["resting"] & (tr_act == k))).to(dev) for k in np.unique(tr_act)}
    tr_act_t = torch.from_numpy(tr_act).to(dev)
    print(f"train {len(tr['event'])} transitions, held-out {len(te['event'])}; unlabelled events: "
          f"{int((tr['event'] < 0).sum())} / {int((te['event'] < 0).sum())}")

    E, A, d = TR["e0"].shape[1], TR["a"].shape[1], args.d
    idm, fdm = IDM(E, A, d).to(dev), FDM(E, A, d).to(dev)
    opt = torch.optim.Adam(list(idm.parameters()) + list(fdm.parameters()), lr=args.lr)
    N = len(tr["event"])
    log = open(out / "train.log", "w")
    for step in range(args.steps + 1):
        i = torch.randint(0, N, (args.batch,), device=dev)
        e0, e1, a = TR["e0"][i], TR["e1"][i], TR["a"][i]
        mu, ls = idm(e0, e1, a)
        u = mu + ls.exp() * torch.randn_like(mu)                             # reparameterisation
        rec = (fdm(e0, a, u) - e1).pow(2).sum(-1).mean()
        kl = 0.5 * (mu.pow(2) + (2 * ls).exp() - 1 - 2 * ls).sum(-1).mean()
        # transfer: the same u, applied to a resting frame from elsewhere, must be read back as the same u
        r = torch.empty_like(i)
        for k, idx in rest_by_act.items():
            m = tr_act_t[i] == k
            if m.any():
                r[m] = idx[torch.randint(0, len(idx), (int(m.sum()),), device=dev)]
        e_other = TR["e0"][r]
        e_gen = fdm(e_other, a, u)
        mu_back, _ = idm(e_other, e_gen, a)
        tra = (mu_back - mu.detach()).pow(2).sum(-1).mean()
        loss = rec + args.beta * kl + args.transfer * tra
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0:
            msg = f"step {step:5d}  next-frame {rec.item():9.3f}  KL {kl.item():6.3f} nats  transfer {tra.item():.4f}"
            print(msg); log.write(msg + "\n"); log.flush()

    # ---------------------------------------------------------------- scoring (true events used only here)
    idm.eval(); fdm.eval()
    with torch.no_grad():
        mu_te, ls_te = idm(TE["e0"], TE["e1"], TE["a"])
        mu_tr, _ = idm(TR["e0"], TR["e1"], TR["a"])
        err = lambda pred: (pred - TE["e1"]).pow(2).sum(-1)
        moving = torch.from_numpy(te["event"] > 0).to(dev)
        res = {
            "copy (e_t+1 = e_t)": err(TE["e0"]),
            "FDM, u = 0 (no latent action)": err(fdm(TE["e0"], TE["a"], torch.zeros_like(mu_te))),
            "FDM, u = IDM mean": err(fdm(TE["e0"], TE["a"], mu_te)),
        }
        kl_te = 0.5 * (mu_te.pow(2) + (2 * ls_te).exp() - 1 - 2 * ls_te).sum(-1)
    report = {"args": vars(args), "next_frame_error_moving_heldout": {k: float(v[moving].mean()) for k, v in res.items()}}
    print("\nheld-out next-frame squared error, moving transitions only:")
    for k, v in report["next_frame_error_moving_heldout"].items():
        print(f"  {k:32s} {v:10.2f}")
    # information in u per event and per swap phase (KL in nats = what u carries)
    ev_te, ph_te = te["event"], te["phase"]
    kl_np = kl_te.cpu().numpy()
    report["kl_nats_by_event"] = {EVENTS[k]: float(kl_np[ev_te == k].mean()) for k in range(6) if (ev_te == k).any()}
    report["kl_nats_by_swap_phase"] = {int(p): float(kl_np[ph_te == p].mean()) for p in range(5) if (ph_te == p).any()}
    print("KL (nats carried by u) by event:", {k: round(v, 2) for k, v in report["kl_nats_by_event"].items()})
    print("KL by swap phase 0..4:", {k: round(v, 2) for k, v in report["kl_nats_by_swap_phase"].items()})

    # linear probe: event from the IDM mean (train transitions -> held-out)
    ytr, yte = torch.from_numpy(tr["event"]).to(dev), torch.from_numpy(ev_te).to(dev)
    keep_tr, keep_te = ytr >= 0, yte >= 0
    probe = nn.Linear(d, 6).to(dev)
    popt = torch.optim.Adam(probe.parameters(), lr=1e-2)
    for _ in range(2000):
        l = nn.functional.cross_entropy(probe(mu_tr[keep_tr]), ytr[keep_tr]); popt.zero_grad(); l.backward(); popt.step()
    with torch.no_grad():
        pred = probe(mu_te[keep_te]).argmax(-1)
    yk = yte[keep_te]
    conf = np.zeros((6, 6), int)
    for a_, b_ in zip(yk.cpu().numpy(), pred.cpu().numpy()):
        conf[a_, b_] += 1
    report["probe_accuracy_heldout"] = float((pred == yk).float().mean())
    sw = (yk >= 1) & (yk <= 3)
    report["probe_accuracy_swaps_heldout"] = float((pred[sw] == yk[sw]).float().mean())
    ph0 = torch.from_numpy(ph_te).to(dev)[keep_te] == 0
    report["probe_accuracy_swap_first_step"] = float((pred[sw & ph0] == yk[sw & ph0]).float().mean())
    report["confusion_heldout"] = conf.tolist()
    print(f"linear probe of the event from u (held-out): all {report['probe_accuracy_heldout']:.3f}, swaps "
          f"{report['probe_accuracy_swaps_heldout']:.3f}, first step of a swap {report['probe_accuracy_swap_first_step']:.3f}")
    print("confusion (rows true, cols predicted):", EVENTS)
    for k in range(6):
        print(f"  {EVENTS[k]:9s}", conf[k].tolist())

    # the most common resting frame under the hold action (cups down at rest here; chosen generically)
    hold_rest = rest_by_act[1]
    keys = [tuple(np.round(x, 3)) for x in TR["e0"][hold_rest].cpu().numpy()[:, :16]]
    vals, cnt = np.unique(np.array(keys), axis=0, return_counts=True)
    mode_key = vals[cnt.argmax()]
    mode_i = int(hold_rest[[i for i, k in enumerate(keys) if np.allclose(k, mode_key)][0]])
    e_mode = TR["e0"][mode_i]
    print(f"test start frame: the most common resting frame under hold ({cnt.max()} of {len(hold_rest)} such frames)")

    # replay: a held-out swap's latent actions, (a) from its own start frame, (b) from the common resting frame.
    # Scored on the 4 mid-arc frames (the arc's last frame is identical to a resting frame, so it can't be scored)
    with torch.no_grad():
        bank_e1 = TR["e1"]
        res_replay = {"own start": [0, 0], "common resting frame": [0, 0]}
        for i in [i for i in range(len(ev_te)) if ph_te[i] == 0]:
            if not (ph_te[i:i + 5] == np.arange(5)).all():
                continue
            for name, e in (("own start", TE["e0"][i]), ("common resting frame", e_mode)):
                hits = []
                for k in range(4):
                    e = fdm(e[None], TE["a"][i + k:i + k + 1], mu_te[i + k:i + k + 1])[0]
                    nn_i = int(((bank_e1 - e).pow(2).sum(-1)).argmin())
                    hits.append(tr["event"][nn_i] == ev_te[i] and tr["phase"][nn_i] == k)
                res_replay[name][0] += int(all(hits)); res_replay[name][1] += 1
    report["swap_replay_heldout"] = {k: f"{a}/{b}" for k, (a, b) in res_replay.items()}
    print("replayed held-out swaps, all 4 mid-arc frames = same pair and phase:", report["swap_replay_heldout"])

    # sampling futures from the common resting frame, hold action: (a) u ~ N(0, I); (b) u drawn from the bank of
    # real latent actions taken at resting frames under hold (the empirical distribution)
    with torch.no_grad():
        g = torch.Generator(device=dev).manual_seed(0)
        k = 2000
        e0 = e_mode.expand(k, -1)
        a_hold = torch.tensor([1.0, 0.0], device=dev).expand(k, -1)
        bank_u = mu_tr[hold_rest]
        for name, u in (("u ~ N(0, I)", torch.randn(k, d, device=dev, generator=g)),
                        ("u from the bank of real resting-hold actions",
                         bank_u[torch.randint(0, len(bank_u), (k,), device=dev, generator=g)])):
            nn_i = torch.cdist(fdm(e0, a_hold, u), bank_e1).argmin(-1).cpu().numpy()
            c = {EVENTS[c] if c >= 0 else "other": int((tr["event"][nn_i] == c).sum()) for c in range(-1, 6)}
            report.setdefault("samples_from_common_rest_hold", {})[name] = c
            print(f"2000 samples, {name}: nearest real event {c}")
    real = {EVENTS[c]: int((tr["event"][hold_rest.cpu().numpy()] == c).sum()) for c in range(6)}
    report["real_events_after_resting_hold_frames"] = real
    print("   real data, what follows a resting hold frame:", real)

    np.savez(out / "latents.npz", mu_tr=mu_tr.cpu().numpy(), ev_tr=tr["event"], ph_tr=tr["phase"],
             mu_te=mu_te.cpu().numpy(), ev_te=ev_te, ph_te=ph_te, kl_te=kl_np)
    torch.save({"idm": idm.state_dict(), "fdm": fdm.state_dict(), "mean": mean, "std": std, "args": vars(args)},
               out / "idm_fdm.pt")
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
