"""A prior updater driven by latent actions, bolted onto a frozen SMT phase-1 run (docs/SELF_SIGNATURE.md).

    pixi run python scripts/train_latent_prior_phase1.py [--run <stage-1 run>] [--idm <idm_fdm.pt>]

P(z_t, [a_t, u_t]) -> z_{t+1}: the phase-1 PriorUpdater class with the latent action u (frozen frame-space IDM, means)
appended to the real action. Trained like the phase-1 prior: MSE to the frozen teacher's z*_{t+1}, input the teacher's
z*_t; the stage-1 model is never touched. Warm-started from the run's own prior (same trunk; the action projection
gets u columns initialised at zero).

Tracking check on the held-out episodes: from the teacher memory at t, roll P over the REAL latent actions for up to
10 steps, and ask the decoder "ball if we lift now?" from the imagined memory vs from the real teacher memory at the
same frame (both with the real frame as the read token).
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import zarr  # noqa: E402
from torch.amp import autocast  # noqa: E402

from latent_action_idm import IDM  # noqa: E402
from train_mcl_prior_phase1 import load_model  # noqa: E402
from world_model.models.smt_phase1.updater import PriorUpdater  # noqa: E402


def episode_data(model, idm, imean, istd, z, e, dev):
    """Teacher memory per frame (model space), model-space embeddings, actions, IDM means u_t (t -> t+1)."""
    ends = z["episode_ends"][:]; st = np.r_[0, ends[:-1]]
    s, n = int(st[e]), int(ends[e] - st[e])
    raw = z[model.embed_key][s:s + n].astype(np.float32)
    E = (torch.from_numpy(raw).to(dev) - model.embed_mean) / model.embed_std
    act = torch.from_numpy(z["action"][s:s + n].astype(np.float32)).to(dev)
    PA = torch.zeros(1, n, 2, device=dev); PA[0, 1:] = act[:n - 1]
    first = torch.zeros(1, n, dtype=torch.bool, device=dev); first[0, 0] = True
    Ei = torch.from_numpy((raw - imean) / istd).to(dev)
    with torch.no_grad():
        Z = model.encode_windows(E[None], PA, first)[0].float()
        U, _ = idm(Ei[:-1], Ei[1:], act[:-1])
    return Z, E, act, U


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="logdir/world_model/smt_phase1_all3u/naive")
    ap.add_argument("--idm", default="logdir/latent_actions/smoke_b10/idm_fdm.pt")
    ap.add_argument("--zarr", default="data/memtest4_all3u.zarr")
    ap.add_argument("--n_train", type=int, default=768)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device
    torch.manual_seed(0)
    out = ROOT / args.run / "latent_prior"
    out.mkdir(parents=True, exist_ok=True)

    model = load_model(Path(args.run) / "checkpoints" / "wm_latest.pt", dev)
    amp = dict(device_type=dev.split(":")[0], dtype=model.amp_dtype)
    ck = torch.load(ROOT / args.idm, map_location=dev, weights_only=False)
    imean, istd, d = ck["mean"], ck["std"], ck["args"]["d"]
    idm = IDM(1024, 2, d).to(dev); idm.load_state_dict(ck["idm"]); idm.eval()

    z = zarr.open_group(str(ROOT / args.zarr), mode="r")
    n_ep = len(z["episode_ends"])
    Zs, As, Us, Zn = [], [], [], []
    for e in range(args.n_train):
        Z, _, act, U = episode_data(model, idm, imean, istd, z, e, dev)
        Zs.append(Z[:-1]); Zn.append(Z[1:]); As.append(act[:-1]); Us.append(U)
    Zs, Zn, As, Us = torch.cat(Zs), torch.cat(Zn), torch.cat(As), torch.cat(Us)
    print(f"{len(Zs)} training transitions; u norm: moving {Us.norm(dim=-1)[Us.norm(dim=-1) > 1].numel()} with |u| > 1")

    P = PriorUpdater(act_dim=2 + d, depth=model.prior.rnn.layers.__len__() // 2 if hasattr(model.prior.rnn, "layers") else 8,
                     d_model=model.prior.d_model, n_registers=model.prior.n_registers).to(dev)
    # warm start: the run's own prior trunk; the action projection keeps its 2 real-action columns, u columns start at 0
    sd = {k: v for k, v in model.prior.state_dict().items() if not k.startswith("act_in")}
    missing = P.load_state_dict(sd, strict=False)
    with torch.no_grad():
        P.act_in.weight.zero_(); P.act_in.weight[:, :2] = model.prior.act_in.weight; P.act_in.bias.copy_(model.prior.act_in.bias)
    print("warm start: missing keys", [k for k in missing.missing_keys if not k.startswith("act_in")][:5])
    opt = torch.optim.AdamW(P.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-2)
    for step in range(args.steps + 1):
        i = torch.randint(0, len(Zs), (args.batch,), device=dev)
        with autocast(**amp):
            pred = P.step(Zs[i], torch.cat([As[i], Us[i]], -1))
        loss = (pred.float() - Zn[i]).pow(2).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(P.parameters(), 1.0); opt.step()
        if step % 500 == 0:
            with torch.no_grad(), autocast(**amp):
                base = (model.prior.step(Zs[i], As[i]).float() - Zn[i]).pow(2).mean()
            print(f"step {step:5d}  latent prior MSE {loss.item():.5f}   (the run's own prior, no u: {base.item():.5f})")
    torch.save({"P": P.state_dict(), "d": d, "idm": args.idm, "run": args.run}, out / "latent_prior.pt")

    # ---- tracking check, held-out episodes
    dec, F_ = model.reward_decoder, int(model.fut_len)
    A_LIFT = torch.tensor([0., 1.], device=dev)

    @torch.no_grad()
    def lift_now(zz, e_now):
        X = e_now[:, None].expand(-1, F_, -1)
        A = A_LIFT.expand(len(zz), F_, -1)
        with autocast(**amp):
            return torch.sigmoid(dec.ball_logit(dec(zz, X, A)).float()[:, 1])

    H = 10
    diff = {h: [] for h in range(1, H + 1)}
    side = {h: [] for h in range(1, H + 1)}
    diff_old = {h: [] for h in range(1, H + 1)}
    for e in range(args.n_train, n_ep):
        Z, E, act, U = episode_data(model, idm, imean, istd, z, e, dev)
        n = len(Z)
        p_real = lift_now(Z, E)
        for t0 in range(0, n - 1):
            zi, zo = Z[t0:t0 + 1], Z[t0:t0 + 1]
            for h in range(1, H + 1):
                t = t0 + h
                if t >= n or act[t - 1, 1] > .5:          # stop at a real lift: its u carries the revealed outcome
                    break
                with torch.no_grad(), autocast(**amp):
                    zi = P.step(zi, torch.cat([act[t - 1:t], U[t - 1:t]], -1)).float()
                    zo = model.prior.step(zo, act[t - 1:t]).float()
                pi, po = lift_now(zi, E[t:t + 1])[0], lift_now(zo, E[t:t + 1])[0]
                diff[h].append(float((pi - p_real[t]).abs())); side[h].append(bool((pi > .5) == (p_real[t] > .5)))
                diff_old[h].append(float((po - p_real[t]).abs()))
    print("\nheld-out tracking: 'ball if we lift now?' from the imagined memory vs the real teacher memory")
    print(f"{'steps ahead':>11s} {'n':>6s} {'|dp| latent prior':>18s} {'same side of 0.5':>17s} {'|dp| own prior (no u)':>22s}")
    for h in range(1, H + 1):
        if diff[h]:
            print(f"{h:11d} {len(diff[h]):6d} {np.mean(diff[h]):18.3f} {np.mean(side[h]):17.3f} {np.mean(diff_old[h]):22.3f}")
    print(f"-> {out / 'latent_prior.pt'}")


if __name__ == "__main__":
    main()
