"""Exclusive "where is the ball" classes for every still frame, from the decoder + imagined latent actions.

    pixi run python scripts/psr_labels_phase1.py [--lap <latent_action_prior.pt>] [--out <npz>]

For every at-rest hold frame (cups just down, each swap's end, idle frames, the lift onset; mid-swap frames
skipped -- the same frames as build_self_slot0_labels.py):

    1. ask the decoder "ball if we lift now?" on the memory as it is      -> yes: class "here" (no imagination)
    2. otherwise press each latent action for one block in imagination
       (latent-action prior), and ask again                                -> the ONE action that brings the
                                                                              ball out is the class
       more than one action brings it out -> conflict (no class); none -> no class

A class is named by its action ("here", "after u9", ...). The hidden slot is used only to score: the action
that swaps slot 0 with slot k should collect the frames whose ball is in slot k; actions that do not touch
slot 0 (1<->2, nothing) should never bring the ball out.

Reported for the training episodes on the TEACHER memory (what an alignment loss would act on) and for the
test episodes on the POSTERIOR memory. Labels (no ground truth inside `cls`) are saved for later use.
"""
import argparse
import sys
from collections import Counter
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
    ap.add_argument("--lap", default="logdir/world_model/smt_phase1_memtest4/window60_phase1/latent_action_prior/latent_action_prior.pt")
    ap.add_argument("--actions", nargs="+", default=["9:0-1", "8:0-2", "12:1-2", "6:none"],
                    help="latent action:what it is (slot pair or none) -- the pair is used ONLY to score; or 'auto'")
    ap.add_argument("--lap_eval", default=None, help="auto: the latent-action run's eval.json (code use; names for scoring)")
    ap.add_argument("--train_zarr", default="data/memtest4.zarr")
    ap.add_argument("--n_train", type=int, default=225)
    ap.add_argument("--test_zarr", default="data/memtest4_eval.zarr")
    ap.add_argument("--out", default="logdir/world_model/smt_phase1_memtest4/window60_phase1/latent_action_prior/psr_labels_train.npz")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device
    model = load_model(args.run, dev)
    amp = dict(device_type=dev.split(":")[0], dtype=model.amp_dtype)
    ck = torch.load(ROOT / args.lap, map_location=dev)
    lap = LatentActionPrior(model.prior, model.embed_dim, block=ck["block"], n_codes=ck["n_codes"]).to(dev).eval()
    lap.load_state_dict(ck["lap"])
    auto = args.actions == ["auto"]
    if auto:
        # every latent action the inverse model uses (>1% of test blocks, read off frames only); the pair is the
        # ground-truth NAME of each, used only to score -- the two label actions are chosen without it, below
        import json
        ev = json.load(open(ROOT / args.lap_eval))
        nm = {int(k): v for k, v in ev["meaning"].items()}
        acts = [(c, None if not nm.get(c, "idle").startswith("swap") else tuple(int(x) for x in nm[c].split(" ")[1].split("<->")))
                for c, sh in enumerate(ev["share"]) if sh > 0.01]
    else:
        acts = [(int(a.split(":")[0]), None if a.split(":")[1] == "none" else tuple(int(x) for x in a.split(":")[1].split("-")))
                for a in args.actions]
    key = model.embed_key
    norm = lambda v: (torch.from_numpy(np.asarray(v, np.float32)).to(dev) - model.embed_mean) / model.embed_std
    A_HOLD, A_LIFT = torch.tensor([1., 0.], device=dev), torch.tensor([0., 1.], device=dev)

    # the "ball if we lift now?" query (as build_self_slot0_labels.py)
    ztr = zarr.open_group(str(ROOT / args.train_zarr), mode="r")
    ends = ztr["episode_ends"][:]
    starts = np.concatenate([[0], ends[:-1]])
    e0 = next(e for e in range(len(ends)) if not ztr["ball_under_target"][ends[e] - 1])
    on0 = int(np.flatnonzero(ztr["phase"][starts[e0]:ends[e0]] == LIFT_PH)[0])
    REST, LIFT = norm(ztr[key][starts[e0] + on0]), norm(ztr[key][starts[e0] + on0 + 1:ends[e0]])
    F_, dec = int(model.fut_len), model.reward_decoder
    tx = [REST] * (IDLE + 1) + list(LIFT)
    ta = [A_HOLD] * IDLE + [A_LIFT] * (1 + len(LIFT))
    while len(tx) < F_ - 1:
        tx.append(LIFT[-1]); ta.append(A_LIFT)
    TX, TA, READ = torch.stack(tx[:F_ - 1]), torch.stack(ta[:F_ - 1]), 1 + IDLE + 1

    @torch.no_grad()
    def lift_now(z):
        out = []
        for i in range(0, len(z), 512):
            zz = z[i:i + 512]
            X = torch.cat([REST.expand(len(zz), 1, -1), TX[None].expand(len(zz), -1, -1)], 1)
            A = torch.cat([A_HOLD.expand(len(zz), 1, -1), TA[None].expand(len(zz), -1, -1)], 1)
            with autocast(**amp):
                out.append(torch.sigmoid(dec.ball_logit(dec(zz, X, A)).float()[:, READ]))
        return torch.cat(out).cpu().numpy()

    @torch.no_grad()
    def press(z, code):
        out = []
        for i in range(0, len(z), 512):
            zz = z[i:i + 512]
            with autocast(**amp):
                steps = int(ck["durations"][code]) if "durations" in ck else lap.block   # still-frame segments: learned length
                path = lap.rollout(zz, A_HOLD.expand(len(zz), steps, -1),
                                   torch.full((len(zz),), code, device=dev, dtype=torch.long))
            out.append(path[:, -1].float())
        return torch.cat(out)

    def still_frames(zarr_path, episodes, teacher):
        zg = zarr.open_group(str(ROOT / zarr_path), mode="r")
        en = zg["episode_ends"][:]
        st = np.concatenate([[0], en[:-1]])
        seqs, ph, slot, act = sequences(ROOT / zarr_path), zg["phase"][:], zg["ball_slot"][:], zg["action"][:].astype(np.float32)
        Z, info = [], []
        for e in episodes:
            s, n = int(st[e]), int(en[e] - st[e])
            E = norm(zg[key][s:s + n])[None]
            PA = torch.zeros(1, n, 2, device=dev)
            PA[0, 1:] = torch.from_numpy(act[s:s + n - 1]).to(dev)
            first = torch.zeros(1, n, dtype=torch.bool, device=dev); first[0, 0] = True
            with torch.no_grad():
                if teacher:
                    Zt = model.encode_windows(E, PA, first)[0]
                else:
                    zw = model.encode_windows(E[:, :1], PA[:, :1], first[:, :1])
                    zs, z = [zw], zw[:, -1]
                    with autocast(**amp):
                        for t in range(1, n):
                            z = model.posterior.step(z, E[:, t], PA[:, t]).float(); zs.append(z.unsqueeze(1))
                    Zt = torch.cat(zs, 1)[0].float()
            p = ph[s:s + n]
            h0, on, k = int(np.flatnonzero(p == HOLD_PH)[0]), int(np.flatnonzero(p == LIFT_PH)[0]), len(seqs[e][1])
            for t in range(h0 + MOVE - 1, on + 1):
                j = t - (h0 + MOVE)
                if 0 <= j < SWAP * k and j % SWAP != SWAP - 1:
                    continue                                              # mid-swap
                Z.append(Zt[t])
                info.append((e, t, int(slot[s + t]), 0 if j < 0 else min(k, (j + 1) // SWAP)))
        return torch.stack(Z), np.array(info)

    def run(label, zarr_path, episodes, teacher):
        Z, info = still_frames(zarr_path, episodes, teacher)
        truth, done = info[:, 2], info[:, 3]
        p0 = lift_now(Z)
        here = p0 > 0.5
        pa = {c: lift_now(press(Z, c)) for c, _ in acts}
        print(f"\n=== {label}: {len(Z)} still frames ===")
        print(f"1. decoder directly (no imagination): says 'ball here' on {here.mean():.3f} (truly there {np.mean(truth == 0):.3f}); "
              f"right {np.mean(here == (truth == 0)):.3f}; of its 'ball here', truly slot 0: {np.mean(truth[here] == 0):.3f}; "
              f"of its 'no', truly not slot 0: {np.mean(truth[~here] != 0):.3f}")
        print(f"2. frames where it says 'no' ({(~here).sum()}): press each latent action, ask again")
        print(f"   {'action':14s} {'says ball':>10s}   {'of those, ball truly in the slot this action brings to 0':s}")
        for c, pr in acts:
            yes = (~here) & (pa[c] > 0.5)
            target = None if pr is None or 0 not in pr else (pr[1] if pr[0] == 0 else pr[0])
            right = np.mean(truth[yes] == target) if yes.any() and target is not None else float("nan")
            what = f"slot {target}" if target is not None else "(should never)"
            print(f"   u{c:<3d} {('nothing' if pr is None else f'{pr[0]}<->{pr[1]}'):9s} {yes.sum() / max((~here).sum(), 1):10.3f}   "
                  f"{what}: {right:.3f}" if target is not None else
                  f"   u{c:<3d} {('nothing' if pr is None else f'{pr[0]}<->{pr[1]}'):9s} {yes.sum() / max((~here).sum(), 1):10.3f}   {what}")
        # exclusive classes
        yes_mat = np.stack([(~here) & (pa[c] > 0.5) for c, _ in acts], 1)
        n_yes = yes_mat.sum(1)
        cls = np.full(len(Z), -1)                                         # -1 none, -2 conflict, 0 here, 1+i action i
        cls[here] = 0
        one = (~here) & (n_yes == 1)
        cls[one] = 1 + yes_mat[one].argmax(1)
        cls[(~here) & (n_yes > 1)] = -2
        names = {0: "here (no imagination)"} | {1 + i: f"after u{c} ({'nothing' if pr is None else f'{pr[0]}<->{pr[1]}'})"
                                                for i, (c, pr) in enumerate(acts)} | {-2: "conflict (>1 action)", -1: "no class"}
        expect = {0: 0} | {1 + i: (None if pr is None or 0 not in pr else (pr[1] if pr[0] == 0 else pr[0]))
                          for i, (c, pr) in enumerate(acts)}
        print("3. exclusive classes")
        print(f"   {'class':26s} {'share':>6s} {'n':>6s}   true slot 0 / 1 / 2       class accuracy")
        for k in [0] + [1 + i for i in range(len(acts))] + [-2, -1]:
            m = cls == k
            if not m.any():
                continue
            dist = [np.mean(truth[m] == s_) for s_ in range(3)]
            acc = f"{np.mean(truth[m] == expect[k]):.3f}" if k in expect and expect[k] is not None else "-"
            print(f"   {names[k]:26s} {m.mean():6.3f} {m.sum():6d}   {dist[0]:.2f} / {dist[1]:.2f} / {dist[2]:.2f}        {acc}")
        lab = np.array([expect.get(k) if k in expect and expect.get(k) is not None else -1 for k in cls])
        has = lab >= 0
        print(f"   frames with a slot class: {has.mean():.3f}; accuracy of those classes {np.mean(lab[has] == truth[has]):.3f}")
        print("   by swaps done:  " + "  ".join(
            f"{d}: cov {has[done == d].mean():.2f} acc {np.mean(lab[(done == d) & has] == truth[(done == d) & has]):.2f}"
            for d in sorted(set(done.tolist())) if ((done == d) & has).any()))
        for s_ in range(3):
            m = truth == s_
            print(f"   ball truly in slot {s_} (n {m.sum()}): given " + ", ".join(
                f"{names[k].split(' (')[0]} {np.mean(cls[m] == k):.2f}" for k in [0] + [1 + i for i in range(len(acts))] + [-2, -1]))
        return info, cls, p0, pa

    info, cls, p0, pa = run("TRAINING episodes, teacher memory", args.train_zarr, range(args.n_train), True)
    # the two "swap involving slot 0" actions, chosen WITHOUT labels: they flip the decoder's own answer most
    # ("ball here" -> no, and "no" -> ball); "nothing" and the swap of the other two cups should leave it alone
    here = p0 > 0.5
    # "nothing" is excluded first as the action the inverse model reads off the frames most often (idle blocks
    # dominate any hold), since its drift can flip the answer about as often as a weakly-imagined swap does
    flip = {c: float(np.mean(pa[c][here] < 0.5) + np.mean(pa[c][~here] > 0.5)) for c, _ in acts}
    nothing = int(np.argmax(ev["share"])) if auto else None
    pair_codes = [c for c in sorted(flip, key=flip.get, reverse=True) if c != nothing][:2]
    if auto:
        print(f"most-used latent action (taken as 'nothing', excluded): u{nothing} ({max(ev['share']):.2f} of blocks)")
    print("\nlabel-free choice of the two swap actions (flip rate of the decoder's own answer): "
          + ", ".join(f"u{c} {flip[c]:.3f}" + (" <- picked" if c in pair_codes else "") for c in flip)
          + "   | their ground-truth names (scoring only): "
          + ", ".join(f"u{c} = {dict(acts)[c]}" for c in pair_codes))
    np.savez(ROOT / args.out, episode=info[:, 0], t=info[:, 1], cls=cls, p_here=p0, pair_codes=np.array(pair_codes),
             **{f"p_after_u{c}": pa[c] for c, _ in acts}, truth=info[:, 2], swaps_done=info[:, 3],
             note=np.array("cls: 0 here, 1+i after action i (order of --actions), -2 conflict, -1 none; truth for scoring only"))
    n_test = len(zarr.open_group(str(ROOT / args.test_zarr), mode="r")["episode_ends"])
    run("TEST episodes, posterior memory", args.test_zarr, range(n_test), False)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
