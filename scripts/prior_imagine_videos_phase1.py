"""Let the PRIOR imagine from a "no ball here" moment, render what it imagines, and decode its belief.

    pixi run python scripts/prior_imagine_videos_phase1.py --episode <eval ep> [--holds 0 7 14] --outdir <dir>

From the last still hold frame before the lift (ground truth: ball NOT under the lifted cup, slot 0), the
posterior's state is handed to the prior, which advances on the action flags alone (hold ... then lift) and
never sees a frame. The prior is deterministic, so the three rollouts differ only in how long they hold
before lifting: 0, 7 (one swap's length), 14 (two swaps).

Rendering: the stage-1 run has no fitted readout, so a ridge readout (memory -> frame embedding) is fitted
here on posterior states of the run's own training episodes, and the frames are drawn by the P1-250 pixel
decoder (same frozen encoder and normalisation -- checked). Decoded value per step = the reward decoder's
answer to "if we lift right now, is the ball there?" (the spliced no-swap-then-lift query that the
Align-self labels use), on the imagined state.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import zarr  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from tensordict import TensorDict  # noqa: E402
from torch.amp import autocast  # noqa: E402

from info_gathering.dataset_smt_prelim import HALFWAY, dense_ball_slot  # noqa: E402
from info_gathering.swap_sequences import sequences  # noqa: E402
from world_model.utils.checkpoint import CheckpointManager, load_frozen_models  # noqa: E402

P1_DECODER = ("logdir/world_model/smt_phase1_memtest4/2026-09-20_13-58-25_memtest4_phase1/readout/2026-09-20_14-32-18/"
              "pixel_decoder_phase1/2026-09-20_15-09-31/checkpoints/decoder_latest.pt")
HOLD_PH, LIFT_PH, IDLE = 1, 2, 3
A_HOLD, A_LIFT = [1., 0.], [0., 1.]
SURF, INK, MUTED, GOOD, BAD, BALL = (18, 18, 22), (235, 235, 235), (150, 150, 150), (120, 200, 120), (230, 70, 70), (250, 200, 40)


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def episode(z, e, key):
    ends = z["episode_ends"][:]
    s, n = (0 if e == 0 else int(ends[e - 1])), int(ends[e] - (0 if e == 0 else ends[e - 1]))
    return s, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="logdir/world_model/smt_phase1_memtest4/window60_alignself_phase1")
    ap.add_argument("--train_zarr", default="data/memtest4.zarr")
    ap.add_argument("--n_train", type=int, default=225)
    ap.add_argument("--zarr", default="data/memtest4_eval.zarr")
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--holds", type=int, nargs="+", default=[0, 7, 14])
    ap.add_argument("--lift_len", type=int, default=10)
    ap.add_argument("--outdir", default="figures/prior_imagine")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--until", action="store_true", help="imagine with one action until the decoder says ball, then lift")
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--ckpt", default=None, help="model checkpoint (default <run>/checkpoints/wm_latest.pt); a "
                    "readout-stage checkpoint renders with its TRAINED readout instead of a ridge fit")
    ap.add_argument("--mcl", default=None, help="branching prior (scripts/train_mcl_prior_phase1.py): "
                    "--until then SAMPLES a code every step instead of using the deterministic prior")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--lap", default=None, help="latent-action prior (train_latent_action_prior_phase1.py): --until "
                    "then PRESSES each --press action for one block, then lifts with the deterministic prior")
    ap.add_argument("--press", nargs="+", default=["9:0-1", "8:0-2", "12:1-2", "6:none"],
                    help="latent action:what it means (slot pair a-b, or none), e.g. 9:0-1")
    ap.add_argument("--check_every", type=int, default=1, help="ask the decoder only every N imagined steps "
                    "(6 = at the end of each swap block, when the cups are at rest)")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--gt_video", action="store_true", help="--until: real vs imagined side by side, each with its "
                    "own ground truth and decoder value (imagined truth = start slot moved by the imagined swaps)")
    ap.add_argument("--start", default="onset-1", help="start frame: an int, or onset-1, or swap:<j> = the at-rest "
                    "frame just before swap j (0-based)")
    args = ap.parse_args()
    dev = args.device

    st = CheckpointManager.load(ROOT / (args.ckpt or Path(args.run) / "checkpoints" / "wm_latest.pt"))
    cfg = OmegaConf.create(st["config"])
    tr, _ = instantiate(cfg.data)
    model = instantiate(cfg.model, action_dim=tr.action_dim, device=dev, _recursive_=False)
    model.load_train_state_dict(st["model"])
    model.requires_grad_(False).eval()
    model.observe_mode = "recurrent"
    p1, pix, *_ = load_frozen_models(str(ROOT / P1_DECODER), dev)
    assert torch.equal(p1.embed_mean, model.embed_mean) and torch.equal(p1.embed_std, model.embed_std), \
        "pixel decoder was fitted under a different embedding normalisation"
    del p1
    key = model.embed_key
    norm = lambda v: (torch.from_numpy(np.asarray(v, np.float32)).to(dev) - model.embed_mean) / model.embed_std

    def posterior(zg, e):
        s, n = episode(zg, e, key)
        E = norm(zg[key][s:s + n])[None]
        PA = torch.zeros(1, n, 2, device=dev)
        PA[0, 1:] = torch.from_numpy(zg["action"][s:s + n - 1].astype(np.float32)).to(dev)
        first = torch.zeros(1, n, dtype=torch.bool, device=dev); first[0, 0] = True
        with torch.no_grad():
            Z = _observe(model, E, PA, first)
        return Z[0].float(), E[0], s, n

    # ---- ridge readout: memory -> normalised frame embedding, fitted on the run's own training episodes
    trained_readout = "readout_step" in st
    ztr = zarr.open_group(str(ROOT / args.train_zarr), mode="r")
    Xs, Ys = [], []
    for e in range(0 if trained_readout else args.n_train):
        Z, E, _, _ = posterior(ztr, e)
        Xs.append(Z.flatten(1)); Ys.append(E)
    X, Y = (torch.cat(Xs), torch.cat(Ys)) if Xs else (torch.zeros(2, 1, device=dev), torch.zeros(2, 1, device=dev))
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = torch.cat([(X - mu) / sd, torch.ones(len(X), 1, device=dev)], 1)
    zev = zarr.open_group(str(ROOT / args.zarr), mode="r")
    Xv, Yv = [], []
    for e in range(0, 0 if trained_readout else 315, 9):
        Z, E, _, _ = posterior(zev, e)
        Xv.append(Z.flatten(1)); Yv.append(E)
    best = None
    if not trained_readout:
        Xv = torch.cat([(torch.cat(Xv) - mu) / sd, torch.ones(sum(len(v) for v in Yv), 1, device=dev)], 1)
        Yv = torch.cat(Yv)
        G, B = Xn.T @ Xn, Xn.T @ Y
    for lam in (() if trained_readout else (1e0, 1e1, 1e2, 1e3)):
        W = torch.linalg.solve(G + lam * torch.eye(len(G), device=dev), B)
        r2 = 1 - ((Xv @ W - Yv) ** 2).sum() / ((Yv - Yv.mean(0)) ** 2).sum()
        print(f"ridge lambda {lam:g}: test-set R2 {float(r2):.3f}")
        if best is None or r2 > best[0]:
            best = (float(r2), W)
    if trained_readout:
        r2 = float("nan")
        readout = lambda z: model.readout(z.unsqueeze(0))[0].float()
    else:
        r2, W = best
        readout = lambda z: (((z.flatten(1) - mu) / sd) @ W[:-1] + W[-1])

    @torch.no_grad()
    def render(emb):
        f = pix.render(TensorDict({"embed": emb[None]}, batch_size=(1, len(emb))))[0]
        return (f.clamp(0, 1) * 255).byte().cpu().numpy()

    # ---- the "lift right now?" query (as scripts/build_self_slot0_labels.py)
    F, dec = int(model.fut_len), model.reward_decoder
    ends = ztr["episode_ends"][:]
    e0 = next(e for e in range(len(ends)) if not ztr["ball_under_target"][ends[e] - 1])
    s0, n0 = episode(ztr, e0, key)
    on0 = int(np.flatnonzero(ztr["phase"][s0:s0 + n0] == LIFT_PH)[0])
    REST, LIFT = norm(ztr[key][s0 + on0]), norm(ztr[key][s0 + on0 + 1:s0 + n0])
    AH, AL = torch.tensor(A_HOLD, device=dev), torch.tensor(A_LIFT, device=dev)
    tx = [REST] * (IDLE + 1) + list(LIFT)
    ta = [AH] * IDLE + [AL] * (1 + len(LIFT))
    while len(tx) < F - 1:
        tx.append(LIFT[-1]); ta.append(AL)
    TX, TA = torch.stack(tx[:F - 1]), torch.stack(ta[:F - 1])
    READ = 1 + IDLE + 1

    @torch.no_grad()
    def lift_now(zs):
        X_ = torch.cat([REST.expand(len(zs), 1, -1), TX[None].expand(len(zs), -1, -1)], 1)
        A_ = torch.cat([AH.expand(len(zs), 1, -1), TA[None].expand(len(zs), -1, -1)], 1)
        with autocast(dev.split(":")[0], dtype=model.amp_dtype):
            lg = dec.ball_logit(dec(zs, X_, A_)).float()[:, READ]
        return torch.sigmoid(lg).cpu().numpy()

    # ---- real swap clips by slot pair: mean embedding at offsets 0..6 of a swap block (cups down, so the
    # frames are identical across episodes). Used only to NAME the swap an imagined block looks like.
    templates = {}
    if args.gt_video:
        tr_seqs = sequences(ROOT / args.train_zarr)
        tr_ph = ztr["phase"][:]
        acc = {}
        for e_ in range(args.n_train):
            s_, n_ = episode(ztr, e_, key)
            h_ = int(np.flatnonzero(tr_ph[s_:s_ + n_] == HOLD_PH)[0])
            for j, pr in enumerate(tr_seqs[e_][1]):
                b = s_ + h_ + 4 + 6 * j
                acc.setdefault(pr, []).append(ztr[key][b:b + 7])
        templates = {pr: norm(np.mean(v, 0)) for pr, v in acc.items()}
        templates[None] = REST[None].expand(7, -1)

    # ---- the start point and the three imagined rollouts
    e = args.episode
    Z, E, s, n = posterior(zev, e)
    ph, slot = zev["phase"][s:s + n], zev["ball_slot"][s:s + n]
    rgb = zev["rgb"][s:s + n]
    onset = int(np.flatnonzero(ph == LIFT_PH)[0])
    h0 = int(np.flatnonzero(ph == HOLD_PH)[0])
    t0 = (onset - 1 if args.start == "onset-1" else
          h0 + 4 + 6 * int(args.start.split(":")[1]) if args.start.startswith("swap:") else int(args.start))
    truth = int(slot[t0])
    trained = set(sequences(ROOT / args.train_zarr)[:args.n_train])
    seq = sequences(ROOT / args.zarr)[e]
    seen = seq in trained
    print(f"ep {e}: sequence start {seq[0]} swaps {seq[1]} ({'seen' if seen else 'UNSEEN'}), "
          f"start point t={t0} (onset {onset}), true ball slot {truth}, readout R2 {r2:.3f}")
    p_start = float(lift_now(Z[t0:t0 + 1])[0])
    print(f"posterior at the start point: P(ball under the lifted cup) = {p_start:.2f}")
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    if args.until:
        # keep imagining with a fixed action until the decoder says "ball", then imagine the lift
        mcl = None
        if args.mcl:
            from world_model.models.smt_phase1.mcl_prior import MCLPrior
            ck = torch.load(ROOT / args.mcl, map_location=dev)
            mcl = MCLPrior(model.prior, n_codes=ck["n_codes"]).to(dev).eval()
            mcl.load_state_dict(ck["mcl"])
            block = int(ck.get("block", 1))           # draw a code once per block and hold it
        runs = ([(f"sample{k}", A_HOLD, k) for k in range(args.samples)] if mcl else [("hold", A_HOLD, 0), ("zero", [0., 0.], 0)])
        lap, meaning = None, {}
        if args.lap:
            from world_model.models.smt_phase1.latent_action_prior import LatentActionPrior
            ck = torch.load(ROOT / args.lap, map_location=dev)
            lap = LatentActionPrior(model.prior, model.embed_dim, block=ck["block"], n_codes=ck["n_codes"]).to(dev).eval()
            lap.load_state_dict(ck["lap"])
            for sp in args.press:
                c_, m_ = sp.split(":")
                meaning[int(c_)] = None if m_ == "none" else tuple(int(x) for x in m_.split("-"))
            runs = [(f"press_u{c_}_{'nothing' if m_ is None else f'{m_[0]}-{m_[1]}'}", A_HOLD, c_) for c_, m_ in meaning.items()]

        held = {"code": None, "left": 0}

        def advance(z, a_t, gen, new_phase=False):
            with torch.no_grad(), autocast(dev.split(":")[0], dtype=model.amp_dtype):
                if lap is not None:                   # pressed action while holding; the lift was never trained
                    if a_t[0, 1] > 0:
                        return model.prior.step(z, a_t).float(), None
                    return lap.step(z, a_t, torch.tensor([held["code"]], device=dev)).float(), held["code"]
                if mcl is None:
                    return model.prior.step(z, a_t).float(), None
                if held["left"] == 0 or new_phase:
                    probs = torch.softmax(mcl.code_logits(z, a_t).float(), -1)
                    held["code"] = torch.multinomial(probs, 1, generator=gen).squeeze(-1)
                    held["left"] = block
                held["left"] -= 1
                return mcl.step(z, a_t, held["code"]).float(), int(held["code"][0])

        if mcl is not None:
            with torch.no_grad(), autocast(dev.split(":")[0], dtype=model.amp_dtype):
                pc = torch.softmax(mcl.code_logits(Z[t0:t0 + 1], torch.tensor([A_HOLD], device=dev)).float(), -1)[0]
            print("  p(code | start state, hold): " + " ".join(f"c{k} {v:.2f}" for k, v in enumerate(pc.tolist())))
        for name, a, seed in runs:
            gen = torch.Generator(device=dev).manual_seed(seed)
            codes = [None]
            z, zs, ps, moves = Z[t0:t0 + 1], [Z[t0:t0 + 1]], [p_start], [0.0]
            held["left"] = 0
            if lap is not None:
                held["code"] = seed
            hit = None
            a_t = torch.tensor(a, device=dev)[None]
            for i in range(1, args.max_steps + 1):
                z_new, c = advance(z, a_t, gen)
                codes.append(c)
                moves.append(float((z_new - z).norm() / z.norm()))
                z = z_new
                zs.append(z); ps.append(float(lift_now(z)[0]))
                if lap is not None and i == lap.block:      # one pressed block, then lift
                    hit = i if ps[-1] > 0.5 else None
                    break
                if lap is None and ps[-1] > 0.5 and i % args.check_every == 0:
                    hit = i
                    break
            n_hold = len(zs) - 1
            al = torch.tensor(A_LIFT, device=dev)[None]
            for li in range(args.lift_len):
                z, c = advance(z, al, gen, new_phase=li == 0)
                codes.append(c)
                zs.append(z); ps.append(float(lift_now(z)[0])); moves.append(float("nan"))
            zi = torch.cat(zs)
            frames = render(readout(zi))
            p = np.array(ps)
            labels = ["start (posterior)"] + ["hold"] * n_hold + ["LIFT"] * args.lift_len
            if mcl is not None:
                labels = [l if codes[i] is None else f"{l} c{codes[i]}" for i, l in enumerate(labels)]
                print(f"  {name}: codes drawn (hold) " + "".join(str(c) for c in codes[1:n_hold + 1]) + "  (lift) " + "".join(str(c) for c in codes[n_hold + 1:]))
            print(f"  action {name} {a}: {'decoder says BALL at imagined step ' + str(hit) if hit else f'never says ball in {args.max_steps} steps'}"
                  f"; max P(ball) during hold {max(ps[1:n_hold + 1]):.3f};"
                  f" state change per step: first {moves[1]:.3f}, step 10 {moves[min(10, n_hold)]:.4f}, last {moves[n_hold]:.5f}")
            print("    P(ball) every 10 steps: " + " ".join(f"{v:.2f}" for v in p[:n_hold + 1:10])
                  + " | lift: " + " ".join(f"{v:.2f}" for v in p[n_hold + 1:]))
            real_idx = [min(t0 + i, n - 1) for i in range(len(zi))]
            if args.gt_video:
                with torch.no_grad():
                    emb_im = readout(zi).float()
                p_real = lift_now(Z[[min(t0 + i, n - 1) for i in range(len(zi))]])
                dense = dense_ball_slot(slot[None].astype(np.int64), np.array([n]))[0]  # switches at the halfway point
                real_slot = [int(dense[k]) for k in real_idx]
                k_sw = len(seq[1])
                mov_r = [0 < k - (h0 + 4) < 6 * k_sw and (k - (h0 + 4)) % 6 != 0 for k in real_idx]
                lift_r = [k >= onset for k in real_idx]
                p_on = float(lift_now(Z[onset - 1:onset])[0])
                p_real = np.where(lift_r, p_on, p_real)                       # in the lift: the call made at its onset
                imp, named, cur = [truth], [], truth
                for blk in range(n_hold // 6):
                    seg = emb_im[6 * blk + 1:6 * blk + 6]                       # the 5 moving frames of the block
                    cands = []
                    for pr, T in templates.items():
                        for lag in (0, 1):                                      # imagination may run a frame late
                            cands.append((float((seg - T[1 - lag:6 - lag]).pow(2).mean()), pr, lag))
                    d0, pr, _ = min(cands)
                    if lap is not None:
                        print(f"  {name}: imagined frames look like {'no swap' if pr is None else f'{pr[0]}<->{pr[1]}'}")
                        pr = meaning[seed]                     # truth follows the PRESSED swap
                    named.append(pr)
                    if pr is not None and cur in pr:
                        cur = pr[1] if cur == pr[0] else pr[0]
                    imp += [imp[-1]] * (HALFWAY) + [cur] * (6 - HALFWAY)
                imp += [cur] * (len(zi) - len(imp))
                mov_i = [0 < i <= 6 * len(named) and i % 6 != 0 and named[(i - 1) // 6] is not None for i in range(len(zi))]
                lift_i = [i > n_hold for i in range(len(zi))]
                p_show = np.where(lift_i, p[n_hold], p)
                print(f"  {name}: imagined blocks look like " + ", ".join("no swap" if q is None else f"{q[0]}<->{q[1]}" for q in named)
                      + f"  -> ball should end in slot {cur}")
                vid = [gt_panel(rgb[real_idx[i]], frames[i], real_slot[i], imp[i], p_real[i], p_show[i], p_real, p_show, i,
                                labels[i], named[:(i - 1) // 6 + (1 if i >= 6 else 0)] if i else [], e, seen, seq, name, t0, n,
                                (mov_r[i], mov_i[i]), (lift_r[i], lift_i[i]))
                       for i in range(len(zi))]
                imageio.mimwrite(out / f"ep{e:03d}_gt_{name}.mp4", vid, fps=args.fps, macro_block_size=1)
                print(f"  -> {out / f'ep{e:03d}_gt_{name}.mp4'}")
                continue
            vid = [panel(rgb[t0], rgb[real_idx[i]], frames[i], p, i, labels[i], truth, f"{name}-until-ball", e, seen, seq, r2)
                   for i in range(len(zi))]
            imageio.mimwrite(out / f"ep{e:03d}_until_{name}.mp4", vid, fps=args.fps, macro_block_size=1)
            keep = sorted(set(list(range(0, n_hold + 1, max(1, n_hold // 14))) + list(range(n_hold + 1, len(zi)))))
            row = sheet(frames[keep], p[keep], [labels[k] for k in keep], f"{name}\n{n_hold} steps", every=len(keep) < len(zi))
            Image.fromarray(row).save(out / f"ep{e:03d}_until_{name}_sheet.png")
            print(f"  -> {out / f'ep{e:03d}_until_{name}.mp4'}")
        return
    sheets = []
    for h in args.holds:
        acts = [A_HOLD] * h + [A_LIFT] * args.lift_len
        pa = torch.tensor(acts, device=dev)[None]
        imag = model.imagine(TensorDict({"z": Z[None, :t0 + 1]}, batch_size=(1, t0 + 1)), pa)
        zi = torch.cat([Z[t0:t0 + 1], imag["z"][0].float()])            # step 0 = the start point itself
        frames = render(readout(zi))
        p = lift_now(zi)
        real_idx = [min(t0 + i, n - 1) for i in range(len(zi))]
        labels = ["start (posterior)"] + ["hold"] * h + ["LIFT"] * args.lift_len
        print(f"  hold {h:2d}: P(ball) per step " + " ".join(f"{v:.2f}" for v in p))
        vid = []
        for i in range(len(zi)):
            vid.append(panel(rgb[t0], rgb[real_idx[i]], frames[i], p, i, labels[i], truth, h, e, seen, seq, r2))
        path = out / f"ep{e:03d}_hold{h:02d}.mp4"
        imageio.mimwrite(path, vid, fps=3, macro_block_size=1)
        sheets.append(sheet(frames, p, labels, h))
        print(f"  -> {path}")
    wmax = max(s_.shape[1] for s_ in sheets)
    img = np.concatenate([np.pad(s_, ((0, 0), (0, wmax - s_.shape[1]), (0, 0)), constant_values=SURF[0])
                          for s_ in sheets], 0)
    head = Image.new("RGB", (img.shape[1], 40), SURF)
    ImageDraw.Draw(head).text((8, 8), f"ep {e}: prior imagines from t={t0} (truth: ball in slot {truth}, lifted cup = "
                              f"slot 0). Number = P(ball there if we lift now), decoded from the imagined state.",
                              font=font(16), fill=INK)
    Image.fromarray(np.concatenate([np.asarray(head), img], 0)).save(out / f"ep{e:03d}_sheet.png")
    print(f"  -> {out / f'ep{e:03d}_sheet.png'}")


def _observe(model, E, PA, first):
    """model.observe on cached embeddings (identical to encoding the frames: verified by the cache)."""
    warm = max(1, min(model.eval_warmup, E.shape[1]))
    z_w = model.encode_windows(E[:, :warm], PA[:, :warm], first[:, :warm])
    zs, z = [z_w], z_w[:, -1]
    with torch.no_grad(), autocast(model.device.type, dtype=model.amp_dtype):
        for t in range(warm, E.shape[1]):
            z = model.posterior.step(z, E[:, t], PA[:, t]).float()
            zs.append(z.unsqueeze(1))
    return torch.cat(zs, 1)


def cups(d, x, y, w, truth, label):
    d.text((x, y), label, font=font(13), fill=MUTED)
    for k in range(3):
        cx = x + 10 + k * (w // 3)
        d.rectangle([cx, y + 22, cx + w // 3 - 20, y + 52], outline=INK, width=2)
        d.text((cx + 4, y + 56), f"slot {k}" + (" (lifted)" if k == 0 else ""), font=font(11), fill=MUTED)
        if k == truth:
            r = 9
            mx = cx + (w // 3 - 20) // 2
            d.ellipse([mx - r, y + 32, mx + r, y + 50], fill=BALL)


def panel(start, real, imag, p, i, lab, truth, h, e, seen, seq, r2):
    S, gap = 260, 14
    img = Image.new("RGB", (3 * S + 4 * gap, S + 210), SURF)
    d = ImageDraw.Draw(img)
    d.text((gap, 8), f"ep {e} ({'seen' if seen else 'unseen'} sequence {seq[0]}{list(seq[1])}) - prior imagines: "
                     f"{h if isinstance(h, str) else f'hold {h} then lift'}", font=font(17), fill=INK)
    d.text((gap, 32), f"imagined step {i}/{len(p) - 1}   action: {lab}", font=font(15), fill=BAD if i else GOOD)
    y0 = 58
    for j, (arr, txt) in enumerate(((start, "real frame at the start point"),
                                    (real, "real episode, same step (prior never sees it)"),
                                    (imag, "IMAGINED by the prior"))):
        x = gap + j * (S + gap)
        img.paste(Image.fromarray(arr).resize((S, S), Image.Resampling.NEAREST), (x, y0))
        d.text((x, y0 + S + 4), txt, font=font(12), fill=BAD if j == 2 else MUTED)
    by = y0 + S + 26
    cups(d, gap, by, 260, truth, "ground truth (where the ball really is):")
    x = gap + 300
    d.text((x, by), "decoded: P(ball under the lifted cup if we lift now)", font=font(13), fill=MUTED)
    W_, H_ = 3 * S + 2 * gap - 300, 80
    d.rectangle([x, by + 22, x + W_, by + 22 + H_], outline=(70, 70, 76))
    d.line([x, by + 22 + H_ // 2, x + W_, by + 22 + H_ // 2], fill=(55, 55, 60))
    pts = [(x + int(W_ * k / max(len(p) - 1, 1)), by + 22 + int(H_ * (1 - v))) for k, v in enumerate(p)]
    d.line(pts, fill=(120, 120, 140), width=2)
    cx, cy = pts[i]
    ok = (p[i] > 0.5) == (truth == 0)
    d.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=GOOD if ok else BAD)
    d.text((x, by + 26 + H_), f"now {p[i]:.2f} -> says {'BALL' if p[i] > 0.5 else 'no ball'}   "
                              f"({'correct' if ok else 'WRONG'}; truth: slot {truth})", font=font(14),
           fill=GOOD if ok else BAD)
    return np.asarray(img)


def gt_panel(real, imag, real_slot, imp_slot, pr, pi, p_real, p_im, i, lab, named, e, seen, seq, name, t0, n,
             moving=(False, False), lifting=(False, False)):
    """Real (left) and imagined (right), each with its own ground truth and decoder value."""
    S, gap, W = 300, 24, 2 * 300 + 3 * 24
    img = Image.new("RGB", (W, S + 330), SURF)
    d = ImageDraw.Draw(img)
    d.text((gap, 8), f"ep {e} ({'seen' if seen else 'unseen'}: start {seq[0]}, swaps {list(seq[1])}) - {name}",
           font=font(16), fill=INK)
    d.text((gap, 30), f"step {i}   imagined action: {lab}", font=font(15), fill=GOOD if i == 0 else BAD)
    y0 = 56
    imag_txt = "imagined swaps: " + (", ".join("none" if q is None else f"{q[0]}<->{q[1]}" for q in named) or "-")
    for j, (arr, title, gt, pv, gtlab) in enumerate((
            (real, f"REAL episode (frame {min(t0 + i, n - 1)})", real_slot, pr, "ground truth: ball is in"),
            (imag, "IMAGINED by the branching prior", imp_slot, pi, "ball SHOULD be in (start slot + imagined swaps)"))):
        x = gap + j * (S + gap)
        img.paste(Image.fromarray(arr).resize((S, S), Image.Resampling.NEAREST), (x, y0))
        d.text((x, y0 + S + 4), title, font=font(13), fill=MUTED if j == 0 else BAD)
        if j == 1:
            d.rectangle([x - 2, y0 - 2, x + S + 1, y0 + S + 1], outline=BAD, width=2)
        cups(d, x, y0 + S + 26, S, gt, gtlab)
        ok = (pv > 0.5) == (gt == 0)
        q = "decoder's call for this lift (made at its onset)" if lifting[j] else "decoder: P(ball if we lift now)"
        d.text((x, y0 + S + 104), f"{q} = {pv:.2f}", font=font(13), fill=INK)
        if moving[j]:
            d.text((x, y0 + S + 124), f"says {'BALL' if pv > 0.5 else 'no ball'}  (mid-swap: not scored)",
                   font=font(14), fill=MUTED)
        else:
            d.text((x, y0 + S + 124), f"says {'BALL' if pv > 0.5 else 'no ball'} -> {'correct' if ok else 'WRONG'}",
                   font=font(14), fill=GOOD if ok else BAD)
    d.text((gap + S + gap, y0 + S + 146), imag_txt[:60], font=font(13), fill=MUTED)
    # both decoder traces
    px, py, pw, ph_ = gap, y0 + S + 172, W - 2 * gap, 90
    d.rectangle([px, py, px + pw, py + ph_], outline=(70, 70, 76))
    d.line([px, py + ph_ // 2, px + pw, py + ph_ // 2], fill=(55, 55, 60))
    m = max(len(p_im) - 1, 1)
    for trace, col in ((p_real, (120, 150, 230)), (p_im, BAD)):
        pts = [(px + int(pw * k / m), py + int(ph_ * (1 - v))) for k, v in enumerate(trace)]
        d.line(pts, fill=col, width=2)
    cx = px + int(pw * i / m)
    d.line([cx, py, cx, py + ph_], fill=(200, 200, 200))
    d.text((px, py + ph_ + 4), "P(ball if we lift now):  blue = real (posterior)   red = imagined", font=font(12), fill=MUTED)
    return np.asarray(img)


def sheet(frames, p, labels, h, every=False):
    S = 96
    row = Image.new("RGB", (S * len(frames) + 160, S + 40), SURF)
    d = ImageDraw.Draw(row)
    d.text((6, S // 2), h if isinstance(h, str) else f"hold {h}\nthen lift", font=font(14), fill=INK)
    for i, f in enumerate(frames):
        x = 160 + i * S
        row.paste(Image.fromarray(f).resize((S - 4, S - 4), Image.Resampling.NEAREST), (x, 2))
        tag = "s" if i == 0 and labels[i].startswith("start") else ("L" if labels[i].startswith("LIFT") else "h")
        tag += labels[i].split(" ")[-1] if labels[i].split(" ")[-1].startswith("c") else ""
        d.text((x + 2, S + 2), f"{tag} {p[i]:.2f}",
               font=font(12), fill=GOOD if p[i] <= 0.5 else BAL)
    return np.asarray(row)


BAL = BALL

if __name__ == "__main__":
    main()
