"""docs/TWIN_FINETUNE.md: finetune ONLY the updaters, on unlabelled episodes, against a frozen dictionary.

    frozen   frame encoder, teacher, reward head, readout, pixel decoder   (the "dictionary")
    trained  the online posterior + prior updaters
    twin     an EMA copy of the online updaters; never receives gradients

One step, for a batch of finetune episodes, is the eval's own computation made differentiable:

    z_0  = teacher(window ending at frame 0)                    frozen, eval_warmup 1
    z_t  = posterior(z_{t-1}, e_t, a_{t-1})       t = 1..onset   onset = first show frame
    y_k  = prior(y_{k-1}, a_{onset+k-1}), y_0 = z_onset          k = 1..9, no frames
    x_k  = decoder(readout(y_k))                                 the imagined frame onset+k

    L_reveal = BCE( p_ball(x_k), the eval detector's verdict on the REAL frame onset+k ),  k >= 2
    L_recon  = mean_t || readout(z_t) - e_t ||^2,   t = 0..onset           stay readable
    L_twin   = mean || state - stopgrad(twin's state) ||^2, every state    stay honest

The reveal target is computed from pixels. No hidden-state label (ball_slot, ball_under_target)
enters training; the label is read once, at load, only to assert the pixel verdict agrees with it.

At the end the online updaters (and, separately, the twin) are written into a copy of the
readout-run checkpoint, with a pixel-decoder checkpoint pointing at it, so eval-smt-phase1 and
onset-probe-phase1 run on them unchanged:
    <logdir>/export/pixel_decoder        decoder_run for eval-smt-phase1
    <logdir>/export                      wm_run for onset-probe-phase1

    pixi run finetune-twin-phase1 decoder_run=<C0 pixel decoder run> name=f3
"""
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import zarr  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from torch.amp import autocast  # noqa: E402

from info_gathering.ball_detector import ball_pixels, yellowness  # noqa: E402
from world_model.models.smt_prelim.embed_cache import read_embed_attrs  # noqa: E402
from world_model.utils.checkpoint import CheckpointManager, load_frozen_models  # noqa: E402
from world_model.utils.seeding import seed_everything  # noqa: E402

SHOW_PHASE = 2
LIFT_ACTION = (0.0, 1.0)


def soft_ball_logit(frames, cfg):
    """frames (..., H, W, 3) in [0, 1] -> (logit of p(ball), the statistic), both (...).

    The eval detector calls a frame "ball" when at least min_pixels (3) pixels have yellowness
    above threshold (0.15) -- which holds exactly when the 3rd-largest yellowness exceeds 0.15.
    That order statistic is differentiable (its gradient lands on the pixel that holds it, so a
    frame with no yellow at all still gets pushed), and it makes the loss's call identical to the
    eval's by construction:  logit > 0  <=>  the hard verdict.

    (A summed per-pixel sigmoid count was tried first and fails: ~4000 background pixels each
    leak sigmoid(-3) ~ 0.05, so empty frames scored ~30 "pixels" and agreed with the hard
    detector on 12-37% of frames.)
    """
    kth = torch.topk(yellowness(frames).flatten(-2), int(cfg.k), dim=-1).values[..., -1]
    return (kth - float(cfg.threshold)) / float(cfg.scale), kth


class FinetuneEpisodes:
    """The finetune pool, on the device: what observe()/imagine() would be fed, plus pixel targets."""

    def __init__(self, zarr_path, model, detector, horizon, device):
        z = zarr.open_group(str(ROOT / zarr_path), mode="r")
        # the cached RAW encoder output, normalised with THIS model's buffers: exactly what
        # model._embed(image) hands the posterior at eval (checked in check_against_eval)
        read_embed_attrs(ROOT / zarr_path, model.embed_key, model.encoder_checkpoint)
        ends = z["episode_ends"][:]
        starts = np.concatenate([[0], ends[:-1]])
        lengths = ends - starts
        raw, action, phase, rgb = z[model.embed_key][:], z["action"][:], z["phase"][:], z["rgb"]
        label_end = z["ball_under_target"][:][ends - 1].astype(bool)     # sanity check ONLY
        n, steps, horizon = len(ends), int(lengths.max()), int(horizon)
        embed = np.zeros((n, steps, raw.shape[1]), dtype=np.float32)
        prev = np.zeros((n, steps, action.shape[1]), dtype=np.float32)
        onset = np.zeros(n, dtype=np.int64)
        target = np.zeros((n, horizon), dtype=bool)
        for e, (s, t) in enumerate(zip(starts, ends)):
            m = int(t - s)
            embed[e, :m] = raw[s:t]
            prev[e, 1:m] = action[s:t - 1]        # prev_action: zeros at t=0, else a_{t-1}
            on = int(np.flatnonzero(phase[s:t] == SHOW_PHASE)[0])
            if on + horizon != m - 1:
                raise AssertionError(f"episode {e}: onset {on} + horizon {horizon} != last frame {m - 1}")
            if not (prev[e, on + 1:on + 1 + horizon] == np.array(LIFT_ACTION)).all():
                raise AssertionError(f"episode {e}: the imagined steps are not all fed the lift action")
            frames = torch.from_numpy(rgb[s + on + 1:s + on + 1 + horizon].astype(np.float32) / 255.0)
            target[e] = (ball_pixels(frames, float(detector.threshold)) >= int(detector.min_pixels)).numpy()
            onset[e] = on
        # the pixels must tell the same story as the generator: nothing at show offset 1, the
        # episode's answer from offset 2 on. Used to CHECK the data, never as a training target.
        self.offset1_visible = int(target[:, 0].sum())
        agree = (target[:, 1:] == label_end[:, None]).all(1)
        if not agree.all():
            raise AssertionError(f"pixel reveal disagrees with the label on {int((~agree).sum())} episodes")
        self.num_episodes, self.horizon = n, horizon
        self.lengths = torch.from_numpy(lengths.astype(np.int64)).to(device)
        self.onset = torch.from_numpy(onset).to(device)
        mean, std = model.embed_mean.to(device), model.embed_std.to(device)
        self.embed = (torch.from_numpy(embed).to(device) - mean) / std
        self.prev = torch.from_numpy(prev).to(device)
        self.target = torch.from_numpy(target).to(device)
        self.positive_rate = float(target[:, -1].mean())
        self._z, self._starts, self._ends = z, starts, ends

    def images(self, episodes):
        """Raw frames of whole episodes, (B, T, H, W, 3) uint8 -- only for the eval-path check."""
        clips = [self._z["rgb"][int(self._starts[e]):int(self._ends[e])] for e in episodes]
        return torch.from_numpy(np.stack(clips))


def rollout(model, posterior, prior, data, idx, horizon, amp_dtype):
    """observe() to each episode's onset, then imagine() the lift. Differentiable through both updaters.

    Returns Z (B, T+1, K, d) observed states t=0..max onset, mask (B, T+1) (t <= own onset),
    Y (B, horizon, K, d) imagined states for frames onset+1..onset+horizon.
    """
    device_type = data.embed.device.type
    on = data.onset[idx]
    last = int(on.max())
    E, PA = data.embed[idx, :last + 1], data.prev[idx, :last + 1]
    is_first = torch.ones(len(idx), 1, dtype=torch.bool, device=E.device)
    z = model.encode_windows(E[:, :1], PA[:, :1], is_first)[:, -1]       # frozen teacher, no grad
    states = [z]
    with autocast(device_type, dtype=amp_dtype):
        for t in range(1, last + 1):
            z = posterior.step(z, E[:, t], PA[:, t]).float()               # exactly observe()'s step
            states.append(z)
    Z = torch.stack(states, 1)
    rows = torch.arange(len(idx), device=E.device)
    y = Z[rows, on]
    imagined = []
    with autocast(device_type, dtype=amp_dtype):
        for k in range(horizon):
            y = prior.step(y, data.prev[idx, on + 1 + k]).float()          # exactly imagine()'s step
            imagined.append(y)
    mask = torch.arange(last + 1, device=E.device)[None] <= on[:, None]
    return Z, mask, torch.stack(imagined, 1)


def render(model, decoder, states, amp_dtype):
    """states (..., K, d) -> (embedding, frames in [0, 1]). Readout as imagine() runs it (bf16);
    the decoder in fp32 so its backward cannot underflow (the eval renders it in fp16)."""
    with autocast(states.device.type, dtype=amp_dtype):
        emb = model.readout(states).float()
    frames = decoder._dist({"embed": emb}).mode()
    return emb, frames.clamp(0.0, 1.0)


@torch.no_grad()
def check_against_eval(model, decoder, data, cfg, amp_dtype):
    """The training rollout must reproduce the eval's own path (model.observe / model.imagine /
    decoder.render) on real finetune episodes. Returns the measured gaps; raises if they are not tiny."""
    lengths = data.lengths.cpu().numpy()
    common = np.bincount(lengths).argmax()
    eps = np.flatnonzero(lengths == common)[:8]
    image = data.images(eps).to(data.embed.device)
    T = image.shape[1]
    batch = {"image": image, "prev_action": data.prev[eps, :T],
             "is_first": torch.zeros(len(eps), T, dtype=torch.bool, device=image.device)}
    batch["is_first"][:, 0] = True
    post = model.observe(batch)
    on = int(data.onset[eps[0]])
    imag = model.imagine(post[:, :on + 1], batch["prev_action"][:, on + 1:on + 1 + data.horizon])
    ref_frames = decoder.render(imag)
    idx = torch.as_tensor(eps, device=data.embed.device)
    Z, mask, Y = rollout(model, model.posterior, model.prior, data, idx, data.horizon, amp_dtype)
    _, frames = render(model, decoder, Y, amp_dtype)
    gap = {"embed": float((post["embed"][:, :on + 1] - data.embed[idx, :on + 1]).abs().max()),
           "z_onset": float((post["z"][:, on] - Z[:, on]).abs().max()),
           "z_imagined": float((imag["z"] - Y).abs().max()),
           "frames": float((ref_frames - frames).abs().max())}
    hard = lambda f: (ball_pixels(f, float(cfg.detector.threshold)) >= int(cfg.detector.min_pixels)).any(-1)
    gap["verdict_agree"] = float((hard(ref_frames) == hard(frames)).float().mean())
    if gap["embed"] > 1e-4 or gap["z_onset"] > 1e-3 or gap["z_imagined"] > 1e-3:
        raise AssertionError(f"training rollout does not reproduce the eval path: {gap}")
    if gap["verdict_agree"] < 1.0:
        # the eval renders in fp16 and training in fp32, so a frame sitting on the 3-pixel line can
        # flip; the STATES above are what must match exactly
        print(f"warning: fp32 and fp16 renders disagree on {1 - gap['verdict_agree']:.0%} of the check episodes")
    return gap


def export(dec_state, wm_state, posterior, prior, out_dir, note):
    """A world-model checkpoint with these updaters + a decoder checkpoint pointing at it."""
    wm = copy.deepcopy(wm_state)
    modules = wm["model"]["modules"]
    for prefix, module in (("posterior.", posterior), ("prior.", prior)):
        new = {prefix + k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
        old = {k for k in modules if k.startswith(prefix)}
        if old != set(new):
            raise AssertionError(f"{prefix} keys differ between the checkpoint and the finetuned module")
        for k, v in new.items():
            if modules[k].shape != v.shape:
                raise AssertionError(f"{k}: shape {tuple(v.shape)} != {tuple(modules[k].shape)}")
            modules[k] = v
    wm["twin_ft"] = note
    wm_path = CheckpointManager(Path(out_dir) / "checkpoints", "wm").save("latest", wm)
    dec = dict(dec_state)
    dec["wm_checkpoint"] = str(wm_path.resolve())
    dec["twin_ft"] = note
    CheckpointManager(Path(out_dir) / "pixel_decoder" / "checkpoints", "decoder").save("latest", dec)
    return wm_path


@hydra.main(version_base=None, config_path="../configs/smt_phase1", config_name="finetune_twin_phase1")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    device = torch.device(cfg.device)
    logdir = Path(cfg.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    amp_dtype = getattr(torch, str(cfg.amp_dtype))
    L = cfg.loss

    model, decoder, _, _, dec_state, wm_state = load_frozen_models(cfg.decoder_checkpoint, cfg.device)
    model.requires_grad_(False).eval()
    decoder.requires_grad_(False).eval()
    data = FinetuneEpisodes(cfg.ft_zarr, model, cfg.detector, cfg.horizon, device)
    print(f"finetune pool {cfg.ft_zarr}: {data.num_episodes} episodes, reveal positive rate "
          f"{data.positive_rate:.3f}, ball visible at show offset 1 in {data.offset1_visible} episodes")
    gap = check_against_eval(model, decoder, data, cfg, amp_dtype)
    print(f"ok  training rollout == eval path: {gap}")

    online = {"posterior": model.posterior, "prior": model.prior}
    twin = {k: copy.deepcopy(m).requires_grad_(False).eval() for k, m in online.items()}
    for m in online.values():
        m.requires_grad_(True)
    params = [p for m in online.values() for p in m.parameters()]
    optimizer = torch.optim.AdamW(params, lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    checkpoints = CheckpointManager(logdir / "checkpoints", "ft")
    start = 0
    if cfg.resume == "auto" and checkpoints.latest() is not None:
        state = CheckpointManager.load(checkpoints.latest())
        for k in online:
            online[k].load_state_dict(state["online"][k])
            twin[k].load_state_dict(state["twin"][k])
        optimizer.load_state_dict(state["optimizer"])
        start = int(state["step"])
        print(f"resumed from {checkpoints.latest()} at step {start}")

    def state_fn(step):
        return {"step": step, "config": OmegaConf.to_container(cfg, resolve=True),
                "online": {k: m.state_dict() for k, m in online.items()},
                "twin": {k: m.state_dict() for k, m in twin.items()},
                "optimizer": optimizer.state_dict(), "decoder_checkpoint": str(Path(cfg.decoder_checkpoint).resolve())}

    k0 = int(L.reveal_from_offset) - 1          # imagined index k-1 holds show offset k
    hard = lambda f: ball_pixels(f, float(cfg.detector.threshold)) >= int(cfg.detector.min_pixels)
    need_twin = float(L.w_twin) > 0
    metrics_file = (logdir / "metrics.jsonl").open("a")
    print(f"finetune {cfg.name}: steps {cfg.steps} batch {cfg.batch_size} lr {cfg.lr} ema {cfg.ema} | "
          f"w_reveal {L.w_reveal} w_recon {L.w_recon} w_twin {L.w_twin} | logdir {logdir}", flush=True)
    t0, sums, count = time.time(), {}, 0
    for step in range(start + 1, int(cfg.steps) + 1):
        rng = np.random.default_rng(int(cfg.seed) * 100003 + step)     # reproducible across resumes
        idx = torch.as_tensor(rng.choice(data.num_episodes, int(cfg.batch_size), replace=False), device=device)
        Z, mask, Y = rollout(model, online["posterior"], online["prior"], data, idx, data.horizon, amp_dtype)
        _, frames = render(model, decoder, Y, amp_dtype)
        logit, kth = soft_ball_logit(frames, cfg.soft_ball)
        target = data.target[idx]
        l_reveal = F.binary_cross_entropy_with_logits(logit[:, k0:], target[:, k0:].float())
        with autocast(device.type, dtype=amp_dtype):
            readout_obs = model.readout(Z).float()
        err = (readout_obs - data.embed[idx, :Z.shape[1]]).pow(2).mean(-1)
        l_recon = (err * mask).sum() / mask.sum()
        log_now = step % int(cfg.log_every) == 0 or step == start + 1
        if need_twin or log_now:
            with torch.no_grad():
                Zt, _, Yt = rollout(model, twin["posterior"], twin["prior"], data, idx, data.horizon, amp_dtype)
            d_obs = (Z - Zt).pow(2).mean((-1, -2))
            d_im = (Y - Yt).pow(2).mean((-1, -2))
            l_twin = ((d_obs * mask).sum() + d_im.sum()) / (mask.sum() + d_im.numel())
        else:
            l_twin = torch.zeros((), device=device)
        total = float(L.w_reveal) * l_reveal + float(L.w_recon) * l_recon + float(L.w_twin) * l_twin
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        if step == start + 1:
            # the gradient contract: only the online updaters may move
            moved = [n for n, p in model.named_parameters() if p.grad is not None
                     and not n.startswith(("posterior.", "prior."))]
            moved += [f"decoder.{n}" for n, p in decoder.named_parameters() if p.grad is not None]
            moved += [f"twin.{k}.{n}" for k, m in twin.items() for n, p in m.named_parameters() if p.grad is not None]
            if moved:
                raise AssertionError(f"gradients reached frozen parameters: {moved[:5]}")
            if not all(p.grad is not None for p in params if p.requires_grad):
                missing = [n for k, m in online.items() for n, p in m.named_parameters() if p.grad is None]
                print(f"note: {len(missing)} updater tensors got no gradient this step: {missing[:4]}")
            print("ok  gradient contract: only the online posterior/prior receive gradients", flush=True)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, float(cfg.clip_grad_norm))
        optimizer.step()
        with torch.no_grad():
            for k in online:
                for p_t, p_o in zip(twin[k].parameters(), online[k].parameters()):
                    p_t.lerp_(p_o, 1.0 - float(cfg.ema))
                for b_t, b_o in zip(twin[k].buffers(), online[k].buffers()):
                    b_t.copy_(b_o)

        with torch.no_grad():
            hard_frames = hard(frames)
            pred = hard_frames.any(-1)                          # the eval's call: any imagined frame
            label = target[:, -1]
            row = {"loss/total": total.item(), "loss/reveal": l_reveal.item(), "loss/recon": l_recon.item(),
                   "loss/twin": l_twin.item(), "opt/grad_norm": float(grad_norm),
                   "batch/acc": float((pred == label).float().mean()),
                   "batch/soft_hard_agree": float(((logit > 0) == hard_frames).float().mean()),
                   "batch/kth_yellow_pos": float(kth[:, k0:][target[:, k0:]].mean()) if target[:, k0:].any() else float("nan"),
                   "batch/kth_yellow_neg": float(kth[:, k0:][~target[:, k0:]].mean()) if (~target[:, k0:]).any() else float("nan")}
        for k, v in row.items():
            if np.isfinite(v):
                sums[k] = sums.get(k, 0.0) + v
        count += 1
        if log_now:
            avg = {k: v / count for k, v in sums.items()}
            avg["twin/dist"] = l_twin.item()
            avg["step"] = step
            avg["sec_per_step"] = (time.time() - t0) / count
            metrics_file.write(json.dumps(avg) + "\n")
            metrics_file.flush()
            print(f"[{step}] " + " ".join(f"{k}={v:.4g}" for k, v in avg.items() if k != "step"), flush=True)
            t0, sums, count = time.time(), {}, 0
        if step % int(cfg.ckpt_every) == 0 or step == int(cfg.steps):
            checkpoints.save("latest", state_fn(step))

    note = {"name": str(cfg.name), "steps": int(cfg.steps), "ft_zarr": str(cfg.ft_zarr),
            "source_decoder_checkpoint": str(Path(cfg.decoder_checkpoint).resolve()),
            "config": OmegaConf.to_container(cfg, resolve=True)}
    path = export(dec_state, wm_state, online["posterior"], online["prior"], logdir / "export",
                  {**note, "updaters": "online"})
    print(f"exported online updaters -> {path}")
    if bool(cfg.export_twin):
        path = export(dec_state, wm_state, twin["posterior"], twin["prior"], logdir / "export_twin",
                      {**note, "updaters": "twin"})
        print(f"exported twin updaters -> {path}")


if __name__ == "__main__":
    main()
