"""OGBar-SMT stage 1 (docs/OGBAR_SMT.md): teacher + future-embedding decoder + stochastic Q and P, one step per anchor.

    pixi run python scripts/verify_smt_ogbar.py                      # run first: contract, causality, memory
    pixi run python scripts/train_smt_ogbar.py
    pixi run python scripts/train_smt_ogbar.py model.z_dist=deterministic tag=det     # phase-1 control
    pixi run python scripts/train_smt_ogbar.py trainer.resume=<logdir>/checkpoints/wm_latest.pt

Batches of 128 random anchors from any episode; each anchor carries its whole episode at its offset
(~39 MB of fp16 patches), read by 6 loader workers into shared memory (no pinned copy: pinning 5 GB per
batch costs ~7 s). Writes <logdir>/checkpoints/wm_latest.pt, wm_step<N>.pt every keep_every steps.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hydra  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from info_gathering.ogbar_smt_data import AnchorBatches, OgbarSmtAnchors  # noqa: E402
from world_model.models.smt_ogbar import SmtOgbar  # noqa: E402
from world_model.utils.checkpoint import CheckpointManager  # noqa: E402
from world_model.utils.logger import Logger  # noqa: E402
from world_model.utils.seeding import seed_everything  # noqa: E402


@hydra.main(version_base=None, config_path="../configs/smt_ogbar", config_name="train_ogbar_v3_400")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    logdir = Path(cfg.logdir)
    checkpoints = CheckpointManager(logdir / "checkpoints", "wm")
    state = CheckpointManager.load(cfg.trainer.resume) if cfg.trainer.resume else None

    episodes = cfg.data.get("episodes")
    if episodes is not None and not OmegaConf.is_list(episodes) and not isinstance(episodes, (list, tuple)):
        episodes = [int(e) for e in str(episodes).split(",")]
    data = OgbarSmtAnchors(ROOT / cfg.data.cache_dir, ROOT / cfg.data.zarr_path, cfg.data.frameskip,
                           episodes=None if episodes is None else [int(e) for e in episodes],
                           privileged=bool(cfg.data.get("privileged", False)))      # c_view for the align loss only
    device = torch.device(cfg.device)
    import numpy as np
    a_mu, a_sd = data.action_raw.mean(0), np.maximum(data.action_raw.std(0), 1e-6)
    action_stats = {"mean": np.tile(a_mu, data.k).tolist(), "std": np.tile(a_sd, data.k).tolist()}
    model = SmtOgbar(action_dim=data.action_dim, action_stats=action_stats, **OmegaConf.to_container(cfg.model)).to(device)
    o = cfg.optim
    opt = torch.optim.AdamW(model.parameters(), lr=o.lr, betas=tuple(o.betas), weight_decay=o.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, o.warmup_steps)))
    step = 0
    if state:
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        step = int(state["step"])
        print(f"resumed from {cfg.trainer.resume} at step {step}")
    print(f"{len(data)} anchors from {len(data.ends)} episodes; {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M "
          f"parameters", flush=True)

    config = OmegaConf.to_container(cfg, resolve=True)
    logger = Logger(logdir, config, cfg.wandb, job_type="world_model", name=f"smt_ogbar/{logdir.name}",
                    group=cfg.wandb.group, run_id=state["wandb_run_id"] if state else None, tensorboard=cfg.tensorboard)

    def save(name):
        return checkpoints.save(name, dict(model=model.state_dict(), optimizer=opt.state_dict(),
                                           scheduler=sched.state_dict(), step=step, config=config,
                                           wandb_run_id=logger.run_id))

    def batches():
        g = torch.Generator().manual_seed(cfg.seed + step)
        while True:
            sampler = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(data, generator=g),
                                                    cfg.loader.batch_size, drop_last=True)
            yield from torch.utils.data.DataLoader(AnchorBatches(data), batch_size=None, sampler=sampler,
                                                   num_workers=cfg.loader.num_workers,
                                                   prefetch_factor=cfg.loader.prefetch_factor, pin_memory=False)

    model.train()
    sums, counts, count, t0, t_data = {}, {}, 0, time.time(), 0.0
    tw = time.time()
    for batch in batches():
        batch = {k: v.to(device) for k, v in batch.items()}
        batch["patch"] = batch["patch"].to(torch.bfloat16)
        t_data += time.time() - tw
        with torch.autocast("cuda", dtype=torch.bfloat16):
            total, losses, metrics = model(batch, diagnostics=(step + 1) % cfg.trainer.log_every == 0)
        opt.zero_grad(set_to_none=True)
        total.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), o.clip_grad_norm)
        opt.step()
        sched.step()
        step += 1
        logs = {**{f"loss/{k}": v for k, v in losses.items()}, **{f"metric/{k}": v for k, v in metrics.items()},
                "opt/loss": total, "opt/grad_norm": gn}
        for k, v in logs.items():                      # memory_gain only exists on diagnostic steps
            sums[k] = sums.get(k, 0.0) + float(v.detach() if torch.is_tensor(v) else v)
            counts[k] = counts.get(k, 0) + 1
        count += 1
        if step % cfg.trainer.log_every == 0:
            vals = {k: v / counts[k] for k, v in sums.items()}
            dt = time.time() - t0
            vals.update({"perf/s_per_step": dt / count, "perf/data_wait_frac": t_data / dt, "opt/lr": sched.get_last_lr()[0],
                         "perf/gpu_peak_gb": torch.cuda.max_memory_allocated(device) / 1e9})
            logger.scalars(step, vals)
            sums, counts, count, t0, t_data = {}, {}, 0, time.time(), 0.0
        if step % cfg.trainer.ckpt_every == 0:
            save("latest")
        if step % cfg.trainer.keep_every == 0:
            save(f"step{step}")
        if step >= cfg.trainer.steps:
            break
        tw = time.time()
    save("latest")
    print(f"done at step {step}", flush=True)


if __name__ == "__main__":
    main()
