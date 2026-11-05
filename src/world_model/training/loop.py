"""Generic training loop: step, log, periodic callbacks, checkpoint."""
import time


def cycle(loader):
    """Endless batches, reshuffled every pass."""
    while True:
        yield from loader


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def train_loop(step_fn, loader, device, logger, checkpoints, state_fn, steps, log_every, ckpt_every, start_step=0,
               callbacks=(), prefix="train/"):
    """Call step_fn(batch) -> metrics until `steps` updates are done.

    callbacks: (every, fn(step)) pairs, e.g. holdout evaluation or rollout videos.
    """
    if start_step >= steps:
        print(f"already at step {start_step} >= {steps}, nothing to do")
        return
    batches = cycle(loader)
    last_step, last_time = start_step, time.time()
    for step in range(start_step + 1, steps + 1):
        metrics = step_fn(to_device(next(batches), device))
        if step % log_every == 0:
            # read metrics right away: compiled steps may reuse their output buffers on the next call
            now = time.time()
            metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
            metrics["perf/updates_per_sec"] = (step - last_step) / (now - last_time)
            logger.scalars(step, metrics)
            last_step, last_time = step, now
        for every, fn in callbacks:
            if step % every == 0:
                fn(step)
        if step % ckpt_every == 0 or step == steps:
            checkpoints.save("latest", state_fn(step))
