"""Open-loop rollouts: observe real frames, imagine the continuation from actions, decode."""
import torch


def action_onsets(action):
    """Steps where the action switches to a new non-zero value, shared by all clips.

    action: (B, T, A). An onset is a non-zero action differing from the previous step's, so a
    change from one non-zero action straight to another (memtest4's cover -> lift) counts as
    well as a change out of zero. A clip already acting at step 0 has an onset at 0.
    """
    prev = torch.cat([torch.zeros_like(action[:, :1]), action[:, :-1]], dim=1)
    starts = (action.abs().sum(-1) > 0) & (action != prev).any(-1)
    if not bool((starts == starts[:1]).all()):
        raise ValueError("clips in one rollout batch must share their action onsets")
    return [int(t) for t in torch.nonzero(starts[0]).flatten()]


@torch.no_grad()
def open_loop_rollout(model, decoder, batch, context, horizon=None):
    """Observe the first `context` steps of whole-episode clips, then imagine up to `horizon` more.

    To test an action (e.g. a lift starting at onset step o), use context = o + 1: the model sees
    every real frame up to the step the action is taken, and imagines its consequences.

    Returns
        panel    (B, context + H, 3H, W, 3) float in [0, 1]: truth / model / error stacked vertically,
                 model = reconstruction for t < context, imagination after
        metrics  pixel MSE over the observed and the imagined steps
    """
    T = batch["image"].shape[1]
    if not 0 < context < T:
        raise ValueError(f"context {context} must be in (0, {T})")
    end = T if horizon is None else min(T, context + int(horizon))
    post = model.observe({k: batch[k][:, :context] for k in ("image", "prev_action", "is_first")})
    # prev_action[:, context] is the action taken at step context - 1, which produces the first imagined frame
    imag = model.imagine(post, batch["prev_action"][:, context:end])
    pred = torch.cat([decoder.render(post), decoder.render(imag)], dim=1)
    truth = batch["image"][:, :end].float() / 255.0
    # (T,)
    mse = (pred - truth).pow(2).mean(dim=(0, 2, 3, 4))
    metrics = {"mse/observed": mse[:context].mean(), "mse/imagined": mse[context:].mean()}
    error = (pred - truth + 1.0) / 2.0
    return torch.cat([truth, pred, error], dim=2), metrics
