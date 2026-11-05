"""The SMT objective, adapted to an RSSM-shaped memory.

Mirrors forward_smt (smt/src/model.py:294-330): encode the window at t and the window slid one
step forward, predict the second from the first with the one-step cell, and separately decode
the future from the state. The only structural change is that the memory has a categorical half,
so the dynamics term is MSE on `deter` plus KL on the stoch logits instead of a single MSE.

Gradient paths into the teacher encoder, all three required (nothing here detaches them):
  1. target side   dyn/kl_rep -> m_{t+1} -> encoder      "be a next-state the cell can hit"
  2. input side    dyn/kl_dyn -> cell -> m_t -> encoder  "be a state the cell can step from"
  3. decoder       dec -> from_state(m_t) -> encoder     "carry what predicts the future"
The only detaches are the two inside rssm.kl_loss, which are the KL-balancing split.
"""
import torch

from .blocks import masked_mean, state_feat, uniformity_loss


def split_batch(batch):
    """The anchor item -> the tensors forward_smt needs. See dataset_smt_prelim.SmtAnchors."""
    e_win, a_win, win_mask = batch["e_win"], batch["a_win"], batch["win_mask"]
    e_seq, a_seq, seq_mask = batch["e_seq"], batch["a_seq"], batch["seq_mask"]
    return dict(
        e_ctx=e_win[:, :-1], a_ctx=a_win[:, :-1], ctx_mask=win_mask[:, :-1],
        e_ctx_next=e_win[:, 1:], a_ctx_next=a_win[:, 1:], ctx_mask_next=win_mask[:, 1:],
        e_next=e_win[:, -1], a_next=a_win[:, -1],            # e_{t+1} and a_t
        e_fut_in=e_seq[:, :-1], a_fut_in=a_seq[:, :-1],      # e_{t..t+F-1}, a_{t..t+F-1}
        y_fut=e_seq[:, 1:], fut_mask=seq_mask[:, 1:],        # e_{t+1..t+F}
    )


def smt_losses(encoder, decoder, predictor, batch, cfg, step=0):
    """Returns (losses, metrics, state). Losses are unscaled; the caller applies coefficients."""
    parts = split_batch(batch)
    rssm = predictor.rssm

    deter_t, logit_t, z_t = encoder(parts["e_ctx"], parts["a_ctx"], parts["ctx_mask"])
    deter_t1, logit_t1, _ = encoder(parts["e_ctx_next"], parts["a_ctx_next"], parts["ctx_mask_next"])

    stoch_t = rssm.get_dist(logit_t).rsample()               # straight-through; carries gradient to logit_t
    _, deter_p, logit_p = predictor.step(stoch_t, deter_t, parts["a_next"], parts["e_next"])

    y_hat = decoder(deter_t, logit_t, parts["e_fut_in"], parts["a_fut_in"])

    # --- predictive state: decode the future from the memory -------------------------------
    fut_mask = parts["fut_mask"].float()
    per_step = (y_hat - parts["y_fut"]).pow(2).mean(-1)       # (B, F)
    gamma = float(cfg.get("gamma", 1.0))
    decay = gamma ** torch.arange(per_step.shape[1], device=per_step.device)

    # Weight each prediction by what the MEMORY buys, not by how large the error is.
    # Measured on this data: cup kinematics are 96% of the (anchor, position) pairs and 99% of
    # the squared error, yet they are fully determined by (frame, action) -- the memory is only
    # needed for the ~1% of pairs where the ball appears. Unweighted MSE therefore spends its
    # gradient on motion that needs no memory at all.
    #
    # The weight is the same decoder's error with z zeroed: high exactly where the memoryless
    # path cannot answer. Detached, and normalised so the loss keeps its scale. It uses no
    # labels -- weighting by the ball detector would hand the model the eval target.
    weight = torch.ones_like(per_step)
    mode = str(cfg.get("reweight", "none"))
    if mode == "nomem" and int(step) >= int(cfg.get("reweight_warmup", 0)):
        with torch.no_grad():
            nomem = decoder.decode_without_memory(deter_t, logit_t, parts["e_fut_in"], parts["a_fut_in"])
            w = (nomem - parts["y_fut"]).pow(2).mean(-1)
            w = w.clamp(max=float(cfg.get("reweight_clip", 1e9))) + float(cfg.get("reweight_floor", 0.0))
            weight = w / masked_mean(w, fut_mask).clamp(min=1e-8)
    elif mode not in ("none", "nomem"):
        raise ValueError(f"unknown reweight mode {mode!r}")
    loss_dec = masked_mean(per_step * decay * weight, fut_mask)

    # --- one-step dynamics ------------------------------------------------------------------
    loss_dyn_deter = (deter_p - deter_t1).pow(2).mean()
    kl_free = float(cfg.get("kl_free", 0.0))
    # post = the teacher's next state, prior = the predictor's guess. kl_loss detaches the post
    # for `dyn` (trains the predictor) and the prior for `rep` (trains the teacher).
    kl_dyn, kl_rep = rssm.kl_loss(logit_t1, logit_p, kl_free)

    losses = {
        "dec": loss_dec,
        "dyn_deter": loss_dyn_deter,
        "kl_dyn": kl_dyn.mean(),
        "kl_rep": kl_rep.mean(),
        "unif": uniformity_loss(z_t, sample=int(cfg.get("unif_sample", 512))),
    }

    with torch.no_grad():
        # SMT's loss_decoder0 (model.py:315): the one-step read-out, the number that says what
        # the predictor's state is worth
        dec0 = masked_mean(per_step[:, 0], fut_mask[:, 0])
        # is the memory read at all? same forward with the state zeroed -- logging only
        y_nozed = decoder.decode_without_memory(deter_t, logit_t, parts["e_fut_in"], parts["a_fut_in"])
        dec0_nozed = masked_mean((y_nozed - parts["y_fut"]).pow(2).mean(-1)[:, 0], fut_mask[:, 0])
        # what copying the input would score, the baseline the data makes easy
        copy = (parts["e_fut_in"] - parts["y_fut"]).pow(2).mean(-1)
        moving = (copy[:, 0] > 1e-8).float()                  # the 39% of steps where a frame changes
        residual = (deter_p - deter_t1).pow(2).mean()
        metrics = {
            "dec0": dec0,
            "dec0_nozed": dec0_nozed,
            "dec0_nozed_gap": dec0_nozed - dec0,
            "dec0_copy": masked_mean(copy[:, 0], fut_mask[:, 0]),
            "dec0_moving": masked_mean(per_step[:, 0], fut_mask[:, 0] * moving),
            "dec0_copy_moving": masked_mean(copy[:, 0], fut_mask[:, 0] * moving),
            "deter_r2": 1.0 - residual / deter_t1.var().clamp(min=1e-8),
            "deter_cos": torch.cosine_similarity(deter_p, deter_t1, dim=-1).mean(),
            "deter_std": deter_t.std(0).mean(),
            "kl_raw": rssm.kl_loss(logit_t1, logit_p, 0.0)[0].mean(),
            "teacher_entropy": rssm.get_dist(logit_t).entropy().mean(),
            "state_norm": state_feat(deter_t, logit_t).norm(dim=-1).mean(),
            # is the reweighting actually moving mass onto the frames that change?
            "weight_moving": masked_mean(weight, fut_mask * (copy > 1e-8).float()),
            "weight_static": masked_mean(weight, fut_mask * (copy <= 1e-8).float()),
        }
        if "meta/phase" in batch:                             # labels are diagnostics only
            phase = batch["meta/phase"]
            for value, name in ((0, "reveal"), (1, "hold"), (2, "show")):
                sel = (phase == value).float()
                if float(sel.sum()) > 0:
                    metrics[f"dec0/{name}"] = masked_mean(per_step[:, 0], fut_mask[:, 0] * sel)

    return losses, metrics, (deter_t, logit_t, stoch_t)


def smt_token_losses(encoder, decoder, updater, batch, cfg, step=0):
    """SMT as written: memory is K tokens, the updater is a transformer, the match is MSE.

    No categorical, so no KL terms -- this is `forward_smt` (smt/src/model.py:294-330) with an
    action token added. Uses TeacherEncoder.tokens / TeacherDecoder.forward_tokens and
    SmtBeliefUpdater.step, i.e. the observation is a token the memory attends to rather than a
    sample from a 32x16 categorical.
    """
    parts = split_batch(batch)

    z_t = encoder.tokens(parts["e_ctx"], parts["a_ctx"], parts["ctx_mask"])
    z_t1 = encoder.tokens(parts["e_ctx_next"], parts["a_ctx_next"], parts["ctx_mask_next"])
    z_pred = updater.step(z_t, parts["e_next"], parts["a_next"])

    y_hat, ball_logit = decoder.predict_tokens(z_t, parts["e_fut_in"], parts["a_fut_in"])

    fut_mask = parts["fut_mask"].float()
    per_step = (y_hat - parts["y_fut"]).pow(2).mean(-1)
    gamma = float(cfg.get("gamma", 1.0))
    decay = gamma ** torch.arange(per_step.shape[1], device=per_step.device)

    weight = torch.ones_like(per_step)
    mode = str(cfg.get("reweight", "none"))
    if mode == "nomem" and int(step) >= int(cfg.get("reweight_warmup", 0)):
        with torch.no_grad():
            nomem = decoder.decode_tokens_without_memory(z_t, parts["e_fut_in"], parts["a_fut_in"])
            w = (nomem - parts["y_fut"]).pow(2).mean(-1)
            w = w.clamp(max=float(cfg.get("reweight_clip", 1e9))) + float(cfg.get("reweight_floor", 0.0))
            weight = w / masked_mean(w, fut_mask).clamp(min=1e-8)

    # --- auxiliary: will the ball be visible at this future step? --------------------------
    # Supervised ONLY during the final lift, once the cup has begun to rise; everywhere else this
    # is SMT's no-prediction mask (model.py:61-67). The latent objective spends 99% of its
    # gradient on cup kinematics that need no memory, so this asks the question directly.
    y_ball = batch["ball_seq"][:, 1:]                      # aligned with y_fut
    ball_mask = batch["ball_lift_seq"][:, 1:].float() * fut_mask
    ball_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        ball_logit.float(), y_ball.float(), reduction="none")
    loss_ball = masked_mean(ball_bce, ball_mask)

    losses = {
        "dec": masked_mean(per_step * decay * weight, fut_mask),
        "dyn": (z_pred - z_t1).pow(2).mean(),              # SMT's loss_dyn, model.py:323-324
        "unif": uniformity_loss(z_t, sample=int(cfg.get("unif_sample", 512))),
        "ball": loss_ball,
    }

    with torch.no_grad():
        dec0 = masked_mean(per_step[:, 0], fut_mask[:, 0])
        nozed = decoder.decode_tokens_without_memory(z_t, parts["e_fut_in"], parts["a_fut_in"])
        dec0_nozed = masked_mean((nozed - parts["y_fut"]).pow(2).mean(-1)[:, 0], fut_mask[:, 0])
        copy = (parts["e_fut_in"] - parts["y_fut"]).pow(2).mean(-1)
        moving = (copy[:, 0] > 1e-8).float()
        residual = (z_pred - z_t1).pow(2).mean()
        # does the updater actually respond to what it is shown? the RSSM cell scored 0.0 here
        shuffled = updater.step(z_t, parts["e_next"].flip(0), parts["a_next"])
        metrics = {
            "dec0": dec0,
            "dec0_nozed": dec0_nozed,
            "dec0_nozed_gap": dec0_nozed - dec0,
            "dec0_copy": masked_mean(copy[:, 0], fut_mask[:, 0]),
            "dec0_moving": masked_mean(per_step[:, 0], fut_mask[:, 0] * moving),
            "dec0_copy_moving": masked_mean(copy[:, 0], fut_mask[:, 0] * moving),
            "z_r2": 1.0 - residual / z_t1.var().clamp(min=1e-8),
            "z_std": z_t.std(0).mean(),
            "obs_sensitivity": (z_pred - shuffled).abs().mean(),
            "weight_moving": masked_mean(weight, fut_mask * (copy > 1e-8).float()),
            "weight_static": masked_mean(weight, fut_mask * (copy <= 1e-8).float()),
        }
        correct = ((ball_logit > 0).float() == y_ball).float()
        metrics["ball_acc"] = masked_mean(correct, ball_mask)
        metrics["ball_pos_rate"] = masked_mean(y_ball, ball_mask)
        metrics["ball_frac_supervised"] = ball_mask.sum() / ball_mask.numel()
        # THE number: accuracy at the FIRST supervised position of each anchor. Later positions
        # hand the decoder an input frame that already shows the ball, so they are copyable; the
        # first one is the only place the answer must come from memory.
        has = ball_mask.sum(1) > 0
        first = ball_mask.argmax(1)
        if bool(has.any()):
            rows = torch.nonzero(has).squeeze(-1)
            picked = first[rows]
            metrics["ball_acc_first"] = correct[rows, picked].mean()
            metrics["ball_n_first"] = has.float().sum()
        if "meta/phase" in batch:
            phase = batch["meta/phase"]
            for value, name in ((0, "reveal"), (1, "hold"), (2, "show")):
                sel = (phase == value).float()
                if float(sel.sum()) > 0:
                    metrics[f"dec0/{name}"] = masked_mean(per_step[:, 0], fut_mask[:, 0] * sel)

    return losses, metrics, z_t
