"""DMT: train the updaters on the states they produce themselves, not only on teacher states.

SMT's second stage (`forward_dmt`, submodules/smt/src/model.py:353). Stage 1 applies each
updater ONCE, from a teacher encoding, so it is only ever fit on the teacher's state
distribution. At eval it runs recurrently on its own output, which stage 1 never showed it.
DMT closes that gap the way DAgger does: roll the updater out, and correct every state it
visits against the teacher's encoding of the same step.

    z_enc[j] = teacher(window ending at t0+j)        oracle states, teacher FROZEN
    z_rnn[0] = z_enc[0]
    z_rnn[j] = posterior.step(z_rnn[j-1], e[t0+j], a[t0+j])      its OWN states

    L_dyn   = mean_j || z_rnn[j] - z_enc[j] ||^2                 pull the rollout back
    L_prior = mean_j || prior.step(z_rnn[j-1], a) - z_enc[j] ||^2
    L_slot  = CE( head(sg(z_rnn[j])), slot[t0+j] )               head adapts to rolled states

detach_rnn (SMT's default True) cuts the gradient between rollout steps, so each step is a
one-step correction rather than BPTT -- the point of SMT is to avoid backprop through time.
detach_readout (also True by default) keeps the read-out loss from reshaping the updater.

DELTA: posterior.step / prior.step ARE the gated step when arch.delta_mask is on (updater.py),
so every rollout here runs the same (1 - m) * z + m * z_hat equation stage 1 trained.

NOTE ON WHAT THIS CAN AND CANNOT FIX: DMT changes which states the updater is trained on, but
those states still come from the training episodes. Rolling out over 1-swap episodes visits
only 1-swap states, so DMT alone cannot supply the post-2-swap states a depth-1 dataset lacks.
"""
import torch
import torch.nn.functional as F

from ..smt_prelim.blocks import sliding_windows


@torch.no_grad()
def oracle_bank(model, data, device, dtype=torch.float16):
    """Teacher encodings for every (episode, step). The teacher is frozen in DMT, so these never
    change -- computing them once turns each training step into updater-only work."""
    from ..smt_prelim.blocks import window_mask
    steps = data.embed.shape[1]
    bank = torch.zeros(data.num_episodes, steps, model.arch["n_registers"], model.arch["d_model"],
                       dtype=dtype, device=device)
    for e in range(data.num_episodes):
        emb = torch.as_tensor(data.embed[e], device=device)[None]
        act = torch.as_tensor(data.prev_action[e], device=device)[None]
        real = torch.zeros(1, steps, dtype=torch.bool, device=device); real[0, 0] = True
        bank[e] = model.encode_windows(emb, act, real)[0].to(dtype)
    return bank


def dmt_losses(model, batch, cfg, bank=None):
    """Returns (losses, metrics). Teacher frozen; posterior, prior and the reward head train.

    bank: optional precomputed teacher states, indexed [episode, step] (see oracle_bank).
    """
    e_full, a_full = batch["e_full"], batch["a_full"]
    real, slot = batch["real_full"], batch["slot_full"].long()
    C, R = model.ctx_len, e_full.shape[1] - model.ctx_len
    batch_size = e_full.shape[0]

    # oracle states: the teacher's encoding of the window ending at each rolled step
    steps = R + 1
    if bank is not None:
        idx = batch["t"][:, None] + torch.arange(steps, device=e_full.device)[None]
        z_enc = bank[batch["episode"][:, None], idx].to(e_full.dtype)
    else:
        e_win = sliding_windows(e_full, C)[:, C - 1:]                  # (B, R+1, C, E)
        a_win = sliding_windows(a_full, C)[:, C - 1:]
        m_win = sliding_windows(real.float().unsqueeze(-1), C)[:, C - 1:].squeeze(-1) > 0.5
        with torch.no_grad():                                          # hot_teacher_encoder=False
            z_enc = model.encoder.tokens(e_win.reshape(-1, C, e_full.shape[-1]),
                                         a_win.reshape(-1, C, a_full.shape[-1]),
                                         m_win.reshape(-1, C))
        z_enc = z_enc.view(batch_size, steps, *z_enc.shape[1:])

    detach_rnn = bool(cfg.get("detach_rnn", True))
    detach_readout = bool(cfg.get("detach_readout", True))
    z = z_enc[:, 0]
    dyn, prior_err, rolled = [], [], []
    for j in range(1, steps):
        src = z.detach() if detach_rnn else z
        obs, act = e_full[:, C - 1 + j], a_full[:, C - 1 + j]
        z = model.posterior.step(src, obs, act)
        dyn.append((z - z_enc[:, j]).pow(2).mean())
        prior_err.append((model.prior.step(src.detach(), act) - z_enc[:, j]).pow(2).mean())
        rolled.append(z)

    rolled = torch.stack(rolled, dim=1)                                # (B, R, K, d)
    head_in = rolled.detach() if detach_readout else rolled
    flat = head_in.reshape(-1, *head_in.shape[2:])
    dummy_e = torch.zeros(flat.shape[0], 1, model.embed_dim, device=flat.device, dtype=e_full.dtype)
    logits = model.reward_decoder(flat, dummy_e, a_full[:, C:].reshape(-1, 1, a_full.shape[-1]))[:, 0]
    valid = real[:, C:].reshape(-1).float()
    # The read-out loss must match the arm's head. "slot" is 3-way CE; "binary" is one logit,
    # "is the ball under the cup that gets lifted", and needs BCE -- cross_entropy on a 1-logit
    # head raises "Expected floating point type for target with class probabilities, got Long".
    head = getattr(model.reward_decoder, "head", "slot")
    if head == "binary":
        tgt = batch["ball_full"][:, C:].reshape(-1).float()
        per = F.binary_cross_entropy_with_logits(logits.float().reshape(-1), tgt, reduction="none")
    elif head == "slot":
        tgt = slot[:, C:].reshape(-1)
        per = F.cross_entropy(logits.float(), tgt, reduction="none")
    else:
        raise ValueError(f"DMT read-out loss does not support reward_head={head!r}")
    target = tgt
    slot_ce = (per * valid).sum() / valid.clamp(min=1).sum()

    losses = {"dyn": torch.stack(dyn).mean(), "prior": torch.stack(prior_err).mean(), "slot": slot_ce}
    with torch.no_grad():
        var = z_enc[:, 1:].var().clamp(min=1e-8)
        pred = (logits.reshape(-1) > 0).float() if head == "binary" else logits.argmax(-1)
        correct = ((pred == target).float() * valid).sum() / valid.clamp(min=1).sum()
        metrics = {"dmt/r2_end": 1.0 - dyn[-1] / var,                  # drift at the END of the rollout
                   "dmt/r2_first": 1.0 - dyn[0] / var,
                   "dmt/slot_acc_rolled": correct,
                   "dmt/roll_len": float(steps - 1)}
    return losses, metrics


def dmt_total(losses, cfg):
    return (float(cfg.get("coef_dyn", 1.0)) * losses["dyn"]
            + float(cfg.get("coef_prior", 1.0)) * losses["prior"]
            + float(cfg.get("coef_slot", 1.0)) * losses["slot"])
