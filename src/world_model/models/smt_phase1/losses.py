"""The phase-1 objective: reward BCE at the lift, plus two one-step dynamics terms.

    z_t      = encoder(window ending at t)
    z_target = encoder(window ending at t+1)         SAME weights, slid one step
    z_post   = posterior.step(z_t, e_{t+1}, a_t)
    z_prior  = prior.step(sg(z_t), a_t)
    logits   = reward_decoder(z_t, e_fut, a_fut)     (B, F+1) -> r_t .. r_{t+F}

    reward = masked_mean(BCEWithLogits(logits, y_reward), reward_mask)
    post   = (z_post  - z_target).pow(2).mean()
    prior  = (z_prior - sg(z_target)).pow(2).mean()
    unif   = uniformity_loss(z_t)

Gradient paths, the contract scripts/verify_smt_phase1.py asserts (docs/SMT_PHASE1.md 3):

    term           | must reach                 | must NOT reach
    reward         | encoder, reward_decoder    | posterior, prior
    post           | encoder, posterior         | reward_decoder, prior
    prior          | prior                      | encoder, posterior, reward_decoder
    unif           | encoder                    | everything else
    delta_l1       | posterior.mask_net         | everything else       (DELTA only)
    delta_l1_prior | prior.mask_net             | everything else       (DELTA only)

Why the asymmetry. L_post with a LIVE target shapes the encoder to produce states that are
one-step reachable -- SMT's detach_rnn=False, and wanted. L_prior must not shape the encoder at
all: through its target it would push the encoder to make states predictable without seeing the
observation, i.e. to discard exactly the observation-specific information we need, and through
its INPUT it would push z_t to pre-encode a frame it has not seen. Detaching both makes the
prior chase the encoder and never the reverse. This is the asymmetry Dreamer gets from KL
balancing (dyn 1.0 / rep 0.1), expressed on vectors instead of categoricals.

(docs/SMT_PHASE1.md 3 writes `prior.step(z_t, a_next)` in its code block but its gradient table --
the normative, asserted one -- lists `prior -> prior only`, and the prose says the prior must
chase the encoder "never the reverse". The table wins; `detach_prior_input: false` restores the
code block's version if that reading is ever wanted.)

On .pow(2): MSE is SMT's own choice (model.py:31-38) and on this geometry it is equivalent to
cosine, since every token is rms-normed -- ||a-b||^2 = 2d - 2<a,b>. It is NOT neutral. It weights
all 2560 dimensions equally and the ball occupies almost none of that norm: measured in
smt_prelim, states with DIFFERENT ball answers sat 45.6 apart against 44.1 for matching ones, on
vectors of norm 50.6. That is how z_r2 reached 0.9999 while the answer still failed at 3-4 swaps.
The z_dist_same / z_dist_diff metrics below watch for exactly that; if it recurs, change the
dynamics metric first (whitened MSE, or a contrastive term scoring discriminability).
"""
import numpy as np
import torch
import torch.nn.functional as F

from ..smt_prelim.blocks import masked_mean, uniformity_loss


def split_batch(batch):
    """An anchor item -> the tensors the objective needs. See dataset_smt_prelim.SmtAnchors.

    e_win is the C+1 steps ending at t+1 and e_seq the F+1 steps starting at t, so:
      e_win[:, :-1]  window ending at t          e_win[:, 1:]   window ending at t+1
      e_win[:, -1]   e_{t+1}                     a_win[:, -1]   prev_action[t+1] == a_t
      e_seq[:, :-1]  e_t .. e_{t+F-1}            a_seq[:, :-1]  a_t .. a_{t+F-1}
      ball_seq       r_t .. r_{t+F}  (F+1 targets, one per read-out)
    """
    e_win, a_win, win_mask = batch["e_win"], batch["a_win"], batch["win_mask"]
    e_seq, a_seq = batch["e_seq"], batch["a_seq"]
    return dict(
        e_ctx=e_win[:, :-1], a_ctx=a_win[:, :-1], ctx_mask=win_mask[:, :-1],
        e_ctx_next=e_win[:, 1:], a_ctx_next=a_win[:, 1:], ctx_mask_next=win_mask[:, 1:],
        e_next=e_win[:, -1], a_next=a_win[:, -1],
        e_fut=e_seq[:, :-1], a_fut=a_seq[:, :-1],
        # ball_lift_seq is already ANDed with seq_mask by the dataset, so no padded position
        # can be supervised; ball_seq right-clamps to the last real step exactly as e_seq does.
        y_reward=batch["ball_seq"], reward_mask=batch["ball_lift_seq"].float(),
    )


def _pair_distances(z, answer, pairs=512):
    """Mean ||z_i - z_j|| for anchors whose ball answer matches, and for those where it differs.

    The smt_prelim failure signature: these two are nearly equal, meaning the direction the task
    depends on carries none of the state's norm and MSE has nothing to pull on.
    """
    flat = z.reshape(z.shape[0], -1)
    n = flat.shape[0]
    if n < 2:
        return None, None
    i = torch.randint(0, n, (pairs,), device=flat.device)
    j = torch.randint(0, n, (pairs,), device=flat.device)
    keep = i != j
    if not bool(keep.any()):
        return None, None
    i, j = i[keep], j[keep]
    dist = (flat[i] - flat[j]).norm(dim=-1)
    same = (answer[i] == answer[j]).float()
    if float(same.sum()) == 0 or float((1 - same).sum()) == 0:
        return None, None
    return (dist * same).sum() / same.sum(), (dist * (1 - same)).sum() / (1 - same).sum()


_DTW_CACHE = {}
_LABEL_CACHE = {}
_PAIR_CACHE = {}


def _pair_index(path, prefix):
    """{anchor key -> set of partner keys} for the pairs stored under `prefix` (\"\" positives, \"neg_\" negatives)."""
    key = (path, prefix)
    if key not in _PAIR_CACHE:
        d = np.load(path)
        part = {}
        if f"{prefix}x_episode" in d.files:
            for ex, tx, ey, ty in zip(d[f"{prefix}x_episode"], d[f"{prefix}x_t"], d[f"{prefix}y_episode"], d[f"{prefix}y_t"]):
                a, b = int(ex) * 10000 + int(tx), int(ey) * 10000 + int(ty)
                part.setdefault(a, set()).add(b); part.setdefault(b, set()).add(a)
        _PAIR_CACHE[key] = part
    return _PAIR_CACHE[key]


def _pairs_in_batch(keys, part):
    where = {}
    for i, k in enumerate(keys):
        where.setdefault(k, []).append(i)
    ii, jj = [], []
    for i, k in enumerate(keys):
        for kb in part.get(k, ()):
            for j in where.get(kb, ()):
                if j > i:
                    ii.append(i); jj.append(j)
    return ii, jj


def _soft_index(path):
    """{anchor key -> {partner key: s}} over BOTH lists of a soft pair file (scripts/soft_pairs_phase1.py)."""
    key = (path, "soft")
    if key not in _PAIR_CACHE:
        d = np.load(path)
        part = {}
        for pre in ("", "neg_"):
            for ex, tx, ey, ty, w in zip(d[f"{pre}x_episode"], d[f"{pre}x_t"], d[f"{pre}y_episode"], d[f"{pre}y_t"],
                                         d[f"{pre}w"]):
                a, b = int(ex) * 10000 + int(tx), int(ey) * 10000 + int(ty)
                part.setdefault(a, {})[b] = float(w); part.setdefault(b, {})[a] = float(w)
        _PAIR_CACHE[key] = part
    return _PAIR_CACHE[key]


def _soft_align(z_t, batch, cfg):
    """align_label "soft": every pair found in the batch is pulled with weight s and pushed with weight 1 - s.

    pull = mean over pairs of s * 0.25 * mean (z_x - z_y)^2           (Align-Walk's pull, scaled by s)
    push = log( sum_pairs,tokens (1 - s) exp(-t ||u_x - u_y||^2) / sum (1 - s) )   (the uniformity loss on
           unit-norm tokens, token k of x against token k of y, each pair weighted by 1 - s)
    Returns (pull, push, n pairs, sum of push weights).
    """
    part = _soft_index(str(cfg["align_pair_path"]))
    keys = (batch["episode"].long() * 10000 + batch["t"].long()).tolist()
    where = {}
    for i, k in enumerate(keys):
        where.setdefault(k, []).append(i)
    ii, jj, ww = [], [], []
    for i, k in enumerate(keys):
        for kb, w in part.get(k, {}).items():
            for j in where.get(kb, ()):
                if j > i:
                    ii.append(i); jj.append(j); ww.append(w)
    zero = z_t.sum() * 0.0
    if not ii:
        return zero, zero, 0, 0.0
    dev = z_t.device
    i_, j_, s = torch.tensor(ii, device=dev), torch.tensor(jj, device=dev), torch.tensor(ww, device=dev)
    flat = z_t.flatten(1).float()
    pull = (s * 0.25 * (flat[i_] - flat[j_]).pow(2).mean(-1)).mean()
    wp = (1.0 - s).clamp(min=0.0)
    if float(wp.sum()) <= 1e-6:
        return pull, zero, len(ii), 0.0
    u = z_t.float() / (z_t.float().norm(dim=-1, keepdim=True) + 1e-8)
    d2 = (u[i_] - u[j_]).pow(2).sum(-1)                                   # (n pairs, K tokens)
    t = float(cfg.get("align_neg_t", 2.0))
    push = ((wp[:, None] * d2.mul(-t).exp()).sum() / (wp.sum() * d2.shape[1])).log()
    return pull, push, len(ii), float(wp.sum())


def _pair_align(z_t, batch, cfg):
    """align_label "pairs": pull together anchors listed as positive pairs, push apart negative pairs
    (scripts/splice_pairs_phase1.py; the batch sampler, loader.pair_file, puts partners together).

    pull: each positive pair is a 2-member class, 0.25 * mean (z_i - z_j)^2 -- the (flat - class mean)^2 the
          class labels use.
    push: align_neg "hinge" = mean over negative pairs of relu(align_margin - ||z_i - z_j||)^2 (dtwpair's hinge);
          align_neg "unif"  = the uniformity loss restricted to negative pairs (never zero).
    Returns (pull, push, n positive pairs in the batch, n negative pairs in the batch).
    """
    path = str(cfg["align_pair_path"])
    keys = (batch["episode"].long() * 10000 + batch["t"].long()).tolist()
    flat = z_t.flatten(1)
    zero = z_t.sum() * 0.0
    ii, jj = _pairs_in_batch(keys, _pair_index(path, ""))
    pull = zero if not ii else 0.25 * (flat[torch.tensor(ii, device=z_t.device)]
                                       - flat[torch.tensor(jj, device=z_t.device)]).pow(2).mean()
    ni, nj = _pairs_in_batch(keys, _pair_index(path, "neg_"))
    if not ni:
        push = zero
    elif str(cfg.get("align_neg", "hinge")) == "unif":
        # the uniformity loss (blocks.uniformity_loss: unit-norm tokens, log mean exp(-t ||u_i - u_j||^2)) over
        # NEGATIVE pairs only, token k of x against token k of y. Never zero, strongest on the closest pairs.
        u = z_t.float() / (z_t.float().norm(dim=-1, keepdim=True) + 1e-8)
        d2 = (u[torch.tensor(ni, device=z_t.device)] - u[torch.tensor(nj, device=z_t.device)]).pow(2).sum(-1)
        push = d2.mul(-float(cfg.get("align_neg_t", 2.0))).exp().mean().log()
    else:
        d = (flat[torch.tensor(ni, device=z_t.device)] - flat[torch.tensor(nj, device=z_t.device)]).norm(dim=-1)
        push = torch.relu(float(cfg.get("align_margin", 20.0)) - d).pow(2).mean()
    return pull, push, len(ii), len(ni)


def _file_labels(batch, cfg, device):
    """align_label "file": (eligible, class) per anchor from a precomputed (episode, t) -> label table.

    Built by scripts/build_self_slot0_labels.py: a trained model's OWN answer to "is the ball in slot 0
    right now?" at every at-rest hold frame (cups down, swap ends, idle frames, onset). Every frame in
    the table is eligible -- including those with swaps still to come -- not only motionless frames.
    """
    path = str(cfg["align_label_path"])
    if path not in _LABEL_CACHE:
        d = np.load(path)
        _LABEL_CACHE[path] = {int(e) * 10000 + int(t): int(l) for e, t, l in zip(d["episode"], d["t"], d["label"])}
    table = _LABEL_CACHE[path]
    keys = (batch["episode"].long() * 10000 + batch["t"].long()).tolist()
    lab = [table.get(k, -1) for k in keys]
    cls = torch.tensor(lab, device=device)
    return cls >= 0, cls.clamp(min=0)


def _dtw_table(path, device):
    """(matrix, {episode*10000+t -> row}, moves_ahead) from scripts/build_dtw_matrix.py, cached."""
    key = (path, str(device))
    if key not in _DTW_CACHE:
        d = np.load(path)
        lut = {int(e) * 10000 + int(t): i for i, (e, t) in enumerate(zip(d["episode"], d["t"]))}
        _DTW_CACHE[key] = (torch.from_numpy(d["dtw"].astype(np.float32)).to(device), lut,
                           torch.from_numpy(d["moves_ahead"]).to(device))
    return _DTW_CACHE[key]


def _dtw_pairs(z_t, batch, cfg):
    """DTW pairing with ALIGN-LIFT's attract+hinge form, which THRESHOLDS instead of regressing.

        pull = mean over pairs with dtw < near  of ||z_i - z_j||^2
        push = mean over pairs with dtw > far   of relu(margin - ||z_i - z_j||)^2

    Why not the regression in _dtw_align: only 1.64% of eligible pairs carry the signal (dtw ~ 0);
    the rest have targets in [7,31] set by future choreography the state cannot know. Regressing
    on all of them fits noise -- measured R^2 = -0.37 over 20k steps. Thresholding keeps the
    informative tails and drops the middle.
    """
    mat, lut, mov = _dtw_table(str(cfg["align_dtw_path"]), z_t.device)
    ep = batch["episode"].detach().cpu().numpy().astype(np.int64)
    tt = batch["t"].detach().cpu().numpy().astype(np.int64)
    rows = np.fromiter((lut.get(int(a) * 10000 + int(b), -1) for a, b in zip(ep, tt)),
                       dtype=np.int64, count=len(ep))
    r = torch.from_numpy(rows).to(z_t.device)
    keep = (r >= 0)
    keep[keep.clone()] = mov[r[keep]]
    k = int(keep.sum())
    zero = z_t.sum() * 0.0
    if k < 2:
        return zero, zero, 0, 0
    zf = z_t.flatten(1)[keep]
    rr = r[keep]
    iu = torch.triu_indices(k, k, offset=1, device=z_t.device)
    dt = mat[rr[iu[0]], rr[iu[1]]]
    dz = (zf[iu[0]] - zf[iu[1]]).norm(dim=-1)
    near = dt < float(cfg.get("align_near", 1.0))
    far = dt > float(cfg.get("align_far", 5.0))
    margin = float(cfg.get("align_margin", 20.0))
    pull = dz[near].pow(2).mean() if bool(near.any()) else zero
    push = torch.relu(margin - dz[far]).pow(2).mean() if bool(far.any()) else zero
    return pull, push, k, int(near.sum())


def _dtw_align(z_t, batch, cfg):
    """Metric regression onto the DTW distance between FUTURE EMBEDDING TRAJECTORIES.

        L = mean over cross-episode pairs of ( ||z_i - z_j|| - lambda * dtw_ij )^2

    No privileged label anywhere: dtw_ij comes from the cached embeddings, and the eligibility
    flag `moves_ahead` (does the future contain motion before the lift) is read off those same
    embeddings. Pairs are restricted to anchors with motion still to come -- in the post-swap
    slack two different non-target slots have IDENTICAL futures, so including slack pairs would
    target distance 0 for them and re-collapse the partition (measured: 1-vs-2 pull 1.00x of
    same-slot unrestricted, 0.23x restricted).

    align_dtw_mode picks how dtw_ij is used:

    "attract" (default) -- soft-weighted alignment, w_ij = exp(-dtw_ij / tau):

        L = sum_ij w_ij * ||z_i - z_j||^2 / D  /  sum_ij w_ij

      Attract-only, because the DTW criterion is SOUND BUT INCOMPLETE: dtw ~ 0 implies the two
      states are equivalent, but a large dtw implies nothing -- the futures may have diverged
      only because the upcoming schedules differed. Repulsion therefore has to come from
      coef_unif, not from dtw.

    "regress" -- metric regression, ( ||z_i - z_j|| - lambda * dtw_ij )^2.  MEASURED TO FAIL:
      mean dtw is 9.46 for same-slot pairs and 9.64 for different-slot pairs, so 98% of the
      target encodes the upcoming SCHEDULE (which z cannot know) and only 1.9% the ball, while
      96.8% of same-slot pairs get pushed apart to a median target distance of 19.4. Kept only
      so the negative result is reproducible.
    """
    mat, lut, mov = _dtw_table(str(cfg["align_dtw_path"]), z_t.device)
    ep = batch["episode"].detach().cpu().numpy().astype(np.int64)
    tt = batch["t"].detach().cpu().numpy().astype(np.int64)
    rows = np.fromiter((lut.get(int(a) * 10000 + int(b), -1) for a, b in zip(ep, tt)),
                       dtype=np.int64, count=len(ep))
    r = torch.from_numpy(rows).to(z_t.device)
    keep = (r >= 0)
    keep[keep.clone()] = mov[r[keep]]
    k = int(keep.sum())
    if k < 2:
        return z_t.sum() * 0.0, 0, 0
    zf = z_t.flatten(1)[keep]
    rr = r[keep]
    epk = torch.from_numpy(ep).to(z_t.device)[keep]
    iu = torch.triu_indices(k, k, offset=1, device=z_t.device)
    cross = epk[iu[0]] != epk[iu[1]]                      # same-episode rest frames are near-copies
    if int(cross.sum()) < 1:
        return z_t.sum() * 0.0, k, 0
    a, b = iu[0][cross], iu[1][cross]
    dt = mat[rr[a], rr[b]]
    if str(cfg.get("align_dtw_mode", "attract")) == "regress":
        dz = (zf[a] - zf[b]).norm(dim=-1)
        lam = float(cfg.get("align_lambda", 2.0))
        return (dz - lam * dt).pow(2).mean(), k, int(cross.sum())
    tau = float(cfg.get("align_tau", 2.0))
    w = torch.exp(-dt / tau)
    d2 = (zf[a] - zf[b]).pow(2).mean(-1)          # per-dimension, so coef_align matches the
    return (w * d2).sum() / w.sum().clamp(min=1e-6), k, int(cross.sum())   # slot/target arms


def _mask_metrics(m, name, batch):
    """Mean gate, fraction of regions open (> 0.5), and -- diagnostics only -- the mean gate on
    motionless vs moving transitions. The prediction is static ~0 (docs/DELTA.md 8)."""
    m = m.float().flatten(1)                                   # (B, regions)
    out = {f"delta/{name}_mean": m.mean(), f"delta/{name}_open": (m > 0.5).float().mean()}
    e = batch["e_win"]
    still = (e[:, -1] - e[:, -2]).abs().amax(-1) < 1e-6       # e_{t+1} == e_t: nothing happened
    if bool(still.any()):
        out[f"delta/{name}_mean_static"] = m[still].mean()
    if bool((~still).any()):
        out[f"delta/{name}_mean_moving"] = m[~still].mean()
    return out


def phase1_losses(encoder, posterior, prior, reward_decoder, batch, cfg):
    """Returns (losses, metrics, z_t). Losses are unscaled; the caller applies coefficients."""
    parts = split_batch(batch)

    z_t = encoder.tokens(parts["e_ctx"], parts["a_ctx"], parts["ctx_mask"])
    z_target = encoder.tokens(parts["e_ctx_next"], parts["a_ctx_next"], parts["ctx_mask_next"])

    # DELTA (docs/DELTA.md 3): the gated step. m is None when delta_mask is off, and then z_post /
    # z_prior are exactly what posterior.step / prior.step returned before the gate existed.
    z_post, m_post = posterior.step_with_mask(z_t, parts["e_next"], parts["a_next"])
    prior_in = z_t.detach() if bool(cfg.get("detach_prior_input", True)) else z_t
    z_prior, m_prior = prior.step_with_mask(prior_in, parts["a_next"])

    logits = reward_decoder(z_t, parts["e_fut"], parts["a_fut"])

    mask = parts["reward_mask"]
    y = parts["y_reward"].float()
    loss_mask = mask
    head = getattr(reward_decoder, "head", "binary")
    if head == "embed":
        # SMT's predictive-state objective, action-conditioned: from the memory and the coming
        # actions, predict the frame embeddings themselves. No ball label anywhere in the head.
        target_e = batch["e_seq"]
        loss_mask = batch["seq_mask"].float()
        bce = (logits.float() - target_e).pow(2).mean(-1)
        slot_correct = None
    elif head == "slot":
        # 3-way: which slot holds the ball.
        #   lift  : the 9 lift frames only, same mask as the binary head
        #   dense : EVERY real future step, label switching at each swap's halfway frame
        if str(cfg.get("slot_supervision", "lift")) == "dense":
            if "slot_dense_seq" not in batch:
                raise KeyError("slot_supervision=dense but the dataset has no dense slot labels "
                               "(irregular ball_slot swaps -- see the [dataset] warning at load)")
            slot = batch["slot_dense_seq"].long()
            loss_mask = batch["seq_mask"].float()
        else:
            slot = batch["slot_seq"].long()
        bce = F.cross_entropy(logits.float().flatten(0, 1), slot.flatten(), reduction="none").view_as(y)
        slot_correct = (logits.argmax(-1) == slot).float()
        slot_mask = loss_mask
        logits = reward_decoder.ball_logit(logits)       # binary view for every metric below
    else:
        slot_correct = None
        bce = F.binary_cross_entropy_with_logits(logits.float(), y, reduction="none")

    # --- alignment: states that SHOULD be identical are pulled together -------------------
    # Eligible = motionless HOLD frames only; every moving frame is excluded, and lift frames
    # (phase 2) are excluded too, so this is disjoint from the reward head's mask. What defines
    # a positive pair is align_label: the 3-way slot (privileged) or the 1-bit lift answer.
    # Wang & Isola's alignment term; the separation hinge replaces blind uniformity, which
    # pushes apart exactly the states this is trying to merge.
    align = z_t.sum() * 0.0
    sep = z_t.sum() * 0.0
    dtw_counts = None
    if float(cfg.get("coef_align", 0.0)) > 0.0 or float(cfg.get("coef_sep", 0.0)) > 0.0:
        e_ctx = parts["e_ctx"]
        still = (e_ctx[:, -1] - e_ctx[:, -2]).abs().amax(-1) < 1e-6        # frame t == frame t-1
        eligible = still & (batch["meta/phase"] == 1) if "meta/phase" in batch else still
        label = str(cfg.get("align_label", "slot"))
        if label == "file":
            # the model's own "ball in slot 0 now?" at every at-rest hold frame, mid-hold included
            eligible, cls = _file_labels(batch, cfg, z_t.device)
            n_cls = 2
            if cfg.get("align_class_weights") is not None:
                # several classes, each pulled together with its own weight (ALIGN-SELF-PREDICTIONS:
                # "ball here" read directly, and "ball comes out after imagined swap k", weaker)
                n_cls = len(cfg["align_class_weights"])
        elif label == "dtw":
            # metric regression onto future-trajectory DTW; no labels, no margin, no sep term.
            align, n_elig, n_pairs = _dtw_align(z_t, batch, cfg)
            dtw_counts = (n_elig, n_pairs)
        elif label == "dtwpair":
            # DTW pairing, ALIGN-LIFT's attract+hinge form. sep IS used here.
            align, sep, n_elig, n_pairs = _dtw_pairs(z_t, batch, cfg)
            dtw_counts = (n_elig, n_pairs)
        elif label == "pairs":
            # positive pairs from splicing real futures (no labels, no classes)
            align, sep, n_pairs, n_neg = _pair_align(z_t, batch, cfg)
            dtw_counts = (n_neg, n_pairs)
        elif label == "soft":
            # soft pairs (scripts/soft_pairs_phase1.py): pull with weight s, push with weight 1 - s
            align, sep, n_pairs, w_push = _soft_align(z_t, batch, cfg)
            dtw_counts = (w_push, n_pairs)
        elif label == "slot":
            # 3 classes: WHICH cup hides the ball. Privileged -- log2(3) bits per rest frame,
            # and it hands over the whole partition the task is about.
            cls, n_cls = batch["slot_dense_seq"][:, 0].long(), 3
        elif label == "target":
            # 2 classes: is the ball under the cup that will be LIFTED -- the task's own output
            # bit, nothing more. Says nothing about which of the other two holds it, so the
            # 3-way partition is not handed over. Positives join rest states at ANY swap depth.
            # Verified identical to (slot_dense_seq == episode_target) at every eligible frame.
            cls, n_cls = batch["meta/ball_under_target"].long(), 2
        else:
            raise ValueError(f"align_label must be 'slot', 'target', 'file', 'dtw' or "
                             f"'dtwpair', got {label!r}")
        flat = z_t.flatten(1)
        # align_positive_only: with align_label="target" the two classes are {ball under the cup
        # that lifts} and {not}. The NEGATIVE class is poison -- the lifted cup is always slot 0,
        # so "not under it" lumps slot 1 with slot 2 and pulling it together merges them. Measured
        # on ALIGN-TARGET: the 1<->2 prototype distance collapsed to 1.90 against a within-class
        # spread of 6.76, while 0<->1 and 0<->2 held at ~16.6; that partition is not closed under
        # the swaps, hence its decay with depth. Positives are a true equivalence (same slot,
        # whatever the history); negatives assert something we do not know, so drop the loss there.
        # The negative class still gets a MEAN for the sep hinge (align_positive_sep), just never
        # the pull-together term. That distinction is the whole point: L_align on a class squashes
        # its members onto a common point, which is what merged slots 1 and 2; the hinge on the
        # means instead pushes every negative in the SAME direction, a uniform translation of the
        # group, which leaves its internal spread untouched. (Dual of docs/ALIGN.md 3(c): L_align is
        # translation-invariant per class, so a loss on the means acts only by translation.)
        # Without it there is no repulsion at all: ALIGN-POS measured 0<->1 and 0<->2 collapsing
        # from ~16.6 to ~7, costing 1-2 swap accuracy (1.00 -> 0.89 / 0.60).
        pos_only = bool(cfg.get("align_positive_only", False))
        pos_sep = bool(cfg.get("align_positive_sep", False))
        class_w = cfg.get("align_class_weights") if label in ("file", "slot") else None   # per-class pull weights
        means, counts = [], []
        for s in ([] if label in ("dtw", "dtwpair", "pairs", "soft") else range(n_cls)):
            m = eligible & (cls == s)
            if int(m.sum()) > 1:
                mu = flat[m].mean(0, keepdim=True)
                if class_w is not None:
                    if float(class_w[s]) > 0:
                        align = align + float(class_w[s]) * (flat[m] - mu).pow(2).mean()
                elif not (pos_only and s != 1):
                    align = align + (flat[m] - mu).pow(2).mean()       # PULL: positives only
                elif not pos_sep:
                    continue                                            # no mean, no hinge
                means.append(mu.squeeze(0)); counts.append(int(m.sum()))  # PUSH: both means
        if len(means) > 1 and label not in ("dtw", "dtwpair", "pairs", "soft"):
            margin = float(cfg.get("align_margin", 20.0))
            pairs = [(i, j) for i in range(len(means)) for j in range(i + 1, len(means))]
            sep = sum(torch.relu(margin - (means[i] - means[j]).norm()).pow(2) for i, j in pairs) / len(pairs)

    prior_target = z_target.detach() if bool(cfg.get("detach_prior_target", True)) else z_target
    losses = {
        "reward": masked_mean(bce, loss_mask),
        "post": (z_post - z_target).pow(2).mean(),
        "prior": (z_prior - prior_target).pow(2).mean(),
        "unif": uniformity_loss(z_t, sample=int(cfg.get("unif_sample", 512))),
        "align": align,
        "sep": sep,
        # L1 on the gates (m >= 0, so the mean IS the L1). Each reaches only its own MaskNet:
        # the gate reads the event alone, never z, so no path back to the encoder or the trunk.
        "delta_l1": m_post.float().mean() if m_post is not None else z_t.sum() * 0.0,
        "delta_l1_prior": m_prior.float().mean() if m_prior is not None else z_t.sum() * 0.0,
    }

    with torch.no_grad():
        if head == "embed":
            var = batch["e_seq"].var().clamp(min=1e-8)
            metrics = {"recon_mse": masked_mean(bce, loss_mask),
                       "recon_r2": 1.0 - masked_mean(bce, loss_mask) / var,
                       "z_post_r2": 1.0 - (z_post - z_target).pow(2).mean() / z_target.var().clamp(min=1e-8),
                       "z_std": z_t.std(0).mean(),
                       "obs_sensitivity": (z_post - posterior.step(z_t, parts["e_next"].flip(0),
                                                                   parts["a_next"])).abs().mean()}
            return losses, metrics, z_t
        correct = ((logits > 0).float() == y).float()
        residual_post = (z_post - z_target).pow(2).mean()
        residual_prior = (z_prior - z_target).pow(2).mean()
        variance = z_target.var().clamp(min=1e-8)
        # does the posterior respond to what it is shown? the RSSM cell scored exactly 0.0 here
        shuffled = posterior.step(z_t, parts["e_next"].flip(0), parts["a_next"])
        metrics = {
            "reward_acc": masked_mean(correct, mask),
            "reward_pos_rate": masked_mean(y, mask),
            "reward_frac_supervised": mask.sum() / mask.numel(),
            "reward_n_anchors": (mask.sum(1) > 0).float().mean(),
            "z_post_r2": 1.0 - residual_post / variance,
            "z_prior_r2": 1.0 - residual_prior / variance,
            "z_post_cos": torch.cosine_similarity(z_post.flatten(1), z_target.flatten(1), dim=-1).mean(),
            "z_prior_cos": torch.cosine_similarity(z_prior.flatten(1), z_target.flatten(1), dim=-1).mean(),
            "z_std": z_t.std(0).mean(),
            "z_norm": z_t.flatten(1).norm(dim=-1).mean(),
            "obs_sensitivity": (z_post - shuffled).abs().mean(),
        }
        if m_post is not None:
            metrics.update(_mask_metrics(m_post, "post", batch))
            metrics.update(_mask_metrics(m_prior, "prior", batch))
        if dtw_counts is not None:
            metrics["dtw_eligible"] = float(dtw_counts[0])      # rest anchors with motion ahead
            metrics["dtw_pairs"] = float(dtw_counts[1])         # cross-episode pairs scored
        if slot_correct is not None:
            metrics["slot_acc"] = masked_mean(slot_correct, slot_mask)       # on the supervised steps
            metrics["slot_acc_lift"] = masked_mean(slot_correct, mask)       # lift frames only
        # THE number: accuracy at the FIRST supervised position of each anchor. Later positions
        # hand the decoder observation tokens that already show the ball; the first one is the
        # only place the answer must come from the memory.
        has = mask.sum(1) > 0
        if bool(has.any()):
            rows = torch.nonzero(has).squeeze(-1)
            metrics["reward_acc_first"] = correct[rows, mask.argmax(1)[rows]].mean()
            # memory-only read-outs (r_t, r_{t+1}) against the ones that accumulate observations.
            # A large gap means the decoder is leaning on tokens rather than on the state.
            index = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
            early, late = mask * (index <= 1), mask * (index > 1)
            if float(early.sum()) > 0:
                metrics["reward_acc_early"] = masked_mean(correct, early)
            if float(late.sum()) > 0:
                metrics["reward_acc_late"] = masked_mean(correct, late)
            # the MSE-geometry watch: is the ball answer visible in the state's bulk distance?
            answer = y[rows, mask.argmax(1)[rows]]
            same, diff = _pair_distances(z_t[rows], answer)
            if same is not None:
                metrics["z_dist_same"] = same
                metrics["z_dist_diff"] = diff
                metrics["z_dist_gap"] = diff - same
        if "meta/phase" in batch:                      # labels are diagnostics only
            phase = batch["meta/phase"]
            for value, name in ((0, "reveal"), (1, "hold"), (2, "show")):
                sel = (phase == value).float()
                if float(sel.sum()) > 0:
                    metrics[f"reward_acc/{name}"] = masked_mean(correct, mask * sel[:, None])

    return losses, metrics, z_t
