"""PHASE 1 gates 1-3 (docs/SMT_PHASE1.md 6), asserted before any training run.

    1  dataset       fut_len 20 -> exactly 9 supervised positions per episode at show offsets
                     1-9, none in the pad, positive rate ~0.36, ~68% anchor coverage
    2  gradients     the four-term attribution table. A stray .detach() silently removes a path
                     and looks healthy right up to the final eval; this has happened twice here
    3  shapes and    the reward decoder is causal (perturbing future position j leaves every
       causality     read-out at or before j bit-identical); the posterior responds to its
                     embedding input; the prior cannot see one at all
    4  DELTA         (only when arch.delta_mask != off, docs/DELTA.md 5) a shut gate returns z_t
                     exactly, an open gate returns z_hat exactly, and the L1 terms reach only
                     their own MaskNet

    pixi run verify-smt-phase1
    pixi run verify-smt-phase1 gates=1          # dataset only, no GPU / no frozen encoder
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import DictConfig  # noqa: E402
from torch.utils.data import default_collate  # noqa: E402

from world_model.models.smt_phase1.losses import phase1_losses, split_batch  # noqa: E402
from world_model.training.loop import to_device  # noqa: E402
from world_model.utils.seeding import seed_everything  # noqa: E402

SHOW_OFFSETS = list(range(1, 10))

# term -> (must reach, must not reach). docs/SMT_PHASE1.md 3.
CONTRACT = {
    "reward": ({"encoder", "reward_decoder"}, {"posterior", "prior"}),
    "post": ({"encoder", "posterior"}, {"reward_decoder", "prior"}),
    "prior": ({"prior"}, {"encoder", "posterior", "reward_decoder"}),
    "unif": ({"encoder"}, {"posterior", "prior", "reward_decoder"}),
    "align": ({"encoder"}, {"posterior", "prior", "reward_decoder"}),
    "sep": ({"encoder"}, {"posterior", "prior", "reward_decoder"}),
    # DELTA: the gate reads the event alone, so its L1 has no path to z, the trunk or the teacher.
    # Module-level here; gate 4 then checks it reaches ONLY the mask_net inside each updater.
    "delta_l1": ({"posterior"}, {"encoder", "prior", "reward_decoder"}),
    "delta_l1_prior": ({"prior"}, {"encoder", "posterior", "reward_decoder"}),
}


def gate_dataset(train_set, holdout_set, sample=2000):
    data = train_set.data
    lift, target, phase = data.ball_lift, data.ball_target, data.meta["phase"]

    per_episode = {int(lift[e, :int(n)].sum()) for e, n in enumerate(data.real_len)}
    assert per_episode == {9}, f"expected 9 supervised frames per episode, got {sorted(per_episode)}"
    assert (phase[lift] == 2).all(), "a supervised frame is outside the show phase"
    assert (data.meta["pad"][lift] == 0).all(), "a supervised frame landed in the pad"
    for e, n in enumerate(data.real_len):
        n = int(n)
        offsets = (np.flatnonzero(lift[e, :n]) - (n - 10)).tolist()
        assert offsets == SHOW_OFFSETS, f"episode {e} supervises offsets {offsets}"
        assert (target[e, lift[e]] == target[e, n - 1]).all(), "the target varies within the lift"
    positive = float(target[lift].mean())
    assert 0.30 < positive < 0.42, f"positive rate {positive:.3f} is not the expected ~0.36"

    stride = max(1, len(train_set) // sample)
    picked = range(0, len(train_set), stride)
    covered, seen = 0, 0
    for i in picked:
        item = train_set[int(i)]
        mask = item["ball_lift_seq"]
        assert item["ball_seq"].shape[0] == train_set.fut_len + 1, "ball_seq is not F+1 long"
        assert not bool(mask[~item["seq_mask"]].any()), "a supervised position landed in the pad"
        covered += int(bool(mask.any()))
        seen += 1
    coverage = covered / seen
    assert coverage > 0.60, f"anchor coverage {coverage:.3f} is far below the measured 0.684"

    print(f"  gate 1 OK  9 supervised frames/episode at show offsets 1-9, positive rate "
          f"{positive:.3f}, {coverage:.3f} of {seen} sampled anchors see a lift position "
          f"(fut_len {train_set.fut_len}); {len(train_set)} train / {len(holdout_set)} holdout anchors")
    return {"positive_rate": positive, "anchor_coverage": coverage}


def gate_gradients(model, batch):
    losses, _, _ = phase1_losses(model.encoder, model.posterior, model.prior,
                                 model.reward_decoder, batch, model.loss_cfg)
    groups = model.modules_by_term()
    for term, (must, must_not) in CONTRACT.items():
        if term in ("align", "sep", "delta_l1", "delta_l1_prior") and float(losses[term].detach()) == 0.0:
            print(f"  gate 2 --  {term:<7} inactive (coefficient 0 in this config)"); continue
        for module in groups.values():
            module.zero_grad(set_to_none=True)
        losses[term].backward(retain_graph=True)
        reached = {name for name, module in groups.items()
                   if any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in module.parameters())}
        missing, stray = must - reached, reached & must_not
        assert not missing, f"loss '{term}' does not reach {sorted(missing)} (reached {sorted(reached)})"
        assert not stray, f"loss '{term}' leaks into {sorted(stray)}"
        print(f"  gate 2 OK  {term:<7} -> {sorted(reached)}")
    for module in groups.values():
        module.zero_grad(set_to_none=True)


@torch.no_grad()
def gate_shapes(model, batch):
    parts = split_batch(batch)
    z = model.encoder.tokens(parts["e_ctx"], parts["a_ctx"], parts["ctx_mask"])
    fut_len = parts["a_fut"].shape[1]
    assert z.shape[1:] == (model.arch["n_registers"], model.arch["d_model"]), f"memory is {tuple(z.shape)}"

    logits = model.reward_decoder(z, parts["e_fut"], parts["a_fut"])
    assert tuple(logits.shape[:2]) == (z.shape[0], fut_len + 1), \
        f"reward logits {tuple(logits.shape)} != (B, {fut_len + 1})"

    # causality: corrupt future position j; read-outs r_t..r_{t+j} must not move at all.
    # r_{t+k} is read at the action token a_{t+k-1}, so the last read-out that predates
    # a_j / e_j is r_{t+j}, at index j.
    j = fut_len // 2
    e_bad, a_bad = parts["e_fut"].clone(), parts["a_fut"].clone()
    e_bad[:, j] = torch.randn_like(e_bad[:, j]) * 10.0
    a_bad[:, j] = torch.randn_like(a_bad[:, j]) * 10.0
    perturbed = model.reward_decoder(z, e_bad, a_bad)
    head = (logits[:, :j + 1] - perturbed[:, :j + 1]).abs().max()
    tail = (logits[:, j + 1:] - perturbed[:, j + 1:]).abs().max()
    assert float(head) == 0.0, f"reward decoder is not causal: r_t..r_t+{j} moved by {float(head):.3e}"
    assert float(tail) > 0.0, "perturbing a future token changed nothing downstream either"
    if not model.reward_decoder.use_obs:
        # no-obs decoder: future embeddings must not reach ANY read-out
        e_only = model.reward_decoder(z, torch.randn_like(parts["e_fut"]) * 10.0, parts["a_fut"])
        leak = float((logits - e_only).abs().max())
        assert leak == 0.0, f"decoder_obs=false but future embeddings moved the logits by {leak:.3e}"
        print(f"  gate 3 OK  decoder_obs=false: randomising every future embedding moves logits by {leak:.1e}")
    print(f"  gate 3 OK  reward decoder causal at j={j} (head delta {float(head):.1e}, "
          f"tail delta {float(tail):.3f}); memory {tuple(z.shape[1:])}, logits {tuple(logits.shape)}")

    # the posterior must respond to the observation; the RSSM cell it replaces passed exactly 0.0
    z_post = model.posterior.step(z, parts["e_next"], parts["a_next"])
    flipped = model.posterior.step(z, parts["e_next"].flip(0), parts["a_next"])
    sensitivity = float((z_post - flipped).abs().mean())
    assert sensitivity > 1e-4, f"posterior obs_sensitivity {sensitivity:.3e} -- it is blind to what it sees"

    # the prior's insensitivity is structural, not learned: step() has no embedding parameter
    import inspect
    params = list(inspect.signature(model.prior.step).parameters)
    assert params == ["z", "action"], f"PriorUpdater.step takes {params}; it must never see an embedding"
    z_prior = model.prior.step(z, parts["a_next"])
    assert z_prior.shape == z.shape, f"prior returned {tuple(z_prior.shape)}"
    print(f"  gate 3 OK  posterior obs_sensitivity {sensitivity:.4f}; prior.step{tuple(params)} "
          f"cannot be shown an embedding")


def gate_delta(model, batch):
    """docs/DELTA.md 5 smoke tests 2-3. Test 1 (off == baseline, bit for bit) is structural: with
    delta_mask off no MaskNet exists and gate() returns z_hat itself."""
    if model.posterior.mask_net is None:
        print("  gate 4 --  delta_mask off: no MaskNet built, step() is the ungated updater")
        return
    parts = split_batch(batch)
    with torch.no_grad():
        z = model.encoder.tokens(parts["e_ctx"], parts["a_ctx"], parts["ctx_mask"]).float()
        for name, mod, args in (("posterior", model.posterior, (parts["e_next"], parts["a_next"])),
                                ("prior", model.prior, (parts["a_next"],))):
            net = mod.mask_net
            weight, bias = net.out.weight.data.clone(), net.out.bias.data.clone()
            net.out.weight.data.zero_()
            net.out.bias.data.fill_(-1e4)                        # sigmoid -> exactly 0
            z_shut, m = mod.step_with_mask(z, *args)
            assert float(m.max()) == 0.0, f"{name}: gate did not shut ({float(m.max())})"
            assert torch.equal(z_shut, z), f"{name}: a shut gate moved z by {float((z_shut - z).abs().max()):.3e}"
            net.out.bias.data.fill_(1e4)                         # sigmoid -> exactly 1
            z_open, m = mod.step_with_mask(z, *args)
            mod.mask_net = None                                  # the ungated proposal z_hat
            z_hat = mod.step(z, *args)
            mod.mask_net = net
            net.out.weight.data.copy_(weight)
            net.out.bias.data.copy_(bias)
            assert float(m.min()) == 1.0, f"{name}: gate did not open ({float(m.min())})"
            assert torch.equal(z_open, z_hat), f"{name}: an open gate is not z_hat"
            print(f"  gate 4 OK  {name}: m=0 returns z_t exactly, m=1 returns z_hat exactly "
                  f"(mask {model.posterior.delta_mask}, shape {tuple(m.shape[1:])})")

    losses, _, _ = phase1_losses(model.encoder, model.posterior, model.prior,
                                 model.reward_decoder, batch, model.loss_cfg)
    named = dict(model.named_parameters())
    for term, owner in (("delta_l1", "posterior.mask_net."), ("delta_l1_prior", "prior.mask_net.")):
        model.zero_grad(set_to_none=True)
        losses[term].backward(retain_graph=True)
        reached = {n for n, p in named.items() if p.grad is not None and bool(p.grad.abs().sum() > 0)}
        stray = {n for n in reached if not n.startswith(owner)}
        assert reached, f"{term} reaches nothing"
        assert not stray, f"{term} leaks into {sorted(stray)[:5]}"
        print(f"  gate 4 OK  {term:<14} -> only {owner[:-1]} ({len(reached)} tensors)")
    model.zero_grad(set_to_none=True)


@hydra.main(version_base=None, config_path="../configs/smt_phase1", config_name="verify_phase1")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    gates = {int(g) for g in str(cfg.gates).split(",")}

    train_set, holdout_set = instantiate(cfg.data_anchors)
    print(f"verifying smt_phase1 | ctx {cfg.model.ctx_len} fut {cfg.model.fut_len} "
          f"eval_warmup {cfg.model.eval_warmup}")
    if 1 in gates:
        gate_dataset(train_set, holdout_set)
    if not gates - {1}:
        return

    model = instantiate(cfg.model, action_dim=train_set.action_dim, device=cfg.device, _recursive_=False)
    print(f"  built {sum(p.numel() for p in model.trainable()) / 1e6:.2f}M trainable parameters")
    # an anchor batch that is guaranteed to contain supervised reward positions, so the reward
    # term is not silently zero when its gradient path is under test
    rng = np.random.default_rng(0)
    picked, tried = [], 0
    while len(picked) < 16 and tried < 20000:
        i = int(rng.integers(len(train_set)))
        tried += 1
        if bool(train_set[i]["ball_lift_seq"].any()):
            picked.append(train_set[i])
    assert len(picked) == 16, "could not find 16 anchors with a supervised reward position"
    batch = to_device(default_collate(picked), cfg.device)

    if 2 in gates:
        gate_gradients(model, batch)
    if 3 in gates:
        gate_shapes(model, batch)
    if 4 in gates:
        gate_delta(model, batch)
    print("all requested gates pass")


if __name__ == "__main__":
    main()
