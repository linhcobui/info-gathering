"""PHASE 1: a plain-vector belief maintained recurrently, over frozen R2-Dreamer embeddings.

    memory  z : (K=10, d=256) rms-normed vectors. 2560 floats. No deter, no logits, no
                categorical, no sampling.

Stage 1 trains the teacher encoder, the two updaters and the reward decoder on anchors
(train_smt_phase1.py). Stage 2a fits StateReadout on frozen stage-1 states
(train_readout_phase1.py); stage 2b fits the pixel head on real embeddings (train_decoder.py);
stage 3 is scripts/eval_memtest1.py, unchanged -- everything below the `WorldModel` ABC.

The eval is the point of the redesign. observe() teacher-encodes ONE window (eval_warmup) and
then runs the POSTERIOR over real embeddings for every remaining step; imagine() runs the PRIOR
with no observations at all and turns each state back into a frame through the readout. The
reward decoder is not used at eval, at all. That is what ctx_len 60 made untestable in
smt_prelim, where the teacher re-encoded the whole prefix at every step.
"""
import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from torch.amp import autocast

from world_model.models.base import WorldModel

from ..r2dreamer.native_config import load_native_config
from ..smt_prelim.blocks import sliding_windows, window_mask
from ..smt_prelim.bridge import to_f32
from ..smt_prelim.embed_cache import build_encoder, exact_fp32, read_embed_attrs
from ..smt_prelim.pixel_decoder import EmbedPixelDecoder
from ..smt_prelim.teacher import TeacherEncoder
from .decoder import RewardDecoder
from .losses import phase1_losses
from .readout import StateReadout
from .updater import PosteriorUpdater, PriorUpdater

DEFAULT_ARCH = {"d_model": 256, "n_registers": 10, "encoder_depth": 8, "posterior_depth": 8,
                "prior_depth": 8, "decoder_depth": 4, "readout_depth": 4, "n_heads": 8,
                "rotary_pos_emb": True, "use_rmsnorm": True, "decoder_obs": True, "decoder_layout": "interleaved", "reward_head": "binary",
                "delta_mask": "off", "delta_mask_init": 3.0, "delta_mask_hidden": 256}
DEFAULT_LOSS = {"coef_reward": 1.0, "coef_post": 1.0, "coef_prior": 1.0, "coef_unif": 1.0e-3, "coef_align": 0.0, "coef_sep": 0.0, "align_margin": 20.0,
                "unif_sample": 512, "slot_supervision": "lift", "detach_prior_input": True, "detach_prior_target": True,
                "coef_delta": 0.0, "delta_warmup_steps": 5000}
DEFAULT_OPTIM = {"lr": 3.0e-4, "betas": (0.9, 0.95), "weight_decay": 1.0e-2,
                 "clip_grad_norm": 1.0, "amp_dtype": "bfloat16"}


def _plain(node, default):
    merged = dict(default)
    if node is not None:
        merged.update(OmegaConf.to_container(node, resolve=True) if OmegaConf.is_config(node) else dict(node))
    return merged


class SmtPhase1WM(WorldModel):
    def __init__(self, action_dim, image_size, native, encoder_checkpoint, embed_key,
                 embed_stats_zarr, ctx_len=60, fut_len=20, device="cuda:0", name="smt_phase1",
                 expect_step=None, arch=None, loss=None, optim=None,
                 eval_warmup=1, observe_mode="recurrent", window=None, oracle_obs=False):
        # `window` is consumed by the data configs via ${model.window}; the model ignores it
        super().__init__()
        self.name = str(name)
        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.action_dim = int(action_dim)
        self.ctx_len, self.fut_len = int(ctx_len), int(fut_len)
        # eval_warmup 1 is the strict setting: the teacher encodes only the window at t = 0
        # (one real frame after left-padding) and the updaters carry everything after. Raising
        # it makes the test easier; eval_warmup == ctx_len reduces to smt_prelim's untestable
        # configuration, where the onset state was always a full-prefix teacher encoding.
        self.eval_warmup = int(eval_warmup)
        self.observe_mode = str(observe_mode)
        self.encoder_checkpoint = str(encoder_checkpoint)
        self.embed_key = str(embed_key)
        self.loss_cfg = _plain(loss, DEFAULT_LOSS)
        self.optim_cfg = _plain(optim, DEFAULT_OPTIM)
        self._step = 0

        native = OmegaConf.to_container(native, resolve=True) if OmegaConf.is_config(native) else dict(native)
        self.native = load_native_config(native["size"], native.get("overrides"), str(device))
        self.frozen_encoder, _ = build_encoder(self.encoder_checkpoint, native, self.device,
                                               self.image_size, expect_step)

        # the normalisation the cached training embeddings were built with, so observe() emits
        # exactly what the model was trained on
        # ORACLE mode: the "embedding" is a symbolic observation written by
        # scripts/build_oracle_state.py, not an encoder output, so there is no source checkpoint
        # to check against. It carries exactly the pixels' information (verified: a hand-written
        # tracker scores 878/878, and the ball slot never appears during the hold).
        self.oracle_obs = bool(oracle_obs)
        attrs = read_embed_attrs(embed_stats_zarr, self.embed_key,
                                 None if self.oracle_obs else self.encoder_checkpoint)
        self.embed_dim = int(attrs["embed_dim"])
        self.register_buffer("embed_mean", torch.tensor(attrs["mean"], dtype=torch.float32))
        self.register_buffer("embed_std", torch.tensor(np.maximum(attrs["std"], 1e-6), dtype=torch.float32))

        cfg = _plain(arch, DEFAULT_ARCH)
        self.arch = cfg
        shared = dict(d_model=cfg["d_model"], n_heads=cfg["n_heads"], n_registers=cfg["n_registers"],
                      rotary_pos_emb=cfg["rotary_pos_emb"], use_rmsnorm=cfg["use_rmsnorm"])
        # heads=False: phase 1 has no RSSM to regress onto, so to_deter / to_logit -- 6.55M
        # parameters that received no gradient in smt_prelim's tokens mode -- are not built
        self.encoder = TeacherEncoder(embed_dim=self.embed_dim, act_dim=self.action_dim,
                                      depth=cfg["encoder_depth"], heads=False, **shared)
        # DELTA (docs/DELTA.md): the event gate on both updaters; "off" builds no MaskNet at all
        delta = dict(delta_mask=cfg["delta_mask"], delta_mask_init=cfg["delta_mask_init"],
                     delta_mask_hidden=cfg["delta_mask_hidden"])
        self.posterior = PosteriorUpdater(embed_dim=self.embed_dim, act_dim=self.action_dim,
                                          depth=cfg["posterior_depth"], **shared, **delta)
        self.prior = PriorUpdater(act_dim=self.action_dim, depth=cfg["prior_depth"], **shared, **delta)
        self.reward_decoder = RewardDecoder(embed_dim=self.embed_dim, act_dim=self.action_dim,
                                            depth=cfg["decoder_depth"],
                                            use_obs=bool(cfg["decoder_obs"]),
                                            head=str(cfg["reward_head"]),
                                            layout=str(cfg["decoder_layout"]), **shared)
        # stage 2: frozen during stage 1 training and excluded from the optimiser below
        self.readout = StateReadout(embed_dim=self.embed_dim, depth=cfg["readout_depth"],
                                    d_model=cfg["d_model"], n_heads=cfg["n_heads"],
                                    n_registers=cfg["n_registers"],
                                    rotary_pos_emb=cfg["rotary_pos_emb"],
                                    use_rmsnorm=cfg["use_rmsnorm"])
        self.to(self.device)

        self.amp_dtype = getattr(torch, str(self.optim_cfg["amp_dtype"]))
        self._optimizer = torch.optim.AdamW(
            self.trainable(), lr=float(self.optim_cfg["lr"]), betas=tuple(self.optim_cfg["betas"]),
            weight_decay=float(self.optim_cfg["weight_decay"]))

    # ------------------------------------------------------------------ modules

    def modules_by_term(self):
        """The four stage-1 modules, by the name the gradient-attribution test uses."""
        return {"encoder": self.encoder, "posterior": self.posterior, "prior": self.prior,
                "reward_decoder": self.reward_decoder}

    def trainable(self):
        """Stage 1 only: the readout is fitted later, on states these weights have frozen."""
        return [p for m in self.modules_by_term().values() for p in m.parameters()]

    # ------------------------------------------------------------------ embeddings

    @torch.no_grad()
    def _embed(self, image):
        """image (B, T, H, W, 3) uint8 -> normalised embeddings (B, T, D).

        exact_fp32 mirrors the cache: with TF32 on, the same frame embeds ~1.7e-3 away from its
        cached value, which would silently shift eval inputs off the training manifold.
        """
        with exact_fp32():
            embed = self.frozen_encoder({"image": to_f32(image) / 255.0}).float()
        return (embed - self.embed_mean) / self.embed_std

    def _oracle(self, batch):
        """The cached symbolic observation, normalised the same way the embedding cache is."""
        v = batch[f"meta/{self.embed_key}"].float()
        return (v - self.embed_mean) / self.embed_std

    # ------------------------------------------------------------------ WorldModel ABC

    def _losses(self, batch):
        return phase1_losses(self.encoder, self.posterior, self.prior, self.reward_decoder,
                             batch, self.loss_cfg)

    def delta_coef(self):
        """lambda_t: warmed up linearly from 0, so a shut-forever mask cannot win before
        tracking is learned (docs/DELTA.md 3)."""
        lam = float(self.loss_cfg["coef_delta"])
        warm = int(self.loss_cfg["delta_warmup_steps"])
        return lam if warm <= 0 else lam * min(1.0, self._step / warm)

    def _total(self, losses):
        cfg = self.loss_cfg
        total = (cfg["coef_reward"] * losses["reward"] + cfg["coef_post"] * losses["post"]
                 + cfg["coef_prior"] * losses["prior"] + cfg["coef_unif"] * losses["unif"]
                 + cfg["coef_align"] * losses["align"] + cfg["coef_sep"] * losses["sep"])
        lam = self.delta_coef()
        if lam > 0.0:      # skipped entirely when off, so the baseline total is unchanged
            total = total + lam * (losses["delta_l1"] + losses["delta_l1_prior"])
        return total

    def train_step(self, batch):
        self.train()
        self._step += 1
        self._optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, dtype=self.amp_dtype):
            losses, metrics, _ = self._losses(batch)
        total = self._total(losses)
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.trainable(), float(self.optim_cfg["clip_grad_norm"]))
        self._optimizer.step()
        out = {f"loss/{k}": v for k, v in losses.items()}
        out.update(metrics)
        out["opt/loss"] = total
        out["opt/grad_norm"] = grad_norm
        if self.posterior.mask_net is not None:
            out["delta/lambda"] = self.delta_coef()
        return out

    @torch.no_grad()
    def eval_step(self, batch):
        self.eval()
        with autocast(self.device.type, dtype=self.amp_dtype):
            losses, metrics, _ = self._losses(batch)
        out = {f"loss/{k}": v for k, v in losses.items()}
        out.update(metrics)
        out["opt/loss"] = self._total(losses)
        return out

    @torch.no_grad()
    def encode_windows(self, embed, prev_action, is_first):
        """The teacher's K memory tokens for the window ending at every step: (B, T, K, d)."""
        batch_size, steps = embed.shape[0], embed.shape[1]
        e_ctx = sliding_windows(embed, self.ctx_len)
        a_ctx = sliding_windows(prev_action, self.ctx_len)
        mask = window_mask(steps, self.ctx_len, is_first)
        flat = batch_size * steps
        with autocast(self.device.type, dtype=self.amp_dtype):
            z = self.encoder.tokens(e_ctx.reshape(flat, self.ctx_len, -1),
                                    a_ctx.reshape(flat, self.ctx_len, -1),
                                    mask.reshape(flat, self.ctx_len))
        return z.float().view(batch_size, steps, *z.shape[1:])

    @torch.no_grad()
    def observe(self, batch):
        """Latents for every step, causal: the state at t never depends on a frame after t.

        observe_mode:
          recurrent  the teacher encodes the first `eval_warmup` windows and the POSTERIOR is
                     then chained over every remaining real embedding. This is the condition
                     phase 1 exists to test -- with eval_warmup 1 the state crosses ~40 real
                     steps without a single re-encoding.
          window     the teacher re-encodes a fresh window at every step and no updater is used,
                     so memory is capped at ctx_len. The ceiling, and the ablation.

        `embed` is the REAL embedding here, so the pixel head (stage 2b) is fitted on true
        features and `recon` stays the ceiling it was. Imagined embeddings come from the readout.
        """
        self.eval()
        embed = self._oracle(batch) if self.oracle_obs else self._embed(batch["image"])
        prev_action = batch["prev_action"].float()
        is_first = batch["is_first"]
        batch_size, steps = embed.shape[0], embed.shape[1]

        if self.observe_mode == "window":
            z = self.encode_windows(embed, prev_action, is_first)
            return TensorDict({"z": z, "embed": embed}, batch_size=(batch_size, steps))
        if self.observe_mode != "recurrent":
            raise ValueError(f"unknown observe_mode {self.observe_mode!r}")

        warm = max(1, min(self.eval_warmup, steps))
        z_w = self.encode_windows(embed[:, :warm], prev_action[:, :warm], is_first[:, :warm])
        zs, z = [z_w], z_w[:, -1]
        with autocast(self.device.type, dtype=self.amp_dtype):
            for step in range(warm, steps):
                z = self.posterior.step(z, embed[:, step], prev_action[:, step]).float()
                if bool(is_first[:, step].any()):      # a new episode mid-clip restarts the state
                    reset = is_first[:, step].view(-1, 1, 1)
                    z = torch.where(reset, torch.zeros_like(z), z)
                zs.append(z.unsqueeze(1))
        return TensorDict({"z": torch.cat(zs, 1), "embed": embed}, batch_size=(batch_size, steps))

    @torch.no_grad()
    def imagine(self, context, prev_action):
        """Open loop: the PRIOR advances the state on the action alone, and never sees a frame.

        There is no predicted embedding fed back anywhere -- the prior has no embedding input,
        so the imagined trajectory cannot be contaminated by a bad reconstruction. The readout
        turns each state into an embedding only so the pixel decoder has something to render.
        """
        self.eval()
        z = context["z"][:, -1].float()
        zs, embeds = [], []
        with autocast(self.device.type, dtype=self.amp_dtype):
            for i in range(prev_action.shape[1]):
                z = self.prior.step(z, prev_action[:, i].float()).float()
                zs.append(z)
                embeds.append(self.readout(z).float())
        return TensorDict({"z": torch.stack(zs, 1), "embed": torch.stack(embeds, 1)},
                          batch_size=tuple(prev_action.shape[:2]))

    def build_decoder(self):
        return EmbedPixelDecoder(self.native.decoder, self.embed_dim, self.image_size).to(self.device)

    def train_state_dict(self):
        return {"modules": self.state_dict(), "optimizer": self._optimizer.state_dict(),
                "step": self._step}

    def load_train_state_dict(self, state):
        self.load_state_dict(state["modules"])
        self._step = int(state.get("step", 0))
        if "optimizer" in state:
            self._optimizer.load_state_dict(state["optimizer"])
