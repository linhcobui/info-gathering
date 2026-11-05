"""PRELIMINARY v0: SMT-supervised, RSSM-shaped memory over frozen R2-Dreamer embeddings.

A transformer encoder turns a short window of frozen frame embeddings into a Markovian state
(deter + categorical stoch); the native RSSM cell is then trained by supervised one-step
regression onto that state, instead of by backpropagation through a whole-episode unroll.
"""
import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from torch.amp import autocast

from world_model.models.base import WorldModel

from ..r2dreamer.native_config import load_native_config
from .blocks import sliding_windows, window_mask
from .bridge import to_f32
from .embed_cache import build_encoder, exact_fp32, read_embed_attrs
from .losses import smt_losses, smt_token_losses
from .pixel_decoder import EmbedPixelDecoder
from .predictor import OneStepPredictor
from .updater import SmtBeliefUpdater
from .teacher import TeacherDecoder, TeacherEncoder

DEFAULT_LOSS = {"coef_dec": 1.0, "coef_dyn": 0.1, "coef_dyn_tokens": 1.0, "coef_kl_dyn": 1.0, "coef_kl_rep": 0.1,
                "coef_unif": 1.0e-3, "gamma": 1.0, "kl_free": 0.0, "unif_sample": 512,
                "reweight": "none", "reweight_warmup": 2000, "reweight_clip": 1.0e9,
                "reweight_floor": 0.0, "coef_ball": 1.0}
DEFAULT_OPTIM = {"lr": 3.0e-4, "betas": (0.9, 0.95), "weight_decay": 1.0e-2,
                 "clip_grad_norm": 1.0, "amp_dtype": "bfloat16"}


def _plain(node, default):
    merged = dict(default)
    if node is not None:
        merged.update(OmegaConf.to_container(node, resolve=True) if OmegaConf.is_config(node) else dict(node))
    return merged


class SmtPrelimWM(WorldModel):
    def __init__(self, action_dim, image_size, native, encoder_checkpoint, embed_key,
                 embed_stats_zarr, ctx_len=10, fut_len=10, device="cuda:0", name="smt_prelim",
                 expect_step=None, teacher=None, loss=None, optim=None,
                 eval_stoch="mode", imagine_mode="markov", observe_mode="recurrent",
                 state_mode="rssm", window=None):
        # `window` is consumed by the data configs via ${model.window}; the model ignores it,
        # as R2DreamerWM does
        super().__init__()
        self.name = str(name)
        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.action_dim = int(action_dim)
        self.ctx_len, self.fut_len = int(ctx_len), int(fut_len)
        self.eval_stoch, self.imagine_mode = str(eval_stoch), str(imagine_mode)
        self.observe_mode = str(observe_mode)
        # tokens = SMT as written (memory is K tokens, updater is a transformer);
        # rssm  = the Dreamer cell, whose only observation channel is a categorical
        self.state_mode = str(state_mode)
        self.encoder_checkpoint = str(encoder_checkpoint)
        self.embed_key = str(embed_key)
        self.loss_cfg = _plain(loss, DEFAULT_LOSS)
        self._step = 0
        self.optim_cfg = _plain(optim, DEFAULT_OPTIM)

        native = OmegaConf.to_container(native, resolve=True) if OmegaConf.is_config(native) else dict(native)
        self.native = load_native_config(native["size"], native.get("overrides"), str(device))
        self.frozen_encoder, _ = build_encoder(self.encoder_checkpoint, native, self.device,
                                               self.image_size, expect_step)

        # the normalisation the cached training embeddings were built with, so observe() emits
        # exactly what the model was trained on
        attrs = read_embed_attrs(embed_stats_zarr, self.embed_key, self.encoder_checkpoint)
        self.embed_dim = int(attrs["embed_dim"])
        self.register_buffer("embed_mean", torch.tensor(attrs["mean"], dtype=torch.float32))
        self.register_buffer("embed_std", torch.tensor(np.maximum(attrs["std"], 1e-6), dtype=torch.float32))

        rssm_cfg = self.native.rssm
        teacher_cfg = _plain(teacher, {"d_model": 256, "n_registers": 10, "encoder_depth": 8,
                                       "decoder_depth": 4, "n_heads": 8, "rotary_pos_emb": True,
                                       "use_rmsnorm": True})
        shared = dict(embed_dim=self.embed_dim, act_dim=self.action_dim,
                      d_model=teacher_cfg["d_model"], n_heads=teacher_cfg["n_heads"],
                      n_registers=teacher_cfg["n_registers"],
                      rotary_pos_emb=teacher_cfg["rotary_pos_emb"],
                      use_rmsnorm=teacher_cfg["use_rmsnorm"])
        self.teacher_enc = TeacherEncoder(depth=teacher_cfg["encoder_depth"], deter=int(rssm_cfg.deter),
                                          stoch=int(rssm_cfg.stoch), discrete=int(rssm_cfg.discrete),
                                          **shared)
        self.teacher_dec = TeacherDecoder(depth=teacher_cfg["decoder_depth"],
                                          feat_dim=int(rssm_cfg.deter) + int(rssm_cfg.stoch) * int(rssm_cfg.discrete),
                                          twin_ball_decoder=bool(teacher_cfg.get("twin_ball_decoder", False)),
                                          **shared)
        self.predictor = OneStepPredictor(rssm_cfg, self.embed_dim, self.action_dim)
        self.updater = SmtBeliefUpdater(embed_dim=self.embed_dim, act_dim=self.action_dim,
                                        d_model=teacher_cfg["d_model"], n_heads=teacher_cfg["n_heads"],
                                        n_registers=teacher_cfg["n_registers"],
                                        depth=teacher_cfg.get("updater_depth", 8),
                                        rotary_pos_emb=teacher_cfg["rotary_pos_emb"],
                                        use_rmsnorm=teacher_cfg["use_rmsnorm"])
        self.to(self.device)

        self.amp_dtype = getattr(torch, str(self.optim_cfg["amp_dtype"]))
        self._optimizer = torch.optim.AdamW(
            self._trainable(),                              # rssm mode excludes the unused prior head
            lr=float(self.optim_cfg["lr"]), betas=tuple(self.optim_cfg["betas"]),
            weight_decay=float(self.optim_cfg["weight_decay"]))

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

    def _cell(self):
        return self.updater if self.state_mode == "tokens" else self.predictor

    def _trainable(self):
        return (list(self.teacher_enc.parameters()) + list(self.teacher_dec.parameters())
                + list(self._cell().parameters()))

    def _losses(self, batch):
        if self.state_mode == "tokens":
            return smt_token_losses(self.teacher_enc, self.teacher_dec, self.updater,
                                    batch, self.loss_cfg, self._step)
        return smt_losses(self.teacher_enc, self.teacher_dec, self.predictor,
                          batch, self.loss_cfg, self._step)

    def _total(self, losses):
        cfg = self.loss_cfg
        if self.state_mode == "tokens":
            return (cfg["coef_dec"] * losses["dec"] + cfg["coef_dyn_tokens"] * losses["dyn"]
                    + cfg["coef_unif"] * losses["unif"] + cfg["coef_ball"] * losses["ball"])
        return (cfg["coef_dec"] * losses["dec"] + cfg["coef_dyn"] * losses["dyn_deter"]
                + cfg["coef_kl_dyn"] * losses["kl_dyn"] + cfg["coef_kl_rep"] * losses["kl_rep"]
                + cfg["coef_unif"] * losses["unif"])

    # ------------------------------------------------------------------ WorldModel ABC

    def train_step(self, batch):
        self.train()
        self._step += 1
        self._optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, dtype=self.amp_dtype):
            losses, metrics, _ = self._losses(batch)
        total = self._total(losses)
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self._trainable(), float(self.optim_cfg["clip_grad_norm"]))
        self._optimizer.step()
        out = {f"loss/{k}": v for k, v in losses.items()}
        out.update(metrics)
        out["opt/loss"] = total
        out["opt/grad_norm"] = grad_norm
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

    def _stoch(self, logit):
        dist = self.predictor.rssm.get_dist(logit)
        return dist.mode if self.eval_stoch == "mode" else dist.rsample()

    @torch.no_grad()
    def _encode_windows(self, embed, prev_action, is_first):
        """Teacher-encode the window ending at every step of a (B, T, ...) slice."""
        batch_size, steps = embed.shape[0], embed.shape[1]
        e_ctx = sliding_windows(embed, self.ctx_len)
        a_ctx = sliding_windows(prev_action, self.ctx_len)
        mask = window_mask(steps, self.ctx_len, is_first)
        flat = batch_size * steps
        with autocast(self.device.type, dtype=self.amp_dtype):
            deter, logit, _ = self.teacher_enc(e_ctx.reshape(flat, self.ctx_len, -1),
                                               a_ctx.reshape(flat, self.ctx_len, -1),
                                               mask.reshape(flat, self.ctx_len))
        return (deter.float().view(batch_size, steps, -1),
                logit.float().view(batch_size, steps, self.teacher_enc.stoch, self.teacher_enc.discrete))

    @torch.no_grad()
    def _encode_window_tokens(self, embed, prev_action, is_first):
        """The teacher's K memory tokens for the window ending at every step: (B, T, K, d)."""
        batch_size, steps = embed.shape[0], embed.shape[1]
        e_ctx = sliding_windows(embed, self.ctx_len)
        a_ctx = sliding_windows(prev_action, self.ctx_len)
        mask = window_mask(steps, self.ctx_len, is_first)
        flat = batch_size * steps
        with autocast(self.device.type, dtype=self.amp_dtype):
            z = self.teacher_enc.tokens(e_ctx.reshape(flat, self.ctx_len, -1),
                                        a_ctx.reshape(flat, self.ctx_len, -1),
                                        mask.reshape(flat, self.ctx_len))
        return z.float().view(batch_size, steps, *z.shape[1:])

    @torch.no_grad()
    def _observe_tokens(self, embed, prev_action, is_first):
        """SMT's read_with_teacher_rnn: warm up with the teacher, then update the belief."""
        batch_size, steps = embed.shape[0], embed.shape[1]
        if self.observe_mode == "window" or steps <= self.ctx_len:
            z = self._encode_window_tokens(embed, prev_action, is_first)
            return TensorDict({"z": z, "embed": embed}, batch_size=(batch_size, steps))

        warm = self.ctx_len
        z_w = self._encode_window_tokens(embed[:, :warm], prev_action[:, :warm], is_first[:, :warm])
        zs, z = [z_w], z_w[:, -1]
        with autocast(self.device.type, dtype=self.amp_dtype):
            for step in range(warm, steps):
                z = self.updater.step(z, embed[:, step], prev_action[:, step]).float()
                if bool(is_first[:, step].any()):
                    reset = is_first[:, step].view(-1, 1, 1)
                    z = torch.where(reset, torch.zeros_like(z), z)
                zs.append(z.unsqueeze(1))
        return TensorDict({"z": torch.cat(zs, 1), "embed": embed}, batch_size=(batch_size, steps))

    @torch.no_grad()
    def observe(self, batch):
        """Latents for every step, causal: the state at t never depends on a frame after t.

        observe_mode:
          recurrent  the teacher encodes the first ctx_len steps, then the one-step cell is
                     unrolled over the remaining real embeddings -- SMT's read_with_teacher_rnn
                     (smt/src/model.py:415-433). Memory can outlast the encoder's window,
                     which is the entire point of training the cell.
          window     the teacher re-encodes a fresh window at every step and the cell is never
                     used, so memory is capped at ctx_len. Kept as the ablation.
        """
        self.eval()
        embed = self._embed(batch["image"])
        prev_action = batch["prev_action"].float()
        is_first = batch["is_first"]
        batch_size, steps = embed.shape[0], embed.shape[1]

        if self.state_mode == "tokens":
            return self._observe_tokens(embed, prev_action, is_first)

        if self.observe_mode == "window" or steps <= self.ctx_len:
            deter, logit = self._encode_windows(embed, prev_action, is_first)
            return TensorDict({"deter": deter, "logit": logit, "stoch": self._stoch(logit),
                               "embed": embed}, batch_size=(batch_size, steps))

        warm = self.ctx_len
        deter_w, logit_w = self._encode_windows(embed[:, :warm], prev_action[:, :warm], is_first[:, :warm])
        deters, logits = [deter_w], [logit_w]
        deter, logit = deter_w[:, -1], logit_w[:, -1]
        stoch = self._stoch(logit)
        with autocast(self.device.type, dtype=self.amp_dtype):
            for t in range(warm, steps):
                _, deter, logit = self.predictor.step(stoch, deter, prev_action[:, t], embed[:, t])
                deter, logit = deter.float(), logit.float()
                if bool(is_first[:, t].any()):                # a new episode mid-clip restarts the state
                    reset = is_first[:, t].view(-1, *([1] * (deter.dim() - 1)))
                    deter = torch.where(reset, torch.zeros_like(deter), deter)
                    logit = torch.where(reset.view(-1, 1, 1), torch.zeros_like(logit), logit)
                stoch = self._stoch(logit)
                deters.append(deter.unsqueeze(1))
                logits.append(logit.unsqueeze(1))
        deter, logit = torch.cat(deters, 1), torch.cat(logits, 1)
        return TensorDict({"deter": deter, "logit": logit, "stoch": self._stoch(logit),
                           "embed": embed}, batch_size=(batch_size, steps))

    @torch.no_grad()
    def imagine(self, context, prev_action):
        """Open loop: after the context, no real frame or embedding is ever used again.

        Per step the decoder predicts the next embedding from the current state and the action
        taken, then the one-step cell advances the state on that predicted embedding -- the same
        (state, action, next embedding) triple the dynamics loss is trained on.
        """
        self.eval()
        if self.state_mode == "tokens":
            z, embed = context["z"][:, -1].float(), context["embed"][:, -1].float()
            zs, embeds = [], []
            with autocast(self.device.type, dtype=self.amp_dtype):
                for i in range(prev_action.shape[1]):
                    action = prev_action[:, i].float()
                    embed = self.teacher_dec.decode_step_tokens(z, embed, action).float()
                    z = self.updater.step(z, embed, action).float()
                    zs.append(z)
                    embeds.append(embed)
            return TensorDict({"z": torch.stack(zs, 1), "embed": torch.stack(embeds, 1)},
                              batch_size=tuple(prev_action.shape[:2]))

        deter, logit = context["deter"][:, -1].float(), context["logit"][:, -1].float()
        stoch, embed = context["stoch"][:, -1].float(), context["embed"][:, -1].float()
        deters, logits, stochs, embeds = [], [], [], []
        with autocast(self.device.type, dtype=self.amp_dtype):
            for i in range(prev_action.shape[1]):
                action = prev_action[:, i].float()
                embed = self.teacher_dec.decode_step(deter, logit, embed, action).float()
                # obs_step samples the posterior with gumbel_softmax (rssm.py:177), so the
                # stoch it returns is a draw. Carry the mode forward instead, or the whole
                # rollout is stochastic and eval F1 wobbles run to run.
                _, deter, logit = self.predictor.step(stoch, deter, action, embed)
                deter, logit = deter.float(), logit.float()
                stoch = self._stoch(logit)
                deters.append(deter)
                logits.append(logit)
                stochs.append(stoch)
                embeds.append(embed)
        return TensorDict({"deter": torch.stack(deters, 1), "logit": torch.stack(logits, 1),
                           "stoch": torch.stack(stochs, 1), "embed": torch.stack(embeds, 1)},
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
