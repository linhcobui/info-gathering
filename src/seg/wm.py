"""The world-model interface the pipeline needs, and the adapter for the frozen DINO-WM checkpoint.

Context = the history_len (H) frames ending at t. Shapes: vis (b, H, P, D), prop (b, H, p), act (b, H, A) where
act[:, i] is the action leaving context frame i, so act[:, -1] = a_t.
"""
import sys

import torch
from einops import rearrange

from .config import ROOT


class WorldModel:
    history_len: int
    device: torch.device

    def predict_next(self, vis, prop, act, return_features=False):
        """Predicted e_{t+1} (b, P, D) float32; with return_features also (b, F) features (or None)."""
        raise NotImplementedError

    def rollout(self, vis, prop, act):
        """Open loop from a real context: act (b, H + n - 1, A) = a_{t-H+1} .. a_{t+n-1}.

        Returns predicted e_{t+1} .. e_{t+n} (b, n, P, D). Default: feed each prediction back as a context frame,
        repeating the last real proprio (for stub models that ignore proprio).
        """
        H = self.history_len
        n = act.shape[1] - H + 1
        vis, out = vis.float(), []
        for i in range(n):
            e = self.predict_next(vis[:, -H:], prop[:, -H:], act[:, i:i + H])
            out.append(e)
            vis = torch.cat([vis, e[:, None]], 1)
            prop = torch.cat([prop, prop[:, -1:]], 1)
        return torch.stack(out, 1)


class DinoWMAdapter(WorldModel):
    """The trained DINO-WM (src/world_model/models/dino_wm), frozen and in eval mode. Never trained or modified."""

    def __init__(self, ckpt_path, device, bf16=True, feature_layer=-1):
        sys.path.insert(0, str(ROOT / "scripts"))
        from omegaconf import OmegaConf
        from train_dino_wm import build_model
        from world_model.utils.checkpoint import CheckpointManager

        state = CheckpointManager.load(str(ckpt_path))
        cfg = OmegaConf.create(state["config"])
        self.device = torch.device(device)
        a_dim, p_dim = len(state["stats"]["action_mean"]), len(state["stats"]["proprio_mean"])
        self.model = build_model(cfg, state["stats"], a_dim, p_dim).to(self.device).eval().requires_grad_(False)
        self.model.load_state_dict(state["model"])
        self.history_len, self.emb_dim = self.model.num_hist, self.model.emb_dim
        self.bf16, self.feature_layer = bf16, feature_layer
        self.step = int(state["step"])

    def _autocast(self):
        return torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.bf16)

    def _predictor(self, z):
        """ViTPredictor.forward, also returning the output of layer feature_layer (b, t, p, dim)."""
        p = self.model.predictor
        t = z.shape[1]
        x = rearrange(z, "b t p d -> b (t p) d")
        n = x.shape[1]
        x = p.emb_dropout(x + p.pos_embedding[:, :n])
        mask = p.mask[:n, :n]
        stop, feat = self.feature_layer % len(p.layers), None
        for i, (attn, ff) in enumerate(p.layers):
            x = attn(x, mask) + x
            x = ff(x) + x
            if i == stop:
                feat = x
        x = p.norm(x)
        return rearrange(x, "b (t p) d -> b t p d", t=t), rearrange(feat, "b (t p) d -> b t p d", t=t)

    @torch.no_grad()
    def predict_next(self, vis, prop, act, return_features=False):
        with self._autocast():
            z = self.model.encode(vis, prop, act)
            out, feat = self._predictor(z)
        e = out[:, -1, :, :self.emb_dim].float()
        return (e, feat[:, -1].float().mean(1)) if return_features else e

    @torch.no_grad()
    def rollout(self, vis, prop, act):
        with self._autocast():
            v, _ = self.model.rollout(vis, prop, act)
        return v[:, self.history_len:].float()

    @torch.no_grad()
    def check_against_model(self, vis, prop, act, atol=1e-3):
        """The re-implemented predictor pass must equal DinoWM.predict (guards against drift in model.py)."""
        with self._autocast():
            ref = self.model.predict(self.model.encode(vis, prop, act))[:, -1, :, :self.emb_dim].float()
        diff = (ref - self.predict_next(vis, prop, act)).abs().max().item()
        if diff > atol:
            raise RuntimeError(f"adapter predictor pass differs from DinoWM.predict by {diff:.2e}")
        return diff
