"""Offline, world-model-only Dreamer."""
import torch

from .bridge import Dreamer


class OfflineDreamer(Dreamer):
    """Dreamer trained on a fixed dataset: world model only.

    Keeps the recipe's world-model losses (dyn/rep KL + the representation loss) and
    drops everything that needs reward or a policy (rew, con, policy, value, repval).
    Representation loss, as in the native recipe:
        rep_loss="r2dreamer"  Barlow term between latent features and encoder embeddings (no decoder)
        rep_loss="dreamer"    DreamerV3: pixel reconstruction through the native decoder
    Dreamer.update() -- LaProp, AGC, GradScaler, LR warmup, torch.compile -- is inherited
    unchanged. The unused actor/critic/reward/cont heads are still built by
    Dreamer.__init__ but never receive gradients, and LaProp and AGC skip them.
    """

    def __init__(self, config, obs_space, act_space):
        if config.rep_loss not in ("r2dreamer", "dreamer"):
            # the other variants' losses are not ported to wm_losses()
            raise ValueError(f"OfflineDreamer supports rep_loss 'r2dreamer' or 'dreamer', got {config.rep_loss!r}")
        super().__init__(config, obs_space, act_space)

    def wm_losses(self, data, initial):
        """World-model losses for one batch; Dreamer._cal_grad up to where the actor-critic starts."""
        # data: dict of (B, T, *), initial: (stoch: (B, S, K), deter: (B, D))
        B, T = data.shape
        # (B, T, E)
        embed = self.encoder(data)
        # (B, T, S, K), (B, T, D), (B, T, S, K)
        post_stoch, post_deter, post_logit = self.rssm.observe(embed, data["action"], initial, data["is_first"])
        # (B, T, S, K)
        _, prior_logit = self.rssm.prior(post_deter)
        dyn_loss, rep_loss = self.rssm.kl_loss(post_logit, prior_logit, self.kl_free)
        losses = {"dyn": torch.mean(dyn_loss), "rep": torch.mean(rep_loss)}

        if self.rep_loss == "dreamer":
            # DreamerV3 reconstruction, verbatim from Dreamer._cal_grad (rep_loss == "dreamer")
            losses.update({key: torch.mean(-dist.log_prob(data[key]))
                           for key, dist in self.decoder(post_stoch, post_deter).items()})
            metrics = {
                "dyn_entropy": torch.mean(self.rssm.get_dist(prior_logit).entropy()),
                "rep_entropy": torch.mean(self.rssm.get_dist(post_logit).entropy()),
            }
            return losses, metrics, (post_stoch, post_deter)

        # Barlow Twins redundancy reduction between latent features and encoder embeddings,
        # verbatim from Dreamer._cal_grad (rep_loss == "r2dreamer")
        # (B, T, F)
        feat = self.rssm.get_feat(post_stoch, post_deter)
        # (B*T, E)
        x1 = self.prj(feat.reshape(B * T, -1))
        x2 = embed.reshape(B * T, -1).detach()  # this detach is important
        x1_norm = (x1 - x1.mean(0)) / (x1.std(0) + 1e-8)
        x2_norm = (x2 - x2.mean(0)) / (x2.std(0) + 1e-8)
        c = torch.mm(x1_norm.T, x2_norm) / (B * T)
        invariance_loss = (torch.diagonal(c) - 1.0).pow(2).sum()
        off_diag_mask = ~torch.eye(x1.shape[-1], dtype=torch.bool, device=x1.device)
        redundancy_loss = c[off_diag_mask].pow(2).sum()
        losses["barlow"] = invariance_loss + self.barlow_lambd * redundancy_loss

        metrics = {
            "dyn_entropy": torch.mean(self.rssm.get_dist(prior_logit).entropy()),
            "rep_entropy": torch.mean(self.rssm.get_dist(post_logit).entropy()),
        }
        return losses, metrics, (post_stoch, post_deter)

    def _cal_grad(self, data, initial):
        losses, metrics, post = self.wm_losses(data, initial)
        total_loss = sum(v * self._loss_scales[k] for k, v in losses.items())
        self._scaler.scale(total_loss).backward()
        metrics.update({f"loss/{name}": loss for name, loss in losses.items()})
        metrics["opt/loss"] = total_loss
        return post, metrics


class BatchAsBuffer:
    """Stands in for r2dreamer's replay Buffer so Dreamer.update() runs unmodified on a DataLoader batch.

    Dreamer.update() calls sample() once, then update() to store the new latents as
    replay context. Every window here starts from the zero state, so nothing is stored.
    """

    def __init__(self, data, initial):
        self._data, self._initial = data, initial

    def sample(self):
        return self._data, None, self._initial

    def update(self, index, stoch, deter):
        pass
