"""PHASE 1: a plain-vector belief, two one-step updaters, and a reward-only decoder.

The direct response to what smt_prelim measured (docs/SMT_PRELIM.md 6, docs/SMT_PHASE1.md 1):

  * the RSSM half of the state was dead weight -- so the memory here is K rms-normed vectors
    and nothing else;
  * the latent-prediction loss spent 99% of its gradient on cup kinematics that need no
    memory -- so the decoder predicts only the reward, and only during the lift;
  * the belief updater was never actually exercised, because ctx_len 60 meant the teacher
    re-encoded the whole prefix at every eval step -- so eval warms up for ONE step and the
    updaters carry the state from there, observing with the posterior and imagining with the
    prior.

Everything upstream of the memory is reused unchanged: the frozen R2-Dreamer encoder, its
cached embeddings, the anchor dataset, and smt_prelim's blocks / teacher / pixel decoder.
"""
from .model import SmtPhase1WM

__all__ = ["SmtPhase1WM"]
