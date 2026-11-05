"""PRELIMINARY: SMT-supervised memory over frozen R2-Dreamer embeddings.

Preliminary v0 of an alternative to the RSSM's BPTT training: a transformer encoder
produces a Markovian state from a short window of frozen frame embeddings, and the
RSSM cell is then trained by supervised one-step regression onto that state rather
than by backpropagation through a whole-episode unroll.
"""
from .model import SmtPrelimWM

__all__ = ["SmtPrelimWM"]
