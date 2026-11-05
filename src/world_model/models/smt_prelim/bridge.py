"""Native r2dreamer modules for the smt_prelim backend.

Imports the r2dreamer bridge purely for its sys.path side effect, then pulls in the
extra native modules this backend needs (the RSSM cell and the one-hot distribution).
That insert is idempotent, so there is still exactly one path-hacking backend per
process -- see world_model/models/r2dreamer/bridge.py.
"""
from ..r2dreamer.bridge import R2DREAMER_ROOT, networks, to_f32  # noqa: F401

import distributions as dists  # noqa: E402
from rssm import RSSM  # noqa: E402

__all__ = ["R2DREAMER_ROOT", "RSSM", "dists", "networks", "to_f32"]
