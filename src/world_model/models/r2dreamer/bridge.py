"""The only module that imports from submodules/r2dreamer.

r2dreamer is not an installable package: its modules are top-level names
(dreamer, networks, tools, optim, ...), so the submodule is put on sys.path
here. Load at most one backend that does this per process.
"""
import sys
from pathlib import Path

R2DREAMER_ROOT = Path(__file__).resolve().parents[4] / "submodules" / "r2dreamer"
if not (R2DREAMER_ROOT / "dreamer.py").exists():
    raise ImportError(f"r2dreamer submodule not found at {R2DREAMER_ROOT}; run: git submodule update --init")
if str(R2DREAMER_ROOT) not in sys.path:
    sys.path.insert(0, str(R2DREAMER_ROOT))

import networks  # noqa: E402
from dreamer import Dreamer  # noqa: E402
from tools import to_f32  # noqa: E402

__all__ = ["R2DREAMER_ROOT", "Dreamer", "networks", "to_f32"]
