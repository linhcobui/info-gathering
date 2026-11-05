"""memtest2 datasets.

memtest2 stores the same shape of data as memtest1 -- variable-length episodes in one zarr,
each ending with the lifted cup up and motionless -- so it uses the same loader, which pads to
the longest episode by repeating the final frame and flags the repeats in `meta/pad`.
Only the action width (2 instead of 1) and the meta keys differ, and both come from the store
and the config.
"""
from .dataset_memtest1 import load_memtest1 as load_memtest2, memtest1_datasets as memtest2_datasets

META_KEYS = ("ball_under_target", "ball_slot", "phase")

__all__ = ["load_memtest2", "memtest2_datasets", "META_KEYS"]
