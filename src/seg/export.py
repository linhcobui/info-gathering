"""Step 6a: candidate_tests.json (sorted by resolvability) and the frame images the VLM-narration stage reads.

frame_paths: frames t_start - margin_before .. t_end (the window and a little lead-in). outcome_frame_paths:
t_end + 1 .. t_end + n_outcome. Both clipped to the episode. Paths are relative to the repo root when the output
directory is inside it.
"""
import os

import numpy as np
from PIL import Image

from .config import ROOT


def _rel(p):
    p = os.path.abspath(p)
    return os.path.relpath(p, ROOT) if p.startswith(str(ROOT) + os.sep) else p


def save_frames(ep, ts, out, fmt):
    d = out / "frames" / ep.episode_id.replace("/", "_")
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for t in ts:
        p = d / f"t{int(t):03d}_raw{int(ep.frame_index[t]):06d}.{fmt}"
        if not p.exists():
            img = ep.frame_rgb(int(t))
            if img is None:
                continue
            Image.fromarray(np.asarray(img, dtype=np.uint8)).save(p)
        paths.append(_rel(p))
    return paths


def sort_key(rec):
    r = rec.get("resolvability")
    return (r is None, -(r if r is not None else 0.0), -rec["onset_score"])


def candidate_tests(records, episodes, cfg, out):
    by_id = {e.episode_id: e for e in episodes}
    recs = sorted(records, key=sort_key)
    cap = cfg.export.max_windows_with_frames
    tests = []
    for rank, r in enumerate(recs):
        ep = by_id[r["episode_id"]]
        win = np.arange(max(0, r["t_start"] - cfg.export.margin_before), r["t_end"] + 1)
        post = np.arange(r["t_end"] + 1, min(ep.T, r["t_end"] + 1 + cfg.export.n_outcome))
        with_frames = cap is None or rank < cap
        tests.append(dict(
            episode_id=r["episode_id"], t_start=r["t_start"], t_end=r["t_end"], onset_score=r["onset_score"],
            resolvability=r["resolvability"], resolvability_pct=r["resolvability_pct"],
            frame_paths=save_frames(ep, win, out, cfg.export.image_format) if with_frames else [],
            outcome_frame_paths=save_frames(ep, post, out, cfg.export.image_format) if with_frames else [],
            episode_return=None, episode_success=None,                      # no reward for now (user, 2026-11-03)
            gap=r["gap"], K_used=r["K_used"], skipped=r["skipped"], n_controls=r["n_controls"],
            wm_train_episode=r["wm_train_episode"], raw_frame_start=int(ep.frame_index[r["t_start"]]),
            raw_frame_end=int(ep.frame_index[r["t_end"]])))
    return tests
