"""Config loading, output paths and provenance (config + git state written next to every output)."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]


def load_config(path=None, overrides=()):
    cfg = OmegaConf.load(path or ROOT / "configs/segmentation.yaml")
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg


def out_dir(cfg):
    p = Path(cfg.paths.out_dir)
    p = p if p.is_absolute() else ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def git_info():
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:                                    # no git: provenance still has the source hash
            return None
    status = run("status", "--porcelain", "--", "src/seg", "scripts", "configs/segmentation.yaml")
    h = hashlib.sha256()
    for p in sorted((ROOT / "src/seg").glob("*.py")):        # src/seg is untracked, so hash what actually ran
        h.update(p.read_bytes())
    return dict(git_head=run("rev-parse", "HEAD"), seg_files_dirty=bool(status), seg_source_sha256=h.hexdigest()[:16])


def write_meta(path, cfg, **extra):
    """Sidecar <path>.meta.json: when, how (argv), git state, the full config, plus `extra`."""
    meta = dict(written=time.strftime("%Y-%m-%d %H:%M:%S"), argv=sys.argv, **git_info(),
                config=OmegaConf.to_container(cfg, resolve=True), **extra)
    Path(str(path) + ".meta.json").write_text(json.dumps(meta, indent=1, default=str))
