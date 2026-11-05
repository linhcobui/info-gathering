"""Posterior memory at MOTIONLESS hold frames of the test set, for several stage-1 runs -> npz for t-SNE.

    pixi run python scripts/rest_states_extract.py --out <npz> NAME=<run dir>:<train zarr>:<n trained> ...

Per state: the ball's slot now, the episode's swap count and start slot, and whether the episode's
swap sequence is in THAT run's training set (each run is judged against its own training data).
Plot with scripts/rest_states_tsne.py (system python3: needs scikit-learn).
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from torch.utils.data import default_collate  # noqa: E402

from info_gathering.dataset_memtest1 import memtest1_datasets  # noqa: E402
from info_gathering.dataset_smt_prelim import dense_ball_slot  # noqa: E402
from info_gathering.swap_sequences import sequences  # noqa: E402
from world_model.training.loop import to_device  # noqa: E402
from world_model.utils.checkpoint import CheckpointManager  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="NAME=<run dir>:<training zarr>:<number of trained episodes>")
    ap.add_argument("--zarr", default="data/memtest4_eval.zarr")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    ds, _ = memtest1_datasets(str(ROOT / args.zarr), image_size=64, holdout_episodes=0,
                              meta_keys=("ball_under_target", "ball_slot", "phase"))
    A = ds.arrays
    rl = A.steps - A.meta["pad"].sum(1)
    dense = dense_ball_slot(A.meta["ball_slot"], rl)
    ev = sequences(ROOT / args.zarr)
    out = {"depth": np.array([len(s[1]) for s in ev]), "start": np.array([s[0] for s in ev])}
    names = []
    for i, spec in enumerate(args.runs):
        name, rest = spec.split("=", 1)
        run, trz, ntr = rest.split(":")
        names.append(name)
        trained = set(sequences(ROOT / trz)[:int(ntr)])
        out[f"seen{i}"] = np.array([s in trained for s in ev])
        st = CheckpointManager.load(Path(run) / "checkpoints" / "wm_latest.pt")
        cfg = OmegaConf.create(st["config"])
        tr, _ = instantiate(cfg.data)
        model = instantiate(cfg.model, action_dim=tr.action_dim, device=args.device, _recursive_=False)
        model.load_train_state_dict(st["model"])
        model.requires_grad_(False).eval()
        model.observe_mode = "recurrent"
        Z, S, E = [], [], []
        for e in range(A.num_episodes):
            n = int(rl[e])
            b = to_device(default_collate([ds[e]]), args.device)
            b = {q: v[:, :n] if v.dim() > 1 else v for q, v in b.items()}
            with torch.no_grad():
                emb = model._embed(b["image"])[0]
                z = model.observe(b)["z"][0].float().flatten(1).cpu().numpy()
            still = np.r_[False, (torch.norm(emb[1:] - emb[:-1], dim=-1) < 1e-6).cpu().numpy()]
            m = still & (A.meta["phase"][e, :n] == 1)
            Z.append(z[m]); S.append(dense[e, :n][m]); E.append(np.full(int(m.sum()), e))
        out[f"z{i}"], out[f"s{i}"], out[f"e{i}"] = np.concatenate(Z), np.concatenate(S), np.concatenate(E)
        print(f"{name}: {len(out[f's{i}'])} rest states, {int((~out[f'seen{i}']).sum())} unseen episodes", flush=True)
        del model
        torch.cuda.empty_cache()
    np.savez(args.out, names=np.array(names), **out)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
