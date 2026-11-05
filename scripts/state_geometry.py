"""Per-arm geometry of the memory state at motionless hold frames.

within  = sqrt of the per-slot mean squared deviation from the slot mean (what L_align minimises)
between = the three pairwise distances between slot means (what L_sep hinges at align_margin)
"""
import sys, os, json; sys.path.insert(0, "src")
import numpy as np, torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import default_collate
from info_gathering.dataset_memtest1 import memtest1_datasets
from info_gathering.dataset_smt_prelim import dense_ball_slot
from world_model.training.loop import to_device
from world_model.utils.checkpoint import CheckpointManager

DEV = "cuda:1"
ARMS = json.loads(os.environ["ARMS"])
DATA = os.environ.get("DATA", "data/memtest4_big.zarr")

ds, _ = memtest1_datasets(DATA, image_size=64, holdout_episodes=0,
                          meta_keys=("ball_under_target", "ball_slot", "phase"))
A = ds.arrays; rl = A.steps - A.meta["pad"].sum(1); dn = dense_ball_slot(A.meta["ball_slot"], rl)

print(f"{'arm':<26} {'within':>8} {'between (0-1, 0-2, 1-2)':>26} {'|z|':>8} {'ratio':>7}")
for name, ckpt in ARMS.items():
    st = CheckpointManager.load(ckpt)
    cfg = OmegaConf.create(st["config"])
    tr, _h = instantiate(cfg.data)
    model = instantiate(cfg.model, action_dim=tr.action_dim, device=DEV, _recursive_=False)
    model.load_train_state_dict(st["model"]); model.requires_grad_(False).eval()
    Z, S = [], []
    for e in range(0, 200, 4):
        n = int(rl[e]); b = to_device(default_collate([ds[e]]), DEV)
        b = {q: v[:, :n] if v.dim() > 1 else v for q, v in b.items()}
        emb = model._embed(b["image"])[0]
        still = np.r_[False, (torch.norm(emb[1:] - emb[:-1], dim=-1) < 1e-6).cpu().numpy()]
        model.observe_mode = "recurrent"
        z = model.observe(b)["z"][0].float().flatten(1).cpu().numpy()
        m = still & (A.meta["phase"][e, :n] == 1)
        Z.append(z[m]); S.append(dn[e, :n][m])
    Z = np.concatenate(Z); S = np.concatenate(S)
    mu = np.stack([Z[S == s].mean(0) for s in range(3)])
    within = float(np.sqrt(np.mean([((Z[S == s] - mu[s]) ** 2).sum(1).mean() for s in range(3)])))
    btw = [float(np.linalg.norm(mu[i] - mu[j])) for i in range(3) for j in range(i + 1, 3)]
    nrm = float(np.linalg.norm(Z, axis=1).mean())
    print(f"{name:<26} {within:8.2f}   {btw[0]:7.2f} {btw[1]:7.2f} {btw[2]:7.2f}      "
          f"{nrm:8.2f} {np.mean(btw)/max(within,1e-9):7.2f}")
    del model; torch.cuda.empty_cache()
