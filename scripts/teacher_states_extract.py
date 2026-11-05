"""TEACHER memory z*_t at rest frames of shown-start test episodes, for several stage-1 runs -> npz for t-SNE.

    pixi run python scripts/teacher_states_extract.py --zarr data/memtest4_eval.zarr --out <npz> NAME=<run dir> ...

z*_t = the teacher encoder over the window of frames ending at t (no posterior chaining). Rest frame = a hold frame
identical to the previous frame. Saved in the rest_states_extract.py layout (names, z{j}, s{j} = ball slot now,
e{j} = episode), so scripts/tsne_ball_slot.py plots it. Ball slot: scoring/colour only.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import zarr  # noqa: E402

from train_mcl_prior_phase1 import load_model  # noqa: E402

HOLD = np.array([1., 0.], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="NAME=<stage-1 run dir>")
    ap.add_argument("--zarr", default="data/memtest4_eval.zarr")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device
    g = zarr.open_group(str(ROOT / args.zarr), mode="r")
    ends = g["episode_ends"][:]; starts = np.r_[0, ends[:-1]]
    out, names = {}, []
    for j, spec in enumerate(args.runs):
        name, run = spec.split("=", 1)
        names.append(name)
        m = load_model(f"{run}/checkpoints/wm_latest.pt", dev)
        Z, S, Ep = [], [], []
        for e in range(len(ends)):
            s, n = int(starts[e]), int(ends[e] - starts[e])
            act = g["action"][s:s + n].astype(np.float32)
            PA = np.zeros_like(act); PA[1:] = act[:-1]
            E = (torch.from_numpy(np.asarray(g[m.embed_key][s:s + n], np.float32)).to(dev) - m.embed_mean) / m.embed_std
            f = torch.zeros(1, n, dtype=torch.bool, device=dev); f[0, 0] = True
            with torch.no_grad():
                zt = m.encode_windows(E[None], torch.from_numpy(PA).to(dev)[None], f)[0].float()
            still = np.r_[False, (E[1:] - E[:-1]).abs().amax(-1).cpu().numpy() < 1e-6]
            rest = np.flatnonzero(still & (PA == HOLD).all(1))
            Z.append(zt[rest].flatten(1).cpu().numpy()); S.append(g["ball_slot"][s:s + n][rest]); Ep.append(np.full(len(rest), e))
        out[f"z{j}"], out[f"s{j}"], out[f"e{j}"] = np.concatenate(Z), np.concatenate(S).astype(int), np.concatenate(Ep)
        print(f"{name}: {len(out[f'z{j}'])} rest states from {len(ends)} episodes", flush=True)
        del m; torch.cuda.empty_cache()
    np.savez(ROOT / args.out, names=np.array(names), **out)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
