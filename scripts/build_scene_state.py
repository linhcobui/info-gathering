"""Scene-state observation: the cups' actual xyz, with cup IDENTITY hidden.

    OBSERVABLE (valid every frame)
      [0:3]    phase one-hot
      [3:12]   three cup positions, SORTED BY X        <- identity is never handed over; the
                                                          model must track cups by continuity,
                                                          exactly as a viewer does
      [12]     ball_shown, taken from the RENDER via ball_detector (not from any rule of mine)
    NON-OBSERVABLE (zero unless ball_shown)
      [13:16]  ball xyz

Poses are reconstructed exactly from the stored episode seeds (verified: 40/40 episodes
reproduce ball_slot and cup_z_max). No swap-pair channel and no swap-progress channel: unlike
the earlier hand-built oracle, the swap is implicit in cup motion, so the boundary frames where
the cups are stationary carry no swap information -- as in the pixels.

The earlier oracle set ball_shown from (phase==show AND ball_under_target), which fired at the
lift ONSET frame while the cup was still down, handing the probe its answer. Reading the render
makes that class of mistake impossible.
"""
import sys, json; sys.path.insert(0, "src")
import numpy as np, torch, zarr
from info_gathering.memtest4 import sample_episode
from info_gathering.ball_detector import ball_pixels
D = 16

def build(zp):
    g = zarr.open_group(zp, mode="a")
    cfg = json.loads(g.attrs["config"])
    ends = g["episode_ends"][:]; st = np.r_[0, ends[:-1]]
    seeds, bstart = g["episode_seeds"][:], g["episode_ball_start"][:]
    nsw, ph = g["episode_swapped"][:], g["phase"][:]
    hold = g["episode_hold"][:]
    bs, cz = g["ball_slot"][:], g["cup_z_max"][:]
    out = np.zeros((len(ph), D), dtype=np.float32)
    bad = 0
    for e in range(len(ends)):
        a, b = int(st[e]), int(ends[e])
        # memtest4_big samples hold_frames inside sample_episode; the eval sets pass it in,
        # which shifts the RNG stream. Try both and keep whichever reproduces ball_slot exactly.
        ep = None
        for args in ((int(bstart[e]), int(nsw[e])), (int(bstart[e]), int(nsw[e]), int(hold[e]))):
            try:
                cand = sample_episode(cfg, np.random.default_rng(int(seeds[e])), *args)
            except Exception:
                continue
            if len(cand["ball_slot"]) == b - a and np.array_equal(cand["ball_slot"], bs[a:b]):
                ep = cand; break
        if ep is None:
            bad += 1; continue
        T = b - a; v = out[a:b]
        v[np.arange(T), ph[a:b]] = 1.0
        cup = ep["cup_xyz"]                                   # (T, 3, 3), indexed by IDENTITY
        order = np.argsort(cup[:, :, 0], axis=1)              # sort by x -> identity destroyed
        v[:, 3:12] = np.take_along_axis(cup, order[:, :, None], axis=1).reshape(T, 9)
        rgb = torch.from_numpy(g["rgb"][a:b].astype(np.float32) / 255.0)
        shown = (ball_pixels(rgb) >= 30).numpy()              # straight from the render
        v[:, 12] = shown.astype(np.float32)
        v[shown, 13:16] = ep["ball_xyz"][shown]
    key = "embed_scene_state"
    if key in g: del g[key]
    g.create_dataset(key, data=out, chunks=(4096, D), dtype="float32")
    g.attrs[key] = {"embed_dim": D, "source_checkpoint": "SCENE-STATE (no encoder)",
                    "mean": out.mean(0).tolist(), "std": (out.std(0) + 1e-3).tolist(),
                    "note": "cup xyz sorted by x (identity hidden) + render-derived ball visibility"}
    # leak check: the ball must NOT be visible at the lift onset, where the cup is still down
    leak = 0
    for e in range(len(ends)):
        a, b = int(st[e]), int(ends[e]); p = ph[a:b]
        onset = int(np.flatnonzero(p == 2)[0])
        leak += int(out[a + onset, 12] > 0)
    print(f"{zp}: {out.shape}, {bad} episodes failed to reconstruct, "
          f"ball visible AT THE ONSET in {leak}/{len(ends)} episodes")
    return out

for zp in ("data/memtest4_big.zarr", "data/memtest4_eval.zarr", "data/memtest4_eval5.zarr"):
    build(zp)
