"""Write an ORACLE STATE observation into the memtest4 zarrs, in place of pixel embeddings.

The channels are ordered OBSERVABLE -> NON-OBSERVABLE, and the split is explicit rather than
folded together:

  ---- observable, valid at every frame ----------------------------------------------
    [0:3]   phase one-hot                 reveal / hold / show
    [3]     cup_z_max, scaled             how high the cups are
    [4:8]   swapping-pair one-hot         none, (0,1), (0,2), (1,2)   -- you SEE which two move
    [8]     progress within the swap      0..1
    [9]     ball_shown flag               is the scene revealing a ball RIGHT NOW
  ---- non-observable: the hidden variable, gated by [9] -------------------------------
    [10:13] ball slot one-hot             ZERO unless ball_shown; never nonzero during the hold

Ball POSITION is never observable while the cups are down: dims 10:13 are identically zero for
every hold frame. They are nonzero only where the rendered scene actually shows a ball -- during
the reveal (cups up), and at the lift if the ball is under the cup that rose. Visibility [9] is
an observable fact about the scene; the slot [10:13] is the hidden variable it occasionally
exposes. This is a perfect, noiseless observation -- not privileged state.

The swapped pair is observable (identical cups visibly exchange places); it is computed here from
the dense slot trajectory only because that is the convenient way to recover an observable fact.
"""
import sys; sys.path.insert(0, "src")
import numpy as np, torch, zarr
from info_gathering.ball_detector import ball_pixels
from info_gathering.dataset_smt_prelim import dense_ball_slot
D = 13
PAIRS = {(0, 1): 1, (0, 2): 2, (1, 2): 3}

def build(zp):
    g = zarr.open_group(zp, mode="a")
    ends = g["episode_ends"][:]; starts = np.r_[0, ends[:-1]]
    ph = g["phase"][:]; bs = g["ball_slot"][:]; but = g["ball_under_target"][:]
    cz = g["cup_z_max"][:].astype(np.float32); sw = g["episode_swapped"][:]
    L = np.diff(np.r_[0, ends]); n = len(ends)
    BS = np.zeros((n, L.max()), dtype=np.int64)
    for e in range(n): BS[e, :L[e]] = bs[starts[e]:ends[e]]
    dn = dense_ball_slot(BS, L)
    out = np.zeros((len(ph), D), dtype=np.float32)
    czs = cz / max(float(cz.max()), 1e-6)
    for e in range(n):
        a, b = int(starts[e]), int(ends[e]); p = ph[a:b]
        h0 = int(np.flatnonzero(p == 1)[0]); k = int(sw[e]); lift = int(np.flatnonzero(p == 2)[0])
        v = out[a:b]
        v[np.arange(b - a), p] = 1.0                                  # phase one-hot
        v[:, 3] = czs[a:b]
        v[:, 4] = 1.0                                                 # default: no swap
        for j in range(k):
            s0 = h0 + 4 + 6 * j; s1 = s0 + 6                          # this swap's frames
            before = int(dn[e, s0]); after = int(dn[e, min(s1, b - a - 1)])
            pair = tuple(sorted((before, after))) if before != after else tuple(sorted({0,1,2} - {before}))
            v[s0:s1, 4] = 0.0
            v[s0:s1, 4 + PAIRS[pair]] = 1.0
            v[s0:s1, 8] = np.linspace(0, 1, s1 - s0, dtype=np.float32)
        # Visibility comes from the RENDER, never from a rule. The previous version used
        # (phase == show AND ball_under_target), which is TRUE at the lift ONSET -- where the cup
        # has not begun to rise (cup_z equals the previous frame) and the render holds zero ball
        # pixels. The ball actually appears at show-offset 2, unanimously across episodes. Since
        # the onset probe reads the state AT the onset, and that state has consumed the onset
        # frame, `shown[onset]` was identically equal to the label. Measured: 113/315 eval
        # episodes, exactly the ball_under_target ones.
        rgb = torch.from_numpy(g["rgb"][a:b].astype(np.float32) / 255.0)
        shown = (ball_pixels(rgb) >= 30).numpy()
        v[:, 9] = shown.astype(np.float32)                            # observable: "a ball is visible"
        # non-observable: the slot, gated to the frames where the scene exposes it
        idx = np.flatnonzero(shown)
        v[idx, 10 + dn[e, :b - a][idx]] = 1.0
    key = "embed_oracle_state"
    if key in g: del g[key]
    g.create_dataset(key, data=out, chunks=(4096, D), dtype="float32")
    g.attrs[key] = {"embed_dim": D, "source_checkpoint": "ORACLE-STATE (no encoder)",
                    "mean": out.mean(0).tolist(), "std": (out.std(0) + 1e-6).tolist(),
                    "note": "symbolic observation with the same information content as the pixels"}
    leak = 0
    for e in range(n):
        a, b = int(starts[e]), int(ends[e]); p = ph[a:b]
        onset = int(np.flatnonzero(p == 2)[0])
        leak += int(out[a + onset, 9] > 0)
    print(f"{zp}: wrote {key} {out.shape}; ball visible AT THE ONSET in {leak}/{n} episodes")
    return out

for zp in ("data/memtest4_big.zarr", "data/memtest4_eval.zarr", "data/memtest4_eval5.zarr"):
    build(zp)
