"""Build the stores docs/TWIN_FINETUNE.md needs, from existing renders. Nothing is re-rendered.

    data/memtest4_s3.zarr     STAGE 1: memtest4_big's episodes with <= 3 swaps, order kept
                              (its last 25 are the anchor holdout, as before)
    data/memtest4_ft.zarr     FINETUNE pool: memtest4_ft_raw minus every 4-swap episode whose
                              sequence is in the held-out half H. Used without labels.
    data/memtest4_s3ft.zarr   CONTROL C1 ("is it just more data?"): S1's training episodes, then
                              all of FT, then S1's 25 holdout episodes last -- so the anchor holdout
                              is the same 25 episodes as S1's
    data/memtest4_ft_split.json   H, provenance, and the coverage this produces on the test sets

H, "incomplete 4 on permutations": a seeded random HALF of the 243 possible 4-swap sequences,
stratified so that half of the 4-swap sequences that occur in data/memtest4_eval.zarr fall in it.
Test 4-swap episodes then split into "sequence seen in finetuning" and "never seen", in about equal
numbers, and 5 swaps (data/memtest4_eval5.zarr) stays pure extrapolation.

Embeddings are NOT copied: run cache-embeddings-prelim on the new stores afterwards, so each gets
its own mean/std and provenance tag.

    pixi run python scripts/build_twin_ft_data.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import zarr  # noqa: E402

from info_gathering.swap_sequences import all_sequences, final_slot, sequences  # noqa: E402

BIG, FT_RAW = "data/memtest4_big.zarr", "data/memtest4_ft_raw.zarr"
EVAL, EVAL5 = "data/memtest4_eval.zarr", "data/memtest4_eval5.zarr"
S1, FT, S1FT = "data/memtest4_s3.zarr", "data/memtest4_ft.zarr", "data/memtest4_s3ft.zarr"
SPLIT = "data/memtest4_ft_split.json"
HOLDOUT = 25          # the anchor datasets' holdout_episodes
MAX_S1_SWAPS = 3
SEED = 0

PER_FRAME = {"rgb": "uint8", "action": "uint8", "phase": "uint8", "ball_slot": "uint8",
             "ball_under_target": "uint8", "cup_z_max": "float32"}
PER_EPISODE = {"episode_seeds": "int64", "episode_ball_start": "uint8", "episode_swapped": "uint8",
               "episode_target": "uint8", "episode_hold": "int64"}
COPY_ATTRS = ("fps", "resolution", "phases")


def write_store(dst, parts, note):
    """Concatenate whole episodes from (src, [episode indices]) parts into a fresh store.

    Same array names, dtypes and chunking as memtest4.Memtest4Recorder, so every loader reads it
    unchanged. Embedding caches are deliberately left behind.
    """
    frames = {k: [] for k in PER_FRAME}
    per_ep = {k: [] for k in PER_EPISODE}
    lengths, provenance, configs, attrs = [], [], [], None
    for src, episodes in parts:
        z = zarr.open_group(str(ROOT / src), mode="r")
        ends = z["episode_ends"][:]
        starts = np.concatenate([[0], ends[:-1]])
        arrays = {k: z[k][:] for k in PER_FRAME}
        episode_arrays = {k: z[k][:] for k in PER_EPISODE}
        for e in episodes:
            s, t = int(starts[e]), int(ends[e])
            for k in PER_FRAME:
                frames[k].append(arrays[k][s:t])
            for k in PER_EPISODE:
                per_ep[k].append(episode_arrays[k][e])
            lengths.append(t - s)
            provenance.append([src, int(e)])
        configs += list(z.attrs.get("configs", [z.attrs["config"]]))
        if attrs is None:
            attrs = {k: z.attrs[k] for k in COPY_ATTRS}
            attrs.update({"config": z.attrs["config"], "master_seed": z.attrs["master_seed"]})
    out = zarr.open_group(str(ROOT / dst), mode="w")
    res = int(attrs["resolution"])
    chunks = {"rgb": (512, res, res, 3), "action": (4096, 2)}
    for k, dtype in PER_FRAME.items():
        data = np.concatenate(frames[k]).astype(dtype)
        out.create_dataset(k, data=data, chunks=chunks.get(k, (4096,)), dtype=dtype)
    out.create_dataset("episode_ends", data=np.cumsum(lengths).astype("int64"))
    for k, dtype in PER_EPISODE.items():
        out.create_dataset(k, data=np.array(per_ep[k], dtype=dtype))
    # every distinct generator config, so swap_sequences.replay can rebuild any episode
    attrs.update({"configs": list(dict.fromkeys(configs)), "n_episodes": len(lengths),
                  "subset_note": note, "provenance": provenance})
    out.attrs.update(attrs)
    return len(lengths), int(np.sum(lengths))


def check_copy(dst, parts):
    """Every copied episode is byte-identical to its source, frames and labels."""
    z = zarr.open_group(str(ROOT / dst), mode="r")
    ends = z["episode_ends"][:]
    starts = np.concatenate([[0], ends[:-1]])
    i = 0
    for src, episodes in parts:
        zs = zarr.open_group(str(ROOT / src), mode="r")
        se = zs["episode_ends"][:]
        ss = np.concatenate([[0], se[:-1]])
        for e in episodes:
            a, b = slice(int(starts[i]), int(ends[i])), slice(int(ss[e]), int(se[e]))
            for k in ("action", "phase", "ball_slot", "ball_under_target", "cup_z_max"):
                assert np.array_equal(z[k][a], zs[k][b]), f"{dst} episode {i}: {k} differs from {src}[{e}]"
            for k in PER_EPISODE:
                assert z[k][i] == zs[k][e], f"{dst} episode {i}: {k} differs"
            i += 1
    # rgb: a strided sample of episodes (the full copy is ~0.5 GB per store)
    rgb_d, i = z["rgb"], 0
    for src, episodes in parts:
        zs = zarr.open_group(str(ROOT / src), mode="r")
        se = zs["episode_ends"][:]
        ss = np.concatenate([[0], se[:-1]])
        for j, e in enumerate(episodes):
            if j % 7 == 0:
                assert np.array_equal(rgb_d[int(starts[i]):int(ends[i])], zs["rgb"][int(ss[e]):int(se[e])]), \
                    f"{dst} episode {i}: rgb differs"
            i += 1
    assert i == len(ends)


def main():
    rng = np.random.default_rng(SEED)
    seq = {k: sequences(ROOT / k) for k in (BIG, FT_RAW, EVAL, EVAL5)}

    # ---- S1: the existing 0-4 training store, minus its 4-swap episodes
    s1_eps = [e for e, s in enumerate(seq[BIG]) if len(s[1]) <= MAX_S1_SWAPS]
    s1_train, s1_hold = s1_eps[:-HOLDOUT], s1_eps[-HOLDOUT:]

    # ---- H: half of all 4-swap sequences, half of the eval set's 4-swap sequences inside it
    space = all_sequences(4)
    in_eval = sorted({s for s in seq[EVAL] if len(s[1]) == 4})
    rest = sorted(set(space) - set(in_eval))
    pick_eval = [in_eval[i] for i in rng.permutation(len(in_eval))[:len(in_eval) // 2]]
    pick_rest = [rest[i] for i in rng.permutation(len(rest))[:len(rest) // 2]]
    held = set(pick_eval) | set(pick_rest)

    # ---- FT: every raw episode except 4-swap ones in H
    ft_eps = [e for e, s in enumerate(seq[FT_RAW]) if not (len(s[1]) == 4 and s in held)]
    assert not any(seq[FT_RAW][e] in held for e in ft_eps)

    built = {}
    built[S1] = write_store(S1, [(BIG, s1_eps)], f"memtest4_big episodes with <= {MAX_S1_SWAPS} swaps")
    built[FT] = write_store(FT, [(FT_RAW, ft_eps)], "memtest4_ft_raw minus held-out 4-swap sequences")
    c1_parts = [(BIG, s1_train), (FT_RAW, ft_eps), (BIG, s1_hold)]
    built[S1FT] = write_store(S1FT, c1_parts, "C1: S1 train + FT + S1 holdout (last 25)")
    for dst, parts in ((S1, [(BIG, s1_eps)]), (FT, [(FT_RAW, ft_eps)]), (S1FT, c1_parts)):
        check_copy(dst, parts)
    # the new stores must replay to the same sequences they were selected for
    assert sequences(ROOT / S1) == [seq[BIG][e] for e in s1_eps]
    assert sequences(ROOT / FT) == [seq[FT_RAW][e] for e in ft_eps]
    assert sequences(ROOT / S1FT) == ([seq[BIG][e] for e in s1_train] + [seq[FT_RAW][e] for e in ft_eps]
                                      + [seq[BIG][e] for e in s1_hold])

    # ---- coverage report
    seen_s1 = {seq[BIG][e] for e in s1_train}
    seen_ft = {seq[FT_RAW][e] for e in ft_eps}
    report = {"seed": SEED, "held_out_4swap": sorted([s[0], [list(p) for p in s[1]]] for s in held),
              "n_held_out": len(held), "n_space": len(space),
              "s1_episodes_from_big": s1_eps, "ft_episodes_from_raw": ft_eps,
              "stores": {k: {"episodes": n, "frames": f} for k, (n, f) in built.items()}}
    lines = []
    for name, store in (("S1", [seq[BIG][e] for e in s1_eps]), ("FT", [seq[FT_RAW][e] for e in ft_eps])):
        depth = np.array([len(s[1]) for s in store])
        lines.append(f"{name}: {len(store)} episodes, by depth {np.bincount(depth).tolist()}, "
                     f"distinct 4-swap sequences {len({s for s in store if len(s[1]) == 4})}")
    for tag, store in (("eval", seq[EVAL]), ("eval5", seq[EVAL5])):
        rows = {}
        for k in range(6):
            idx = [i for i, s in enumerate(store) if len(s[1]) == k]
            if not idx:
                continue
            s1 = sum(store[i] in seen_s1 for i in idx)
            ft_only = sum(store[i] in seen_ft and store[i] not in seen_s1 for i in idx)
            unseen = sum(store[i] not in seen_s1 and store[i] not in seen_ft for i in idx)
            in_h = sum(store[i] in held for i in idx)
            pos = {"seen": [final_slot(store[i]) == 0 for i in idx if store[i] in seen_s1 | seen_ft],
                   "unseen": [final_slot(store[i]) == 0 for i in idx if store[i] not in seen_s1 | seen_ft]}
            rows[k] = {"n": len(idx), "seen_s1": s1, "seen_ft_only": ft_only, "unseen": unseen,
                       "in_H": in_h, "pos_rate_seen": float(np.mean(pos["seen"])) if pos["seen"] else None,
                       "pos_rate_unseen": float(np.mean(pos["unseen"])) if pos["unseen"] else None}
            lines.append(f"{tag} depth {k}: n {len(idx):3d} | seen in S1 {s1:3d} | seen only in FT {ft_only:3d}"
                         f" | unseen {unseen:3d} (of which held out {in_h}) | positive rate seen "
                         f"{rows[k]['pos_rate_seen'] if rows[k]['pos_rate_seen'] is None else round(rows[k]['pos_rate_seen'], 3)}"
                         f" / unseen {rows[k]['pos_rate_unseen'] if rows[k]['pos_rate_unseen'] is None else round(rows[k]['pos_rate_unseen'], 3)}")
        report[f"coverage_{tag}"] = rows
    report["summary"] = lines
    (ROOT / SPLIT).write_text(json.dumps(report, indent=1))
    for k, (n, f) in built.items():
        print(f"{k}: {n} episodes, {f} frames")
    print("\n".join(lines))
    print(f"-> {SPLIT}")


if __name__ == "__main__":
    main()
