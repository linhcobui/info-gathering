# New machine: setup checklist (handoff, 2026-11-04)

Written by the Claude agent on the old machine for the Claude agent on the new one. The user has already cloned the repo (with submodules) and has `pixi` working. Work through the list in order. Ask the user before anything that launches training or long generation (see "Working rules").

**What came over and what didn't.** Only the git repo moved. The following stayed on the old machine and do **not** exist here:
- `data/`: all datasets and DINO caches;
- `logdir/`: every checkpoint, including the trained DINO-WM;
- `outputs/`: the segmentation results;
- `figures/`.

The user wants the datasets regenerated here. Any path in older docs or notes pointing into those folders refers to the old machine. Start with [docs/README.md](README.md), the index of all design and results docs.

## 1. Install the memory notes

The old agent's memory notes (27 project and preference notes, plus a note about this move) travel as a tarball the user copies over: `claude-memory-export.tar.gz`. It is not in git, because the repo is public.

1. Ask the user where they put the tarball, then unpack it: `tar -xzf claude-memory-export.tar.gz` gives a `claude-memory-export/` folder.
2. Copy its `*.md` files into this machine's memory directory for this project: `~/.claude/projects/<slug>/memory/`. Here `<slug>` is the absolute repo path with every `/` replaced by `-`; for example `/home/linhco/info-gathering` → `-home-linhco-info-gathering`. If your system prompt names a memory directory, use that one.
3. If the directory already has a `MEMORY.md`, merge the index lines rather than overwriting it.
4. Read `MEMORY.md`. The notes `new-machine-2026-11` and `repo-layout-2026-11` describe this move.

The RAM-limit note was deliberately left out: that limit belonged to the old shared machine.

## 2. Check the machine

Run `nvidia-smi` (GPU count and model), `free -g` and `df -h .`, and tell the user what you find.

- **Disk:** regenerating everything needs about 100 GB, of which the full DINO cache of the 400-episode store is 78 GB.
- **CUDA:** torch is pinned to 2.8.0 from the CUDA 12.8 wheel index (`pixi.toml`). Check that it sees the GPU:
  ```
  pixi run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  ```

## 3. Remove hard-coded GPU choices

The old machine had two 4090s shared with other users. The code defaults to `cuda:1` in these places:

| Where | What to change |
|---|---|
| `argparse` defaults (`--device`, default `"cuda:1"`) in `scripts/`: `belief_midframe_ogbar.py`, `branch_eval_ogbar.py`, `compare_from_cut.py`, `delta_extract.py`, `dino_wm_videos.py`, `dino_wm_watch_imagine.py`, `finetune_bptt_phase1.py`, `finetune_onestep_phase1.py`, `grasp_cut_compare.py`, `ogbar_frame_duplicates.py`, `ogbar_near_pairs.py`, `probe_hold_eval.py`, `smt_ogbar_watch_imagine.py`, `state_geometry.py`, `swap_test_ogbar.py`, `tilt_onset_table.py`, `tsne_belief_ogbar.py` | default to `"cuda:0"` |
| `device: cuda:1` in `configs/smt_phase1/`: `train_memtest4_big_aligndtwattract_phase1.yaml`, `train_memtest4_big_aligndtwpair_phase1.yaml`, `train_memtest4_s3ft_noobs_phase1.yaml` | `cuda:0` |
| `scripts/window60_report.sh`: `DEV=${DEV:-cuda:1}` | `cuda:0` |
| `scripts/all3_chain.sh`: runs two trainings in parallel on `cuda:0` and `cuda:1` | with one GPU, make them run one after another on `cuda:0` |

Re-list them first, in case anything changed: `grep -rln "cuda:1" src scripts configs tests`.

OGBar rendering uses `MUJOCO_EGL_DEVICE_ID=0` in `scripts/generate_ogbar.py`, `generate_ogbar_eval.py`, `branch_episode_ogbar.py`, `mpc_cem_ogbar.py`, `render_ogbar_showcase.py` and in `docs/OGBAR.md`. That's fine as long as GPU 0 exists; just confirm EGL rendering works (the first generation run will tell).

Commit this as one change ("Default to cuda:0 for the new machine").

## 4. Remove the RAM guard

`scripts/ram_guard.sh` watched the old shared machine's 125 GB RAM limit and is no longer needed. Delete it: nothing else references it. Commit.

## 5. One-time downloads and logins

- **DINOv2 weights:** `scripts/cache_dinov2_ogbar.py` loads them from the local torch hub copy, `~/.cache/torch/hub/facebookresearch_dinov2_main` (`source="local"`). Populate it once:
  ```
  pixi run python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')"
  ```
- **wandb:** `pixi run wandb login`. Training logs to project `info-gathering-wm`.
- **GitHub, to push from here:** `gh auth login`, then `gh auth setup-git`.

## 6. Regenerate the OGBar data (ask the user first)

These are long jobs; see [OGBAR.md](OGBAR.md) "Generating data". The old stores were made with the same config. The regenerated episodes come from the same distribution but are **not identical**: the probe's random draws now happen in a different order. So the numbers in [surprise_detectors.md](surprise_detectors.md) were measured on the old data.

| Step | Command | Size |
|---|---|---|
| Training store, 400 play episodes, seed 0 | `MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 pixi run python scripts/generate_ogbar.py --config configs/data/ogbar_slot_v3_flat_noslip.yaml --n 400 --workers 8 --zarr data/ogbar_v3_400.zarr --video-dir data/videos_ogbar_v3_400` | 5.5 GB |
| Held-out store, 200 episodes, seed 1 | the same with `--seed 1 --n 200 --zarr data/ogbar_v3_heldout200.zarr --video-dir data/videos_ogbar_v3_heldout200` | 2.8 GB |
| 5 probe-only eval episodes | `MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 pixi run python scripts/generate_ogbar_eval.py` | small |
| DINO cache, training store (every raw frame; the WM trains on all 5 offsets) | `pixi run python scripts/cache_dinov2_ogbar.py --src data/ogbar_v3_400.zarr` | 78 GB |
| DINO cache, held-out store (offset 0 only) | `pixi run python scripts/cache_dinov2_ogbar.py --src data/ogbar_v3_heldout200.zarr --stride 5` | 7.4 GB |

Set `--workers` to the CPU count. The cache script ends with a self-check (re-encodes 256 frames one at a time); it must pass.

The shell-game (memtest) datasets are only needed if the user goes back to that project. Their generators are the `generate-memtest*` tasks in `pixi.toml`.

## 7. Retrain the DINO-WM (ask the user first)

```
pixi run python scripts/train_dino_wm.py           # configs/dino_wm/train_ogbar_v3_400.yaml
```

This is 100k steps at batch 32, which took about 8.5 hours and 15 GB of GPU memory on one 4090. The run goes to `logdir/world_model/dino_wm_ogbar/<date>_proprio/`.

Every config and script that names the old run `logdir/world_model/dino_wm_ogbar/v3_400_proprio` needs the new path. The main one is `paths.wm_ckpt` in `configs/segmentation.yaml`. Find the rest with `grep -rn "v3_400_proprio" configs scripts`.

## 8. Then: the error-spike pipeline

The commands are in [surprise_detectors.md](surprise_detectors.md) "How to reproduce"; the design is in [recon.md](recon.md) and [assumptions.md](assumptions.md). The tests run without any data: `pixi run python -m pytest tests/ -q` should show 19 passed. Run them right after step 3, as a check that the environment works.

Where it stopped: the best detector is the "jump" detector on the 4-step, worst-16-patch error. Three things are open:
- a fresh, untouched test set, because the designs were chosen on privileged scores;
- running steps 4–6 (resolvability, candidate export) on the jump detector;
- longer horizons (h = 6–8).

## 9. Small leftovers

- `README.md` (root) still describes only the original shell-game datasets. Ask the user whether to update it.
- `figures/` is gitignored and empty here. Docs that embed figures show broken images until the figures are regenerated.

## Working rules the user has set (details in the memory notes)

- **Don't launch training or long jobs on a design statement alone.** Wait for an explicit go.
- **Plain language:** define every symbol, and say what each table number is (which loss, which representation, averaged over what).
- **Link every output file** you make with a clickable markdown link.
- **Figures follow [FIGURES.md](FIGURES.md)** (`scripts/figure_style.py`). Skip purely cosmetic figure fixes.
- **Run long tool calls in the background** so the user can keep talking.
