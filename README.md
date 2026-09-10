# info-gathering

Offline dataset generation in MuJoCo for **world-model memory benchmarks**, built around a
cups-and-ball *shell game*. The environment reveals a ball's location, hides it under cups,
shuffles the cups through a series of swaps, and lifts a cup to reveal what's underneath.
A model trained on this data must **track a hidden object through occlusion and motion** —
a direct probe of memory in a learned world model.

---
![Screenshot](assets/Screenshot%202026-09-10%20154029.png)
## Motivation

Standard next-frame world models can look sharp while failing at object permanence: once an
object is occluded, the model has no incentive to remember where it went. The shell game
isolates exactly this capability. The ball is shown, then hidden; the only way to predict what
a lift reveals is to have maintained a latent belief about the ball's slot across the swaps.
Because every frame is generated from known ground truth, the dataset comes with exact labels
(`ball_slot`, `cup_order`) for probing whether a model's latent actually encodes the hidden
state.

Two task variants:

- **Memory Test -1** (minimal): 1 cup, either hiding a ball or empty. Cover, wait, lift.
  Tests: "did you remember whether a ball was there?"
- **Memory Test 0** (main task): 3 cups, 1 ball. Reveal → cover → repeated blocks of
  *5 swaps + 1 lift*. The ball rides its cup through swaps and stays hidden until a lift.
  Tests: "did you track the ball through the shuffles?"

---

## How the task works

An episode is a deterministic function of a single integer seed. The timeline is:

```
reveal   cups raised, ball visible in its starting slot
cover    all cups lower over their slots (ball now hidden)
┌─ block (repeated blocks_per_episode times) ─────────────┐
│  pause                                                   │
│  swap  ×5   two slots exchange cups via planar arcs      │
│  pause                                                   │
│  lift  cup in slot 2 rises, pauses, lowers (reveal)      │
└──────────────────────────────────────────────────────────┘
```

Key mechanics:

- **Slots vs identities.** Three fixed floor positions (slots 0,1,2). Each cup has a stable
  *identity*; a swap exchanges which identities occupy two slots. `cup_order` records the
  identity sitting in each slot at every frame; `ball_slot` records which slot holds the ball.
- **Swaps are planar arcs.** The two swapping cups slide along the floor (never lifted). The
  cup moving rightward arcs *toward* the camera (+y); the one moving leftward arcs *away* (−y),
  so their paths don't intersect. The ball rides whichever cup it's under.
- **The lift is fixed to slot 2** for now, encoded as action `[0,0,1]`; every other frame is
  the no-op `[0,0,0]`. Whether a lift reveals the ball depends on whether the swaps happened
  to walk the ball into slot 2 (≈1/3 of lifts, by symmetry).
- **Everything is kinematic.** Cups are MuJoCo *mocap* bodies; the ball is posed by direct
  `qpos` writes with `mj_forward` (no physics stepping). This guarantees that rendered pixels
  always agree with the `ball_slot` label — critical for a benchmark whose whole point is
  label correctness.

---

## Repository layout

```
info-gathering/
├── pixi.toml / pixi.lock        environment (mujoco, zarr<3, imageio, tqdm, …)
├── configs/
│   └── memtest0.yaml            ALL hyperparameters: timing, geometry, colors, camera, seed, output paths
├── submodules/
│   └── object_sim/              git submodule (vikashplus/object_sim) — the cup mesh + collision decomposition
├── src/info_gathering/          importable library
│   ├── scene_builder.py         builds the MuJoCo XML from config (cups, colors, camera, OGBench-style floor)
│   ├── choreography.py          atomic primitives (hold / vertical_move / swap_arc) + generate_plan + choreograph
│   ├── rendering.py             build_model + render_episode (kinematic posing, EGL offscreen render)
│   └── recorder.py              ZarrRecorder: flattened zarr writer with episode boundaries + seed log
├── scripts/                     thin entry points
│   ├── test_render.py           milestone: compile scene, render one frame → PNG
│   ├── preview_episode.py       one episode → MP4 (fixed seed, for tuning look/camera)
│   ├── print_episode.py         print an episode's GT timeline (segmented, human-readable)
│   └── generate_dataset.py      N episodes → zarr + sample MP4 previews (master→per-episode seeds, tqdm)
└── data/                        (gitignored) outputs; large datasets live on /data instead
```

**Design principle:** `src/` is the library (imported, testable, reused); `scripts/` are thin
runnable entry points that wire library calls to I/O. All tunable numbers live in
`configs/memtest0.yaml` — no magic constants in code.

---

## Environment setup

The environment is managed with [pixi](https://pixi.sh).

```bash
# from the repo root
pixi install
pixi shell
```

Notes specific to the deployment machine (`sencha`, headless via SSH):

- **Headless rendering** uses EGL. Scripts set `MUJOCO_GL=egl` automatically; a working
  NVIDIA GPU + driver is required.
- **`/data` is NTFS (fuseblk)** and rejects the file-timestamp operations pixi needs, so the
  **pixi env lives in `$HOME`**, not on `/data`. Writing plain data files to `/data` is fine —
  only package installs break there.
- **zarr is pinned `<3`.** zarr v3 changed `create_dataset` to require an explicit `shape=`,
  which broke the recorder; v2 is also what the broader offline-RL ecosystem
  (diffusion-policy, lerobot) expects.

---

## Usage

### Sanity-render one frame
```bash
python scripts/test_render.py
# → data/test_frame.png
```

### Preview one episode as a video
```bash
python scripts/preview_episode.py
# → data/preview.mp4   (fixed seed; edit configs/memtest0.yaml to retune camera/colors)
```

### Inspect an episode's ground truth
```bash
python scripts/print_episode.py --ep 0
# prints a segmented timeline: phase, ball_slot, cup_order, action, and per-lift reveals
```

### Generate the dataset
```bash
# smoke test first
python scripts/generate_dataset.py --n 20 --preview-every 5

# full run (long) — launch under tmux so it survives disconnects
tmux new -s gen
python scripts/generate_dataset.py --preview-every 50
# detach: Ctrl-b then d   |   reattach: tmux attach -t gen
```

Outputs (paths set in config, recommended on `/data`):
- `memtest0.zarr` — the dataset
- `videos/ep_XXXXX.mp4` — periodic previews for spot-checking variety

---

## Dataset format

A **flattened** zarr store: all episodes concatenated along a single frame axis, with explicit
boundaries. This cleanly handles variable-length episodes (episodes differ in length as
`blocks_per_episode` / swap counts change).

```
memtest0.zarr/
├── rgb            (total_frames, 224, 224, 3)  uint8   head-on RGB
├── action         (total_frames, 3)            uint8   [0,0,0] or [0,0,1]
├── ball_slot      (total_frames,)              uint8   which slot (0/1/2) holds the ball
├── cup_order      (total_frames, 3)            uint8   cup identity occupying each slot
├── phase          (total_frames,)              uint8   0=reveal 1=cover 2=pause 3=swap 4=lift
├── episode_ends   (n_episodes,)                int64   cumulative frame index where each episode ends
└── episode_seeds  (n_episodes,)                int64   per-episode seed (full reproducibility)
attrs: config (full JSON snapshot), master_seed, n_episodes, fps, resolution
```

Slicing episode `i`:
```python
import zarr
z = zarr.open_group("memtest0.zarr", mode="r")
ends = z["episode_ends"][:]
start = 0 if i == 0 else ends[i-1]
end   = ends[i]
frames = z["rgb"][start:end]         # (T, 224, 224, 3)
labels = z["ball_slot"][start:end]   # (T,)
```

**Reproducibility.** A master seed derives one seed per episode
(`master_rng.integers(...)`), and every per-episode seed is stored in `episode_seeds`. Any
episode can be regenerated exactly from its seed; the full config snapshot in `attrs` makes the
dataset self-describing.

---

## Configuration reference (`configs/memtest0.yaml`)

| Group | Keys | Meaning |
|---|---|---|
| Structure | `blocks_per_episode`, `swaps_per_block`, `lift_slot`, `ball_start_slot` | how many swap+lift blocks; which slot lifts; ball start (null = uniform random) |
| Timing (frames) | `reveal_frames`, `cover_frames`, `swap_frames`, `lift_up/pause/down_frames`, `pause_frames` | per-phase durations |
| Geometry (m) | `cup_spacing`, `cup_rest_z`, `cup_up_z`, `arc_height_front/back`, `ball_radius`, `ball_z` | slot spacing, resting/raised cup heights, swap arc bulge, ball size |
| Appearance | `cup_color_mode` (`same`/`distinct`), `cup_color_same`, `cup_colors_distinct`, `ball_rgba`, `floor_texrepeat`, `sky_rgb1/2` | matte cups; same-mode is red; OGBench-style checker floor |
| Camera | `camera`, `camera_pos`, `camera_xyaxes` | head-on camera pose (tunable; smaller up-vector y = lower horizon) |
| Rendering | `resolution`, `fps` | 224×224, 20 fps |
| Seeds/output | `seed`, `n_episodes`, `zarr_path`, `video_dir` | master seed; dataset size; output locations |

---

## Roadmap

### Near-term (dataset correctness & scale)
- **`inspect_data.py` — correctness gate (next).** Render `ball_slot` / `phase` / `cup_order`
  overlaid onto frames and export as annotated videos, so labels can be verified against pixels
  across many episodes. *This is the mandatory check before training anything:* a world model
  trained on subtly-misaligned labels silently learns the wrong invariant.
- **Ball-cover audit.** Confirm the (now larger, `ball_radius 0.03`) ball is fully occluded
  under a lowered cup from the camera's viewpoint in every phase; tighten `ball_radius` /
  `cup_rest_z` if it peeks.
- **Scale generation.** Move outputs to `/data`, generate the full set (target: a few thousand
  episodes), record on-disk size and per-episode length distribution.

### Task richness (harder / more diagnostic variants)
- **Randomized lift target** (not always slot 2) so the model can't shortcut by only watching
  one position; action becomes a genuine one-hot over slots.
- **Balanced reveal ratio** — optionally bias the ball's walk or lift choice so positive/negative
  reveals are ~50/50 rather than ~1/3, for cleaner probing.
- **Difficulty knobs**: more swaps per block, faster swaps, `cup_color_mode: same` (removes color
  as a cue, forcing pure positional tracking), variable `blocks_per_episode`.
- **Domain randomization** for generalization: jittered camera pose, lighting, floor texture,
  cup color per episode.
- **Memory Test -1** generator (1-cup variant) as a minimal-capacity baseline.

### Consumption / modeling
- **PyTorch `Dataset`/`DataLoader`** over the zarr: window sampling, episode-aware batching
  (never crossing `episode_ends`), train/val split by episode.
- **Probing harness**: train a linear/MLP probe from a world model's latent to `ball_slot` at
  lift frames — the quantitative memory metric this dataset exists to produce.
- **Baselines**: a memoryless next-frame model (should fail at lifts) vs a recurrent/latent-state
  model (should succeed), to validate the benchmark discriminates.

### Infrastructure / hygiene
- Move dataset outputs off `$HOME` onto `/data` by default in config.
- Optional: switch `.pixi` env location / document the NTFS constraint for new machines.
- Unit-test `choreography.py` invariants (e.g. `ball_slot` only changes on swaps that involve
  the ball's slot; `cup_order` is always a permutation).
- Commit `pixi.lock` alongside `pixi.toml` for a reproducible environment.

---

## Known constraints & gotchas

- **NTFS `/data`**: no pixi/conda envs there (timestamp ops fail); data files are fine.
- **`$HOME` on nearly-full `/`**: keep large datasets on `/data`; watch `df -h /`.
- **EGL shutdown noise**: a harmless `EGLError` traceback can print at interpreter exit *after*
  files are written; the renderer is closed explicitly in library code to minimize it.
- **Mesh path resolution**: the cup's `assets.xml` references meshes relative to a specific
  `meshdir`; the scene builder passes an absolute `submodules/object_sim` path so the STLs
  resolve regardless of where a script is run from.

---

## Acknowledgements

Cup mesh and collision decomposition from
[vikashplus/object_sim](https://github.com/vikashplus/object_sim) (Apache-2.0). Floor/skybox
styling follows the MuJoCo Menagerie / OGBench standard scene convention.
