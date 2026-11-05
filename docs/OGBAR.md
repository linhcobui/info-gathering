# OGBar: OGBench cube-single with a bar whose centre of mass is hidden

**Status (2026-10-14).**

- **Current setup: v3**, config [`configs/data/ogbar_slot_v3_flat_noslip.yaml`](../configs/data/ogbar_slot_v3_flat_noslip.yaml). A round-ended bar has to be slid sideways into a raised slot.
- **Data policy:** scripted "play" episodes (`src/ogbar/play.py`): probes, slide-ins, slide-outs, smashes and free moves.
- **First dataset (complete):** 400 episodes at 224 px, `data/ogbar_v3_400.zarr`, 396,972 frames, 5.5 GB (debug videos of the first 10 in `data/videos_ogbar_v3_400/`). 391 episodes run the full 1000 steps; 9 end early (279–909 steps) because the bar ended up out of the arm's reach (`episode/unreachable`).
- **Open issues** are listed under [Known issues](#known-issues).

## Purpose

We want an offline dataset for world-model training in which an object has a property that **cannot be seen but changes how it behaves**.

- **The hidden property.** The bar looks the same from end to end. Its mass sits in two hidden point masses, so its centre of mass (COM, the point it balances on) changes from episode to episode.
- **Why it matters.**
  - Gripped within about 4.5 cm of its COM, the bar hangs level in the gripper; gripped further away, it tilts.
  - The slot only accepts a level bar, so the slide-in mostly succeeds only when gripped near the COM.
- **What a model has to learn.** To predict what happens, a world model has to infer the COM from what it has seen, for example from which way and how far the bar tilted when it was lifted.

## Where things live

| What | Where |
|---|---|
| Package (standalone copy; the OGBench submodule is never edited or imported) | `src/ogbar/` |
| Environment | `src/ogbar/envs/bar_env.py` (`BarEnv`) |
| Arm, IK and pixel rendering (copied from OGBench, small edits) | `src/ogbar/envs/manipspace_env.py`, `src/ogbar/envs/env.py` |
| Scripted oracle (approach, grasp, carry, slide-in) | `src/ogbar/oracle.py` (`BarMarkovOracle`) |
| Play-episode policy (what to do when, probes, pushes) | `src/ogbar/play.py` (`PlayEpisode`) |
| Mass model | `src/ogbar/mass.py` |
| Generator and zarr recorder | `src/ogbar/generate.py`, `src/ogbar/recorder.py`, `scripts/generate_ogbar.py` |
| **Config (current)** | `configs/data/ogbar_slot_v3_flat_noslip.yaml` |
| Variant configs kept for comparison | `ogbar_slot_v3.yaml` (rod fingertips), `ogbar_slot_v3_flat.yaml` (flat pads, no no-slip) |
| Earlier versions | `ogbar_slot_v1.yaml` (`ogbar_single.yaml` is identical), `ogbar_slot_v2.yaml` |

Copied files carry a header naming their OGBench source (commit a977253, MIT; licence in `src/ogbar/LICENSE_OGBENCH`). Robot meshes are read from the submodule without editing it.

## Environment

### The bar and its hidden mass

| | cube-single | OGBar v3 |
|---|---|---|
| Shape | 4 cm cube | Capsule: a cylinder 4 cm across, **40 cm long** overall (a 36 cm straight part plus a 2 cm-radius half-sphere at each end). |
| Mass | 79.4 g, uniform | **200 g in total.** The shell itself has no mass. Two hidden point masses sit on the centreline at ±18 cm (the centres of the end half-spheres): W_L (left) and W_R (right), W_L + W_R = 200 g. |
| Hidden property | none | Each episode draws **W_R uniformly from 30–170 g**. The COM offset from the bar's middle is c = (W_R − W_L) / 200 g × 18 cm, so c is uniform on **±12.6 cm**. Both ends look identical. |
| Inertia | solid cube | Each point mass is a 1 cm ball, so the bar has rotational inertia about every axis. Mass, COM and inertia are written into the model at the start of each episode. |

### Gripper and contact physics

| | cube-single | OGBar v3 |
|---|---|---|
| Fingertips | flat pads | Same flat pads (two boxes per finger). Contact priority 1, so the pad's own values set every pad–bar contact: sliding friction 0.7 (inner box) / 0.6 (outer box), twist friction 0.005 m, `condim` 4, elliptic friction cone. |
| Grip strength | `forcerange` 5 | **0.75** (`physics.gripper_forcerange`), a weak grip. |
| No-slip solver | off | **On** (`physics.noslip_iterations: 10`). See below. |
| Sliding friction elsewhere | per-geom defaults | 0.4 for every geom except the gripper (`physics.sliding_friction`). The bar's own value is 0.7 (`physics.bar_sliding_friction`); MuJoCo takes the larger of two equal-priority values, so bar–floor and bar–slot contacts use 0.7. |
| Rolling resistance | none | Bar `condim` 6 with rolling friction 0.003 m, so a released cylinder doesn't roll away forever. |
| Walls | invisible walls | **None.** The episode ends (`terminal`) once the *whole* bar is outside x 0.17–0.68, \|y\| < 0.38 (`bar.boundary`). |
| Collision | default | MuJoCo's midphase is disabled (`physics.disable_midphase`); it once dropped contacts between the bar and static fixtures. |
| Gripper verticality | IK target only | `physics.ik_tilt_ki: 1.0` adds a closed-loop correction that holds the gripper within about 0.04° of vertical (the wrist otherwise sags 0.4–0.8°). |

**What no-slip does.**

- **Without it.** MuJoCo's contacts are slightly soft, so friction can never fully stop motion: under a steady twist, a held bar slowly creeps around the fingers (about 0.5–3°/s), at any friction value (tested up to 4.0).
- **With it.** The no-slip pass recomputes the friction forces as hard limits after each step. Friction then either holds completely or slips, like textbook friction.

**How far from the COM the bar holds level** (flat pads + no-slip; lift 15 cm, hold 2 s, no noise, 3 mass splits × both sides, every 0.5 cm; [`figures/ogbar_hold_level_noslip.png`](../figures/ogbar_hold_level_noslip.png), video in `data/videos_ogbar_hold_level/`):

| Grip distance from COM | Tilt after a 2 s hold |
|---|---|
| 0–4.5 cm | level: median ≤ 2.5°, worst ≤ 5°, not turning (0.05°/s) |
| 5–6.5 cm | borderline: median 4–17°, depending on mass split and side |
| ≥ 7 cm | swings to 25–50° |

![Bar tilt after a 2 s hold against grip distance from the COM, with and without no-slip](../figures/ogbar_hold_level_noslip.png)

![Eight lifts of the same bar (100/100 g), gripped 0, 2, 4, 5, 6, 7, 8 and 10 cm from the COM, filmed side-on; the green ring marks the COM](../figures/ogbar_hold_level_noslip_still.png)

The cutoff is set by pad friction × grip force. With no-slip, pad friction 0.5 moves it to about 5 cm and 1.0 to about 7 cm.

### The slot

- **Position.** x = 0.425, y = −0.20 (left of the side camera's view).
- **Height.** Raised 5 cm (`slot.height` 0.065): a seated bar's centre is 8.4 cm above the floor.
- **Shape.** Just **two end blocks**, one at each round tip of the seated bar, with nothing between them, so the bar can be gripped anywhere along it.
  - **Pockets.** Each block is a half-ring around the tip: a semicircular pocket of radius 2 cm + 1.5 mm clearance, 1.5 cm wall, built from 12 convex segments. It runs along y with a roof over the whole tip and a back stop on the −y side. Blocks are 12.4 cm deep (y).
  - **Bases.** Each pocket sits on a solid base that fills the space under it over the block's full depth, so a bar cannot get underneath.
- **Insertion.** **Only by sliding** (`slot.insert: slide`): sideways along −y, level and at pocket height, with both tips entering both pockets at once.

### Start poses and cameras

- **Start.** The bar starts on the floor to the right of the slot: the whole bar inside x 0.18–0.67, y −0.07–0.37, any yaw. The arm starts at a random pose.
- **Visibility.** Start poses and floor targets must project fully inside both the side and the front camera images.
- **`rgb` (the world-model input).** Side camera `side_pixels` ("camera D"): from the +y side, looking along −y at the slot (`env.pixel_camera`). 224×224, arm drawn semi-transparent.
- **Front camera.** `front_pixels` is not recorded (`env.record_front: false`).
- **Debug renders** (masses, COM and grasp point drawn) use a separate renderer and never touch the recorded pixels.

### Success signal

`BarEnv._compute_success()` is `bar_in_slot()`. The bar counts as seated when all of these hold:
- it touches the slot;
- it touches neither fingertip nor the floor;
- its middle is within 1 cm of the slot centre;
- it is within 3° of level.

It is recorded per frame as `success`. A bar can count as seated for a moment and then fall out, so for a strict label require `success` to hold for N frames after release. `terminal` is not success: it marks only the last frame of an episode that ended early.

## Data policy: play episodes

Each episode is exactly **1000 steps** at 20 Hz (200 world-model steps at frameskip 5). The policy draws a **plan**, a shuffled list of units:
- 0–4 probes (uniform);
- a slide-in "pair" with probability 0.6;
- a smash with probability 0.5;
- Binomial(3, 0.25) free moves;
- at least one unit in total.

When a plan finishes, another is drawn. Whatever segment is running at step 1000 is cut off.

**Rule: a slide-in only after a lift.**
- The bar must have been gripped and lifted at least once earlier in the episode (by a probe, free move or smash).
- A slide-in drawn before the first lifting unit moves to just after it. If nothing has lifted the bar yet, it is skipped.
- So not every episode has a slide-in.

**Early end.** An episode ends early only if the bar leaves the workspace boundary, or no point of its straight part is reachable (`episode/unreachable`).

### Segments

| Segment | What happens |
|---|---|
| **probe** | Grip at a random point (±12 cm from the bar's middle), lift 10–25 cm, hold 2 s, lower it back where it was, let go. The held tilt is recorded (only while both fingertips hold the bar; a probe that drops the bar records none). |
| **slide_in** | Grip at the COM (see [Known issues](#known-issues) for which COM). Carry to a staging pose 12.5 cm in front of the slot, 8 cm above pocket height. Lower to pocket height, wait up to 30 steps for the bar to hang within 2° of level, then slide in along −y. Release when within 3 mm of seated or stalled. No set-down or regrip (`oracle.slot_setdown: false`). Always followed by a slide-out if it seated. |
| **slide_out** | **No grasp.** The gripper opens, turns so its two fingertips (about 10 cm apart) sit side by side along the bar, comes down behind the seated bar's middle between the two blocks, and pushes it out along +y. The fingertips sit about 7 mm below the bar's axis (`play.push_dz: 0.007`) and follow the bar's height as it drops off the blocks onto the floor. Up to 3 passes. |
| **smash** | Grip at a random point (±16 cm), bring the gripper to pocket height in front of the slot and push straight in, with no levelling. One push per smash. |
| **clear** | The same push as the slide-out, used whenever the bar is stuck at the slot without being seated (touching it, or lying in or under a pocket). Sideways along the bar's normal from its −y side, or along its length from its −y end if it lies within 30° of the y axis. It runs before every segment and after a smash or failed slide-in. |
| **free** | Grip at a random point (±16 cm, or ±14 cm if the bar is on the slot) and carry it to a random floor pose. Carrying is capped at the probes' lift speed (`play.free_max_action: 0.25`). |

**Shared behaviour:**
- **Grasp approach.** Grasps use a fine proportional approach and close only within 1.5 mm (along the bar) of the chosen point (`oracle.precise_grasp_axis_tol`). This uses no hidden information.
- **Retries.** After a slip or a miss, a retry takes its grasp point from the same segment's rule: the COM for slide-in, the segment's range otherwise.
- **Reach.**
  - A grasp point must lie inside the arm's workspace, with x ≥ 0.31: nearer the base, the gripper can't reach the floor.
  - This is re-checked every step while approaching, since the bar may still be rolling.
  - If nothing in range is reachable, the draw widens to ±16 cm.
- **Grasp yaw.** The oracle picks whichever of the two equivalent grasp yaws (180° apart) keeps the turn while carrying short and inside the gripper's ±171° limit.
- **Empty travel.** With nothing in hand, the gripper travels at the 0.35 m ceiling, using OGBench's full-speed controller.
- **Action noise** (OGBench "noisy" style), on every action including the scripted parts:
  - each episode draws a level ξ ~ U(0, 0.1);
  - each step adds Gaussian noise with standard deviation ξ × (1, 1, 1, 3, 1) on (dx, dy, dz, dyaw, gripper);
  - no random actions.

**COM estimate from probes** (`com_estimate`).
- **Fit.** A straight line of held tilt against grip position, tilt ≈ k (s − c), gives c = −intercept / slope, clipped to ±14 cm.
- **Fallbacks.** With fewer than 2 distinct probe positions, it uses the grip position of the most level probe. With no probe readings, there is no estimate.

### Measured behaviour (current config and code)

| Test | Result |
|---|---|
| Slide-in seated, by grip distance from the true COM (no noise, 16 per cell, scored 100 steps after release) | 0–1 cm 81%, 1–2 cm 62%, 2–4 cm 44%, 4–6 cm 31%, 6–8 cm 0% |
| Slide-in seated, dataset noise (48 episodes) | 36/48; median 274 steps, one 600-step timeout; lopsided bars (\|c\| > 8 cm) 8/13 |
| Slide-out after a seated slide-in | 36/36 out; one pass in 35; median 78 steps, max 176 |
| Smash (48 episodes) | 5/48 seated; median 233 steps, max 315 |
| Clear after a smash | the bar was stuck in 29 of 48 smashes; all 29 cleared, in 52–136 steps |
| Free move right after smash + clear | 47/48 placed |
| Carry speed while holding the bar | free 4.7 mm/step median, probe lift/lower 4.0 |
| Bar falling out of the gripper (10 episodes) | 7 of 45 grips. 6 of those during a push into the slot (a jammed bar dragging off the fingers), 1 on a lift; probes 0/18. |

## What gets recorded

The store is a flattened zarr: per-frame arrays are concatenated along time, and `episode_ends` / `episode_seeds` mark episode boundaries. Frame t is the state *before* action t. The layout follows the repo's memtest recorders.

**Model inputs:**

| Key | Shape | What |
|---|---|---|
| `rgb` | (224, 224, 3) uint8 | side camera |
| `action` | (5,) float32 | executed action (with noise): dx, dy, dz, dyaw, gripper, each in [−1, 1] |
| `state` | (28,) float32 | OGBench state vector. It includes the bar's exact pose; leave it out for a pixels-only model. |

**Privileged, per frame** (listed in the store attribute `privileged_keys`; never model inputs):

| Group | Keys |
|---|---|
| Hidden COM | `bar_com_world` (world position of the true COM), `bar_com_linvel` |
| Bar | `bar_pos`, `bar_quat`, `bar_tilt`, `bar_yaw_mod_pi`, `bar_linvel`, `bar_angvel` |
| Grip | `bar_in_hand_pose`, `grasped` (both fingertips touching), `pad_contact_l`, `pad_contact_r`, `effector_pos`, `effector_yaw`, `gripper_opening`, `gripper_contact` |
| Contacts and slot | `bar_floor_contact`, `bar_slot_contact`, `bar_in_slot`, `success`, `terminal` |
| Full sim state | `qpos` (21), `qvel` (20), `ctrl` (7) |
| Policy labels | `action_clean` (before noise), `segment_type` (index into attr `play_segments`), `segment_idx`, `oracle_phase` (attr `oracle_phases`), `probe_stage` (attr `probe_stages`), `pickup_idx`, `grasp_offset_s` (grasp point in use), `com_estimate` (see below), `target_pos`, `target_yaw` |

**Tables:**

| Table | Keys |
|---|---|
| `episode/` | `w_left`, `w_right`, `com_offset_c` (true COM offset), `inertia_diag`, `noise_xi`, `com_estimate`, `n_segments`, `n_probes`, `n_pickups`, `n_slide_in_seated`, `n_smash_seated`, `out_of_bounds`, `unreachable`, `length` |
| `segment/` | `episode`, `type`, `t_start`, `t_end`, `complete` (false if cut off at step 1000), `s_planned`, `s_last`, `com_from_estimate`, `probe_lift`, `probe_tilt_deg`, `probe_slipped`, `seated`, `out_of_slot`, `placed` |
| `pickup/` (one row per grasp attempt) | `episode`, `segment_idx` (row of `segment/` within the episode), `t_start`, `t_grasp` (−1 if never gripped), `t_end`, `s`, `outcome` (attr `pickup_outcomes`: placed / slipped / missed / timeout), `max_tilt` |

**`com_estimate` per frame** is only set once a slide-in has used the probe estimate (≥ 2 probes before it). Before that it is NaN, even if probes have happened. The end-of-episode estimate is in `episode/com_estimate`.

**Attributes:** `config` (the config used, as JSON) and `configs` (one per `--append`), `master_seed`, `n_episodes`, `fps`, `resolution`, `cameras`, `total_mass`, `mass_offset_d`, the name lists above, `privileged_keys`, `ogbench_source_commit`.

**Storage.**
- **Images.** Blosc zstd level 5 without shuffle, lossless, in chunks of 53 frames (about 8 MB raw). That's about 14.5 MB per 1000-frame episode, about 10% of raw; zarr's default lz4 needed 34 MB.
- **Everything else** is under 1 MB per episode.
- **Totals.** 400 episodes ≈ 6 GB; 1000 ≈ 15 GB.

**Loading.** Frameskip is applied in the loader: k = 5 gives 200 steps with a 25-D action (5 × 5). Episodes that end early are shorter than 1000 frames. The loader (`info_gathering.dataset`) has not yet been checked with this store (224 px, float actions in [−1, 1]).

## Generating data

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 pixi run python scripts/generate_ogbar.py \
    --config configs/data/ogbar_slot_v3_flat_noslip.yaml --n 400 --workers 8 \
    --zarr data/ogbar_v3_400.zarr --video-dir data/videos_ogbar_v3_400
```

- **Debug videos.** The first `debug.n_episodes` (10) episodes get one, showing the masses, COM and grasp point. `--append` adds episodes to an existing store.
- **Seeds.** Each episode's seed comes from the master seed and its index. An episode is rejected and redrawn with a derived seed if the bar's COM moves faster than 5 m/s or goes below the table.
- **GPU.** Pin rendering to a GPU with free memory (`MUJOCO_EGL_DEVICE_ID`). EGL fails ("framebuffer not complete") on a full GPU.

### Eval episodes (`data/ogbar_v3_eval5.zarr`, 2026-10-15)

`scripts/generate_ogbar_eval.py` writes 5 probe-only episodes with the same env, recorder and privileged keys, plus a debug video each (`data/videos_ogbar_v3_eval5/`). The DINOv2 tokens are in `data/ogbar_v3_eval5_dinov2/`. Fixed instead of sampled (also stored under the store's `config.eval`):

- **COM:** one episode per W_R = 30, 65, 100, 135, 170 g, so c = −12.6, −6.3, 0, +6.3, +12.6 cm, the training range end to end.
- **Start pose:** x 0.44, y 0.10, yaw 0. The bar lies along x, side-on to camera D, so its tilt shows. Each probe puts it back down; yaw stays within 3° and the bar drifts at most 6 cm over the episode.
- **Plan:** 5 probes and nothing else, grasping at s = −12, +12, 0, −6, +6 cm in that order (fixed, retries too), lift 15 cm.
- **No action noise.** Each episode ends after its 5th probe: 1094–1199 steps.

Held tilt per probe (deg; positive when s > c):

| c | s −12 | +12 | 0 | −6 | +6 |
|---|---|---|---|---|---|
| −12.6 | +1.7 | +31.6 | +49.6 | +46.7 | +45.6 |
| −6.3 | −5.4 | +30.5 | +32.7 | +2.0 | +44.8 |
| 0 | −33.8 | +30.6 | +1.6 | −42.7 | +22.6 |
| +6.3 | −31.7 | +5.4 | −6.1 | −43.2 | +1.3 |
| +12.6 | −33.8 | +1.0 | −50.7 | −44.7 | −4.6 |

All 25 probes held, with no slips. At a 15 cm lift a long lever lets the low end touch the floor, which caps the tilt near 30–35° for grasps at ±12 cm.

## Known issues

1. **The true COM can leak into the actions.** A slide-in grips at the *true* COM unless at least 2 probes came earlier in the episode, which is most episodes. The recorded actions therefore reveal the hidden COM. Options: only slide in after ≥ 2 probes, or always grip at the probe estimate.
2. **The probe COM estimate is weak.** With no-slip, the bar hangs level anywhere within about 4.5 cm of the COM, so the straight-line tilt fit has a dead zone. In 10-episode checks, some estimates were off by about 10 cm.
3. **Jerky empty-gripper travel.** Approach and retreat (oracle phases 1 and 9) use OGBench's controller, which moves at full speed until the waypoint and turns the wrist at its maximum rate. There is also a one-step jump when a probe takes over after grasping.
4. **Smashes sometimes seat** (5/48) with flat pads + no-slip. They were 0 with the earlier rod fingertips.
5. **Reach.** Lopsided bars put the COM near one end, which is sometimes hard for the arm to reach. A bar knocked next to the robot base ends the episode early (`unreachable`).
6. **Per-frame `com_estimate` lags.** It isn't updated after each probe, only when a slide-in uses it. It would be a small change to update it after every probe.
7. **Direction bias along world x** (measured on the v2 bar). Tilt comes out 20–40% larger in one direction for bars lying along x. Likely cause: the arm's joint servos jitter about world y.

## History

| Version | Bar | Task | Why it changed |
|---|---|---|---|
| Original design | 4×4×24 cm, 80 g, masses at ±10 cm | pick and place | the hidden mass barely showed with OGBench's grip |
| Platform / fence / cradle (2026-10-11 to 13) | capsule, 24 → 30 → 40 cm, 100 → 200 g | onto a board, over a fence, onto a short cradle | success depended too little on the COM, or the arm hit the fixture |
| Slot v1 (`ogbar_slot_v1.yaml`) | 40 cm square bar, pyramid ends | lowered into V-grooves | near-misses were levered flat by the slopes |
| Slot v2 (`ogbar_slot_v2.yaml`) | same | slid into full V-grooves | replaced by round ends |
| **Slot v3** | 40 cm capsule, 200 g, masses 30–170 g at ±18 cm | slid into round pockets | current |

**Changes within v3 (2026-10-14, user):**
- Rod fingertips → flat pads, plus no-slip.
- Slot raised 5 cm on solid bases.
- Play episodes with the slide-in rule.
- Push-based slide-out.
- Slower, rarer free moves and more probes.
- 224 px side view only.
