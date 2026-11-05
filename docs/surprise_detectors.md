# Surprise detectors on the frozen DINO-WM (OGBar v3)

**Status (2026-11-04).** Steps 1–3 of the error-spike pipeline have been run on real data, followed by a search over detector designs. The current best is the **jump detector** on the **4-step, worst-16-patch error**. Steps 4–6 (resolvability, candidate-test export) have not been run on it yet. Setup and earlier decisions: [recon.md](recon.md), [assumptions.md](assumptions.md).

## Goal

Find the moments in the OGBar play data where a deterministic world model *cannot* predict what happens next because of hidden state: the bar's centre of mass (COM). These moments become candidate test windows for a later VLM-narration stage. The DINO-WM is never retrained or modified. Everything here is a small model `g` trained on the WM's errors.

## Symbols

| Symbol | Meaning |
|---|---|
| t | model step = 5 raw steps (frameskip 5); an episode is ≤ 200 steps |
| `e_t` | DINOv2 ViT-S/14 patch tokens of the side-camera frame at t: 256 patches × 384 numbers |
| `a_t` | action from t to t+1: 5 raw 5-D actions (dx, dy, dz, dyaw, gripper) concatenated, 25 numbers |
| H | WM context length = 3 frames (`e_{t-2}, e_{t-1}, e_t`) |
| h | prediction horizon: the WM predicts frame t+h from the real context ending at t, open loop with the logged actions `a_t … a_{t+h-1}` (h = 1 is the one-step, teacher-forced prediction) |
| `d_p` | per-patch error: L2 distance between the WM's predicted token and the real token of patch p |
| `err_l2_mean` | mean of `d_p` over all 256 patches |
| `err_l2_topk` | mean of `d_p` over the 16 patches with the largest error ("worst-16") |
| λ | threshold for a "large" error |
| `g` | small model predicting the WM's error from the context frames and actions, before the frame exists |
| s, c | grasp point and true COM, both measured along the bar from its middle (cm); c is uniform on ±12.6 cm. **Privileged: used only to score.** |

## Data

| Store | Episodes | Role |
|---|---|---|
| `data/ogbar_v3_400.zarr` | 400 (seed 0) | The WM trained on all of them, so their errors are **in-sample**. `g` never trains on them. |
| `data/ogbar_v3_heldout200.zarr` | 200 (seed 1, same config and policy) | Held-out. Split by episode into `g` train / val / test = 120 / 40 / 40. |

Unless stated otherwise, every number below is on held-out episodes.

## What we tried, in order

### 1. One-step error, λ = 90th percentile (`outputs/v1_l2mean/`)

`g` = an MLP on patch-mean frame features plus `a_t`, trained to predict `y_t = 1[err_l2_mean_t > λ]`.

- **`g`:** test AUROC 0.91, AUPRC 0.60 (positive rate 0.10).
- **But the error was a slow level, not peaks.** `g` rarely went above 0.5, so there were 0–1 windows per episode.
- **`g` ignored the action:** removing `a_t` cost only 0.008 AUROC.

### 2. Worst-16 patches (`outputs/v2_topk/`)

Averaging only the 16 worst patches gave more peaks: 10.5 per episode instead of 7.6, each 1.9 steps long instead of 2.6. But `g` predicted these peaks worse (AUROC 0.83, AUPRC 0.40), so its windows lined up with them less.

### 3. λ = the average error (user, `outputs/v3_mean/`)

`y_t = 1[err_t > mean error over held-out g-train steps]`, so about half of all steps are positive.

- **`g`:** AUROC 0.94, AUPRC 0.93.
- **But high `g` came in long stretches** (for example steps 3–68 of ep073).

**Cause: distance to the camera.** `err_l2_mean` averages over all patches. The arm and bar fill more patches when they're near the side camera, and every covered patch carries some error.

Spearman correlation of `err_l2_mean` with privileged state, over 39,138 held-out steps:

| Gripper y (+ = towards the camera) | Bar y | Copy-last scene change | Gripper speed | Bar tilt change |
|---|---|---|---|---|
| +0.69 | +0.62 | +0.71 | −0.07 | +0.33 |

Steps above the mean were 11% in the farthest fifth of gripper y and 88% in the nearest.

### 4. Changed patches only, and the CLS token (dropped)

| Metric | Correlation with gripper y | Correlation with tilt change | Steps above mean: lift / hold / lower |
|---|---|---|---|
| `err_l2_mean` | 0.69 | 0.33 | 70% / 36% / 68% |
| changed patches only (token moved by > 15) | 0.45 | 0.10 | 58% / **78%** / 59% |
| CLS (WM prediction decoded, then re-encoded with DINOv2) | 0.31 | 0.30 | 55% / 36% / 57% |

- **Changed patches only** lost the link to bar motion, and the static hold became its highest stage.
- **CLS** reduced the camera-distance effect, but the user dropped it as too odd (it goes through the decoder and back through DINOv2).

### 5. The one-step error does not see the hidden COM

In a probe lift the bar swings only when gripped far from its COM. I averaged `err_l2_mean` over frames 0–5 after the lift starts (one-step predictions) and binned it by |s − c|:

| \|s − c\| | Probes | Median max tilt | Error |
|---|---|---|---|
| < 3 cm (stays level) | 91 | 1.7° | 9.42 |
| 3–6 cm | 75 | 4.2° | 9.31 |
| 6–10 cm | 84 | 28.2° | 9.15 |
| > 10 cm (swings) | 161 | 22.7° | 9.14 |

The error is flat. The lift does raise it, peaking at the lift's first step, but that's the motion starting, not the hidden COM. Once the first tilted frame is in the 3-frame context, the WM extrapolates the rest of the swing.

### 6. Open-loop errors over h = 1…4 steps: the hidden state appears

The grasp point s is visible. Gripped near an end (|s| > 6 cm), the bar almost always swings towards the middle, so a swing is predictable. Gripped near the middle, the outcome depends on which side the COM is. So the right comparison is predictable outcomes against open ones.

The table shows error over lift frames 0–5, each frame predicted h steps ahead. Values are mean ± standard error over held-out probes; "swings" = max tilt > 10°.

| Case | Probes | `err_l2_mean` h=1 | `err_l2_mean` h=4 | worst-16 h=1 | worst-16 h=4 |
|---|---|---|---|---|---|
| End grasp, swings (**predictable**) | 127 | 9.05 ± 0.14 | 10.67 ± 0.17 | 19.5 ± 0.3 | **26.2 ± 0.4** |
| End grasp, stays level (surprise) | 71 | 9.65 ± 0.17 | 11.84 ± 0.25 | 20.7 ± 0.3 | **30.3 ± 0.7** |
| Middle grasp, stays level (open) | 91 | 9.10 ± 0.14 | 11.77 ± 0.22 | 20.1 ± 0.2 | **32.1 ± 0.7** |
| Middle grasp, swings (open) | 122 | 9.25 ± 0.14 | 11.67 ± 0.21 | 20.5 ± 0.3 | **31.1 ± 0.6** |

- **This looks like a deterministic WM predicting the most likely outcome.** It's wrong when the likely outcome doesn't happen (end grasp, level) and when there is no likely outcome (middle grasp).
- **The effect grows with h.** It's about 5% at h = 1 and 15–20% at h = 4, roughly 4–5 standard errors.
- **The worst-16 metric shows it more strongly** than the full mean, because the bar covers few patches.

### 7. `g` on the 4-step worst-16 error, λ = mean (`outputs/v5_h4_topk/`)

Here `g` sees the 3 context frames and the 4 logged actions `a_t … a_{t+3}`.

- λ = 23.95; 43% of held-out steps are above it.
- **`g`:** test AUROC 0.85, AUPRC 0.81, ECE 0.022.
- **The action matters for the first time:** removing it drops AUROC by 0.042.
- 6.4 windows per episode, 8 steps long.

### 8. Detector search (`outputs/detectors/`, `scripts/seg_detectors.py`)

The goal, from the user: fire when a surprise **can** occur, don't punish much when it doesn't, ignore the slow level, and produce a binary output. All variants use the 4-step worst-16 error and the inputs from §7.

| Variant | Target and loss |
|---|---|
| exceed mean | BCE (binary cross-entropy) on `1[err_t > mean]` (§7) |
| q90 − q50 spread | Pinball (quantile) loss on log `err_t` at τ = 0.5 and 0.9. Signal = q90 − q50. At τ = 0.9 a miss costs 9× a false alarm. |
| q95 − q50 spread | the same at τ = 0.95 |
| q90 alone | the upper quantile without subtracting the median |
| **jump** (derivative) | BCE on `1[log err_t − median(log err_{t−1..t−4}) > κ]`, with κ = the 90th percentile of that jump over `g`-train rows |
| jump / spread with attention pooling | one learned query attends over each frame's 256 patches |
| jump / spread with a transformer | over the 3 frame tokens and an action token |
| jump / spread with grid pooling | keeps the 16 × 16 patch layout: per-patch projection to 32 numbers, then two stride-2 convolutions |

**How every variant is scored, identically:**
- the signal becomes a percentile rank over held-out val steps;
- it fires in hysteresis windows (on above 0.8, off below 0.65), so about 20% of steps are firing;
- scoring is on held-out val + test (80 episodes, 174 probe lifts).

Columns of the table below:
- **Lift fire rate:** fraction of probe-lift frames 0–5 inside a window. Cases: predictable = end grasp that swings; surprise = end grasp that stays level; open = middle grasp.
- **Elsewhere:** the same fraction for all other frames.
- **Grasp precision:** fraction of windows within 2 steps of a grasp onset, with the value for random windows of the same lengths in brackets.
- **Camera:** Spearman correlation of the signal with gripper y.

| Variant | Camera | Lift fire rate (predictable / surprise / open) | Elsewhere | Lift ÷ elsewhere | Open ÷ predictable | Grasp precision (random) |
|---|---|---|---|---|---|---|
| exceed mean | 0.68 | .33 / .37 / .39 | .17 | 2.2 | 1.16 | .38 (.31) |
| q90 − q50 spread | −0.34 | .50 / .45 / .56 | .19 | 2.6 | 1.11 | .37 (.30) |
| q95 − q50 spread | −0.45 | .50 / .33 / .48 | .18 | 2.4 | 0.98 | .44 (.33) |
| q90 alone | 0.49 | .48 / .52 / .55 | .19 | 2.7 | 1.14 | .40 (.30) |
| **jump** | **−0.21** | **.74 / .67 / .81** | .20 | **3.6** | 1.10 | **.47 (.27)** |
| jump, attention | −0.10 | .69 / .60 / .78 | .21 | 3.2 | 1.13 | .43 (.24) |
| jump, transformer | −0.27 | .59 / .54 / .74 | .23 | 2.7 | **1.25** | .40 (.25) |
| jump, grid | −0.21 | .65 / .56 / .72 | .21 | 3.0 | 1.11 | .41 (.25) |
| spread, attention | −0.39 | .68 / .57 / .70 | .19 | 3.3 | 1.03 | .47 (.31) |
| spread, transformer | −0.37 | .60 / .52 / .68 | .22 | 2.7 | 1.12 | .36 (.28) |
| spread, grid | −0.48 | .54 / .46 / .60 | .19 | 2.8 | 1.11 | .38 (.31) |

**Ceiling.** The *realised* labels, which a perfect `g` would reproduce, separate the lift cases much more strongly. Fraction of lift frames positive:

| Label | Predictable | Surprise | Open | Elsewhere | Open ÷ predictable |
|---|---|---|---|---|---|
| exceed mean | .58 | .76 | .76 | .41 | 1.31 |
| jump | .19 | .29 | .40 | .08 | **2.07** |

**Findings**
1. **The jump detector is the best event detector.** It fires on 67–81% of lift frames against 20% elsewhere. Its windows meet a grasp onset 1.8 times as often as random windows. It doesn't follow camera distance.
2. **The quantile spread cancels the slow level but overshoots.** Its correlation with camera distance turns negative: a log-ratio gets large when the typical error is small. It is also weaker at events.
3. **No variant anticipates *which* lifts have an open outcome.** They reach 1.10–1.25× against the target's 2.07×, even with spatial inputs or a transformer. The signal is in the target; `g` can't foresee it from the 3 context frames and the 4 actions.
4. **Richer inputs don't help.** Mean pooling with an MLP is as good as the alternatives.

Videos of the jump detector: [ep058](../outputs/detectors/jump_classifier/videos/heldout200_ep058_g.mp4), [ep073](../outputs/detectors/jump_classifier/videos/heldout200_ep073_g.mp4), [ep184](../outputs/detectors/jump_classifier/videos/heldout200_ep184_g.mp4), [v3_400 ep056](../outputs/detectors/jump_classifier/videos/v3_400_ep056_g.mp4) (the WM's training set).

Each video frame shows:
- the real frame;
- the WM's prediction of that frame from 4 steps back, decoded to pixels;
- the realised 4-step error with λ;
- the detector's percentile with its windows shaded;
- privileged labels (segment, grasp, s, c) and grasp onsets (blue ticks), for the viewer only.

## Audit: does the jump detector use anything privileged?

No. What it uses:

| Part | Source | Privileged? |
|---|---|---|
| Inputs: the 3 context frames | the 256 cached DINOv2 patch tokens of the camera image, averaged per frame | No |
| Inputs: actions | the 4 logged, executed actions `a_t … a_{t+3}`, with noise (`action`, not `action_clean`) | No |
| Inputs: proprio | not used (`g.use_proprio: false`) | n/a |
| Label | the WM's own 4-step errors (the jump of log `err_l2_topk` above the median of the previous 4 steps) | No |
| Training | early stopping on val labels, also WM errors only | No |
| Firing threshold | percentile of its own output over held-out val steps | No |

Caveats:
1. **It sees 4 future actions.** `a_t … a_{t+3}` are the robot's own plan, not hidden state, but they are information beyond t. That's how it knows a lift is coming.
2. **The actions leak the COM in slide-ins.** 248 of 273 slide-ins in the training store grip at the *true* COM (fewer than 2 probes before them; [OGBAR.md](OGBAR.md) "Known issues" #1), and the held-out store uses the same policy. A model that reads actions could exploit that. Probe grasps are random, so the lift results above don't depend on it. A fix would be to regenerate the data with slide-ins gripping at the probe estimate.
3. **Design choices were made by looking at privileged scores.** The error metric (worst-16), the horizon (h = 4) and the choice of the jump detector over 10 other variants were all guided by privileged checks (true COM, grasp point, tilt, grasp onsets) on held-out val + test episodes. The pipeline itself is label-free, but the reported numbers for the winner are **optimistic**, because the same episodes were used to choose it. A clean estimate needs a fresh, untouched held-out set (e.g. 100 more episodes with a new seed).
4. **Privileged keys are used only for scoring and display:** `scripts/seg_signal_check.py`, `seg_metric_compare.py`, `seg_lift_check.py`, `seg_detectors.py` (scoring), and the text panel and ticks in `seg_video.py`. Nothing upstream reads them.

## Open issues and next steps

- **A fresh test set** (caveat 3), before trusting any number above.
- **Run steps 4–6 on the jump detector.** Resolvability should separate windows that reveal hidden state from those that don't, which is exactly the distinction `g` itself can't make (finding 3).
- **A longer horizon** (h = 6–8). The hidden-state effect grew from h = 1 to 4. The rollout code supports it (`--h-max`).
- **Make "open vs predictable" learnable.** For example, explicit features for where the gripper is grasping relative to the bar, computed from the image.
- **Fix the slide-in COM leak** in the data policy (caveat 2).

## How to reproduce

```bash
pixi run python scripts/compute_wm_errors.py                                   # one-step errors (all metrics) -> outputs/
pixi run python scripts/compute_rollout_errors.py --h-max 4 --stores heldout200 # open-loop errors, h = 1..4
pixi run python scripts/compute_rollout_errors.py --h-max 4 --stores v3_400
pixi run python scripts/rollout_to_errors.py --h 4 --out outputs/v5_h4_topk     # the h = 4 errors table
O="paths.out_dir=outputs/v5_h4_topk error_metric=err_l2_topk error_horizon=4 g.n_actions=4"
pixi run python scripts/make_labels.py $O                                       # split + lambda (= mean)
pixi run python scripts/train_g.py $O                                           # the exceed-mean g
pixi run python scripts/seg_detectors.py                                        # detector variants (mean pooling)
pixi run python scripts/seg_detectors.py --pooling grid --tag " grid" --variants jump "q90 - q50" g.batch_size=128
pixi run python scripts/seg_video.py paths.out_dir=outputs/detectors/jump_classifier error_metric=err_l2_topk \
    error_horizon=4 g.n_actions=4 on_thresh=0.8 off_thresh=0.65 --signal-label "jump detector" \
    --episodes heldout200/ep058 heldout200/ep073 heldout200/ep184 v3_400/ep056
```

Diagnostics (privileged scoring): `scripts/seg_lift_check.py` (§5–6), `scripts/seg_metric_compare.py` (§4), `scripts/seg_signal_check.py` (§2), `scripts/seg_error_hist.py` (error histograms: [pooled](figures/error_hist_pooled.png), [per episode](figures/error_hist_episodes.png)).

| Output | What |
|---|---|
| `outputs/wm_errors.parquet` | one-step errors, every metric, all 600 episodes |
| `outputs/wm_rollout_errors_{heldout200,v3_400}.parquet` | open-loop errors, h = 1..4 |
| `outputs/v1_l2mean/`, `v2_topk/`, `v3_mean/`, `v4_cls/`, `v5_h4_topk/` | labels, `g`, videos for §1, 2, 3, 4, 7 |
| `outputs/detectors/` | §8: per-variant scores, `summary*.csv`, jump-detector videos |
