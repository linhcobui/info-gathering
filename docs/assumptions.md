# Assumptions: error-spike segmentation with the existing DINO-WM

Each entry: what was assumed, and who decided (user = confirmed by the user; default = my default, open to change).

## Data

1. **Held-out set = 200 new play episodes** (`data/ogbar_v3_heldout200.zarr`), same config and policy code as `ogbar_v3_400`, master seed 1 (the training store used seed 0; episode seeds are `SeedSequence([master, index, retry])`, so the two sets are disjoint). λ and g are fit on these, split by episode. (user, 2026-11-03)
2. **The policy code is the same as for the training store.** Edits to `src/ogbar/` after the training store was recorded (2026-10-15, -21, -27) only add opt-in eval options (fixed start pose, fixed probe/smash grasp points, restoring a simulator state), all off by default. One side effect: the 10-15 edit draws a probe's lift height before its grasp point, so a given seed gives different values than before, from the same distribution. (default, checked from the edit history)
3. **Offset 0 only.** Model frame t = raw frame `episode_start + 5t`. The WM trained on all 5 offsets, so offset 0 is in-sample for the training store. The held-out cache holds offset-0 frames only (`--stride 5`). (default)
4. **No reward for now.** `episode_return` and `episode_success` in `candidate_tests.json` are null, and the success correlation in the summary table is skipped. (user, 2026-11-03)
5. **No masked metric.** `err_l2_masked` stays an unused `mask_fn` hook. (user, 2026-11-03)

## World model

6. **The error is computed on the visual part only** (the first 384 numbers of each predicted token), not on the predicted proprio embedding. (default)
7. **The WM conditions on more than `a_t`:** `ê_{t+1} = f_WM(e_{t-2:t}, p_{t-2:t}, a_{t-2:t})`, with proprio `p` and the 2 previous actions. Inference passes it exactly what it trained on. (fact, recon §1)
8. **`err[t]` is the error of predicting frame t+1 from context ending at t**, so valid t = H−1 … T−2 (t = 2 … 198 for a 200-step episode). (default, following the task's notation)
9. **The adapter re-implements the predictor's forward pass** so that the features of an intermediate layer come out of the same pass. It is checked against `DinoWM.predict` on every run of `compute_wm_errors.py` (max |diff| 0.0 on the smoke test). The WM's weights and code are untouched. (default)

## Threshold and g

10. **λ is fit on the held-out g-train split only** (`lambda_fit_splits: [train]`), so the g val/test episodes never touch the threshold. (default)
11. **Class weighting distorts g's probabilities**, so g is Platt-scaled on the val split (fit `a`, `b` in `sigmoid(a·logit + b)` with unweighted BCE). Calibration is reported for both the raw and the calibrated outputs. On the toy run, ECE went from 0.24 (raw) to 0.004 (calibrated). (default)
12. **g sees the same 3 frames as the WM** (`history_len`), plus `a_t`. Past actions and proprio are not inputs (`g.use_proprio` is off by default). (default, my recommendation, not answered)
13. **`wm` pooling** = the predictor's layer-4 output, averaged over the last context frame's patches. It already contains `a_t` (the action is part of every token), so with this pooling the no-action ablation does not cleanly remove `a_t`; the metrics note this. (default)

## Windows and resolvability

14. **Windows are placed on frames, not on error steps.** g_t scores the prediction of frame t+1, so the series that gets segmented is q_τ = g_{τ−1}. A window [t_start, t_end] then holds the surprising frames themselves. That's what makes the task's recipe literally right: blind context ends at t_start − 1 (before the outcome), informed context ends at t_end (outcome seen). Without the shift, the informed context would end one frame before the outcome. (default)
15. **`max_len` cap**: a longer window keeps its max_len-step stretch with the highest mean q (it is not split). (default)
16. **Entropy method**: binary entropy h(q) = −q log q − (1−q) log(1−q), smoothed over `entropy.window` steps. Its peaks sit where q crosses 0.5, i.e. at the edges of a high-q stretch. They cut the episode into pieces, and a piece is a window if its mean q > `on_thresh`. This is my reading of "windowed entropy, boundaries at peaks". (default)
17. **Controls are drawn from episodes with the same `wm_train` flag as the window** (`control_episodes: same_split`). In-sample windows are compared with in-sample controls, and held-out windows with held-out controls. `heldout` and `all` are options. A control's whole span (its blind context through u + K) avoids every candidate window. (default)
18. **Truncated horizons**: controls are always rolled out for the full K steps. For a window truncated to K_used, the control gaps are recomputed over their first K_used steps. Windows with K_used < `resolve_min_horizon` (3) are skipped and listed as skipped. (default)
19. **Resolvability with H = 3 means "revealed and still visible", not "revealed and remembered".** The informed context only helps while the evidence (e.g. the tilt) is still in its 3 frames. Long-range effects (a probe at step 50 informing a slide-in at step 120) cannot show up with this WM. (default, my recommendation, not answered)

## Outputs

20. **`frame_paths`** = t_start − 3 … t_end (lead-in before the window, none after, so they don't overlap the outcome frames). **`outcome_frame_paths`** = t_end + 1 … t_end + 4. Images are PNGs exported from the zarr into `outputs/frames/`. (default)
21. **Diagnostics go to `outputs/diagnostics/`**, as the task says, not `figures/<topic>/` as FIGURES.md would. They use the house style. Privileged grasp onsets (from `grasped`) are drawn in the per-episode plots and counted in the summary table (`frac_windows_near_grasp`). They are diagnostics only; nothing upstream reads them. (default)
22. **Code layout**: the package is `src/seg/`, the scripts are in `scripts/`, the tests in `tests/`, the config is `configs/segmentation.yaml`. `outputs/` is not in `.gitignore`. (default, my recommendation, not answered)
23. **New dependencies**: `pandas`, `pyarrow` (parquet) and `pytest`, added as PyPI dependencies. The package diff showed only these and their own small dependencies (`iniconfig`, `pluggy`); no existing version changed. AUROC, AUPRC and ECE are computed in NumPy (no scikit-learn). (default)

## Revision 2026-11-03: a peakier error signal (user: "more peaks")

24. **Default error metric is now `err_l2_topk`** (mean per-patch L2 over the 16 worst of 256 patches), not `err_l2_mean`. On held-out episodes it gives 10.5 runs above λ per episode (vs 7.6), runs of 1.9 steps (vs 2.6), and no correlation with gripper speed (Spearman 0.01). `g` still predicts it: test AUROC 0.83, AUPRC 0.40, at a positive rate of 0.10. `err_l2_max` gives more peaks (14.4 per episode), but `g` can't predict them (AUROC 0.71). Detrending (minus a rolling median) gives the most peaks, but they correlate with arm motion (Spearman 0.16–0.19). Scores: `outputs/diagnostics/signal_check.csv` from `scripts/seg_signal_check.py` (privileged; for choosing only).
25. **Window thresholds lowered to on 0.2 / off 0.12.** `g` is calibrated, so 0.2 is twice the 10% base rate. This gives about 6 windows per held-out test episode (vs 1.6 at 0.5). Precision against privileged events barely changes with the threshold, and it is close to chance: windows meet a >5° one-step tilt change 28% of the time vs 26% for random windows of the same length, and a grasp onset 38% vs 28%.
26. **The WM memorised its training episodes' local errors.** Under `err_l2_topk`, only 3.4% of WM-train steps exceed the held-out λ, vs 10% of held-out steps (`err_l2_mean`: 6% vs 10%).
27. **λ = the average error (user, 2026-11-03).** `lambda_stat: mean`: λ is the mean of `err_l2_mean` over the held-out g-train episodes (7.98), so g predicts P(err_t > average error | e_{t-2:t}, a_t). This replaces the 90th percentile and makes `err_l2_mean` the default again, with window thresholds back at 0.5 / 0.3. Output: `outputs/v3_mean/`. (Items 24 and 25 describe the earlier top-k variant, kept in `outputs/v2_topk/`.)
