# Recon: the existing DINO-WM on OGBar (Step 0 of the error-spike segmentation task)

Written 2026-11-03. Every fact below was read from the code, the checkpoint or the stores; nothing is guessed.

## Symbols

| symbol | meaning |
|---|---|
| raw step | one environment step at 20 Hz; a stored frame is the state *before* the raw action with the same index |
| t | model step = 5 raw steps (frameskip k = 5, 4 Hz) |
| `e_t` | DINOv2 patch tokens of the side-view frame at model step t, shape (P, D) = (256, 384) |
| `p_t` | proprioception at t: `state[:19]` of the store (joint pos/vel, effector pos, cos/sin yaw, gripper opening, gripper contact) |
| `a_t` | action from step t to t+1: the 5 raw 5-D actions (dx, dy, dz, dyaw, gripper) concatenated, 25 numbers |
| H | WM history length = 3 frames |

## 1. Where the WM lives and how to call it

| what | where |
|---|---|
| model (predictor, decoder, rollout) | [src/world_model/models/dino_wm/model.py](../src/world_model/models/dino_wm/model.py), class `DinoWM` |
| builder | `build_model(cfg, stats, action_dim, proprio_dim)` in [scripts/train_dino_wm.py](../scripts/train_dino_wm.py) |
| training config | [configs/dino_wm/train_ogbar_v3_400.yaml](../configs/dino_wm/train_ogbar_v3_400.yaml) |
| checkpoint | `logdir/world_model/dino_wm_ogbar/v3_400_proprio/checkpoints/wm_latest.pt` (step 100,000; keys `model`, `optimizers`, `step`, `stats`, `config`, `wandb_run_id`) |
| loading | `state = CheckpointManager.load(path)` ([src/world_model/utils/checkpoint.py](../src/world_model/utils/checkpoint.py)); `cfg = OmegaConf.create(state["config"])`; `model = build_model(cfg, state["stats"], 25, 19).eval(); model.load_state_dict(state["model"])` |
| data loader | [src/info_gathering/ogbar_dino.py](../src/info_gathering/ogbar_dino.py), `OgbarDinoWindows`; `episode_sequence(e)` returns a whole episode at stride 5 |
| working examples of inference | `one_step` and `open_loop` in [scripts/dino_wm_videos.py](../scripts/dino_wm_videos.py) |

**`eval()` is required**: the predictor has dropout 0.1.

**Token layout.** `model.encode(visual, proprio, action)` turns (b, t, 256, 384) tokens, (b, t, 19) proprio and (b, t, 25) actions into z of shape (b, t, 256, 404). Each patch token is [384 DINO | 10 proprio embedding | 10 action embedding]; the two embeddings are tiled over the 256 patches of their frame. Actions and proprio are normalised inside the model with the train-set mean/std stored in the checkpoint (`state["stats"]`).

**One-step prediction** (teacher forcing, as trained):
```python
z = model.encode(vis[t-2:t+1][None], prop[t-2:t+1][None], act[t-2:t+1][None])   # frames t-2, t-1, t
z_hat = model.predict(z)[:, -1]                    # (1, 256, 404): prediction of frame t+1
e_hat = z_hat[..., :384]                           # visual part; [384:394] = predicted proprio embedding
```
The predictor is frame-causal and predicts the next frame at every position, so `predict(z)[:, -1]` is the one that has read all 3 context frames. **Note:** `ê_{t+1}` depends on `e_{t-2:t}`, `p_{t-2:t}` and `a_{t-2:t}`, i.e. on the past two actions as well as `a_t`, and on proprio. So the WM is really `f_WM(e_{t-2:t}, p_{t-2:t}, a_{t-2:t})`.

**Open-loop rollout:** `model.rollout(visual0, proprio0, actions)` with `visual0`, `proprio0` = the n real context frames (b, n, …) and `actions` (b, n + K, 25), where `actions[:, i]` leaves frame i. It returns `(visual, proprio_emb)` with n + K + 1 frames: the first n are the real ones, the rest are predicted. Each predicted frame gets its own predicted proprio embedding and the logged action's embedding (the action part is overwritten with the real action). Example: `open_loop` in dino_wm_videos.py.

The training loss is MSE on the visual + proprio parts; the reported 1-step visual latent MSE is 0.227 (copy-last-frame baseline 0.657), measured on training data.

## 2. Embeddings and preprocessing

- **Encoder:** frozen DINOv2 ViT-S/14 (`dinov2_vits14`, local torch.hub copy at `~/.cache/torch/hub/facebookresearch_dinov2_main`). Keys `x_norm_patchtokens` (used by the WM) and `x_norm_clstoken` (cached, unused by the WM).
- **Shape:** P = 256 patches (16 × 16 grid) × D = 384. (DINO-WM itself uses 196 px → 14 × 14; here 224 px.)
- **Preprocessing:** 224 × 224 RGB uint8 / 255, ImageNet mean/std normalisation, fp32 compute, stored fp16.
- **Cached on disk, not computed on the fly:**
  - training store: `data/ogbar_v3_400_dinov2/patch.npy`, (396,972, 256, 384) fp16 memmap, 78 GB, **every raw frame** (the WM trained on windows starting at any raw offset). Also `cls.npy`, `episode_ends.npy`, `meta.json`. Built by [scripts/cache_dinov2_ogbar.py](../scripts/cache_dinov2_ogbar.py).
  - eval stores: `data/ogbar_v3_eval5_dinov2/`, `data/ogbar_v3_ep033_branches_dinov2/` (every raw frame).
  - held-out store (new, this task): `data/ogbar_v3_heldout200_dinov2/` with `--stride 5` (offset-0 model frames only; row i = source frame `frame_index.npy[i]`).
- The full cache does not fit in RAM (125 GB shared limit); read per episode from the memmap.

## 3. History, frameskip, actions

- **H = 3** (`model.num_hist`), `num_pred` = 1.
- **Frameskip 5**, applied in the loader: model frame t = raw frame `episode_start + offset + 5t`. The WM was trained on **all 5 raw offsets**; this task uses **offset 0** only (200 model steps per full episode).
- **Action:** 25-D = 5 raw 5-D actions (dx, dy, dz, dyaw, gripper, each in [−1, 1], *with* the dataset's action noise) between frame t and t+1, concatenated. Normalised inside the model. This chunking is the only "chunking"; the WM predicts one model step per call.
- **Proprio:** 19-D `state[:19]`, used by this checkpoint (`use_proprio: true`).

## 4. Dataset format

- **Store:** flattened zarr; per-frame arrays concatenated along time; `episode_ends` (exclusive end index of each episode) and `episode_seeds`. Per-frame keys: `rgb` (224², side camera D), `action` (5,), `state` (28,; includes the bar's exact pose, so only `state[:19]` is a model input), plus privileged keys (`bar_com_world`, `bar_tilt`, `grasped`, `success`, `segment_type`, `probe_stage`, …). Tables `episode/` (incl. `com_offset_c`), `segment/`, `pickup/`. Full list: [OGBAR.md](OGBAR.md#what-gets-recorded).
- **Rewards:** there is no `reward` key. `success` (bar seated) is per frame. Per the user (2026-11-03), **no reward is used in this task for now**.
- **Training store `data/ogbar_v3_400.zarr`:** 400 episodes, 396,972 raw frames, master seed 0. 391 run 1000 raw steps; 9 end early (279–909). At offset 0: 79,391 model steps (55–200 per episode); 78,191 valid one-step targets (t = 2 … T−2).
- **WM training split: none. All 400 episodes were in the WM's training set** (`OgbarDinoWindows(episodes=None)` in train_dino_wm.py; ~8 epochs). So every error on this store is in-sample.
- **Held-out data:**
  - `data/ogbar_v3_heldout200.zarr` (new, this task): 200 play episodes, same config (`configs/data/ogbar_slot_v3_flat_noslip.yaml`, identical to the one recorded in the training store's attrs) and same policy code, **master seed 1** (independent episode seeds). This is the held-out set for λ and g.
  - `data/ogbar_v3_eval5.zarr`: 5 probe-only episodes, fixed COM, no action noise: a different distribution, diagnostics only.
  - `data/ogbar_v3_ep033_branches.zarr`: 8 counterfactual branches of *training* episode 33 (share its history): not held-out.
- **Known data caveat:** 248 of the 273 slide-ins in the training store grip at the *true* COM, so the action sequence before a slide-in reveals c (OGBAR_SMT.md "Data"). The held-out store has the same policy, so the same leak.

## 5. Masks

None exist (no SAM or other object masks). Per the user (2026-11-03), **no masked metric**: `err_l2_masked` is left as an unused hook.
