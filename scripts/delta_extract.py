"""DELTA diagnostics, stage 1 of 2: run every arm over the eval sets and dump what it did.

For each arm and each eval zarr this records, per episode and per frame:

    teacher z      the 60-frame window encoding at every step (one shot, no updater)
    posterior z    one encode at t = 0, then the (gated) posterior chained over every real frame
    gate m         the posterior's DELTA gate at every step, per memory token (dim mode: the
                   per-token mean and open fraction, plus the full 2560-d mean per event type)
    |dz| per token ||z_t - z_{t-1}|| for each memory token, teacher and posterior -- the
                   basis-native "which regions changed", which exists for ungated arms too
    onset reads    the reward head at the lift onset (r0 memory-only, r1 memory + a_t) from the
                   teacher, posterior and prior (posterior, then 9 prior steps) states

Labels come from the zarr plus the swap schedule reconstructed from the episode seeds (the slot
pairs are not stored), verified frame-for-frame against the stored ball_slot. Training never
sees any of these labels; this script only reads trained models.

Stage 2 (scripts/delta_analyze.py, system python with scikit-learn) does the probes, the
t-SNEs and the heatmaps from the .npz written here.

    pixi run python scripts/delta_extract.py --out logdir/delta_diag/run1 --device cuda:1 \
        --arm D-base=logdir/world_model/.../2026-09-21_11-49-55_memtest4_big_noobs_phase1 ...
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import zarr  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from torch.amp import autocast  # noqa: E402

from info_gathering.dataset_smt_prelim import load_embedded_episodes  # noqa: E402
from info_gathering.memtest4 import sample_episode  # noqa: E402
from world_model.utils.checkpoint import CheckpointManager  # noqa: E402

META = ("ball_under_target", "ball_slot", "phase", "cup_z_max")
EVENTS = ("reveal", "descent", "swap(0,1)", "swap(0,2)", "swap(1,2)", "static", "lift_onset",
          "lift", "show_static")
_SCHEDULES = {}


def schedules(zarr_path):
    """Per episode: start slot and [(first_frame, slot_a, slot_b)] for every swap, from the seed."""
    if zarr_path in _SCHEDULES:
        return _SCHEDULES[zarr_path]
    g = zarr.open_group(str(ROOT / zarr_path), mode="r")
    cfg = json.loads(g.attrs["config"])
    ends = g["episode_ends"][:]
    starts = np.r_[0, ends[:-1]]
    seeds, bstart = g["episode_seeds"][:], g["episode_ball_start"][:]
    nsw, hold, slot = g["episode_swapped"][:], g["episode_hold"][:], g["ball_slot"][:]
    first_swap = int(cfg["reveal_frames"]) + int(cfg["move_frames"])
    step = int(cfg["swap_frames"]) + int(cfg.get("swap_gap", 0))
    out = []
    for e in range(len(ends)):
        a, b = int(starts[e]), int(ends[e])
        ep = None
        # memtest4_big samples hold inside sample_episode; the eval sets pass it in (see
        # build_scene_state.py). Keep whichever reproduces ball_slot exactly.
        for args in ((int(bstart[e]), int(nsw[e])), (int(bstart[e]), int(nsw[e]), int(hold[e]))):
            try:
                cand = sample_episode(cfg, np.random.default_rng(int(seeds[e])), *args)
            except Exception:
                continue
            if len(cand["ball_slot"]) == b - a and np.array_equal(cand["ball_slot"], slot[a:b]):
                ep = cand
                break
        if ep is None:
            raise RuntimeError(f"{zarr_path} episode {e} does not reproduce from its seed")
        where = np.arange(3)                       # identity -> slot
        swaps = []
        for k, (ia, ib) in enumerate(ep["pairs"]):
            sa, sb = sorted((int(where[ia]), int(where[ib])))
            swaps.append((first_swap + k * step, sa, sb))
            where[[ia, ib]] = where[[ib, ia]]
        out.append({"start": int(bstart[e]), "swaps": swaps, "n": b - a})
    _SCHEDULES[zarr_path] = (out, cfg)
    return out, cfg


def seq_key(s):
    return (s["start"],) + tuple((a, b) for _, a, b in s["swaps"])


def frame_labels(data, sched, cfg, moving):
    """event id, swaps-so-far per frame. moving[e, t]: e_t differs from e_{t-1}."""
    E, T = data.meta["phase"].shape
    event = np.full((E, T), -1, dtype=np.int64)
    done = np.zeros((E, T), dtype=np.int64)
    R, M, S = int(cfg["reveal_frames"]), int(cfg["move_frames"]), int(cfg["swap_frames"])
    for e in range(E):
        n = int(data.real_len[e])
        ph = data.meta["phase"][e]
        onset = int(np.flatnonzero(ph[:n] == 2)[0])
        for t in range(1, n):
            if ph[t] == 0:
                event[e, t] = EVENTS.index("reveal")
            elif ph[t] == 2:
                event[e, t] = EVENTS.index("lift_onset" if t == onset else
                                           ("lift" if moving[e, t] else "show_static"))
            elif R <= t < R + M:
                event[e, t] = EVENTS.index("descent")
            else:
                event[e, t] = EVENTS.index("static")
                for first, a, b in sched[e]["swaps"]:
                    if first <= t < first + S and moving[e, t]:
                        event[e, t] = EVENTS.index(f"swap({a},{b})")
        for first, _, _ in sched[e]["swaps"]:
            done[e, first + S - 1:] += 1                  # the swap has landed on its last frame
    return event, done


@torch.no_grad()
def run_arm(name, run_dir, zarr_paths, device, out, chunk=16):
    ckpt = Path(run_dir) / "checkpoints" / "wm_latest.pt"
    state = CheckpointManager.load(str(ckpt))
    cfg = OmegaConf.create(state["config"])
    model = instantiate(cfg.model, action_dim=2, device=device, _recursive_=False)
    model.load_train_state_dict(state["model"])
    model.requires_grad_(False).eval()
    mode = model.posterior.delta_mask
    K, D = model.arch["n_registers"], model.arch["d_model"]
    fut = model.fut_len
    mean, std = model.embed_mean.cpu().numpy(), model.embed_std.cpu().numpy()

    train_zarr = str(cfg.data_anchors.zarr_path)
    tr_sched, _ = schedules(train_zarr)
    n_train = len(tr_sched) - int(cfg.data_anchors.holdout_episodes)
    seen_keys = {seq_key(s) for s in tr_sched[:n_train]}
    meta = {"arm": name, "run": str(run_dir), "step": int(state["step"]), "mask": mode,
            "train_zarr": train_zarr, "n_train": n_train,
            "coef_delta": float(model.loss_cfg.get("coef_delta", 0.0))}

    # the prior's gate is a function of the action alone; memtest4 has three actions
    if model.prior.mask_net is not None:
        acts = torch.tensor([[0., 0.], [1., 0.], [0., 1.]], device=device)
        meta["prior_gate"] = model.prior.mask_net(acts).float().cpu().numpy().reshape(3, K, -1).mean(-1).tolist()

    # the arm's own training set too: stage 2 fits decoder-free probes on its onset states
    for zp in list(zarr_paths) + [train_zarr]:
        tag = Path(zp).stem
        data = load_embedded_episodes(str(ROOT / zp), model.embed_key, META, normalize=False)
        sched, scfg = schedules(zp)
        emb = ((data.embed - mean) / std).astype(np.float32)
        E, T = emb.shape[:2]
        moving = np.zeros((E, T), dtype=bool)
        moving[:, 1:] = np.abs(data.embed[:, 1:] - data.embed[:, :-1]).max(-1) > 1e-6
        event, done = frame_labels(data, sched, scfg, moving)
        onset = np.array([int(np.flatnonzero(data.meta["phase"][e, :int(n)] == 2)[0])
                          for e, n in enumerate(data.real_len)])

        zt_all = np.zeros((E, T, K, D), dtype=np.float16)
        zp_all = np.zeros((E, T, K, D), dtype=np.float16)
        gate = np.full((E, T, K), np.nan, dtype=np.float32)
        gate_open = np.full((E, T, K), np.nan, dtype=np.float32)
        gate_dim_sum = np.zeros((len(EVENTS), K * D), dtype=np.float64)
        gate_dim_cnt = np.zeros(len(EVENTS), dtype=np.int64)
        reads = {f"{s}_{h}": np.zeros(E, dtype=np.float32)
                 for s in ("teacher", "posterior", "prior") for h in ("r0", "r1")}
        for lo in range(0, E, chunk):
            idx = np.arange(lo, min(E, lo + chunk))
            e_t = torch.from_numpy(emb[idx]).to(device)
            pa = torch.from_numpy(data.prev_action[idx]).to(device)
            first = torch.zeros(len(idx), T, dtype=torch.bool, device=device)
            first[:, 0] = True
            zt = model.encode_windows(e_t, pa, first)                     # (B, T, K, D) float
            z = zt[:, 0]
            zs, ms = [z], []
            with autocast(model.device.type, dtype=model.amp_dtype):
                for t in range(1, T):
                    z, m = model.posterior.step_with_mask(z, e_t[:, t], pa[:, t])
                    z = z.float()
                    zs.append(z)
                    ms.append(None if m is None else m.float())
            zpost = torch.stack(zs, 1)
            zt_all[idx] = zt.cpu().numpy().astype(np.float16)
            zp_all[idx] = zpost.cpu().numpy().astype(np.float16)
            if ms[0] is not None:
                mm = torch.stack(ms, 1)                                     # (B, T-1, K, 1|D)
                gate[idx, 1:] = mm.mean(-1).cpu().numpy()
                gate_open[idx, 1:] = (mm > 0.5).float().mean(-1).cpu().numpy()
                if mode == "dim":
                    flat = mm.flatten(2).cpu().numpy().astype(np.float64)   # (B, T-1, K*D)
                    ev = event[idx, 1:]
                    for k in range(len(EVENTS)):
                        sel = ev == k
                        gate_dim_sum[k] += flat[sel].sum(0)
                        gate_dim_cnt[k] += int(sel.sum())

            # onset reads: teacher, posterior, and posterior followed by 9 prior steps
            on = torch.from_numpy(onset[idx]).to(device)
            ar = torch.arange(len(idx), device=device)
            states = {"teacher": zt[ar, on], "posterior": zpost[ar, on]}
            zpr = states["posterior"]
            lift_a = torch.tensor([0., 1.], device=device).expand(len(idx), 2)
            with autocast(model.device.type, dtype=model.amp_dtype):
                for _ in range(9):
                    zpr = model.prior.step(zpr, lift_a).float()
            states["prior"] = zpr
            n_real = torch.from_numpy(data.real_len[idx]).to(device)
            fidx = on[:, None] + torch.arange(fut, device=device)[None]
            fidx = torch.minimum(fidx, (n_real - 1)[:, None])
            act = torch.from_numpy(data.action[idx]).to(device)
            e_fut = e_t[ar[:, None], fidx]
            a_fut = act[ar[:, None], fidx]
            with autocast(model.device.type, dtype=model.amp_dtype):
                for s, zz in states.items():
                    raw = model.reward_decoder(zz, e_fut, a_fut)
                    lg = model.reward_decoder.ball_logit(raw).float()
                    reads[f"{s}_r0"][idx] = lg[:, 0].cpu().numpy()
                    reads[f"{s}_r1"][idx] = lg[:, 1].cpu().numpy()

        # per-token |dz|, teacher and posterior
        dz_t = np.zeros((E, T, K), dtype=np.float32)
        dz_p = np.zeros((E, T, K), dtype=np.float32)
        dz_t[:, 1:] = np.linalg.norm((zt_all[:, 1:].astype(np.float32) - zt_all[:, :-1]), axis=-1)
        dz_p[:, 1:] = np.linalg.norm((zp_all[:, 1:].astype(np.float32) - zp_all[:, :-1]), axis=-1)

        # rest states: motionless HOLD frames after the cups are down -- the rendered scene is
        # identical across episodes there, so only the hidden ball slot (and history) can differ
        R, M = int(scfg["reveal_frames"]), int(scfg["move_frames"])
        tt = np.arange(T)[None].repeat(E, 0)
        real = tt < data.real_len[:, None]
        rest = real & (data.meta["phase"] == 1) & ~moving & (tt >= R + M)
        re, rt = np.nonzero(rest)
        on_e = np.arange(E)

        keys = [seq_key(s) for s in sched]
        seen = np.array([k in seen_keys for k in keys])
        np.savez_compressed(
            out / f"{name}__{tag}.npz",
            rest_ep=re, rest_t=rt,
            rest_teacher=zt_all[re, rt].reshape(len(re), -1),
            rest_posterior=zp_all[re, rt].reshape(len(re), -1),
            onset_teacher=zt_all[on_e, onset].reshape(E, -1),
            onset_posterior=zp_all[on_e, onset].reshape(E, -1),
            ball_slot=data.meta["ball_slot"].astype(np.int64), done=done, event=event,
            phase=data.meta["phase"], real_len=data.real_len, onset=onset,
            label=data.meta["ball_under_target"][on_e, onset].astype(bool),
            n_swaps=np.array([len(s["swaps"]) for s in sched]),
            start=np.array([s["start"] for s in sched]), seen=seen,
            gate=gate, gate_open=gate_open, dz_teacher=dz_t, dz_posterior=dz_p,
            gate_dim_mean=(gate_dim_sum / np.maximum(gate_dim_cnt, 1)[:, None]).astype(np.float32),
            gate_dim_cnt=gate_dim_cnt,
            **reads)
        print(f"[{name}] {tag}: {E} episodes, {len(re)} rest states, {int(seen.sum())} seen, "
              f"mask {mode}", flush=True)
    (out / f"{name}__meta.json").write_text(json.dumps(meta, indent=2))
    del model
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", action="append", required=True, help="NAME=RUN_DIR")
    p.add_argument("--zarr", action="append", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", required=True)
    p.add_argument("--chunk", type=int, default=16, help="episodes per GPU batch")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    zarrs = args.zarr or ["data/memtest4_eval.zarr", "data/memtest4_eval5.zarr"]
    for spec in args.arm:
        name, run = spec.split("=", 1)
        run_arm(name, run, zarrs, args.device, out, args.chunk)
    (out / "events.json").write_text(json.dumps(list(EVENTS)))


if __name__ == "__main__":
    main()
