"""docs/ONESTEP_FT.md: finetune ONLY the posterior against the frozen reward decoder, one step of credit.

    pixi run python scripts/finetune_onestep_phase1.py [--run <stage-1 run dir>] [--out <dir>]

One optimiser step per rollout. All training episodes are rolled out together, t = 0 .. T:

    z_0 = teacher(window ending at frame 0)                          frozen, no gradient
    z_t = posterior( detach(z_{t-1}), x_t, a_{t-1} )                 current weights
    r   = reward_decoder( z_t, (x_t,a_t) .. (x_{t+F-1},a_{t+F-1}) )  frozen (paired layout)
    L_t = sum BCE( r, ball revealed ) over scored read-outs / (total scored, known up front)
    L_t.backward() right away          -> grads accumulate = sum of one-step gradients, no BPTT
    optimizer.step() once at the end

Scored read-out: its frame is lift offset 1-9 of that episode, inside the episode, and the state is
at or before that episode's onset. Training data: the TEST set's 4-swap episodes (an ablation).

Stops when the eval loss (the objective on all training episodes, no gradient) has not improved by
--min_delta (relative) for --patience evaluations, or at --max_steps. Then runs the accuracy evals.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import zarr  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from torch.amp import autocast  # noqa: E402

from info_gathering.swap_sequences import sequences  # noqa: E402
from world_model.models.smt_prelim.embed_cache import read_embed_attrs  # noqa: E402
from world_model.utils.checkpoint import CheckpointManager  # noqa: E402

SHOW, HOLD, MOVE = 2, 1, 4


class Episodes:
    """Whole episodes from a store, padded and batched: normalised cached embeddings + actions."""

    def __init__(self, zarr_path, model, keep, device):
        z = zarr.open_group(str(ROOT / zarr_path), mode="r")
        read_embed_attrs(ROOT / zarr_path, model.embed_key, model.encoder_checkpoint)
        ends = z["episode_ends"][:]
        starts = np.concatenate([[0], ends[:-1]])
        raw, act, phase, but = z[model.embed_key][:], z["action"][:], z["phase"][:], z["ball_under_target"][:]
        eps = [e for e in range(len(ends)) if keep(e)]
        L = int(max(ends[e] - starts[e] for e in eps))
        n = len(eps)
        E = np.zeros((n, L, raw.shape[1]), np.float32)
        A = np.zeros((n, L, act.shape[1]), np.float32)
        self.length, self.onset, self.h0 = (np.zeros(n, np.int64) for _ in range(3))
        self.label = np.zeros(n, np.float32)
        for i, e in enumerate(eps):
            s, t = int(starts[e]), int(ends[e])
            E[i, :t - s], E[i, t - s:] = raw[s:t], raw[t - 1]      # pad: repeat the last frame
            A[i, :t - s], A[i, t - s:] = act[s:t], act[t - 1]
            ph = phase[s:t]
            self.length[i], self.onset[i] = t - s, int(np.flatnonzero(ph == SHOW)[0])
            self.h0[i], self.label[i] = int(np.flatnonzero(ph == HOLD)[0]), float(but[t - 1])
        PA = np.zeros_like(A)
        PA[:, 1:] = A[:, :-1]                                       # prev_action: zeros at t = 0
        mean, std = model.embed_mean.cpu().numpy(), model.embed_std.cpu().numpy()
        self.E = torch.from_numpy((E - mean) / std).to(device)
        self.A, self.PA = torch.from_numpy(A).to(device), torch.from_numpy(PA).to(device)
        self.n, self.L, self.ids = n, L, eps


def scored_mask(data, idx, t, F_len, device):
    """(B, F+1) bool: which read-outs r_t .. r_{t+F} of the state at t are scored."""
    on = torch.from_numpy(data.onset[idx]).to(device)
    ln = torch.from_numpy(data.length[idx]).to(device)
    frame = t + torch.arange(F_len + 1, device=device)[None]
    return (frame >= on[:, None] + 1) & (frame <= on[:, None] + 9) & (frame < ln[:, None]) & (t <= on)[:, None]


def run_episodes(model, data, idx, F_len, amp, train):
    """Roll the posterior t = 0..T over episodes idx. train=True: backward each L_t as it goes.

    Returns (loss value, number of scored read-outs, states (B, T+1, K, d) detached).
    """
    dev = data.E.device
    idx = np.asarray(idx)
    last = int(data.onset[idx].max())
    total = sum(int(scored_mask(data, idx, t, F_len, dev).sum()) for t in range(1, last + 1))
    E, A, PA = data.E[idx], data.A[idx], data.PA[idx]
    label = torch.from_numpy(data.label[idx]).to(dev)
    z = model.encode_windows(E[:, :1], PA[:, :1], torch.ones(len(idx), 1, dtype=torch.bool, device=dev))[:, -1]
    states, loss_sum = [z.detach()], 0.0
    with torch.set_grad_enabled(train):
        for t in range(1, last + 1):
            with autocast(dev.type, dtype=amp):
                z = model.posterior.step(z.detach(), E[:, t], PA[:, t]).float()
            states.append(z.detach())
            m = scored_mask(data, idx, t, F_len, dev)
            if not m.any():
                continue
            fidx = torch.clamp(torch.arange(t, t + F_len, device=dev), max=data.L - 1)
            with autocast(dev.type, dtype=amp):
                lg = model.reward_decoder.ball_logit(model.reward_decoder(z, E[:, fidx], A[:, fidx])).float()
            bce = F.binary_cross_entropy_with_logits(lg, label[:, None].expand_as(lg), reduction="none")
            loss_t = (bce * m).sum() / total
            if train:
                loss_t.backward()                     # frees this step's graph; grads accumulate
            loss_sum += float(loss_t)
    return loss_sum, total, torch.stack(states, 1)


@torch.no_grad()
def accuracy(model, data, F_len, amp):
    """Decoder accuracy at the first scored lift frame, read at the onset and from frame 13."""
    _, _, Z = run_episodes(model, data, np.arange(data.n), F_len, amp, train=False)
    out = {}
    for name, t_of in (("onset", data.onset), ("frame13", data.h0 + MOVE - 1)):
        ok = []
        for i in range(data.n):
            t = int(t_of[i])
            fidx = np.clip(np.arange(t, t + F_len), 0, data.L - 1)
            with autocast(Z.device.type, dtype=amp):
                lg = model.reward_decoder.ball_logit(model.reward_decoder(
                    Z[i:i + 1, t], data.E[i:i + 1, fidx], data.A[i:i + 1, fidx])).float()[0]
            ok.append(bool(lg[int(data.onset[i]) + 1 - t] > 0) == bool(data.label[i]))
        out[name] = float(np.mean(ok))
    return out


def checks(model, data, F_len, amp):
    """docs/ONESTEP_FT.md checks 2-4 (check 1 is the step-0 accuracy printout)."""
    dev = data.E.device
    params = list(model.posterior.parameters())
    # 2. gradient contract
    model.zero_grad(set_to_none=True)
    run_episodes(model, data, np.arange(4), F_len, amp, train=True)
    stray = [n for n, p in model.named_parameters() if p.grad is not None and not n.startswith("posterior.")]
    assert not stray, f"gradients reached frozen modules: {stray[:4]}"
    print("ok  check 2: only the posterior receives gradients")
    # 3. one-step credit: the gradient of L_t from the detached rollout == recomputing that one step
    idx = np.array([0])
    _, _, Z = run_episodes(model, data, idx, F_len, amp, train=False)
    t = int(data.onset[0])
    fidx = torch.clamp(torch.arange(t, t + F_len, device=dev), max=data.L - 1)
    label = torch.from_numpy(data.label[idx]).to(dev)
    m = scored_mask(data, idx, t, F_len, dev)

    def grad_of(z_prev):
        model.zero_grad(set_to_none=True)
        with autocast(dev.type, dtype=amp):
            z = model.posterior.step(z_prev, data.E[idx][:, t], data.PA[idx][:, t]).float()
            lg = model.reward_decoder.ball_logit(model.reward_decoder(z, data.E[idx][:, fidx], data.A[idx][:, fidx])).float()
        ((F.binary_cross_entropy_with_logits(lg, label[:, None].expand_as(lg), reduction="none") * m).sum()).backward()
        return torch.cat([p.grad.flatten() for p in params])

    g_step = grad_of(Z[:, t - 1].clone())
    # the same step reached by a fresh detached rollout (states recomputed, not reused)
    _, _, Z2 = run_episodes(model, data, idx, F_len, amp, train=False)
    g_roll = grad_of(Z2[:, t - 1].detach())
    gap = float((g_step - g_roll).abs().max() / g_step.abs().max().clamp(min=1e-12))
    assert gap < 1e-4, f"one-step gradient differs between rollout and single step: rel {gap:.2e}"
    print(f"ok  check 3: L_t's gradient equals recomputing only the step at t (rel gap {gap:.1e}); "
          f"inputs are detached, so nothing reaches earlier steps")
    # 4. batched == unbatched
    both = np.array([0, 1])
    lb, _, Zb = run_episodes(model, data, both, F_len, amp, train=False)
    for j, i in enumerate(both):
        _, _, Zi = run_episodes(model, data, np.array([i]), F_len, amp, train=False)
        T = Zi.shape[1]
        d = float((Zb[j, :T] - Zi[0]).abs().max())
        assert d < 5e-2, f"batched vs single episode {i}: state gap {d:.3e}"
        print(f"ok  check 4: episode {i} batched vs alone, max state gap {d:.1e} (bf16)")
    model.zero_grad(set_to_none=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="logdir/world_model/smt_phase1_memtest4/window60_phase1")
    ap.add_argument("--out", default="logdir/world_model/smt_phase1_memtest4/window60_phase1/onestep_ft")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max_steps", type=int, default=5000)
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--patience", type=int, default=10, help="evals without improvement before stopping")
    ap.add_argument("--min_delta", type=float, default=0.01, help="relative improvement that counts")
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()
    torch.manual_seed(0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    state = CheckpointManager.load(Path(args.run) / "checkpoints" / "wm_latest.pt")
    cfg = OmegaConf.create(state["config"])
    train_set, _ = instantiate(cfg.data)
    model = instantiate(cfg.model, action_dim=train_set.action_dim, device=args.device, _recursive_=False)
    model.load_train_state_dict(state["model"])
    model.requires_grad_(False).eval()
    model.posterior.requires_grad_(True)
    F_len, amp = int(model.fut_len), model.amp_dtype

    sw = zarr.open_group(str(ROOT / "data/memtest4_eval.zarr"), mode="r")["episode_swapped"][:]
    seen = set(sequences(ROOT / cfg.data_anchors.zarr_path)[:-int(cfg.data_anchors.holdout_episodes)])
    ev = sequences(ROOT / "data/memtest4_eval.zarr")
    train = Episodes("data/memtest4_eval.zarr", model, lambda e: sw[e] == 4, args.device)
    five = Episodes("data/memtest4_eval5.zarr", model, lambda e: True, args.device)
    sets = {"4sw trained-on": train,
            "4sw seen order": Episodes("data/memtest4_eval.zarr", model, lambda e: sw[e] == 4 and ev[e] in seen, args.device),
            "4sw unseen order": Episodes("data/memtest4_eval.zarr", model, lambda e: sw[e] == 4 and ev[e] not in seen, args.device),
            "5sw (never trained)": five,
            "0-3sw test": Episodes("data/memtest4_eval.zarr", model, lambda e: sw[e] <= 3, args.device)}

    def accuracies(tag):
        row = {name: accuracy(model, d, F_len, amp) for name, d in sets.items()}
        print(f"\n== accuracy {tag} (first scored lift frame; read at onset / from frame 13)")
        for k, v in row.items():
            print(f"   {k:<22} onset {v['onset']:.3f}   frame13 {v['frame13']:.3f}   (n={sets[k].n})")
        return row

    print(f"one-step finetune of the posterior on {train.n} test 4-swap episodes | lr {args.lr}", flush=True)
    before = accuracies("BEFORE (check 1: must match the earlier measurements)")
    checks(model, train, F_len, amp)

    params = list(model.posterior.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    all_idx = np.arange(train.n)
    log, best, since, t0 = [], float("inf"), 0, time.time()
    for step in range(1, args.max_steps + 1):
        opt.zero_grad(set_to_none=True)
        train_loss, _, _ = run_episodes(model, train, all_idx, F_len, amp, train=True)
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        opt.step()
        if step % args.eval_every == 0 or step == 1:
            with torch.no_grad():
                ev_loss, _, _ = run_episodes(model, train, all_idx, F_len, amp, train=False)
                five_loss, _, _ = run_episodes(model, five, np.arange(five.n), F_len, amp, train=False)
            row = {"step": step, "eval_loss": ev_loss, "five_swap_loss": five_loss, "train_loss": train_loss,
                   "grad_norm": gn, "sec": time.time() - t0}
            log.append(row)
            (out / "log.jsonl").open("a").write(json.dumps(row) + "\n")
            improved = ev_loss < best * (1 - args.min_delta)
            if improved:
                best, since = ev_loss, 0
                torch.save(model.posterior.state_dict(), out / "posterior_best.pt")
            else:
                since += 1
            print(f"[{step:5d}] eval loss {ev_loss:.4f} (best {best:.4f}, {since}/{args.patience} without "
                  f"improvement) | 5-swap loss {five_loss:.4f} | grad-norm {gn:.3f} | {time.time() - t0:.0f}s", flush=True)
            if since >= args.patience:
                print(f"plateau: no {args.min_delta:.0%} improvement for {args.patience} evals -> stop at step {step}")
                break
    torch.save(model.posterior.state_dict(), out / "posterior_last.pt")
    after = accuracies(f"AFTER (step {step}, last weights)")
    (out / "result.json").write_text(json.dumps({"before": before, "after": after, "log": log,
                                                 "args": vars(args)}, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
