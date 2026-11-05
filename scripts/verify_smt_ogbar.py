"""Checks for the OGBar-SMT model (docs/OGBAR_SMT.md, "Order of work" step 3), on real cached batches.

  1. gradient contract: each loss reaches exactly the modules in the table, and nothing else
  2. causality: perturbing step t+j+1 onwards leaves the decoder outputs for j' <= j and z*_t identical
  3. Q responds to x_{t+1}; P has no x_{t+1} input at all
  4. shapes, peak GPU memory and speed at the training batch size

    pixi run python scripts/verify_smt_ogbar.py                       # batch 128
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
import yaml  # noqa: E402

from info_gathering.ogbar_smt_data import AnchorBatches, OgbarSmtAnchors, collate  # noqa: E402
from world_model.models.smt_ogbar import SmtOgbar  # noqa: E402

CONTRACT = {          # term -> (must reach, must NOT reach)
    "decoder": ({"teacher", "decoder", "pool"}, {"posterior", "prior"}),
    "post": ({"posterior", "teacher"}, {"prior", "decoder"}),
    "prior": ({"prior"}, {"teacher", "posterior", "decoder", "pool"}),
    "unif": ({"teacher", "pool"}, {"posterior", "prior", "decoder"}),
    "align": ({"teacher", "pool"}, {"posterior", "prior", "decoder"}),     # checked only when coef_align > 0
    "align_push": ({"teacher", "pool"}, {"posterior", "prior", "decoder"}),
}


def to_dev(batch, device):
    out = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    out["patch"] = out["patch"].to(torch.bfloat16)
    return out


def reached(model):
    return {name for name, m in model.modules_by_term().items()
            if any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smt_ogbar/train_ogbar_v3_400.yaml")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / args.config))
    device = torch.device(args.device)
    data = OgbarSmtAnchors(ROOT / cfg["data"]["cache_dir"], ROOT / cfg["data"]["zarr_path"], cfg["data"]["frameskip"],
                           privileged=bool(cfg["data"].get("privileged", False)))
    g = torch.Generator().manual_seed(0)
    small = to_dev(collate([data[int(i)] for i in torch.randint(len(data), (4,), generator=g)]), device)
    torch.manual_seed(0)
    model = SmtOgbar(action_dim=data.action_dim, **cfg["model"]).to(device)
    print(f"{len(data)} anchors; {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    fails = []

    # 1. gradient contract (fp32, small batch)
    sm = dict(small, patch=small["patch"].float())
    for term, (must, must_not) in CONTRACT.items():
        if term.startswith("align") and model.cfg["coef_align"] <= 0:
            continue
        model.zero_grad(set_to_none=True)
        _, losses, _ = model(sm)
        losses[term].backward()
        got = reached(model)
        ok = must <= got and not (must_not & got)
        print(f"[{'ok' if ok else 'FAIL'}] {term:8s} reaches {sorted(got)}")
        if not ok:
            fails.append(f"contract {term}: reaches {sorted(got)}")
    model.zero_grad(set_to_none=True)

    # 2. causality (eval: no dropout; fp32)
    model.eval()
    with torch.no_grad():
        b = sm
        t, L = b["t"], b["length"]
        pooled = model.pool(b["patch"])
        b = dict(b, action=model.norm_action(b["action"]))
        z = model.teacher_windows(pooled, b["action"], t.unsqueeze(1))[:, 0]
        hid, _ = model.decode(z, pooled, b["action"], t, L)
        j0 = 5
        pert = b["patch"].clone()
        for i in range(len(t)):
            pert[i, int(t[i]) + j0 + 1:] += torch.randn_like(pert[i, int(t[i]) + j0 + 1:])
        pooled2 = model.pool(pert)
        z2 = model.teacher_windows(pooled2, b["action"], t.unsqueeze(1))[:, 0]
        hid2, _ = model.decode(z2, pooled2, b["action"], t, L)
        dz = float((z2 - z).abs().max())
        dh_before = float((hid2[:, :j0 + 1] - hid[:, :j0 + 1]).abs().max())
        dh_after = float((hid2[:, j0 + 1:] - hid[:, j0 + 1:]).abs().max())
        ok = dz == 0 and dh_before == 0 and dh_after > 0
        print(f"[{'ok' if ok else 'FAIL'}] causality: perturb steps >= t+{j0 + 1}: z*_t max|diff| {dz:.1e}, "
              f"decoder j<={j0} {dh_before:.1e}, j>{j0} {dh_after:.1e}")
        if not ok:
            fails.append("causality")
        # z*_{t+1} must see x_{t+1}
        pert = b["patch"].clone()
        ar = torch.arange(len(t), device=device)
        pert[ar, t + 1] += torch.randn_like(pert[ar, t + 1])
        zn = model.teacher_windows(model.pool(b["patch"]), b["action"], (t + 1).unsqueeze(1))
        zn2 = model.teacher_windows(model.pool(pert), b["action"], (t + 1).unsqueeze(1))
        print(f"       z*_t+1 responds to x_t+1: max|diff| {float((zn2 - zn).abs().max()):.2e}")

        # 3. Q responds to x_{t+1}; P has no such input
        x_t, x_n, a_t = b["patch"][ar, t], b["patch"][ar, t + 1], b["action"][ar, t]
        mq, _ = model.posterior(z, x_t, a_t, x_n)
        mq2, _ = model.posterior(z, x_t, a_t, x_n.flip(0))
        sens = float((mq - mq2).abs().mean())
        ok = sens > 0 and not model.prior.sees_next and not hasattr(model.prior, "next_in")
        print(f"[{'ok' if ok else 'FAIL'}] Q responds to x_t+1 (mean|diff| {sens:.3e}); P has no x_t+1 input")
        if not ok:
            fails.append("Q/P inputs")
        if model.cfg["updater_obs_input"] == "none":      # Q reads (z_t, a_t, x_t+1) and P (z_t, a_t): never x_t
            mq3, _ = model.posterior(z, x_t.flip(0), a_t, x_n)
            mp, _ = model.prior(z, x_t, a_t)
            mp3, _ = model.prior(z, x_t.flip(0), a_t)
            dq, dp = float((mq3 - mq).abs().max()), float((mp3 - mp).abs().max())
            ok = dq == 0 and dp == 0
            print(f"[{'ok' if ok else 'FAIL'}] Q and P ignore x_t: max|diff| Q {dq:.1e}, P {dp:.1e}")
            if not ok:
                fails.append("x_t reaches Q or P")
    model.train()

    # 4. memory and speed at the training batch size, real data, bf16 autocast
    sampler = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(data), args.batch, drop_last=True)
    loader = torch.utils.data.DataLoader(AnchorBatches(data), batch_size=None, sampler=sampler, num_workers=args.workers,
                                         prefetch_factor=1, pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    torch.cuda.reset_peak_memory_stats(device)
    it = iter(loader)
    times, waits = [], []
    for i in range(10):
        t0 = time.time()
        batch = to_dev(next(it), device)
        torch.cuda.synchronize(device)
        t1 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            total, losses, metrics = model(batch)
        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()
        torch.cuda.synchronize(device)
        if i >= 1:
            waits.append(t1 - t0)
            times.append(time.time() - t1)
    print(f"batch {args.batch}: patch {tuple(batch['patch'].shape)}; peak GPU {torch.cuda.max_memory_allocated(device) / 1e9:.1f} GB; "
          f"compute {sum(times) / len(times):.2f} s/step, waiting for data {sum(waits) / len(waits):.2f} s/step")
    print("losses " + " ".join(f"{k}={float(v.detach()):.4f}" for k, v in losses.items()) + " | "
          + " ".join(f"{k}={float(v):.3f}" for k, v in metrics.items()))
    if fails:
        raise SystemExit(f"FAILED: {fails}")
    print("all checks passed")


if __name__ == "__main__":
    main()
