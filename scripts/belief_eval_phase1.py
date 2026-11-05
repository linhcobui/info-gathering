"""ALL3-U belief eval: is the decoder's answer at the final lift the HONEST one when the start was never shown?

    pixi run python scripts/belief_eval_phase1.py --run <stage-1 run dir> [--zarr data/memtest4_evalu.zarr]
    pixi run python scripts/belief_eval_phase1.py --table "NAME=<run dir>" ... [--md out.md]

For every closed-start episode, the exact Bayes answer p* to "ball under the lifted cup?" at the final lift onset,
from the true information seen so far (scripts/belief_labels_phase1.truth_labels): no peek -> 1/3; peek showed the
ball -> certain; peek showed the cup empty -> the ball is in one of the other two, then the swaps -> 0 or 1/2.
Against it, the frozen reward decoder's p from the chained posterior (warm-up once, posterior over every real frame,
read at the onset before the cup moves: the posterior_r1 read-out of onset_probe_phase1.py) and, for diagnosis,
from a fresh teacher window. An episode PASSES when |p - p*| <= 0.1.
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

TOL = 0.1
CATS = [("1/3: nothing seen", (3,)),                       # belief classes, as belief_labels_phase1.SETS
        ("1/2: one of two, the lifted cup is one", (5, 6)),
        ("0: one of the other two (empty peek)", (4,)),
        ("0/1: known slot", (0, 1, 2))]
PSTAR = {0: 1.0, 1: 0.0, 2: 0.0, 3: 1 / 3, 4: 0.0, 5: 0.5, 6: 0.5}     # belief class -> P(slot 0)


def evaluate(args):
    import json
    import torch
    import zarr
    from torch.utils.data import default_collate
    from belief_labels_phase1 import truth_labels
    from info_gathering.dataset_memtest1 import memtest1_datasets
    from info_gathering.swap_sequences import sequences
    from onset_probe_phase1 import read_reward
    from train_mcl_prior_phase1 import load_model
    from world_model.training.loop import to_device

    run = ROOT / args.run
    model = load_model(Path(args.run) / "checkpoints" / "wm_latest.pt", args.device)
    F_ = int(model.fut_len)
    z = zarr.open_group(str(ROOT / args.zarr), mode="r")
    cfg = json.loads(z.attrs["config"])
    peek_len = 2 * int(cfg["move_frames"]) + int(cfg.get("peek_up_frames", 2))
    show = int(cfg["show_frames"])
    ends = z["episode_ends"][:]
    st = np.r_[0, ends[:-1]]
    seqs = sequences(ROOT / args.zarr)
    cup_rest = float(z["cup_z_max"][:].min())
    sw, peek = z["episode_swapped"][:], z["episode_peek"][:]

    ds, _ = memtest1_datasets(str(ROOT / args.zarr), image_size=64, holdout_episodes=0,
                              meta_keys=("ball_under_target", "ball_slot", "phase"))
    real_len = ds.arrays.steps - ds.arrays.meta["pad"].sum(1)
    rows = []
    for h in np.unique(real_len):
        eps = np.flatnonzero(real_len == h)
        batch = to_device(default_collate([ds[int(e)] for e in eps]), args.device)
        onset = int(h) - show                                    # the final lift: always the last show_frames
        assert (batch["action"][:, onset, 1] == 1).all() and (batch["action"][:, onset - 1, 1] == 0).all()
        p = {}
        for mode in ("recurrent", "window"):
            model.observe_mode = mode
            post = model.observe(batch)
            _, r1, _ = read_reward(model, post["z"][:, onset].float(), post["embed"], batch["action"], onset, F_, int(h))
            p[mode] = torch.sigmoid(r1.float()).cpu().numpy()
        model.observe_mode = "recurrent"
        for i, e in enumerate(eps):
            s, n = int(st[e]), int(ends[e] - st[e])
            cls = truth_labels(z, int(e), s, n, seqs, cup_rest, peek_len)[onset]
            label = bool(batch["meta/ball_under_target"][i, onset])
            rows.append(dict(episode=int(e), swaps=int(sw[e]), peek_after=int(peek[e]), belief_class=int(cls),
                             p_star=PSTAR[cls], label=int(label), p_posterior=float(p["recurrent"][i]),
                             p_teacher=float(p["window"][i])))
    out = run / args.out_name
    out.mkdir(parents=True, exist_ok=True)
    with (out / "per_episode.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} closed-start episodes -> {out / 'per_episode.csv'}")
    print(render({Path(args.run).name: rows}))


def load(run, out_name="belief_eval"):
    with (ROOT / run / out_name / "per_episode.csv").open() as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def render(runs, md=False):
    """Pass rate (|p - p*| <= 0.1) and mean p, by honest-answer category and depth band, posterior read-out."""
    bands = [("0-3 (trained)", lambda d: d <= 3), ("4-8 (OOD)", lambda d: d >= 4)]
    names = list(runs)
    lines = []
    head = "| honest answer | swaps | n | " + " | ".join(f"{n_} pass · mean p" for n_ in names) + " |"
    lines += [head, "|" + "---|" * (3 + len(names))]
    for cname, classes in CATS:
        for bname, bf in bands:
            cells, n_ = [], 0
            for nm in names:
                sel = [r for r in runs[nm] if int(r["belief_class"]) in classes and bf(r["swaps"])]
                n_ = len(sel)
                if not sel:
                    cells.append("–"); continue
                ok = np.mean([abs(r["p_posterior"] - r["p_star"]) <= TOL for r in sel])
                mp = np.mean([r["p_posterior"] for r in sel]) if not cname.startswith("0/1") else \
                    np.mean([abs(r["p_posterior"] - r["p_star"]) for r in sel])
                cells.append(f"{ok:.2f} · {mp:.2f}" + ("ᵉ" if cname.startswith("0/1") else ""))
            lines.append(f"| {cname} | {bname} | {n_} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(f"pass = |p − p*| ≤ {TOL} on the posterior chained to the final lift onset. "
                 "Mean p is the decoder's average answer; ᵉ = mean |p − p*| instead, since the certain cases mix 0 and 1.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--zarr", default="data/memtest4_evalu.zarr")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--table", nargs="*")
    ap.add_argument("--md")
    ap.add_argument("--out_name", default="belief_eval",
                    help="output folder inside the run (e.g. belief_eval_train for --zarr data/memtest4_all3u.zarr)")
    args = ap.parse_args()
    if args.table:
        runs = {spec.split("=", 1)[0]: load(spec.split("=", 1)[1], args.out_name) for spec in args.table}
        text = render(runs)
        print(text)
        if args.md:
            Path(ROOT / args.md).write_text(text + "\n")
        return
    evaluate(args)


if __name__ == "__main__":
    main()
