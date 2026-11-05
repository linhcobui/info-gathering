"""DELTA diagnostics, stage 2 of 2: probes, t-SNEs and heatmaps from delta_extract.py's dumps.

Runs under the SYSTEM python (needs scikit-learn + matplotlib, which the pixi env lacks):

    python3 scripts/delta_analyze.py logdir/delta_diag/<run> [--arms D-base,D-tok-lo,...]

Writes report.md, results.json and figures/ into the same directory.

    onset         reward head at the lift onset, r1 (memory + a_t, no frame), F1 / acc by
                  depth, seen/unseen, and five swaps -- teacher, posterior, prior
    history       docs/DELTA.md 1: among rest states sharing the ball slot, probe swaps-so-far and
                  start slot (episode-grouped CV, PCA-64 + logistic regression), and the share
                  of within-slot variance explained by depth
    blocks        docs/DELTA.md 7.2: slot and swaps-so-far probed from ONE memory token at a time
    gates         docs/DELTA.md 7.1: mean gate per (event type x token); |dz| per (event x token)
    t-SNE         rest states, teacher vs posterior, coloured by slot / swaps-so-far / start
"""
import argparse
import json
import os
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")     # the box is shared; unbounded BLAS threads thrash it

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
# No StandardScaler anywhere: z is rms-normed per token, so raw coordinates are the native
# metric, and per-dim scaling blows up the fp16 jitter of tokens a sparse gate never writes.

SOURCES = ("teacher", "posterior", "prior")
SLOT_COLORS = ["#1f77b4", "#d62728", "#2ca02c"]


def f1_acc(pred, y):
    pred, y = np.asarray(pred, bool), np.asarray(y, bool)
    if len(y) == 0:
        return None
    tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
    f1 = 2 * tp / max(2 * tp + fp + fn, 1) if (2 * tp + fp + fn) else 1.0
    return {"f1": f1, "acc": float((pred == y).mean()), "n": int(len(y))}


def grouped_probe(X, y, groups, n_splits=5, pca=64):
    """Episode-grouped CV accuracy of PCA + logistic regression, and the majority baseline."""
    classes = np.unique(y)
    if len(classes) < 2 or len(np.unique(groups)) < n_splits:
        return None, None
    correct, total, maj = 0, 0, 0
    for tr, te in GroupKFold(n_splits).split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            pred = np.full(len(te), np.bincount(y[tr]).argmax())
        else:
            k = min(pca, X.shape[1], len(tr) - 1)
            clf = make_pipeline(PCA(k, svd_solver="randomized", random_state=0),
                                LogisticRegression(max_iter=2000, C=1.0))
            clf.fit(X[tr], y[tr])
            pred = clf.predict(X[te])
        correct += int((pred == y[te]).sum())
        maj += int((y[te] == np.bincount(y[tr]).argmax()).sum())
        total += len(te)
    return correct / total, maj / total


def within_slot(X, slot, ep, target):
    """Probe `target` separately inside each ball-slot group; pooled accuracy and majority."""
    acc_n, maj_n, n = 0.0, 0.0, 0
    for s in range(3):
        sel = slot == s
        a, m = grouped_probe(X[sel], target[sel], ep[sel])
        if a is None:
            continue
        acc_n += a * sel.sum(); maj_n += m * sel.sum(); n += sel.sum()
    return (acc_n / n, maj_n / n) if n else (None, None)


def depth_share(X, slot, depth):
    """Share of within-slot variance explained by depth: sum SS_between(depth) / sum SS_within-slot."""
    ssb, sst = 0.0, 0.0
    for s in range(3):
        Xs, ds = X[slot == s], depth[slot == s]
        if len(Xs) < 2:
            continue
        mu = Xs.mean(0)
        sst += ((Xs - mu) ** 2).sum()
        for d in np.unique(ds):
            Xd = Xs[ds == d]
            ssb += len(Xd) * ((Xd.mean(0) - mu) ** 2).sum()
    return ssb / max(sst, 1e-12)


def load(root, arm, tag):
    p = root / f"{arm}__{tag}.npz"
    return dict(np.load(p, allow_pickle=False)) if p.exists() else None


def onset_tables(arms, root):
    res = {}
    for arm in arms:
        d, d5 = load(root, arm, "memtest4_eval"), load(root, arm, "memtest4_eval5")
        r = {}
        for s in SOURCES:
            pred, y = d[f"{s}_r1"] > 0, d["label"]
            row = {"all": f1_acc(pred, y), "seen": f1_acc(pred[d["seen"]], y[d["seen"]]),
                   "unseen": f1_acc(pred[~d["seen"]], y[~d["seen"]])}
            for k in range(5):
                sel = d["n_swaps"] == k
                row[f"d{k}"] = f1_acc(pred[sel], y[sel])
                row[f"d{k}_unseen"] = f1_acc(pred[sel & ~d["seen"]], y[sel & ~d["seen"]])
            sel = (d["n_swaps"] >= 2) & ~d["seen"]
            row["d2-4_unseen"] = f1_acc(pred[sel], y[sel])
            if d5 is not None:
                row["d5"] = f1_acc(d5[f"{s}_r1"] > 0, d5["label"])
            r[s] = row
        res[arm] = r
    return res


def native_probe(arms, root, metas):
    """Decoder-free read of the ONSET state: logistic regression (PCA-64) fit on the arm's own
    TRAINING-split onset states, tested on eval / eval5 -- separately for teacher and posterior.

    The reward decoder was only ever trained on teacher states. If the gated posterior drifts off
    that manifold the decoder misreads it even when the answer is in the state; this probe is
    fit on the source's own states, so it asks only whether the answer is linearly there.
    (Diagnostic only: the probe uses the label; training never did.)
    """
    res = {}
    for arm in arms:
        m = metas[arm]
        tr = load(root, arm, Path(m["train_zarr"]).stem)
        if tr is None:
            continue
        n_tr = int(m["n_train"])
        d, d5 = load(root, arm, "memtest4_eval"), load(root, arm, "memtest4_eval5")
        r = {}
        for s in ("teacher", "posterior"):
            X = tr[f"onset_{s}"][:n_tr].astype(np.float32)
            y = tr["label"][:n_tr].astype(int)
            clf = make_pipeline(PCA(min(64, len(X) - 1), svd_solver="randomized",
                                                      random_state=0),
                                LogisticRegression(max_iter=3000, C=1.0))
            clf.fit(X, y)
            pred = clf.predict(d[f"onset_{s}"].astype(np.float32)).astype(bool)
            lab = d["label"]
            row = {"all": f1_acc(pred, lab), "seen": f1_acc(pred[d["seen"]], lab[d["seen"]]),
                   "unseen": f1_acc(pred[~d["seen"]], lab[~d["seen"]])}
            for k in range(5):
                sel = d["n_swaps"] == k
                row[f"d{k}"] = f1_acc(pred[sel], lab[sel])
            if d5 is not None:
                row["d5"] = f1_acc(clf.predict(d5[f"onset_{s}"].astype(np.float32)).astype(bool),
                                   d5["label"])
            r[s] = row
        res[arm] = r
    return res


def closure_table(arms, root):
    """R^2 of the posterior ONSET state against the teacher's, by depth (eval) and at 5 swaps."""
    res = {}
    for arm in arms:
        d, d5 = load(root, arm, "memtest4_eval"), load(root, arm, "memtest4_eval5")
        zt, zp = d["onset_teacher"].astype(np.float32), d["onset_posterior"].astype(np.float32)
        var = ((zt - zt.mean(0)) ** 2).sum(1).mean()
        err = ((zp - zt) ** 2).sum(1)
        r = {f"d{k}": float(1 - err[d["n_swaps"] == k].mean() / var) for k in range(5)}
        r["all"] = float(1 - err.mean() / var)
        if d5 is not None:
            t5, p5 = d5["onset_teacher"].astype(np.float32), d5["onset_posterior"].astype(np.float32)
            r["d5"] = float(1 - ((p5 - t5) ** 2).sum(1).mean() / var)
        res[arm] = r
    return res


def history_tables(arms, root):
    res = {}
    for arm in arms:
        d = load(root, arm, "memtest4_eval")
        ep, t = d["rest_ep"], d["rest_t"]
        slot = d["ball_slot"][ep, t]
        done = d["done"][ep, t]
        start = d["start"][ep]
        r = {}
        for s in ("teacher", "posterior"):
            X = d[f"rest_{s}"].astype(np.float32)
            a_done, m_done = within_slot(X, slot, ep, done)
            a_start, m_start = within_slot(X, slot, ep, start)
            a_slot, m_slot = grouped_probe(X, slot, ep)
            r[s] = {"swaps_so_far": a_done, "swaps_so_far_majority": m_done,
                    "start_slot": a_start, "start_slot_majority": m_start,
                    "depth_var_share": depth_share(X, slot, done),
                    "slot": a_slot, "slot_majority": m_slot, "n": int(len(X))}
        res[arm] = r
    return res


def block_tables(arms, root, K=10):
    res = {}
    for arm in arms:
        d = load(root, arm, "memtest4_eval")
        ep, t = d["rest_ep"], d["rest_t"]
        slot, done = d["ball_slot"][ep, t], d["done"][ep, t]
        X = d["rest_posterior"].astype(np.float32).reshape(len(ep), K, -1)
        rows = []
        for k in range(K):
            a_slot, _ = grouped_probe(X[:, k], slot, ep, pca=32)
            a_done, _ = within_slot(X[:, k], slot, ep, done)
            rows.append({"token": k, "slot": a_slot, "swaps_so_far_within_slot": a_done})
        res[arm] = rows
    return res


def event_matrix(d, key, events, K=10):
    """mean of d[key] (E, T, K) per event type -> (n_events, K)."""
    ev = d["event"]
    M = np.full((len(events), K), np.nan)
    for i in range(len(events)):
        sel = ev == i
        if sel.any():
            M[i] = np.nanmean(d[key][sel], axis=0)
    return M


def plot_heat(M, events, title, path, vmax=None, fmt="{:.2f}"):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    im = ax.imshow(M, aspect="auto", cmap="magma", vmin=0, vmax=vmax)
    ax.set_yticks(range(len(events))); ax.set_yticklabels(events)
    ax.set_xticks(range(M.shape[1])); ax.set_xlabel("memory token")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                v = M[i, j]
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=6.5,
                        color="white" if v < 0.6 * (vmax or np.nanmax(M)) else "black")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def tsne_figure(arm, d, path, max_points=1600, seed=0):
    ep, t = d["rest_ep"], d["rest_t"]
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(len(ep), size=min(max_points, len(ep)), replace=False))
    ep, t = ep[pick], t[pick]
    colourings = {
        "ball slot": (d["ball_slot"][ep, t], "tab10", 3),
        "swaps so far": (d["done"][ep, t], "viridis", 5),
        "start slot": (d["start"][ep], "tab10", 3),
    }
    unseen = ~d["seen"][ep]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.4))
    for r, s in enumerate(("teacher", "posterior")):
        X = d[f"rest_{s}"][pick].astype(np.float32)
        X = PCA(50, random_state=seed).fit_transform(X)
        Y = TSNE(2, perplexity=30, init="pca", random_state=seed, n_iter=1500).fit_transform(X)
        for c, (cname, (lab, cmap, n)) in enumerate(colourings.items()):
            ax = axes[r, c]
            if cmap == "tab10":
                cols = np.array(SLOT_COLORS)[lab]
            else:
                cols = plt.get_cmap(cmap)(lab / max(n - 1, 1))
            ax.scatter(Y[~unseen, 0], Y[~unseen, 1], c=cols[~unseen], s=5, alpha=0.75, linewidths=0)
            ax.scatter(Y[unseen, 0], Y[unseen, 1], c=cols[unseen], s=22, marker="x", linewidths=0.8)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{s} — {cname}", fontsize=10)
            handles = [plt.Line2D([], [], marker="o", ls="", color=(SLOT_COLORS[i] if cmap == "tab10"
                                   else plt.get_cmap(cmap)(i / max(n - 1, 1))), label=str(i))
                       for i in range(n)]
            ax.legend(handles=handles, fontsize=7, loc="best", title=cname, title_fontsize=7)
    fig.suptitle(f"{arm}: rest states (motionless hold frames), memtest4_eval. "
                 f"x = UNSEEN swap sequence", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def tsne_onset_grid(arms, root, path, seed=0):
    """One panel per (arm, state source): ONSET states, eval + eval5, coloured by ball slot."""
    fig, axes = plt.subplots(2, len(arms), figsize=(3.6 * len(arms), 7.2), squeeze=False)
    for c, arm in enumerate(arms):
        d, d5 = load(root, arm, "memtest4_eval"), load(root, arm, "memtest4_eval5")
        for r, s in enumerate(("teacher", "posterior")):
            X = np.concatenate([d[f"onset_{s}"], d5[f"onset_{s}"]]).astype(np.float32)
            slot = np.concatenate([d["ball_slot"][np.arange(len(d["onset"])), d["onset"]],
                                   d5["ball_slot"][np.arange(len(d5["onset"])), d5["onset"]]])
            kind = np.concatenate([np.where(d["seen"], 0, 1), np.full(len(d5["onset"]), 2)])
            X = PCA(50, random_state=seed).fit_transform(X)
            Y = TSNE(2, perplexity=30, init="pca", random_state=seed, n_iter=1500).fit_transform(X)
            ax = axes[r, c]
            for k, (mk, sz, lab) in enumerate((("o", 8, "seen"), ("x", 26, "unseen 2-4"),
                                               ("^", 26, "5 swaps"))):
                sel = kind == k
                ax.scatter(Y[sel, 0], Y[sel, 1], c=np.array(SLOT_COLORS)[slot[sel]], s=sz,
                           marker=mk, linewidths=0.8 if mk != "o" else 0, alpha=0.8, label=lab)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{arm} — {s}", fontsize=10)
            if r == 0 and c == 0:
                ax.legend(fontsize=7, loc="best")
    fig.suptitle("Lift-onset states (eval 315 + eval5 63), colour = true ball slot "
                 "(blue 0, red 1, green 2)", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fmt(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def cell(m):
    return "—" if m is None else f"{m['f1']:.3f} ({m['n']})"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--arms", default=None)
    p.add_argument("--no-tsne", action="store_true")
    p.add_argument("--quick", action="store_true", help="onset + history + gates only")
    args = p.parse_args()
    root = Path(args.root)
    events = json.loads((root / "events.json").read_text())
    found = sorted({f.name.split("__")[0] for f in root.glob("*__meta.json")})
    arms = args.arms.split(",") if args.arms else found
    metas = {a: json.loads((root / f"{a}__meta.json").read_text()) for a in arms}
    figs = root / "figures"
    figs.mkdir(exist_ok=True)

    onset = onset_tables(arms, root)
    hist = history_tables(arms, root)
    blocks = {} if args.quick else block_tables(arms, root)

    L = ["# DELTA diagnostics", "",
         "Eval set `memtest4_eval` (315 episodes, 271 seen / 44 unseen for 0–4-swap-trained arms); "
         "`memtest4_eval5` (63 episodes, five swaps, all unseen). Onset read-out r1 = memory + "
         "lift action, no frame. teacher = 60-frame window encode at the onset; posterior = one "
         "encode at t=0, then the gated posterior over every frame; prior = posterior, then 9 "
         "gated prior steps.", "",
         "| arm | run | step | mask | λ |", "|---|---|---|---|---|"]
    for a in arms:
        m = metas[a]
        L.append(f"| {a} | `{Path(m['run']).name}` | {m['step']} | {m['mask']} | {m['coef_delta']} |")

    L += ["", "## Onset probe — F1 (n)", ""]
    for s in SOURCES:
        L += [f"### {s}", "",
              "| arm | all | seen | UNSEEN | 0 | 1 | 2 | 3 | 4 | 4 UNSEEN | 2–4 UNSEEN | 5 swaps | acc all | acc 5 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for a in arms:
            r = onset[a][s]
            L.append(f"| {a} | {cell(r['all'])} | {cell(r['seen'])} | {cell(r['unseen'])} | "
                     + " | ".join(fmt(r[f'd{k}']['f1']) for k in range(5))
                     + f" | {cell(r['d4_unseen'])} | {cell(r['d2-4_unseen'])} | {cell(r.get('d5'))} | "
                     f"{fmt(r['all']['acc'])} | {fmt(r['d5']['acc']) if r.get('d5') else '—'} |")
        L.append("")

    native = native_probe(arms, root, metas)
    if native:
        L += ["### decoder-free onset probe — F1 (n)", "",
              "Logistic regression (PCA-64) on the arm's own TRAINING-split onset states, per "
              "source, tested on eval. Asks whether the answer is linearly in the state, "
              "independent of the reward decoder (which only ever saw teacher states).", "",
              "| arm | state | all | seen | UNSEEN | 0 | 1 | 2 | 3 | 4 | 5 swaps |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for a in native:
            for s in ("teacher", "posterior"):
                r = native[a][s]
                L.append(f"| {a} | {s} | {cell(r['all'])} | {cell(r['seen'])} | {cell(r['unseen'])} | "
                         + " | ".join(fmt(r[f'd{k}']['f1']) for k in range(5))
                         + f" | {cell(r.get('d5'))} |")
        L.append("")
    closure = closure_table(arms, root)
    L += ["### closure at the onset — R² of posterior state vs teacher state", "",
          "1 − mean‖z_post − z_teacher‖² / Var(z_teacher), variance over eval onset states.", "",
          "| arm | all | 0 | 1 | 2 | 3 | 4 | 5 swaps |", "|---|---|---|---|---|---|---|---|"]
    for a in arms:
        c = closure[a]
        L.append(f"| {a} | {c['all']:.3f} | " + " | ".join(f"{c[f'd{k}']:.3f}" for k in range(5))
                 + f" | {c.get('d5', float('nan')):.3f} |")
    L.append("")
    L += ["## History leakage (docs/DELTA.md 1) — rest states, memtest4_eval", "",
          "Probes are fit WITHIN each ball-slot group (episode-grouped 5-fold CV, PCA-64 + "
          "logistic regression), then pooled. Majority baseline in parentheses. Lower = less "
          "history. `depth share` = within-slot variance explained by swaps-so-far.", "",
          "| arm | state | n | slot probe | swaps-so-far | start slot | depth share |",
          "|---|---|---|---|---|---|---|"]
    for a in arms:
        for s in ("teacher", "posterior"):
            h = hist[a][s]
            L.append(f"| {a} | {s} | {h['n']} | {fmt(h['slot'])} | {fmt(h['swaps_so_far'])} "
                     f"({fmt(h['swaps_so_far_majority'])}) | {fmt(h['start_slot'])} "
                     f"({fmt(h['start_slot_majority'])}) | {100 * h['depth_var_share']:.1f}% |")

    L += ["", "## Blocks (docs/DELTA.md 7.2) — one memory token at a time, posterior rest states", "",
          "Slot probe from that token alone (chance ~0.33) / swaps-so-far within slot.", "",
          "| arm | " + " | ".join(f"tok {k}" for k in range(10)) + " |",
          "|---|" + "---|" * 10]
    for a in blocks:
        L.append(f"| {a} | " + " | ".join(f"{fmt(b['slot'], 2)} / {fmt(b['swaps_so_far_within_slot'], 2)}"
                                          for b in blocks[a]) + " |")

    L += ["", "## Gates and per-token change by event type (docs/DELTA.md 7.1)", ""]
    gate_res = {}
    for a in arms:
        d = load(root, a, "memtest4_eval")
        dz = event_matrix(d, "dz_posterior", events)
        dzt = event_matrix(d, "dz_teacher", events)
        plot_heat(dz, events, f"{a}: posterior ||z_t - z_t-1|| per token", figs / f"{a}_dz_posterior.png",
                  fmt="{:.1f}")
        plot_heat(dzt, events, f"{a}: teacher ||z_t - z_t-1|| per token", figs / f"{a}_dz_teacher.png",
                  fmt="{:.1f}")
        g = {"dz_posterior": dz.tolist(), "dz_teacher": dzt.tolist()}
        L.append(f"**{a}** — posterior |Δz| per token (rows = event, mean over frames):")
        L.append("")
        L.append("| event | " + " | ".join(str(k) for k in range(10)) + " | total |")
        L.append("|---|" + "---|" * 11)
        for i, e in enumerate(events):
            if np.isfinite(dz[i]).all():
                L.append(f"| {e} | " + " | ".join(f"{v:.2f}" for v in dz[i]) + f" | {dz[i].sum():.2f} |")
        L.append("")
        if np.isfinite(d["gate"]).any():
            G = event_matrix(d, "gate", events)
            O = event_matrix(d, "gate_open", events)
            plot_heat(G, events, f"{a}: posterior gate m by event (colour auto-scaled)",
                      figs / f"{a}_gate.png", fmt="{:.3f}")
            g.update({"gate": G.tolist(), "gate_open": O.tolist()})
            L.append(f"**{a}** — posterior gate m (mean) per token:")
            L.append("")
            L.append("| event | " + " | ".join(str(k) for k in range(10)) + " | frac>0.5 |")
            L.append("|---|" + "---|" * 11)
            for i, e in enumerate(events):
                if np.isfinite(G[i]).all():
                    L.append(f"| {e} | " + " | ".join(f"{v:.3f}" for v in G[i]) + f" | {np.nanmean(O[i]):.3f} |")
            L.append("")
            # WHEN within a swap does the gate open? position = frames since the swap's first
            # moving frame (runs of consecutive swap-event frames)
            sw = np.isin(d["event"], [events.index(e) for e in events if e.startswith("swap")])
            pos = np.full(sw.shape, -1)
            for e_i in range(sw.shape[0]):
                run = 0
                for t_i in range(sw.shape[1]):
                    run = run + 1 if sw[e_i, t_i] else 0
                    if sw[e_i, t_i]:
                        pos[e_i, t_i] = (run - 1) % 6
            rows = []
            for k in range(6):
                sel = pos == k
                if sel.any():
                    gk = d["gate"][sel]
                    rows.append(f"| {k} | {int(sel.sum())} | {np.nanmean(gk):.3f} | {np.nanmean(gk.max(-1)):.3f} | "
                                f"{np.nanmean((gk > 0.5).sum(-1)):.2f} |")
            g["gate_by_swap_position"] = rows
            L += [f"Gate by frame position within a swap ({a}):", "",
                  "| position | n | mean m | max-token m | tokens > 0.5 |", "|---|---|---|---|---|"] + rows + [""]
            if "prior_gate" in metas[a]:
                pg = np.array(metas[a]["prior_gate"])
                L.append("Prior gate m_p(a) per token: " + "; ".join(
                    f"{n} [" + " ".join(f"{v:.3f}" for v in pg[i]) + "]"
                    for i, n in enumerate(("reveal [0,0]", "hold [1,0]", "show [0,1]"))))
                L.append("")
            if metas[a]["mask"] == "dim":
                gd = d["gate_dim_mean"]
                fig, ax = plt.subplots(figsize=(12, 3.6))
                ax.imshow(gd, aspect="auto", cmap="magma", vmin=0, vmax=1, interpolation="nearest")
                ax.set_yticks(range(len(events))); ax.set_yticklabels(events)
                ax.set_xticks(np.arange(0, 2560, 256)); ax.set_xlabel("dimension (token*256 + channel)")
                ax.set_title(f"{a}: dim gate by event"); fig.tight_layout()
                fig.savefig(figs / f"{a}_gate_dims.png", dpi=130); plt.close(fig)
        gate_res[a] = g

    if not (args.no_tsne or args.quick):
        for a in arms:
            tsne_figure(a, load(root, a, "memtest4_eval"), figs / f"{a}_tsne_rest.png")
        tsne_onset_grid(arms, root, figs / "tsne_onset_all_arms.png")
        L += ["## t-SNE", "", "- `figures/<arm>_tsne_rest.png`: rest states, teacher (top) vs "
              "posterior (bottom), coloured by ball slot / swaps so far / start slot; x = unseen.",
              "- `figures/tsne_onset_all_arms.png`: lift-onset states, every arm, colour = slot.", ""]

    (root / "report.md").write_text("\n".join(L) + "\n")
    (root / "results.json").write_text(json.dumps(
        {"onset": onset, "native_probe": native, "closure": closure, "history": hist, "blocks": blocks, "gates": gate_res, "meta": metas},
        indent=1, default=float))
    print("\n".join(L))


if __name__ == "__main__":
    main()
