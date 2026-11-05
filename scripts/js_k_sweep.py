"""Precision / coverage of "same if max JS < thr" as a function of K (number of borrowed futures per pair).

    python3 scripts/js_k_sweep.py figures/js_pairs/dump_k64.npz [--thr 0.05]

Uses ONE dump with many futures per pair and evaluates each K by keeping each pair's first K futures (future 0 is
y's own). Precision / coverage / false-same on pairs where both frames come from shown-start episodes; uncertain
same-belief pairs reported separately. Truth (hidden slot, privileged belief class) is used only to score.
"""
import sys

import numpy as np


def H(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def main():
    d = np.load(sys.argv[1])
    thr = float(sys.argv[sys.argv.index("--thr") + 1]) if "--thr" in sys.argv else 0.05
    D = np.nanmean(d["P_all"], 0)
    Js, Jp, Jf = d["J_side"], d["J_pair"], d["J_fut"]
    rx = np.flatnonzero(Js == 0); ry = rx + 1
    dx, dy, pid, fut = D[rx], D[ry], Jp[rx], Jf[rx]
    ok = ~np.isnan(dx) & ~np.isnan(dy)
    js = np.where(ok, H((dx + dy) / 2) - (H(dx) + H(dy)) / 2, -1.0)          # -1 = no test at that step
    jmax_f = js.max(1)                                                        # most decisive test within each future
    n = len(d["slot_x"])
    same = d["slot_x"] == d["slot_y"]; shown = d["shown_x"] & d["shown_y"]
    cx, cy = d["cls_x"], d["cls_y"]; unc = (cx == cy) & (cx >= 3)
    print(f"{n} pairs, up to {Jf.max() + 1} futures each; rule: same if max JS over all tests < {thr} bits")
    print(f"{'K':>4s} {'tests/pair':>10s} {'precision':>10s} {'coverage':>9s} {'false-same':>11s} {'uncertain same-belief called same':>34s}")
    for K in (1, 2, 4, 8, 16, 32, 64):
        if K > Jf.max() + 1:
            break
        sel = fut < K
        mx = np.full(n, -1.0); np.maximum.at(mx, pid[sel], jmax_f[sel])
        nt = np.bincount(pid[sel], weights=ok[sel].sum(1), minlength=n)
        has = nt > 0
        called = has & (mx < thr)
        S = has & shown
        prec = (called & S & same).sum() / max((called & S).sum(), 1)
        print(f"{K:4d} {np.median(nt[has]):10.0f} {prec:10.3f} {(called & S & same).sum() / (S & same).sum():9.3f} "
              f"{(called & S & ~same).sum() / (S & ~same).sum():11.3f} {(called & unc).sum() / max((has & unc).sum(), 1):34.2f}")


if __name__ == "__main__":
    main()
