"""Precision / coverage of POSITIVE pairs judged at real lifts (soft_pairs_phase1.py --tests lifts --dump).

    python3 scripts/js_lifts_analysis.py figures/js_pairs/dump_lifts_k64.npz [--thr 0.05]

A test = one real lift onset in one borrowed future; both memories are run there by the posterior and the decoder
answers p_x, p_y (read before the cup rises). JS in bits between the two answers. POSITIVE = every test < thr.
Scored with the hidden slot (shown starts) and the privileged belief class (all starts) ONLY here.
  precision  of the pairs called positive, the fraction truly the same (same cup / same belief class)
  coverage   of the truly-same pairs that have at least one test, the fraction called positive
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
    jf = np.where(ok, H((dx + dy) / 2) - (H(dx) + H(dy)) / 2, -1.0).max(1)        # most decisive test per future
    n = len(d["slot_x"])
    same = d["slot_x"] == d["slot_y"]; shown = d["shown_x"] & d["shown_y"]
    cx, cy = d["cls_x"], d["cls_y"]; lab = (cx >= 0) & (cy >= 0); bsame = cx == cy
    ph, mv = d["phase_x"], d["moving_x"]
    kinds = [("all", np.ones(n, bool)), ("opening reveal (cups up)", ph == 0), ("hold, cups still", (ph == 1) & ~mv),
             ("hold, cups moving (mid-swap)", (ph == 1) & mv), ("lift / peek", ph == 2)]

    def row(name, called, has, m):
        S, L = has & shown & m, has & lab & m
        ps = (called & S & same).sum() / max((called & S).sum(), 1)
        cs = (called & S & same).sum() / max((S & same).sum(), 1)
        pb = (called & L & bsame).sum() / max((called & L).sum(), 1)
        cb = (called & L & bsame).sum() / max((L & bsame).sum(), 1)
        print(f"{name:30s} {(called & m).sum():6d} | {(called & S).sum():6d} {ps:7.3f} {cs:7.3f} | {(called & L).sum():6d} {pb:7.3f} {cb:7.3f}")

    hdr = (f"{'':30s} {'called':>6s} | {'shown starts: n':>6s} {'prec':>7s} {'cover':>7s} | "
           f"{'all starts (belief): n':>6s} {'prec':>7s} {'cover':>7s}")
    print(f"{n} pairs; POSITIVE if every test has JS < {thr} bits")
    print("K sweep (first K futures per pair; future 0 is y's own):")
    print(f"{'K':>30s} " + hdr[31:])
    for K in (1, 2, 4, 8, 16, 32, 64):
        if K > Jf.max() + 1:
            break
        sel = fut < K
        mx = np.full(n, -1.0); np.maximum.at(mx, pid[sel], jf[sel])
        nt = np.bincount(pid[sel], weights=ok[sel].sum(1), minlength=n)
        has = nt > 0
        row(f"K={K} (tests/pair {np.median(nt[has]):.0f})", has & (mx < thr), has, np.ones(n, bool))
    mx = np.full(n, -1.0); np.maximum.at(mx, pid, jf)
    has = np.bincount(pid, weights=ok.sum(1), minlength=n) > 0
    print(f"\nK = all, by where x sits (phase used for scoring only):\n{'':30s} " + hdr[31:])
    for nm, m in kinds:
        row(nm, has & (mx < thr), has, m)


if __name__ == "__main__":
    main()
