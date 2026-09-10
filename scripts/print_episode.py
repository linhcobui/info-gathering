"""Print the ground-truth timeline of an episode in a readable, segmented form."""
import os, sys, json, argparse
import numpy as np
import zarr

PHASE = {0: "reveal", 1: "cover", 2: "pause", 3: "swap", 4: "lift"}

def ep_bounds(episode_ends, i):
    start = 0 if i == 0 else int(episode_ends[i - 1])
    end = int(episode_ends[i])
    return start, end

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", default="data/memtest0.zarr")
    ap.add_argument("--ep", type=int, default=0, help="episode index")
    args = ap.parse_args()

    z = zarr.open_group(args.zarr, mode="r")
    ends = z["episode_ends"][:]
    n = len(ends)
    if args.ep >= n:
        print(f"only {n} episodes (0..{n-1})"); return

    s, e = ep_bounds(ends, args.ep)
    ball = z["ball_slot"][s:e]
    order = z["cup_order"][s:e]
    act = z["action"][s:e]
    phase = z["phase"][s:e]
    T = e - s
    seed = int(z["episode_seeds"][args.ep])

    print(f"\n=== Episode {args.ep}  (seed={seed}, frames {s}..{e-1}, T={T}) ===")
    print(f"{'frames':>12} | {'phase':<7} | {'ball':^4} | {'cup_order':^11} | action")
    print("-" * 60)

    # collapse consecutive identical (phase, ball, order, action) rows into segments
    def key(t):
        return (int(phase[t]), int(ball[t]), tuple(order[t].tolist()), tuple(act[t].tolist()))

    seg_start = 0
    for t in range(1, T + 1):
        if t == T or key(t) != key(seg_start):
            k = key(seg_start)
            ph, b, od, ac = k
            frng = f"{seg_start:>4}-{t-1:<4}"
            print(f"{frng:>12} | {PHASE[ph]:<7} | {b:^4} | {str(list(od)):^11} | {list(ac)}")
            seg_start = t

    # summary: where does the ball end up, and what does each lift reveal?
    print("-" * 60)
    lifts = np.where((act[:, 2] == 1) & (phase == 4))[0]
    # find the pause-up frame of each lift (middle) to report reveal
    print("Lifts (action [0,0,1] on slot 2):")
    # group contiguous lift frames
    if len(lifts):
        groups = np.split(lifts, np.where(np.diff(lifts) > 1)[0] + 1)
        for g in groups:
            mid = g[len(g)//2]
            revealed = "BALL" if ball[mid] == 2 else "empty"
            print(f"  frames {g[0]}-{g[-1]}: slot 2 held cup_order[2]={order[mid,2]}, "
                  f"ball in slot {ball[mid]} -> lift reveals {revealed}")
    else:
        print("  (none)")
    print()

if __name__ == "__main__":
    main()
