"""Step 4: candidate windows where the expected error is high.

Time convention. g_t scores err_t, the prediction of frame t+h from context ending at t (h = cfg.error_horizon; 1 =
the one-step error, h > 1 = open loop over h steps). Windows are placed on the *frames* that are expected to
surprise the WM: q_tau = g_{tau-h}. So a window [t_start, t_end] (inclusive model
frames) holds the outcome itself, the frame before it (t_start - 1) is the last one that hasn't shown it, and
resolvability (step 5) can use the task's literal recipe: blind context ends at t_start - 1, informed at t_end.

Methods (cfg.seg_method):
    hysteresis  open when q > on_thresh, stay open while q >= off_thresh
    entropy     binary entropy h(q) smoothed over entropy.window steps; its peaks (where q crosses 0.5, i.e. the
                edges of a high-q stretch) cut the episode into pieces, and a piece is a window if its mean q > on_thresh
Then: merge windows less than merge_gap steps apart, drop those shorter than min_len, cap at max_len (keep the
max_len-step stretch with the highest mean q). Windows never cross episodes.
"""
from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import find_peaks


@dataclass
class Segment:
    episode_id: str
    t_start: int
    t_end: int
    onset_score: float

    @property
    def length(self):
        return self.t_end - self.t_start + 1

    def as_dict(self):
        return asdict(self)


def smooth(q, w):
    """Centred moving average over w steps; edges average over the steps that exist."""
    if w <= 1:
        return np.asarray(q, dtype=float)
    k = np.ones(int(w))
    return np.convolve(q, k, "same") / np.convolve(np.ones(len(q)), k, "same")


def hysteresis(q, on, off):
    """Index intervals [s, e] (inclusive) of q's runs: open at q > on, close once q < off."""
    if off > on:
        raise ValueError(f"off_thresh {off} must not exceed on_thresh {on}")
    out, start = [], None
    for i, v in enumerate(q):
        if start is None and v > on:
            start = i
        elif start is not None and v < off:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(q) - 1))
    return out


def binary_entropy(q):
    q = np.clip(np.asarray(q, dtype=float), 1e-12, 1 - 1e-12)
    return -(q * np.log2(q) + (1 - q) * np.log2(1 - q))


def entropy_windows(q, on, window, min_prominence):
    h = smooth(binary_entropy(q), window)
    peaks, _ = find_peaks(np.concatenate([[0.0], h, [0.0]]), prominence=min_prominence)
    cuts = np.unique(np.concatenate([[0], peaks - 1, [len(q)]]))     # piece i = [cuts[i], cuts[i+1])
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b > a and np.mean(q[a:b]) > on:
            out.append((int(a), int(b) - 1))
    return out


def merge(intervals, gap):
    """Merge intervals separated by fewer than `gap` steps (steps strictly between them)."""
    out = []
    for s, e in sorted(intervals):
        if out and s - out[-1][1] - 1 < gap:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def cap(intervals, q, max_len):
    out = []
    for s, e in intervals:
        if e - s + 1 > max_len:
            m = np.convolve(q[s:e + 1], np.ones(max_len), "valid")
            s = s + int(np.argmax(m))
            e = s + max_len - 1
        out.append((s, e))
    return out


def segment_series(q, cfg):
    """Index intervals for one episode's q series (already in frame order, contiguous)."""
    q = np.asarray(q, dtype=float)
    qs = smooth(q, cfg.smooth_window)
    if cfg.seg_method == "hysteresis":
        iv = hysteresis(qs, cfg.on_thresh, cfg.off_thresh)
    elif cfg.seg_method == "entropy":
        iv = entropy_windows(qs, cfg.on_thresh, cfg.entropy.window, cfg.entropy.min_prominence)
    else:
        raise ValueError(f"seg_method must be hysteresis or entropy, got {cfg.seg_method}")
    iv = merge(iv, cfg.merge_gap)
    iv = [(s, e) for s, e in iv if e - s + 1 >= cfg.min_len]
    return cap(iv, qs, cfg.max_len), qs


def onset(qs, s, e, how):
    if how == "peak":
        return float(np.max(qs[s:e + 1]))
    if how == "mean":
        return float(np.mean(qs[s:e + 1]))
    raise ValueError(f"onset_score must be peak or mean, got {how}")


def segment_all(scores, cfg):
    """scores: table with episode_id, t, g (one row per error step) -> list[Segment] in frame coordinates."""
    segs = []
    for eid, df in scores.sort_values(["episode_id", "t"]).groupby("episode_id", sort=False):
        frames = df.t.to_numpy() + cfg.error_horizon     # q_tau = g_{tau - h}
        if len(frames) and not np.array_equal(frames, np.arange(frames[0], frames[0] + len(frames))):
            raise ValueError(f"{eid}: g scores are not contiguous in t")
        iv, qs = segment_series(df.g.to_numpy(), cfg)
        segs += [Segment(eid, int(frames[s]), int(frames[e]), onset(qs, s, e, cfg.onset_score)) for s, e in iv]
    return segs
