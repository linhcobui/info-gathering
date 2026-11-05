"""Step 2: episode-level split, the error threshold lambda and labels y_t = 1[err_t > lambda].

The WM trained on every wm_train episode, so their errors are optimistic: lambda is fit on held-out episodes only
(`lambda_fit_splits` of the held-out split), and g trains on held-out episodes only.
"""
import numpy as np
import pandas as pd

WM_TRAIN = "wm_train"          # split name of the episodes the WM trained on
SPLITS = ("train", "val", "test")


def episode_split(episodes_table, fractions, seed):
    """{episode_id: 'train' | 'val' | 'test' | 'wm_train'}; held-out episodes shuffled with `seed`, by episode."""
    eps = episodes_table.drop_duplicates("episode_id")
    held = np.array(sorted(eps.loc[~eps.wm_train_episode, "episode_id"]))
    held = held[np.random.default_rng(seed).permutation(len(held))]
    f = np.asarray(fractions, dtype=float)
    cuts = np.round(np.cumsum(f / f.sum()) * len(held)).astype(int)
    split = {e: WM_TRAIN for e in eps.loc[eps.wm_train_episode, "episode_id"]}
    for name, part in zip(SPLITS, np.split(held, cuts[:-1])):
        split.update({e: name for e in part})
    return split


def normalised_metric(errors, metric, mode, window=9):
    """The metric used for thresholding: raw (global), z-scored within each episode (episode), or minus a centred
    rolling median over `window` steps of the same episode (detrend: only spikes above the local level count)."""
    x = errors[metric].astype(np.float64)
    if mode == "global":
        return x.to_numpy()
    if mode == "detrend":
        order = errors.sort_values(["episode_id", "t"]).index
        xs = x.loc[order]
        med = xs.groupby(errors.episode_id.loc[order]).transform(
            lambda v: v.rolling(int(window), center=True, min_periods=1).median())
        return (xs - med).reindex(errors.index).to_numpy()
    if mode == "episode":
        g = x.groupby(errors.episode_id)
        return ((x - g.transform("mean")) / g.transform("std").replace(0, np.nan).fillna(1.0)).to_numpy()
    raise ValueError(f"normalise must be global, episode or detrend, got {mode}")


def make_labels(errors, cfg):
    """-> (labels table, threshold info). Adds `split` from cfg.split_fractions / cfg.seed."""
    if cfg.error_metric == "err_l2_masked" and errors.err_l2_masked.isna().all():
        raise ValueError("error_metric err_l2_masked needs a mask_fn; no masks were computed")
    split = episode_split(errors, cfg.split_fractions, cfg.seed)
    s = errors.episode_id.map(split)
    v = normalised_metric(errors, cfg.error_metric, cfg.normalise, cfg.detrend_window)
    fit = s.isin(list(cfg.lambda_fit_splits)).to_numpy() & np.isfinite(v)
    if cfg.lambda_ is not None:
        lam = float(cfg.lambda_)
    else:
        if not fit.any():
            raise ValueError(f"no held-out rows in splits {list(cfg.lambda_fit_splits)} to fit lambda on")
        if cfg.lambda_stat == "mean":
            lam = float(np.mean(v[fit]))
        elif cfg.lambda_stat == "quantile":
            lam = float(np.quantile(v[fit], cfg.lambda_quantile))
        else:
            raise ValueError(f"lambda_stat must be mean or quantile, got {cfg.lambda_stat}")
    y = (v > lam).astype(np.int8)
    labels = pd.DataFrame(dict(episode_id=errors.episode_id, t=errors.t, split=s, wm_train_episode=errors.wm_train_episode,
                               err_value=v.astype(np.float32), y=y))
    pos = labels.groupby("split").y.mean().to_dict()
    info = dict(lambda_=lam, metric=cfg.error_metric, normalise=cfg.normalise, stat=cfg.lambda_stat,
                quantile=cfg.lambda_quantile,
                fixed=cfg.lambda_ is not None, fit_splits=list(cfg.lambda_fit_splits), n_fit_rows=int(fit.sum()),
                positive_rate_by_split={k: float(p) for k, p in pos.items()},
                n_episodes_by_split=pd.Series(split).value_counts().to_dict())
    return labels, split, info
