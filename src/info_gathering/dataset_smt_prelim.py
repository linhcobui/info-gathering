"""PRELIMINARY: anchor windows over cached frame embeddings, for SMT-style training.

One item per (episode, anchor t): the context window ending at t, the same window slid one
step forward (whose encoding is the one-step target), and the future window the decoder
must predict. Pixels are never loaded -- the embeddings are cached in the zarr by
world_model.models.smt_prelim.embed_cache.

Padding follows dataset_memtest1.load_memtest1 (repeat the final real value, flag it in
meta/pad) and prev_action follows EpisodeWindows.__getitem__ in dataset.py: zeros at the
episode start, otherwise the action taken at the previous step. This module re-implements
those two rules rather than calling load_memtest1, which would materialise a ~180 MB image
array into every DataLoader worker for no reason.
"""
from dataclasses import dataclass

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

META_KEYS = ("ball_under_target", "ball_slot", "phase", "cup_z_max")
SHOW_PHASE = 2          # memtest4.PHASES: the final lift, action [0, 1], always terminal


@dataclass
class EmbeddedEpisodes:
    """Per-episode arrays, padded to the longest episode by repeating the final value."""

    embed: np.ndarray        # (E, T, D) float32
    action: np.ndarray       # (E, T, A) float32  -- the action taken at step t
    prev_action: np.ndarray  # (E, T, A) float32  -- the action leading into step t
    real_len: np.ndarray     # (E,) int64
    meta: dict               # key -> (E, T, ...), including "pad" (E, T) uint8
    fps: float
    embed_mean: np.ndarray   # (D,) float32
    embed_std: np.ndarray    # (D,) float32
    step_delta: float        # median ||e_{t+1} - e_t|| over real steps
    ball_target: np.ndarray  # (E, T) float32  1 if the ball is under the cup at target_slot
    ball_lift: np.ndarray    # (E, T) bool     the cup has started to rise: supervise here only
    ball_slot_dense: np.ndarray = None  # (E, T) int64 ball slot, switching at a swap's HALFWAY frame

    @property
    def num_episodes(self):
        return self.embed.shape[0]

    @property
    def steps(self):
        return self.embed.shape[1]

    @property
    def action_dim(self):
        return self.action.shape[-1]

    @property
    def embed_dim(self):
        return self.embed.shape[-1]


def load_embedded_episodes(zarr_path, embed_key, meta_keys=META_KEYS, normalize=True,
                           lift_z_eps=1e-3):
    z = zarr.open_group(str(zarr_path), mode="r")
    if embed_key not in z:
        raise FileNotFoundError(f"{zarr_path} has no {embed_key!r}; run cache-embeddings-prelim first")
    attrs = z.attrs.get(embed_key) or {}

    ends = z["episode_ends"][:]
    starts = np.concatenate([[0], ends[:-1]])
    n_episodes, steps = len(ends), int((ends - starts).max())
    flat_embed, flat_action = z[embed_key][:], z["action"][:]
    dim, action_dim = flat_embed.shape[1], flat_action.shape[1]

    embed = np.zeros((n_episodes, steps, dim), dtype=np.float32)
    action = np.zeros((n_episodes, steps, action_dim), dtype=np.float32)
    meta = {k: np.zeros((n_episodes, steps) + z[k].shape[1:], dtype=z[k].dtype) for k in meta_keys}
    meta["pad"] = np.zeros((n_episodes, steps), dtype=np.uint8)
    real_len = (ends - starts).astype(np.int64)

    for e, (start, stop) in enumerate(zip(starts, ends)):
        n = int(stop - start)
        embed[e, :n] = flat_embed[start:stop]
        embed[e, n:] = flat_embed[stop - 1]          # the episode ends on a static frame
        action[e, :n] = flat_action[start:stop]
        action[e, n:] = flat_action[stop - 1]
        for key in meta_keys:
            values = z[key][start:stop]
            meta[key][e, :n] = values
            meta[key][e, n:] = values[-1]
        meta["pad"][e, n:] = 1

    prev_action = np.zeros_like(action)
    prev_action[:, 1:] = action[:, :-1]              # zeros at t == 0: every window starts an episode

    mean = np.asarray(attrs.get("mean", np.zeros(dim)), dtype=np.float32)
    std = np.asarray(attrs.get("std", np.ones(dim)), dtype=np.float32)
    std = np.maximum(std, 1e-6)
    if normalize:
        embed = (embed - mean) / std

    real = np.concatenate([np.linalg.norm(embed[e, 1:int(n)] - embed[e, :int(n) - 1], axis=-1)
                           for e, n in enumerate(real_len)])

    # The auxiliary "will the ball show" target, supervised ONLY during the final lift and only
    # once the cup has begun to rise. The lift action lands one frame before the cup moves, so at
    # show offset 0 the ball is visible in 0% of episodes -- supervising it would teach "no ball"
    # on every episode, positives included. cup_z_max is the tallest cup's height: cup_rest_z when
    # down, cup_up_z when raised, so "above rest" selects show offsets 1..9.
    needed = {"ball_under_target", "phase", "cup_z_max"}
    missing = needed - set(meta)
    if missing:
        # loud, not silent: a missing key here would leave the ball head with an all-False mask
        # and no gradient, which looks like a healthy run right up to the final eval
        raise KeyError(f"meta_keys is missing {sorted(missing)}; the auxiliary ball head needs "
                       f"{sorted(needed)}. Add them to the dataset config's meta_keys.")
    cup_z = meta["cup_z_max"].astype(np.float32)
    rest = float(cup_z[meta["pad"] == 0].min())              # cup_rest_z, read off the data
    ball_target = meta["ball_under_target"].astype(np.float32)
    ball_lift = (meta["phase"] == SHOW_PHASE) & (cup_z > rest + lift_z_eps) & (meta["pad"] == 0)
    if not ball_lift.any():
        raise ValueError(f"no supervised lift frames found in {zarr_path}")

    ball_slot_dense = None
    if "ball_slot" in meta:
        try:
            ball_slot_dense = dense_ball_slot(meta["ball_slot"], real_len)
        except ValueError as err:
            # e.g. data/memtest4.zarr still carries the OLD x-based labels (a 0<->2 swap reads slot 1
            # mid-arc). Lift-frame labels are unaffected, so only the dense slot target is withheld;
            # a run that asks for it fails loudly in the loss instead.
            print(f"[dataset] {zarr_path}: dense slot labels unavailable -- {err}")

    return EmbeddedEpisodes(embed=embed, action=action, prev_action=prev_action, real_len=real_len,
                            meta=meta, fps=float(z.attrs["fps"]), embed_mean=mean, embed_std=std,
                            step_delta=float(np.median(real)),
                            ball_target=ball_target, ball_lift=ball_lift,
                            ball_slot_dense=ball_slot_dense)


SWAP_FRAMES = 6          # memtest4 swap_frames
HALFWAY = 3              # swap_arc eases with smoothstep over linspace(0,1,6): progress
                         # 0, .10, .35, .65, .90, 1.0 -> the arc passes halfway between frames 2 and 3


def dense_ball_slot(ball_slot, real_len, swap_frames=SWAP_FRAMES, halfway=HALFWAY):
    """Per-frame ball slot that switches when the carrying cup crosses the swap's halfway point.

    The stored ball_slot keeps the start slot for frames 0..4 of a swap and switches on frame 5
    (memtest4.py). Here frames 0..halfway-1 keep the old slot and halfway..5 carry the new one.
    Swaps that leave the ball alone change nothing.
    """
    dense = ball_slot.astype(np.int64).copy()
    for e, n in enumerate(real_len):
        n = int(n)
        seq = ball_slot[e, :n]
        for c in np.flatnonzero(seq[1:] != seq[:-1]) + 1:      # c = the swap's LAST frame
            first = c - (swap_frames - 1)
            if first < 1 or (seq[first:c] != seq[first]).any() or seq[first - 1] != seq[first]:
                raise ValueError(f"episode {e}: ball_slot change at {c} is not a {swap_frames}-frame "
                                 f"swap ending on its last frame; the halfway relabel would be wrong")
            dense[e, first + halfway:c] = seq[c]
    return dense


class SmtAnchors(Dataset):
    """Every valid anchor t of every listed episode.

    An anchor needs both e_{t+1} (the predictor's input and the target window's last frame)
    and at least one future target, so t runs over [0, real_len - 2].
    """

    def __init__(self, data: EmbeddedEpisodes, episodes, ctx_len=10, fut_len=10):
        self.data, self.ctx_len, self.fut_len = data, int(ctx_len), int(fut_len)
        self.index = np.array([(e, t) for e in episodes
                               for t in range(int(data.real_len[e]) - 1)], dtype=np.int64)

    def __len__(self):
        return len(self.index)

    @property
    def action_dim(self):
        return self.data.action_dim

    @property
    def embed_dim(self):
        return self.data.embed_dim

    def __getitem__(self, i):
        e, t = (int(v) for v in self.index[i])
        data, C, F = self.data, self.ctx_len, self.fut_len
        n = int(data.real_len[e])

        # context: the C+1 steps ending at t+1, left-padded where the episode has not started
        win = np.arange(t - C + 1, t + 2)
        win_mask = win >= 0
        win_idx = np.clip(win, 0, None)
        e_win = np.where(win_mask[:, None], data.embed[e, win_idx], 0.0).astype(np.float32)
        a_win = np.where(win_mask[:, None], data.prev_action[e, win_idx], 0.0).astype(np.float32)

        # future: steps t .. t+F, clamped to the last real step so padding repeats a static frame
        seq = np.arange(t, t + F + 1)
        seq_mask = seq < n
        seq_idx = np.clip(seq, 0, n - 1)

        item = {
            "e_win": torch.from_numpy(e_win),
            "a_win": torch.from_numpy(a_win),
            "win_mask": torch.from_numpy(win_mask),
            "e_seq": torch.from_numpy(data.embed[e, seq_idx].astype(np.float32)),
            "a_seq": torch.from_numpy(data.action[e, seq_idx].astype(np.float32)),
            "seq_mask": torch.from_numpy(seq_mask),
            # the auxiliary head's target and its no-prediction mask, aligned with e_seq so that
            # ball_seq[1:] lines up with y_fut = e_seq[1:]. The right-clamp repeats the last real
            # step exactly as e_seq does; seq_mask then removes those positions from the loss.
            "ball_seq": torch.from_numpy(data.ball_target[e, seq_idx].astype(np.float32)),
            "ball_lift_seq": torch.from_numpy(data.ball_lift[e, seq_idx] & seq_mask),
            # which slot (0/1/2) the ball is in at each future step -- the 3-way target of the
            # slot head (smt_phase1 arch.reward_head: slot); same alignment and mask as ball_seq
            "slot_seq": torch.from_numpy(data.meta["ball_slot"][e, seq_idx].astype(np.int64)),
            # dense version: switches at the swap's halfway frame; supervised on every real step
            **({"slot_dense_seq": torch.from_numpy(data.ball_slot_dense[e, seq_idx])}
               if data.ball_slot_dense is not None else {}),
            "episode": torch.tensor(e, dtype=torch.int64),
            "t": torch.tensor(t, dtype=torch.int64),
            "real_len": torch.tensor(n, dtype=torch.int64),
        }
        for key, values in data.meta.items():
            item[f"meta/{key}"] = torch.from_numpy(np.asarray(values[e, t]))
        return item


def smt_prelim_datasets(zarr_path, embed_key, ctx_len=10, fut_len=10, holdout_episodes=25,
                        meta_keys=META_KEYS, normalize=True):
    """(train, holdout) anchor datasets; the last `holdout_episodes` episodes are held out."""
    data = load_embedded_episodes(zarr_path, embed_key, tuple(meta_keys), normalize)
    split = data.num_episodes - int(holdout_episodes)
    return (SmtAnchors(data, range(split), ctx_len, fut_len),
            SmtAnchors(data, range(split, data.num_episodes), ctx_len, fut_len))


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/memtest4.zarr"
    key = sys.argv[2] if len(sys.argv) > 2 else "embed_r2d_memtest4_wm15000"
    train, holdout = smt_prelim_datasets(path, key)
    data = train.data
    print(f"{path}: {data.num_episodes} episodes x {data.steps} padded steps, "
          f"embed {data.embed_dim}, action {data.action_dim}, median step delta {data.step_delta:.3f}")
    print(f"anchors: {len(train)} train / {len(holdout)} holdout")

    raw = load_embedded_episodes(path, key, normalize=False)
    checked = 0
    for i in np.random.default_rng(0).choice(len(train), size=512, replace=False):
        item = train[int(i)]
        e, t, n = int(item["episode"]), int(item["t"]), int(item["real_len"])
        # the identities the model relies on
        assert t < n - 1, "an anchor landed in the pad"
        assert np.allclose(item["e_win"][-1].numpy(), data.embed[e, t + 1]), "e_win[-1] is not e_{t+1}"
        assert np.allclose(item["e_seq"][0].numpy(), data.embed[e, t]), "e_seq[0] is not e_t"
        assert torch.equal(item["a_win"][-1], item["a_seq"][0]), "prev_action[t+1] != action[t]"
        assert bool(item["win_mask"][-1]) and bool(item["seq_mask"][0])
        assert item["win_mask"].numpy().tolist() == (np.arange(t - 10 + 1, t + 2) >= 0).tolist()
        assert item["seq_mask"].numpy().tolist() == (np.arange(t, t + 11) < n).tolist()
        assert not item["win_mask"][:-1][~(np.arange(t - 9, t + 1) >= 0)].any()
        assert data.meta["pad"][e, t] == 0, "anchor label comes from a padded step"
        # normalisation is invertible back to the cached embedding
        assert np.allclose(item["e_seq"][0].numpy() * data.embed_std + data.embed_mean,
                           raw.embed[e, t], atol=1e-4)
        checked += 1
    print(f"ok  {checked} anchors pass the index, mask and normalisation identities")

    # --- the auxiliary ball head's target and mask ---
    lift, tgt, phase = data.ball_lift, data.ball_target, data.meta["phase"]
    per_episode = [int(lift[e, :int(n)].sum()) for e, n in enumerate(data.real_len)]
    assert set(per_episode) == {9}, f"expected 9 supervised frames per episode, got {set(per_episode)}"
    assert (phase[lift] == 2).all(), "a supervised frame is outside the show phase"
    assert (data.meta["pad"][lift] == 0).all(), "a supervised frame landed in the pad"
    for e, n in enumerate(data.real_len):
        n = int(n)
        offsets = np.flatnonzero(lift[e, :n]) - (n - 10)
        assert offsets.tolist() == list(range(1, 10)), f"episode {e} supervises offsets {offsets.tolist()}"
        assert (tgt[e, lift[e]] == tgt[e, n - 1]).all(), "target varies within the lift"
    print(f"ok  ball head: exactly 9 frames per episode, show offsets 1-9, constant target, "
          f"positive rate {tgt[lift].mean():.3f}")

    anchors_with_signal = sum(int(train[int(i)]["ball_lift_seq"].any())
                              for i in range(0, len(train), max(1, len(train) // 2000)))
    sampled = len(range(0, len(train), max(1, len(train) // 2000)))
    print(f"ok  {anchors_with_signal}/{sampled} sampled anchors see at least one supervised "
          f"position in their {train.fut_len}-step future")

    shapes = {k: tuple(v.shape) for k, v in train[0].items()}
    print("item shapes:", shapes)


class SmtRollouts(Dataset):
    """Contiguous slices for DMT: one item per (episode, t0), covering the steps a rollout needs.

    forward_dmt (smt/src/model.py:353) encodes an oracle memory at every step of a window and
    rolls the RNN over the same window from the first oracle state. To build the oracle states we
    need the teacher's context window at each of the roll_len+1 steps, i.e. raw steps
    t0-ctx_len+1 .. t0+roll_len. The windows themselves are cut with sliding_windows in the loss,
    so the item stays one contiguous slice rather than roll_len+1 overlapping copies.
    """

    def __init__(self, data: EmbeddedEpisodes, episodes, ctx_len=60, roll_len=16):
        self.data, self.ctx_len, self.roll_len = data, int(ctx_len), int(roll_len)
        self.index = np.array([(e, t) for e in episodes
                               for t in range(int(data.real_len[e]) - int(roll_len))], dtype=np.int64)

    def __len__(self):
        return len(self.index)

    @property
    def action_dim(self):
        return self.data.action_dim

    @property
    def embed_dim(self):
        return self.data.embed_dim

    def __getitem__(self, i):
        e, t0 = (int(v) for v in self.index[i])
        data, C, R = self.data, self.ctx_len, self.roll_len
        n = int(data.real_len[e])
        idx = np.arange(t0 - C + 1, t0 + R + 1)
        real = (idx >= 0) & (idx < n)                    # left pad before the episode, right pad after
        clipped = np.clip(idx, 0, n - 1)
        return {
            "e_full": torch.from_numpy(np.where(real[:, None], data.embed[e, clipped], 0.0).astype(np.float32)),
            "a_full": torch.from_numpy(np.where(real[:, None], data.prev_action[e, clipped], 0.0).astype(np.float32)),
            "real_full": torch.from_numpy(real),
            "slot_full": torch.from_numpy(data.ball_slot_dense[e, clipped]),
            # binary read-out target, for arms whose reward_head is "binary" rather than "slot"
            "ball_full": torch.from_numpy(
                data.meta["ball_under_target"][e, clipped].astype(np.float32)),
            "episode": torch.tensor(e, dtype=torch.int64),
            "t": torch.tensor(t0, dtype=torch.int64),
        }


def smt_rollout_datasets(zarr_path, embed_key, ctx_len=60, roll_len=16, holdout_episodes=25,
                         meta_keys=META_KEYS, normalize=True):
    """(train, holdout) rollout datasets; the last `holdout_episodes` episodes are held out."""
    data = load_embedded_episodes(zarr_path, embed_key, tuple(meta_keys), normalize)
    if data.ball_slot_dense is None:
        raise ValueError(f"{zarr_path} has no dense slot labels; DMT's read-out loss needs them")
    split = data.num_episodes - int(holdout_episodes)
    return (SmtRollouts(data, range(split), ctx_len, roll_len),
            SmtRollouts(data, range(split, data.num_episodes), ctx_len, roll_len))
