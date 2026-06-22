"""MovieLens data preparation for the NextWatch sequential recommender.

Implements the audit-critical contract (see ARCHITECTURE.md / AUDIT.md):
  * deterministic ordering by (timestamp, movieId)            [L1]
  * leave-one-out split: last=test, 2nd-last=val, rest=train  [C1]
  * ALL-PREFIX windows for TRAINING, LOO only for EVAL        [C1]
  * MIN_COUNT and popularity computed on the TRAIN pool only  [C2]
  * index 0 reserved for PAD; real movie ids start at 1       [C3]
  * fixed rating scale (r-0.5)/4.5, no fitted scaler          [C2]

The E0 gate uses movie-id sequences only (genre/rating features are an
ablation, E6); this module exposes the ids and the train-only popularity that
both the model and the baselines consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PAD = 0  # reserved padding / mask index; real movie ids are >= 1


@dataclass
class Dataset:
    """Everything downstream (model, baselines, eval) needs, leakage-safe."""

    X_train: np.ndarray            # (n_windows, max_len) padded movie-id prefixes
    y_train: np.ndarray            # (n_windows,) next movie id
    val_hist: list[list[int]]      # per-user history for validation
    val_target: list[int]          # per-user held-out val movie id
    test_hist: list[list[int]]
    test_target: list[int]
    seen_val: list[set[int]]       # val history to exclude at val eval (NOT the val target)
    seen_test: list[set[int]]      # test history to exclude at test eval (NOT the test target)
    popularity: np.ndarray         # (n_items+1,) train-only counts, index 0 unused
    n_items: int                   # V (number of real items; valid ids 1..V)
    max_len: int
    id_to_movie: dict[int, int]    # contiguous id -> original MovieLens movieId
    n_oov_targets: dict[str, int] = field(default_factory=dict)  # targets dropped by MIN_COUNT


def load_ml1m(ratings_path: str, sep: str = "::") -> pd.DataFrame:
    """Load ml-1m ratings.dat -> DataFrame[userId, movieId, rating, timestamp]."""
    df = pd.read_csv(
        ratings_path,
        sep=sep,
        engine="python",
        names=["userId", "movieId", "rating", "timestamp"],
        encoding="latin-1",
    )
    # No duplicate (user, movie) in ml-1m, but assert it so a future data merge
    # can never silently leak a repeated target into the history.    [L1]
    assert not df.duplicated(["userId", "movieId"]).any(), "duplicate (user, movie) rows"
    return df


def build_dataset(
    ratings_path: str,
    min_count: int = 5,
    max_len: int = 50,
    max_windows_per_user: int | None = 30,
    seed: int = 42,
) -> Dataset:
    """Build leakage-safe train windows + LOO eval sets from ml-1m.

    max_windows_per_user caps the most-recent prefix targets per user so the E0
    gate trains quickly; set None for the full all-prefix set in real runs.
    """
    rng = np.random.default_rng(seed)
    df = load_ml1m(ratings_path)

    # Deterministic per-user chronological order.                     [L1]
    df = df.sort_values(["userId", "timestamp", "movieId"], kind="stable")
    user_seqs = df.groupby("userId")["movieId"].apply(list)

    # Leave-one-out split BEFORE any statistic is computed.           [C1][C2]
    train_pool, val_pair, test_pair = {}, {}, {}
    for uid, seq in user_seqs.items():
        if len(seq) < 3:
            continue  # need history + val + test
        train_pool[uid] = seq[:-2]
        val_pair[uid] = (seq[:-2], seq[-2])      # history, target
        test_pair[uid] = (seq[:-1], seq[-1])     # history (incl. val item), target

    # MIN_COUNT and popularity from the TRAIN POOL ONLY.              [C2]
    counts: dict[int, int] = {}
    for seq in train_pool.values():
        for m in seq:
            counts[m] = counts.get(m, 0) + 1
    kept = sorted(m for m, c in counts.items() if c >= min_count)

    # Contiguous ids: PAD=0, real items 1..V.                         [C3]
    movie_to_id = {m: i + 1 for i, m in enumerate(kept)}
    id_to_movie = {i + 1: m for i, m in enumerate(kept)}
    n_items = len(kept)

    popularity = np.zeros(n_items + 1, dtype=np.int64)
    for m, c in counts.items():
        if m in movie_to_id:
            popularity[movie_to_id[m]] = c

    def encode(seq: list[int]) -> list[int]:
        return [movie_to_id[m] for m in seq if m in movie_to_id]

    # ---- TRAIN: all-prefix windows (capped) ----                    [C1]
    X_rows, y_rows = [], []
    for uid, seq in train_pool.items():
        ids = encode(seq)
        if len(ids) < 2:
            continue
        targets = range(1, len(ids))  # predict ids[t] from ids[:t]
        if max_windows_per_user is not None and len(ids) - 1 > max_windows_per_user:
            targets = range(len(ids) - max_windows_per_user, len(ids))
        for t in targets:
            hist = ids[:t][-max_len:]
            X_rows.append(_left_pad(hist, max_len))
            y_rows.append(ids[t])
    X_train = np.asarray(X_rows, dtype=np.int32)
    y_train = np.asarray(y_rows, dtype=np.int32)
    perm = rng.permutation(len(X_train))
    X_train, y_train = X_train[perm], y_train[perm]

    # ---- EVAL: leave-one-out, drop users whose target fell out of catalog ----
    # Each split masks ITS OWN history only, so the held-out target is never
    # masked: val masks 1..n-2 (target n-1), test masks 1..n-1 (target n).  [M2]
    val_hist, val_target, test_hist, test_target = [], [], [], []
    seen_val, seen_test = [], []
    n_oov = {"val": 0, "test": 0}
    for uid in test_pair:
        v_hist, v_tgt = val_pair[uid]
        t_hist, t_tgt = test_pair[uid]
        if t_tgt not in movie_to_id:
            n_oov["test"] += 1
            continue  # target unrankable -> automatic miss, excluded from denom honestly
        enc_t_hist = encode(t_hist)
        if not enc_t_hist:
            continue
        test_hist.append(enc_t_hist[-max_len:])
        test_target.append(movie_to_id[t_tgt])
        seen_test.append(set(enc_t_hist))
        enc_v_hist = encode(v_hist)
        if v_tgt in movie_to_id and enc_v_hist:
            val_hist.append(enc_v_hist[-max_len:])
            val_target.append(movie_to_id[v_tgt])
            seen_val.append(set(enc_v_hist))
        else:
            n_oov["val"] += 1
            val_hist.append(enc_t_hist[-max_len:])  # keep aligned; rarely hit
            val_target.append(movie_to_id[t_tgt])
            seen_val.append(set(enc_t_hist))

    return Dataset(
        X_train=X_train, y_train=y_train,
        val_hist=val_hist, val_target=val_target,
        test_hist=test_hist, test_target=test_target,
        seen_val=seen_val, seen_test=seen_test, popularity=popularity, n_items=n_items,
        max_len=max_len, id_to_movie=id_to_movie, n_oov_targets=n_oov,
    )


def _left_pad(ids: list[int], max_len: int) -> list[int]:
    """Pre-pad with PAD so recent items sit at the end."""
    if len(ids) >= max_len:
        return ids[-max_len:]
    return [PAD] * (max_len - len(ids)) + ids


def pad_histories(histories: list[list[int]], max_len: int) -> np.ndarray:
    return np.asarray([_left_pad(h, max_len) for h in histories], dtype=np.int32)
