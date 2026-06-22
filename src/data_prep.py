"""MovieLens data preparation for the NextWatch sequential recommender.

Implements the audit-critical contract (see ARCHITECTURE.md / AUDIT.md):
  * deterministic ordering by (timestamp, movieId)            [L1]
  * leave-one-out split: last=test, 2nd-last=val, rest=train  [C1]
  * ALL-PREFIX windows for TRAINING, LOO only for EVAL        [C1]
  * MIN_COUNT and popularity computed on the TRAIN pool only  [C2]
  * index 0 reserved for PAD; real movie ids start at 1       [C3]
  * fixed rating scale (r-0.5)/4.5, no fitted scaler          [C2]

Movie-id sequences feed the embedding; the parallel rating sequences feed the
hybrid model's rating channel. Genre is looked up inside the model from the
genre_matrix, so it is not materialised per timestep here.
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
    X_train_rat: np.ndarray        # (n_windows, max_len) scaled ratings, aligned with X_train
    y_train: np.ndarray            # (n_windows,) next movie id
    val_hist: list[list[int]]      # per-user history (ids) for validation
    val_hist_rat: list[list[float]]
    val_target: list[int]          # per-user held-out val movie id
    test_hist: list[list[int]]
    test_hist_rat: list[list[float]]
    test_target: list[int]
    seen_val: list[set[int]]       # val history to exclude at val eval (NOT the val target)
    seen_test: list[set[int]]      # test history to exclude at test eval (NOT the test target)
    popularity: np.ndarray         # (n_items+1,) train-only counts, index 0 unused
    n_items: int                   # V (number of real items; valid ids 1..V)
    max_len: int
    id_to_movie: dict[int, int]    # contiguous id -> original MovieLens movieId
    genre_matrix: np.ndarray       # (n_items+1, n_genres) multi-hot, index 0 = PAD (zeros)
    genre_names: list[str]
    train_user_items: list[list[int]]  # encoded train-pool items per train user (for item-kNN)
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


def load_genres(movies_path: str, movie_to_id: dict[int, int]) -> tuple[np.ndarray, list[str]]:
    """Build a (n_items+1, n_genres) multi-hot matrix aligned to contiguous ids."""
    rows = []
    with open(movies_path, encoding="latin-1") as fh:
        for line in fh:
            mid_str, _title, genres = line.rstrip("\n").split("::")
            mid = int(mid_str)
            if mid in movie_to_id:
                rows.append((movie_to_id[mid], genres.split("|")))
    names = sorted({g for _, gs in rows for g in gs})
    gidx = {g: i for i, g in enumerate(names)}
    mat = np.zeros((len(movie_to_id) + 1, len(names)), dtype=np.float32)
    for cid, gs in rows:
        for g in gs:
            mat[cid, gidx[g]] = 1.0
    return mat, names


def build_dataset(
    ratings_path: str,
    movies_path: str = "data/ml-1m/movies.dat",
    min_count: int = 5,
    max_len: int = 50,
    max_windows_per_user: int | None = 30,
    seed: int = 42,
) -> Dataset:
    """Build leakage-safe train windows + LOO eval sets from ml-1m.

    max_windows_per_user caps the most-recent prefix targets per user so runs
    train quickly; set None for the full all-prefix set in final runs.
    """
    rng = np.random.default_rng(seed)
    df = load_ml1m(ratings_path)
    df["rating_scaled"] = (df["rating"].astype(np.float32) - 0.5) / 4.5  # fixed scale [C2]

    # Deterministic per-user chronological order.                     [L1]
    df = df.sort_values(["userId", "timestamp", "movieId"], kind="stable")
    grouped = df.groupby("userId")
    user_movies = grouped["movieId"].apply(list)
    user_rats = grouped["rating_scaled"].apply(list)

    # Leave-one-out split BEFORE any statistic is computed.           [C1][C2]
    # Stored as (movies, ratings[, target]) so the rating channel follows ids.
    train_pool, val_pair, test_pair = {}, {}, {}
    for uid in user_movies.index:
        movies, rats = user_movies[uid], user_rats[uid]
        if len(movies) < 3:
            continue
        train_pool[uid] = (movies[:-2], rats[:-2])
        val_pair[uid] = (movies[:-2], rats[:-2], movies[-2])
        test_pair[uid] = (movies[:-1], rats[:-1], movies[-1])

    # MIN_COUNT and popularity from the TRAIN POOL ONLY.              [C2]
    counts: dict[int, int] = {}
    for movies, _ in train_pool.values():
        for m in movies:
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

    def encode(movies: list[int], rats: list[float]) -> tuple[list[int], list[float]]:
        ids, rs = [], []
        for m, r in zip(movies, rats):
            if m in movie_to_id:
                ids.append(movie_to_id[m])
                rs.append(r)
        return ids, rs

    # ---- TRAIN: all-prefix windows (capped) ----                    [C1]
    X_rows, Xr_rows, y_rows = [], [], []
    train_user_items: list[list[int]] = []
    for movies, rats in train_pool.values():
        ids, rs = encode(movies, rats)
        if len(ids) < 2:
            continue
        train_user_items.append(ids)  # full train-pool items for item-kNN
        targets = range(1, len(ids))  # predict ids[t] from ids[:t]
        if max_windows_per_user is not None and len(ids) - 1 > max_windows_per_user:
            targets = range(len(ids) - max_windows_per_user, len(ids))
        for t in targets:
            X_rows.append(_pad_int(ids[:t][-max_len:], max_len))
            Xr_rows.append(_pad_float(rs[:t][-max_len:], max_len))
            y_rows.append(ids[t])
    X_train = np.asarray(X_rows, dtype=np.int32)
    X_train_rat = np.asarray(Xr_rows, dtype=np.float32)
    y_train = np.asarray(y_rows, dtype=np.int32)
    perm = rng.permutation(len(X_train))
    X_train, X_train_rat, y_train = X_train[perm], X_train_rat[perm], y_train[perm]

    # ---- EVAL: leave-one-out, drop users whose target fell out of catalog ----
    # Each split masks ITS OWN history only, so the held-out target is never
    # masked: val masks 1..n-2 (target n-1), test masks 1..n-1 (target n).  [M2]
    val_hist, val_hist_rat, val_target = [], [], []
    test_hist, test_hist_rat, test_target = [], [], []
    seen_val, seen_test = [], []
    n_oov = {"val": 0, "test": 0}
    for uid in test_pair:
        v_movies, v_rats, v_tgt = val_pair[uid]
        t_movies, t_rats, t_tgt = test_pair[uid]
        if t_tgt not in movie_to_id:
            n_oov["test"] += 1
            continue  # target unrankable -> automatic miss, excluded from denom honestly
        t_ids, t_rs = encode(t_movies, t_rats)
        if not t_ids:
            continue
        test_hist.append(t_ids[-max_len:])
        test_hist_rat.append(t_rs[-max_len:])
        test_target.append(movie_to_id[t_tgt])
        seen_test.append(set(t_ids))
        v_ids, v_rs = encode(v_movies, v_rats)
        if v_tgt in movie_to_id and v_ids:
            val_hist.append(v_ids[-max_len:])
            val_hist_rat.append(v_rs[-max_len:])
            val_target.append(movie_to_id[v_tgt])
            seen_val.append(set(v_ids))
        else:
            n_oov["val"] += 1  # keep aligned; rarely hit
            val_hist.append(t_ids[-max_len:])
            val_hist_rat.append(t_rs[-max_len:])
            val_target.append(movie_to_id[t_tgt])
            seen_val.append(set(t_ids))

    genre_matrix, genre_names = load_genres(movies_path, movie_to_id)

    return Dataset(
        X_train=X_train, X_train_rat=X_train_rat, y_train=y_train,
        val_hist=val_hist, val_hist_rat=val_hist_rat, val_target=val_target,
        test_hist=test_hist, test_hist_rat=test_hist_rat, test_target=test_target,
        seen_val=seen_val, seen_test=seen_test, popularity=popularity, n_items=n_items,
        max_len=max_len, id_to_movie=id_to_movie,
        genre_matrix=genre_matrix, genre_names=genre_names,
        train_user_items=train_user_items, n_oov_targets=n_oov,
    )


def _pad_int(ids: list[int], max_len: int) -> list[int]:
    """Pre-pad with PAD so recent items sit at the end."""
    if len(ids) >= max_len:
        return ids[-max_len:]
    return [PAD] * (max_len - len(ids)) + ids


def _pad_float(vals: list[float], max_len: int) -> list[float]:
    if len(vals) >= max_len:
        return vals[-max_len:]
    return [0.0] * (max_len - len(vals)) + vals


def pad_histories(histories: list[list[int]], max_len: int) -> np.ndarray:
    return np.asarray([_pad_int(h, max_len) for h in histories], dtype=np.int32)


def pad_ratings(histories: list[list[float]], max_len: int) -> np.ndarray:
    return np.asarray([_pad_float(h, max_len) for h in histories], dtype=np.float32)
