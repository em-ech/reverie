"""Build the neural collaborative-filtering training set from the Letterboxd data.

Turns the big Kaggle Letterboxd ratings (samlearner) into compact, integer-indexed
arrays for a Keras embedding model, slots one or more real people in as extra users
(their Letterboxd exports, each with a temporal holdout by rating date), and builds a
per-movie content feature matrix for the model's frozen content tower.

One-time data step. Big CSVs stay in ~/Downloads (passed by path); outputs land in
artifacts/ncf/ (gitignored).

Usage
-----
    conda activate deep-learning
    python -m scripts.build_ncf_dataset \
        --ratings ~/Downloads/ratings_export.csv \
        --movies  ~/Downloads/movie_data.csv \
        --user em=data/letterboxd_ratings.csv \
        --user michael=data/michael_ratings.csv \
        --min-movie 30 --min-user 20 --out artifacts/ncf
    # add --subsample 3000000 for a fast dev run
"""

from __future__ import annotations

import argparse
import ast
import json
import os

import numpy as np
import pandas as pd

from src.importers import _normalise_title

NUMERIC = ["year_released", "vote_average", "log_vote_count", "runtime"]


def _title_year_key(title, year) -> str:
    if not isinstance(title, str) or not title.strip():
        return "\x00"  # sentinel that never matches a real title
    y = pd.to_numeric(year, errors="coerce")
    y = "" if pd.isna(y) else str(int(y))
    return f"{_normalise_title(title)}_{y}"


def _parse_genres(s) -> list[str]:
    if not isinstance(s, str) or not s.strip():
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            v = parser(s)
            return [str(g) for g in v] if isinstance(v, list) else []
        except (ValueError, SyntaxError):
            continue
    return []


def _load_user_films(path: str, key_to_slug: dict) -> pd.DataFrame:
    """A person's Letterboxd ratings.csv -> matched, date-sorted (slug, rating_val)."""
    d = pd.read_csv(path).dropna(subset=["Rating"]).copy()
    d["slug"] = [_title_year_key(t, y) for t, y in zip(d.Name, d.Year)]
    d["slug"] = d["slug"].map(key_to_slug)
    d = d.dropna(subset=["slug"]).copy()
    d["rating_val"] = (pd.to_numeric(d.Rating) * 2).round().clip(1, 10).astype(int)
    d["Date"] = pd.to_datetime(d.Date, errors="coerce")
    return d.sort_values("Date")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", required=True)
    ap.add_argument("--movies", required=True)
    ap.add_argument("--user", action="append", default=[], metavar="NAME=PATH",
                    help="a real person's Letterboxd ratings.csv to slot in (repeatable)")
    ap.add_argument("--min-movie", type=int, default=30)
    ap.add_argument("--min-user", type=int, default=20)
    ap.add_argument("--user-weight-cap", type=float, default=25.0,
                    help="max sample-weight multiplier for a slotted-in person's rows")
    ap.add_argument("--subsample", type=int, default=0, help="rows to sample for dev")
    ap.add_argument("--out", default="artifacts/ncf")
    args = ap.parse_args()
    rng = np.random.default_rng(42)
    user_specs = dict(s.split("=", 1) for s in args.user)

    # --- 1. ratings -----------------------------------------------------------
    print("loading ratings...")
    r = pd.read_csv(
        args.ratings, usecols=["movie_id", "rating_val", "user_id"],
        dtype={"movie_id": "category", "user_id": "category", "rating_val": "int8"},
    )
    if args.subsample and args.subsample < len(r):
        r = r.sample(args.subsample, random_state=42).reset_index(drop=True)

    mc = r["movie_id"].value_counts()
    uc = r["user_id"].value_counts()
    keep_movies = set(mc[mc >= args.min_movie].index)
    keep_users = set(uc[uc >= args.min_user].index)
    r = r[r["movie_id"].isin(keep_movies) & r["user_id"].isin(keep_users)]
    r["movie_id"] = r["movie_id"].astype(str)
    r["user_id"] = r["user_id"].astype(str)
    print(f"  kept {len(r):,} ratings | {len(keep_movies):,} movies | {len(keep_users):,} users")

    # --- 2. movie metadata + real-person matching ----------------------------
    print("loading movie metadata...")
    mv = pd.read_csv(
        args.movies,
        usecols=["movie_id", "movie_title", "year_released", "genres", "vote_average",
                 "vote_count", "runtime"],
        engine="python", on_bad_lines="skip",
    ).drop_duplicates("movie_id")
    mv["key"] = [_title_year_key(t, y) for t, y in zip(mv.movie_title, mv.year_released)]
    key_to_slug = dict(zip(mv["key"], mv["movie_id"]))

    ext = {name: _load_user_films(path, key_to_slug) for name, path in user_specs.items()}
    for name, d in ext.items():
        print(f"  {name}: {len(d)} films matched")

    # --- 3. vocab + integer indices ------------------------------------------
    ext_slugs = set().union(*[set(d.slug) for d in ext.values()]) if ext else set()
    vocab = sorted(keep_movies | ext_slugs)   # keep real people's films even if sparse
    movie_to_idx = {"<UNK>": 0}
    movie_to_idx.update({s: i + 1 for i, s in enumerate(vocab)})
    user_to_idx = {u: i for i, u in enumerate(sorted(keep_users))}
    ext_idx = {}
    for name in ext:
        ext_idx[name] = len(user_to_idx)
        user_to_idx[f"__{name}__"] = ext_idx[name]
    n_movies, n_users = len(movie_to_idx), len(user_to_idx)
    print(f"  vocab: {n_movies:,} movies (incl. UNK) | {n_users:,} users (+{len(ext)} real people)")

    # --- 4. global rating arrays + split -------------------------------------
    gui = r["user_id"].map(user_to_idx).to_numpy(np.int32)
    gmi = r["movie_id"].map(movie_to_idx).to_numpy(np.int32)
    gy = r["rating_val"].to_numpy(np.int8)
    p = rng.random(len(r))
    masks = {"train": p < 0.8, "val": (p >= 0.8) & (p < 0.9), "test": p >= 0.9}

    # Each real person: most-recent 20% held out; the rest folds into global train.
    # Their rows are up-weighted so a light user's ~250 ratings train their embedding
    # roughly as hard as a native user's (median count) thousands.
    native_median = float(uc[uc >= args.min_user].median())
    ext_train, ext_tests, ext_w = [], {}, []
    for name, d in ext.items():
        n_test = max(1, int(len(d) * 0.2))
        te, trn = d.iloc[-n_test:], d.iloc[:-n_test]
        uidx = ext_idx[name]
        w = float(np.clip(native_median / max(len(trn), 1), 1.0, args.user_weight_cap))
        ext_train.append((
            np.full(len(trn), uidx, np.int32),
            trn.slug.map(movie_to_idx).to_numpy(np.int32),
            trn.rating_val.to_numpy(np.int8),
        ))
        ext_w.append(w)
        ext_tests[name] = (
            np.full(len(te), uidx, np.int32),
            te.slug.map(movie_to_idx).to_numpy(np.int32),
            te.rating_val.to_numpy(np.int8),
        )
        print(f"  {name}: {len(trn)} train / {len(te)} test (most recent), weight x{w:.1f}")

    splits, train_weight = {}, None
    for name, m in masks.items():
        u, mi, y = gui[m], gmi[m], gy[m]
        if name == "train":
            w = np.ones(len(y), np.float32)
            if ext_train:
                u = np.concatenate([u] + [a[0] for a in ext_train])
                mi = np.concatenate([mi] + [a[1] for a in ext_train])
                y = np.concatenate([y] + [a[2] for a in ext_train])
                w = np.concatenate([w] + [np.full(len(a[2]), ew, np.float32)
                                          for a, ew in zip(ext_train, ext_w)])
            train_weight = w
        splits[name] = (u, mi, y)

    # --- 5. content features (per movie, frozen tower input) -----------------
    print("building content features...")
    mvv = mv[mv["movie_id"].isin(set(vocab))].drop_duplicates("movie_id").set_index("movie_id")
    genre_set: set[str] = set()
    for g in mvv["genres"].dropna():
        genre_set.update(_parse_genres(g))
    genre_names = sorted(genre_set)
    G = len(genre_names)
    gidx = {g: i for i, g in enumerate(genre_names)}

    feat = np.zeros((n_movies, G + len(NUMERIC)), dtype=np.float32)
    med = {
        "year_released": pd.to_numeric(mvv.year_released, errors="coerce").median(),
        "vote_average": pd.to_numeric(mvv.vote_average, errors="coerce").median(),
        "log_vote_count": float(np.log1p(pd.to_numeric(mvv.vote_count, errors="coerce")).median()),
        "runtime": pd.to_numeric(mvv.runtime, errors="coerce").median(),
    }
    for slug, idx in movie_to_idx.items():
        if slug == "<UNK>" or slug not in mvv.index:
            continue
        row = mvv.loc[slug]
        for g in _parse_genres(row.genres):
            if g in gidx:
                feat[idx, gidx[g]] = 1.0
        yr = pd.to_numeric(row.year_released, errors="coerce")
        va = pd.to_numeric(row.vote_average, errors="coerce")
        vc = pd.to_numeric(row.vote_count, errors="coerce")
        rt = pd.to_numeric(row.runtime, errors="coerce")
        feat[idx, G + 0] = med["year_released"] if pd.isna(yr) else yr
        feat[idx, G + 1] = med["vote_average"] if pd.isna(va) else va
        feat[idx, G + 2] = med["log_vote_count"] if pd.isna(vc) else float(np.log1p(vc))
        feat[idx, G + 3] = med["runtime"] if pd.isna(rt) else rt

    num = feat[1:, G:]
    mean, std = num.mean(0), num.std(0)
    std[std == 0] = 1.0
    feat[1:, G:] = (num - mean) / std

    # --- 6. baseline stats (train-only) --------------------------------------
    tu, tm, ty = splits["train"]
    gmean = float(ty.mean())
    msum = np.bincount(tm, weights=ty, minlength=n_movies)
    mcnt = np.bincount(tm, minlength=n_movies)
    movie_mean = np.where(mcnt > 0, msum / np.maximum(mcnt, 1), gmean).astype(np.float32)
    usum = np.bincount(tu, weights=ty, minlength=n_users)
    ucnt = np.bincount(tu, minlength=n_users)
    user_mean = np.where(ucnt > 0, usum / np.maximum(ucnt, 1), gmean).astype(np.float32)

    # --- 7. write artifacts ---------------------------------------------------
    os.makedirs(args.out, exist_ok=True)
    for name, (u, mi, y) in splits.items():
        if name == "train":
            np.savez(f"{args.out}/{name}.npz", user=u, movie=mi, rating=y, weight=train_weight)
        else:
            np.savez(f"{args.out}/{name}.npz", user=u, movie=mi, rating=y)
    for name, (u, mi, y) in ext_tests.items():
        np.savez(f"{args.out}/{name}_test.npz", user=u, movie=mi, rating=y)
    np.save(f"{args.out}/movie_features.npy", feat)
    np.savez(f"{args.out}/baselines.npz", global_mean=np.float32(gmean),
             movie_mean=movie_mean, user_mean=user_mean)
    with open(f"{args.out}/meta.json", "w") as fh:
        json.dump({
            "n_users": n_users, "n_movies": n_movies, "global_mean": gmean,
            "genre_names": genre_names, "n_features": int(feat.shape[1]),
            "rating_scale": "1-10",
            "external_users": ext_idx,
            "sizes": {k: int(len(v[2])) for k, v in splits.items()}
            | {f"{n}_test": int(len(t[2])) for n, t in ext_tests.items()},
        }, fh, indent=2)
    with open(f"{args.out}/movie_index.json", "w") as fh:
        json.dump(movie_to_idx, fh)
    with open(f"{args.out}/user_index.json", "w") as fh:
        json.dump(user_to_idx, fh)

    print(f"Done. -> {args.out}")
    print(f"  train {len(splits['train'][2]):,} | val {len(splits['val'][2]):,} | "
          f"test {len(splits['test'][2]):,} | real people: {list(ext_idx)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
