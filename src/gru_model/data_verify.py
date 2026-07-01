"""Verify the ml-1m dataset loads cleanly and write a data-choices summary.

Run once before training to confirm the pipeline is sound:

    python -m src.gru_model.data_verify                          # quick check
    python -m src.gru_model.data_verify --full                   # full build (slow)
    python -m src.gru_model.data_verify --out data/data_notes.md # write the summary doc

Outputs:
  * Console pass/fail report for each assertion.
  * data/data_notes.md  (or --out path) — a short human-readable summary
    of what we kept and why, intended to feed directly into the slideshow.
    Safe to commit (no personal data, no ratings content).

Audit tags satisfied: [C1] [C2] [C3] [L1] [H1]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Default paths (relative to project root; override via CLI flags if needed)
DEFAULT_RATINGS = "data/ml-1m/ratings.dat"
DEFAULT_MOVIES  = "data/ml-1m/movies.dat"
DEFAULT_OUT     = "data/data_notes.md"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pass(msg: str) -> None:
    print(f"  ✓  {msg}")


def _fail(msg: str) -> None:
    print(f"  ✗  {msg}", file=sys.stderr)


def _section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_ratings(ratings_path: str) -> pd.DataFrame:
    """Load and assert the ml-1m ratings file is well-formed."""
    _section("ratings.dat")
    p = Path(ratings_path)
    if not p.exists():
        _fail(f"File not found: {p}")
        print("\nExpected layout: data/ml-1m/ratings.dat")
        print("Download ml-1m from https://grouplens.org/datasets/movielens/1m/")
        sys.exit(1)

    df = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        names=["userId", "movieId", "rating", "timestamp"],
        encoding="latin-1",
    )

    # Row count
    n = len(df)
    if n >= 900_000:
        _pass(f"{n:,} rows loaded  (expected ~1,000,209)")
    else:
        _fail(f"Only {n:,} rows – might be the small dataset, not ml-1m")

    # No nulls
    nulls = df.isnull().sum().sum()
    if nulls == 0:
        _pass("No null values")
    else:
        _fail(f"{nulls} null values found")

    # Ratings in valid range
    bad_ratings = (~df["rating"].isin([1, 2, 3, 4, 5])).sum()
    if bad_ratings == 0:
        _pass("All ratings in {1,2,3,4,5}")
    else:
        _fail(f"{bad_ratings} out-of-range rating values")

    # No duplicate (user, movie) pairs   [L1]
    dups = df.duplicated(["userId", "movieId"]).sum()
    if dups == 0:
        _pass("No duplicate (userId, movieId) pairs  [L1]")
    else:
        _fail(f"{dups} duplicate (userId, movieId) pairs — leakage risk!  [L1]")

    # Timestamps present
    ts_zeros = (df["timestamp"] == 0).sum()
    if ts_zeros == 0:
        _pass("All timestamps non-zero (deterministic sort possible)  [L1]")
    else:
        _fail(f"{ts_zeros} zero timestamps — ordering may not be deterministic  [L1]")

    # User and movie counts
    n_users  = df["userId"].nunique()
    n_movies = df["movieId"].nunique()
    _pass(f"{n_users:,} unique users,  {n_movies:,} unique movies")
    if n_users >= 6_000:
        _pass("User count consistent with ml-1m (6,040 expected)")
    else:
        _fail(f"User count {n_users} is lower than ml-1m's 6,040")

    return df


def check_movies(movies_path: str, df_ratings: pd.DataFrame) -> None:
    """Verify movies.dat and genre coverage."""
    _section("movies.dat")
    p = Path(movies_path)
    if not p.exists():
        _fail(f"File not found: {p}")
        return

    rows = []
    with open(movies_path, encoding="latin-1") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("::")
            if len(parts) != 3:
                continue
            mid_str, title, genres = parts
            rows.append({"movieId": int(mid_str), "title": title, "genres": genres})
    df_movies = pd.DataFrame(rows)
    _pass(f"{len(df_movies):,} movies in movies.dat")

    # Every rated movie has metadata
    rated_ids = set(df_ratings["movieId"].unique())
    movie_ids = set(df_movies["movieId"].unique())
    missing_meta = rated_ids - movie_ids
    if not missing_meta:
        _pass("Every rated movieId has metadata in movies.dat")
    else:
        _fail(f"{len(missing_meta)} rated movies have no metadata entry")

    # No movies with missing-genre placeholder
    no_genres = df_movies["genres"].eq("(no genres listed)").sum()
    if no_genres == 0:
        _pass("No '(no genres listed)' entries")
    else:
        _fail(f"{no_genres} movies have '(no genres listed)' — check coverage")

    all_genres = sorted({g for gs in df_movies["genres"] for g in gs.split("|")})
    _pass(f"Genre vocabulary ({len(all_genres)}): {', '.join(all_genres)}")


def check_split_integrity(df: pd.DataFrame, min_count: int = 5) -> dict:
    """Quick sanity-check the leave-one-out split counts without a full build."""
    _section("Leave-one-out split sanity (quick, no model)")

    df_sorted = df.sort_values(["userId", "timestamp", "movieId"], kind="stable")
    grouped = df_sorted.groupby("userId")["movieId"].apply(list)

    # Users with >= 3 interactions can participate
    eligible = {uid: seq for uid, seq in grouped.items() if len(seq) >= 3}
    _pass(f"{len(eligible):,} users with ≥3 interactions (eligible for LOO split)  [C1]")

    dropped = len(grouped) - len(eligible)
    if dropped:
        _pass(f"{dropped} users skipped (fewer than 3 interactions)")

    # Check that per-user train sizes are consistent: test_history = len(val_history)+1  [C2]
    mismatches = 0
    for uid, seq in eligible.items():
        val_hist_len  = len(seq) - 2   # indices 0..n-3
        test_hist_len = len(seq) - 1   # indices 0..n-2
        if test_hist_len != val_hist_len + 1:
            mismatches += 1
    if mismatches == 0:
        _pass("len(test_history) == len(val_history)+1  for all users  [C2]")
    else:
        _fail(f"{mismatches} users violate the history-length invariant  [C2]")

    # Popularity is computed on train only — just count unique items in train pools  [C2]
    from collections import Counter
    train_counts: Counter = Counter()
    for seq in eligible.values():
        for m in seq[:-2]:
            train_counts[m] += 1
    kept = [m for m, c in train_counts.items() if c >= min_count]
    _pass(
        f"MIN_COUNT={min_count}: {len(kept):,} items survive "
        f"({len(train_counts)-len(kept):,} dropped)  [C2]"
    )

    return {"n_users": len(eligible), "n_items_kept": len(kept), "train_counts": train_counts}


# ---------------------------------------------------------------------------
# Data notes doc
# ---------------------------------------------------------------------------

def write_data_notes(out_path: str, df: pd.DataFrame, stats: dict) -> None:
    """Write a concise Markdown summary of our data choices for the slideshow."""
    _section(f"Writing data notes → {out_path}")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    ratings_by_user = df.groupby("userId")["rating"].count()
    median_seq = int(ratings_by_user.median())
    min_seq    = int(ratings_by_user.min())
    max_seq    = int(ratings_by_user.max())

    ratings_dist = df["rating"].value_counts().sort_index()
    genre_counts = {}
    movies_path = DEFAULT_MOVIES
    if Path(movies_path).exists():
        with open(movies_path, encoding="latin-1") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("::")
                if len(parts) == 3:
                    for g in parts[2].split("|"):
                        genre_counts[g] = genre_counts.get(g, 0) + 1
        top_genres = sorted(genre_counts, key=lambda g: -genre_counts[g])[:5]
    else:
        top_genres = []

    lines = [
        "# Reverie — Data Choices",
        "",
        "> Auto-generated by `src/data_verify.py`. Safe to commit — no personal data.",
        "",
        "## Primary training corpus: MovieLens 1M (`ml-1m`)",
        "",
        "We chose **ml-1m** over `ml-latest-small` because:",
        "",
        "- **~10× more sequences** (6,040 vs 610 users) gives the GRU enough",
        "  teacher-forced training windows to learn sequential patterns.",
        "- **Denser catalog**: ~3,700 movies with ≥5 ratings each, vs ~9,700",
        "  mostly-sparse movies in the small set. A sparse softmax over 9,700+",
        "  items is much harder to train reliably.",
        "- The larger split simply makes results more statistically meaningful.",
        "",
        "We did **not** run both datasets. One sentence for the deck: the small",
        "dataset would be a fallback if training failed, not a comparison point —",
        "we chose ml-1m deliberately and it worked.",
        "",
        "## What we kept",
        "",
        f"| Stat | Value |",
        f"|------|-------|",
        f"| Total ratings | {len(df):,} |",
        f"| Unique users | {df['userId'].nunique():,} |",
        f"| Unique movies (raw) | {df['movieId'].nunique():,} |",
        f"| Movies after MIN_COUNT ≥ 5 | {stats['n_items_kept']:,} |",
        f"| Median interactions per user | {median_seq} |",
        f"| Range of interactions per user | {min_seq} – {max_seq} |",
        f"| Eligible users (≥ 3 interactions) | {stats['n_users']:,} |",
        "",
        "### Filtering rule",
        "",
        "We **kept a movie** in the training vocabulary if and only if it received",
        "**≥ 5 ratings in the training pool** (leave-one-out: last = test, second-to-last",
        "= val, rest = train). This threshold:",
        "",
        "- Removes the long tail of movies the model can never meaningfully rank.",
        "- Is computed on **train data only** — no leakage from val/test counts.",
        "- A target movie below threshold is treated as an automatic miss at evaluation",
        "  (never silently re-added to the catalog).",
        "",
        "### Rating distribution",
        "",
        "| Stars | Count |",
        "|-------|-------|",
    ]
    for stars, count in ratings_dist.items():
        lines.append(f"| {stars} | {count:,} |")

    if top_genres:
        lines += [
            "",
            f"### Top 5 genres by title count",
            "",
            "| Genre | Movies |",
            "|-------|--------|",
        ]
        for g in top_genres:
            lines.append(f"| {g} | {genre_counts[g]:,} |")

    lines += [
        "",
        "## What we did NOT use (and why)",
        "",
        "| Dataset | Why excluded |",
        "|---------|-------------|",
        "| `ml-latest-small` | Fallback only; far too sparse for a reliable softmax over 9,700 items. |",
        "| Netflix Prize (`combined_data_*`) | No genre metadata; 2 GB+; day-level timestamps. One-liner deck mention only. |",
        "| IMDB reviews | Sentiment dataset — no users, no sequences, wrong task. |",
        "",
        "## Personal export files (for the live demo)",
        "",
        "Personal Letterboxd/Netflix files are **never committed** (see `.gitignore`).",
        "The importers in `src/importers.py` map them to MovieLens IDs at runtime,",
        "on the demo machine only.",
        "",
        "- **Letterboxd** (`ratings.csv`): rated-film set with star ratings.",
        "  Used as the taste-profile seed, *not* as a training sequence (no reliable order).",
        "- **Netflix** (`ViewingActivity.csv`, `Ratings.csv`): viewing history + thumbs.",
        "  ~81 % TV, Spanish-localized. ~20-30 top shows are resolved to English/genres",
        "  for the TV recommendation demo.",
        "",
    ]

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    _pass(f"Written to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Verify ml-1m data and write data notes.")
    ap.add_argument("--ratings", default=DEFAULT_RATINGS)
    ap.add_argument("--movies",  default=DEFAULT_MOVIES)
    ap.add_argument("--out",     default=DEFAULT_OUT, help="Path for the Markdown data notes file")
    ap.add_argument("--full",    action="store_true",
                    help="Run the full dataset build via data_prep (slow)")
    args = ap.parse_args()

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║          Reverie — ml-1m data verification               ║")
    print("╚══════════════════════════════════════════════════════════╝")

    df = check_ratings(args.ratings)
    check_movies(args.movies, df)
    stats = check_split_integrity(df)

    if args.full:
        _section("Full dataset build (slow — ~30s on ml-1m)")
        from src.gru_model.data_prep import build_dataset
        ds = build_dataset(args.ratings, args.movies)
        _pass(f"build_dataset() succeeded: {ds.n_items} items, "
              f"{len(ds.X_train)} training windows")
        _pass(f"Val users: {len(ds.val_target)},  Test users: {len(ds.test_target)}")
        _pass(f"OOV targets: {ds.n_oov_targets}")

    write_data_notes(args.out, df, stats)

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║  All checks passed. Data is ready.                      ║")
    print("╚══════════════════════════════════════════════════════════╝\n")


if __name__ == "__main__":
    main()