"""Personal-history importers for the Reverie demo.

Converts a user's Letterboxd or Netflix export into the
  list[tuple[movieId, rating_stars]]
format consumed by recommend.recommend_movies() and taste_vector().

PRIVACY RULES (enforced here):
  * Files are read-only from disk; nothing is written back.
  * No personal file is ever uploaded to GitHub — all personal paths are
    .gitignored at the repo level. The functions here run on the demo machine
    only and return plain Python objects.
  * The importer strips every column that is not needed for inference.

Usage
-----
    from src.importers import load_letterboxd, load_netflix

    # --- Letterboxd ---
    lbx = load_letterboxd("path/to/letterboxd_export/ratings.csv",
                          movie_to_id=state["movie_to_id"])
    # lbx: list[tuple[int, float]]  (movieId, stars)

    # --- Netflix ---
    nfx_movies, nfx_tv = load_netflix(
        viewing_path="path/to/NetflixViewingHistory/ViewingActivity.csv",
        ratings_path="path/to/NetflixRatings/Ratings.csv",   # optional
        movie_to_id=state["movie_to_id"],
        tv_title_map=tv_title_map,    # from src.tv_catalog.load_tv_catalog()
    )
    # nfx_movies: list[tuple[int, float]]   (movieId, stars) — feeds recommend_movies
    # nfx_tv:     list[tuple[str, float]]   (english_title, stars) — feeds score_tv

Both functions return results sorted chronologically where timestamps are
available, falling back to file order.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_stars(val: str | float | int | None, default: float = 4.0) -> float:
    """Normalize any rating representation to a 0.5–5.0 star float.

    Handles:
      * Letterboxd  "5", "4.5", "3", ""  (already in stars)
      * Netflix     1 (thumb down) → 2.0,  2 (thumb up) → 4.0  (pre-2017 numeric)
                    "thumb_down" / "thumb_up"  (post-2017 string)
    Falls back to *default* for any unrecognised or missing value.
    """
    if val is None or (isinstance(val, str) and not val.strip()):
        return default
    if isinstance(val, (int, float)):
        v = float(val)
        if v <= 2:   # Netflix binary: 1 = dislike, 2 = like
            return 2.0 if v == 1 else 4.0
        return max(0.5, min(5.0, v))
    s = str(val).strip().lower()
    if s in ("thumb_down", "thumbsdown", "dislike", "not_interested"):
        return 2.0
    if s in ("thumb_up", "thumbsup", "like", "thumbs_up"):
        return 4.0
    try:
        return max(0.5, min(5.0, float(s)))
    except ValueError:
        return default


def _normalise_title(title: str) -> str:
    """Lower-case, strip year suffixes and punctuation for fuzzy matching."""
    t = title.lower().strip()
    t = re.sub(r"\s*\(\d{4}\)\s*$", "", t)     # remove trailing " (2001)"
    t = re.sub(r"[^\w\s]", " ", t)             # punctuation → space
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _build_title_lookup(movie_to_id: dict[int, int], movies_dat_path: str | None) -> dict[str, int]:
    """Build normalised_title → movieId from movies.dat if available."""
    lookup: dict[str, int] = {}
    if not movies_dat_path:
        return lookup
    p = Path(movies_dat_path)
    if not p.exists():
        return lookup
    with open(p, encoding="latin-1") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("::")
            if len(parts) != 3:
                continue
            mid_str, title, _ = parts
            mid = int(mid_str)
            if mid in movie_to_id:
                lookup[_normalise_title(title)] = mid
    return lookup


# ---------------------------------------------------------------------------
# Letterboxd importer
# ---------------------------------------------------------------------------

def load_letterboxd(
    ratings_csv: str,
    movie_to_id: dict[int, int],
    movies_dat_path: str = "data/ml-1m/movies.dat",
) -> list[tuple[int, float]]:
    """Read a Letterboxd ratings.csv and return (movieId, stars) pairs.

    Matching strategy (in order):
      1. Exact title + year match against movies.dat.
      2. Normalised-title match (punctuation-insensitive, no year).
      3. Skip rows that don't map (logged to stderr via a counter).

    Letterboxd exports have no reliable chronological ordering (most users
    bulk-log on a single date), so the returned list preserves file order.
    The caller should NOT assume it is an ordered watch sequence.

    Parameters
    ----------
    ratings_csv      : path to the Letterboxd `ratings.csv` export file
    movie_to_id      : {original_movieId: contiguous_id} from the trained model state
    movies_dat_path  : path to ml-1m movies.dat for title lookup
    """
    p = Path(ratings_csv)
    if not p.exists():
        raise FileNotFoundError(f"Letterboxd ratings file not found: {p}")

    title_lookup = _build_title_lookup(movie_to_id, movies_dat_path)

    results: list[tuple[int, float]] = []
    missed = 0
    total  = 0

    with open(p, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            total += 1
            raw_title = row.get("Name", row.get("Title", "")).strip()
            year_str  = row.get("Year", "").strip()
            stars_str = row.get("Rating", "").strip()
            stars     = _parse_stars(stars_str)

            # Strategy 1: title (year)
            full_title = f"{raw_title} ({year_str})" if year_str else raw_title
            norm_full  = _normalise_title(full_title)
            mid = title_lookup.get(norm_full)

            # Strategy 2: title only (no year)
            if mid is None:
                norm_bare = _normalise_title(raw_title)
                mid = title_lookup.get(norm_bare)

            if mid is not None and mid in movie_to_id:
                results.append((mid, stars))
            else:
                missed += 1

    match_rate = 100 * (total - missed) / max(total, 1)
    print(
        f"[importers] Letterboxd: {total} rows → "
        f"{len(results)} matched ({match_rate:.0f}%), "
        f"{missed} unmatched (not in ml-1m catalog)"
    )
    return results


# ---------------------------------------------------------------------------
# Netflix importer
# ---------------------------------------------------------------------------

def load_netflix(
    viewing_path: str,
    movie_to_id: dict[int, int],
    ratings_path: str | None = None,
    tv_title_map: dict[str, str] | None = None,
    movies_dat_path: str = "data/ml-1m/movies.dat",
    default_watch_stars: float = 3.5,
) -> tuple[list[tuple[int, float]], list[tuple[str, float]]]:
    """Read Netflix export files and split into movie and TV histories.

    Parameters
    ----------
    viewing_path        : path to ViewingActivity.csv
    movie_to_id         : {original_movieId: contiguous_id} from model state
    ratings_path        : optional path to Ratings.csv (thumb ratings)
    tv_title_map        : {spanish_title: english_title} from tv_catalog; if None,
                          TV items are matched by raw title
    movies_dat_path     : path to ml-1m movies.dat for title lookup
    default_watch_stars : star value assigned to a watched item with no rating

    Returns
    -------
    movies_history : list[tuple[movieId, stars]] sorted chronologically
    tv_history     : list[tuple[english_title, stars]] sorted chronologically

    Notes
    -----
    The ViewingActivity.csv from Netflix contains a "Title" column that is
    usually "Series: Season X: Episode Y" for TV and just a movie title for
    films. A row is classified as TV if a colon is present after the first
    word (the series name pattern), or if it matches a key in tv_title_map.
    Spanish-localised titles are resolved via tv_title_map before matching.
    """
    vp = Path(viewing_path)
    if not vp.exists():
        raise FileNotFoundError(f"Netflix ViewingActivity not found: {vp}")

    title_lookup = _build_title_lookup(movie_to_id, movies_dat_path)

    # Build a star-rating lookup from Ratings.csv if available
    # Netflix Ratings.csv: columns vary by export year; look for Title + Rating
    thumb_map: dict[str, float] = {}
    if ratings_path:
        rp = Path(ratings_path)
        if rp.exists():
            with open(rp, newline="", encoding="utf-8-sig") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    title = row.get("Title", row.get("title", "")).strip()
                    rating_val = row.get("Rating", row.get("rating", "")).strip()
                    if title:
                        thumb_map[title.lower()] = _parse_stars(rating_val)
        else:
            print(f"[importers] Netflix ratings file not found, skipping: {rp}")

    tv_map = {k.lower(): v for k, v in (tv_title_map or {}).items()}

    # Parse viewing activity; rows are newest-first in Netflix exports
    rows_raw: list[dict] = []
    with open(vp, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows_raw.append(row)

    # Reverse so we process oldest → newest (chronological for the model)
    rows_raw = list(reversed(rows_raw))

    movies_history: list[tuple[int, float]] = []
    tv_history:     list[tuple[str, float]] = []
    skipped = 0

    for row in rows_raw:
        raw_title = row.get("Title", row.get("title", "")).strip()
        if not raw_title:
            skipped += 1
            continue

        # Netflix TV rows look like "Dark: Season 1: Episode 1: Secrets"
        # Detect TV by the presence of ": Season" or ": Episode" patterns,
        # or by appearing in the tv_title_map.
        is_tv = bool(
            re.search(r":\s*(Season|Episode|Part|Series|Chapter|Vol\.)\s*\d", raw_title, re.I)
            or raw_title.lower() in tv_map
        )

        # Extract the series name (everything before the first season/episode marker)
        if is_tv:
            series_match = re.match(r"^(.+?):\s*(Season|Episode|Part|Series|Chapter|Vol\.)\s*\d", raw_title, re.I)
            series_name = series_match.group(1).strip() if series_match else raw_title.split(":")[0].strip()
        else:
            series_name = raw_title

        # Star rating: use the thumb map if available
        stars = thumb_map.get(raw_title.lower(), thumb_map.get(series_name.lower(), default_watch_stars))

        if is_tv:
            # Resolve Spanish title → English via tv_title_map
            english_title = tv_map.get(series_name.lower(), series_name)
            tv_history.append((english_title, stars))
        else:
            # Try to map to a MovieLens movieId
            norm = _normalise_title(raw_title)
            mid  = title_lookup.get(norm)
            if mid is None:
                norm2 = _normalise_title(series_name)
                mid = title_lookup.get(norm2)
            if mid is not None and mid in movie_to_id:
                movies_history.append((mid, stars))
            else:
                skipped += 1

    # De-duplicate: keep last-seen rating per movieId / TV title
    def _dedup_movies(seq: list[tuple[int, float]]) -> list[tuple[int, float]]:
        seen: dict[int, float] = {}
        for mid, stars in seq:
            seen[mid] = stars          # last occurrence wins (most recent watch)
        return list(seen.items())

    def _dedup_tv(seq: list[tuple[str, float]]) -> list[tuple[str, float]]:
        seen: dict[str, float] = {}
        for title, stars in seq:
            seen[title] = stars
        return list(seen.items())

    movies_history = _dedup_movies(movies_history)
    tv_history     = _dedup_tv(tv_history)

    print(
        f"[importers] Netflix: {len(rows_raw)} rows → "
        f"{len(movies_history)} movies matched, "
        f"{len(tv_history)} TV series resolved, "
        f"{skipped} skipped/unmatched"
    )
    return movies_history, tv_history


# ---------------------------------------------------------------------------
# Quick smoke test (run as script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json, sys

    print("Importers smoke test — pass --letterboxd or --netflix flags")
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--letterboxd", help="Path to Letterboxd ratings.csv")
    ap.add_argument("--netflix_view", help="Path to Netflix ViewingActivity.csv")
    ap.add_argument("--netflix_ratings", help="Path to Netflix Ratings.csv (optional)")
    ap.add_argument("--movie_index", default="artifacts/movie_index.json")
    ap.add_argument("--movies_dat",  default="data/ml-1m/movies.dat")
    args = ap.parse_args()

    if not (args.letterboxd or args.netflix_view):
        print("Nothing to test — provide --letterboxd or --netflix_view")
        sys.exit(0)

    with open(args.movie_index) as fh:
        idx = json.load(fh)
    movie_to_id = {int(k): int(v) for k, v in idx["movie_to_id"].items()}

    if args.letterboxd:
        result = load_letterboxd(args.letterboxd, movie_to_id, args.movies_dat)
        print(f"  → {len(result)} movies from Letterboxd")
        for mid, stars in result[:5]:
            print(f"     movieId={mid}  stars={stars}")

    if args.netflix_view:
        movies, tv = load_netflix(
            args.netflix_view, movie_to_id,
            ratings_path=args.netflix_ratings,
            movies_dat_path=args.movies_dat,
        )
        print(f"  → {len(movies)} movies,  {len(tv)} TV series from Netflix")
        for mid, stars in movies[:5]:
            print(f"     movieId={mid}  stars={stars}")
        for title, stars in tv[:5]:
            print(f"     TV: '{title}'  stars={stars}")