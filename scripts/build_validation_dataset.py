"""Build the temporal-holdout validation dataset from a Letterboxd export.

Reads a Letterboxd `ratings.csv` (Name, Year, Rating), looks each film up on
TMDB to get its genres, maps them to the 18 MovieLens genres, and writes
`data/validation_dataset.csv` with one row per rated film:

    title, year, rating, rated_3plus, <18 genre columns>

Used by `notebooks/temporal_validation.ipynb` to test whether genre-based taste
learned from PRE-2020 ratings predicts which POST-2020 films get rated 3+.

This is a ONE-TIME data step (network-bound, like build_posters). Personal
ratings stay local; `data/validation_dataset.csv` is gitignored.

Usage
-----
    conda activate deep-learning
    TMDB_API_KEY=xxxx python -m scripts.build_validation_dataset \
        --letterboxd /path/to/letterboxd/ratings.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import requests

from src.importers import _normalise_title

SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
PACING_SECONDS = 0.28
MAX_RETRIES = 3

# The 18 MovieLens (ml-1m) genres, in the model's canonical order.
ML_GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

# TMDB genre id -> one or more ml-1m genres. (TMDB has no Film-Noir; History and
# TV Movie map to the closest ml-1m bucket or are dropped.)
TMDB_ID_TO_ML: dict[int, list[str]] = {
    28: ["Action"],
    12: ["Adventure"],
    16: ["Animation", "Children's"],
    35: ["Comedy"],
    80: ["Crime"],
    99: ["Documentary"],
    18: ["Drama"],
    10751: ["Children's"],          # Family
    14: ["Fantasy"],
    36: ["Drama"],                  # History
    27: ["Horror"],
    10402: ["Musical"],             # Music
    9648: ["Mystery"],
    10749: ["Romance"],
    878: ["Sci-Fi"],                # Science Fiction
    53: ["Thriller"],
    10752: ["War"],
    37: ["Western"],
    10770: [],                      # TV Movie -> drop
}


def _request(session: requests.Session, params: dict) -> dict | None:
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(SEARCH_URL, params=params, timeout=15)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 429:
            time.sleep(int(r.headers.get("Retry-After", "2")) + 1)
            continue
        if r.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        if r.status_code != 200:
            return None
        return r.json()
    return None


def _pick(results: list[dict], title: str, year: str) -> dict | None:
    if not results:
        return None
    if year:
        year_hits = [r for r in results if (r.get("release_date") or "")[:4] == year]
        if year_hits:
            results = year_hits
    norm = _normalise_title(title)
    exact = [r for r in results if _normalise_title(r.get("title", "")) == norm]
    return exact[0] if exact else results[0]


def _genre_vector(genre_ids: list[int]) -> list[int]:
    present = set()
    for gid in genre_ids:
        present.update(TMDB_ID_TO_ML.get(gid, []))
    return [1 if g in present else 0 for g in ML_GENRES]


def _read_letterboxd(path: str) -> list[tuple[str, str, float]]:
    """(title, year, rating) for every rated row."""
    rows: list[tuple[str, str, float]] = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("Name") or row.get("Title") or "").strip()
            year = (row.get("Year") or "").strip()
            rating_raw = (row.get("Rating") or "").strip()
            if not name or not rating_raw:
                continue
            try:
                rating = float(rating_raw)
            except ValueError:
                continue
            rows.append((name, year, rating))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the validation dataset from Letterboxd.")
    ap.add_argument("--letterboxd", required=True, help="Path to Letterboxd ratings.csv")
    ap.add_argument("--out", default="data/validation_dataset.csv")
    args = ap.parse_args()

    api_key = os.environ.get("TMDB_API_KEY")
    if not api_key:
        print("ERROR: set TMDB_API_KEY (themoviedb.org -> Settings -> API).", file=sys.stderr)
        return 1

    films = _read_letterboxd(args.letterboxd)
    print(f"{len(films)} rated films in the export; resolving genres via TMDB...")

    session = requests.Session()
    out_rows: list[dict] = []
    missed = 0
    for i, (title, year, rating) in enumerate(films, 1):
        params = {"api_key": api_key, "query": title, "include_adult": "false",
                  "language": "en-US"}
        if year:
            params["year"] = year
        data = _request(session, params)
        best = _pick(data.get("results", []), title, year) if data else None
        if not best or not best.get("genre_ids"):
            missed += 1
        else:
            vec = _genre_vector(best.get("genre_ids", []))
            if sum(vec) == 0:
                missed += 1
            else:
                row = {"title": title, "year": year, "rating": rating,
                       "rated_3plus": int(rating >= 3)}
                row.update({g: v for g, v in zip(ML_GENRES, vec)})
                out_rows.append(row)
        if i % 50 == 0:
            print(f"  {i}/{len(films)} resolved ({len(out_rows)} kept)")
        time.sleep(PACING_SECONDS)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fields = ["title", "year", "rating", "rated_3plus", *ML_GENRES]
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    pre = sum(1 for r in out_rows if r["year"].isdigit() and int(r["year"]) < 2020)
    post = len(out_rows) - pre
    pos = sum(r["rated_3plus"] for r in out_rows)
    print(f"Done. {len(out_rows)} films kept ({missed} unresolved). "
          f"pre-2020: {pre}, 2020+: {post}. rated 3+: {pos} "
          f"({100 * pos / max(len(out_rows), 1):.0f}%). -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
