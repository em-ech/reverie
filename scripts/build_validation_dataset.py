"""Build the temporal-holdout validation dataset from a Letterboxd export.

Reads a Letterboxd `ratings.csv` (Name, Year, Rating), looks each film up on
TMDB, and writes `data/validation_dataset.csv` with one row per rated film:

    title, year, rating, rated_3plus,
    <18 ml-1m genre columns>,
    vote_average, vote_count, popularity, runtime, is_foreign,
    director, lead_actor, keywords

Director/actor *affinity* and keyword features are derived later, in the
notebook, from the TRAINING films only (no leakage). This script just records
the raw director / lead actor / keyword names.

One-time data step (network-bound, ~2 TMDB calls per film). Personal ratings
stay local; `data/validation_dataset.csv` is gitignored.

Usage
-----
    conda activate deep-learning
    TMDB_API_KEY=xxxx python -m scripts.build_validation_dataset \
        --letterboxd data/letterboxd_ratings.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import requests

from src.catalog.importers import _normalise_title

SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
DETAILS_URL = "https://api.themoviedb.org/3/movie/{id}"
PACING_SECONDS = 0.28
MAX_RETRIES = 3

ML_GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

# TMDB genre id -> one or more ml-1m genres.
TMDB_ID_TO_ML: dict[int, list[str]] = {
    28: ["Action"], 12: ["Adventure"], 16: ["Animation", "Children's"],
    35: ["Comedy"], 80: ["Crime"], 99: ["Documentary"], 18: ["Drama"],
    10751: ["Children's"], 14: ["Fantasy"], 36: ["Drama"], 27: ["Horror"],
    10402: ["Musical"], 9648: ["Mystery"], 10749: ["Romance"], 878: ["Sci-Fi"],
    53: ["Thriller"], 10752: ["War"], 37: ["Western"], 10770: [],
}


def _get(session: requests.Session, url: str, params: dict) -> dict | None:
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(url, params=params, timeout=15)
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


def _genre_vector(genres: list[dict]) -> list[int]:
    present: set[str] = set()
    for g in genres:
        present.update(TMDB_ID_TO_ML.get(g.get("id"), []))
    return [1 if g in present else 0 for g in ML_GENRES]


def _director(credits: dict) -> str:
    for c in credits.get("crew", []):
        if c.get("job") == "Director":
            return c.get("name", "")
    return ""


def _lead_actor(credits: dict) -> str:
    cast = sorted(credits.get("cast", []), key=lambda c: c.get("order", 999))
    return cast[0].get("name", "") if cast else ""


def _read_letterboxd(path: str) -> list[tuple[str, str, float]]:
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
    print(f"{len(films)} rated films; fetching genres + details from TMDB...")

    session = requests.Session()
    out_rows: list[dict] = []
    missed = 0
    for i, (title, year, rating) in enumerate(films, 1):
        params = {"api_key": api_key, "query": title, "include_adult": "false",
                  "language": "en-US"}
        if year:
            params["year"] = year
        search = _get(session, SEARCH_URL, params)
        best = _pick(search.get("results", []), title, year) if search else None
        time.sleep(PACING_SECONDS)

        details = None
        if best and best.get("id"):
            details = _get(
                session, DETAILS_URL.format(id=best["id"]),
                {"api_key": api_key, "append_to_response": "credits,keywords",
                 "language": "en-US"},
            )
            time.sleep(PACING_SECONDS)

        if not details:
            missed += 1
        else:
            vec = _genre_vector(details.get("genres", []))
            credits = details.get("credits", {})
            kws = [k.get("name", "") for k in details.get("keywords", {}).get("keywords", [])]
            row = {
                "title": title, "year": year, "rating": rating,
                "rated_3plus": int(rating >= 3),
                "vote_average": details.get("vote_average") or 0,
                "vote_count": details.get("vote_count") or 0,
                "popularity": details.get("popularity") or 0,
                "runtime": details.get("runtime") or 0,
                "is_foreign": int((details.get("original_language") or "en") != "en"),
                "director": _director(credits),
                "lead_actor": _lead_actor(credits),
                "keywords": "|".join(k for k in kws if k),
            }
            row.update({g: v for g, v in zip(ML_GENRES, vec)})
            out_rows.append(row)

        if i % 50 == 0:
            print(f"  {i}/{len(films)} done ({len(out_rows)} kept)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fields = ["title", "year", "rating", "rated_3plus", *ML_GENRES,
              "vote_average", "vote_count", "popularity", "runtime", "is_foreign",
              "director", "lead_actor", "keywords"]
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    pre = sum(1 for r in out_rows if r["year"].isdigit() and int(r["year"]) < 2020)
    post = len(out_rows) - pre
    print(f"Done. {len(out_rows)} films kept ({missed} unresolved). "
          f"pre-2020: {pre}, 2020+: {post}. -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
