"""Offline TMDB poster fetcher for the Reverie demo.

Matches every recommendable MovieLens (ml-1m) title to a TMDB movie and caches
its poster + backdrop image URLs to `artifacts/posters.json`, so the API/UI can
show real posters without any API key at serve time.

This is a ONE-TIME build step, not part of training or serving. The model and
its artifacts are never touched (read-only against movies.dat + movie_to_id).

Usage
-----
    conda activate deep-learning
    TMDB_API_KEY=xxxxxxxx python -m src.catalog.build_posters
    # resumable: re-run to fill in any movies still missing (e.g. after a 429)

Get a free key at themoviedb.org -> Settings -> API.

Output (`artifacts/posters.json`)
---------------------------------
    { "1": {"tmdb_id": 862,
            "poster_url":   "https://image.tmdb.org/t/p/w500/<hash>.jpg",
            "backdrop_url": "https://image.tmdb.org/t/p/w1280/<hash>.jpg"},
      ... }
A null poster_url/backdrop_url is recorded for misses so we don't re-query them.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import requests

from src.gru_model import recommend as rec

MOVIES_DAT = "data/ml-1m/movies.dat"
OUT_PATH = "artifacts/posters.json"

TMDB_SEARCH = "https://api.themoviedb.org/3/search/movie"
IMG_BASE = "https://image.tmdb.org/t/p/"
POSTER_SIZE = "w500"
BACKDROP_SIZE = "w1280"

PACING_SECONDS = 0.28   # ~36 req / 10s, under TMDB's ~40/10s limit
MAX_RETRIES = 3
CHECKPOINT_EVERY = 200

_ARTICLE_RE = re.compile(r"^(.*),\s+(The|A|An|La|Le|Les|Il|El|Das|Der|Die)\s*$")


def _parse_movies_dat() -> dict[int, tuple[str, str]]:
    """movieId -> (clean_title, year), restricted to recommendable items.

    Reorders MovieLens' trailing article so "Matrix, The (1999)" becomes
    ("The Matrix", "1999") — the form TMDB search expects.
    """
    recommendable = set(rec.load()["movie_to_id"].keys())
    out: dict[int, tuple[str, str]] = {}
    with open(MOVIES_DAT, encoding="latin-1") as fh:
        for line in fh:
            mid_str, raw, _genres = line.rstrip("\n").split("::")
            mid = int(mid_str)
            if mid not in recommendable:
                continue
            year = raw[-5:-1] if raw.endswith(")") else ""
            title = raw[:-7].strip() if year else raw.strip()
            m = _ARTICLE_RE.match(title)
            if m:
                title = f"{m.group(2)} {m.group(1)}".strip()
            out[mid] = (title, year)
    return out


def _normalise(title: str) -> str:
    """Lower-case, drop punctuation, collapse whitespace (for title matching)."""
    t = re.sub(r"[^\w\s]", " ", title.lower())
    return re.sub(r"\s+", " ", t).strip()


def _request(session: requests.Session, params: dict) -> dict | None:
    """GET the TMDB search endpoint with rate-limit + backoff handling."""
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(TMDB_SEARCH, params=params, timeout=15)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", "2"))
            time.sleep(wait + 1)
            continue
        if r.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        if r.status_code != 200:
            return None
        return r.json()
    return None


def _pick(results: list[dict], title: str, year: str) -> dict | None:
    """Choose the best TMDB result: exact year first, then exact title, else top."""
    if not results:
        return None
    if year:
        year_hits = [r for r in results if (r.get("release_date") or "")[:4] == year]
        if year_hits:
            results = year_hits
    norm = _normalise(title)
    exact = [r for r in results if _normalise(r.get("title", "")) == norm]
    return exact[0] if exact else results[0]


def _img(path: str | None, size: str) -> str | None:
    return f"{IMG_BASE}{size}{path}" if path else None


def _save(cache: dict, path: Path) -> None:
    path.write_text(json.dumps(cache, indent=0, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    api_key = os.environ.get("TMDB_API_KEY")
    if not api_key:
        print("ERROR: set TMDB_API_KEY (themoviedb.org -> Settings -> API).", file=sys.stderr)
        return 1

    movies = _parse_movies_dat()
    out_path = Path(OUT_PATH)
    cache: dict[str, dict] = {}
    if out_path.exists():
        cache = json.loads(out_path.read_text(encoding="utf-8"))

    # Re-fetch entries that predate the `rating` field so a re-run tops them up.
    todo = [
        (mid, t, y) for mid, (t, y) in movies.items()
        if str(mid) not in cache or "rating" not in cache[str(mid)]
    ]
    print(f"{len(movies)} recommendable movies; {len(cache)} cached; {len(todo)} to fetch.")

    session = requests.Session()
    hits = sum(1 for v in cache.values() if v.get("poster_url"))
    for i, (mid, title, year) in enumerate(todo, 1):
        params = {"api_key": api_key, "query": title, "include_adult": "false",
                  "language": "en-US"}
        if year:
            params["year"] = year
        data = _request(session, params)
        best = _pick(data.get("results", []), title, year) if data else None
        if best:
            cache[str(mid)] = {
                "tmdb_id": best.get("id"),
                "poster_url": _img(best.get("poster_path"), POSTER_SIZE),
                "backdrop_url": _img(best.get("backdrop_path"), BACKDROP_SIZE),
                "rating": best.get("vote_average"),  # TMDB audience score, 0-10
            }
            if cache[str(mid)]["poster_url"]:
                hits += 1
        else:
            cache[str(mid)] = {"tmdb_id": None, "poster_url": None, "backdrop_url": None, "rating": None}

        if i % CHECKPOINT_EVERY == 0:
            _save(cache, out_path)
            print(f"  {i}/{len(todo)} fetched, {hits} with posters (checkpoint saved)")
        time.sleep(PACING_SECONDS)

    _save(cache, out_path)
    with_posters = sum(1 for v in cache.values() if v.get("poster_url"))
    print(f"Done. {len(cache)} movies cached, {with_posters} with posters "
          f"({100 * with_posters / max(len(cache), 1):.0f}%). -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
