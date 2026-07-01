"""Movie metadata + poster enrichment + Netflix-style match scoring.

The single authority for turning a movieId (and optional model score) into the
title/genre/poster/rating payload the frontend renders. Shared by every router
and service that returns movies, so this logic lives in exactly one place.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from app.config import settings
from src.gru_model import recommend as rec

MOVIES_DAT = "data/ml-1m/movies.dat"
POSTERS_JSON = "artifacts/posters.json"
PROVIDERS_JSON = "artifacts/providers.json"
MODERN_CATALOG = "artifacts/modern/catalog.json"

# Netflix-style match-% display band over the returned top-N (log-normalized).
_MATCH_CEIL = 0.99
_MATCH_FLOOR = 0.80

_titles: dict[int, dict] = {}
_posters: dict[int, dict] = {}
_providers: dict[int, dict] = {}
_recommendable: set[int] = set()


def is_modern() -> bool:
    return settings.catalog_mode == "modern"


def load_metadata() -> None:
    """Populate the title + poster (+ provider) caches once at startup. In modern
    mode the catalog is the Letterboxd films the NCF model can score; otherwise
    it is MovieLens ml-1m for the GRU demo."""
    if is_modern():
        _load_modern()
    else:
        _load_titles()
        _load_posters()
        _load_providers()


def _load_modern() -> None:
    """The modern catalog: tmdb_id -> title/year/genres/poster/rating, built by
    scripts/build_modern_catalog.py."""
    if not Path(MODERN_CATALOG).exists():
        raise FileNotFoundError(
            f"Modern catalog not found at {MODERN_CATALOG}. Build it with "
            "`python -m scripts.build_modern_catalog --movies <movie_data.csv>`, "
            "or set REVERIE_CATALOG_MODE=ml1m for the MovieLens demo."
        )
    with open(MODERN_CATALOG, encoding="utf-8") as fh:
        catalog = json.load(fh)
    for k, v in catalog.items():
        mid = int(k)
        _titles[mid] = {
            "movieId": mid, "title": v["title"], "year": v["year"], "genres": v["genres"],
        }
        _posters[mid] = {
            "poster_url": v.get("poster_url"), "backdrop_url": None, "rating": v.get("rating"),
        }
        _recommendable.add(mid)


def _load_titles() -> None:
    recommendable = recommendable_ids()
    with open(MOVIES_DAT, encoding="latin-1") as fh:
        for line in fh:
            mid_str, title, genres = line.rstrip("\n").split("::")
            mid = int(mid_str)
            if mid in recommendable:
                year = title[-5:-1] if title.endswith(")") else ""
                _titles[mid] = {
                    "movieId": mid,
                    "title": title[:-7].strip() if year else title,
                    "year": year,
                    "genres": genres.split("|"),
                }


def _load_posters() -> None:
    p = Path(POSTERS_JSON)
    if not p.exists():
        return
    with open(p, encoding="utf-8") as fh:
        raw = json.load(fh)
    for k, v in raw.items():
        _posters[int(k)] = {
            "poster_url": v.get("poster_url"),
            "backdrop_url": v.get("backdrop_url"),
            "rating": v.get("rating"),  # TMDB audience score, 0-10 (None if absent)
        }


def _load_providers() -> None:
    """movieId -> {US: {...}, ES: {...}} streaming availability (optional)."""
    p = Path(PROVIDERS_JSON)
    if not p.exists():
        return
    with open(p, encoding="utf-8") as fh:
        raw = json.load(fh)
    for k, v in raw.items():
        if v:  # skip movies with no providers in any region
            _providers[int(k)] = v


def recommendable_ids() -> set[int]:
    """Ids the app can recommend/encode: modern catalog tmdb_ids, or the frozen
    GRU's MovieLens ids in ml1m mode."""
    if is_modern():
        return _recommendable
    return set(rec.load()["movie_to_id"].keys())


def catalog_size() -> int:
    return len(_titles)


def poster_count() -> int:
    return len(_posters)


def titles() -> dict[int, dict]:
    return _titles


def enrich(movie_id: int, score: float | None = None) -> dict:
    """Title + genre + poster/rating metadata; optional raw model score."""
    meta = _titles.get(
        movie_id, {"movieId": movie_id, "title": str(movie_id), "year": "", "genres": []}
    )
    poster = _posters.get(
        movie_id, {"poster_url": None, "backdrop_url": None, "rating": None}
    )
    out = {**meta, **poster, "providers": _providers.get(movie_id)}
    if score is not None:
        out["score"] = round(float(score), 4)
    return out


def top_titles(history: list[tuple[int, float]], k: int = 2) -> list[str]:
    """Titles of the highest-rated films in a history, for the taste blurb."""
    top = sorted(history, key=lambda h: h[1], reverse=True)[:k]
    return [enrich(mid)["title"] for mid, _ in top]


def match_from_ratings(ratings: list[float]) -> list[int]:
    """Map NCF predicted ratings (1-10) onto the Netflix-style 80-99% match band,
    monotonic in the predicted rating. Used by the modern (collaborative) path."""
    if not ratings:
        return []
    hi, lo = max(ratings), min(ratings)
    if hi == lo:
        return [round(_MATCH_CEIL * 100)] * len(ratings)
    span = _MATCH_CEIL - _MATCH_FLOOR
    return [round((_MATCH_FLOOR + span * (r - lo) / (hi - lo)) * 100) for r in ratings]


def match_scores(scores: list[float]) -> list[int]:
    """Map raw softmax scores onto a Netflix-style 80-99% match band, monotonic
    in score; degenerate inputs (one item or all-equal) return the ceiling."""
    logs = [math.log(s) if s > 0 else None for s in scores]
    finite = [l for l in logs if l is not None]
    if not finite:
        return [round(_MATCH_CEIL * 100)] * len(scores)
    hi, lo = max(finite), min(finite)
    if hi == lo:
        return [round(_MATCH_CEIL * 100)] * len(scores)
    out: list[int] = []
    for l in logs:
        if l is None:
            out.append(round(_MATCH_FLOOR * 100))
        else:
            r = (l - lo) / (hi - lo)
            out.append(round((_MATCH_FLOOR + (_MATCH_CEIL - _MATCH_FLOOR) * r) * 100))
    return out
