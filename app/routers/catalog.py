"""Catalog, recommendations, and watchlist import.

These endpoints are PUBLIC (no auth) so the original single-user demo keeps
working. They wrap the frozen model in src/recommend.py via app.enrich.
"""

from __future__ import annotations

import contextlib
import io
import os
import random
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app import enrich
from app.services.taste_service import taste_blurb
from src import importers
from src import ncf_recommend as ncf
from src import recommend as rec

router = APIRouter(tags=["catalog"])

MOVIES_DAT = enrich.MOVIES_DAT


class HistItem(BaseModel):
    movieId: int
    rating: float = 4.0  # stars


class RecRequest(BaseModel):
    history: list[HistItem] = []
    n: int = 12


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "catalog_size": enrich.catalog_size(), "posters": enrich.poster_count()}


@router.get("/catalog")
def catalog(q: str = "", limit: int = 20, genre: str = "") -> list[dict]:
    """Title search for building a watch history (autocomplete), with an
    optional genre filter used to populate the genre rows in the UI."""
    ql = q.lower().strip()
    gl = genre.strip()
    items = (
        m for m in enrich.titles().values()
        if (not ql or ql in m["title"].lower())
        and (not gl or gl in m["genres"])
    )
    out = sorted(items, key=lambda m: m["title"])[:limit]
    return [enrich.enrich(m["movieId"]) for m in out]


@router.get("/catalog/browse")
def browse(genre: str = "", limit: int = 30, exclude: str = "") -> list[dict]:
    """A poster-rich, acclaimed pool to swipe or scroll through while building a
    history (cold start). Sorted by audience rating, then shuffled within the
    top band so the deck feels fresh. Optional genre filter and excluded ids."""
    gl = genre.strip()
    drop = {int(x) for x in exclude.split(",") if x.strip().lstrip("-").isdigit()}
    pool = [
        e
        for m in enrich.titles().values()
        if (e := enrich.enrich(m["movieId"]))["movieId"] not in drop
        and e.get("poster_url")
        and (not gl or gl in e["genres"])
    ]
    pool.sort(key=lambda e: (e.get("rating") or 0.0), reverse=True)
    top = pool[: max(limit * 4, 80)]  # rating-ranked band, then shuffled for variety
    random.shuffle(top)
    return top[:limit]


def _recommend_payload(history: list[tuple[int, float]], n: int) -> dict:
    if enrich.is_modern():
        return _modern_payload(history, n)
    recs = rec.recommend_movies(history, n=n)
    matches = enrich.match_scores([s for _, s in recs])
    taste = rec.taste_vector(history) if history else None
    st = rec.load()
    taste_dict = None if taste is None else {
        g: round(float(v), 4) for g, v in zip(st["cfg"]["genre_names"], taste)
    }
    return {
        "recommendations": [
            {**enrich.enrich(mid, s), "match": m} for (mid, s), m in zip(recs, matches)
        ],
        "taste": taste_dict,
        "blurb": taste_blurb(taste_dict, enrich.top_titles(history)),
    }


def _modern_payload(history: list[tuple[int, float]], n: int) -> dict:
    """Modern catalog served by the collaborative NCF model (nearest-favorites)."""
    recs = ncf.rank_for_history(history, n=n)
    matches = enrich.match_from_ratings([r for _, r in recs])
    genres_of = lambda mid: enrich.enrich(mid)["genres"]
    # Radar + blurb describe what the model expects you to enjoy next, so derive
    # them from the recommended films' genres (coherent with the picks shown).
    taste = ncf.taste_from_movies([mid for mid, _ in recs], genres_of) if recs else None
    return {
        "recommendations": [
            {**enrich.enrich(mid, r), "match": m} for (mid, r), m in zip(recs, matches)
        ],
        "taste": taste or None,
        "blurb": taste_blurb(taste or None, enrich.top_titles(history)),
    }


@router.post("/recommend")
def recommend(req: RecRequest) -> dict:
    history = [(h.movieId, h.rating) for h in req.history]
    return _recommend_payload(history, req.n)


def _detect_source(header: str, source: str) -> str:
    if source != "auto":
        return source
    h = header.lower()
    if "name" in h and "rating" in h:
        return "letterboxd"
    if "title" in h and ("start time" in h or "profile name" in h or "duration" in h):
        return "netflix"
    raise HTTPException(
        status_code=400,
        detail="Unrecognized CSV format (expected Letterboxd ratings.csv or Netflix ViewingActivity.csv)",
    )


def _modern_import(detected: str, raw: bytes, ratings_raw: bytes | None) -> list[tuple[int, float]]:
    """Match a Letterboxd / Netflix export to the modern catalog by normalized
    title + year. Letterboxd carries star ratings; Netflix is title-only seen
    history, recorded at a neutral 3.5. Returns (tmdb_id, stars 0.5-5)."""
    import pandas as pd

    from src.importers import _normalise_title

    index: dict[tuple[str, str], int] = {}
    for mid, m in enrich.titles().items():
        nt = _normalise_title(m["title"])
        index[(nt, str(m["year"]))] = mid
        index.setdefault((nt, ""), mid)  # year-less fallback

    def match(title, year) -> int | None:
        if not isinstance(title, str) or not title.strip():
            return None
        nt = _normalise_title(title)
        y = ""
        try:
            y = str(int(float(year)))
        except (TypeError, ValueError):
            y = ""
        return index.get((nt, y)) or index.get((nt, ""))

    df = pd.read_csv(io.BytesIO(raw), on_bad_lines="skip")
    out: dict[int, float] = {}  # dedupe, last rating wins
    if detected == "letterboxd":
        df = df.dropna(subset=["Rating"]) if "Rating" in df.columns else df
        for _, r in df.iterrows():
            mid = match(r.get("Name"), r.get("Year"))
            if mid is not None:
                out[mid] = float(min(max(r["Rating"], 0.5), 5.0))
    else:  # netflix: title-only, seen at a neutral rating
        col = "Title" if "Title" in df.columns else df.columns[0]
        for t in df[col].dropna():
            mid = match(str(t).split(":")[0], None)
            if mid is not None:
                out.setdefault(mid, 3.5)
    return list(out.items())


def _temp_csv(raw: bytes) -> str:
    """Write upload bytes to a SYSTEM temp file (never the repo). Caller unlinks."""
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    try:
        tf.write(raw)
    finally:
        tf.close()
    return tf.name


@router.post("/import")
async def import_watchlist(
    file: UploadFile = File(...),
    ratings_file: UploadFile | None = File(None),
    source: str = Form("auto"),
) -> dict:
    """Turn a Letterboxd / Netflix export into a Reverie watch history.

    Privacy: processed from a system temp file deleted immediately afterwards;
    nothing is written under the repo and contents are never logged."""
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    text = raw.decode("utf-8-sig", errors="replace")
    lines = text.splitlines()
    header = lines[0] if lines else ""
    total = max(0, len(lines) - 1)
    detected = _detect_source(header, source)

    if enrich.is_modern():
        ratings_raw = await ratings_file.read() if ratings_file is not None else None
        pairs = _modern_import(detected, raw, ratings_raw)
        history = [
            {**enrich.enrich(mid), "rating": round(float(stars), 1)} for mid, stars in pairs
        ]
        return {"source": detected, "total": total, "matched": len(history), "history": history}

    movie_to_id = rec.load()["movie_to_id"]
    sink = io.StringIO()  # swallow importer prints (privacy + noise)

    if detected == "letterboxd":
        path = _temp_csv(raw)
        try:
            with contextlib.redirect_stdout(sink):
                pairs = importers.load_letterboxd(path, movie_to_id, MOVIES_DAT)
        finally:
            os.unlink(path)
    else:  # netflix
        view_path = _temp_csv(raw)
        ratings_path = None
        if ratings_file is not None:
            rraw = await ratings_file.read()
            ratings_path = _temp_csv(rraw) if rraw else None
        try:
            with contextlib.redirect_stdout(sink):
                movies_hist, _tv = importers.load_netflix(
                    view_path, movie_to_id,
                    ratings_path=ratings_path, movies_dat_path=MOVIES_DAT,
                )
            pairs = movies_hist
        finally:
            os.unlink(view_path)
            if ratings_path:
                os.unlink(ratings_path)

    history = [
        {**enrich.enrich(mid), "rating": round(float(stars), 1)} for mid, stars in pairs
    ]
    return {
        "source": detected,
        "total": total,
        "matched": len(history),
        "history": history,
    }
