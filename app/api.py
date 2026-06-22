"""FastAPI service exposing the Reverie recommender to the React frontend.

Wraps src/recommend.py (stack-agnostic inference). Loads the model + movie
titles once at startup and warms it so the first user request is fast.   [M6]

Run (dev):  uvicorn app.api:app --reload --port 8000
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src import recommend as rec

MOVIES_DAT = "data/ml-1m/movies.dat"
_titles: dict[int, dict] = {}


def _load_titles() -> None:
    """movieId -> {title, year, genres} for recommendable items only."""
    st = rec.load()
    recommendable = set(st["movie_to_id"].keys())
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


@asynccontextmanager
async def lifespan(_: FastAPI):
    _load_titles()
    rec.recommend_movies([])  # warm the model graph before the first request
    yield


app = FastAPI(title="Reverie API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:4173"],
    allow_methods=["*"], allow_headers=["*"],
)


class HistItem(BaseModel):
    movieId: int
    rating: float = 4.0  # stars


class RecRequest(BaseModel):
    history: list[HistItem] = []
    n: int = 12


def _enrich(movie_id: int, score: float) -> dict:
    meta = _titles.get(movie_id, {"movieId": movie_id, "title": str(movie_id), "year": "", "genres": []})
    return {**meta, "score": round(float(score), 4)}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "catalog_size": len(_titles)}


@app.get("/catalog")
def catalog(q: str = "", limit: int = 20) -> list[dict]:
    """Title search for building a watch history (autocomplete)."""
    ql = q.lower().strip()
    items = (m for m in _titles.values() if not ql or ql in m["title"].lower())
    out = sorted(items, key=lambda m: m["title"])[:limit]
    return out


@app.post("/recommend")
def recommend(req: RecRequest) -> dict:
    history = [(h.movieId, h.rating) for h in req.history]
    recs = rec.recommend_movies(history, n=req.n)
    taste = rec.taste_vector(history) if history else None
    st = rec.load()
    return {
        "recommendations": [_enrich(mid, s) for mid, s in recs],
        "taste": None if taste is None else {
            g: round(float(v), 4) for g, v in zip(st["cfg"]["genre_names"], taste)
        },
    }
