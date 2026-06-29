"""Reverie API — app factory.

Wraps the frozen recommender (src/recommend.py) in a stateful multi-user layer:
public catalog/recommend/import endpoints plus auth + per-user history. The
model + title/poster metadata load once at startup and are warmed before the
first request.

Run (dev):  uvicorn app.api:app --reload --port 8000
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import enrich
from app.db import init_db
from app.routers import auth, catalog, friends, history
from src import recommend as rec


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()                    # create SQLite tables (idempotent)
    enrich.load_metadata()       # titles + posters
    rec.recommend_movies([])     # warm the model graph before the first request
    yield


app = FastAPI(title="Reverie API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:4173"],
    allow_methods=["*"], allow_headers=["*"],
)

app.include_router(catalog.router)
app.include_router(auth.router)
app.include_router(history.router)
app.include_router(friends.router)
