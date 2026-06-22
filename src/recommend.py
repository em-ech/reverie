"""Inference contract: recommend(history) -> top_n.   [serving seam]

Stack-agnostic (callable from FastAPI or Streamlit). Loads the exported artifact
once, rebuilds the model architecture in code, and exposes:
  * recommend_movies  - next-movie ranking from the softmax head
  * taste_vector      - derived, mean-centered, top-K genre profile   [H2]
  * score_tv          - content-cosine bridge to a TV/movie catalog    [H2, M5]
"""

from __future__ import annotations

import functools
import json

import numpy as np

from src.data_prep import pad_histories, pad_ratings
from src.model import build_hybrid_gru

ART = "artifacts"


@functools.lru_cache(maxsize=1)
def load():
    """Load + cache the model and its companion arrays/maps."""
    with open(f"{ART}/model_config.json") as fh:
        cfg = json.load(fh)
    genre_matrix = np.load(f"{ART}/genre_matrix.npy")
    model = build_hybrid_gru(
        cfg["n_items"], cfg["max_len"], genre_matrix,
        embed_dim=cfg["embed_dim"], rnn_units=cfg["rnn_units"],
        use_genre=cfg["use_genre"], use_rating=cfg["use_rating"],
    )
    model.load_weights(f"{ART}/weights.weights.h5")
    with open(f"{ART}/movie_index.json") as fh:
        idx = json.load(fh)
    movie_to_id = {int(k): int(v) for k, v in idx["movie_to_id"].items()}
    id_to_movie = {int(k): int(v) for k, v in idx["id_to_movie"].items()}
    return {
        "cfg": cfg, "model": model, "genre_matrix": genre_matrix,
        "genre_marginal": np.load(f"{ART}/genre_marginal.npy"),
        "popularity": np.load(f"{ART}/popularity_train.npy"),
        "movie_to_id": movie_to_id, "id_to_movie": id_to_movie,
    }


def _encode(history: list[tuple[int, float]], st) -> tuple[list[int], list[float], set[int]]:
    """history: list of (movieId, rating_stars). Map to ids + fixed-scaled rating."""
    ids, rats = [], []
    for movie_id, stars in history:
        cid = st["movie_to_id"].get(int(movie_id))
        if cid is not None:
            ids.append(cid)
            rats.append((float(stars) - 0.5) / 4.5)
    return ids, rats, set(ids)


def _softmax(ids: list[int], rats: list[float], st) -> np.ndarray:
    L = st["cfg"]["max_len"]
    x_ids = pad_histories([ids], L)
    x_rat = pad_ratings([rats], L)
    return st["model"].predict([x_ids, x_rat], verbose=0)[0]  # (n_items+1,)


def recommend_movies(history: list[tuple[int, float]], n: int = 10) -> list[tuple[int, float]]:
    """Top-n next movies (original MovieLens ids), seen items + PAD excluded."""
    st = load()
    if not history:  # cold start -> most popular
        probs = st["popularity"].astype(np.float64)
        seen = set()
    else:
        ids, rats, seen = _encode(history, st)
        probs = _softmax(ids, rats, st)
    probs = probs.copy()
    probs[0] = -np.inf
    for c in seen:
        probs[c] = -np.inf
    top = np.argsort(probs)[::-1][:n]
    return [(st["id_to_movie"][int(c)], float(probs[c])) for c in top]


def taste_vector(history: list[tuple[int, float]], top_k: int = 100) -> np.ndarray:
    """Mean-centered expected next-genre profile from the softmax top-K.   [H2]"""
    st = load()
    if not history:
        return np.zeros(st["genre_matrix"].shape[1], dtype=np.float32)
    ids, rats, _ = _encode(history, st)
    probs = _softmax(ids, rats, st)
    probs[0] = 0.0
    top = np.argsort(probs)[::-1][:top_k]
    p = probs[top] / max(probs[top].sum(), 1e-9)
    raw = (p[:, None] * st["genre_matrix"][top]).sum(0)
    return (raw - st["genre_marginal"]).astype(np.float32)


def score_tv(history, tv_genre_matrix: np.ndarray) -> np.ndarray:
    """Cosine of the taste vector against mean-centered catalog genre vectors.

    tv_genre_matrix must already be in the shared genre vocabulary.   [M5]
    """
    st = load()
    taste = taste_vector(history)
    cand = tv_genre_matrix - st["genre_marginal"]
    tn = np.linalg.norm(taste) + 1e-9
    cn = np.linalg.norm(cand, axis=1) + 1e-9
    return (cand @ taste) / (cn * tn)
