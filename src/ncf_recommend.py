"""Serving seam for the collaborative NCF model: rank the modern catalog for a
viewer's watch history.

The model's strength is its trained **movie embedding space** (films co-loved on
Letterboxd sit near each other). We rank by similarity to the viewer's FAVORITE
films, scored as the mean cosine to their nearest few favorites, not the mean of
all watched films (a broad history averages into a mushy centroid that recommends
generic titles). A mild recency nudge keeps the picks modern. Films map
tmdb_id <-> Letterboxd slug <-> the model's integer index.

    from src import ncf_recommend as ncf
    ncf.rank_for_history([(550, 4.5), (27205, 5.0)], n=12, exclude=[550])
"""

from __future__ import annotations

import functools
import json

import numpy as np

from src.ncf_data import load_features, load_meta
from src.ncf_model import build_ncf

NCF_DIR = "artifacts/ncf"
MODERN_DIR = "artifacts/modern"
EMB_DIM = 50
NEAREST = 5          # score a candidate by its mean cosine to this many nearest favorites
RECENCY_W = 0.15     # weight of the modern nudge (0 = ignore year)


def _stars_to_scale(stars: float) -> float:
    """Letterboxd stars (0.5-5) -> the model's 1-10 rating scale."""
    return float(min(max(stars * 2.0, 1.0), 10.0))


@functools.lru_cache(maxsize=1)
def load() -> dict:
    """Load the trained movie embeddings + catalog maps. Cached: the heavy work
    (building the model to read its embedding weights) happens once per process."""
    meta = load_meta(NCF_DIR)
    feats = load_features(NCF_DIR)
    with open(f"{NCF_DIR}/model_config.json") as fh:
        cfg = json.load(fh)

    model = build_ncf(meta["n_users"], meta["n_movies"], feats,
                      emb_dim=cfg["emb_dim"], global_mean=float(meta["global_mean"]))
    model.load_weights(f"{NCF_DIR}/ncf.weights.h5")
    memb = model.get_layer("movie_emb").get_weights()[0]
    memb_norm = memb / (np.linalg.norm(memb, axis=1, keepdims=True) + 1e-9)

    with open(f"{NCF_DIR}/movie_index.json") as fh:
        movie_index = json.load(fh)  # slug -> idx
    with open(f"{MODERN_DIR}/slug_to_tmdb.json") as fh:
        slug_to_tmdb = json.load(fh)
    with open(f"{MODERN_DIR}/catalog.json") as fh:
        catalog = json.load(fh)  # already popularity-ordered

    tmdb_to_idx: dict[int, int] = {}
    idx_to_tmdb: dict[int, int] = {}
    year_by_idx = np.zeros(meta["n_movies"], dtype=np.float32)
    for slug, tmdb in slug_to_tmdb.items():
        idx = movie_index.get(slug)
        if idx:
            tmdb_to_idx[int(tmdb)] = idx
            idx_to_tmdb[idx] = int(tmdb)
            yr = catalog[str(int(tmdb))].get("year")
            year_by_idx[idx] = int(yr) if yr else 0

    cand_idx = np.array(sorted(idx_to_tmdb), dtype=np.int32)
    return {
        "memb_norm": memb_norm, "global_mean": float(meta["global_mean"]),
        "tmdb_to_idx": tmdb_to_idx, "idx_to_tmdb": idx_to_tmdb,
        "cand_idx": cand_idx, "year_by_idx": year_by_idx,
        "popular_tmdb": [int(t) for t in catalog], "genre_names": meta["genre_names"],
    }


def genre_names() -> list[str]:
    """The model's genre vocabulary (for taste vectors / the Blend radar)."""
    return load()["genre_names"]


def _favorite_idxs(rated: list[tuple[int, float]]) -> list[int]:
    """The films that define a viewer's taste: those rated near their top. Falls
    back to the whole set when ratings carry no signal (e.g. Netflix imports that
    are all the same neutral value)."""
    if not rated:
        return []
    hi = max(stars for _, stars in rated)
    thresh = max(4.0, hi - 0.5)  # within half a star of their best
    fav = [idx for idx, stars in rated if stars >= thresh]
    if len(fav) < 8:  # too thin a signal -> use everything they watched
        fav = [idx for idx, _ in rated]
    return fav


def rank_for_history(
    history: list[tuple[int, float]],
    n: int = 12,
    exclude: list[int] | None = None,
    recency: float = RECENCY_W,
) -> list[tuple[int, float]]:
    """history: list of (tmdb_id, stars 0.5-5). Returns the top-n (tmdb_id,
    score), seen + excluded films removed. Score = mean cosine to the viewer's
    nearest favorites, plus a recency nudge."""
    st = load()
    memb_norm = st["memb_norm"]
    tmdb_to_idx = st["tmdb_to_idx"]
    exclude_set = {int(t) for t in (exclude or [])}

    rated: list[tuple[int, float]] = []
    for tmdb, stars in history:
        idx = tmdb_to_idx.get(int(tmdb))
        if idx is not None:
            rated.append((idx, float(stars)))
        exclude_set.add(int(tmdb))

    # Cold start (no scorable history): popularity order, minus excluded.
    if not rated:
        out = [t for t in st["popular_tmdb"] if t not in exclude_set][:n]
        return [(t, st["global_mean"]) for t in out]

    fav = np.array(_favorite_idxs(rated), dtype=np.int32)
    seen_idx = {idx for idx, _ in rated} | {
        tmdb_to_idx[t] for t in exclude_set if t in tmdb_to_idx
    }
    cand = np.array([i for i in st["cand_idx"] if i not in seen_idx], dtype=np.int32)
    if cand.size == 0:
        return []

    # Mean cosine to the k nearest favorites (focused, not a mushy global mean).
    sims = memb_norm[cand] @ memb_norm[fav].T  # (n_cand, n_fav)
    k = min(NEAREST, fav.shape[0])
    base = np.sort(sims, axis=1)[:, -k:].mean(axis=1)

    years = st["year_by_idx"][cand]
    recency01 = np.clip((years - 2000.0) / 25.0, 0.0, 1.0)  # 2000 -> 0, 2025 -> 1
    score = base + recency * recency01

    top = np.argsort(-score)[:n]
    idx_to_tmdb = st["idx_to_tmdb"]
    return [(idx_to_tmdb[int(cand[i])], float(score[i])) for i in top]


def taste_genres(history, genres_of) -> dict[str, float]:
    """A rating-weighted genre profile over the seen films, for the Blend radar
    (comparing two viewers). genres_of: callable mapping a tmdb_id to its genres."""
    acc: dict[str, float] = {}
    total = 0.0
    for tmdb, stars in history:
        w = _stars_to_scale(stars)
        total += w
        for g in genres_of(int(tmdb)):
            acc[g] = acc.get(g, 0.0) + w
    if total <= 0:
        return {}
    return {g: round(v / total, 4) for g, v in acc.items()}


def taste_from_movies(movie_ids, genres_of) -> dict[str, float]:
    """Genre distribution of a set of films (e.g. the recommendations), for the
    'what we expect you to enjoy next' radar. Honest to that label and coherent
    with the picks actually shown."""
    acc: dict[str, float] = {}
    for mid in movie_ids:
        for g in genres_of(int(mid)):
            acc[g] = acc.get(g, 0.0) + 1.0
    total = sum(acc.values())
    if total <= 0:
        return {}
    return {g: round(v / total, 4) for g, v in acc.items()}
