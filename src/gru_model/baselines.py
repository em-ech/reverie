"""Non-neural baselines the RNN must beat. Train-only popularity.   [H5, C2]

E0 uses most-popular; most-recent-genre and item-kNN are added for the full
headline comparison.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


def most_popular_scores(popularity: np.ndarray, n_users: int) -> np.ndarray:
    """Same train-only popularity vector for every user.

    Seen-item masking is applied uniformly by the evaluator, so a popular item
    the user already watched cannot unfairly fill their top-K.   [M2]
    """
    return np.tile(popularity.astype(np.float64), (n_users, 1))


def recent_genre_scores(
    genre_matrix: np.ndarray,      # (n_items+1, n_genres)
    popularity: np.ndarray,        # (n_items+1,) train-only
    histories: list[list[int]],    # per eval user, train history ids
    last_k: int = 5,
) -> np.ndarray:
    """Popularity restricted to genres the user watched in their last-K items.

    A genuinely strong content+popularity hybrid (not a strawman): an item only
    scores if it shares >=1 genre with the user's recent taste, then ranks by
    train-only popularity.   [H5, M1]
    """
    n_users, n_items1 = len(histories), genre_matrix.shape[0]
    recent = np.zeros((n_users, genre_matrix.shape[1]), dtype=np.float32)
    for i, h in enumerate(histories):
        if h:
            recent[i] = genre_matrix[h[-last_k:]].max(axis=0)
    overlap = recent @ genre_matrix.T            # (n_users, n_items+1)
    return (overlap > 0).astype(np.float64) * popularity[None, :]


def item_knn_scores(
    train_user_items: list[list[int]],
    n_items: int,
    histories: list[list[int]],
) -> np.ndarray:
    """Item-based kNN: cosine item-item similarity from the train user-item
    matrix, scored against each eval user's history.   [H5]

    The baseline shallow recommenders famously beat by personalization; if the
    RNN does not beat this, that is the project's headline finding.
    """
    n_train = len(train_user_items)
    rows, cols = [], []
    for u, items in enumerate(train_user_items):
        for it in items:
            rows.append(u)
            cols.append(it)
    mat = sparse.csr_matrix(
        (np.ones(len(rows), np.float32), (rows, cols)),
        shape=(n_train, n_items + 1),
    )
    sim = cosine_similarity(mat.T, dense_output=True)   # (n_items+1, n_items+1)
    np.fill_diagonal(sim, 0.0)

    n_users = len(histories)
    hrows, hcols = [], []
    for i, h in enumerate(histories):
        for it in h:
            hrows.append(i)
            hcols.append(it)
    hist_mat = sparse.csr_matrix(
        (np.ones(len(hrows), np.float32), (hrows, hcols)),
        shape=(n_users, n_items + 1),
    )
    return hist_mat @ sim    # (n_users, n_items+1)
