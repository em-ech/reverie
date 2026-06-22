"""Non-neural baselines the RNN must beat. Train-only popularity.   [H5, C2]

E0 uses most-popular; most-recent-genre and item-kNN are added for the full
headline comparison.
"""

from __future__ import annotations

import numpy as np


def most_popular_scores(popularity: np.ndarray, n_users: int) -> np.ndarray:
    """Same train-only popularity vector for every user.

    Seen-item masking is applied uniformly by the evaluator, so a popular item
    the user already watched cannot unfairly fill their top-K.   [M2]
    """
    return np.tile(popularity.astype(np.float64), (n_users, 1))
