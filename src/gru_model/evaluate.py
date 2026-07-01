"""Leave-one-out ranking evaluation with full-catalog ranking.

Metrics under one held-out positive per user: HR@K (== Recall@K), MRR, NDCG@K.
PAD (index 0) and the user's seen items are excluded from candidates for EVERY
model, identically.   [M2, C3]   Bootstrap CIs over users for significance. [H6]
"""

from __future__ import annotations

import numpy as np

NEG_INF = -1e9


def ranks_from_scores(
    scores: np.ndarray,          # (n_users, n_items+1) higher = better
    targets: list[int],          # held-out true id per user
    seen: list[set[int]],        # items to mask per user
) -> np.ndarray:
    """Return the 1-based rank of each user's true item over the full catalog."""
    s = scores.copy()
    s[:, 0] = NEG_INF  # never rank PAD
    for i, seen_set in enumerate(seen):
        if seen_set:
            s[i, list(seen_set)] = NEG_INF
    tgt = np.asarray(targets)
    target_scores = s[np.arange(len(tgt)), tgt]
    # rank = 1 + (number of items strictly better than the target)
    return 1 + (s > target_scores[:, None]).sum(axis=1)


def metrics_from_ranks(ranks: np.ndarray, k: int = 10) -> dict[str, float]:
    hit = ranks <= k
    return {
        f"HR@{k}": float(hit.mean()),
        "MRR": float((1.0 / ranks).mean()),
        f"NDCG@{k}": float((hit / np.log2(ranks + 1)).mean()),
    }


def bootstrap_ci(
    per_user: np.ndarray, n: int = 1000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """95% CI of the mean of a per-user metric, by resampling users."""
    rng = np.random.default_rng(seed)
    means = [rng.choice(per_user, size=len(per_user), replace=True).mean() for _ in range(n)]
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
