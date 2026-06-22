"""E0 GO/NO-GO gate: does a bare GRU beat most-popular on ml-1m?   [H1]

Evaluated on the VALIDATION split only (the test set is opened once, later, on a
frozen config). Reports HR@10 + MRR for both models and a bootstrap CI on the
PAIRED per-user reciprocal-rank difference. Win = CI of the difference excludes 0.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import keras

from src.data_prep import build_dataset, pad_histories
from src.model import build_gru
from src.baselines import most_popular_scores
from src.evaluate import ranks_from_scores, metrics_from_ranks, bootstrap_ci

RATINGS = "data/ml-1m/ratings.dat"
SEED = 42
MAX_LEN = 50


def main() -> None:
    keras.utils.set_random_seed(SEED)

    print("Building dataset (ml-1m, train-only stats, all-prefix windows)...")
    ds = build_dataset(RATINGS, min_count=5, max_len=MAX_LEN, max_windows_per_user=30, seed=SEED)
    print(f"  items V              : {ds.n_items}")
    print(f"  train windows        : {len(ds.X_train):,}")
    print(f"  eval users (val/test): {len(ds.val_hist)} / {len(ds.test_hist)}")
    print(f"  targets dropped (OOV): {ds.n_oov_targets}")

    print("\nTraining GRU (early stop on val loss)...")
    model = build_gru(ds.n_items, MAX_LEN, embed_dim=32, rnn_units=128)
    X_val = pad_histories(ds.val_hist, MAX_LEN)
    y_val = np.asarray(ds.val_target, dtype=np.int32)
    model.fit(
        ds.X_train, ds.y_train,
        validation_data=(X_val, y_val),
        epochs=60, batch_size=256, verbose=2,
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True)],
    )

    print("\nEvaluating on VALIDATION (full-catalog ranking)...")
    rnn_scores = model.predict(X_val, batch_size=512, verbose=0)
    pop_scores = most_popular_scores(ds.popularity, len(ds.val_hist))

    rnn_ranks = ranks_from_scores(rnn_scores, ds.val_target, ds.seen_val)
    pop_ranks = ranks_from_scores(pop_scores, ds.val_target, ds.seen_val)

    rnn_m = metrics_from_ranks(rnn_ranks)
    pop_m = metrics_from_ranks(pop_ranks)
    print(f"  random ref   : HR@10~={10 / ds.n_items:.4f} (sanity floor)")
    print(f"  RNN          : {fmt(rnn_m)}")
    print(f"  most-popular : {fmt(pop_m)}")

    # Paired significance on per-user reciprocal rank.
    diff = (1.0 / rnn_ranks) - (1.0 / pop_ranks)
    lo, hi = bootstrap_ci(diff, n=1000, seed=SEED)
    print(f"\n  paired MRR diff (RNN - popular): {diff.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")

    won = lo > 0
    print("\n" + ("=" * 60))
    if won:
        print("E0 VERDICT: PASS - RNN beats most-popular (CI excludes 0). Proceed.")
    else:
        print("E0 VERDICT: NOT YET - escalate (logit-adj, MIN_COUNT, more windows)")
        print("or invoke the pre-registered fallback narrative. See ARCHITECTURE.md S8.")
    print("=" * 60)
    sys.exit(0 if won else 1)


def fmt(m: dict[str, float]) -> str:
    return "  ".join(f"{k}={v:.4f}" for k, v in m.items())


if __name__ == "__main__":
    main()
