"""Headline baseline comparison on VALIDATION (test set stays sealed until the
frozen, tuned config).   [H5, H6]

Compares the GRU against most-popular, recent-genre popularity, and item-kNN.
Reports HR@10 + MRR with bootstrap 95% CIs, plus a paired Wilcoxon signed-rank
test on per-user reciprocal rank (RNN vs each baseline).
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import keras
from scipy.stats import wilcoxon

from src.data_prep import build_dataset, pad_histories
from src.model import build_gru
from src.baselines import most_popular_scores, recent_genre_scores, item_knn_scores
from src.evaluate import ranks_from_scores, metrics_from_ranks, bootstrap_ci

RATINGS = "data/ml-1m/ratings.dat"
SEED = 42
MAX_LEN = 50


def main() -> None:
    keras.utils.set_random_seed(SEED)
    ds = build_dataset(RATINGS, min_count=5, max_len=MAX_LEN, max_windows_per_user=30, seed=SEED)
    print(f"items={ds.n_items}  train_windows={len(ds.X_train):,}  val_users={len(ds.val_hist)}")

    print("Training GRU (seed 42, early stop on val loss)...")
    model = build_gru(ds.n_items, MAX_LEN, embed_dim=32, rnn_units=128)
    X_val = pad_histories(ds.val_hist, MAX_LEN)
    y_val = np.asarray(ds.val_target, dtype=np.int32)
    model.fit(
        ds.X_train, ds.y_train, validation_data=(X_val, y_val),
        epochs=15, batch_size=256, verbose=2,
        callbacks=[keras.callbacks.EarlyStopping("val_loss", patience=2, restore_best_weights=True)],
    )

    seen, tgt = ds.seen_val, ds.val_target
    scores = {
        "RNN (GRU)":        model.predict(X_val, batch_size=512, verbose=0),
        "most-popular":     most_popular_scores(ds.popularity, len(ds.val_hist)),
        "recent-genre pop": recent_genre_scores(ds.genre_matrix, ds.popularity, ds.val_hist),
        "item-kNN":         item_knn_scores(ds.train_user_items, ds.n_items, ds.val_hist),
    }
    ranks = {name: ranks_from_scores(s, tgt, seen) for name, s in scores.items()}
    rnn_rr = 1.0 / ranks["RNN (GRU)"]

    print(f"\n{'model':<18}{'HR@10':>16}{'MRR':>16}{'vs RNN (Wilcoxon p)':>24}")
    print("-" * 74)
    for name, r in ranks.items():
        m = metrics_from_ranks(r)
        hr_lo, hr_hi = bootstrap_ci((r <= 10).astype(float), seed=SEED)
        mrr_lo, mrr_hi = bootstrap_ci(1.0 / r, seed=SEED)
        if name == "RNN (GRU)":
            p = "-"
        else:
            stat, pval = wilcoxon(rnn_rr, 1.0 / r)
            p = f"{pval:.2e}"
        print(f"{name:<18}{m['HR@10']:.4f} [{hr_lo:.3f},{hr_hi:.3f}]"
              f"{m['MRR']:.4f} [{mrr_lo:.3f},{mrr_hi:.3f}]{p:>14}")

    print("\nRNN wins iff it beats ALL three on HR@10/MRR with non-overlapping CIs and p<0.05.")


if __name__ == "__main__":
    main()
