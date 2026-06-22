"""E6 ablation: do genre and rating input features help?   [M3, E6]

Trains four input variants under one protocol and compares on VALIDATION:
  id-only  |  +genre  |  +rating  |  full hybrid (genre + rating)
A feature is justified only if it improves val HR@10 / MRR. Test set untouched.
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import keras

from src.data_prep import build_dataset, pad_histories, pad_ratings
from src.model import build_hybrid_gru
from src.evaluate import ranks_from_scores, metrics_from_ranks

RATINGS = "data/ml-1m/ratings.dat"
SEED = 42
MAX_LEN = 50

CONFIGS = [
    ("id-only", dict(use_genre=False, use_rating=False)),
    ("+genre", dict(use_genre=True, use_rating=False)),
    ("+rating", dict(use_genre=False, use_rating=True)),
    ("full hybrid", dict(use_genre=True, use_rating=True)),
]


def main() -> None:
    ds = build_dataset(RATINGS, min_count=5, max_len=MAX_LEN, max_windows_per_user=30, seed=SEED)
    print(f"items={ds.n_items}  train_windows={len(ds.X_train):,}  val_users={len(ds.val_hist)}")

    X_val_ids = pad_histories(ds.val_hist, MAX_LEN)
    X_val_rat = pad_ratings(ds.val_hist_rat, MAX_LEN)
    y_val = np.asarray(ds.val_target, dtype=np.int32)

    results = []
    for name, flags in CONFIGS:
        print(f"\n=== {name} ===")
        keras.utils.set_random_seed(SEED)  # same init/shuffle across variants
        model = build_hybrid_gru(ds.n_items, MAX_LEN, ds.genre_matrix, **flags)
        model.fit(
            [ds.X_train, ds.X_train_rat], ds.y_train,
            validation_data=([X_val_ids, X_val_rat], y_val),
            epochs=60, batch_size=256, verbose=2,
            callbacks=[keras.callbacks.EarlyStopping("val_loss", patience=10, restore_best_weights=True)],
        )
        scores = model.predict([X_val_ids, X_val_rat], batch_size=512, verbose=0)
        ranks = ranks_from_scores(scores, ds.val_target, ds.seen_val)
        m = metrics_from_ranks(ranks)
        results.append((name, m))
        print(f"  {name}: HR@10={m['HR@10']:.4f}  MRR={m['MRR']:.4f}  NDCG@10={m['NDCG@10']:.4f}")

    print(f"\n{'variant':<14}{'HR@10':>10}{'MRR':>10}{'NDCG@10':>10}")
    print("-" * 44)
    for name, m in results:
        print(f"{name:<14}{m['HR@10']:>10.4f}{m['MRR']:>10.4f}{m['NDCG@10']:>10.4f}")
    print("\nKeep a feature only if it improves on id-only (E0 id-only ~HR@10 0.257).")


if __name__ == "__main__":
    main()
