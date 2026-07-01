"""Final test-set evaluation — run ONCE on the frozen config (Lea's task).

Locked config from Stephan's E3/E4/E5:
    MAX_LEN=20, EMBED_DIM=32, RNN_UNITS=64, CELL="gru"

Trains 3 independent seeds and reports mean ± std HR@10, MRR, NDCG@10
on the TEST set (never touched during E3/E4/E5 tuning).

This is the number that goes on the results slide.
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import keras

from src.gru_model.data_prep import build_dataset, pad_histories, pad_ratings
from src.gru_model.model import build_hybrid_gru
from src.gru_model.evaluate import ranks_from_scores, metrics_from_ranks, bootstrap_ci
from src.gru_model import track

RATINGS = "data/ml-1m/ratings.dat"

# Locked config — DO NOT change these (decided by Stephan in E3/E4/E5)
MAX_LEN   = 20
EMBED_DIM = 32
RNN_UNITS = 64
CELL      = "gru"

SEEDS = [42, 0, 7]


def main() -> None:
    # Dataset split is deterministic (leave-one-out by timestamp) — build once.
    ds = build_dataset(
        RATINGS, min_count=5, max_len=MAX_LEN,
        max_windows_per_user=30, seed=42,
    )
    print(f"items={ds.n_items}  train_windows={len(ds.X_train):,}  "
          f"test_users={len(ds.test_hist)}")

    X_test_ids = pad_histories(ds.test_hist, MAX_LEN)
    X_test_rat = pad_ratings(ds.test_hist_rat, MAX_LEN)
    y_test     = np.asarray(ds.test_target, dtype=np.int32)

    all_metrics: list[dict] = []

    for seed in SEEDS:
        print(f"\n{'='*55}")
        print(f"  seed = {seed}")
        print(f"{'='*55}")

        keras.utils.set_random_seed(seed)
        model = build_hybrid_gru(
            ds.n_items, MAX_LEN, ds.genre_matrix,
            embed_dim=EMBED_DIM, rnn_units=RNN_UNITS,
            use_genre=True, use_rating=True, cell=CELL,
        )

        X_val_ids = pad_histories(ds.val_hist, MAX_LEN)
        X_val_rat = pad_ratings(ds.val_hist_rat, MAX_LEN)
        y_val     = np.asarray(ds.val_target, dtype=np.int32)

        model.fit(
            [ds.X_train, ds.X_train_rat], ds.y_train,
            validation_data=([X_val_ids, X_val_rat], y_val),
            epochs=60, batch_size=256, verbose=2,
            callbacks=[keras.callbacks.EarlyStopping(
                "val_loss", patience=10, restore_best_weights=True,
            )],
        )

        scores = model.predict([X_test_ids, X_test_rat], batch_size=512, verbose=0)
        ranks  = ranks_from_scores(scores, ds.test_target, ds.seen_test)
        m      = metrics_from_ranks(ranks)
        all_metrics.append(m)

        print(f"  seed={seed}  HR@10={m['HR@10']:.4f}  MRR={m['MRR']:.4f}  NDCG@10={m['NDCG@10']:.4f}")
        track.record(
            f"TEST seed={seed}", m,
            config=f"L{MAX_LEN}/d{EMBED_DIM}/u{RNN_UNITS}/{CELL}/p10/e60",
            notes="OFFICIAL TEST RUN — frozen config",
        )

    # Aggregate across seeds
    hr   = np.array([m["HR@10"]   for m in all_metrics])
    mrr  = np.array([m["MRR"]     for m in all_metrics])
    ndcg = np.array([m["NDCG@10"] for m in all_metrics])

    print(f"\n{'='*55}")
    print("  FINAL TEST RESULTS (frozen config, test set)")
    print(f"{'='*55}")
    print(f"  HR@10  : {hr.mean():.4f} ± {hr.std():.4f}  (seeds {SEEDS})")
    print(f"  MRR    : {mrr.mean():.4f} ± {mrr.std():.4f}")
    print(f"  NDCG@10: {ndcg.mean():.4f} ± {ndcg.std():.4f}")
    print(f"\n  → Slide number: HR@10 = {hr.mean():.3f} ± {hr.std():.3f} (mean ± std, 3 seeds, test set)")


if __name__ == "__main__":
    main()
