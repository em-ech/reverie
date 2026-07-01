"""E3 ablation: how much history does the model need?

Trains the full hybrid GRU with three history window sizes on VALIDATION only.
Test set is never touched. Decision: smallest max_len within noise of the best.
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import keras

from src.gru_model.data_prep import build_dataset, pad_histories, pad_ratings
from src.gru_model.model import build_hybrid_gru
from src.gru_model.evaluate import ranks_from_scores, metrics_from_ranks
from src.gru_model import track

RATINGS = "data/ml-1m/ratings.dat"
SEED = 42
EMBED_DIM = 32
RNN_UNITS = 128

HISTORY_LENGTHS = [10, 20, 50]


def main() -> None:
    results = []

    for max_len in HISTORY_LENGTHS:
        print(f"\n{'='*55}")
        print(f"  max_len = {max_len}")
        print(f"{'='*55}")

        # Dataset must be rebuilt per max_len — truncation/padding changes.
        ds = build_dataset(
            RATINGS, min_count=5, max_len=max_len,
            max_windows_per_user=30, seed=SEED,
        )
        print(f"  items={ds.n_items}  train_windows={len(ds.X_train):,}  val_users={len(ds.val_hist)}")

        X_val_ids = pad_histories(ds.val_hist, max_len)
        X_val_rat = pad_ratings(ds.val_hist_rat, max_len)
        y_val = np.asarray(ds.val_target, dtype=np.int32)

        keras.utils.set_random_seed(SEED)
        model = build_hybrid_gru(
            ds.n_items, max_len, ds.genre_matrix,
            embed_dim=EMBED_DIM, rnn_units=RNN_UNITS,
            use_genre=True, use_rating=True,
        )
        model.fit(
            [ds.X_train, ds.X_train_rat], ds.y_train,
            validation_data=([X_val_ids, X_val_rat], y_val),
            epochs=60, batch_size=256, verbose=2,
            callbacks=[keras.callbacks.EarlyStopping(
                "val_loss", patience=10, restore_best_weights=True,
            )],
        )

        scores = model.predict([X_val_ids, X_val_rat], batch_size=512, verbose=0)
        m = metrics_from_ranks(ranks_from_scores(scores, ds.val_target, ds.seen_val))
        results.append((max_len, m))
        print(f"  L={max_len}: HR@10={m['HR@10']:.4f}  MRR={m['MRR']:.4f}  NDCG@10={m['NDCG@10']:.4f}")

        track.record(
            f"E3 max_len={max_len}", m,
            config=f"L{max_len}/d{EMBED_DIM}/u{RNN_UNITS}/p10/e60",
            notes="E3 history length sweep",
        )

    # Summary table
    print(f"\n{'max_len':>10}{'HR@10':>10}{'MRR':>10}{'NDCG@10':>10}")
    print("-" * 40)
    best_hr = max(m["HR@10"] for _, m in results)
    for length, m in results:
        marker = " <-- best" if m["HR@10"] == best_hr else ""
        print(f"{length:>10}{m['HR@10']:>10.4f}{m['MRR']:>10.4f}{m['NDCG@10']:>10.4f}{marker}")

    best_len = min(
        (length for length, m in results if best_hr - m["HR@10"] <= 0.005),
        key=lambda l: l,
    )
    print(f"\nDecision: use max_len={best_len} "
          f"(smallest within 0.005 HR@10 of best {best_hr:.4f})")
    print("Update MAX_LEN in src/train.py if this differs from 50.")


if __name__ == "__main__":
    main()
