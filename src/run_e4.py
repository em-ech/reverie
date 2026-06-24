"""E4 ablation: how big does the model need to be?

Trains the full hybrid GRU across a grid of embed_dim x rnn_units on VALIDATION.
max_len=20 is locked from E3. Decision: smallest model within noise of the best.
Also tracks the train-val loss gap to catch overfitting as size grows.
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import keras

from src.data_prep import build_dataset, pad_histories, pad_ratings
from src.model import build_hybrid_gru
from src.evaluate import ranks_from_scores, metrics_from_ranks
from src import track

RATINGS = "data/ml-1m/ratings.dat"
SEED = 42
MAX_LEN = 20  # locked from E3

# Grid: embed_dim x rnn_units  (6 variants, ~30-40 min total on CPU)
CONFIGS = [
    (16,  64),
    (16, 128),
    (32,  64),
    (32, 128),  # current default
    (64, 128),
    (64, 256),
]


def main() -> None:
    # Build dataset once — max_len is fixed, so it doesn't need to change.
    ds = build_dataset(
        RATINGS, min_count=5, max_len=MAX_LEN,
        max_windows_per_user=30, seed=SEED,
    )
    print(f"items={ds.n_items}  train_windows={len(ds.X_train):,}  val_users={len(ds.val_hist)}")

    X_val_ids = pad_histories(ds.val_hist, MAX_LEN)
    X_val_rat = pad_ratings(ds.val_hist_rat, MAX_LEN)
    y_val = np.asarray(ds.val_target, dtype=np.int32)

    results = []

    for embed_dim, rnn_units in CONFIGS:
        label = f"d{embed_dim}/u{rnn_units}"
        n_params = (ds.n_items + 1) * embed_dim + 3 * rnn_units * (embed_dim + rnn_units + 1)
        print(f"\n{'='*55}")
        print(f"  embed_dim={embed_dim}  rnn_units={rnn_units}  (~{n_params/1e3:.0f}k params)")
        print(f"{'='*55}")

        keras.utils.set_random_seed(SEED)
        model = build_hybrid_gru(
            ds.n_items, MAX_LEN, ds.genre_matrix,
            embed_dim=embed_dim, rnn_units=rnn_units,
            use_genre=True, use_rating=True,
        )

        history = model.fit(
            [ds.X_train, ds.X_train_rat], ds.y_train,
            validation_data=([X_val_ids, X_val_rat], y_val),
            epochs=60, batch_size=256, verbose=2,
            callbacks=[keras.callbacks.EarlyStopping(
                "val_loss", patience=10, restore_best_weights=True,
            )],
        )

        scores = model.predict([X_val_ids, X_val_rat], batch_size=512, verbose=0)
        m = metrics_from_ranks(ranks_from_scores(scores, ds.val_target, ds.seen_val))

        # Train-val gap at the best epoch (overfitting evidence).
        best_epoch = int(np.argmin(history.history["val_loss"]))
        train_loss = history.history["loss"][best_epoch]
        val_loss   = history.history["val_loss"][best_epoch]
        gap = val_loss - train_loss

        results.append((embed_dim, rnn_units, m, gap, best_epoch + 1))
        print(f"  {label}: HR@10={m['HR@10']:.4f}  MRR={m['MRR']:.4f}  "
              f"gap={gap:.4f}  best_ep={best_epoch+1}")

        track.record(
            f"E4 {label}", m,
            config=f"L{MAX_LEN}/{label}/p10/e60",
            notes=f"E4 capacity sweep, gap={gap:.4f}",
        )

    # Summary table
    print(f"\n{'embed':>6}{'units':>6}{'HR@10':>10}{'MRR':>10}{'NDCG@10':>10}"
          f"{'val-train gap':>15}{'best_ep':>9}")
    print("-" * 66)
    best_hr = max(m["HR@10"] for _, _, m, _, _ in results)
    for embed_dim, rnn_units, m, gap, ep in results:
        marker = " <-- best" if m["HR@10"] == best_hr else ""
        print(f"{embed_dim:>6}{rnn_units:>6}{m['HR@10']:>10.4f}{m['MRR']:>10.4f}"
              f"{m['NDCG@10']:>10.4f}{gap:>15.4f}{ep:>9}{marker}")

    # Pick smallest model within 0.005 HR@10 of best, sorted by total params (smallest first).
    candidates = [
        (embed_dim, rnn_units, m)
        for embed_dim, rnn_units, m, _, _ in results
        if best_hr - m["HR@10"] <= 0.005
    ]
    best_embed, best_units, _ = min(candidates, key=lambda x: x[0] * x[1])
    print(f"\nDecision: embed_dim={best_embed}, rnn_units={best_units} "
          f"(smallest within 0.005 HR@10 of best {best_hr:.4f})")
    print("Update EMBED_DIM and RNN_UNITS in src/train.py.")


if __name__ == "__main__":
    main()
