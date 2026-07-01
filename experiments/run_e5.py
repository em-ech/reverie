"""E5 ablation: GRU vs LSTM — which cell type wins?

Trains two identical models differing only in cell type on VALIDATION.
max_len=20 and embed_dim=32/rnn_units=64 are locked from E3/E4.
Decision: pick higher val HR@10; tie (<0.003) goes to GRU (fewer params).
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import keras

from src.gru_model.data_prep import build_dataset, pad_histories, pad_ratings
from src.gru_model.model import build_hybrid_gru
from src.gru_model.evaluate import ranks_from_scores, metrics_from_ranks
from src.gru_model import track

RATINGS = "data/ml-1m/ratings.dat"
SEED = 42
MAX_LEN = 20    # locked E3
EMBED_DIM = 32  # locked E4
RNN_UNITS = 64  # locked E4

CELLS = ["gru", "lstm"]


def main() -> None:
    ds = build_dataset(
        RATINGS, min_count=5, max_len=MAX_LEN,
        max_windows_per_user=30, seed=SEED,
    )
    print(f"items={ds.n_items}  train_windows={len(ds.X_train):,}  "
          f"val_users={len(ds.val_hist)}")

    X_val_ids = pad_histories(ds.val_hist, MAX_LEN)
    X_val_rat = pad_ratings(ds.val_hist_rat, MAX_LEN)
    y_val = np.asarray(ds.val_target, dtype=np.int32)

    results = []

    for cell in CELLS:
        print(f"\n{'='*55}")
        print(f"  cell = {cell.upper()}")
        print(f"{'='*55}")

        keras.utils.set_random_seed(SEED)
        model = build_hybrid_gru(
            ds.n_items, MAX_LEN, ds.genre_matrix,
            embed_dim=EMBED_DIM, rnn_units=RNN_UNITS,
            use_genre=True, use_rating=True, cell=cell,
        )

        t0 = time.time()
        history = model.fit(
            [ds.X_train, ds.X_train_rat], ds.y_train,
            validation_data=([X_val_ids, X_val_rat], y_val),
            epochs=60, batch_size=256, verbose=2,
            callbacks=[keras.callbacks.EarlyStopping(
                "val_loss", patience=10, restore_best_weights=True,
            )],
        )
        elapsed = time.time() - t0
        n_epochs = len(history.history["loss"])

        scores = model.predict(
            [X_val_ids, X_val_rat], batch_size=512, verbose=0,
        )
        m = metrics_from_ranks(
            ranks_from_scores(scores, ds.val_target, ds.seen_val)
        )

        best_ep = int(np.argmin(history.history["val_loss"])) + 1
        sec_per_epoch = elapsed / n_epochs

        results.append((cell, m, best_ep, sec_per_epoch))
        print(f"  {cell.upper()}: HR@10={m['HR@10']:.4f}  MRR={m['MRR']:.4f}"
              f"  best_ep={best_ep}  {sec_per_epoch:.0f}s/epoch")

        track.record(
            f"E5 cell={cell}", m,
            config=f"L{MAX_LEN}/d{EMBED_DIM}/u{RNN_UNITS}/{cell}/p10/e60",
            notes=f"E5 cell sweep, best_ep={best_ep}",
        )

    # Summary table
    print(f"\n{'cell':<6}{'HR@10':>10}{'MRR':>10}{'NDCG@10':>10}"
          f"{'best_ep':>9}{'s/epoch':>9}")
    print("-" * 54)
    best_hr = max(m["HR@10"] for _, m, _, _ in results)
    for cell, m, ep, spe in results:
        marker = " <-- best" if m["HR@10"] == best_hr else ""
        print(f"{cell.upper():<6}{m['HR@10']:>10.4f}{m['MRR']:>10.4f}"
              f"{m['NDCG@10']:>10.4f}{ep:>9}{spe:>9.0f}{marker}")

    gru_hr  = next(m["HR@10"] for c, m, _, _ in results if c == "gru")
    lstm_hr = next(m["HR@10"] for c, m, _, _ in results if c == "lstm")
    diff = abs(gru_hr - lstm_hr)

    if diff < 0.003:
        winner = "gru"
        reason = f"tied (diff={diff:.4f} < 0.003) → GRU wins (fewer params)"
    else:
        winner = "gru" if gru_hr > lstm_hr else "lstm"
        reason = f"higher val HR@10 ({max(gru_hr, lstm_hr):.4f})"

    print(f"\nDecision: cell={winner.upper()} — {reason}")
    print("Update CELL in src/train.py if not already set to gru.")


if __name__ == "__main__":
    main()
