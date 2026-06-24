"""Train the full hybrid GRU and export a portable artifact.

Saves weights + architecture-in-code config (not a full .keras file) for TF
version robustness, plus the encoders and the train-only statistics that
recommend.py needs.   [M6]
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import keras

from src.data_prep import build_dataset, pad_histories, pad_ratings
from src.model import build_hybrid_gru
from src.evaluate import ranks_from_scores, metrics_from_ranks
from src import track

RATINGS = "data/ml-1m/ratings.dat"
ART = "artifacts"
SEED = 42
# E3 winner: L=20 (0.2779) > L=50 (0.2739) > L=10 (0.2707)
MAX_LEN = 20
# E4 winner: d32/u64 smallest within 0.005 of best d64/u128 (0.2782)
EMBED_DIM = 32
RNN_UNITS = 64


def main() -> None:
    os.makedirs(ART, exist_ok=True)
    keras.utils.set_random_seed(SEED)
    ds = build_dataset(RATINGS, min_count=5, max_len=MAX_LEN, max_windows_per_user=30, seed=SEED)
    print(f"items={ds.n_items}  train_windows={len(ds.X_train):,}")

    model = build_hybrid_gru(
        ds.n_items, MAX_LEN, ds.genre_matrix,
        embed_dim=EMBED_DIM, rnn_units=RNN_UNITS, use_genre=True, use_rating=True,
    )
    X_val = [pad_histories(ds.val_hist, MAX_LEN), pad_ratings(ds.val_hist_rat, MAX_LEN)]
    y_val = np.asarray(ds.val_target, dtype=np.int32)
    history = model.fit(
        [ds.X_train, ds.X_train_rat], ds.y_train,
        validation_data=(X_val, y_val), epochs=60, batch_size=256, verbose=2,
        callbacks=[keras.callbacks.EarlyStopping("val_loss", patience=10, restore_best_weights=True)],
    )

    # Track progress over time: val metrics + train/val loss curve.  [feedback_show_experiments]
    val_scores = model.predict(X_val, batch_size=512, verbose=0)
    val_metrics = metrics_from_ranks(ranks_from_scores(val_scores, ds.val_target, ds.seen_val))
    track.save_history(history, "artifact_full_hybrid")
    track.record("artifact train (full hybrid)", val_metrics,
                 config=f"p10/e60, d{EMBED_DIM}, u{RNN_UNITS}", notes="saved artifact")
    print(f"  val: {val_metrics}")

    # Popularity-weighted "average viewer" genre profile, for mean-centering. [H2]
    pop = ds.popularity.astype(np.float64)
    genre_marginal = (pop[:, None] * ds.genre_matrix).sum(0) / max(pop.sum(), 1.0)

    model.save_weights(f"{ART}/weights.weights.h5")
    np.save(f"{ART}/genre_matrix.npy", ds.genre_matrix)
    np.save(f"{ART}/genre_marginal.npy", genre_marginal.astype(np.float32))
    np.save(f"{ART}/popularity_train.npy", ds.popularity)
    with open(f"{ART}/model_config.json", "w") as fh:
        json.dump({
            "n_items": ds.n_items, "max_len": MAX_LEN,
            "embed_dim": EMBED_DIM, "rnn_units": RNN_UNITS,
            "use_genre": True, "use_rating": True,
            "genre_names": ds.genre_names,
        }, fh, indent=2)
    with open(f"{ART}/movie_index.json", "w") as fh:
        # original MovieLens movieId <-> contiguous id
        json.dump({
            "id_to_movie": ds.id_to_movie,
            "movie_to_id": {str(v): k for k, v in ds.id_to_movie.items()},
        }, fh)
    print(f"Artifacts written to {ART}/")


if __name__ == "__main__":
    main()
