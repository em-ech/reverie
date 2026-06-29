"""Train the NCF model on the artifacts from build_ncf_dataset.py.

    python -m scripts.train_ncf --data artifacts/ncf --epochs 20 --batch 2048
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf

from src.ncf_data import load_features, load_meta, load_split
from src.ncf_model import build_ncf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="artifacts/ncf")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--emb", type=int, default=50)
    args = ap.parse_args()
    tf.keras.utils.set_random_seed(42)

    meta = load_meta(args.data)
    feats = load_features(args.data)
    utr, mtr, ytr = load_split(args.data, "train")
    uval, mval, yval = load_split(args.data, "val")
    tw = np.load(f"{args.data}/train.npz").get("weight")
    print(f"train {len(ytr):,} | val {len(yval):,} | {meta['n_users']:,} users x {meta['n_movies']:,} movies"
          + (f" | sample-weighting on (max x{tw.max():.0f})" if tw is not None else ""))

    model = build_ncf(meta["n_users"], meta["n_movies"], feats,
                      emb_dim=args.emb, global_mean=meta["global_mean"])
    model.compile(optimizer="adam", loss="mse",
                  metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")])

    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True)
    model.fit([utr, mtr], ytr, sample_weight=tw, validation_data=([uval, mval], yval),
              epochs=args.epochs, batch_size=args.batch, callbacks=[early], verbose=2)

    os.makedirs(args.data, exist_ok=True)
    model.save_weights(f"{args.data}/ncf.weights.h5")
    with open(f"{args.data}/model_config.json", "w") as fh:
        json.dump({"emb_dim": args.emb, "global_mean": meta["global_mean"],
                   "n_users": meta["n_users"], "n_movies": meta["n_movies"]}, fh)
    print(f"saved weights -> {args.data}/ncf.weights.h5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
