"""Evaluate the trained NCF model: global test RMSE vs naive baselines, and the
personal held-out result for Em (RMSE + the 'will she rate it 4+' classification).

    python -m scripts.eval_ncf --data artifacts/ncf
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from src.ncf_model.baselines import mae, predict_global, predict_movie, predict_user, rmse
from src.ncf_model.data import load_baselines, load_features, load_meta, load_split
from src.ncf_model.model import build_ncf

LOVE = 8  # rating_val >= 8 on the 1-10 scale == 4 stars


def _predict(model, u, m) -> np.ndarray:
    return model.predict([u, m], verbose=0, batch_size=8192).ravel()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="artifacts/ncf")
    args = ap.parse_args()

    meta = load_meta(args.data)
    feats = load_features(args.data)
    gmean, movie_mean, user_mean = load_baselines(args.data)
    with open(f"{args.data}/model_config.json") as fh:
        cfg = json.load(fh)

    model = build_ncf(meta["n_users"], meta["n_movies"], feats,
                      emb_dim=cfg["emb_dim"], global_mean=meta["global_mean"])
    model.load_weights(f"{args.data}/ncf.weights.h5")

    # --- global held-out test ------------------------------------------------
    ut, mt, yt = load_split(args.data, "test")
    pm = _predict(model, ut, mt)
    print("=== GLOBAL test (held-out ratings) ===")
    print(f"  global-mean baseline   RMSE {rmse(yt, predict_global(gmean, len(yt))):.3f}")
    print(f"  movie-mean  baseline   RMSE {rmse(yt, predict_movie(movie_mean, mt)):.3f}")
    print(f"  user-mean   baseline   RMSE {rmse(yt, predict_user(user_mean, ut)):.3f}")
    print(f"  >> neural net          RMSE {rmse(yt, pm):.3f}  MAE {mae(yt, pm):.3f}")

    # --- each real person's personal held-out (temporal) ---------------------
    for name, uidx in meta.get("external_users", {}).items():
        ue, me, ye = load_split(args.data, f"{name}_test")
        pe = _predict(model, ue, me)
        umean = user_mean[uidx]
        print(f"\n=== {name.upper()} personal ({len(ye)} most-recent films, never trained on) ===")
        print(f"  predict-their-average  RMSE {rmse(ye, np.full(len(ye), umean)):.3f}")
        print(f"  movie-mean  baseline   RMSE {rmse(ye, predict_movie(movie_mean, me)):.3f}")
        print(f"  >> neural net          RMSE {rmse(ye, pe):.3f}  MAE {mae(ye, pe):.3f}")
        yb = (ye >= LOVE).astype(int)
        pb = (pe >= LOVE).astype(int)
        print(f"  'will they love it (>=4 stars)':")
        print(f"    confusion [rows=actual, cols=pred]: {confusion_matrix(yb, pb).tolist()}")
        print(classification_report(yb, pb, zero_division=0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
