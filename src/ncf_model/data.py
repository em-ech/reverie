"""Loaders for the NCF artifacts written by scripts/build_ncf_dataset.py."""

from __future__ import annotations

import json

import numpy as np


def load_split(out_dir: str, name: str):
    """Return (user_idx, movie_idx, rating) as (N,1) int / (N,) float arrays."""
    z = np.load(f"{out_dir}/{name}.npz")
    u = z["user"].astype("int32").reshape(-1, 1)
    m = z["movie"].astype("int32").reshape(-1, 1)
    y = z["rating"].astype("float32")
    return u, m, y


def load_meta(out_dir: str) -> dict:
    with open(f"{out_dir}/meta.json") as fh:
        return json.load(fh)


def load_features(out_dir: str) -> np.ndarray:
    return np.load(f"{out_dir}/movie_features.npy")


def load_baselines(out_dir: str):
    z = np.load(f"{out_dir}/baselines.npz")
    return float(z["global_mean"]), z["movie_mean"], z["user_mean"]
