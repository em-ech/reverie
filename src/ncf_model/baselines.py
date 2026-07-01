"""Naive rating-prediction baselines (train-only stats) and RMSE."""

from __future__ import annotations

import numpy as np


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def predict_global(global_mean: float, n: int) -> np.ndarray:
    return np.full(n, global_mean, dtype="float32")


def predict_movie(movie_mean: np.ndarray, movie_idx: np.ndarray) -> np.ndarray:
    return movie_mean[np.asarray(movie_idx).ravel()]


def predict_user(user_mean: np.ndarray, user_idx: np.ndarray) -> np.ndarray:
    return user_mean[np.asarray(user_idx).ravel()]
