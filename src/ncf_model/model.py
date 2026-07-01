"""The neural collaborative-filtering model.

A user embedding and a movie embedding, plus per-user / per-movie bias terms and a
frozen movie content tower (genres + standardized numeric features), concatenated
into a small MLP that predicts a 1-10 star rating. The biases alone recover the
user/movie-mean baselines; the embeddings learn the personalization residual; the
content tower carries cold-start movies the embedding barely saw.

Framed for the course as "a neural network with learned embedding layers predicting
a rating" (embeddings + Dense + regression).
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers


def build_ncf(
    n_users: int,
    n_movies: int,
    movie_features: np.ndarray,
    emb_dim: int = 50,
    l2: float = 1e-6,
    dropout: float = 0.3,
    global_mean: float = 6.5,
) -> tf.keras.Model:
    reg = regularizers.l2(l2)
    n_feat = movie_features.shape[1]

    u_in = layers.Input(shape=(1,), dtype="int32", name="user")
    m_in = layers.Input(shape=(1,), dtype="int32", name="movie")

    u_emb = layers.Flatten()(layers.Embedding(n_users, emb_dim, embeddings_regularizer=reg, name="user_emb")(u_in))
    m_emb = layers.Flatten()(layers.Embedding(n_movies, emb_dim, embeddings_regularizer=reg, name="movie_emb")(m_in))
    content = layers.Flatten()(layers.Embedding(
        n_movies, n_feat,
        embeddings_initializer=tf.keras.initializers.Constant(movie_features),
        trainable=False, name="content",
    )(m_in))

    u_bias = layers.Flatten()(layers.Embedding(n_users, 1, name="user_bias")(u_in))
    m_bias = layers.Flatten()(layers.Embedding(n_movies, 1, name="movie_bias")(m_in))

    x = layers.Concatenate()([u_emb, m_emb, content])
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    mlp = layers.Dense(1, name="mlp_out")(x)

    out = layers.Add()([mlp, u_bias, m_bias])
    out = layers.Lambda(lambda t: t + global_mean, name="add_global")(out)

    return tf.keras.Model([u_in, m_in], out, name="ncf")
