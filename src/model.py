"""GRU next-item model for NextWatch.

Single softmax head over the movie catalog (the derived taste vector is computed
downstream, not a trained head). No recurrent_dropout so the fast cuDNN/Metal
kernel stays enabled.   [H3, H7]
"""

from __future__ import annotations

import keras
import numpy as np
from keras import layers


def build_gru(
    n_items: int,
    max_len: int,
    embed_dim: int = 32,
    rnn_units: int = 128,
    dropout: float = 0.2,
    head_dropout: float = 0.3,
    l2: float = 1e-6,
    cell: str = "gru",
) -> keras.Model:
    """Embedding(mask_zero) -> GRU/LSTM -> Dropout -> softmax over (n_items+1).

    Output dim is n_items+1 because index 0 is the reserved PAD class; it is
    excluded from candidates at evaluation/serving time.   [C3]
    """
    reg = keras.regularizers.l2(l2)
    inp = keras.Input(shape=(max_len,), dtype="int32", name="history")
    x = layers.Embedding(
        input_dim=n_items + 1,
        output_dim=embed_dim,
        mask_zero=True,
        embeddings_regularizer=reg,
        name="movie_embedding",
    )(inp)
    rnn = layers.GRU if cell == "gru" else layers.LSTM
    x = rnn(rnn_units, dropout=dropout, name=f"{cell}_encoder")(x)  # no recurrent_dropout
    x = layers.Dropout(head_dropout)(x)
    out = layers.Dense(n_items + 1, activation="softmax", name="next_movie")(x)

    model = keras.Model(inp, out, name="nextwatch_gru")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
    )
    return model


def build_hybrid_gru(
    n_items: int,
    max_len: int,
    genre_matrix: np.ndarray,
    embed_dim: int = 32,
    rnn_units: int = 128,
    dropout: float = 0.2,
    head_dropout: float = 0.3,
    l2: float = 1e-6,
    cell: str = "gru",
    use_genre: bool = True,
    use_rating: bool = True,
) -> keras.Model:
    """Hybrid input: movie embedding (+ frozen genre lookup) (+ rating channel).

    The mask is computed explicitly from the id sequence and passed to the RNN,
    so padded steps are ignored regardless of how the feature channels are
    concatenated (avoids the Concatenate-drops-the-mask pitfall).   [C3]
    use_genre / use_rating toggle channels for the E6 ablation.
    """
    reg = keras.regularizers.l2(l2)
    ids_in = keras.Input(shape=(max_len,), dtype="int32", name="history")
    rat_in = keras.Input(shape=(max_len,), dtype="float32", name="ratings")

    movie_emb = layers.Embedding(
        input_dim=n_items + 1, output_dim=embed_dim,
        embeddings_regularizer=reg, name="movie_embedding",
    )(ids_in)
    feats = [movie_emb]

    if use_genre:
        # Frozen lookup of the multi-hot genre vector per step. PAD row (0) is
        # zeros, so padded steps contribute nothing even before masking.
        genre_lookup = layers.Embedding(
            input_dim=n_items + 1, output_dim=genre_matrix.shape[1],
            embeddings_initializer=keras.initializers.Constant(genre_matrix),
            trainable=False, name="genre_lookup",
        )(ids_in)
        feats.append(genre_lookup)

    if use_rating:
        feats.append(layers.Reshape((max_len, 1))(rat_in))

    x = layers.Concatenate(name="features")(feats) if len(feats) > 1 else feats[0]

    # Explicit temporal mask from the id sequence (1 where real, 0 at PAD). [C3]
    mask = layers.Lambda(
        lambda t: keras.ops.not_equal(t, 0), output_shape=(max_len,), name="pad_mask",
    )(ids_in)
    rnn = layers.GRU if cell == "gru" else layers.LSTM
    x = rnn(rnn_units, dropout=dropout, name=f"{cell}_encoder")(x, mask=mask)
    x = layers.Dropout(head_dropout)(x)
    out = layers.Dense(n_items + 1, activation="softmax", name="next_movie")(x)

    model = keras.Model([ids_in, rat_in], out, name="nextwatch_hybrid")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy")
    return model
