"""GRU next-item model for NextWatch.

Single softmax head over the movie catalog (the derived taste vector is computed
downstream, not a trained head). No recurrent_dropout so the fast cuDNN/Metal
kernel stays enabled.   [H3, H7]
"""

from __future__ import annotations

import keras
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
