"""Positional encoding — inject sequence order into token embeddings.

Method: self-attention (see 03-attention) is permutation-invariant by
itself — it has no notion of "token 3 comes before token 5" unless told.
Two classic fixes, both added directly to the token embedding:

  - Sinusoidal (Vaswani et al., 2017, original Transformer): fixed, not
    learned. pos_enc[pos, 2i] = sin(pos / 10000^(2i/d)), [pos, 2i+1] = cos(...).
    Different frequency per dimension pair lets the model infer relative
    offsets via linear combinations of sin/cos (no training needed for this
    part), and it generalizes to sequence lengths longer than seen in training.
  - Learned (GPT-2/BERT style): a plain embedding table of shape
    (max_seq_len, d_model), one learned row per position, added the same way.
    Simpler, fits training data better, but can't extrapolate past
    max_seq_len since there's no row learned for unseen positions.

Both are added (not concatenated) to the token embedding: final input to the
transformer = token_embedding + positional_encoding.
"""

import numpy as np


def sinusoidal_positional_encoding(max_seq_len: int, d_model: int) -> np.ndarray:
    """Fixed sin/cos table, shape (max_seq_len, d_model). No training involved."""
    assert d_model % 2 == 0, "sinusoidal encoding pairs sin/cos dims, so d_model must be even"
    position = np.arange(max_seq_len)[:, None]                       # (seq, 1)
    dim_pair = np.arange(0, d_model, 2)                                # (d_model/2,)
    freqs = 1.0 / (10000 ** (dim_pair / d_model))                      # (d_model/2,)
    angles = position * freqs                                          # (seq, d_model/2)

    pe = np.zeros((max_seq_len, d_model))
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles)
    return pe


class LearnedPositionalEncoding:
    """Trainable per-position table, shape (max_seq_len, d_model)."""

    def __init__(self, max_seq_len: int, d_model: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.weight = rng.normal(0, 0.02, size=(max_seq_len, d_model))
        self.max_seq_len = max_seq_len

    def __call__(self, seq_len: int) -> np.ndarray:
        assert seq_len <= self.max_seq_len, "position beyond trained range — can't extrapolate"
        return self.weight[:seq_len]


if __name__ == "__main__":
    max_seq_len, d_model = 16, 8

    pe = sinusoidal_positional_encoding(max_seq_len, d_model)
    print("sinusoidal pe shape:", pe.shape)
    assert pe.shape == (max_seq_len, d_model)
    assert not np.array_equal(pe[0], pe[1])  # distinct positions -> distinct encodings
    print("pos 0 != pos 1:", not np.array_equal(pe[0], pe[1]))
    print("values bounded in [-1, 1] (sin/cos range):", pe.min() >= -1 and pe.max() <= 1)

    try:
        sinusoidal_positional_encoding(4, 7)
        raise SystemExit("odd d_model should have been rejected")
    except AssertionError:
        pass

    learned = LearnedPositionalEncoding(max_seq_len, d_model)
    lpe = learned(seq_len=5)
    print("\nlearned pe shape (seq_len=5):", lpe.shape)
    assert lpe.shape == (5, d_model)

    token_emb = np.random.default_rng(1).normal(0, 0.02, size=(5, d_model))
    model_input = token_emb + lpe  # the actual fusion step used before the transformer
    print("token_emb + positional_emb shape:", model_input.shape)
    assert model_input.shape == token_emb.shape
