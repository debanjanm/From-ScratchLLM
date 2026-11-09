"""Token embedding — lookup table mapping token id -> dense vector.

Method: an embedding table is just a matrix of shape (vocab_size, d_model).
"Embedding" a token id is a row lookup (equivalent to one-hot @ matrix, but
implemented as direct indexing since that's what every framework actually
does — one-hot matmul would waste O(vocab_size) work per token). The matrix
starts random and is learned via backprop like any other weight.

Pros: turns discrete ids into continuous vectors a network can do math on;
similar tokens can end up with similar vectors after training.
Cons: table size is vocab_size * d_model — the single largest parameter
block in small models (e.g. GPT-2 small: 50257 * 768 ≈ 38M params just here).
"""

import numpy as np


class TokenEmbedding:
    """Row-lookup embedding table, weights ~N(0, 0.02) like GPT-2 init."""

    def __init__(self, vocab_size: int, d_model: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.weight = rng.normal(0, 0.02, size=(vocab_size, d_model))

    def __call__(self, ids: list[int]) -> np.ndarray:
        """ids -> (len(ids), d_model) matrix, one row per token."""
        return self.weight[ids]


if __name__ == "__main__":
    vocab_size, d_model = 100, 8
    emb = TokenEmbedding(vocab_size, d_model)

    ids = [3, 17, 3]  # note: id 3 repeats
    vecs = emb(ids)
    print("ids:", ids)
    print("shape:", vecs.shape)
    assert vecs.shape == (len(ids), d_model)
    assert np.array_equal(vecs[0], vecs[2])  # same id -> same vector (pure lookup)
    print("repeated id 3 gives identical vector both times: ", np.array_equal(vecs[0], vecs[2]))
