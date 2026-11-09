"""Scaled dot-product attention — the core operation behind every Transformer.

Method: for queries Q, keys K, values V (each rows = tokens, cols = d_k dims):
    attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

QK^T gives a similarity score between every query and every key (how much
token i should attend to token j). Dividing by sqrt(d_k) prevents scores
from growing large with d_k (large dot products push softmax into
near-one-hot regions with vanishing gradients — the scaling keeps variance
~1 regardless of dimension). Softmax turns scores into a probability
distribution per query row; multiplying by V produces a weighted average of
value vectors, weighted by relevance.

Pros: fully parallel (no recurrence like RNNs), directly models
long-range dependencies in one matmul.
Cons: O(seq_len^2) time/memory — the seq_len x seq_len score matrix is why
long-context attention (sliding window, etc. — see 07) exists.
"""

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax: subtract row max before exp to avoid overflow."""
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                                  mask: np.ndarray = None,
                                  bias: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
    """Q: (seq_q, d_k), K/V: (seq_k, d_k)/(seq_k, d_v). mask: (seq_q, seq_k), True = keep.
    bias: optional (seq_q, seq_k) added to scores before masking (ALiBi plugs in here).
    Returns (output (seq_q, d_v), attention_weights (seq_q, seq_k))."""
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if bias is not None:
        scores = scores + bias
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    weights = softmax(scores, axis=-1)
    return weights @ V, weights


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    seq_len, d_k, d_v = 4, 8, 8
    Q = rng.normal(size=(seq_len, d_k))
    K = rng.normal(size=(seq_len, d_k))
    V = rng.normal(size=(seq_len, d_v))

    out, weights = scaled_dot_product_attention(Q, K, V)
    print("output shape:", out.shape)
    print("weights shape:", weights.shape)
    assert out.shape == (seq_len, d_v)
    assert np.allclose(weights.sum(axis=-1), 1.0)  # each query's weights form a distribution
    print("each row of attention weights sums to 1:", np.allclose(weights.sum(axis=-1), 1.0))
