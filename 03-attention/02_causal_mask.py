"""Causal (autoregressive) mask — stop a token from attending to future tokens.

Method: plain attention (01) lets every query see every key, including ones
that come later in the sequence. A decoder-only LM (GPT-style) predicts
token t+1 from tokens 1..t, so at training time it must NOT be allowed to
peek at t+1..end — otherwise it trivially "cheats" by copying the answer.
Fix: build a boolean mask where position i may attend to position j only if
j <= i (lower-triangular), and pass it into scaled_dot_product_attention's
`mask` argument, which sets disallowed scores to -inf before softmax (so
their attention weight becomes exactly 0).

Pros: enables training on the full sequence in one parallel forward pass
(teacher forcing) while still respecting autoregressive order.
Cons: none really — it's cheap (one triangular matrix) and required
correctness for any decoder-only model, not an optional efficiency knob.
"""

import importlib
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
_sdpa = importlib.import_module("01_scaled_dot_product")


def causal_mask(seq_len: int) -> np.ndarray:
    """(seq_len, seq_len) bool matrix, True where position i may attend to position j (j<=i)."""
    return np.tril(np.ones((seq_len, seq_len), dtype=bool))


if __name__ == "__main__":
    seq_len = 5
    mask = causal_mask(seq_len)
    print("mask:\n", mask.astype(int))
    assert mask[0].sum() == 1   # first token sees only itself
    assert mask[-1].sum() == seq_len  # last token sees everything

    rng = np.random.default_rng(0)
    d_k = 8
    Q = K = V = rng.normal(size=(seq_len, d_k))
    out, weights = _sdpa.scaled_dot_product_attention(Q, K, V, mask=mask)

    print("\nattention weights (row i should be 0 for j > i):\n", np.round(weights, 3))
    assert np.allclose(np.triu(weights, k=1), 0)  # nothing attends to the future
    print("upper triangle (future positions) is all zero:", np.allclose(np.triu(weights, k=1), 0))
