"""Sliding window attention — cap each token's attention to a local window.

Method: full causal attention (02) still costs O(seq_len^2) — every token
attends to ALL previous tokens. Sliding window caps that: token i may only
attend to tokens in [i - window + 1, i] (plus causality, so never the
future either). Build this as just another mask, same shape as the causal
mask, combined with it — it's a banded lower-triangular matrix instead of a
full lower-triangular one. Used by Mistral 7B, Longformer (2020+), often
stacked across layers so information still propagates further than one
window (layer L's window covers ~L * window_size effective range).

Pros: O(seq_len * window) instead of O(seq_len^2) — makes long-context
cheap; local syntax/semantics are usually local anyway, so quality loss is
often small.
Cons: a single layer genuinely cannot connect tokens farther apart than
`window`; long-range dependencies need either many stacked layers or an
occasional full-attention layer mixed in (as some models do).
"""

import importlib
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
_sdpa = importlib.import_module("01_scaled_dot_product")


def sliding_window_mask(seq_len: int, window: int) -> np.ndarray:
    """True where position i may attend to j: causal (j <= i) AND local (i - j < window)."""
    i = np.arange(seq_len)[:, None]
    j = np.arange(seq_len)[None, :]
    return (j <= i) & (i - j < window)


if __name__ == "__main__":
    seq_len, window = 8, 3
    mask = sliding_window_mask(seq_len, window)
    print("mask (window=3):\n", mask.astype(int))

    assert mask[-1].sum() == window       # last token sees exactly `window` tokens (itself + 2 back)
    assert mask[0].sum() == 1             # first token still sees only itself
    print("\nlast row attends to exactly", mask[-1].sum(), f"tokens (window={window})")

    rng = np.random.default_rng(0)
    d_k = 8
    Q = K = V = rng.normal(size=(seq_len, d_k))
    out, weights = _sdpa.scaled_dot_product_attention(Q, K, V, mask=mask)

    # token 7 must have zero weight on token 3 (distance 4, outside window=3)
    print("weight from token 7 on token 3 (distance 4, outside window):", weights[7, 3])
    assert weights[7, 3] == 0.0
    print("weight from token 7 on token 6 (distance 1, inside window):", round(weights[7, 6], 4))
    assert weights[7, 6] > 0.0
