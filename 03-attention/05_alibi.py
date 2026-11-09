"""ALiBi (Attention with Linear Biases) — encode position with a bias, not rotation/addition.

Method: instead of modifying Q/K (RoPE) or adding a positional vector to
the embedding (sinusoidal/learned), ALiBi adds a fixed, non-learned penalty
directly to the raw attention scores, proportional to the distance between
query and key positions:

    scores[i, j] = (Q K^T / sqrt(d_k))[i, j] - slope_h * (i - j)   for j <= i

Each head h gets its own fixed `slope_h` (geometrically spaced, e.g.
1/2, 1/4, 1/8, ... across heads), so different heads penalize distance at
different rates — some heads stay near-local, others stay closer to global.
No extra parameters, no rotation math, applied once at score time. Used by
BLOOM, MPT (2021+), specifically to get strong length extrapolation (train
short, run long) cheaper than RoPE needs.

Pros: dead simple (one subtraction), best-in-class extrapolation to sequence
lengths far beyond training length, zero extra learned parameters.
Cons: bias is a fixed linear function of distance — can't represent more
complex positional relationships RoPE's rotation can.
"""

import importlib
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
_sdpa = importlib.import_module("01_scaled_dot_product")
_causal = importlib.import_module("02_causal_mask")


def alibi_slopes(num_heads: int) -> np.ndarray:
    """Geometric sequence of per-head slopes, e.g. 2^-1, 2^-2, ... for 8 heads (paper's scheme)."""
    ratio = 2 ** (-8 / num_heads)
    return np.array([ratio ** (h + 1) for h in range(num_heads)])


def alibi_bias(seq_len: int, slope: float) -> np.ndarray:
    """(seq_len, seq_len) bias matrix for one head: -slope * (i - j), 0 where j > i (future)."""
    i = np.arange(seq_len)[:, None]
    j = np.arange(seq_len)[None, :]
    distance = np.maximum(i - j, 0)  # only penalize past/self; future is masked separately
    return -slope * distance


if __name__ == "__main__":
    num_heads, seq_len = 4, 6
    slopes = alibi_slopes(num_heads)
    print("per-head slopes:", np.round(slopes, 4))
    assert len(slopes) == num_heads
    assert (np.diff(slopes) < 0).all()  # slopes strictly decrease head to head

    bias = alibi_bias(seq_len, slope=slopes[0])
    print("\nbias matrix for head 0 (most local):\n", np.round(bias, 2))
    assert bias[3, 0] < bias[3, 2]  # farther key (j=0) penalized more than closer key (j=2)
    print("\nfarther key penalized more than closer key:", bias[3, 0] < bias[3, 2])

    rng = np.random.default_rng(0)
    d_k = 8
    Q = K = rng.normal(size=(seq_len, d_k))
    V = rng.normal(size=(seq_len, d_k))
    mask = _causal.causal_mask(seq_len)
    _, w_plain = _sdpa.scaled_dot_product_attention(Q, K, V, mask=mask)
    _, w_alibi = _sdpa.scaled_dot_product_attention(Q, K, V, mask=mask, bias=bias)
    assert np.allclose(w_alibi.sum(axis=-1), 1.0)
    assert np.allclose(np.triu(w_alibi, k=1), 0)        # future still masked
    # bias shifts attention toward nearby keys: weight on the farthest key shrinks
    assert w_alibi[5, 0] < w_plain[5, 0]
    print("\nthrough attention: weight on farthest key (row 5, col 0):",
          round(w_plain[5, 0], 4), "->", round(w_alibi[5, 0], 4), "with ALiBi")
