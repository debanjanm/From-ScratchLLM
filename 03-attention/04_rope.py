"""RoPE (Rotary Position Embedding) — encode position by rotating Q/K vectors.

Method: instead of adding a positional vector to the token embedding (see
02-embeddings sinusoidal/learned), RoPE rotates each Q and K vector by an
angle proportional to its position, applied pairwise across dimensions
(treat every 2 dims as a 2D plane, rotate by pos * theta_i, theta_i decaying
per dimension pair like the sinusoidal frequencies). The key property: the
dot product of a rotated Q at position m and rotated K at position n depends
only on (m - n), the RELATIVE distance — not on m and n individually. So
attention scores naturally encode relative position without ever adding a
positional vector to the residual stream. Used by GPT-NeoX, LLaMA, PaLM,
Mistral (2021+).

Pros: relative-position-aware "for free" inside the dot product, extrapolates
better to longer sequences than learned absolute position tables, no extra
parameters (theta is fixed, not learned).
Cons: only applies inside attention (Q/K), not to V or the residual stream
directly; implementation (interleaved rotation) is fiddlier than "just add a
vector".
"""

import importlib
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
_sdpa = importlib.import_module("01_scaled_dot_product")
_causal = importlib.import_module("02_causal_mask")


def rope_angles(seq_len: int, d_k: int) -> np.ndarray:
    """theta_i decays per dim pair like sinusoidal freqs. Shape: (seq_len, d_k/2)."""
    dim_pair = np.arange(0, d_k, 2)
    theta = 1.0 / (10000 ** (dim_pair / d_k))          # (d_k/2,)
    positions = np.arange(seq_len)[:, None]             # (seq_len, 1)
    return positions * theta                             # (seq_len, d_k/2)


def apply_rope(x: np.ndarray) -> np.ndarray:
    """x: (seq_len, d_k). Rotate each adjacent dim-pair by its position-dependent angle."""
    seq_len, d_k = x.shape
    assert d_k % 2 == 0, "RoPE needs an even head dim to form rotation pairs"
    angles = rope_angles(seq_len, d_k)
    cos, sin = np.cos(angles), np.sin(angles)

    x1, x2 = x[:, 0::2], x[:, 1::2]  # even/odd dims = the pair to rotate
    rotated = np.empty_like(x)
    rotated[:, 0::2] = x1 * cos - x2 * sin
    rotated[:, 1::2] = x1 * sin + x2 * cos
    return rotated


if __name__ == "__main__":
    seq_len, d_k = 6, 8
    rng = np.random.default_rng(0)
    Q = rng.normal(size=(seq_len, d_k))
    K = rng.normal(size=(seq_len, d_k))

    Q_rot, K_rot = apply_rope(Q), apply_rope(K)
    print("Q_rot shape:", Q_rot.shape)
    assert Q_rot.shape == Q.shape

    # core RoPE property: for the SAME q,k content, dot(rotate(q,m), rotate(k,n))
    # depends only on (n - m), not on m,n individually. Demonstrate by holding q,k
    # fixed and sliding both positions together, keeping their offset constant at 2.
    q_vec, k_vec = rng.normal(size=d_k), rng.normal(size=d_k)
    scores = []
    for m in range(6):
        n = m + 2
        q_rot_m = apply_rope(np.tile(q_vec, (n + 1, 1)))[m]  # rotate q_vec as if at position m
        k_rot_n = apply_rope(np.tile(k_vec, (n + 1, 1)))[n]  # rotate k_vec as if at position n
        scores.append(float(q_rot_m @ k_rot_n))
    print("scores for fixed q,k with (n - m) == 2, sliding m=0..5:", np.round(scores, 4))
    assert np.allclose(scores, scores[0], atol=1e-8)
    print("all equal (relative-position invariance holds):", np.allclose(scores, scores[0]))

    # wired into attention: rotate Q and K (never V), then run normal causal attention
    V = rng.normal(size=(seq_len, d_k))
    out, w = _sdpa.scaled_dot_product_attention(Q_rot, K_rot, V, mask=_causal.causal_mask(seq_len))
    assert out.shape == V.shape and np.allclose(w.sum(axis=-1), 1.0)
    print("attention with RoPE-rotated Q/K ran, output shape:", out.shape)
