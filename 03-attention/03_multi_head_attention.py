"""Multi-head attention — run several smaller attentions in parallel, concat, project.

Method: single-head attention (01) forces all of Q/K/V to share one d_model-
sized subspace to capture every kind of relationship (syntax, coreference,
position, ...) at once. Multi-head splits d_model into `num_heads` smaller
d_k = d_model / num_heads chunks, runs scaled dot-product attention
independently per chunk (so each head can specialize), concatenates the
per-head outputs back to d_model, and applies one final output projection
W_O to mix information across heads. Q/K/V projections (W_q, W_k, W_v) are
learned once for the full d_model then reshaped into heads — not one
separate weight matrix per head.

Pros: lets different heads attend to different relationship types (empirically
some heads learn positional patterns, others learn syntax/coreference);
same total compute as one big head (split then concat is free).
Cons: more parameters to manage (4 weight matrices vs 3), no guarantee heads
specialize usefully without enough training data/capacity.
"""

import importlib
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
_sdpa = importlib.import_module("01_scaled_dot_product")


class MultiHeadAttention:
    def __init__(self, d_model: int, num_heads: int, seed: int = 0):
        assert d_model % num_heads == 0, "d_model must divide evenly into heads"
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        rng = np.random.default_rng(seed)
        scale = 0.02
        self.W_q = rng.normal(0, scale, size=(d_model, d_model))
        self.W_k = rng.normal(0, scale, size=(d_model, d_model))
        self.W_v = rng.normal(0, scale, size=(d_model, d_model))
        self.W_o = rng.normal(0, scale, size=(d_model, d_model))

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """(seq, d_model) -> (num_heads, seq, d_k)."""
        seq_len = x.shape[0]
        return x.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)

    def __call__(self, x: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """x: (seq, d_model), self-attention (Q=K=V=x projected). Returns (seq, d_model)."""
        Q, K, V = self._split_heads(x @ self.W_q), self._split_heads(x @ self.W_k), self._split_heads(x @ self.W_v)

        head_outputs = []
        for h in range(self.num_heads):
            out, _ = _sdpa.scaled_dot_product_attention(Q[h], K[h], V[h], mask=mask)
            head_outputs.append(out)

        concat = np.concatenate(head_outputs, axis=-1)  # (seq, d_model)
        return concat @ self.W_o


if __name__ == "__main__":
    seq_len, d_model, num_heads = 5, 16, 4
    mha = MultiHeadAttention(d_model, num_heads)

    x = np.random.default_rng(1).normal(size=(seq_len, d_model))
    out = mha(x)
    print("input shape:", x.shape)
    print("output shape:", out.shape)
    assert out.shape == x.shape  # multi-head attention preserves shape
    print("d_k per head:", mha.d_k, f"({num_heads} heads x {mha.d_k} = {num_heads * mha.d_k} = d_model)")
