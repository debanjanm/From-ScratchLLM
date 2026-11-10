"""Transformer block — attention + feed-forward, wrapped in norms and residuals.

Method (GPT-2/LLaMA "pre-norm" layout), for input x of shape (batch, seq, d):
    x = x + Attention(Norm1(x))      # mix across tokens (causal)
    x = x + FFN(Norm2(x))            # transform each token (MLP or MoE)
Residual connections ("x + ...") give gradients a direct path through depth
and let each sublayer learn a small correction instead of a whole new
representation, which is what makes 100-layer stacks trainable. Pre-norm
(normalize BEFORE each sublayer) trains more stably than the original
post-norm (normalize after the add) because the residual stream itself is
never squashed.

This file also adds the BATCH dimension that 03-attention left out: shapes
are (batch, heads, seq, d_k) and numpy broadcasting applies the same causal
mask to every sequence and head. Softmax and the causal mask are reused from
03-attention.

Stack N of these (plus embeddings in front and a final norm + output
projection behind) and you have GPT: that is stage 05.
"""

import importlib
import pathlib
import sys

import numpy as np

here = pathlib.Path(__file__).parent
sys.path.insert(0, str(here))
sys.path.insert(0, str(here.parent / "03-attention"))
_norm = importlib.import_module("02_rmsnorm")
_mlp = importlib.import_module("03_mlp")
_moe = importlib.import_module("04_moe")
_sdpa = importlib.import_module("01_scaled_dot_product")   # softmax
_causal = importlib.import_module("02_causal_mask")        # causal_mask


class BatchedCausalSelfAttention:
    """Multi-head causal self-attention over (batch, seq, d_model)."""

    def __init__(self, d_model: int, num_heads: int, seed: int = 0):
        assert d_model % num_heads == 0
        self.h, self.d_k = num_heads, d_model // num_heads
        rng = np.random.default_rng(seed)
        self.W_q, self.W_k, self.W_v, self.W_o = (
            rng.normal(0, 0.02, size=(d_model, d_model)) for _ in range(4))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        B, T, D = x.shape
        split = lambda m: (x @ m).reshape(B, T, self.h, self.d_k).transpose(0, 2, 1, 3)  # (B,h,T,d_k)
        q, k, v = split(self.W_q), split(self.W_k), split(self.W_v)

        scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)       # (B,h,T,T)
        scores = np.where(_causal.causal_mask(T), scores, -np.inf)      # mask broadcasts over B,h
        out = _sdpa.softmax(scores) @ v                                  # (B,h,T,d_k)
        return out.transpose(0, 2, 1, 3).reshape(B, T, D) @ self.W_o


class TransformerBlock:
    def __init__(self, d_model: int, num_heads: int, ffn: str = "mlp", seed: int = 0):
        self.norm1, self.norm2 = _norm.RMSNorm(d_model), _norm.RMSNorm(d_model)
        self.attn = BatchedCausalSelfAttention(d_model, num_heads, seed)
        self.ffn = _moe.MoE(d_model, seed=seed) if ffn == "moe" else _mlp.SwiGLU(d_model, seed=seed)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    B, T, D, H = 2, 6, 16, 4
    x = rng.normal(size=(B, T, D))

    for ffn in ("mlp", "moe"):
        block = TransformerBlock(D, H, ffn=ffn)
        y = block(x)
        print(f"{ffn}: {x.shape} -> {y.shape}")
        assert y.shape == x.shape

        # causality: changing the LAST token must not change any earlier position's output
        x2 = x.copy()
        x2[:, -1] += 10.0
        y2 = block(x2)
        assert np.allclose(y[:, :-1], y2[:, :-1])
        assert not np.allclose(y[:, -1], y2[:, -1])

        # residual: with both sublayers' output weights zeroed, the block is the identity
        block.attn.W_o[:] = 0
        for w in ("W2",):
            if ffn == "mlp":
                getattr(block.ffn, w)[:] = 0
            else:
                for e in block.ffn.experts:
                    e.W2[:] = 0
        assert np.allclose(block(x), x)
    print("causal (future tokens can't change the past): True")
    print("residual path: zeroed sublayers -> block is identity: True")

    # stacking blocks keeps shape: this is the body of GPT
    h = x
    for i in range(3):
        h = TransformerBlock(D, H, seed=i)(h)
    assert h.shape == x.shape
    print("3 stacked blocks:", h.shape)
