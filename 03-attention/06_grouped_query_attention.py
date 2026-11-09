"""Grouped-Query / Multi-Query Attention — share K/V projections across query heads.

Method: standard multi-head attention (03) gives every head its own Q, K,
AND V projection — at inference, the K/V for every head must be cached per
generated token (the "KV cache"), and cache size scales with num_heads.
Most of that K/V diversity turns out to be unnecessary. Fixes:

  - Multi-Query Attention (MQA, Shazeer 2019): ALL query heads share a
    single K and V projection. num_kv_heads = 1.
  - Grouped-Query Attention (GQA, Ainslie et al. 2023): a middle ground —
    query heads are split into groups, each group shares one K/V head.
    num_kv_heads is a tunable value between 1 (MQA) and num_heads (full MHA).

Query heads within a group still differ (separate Q projection per head);
only K/V are shared, so groups of queries attend using the same keys/values
but weight them differently.

Pros: KV cache size shrinks by num_heads/num_kv_heads — the main inference
memory/bandwidth bottleneck for autoregressive generation — with much less
quality loss than reducing num_heads itself. Used by LLaMA-2 70B, Mistral,
most 2023+ production LLMs.
Cons: some representational capacity lost vs full MHA (queries in a group
are stuck sharing one K/V view); MQA (extreme case) loses more than GQA.
"""

import importlib
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
_sdpa = importlib.import_module("01_scaled_dot_product")


class GroupedQueryAttention:
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int, seed: int = 0):
        assert d_model % num_heads == 0
        assert num_heads % num_kv_heads == 0, "query heads must split evenly into kv groups"
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = num_heads // num_kv_heads
        self.d_k = d_model // num_heads

        rng = np.random.default_rng(seed)
        scale = 0.02
        self.W_q = rng.normal(0, scale, size=(d_model, d_model))              # full heads
        self.W_k = rng.normal(0, scale, size=(d_model, num_kv_heads * self.d_k))  # fewer heads
        self.W_v = rng.normal(0, scale, size=(d_model, num_kv_heads * self.d_k))
        self.W_o = rng.normal(0, scale, size=(d_model, d_model))

    def __call__(self, x: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        seq_len = x.shape[0]
        Q = (x @ self.W_q).reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)
        K = (x @ self.W_k).reshape(seq_len, self.num_kv_heads, self.d_k).transpose(1, 0, 2)
        V = (x @ self.W_v).reshape(seq_len, self.num_kv_heads, self.d_k).transpose(1, 0, 2)

        head_outputs = []
        for h in range(self.num_heads):
            kv_head = h // self.group_size  # multiple query heads map to the same kv head
            out, _ = _sdpa.scaled_dot_product_attention(Q[h], K[kv_head], V[kv_head], mask=mask)
            head_outputs.append(out)

        concat = np.concatenate(head_outputs, axis=-1)
        return concat @ self.W_o


if __name__ == "__main__":
    seq_len, d_model, num_heads = 5, 16, 4
    x = np.random.default_rng(1).normal(size=(seq_len, d_model))

    gqa = GroupedQueryAttention(d_model, num_heads, num_kv_heads=2)
    out = gqa(x)
    print("GQA (4 query heads, 2 kv heads) output shape:", out.shape)
    assert out.shape == x.shape
    print("KV cache size vs full MHA:", f"{gqa.num_kv_heads}/{gqa.num_heads} heads "
          f"= {gqa.num_kv_heads / gqa.num_heads:.0%} of full MHA cache")

    mqa = GroupedQueryAttention(d_model, num_heads, num_kv_heads=1)  # MQA = GQA's extreme case
    out_mqa = mqa(x)
    print("\nMQA (num_kv_heads=1) output shape:", out_mqa.shape)
    assert out_mqa.shape == x.shape
    assert mqa.group_size == num_heads  # every query head shares the single kv head
