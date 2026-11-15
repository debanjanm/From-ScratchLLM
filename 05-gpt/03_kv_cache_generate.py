"""KV cache + autoregressive generation.

Naive generation re-runs the whole sequence through the model for every new
token: step t processes t positions, so n new tokens cost O(n^2) token-
forwards (and every layer recomputes K and V for tokens it already saw).

But causal attention means a past token's K and V never change. So cache them:
per layer keep K, V for all earlier positions, and for each new token compute
only ITS q, k, v, append k, v to the cache, and attend q over the cached
K, V. Each step now processes exactly one token: O(n) token-forwards total.
No mask is needed in a decode step, since the cache holds only the past.

The catch (why GQA/MQA, stage 03, exist): the cache grows as
layers * 2 * seq * heads * d_k and is the main inference memory cost.

Here the prompt is "prefilled" token by token through the same decode step,
which keeps the code tiny; real systems run the prompt as one parallel pass.
The check at the bottom proves cached and naive generation agree exactly.
"""

import importlib
import pathlib
import sys

import numpy as np

here = pathlib.Path(__file__).parent
sys.path.insert(0, str(here))
_gpt = importlib.import_module("01_gpt_model")
_sampling = importlib.import_module("02_sampling")
_softmax = _gpt._sdpa.softmax


class KVCache:
    def __init__(self, num_layers: int):
        self.k = [None] * num_layers   # each: (batch, heads, seq_so_far, d_k)
        self.v = [None] * num_layers

    def append(self, layer: int, k: np.ndarray, v: np.ndarray):
        self.k[layer] = k if self.k[layer] is None else np.concatenate([self.k[layer], k], axis=2)
        self.v[layer] = v if self.v[layer] is None else np.concatenate([self.v[layer], v], axis=2)
        return self.k[layer], self.v[layer]

    def size(self) -> int:
        return sum(a.size for a in self.k + self.v if a is not None)


def decode_step(model, token_ids: np.ndarray, pos: int, cache: KVCache) -> np.ndarray:
    """token_ids: (batch, 1) new token at position `pos`. Returns logits (batch, 1, vocab)."""
    x = model.tok(token_ids) + model.pos.weight[pos]
    for layer, block in enumerate(model.blocks):
        a = block.attn
        h = block.norm1(x)
        B = h.shape[0]
        split = lambda W: (h @ W).reshape(B, 1, a.h, a.d_k).transpose(0, 2, 1, 3)  # (B,h,1,d_k)
        q, k_new, v_new = split(a.W_q), split(a.W_k), split(a.W_v)

        K, V = cache.append(layer, k_new, v_new)                    # (B,h,pos+1,d_k)
        weights = _softmax(q @ K.transpose(0, 1, 3, 2) / np.sqrt(a.d_k))
        attn_out = (weights @ V).transpose(0, 2, 1, 3).reshape(B, 1, -1) @ a.W_o
        x = x + attn_out
        x = x + block.ffn(block.norm2(x))
    return model.norm_f(x) @ model.tok.weight.T


def generate_naive(model, prompt: list[int], n_new: int, **sample_kw) -> list[int]:
    ids = list(prompt)
    for _ in range(n_new):
        logits = model(np.array([ids]))[0, -1]          # whole sequence, every step
        ids.append(_sampling.sample(logits, **sample_kw))
    return ids


def generate_cached(model, prompt: list[int], n_new: int, **sample_kw) -> list[int]:
    cache = KVCache(len(model.blocks))
    ids = list(prompt)
    for pos, tok in enumerate(prompt):                  # prefill the cache
        logits = decode_step(model, np.array([[tok]]), pos, cache)
    for _ in range(n_new):
        ids.append(_sampling.sample(logits[0, -1], **sample_kw))
        logits = decode_step(model, np.array([[ids[-1]]]), len(ids) - 1, cache)
    return ids[: len(prompt) + n_new]


if __name__ == "__main__":
    vocab, d_model, heads, layers, max_len = 50, 32, 4, 2, 64
    model = _gpt.GPT(vocab, d_model, heads, layers, max_len)
    prompt, n_new = [3, 14, 15, 9], 12

    naive = generate_naive(model, prompt, n_new, temperature=0)     # greedy
    cached = generate_cached(model, prompt, n_new, temperature=0)
    print("naive :", naive)
    print("cached:", cached)
    assert naive == cached                                          # identical output
    # greedy on an untrained, weight-tied model just repeats a token, so also compare seeded sampling
    s_naive = generate_naive(model, prompt, n_new, temperature=1.0, rng=np.random.default_rng(0))
    s_cached = generate_cached(model, prompt, n_new, temperature=1.0, rng=np.random.default_rng(0))
    print("sampled:", s_cached)
    assert s_naive == s_cached and len(set(s_cached[len(prompt):])) > 1

    # the cached step reproduces the full forward pass logits
    ids = np.array([naive])
    full = model(ids)
    cache = KVCache(layers)
    stepped = np.concatenate([decode_step(model, ids[:, t:t + 1], t, cache) for t in range(ids.shape[1])], axis=1)
    assert np.allclose(full, stepped)
    print("cached logits == full-forward logits at every position: True")

    naive_work = sum(range(len(prompt), len(prompt) + n_new))      # positions re-processed per step
    cached_work = len(prompt) + n_new
    print(f"token-forwards: naive {naive_work} vs cached {cached_work}")
    print(f"cache holds {cache.size()} floats for {ids.shape[1]} tokens "
          f"(= layers*2*seq*heads*d_k = {layers * 2 * ids.shape[1] * heads * (d_model // heads)})")
    assert cached_work < naive_work
