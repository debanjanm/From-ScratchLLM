"""Sampling strategies — turn next-token logits into one chosen token.

The model outputs a score per vocab token; how you PICK from it shapes the
text:
  - Greedy: argmax. Deterministic, but repetitive and dull.
  - Temperature T: divide logits by T before softmax. T<1 sharpens (safer),
    T>1 flattens (more random), T->0 approaches greedy.
  - Top-k: keep only the k highest-scoring tokens, renormalize. Cuts off the
    long tail of nonsense tokens, but k is fixed regardless of how confident
    the model is.
  - Top-p (nucleus, Holtzman 2019): keep the smallest set of tokens whose
    probability mass reaches p. The set size adapts: tiny when the model is
    sure, large when it is uncertain.
Order used here: temperature -> top-k -> top-p -> sample (the usual one).
"""

import importlib
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "03-attention"))
_softmax = importlib.import_module("01_scaled_dot_product").softmax


def sample(logits: np.ndarray, temperature: float = 1.0, top_k: int = None,
           top_p: float = None, rng: np.random.Generator = None) -> int:
    """logits: (vocab,). temperature=0 means greedy."""
    if temperature == 0:
        return int(np.argmax(logits))
    rng = rng or np.random.default_rng()
    logits = logits / temperature

    if top_k is not None:
        kth = np.sort(logits)[-min(top_k, logits.size)]
        logits = np.where(logits >= kth, logits, -np.inf)

    probs = _softmax(logits)
    if top_p is not None:
        order = np.argsort(-probs)
        cum = np.cumsum(probs[order])
        keep = order[: np.searchsorted(cum, top_p) + 1]   # smallest prefix reaching p
        mask = np.zeros_like(probs, dtype=bool)
        mask[keep] = True
        probs = np.where(mask, probs, 0.0)
        probs /= probs.sum()

    return int(rng.choice(probs.size, p=probs))


if __name__ == "__main__":
    logits = np.array([4.0, 3.0, 2.0, 1.0, 0.0, -1.0])
    rng = np.random.default_rng(0)
    draw = lambda n=2000, **kw: np.array([sample(logits, rng=rng, **kw) for _ in range(n)])

    assert sample(logits, temperature=0) == 0                    # greedy
    assert set(draw(top_k=1)) == {0}                             # top-1 == greedy
    assert set(draw(top_k=3)) <= {0, 1, 2}                       # only the 3 best ever appear
    print("top-k=3 drew only from tokens:", sorted(set(draw(top_k=3))))

    # nucleus: probs ~ [.64,.23,.09,...]; p=0.8 needs tokens 0 and 1
    probs = _softmax(logits)
    print("probs:", np.round(probs, 3))
    assert set(draw(top_p=0.8)) <= {0, 1}
    print("top-p=0.8 drew only from tokens:", sorted(set(draw(top_p=0.8))))

    # temperature reshapes the distribution: low T concentrates on the top token
    share_top = lambda T: np.mean(draw(temperature=T) == 0)
    cold, normal, hot = share_top(0.3), share_top(1.0), share_top(3.0)
    print(f"share of top token  T=0.3: {cold:.2f}  T=1: {normal:.2f}  T=3: {hot:.2f}")
    assert cold > normal > hot
