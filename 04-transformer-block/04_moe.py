"""Mixture of Experts (MoE) — a sparse drop-in replacement for the MLP.

Method: instead of one big MLP that every token passes through, keep
`n_experts` separate MLPs plus a small router. For each token:
  1. router scores = x @ W_router  (one score per expert),
  2. keep only the top-k experts, softmax their scores into gate weights,
  3. output = sum of gate_i * expert_i(x) over the chosen k experts.
Only k of n experts run per token, so parameters scale with n_experts while
per-token compute scales with k. Used by Mixtral (8 experts, top-2), Switch
Transformer, DeepSeek-MoE, GPT-4 (reported).

This is a LAYER swap, not a new architecture: same input/output shape as the
MLP in 03, so it plugs into the block unchanged.

Pros: far more capacity (parameters) at roughly constant FLOPs per token.
Cons: all experts must sit in memory; routing can collapse onto a few
experts, so real systems add a load-balancing loss (omitted here); sparse
dispatch is awkward on hardware. This snippet loops over experts for
clarity rather than speed.
"""

import importlib
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
_mlp = importlib.import_module("03_mlp")


class MoE:
    def __init__(self, d_model: int, n_experts: int = 4, top_k: int = 2, seed: int = 0):
        assert 1 <= top_k <= n_experts
        self.top_k = top_k
        rng = np.random.default_rng(seed)
        self.W_router = rng.normal(0, 0.02, size=(d_model, n_experts))
        self.experts = [_mlp.MLP(d_model, seed=seed + 1 + i) for i in range(n_experts)]

    def route(self, tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """tokens: (N, d). Returns (chosen expert ids (N, k), gate weights (N, k) summing to 1)."""
        logits = tokens @ self.W_router
        chosen = np.argsort(-logits, axis=-1)[:, :self.top_k]
        top_logits = np.take_along_axis(logits, chosen, axis=-1)
        e = np.exp(top_logits - top_logits.max(axis=-1, keepdims=True))
        return chosen, e / e.sum(axis=-1, keepdims=True)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """x: (..., d_model) -> same shape."""
        shape = x.shape
        tokens = x.reshape(-1, shape[-1])
        chosen, gates = self.route(tokens)

        out = np.zeros_like(tokens)
        for e_id, expert in enumerate(self.experts):
            for slot in range(self.top_k):
                rows = np.where(chosen[:, slot] == e_id)[0]
                if rows.size:   # run this expert only on tokens routed to it
                    out[rows] += gates[rows, slot, None] * expert(tokens[rows])
        return out.reshape(shape)


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    d_model, n_experts, top_k = 16, 4, 2
    moe = MoE(d_model, n_experts, top_k)
    x = rng.normal(size=(2, 6, d_model))

    y = moe(x)
    print("output shape:", y.shape)
    assert y.shape == x.shape

    tokens = x.reshape(-1, d_model)
    chosen, gates = moe.route(tokens)
    assert chosen.shape == (12, top_k) and np.allclose(gates.sum(axis=-1), 1.0)
    print("expert load (tokens per expert):", np.bincount(chosen.ravel(), minlength=n_experts))

    # check one token by hand: output = sum of gate * expert(x) over its chosen experts
    t = 0
    manual = sum(gates[t, s] * moe.experts[chosen[t, s]](tokens[t]) for s in range(top_k))
    assert np.allclose(y.reshape(-1, d_model)[t], manual)

    # top_k=1 degenerates to "each token goes through exactly one expert, gate 1.0"
    one = MoE(d_model, n_experts, top_k=1)
    c1, g1 = one.route(tokens)
    assert np.allclose(g1, 1.0)
    assert np.allclose(one(x).reshape(-1, d_model)[3], one.experts[c1[3, 0]](tokens[3]))
    print("sparse: each token ran", top_k, "of", n_experts, "experts")
