"""Feed-forward (MLP) block — per-token nonlinear transform after attention.

Method: attention mixes information ACROSS tokens; the MLP then transforms
each token ON ITS OWN (identical weights at every position, no cross-token
interaction). Classic form: expand to 4x width, nonlinearity, project back:
    MLP(x) = GELU(x W1 + b1) W2 + b2        (d_model -> 4*d_model -> d_model)
Most of a Transformer's parameters (about 2/3) live here, and it is where
much factual "knowledge" is believed to be stored.

Modern variant, SwiGLU (Shazeer 2020; LLaMA, PaLM): a gated MLP
    SwiGLU(x) = (silu(x W1) * (x W3)) W2
One branch gates the other elementwise. Uses hidden ~ 8/3 * d_model so the
parameter count matches the 4x GELU MLP.

Pros: nonlinearity is what lets stacked attention layers model more than
linear mixing; SwiGLU consistently beats plain GELU at equal params.
Cons: dominates parameter count and FLOPs (MoE, next file, makes it sparse).
"""

import numpy as np


def gelu(x: np.ndarray) -> np.ndarray:
    """tanh approximation of GELU (GPT-2 style)."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1 + np.exp(-x))


class MLP:
    """GPT-2 style: Linear -> GELU -> Linear."""

    def __init__(self, d_model: int, d_hidden: int = None, seed: int = 0):
        d_hidden = d_hidden or 4 * d_model
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.02, size=(d_model, d_hidden))
        self.b1 = np.zeros(d_hidden)
        self.W2 = rng.normal(0, 0.02, size=(d_hidden, d_model))
        self.b2 = np.zeros(d_model)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return gelu(x @ self.W1 + self.b1) @ self.W2 + self.b2


class SwiGLU:
    """LLaMA style: (silu(x W1) * x W3) W2, no biases."""

    def __init__(self, d_model: int, d_hidden: int = None, seed: int = 0):
        d_hidden = d_hidden or int(8 * d_model / 3)
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.02, size=(d_model, d_hidden))   # gate branch
        self.W3 = rng.normal(0, 0.02, size=(d_model, d_hidden))   # value branch
        self.W2 = rng.normal(0, 0.02, size=(d_hidden, d_model))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (silu(x @ self.W1) * (x @ self.W3)) @ self.W2


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    d_model = 16
    x = rng.normal(size=(2, 5, d_model))   # (batch, seq, d_model)

    for name, mlp in [("MLP (GELU)", MLP(d_model)), ("SwiGLU", SwiGLU(d_model))]:
        y = mlp(x)
        n_params = sum(v.size for v in vars(mlp).values())
        print(f"{name}: out {y.shape}, params {n_params}")
        assert y.shape == x.shape

        # per-token: permuting positions permutes the output the same way (no cross-token mixing)
        perm = rng.permutation(5)
        assert np.allclose(mlp(x[:, perm]), y[:, perm])
    print("applied independently per position: True")
    assert abs(gelu(np.array(0.0))) < 1e-12
