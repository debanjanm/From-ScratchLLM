"""RMSNorm — LayerNorm without the mean subtraction (and without beta).

Method: rescale each token vector by its root-mean-square:
    y = x / sqrt(mean(x^2) + eps) * gamma
No mean is computed or subtracted, and there is no shift parameter. The
claim (Zhang & Sennrich, 2019): LayerNorm's benefit comes from re-scaling,
not re-centering, so dropping the centering saves compute with essentially
the same quality. Used by LLaMA, Mistral, Gemma and most modern open LLMs.

Pros: fewer ops and half the parameters of LayerNorm, same stability.
Cons: output is not zero-mean, so it only matches LayerNorm when the input
already has mean ~0.
"""

import importlib
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
_ln = importlib.import_module("01_layernorm")


class RMSNorm:
    def __init__(self, d_model: int, eps: float = 1e-6):
        self.gamma = np.ones(d_model)
        self.eps = eps

    def __call__(self, x: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return x / rms * self.gamma


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 3.0, size=(2, 4, 16))
    y = RMSNorm(16)(x)

    rms_out = np.sqrt((y ** 2).mean(axis=-1))
    print("output RMS (should be ~1):", np.round(rms_out[0], 4))
    print("output mean (NOT 0, unlike LayerNorm):", round(y[0, 0].mean(), 3))
    assert np.allclose(rms_out, 1, atol=1e-4)
    assert abs(y[0, 0].mean()) > 0.1   # no centering happened

    # for zero-mean input the two norms agree
    z = rng.normal(size=(3, 16))
    z = z - z.mean(axis=-1, keepdims=True)
    assert np.allclose(RMSNorm(16, eps=1e-5)(z), _ln.LayerNorm(16)(z), atol=1e-4)
    print("matches LayerNorm exactly when input is already zero-mean: True")
