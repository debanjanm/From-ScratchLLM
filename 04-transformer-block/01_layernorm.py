"""LayerNorm — normalize each token's feature vector to mean 0, variance 1.

Method: for one token vector x of size d_model, compute mean and variance
ACROSS ITS FEATURES (last axis), then
    y = (x - mean) / sqrt(var + eps) * gamma + beta
gamma (scale) and beta (shift) are learned per-feature, so the network can
undo the normalization if it wants. Unlike BatchNorm (statistics across the
batch), LayerNorm uses only the token's own features: no dependence on batch
size or other sequences, identical at train and inference, and works with
variable-length text. That is why Transformers use it.

Pros: stabilizes activations/gradients in deep stacks, batch-size independent.
Cons: computes both mean and variance (RMSNorm, next file, drops the mean
for cheaper compute with similar quality).
"""

import numpy as np


class LayerNorm:
    def __init__(self, d_model: int, eps: float = 1e-5):
        self.gamma = np.ones(d_model)   # learned scale, starts as identity
        self.beta = np.zeros(d_model)   # learned shift, starts as zero
        self.eps = eps

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """x: (..., d_model). Normalizes over the last axis, any leading shape."""
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + self.eps) * self.gamma + self.beta


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 3.0, size=(2, 4, 16))   # (batch, seq, d_model), far from mean 0 / var 1
    y = LayerNorm(16)(x)

    print("input  mean/std (one token):", round(x[0, 0].mean(), 3), round(x[0, 0].std(), 3))
    print("output mean/std (one token):", round(y[0, 0].mean(), 3), round(y[0, 0].std(), 3))
    assert y.shape == x.shape
    assert np.allclose(y.mean(axis=-1), 0, atol=1e-6)
    assert np.allclose(y.std(axis=-1), 1, atol=1e-3)

    # independence: normalizing a token alone gives the same result as inside a batch
    assert np.allclose(LayerNorm(16)(x[0, 0]), y[0, 0])
    print("each token normalized independently of batch/sequence: True")
