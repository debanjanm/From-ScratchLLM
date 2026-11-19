"""AdamW — the optimizer behind nearly every LLM.

Method: plain gradient descent (w -= lr * g) treats every weight alike and
is noisy. Adam keeps two running averages per weight:
    m = b1*m + (1-b1)*g          (momentum: smoothed direction)
    v = b2*v + (1-b2)*g^2        (smoothed squared gradient: per-weight scale)
and steps  w -= lr * m_hat / (sqrt(v_hat) + eps), where m_hat/v_hat undo the
zero-initialization bias early in training (divide by 1 - b^t). Weights with
consistently large gradients get smaller steps; rarely-updated ones get
larger steps.

AdamW (Loshchilov & Hutter) adds DECOUPLED weight decay: shrink weights
directly, w -= lr * wd * w, instead of folding an L2 term into the gradient
(where Adam's rescaling would distort it). Decay is applied only to weight
matrices, not to norm gains or other 1-D params (standard practice).

Minimal on purpose: the fuller optimizer family (SGD, RMSProp, ...) belongs in
the FromScratchDL repo; this is just what GPT training needs.
"""

import numpy as np


class AdamW:
    def __init__(self, params: dict, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, betas[0], betas[1], eps, weight_decay
        self.m = {k: np.zeros_like(w) for k, w in params.items()}
        self.v = {k: np.zeros_like(w) for k, w in params.items()}
        self.t = 0

    def step(self, params: dict, grads: dict, lr: float = None):
        """Update params in place. Pass `lr` to override (e.g. from a schedule)."""
        lr = self.lr if lr is None else lr
        self.t += 1
        for k, w in params.items():
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            m_hat = self.m[k] / (1 - self.b1 ** self.t)
            v_hat = self.v[k] / (1 - self.b2 ** self.t)
            if self.wd and w.ndim >= 2:                      # decoupled decay, matrices only
                w -= lr * self.wd * w
            w -= lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def state_dict(self) -> dict:
        return {"t": self.t, "m": self.m, "v": self.v}

    def load_state_dict(self, state: dict):
        self.t, self.m, self.v = state["t"], state["m"], state["v"]


if __name__ == "__main__":
    # minimize f(w) = sum((w - 3)^2): gradient is 2(w - 3)
    params = {"w": np.zeros((2, 2))}
    opt = AdamW(params, lr=0.1)
    for _ in range(300):
        opt.step(params, {"w": 2 * (params["w"] - 3)})
    print("w after 300 Adam steps (target 3):", np.round(params["w"], 3).tolist())
    assert np.allclose(params["w"], 3, atol=1e-2)

    # decoupled decay: zero gradient, so weights shrink only by the decay term
    decayed = {"mat": np.ones((2, 2)), "gain": np.ones(2)}
    opt = AdamW(decayed, lr=0.1, weight_decay=0.5)
    opt.step(decayed, {k: np.zeros_like(w) for k, w in decayed.items()})
    print("after one decay step: matrix", decayed["mat"][0, 0], "| 1-D gain", decayed["gain"][0])
    assert np.allclose(decayed["mat"], 1 - 0.1 * 0.5) and np.allclose(decayed["gain"], 1.0)
