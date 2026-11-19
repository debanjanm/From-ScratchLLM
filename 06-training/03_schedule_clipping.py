"""Learning-rate schedule and gradient clipping — the two stabilizers of LLM training.

Warmup + cosine decay: Adam's second-moment estimates are unreliable in the
first steps, so a full learning rate there can blow up training. Ramp the LR
linearly from 0 to the peak over `warmup` steps, then decay it along a
half-cosine down to `min_lr` by the final step. Large steps early (explore),
small steps late (settle into a minimum).

Global-norm gradient clipping: one bad batch can produce a huge gradient and
wreck the weights. Compute the norm of ALL gradients together,
    norm = sqrt(sum over every tensor of sum(g^2)),
and if it exceeds `max_norm`, scale every gradient by max_norm / norm. The
direction is preserved; only the step size is capped. (Clipping per tensor
would change the direction, which is why the norm is global.)
"""

import math

import numpy as np


def lr_at(step: int, max_lr: float, warmup: int, total: int, min_lr: float = 0.0) -> float:
    if step < warmup:
        return max_lr * (step + 1) / warmup                      # linear ramp up
    progress = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def clip_grad_norm(grads: dict, max_norm: float) -> float:
    """Scale grads in place so their global L2 norm <= max_norm. Returns the norm BEFORE clipping."""
    norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
    if norm > max_norm:
        scale = max_norm / (norm + 1e-6)
        for g in grads.values():
            g *= scale
    return norm


if __name__ == "__main__":
    max_lr, warmup, total, min_lr = 1e-3, 10, 100, 1e-4
    lrs = [lr_at(s, max_lr, warmup, total, min_lr) for s in range(total)]
    print("lr at steps 0, 9, 10, 55, 99:", [f"{lrs[i]:.2e}" for i in (0, 9, 10, 55, 99)])
    assert lrs[0] < lrs[9] <= max_lr                              # warming up
    assert abs(lrs[10] - max_lr) < 1e-9                           # peak right after warmup
    assert all(a >= b for a, b in zip(lrs[10:], lrs[11:]))        # then monotonic decay
    assert abs(lrs[-1] - min_lr) < 2e-5                           # ends near the floor

    grads = {"a": np.full((2, 2), 3.0), "b": np.full(4, 4.0)}     # norm = sqrt(36 + 64) = 10
    direction = grads["a"][0, 0] / grads["b"][0]
    before = clip_grad_norm(grads, max_norm=1.0)
    after = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
    print(f"norm before {before:.2f}, after clipping to 1.0: {after:.3f}")
    assert abs(before - 10) < 1e-9 and abs(after - 1.0) < 1e-4
    assert abs(grads["a"][0, 0] / grads["b"][0] - direction) < 1e-9   # direction preserved

    small = {"a": np.full(2, 0.1)}
    clip_grad_norm(small, max_norm=1.0)
    assert np.allclose(small["a"], 0.1)                           # below the cap: untouched
