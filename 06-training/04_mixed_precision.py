"""Mixed precision — fp16 compute, fp32 master weights, dynamic loss scaling.

Why: half precision (fp16) halves memory and is much faster on GPUs, but it
has two failure modes this file demonstrates with real numpy float16:
  1. UNDERFLOW: fp16's smallest values are ~6e-8; tiny gradients become 0 and
     learning silently stops. Fix = LOSS SCALING: multiply the loss (hence all
     gradients) by a big factor S before backward, divide by S afterwards in
     fp32. Gradients now sit in fp16's representable range.
  2. LOST UPDATES: fp16 has ~3 decimal digits, so w=1.0 plus an update of 1e-4
     rounds back to 1.0. Fix = keep MASTER WEIGHTS in fp32 and apply updates
     there; use fp16 copies only for the forward/backward math.
Too large an S overflows to inf, so the scaler is DYNAMIC: on inf/nan skip the
step and halve S; after a run of clean steps, double S to probe for headroom.

Scope: numpy on CPU gains nothing from fp16 speed, so this file shows the
numerics and the scaler logic rather than being wired into the training loop.
"""

import numpy as np


class DynamicLossScaler:
    def __init__(self, init_scale=2.0 ** 15, growth_interval=200):
        self.scale, self.growth_interval, self.good_steps = init_scale, growth_interval, 0

    def unscale(self, grads: dict):
        """Divide scaled fp16 grads by the scale in fp32. Returns (grads, finite?)."""
        out = {k: g.astype(np.float32) / self.scale for k, g in grads.items()}
        finite = all(np.isfinite(g).all() for g in out.values())
        return out, finite

    def update(self, finite: bool):
        if not finite:
            self.scale /= 2                       # overflowed: back off, caller skips the step
            self.good_steps = 0
        else:
            self.good_steps += 1
            if self.good_steps % self.growth_interval == 0:
                self.scale *= 2


if __name__ == "__main__":
    tiny = 1e-8                                    # a realistic tiny gradient
    print("fp16(1e-8) =", np.float16(tiny), "-> underflows to zero")
    assert np.float16(tiny) == 0

    S = 2.0 ** 15
    scaled = np.float16(tiny * S)                  # scaled up first, fits in fp16
    recovered = np.float32(scaled) / S             # unscaled in fp32
    print(f"scaled by {S:.0f}: fp16 stores {scaled}, recovered {recovered:.2e}")
    assert scaled != 0 and abs(recovered - tiny) / tiny < 1e-2

    w16, w32 = np.float16(1.0), np.float32(1.0)
    print("fp16: 1.0 + 1e-4 =", w16 + np.float16(1e-4), "| fp32 master:", w32 + np.float32(1e-4))
    assert w16 + np.float16(1e-4) == 1.0 and w32 + np.float32(1e-4) > 1.0

    scaler = DynamicLossScaler(init_scale=65536.0, growth_interval=2)
    with np.errstate(over="ignore"):
        big = {"w": np.array([60000.0], dtype=np.float16) * np.float16(4)}   # overflows fp16 -> inf
    _, ok = scaler.unscale(big)
    scaler.update(ok)
    print("overflow detected, step skipped, scale halved to", scaler.scale)
    assert not ok and scaler.scale == 32768.0
    for _ in range(2):
        scaler.update(True)
    assert scaler.scale == 65536.0                 # grew back after clean steps
    print("scale regrew after clean steps:", scaler.scale)
