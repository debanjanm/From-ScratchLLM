"""LoRA — fine-tune a frozen model by learning a low-rank update to its weights.

Method: full fine-tuning updates every weight W (d_in x d_out) and needs
gradients plus Adam state for all of them. LoRA (Hu et al., 2021) freezes W
and learns only a low-rank correction:
    W_eff = W + (alpha / r) * A @ B        A: (d_in, r),  B: (r, d_out),  r << d
Why it works: the weight change needed to adapt to a new task seems to have
low "intrinsic rank", so r(d_in + d_out) numbers can stand in for d_in*d_out.
B starts at ZERO, so at step 0 the adapter changes nothing: the model begins
exactly as the pretrained one. (A random, B zero: if both were zero, neither
could receive gradient.)

Backward pass: nothing new is needed beyond stage 06. Build W_eff, run the
existing forward/backward to get dW_eff, then the chain rule gives
    dA = s * dW @ B^T          dB = s * A^T @ dW           (s = alpha / r)
and the frozen W simply receives no update.

After training the adapter can be MERGED (W <- W + s*A@B) for inference with
zero extra latency, or kept separate: one frozen base plus many tiny adapters,
one per task. This is why LoRA dominates open-model fine-tuning.

Demo caveat: domain B is a few repeated sentences, so both methods end up
memorizing it; the comparison shows LoRA reaching nearly full fine-tuning's
loss with a fraction of the trainable parameters, not that it generalizes.

Pros: far fewer trainable parameters and optimizer state; the base is never
modified (no catastrophic overwrite); adapters are tiny files.
Cons: less capacity than full fine-tuning on big shifts; extra hyperparameters
(r, alpha, which matrices). The parameter saving is modest on tiny models:
r(d_in + d_out) vs d_in*d_out only shrinks a lot when d >> r.
"""

import importlib
import pathlib
import sys
import time

import numpy as np

root = pathlib.Path(__file__).parent.parent
for folder in ("08-post-training", "06-training", "01-tokenization"):
    sys.path.insert(0, str(root / folder))
_bp = importlib.import_module("01_backprop_gpt")
_opt = importlib.import_module("02_optimizers")
_sched = importlib.import_module("03_schedule_clipping")
_loop = importlib.import_module("06_train_loop")        # CORPUS, get_batch, eval_loss
_chars = importlib.import_module("01_char_tokenizer")


class LoRA:
    """Low-rank adapters over a frozen TrainableGPT. Exposes the same loss_and_grads(ids) API,
    but its gradients are for the adapter parameters only."""

    LINEARS = ("Wq", "Wk", "Wv", "Wo", "W1", "W2", "W3")

    def __init__(self, base, targets=LINEARS, rank=8, alpha=16, seed=0, b_std=0.0):
        self.base, self.rank, self.scale = base, rank, alpha / rank
        rng = np.random.default_rng(seed)
        self.names = [k for k in base.params if k.split(".")[-1] in targets]
        self.params = {}
        for k in self.names:
            d_in, d_out = base.params[k].shape
            self.params[f"{k}.A"] = rng.normal(0, 1 / np.sqrt(d_in), size=(d_in, rank))
            self.params[f"{k}.B"] = rng.normal(0, b_std, size=(rank, d_out)) if b_std else np.zeros((rank, d_out))

    def delta(self, k):
        return self.scale * self.params[f"{k}.A"] @ self.params[f"{k}.B"]

    def effective(self):
        eff = dict(self.base.params)                         # shares arrays; only adapted ones replaced
        for k in self.names:
            eff[k] = self.base.params[k] + self.delta(k)
        return eff

    def _run(self, fn):
        orig, self.base.params = self.base.params, self.effective()
        try:
            return fn()
        finally:
            self.base.params = orig

    def forward(self, ids):
        return self._run(lambda: self.base.forward(ids)[0])

    def loss_and_grads(self, ids, loss_mask=None):
        loss, g = self._run(lambda: self.base.loss_and_grads(ids, loss_mask))
        grads = {}
        for k in self.names:
            A, B = self.params[f"{k}.A"], self.params[f"{k}.B"]
            grads[f"{k}.A"] = self.scale * g[k] @ B.T         # chain rule through W + s*A@B
            grads[f"{k}.B"] = self.scale * A.T @ g[k]
        return loss, grads

    def merged_params(self):
        """Fold the adapters into plain weights: same function, no adapter overhead at inference."""
        return {k: (v + self.delta(k) if k in self.names else v.copy()) for k, v in self.base.params.items()}


def n_params(d):
    return sum(v.size for v in d.values())


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # --- 1. gradient check of dA / dB (B must be non-zero or dA is trivially 0) ---
    small = _bp.TrainableGPT(11, 8, 2, 2, 8, std=0.3)
    lora = LoRA(small, rank=2, alpha=4, b_std=0.3)
    ids = rng.integers(0, 11, size=(2, 6))
    _, grads = lora.loss_and_grads(ids)
    worst, h = 0.0, 1e-5
    for name, w in lora.params.items():
        for _ in range(2):
            idx = tuple(rng.integers(0, s) for s in w.shape)
            old = w[idx]
            w[idx] = old + h; up = lora.loss_and_grads(ids)[0]
            w[idx] = old - h; down = lora.loss_and_grads(ids)[0]
            w[idx] = old
            num = (up - down) / (2 * h)
            worst = max(worst, abs(num - grads[name][idx]) / max(1e-8, abs(num) + abs(grads[name][idx])))
    print(f"adapter gradient check: max relative error {worst:.2e}")
    assert worst < 1e-5

    # --- 2. pretrain a base model on text A, then adapt to a different text B ---
    text_a = _loop.CORPUS
    text_b = ("the lazy dog sleeps in the warm sun. a happy cat naps on the soft mat. "
              "the old man walks down the long road. a red bird sings in the tall tree. ") * 8
    assert set(text_b) <= set(text_a), "domain B must only use characters the base has seen"
    tok = _chars.CharTokenizer(text_a + text_b)
    seq_len = 32
    data_a, data_b = np.array(tok.encode(text_a)), np.array(tok.encode(text_b))

    base = _bp.TrainableGPT(tok.vocab_size, 64, 4, 2, seq_len, seed=0)
    opt = _opt.AdamW(base.params, weight_decay=0.01)
    for step in range(400):
        _, g = base.loss_and_grads(_loop.get_batch(data_a, 16, seq_len, rng))
        _sched.clip_grad_norm(g, 1.0)
        opt.step(base.params, g, lr=_sched.lr_at(step, 3e-3, 20, 400, 3e-4))
    loss_a = _loop.eval_loss(base, data_a, seq_len)
    loss_b0 = _loop.eval_loss(base, data_b, seq_len)
    print(f"\npretrained on A: loss on A {loss_a:.3f} | loss on unseen domain B {loss_b0:.3f}")

    frozen = {k: v.copy() for k, v in base.params.items()}
    full = _bp.TrainableGPT(tok.vocab_size, 64, 4, 2, seq_len, seed=0)       # full fine-tune copy
    full.params = {k: v.copy() for k, v in frozen.items()}
    lora = LoRA(base, rank=8, alpha=16)

    # zero-init property: before any training the adapter changes nothing
    assert abs(_loop.eval_loss(lora, data_b, seq_len) - loss_b0) < 1e-12
    print("step 0: LoRA model == base model (B is zero): True")

    # --- 3. adapt on B: LoRA (adapters only) vs full fine-tuning, same steps ---
    steps = 300
    runs = {"LoRA": (lora, lora.params, 1e-2), "full fine-tune": (full, full.params, 3e-3)}
    rng_b = np.random.default_rng(1)
    batches = [_loop.get_batch(data_b, 16, seq_len, rng_b) for _ in range(steps)]
    for name, (model, params, peak) in runs.items():
        o, t0 = _opt.AdamW(params), time.time()
        for step, b in enumerate(batches):
            _, g = model.loss_and_grads(b)
            _sched.clip_grad_norm(g, 1.0)
            o.step(params, g, lr=_sched.lr_at(step, peak, 20, steps, peak / 10))
        print(f"{name:15s} trained {n_params(params):6,d} params in {time.time() - t0:.0f}s")

    loss_lora = _loop.eval_loss(lora, data_b, seq_len)
    loss_full = _loop.eval_loss(full, data_b, seq_len)
    print(f"\nloss on B: base {loss_b0:.3f} -> LoRA {loss_lora:.3f} | full fine-tune {loss_full:.3f}")
    print(f"LoRA trained {n_params(lora.params) / n_params(frozen):.0%} as many parameters as full fine-tuning "
          f"(adapters on {len(lora.names)} matrices, rank {lora.rank})")

    # --- 4. the properties that make LoRA useful ---
    assert all(np.array_equal(base.params[k], frozen[k]) for k in frozen)    # base never modified
    assert loss_lora < loss_b0 - 0.5                                          # it genuinely adapted
    ids = data_b[:seq_len][None]
    merged = _bp.TrainableGPT(tok.vocab_size, 64, 4, 2, seq_len)
    merged.params = lora.merged_params()
    assert np.allclose(merged.forward(ids)[0], lora.forward(ids))             # merge == adapter, exactly
    ranks = [np.linalg.matrix_rank(lora.delta(k), tol=1e-8) for k in lora.names]
    assert max(ranks) <= lora.rank
    print(f"merged weights reproduce adapter outputs; every weight update has rank <= {max(ranks)} (r = {lora.rank})")
    print("base weights bit-identical to before adaptation: True")
