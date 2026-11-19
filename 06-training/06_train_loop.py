"""Training loop — everything from stage 06 wired together, on a character-level GPT.

Per step:
    1. sample a random batch of windows from the text (inputs; targets are the
       same window shifted by one inside the loss),
    2. forward + hand-written backward  -> loss, gradients      (01),
    3. clip the global gradient norm                              (03),
    4. set the learning rate from the warmup+cosine schedule      (03),
    5. AdamW update                                               (02).
Finally save a checkpoint, reload it, and confirm the loss is unchanged (05),
then sample some text from the trained model with the stage-05 sampler.

The corpus is tiny on purpose so training finishes in about a minute of pure
numpy on a CPU. The corpus is the same few sentences repeated, so the "val"
slice is text the model has effectively seen: val loss tracking train loss
here does NOT show generalization. Real runs need a genuinely held-out set
and far more data; this loop only proves the training machinery works.
"""

import importlib
import pathlib
import sys
import tempfile
import time

import numpy as np

root = pathlib.Path(__file__).parent.parent
for folder in ("06-training", "05-gpt", "01-tokenization"):
    sys.path.insert(0, str(root / folder))
_bp = importlib.import_module("01_backprop_gpt")
_opt = importlib.import_module("02_optimizers")
_sched = importlib.import_module("03_schedule_clipping")
_ckpt = importlib.import_module("05_checkpoint")
_sampling = importlib.import_module("02_sampling")
_chars = importlib.import_module("01_char_tokenizer")

CORPUS = (
    "the quick brown fox jumps over the lazy dog. "
    "a small model learns to predict the next character from the ones before it. "
    "attention lets every position look back at earlier positions. "
    "the optimizer nudges each weight a little after every batch. "
    "if the loss goes down, the model is learning. "
    "the quick brown fox jumps over the lazy dog again and again. "
) * 6


def get_batch(data, batch_size, seq_len, rng):
    starts = rng.integers(0, len(data) - seq_len, size=batch_size)
    return np.stack([data[s:s + seq_len] for s in starts])


def eval_loss(model, data, seq_len, n=8, seed=123):
    rng = np.random.default_rng(seed)
    return float(np.mean([model.loss_and_grads(get_batch(data, 8, seq_len, rng))[0] for _ in range(n)]))


def generate(model, tok, prompt, n_new, temperature=0.8, top_k=8, seed=0):
    rng = np.random.default_rng(seed)
    ids = tok.encode(prompt)
    for _ in range(n_new):
        ctx = np.array([ids[-model.max_len:]])
        logits = model.forward(ctx)[0][0, -1]
        ids.append(_sampling.sample(logits, temperature=temperature, top_k=top_k, rng=rng))
    return tok.decode(ids)


if __name__ == "__main__":
    tok = _chars.CharTokenizer(CORPUS)
    data = np.array(tok.encode(CORPUS))
    split = int(0.85 * len(data))
    train_data, val_data = data[:split], data[split:]

    seq_len, batch_size, steps = 32, 16, 300
    config = dict(vocab=tok.vocab_size, d_model=48, heads=4, layers=2, max_len=seq_len, seed=0)
    model = _ckpt.build(config)
    opt = _opt.AdamW(model.params, weight_decay=0.01)
    rng = np.random.default_rng(0)

    n_params = sum(w.size for w in model.params.values())
    print(f"vocab {tok.vocab_size}, {n_params:,} params, {len(train_data)} train / {len(val_data)} val chars")
    start_loss = eval_loss(model, val_data, seq_len)
    print(f"untrained val loss {start_loss:.3f} (ln vocab = {np.log(tok.vocab_size):.3f})")

    t0 = time.time()
    for step in range(steps):
        loss, grads = model.loss_and_grads(get_batch(train_data, batch_size, seq_len, rng))
        gnorm = _sched.clip_grad_norm(grads, max_norm=1.0)
        lr = _sched.lr_at(step, max_lr=3e-3, warmup=20, total=steps, min_lr=3e-4)
        opt.step(model.params, grads, lr=lr)
        if step % 50 == 0 or step == steps - 1:
            print(f"step {step:3d}  loss {loss:.3f}  lr {lr:.1e}  grad-norm {gnorm:.2f}")
    print(f"trained in {time.time() - t0:.0f}s")

    train_final, val_final = eval_loss(model, train_data, seq_len), eval_loss(model, val_data, seq_len)
    print(f"final train loss {train_final:.3f} | val loss {val_final:.3f}")
    assert train_final < 0.5 * np.log(tok.vocab_size)        # clearly learned, not near-uniform
    assert val_final < start_loss

    with tempfile.TemporaryDirectory() as d:                 # checkpoint round trip
        path = pathlib.Path(d) / "gpt.npz"
        _ckpt.save_checkpoint(path, model.params, opt, steps, config)
        params, _, _, cfg = _ckpt.load_checkpoint(path)
    reloaded = _ckpt.build(cfg)
    reloaded.params = params
    assert abs(eval_loss(reloaded, val_data, seq_len) - val_final) < 1e-12
    print("checkpoint reloaded: identical loss")

    print("\nsample:", repr(generate(reloaded, tok, "the quick ", 120)))
