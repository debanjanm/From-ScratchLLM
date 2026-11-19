"""BERT-style encoder with masked language modeling (MLM).

Method: GPT predicts the NEXT token and so must not look right. An ENCODER
looks both ways: remove the causal mask and every token attends to every
other token. That makes a richer representation of each token (it sees its
full context) but breaks next-token training, since the answer would be
visible. BERT's trick (Devlin et al., 2018) is a fill-in-the-blank objective:
  1. pick ~15% of positions;
  2. of those: 80% -> [MASK], 10% -> a random token, 10% -> left unchanged
     (so the model can't assume "non-[MASK] means correct" and must examine
     every token);
  3. predict the ORIGINAL token at those positions only; loss ignores the rest.
Same weight-tied head, cross-entropy and hand-written backward as stage 06;
only the mask (none) and the loss mask differ from GPT.

Simplifications vs the original BERT, so this stays small and reuses stage 06:
pre-norm RMSNorm + SwiGLU blocks instead of post-norm LayerNorm + GELU, no
segment embeddings, no next-sentence prediction (later work, e.g. RoBERTa,
found it unnecessary).

Runtime: the demo trains 2000 steps in about 1.5 minutes of pure numpy. MLM learns
slower than GPT (few supervised positions), so 500 steps was not enough.
The corpus is repeated text, so accuracy here reflects memorization.

Pros: bidirectional context gives strong embeddings for classification,
tagging, retrieval. Cons: cannot generate text left to right, and only ~15% of
positions provide a learning signal per sequence (GPT learns from all).
"""

import importlib
import pathlib
import sys
import time

import numpy as np

root = pathlib.Path(__file__).parent.parent
for folder in ("07-architectures", "06-training", "01-tokenization"):
    sys.path.insert(0, str(root / folder))
_ga = importlib.import_module("01_general_attention")
_opt = importlib.import_module("02_optimizers")
_sched = importlib.import_module("03_schedule_clipping")
_chars = importlib.import_module("01_char_tokenizer")
softmax, rms_fwd, rms_bwd = _ga.softmax, _ga.rms_fwd, _ga.rms_bwd


class BERT:
    def __init__(self, vocab, d_model, heads, layers, max_len, seed=0, std=0.02):
        self.H, self.L, self.max_len = heads, layers, max_len
        rng = np.random.default_rng(seed)
        self.params = {"tok": rng.normal(0, std, size=(vocab, d_model)),
                       "pos": rng.normal(0, std, size=(max_len, d_model)),
                       "nf": np.ones(d_model)}
        for l in range(layers):
            _ga.init_block(self.params, f"l{l}", d_model, rng, std)

    def forward(self, ids):
        p, T = self.params, ids.shape[1]
        x = p["tok"][ids] + p["pos"][:T]
        caches = []
        for l in range(self.L):
            x, c = _ga.block_fwd(x, p, f"l{l}", self.H, mask=None)    # no mask: bidirectional
            caches.append(c)
        xf, cn = rms_fwd(x, p["nf"])
        return xf @ p["tok"].T, (caches, cn, xf)

    def loss_and_grads(self, inputs, targets, loss_mask):
        """Cross-entropy on the masked positions only (loss_mask: bool, same shape as ids)."""
        p, g = self.params, {}
        logits, (caches, cn, xf) = self.forward(inputs)
        probs = softmax(logits)
        n = loss_mask.sum()
        tgt = targets[..., None]
        picked = np.take_along_axis(probs, tgt, axis=-1)[..., 0]
        loss = float(-(np.log(picked) * loss_mask).sum() / n)

        dl = probs.copy()
        np.put_along_axis(dl, tgt, np.take_along_axis(dl, tgt, axis=-1) - 1, axis=-1)
        dl *= loss_mask[..., None] / n                                  # unmasked positions get 0 gradient

        D = xf.shape[-1]
        g["tok"] = dl.reshape(-1, dl.shape[-1]).T @ xf.reshape(-1, D)
        dx, g["nf"] = rms_bwd(dl @ p["tok"], cn)
        for l in reversed(range(self.L)):
            dx = _ga.block_bwd(dx, caches[l], p, f"l{l}", self.H, g)
        np.add.at(g["tok"], inputs, dx)
        g["pos"] = np.zeros_like(p["pos"])
        g["pos"][:inputs.shape[1]] = dx.sum(axis=0)
        return loss, g


def mask_tokens(ids, mask_id, n_real, rng, p=0.15):
    """BERT corruption. Returns (inputs, targets, loss_mask)."""
    sel = rng.random(ids.shape) < p
    for b in np.where(~sel.any(axis=1))[0]:                              # every row needs >= 1 target
        sel[b, rng.integers(ids.shape[1])] = True
    r = rng.random(ids.shape)
    inputs = ids.copy()
    inputs[sel & (r < 0.8)] = mask_id                                    # 80% -> [MASK]
    rand = sel & (r >= 0.8) & (r < 0.9)                                  # 10% -> random token
    inputs[rand] = rng.integers(0, n_real, size=rand.sum())
    return inputs, ids, sel                                              # last 10%: unchanged


CORPUS = (
    "the quick brown fox jumps over the lazy dog. "
    "a small model learns to predict the next character from the ones before it. "
    "attention lets every position look back at earlier positions. "
    "the optimizer nudges each weight a little after every batch. "
    "if the loss goes down, the model is learning. "
    "the quick brown fox jumps over the lazy dog again and again. "
) * 6

if __name__ == "__main__":
    tok = _chars.CharTokenizer(CORPUS)
    MASK = tok.vocab_size                                                # extra id after the characters
    V = tok.vocab_size + 1
    data = np.array(tok.encode(CORPUS))
    seq_len = 32
    rng = np.random.default_rng(0)
    batch = lambda n: np.stack([data[s:s + seq_len] for s in rng.integers(0, len(data) - seq_len, size=n)])

    # --- gradient check (small model, larger init so gradients are measurable) ---
    small = BERT(V, 8, 2, 2, 8, std=0.3)
    ids = rng.integers(0, tok.vocab_size, size=(2, 6))
    inp, tgt, sel = mask_tokens(ids, MASK, tok.vocab_size, rng, p=0.4)
    _, grads = small.loss_and_grads(inp, tgt, sel)
    assert set(grads) == set(small.params)
    worst, h = 0.0, 1e-5
    for name, w in small.params.items():
        for _ in range(3):
            idx = tuple(rng.integers(0, s) for s in w.shape)
            old = w[idx]
            w[idx] = old + h; up = small.loss_and_grads(inp, tgt, sel)[0]
            w[idx] = old - h; down = small.loss_and_grads(inp, tgt, sel)[0]
            w[idx] = old
            num = (up - down) / (2 * h)
            worst = max(worst, abs(num - grads[name][idx]) / max(1e-8, abs(num) + abs(grads[name][idx])))
    print(f"gradient check: max relative error {worst:.2e}")
    assert worst < 1e-5

    # --- bidirectional: a FUTURE token changes earlier outputs (the opposite of GPT) ---
    x1 = rng.integers(0, tok.vocab_size, size=(1, 8))
    x2 = x1.copy(); x2[0, -1] = (x2[0, -1] + 1) % tok.vocab_size
    assert not np.allclose(small.forward(x1)[0][:, :-1], small.forward(x2)[0][:, :-1])
    print("changing the last token changes earlier positions' outputs: True (bidirectional)")

    # --- train ---
    model = BERT(V, 48, 4, 2, seq_len)
    opt = _opt.AdamW(model.params, weight_decay=0.01)
    steps, t0 = 2000, time.time()
    for step in range(steps):
        inp, tgt, sel = mask_tokens(batch(16), MASK, tok.vocab_size, rng)
        loss, grads = model.loss_and_grads(inp, tgt, sel)
        _sched.clip_grad_norm(grads, 1.0)
        opt.step(model.params, grads, lr=_sched.lr_at(step, 3e-3, 20, steps, 3e-4))
        if step % 250 == 0 or step == steps - 1:
            print(f"step {step:3d}  masked-LM loss {loss:.3f}")
    print(f"trained in {time.time() - t0:.0f}s")

    # --- evaluate: masked-character accuracy on fresh random maskings ---
    inp, tgt, sel = mask_tokens(batch(64), MASK, tok.vocab_size, np.random.default_rng(99))
    pred = model.forward(inp)[0].argmax(-1)
    acc = float((pred == tgt)[sel].mean())
    print(f"masked-token accuracy {acc:.2f} (chance ~ {1 / tok.vocab_size:.2f})")
    assert acc > 0.6                                                     # corpus is repeated text: memorization, not generalization

    # --- fill in the blank ---
    text = CORPUS[:seq_len]
    i = text.index("fox") + 1                                            # the 'o'
    blanked = np.array([[MASK if j == i else tok.stoi[c] for j, c in enumerate(text)]])
    guess = tok.itos[int(model.forward(blanked)[0][0, i].argmax())]
    shown = text[:i] + "[MASK]" + text[i + 1:]
    print(f"\n{shown!r}  ->  predicted {guess!r} (true {text[i]!r})")
