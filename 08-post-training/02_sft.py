"""SFT — supervised fine-tuning: teach a language model to follow instructions.

Method: a pretrained LM only continues text. SFT turns it into an assistant by
continuing to train with the SAME next-token cross-entropy, but on curated
(prompt, response) examples formatted with a fixed template, e.g.
    user: reverse abc\\nbot: cba\\n
and with ONE crucial change: the loss counts only the RESPONSE tokens.
The prompt is context the model is given, not text it should learn to produce;
including it would waste gradient on predicting arbitrary user input (here:
random words, pure noise) and teach the model to imitate users, not answer them.

Implementation: stage 06's loss takes an optional loss_mask over target
positions (1 = counts, 0 = ignored); the mask's gradient is just the usual
cross-entropy gradient multiplied by that 0/1 weight, so the same hand-written
backward pass is reused untouched. Padding (to make a batch rectangular) is
masked out the same way.

Demo: pretrain on plain text, which cannot follow instructions. Then SFT on
random "copy"/"reverse" prompts. Scoring uses FRESH random words the model has
never seen, so the accuracy is a real generalization check (unlike the repeated
corpora of earlier stages). We also compare training with and without the
response-only mask. In the run recorded while building this, response-only
reached 100% on both tasks while the whole-sequence loss reached 100% on
"copy" but only ~66% on "reverse". That is ONE seed, so treat the gap as
suggestive rather than proven; nothing asserts it. Runtime is about 5 minutes
(two 2500-step SFT runs in pure numpy).

Real SFT adds: many diverse instructions, chat-template special tokens
(see 01-tokenization/11), multi-turn data, and usually LoRA (01_lora.py) to
cut cost; LoRA's loss_and_grads accepts the same loss_mask.
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
_loop = importlib.import_module("06_train_loop")          # CORPUS, get_batch
_chars = importlib.import_module("01_char_tokenizer")
_lora = importlib.import_module("01_lora")

TASKS = {"copy": lambda w: w, "reverse": lambda w: w[::-1]}
LETTERS = list("abcdefghij")
SEQ_LEN = 36


def make_example(rng):
    task = rng.choice(list(TASKS))
    word = "".join(rng.choice(LETTERS, size=rng.integers(3, 7)))
    return task, word, f"user: {task} {word}\nbot: ", TASKS[task](word) + "\n"


def encode_example(tok, pad_id, prompt, response, response_only=True):
    """-> (ids padded to SEQ_LEN, loss_mask over targets ids[1:])."""
    p_ids, r_ids = tok.encode(prompt), tok.encode(response)
    ids = p_ids + r_ids
    mask = np.zeros(SEQ_LEN - 1)
    start = len(p_ids) if response_only else 1               # first target position that counts
    mask[start - 1:len(ids) - 1] = 1                          # target t is ids[t]; mask index is t-1
    return np.array(ids + [pad_id] * (SEQ_LEN - len(ids))), mask


def sft_batch(tok, pad_id, n, rng, response_only=True):
    ex = [encode_example(tok, pad_id, *make_example(rng)[2:], response_only) for _ in range(n)]
    return np.stack([e[0] for e in ex]), np.stack([e[1] for e in ex])


def respond(model, tok, prompt, max_new=8):
    ids = tok.encode(prompt)
    out = []
    for _ in range(max_new):
        nxt = int(model.forward(np.array([ids]))[0][0, -1].argmax())
        if nxt >= tok.vocab_size:                              # PAD / unknown: stop
            break
        ids.append(nxt)
        out.append(tok.itos[nxt])
        if out[-1] == "\n":
            break
    return "".join(out)


def accuracy(model, tok, n=100, seed=777):
    rng = np.random.default_rng(seed)
    hits = {t: [] for t in TASKS}
    for _ in range(n):
        task, word, prompt, resp = make_example(rng)
        hits[task].append(respond(model, tok, prompt) == resp)
    return {t: float(np.mean(v)) for t, v in hits.items() if v}


def train_sft(model, tok, pad_id, steps, response_only, seed):
    rng, opt = np.random.default_rng(seed), _opt.AdamW(model.params, weight_decay=0.01)
    for step in range(steps):
        ids, mask = sft_batch(tok, pad_id, 32, rng, response_only)
        _, g = model.loss_and_grads(ids, mask)
        _sched.clip_grad_norm(g, 1.0)
        opt.step(model.params, g, lr=_sched.lr_at(step, 3e-3, 50, steps, 3e-4))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    text_a = _loop.CORPUS
    tok = _chars.CharTokenizer(text_a + "user: copy reverse abcdefghij\nbot: ")
    PAD = tok.vocab_size
    V = tok.vocab_size + 1

    # --- 1. mask mechanics: loss_mask picks exactly which targets are averaged, gradients still correct ---
    small = _bp.TrainableGPT(V, 8, 2, 2, 8, std=0.3)
    ids = rng.integers(0, V, size=(2, 6))
    mask = np.array([[0, 0, 1, 1, 1], [0, 1, 1, 0, 0]], dtype=float)
    loss_m, grads = small.loss_and_grads(ids, mask)
    per_tok = small.loss_and_grads(ids)[0]                                      # unmasked mean, for contrast
    logits = small.forward(ids)[0][:, :-1]
    logp = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
    nll = -np.take_along_axis(logp, ids[:, 1:, None], -1)[..., 0]
    assert np.isclose(loss_m, (nll * mask).sum() / mask.sum())                  # exactly the masked average
    assert not np.isclose(loss_m, per_tok)
    worst, h = 0.0, 1e-5
    for name, w in small.params.items():
        for _ in range(2):
            idx = tuple(rng.integers(0, s) for s in w.shape)
            old = w[idx]
            w[idx] = old + h; up = small.loss_and_grads(ids, mask)[0]
            w[idx] = old - h; down = small.loss_and_grads(ids, mask)[0]
            w[idx] = old
            num = (up - down) / (2 * h)
            worst = max(worst, abs(num - grads[name][idx]) / max(1e-8, abs(num) + abs(grads[name][idx])))
    print(f"masked-loss gradient check: max relative error {worst:.2e}")
    assert worst < 1e-5
    assert np.isclose(_lora.LoRA(small, rank=2).loss_and_grads(ids, mask)[0], loss_m)   # LoRA passes the mask through

    # --- 2. what the training examples look like ---
    p_ids, m = encode_example(tok, PAD, "user: reverse abc\nbot: ", "cba\n")
    shown = "".join(tok.itos[i] if i < tok.vocab_size else "·" for i in p_ids)
    print("\nexample :", repr(shown))
    print("counted :", "".join("#" if x else "-" for x in [0] + list(m)), "(# = response token, in the loss)")

    # --- 3. pretrain on plain text: it can continue text, but not follow instructions ---
    base = _bp.TrainableGPT(V, 64, 4, 2, SEQ_LEN, seed=0)
    data_a = np.array(tok.encode(text_a))
    opt = _opt.AdamW(base.params, weight_decay=0.01)
    for step in range(300):
        _, g = base.loss_and_grads(_loop.get_batch(data_a, 16, SEQ_LEN, rng))
        _sched.clip_grad_norm(g, 1.0)
        opt.step(base.params, g, lr=_sched.lr_at(step, 3e-3, 20, 300, 3e-4))
    acc0 = accuracy(base, tok)
    print(f"\nbefore SFT: {acc0}")
    print("  'user: reverse abc\\nbot: ' ->", repr(respond(base, tok, "user: reverse abc\nbot: ")))

    # --- 4. SFT: response-only loss vs loss on everything ---
    steps = 2500
    results = {}
    for name, resp_only in (("response-only loss", True), ("loss on whole sequence", False)):
        model = _bp.TrainableGPT(V, 64, 4, 2, SEQ_LEN, seed=0)
        model.params = {k: v.copy() for k, v in base.params.items()}
        t0 = time.time()
        train_sft(model, tok, PAD, steps, resp_only, seed=1)
        results[name] = (model, accuracy(model, tok))
        print(f"after SFT, {name:24s}: {results[name][1]}  ({time.time() - t0:.0f}s)")

    model, acc = results["response-only loss"]
    for prompt in ("user: reverse feca\nbot: ", "user: copy jjadb\nbot: "):
        print(f"  {prompt!r} -> {respond(model, tok, prompt)!r}")
    assert all(a < 0.2 for a in acc0.values())                  # base cannot follow instructions
    assert all(a > 0.9 for a in acc.values())                   # SFT model does, on unseen words
