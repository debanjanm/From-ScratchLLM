"""DPO — Direct Preference Optimization: align a model to preferences with no reward model.

Setting: for a prompt x you have a CHOSEN response y_w and a REJECTED one y_l.
RLHF (04) would fit a reward model, then run reinforcement learning against it.
DPO (Rafailov et al., 2023) shows the same objective can be optimized directly,
with a classification-style loss on the policy itself. Let
    logpi(y|x) = sum of the token log-probs of the response,
    d(y) = logpi(y|x) - logpi_ref(y|x)     (how far the policy moved from a frozen reference)
    z = beta * ( d(y_w) - d(y_l) )
    loss = -log sigmoid(z)
Minimizing it raises the chosen response's probability RELATIVE to the rejected
one and relative to the reference. beta controls the leash: larger beta keeps the
policy nearer the reference (it is the implicit KL penalty). At the start the
policy equals the reference, so z = 0 and loss = ln 2.

Gradient (no autograd needed): with s = sigmoid(-z),
    grad = -beta * s * ( grad logpi(y_w) - grad logpi(y_l) ).
That is a SIGNED, weighted sum of sequence log-prob gradients, which stage 06's
backward pass already computes when given per-token weights +beta*s/B on chosen
tokens and -beta*s/B on rejected tokens (normalize=False).

Demo: first train a policy on BOTH response styles for the same prompts
("ok x" and "no x", 50/50), so it has no preference. Then DPO on pairs
(chosen "ok x", rejected "no x") using only prompts with letters a-d, and test
on held-out letters e-f: the learned preference should transfer. Note DPO
typically lowers the log-prob of BOTH responses, just the rejected one more.

Honest caveats from the recorded run (single seed): the preference learned and
transferred (held-out pairs ranked right 100%), but greedy decoding on held-out
letters produced the exact chosen response only ~50% of the time, because
copying the letter broke for some prompts. DPO optimizes a RELATIVE margin, so
it can move probability mass in unintended ways; real systems monitor this and
often add an SFT term on the chosen responses.
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
_sft = importlib.import_module("02_sft")                  # encode_example, SEQ_LEN
_chars = importlib.import_module("01_char_tokenizer")
SEQ_LEN, encode_example = _sft.SEQ_LEN, _sft.encode_example


def seq_logp(model, ids, mask):
    """Sum of target log-probs over the masked (response) positions -> (B,)."""
    logits = model.forward(ids)[0][:, :-1]
    m = logits.max(axis=-1, keepdims=True)
    logp = logits - m - np.log(np.exp(logits - m).sum(axis=-1, keepdims=True))
    tok_lp = np.take_along_axis(logp, ids[:, 1:, None], axis=-1)[..., 0]
    return (tok_lp * mask).sum(axis=1)


def dpo_loss_and_grads(policy, batch, ref_w, ref_l, beta):
    """batch = (ids_w, mask_w, ids_l, mask_l). ref_* are the frozen reference's sequence log-probs."""
    ids_w, mask_w, ids_l, mask_l = batch
    B = ids_w.shape[0]
    pw, pl = seq_logp(policy, ids_w, mask_w), seq_logp(policy, ids_l, mask_l)
    z = beta * ((pw - ref_w) - (pl - ref_l))
    loss = float(np.mean(np.logaddexp(0.0, -z)))              # -log sigmoid(z), stable
    coef = beta / (1.0 + np.exp(z)) / B                       # beta * sigmoid(-z) / B
    ids = np.concatenate([ids_w, ids_l])
    weights = np.concatenate([mask_w * coef[:, None], -mask_l * coef[:, None]])
    _, grads = policy.loss_and_grads(ids, weights, normalize=False)
    return loss, grads, z


def make_pairs(tok, pad, letters, n, rng):
    rows = []
    for c in rng.choice(letters, size=n):
        prompt = f"user: pick {c}\nbot: "
        rows.append((encode_example(tok, pad, prompt, f"ok {c}\n"), encode_example(tok, pad, prompt, f"no {c}\n")))
    return (np.stack([w[0] for w, _ in rows]), np.stack([w[1] for w, _ in rows]),
            np.stack([l[0] for _, l in rows]), np.stack([l[1] for _, l in rows]))


def p_ok(model, tok, letters):
    """Mean probability that the FIRST response character is 'o' (the 'ok' style) for each letter."""
    out = []
    for c in letters:
        ids = np.array([tok.encode(f"user: pick {c}\nbot: ")])
        logits = model.forward(ids)[0][0, -1]
        e = np.exp(logits - logits.max())
        out.append(e[tok.stoi["o"]] / e.sum())
    return float(np.mean(out))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    tok = _chars.CharTokenizer("user: pick abcdef\nbot: ok no")
    PAD, V = tok.vocab_size, tok.vocab_size + 1
    TRAIN, HELD = list("abcd"), list("ef")

    # --- 1. gradient check of the DPO loss (random reference so z != 0) ---
    small = _bp.TrainableGPT(V, 8, 2, 2, SEQ_LEN, std=0.3)
    ref_small = _bp.TrainableGPT(V, 8, 2, 2, SEQ_LEN, seed=5, std=0.3)
    batch = make_pairs(tok, PAD, TRAIN, 3, rng)
    ref_w, ref_l = seq_logp(ref_small, batch[0], batch[1]), seq_logp(ref_small, batch[2], batch[3])
    loss, grads, z = dpo_loss_and_grads(small, batch, ref_w, ref_l, beta=0.5)
    worst, h = 0.0, 1e-5
    for name, w in small.params.items():
        for _ in range(2):
            idx = tuple(rng.integers(0, s) for s in w.shape)
            old = w[idx]
            w[idx] = old + h; up = dpo_loss_and_grads(small, batch, ref_w, ref_l, 0.5)[0]
            w[idx] = old - h; down = dpo_loss_and_grads(small, batch, ref_w, ref_l, 0.5)[0]
            w[idx] = old
            num = (up - down) / (2 * h)
            worst = max(worst, abs(num - grads[name][idx]) / max(1e-8, abs(num) + abs(grads[name][idx])))
    print(f"DPO gradient check: max relative error {worst:.2e}")
    assert worst < 1e-5

    # --- 2. the starting policy: trained on BOTH styles, so no preference ---
    def sft_batch(n, rng_):
        rows = []
        for c in rng_.choice(list("abcdef"), size=n):
            style = "ok" if rng_.random() < 0.5 else "no"
            rows.append(encode_example(tok, PAD, f"user: pick {c}\nbot: ", f"{style} {c}\n"))
        return np.stack([r[0] for r in rows]), np.stack([r[1] for r in rows])

    policy = _bp.TrainableGPT(V, 48, 4, 2, SEQ_LEN, seed=0)
    opt = _opt.AdamW(policy.params, weight_decay=0.01)
    steps = 600
    for step in range(steps):
        ids, mask = sft_batch(32, rng)
        _, g = policy.loss_and_grads(ids, mask)
        _sched.clip_grad_norm(g, 1.0)
        opt.step(policy.params, g, lr=_sched.lr_at(step, 3e-3, 30, steps, 3e-4))
    p0_train, p0_held = p_ok(policy, tok, TRAIN), p_ok(policy, tok, HELD)
    print(f"\nSFT policy, P(first token 'o' i.e. 'ok' style): train letters {p0_train:.2f}, held-out {p0_held:.2f}")
    assert 0.3 < p0_train < 0.7 and 0.3 < p0_held < 0.7

    frozen = {k: v.copy() for k, v in policy.params.items()}

    # --- 3. DPO with two leash strengths ---
    results = {}
    for beta in (0.1, 0.5):
        model = _bp.TrainableGPT(V, 48, 4, 2, SEQ_LEN, seed=0)
        model.params = {k: v.copy() for k, v in frozen.items()}
        ref = _bp.TrainableGPT(V, 48, 4, 2, SEQ_LEN, seed=0)
        ref.params = {k: v.copy() for k, v in frozen.items()}       # frozen reference, never updated
        o = _opt.AdamW(model.params)
        t0, margins = time.time(), []
        for step in range(150):
            batch = make_pairs(tok, PAD, TRAIN, 32, rng)
            ref_w, ref_l = seq_logp(ref, batch[0], batch[1]), seq_logp(ref, batch[2], batch[3])
            loss, g, z = dpo_loss_and_grads(model, batch, ref_w, ref_l, beta)
            if step == 0:
                assert abs(loss - np.log(2)) < 1e-9 and np.allclose(z, 0)   # policy == reference at the start
            _sched.clip_grad_norm(g, 1.0)
            o.step(model.params, g, lr=5e-4)
            margins.append((z > 0).mean())

        test = make_pairs(tok, PAD, HELD, 64, np.random.default_rng(9))        # held-out letters
        d_w = seq_logp(model, test[0], test[1]) - seq_logp(ref, test[0], test[1])
        d_l = seq_logp(model, test[2], test[3]) - seq_logp(ref, test[2], test[3])
        results[beta] = dict(model=model, p_train=p_ok(model, tok, TRAIN), p_held=p_ok(model, tok, HELD),
                             pref_acc=float(((d_w - d_l) > 0).mean()), d_w=float(d_w.mean()), d_l=float(d_l.mean()))
        r = results[beta]
        print(f"beta={beta}: P('ok' style) train {r['p_train']:.2f}, held-out {r['p_held']:.2f} | "
              f"held-out pairs ranked right {r['pref_acc']:.2f} | "
              f"logpi shift chosen {r['d_w']:+.2f}, rejected {r['d_l']:+.2f} ({time.time() - t0:.0f}s)")

    # --- 4. what must hold ---
    r = results[0.1]
    assert r["p_held"] > p0_held + 0.3                    # preference learned AND transfers to unseen letters
    assert r["pref_acc"] > 0.9
    assert r["d_w"] - r["d_l"] > 0                         # chosen gained relative to rejected
    # greedy decoding: does the model produce the exact chosen response ("ok x\n") on held-out letters?
    def exact_chosen(model, letters):
        hits = []
        for c in letters:
            ids = tok.encode(f"user: pick {c}\nbot: ")
            n0 = len(ids)
            for _ in range(5):
                ids.append(int(model.forward(np.array([ids]))[0][0, -1].argmax()))
            hits.append(tok.decode(ids[n0:]).split("\n")[0] == f"ok {c}")
        return float(np.mean(hits)), tok.decode(ids[n0:])

    print("\ngreedy held-out response (exactly 'ok x'?):")
    print(f"  SFT policy : {exact_chosen(policy, HELD)[0]:.2f}")
    for beta, r in results.items():
        acc, last = exact_chosen(r["model"], HELD)
        print(f"  DPO beta={beta}: {acc:.2f}   e.g. 'pick f' -> {last!r}")
    print("Both betas rank held-out pairs correctly, yet greedy output is exact only half the time: the "
          "preference transferred but copying the letter broke on some held-out prompts. beta=0.1 moved "
          f"much further from the reference (rejected logpi {results[0.1]['d_l']:+.1f} vs {results[0.5]['d_l']:+.1f}); "
          "this run does not show beta=0.5 protecting that behaviour.")
