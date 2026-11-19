"""RLHF (simplified) — learn a reward model from preferences, then optimize the policy against it.

The classic pipeline, after SFT (02), in two steps:

1. REWARD MODEL. Take the SFT network, add a scalar head w_r read from the
   hidden state at the LAST token of a (prompt, response), so r(x, y) = h_last . w_r.
   Train it on preference pairs with the Bradley-Terry loss
       loss = -log sigmoid( r(x, y_chosen) - r(x, y_rejected) ),
   i.e. the probability that chosen beats rejected is sigmoid of the reward gap.
   Backprop starts at the hidden state, which is why stage 06's backward gained
   an entry point there (dxf) besides the vocab logits.

2. RL AGAINST THE REWARD (PPO-style). Sample responses from the policy, score
   them with the (now frozen) reward model, and push probability toward
   high-reward responses, with two safeguards:
     - KL PENALTY: reward' = r - beta * (log pi(y|x) - log pi_ref(y|x)). Without it
       the policy drifts into whatever the imperfect reward model over-scores
       ("reward hacking"). The optimum of reward' is  pi*(y) ∝ pi_ref(y) * exp(r(y)/beta),
       which this demo checks numerically.
     - CLIPPED UPDATES (PPO): with ratio rho = pi_new/pi_old and advantage A,
       maximize min(rho*A, clip(rho, 1-eps, 1+eps)*A). Several update epochs reuse one
       batch of samples, and clipping stops any single batch from moving the policy too far.
   Advantages use a leave-one-out baseline (RLOO): K samples per prompt, each one's
   reward minus the mean of the other K-1. No value network is needed.
   The gradient is again a weighted sum of sequence log-prob gradients
   (-(1/N) sum A*rho*active * grad log pi), reusing stage 06 with normalize=False.

Recorded run (single seed): the reward model ranked held-out pairs 100% right; with
beta=0.1 the policy went to ~100% "ok" style, with beta=5.6 it settled near 0.8
(theory 0.73; 128 samples gives about +-0.04 noise, and the reference is only
approximately 50/50, so close agreement is expected but not exact). This demo does
NOT exhibit reward hacking: the reward model is accurate on the narrow responses the
policy produces. Runtime is about 1.5 minutes.

SIMPLIFICATIONS (this is a teaching version, NOT production RLHF): sequence-level
ratio and advantage (not per-token with GAE), no value function, tiny model, and a
reward model trained on one narrow preference ("ok x" beats "no x").
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
_sft = importlib.import_module("02_sft")
_dpo = importlib.import_module("03_dpo")
_chars = importlib.import_module("01_char_tokenizer")
SEQ_LEN, encode_example, seq_logp, make_pairs = _sft.SEQ_LEN, _sft.encode_example, _dpo.seq_logp, _dpo.make_pairs
softmax = _bp.softmax


# ---------------- reward model ----------------

class RewardModel:
    """GPT backbone + scalar head. The head 'wr' lives in the same params dict so one optimizer covers all."""

    def __init__(self, init_from, pad_id, seed=0):
        P = init_from.params
        self.gpt = _bp.TrainableGPT(P["tok"].shape[0], P["tok"].shape[1], init_from.H, init_from.L, init_from.max_len)
        self.gpt.params = {k: v.copy() for k, v in P.items()}               # start from the SFT weights
        self.gpt.params["wr"] = np.random.default_rng(seed).normal(0, 0.02, P["tok"].shape[1])
        self.pad = pad_id

    @property
    def params(self):
        return self.gpt.params

    def score(self, ids):
        """r(x, y) = hidden state at the last real token . wr   -> (rewards, cache, last_idx, hidden)."""
        _, cache = self.gpt.forward(ids)
        last = (ids != self.pad).sum(axis=1) - 1
        h = cache[3][np.arange(len(ids)), last]
        return h @ self.params["wr"], cache, last, h


def reward_loss_and_grads(rm, ids_w, ids_l):
    """Bradley-Terry loss on (chosen, rejected) pairs and its gradients."""
    B = len(ids_w)
    ids = np.concatenate([ids_w, ids_l])
    r, cache, last, h = rm.score(ids)
    z = r[:B] - r[B:]
    loss = float(np.mean(np.logaddexp(0.0, -z)))
    s = 1.0 / (1.0 + np.exp(z))                                  # sigmoid(-z)
    dr = np.concatenate([-s / B, s / B])                          # dloss / dr for chosen, rejected
    dxf = np.zeros_like(cache[3])
    dxf[np.arange(2 * B), last] = dr[:, None] * rm.params["wr"]  # reward readout touches one position
    g = rm.gpt.backward(ids, cache, dxf=dxf)
    g["wr"] = dr @ h
    return loss, g, z


# ---------------- policy optimization ----------------

def rollout(policy, tok, pad, letters, n_prompts, K, rng, max_new=6):
    """Sample K responses for each of n_prompts prompts. Returns padded ids, response-token mask, letters."""
    cs = np.repeat(rng.choice(letters, size=n_prompts), K)
    ids = np.array([tok.encode(f"user: pick {c}\nbot: ") for c in cs])
    L0, nl = ids.shape[1], tok.stoi["\n"]
    done = np.zeros(len(ids), dtype=bool)
    for _ in range(max_new):
        logits = policy.forward(ids)[0][:, -1].copy()
        logits[:, pad] = -np.inf                                   # never sample padding
        cum = softmax(logits).cumsum(axis=1)
        samp = np.minimum((rng.random((len(ids), 1)) > cum).sum(axis=1), pad - 1)
        ids = np.concatenate([ids, np.where(done, pad, samp)[:, None]], axis=1)
        done |= samp == nl                                          # stop after the first newline
    full = np.concatenate([ids, np.full((len(ids), SEQ_LEN - ids.shape[1]), pad)], axis=1)
    mask = np.zeros((len(ids), SEQ_LEN - 1))
    for t in range(L0, L0 + max_new):
        mask[:, t - 1] = full[:, t] != pad
    return full, mask, cs


def ppo_loss_and_grads(policy, ids, mask, logp_old, adv, eps):
    """Clipped surrogate  -mean(min(rho*A, clip(rho)*A))  and its gradients."""
    N = len(ids)
    rho = np.exp(seq_logp(policy, ids, mask) - logp_old)
    loss = float(-np.mean(np.minimum(rho * adv, np.clip(rho, 1 - eps, 1 + eps) * adv)))
    active = ((adv >= 0) & (rho < 1 + eps)) | ((adv < 0) & (rho > 1 - eps))   # unclipped side only
    coef = adv * rho * active / N
    _, g = policy.loss_and_grads(ids, mask * coef[:, None], normalize=False)
    return loss, g, rho


def response_text(tok, row, L0):
    return "".join(tok.itos[i] for i in row[L0:] if i < tok.vocab_size)


def run_rl(sft_params, rm, ref, tok, pad, letters, beta, iters, rng, V, d, heads, layers, K=4, n_prompts=16,
           eps=0.2, epochs=4, lr=3e-4, log=True):
    policy = _bp.TrainableGPT(V, d, heads, layers, SEQ_LEN, seed=0)
    policy.params = {k: v.copy() for k, v in sft_params.items()}
    opt, L0, hist = _opt.AdamW(policy.params), len(tok.encode("user: pick a\nbot: ")), []
    for it in range(iters):
        ids, mask, cs = rollout(policy, tok, pad, letters, n_prompts, K, rng)
        r = rm.score(ids)[0]
        logp_old, logp_ref = seq_logp(policy, ids, mask), seq_logp(ref, ids, mask)
        R = r - beta * (logp_old - logp_ref)                        # KL-penalized reward
        Rg = R.reshape(n_prompts, K)
        adv = (Rg - (Rg.sum(axis=1, keepdims=True) - Rg) / (K - 1)).ravel()   # leave-one-out baseline
        for _ in range(epochs):
            _, g, _ = ppo_loss_and_grads(policy, ids, mask, logp_old, adv, eps)
            _sched.clip_grad_norm(g, 1.0)
            opt.step(policy.params, g, lr=lr)
        texts = [response_text(tok, row, L0) for row in ids]
        hist.append(dict(reward=float(r.mean()), kl=float((logp_old - logp_ref).mean()),
                         p_ok=float(np.mean([t.startswith("ok") for t in texts])),
                         exact=float(np.mean([t == f"ok {c}\n" for t, c in zip(texts, cs)]))))
        if log and (it % 10 == 0 or it == iters - 1):
            h = hist[-1]
            print(f"  beta={beta:<4} iter {it:3d}  RM reward {h['reward']:+.2f}  KL {h['kl']:+.2f}  "
                  f"'ok' style {h['p_ok']:.2f}  exact 'ok x' {h['exact']:.2f}")
    return policy, hist


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    tok = _chars.CharTokenizer("user: pick abcdef\nbot: ok no")
    PAD, V = tok.vocab_size, tok.vocab_size + 1
    TRAIN, HELD = list("abcd"), list("ef")
    d, heads, layers = 48, 4, 2

    # --- 1. gradient checks: reward model loss, and the PPO surrogate (eps huge so nothing clips) ---
    small = _bp.TrainableGPT(V, 8, 2, 2, SEQ_LEN, std=0.3)
    rm_s = RewardModel(small, PAD)
    rm_s.params["wr"] = rng.normal(0, 0.5, 8)
    batch = make_pairs(tok, PAD, TRAIN, 3, rng)
    _, g, _ = reward_loss_and_grads(rm_s, batch[0], batch[2])
    assert set(g) == set(rm_s.params)
    worst, h = 0.0, 1e-5
    for name, w in rm_s.params.items():
        for _ in range(2):
            idx = tuple(rng.integers(0, s) for s in w.shape)
            old = w[idx]
            w[idx] = old + h; up = reward_loss_and_grads(rm_s, batch[0], batch[2])[0]
            w[idx] = old - h; down = reward_loss_and_grads(rm_s, batch[0], batch[2])[0]
            w[idx] = old
            num = (up - down) / (2 * h)
            worst = max(worst, abs(num - g[name][idx]) / max(1e-8, abs(num) + abs(g[name][idx])))
    print(f"reward-model gradient check: max relative error {worst:.2e}")
    assert worst < 1e-5

    ids, mask = batch[0], batch[1]
    old = seq_logp(small, ids, mask)
    for k in small.params:                                          # move off rho = 1 so the check is meaningful
        small.params[k] += 0.02 * rng.normal(size=small.params[k].shape)
    adv = rng.normal(size=len(ids))
    _, gp, rho = ppo_loss_and_grads(small, ids, mask, old, adv, eps=1e9)
    assert not np.allclose(rho, 1.0)
    worst = 0.0
    for name, w in small.params.items():
        for _ in range(2):
            idx = tuple(rng.integers(0, s) for s in w.shape)
            o = w[idx]
            w[idx] = o + h; up = ppo_loss_and_grads(small, ids, mask, old, adv, 1e9)[0]
            w[idx] = o - h; down = ppo_loss_and_grads(small, ids, mask, old, adv, 1e9)[0]
            w[idx] = o
            num = (up - down) / (2 * h)
            worst = max(worst, abs(num - gp[name][idx]) / max(1e-8, abs(num) + abs(gp[name][idx])))
    print(f"PPO surrogate gradient check: max relative error {worst:.2e}")
    assert worst < 1e-5

    # --- 2. SFT policy that likes both styles equally (the starting point, as in DPO) ---
    def sft_batch(n):
        rows = []
        for c in rng.choice(list("abcdef"), size=n):
            style = "ok" if rng.random() < 0.5 else "no"
            rows.append(encode_example(tok, PAD, f"user: pick {c}\nbot: ", f"{style} {c}\n"))
        return np.stack([r[0] for r in rows]), np.stack([r[1] for r in rows])

    sft = _bp.TrainableGPT(V, d, heads, layers, SEQ_LEN, seed=0)
    opt = _opt.AdamW(sft.params, weight_decay=0.01)
    for step in range(600):
        ids, mask = sft_batch(32)
        _, g = sft.loss_and_grads(ids, mask)
        _sched.clip_grad_norm(g, 1.0)
        opt.step(sft.params, g, lr=_sched.lr_at(step, 3e-3, 30, 600, 3e-4))
    ref = _bp.TrainableGPT(V, d, heads, layers, SEQ_LEN, seed=0)
    ref.params = {k: v.copy() for k, v in sft.params.items()}               # frozen reference

    # --- 3. train the reward model on preference pairs (letters a-d), test on held-out e-f ---
    rm = RewardModel(sft, PAD)
    opt = _opt.AdamW(rm.params, weight_decay=0.01)
    for step in range(200):
        b = make_pairs(tok, PAD, TRAIN, 32, rng)
        loss, g, z = reward_loss_and_grads(rm, b[0], b[2])
        _sched.clip_grad_norm(g, 1.0)
        opt.step(rm.params, g, lr=3e-4)
    test = make_pairs(tok, PAD, HELD, 64, np.random.default_rng(9))
    gap = rm.score(test[0])[0] - rm.score(test[2])[0]
    gap_mean = float(gap.mean())
    print(f"reward model: held-out pairs ranked right {np.mean(gap > 0):.2f}, mean reward gap (ok - no) {gap_mean:+.2f}")
    assert np.mean(gap > 0) > 0.95 and gap_mean > 0

    # --- 4. RL against the frozen reward model, two KL strengths ---
    betas = (0.1, max(1.0, round(gap_mean, 1)))
    runs = {}
    for beta in betas:
        t0 = time.time()
        policy, hist = run_rl(sft.params, rm, ref, tok, PAD, TRAIN, beta, iters=40, rng=np.random.default_rng(3),
                              V=V, d=d, heads=heads, layers=layers)
        # evaluate on held-out letters with fresh samples
        ids, mask, cs = rollout(policy, tok, PAD, HELD, 32, 4, np.random.default_rng(4))
        texts = [response_text(tok, r, len(tok.encode("user: pick a\nbot: "))) for r in ids]
        runs[beta] = dict(hist=hist, p_ok=float(np.mean([t.startswith("ok") for t in texts])),
                          kl=float((seq_logp(policy, ids, mask) - seq_logp(ref, ids, mask)).mean()))
        print(f"  -> beta={beta}: held-out 'ok' style {runs[beta]['p_ok']:.2f}, KL to reference {runs[beta]['kl']:+.2f} "
              f"({time.time() - t0:.0f}s)\n")

    # --- 5. the optimum is known: pi*(ok) = sigmoid(gap / beta) for a two-style reference at 50/50 ---
    print("KL-regularized optimum  pi*(ok) = sigmoid(reward_gap / beta)  vs measured:")
    for beta in betas:
        theory = 1 / (1 + np.exp(-gap_mean / beta))
        print(f"  beta={beta:<4} theory {theory:.2f}   measured (held-out) {runs[beta]['p_ok']:.2f}")
    lo, hi = runs[betas[0]], runs[betas[1]]
    assert lo["p_ok"] > 0.9                                      # RL moved the policy toward the rewarded style
    assert hi["kl"] < lo["kl"]                                   # bigger KL penalty -> stays closer to the reference
