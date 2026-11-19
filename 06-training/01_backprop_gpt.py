"""Hand-written backpropagation for the GPT — no autograd, just the chain rule.

Method: every layer is a pair of functions. forward(x) returns the output AND
a cache of whatever the backward pass will need; backward(d_out, cache)
applies the chain rule and returns gradients for the layer's input and its
weights. The model's backward is those backward calls in reverse order, with
one rule for the residual stream: the gradient arriving at "x + f(x)" flows
unchanged into BOTH x and f.

Derivations used (D = feature size, N = batch*seq):
  matmul  Y = X W      dX = dY W^T         dW = X^T dY
  RMSNorm y = g * x/r, r = sqrt(mean(x^2) + eps)
          dx = (g*dy)/r - x * sum(g*dy*x) / (D r^3)      dg = sum(dy * x/r)
  softmax p = softmax(s)       ds = p * (dp - sum(dp * p))   (masked p=0 -> ds=0)
  attention  scores = q k^T / sqrt(d_k); out = p v  (heads kept as an axis)
  SwiGLU  m = silu(xW1) * (xW3); out = m W2;  silu'(a) = s(a) * (1 + a(1 - s(a)))
  embedding  gradient scatter-adds rows:  np.add.at(d_tok, ids, dx)
  tied head  logits = xf E^T  ->  dxf = dlogits E,  dE += dlogits^T xf
  loss       softmax cross-entropy:  dlogits = (softmax - one_hot) / N

Trust but verify: `grad_check` compares every analytic gradient with a
finite-difference estimate (L(w+h) - L(w-h)) / 2h. If a derivation above is
wrong, that check fails. Scope: dense SwiGLU blocks (MoE backward skipped).
"""

import importlib
import pathlib
import sys

import numpy as np

root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(root / "03-attention"))
_sdpa = importlib.import_module("01_scaled_dot_product")
_causal = importlib.import_module("02_causal_mask")
softmax, causal_mask = _sdpa.softmax, _causal.causal_mask

EPS = 1e-6


# ---------- layers: forward returns (out, cache), backward returns grads ----------

def rms_fwd(x, g):
    r = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + EPS)
    return x / r * g, (x, g, r)


def rms_bwd(dy, cache):
    x, g, r = cache
    D = x.shape[-1]
    u = dy * g
    dx = u / r - x * np.sum(u * x, axis=-1, keepdims=True) / (D * r ** 3)
    dg = (dy * x / r).reshape(-1, D).sum(axis=0)
    return dx, dg


def attn_fwd(h, Wq, Wk, Wv, Wo, H):
    B, T, D = h.shape
    dk = D // H
    split = lambda m: (h @ m).reshape(B, T, H, dk).transpose(0, 2, 1, 3)   # (B,H,T,dk)
    q, k, v = split(Wq), split(Wk), split(Wv)
    s = q @ k.transpose(0, 1, 3, 2) / np.sqrt(dk)
    p = softmax(np.where(causal_mask(T), s, -np.inf))
    ctx = (p @ v).transpose(0, 2, 1, 3).reshape(B, T, D)
    return ctx @ Wo, (h, q, k, v, p, ctx)


def attn_bwd(dout, cache, Wq, Wk, Wv, Wo, H):
    h, q, k, v, p, ctx = cache
    B, T, D = h.shape
    dk = D // H
    flat = lambda t: t.reshape(-1, D)
    merge = lambda t: t.transpose(0, 2, 1, 3).reshape(B, T, D)

    dWo = flat(ctx).T @ flat(dout)
    dctx = (dout @ Wo.T).reshape(B, T, H, dk).transpose(0, 2, 1, 3)
    dp = dctx @ v.transpose(0, 1, 3, 2)
    dv = p.transpose(0, 1, 3, 2) @ dctx
    ds = p * (dp - np.sum(dp * p, axis=-1, keepdims=True)) / np.sqrt(dk)
    dq = merge(ds @ k)
    dkey = merge(ds.transpose(0, 1, 3, 2) @ q)
    dv = merge(dv)

    dWq, dWk, dWv = (flat(h).T @ flat(d) for d in (dq, dkey, dv))
    dh = dq @ Wq.T + dkey @ Wk.T + dv @ Wv.T
    return dh, dWq, dWk, dWv, dWo


def ffn_fwd(h, W1, W3, W2):
    a, b = h @ W1, h @ W3
    sig = 1 / (1 + np.exp(-a))
    s = a * sig                                  # silu(a)
    m = s * b
    return m @ W2, (h, a, b, sig, s, m)


def ffn_bwd(dout, cache, W1, W3, W2):
    h, a, b, sig, s, m = cache
    D = h.shape[-1]
    flat = lambda t: t.reshape(-1, t.shape[-1])
    dW2 = flat(m).T @ flat(dout)
    dm = dout @ W2.T
    da = dm * b * sig * (1 + a * (1 - sig))      # through silu
    db = dm * s
    dW1, dW3 = flat(h).T @ flat(da), flat(h).T @ flat(db)
    return da @ W1.T + db @ W3.T, dW1, dW3, dW2


# ---------- the model ----------

class TrainableGPT:
    """Functional GPT: all weights live in the `params` dict so optimizers can update them."""

    def __init__(self, vocab, d_model, heads, layers, max_len, seed=0, std=0.02):
        assert d_model % heads == 0
        self.H, self.L, self.max_len = heads, layers, max_len
        rng = np.random.default_rng(seed)
        n = lambda *shape: rng.normal(0, std, size=shape)
        hid = int(8 * d_model / 3)
        p = {"tok": n(vocab, d_model), "pos": n(max_len, d_model), "nf": np.ones(d_model)}
        for l in range(layers):
            p[f"{l}.n1"], p[f"{l}.n2"] = np.ones(d_model), np.ones(d_model)
            for name in ("Wq", "Wk", "Wv", "Wo"):
                p[f"{l}.{name}"] = n(d_model, d_model)
            p[f"{l}.W1"], p[f"{l}.W3"], p[f"{l}.W2"] = n(d_model, hid), n(d_model, hid), n(hid, d_model)
        self.params = p

    def forward(self, ids):
        p, T = self.params, ids.shape[1]
        assert T <= self.max_len
        x = p["tok"][ids] + p["pos"][:T]
        caches = []
        for l in range(self.L):
            n1, c1 = rms_fwd(x, p[f"{l}.n1"])
            a, ca = attn_fwd(n1, p[f"{l}.Wq"], p[f"{l}.Wk"], p[f"{l}.Wv"], p[f"{l}.Wo"], self.H)
            x1 = x + a
            n2, c2 = rms_fwd(x1, p[f"{l}.n2"])
            f, cf = ffn_fwd(n2, p[f"{l}.W1"], p[f"{l}.W3"], p[f"{l}.W2"])
            x = x1 + f
            caches.append((c1, ca, c2, cf))
        xf, cn = rms_fwd(x, p["nf"])
        return xf @ p["tok"].T, (ids, caches, cn, xf)

    def loss_and_grads(self, ids, loss_mask=None):
        """Next-token cross-entropy over ids (B,T), and gradients for every parameter.

        loss_mask (optional, shape (B, T-1), 1 = count, 0 = ignore) says which TARGET positions
        (ids[:, 1:]) contribute to the loss. None = all of them. Used by SFT to train on
        response tokens only."""
        p, g = self.params, {}
        logits, (ids_, caches, cn, xf) = self.forward(ids)
        B, T, V = logits.shape

        probs = softmax(logits[:, :-1])
        tgt = ids[:, 1:, None]
        w = np.ones((B, T - 1)) if loss_mask is None else np.asarray(loss_mask, dtype=float)
        N = w.sum()
        picked = np.take_along_axis(probs, tgt, axis=-1)[..., 0]
        loss = float(-(np.log(picked) * w).sum() / N)
        dl = probs.copy()
        np.put_along_axis(dl, tgt, np.take_along_axis(dl, tgt, axis=-1) - 1, axis=-1)
        dlogits = np.zeros_like(logits)
        dlogits[:, :-1] = dl * w[..., None] / N

        g["tok"] = dlogits.reshape(-1, V).T @ xf.reshape(-1, xf.shape[-1])
        dx, g["nf"] = rms_bwd(dlogits @ p["tok"], cn)
        for l in reversed(range(self.L)):
            c1, ca, c2, cf = caches[l]
            df, g[f"{l}.W1"], g[f"{l}.W3"], g[f"{l}.W2"] = ffn_bwd(
                dx, cf, p[f"{l}.W1"], p[f"{l}.W3"], p[f"{l}.W2"])
            d, g[f"{l}.n2"] = rms_bwd(df, c2)
            dx1 = dx + d                                          # residual: grad splits to both paths
            dh, g[f"{l}.Wq"], g[f"{l}.Wk"], g[f"{l}.Wv"], g[f"{l}.Wo"] = attn_bwd(
                dx1, ca, p[f"{l}.Wq"], p[f"{l}.Wk"], p[f"{l}.Wv"], p[f"{l}.Wo"], self.H)
            d, g[f"{l}.n1"] = rms_bwd(dh, c1)
            dx = dx1 + d
        np.add.at(g["tok"], ids, dx)                              # embedding lookup scatter
        g["pos"] = np.zeros_like(p["pos"])
        g["pos"][:ids.shape[1]] = dx.sum(axis=0)
        return loss, g


def grad_check(model, ids, per_param=3, h=1e-5, seed=0):
    """Max relative error between analytic and finite-difference gradients."""
    rng = np.random.default_rng(seed)
    _, grads = model.loss_and_grads(ids)
    worst = 0.0
    for name, w in model.params.items():
        for _ in range(per_param):
            idx = tuple(rng.integers(0, s) for s in w.shape)
            old = w[idx]
            w[idx] = old + h
            up = model.loss_and_grads(ids)[0]
            w[idx] = old - h
            down = model.loss_and_grads(ids)[0]
            w[idx] = old
            num, ana = (up - down) / (2 * h), grads[name][idx]
            worst = max(worst, abs(num - ana) / max(1e-8, abs(num) + abs(ana)))
    return worst


if __name__ == "__main__":
    vocab, d_model, heads, layers, max_len = 11, 8, 2, 2, 8
    ids = np.random.default_rng(1).integers(0, vocab, size=(2, 6))

    # larger init than training uses, so gradients are big enough to compare numerically
    model = TrainableGPT(vocab, d_model, heads, layers, max_len, std=0.3)
    loss, grads = model.loss_and_grads(ids)
    print(f"loss {loss:.4f} (ln vocab = {np.log(vocab):.4f}), {len(grads)} gradient tensors")
    assert set(grads) == set(model.params)
    assert all(grads[k].shape == model.params[k].shape for k in grads)

    err = grad_check(model, ids)
    print(f"gradient check: max relative error {err:.2e} over {3 * len(grads)} sampled weights")
    assert err < 1e-5

    # cross-check against the stage-05 forward-only GPT carrying the same weights
    sys.path.insert(0, str(root / "05-gpt"))
    gpt5 = importlib.import_module("01_gpt_model").GPT(vocab, d_model, heads, layers, max_len)
    p = model.params
    gpt5.tok.weight, gpt5.pos.weight, gpt5.norm_f.gamma = p["tok"], p["pos"], p["nf"]
    for l, blk in enumerate(gpt5.blocks):
        blk.norm1.gamma, blk.norm2.gamma = p[f"{l}.n1"], p[f"{l}.n2"]
        blk.attn.W_q, blk.attn.W_k, blk.attn.W_v, blk.attn.W_o = (p[f"{l}.{n}"] for n in ("Wq", "Wk", "Wv", "Wo"))
        blk.ffn.W1, blk.ffn.W3, blk.ffn.W2 = p[f"{l}.W1"], p[f"{l}.W3"], p[f"{l}.W2"]
    assert np.allclose(gpt5(ids), model.forward(ids)[0])
    print("forward matches the stage-05 GPT exactly with the same weights: True")
