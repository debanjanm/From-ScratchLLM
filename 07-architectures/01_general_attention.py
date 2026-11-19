"""General attention — one primitive for self-, causal-, bidirectional- and cross-attention.

GPT (stage 05) hard-wired a causal self-attention. The three Transformer
families differ mostly in WHICH tokens may attend to WHICH, and WHERE the
queries and keys come from:

  decoder-only (GPT)    queries, keys, values all from x;  mask = causal
  encoder-only (BERT)   queries, keys, values all from x;  mask = none (full)
  encoder-decoder (T5)  encoder: full self-attention over the source;
                        decoder: causal self-attention over the target PLUS
                        CROSS-attention: queries from the decoder, keys and
                        values from the encoder's output.

So the only changes needed are (1) take separate inputs xq and xkv (equal for
self-attention) and (2) accept any boolean mask. Everything here, forward AND
backward, is written once and reused by 02 (BERT) and 03 (T5). For
cross-attention the backward pass returns TWO input gradients: dxq flows back
into the decoder, dxkv flows back into the encoder. That second path is how
the encoder learns from the decoder's loss.

Also here: the pre-norm encoder block (attention + SwiGLU, residuals), built
from the norm/FFN layers hand-differentiated in 06-training.
"""

import importlib
import pathlib
import sys

import numpy as np

root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(root / "06-training"))
_bp = importlib.import_module("01_backprop_gpt")
softmax, causal_mask = _bp.softmax, _bp.causal_mask
rms_fwd, rms_bwd, ffn_fwd, ffn_bwd = _bp.rms_fwd, _bp.rms_bwd, _bp.ffn_fwd, _bp.ffn_bwd


def mha_fwd(xq, xkv, Wq, Wk, Wv, Wo, H, mask=None):
    """xq: (B,Tq,D), xkv: (B,Tk,D), mask: bool (Tq,Tk) with True = may attend."""
    B, Tq, D = xq.shape
    Tk, dk = xkv.shape[1], D // H
    heads = lambda t, T: t.reshape(B, T, H, dk).transpose(0, 2, 1, 3)       # (B,H,T,dk)
    q, k, v = heads(xq @ Wq, Tq), heads(xkv @ Wk, Tk), heads(xkv @ Wv, Tk)
    s = q @ k.transpose(0, 1, 3, 2) / np.sqrt(dk)
    if mask is not None:
        s = np.where(mask, s, -np.inf)
    p = softmax(s)
    ctx = (p @ v).transpose(0, 2, 1, 3).reshape(B, Tq, D)
    return ctx @ Wo, (xq, xkv, q, k, v, p, ctx)


def mha_bwd(dout, cache, Wq, Wk, Wv, Wo, H):
    """Returns dxq, dxkv, dWq, dWk, dWv, dWo. For self-attention add dxq + dxkv."""
    xq, xkv, q, k, v, p, ctx = cache
    B, Tq, D = xq.shape
    Tk, dk = xkv.shape[1], D // H
    flat = lambda t: t.reshape(-1, D)
    merge = lambda t, T: t.transpose(0, 2, 1, 3).reshape(B, T, D)

    dWo = flat(ctx).T @ flat(dout)
    dctx = (dout @ Wo.T).reshape(B, Tq, H, dk).transpose(0, 2, 1, 3)
    dp = dctx @ v.transpose(0, 1, 3, 2)
    dv = p.transpose(0, 1, 3, 2) @ dctx
    ds = p * (dp - np.sum(dp * p, axis=-1, keepdims=True)) / np.sqrt(dk)
    dq = merge(ds @ k, Tq)
    dkey = merge(ds.transpose(0, 1, 3, 2) @ q, Tk)
    dv = merge(dv, Tk)

    dWq = flat(xq).T @ flat(dq)
    dWk, dWv = flat(xkv).T @ flat(dkey), flat(xkv).T @ flat(dv)
    return dq @ Wq.T, dkey @ Wk.T + dv @ Wv.T, dWq, dWk, dWv, dWo


# ---------- parameter helpers and the (bidirectional or masked) encoder block ----------

def init_block(P, pre, d, rng, std=0.02, cross=False):
    """Add one block's weights to dict P under prefix `pre`. cross=True adds cross-attention."""
    n = lambda *shape: rng.normal(0, std, size=shape)
    hid = int(8 * d / 3)
    kinds = ["n1", "n2"] + (["n3"] if cross else [])
    for k in kinds:
        P[f"{pre}.{k}"] = np.ones(d)
    for prefix in ([""] + (["c"] if cross else [])):
        for w in ("Wq", "Wk", "Wv", "Wo"):
            P[f"{pre}.{prefix}{w}"] = n(d, d)
    P[f"{pre}.W1"], P[f"{pre}.W3"], P[f"{pre}.W2"] = n(d, hid), n(d, hid), n(hid, d)


def _add(grads, key, g):
    grads[key] = grads.get(key, 0) + g


def block_fwd(x, P, pre, H, mask=None):
    """Pre-norm block: x + MHA(norm(x)), then x + FFN(norm(x)). mask None = bidirectional."""
    n1, c1 = rms_fwd(x, P[f"{pre}.n1"])
    a, ca = mha_fwd(n1, n1, *(P[f"{pre}.{w}"] for w in ("Wq", "Wk", "Wv", "Wo")), H, mask)
    x1 = x + a
    n2, c2 = rms_fwd(x1, P[f"{pre}.n2"])
    f, cf = ffn_fwd(n2, P[f"{pre}.W1"], P[f"{pre}.W3"], P[f"{pre}.W2"])
    return x1 + f, (c1, ca, c2, cf)


def block_bwd(dx, cache, P, pre, H, grads):
    c1, ca, c2, cf = cache
    df, g1, g3, g2 = ffn_bwd(dx, cf, P[f"{pre}.W1"], P[f"{pre}.W3"], P[f"{pre}.W2"])
    d, gn2 = rms_bwd(df, c2)
    dx1 = dx + d
    dxq, dxkv, *gw = mha_bwd(dx1, ca, *(P[f"{pre}.{w}"] for w in ("Wq", "Wk", "Wv", "Wo")), H)
    d, gn1 = rms_bwd(dxq + dxkv, c1)                       # self-attention: both paths hit the same input
    for name, g in zip(("W1", "W3", "W2", "n2", "n1", "Wq", "Wk", "Wv", "Wo"), (g1, g3, g2, gn2, gn1, *gw)):
        _add(grads, f"{pre}.{name}", g)
    return dx1 + d


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    B, Tq, Tk, D, H = 2, 4, 6, 8, 2
    W = [rng.normal(0, 0.5, size=(D, D)) for _ in range(4)]
    xq, xkv = rng.normal(size=(B, Tq, D)), rng.normal(size=(B, Tk, D))

    # 1. cross-attention shapes: output follows the QUERY length, keys/values may differ in length
    out, cache = mha_fwd(xq, xkv, *W, H)
    print("cross-attention out:", out.shape, "(queries from decoder-like input, keys/values from another sequence)")
    assert out.shape == (B, Tq, D) and cache[5].shape == (B, H, Tq, Tk)

    # 2. mask semantics: causal => changing a later token can't affect earlier outputs; full => it can
    x = rng.normal(size=(B, 5, D))
    x2 = x.copy(); x2[:, -1] += 5.0
    causal_a = mha_fwd(x, x, *W, H, causal_mask(5))[0]
    causal_b = mha_fwd(x2, x2, *W, H, causal_mask(5))[0]
    full_a, full_b = mha_fwd(x, x, *W, H)[0], mha_fwd(x2, x2, *W, H)[0]
    assert np.allclose(causal_a[:, :-1], causal_b[:, :-1])
    assert not np.allclose(full_a[:, :-1], full_b[:, :-1])
    print("causal mask: past ignores the future | no mask (BERT/encoder): every token sees every token")

    # 3. finite-difference check of the backward pass, including BOTH input gradients
    target = rng.normal(size=(B, Tq, D))
    loss = lambda a, b, Ws: float(np.sum(mha_fwd(a, b, *Ws, H)[0] * target))
    dxq, dxkv, *dW = mha_bwd(target, cache, *W, H)
    worst, h = 0.0, 1e-6
    checks = [("xq", xq, dxq), ("xkv", xkv, dxkv)] + [(f"W{i}", W[i], dW[i]) for i in range(4)]
    for name, arr, grad in checks:
        for _ in range(4):
            idx = tuple(rng.integers(0, s) for s in arr.shape)
            old = arr[idx]
            arr[idx] = old + h; up = loss(xq, xkv, W)
            arr[idx] = old - h; down = loss(xq, xkv, W)
            arr[idx] = old
            num = (up - down) / (2 * h)
            worst = max(worst, abs(num - grad[idx]) / max(1e-8, abs(num) + abs(grad[idx])))
    print(f"gradient check (cross-attention, inputs + weights): max relative error {worst:.2e}")
    assert worst < 1e-6
