"""T5-style encoder-decoder (seq2seq) trained on a toy task: reverse a digit string.

Method: two stacks joined by cross-attention.
  ENCODER  reads the whole source with bidirectional attention -> a memory of
           contextual vectors (one per source token).
  DECODER  generates the target left to right. Each decoder layer does
           (1) causal self-attention over what it has produced so far,
           (2) CROSS-attention: its queries look into the encoder memory
               (keys/values come from the encoder), then
           (3) an FFN.
Training uses teacher forcing: the decoder input is the target shifted right
behind a [BOS] token, and it must predict the target at every position.
At inference it decodes greedily, feeding back its own previous outputs.
The loss gradient reaches the encoder only through the cross-attention
keys/values (dxkv in 01_general_attention), summed over all decoder layers.

This is the "text-to-text" shape of T5 (translation, summarization: any task as
source string -> target string). Why this toy task: reversing needs the decoder
to ATTEND to a specific source position (output i copies source L-1-i), so it
only works if cross-attention is wired correctly. And unlike the repeated text
in earlier stages, inputs are fresh random strings, so test accuracy on
unseen sequences is a real generalization check.

Simplifications vs real T5: learned absolute positions instead of relative
position bias, pre-norm RMSNorm + SwiGLU, no padding (all lengths equal),
shared token embedding between encoder, decoder and the output head.
"""

import importlib
import pathlib
import sys
import time

import numpy as np

root = pathlib.Path(__file__).parent.parent
for folder in ("07-architectures", "06-training"):
    sys.path.insert(0, str(root / folder))
_ga = importlib.import_module("01_general_attention")
_opt = importlib.import_module("02_optimizers")
_sched = importlib.import_module("03_schedule_clipping")
softmax, causal_mask, rms_fwd, rms_bwd = _ga.softmax, _ga.causal_mask, _ga.rms_fwd, _ga.rms_bwd
ffn_fwd, ffn_bwd, mha_fwd, mha_bwd, _add = _ga.ffn_fwd, _ga.ffn_bwd, _ga.mha_fwd, _ga.mha_bwd, _ga._add


class Seq2Seq:
    def __init__(self, vocab, d_model, heads, enc_layers, dec_layers, max_len, seed=0, std=0.02):
        self.H, self.Le, self.Ld, self.max_len = heads, enc_layers, dec_layers, max_len
        rng = np.random.default_rng(seed)
        n = lambda *shape: rng.normal(0, std, size=shape)
        self.params = {"tok": n(vocab, d_model), "pos_enc": n(max_len, d_model),
                       "pos_dec": n(max_len, d_model),
                       "ne": np.ones(d_model), "nd": np.ones(d_model)}
        for l in range(enc_layers):
            _ga.init_block(self.params, f"e{l}", d_model, rng, std)
        for l in range(dec_layers):
            _ga.init_block(self.params, f"d{l}", d_model, rng, std, cross=True)

    def encode(self, src):
        p = self.params
        x = p["tok"][src] + p["pos_enc"][:src.shape[1]]
        caches = []
        for l in range(self.Le):
            x, c = _ga.block_fwd(x, p, f"e{l}", self.H, mask=None)     # bidirectional
            caches.append(c)
        enc, cn = rms_fwd(x, p["ne"])
        return enc, (caches, cn)

    def decode(self, tgt_in, enc):
        p, H, Tt = self.params, self.H, tgt_in.shape[1]
        cm = causal_mask(Tt)
        x = p["tok"][tgt_in] + p["pos_dec"][:Tt]
        caches = []
        for l in range(self.Ld):
            q = f"d{l}"
            W = lambda pre: [p[f"{q}.{pre}{w}"] for w in ("Wq", "Wk", "Wv", "Wo")]
            n1, c1 = rms_fwd(x, p[f"{q}.n1"])
            a, cs = mha_fwd(n1, n1, *W(""), H, cm)                      # causal self-attention
            x1 = x + a
            n2, c2 = rms_fwd(x1, p[f"{q}.n2"])
            a2, cc = mha_fwd(n2, enc, *W("c"), H, None)                 # cross-attention into the encoder
            x2 = x1 + a2
            n3, c3 = rms_fwd(x2, p[f"{q}.n3"])
            f, cf = ffn_fwd(n3, p[f"{q}.W1"], p[f"{q}.W3"], p[f"{q}.W2"])
            x = x2 + f
            caches.append((c1, cs, c2, cc, c3, cf))
        xd, cn = rms_fwd(x, p["nd"])
        return xd @ p["tok"].T, (caches, cn, xd)

    def loss_and_grads(self, src, tgt_in, tgt_out):
        p, H, g = self.params, self.H, {}
        enc, (ecaches, ecn) = self.encode(src)
        logits, (dcaches, dcn, xd) = self.decode(tgt_in, enc)

        probs = softmax(logits)
        N = tgt_out.size
        tgt = tgt_out[..., None]
        loss = float(-np.log(np.take_along_axis(probs, tgt, axis=-1)).mean())
        dl = probs.copy()
        np.put_along_axis(dl, tgt, np.take_along_axis(dl, tgt, axis=-1) - 1, axis=-1)
        dl /= N

        D = xd.shape[-1]
        g["tok"] = dl.reshape(-1, dl.shape[-1]).T @ xd.reshape(-1, D)
        dx, g["nd"] = rms_bwd(dl @ p["tok"], dcn)
        d_enc = 0
        for l in reversed(range(self.Ld)):
            q = f"d{l}"
            c1, cs, c2, cc, c3, cf = dcaches[l]
            Ws = lambda pre: [p[f"{q}.{pre}{w}"] for w in ("Wq", "Wk", "Wv", "Wo")]
            df, g1, g3, g2 = ffn_bwd(dx, cf, p[f"{q}.W1"], p[f"{q}.W3"], p[f"{q}.W2"])
            d, gn3 = rms_bwd(df, c3)
            dx2 = dx + d
            dxq, dxkv, *gw = mha_bwd(dx2, cc, *Ws("c"), H)              # cross-attention backward:
            d_enc = d_enc + dxkv                                         # keys/values path -> encoder
            for w, gr in zip(("cWq", "cWk", "cWv", "cWo"), gw):
                _add(g, f"{q}.{w}", gr)
            d, gn2 = rms_bwd(dxq, c2)                                    # queries path -> decoder
            dx1 = dx2 + d
            dxq, dxkv, *gw = mha_bwd(dx1, cs, *Ws(""), H)
            for w, gr in zip(("Wq", "Wk", "Wv", "Wo"), gw):
                _add(g, f"{q}.{w}", gr)
            d, gn1 = rms_bwd(dxq + dxkv, c1)
            dx = dx1 + d
            for name, gr in (("W1", g1), ("W3", g3), ("W2", g2), ("n3", gn3), ("n2", gn2), ("n1", gn1)):
                _add(g, f"{q}.{name}", gr)
        np.add.at(g["tok"], tgt_in, dx)
        g["pos_dec"] = np.zeros_like(p["pos_dec"])
        g["pos_dec"][:tgt_in.shape[1]] = dx.sum(axis=0)

        dxe, g["ne"] = rms_bwd(d_enc, ecn)                               # encoder learns only via cross-attention
        for l in reversed(range(self.Le)):
            dxe = _ga.block_bwd(dxe, ecaches[l], p, f"e{l}", H, g)
        np.add.at(g["tok"], src, dxe)
        g["pos_enc"] = np.zeros_like(p["pos_enc"])
        g["pos_enc"][:src.shape[1]] = dxe.sum(axis=0)
        return loss, g

    def greedy(self, src, bos, length):
        enc, _ = self.encode(src)
        ys = np.full((src.shape[0], 1), bos)
        for _ in range(length):
            nxt = self.decode(ys, enc)[0][:, -1].argmax(-1)
            ys = np.concatenate([ys, nxt[:, None]], axis=1)
        return ys[:, 1:]


BOS, VOCAB, LEN = 10, 11, 6          # digits 0-9 plus [BOS]


def make_batch(n, rng):
    src = rng.integers(0, 10, size=(n, LEN))
    tgt = src[:, ::-1]
    tgt_in = np.concatenate([np.full((n, 1), BOS), tgt[:, :-1]], axis=1)   # shifted right
    return src, tgt_in, tgt


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # --- gradient check on every parameter, incl. cross-attention and the shared embedding ---
    small = Seq2Seq(VOCAB, 8, 2, 1, 2, max_len=8, std=0.3)
    s, ti, to = make_batch(2, rng)
    _, grads = small.loss_and_grads(s, ti, to)
    assert set(grads) == set(small.params)
    worst, h = 0.0, 1e-5
    for name, w in small.params.items():
        for _ in range(3):
            idx = tuple(rng.integers(0, k) for k in w.shape)
            old = w[idx]
            w[idx] = old + h; up = small.loss_and_grads(s, ti, to)[0]
            w[idx] = old - h; down = small.loss_and_grads(s, ti, to)[0]
            w[idx] = old
            num = (up - down) / (2 * h)
            worst = max(worst, abs(num - grads[name][idx]) / max(1e-8, abs(num) + abs(grads[name][idx])))
    print(f"gradient check over {len(grads)} tensors: max relative error {worst:.2e}")
    assert worst < 1e-5

    # --- train on random digit strings (infinite fresh data) ---
    model = Seq2Seq(VOCAB, 64, 4, 2, 2, max_len=LEN)
    opt = _opt.AdamW(model.params, weight_decay=0.01)
    steps, t0 = 1500, time.time()
    for step in range(steps):
        loss, grads = model.loss_and_grads(*make_batch(32, rng))
        _sched.clip_grad_norm(grads, 1.0)
        opt.step(model.params, grads, lr=_sched.lr_at(step, 3e-3, 50, steps, 3e-4))
        if step % 250 == 0 or step == steps - 1:
            print(f"step {step:4d}  loss {loss:.3f}")
    print(f"trained in {time.time() - t0:.0f}s")

    # --- test on sequences the model has never seen ---
    src = make_batch(300, np.random.default_rng(12345))[0]
    pred = model.greedy(src, BOS, LEN)
    exact = float((pred == src[:, ::-1]).all(axis=1).mean())
    per_digit = float((pred == src[:, ::-1]).mean())
    print(f"unseen strings: exact-match {exact:.2f}, per-digit {per_digit:.2f} (chance per-digit 0.10)")
    for i in range(3):
        print("  ", "".join(map(str, src[i])), "->", "".join(map(str, pred[i])))
    assert per_digit > 0.9
