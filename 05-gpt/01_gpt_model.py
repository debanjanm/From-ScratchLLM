"""Mini-GPT — decoder-only Transformer language model.

Method: GPT is just the earlier stages wired in a row:
    token ids -> token embedding + learned position embedding   (02)
              -> N x TransformerBlock (causal attention + FFN)  (03, 04)
              -> final norm -> project to vocab logits
Output logits[b, t] score every vocab token as the NEXT token after position
t, so one forward pass yields a next-token prediction at every position
(that is what makes training on whole sequences parallel).

Weight tying: the output projection reuses the token-embedding matrix
(logits = h @ E^T). It saves vocab*d_model parameters and ties "what a token
means going in" to "what predicts it coming out" (GPT-2, LLaMA-small).

Loss: cross-entropy between logits[:, :-1] and the tokens shifted left by
one. An untrained model is near-uniform, so its loss starts at ln(vocab).
Training (backprop, optimizer) is stage 06; the weights here are random.
"""

import importlib
import pathlib
import sys

import numpy as np

root = pathlib.Path(__file__).parent.parent
for folder in ("02-embeddings", "03-attention", "04-transformer-block"):
    sys.path.insert(0, str(root / folder))
_tok = importlib.import_module("01_token_embedding")
_pos = importlib.import_module("02_positional_encoding")
_norm = importlib.import_module("02_rmsnorm")
_block = importlib.import_module("05_transformer_block")
_sdpa = importlib.import_module("01_scaled_dot_product")   # softmax


class GPT:
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_layers: int,
                 max_seq_len: int, seed: int = 0):
        self.max_seq_len = max_seq_len
        self.tok = _tok.TokenEmbedding(vocab_size, d_model, seed)
        self.pos = _pos.LearnedPositionalEncoding(max_seq_len, d_model, seed + 1)
        self.blocks = [_block.TransformerBlock(d_model, num_heads, seed=seed + 10 + i)
                       for i in range(num_layers)]
        self.norm_f = _norm.RMSNorm(d_model)

    def __call__(self, ids: np.ndarray) -> np.ndarray:
        """ids: (batch, seq) ints -> logits (batch, seq, vocab)."""
        T = ids.shape[1]
        assert T <= self.max_seq_len, "sequence longer than the learned position table"
        x = self.tok(ids) + self.pos(T)
        for block in self.blocks:
            x = block(x)
        return self.norm_f(x) @ self.tok.weight.T     # weight tying


def cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
    """Mean next-token loss. logits: (B,T,V), targets: (B,T) ids."""
    probs = _sdpa.softmax(logits, axis=-1)
    picked = np.take_along_axis(probs, targets[..., None], axis=-1)[..., 0]
    return float(-np.log(picked).mean())


def language_model_loss(model: GPT, ids: np.ndarray) -> float:
    """Predict token t+1 from tokens <= t: compare logits[:, :-1] with ids[:, 1:]."""
    return cross_entropy(model(ids)[:, :-1], ids[:, 1:])


if __name__ == "__main__":
    vocab, d_model, heads, layers, max_len = 50, 32, 4, 2, 64
    model = GPT(vocab, d_model, heads, layers, max_len)
    ids = np.random.default_rng(0).integers(0, vocab, size=(3, 10))

    logits = model(ids)
    print("logits shape:", logits.shape)
    assert logits.shape == (3, 10, vocab)

    loss = language_model_loss(model, ids)
    print(f"untrained loss {loss:.4f} vs ln(vocab) {np.log(vocab):.4f}")
    assert abs(loss - np.log(vocab)) < 0.1      # near-uniform predictions

    # causal at model level: editing the last token leaves earlier logits untouched
    ids2 = ids.copy()
    ids2[:, -1] = (ids2[:, -1] + 1) % vocab
    assert np.allclose(model(ids2)[:, :-1], logits[:, :-1])
    print("earlier positions unaffected by a changed future token: True")
