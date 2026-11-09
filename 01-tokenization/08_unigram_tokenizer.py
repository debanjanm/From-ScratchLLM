"""Unigram Language Model tokenizer — start big, prune down (opposite of BPE).

Method: BPE/WordPiece build UP from characters via merges. Unigram LM does
the reverse: start from a large seed vocab (every substring up to some
length), give each piece a probability, then repeat:
  1. EM: segment every word with Viterbi under current probabilities, count
     piece usage, re-estimate probabilities (unused pieces keep a tiny
     smoothed count so they stay candidates for pruning),
  2. prune: for each piece, measure how much total corpus log-likelihood
     DROPS if it were removed (words re-segmented without it); delete the
     ~25% of pieces whose removal hurts least. Single characters are never
     removed, so any string stays segmentable.
Stop once vocab_size is reached. Encoding = Viterbi: the segmentation that
maximizes total probability (globally optimal, not greedy-leftmost).
Used by ALBERT, T5, XLNet (typically paired with SentencePiece preprocessing).

Pros: probabilistic framing means several valid segmentations exist and can
be sampled (subword regularization); encoding is globally optimal for the
learned distribution.
Cons: much slower to train than BPE (Viterbi re-run per piece per prune).

Simplification: hard-EM (Viterbi counts) instead of forward-backward
expected counts, and a fixed 25% prune rate per round.
"""

import math
from collections import Counter

SPACE_MARKER = "▁"
UNK_PENALTY = 20.0  # log-prob penalty for a char never seen in training


class UnigramTokenizer:
    """Probabilistic subword vocab, pruned from a large seed set by likelihood loss."""

    def __init__(self, max_piece_len: int = 6):
        self.probs: dict[str, float] = {}  # piece -> log prob
        self.max_piece_len = max_piece_len

    def _segment(self, word: str, probs: dict = None) -> tuple[list[str], float]:
        """Viterbi: best split of `word` and its total log-prob."""
        probs = self.probs if probs is None else probs
        unk = min(probs.values()) - UNK_PENALTY
        n = len(word)
        best = [-math.inf] * (n + 1)
        best[0] = 0.0
        back = [0] * (n + 1)
        for end in range(1, n + 1):
            for start in range(max(0, end - self.max_piece_len), end):
                piece = word[start:end]
                lp = probs.get(piece, unk if end - start == 1 else None)
                if lp is None:
                    continue
                if best[start] + lp > best[end]:
                    best[end], back[end] = best[start] + lp, start
        pieces, pos = [], n
        while pos > 0:
            pieces.append(word[back[pos]:pos])
            pos = back[pos]
        return pieces[::-1], best[n]

    def train(self, text: str, vocab_size: int, em_iters: int = 3,
              prune_frac: float = 0.25, verbose: bool = False):
        words = Counter(SPACE_MARKER + w for w in text.split())
        chars = {c for w in words for c in w}

        # seed: every substring up to max_piece_len, weighted by word frequency
        seed = Counter()
        for w, f in words.items():
            for i in range(len(w)):
                for j in range(i + 1, min(i + self.max_piece_len, len(w)) + 1):
                    seed[w[i:j]] += f
        total = sum(seed.values())
        self.probs = {p: math.log(c / total) for p, c in seed.items()}

        while True:
            for _ in range(em_iters):  # hard-EM: Viterbi counts -> new probs
                usage = Counter()
                for w, f in words.items():
                    for p in self._segment(w)[0]:
                        usage[p] += f
                smoothed = {p: usage.get(p, 0) + 0.01 for p in self.probs}
                total = sum(smoothed.values())
                self.probs = {p: math.log(c / total) for p, c in smoothed.items()}

            if verbose:
                print(f"vocab_size={len(self.probs)}")
            if len(self.probs) <= vocab_size:
                break

            # loss of removing each piece = corpus log-likelihood drop
            base = {w: self._segment(w)[1] for w in words}
            loss = {}
            for p in self.probs:
                if p in chars:
                    continue
                reduced = {q: lp for q, lp in self.probs.items() if q != p}
                loss[p] = sum(f * (base[w] - self._segment(w, reduced)[1])
                              for w, f in words.items() if p in w)
            n_drop = min(len(self.probs) - vocab_size,
                         max(1, int(len(self.probs) * prune_frac)))
            for p in sorted(loss, key=loss.get)[:n_drop]:
                del self.probs[p]

    def encode(self, text: str) -> list[str]:
        tokens = []
        for word in text.split():
            tokens.extend(self._segment(SPACE_MARKER + word)[0])
        return tokens

    def decode(self, tokens: list[str]) -> str:
        return "".join(tokens).replace(SPACE_MARKER, " ").lstrip()


if __name__ == "__main__":
    corpus = ("the quick brown fox jumps over the lazy dog the fox runs "
              "the dog barks the quick fox jumps again")
    tok = UnigramTokenizer()
    tok.train(corpus, vocab_size=40, verbose=True)

    sample = "the quick fox"
    tokens = tok.encode(sample)
    print("\nsample:", sample)
    print("tokens:", tokens)
    print("decoded:", repr(tok.decode(tokens)))
    assert tok.decode(tokens) == sample
    assert len(tok.probs) <= 40                    # pruned down to target
    assert any(len(p) > 2 for p in tok.probs)      # multi-char pieces survived pruning
    assert tok.decode(tok.encode("zebra")) == "zebra"  # unseen chars still segment

    print(f"\nfinal vocab: {len(tok.probs)} pieces, e.g. {sorted(tok.probs, key=tok.probs.get)[-5:]}")
    print("pruned by likelihood loss; encoding is the globally best Viterbi path, not greedy")
