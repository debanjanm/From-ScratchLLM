"""Adaptive / dynamic tokenizer — vocab grows online from usage, no full retrain.

Method: every tokenizer so far (05-11) freezes its vocab after a one-time
`train()` pass — encountering a new domain (code, a new language, jargon)
later just means worse compression, not vocab growth. An adaptive tokenizer
instead starts from a small frozen base (byte-level BPE), then *during
encoding* tracks which byte-pairs recur often in the NEW text it's seeing.
Once a pair crosses a frequency threshold, it gets promoted into the live
vocab as a new merge — the tokenizer keeps learning after deployment,
without a full offline retrain. This mirrors ideas explored in domain-
adaptive tokenization and incremental vocab expansion research (2022+),
relevant when a base LLM's tokenizer is bad at a new domain (e.g. DNA
sequences, source code, a low-resource language) it wasn't trained on.

Pros: improves compression on new domains over time without retraining the
whole tokenizer (or the model's embedding table needing a full remap).
Cons: vocab drift — the same corpus tokenized before vs after adaptation
produces different ids, which breaks caching/reproducibility unless
versioned; naive online promotion can learn noisy/one-off merges if the
threshold is too low.
"""

from collections import Counter


class AdaptiveTokenizer:
    """Byte-level BPE base vocab that keeps learning new merges as it sees more text."""

    def __init__(self, promote_threshold: int = 5):
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.promote_threshold = promote_threshold
        self._pending_counts: Counter = Counter()  # pairs seen since last promotion

    def _pair_counts(self, ids: list[int]) -> Counter:
        return Counter(zip(ids, ids[1:]))

    def _merge(self, ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        out, i = [], 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    def encode(self, text: str, adapt: bool = True) -> list[int]:
        """Encode with current vocab; optionally observe pair frequency for future promotion."""
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            counts = self._pair_counts(ids)
            pair = min(counts, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = self._merge(ids, pair, self.merges[pair])

        if adapt:
            for pair, count in self._pair_counts(ids).items():
                self._pending_counts[pair] += count
            self._maybe_promote()
        return ids

    def _maybe_promote(self):
        """Promote any pending pair that has crossed the frequency threshold into a real merge."""
        to_promote = [p for p, c in self._pending_counts.items() if c >= self.promote_threshold]
        for pair in to_promote:
            if pair in self.merges:
                continue
            new_id = 256 + len(self.merges)
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            self.merges[pair] = new_id
            del self._pending_counts[pair]
            print(f"[adapted] promoted {pair} -> {new_id} ({self.vocab[new_id]}) "
                  f"after {self.promote_threshold}+ occurrences")

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")


if __name__ == "__main__":
    tok = AdaptiveTokenizer(promote_threshold=3)

    # simulate a new domain (repeated jargon term) the base vocab has no merge for
    domain_texts = ["the neural_net trains fast"] * 4
    print("before adaptation, vocab_size:", len(tok.vocab))

    for t in domain_texts:
        ids = tok.encode(t)

    print("after adaptation, vocab_size:", len(tok.vocab))

    sample = "the neural_net trains"
    ids = tok.encode(sample, adapt=False)
    decoded = tok.decode(ids)
    print("\nsample:", sample)
    print("ids:", ids, f"({len(ids)} tokens)")
    print("decoded:", decoded)
    assert decoded == sample
    print("\nrepeated domain term now compresses into fewer tokens than "
          "it would have under the original frozen base vocab")
