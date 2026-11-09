"""Byte Pair Encoding tokenizer, built from scratch (GPT-2 style, no libraries).

Algorithm:
1. Start with vocab = individual bytes (0-255).
2. Count frequency of adjacent token pairs across corpus.
3. Merge most frequent pair into a new token; repeat until target vocab size.
4. Encoding = greedily apply learned merges in order.
"""

from collections import Counter


class BPETokenizer:
    """Learns a subword vocab by iteratively merging the most frequent adjacent pair.

    Method: start from raw UTF-8 bytes (256 base tokens). Repeatedly find the
    most frequent adjacent token pair in the corpus and merge it into one new
    token, recording the merge rule. Stop once vocab_size is reached.
    Encoding replays the learned merges, in the order they were learned, on
    new text — so frequent subwords collapse into single tokens.

    Pros: no OOV (byte-level fallback always works), vocab size is tunable,
    frequent words/morphemes become single tokens (shorter sequences than
    char-level). This is the method GPT-2/GPT-3 popularized.
    Cons: merge order is greedy/frequency-driven, not probability-optimal
    (unlike Unigram LM); doesn't explicitly model word boundaries (WordPiece/
    SentencePiece address that differently).
    """

    def __init__(self):
        self.merges: dict[tuple[int, int], int] = {}   # (id1, id2) -> new_id
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    def _get_pair_counts(self, ids: list[int]) -> Counter:
        """Count how often each adjacent token pair occurs — merge candidates."""
        counts = Counter()
        for a, b in zip(ids, ids[1:]):
            counts[(a, b)] += 1
        return counts

    def _merge(self, ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        """Replace every non-overlapping occurrence of `pair` with `new_id`, left to right."""
        merged = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                merged.append(new_id)
                i += 2
            else:
                merged.append(ids[i])
                i += 1
        return merged

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        """Learn merge rules from `text` until vocab reaches vocab_size.

        Each iteration: count all adjacent pairs, merge the most frequent one
        into a new token id, and record the rule so encode() can replay it.
        """
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        ids = list(text.encode("utf-8"))

        for i in range(num_merges):
            counts = self._get_pair_counts(ids)
            if not counts:
                break
            pair = max(counts, key=counts.get)
            new_id = 256 + i
            ids = self._merge(ids, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {new_id} "
                      f"({self.vocab[new_id]}) had {counts[pair]} occurrences")

    def encode(self, text: str) -> list[int]:
        """Apply learned merges in the order they were learned (earliest-first)
        until no more merges from the training run apply."""
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            counts = self._get_pair_counts(ids)
            # pick the mergeable pair that was learned earliest
            pair = min(counts, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = self._merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids: list[int]) -> str:
        """Concatenate each token's raw bytes and decode as UTF-8 — lossless for known ids."""
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")


if __name__ == "__main__":
    corpus = "the quick brown fox jumps over the lazy dog. the dog barks at the fox."
    tok = BPETokenizer()
    tok.train(corpus, vocab_size=280, verbose=True)

    sample = "the fox jumps"
    ids = tok.encode(sample)
    decoded = tok.decode(ids)
    print("\noriginal:", sample)
    print("encoded:", ids)
    print("decoded:", decoded)
    assert decoded == sample
