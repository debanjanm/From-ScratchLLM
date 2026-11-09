"""BPE-Dropout — subword regularization by randomly skipping merges at encode time.

Method: every BPE variant so far applies its learned merges deterministically
— the same input string always tokenizes to the same ids. BPE-Dropout (Provilkov
et al., 2019) adds randomness: at encode time, each eligible merge is skipped
with probability `dropout_p` instead of always applied. The same word can
therefore segment differently on different training passes — e.g. "unhappily"
might become ["un", "happily"] one time and ["un", "happi", "ly"] another.

The model then sees multiple valid segmentations of the same word across
training, instead of memorizing one fixed decomposition. At inference,
dropout is disabled (p=0) so encoding reverts to standard deterministic BPE.

Pros: more robust representations (model doesn't over-rely on one exact
subword split), measurable BLEU/quality gains in low-resource NMT, cheap to
add (one skip-probability check per merge) to an existing trained BPE model.
Cons: only useful DURING training (adds noise) — inference must disable it
for reproducibility; increases training-time compute slightly (more distinct
segmentations to process, no vocab caching shortcut).
"""

import random
from collections import Counter


class BPEDropoutTokenizer:
    """Standard BPE (see 05) with an optional per-merge skip probability at encode time."""

    def __init__(self):
        self.merges: dict[tuple[int, int], int] = {}
        self.merge_order: list[tuple[int, int]] = []  # preserves learn order for replay
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

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

    def train(self, text: str, vocab_size: int):
        ids = list(text.encode("utf-8"))
        for i in range(vocab_size - 256):
            counts = self._pair_counts(ids)
            if not counts:
                break
            pair = max(counts, key=counts.get)
            new_id = 256 + i
            ids = self._merge(ids, pair, new_id)
            self.merges[pair] = new_id
            self.merge_order.append(pair)
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

    def encode(self, text: str, dropout_p: float = 0.0, seed: int = None) -> list[int]:
        """dropout_p=0 -> standard deterministic BPE. dropout_p>0 -> randomly skip
        eligible merges, so the same text can yield different segmentations."""
        rng = random.Random(seed)
        ids = list(text.encode("utf-8"))

        for pair in self.merge_order:  # apply in learned order, same as standard BPE
            if pair not in self.merges:
                continue
            if dropout_p > 0 and rng.random() < dropout_p:
                continue  # skip this merge this time — the "dropout"
            new_id = self.merges[pair]
            ids = self._merge(ids, pair, new_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")


if __name__ == "__main__":
    corpus = "unhappily the quick brown fox jumps happily over the lazy dog"
    tok = BPEDropoutTokenizer()
    tok.train(corpus, vocab_size=290)

    sample = "unhappily"
    print("sample:", sample)
    print("deterministic (p=0):", tok.encode(sample, dropout_p=0.0))

    print("\nwith dropout (p=0.3), different seeds -> different segmentations:")
    seen = set()
    for seed in range(5):
        ids = tok.encode(sample, dropout_p=0.3, seed=seed)
        seen.add(tuple(ids))
        assert tok.decode(ids) == sample  # always reversible regardless of segmentation
        print(f"  seed={seed}: {ids} ({len(ids)} tokens)")

    print(f"\n{len(seen)} distinct segmentation(s) observed across 5 seeds — "
          f"same word, same meaning, different subword split each time")
