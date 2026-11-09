"""Byte-level BPE (GPT-2 style) — regex pre-split + byte-level BPE, no cross-word merges.

Method: plain BPE (see 05) runs merges over the *entire* byte stream with no
boundaries, so it can (and does, see 05's output) learn a merge like
"the quick brown fox j" as one token — nonsensical, and it means merges
learned from one word can bleed into unrelated neighboring text.

GPT-2's fix: apply a regex pre-tokenizer FIRST to chunk text into words/
punctuation/contractions (similar spirit to the rule-based tokenizer, 03),
then run byte-level BPE *independently within each chunk* — merges never
cross a chunk boundary. Base vocab is still all 256 byte values, so there's
still zero OOV; the regex only constrains WHERE merges can happen, it
doesn't add rules to the vocab itself.

Pros: no OOV (byte-level base), no degenerate cross-word merges, subword
units align with linguistic boundaries. This is exactly what GPT-2/GPT-3
use.
Cons: regex pre-tokenizer is itself a hand-tuned rule (language-specific
edge cases), slightly more complex than plain BPE.
"""

import re
from collections import Counter

# simplified ASCII approximation of GPT-2's regex (real one uses \p{L}/\p{N}
# Unicode categories via the `regex` package; stdlib `re` can't express those)
GPT2_PATTERN = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\sa-zA-Z0-9]+|\s+"
)


class ByteLevelBPE:
    """BPE over UTF-8 bytes, with merges confined inside regex-split chunks."""

    def __init__(self):
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    def _chunks(self, text: str) -> list[str]:
        return GPT2_PATTERN.findall(text)

    def _pair_counts(self, chunks_ids: list[list[int]]) -> Counter:
        counts = Counter()
        for ids in chunks_ids:
            for a, b in zip(ids, ids[1:]):
                counts[(a, b)] += 1
        return counts

    def _merge_chunk(self, ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        out, i = [], 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        assert vocab_size >= 256
        chunks_ids = [list(chunk.encode("utf-8")) for chunk in self._chunks(text)]

        for i in range(vocab_size - 256):
            counts = self._pair_counts(chunks_ids)
            if not counts:
                break
            pair = max(counts, key=counts.get)
            new_id = 256 + i
            chunks_ids = [self._merge_chunk(ids, pair, new_id) for ids in chunks_ids]
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose:
                print(f"merge {i+1}: {pair} -> {new_id} ({self.vocab[new_id]}), "
                      f"{counts[pair]} occurrences — confined to its chunk")

    def encode(self, text: str) -> list[int]:
        ids = []
        for chunk in self._chunks(text):
            chunk_ids = list(chunk.encode("utf-8"))
            while len(chunk_ids) >= 2:
                counts = self._pair_counts([chunk_ids])
                pair = min(counts, key=lambda p: self.merges.get(p, float("inf")))
                if pair not in self.merges:
                    break
                chunk_ids = self._merge_chunk(chunk_ids, pair, self.merges[pair])
            ids.extend(chunk_ids)
        return ids

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")


if __name__ == "__main__":
    corpus = "the quick brown fox jumps over the lazy dog. the dog barks at the fox."
    tok = ByteLevelBPE()
    tok.train(corpus, vocab_size=280, verbose=True)

    sample = "the fox jumps"
    ids = tok.encode(sample)
    decoded = tok.decode(ids)
    print("\nsample:", sample)
    print("ids:", ids)
    print("decoded:", decoded)
    assert decoded == sample

    print("\ncompare to 05_bpe_tokenizer.py's merges: no token here spans multiple "
          "words like 'the quick brown fox j' — chunk boundaries prevent that")
