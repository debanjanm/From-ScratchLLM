"""LLM-optimized tokenizer (tiktoken/cl100k-style) — byte-level BPE + special tokens.

Method: everything from byte-level BPE (09) — regex pre-split, byte-level
base vocab, chunk-confined merges — PLUS explicit handling for special
tokens (`<|endoftext|>`, `<|im_start|>`, `<|im_end|>`, etc.) that must
*never* be split by BPE, since they carry structural meaning (document
boundaries, chat turn boundaries) rather than natural-language content.

Special tokens are matched and cut out of the text BEFORE the regular BPE
path runs, get their own reserved ids (allocated above the learned-merge
range), and are re-inserted verbatim at decode time. This is exactly how
OpenAI's tiktoken (used by GPT-3.5/4) and similar production tokenizers
handle chat formatting and document separators. The regex pattern used in
real cl100k_base is also more refined than GPT-2's (better number/whitespace
handling) — omitted here for brevity since chunking mechanics are already
covered by file 09.

Pros: structural/control tokens stay atomic and unambiguous (a BPE merge
could otherwise fuse "<|endoftext|>" partially into surrounding text),
which chat-formatted LLM training/inference depends on.
Cons: special-token set is closed and hand-picked at tokenizer-build time;
adding a new one later means retraining/reserving ids, not just appending.
"""

import re
from collections import Counter

GPT2_PATTERN = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\sa-zA-Z0-9]+|\s+"
)

DEFAULT_SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]


class LLMTokenizer:
    """Byte-level BPE (reuses 09's approach) with reserved, unsplittable special tokens."""

    def __init__(self, special_tokens: list[str] = None):
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special_tokens = special_tokens or DEFAULT_SPECIAL_TOKENS
        self._special_pattern = re.compile(
            "(" + "|".join(re.escape(t) for t in self.special_tokens) + ")"
        )
        self._assign_special_ids()  # valid before train(); re-run after merges are learned

    def _assign_special_ids(self):
        """Special ids sit just above the learned-merge range, so they shift as merges are added."""
        next_id = 256 + len(self.merges)
        self.special_to_id = {tok: next_id + i for i, tok in enumerate(self.special_tokens)}
        self.id_to_special = {i: tok for tok, i in self.special_to_id.items()}

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
        """Train byte-level BPE on text with special-token markers stripped out first."""
        # split around specials so they never enter merge stats and add no stray chunks
        plain_parts = [p for p in self._special_pattern.split(text) if p not in self.special_tokens]
        chunks_ids = [list(c.encode("utf-8")) for p in plain_parts for c in self._chunks(p)]

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
                print(f"merge {i+1}: {pair} -> {new_id} ({self.vocab[new_id]})")

        self._assign_special_ids()

    def _bpe_encode_chunk(self, chunk: str) -> list[int]:
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            counts = self._pair_counts([ids])
            pair = min(counts, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = self._merge_chunk(ids, pair, self.merges[pair])
        return ids

    def encode(self, text: str) -> list[int]:
        """Split around special tokens first — they bypass BPE entirely and keep their reserved id."""
        ids = []
        parts = self._special_pattern.split(text)
        for part in parts:
            if part in self.special_to_id:
                ids.append(self.special_to_id[part])
            elif part:
                for chunk in self._chunks(part):
                    ids.extend(self._bpe_encode_chunk(chunk))
        return ids

    def decode(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            if i in self.id_to_special:
                out.append(self.id_to_special[i])
            else:
                out.append(self.vocab[i].decode("utf-8", errors="replace"))
        return "".join(out)


if __name__ == "__main__":
    corpus = "the quick brown fox jumps over the lazy dog. the fox runs fast."
    tok = LLMTokenizer()
    tok.train(corpus, vocab_size=270, verbose=True)
    print("\nspecial token ids:", tok.special_to_id)

    sample = "<|im_start|>the fox jumps<|im_end|>"
    ids = tok.encode(sample)
    decoded = tok.decode(ids)
    print("\nsample:", sample)
    print("ids:", ids)
    print("decoded:", decoded)
    assert decoded == sample
    assert ids[0] == tok.special_to_id["<|im_start|>"] and ids[-1] == tok.special_to_id["<|im_end|>"]
    assert LLMTokenizer().encode("<|endoftext|>") == [256]  # encode works before train()

    print("\nnote <|im_start|>/<|im_end|> encode to single reserved ids, "
          "never split into byte-level pieces the way surrounding text is")
