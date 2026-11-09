"""Pure byte/Unicode tokenizer (ByT5 style) — no merges, no learning, no OOV ever.

Method: encode = raw UTF-8 bytes, one token per byte. That's the entire
tokenizer — no training step, no vocab to learn, vocab_size is fixed at 256
(every possible byte value) regardless of language, script, or domain.
This is the endpoint of "no pre-tokenizer, no merges" — a deliberate step
back from BPE/SentencePiece/Unigram's learned-vocab approach. Used by ByT5,
CANINE-adjacent models, and as a fallback path in tokenizer-free research
(2021+).

Pros: truly universal — works identically on any language/script/emoji/
binary data with zero OOV, zero training needed, trivially reversible.
Cons: sequence length explodes (a single multi-byte character, e.g. many
CJK chars or emoji, becomes 3-4 tokens), so the model needs a much longer
effective context and more compute per unit of text than a learned subword
vocab would need for the same content.
"""


class ByteTokenizer:
    """Fixed 256-entry vocab = every possible byte value. No training."""

    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        """UTF-8 encode, one token per byte — always succeeds, never raises."""
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return bytes(ids).decode("utf-8", errors="replace")


if __name__ == "__main__":
    tok = ByteTokenizer()
    print("vocab_size:", tok.vocab_size, "(fixed, no training)")

    ascii_sample = "hello"
    ids = tok.encode(ascii_sample)
    print("\nASCII sample:", ascii_sample)
    print("ids:", ids, f"({len(ids)} tokens for {len(ascii_sample)} chars — 1:1)")
    assert tok.decode(ids) == ascii_sample

    # multi-byte chars show the sequence-length cost of this method
    multibyte_sample = "héllo 世界 🌍"
    ids = tok.encode(multibyte_sample)
    print("\nmulti-byte sample:", multibyte_sample)
    print("ids:", ids)
    print(f"{len(ids)} tokens for {len(multibyte_sample)} chars — "
          f"non-ASCII chars cost multiple tokens each, but nothing is ever <unk>")
    assert tok.decode(ids) == multibyte_sample
