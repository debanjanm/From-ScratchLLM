"""Tokenizer-free / patch-based encoding (MegaByte/CANINE-style) — no vocab at all.

Method: every prior stage (01-13) is fundamentally a lookup: text -> some
unit (char/word/subword/byte) -> an id from a fixed or learned vocab table.
Tokenizer-free approaches remove that lookup entirely. Raw bytes are grouped
into fixed-size local "patches" (e.g. 4 consecutive bytes = 1 patch) and each
patch is fed to the model as a small vector/group directly — there is no
vocabulary, no merge rules, no training step for the tokenizer at all. The
model's first layer learns to interpret raw byte patches itself, e.g. via a
small per-patch encoder (MegaByte, 2023) or per-character hash embeddings
(CANINE, 2021).

This snippet demonstrates only the *patching* mechanic (grouping raw bytes
into fixed windows) — the actual "no vocab" part is real; the learned
per-patch embedding step is a model concern (see 02-embeddings), not a
tokenizer concern, so it's out of scope here.

Pros: zero vocabulary to build/store/version, trivially handles any
language/script/binary data, sidesteps the entire "what should the vocab be"
design question this whole folder has been exploring.
Cons: no compression at all — patch count still scales directly with raw
byte length (worse than any learned subword method for natural language),
pushes all the "meaning-finding" work onto the model architecture instead.
"""


PAD = -1  # not a valid byte value, so padding is never confused with data


class PatchTokenizer:
    """Groups raw UTF-8 bytes into fixed-size windows. No vocab, nothing learned."""

    def __init__(self, patch_size: int = 4):
        self.patch_size = patch_size

    def encode(self, text: str) -> list[tuple[int, ...]]:
        """Return fixed-size byte-value tuples ('patches'); last one padded with PAD (-1)."""
        raw = list(text.encode("utf-8"))
        patches = []
        for i in range(0, len(raw), self.patch_size):
            chunk = raw[i:i + self.patch_size]
            chunk += [PAD] * (self.patch_size - len(chunk))  # PAD is outside 0-255, so real NUL bytes survive
            patches.append(tuple(chunk))
        return patches

    def decode(self, patches: list[tuple[int, ...]]) -> str:
        raw = bytes(b for patch in patches for b in patch if b != PAD)
        return raw.decode("utf-8", errors="replace")


if __name__ == "__main__":
    tok = PatchTokenizer(patch_size=4)

    sample = "the quick fox"
    patches = tok.encode(sample)
    print("sample:", sample, f"({len(sample)} chars)")
    print("patches:", patches, f"({len(patches)} patches, no vocab lookup involved)")
    print("decoded:", tok.decode(patches))
    assert tok.decode(patches) == sample
    assert tok.decode(tok.encode("a\x00")) == "a\x00"  # NUL bytes survive (PAD != 0)

    print("\ncompare token counts for the same string across every method in this folder:")
    print(f"  char-level (01):      {len(sample)} tokens")
    print(f"  byte-level (10):      {len(sample.encode('utf-8'))} tokens")
    print(f"  patch-based (14):     {len(patches)} 'tokens', each covering "
          f"{tok.patch_size} raw bytes — fewer units, but each carries no learned meaning yet")
