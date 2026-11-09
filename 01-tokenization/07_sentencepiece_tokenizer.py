"""SentencePiece-style tokenizer — language-agnostic, whitespace is just a symbol.

Method: unlike BPE/WordPiece, which first split on whitespace and DROP it,
SentencePiece keeps whitespace as a visible symbol "▁" (U+2581). Text gets a
dummy "▁" prefix, every space becomes "▁", and the stream is cut into
pieces that each start at a "▁" run (so "▁the", "▁cat"). BPE merges then run
inside those pieces only — no merge ever crosses a word boundary — and since
the "▁" marks live in the tokens, decode is exact: join, turn "▁" back into
spaces, drop the dummy prefix. Nothing relies on a language-specific
pre-tokenizer, so the same recipe works for Japanese/Chinese/Thai where
spaces don't separate words (there the whole run is one piece and merges
decide the splits). Used by T5, ALBERT, XLNet, Llama (2018+).

Pros: no external/language-specific pre-tokenizer, fully reversible
(lossless round-trip incl. leading/repeated spaces), consistent across
languages.
Cons: same greedy-frequency merge limitation as BPE (see 05); "▁" variants
("▁the" vs "the") duplicate some vocab entries.
"""

import re
from collections import Counter

SPACE_MARKER = "▁"  # ▁
# a piece = run of ▁ followed by non-▁ chars (a "word"), or a bare ▁ run
PIECE = re.compile(f"{SPACE_MARKER}*[^{SPACE_MARKER}]+|{SPACE_MARKER}+")


class SentencePieceTokenizer:
    """BPE inside ▁-delimited pieces; whitespace preserved as a symbol."""

    def __init__(self):
        self.merges: dict[tuple[str, str], str] = {}
        self.vocab: set[str] = set()

    def _pieces(self, text: str) -> list[str]:
        """Dummy-prefix + space->▁, then cut into ▁-led pieces."""
        marked = SPACE_MARKER + text.replace(" ", SPACE_MARKER)
        return PIECE.findall(marked)

    def _merge(self, symbols: tuple, pair: tuple[str, str], merged: str) -> tuple:
        out, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                out.append(merged)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        return tuple(out)

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        piece_freq = Counter(self._pieces(text))
        splits = {p: tuple(p) for p in piece_freq}
        self.vocab = {s for syms in splits.values() for s in syms}

        while len(self.vocab) < vocab_size:
            counts = Counter()
            for piece, freq in piece_freq.items():
                syms = splits[piece]
                for pair in zip(syms, syms[1:]):
                    counts[pair] += freq
            if not counts:
                break
            pair = max(counts, key=counts.get)
            merged = pair[0] + pair[1]
            splits = {p: self._merge(s, pair, merged) for p, s in splits.items()}
            self.merges[pair] = merged
            self.vocab.add(merged)
            if verbose:
                print(f"merge: {pair!r} -> {merged!r} ({counts[pair]} occurrences)")

    def encode(self, text: str) -> list[str]:
        merge_order = {pair: i for i, pair in enumerate(self.merges)}
        tokens = []
        for piece in self._pieces(text):
            syms = tuple(piece)
            while len(syms) >= 2:
                candidates = [p for p in zip(syms, syms[1:]) if p in merge_order]
                if not candidates:
                    break
                pair = min(candidates, key=merge_order.get)
                syms = self._merge(syms, pair, self.merges[pair])
            tokens.extend(syms)
        return tokens

    def decode(self, tokens: list[str]) -> str:
        """Join, ▁ -> space, drop the dummy prefix space — exact round-trip."""
        return "".join(tokens).replace(SPACE_MARKER, " ")[1:]


if __name__ == "__main__":
    corpus = "the quick brown fox jumps over the lazy dog the fox runs"
    tok = SentencePieceTokenizer()
    tok.train(corpus, vocab_size=45, verbose=True)

    sample = "the fox jumps"
    tokens = tok.encode(sample)
    print("\nsample:", sample)
    print("tokens:", tokens)
    print("decoded:", repr(tok.decode(tokens)))
    assert tok.decode(tokens) == sample
    assert all(SPACE_MARKER not in t[1:] for t in tokens)  # no token spans a word boundary

    # lossless even for odd whitespace and unseen text — plain split()-based tokenizers can't do this
    for odd in ["  two  spaces ", "", "unseen zebra"]:
        assert tok.decode(tok.encode(odd)) == odd
    print("round-trips leading/repeated/trailing spaces and unseen words exactly")
    print(f"word-initial tokens carry '{SPACE_MARKER}': whitespace is a symbol, not a pre-split rule")
