"""WordPiece tokenizer (BERT-style) — merges by likelihood gain, not raw frequency.

Method: pre-split text into words (whitespace). Each word starts as a
sequence of characters, with all but the first prefixed "##" to mark
"continues previous token" (so subwords can be reassembled unambiguously).
Unlike BPE (which merges the MOST FREQUENT pair), WordPiece merges the pair
that gives the highest *likelihood gain*:

    score(a, b) = count(a, b) / (count(a) * count(b))

This favors merging pairs that co-occur more than chance would predict, even
if their raw frequency is lower than some other pair — a rare pair that
ALWAYS appears together scores higher than a common pair that also appears
apart often. Used by BERT, DistilBERT, Electra (2018+).

Pros: likelihood-based merges tend to produce more linguistically coherent
subwords than raw-frequency BPE; explicit ## marks word-continuation so
detokenization knows where word boundaries are.
Cons: still needs a pre-tokenizer (whitespace split) — doesn't handle
languages without spaces; encoding requires greedy longest-match search
(slower than BPE's merge replay).
"""

from collections import Counter


class WordPieceTokenizer:
    """Vocab built via likelihood-maximizing merges over ##-marked character sequences."""

    def __init__(self):
        self.vocab: set[str] = set()

    def _word_to_symbols(self, word: str) -> list[str]:
        """First char stays bare, rest get '##' prefix: 'dog' -> ['d', '##o', '##g']."""
        return [word[0]] + [f"##{c}" for c in word[1:]]

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        """Iteratively merge the pair with highest co-occurrence likelihood."""
        words = text.split()
        word_freq = Counter(words)
        # each distinct word tracked as a mutable symbol sequence
        splits = {w: self._word_to_symbols(w) for w in word_freq}

        self.vocab = {sym for symbols in splits.values() for sym in symbols}

        while len(self.vocab) < vocab_size:
            pair_counts, symbol_counts = Counter(), Counter()
            for word, freq in word_freq.items():
                symbols = splits[word]
                for sym in symbols:
                    symbol_counts[sym] += freq
                for a, b in zip(symbols, symbols[1:]):
                    pair_counts[(a, b)] += freq

            if not pair_counts:
                break

            # score = likelihood gain, not raw frequency — the key WordPiece difference
            def score(pair):
                a, b = pair
                return pair_counts[pair] / (symbol_counts[a] * symbol_counts[b])

            best = max(pair_counts, key=score)
            a, b = best
            merged = a + b[2:] if b.startswith("##") else a + b  # strip ## when fusing

            for word in splits:
                symbols = splits[word]
                new_symbols, i = [], 0
                while i < len(symbols):
                    if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                        new_symbols.append(merged)
                        i += 2
                    else:
                        new_symbols.append(symbols[i])
                        i += 1
                splits[word] = new_symbols

            self.vocab.add(merged)
            if verbose:
                print(f"merge: {a!r} + {b!r} -> {merged!r} (score={score(best):.4f})")

        self._splits = splits  # cache for reference

    def _tokenize_word(self, word: str) -> list[str]:
        """Greedy longest-match-first: find longest vocab prefix, repeat on remainder."""
        symbols = []
        start = 0
        while start < len(word):
            end = len(word)
            match = None
            while end > start:
                piece = word[start:end]
                candidate = piece if start == 0 else f"##{piece}"
                if candidate in self.vocab:
                    match = candidate
                    break
                end -= 1
            if match is None:
                symbols.append("[UNK]")
                break
            symbols.append(match)
            start = end
        return symbols

    def encode(self, text: str) -> list[str]:
        tokens = []
        for word in text.split():
            tokens.extend(self._tokenize_word(word))
        return tokens

    def decode(self, tokens: list[str]) -> str:
        """Strip '##' and fuse continuations back onto the preceding word."""
        words, current = [], ""
        for tok in tokens:
            if tok.startswith("##"):
                current += tok[2:]
            else:
                if current:
                    words.append(current)
                current = tok
        if current:
            words.append(current)
        return " ".join(words)


if __name__ == "__main__":
    corpus = "the dog runs the dog jumps the dog barks the fox runs"
    tok = WordPieceTokenizer()
    tok.train(corpus, vocab_size=25, verbose=True)
    print("\nvocab:", sorted(tok.vocab))

    sample = "the dog runs"
    tokens = tok.encode(sample)
    print("\nsample:", sample)
    print("tokens:", tokens)
    print("decoded:", tok.decode(tokens))
    assert tok.decode(tokens) == sample
