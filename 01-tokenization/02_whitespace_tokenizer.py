"""Whitespace tokenizer — split text purely on whitespace, one token per word.

Method: `text.split()`. That's it. No punctuation handling, no normalization.
Historically the first step up from char-level (pre-1990s / early rule-based
NLP pipelines) — treats "word" as the atomic unit of meaning.

Pros: fast, simple, short sequences (1 token/word vs 1 token/char).
Cons: "dog," and "dog" become different tokens (punctuation glued to word),
contractions ("don't") stay as one token, huge vocab (every inflection/typo
is a new token), unbounded OOV on unseen words. Rule-based tokenizers (next
stage) exist specifically to fix the punctuation problem.
"""


class WhitespaceTokenizer:
    """Vocab = every distinct whitespace-delimited chunk seen in training text."""

    def __init__(self, text: str):
        words = sorted(set(text.split()))
        self.stoi = {w: i for i, w in enumerate(words)}
        self.itos = {i: w for i, w in enumerate(words)}
        self.vocab_size = len(words)
        self.unk_id = None  # no <unk> in this minimal version — see encode()

    def encode(self, text: str) -> list[int]:
        """Split on whitespace, map each chunk to id. Raises KeyError on unseen word."""
        return [self.stoi[w] for w in text.split()]

    def decode(self, ids: list[int]) -> str:
        """Join tokens back with single spaces — NOT guaranteed to recover original
        spacing/punctuation-adjacency, since split() discards whitespace info."""
        return " ".join(self.itos[i] for i in ids)


if __name__ == "__main__":
    corpus = "the quick brown fox jumps over the lazy dog"
    tok = WhitespaceTokenizer(corpus)
    print("vocab_size:", tok.vocab_size)

    sample = "the fox jumps"
    ids = tok.encode(sample)
    print("ids:", ids)
    print("decoded:", tok.decode(ids))
    assert tok.decode(ids) == sample

    # show the punctuation problem this method has, motivating rule-based tokenization
    punct_sample = "the fox jumps."
    try:
        tok.encode(punct_sample)
    except KeyError as e:
        print(f"\nfails on punctuation-glued word: {e} "
              f"('jumps.' != 'jumps' — different token entirely)")
