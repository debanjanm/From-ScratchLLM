"""Rule-based word tokenizer — regex rules split punctuation/contractions from words.

Method: apply a fixed sequence of regex substitutions (Penn Treebank style)
to separate punctuation, contractions, and clitics from word bodies before
whitespace-splitting. E.g. "don't" -> "do n't", "dog." -> "dog .".
Used in classic NLP pipelines (NLTK's word_tokenize, spaCy's early rules)
through the 1990s-2000s, before statistical/learned methods took over.

Pros: fixes the whitespace tokenizer's punctuation-gluing problem, linguistically
motivated splits (e.g. splitting off possessive "'s", negation "n't").
Cons: rules are hand-written per language, don't generalize (new punctuation
patterns, emojis, hashtags break them), still word-level so vocab is large
and OOV is unbounded — motivates learned statistical/subword methods next.
"""

import re


class RuleBasedTokenizer:
    """Applies ordered regex rules to isolate punctuation/contractions, then splits on whitespace."""

    # order matters: contractions before generic punctuation, else "n't" gets mangled
    RULES = [
        (r"n't\b", " n't"),               # don't -> do n't
        (r"'re\b", " 're"),                # you're -> you 're
        (r"'ve\b", " 've"),                # I've -> I 've
        (r"'ll\b", " 'll"),                # I'll -> I 'll
        (r"'s\b", " 's"),                  # dog's -> dog 's
        (r"([.,!?;:])", r" \1 "),          # split off sentence punctuation
        (r"\s{2,}", " "),                  # collapse extra spaces created above
    ]

    def __init__(self, text: str):
        words = sorted(set(self._split(text)))
        self.stoi = {w: i for i, w in enumerate(words)}
        self.itos = {i: w for i, w in enumerate(words)}
        self.vocab_size = len(words)

    def _split(self, text: str) -> list[str]:
        """Apply regex rules in order, then split on whitespace."""
        for pattern, repl in self.RULES:
            text = re.sub(pattern, repl, text)
        return text.strip().split()

    def encode(self, text: str) -> list[int]:
        return [self.stoi[w] for w in self._split(text)]

    def decode(self, ids: list[int]) -> str:
        """Naive space-join — doesn't re-glue punctuation to words (real detokenizers do)."""
        return " ".join(self.itos[i] for i in ids)


if __name__ == "__main__":
    corpus = "the fox doesn't jump. the dog's bark isn't loud."
    tok = RuleBasedTokenizer(corpus)
    print("vocab_size:", tok.vocab_size)
    print("vocab:", sorted(tok.stoi))

    sample = "the fox doesn't jump."
    ids = tok.encode(sample)
    print("\nsample:", sample)
    print("ids:", ids)
    print("decoded:", tok.decode(ids))

    assert tok.decode(ids) == "the fox does n't jump ."  # contraction + punctuation split off
    # show the fix vs whitespace tokenizer: "jump." and "jump" now share a token
    assert tok.stoi["jump"] != tok.stoi.get(".")
    print("\n'jump' and '.' are now separate tokens (whitespace tokenizer would fuse them)")
