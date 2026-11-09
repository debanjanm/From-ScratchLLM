"""Statistical word tokenizer — vocab capped by frequency, rare words -> <unk>.

Method: split words with rule-based regex (reuse prior stage's splitter),
count frequency of every word type across the corpus, keep only the top-K
most frequent as vocab, map everything else to a single <unk> token at
encode time. This is the standard "closed vocabulary" approach used by
pre-neural statistical language models (n-gram LMs) and early neural word
embedding models (word2vec, GloVe, early seq2seq) in the 2000s-2013 era.

Pros: bounded, predictable vocab size regardless of corpus size.
Cons: <unk> throws away information — every rare/unseen word (typos, names,
numbers, new terms) becomes indistinguishable "<unk>". This information loss
is exactly what motivated subword methods (BPE/WordPiece): break rare words
into known pieces instead of discarding them.
"""

import re
from collections import Counter

UNK = "<unk>"

# reuse the same punctuation/contraction splitting rules as the rule-based stage
_RULES = [
    (r"n't\b", " n't"),
    (r"'re\b", " 're"),
    (r"'ve\b", " 've"),
    (r"'ll\b", " 'll"),
    (r"'s\b", " 's"),
    (r"([.,!?;:])", r" \1 "),
    (r"\s{2,}", " "),
]


def _split(text: str) -> list[str]:
    for pattern, repl in _RULES:
        text = re.sub(pattern, repl, text)
    return text.strip().split()


class StatisticalTokenizer:
    """Vocab = top `max_vocab_size` most frequent words; everything else -> <unk>."""

    def __init__(self, text: str, max_vocab_size: int = 20):
        words = _split(text)
        counts = Counter(words)
        # -1 to reserve a slot for <unk>
        most_common = [w for w, _ in counts.most_common(max_vocab_size - 1)]
        vocab = [UNK] + sorted(most_common)

        self.stoi = {w: i for i, w in enumerate(vocab)}
        self.itos = {i: w for i, w in enumerate(vocab)}
        self.vocab_size = len(vocab)
        self.unk_id = self.stoi[UNK]

    def encode(self, text: str) -> list[int]:
        """Unseen/low-frequency words fall back to <unk> instead of raising."""
        return [self.stoi.get(w, self.unk_id) for w in _split(text)]

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.itos[i] for i in ids)


if __name__ == "__main__":
    corpus = ("the fox jumps over the dog. the dog barks at the fox. "
              "the fox runs away quickly into the forest.")
    tok = StatisticalTokenizer(corpus, max_vocab_size=8)
    print("vocab_size:", tok.vocab_size)
    print("vocab:", sorted(tok.stoi))

    sample = "the fox jumps into a river"   # "river" and "a" are rare/unseen -> <unk>
    ids = tok.encode(sample)
    print("\nsample:", sample)
    print("ids:", ids)
    print("decoded:", tok.decode(ids))

    assert ids[3:] == [tok.unk_id] * 3  # "into", "a", "river" all unseen/rare
    assert tok.decode(ids).split()[-1] == UNK

    print(f"\ninformation lost: 'river' and 'a' both collapse to '{UNK}' "
          f"(id {tok.unk_id}) — indistinguishable from each other or any other OOV word")
