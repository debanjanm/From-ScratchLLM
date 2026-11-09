"""Character-level tokenizer. Simplest possible baseline: one token per char."""


class CharTokenizer:
    """Maps each unique character in a corpus to an integer id.

    Method: vocab = sorted set of distinct chars seen in training text.
    No merging, no learned structure — every character is its own token.

    Pros: trivial to build, tiny vocab, zero OOV for chars seen at train time.
    Cons: sequences get long (1 token per char), so the model must learn
    spelling before it can learn meaning. Historically the starting point
    (early char-level RNN LMs) before word/subword methods took over.
    """

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, text: str) -> list[int]:
        """Map each char to its id. Raises KeyError on a char unseen at train time (no <unk>)."""
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        """Map ids back to chars and join — lossless by construction."""
        return "".join(self.itos[i] for i in ids)


if __name__ == "__main__":
    text = "hello world"
    tok = CharTokenizer(text)
    ids = tok.encode(text)
    print("vocab_size:", tok.vocab_size)
    print("ids:", ids)
    print("decoded:", tok.decode(ids))
    assert tok.decode(ids) == text
