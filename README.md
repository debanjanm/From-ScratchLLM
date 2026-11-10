# From-ScratchLLM
LLM components built from scratch, one numbered folder per stage. Each file explains the method and runs a self-checking demo.

- `01-tokenization/` — char, whitespace, BPE, WordPiece, SentencePiece, Unigram, byte-level BPE, tiktoken-style, and more
- `02-embeddings/` — token embeddings, positional encoding
- `03-attention/` — scaled dot-product, causal mask, multi-head, RoPE, ALiBi, GQA, sliding window
- `04-transformer-block/` — LayerNorm, RMSNorm, MLP/SwiGLU, MoE, full pre-norm block

Setup: `pip install -r requirements.txt` (numpy; tokenization is stdlib-only). Run any file directly: `python3 03-attention/01_scaled_dot_product.py`.

Roadmap: mini-GPT, training, architectures, post-training, efficiency, eval. See [PLAN.md](PLAN.md).
