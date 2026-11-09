# From-ScratchLLM Plan

Build LLM components from scratch, stdlib for tokenization, numpy from 02 onward (`requirements.txt`), one numbered folder per stage. Each file: docstring explaining method + why, runnable `__main__` demo with assert.

## Status

- [x] `01-tokenization/` — char, whitespace, rule-based, statistical, BPE, WordPiece, SentencePiece, Unigram LM, byte-level BPE, byte/unicode, tiktoken-style, adaptive, BPE-dropout, tokenizer-free patches (14 files)
- [x] `02-embeddings/` — token embedding, positional encoding (sinusoidal, learned)
- [x] `03-attention/` — scaled dot-product, multi-head, causal mask, RoPE, ALiBi, GQA, MQA, sliding window

## Next

- [ ] `04-transformer-block/` — LayerNorm/RMSNorm, MLP, residuals, full block
- [ ] `05-gpt/` — mini-GPT, KV cache, sampling (greedy/top-k/top-p/temperature)
- [ ] `06-training/` — custom loop, LR schedule, grad clipping, mixed precision, checkpointing
- [ ] `07-architectures/` — encoder-only (BERT), encoder-decoder (T5)
  - MoE is a layer swap (replaces MLP), not an architecture — build it in `04-transformer-block/` instead, wire into 07 examples
- [ ] `08-post-training/` — LoRA, SFT, DPO, simplified RLHF
- [ ] `09-efficiency/` — quantization, pruning, distillation
- [ ] `10-eval/` — perplexity, benchmark harness

Design notes:
- Attention files (03) are 2-D `(seq, d)` for clarity; batching is added in `04-transformer-block/` via numpy broadcasting.
- `_merge`-style helpers are intentionally copy-pasted across tokenizer files so each stays self-contained.

Shared math primitives (autograd, optimizers, backprop) live in the separate FromScratchDL repo — import, don't duplicate.
