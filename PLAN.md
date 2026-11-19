# From-ScratchLLM Plan

Build LLM components from scratch, stdlib for tokenization, numpy from 02 onward (`requirements.txt`), one numbered folder per stage. Each file: docstring explaining method + why, runnable `__main__` demo with assert.

## Status

- [x] `01-tokenization/` — char, whitespace, rule-based, statistical, BPE, WordPiece, SentencePiece, Unigram LM, byte-level BPE, byte/unicode, tiktoken-style, adaptive, BPE-dropout, tokenizer-free patches (14 files)
- [x] `02-embeddings/` — token embedding, positional encoding (sinusoidal, learned)
- [x] `03-attention/` — scaled dot-product, multi-head, causal mask, RoPE, ALiBi, GQA, MQA, sliding window
- [x] `04-transformer-block/` — LayerNorm, RMSNorm, MLP/SwiGLU, MoE, pre-norm block with batching + residuals
- [x] `05-gpt/` — mini-GPT (weight-tied, untrained), sampling (greedy/temperature/top-k/top-p), KV cache + generation
- [x] `06-training/` — hand-written numpy backprop for the GPT (gradient-checked), AdamW, LR schedule, grad clipping, loss-scaling demo, checkpointing, char-level training loop (dense SwiGLU only; MoE backward skipped)
- [x] `07-architectures/` — general attention (cross/bidirectional, with backward), BERT masked-LM, T5-style encoder-decoder; all gradient-checked and trained
- [x] `08-post-training/` — LoRA, SFT (response-only loss), DPO, simplified RLHF (reward model + PPO-clip with KL penalty); all gradient-checked

## Next

- [ ] `09-efficiency/` — quantization, pruning, distillation
- [ ] `10-eval/` — perplexity, benchmark harness

## Backlog (unscheduled)

- RNN/LSTM baseline for comparison with the Transformer
- Flash-attention-style blockwise softmax (memory-efficient attention)
- Tokenizer and model config serialization (save/load)
- Compare these tokenizers against tiktoken / Hugging Face tokenizers
- Multilingual / script-aware tokenizers (CJK, Indic); image and audio tokenizers (stretch)

Design notes:
- Attention files (03) are 2-D `(seq, d)` for clarity; batching is added in `04-transformer-block/` via numpy broadcasting.
- `_merge`-style helpers are intentionally copy-pasted across tokenizer files so each stays self-contained.

Shared math primitives (autograd, optimizers, backprop) live in the separate FromScratchDL repo — import, don't duplicate.
