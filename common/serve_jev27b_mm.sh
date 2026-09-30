#!/usr/bin/env bash
# Serve autotrust/JEV-27B-VL (JEV-27B with vision): one engine, both systems, text and images.
#   System 1 = model "jev-decision"      (JEV adapter + decision head on the multimodal Qwen3.8-27B)
#   System 2 = model "autotrust/JEV-27B" (the unmodified Qwen3.8-27B with vision; name kept so jev_client works unchanged)
# Notes:
#   --max-num-seqs 8               required: with more than 8 sequences in a batch, vLLM's LoRA path for this multimodal
#                                  model class returns wrong System 1 probabilities
#   --trust-request-chat-template  jev_client.decide_mm renders the raw decision prompt with its own template
# To build the same thing from Qwen/Qwen3.8-27B + autotrust/JEV-27B yourself, see common/make_mm_adapter.py.
set -e
MODEL_DIR=${MODEL_DIR:-JEV-27B-VL}
[ -d "$MODEL_DIR" ] || hf download autotrust/JEV-27B-VL --local-dir "$MODEL_DIR"
exec vllm serve "$MODEL_DIR" --served-model-name autotrust/JEV-27B \
  --enable-lora --max-lora-rank 32 --lora-modules jev-decision="$MODEL_DIR/adapter_vllm" \
  --logprobs-mode processed_logprobs --max-model-len 16384 --enable-prefix-caching --mamba-cache-mode align \
  --limit-mm-per-prompt '{"image": 8}' --max-num-seqs 8 --trust-request-chat-template \
  --port ${PORT:-8000}
