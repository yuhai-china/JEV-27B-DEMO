#!/usr/bin/env bash
# Serve JEV-27B with image input: one engine, both systems, text and images.
#   base    Qwen/Qwen3.8-27B (multimodal; its language weights are identical to autotrust/JEV-27B)
#   System 1 = model "jev-decision"      JEV adapter + decision head, layer names moved to the language model
#   System 2 = model "autotrust/JEV-27B" the unmodified Qwen3.8-27B, now with vision
# Notes:
#   --trust-request-chat-template  jev_client.decide_mm renders the raw decision prompt with its own template
#   --max-num-seqs 8               works around a vLLM LoRA batching issue with the multimodal model class
#                                  (wrong System 1 outputs when more than 8 sequences share a batch)
#   --enforce-eager                optional; System 1 is prefill-only, so CUDA graphs barely matter for it
set -e
BASE=${BASE:-Qwen3.8-27B}
JEV=${JEV:-JEV-27B}
[ -d "$BASE" ] || hf download Qwen/Qwen3.8-27B --local-dir "$BASE"
[ -d "$JEV" ] || hf download autotrust/JEV-27B --local-dir "$JEV" --include "adapter_vllm/*" "calibration.json"
[ -f "$JEV-mm/adapter_vllm/adapter_model.safetensors" ] || python "$(dirname "$0")/make_mm_adapter.py" "$JEV/adapter_vllm" "$JEV-mm/adapter_vllm"
exec vllm serve "$BASE" --served-model-name autotrust/JEV-27B \
  --enable-lora --max-lora-rank 32 --lora-modules jev-decision="$JEV-mm/adapter_vllm" \
  --logprobs-mode processed_logprobs --max-model-len 16384 --enable-prefix-caching --mamba-cache-mode align \
  --limit-mm-per-prompt '{"image": 8}' --max-num-seqs 8 --enforce-eager --trust-request-chat-template \
  --port ${PORT:-8000}
