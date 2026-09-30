#!/usr/bin/env bash
# Serve autotrust/JEV-27B with vLLM: one engine, both systems.
#   System 2 = model "autotrust/JEV-27B" (the unmodified Qwen3.8-27B: chat / reasoning)
#   System 1 = model "jev-decision"     (LoRA + 24-slot decision head expressed as an lm_head LoRA)
# ~54 GB of weights; one 80 GB+ GPU (H100 / H200 / B200). Start-up takes 3-8 minutes (CUDA graphs with LoRA).
set -e
MODEL_DIR=${MODEL_DIR:-JEV-27B}
[ -d "$MODEL_DIR" ] || hf download autotrust/JEV-27B --local-dir "$MODEL_DIR"
exec vllm serve "$MODEL_DIR" --served-model-name autotrust/JEV-27B \
  --enable-lora --max-lora-rank 32 --lora-modules jev-decision="$MODEL_DIR/adapter_vllm" \
  --logprobs-mode processed_logprobs --max-model-len 16384 \
  --enable-prefix-caching --mamba-cache-mode align --port ${PORT:-8000}
