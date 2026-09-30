"""Make the JEV-27B System 1 adapter usable on the multimodal Qwen3.8-27B checkpoint.

autotrust/JEV-27B ships the text-only model. Its language weights are identical to Qwen/Qwen3.8-27B, which also has a
vision encoder. The adapter only needs its layer names moved under `model.language_model.` (the lm_head LoRA that
carries the decision head keeps its name).

    python common/make_mm_adapter.py JEV-27B/adapter_vllm JEV-27B-mm/adapter_vllm
"""
import os, shutil, sys

from safetensors.torch import load_file, save_file

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
sd = load_file(os.path.join(src, "adapter_model.safetensors"))
save_file({k.replace("base_model.model.model.layers.", "base_model.model.model.language_model.layers."): v for k, v in sd.items()},
          os.path.join(dst, "adapter_model.safetensors"))
for f in ("adapter_config.json", "decision_head.json"):
    shutil.copy(os.path.join(src, f), os.path.join(dst, f))
print(f"{len(sd)} tensors -> {dst}")
