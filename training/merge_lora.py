"""
Merges the fine-tuned LoRA adapter into the base Qwen3-1.7B model, producing
full merged weights ready for vLLM serving (vLLM can serve LoRA adapters
directly too, but merging keeps the serving image simpler and avoids the
--enable-lora flag / adapter-loading complexity for a single fixed adapter).

Usage (inside the job container, or locally with a GPU):
    python training/merge_lora.py \
        --adapter-dir /mnt/output/foodlens-qwen3-1.7b-lora \
        --output-dir /mnt/output/foodlens-qwen3-1.7b-merged

Output: a full model directory (config.json, tokenizer files, safetensors
weights) that can be loaded by vLLM or transformers directly, no PEFT needed
at serving time.
"""
from __future__ import annotations

import argparse
import logging

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("merge_lora")

BASE_MODEL = "Qwen/Qwen3-1.7B"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", required=True, help="Path to the trained LoRA adapter")
    parser.add_argument("--output-dir", required=True, help="Where to save the merged full model")
    parser.add_argument("--base-model", default=BASE_MODEL)
    args = parser.parse_args()

    log.info("Loading base model: %s", args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype="bfloat16")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    log.info("Loading LoRA adapter from: %s", args.adapter_dir)
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)

    log.info("Merging adapter into base weights")
    merged_model = model.merge_and_unload()

    log.info("Saving merged model to: %s", args.output_dir)
    merged_model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    log.info("Done. Merged model ready for vLLM serving at %s", args.output_dir)


if __name__ == "__main__":
    main()
