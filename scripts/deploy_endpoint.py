"""
Deploys the merged FoodLens LoRA fine-tune as a Nebius Serverless AI Endpoint
(pay-per-request, OpenAI-compatible /v1/chat/completions).

Prerequisites:
    1. training/finetune_lora.py has produced an adapter directory (e.g. from
       a completed `submit_nebius_job.py --stage finetune` run, downloaded
       from the Nebius bucket to ./foodlens-qwen3-1.7b-lora)
    2. Merge the adapter into the base model weights:
           python scripts/deploy_endpoint.py --merge \
               --adapter-dir ./foodlens-qwen3-1.7b-lora \
               --merged-dir ./foodlens-qwen3-1.7b-merged
    3. Deploy the merged model as an endpoint:
           python scripts/deploy_endpoint.py --deploy \
               --model-dir ./foodlens-qwen3-1.7b-merged \
               --endpoint-name foodlens-qwen3-1.7b

Once deployed, set NEBIUS_FINE_TUNED_MODEL in .env to the endpoint's model
name/ID and restart the API - app/services/nebius_client.py will
automatically prefer it over the base model (see Settings.active_model).
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys


def merge_adapter(adapter_dir: str, merged_dir: str, base_model: str) -> None:
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    print(f"Loading base model + LoRA adapter from {adapter_dir} ...")
    model = AutoPeftModelForCausalLM.from_pretrained(adapter_dir)
    merged = model.merge_and_unload()
    merged.save_pretrained(merged_dir)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.save_pretrained(merged_dir)
    print(f"Merged model saved to {merged_dir}")


def deploy(model_dir: str, endpoint_name: str, dry_run: bool) -> None:
    cmd = [
        "nebius", "ai", "endpoint", "create",
        "--name", endpoint_name,
        "--model-path", model_dir,
        "--serving-engine", "vllm",
        "--scale-to-zero",
    ]
    print("Running:\n  " + " ".join(shlex.quote(c) for c in cmd))
    if dry_run:
        return
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--adapter-dir", default="./foodlens-qwen3-1.7b-lora")
    parser.add_argument("--merged-dir", default="./foodlens-qwen3-1.7b-merged")
    parser.add_argument("--model-dir", default="./foodlens-qwen3-1.7b-merged")
    parser.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--endpoint-name", default="foodlens-qwen3-1.7b")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.merge and not args.deploy:
        parser.error("Pass --merge and/or --deploy")

    if args.merge:
        merge_adapter(args.adapter_dir, args.merged_dir, args.base_model)
    if args.deploy:
        deploy(args.model_dir, args.endpoint_name, args.dry_run)


if __name__ == "__main__":
    main()
