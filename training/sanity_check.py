"""
Fast sanity check for the FoodLens training job environment on Nebius.
Run this BEFORE the full fine-tune job to catch GPU/import/connectivity
issues in under a minute instead of 40 minutes into a real run.
"""
from __future__ import annotations

import argparse
import json
import sys


def check(label: str, fn) -> bool:
    try:
        result = fn()
        print(f"[PASS] {label}: {result}")
        return True
    except Exception as exc:
        print(f"[FAIL] {label}: {type(exc).__name__}: {exc}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", default=None)
    args = parser.parse_args()

    results = []

    def gpu_check():
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False")
        return f"{torch.cuda.get_device_name(0)} ({torch.cuda.device_count()} device(s))"

    results.append(check("GPU visible to PyTorch", gpu_check))

    def imports_check():
        import peft
        import transformers
        import trl
        import datasets
        from trl import SFTConfig, SFTTrainer  # noqa: F401
        return (f"transformers={transformers.__version__}, peft={peft.__version__}, "
                f"trl={trl.__version__}, datasets={datasets.__version__}")

    results.append(check("Required packages import (incl. SFTTrainer)", imports_check))

    def hf_connectivity_check():
        from transformers import AutoTokenizer, AutoConfig
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
        cfg = AutoConfig.from_pretrained("Qwen/Qwen3-1.7B")
        return f"tokenizer vocab_size={tok.vocab_size}, config model_type={cfg.model_type}"

    results.append(check("HuggingFace Hub connectivity + model architecture support", hf_connectivity_check))

    if args.train_file:
        def dataset_check():
            with open(args.train_file) as f:
                lines = [json.loads(line) for i, line in enumerate(f) if i < 5]
            return f"read {len(lines)} sample rows from {args.train_file}"

        results.append(check("Training dataset readable", dataset_check))

    print()
    if all(results):
        print("All checks passed - safe to launch the full fine-tune job.")
        sys.exit(0)
    else:
        print("One or more checks FAILED - fix before launching the full fine-tune job.")
        sys.exit(1)


if __name__ == "__main__":
    main()
