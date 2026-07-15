"""
LoRA fine-tune of Qwen3-1.7B on the FoodLens instruction dataset.

Runs as the entrypoint of docker/Dockerfile.job on a Nebius Serverless AI Job
(single H100 or L40S is enough for a 1.7B model at this LoRA rank; ~2 hours
for 3 epochs over the full Open Food Facts-augmented dataset).

Usage (inside the job container, or locally with a GPU):
    python training/finetune_lora.py \
        --train-file training/dataset/train.jsonl \
        --val-file training/dataset/train_val.jsonl \
        --output-dir /mnt/output/foodlens-qwen3-1.7b-lora \
        --epochs 3

Output: a LoRA adapter directory that gets merged and deployed as a Nebius
Serverless AI Endpoint (see submit_nebius_job.py / README.md "Deploying the
fine-tuned model").
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("finetune_lora")

BASE_MODEL = "Qwen/Qwen3-1.7B"


def build_prompt(example: dict, tokenizer) -> str:
    """Renders a {"messages": [...]} example using the model's chat template."""
    return tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file", default=None)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--bf16", action="store_true", default=True)
    args = parser.parse_args()

    # Imports deferred so `--help` and dataset-only smoke tests don't require torch/CUDA.
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    log.info("Loading tokenizer + base model: %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    log.info("Loading dataset: %s", args.train_file)
    data_files = {"train": args.train_file}
    if args.val_file:
        data_files["validation"] = args.val_file
    dataset = load_dataset("json", data_files=data_files)

    def format_example(example):
        return {"text": build_prompt(example, tokenizer)}

    dataset = dataset.map(format_example, remove_columns=dataset["train"].column_names)

    local_output_dir = "/workspace/local_checkpoint"
    os.makedirs(local_output_dir, exist_ok=True)
    sft_config = SFTConfig(
        output_dir=local_output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        bf16=args.bf16,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if args.val_file else "no",
        dataset_text_field="text",
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
    )

    log.info("Starting training: %d epochs", args.epochs)
    trainer.train()

    log.info("Saving LoRA adapter locally to %s", local_output_dir)
    trainer.save_model(local_output_dir)
    tokenizer.save_pretrained(local_output_dir)
    log.info("Copying LoRA adapter to mounted output dir %s", args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    shutil.copytree(local_output_dir, args.output_dir, dirs_exist_ok=True)
    log.info("Done. Merge + deploy with training/submit_nebius_job.py --stage deploy")


if __name__ == "__main__":
    main()
