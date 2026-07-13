"""
Submits FoodLens training steps as Nebius Serverless AI Jobs.

This is a thin wrapper around the `nebius` CLI (https://docs.nebius.com/cli).
Install and authenticate it first:

    pip install nebius-cli   # or the installer from Nebius docs
    nebius profile create --name foodlens
    export NEBIUS_PROJECT_ID=<your-project-id>

Then build & push the job image once:

    docker build -f docker/Dockerfile.job -t <registry>/foodlens-job:latest .
    docker push <registry>/foodlens-job:latest

Usage:
    # Step 1: build the training dataset (CPU job)
    python training/submit_nebius_job.py --stage prepare --image <registry>/foodlens-job:latest

    # Step 2: LoRA fine-tune on a single H100 (~2h for 3 epochs on the seed+OFF dataset)
    python training/submit_nebius_job.py --stage finetune --image <registry>/foodlens-job:latest \
        --gpu h100 --gpu-count 1

Each stage prints the exact `nebius job create` command it will run (and runs
it, unless --dry-run is passed) so the invocation is auditable and reproducible
from the README alone, even without this script.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys

STAGE_COMMANDS = {
    "prepare": [
        "python", "training/prepare_dataset.py",
        "--out", "/mnt/output/dataset/train.jsonl",
        "--off-sample-size", "20000",
    ],
    "finetune": [
        "python", "training/finetune_lora.py",
        "--train-file", "/mnt/output/dataset/train.jsonl",
        "--val-file", "/mnt/output/dataset/train_val.jsonl",
        "--output-dir", "/mnt/output/foodlens-qwen3-1.7b-lora",
        "--epochs", "3",
    ],
}

GPU_PRESETS = {
    "none": [],
    "l40s": ["--resource", "gpu-l40s-1"],
    "h100": ["--resource", "gpu-h100-1"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=STAGE_COMMANDS.keys(), required=True)
    parser.add_argument("--image", required=True, help="Container image pushed from docker/Dockerfile.job")
    parser.add_argument("--gpu", choices=GPU_PRESETS.keys(), default="none",
                         help="'none' for the CPU-only prepare stage, 'h100' or 'l40s' for finetune")
    parser.add_argument("--project-id", default=os.environ.get("NEBIUS_PROJECT_ID", ""))
    parser.add_argument("--bucket", default=os.environ.get("NEBIUS_BUCKET", "foodlens-datasets"))
    parser.add_argument("--dry-run", action="store_true", help="Print the command without executing it")
    args = parser.parse_args()

    if not args.project_id:
        print("NEBIUS_PROJECT_ID not set (env var or --project-id). See README setup steps.", file=sys.stderr)
        sys.exit(1)

    command_override = " ".join(shlex.quote(part) for part in STAGE_COMMANDS[args.stage])

    cmd = [
        "nebius", "job", "create",
        "--project-id", args.project_id,
        "--name", f"foodlens-{args.stage}",
        "--image", args.image,
        "--mount", f"bucket={args.bucket},path=/mnt/output",
        "--command", command_override,
        *GPU_PRESETS[args.gpu],
    ]

    print("Running:\n  " + " ".join(shlex.quote(c) for c in cmd))
    if args.dry_run:
        return

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
