"""
Submits FoodLens training steps as Nebius Serverless AI Jobs, and prints the
deploy command for hosting the fine-tuned model as a Nebius Serverless AI
Endpoint.

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
    python3 training/submit_nebius_job.py --stage prepare --image <registry>/foodlens-job:latest

    # Step 2: LoRA fine-tune on a single H100 (~2h for 3 epochs on the seed+OFF dataset)
    python3 training/submit_nebius_job.py --stage finetune --image <registry>/foodlens-job:latest \
        --gpu h100 --gpu-count 1

    # Step 3: merge the LoRA adapter into full weights (CPU or small GPU job)
    python3 training/submit_nebius_job.py --stage merge --image <registry>/foodlens-job:latest

    # Step 4: print the endpoint deploy command (does NOT execute it - endpoints
    # bill continuously while running, so this is reviewed and run manually)
    python3 training/submit_nebius_job.py --stage deploy --image <registry>/foodlens-serve:latest

Each stage prints the exact `nebius` command it will run (and runs it, unless
--dry-run is passed, EXCEPT for the deploy stage which always only prints -
see above) so the invocation is auditable and reproducible from the README
alone, even without this script.
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
    "merge": [
        "python", "training/merge_lora.py",
        "--adapter-dir", "/mnt/output/foodlens-qwen3-1.7b-lora",
        "--output-dir", "/mnt/output/foodlens-qwen3-1.7b-merged",
    ],
}

GPU_PRESETS = {
    "none": [],
    "l40s": ["--resource", "gpu-l40s-1"],
    "h100": ["--resource", "gpu-h100-1"],
}


def print_deploy_command(image: str, bucket: str, subnet_id: str) -> None:
    """Print (never execute) the nebius ai endpoint create command.

    Endpoints run continuously and bill for as long as they're up, unlike
    jobs which stop billing once finished - so this is always reviewed and
    run manually, never auto-executed by this script.
    """
    cmd = [
        "nebius", "ai", "endpoint", "create",
        "--name", "foodlens-endpoint",
        "--image", image,
        "--platform", "gpu-l40s-a",
        "--preset", "1gpu-16vcpu-64gb",
        "--container-port", "8000",
        "--volume", f"{bucket}:/mnt/models:ro",
        "--subnet-id", subnet_id,
        "--public",
    ]
    print("Deploy command (review before running - this creates a continuously")
    print("billed endpoint, so it is never auto-executed by this script):")
    print("")
    print("  " + " ".join(shlex.quote(c) for c in cmd))
    print("")
    print("Once running, check status with:")
    print("  nebius ai endpoint get <endpoint_id>")
    print("")
    print("Stop it when not in use to pause billing:")
    print("  nebius ai endpoint stop <endpoint_id>")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=[*STAGE_COMMANDS.keys(), "deploy"], required=True)
    parser.add_argument("--image", required=True, help="Container image (job image for prepare/finetune/merge, serve image for deploy)")
    parser.add_argument("--gpu", choices=GPU_PRESETS.keys(), default="none",
                         help="'none' for CPU-only stages, 'h100' or 'l40s' for finetune")
    parser.add_argument("--project-id", default=os.environ.get("NEBIUS_PROJECT_ID", ""))
    parser.add_argument("--bucket", default=os.environ.get("NEBIUS_BUCKET", "foodlens-datasets"))
    parser.add_argument("--subnet-id", default=os.environ.get("NEBIUS_SUBNET_ID", ""))
    parser.add_argument("--dry-run", action="store_true", help="Print the command without executing it")
    args = parser.parse_args()

    if args.stage == "deploy":
        print_deploy_command(args.image, args.bucket, args.subnet_id)
        return

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

    print("Running:")
    print("  " + " ".join(shlex.quote(c) for c in cmd))
    if args.dry_run:
        return

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
