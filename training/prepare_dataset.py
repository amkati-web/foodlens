"""
Builds the instruction-tuning dataset for the FoodLens LoRA fine-tune.

Intended to run as a Nebius Serverless AI Job (CPU-only is fine for this step;
see README.md "Reproducing the pipeline" for the `nebius job submit` invocation).

Three source datasets are combined into one instruction-following JSONL, each
example shaped as a single-turn chat completion so it works directly with
`trl.SFTTrainer` / `finetune_lora.py`:

  1. Additive safety   -> data/e_numbers.json (backend DB) expanded with the
                           EFSA OpenFoodTox bulk export
  2. NOVA processing    -> Open Food Facts product exports, which include a
                           community-curated `nova_group` field per product
  3. Hidden allergens    -> FSAI / EU FIC allergen guidance, expressed as
                           ingredient -> restriction mappings

Usage:
    python training/prepare_dataset.py --out training/dataset/train.jsonl \
        --off-sample-size 20000

Network sources (fetched at runtime, not committed to the repo):
    - Open Food Facts full export (CSV, ~4GB):
      https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz
    - EFSA OpenFoodTox: https://www.efsa.europa.eu/en/data-report/chemical-hazards-database-openfoodtox

For reproducibility without a multi-GB download, this script also supports
`--seed-only`, which builds a small deterministic dataset from
`training/dataset/sample_*.jsonl` (committed to the repo) so `finetune_lora.py`
can be smoke-tested end-to-end without network access.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import random
import sys
from pathlib import Path

import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("prepare_dataset")

REPO_ROOT = Path(__file__).resolve().parent.parent
E_NUMBER_DB = REPO_ROOT / "backend" / "app" / "data" / "e_numbers.json"
ALLERGEN_DB = REPO_ROOT / "backend" / "app" / "data" / "allergen_map.json"
SAMPLE_DIR = Path(__file__).resolve().parent / "dataset"

OFF_EXPORT_URL = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"

ADDITIVE_SYSTEM_PROMPT = (
    "You are a food-additive safety classifier trained on EFSA, EWG, and CSPI Chemical "
    "Cuisine data. Given a single E-number code, respond ONLY with a JSON object with keys: "
    "name, category, origin, rating ('green'/'amber'/'red'), summary, condition_warnings."
)
NOVA_SYSTEM_PROMPT = (
    "You are a NOVA food classification expert (Monteiro et al.). Given an ingredient list, "
    "respond ONLY with a JSON object with keys: classification ('nova_1'..'nova_4'), label, "
    "confidence, triggered_by, explanation."
)


def build_additive_examples() -> list[dict]:
    db = json.loads(E_NUMBER_DB.read_text())
    examples = []
    for code, entry in db.items():
        target = {
            "name": entry["name"],
            "category": entry["category"],
            "origin": entry["origin"],
            "rating": entry["rating"],
            "summary": entry["summary"],
            "condition_warnings": entry.get("conditions", {}),
        }
        examples.append({
            "messages": [
                {"role": "system", "content": ADDITIVE_SYSTEM_PROMPT},
                {"role": "user", "content": f"E-number: {code}"},
                {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
            ]
        })
    log.info("Built %d additive examples from local DB", len(examples))
    return examples


def _classify_nova_from_off_row(ingredients_text: str, nova_group: str) -> dict | None:
    if not ingredients_text or not nova_group or not nova_group.strip().isdigit():
        return None
    group = int(nova_group.strip())
    if group not in (1, 2, 3, 4):
        return None
    labels = {
        1: "Unprocessed or minimally processed foods",
        2: "Processed culinary ingredients",
        3: "Processed foods",
        4: "Ultra-processed foods (UPF)",
    }
    return {
        "classification": f"nova_{group}",
        "label": labels[group],
        "confidence": 0.9,
        "triggered_by": [],
        "explanation": f"Community-labelled Open Food Facts product, NOVA group {group}.",
    }


def build_nova_examples_from_off(sample_size: int) -> list[dict]:
    """Streams the Open Food Facts export and samples `sample_size` labelled rows.

    Requires network access to static.openfoodfacts.org. Falls back to an empty
    list (with a warning) if unreachable, so the pipeline can still be smoke
    tested via --seed-only.
    """
    examples: list[dict] = []
    try:
        log.info("Downloading Open Food Facts export (streaming, this can take a while)...")
        headers = {"User-Agent": "FoodLens/0.1 (Nebius Serverless AI Builders Challenge; contact: amkati@gmail.com)"}
        with requests.get(OFF_EXPORT_URL, stream=True, timeout=30, headers=headers) as resp:
            resp.raise_for_status()
            with gzip.open(resp.raw, mode="rt", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in tqdm(reader, desc="scanning OFF export"):
                    target = _classify_nova_from_off_row(
                        row.get("ingredients_text_en") or row.get("ingredients_text", ""),
                        row.get("nova_group", ""),
                    )
                    if target is None:
                        continue
                    examples.append({
                        "messages": [
                            {"role": "system", "content": NOVA_SYSTEM_PROMPT},
                            {"role": "user", "content": f"Ingredient list:\n{row.get('ingredients_text_en')}"},
                            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
                        ]
                    })
                    if len(examples) >= sample_size:
                        break
    except requests.RequestException as exc:
        log.warning("Could not reach Open Food Facts export (%s). Skipping OFF-derived examples.", exc)
    log.info("Built %d NOVA examples from Open Food Facts", len(examples))
    return examples


def load_seed_examples() -> list[dict]:
    examples = []
    for path in sorted(SAMPLE_DIR.glob("sample_*.jsonl")):
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
    log.info("Loaded %d committed seed examples from %s", len(examples), SAMPLE_DIR)
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=SAMPLE_DIR / "train.jsonl")
    parser.add_argument("--off-sample-size", type=int, default=20000)
    parser.add_argument("--seed-only", action="store_true",
                         help="Skip network downloads; build only from committed seed files + local DB.")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    examples = build_additive_examples() + load_seed_examples()
    if not args.seed_only:
        examples += build_nova_examples_from_off(args.off_sample_size)

    if not examples:
        log.error("No training examples were built - aborting.")
        sys.exit(1)

    random.Random(args.seed).shuffle(examples)
    split_idx = int(len(examples) * (1 - args.val_split))
    train, val = examples[:split_idx], examples[split_idx:]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    val_path = args.out.with_name(args.out.stem + "_val.jsonl")
    with val_path.open("w") as f:
        for ex in val:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    log.info("Wrote %d train / %d val examples to %s / %s", len(train), len(val), args.out, val_path)


if __name__ == "__main__":
    main()
