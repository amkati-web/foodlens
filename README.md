# 🔬 FoodLens — Intelligent Food Label Analyzer

Photograph or paste a food ingredient list. FoodLens returns a unified safety report:
**additive safety** (E-numbers), **NOVA ultra-processing classification**, and
**personal allergen / dietary-restriction conflicts** — combining a curated rule
database with a Qwen3-1.7B LoRA fine-tune served on **Nebius Serverless AI**.

Built for the **Nebius Serverless AI Builders Challenge**.

![status](https://img.shields.io/badge/status-MVP-yellow) ![license](https://img.shields.io/badge/license-MIT-blue)

---

## Why this needs fine-tuning, not just prompting

A base LLM doesn't reliably know rare E-numbers, regional additive names, or the
specific NOVA classification heuristics from Monteiro et al. — and it will
confidently hallucinate a safety rating rather than say "I don't know." Three
distinct learned tasks are fine-tuned into a single small model:

| Task | Learns | Source data |
|---|---|---|
| Additive classifier | E-number → safety profile → condition warnings | EFSA OpenFoodTox, EWG, CSPI Chemical Cuisine |
| NOVA classifier | ingredient patterns → processing level (1-4) | Open Food Facts (1M+ products with community `nova_group` labels), Monteiro et al. |
| Hidden allergen detector | "modified starch" ⇒ maybe gluten, "casein" ⇒ dairy | FSAI / EU FIC allergen guidance |

The app itself is **database-first**: known E-numbers and allergen keywords are
answered instantly from a curated local JSON database (`backend/app/data/`) with
zero inference cost. The Nebius-hosted model is only called to **fill gaps** —
unrecognised E-numbers, ambiguous ingredient lists, and uncertain NOVA calls —
so production inference cost stays low while coverage stays high.

## Architecture

```
┌─────────────┐      photo / paste       ┌──────────────────────┐
│  Frontend    │ ───────────────────────▶ │  FastAPI backend      │
│ (static SPA) │ ◀─────────────────────── │  /api/ocr             │
└─────────────┘        JSON report        │  /api/analyze         │
                                           │                        │
                              ┌────────────┼────────────┐           │
                              │  additive_service.py     │           │
                              │  nova_service.py         │──DB hit──▶│  local JSON DB
                              │  allergen_service.py     │           │  (instant, free)
                              │            │  DB miss / low confidence
                              │            ▼
                              │   nebius_client.py ──▶ Nebius Serverless AI Endpoint
                              │                          (fine-tuned Qwen3-1.7B LoRA)
                              └───────────────────────────────────────┘
```

**Nebius Serverless AI usage:**
- **Jobs** — dataset processing (`training/prepare_dataset.py`) and LoRA fine-tuning
  (`training/finetune_lora.py`), submitted via `training/submit_nebius_job.py`
- **Endpoints** — the merged fine-tune served pay-per-request behind an
  OpenAI-compatible API, called from `backend/app/services/nebius_client.py`

## Repository layout

```
backend/            FastAPI app (OCR + analysis API, rule DB, Nebius client)
frontend/            Static SPA (photo capture, chip-based profile, results view)
training/            Dataset prep + LoRA fine-tuning pipeline for Nebius Jobs
scripts/             Adapter merge + Nebius Endpoint deployment
docker/              Dockerfile.api (serving) and Dockerfile.job (training)
.github/workflows/  CI: lint, unit tests, dataset smoke test, Docker build
```

## Quickstart (local, rule-based only — no Nebius account needed)

```bash
git clone <your-fork-url> foodlens && cd foodlens
cp .env.example .env                 # leave NEBIUS_API_KEY blank to run DB-only
pip install -r backend/requirements.txt
# system dependency for OCR:
#   Debian/Ubuntu: sudo apt-get install tesseract-ocr
#   macOS:         brew install tesseract
cd backend && uvicorn app.main:app --reload --port 8000
```

Open `http://localhost:8000` — the FastAPI app also serves the static frontend.
Try pasting: `Water, sugar, modified maize starch, E102, E621, malt extract, colour`

Run the test suite:
```bash
cd backend && pip install -r requirements-dev.txt && pytest tests/ -v
```

## Quickstart (Docker)

```bash
cp .env.example .env
docker compose up --build
# -> http://localhost:8000
```

## Enabling the Nebius-hosted LLM fallback

1. Create a Nebius AI Studio account and API key: https://studio.nebius.com
2. Set in `.env`:
   ```
   NEBIUS_API_KEY=sk-...
   NEBIUS_BASE_URL=https://api.studio.nebius.com/v1
   NEBIUS_LLM_MODEL=Qwen/Qwen3-1.7B
   ```
3. Restart the API. `GET /api/health` will report `"llm_available": true`.

Without a fine-tuned model deployed, this calls the base Qwen3-1.7B model with
strict JSON-mode prompting (still useful, just less consistent on rare
E-numbers/regional naming than the fine-tune below).

## Reproducing the fine-tuning pipeline on Nebius

**Cost/runtime estimate:** dataset prep is a CPU job (~10-20 min depending on
Open Food Facts export size sampled); LoRA fine-tuning of Qwen3-1.7B on a
single H100 runs ~2 hours for 3 epochs over ~15-20k examples.

```bash
# 0. Install & authenticate the Nebius CLI (see https://docs.nebius.com/cli)
pip install nebius-cli
nebius profile create --name foodlens
export NEBIUS_PROJECT_ID=<your-project-id>

# 1. Build & push the training container
docker build -f docker/Dockerfile.job -t <registry>/foodlens-job:latest .
docker push <registry>/foodlens-job:latest

# 2. Dataset processing job (CPU) - combines the local additive DB,
#    committed seed examples, and a sample of Open Food Facts NOVA labels
python training/submit_nebius_job.py --stage prepare --image <registry>/foodlens-job:latest

# 3. LoRA fine-tune job (1x H100)
python training/submit_nebius_job.py --stage finetune --image <registry>/foodlens-job:latest --gpu h100

# 4. Merge the adapter and deploy as a Nebius Serverless AI Endpoint
python scripts/deploy_endpoint.py --merge --adapter-dir ./foodlens-qwen3-1.7b-lora \
    --merged-dir ./foodlens-qwen3-1.7b-merged
python scripts/deploy_endpoint.py --deploy --model-dir ./foodlens-qwen3-1.7b-merged \
    --endpoint-name foodlens-qwen3-1.7b

# 5. Point the API at the fine-tune
echo "NEBIUS_FINE_TUNED_MODEL=foodlens-qwen3-1.7b" >> .env
```

You can smoke-test the dataset step without any network access or GPU:
```bash
python training/prepare_dataset.py --seed-only --out training/dataset/train.jsonl
```
This builds 61 examples from the committed local additive DB + seed JSONL files
(`training/dataset/sample_*.jsonl`) and is what CI runs on every push.

## API reference

### `POST /api/ocr`
`multipart/form-data`, field `file` (image). Returns `{extracted_text, confidence}`.

### `POST /api/analyze`
```json
{
  "ingredient_text": "Water, sugar, E102, malt extract",
  "profile": {
    "conditions": ["adhd"],
    "restrictions": ["gluten_free"]
  },
  "use_llm": true
}
```
Returns additive findings (rating, safety summary, condition-specific warnings),
NOVA classification with explanation, allergen/restriction conflicts, and a
plain-language overall summary.

### `GET /api/health`
Reports whether the Nebius LLM fallback is configured and which model is active.

## Data sources

- **EFSA OpenFoodTox** — E-number safety assessments (seed set of 55 common
  additives committed in `backend/app/data/e_numbers.json`; `training/prepare_dataset.py`
  documents how to expand this from the full EFSA export)
- **Open Food Facts** — 1M+ products with community-curated `nova_group` labels,
  streamed directly in `training/prepare_dataset.py`
- **FSAI / EU Food Information for Consumers Regulation** — allergen hidden-name
  mappings (`backend/app/data/allergen_map.json`)
- **Monteiro et al.** — NOVA classification methodology (`backend/app/data/nova_rules.json`)

## Limitations & responsible use

FoodLens is an informational tool, **not medical advice**. Additive safety
ratings reflect current regulatory/scientific consensus at the time the seed
database was written and may not capture the latest research or your specific
medical situation — always consult a healthcare professional for dietary
decisions related to a diagnosed condition or allergy. OCR accuracy depends on
photo quality; always review extracted text before relying on the analysis.

## License

MIT — see [LICENSE](LICENSE).
