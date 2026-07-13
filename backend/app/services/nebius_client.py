"""
Thin wrapper around Nebius AI Studio's OpenAI-compatible /v1 API.

Nebius Serverless AI Endpoints expose an OpenAI-compatible chat completions
API, so we reuse the `openai` SDK pointed at Nebius's base URL. This client is
used for:
  - filling gaps in the local E-number / NOVA / allergen databases
  - (once trained) calling the fine-tuned Qwen3-1.7B LoRA endpoint deployed
    via `training/submit_nebius_job.py` + Nebius Endpoints

If NEBIUS_API_KEY is not set, calls fail closed (return None) so the rest of
the app degrades gracefully to rule-based-only results.
"""
from __future__ import annotations

import json
import logging

from openai import APIError, AsyncOpenAI

from app.config import get_settings

logger = logging.getLogger("foodlens.nebius")

_settings = get_settings()

_client: AsyncOpenAI | None = None
if _settings.nebius_api_key:
    _client = AsyncOpenAI(
        api_key=_settings.nebius_api_key,
        base_url=_settings.nebius_base_url,
    )
else:
    logger.warning(
        "NEBIUS_API_KEY not set - LLM-augmented analysis is disabled, "
        "falling back to rule-based/database-only results."
    )


async def chat_json(system_prompt: str, user_prompt: str, *, temperature: float = 0.1) -> dict | None:
    """Call the Nebius-hosted model and parse a strict-JSON response.

    Returns None on any failure (missing key, network error, bad JSON) so
    callers can fall back to rule-based logic instead of crashing the request.
    """
    if _client is None:
        return None

    try:
        response = await _client.chat.completions.create(
            model=_settings.active_model,
            temperature=temperature,
            max_tokens=800,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
    except APIError as exc:
        logger.error("Nebius API error: %s", exc)
        return None
    except Exception:  # noqa: BLE001 - defensive: never let LLM issues break the request
        logger.exception("Unexpected error calling Nebius endpoint")
        return None

    content = response.choices[0].message.content
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Nebius model returned non-JSON content: %r", content)
        return None


def is_available() -> bool:
    return _client is not None
