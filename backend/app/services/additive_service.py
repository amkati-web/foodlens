from __future__ import annotations

import json
import re
from pathlib import Path

from app.models.schemas import AdditiveFinding, UserProfile
from app.services import nebius_client

_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "e_numbers.json"
_E_NUMBER_DB: dict = json.loads(_DATA_PATH.read_text())

_E_NUMBER_RE = re.compile(r"\bE\s?-?\s?(\d{3}[a-zA-Z]?)\b", re.IGNORECASE)


def _normalise_code(raw: str) -> str:
    digits = re.sub(r"[^0-9a-zA-Z]", "", raw)
    return f"E{digits.upper()}"


def extract_e_numbers(text: str) -> list[str]:
    codes = set()
    for match in _E_NUMBER_RE.finditer(text):
        codes.add(_normalise_code(match.group(1)))
    return sorted(codes)


def _lookup_local(code: str) -> AdditiveFinding | None:
    entry = _E_NUMBER_DB.get(code)
    if not entry:
        return None
    return AdditiveFinding(
        code=code,
        name=entry["name"],
        category=entry["category"],
        origin=entry["origin"],
        rating=entry["rating"],
        summary=entry["summary"],
        condition_warnings=entry.get("conditions", {}),
        source="database",
    )


_LLM_SYSTEM_PROMPT = """You are a food-additive safety classifier trained on EFSA, EWG, and CSPI \
Chemical Cuisine data. Given a single E-number code, respond ONLY with a JSON object with keys: \
name (string), category (string, e.g. 'colour', 'preservative', 'sweetener', 'emulsifier', \
'flavour enhancer', 'acidity regulator', 'thickener'), origin (string, e.g. 'natural', 'synthetic'), \
rating (one of 'green','amber','red' for low/moderate/high concern), summary (1-2 sentence factual \
safety summary), condition_warnings (object mapping any of 'diabetes','adhd','allergies', \
'hypertension','pregnancy','pku' to a short warning string, omit keys with no relevant warning). \
If you do not recognise the code, respond with {"unknown": true}."""


async def _lookup_llm(code: str) -> AdditiveFinding | None:
    result = await nebius_client.chat_json(_LLM_SYSTEM_PROMPT, f"E-number: {code}")
    if not result or result.get("unknown"):
        return None
    try:
        return AdditiveFinding(
            code=code,
            name=result["name"],
            category=result.get("category", "unknown"),
            origin=result.get("origin", "unknown"),
            rating=result.get("rating", "unknown"),
            summary=result.get("summary", ""),
            condition_warnings=result.get("condition_warnings", {}),
            source="llm",
        )
    except (KeyError, TypeError):
        return None


async def analyze_additives(text: str, profile: UserProfile, use_llm: bool) -> list[AdditiveFinding]:
    codes = extract_e_numbers(text)
    findings: list[AdditiveFinding] = []
    for code in codes:
        finding = _lookup_local(code)
        if finding is None and use_llm:
            finding = await _lookup_llm(code)
        if finding is None:
            finding = AdditiveFinding(
                code=code, name="Unknown additive", category="unknown", origin="unknown",
                rating="unknown", summary="Not found in the local database or LLM fallback.",
                source="database",
            )
        findings.append(finding)

    if profile.conditions:
        allowed = set(profile.conditions)
        for f in findings:
            f.condition_warnings = {
                k: v for k, v in f.condition_warnings.items() if k in allowed
            }
    return findings
