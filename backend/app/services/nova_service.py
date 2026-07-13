from __future__ import annotations

import json
from pathlib import Path

from app.models.schemas import NovaResult
from app.services import nebius_client

_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "nova_rules.json"
_NOVA_RULES: dict = json.loads(_DATA_PATH.read_text())


def _rule_based_classify(text: str) -> NovaResult:
    lowered = text.lower()
    triggers = [
        term for term in _NOVA_RULES["nova_4"]["trigger_terms"] if term in lowered
    ]
    ingredient_count = len([p for p in lowered.replace("\n", ",").split(",") if p.strip()])

    if triggers:
        confidence = min(0.55 + 0.08 * len(triggers), 0.92)
        return NovaResult(
            classification="nova_4",
            label=_NOVA_RULES["nova_4"]["label"],
            confidence=round(confidence, 2),
            triggered_by=triggers,
            explanation=(
                f"Detected {len(triggers)} ultra-processing marker(s) "
                f"({', '.join(triggers[:5])}{'...' if len(triggers) > 5 else ''}) "
                "typical of industrial formulations not used in home cooking."
            ),
            source="rules",
        )

    if ingredient_count <= 1:
        return NovaResult(
            classification="nova_1",
            label=_NOVA_RULES["nova_1"]["label"],
            confidence=0.55,
            triggered_by=[],
            explanation="Single ingredient with no processing markers detected.",
            source="rules",
        )

    if ingredient_count <= 3:
        return NovaResult(
            classification="nova_3",
            label=_NOVA_RULES["nova_3"]["label"],
            confidence=0.45,
            triggered_by=[],
            explanation=(
                f"{ingredient_count} recognisable ingredients with no cosmetic/functional "
                "additive markers detected \u2014 consistent with a processed (not ultra-processed) food."
            ),
            source="rules",
        )

    return NovaResult(
        classification="unknown",
        label="Uncertain",
        confidence=0.3,
        triggered_by=[],
        explanation=(
            "No strong ultra-processing markers found, but the ingredient list is long enough "
            "that a confident rule-based call cannot be made. Enable LLM analysis for a better estimate."
        ),
        source="rules",
    )


_LLM_SYSTEM_PROMPT = """You are a NOVA food classification expert (Monteiro et al. classification \
system). Given an ingredient list, respond ONLY with a JSON object with keys: classification (one of \
'nova_1','nova_2','nova_3','nova_4'), label (short name of the group), confidence (0-1 float), \
triggered_by (array of ingredient strings that most influenced the decision), explanation \
(1-3 sentence justification referencing specific ingredients)."""


async def _llm_classify(text: str) -> NovaResult | None:
    result = await nebius_client.chat_json(_LLM_SYSTEM_PROMPT, f"Ingredient list:\n{text}")
    if not result:
        return None
    try:
        return NovaResult(
            classification=result["classification"],
            label=result.get("label", ""),
            confidence=float(result.get("confidence", 0.5)),
            triggered_by=result.get("triggered_by", []),
            explanation=result.get("explanation", ""),
            source="llm",
        )
    except (KeyError, TypeError, ValueError):
        return None


async def analyze_nova(text: str, use_llm: bool) -> NovaResult:
    rule_result = _rule_based_classify(text)
    # Only fall back to the LLM when the rule engine is unsure, to save inference cost/latency
    if use_llm and (rule_result.classification == "unknown" or rule_result.confidence < 0.6):
        llm_result = await _llm_classify(text)
        if llm_result is not None:
            return llm_result
    return rule_result
