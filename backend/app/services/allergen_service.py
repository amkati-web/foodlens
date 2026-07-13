from __future__ import annotations

import json
import re
from pathlib import Path

from app.models.schemas import AllergenFinding, UserProfile

_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "allergen_map.json"
_ALLERGEN_DB: dict = json.loads(_DATA_PATH.read_text())


def _split_ingredients(text: str) -> list[str]:
    parts = re.split(r"[,\n;]|(?:\band\b)", text, flags=re.IGNORECASE)
    return [p.strip().lower() for p in parts if p.strip()]


def analyze_allergens(text: str, profile: UserProfile) -> list[AllergenFinding]:
    if not profile.restrictions:
        return []

    ingredients = _split_ingredients(text)
    lowered_text = text.lower()
    findings: list[AllergenFinding] = []

    for restriction in profile.restrictions:
        rule = _ALLERGEN_DB.get(restriction)
        if not rule:
            continue
        matched: set[str] = set()
        for flag in rule["flags"]:
            flag_l = flag.lower()
            if flag_l in lowered_text:
                # find the most specific ingredient phrase that contains the flag
                hit = next((ing for ing in ingredients if flag_l in ing), flag_l)
                matched.add(hit)
        for m in sorted(matched):
            findings.append(
                AllergenFinding(restriction=restriction, matched_ingredient=m, note=rule["note"])
            )
    return findings
