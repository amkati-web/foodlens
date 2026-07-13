from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Condition = Literal[
    "diabetes", "adhd", "allergies", "hypertension", "pregnancy", "pku"
]
DietaryRestriction = Literal[
    "gluten_free", "lactose_free", "vegan", "vegetarian",
    "nut_allergy", "peanut_allergy", "halal", "kosher",
    "soy_allergy", "egg_allergy",
]


class UserProfile(BaseModel):
    conditions: list[Condition] = Field(default_factory=list)
    restrictions: list[DietaryRestriction] = Field(default_factory=list)


class AnalyzeRequest(BaseModel):
    ingredient_text: str = Field(..., min_length=2, description="Raw ingredient list, comma or newline separated")
    profile: UserProfile = Field(default_factory=UserProfile)
    use_llm: bool = Field(
        default=True,
        description="If true, augment rule-based results with the Nebius-hosted LLM for ingredients not found in the local database.",
    )


class AdditiveFinding(BaseModel):
    code: str
    name: str
    category: str
    origin: str
    rating: Literal["green", "amber", "red", "unknown"]
    summary: str
    condition_warnings: dict[str, str] = Field(default_factory=dict)
    source: Literal["database", "llm"] = "database"


class AllergenFinding(BaseModel):
    restriction: DietaryRestriction
    matched_ingredient: str
    note: str


class NovaResult(BaseModel):
    classification: Literal["nova_1", "nova_2", "nova_3", "nova_4", "unknown"]
    label: str
    confidence: float
    triggered_by: list[str] = Field(default_factory=list)
    explanation: str
    source: Literal["rules", "llm"] = "rules"


class AnalyzeResponse(BaseModel):
    additives: list[AdditiveFinding]
    allergens: list[AllergenFinding]
    nova: NovaResult
    overall_summary: str


class OcrResponse(BaseModel):
    extracted_text: str
    confidence: float | None = None
