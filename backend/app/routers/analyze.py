from __future__ import annotations

from fastapi import APIRouter

from app.models.schemas import AnalyzeRequest, AnalyzeResponse
from app.services import additive_service, allergen_service, nova_service

router = APIRouter(prefix="/api", tags=["analyze"])


def _build_summary(response: AnalyzeResponse) -> str:
    red_flags = [a for a in response.additives if a.rating == "red"]
    amber_flags = [a for a in response.additives if a.rating == "amber"]
    parts = [f"NOVA group: {response.nova.classification.upper()} ({response.nova.label})."]

    if red_flags:
        parts.append(
            f"{len(red_flags)} additive(s) flagged high-concern: "
            + ", ".join(f"{a.code} ({a.name})" for a in red_flags) + "."
        )
    if amber_flags:
        parts.append(f"{len(amber_flags)} additive(s) flagged moderate-concern.")
    if response.allergens:
        restrictions = sorted({a.restriction for a in response.allergens})
        parts.append(
            "Possible conflicts with your dietary restrictions: " + ", ".join(restrictions) + "."
        )
    if not red_flags and not amber_flags and not response.allergens:
        parts.append("No major concerns detected against the profile provided.")
    return " ".join(parts)


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    additives = await additive_service.analyze_additives(
        request.ingredient_text, request.profile, request.use_llm
    )
    allergens = allergen_service.analyze_allergens(request.ingredient_text, request.profile)
    nova = await nova_service.analyze_nova(request.ingredient_text, request.use_llm)

    response = AnalyzeResponse(
        additives=additives, allergens=allergens, nova=nova, overall_summary=""
    )
    response.overall_summary = _build_summary(response)
    return response
