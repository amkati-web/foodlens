import pytest

from app.services import nova_service


@pytest.mark.asyncio
async def test_nova_4_detected_from_trigger_terms():
    text = "Water, sugar, modified maize starch, flavouring, colour, emulsifier"
    result = await nova_service.analyze_nova(text, use_llm=False)
    assert result.classification == "nova_4"
    assert result.confidence > 0.5
    assert len(result.triggered_by) > 0


@pytest.mark.asyncio
async def test_nova_1_single_ingredient():
    result = await nova_service.analyze_nova("Rolled oats", use_llm=False)
    assert result.classification == "nova_1"


@pytest.mark.asyncio
async def test_nova_3_few_plain_ingredients():
    result = await nova_service.analyze_nova("Tomatoes, salt, olive oil", use_llm=False)
    assert result.classification == "nova_3"
