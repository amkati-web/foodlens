import pytest

from app.models.schemas import UserProfile
from app.services import additive_service


def test_extract_e_numbers_various_formats():
    text = "Contains E102, E-621, e 951 and sugar"
    codes = additive_service.extract_e_numbers(text)
    assert codes == ["E102", "E621", "E951"]


def test_extract_e_numbers_ignores_non_matches():
    text = "Just sugar, salt, and water"
    assert additive_service.extract_e_numbers(text) == []


@pytest.mark.asyncio
async def test_analyze_additives_known_code_no_llm():
    profile = UserProfile(conditions=["adhd"])
    findings = await additive_service.analyze_additives("E102, water", profile, use_llm=False)
    assert len(findings) == 1
    assert findings[0].code == "E102"
    assert findings[0].source == "database"
    assert "adhd" in findings[0].condition_warnings


@pytest.mark.asyncio
async def test_analyze_additives_filters_conditions_not_in_profile():
    profile = UserProfile(conditions=["diabetes"])
    findings = await additive_service.analyze_additives("E102", profile, use_llm=False)
    assert findings[0].condition_warnings == {}


@pytest.mark.asyncio
async def test_analyze_additives_unknown_code_without_llm():
    profile = UserProfile()
    findings = await additive_service.analyze_additives("E999", profile, use_llm=False)
    assert findings[0].rating == "unknown"
