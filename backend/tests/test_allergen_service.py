from app.models.schemas import UserProfile
from app.services import allergen_service


def test_detects_hidden_gluten_source():
    text = "Water, malt extract, sugar, salt"
    profile = UserProfile(restrictions=["gluten_free"])
    findings = allergen_service.analyze_allergens(text, profile)
    assert any(f.restriction == "gluten_free" for f in findings)


def test_no_restrictions_returns_empty():
    text = "Water, malt extract, sugar, salt"
    profile = UserProfile()
    assert allergen_service.analyze_allergens(text, profile) == []


def test_detects_vegan_conflict_carmine():
    text = "Sugar, water, carmine, citric acid"
    profile = UserProfile(restrictions=["vegan"])
    findings = allergen_service.analyze_allergens(text, profile)
    assert any(f.restriction == "vegan" for f in findings)


def test_no_conflict_for_clean_ingredient_list():
    text = "Water, sea salt, black pepper"
    profile = UserProfile(restrictions=["lactose_free", "nut_allergy"])
    assert allergen_service.analyze_allergens(text, profile) == []
