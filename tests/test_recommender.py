from pathlib import Path

from app.models import RecommendationRequest
from app.recommender import RecipeRecommender
from app.repository import RecipeRepository


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "recipes.jsonl"


def build_recommender() -> RecipeRecommender:
    repository = RecipeRepository(DATA_PATH)
    repository.load()
    return RecipeRecommender(repository.recipes)


def test_dataset_loads_and_ids_are_unique() -> None:
    repository = RecipeRepository(DATA_PATH)
    repository.load()
    assert len(repository.recipes) >= 25
    assert len({recipe.id for recipe in repository.recipes}) == len(repository.recipes)


def test_gujarati_vegan_fat_loss_dinner_is_strictly_vegan() -> None:
    recommender = build_recommender()
    request = RecommendationRequest(
        query="I am Gujarati and vegan. Suggest a flavorful fat-loss dinner.", top_k=5
    )
    preferences, results = recommender.recommend(request)
    assert preferences.region == "Gujarati"
    assert "vegan" in preferences.diets
    assert results
    assert all("vegan" in item.recipe.dietary_tags for item in results)
    assert results[0].recipe.region == "Gujarati"


def test_non_vegetarian_is_not_misread_as_vegetarian() -> None:
    recommender = build_recommender()
    preferences, results = recommender.recommend(
        RecommendationRequest(query="Give me a non-vegetarian high-protein dinner", top_k=10)
    )
    assert preferences.diets == ["non vegetarian"]
    assert any("non-vegetarian" in item.recipe.dietary_tags for item in results)


def test_vegetarian_request_excludes_egg_fish_and_chicken() -> None:
    recommender = build_recommender()
    _, results = recommender.recommend(
        RecommendationRequest(query="pure vegetarian dinner", top_k=10)
    )
    assert results
    for item in results:
        assert "eggetarian" not in item.recipe.dietary_tags
        assert "non-vegetarian" not in item.recipe.dietary_tags


def test_allergy_and_excluded_ingredient_are_hard_filters() -> None:
    recommender = build_recommender()
    _, results = recommender.recommend(
        RecommendationRequest(
            query="vegan high protein dinner",
            allergies=["soy"],
            exclude_ingredients=["mushroom"],
            top_k=10,
        )
    )
    assert results
    for item in results:
        assert "soy" not in item.recipe.allergens
        assert all("mushroom" not in ingredient.name.lower() for ingredient in item.recipe.ingredients)


def test_numeric_constraints_are_hard_filters() -> None:
    recommender = build_recommender()
    _, results = recommender.recommend(
        RecommendationRequest(
            query="quick high protein meal",
            max_calories=300,
            min_protein_g=13,
            max_time_minutes=45,
            top_k=10,
        )
    )
    assert results
    assert all(item.recipe.nutrition.calories <= 300 for item in results)
    assert all(item.recipe.nutrition.protein_g >= 13 for item in results)
    assert all(item.recipe.total_time_minutes <= 45 for item in results)

