from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .models import (
    InterpretedPreferences,
    Recipe,
    RecommendationItem,
    RecommendationRequest,
)


TOKEN_RE = re.compile(r"[a-z0-9]+")

DIET_ALIASES = {
    "vegan": ("vegan", "plant based", "plant-based"),
    "jain": ("jain", "no onion", "no garlic"),
    "eggetarian": ("eggetarian", "eggitarian", "ovo vegetarian"),
    "vegetarian": ("vegetarian", "pure veg", "veg only"),
    "non-vegetarian": ("non vegetarian", "non-vegetarian", "non veg", "chicken", "fish", "prawn"),
}

GOAL_ALIASES = {
    "fat-loss": ("fat loss", "weight loss", "lose weight", "cutting", "low calorie"),
    "muscle-gain": ("muscle gain", "bulk", "bulking", "gain muscle"),
    "high-protein": ("high protein", "more protein", "protein rich"),
    "high-fiber": ("high fiber", "high fibre", "constipation"),
    "balanced": ("balanced", "healthy", "everyday"),
}

MEAL_ALIASES = {
    "breakfast": ("breakfast", "morning"),
    "lunch": ("lunch", "afternoon meal"),
    "dinner": ("dinner", "night meal"),
    "snack": ("snack", "tea time", "evening snack"),
}

REGION_ALIASES = {
    "Gujarati": ("gujarati", "gujarat", "kathiawadi"),
    "Punjabi": ("punjabi", "punjab"),
    "Rajasthani": ("rajasthani", "rajasthan"),
    "Maharashtrian": ("maharashtrian", "maharashtra", "marathi"),
    "Bengali": ("bengali", "bengal"),
    "South Indian": ("south indian",),
    "Tamil": ("tamil", "tamil nadu"),
    "Kerala": ("kerala", "malayali"),
    "Andhra": ("andhra", "telugu"),
    "Kashmiri": ("kashmiri", "kashmir"),
    "Goan": ("goan", "goa"),
    "Northeast Indian": ("northeast", "north east", "assamese", "naga", "manipuri"),
    "Hyderabadi": ("hyderabadi", "hyderabad"),
}

STOP_WORDS = {
    "a", "an", "and", "are", "can", "for", "from", "i", "in", "is", "it", "make",
    "me", "my", "of", "on", "please", "recipe", "something", "that", "the", "to", "want",
    "what", "with",
}


def normalize(value: str) -> str:
    return " ".join(TOKEN_RE.findall(value.lower()))


def tokens(value: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(value.lower()) if token not in STOP_WORDS]


def _first_alias_match(text: str, mapping: dict[str, tuple[str, ...]]) -> str | None:
    normalized = normalize(text)
    for canonical, aliases in mapping.items():
        if any(normalize(alias) in normalized for alias in aliases):
            return canonical
    return None


def _all_alias_matches(text: str, mapping: dict[str, tuple[str, ...]]) -> list[str]:
    normalized = normalize(text)
    return [
        canonical
        for canonical, aliases in mapping.items()
        if any(normalize(alias) in normalized for alias in aliases)
    ]


@dataclass(frozen=True)
class ScoredRecipe:
    recipe: Recipe
    score: float
    reasons: list[str]


class RecipeRecommender:
    def __init__(self, recipes: list[Recipe]):
        self.recipes = recipes
        self._documents = {recipe.id: self._recipe_tokens(recipe) for recipe in recipes}
        self._idf = self._build_idf(self._documents.values())

    @staticmethod
    def _recipe_text(recipe: Recipe) -> str:
        ingredient_names = " ".join(ingredient.name for ingredient in recipe.ingredients)
        return " ".join(
            [
                recipe.name,
                recipe.region,
                " ".join(recipe.dietary_tags),
                " ".join(recipe.meal_types),
                " ".join(recipe.health_goals),
                ingredient_names,
                " ".join(recipe.tips),
            ]
        )

    def _recipe_tokens(self, recipe: Recipe) -> Counter[str]:
        return Counter(tokens(self._recipe_text(recipe)))

    @staticmethod
    def _build_idf(documents: object) -> dict[str, float]:
        docs = list(documents)  # type: ignore[arg-type]
        document_frequency: Counter[str] = Counter()
        for document in docs:
            document_frequency.update(document.keys())
        total = len(docs)
        return {
            token: math.log((total + 1) / (frequency + 1)) + 1
            for token, frequency in document_frequency.items()
        }

    def _tfidf_cosine(self, query: str, recipe_id: str) -> float:
        query_counts = Counter(tokens(query))
        document_counts = self._documents[recipe_id]
        if not query_counts or not document_counts:
            return 0.0

        def weight(counts: Counter[str], token: str) -> float:
            return counts[token] * self._idf.get(token, math.log(len(self.recipes) + 1) + 1)

        shared = query_counts.keys() & document_counts.keys()
        dot = sum(weight(query_counts, token) * weight(document_counts, token) for token in shared)
        query_norm = math.sqrt(sum(weight(query_counts, token) ** 2 for token in query_counts))
        doc_norm = math.sqrt(sum(weight(document_counts, token) ** 2 for token in document_counts))
        return dot / (query_norm * doc_norm) if query_norm and doc_norm else 0.0

    def interpret(self, request: RecommendationRequest) -> InterpretedPreferences:
        combined_diet_text = " ".join([request.query, *request.dietary_preferences])
        diets = _all_alias_matches(combined_diet_text, DIET_ALIASES)
        normalized_diet_text = normalize(combined_diet_text)
        explicitly_restrictive = any(
            phrase in normalized_diet_text
            for phrase in ("vegan", "jain", "eggetarian", "eggitarian", "pure veg", "veg only")
        )
        if "non vegetarian" in normalized_diet_text and not explicitly_restrictive:
            diets = ["non-vegetarian"]
        # A mention of a specific animal protein should not override an explicit restrictive diet.
        if any(diet in diets for diet in ("vegan", "jain", "vegetarian", "eggetarian")):
            diets = [diet for diet in diets if diet != "non-vegetarian"]

        region = request.region or _first_alias_match(request.query, REGION_ALIASES)
        goal = request.health_goal or _first_alias_match(request.query, GOAL_ALIASES)
        meal = request.meal_type or _first_alias_match(request.query, MEAL_ALIASES)
        return InterpretedPreferences(
            diets=list(dict.fromkeys(normalize(diet) for diet in diets)),
            allergies=list(dict.fromkeys(normalize(item) for item in request.allergies)),
            excluded_ingredients=list(
                dict.fromkeys(normalize(item) for item in request.exclude_ingredients)
            ),
            region=region,
            health_goal=normalize(goal) if goal else None,
            meal_type=normalize(meal) if meal else None,
        )

    @staticmethod
    def _diet_allowed(recipe: Recipe, diets: list[str]) -> bool:
        tags = {normalize(tag) for tag in recipe.dietary_tags}
        for diet in diets:
            if diet == "vegan" and "vegan" not in tags:
                return False
            if diet == "jain" and "jain" not in tags:
                return False
            if diet == "vegetarian" and not tags.intersection({"vegan", "vegetarian", "jain"}):
                return False
            if diet == "eggetarian" and "non vegetarian" in tags:
                return False
        return True

    @staticmethod
    def _contains_restricted(recipe: Recipe, restricted: list[str]) -> bool:
        ingredient_text = normalize(" ".join(item.name for item in recipe.ingredients))
        allergen_text = {normalize(allergen) for allergen in recipe.allergens}
        for item in restricted:
            singular = item[:-1] if item.endswith("s") else item
            if (
                item in ingredient_text
                or singular in ingredient_text
                or item in allergen_text
                or singular in allergen_text
                or any(item in allergen or allergen in item for allergen in allergen_text)
            ):
                return True
        return False

    def recommend(
        self, request: RecommendationRequest
    ) -> tuple[InterpretedPreferences, list[RecommendationItem]]:
        preferences = self.interpret(request)
        candidates: list[ScoredRecipe] = []
        requested_region = normalize(preferences.region or "")
        requested_goal = normalize(preferences.health_goal or "")
        requested_meal = normalize(preferences.meal_type or "")
        available = {normalize(item) for item in request.available_ingredients}

        for recipe in self.recipes:
            if not self._diet_allowed(recipe, preferences.diets):
                continue
            if self._contains_restricted(
                recipe, [*preferences.allergies, *preferences.excluded_ingredients]
            ):
                continue
            if request.max_calories is not None and recipe.nutrition.calories > request.max_calories:
                continue
            if request.min_protein_g is not None and recipe.nutrition.protein_g < request.min_protein_g:
                continue
            if request.max_time_minutes is not None and recipe.total_time_minutes > request.max_time_minutes:
                continue

            score = self._tfidf_cosine(request.query, recipe.id) * 40
            reasons: list[str] = []
            recipe_goals = {normalize(item) for item in recipe.health_goals}
            recipe_meals = {normalize(item) for item in recipe.meal_types}

            if requested_region and requested_region == normalize(recipe.region):
                score += 25
                reasons.append(f"matches the requested {recipe.region} cuisine")
            if requested_goal and requested_goal in recipe_goals:
                score += 20
                reasons.append(f"supports the {preferences.health_goal} goal")
            if requested_meal and requested_meal in recipe_meals:
                score += 14
                reasons.append(f"works well for {preferences.meal_type}")
            if preferences.diets:
                score += 12
                reasons.append("meets the dietary preference")

            if available:
                ingredient_names = {normalize(item.name) for item in recipe.ingredients}
                matched = sum(
                    1
                    for requested in available
                    if any(requested in ingredient or ingredient in requested for ingredient in ingredient_names)
                )
                if matched:
                    coverage = matched / len(available)
                    score += 18 * coverage
                    reasons.append(f"uses {matched} of the available ingredients")

            if recipe.nutrition.protein_g >= 15:
                score += 3
            if recipe.nutrition.fiber_g >= 6:
                score += 2
            if not reasons:
                reasons.append("is a strong text match for the request")
            candidates.append(ScoredRecipe(recipe, round(score, 2), reasons))

        candidates.sort(key=lambda item: (-item.score, item.recipe.nutrition.calories, item.recipe.name))
        return preferences, [
            RecommendationItem(recipe=item.recipe, score=item.score, reasons=item.reasons)
            for item in candidates[: request.top_k]
        ]
