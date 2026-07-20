from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from .config import Settings
from .models import RecommendationItem, RecommendationRequest


logger = logging.getLogger(__name__)


def deterministic_message(items: list[RecommendationItem]) -> str:
    if not items:
        return (
            "I could not find a recipe that safely matches every filter. Try increasing the time "
            "or calorie limit, but keep dietary restrictions and allergies unchanged."
        )
    best = items[0]
    nutrition = best.recipe.nutrition
    return (
        f"My best match is {best.recipe.name}. It is approximately {nutrition.calories} kcal "
        f"with {nutrition.protein_g:g} g protein per serving. "
        f"Why it fits: {', '.join(best.reasons)}."
    )


class GeminiPersonalizer:
    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    def configured(self) -> bool:
        return bool(self.settings.gemini_api_key)

    def _prompt(
        self, request: RecommendationRequest, recommendations: list[RecommendationItem]
    ) -> str:
        safe_recipes: list[dict[str, Any]] = []
        for item in recommendations:
            recipe = item.recipe
            safe_recipes.append(
                {
                    "name": recipe.name,
                    "region": recipe.region,
                    "dietary_tags": recipe.dietary_tags,
                    "meal_types": recipe.meal_types,
                    "health_goals": recipe.health_goals,
                    "ingredients": [ingredient.model_dump() for ingredient in recipe.ingredients],
                    "instructions": recipe.instructions,
                    "nutrition_estimate_per_serving": recipe.nutrition.model_dump(),
                    "allergens": recipe.allergens,
                    "substitutions": recipe.substitutions,
                    "selection_reasons": item.reasons,
                }
            )
        return f"""
You are Fit Rasoi, a careful Indian recipe recommendation assistant.

User request: {request.query}
Requested response language: {request.language}

Rules:
- Recommend only from the VERIFIED CANDIDATES below. Never invent another recipe.
- Never change or contradict dietary tags, allergen data, ingredients, or nutrition values.
- Say that nutrition is estimated when mentioning it.
- Do not make medical promises. If the request concerns a medical condition, give a brief
  suggestion to consult a qualified clinician or dietitian.
- Keep the answer practical and under 220 words.
- Start with the best match, explain why, then offer the other candidates briefly.

VERIFIED CANDIDATES:
{json.dumps(safe_recipes, ensure_ascii=False)}
""".strip()

    def _generate_sync(self, prompt: str) -> str:
        from google import genai

        client = genai.Client(api_key=self.settings.gemini_api_key)
        interactions = getattr(client, "interactions", None)
        if interactions is not None:
            response = interactions.create(
                model=self.settings.gemini_model,
                input=prompt,
                store=False,
                timeout=self.settings.gemini_timeout_seconds,
            )
            return response.output_text.strip()

        # Compatibility path for older google-genai SDK releases.
        response = client.models.generate_content(
            model=self.settings.gemini_model,
            contents=prompt,
        )
        return response.text.strip()

    async def personalize(
        self, request: RecommendationRequest, recommendations: list[RecommendationItem]
    ) -> tuple[str, bool]:
        fallback = deterministic_message(recommendations)
        if not request.use_ai or not self.configured or not recommendations:
            return fallback, False
        try:
            message = await asyncio.wait_for(
                asyncio.to_thread(self._generate_sync, self._prompt(request, recommendations)),
                timeout=self.settings.gemini_timeout_seconds,
            )
            return (message or fallback), bool(message)
        except Exception as exc:
            logger.warning("Gemini personalization failed; using deterministic response: %s", exc)
            return fallback, False
