from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Ingredient(BaseModel):
    name: str
    amount: str


class Nutrition(BaseModel):
    calories: int = Field(ge=0)
    protein_g: float = Field(ge=0)
    carbs_g: float = Field(ge=0)
    fat_g: float = Field(ge=0)
    fiber_g: float = Field(ge=0)
    note: str = "Estimated per serving; actual values vary by brands and preparation."


class Recipe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9-]+$")
    name: str
    region: str
    dietary_tags: list[str]
    meal_types: list[str]
    health_goals: list[str]
    ingredients: list[Ingredient]
    instructions: list[str]
    nutrition: Nutrition
    servings: int = Field(ge=1, le=20)
    prep_time_minutes: int = Field(ge=0)
    cook_time_minutes: int = Field(ge=0)
    difficulty: Literal["easy", "medium", "advanced"]
    spice_level: Literal["mild", "medium", "hot"]
    allergens: list[str] = []
    substitutions: list[str] = []
    tips: list[str] = []

    @property
    def total_time_minutes(self) -> int:
        return self.prep_time_minutes + self.cook_time_minutes


class RecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=2, max_length=1000)
    dietary_preferences: list[str] = []
    allergies: list[str] = []
    exclude_ingredients: list[str] = []
    available_ingredients: list[str] = []
    region: str | None = None
    health_goal: str | None = None
    meal_type: str | None = None
    max_calories: int | None = Field(default=None, ge=50, le=3000)
    min_protein_g: float | None = Field(default=None, ge=0, le=250)
    max_time_minutes: int | None = Field(default=None, ge=5, le=600)
    top_k: int = Field(default=3, ge=1, le=10)
    language: str = Field(default="English", max_length=50)
    use_ai: bool = True

    @field_validator(
        "dietary_preferences",
        "allergies",
        "exclude_ingredients",
        "available_ingredients",
        mode="before",
    )
    @classmethod
    def remove_empty_values(cls, value: object) -> object:
        if not isinstance(value, list):
            return value
        return [str(item).strip() for item in value if str(item).strip()]


class InterpretedPreferences(BaseModel):
    diets: list[str]
    allergies: list[str]
    excluded_ingredients: list[str]
    region: str | None
    health_goal: str | None
    meal_type: str | None


class RecommendationItem(BaseModel):
    recipe: Recipe
    score: float
    reasons: list[str]


class RecommendationResponse(BaseModel):
    interpreted_preferences: InterpretedPreferences
    recommendations: list[RecommendationItem]
    assistant_message: str
    ai_used: bool
    disclaimer: str = (
        "Nutrition values are estimates. For medical conditions, allergies, pregnancy, "
        "or therapeutic diets, consult a qualified clinician or dietitian."
    )


class FeedbackRequest(BaseModel):
    recipe_id: str
    rating: int = Field(ge=1, le=5)
    session_id: str | None = Field(default=None, max_length=100)
    comment: str | None = Field(default=None, max_length=1000)


class FeedbackRecord(FeedbackRequest):
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HealthResponse(BaseModel):
    status: str
    recipes_loaded: int
    gemini_configured: bool
    version: str

