from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware

from .config import Settings, settings
from .gemini import GeminiPersonalizer
from .models import (
    FeedbackRecord,
    FeedbackRequest,
    HealthResponse,
    Recipe,
    RecommendationRequest,
    RecommendationResponse,
)
from .recommender import RecipeRecommender, normalize
from .repository import RecipeRepository


feedback_lock = asyncio.Lock()


def create_app(app_settings: Settings = settings) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        repository = RecipeRepository(app_settings.recipe_data_path)
        repository.load()
        app.state.repository = repository
        app.state.recommender = RecipeRecommender(repository.recipes)
        app.state.personalizer = GeminiPersonalizer(app_settings)
        yield

    app = FastAPI(
        title=app_settings.app_name,
        version=app_settings.app_version,
        description=(
            "A grounded Indian recipe recommendation API with strict dietary filtering, "
            "local TF-IDF ranking, and optional Gemini personalization."
        ),
        lifespan=lifespan,
    )
    app.state.settings = app_settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(app_settings.cors_origins),
        allow_credentials="*" not in app_settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", tags=["system"])
    async def root() -> dict[str, str]:
        return {"name": app_settings.app_name, "docs": "/docs", "health": "/health"}

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health(request: Request) -> HealthResponse:
        return HealthResponse(
            status="ok",
            recipes_loaded=len(request.app.state.repository.recipes),
            gemini_configured=request.app.state.personalizer.configured,
            version=app_settings.app_version,
        )

    @app.get("/api/v1/recipes", response_model=list[Recipe], tags=["recipes"])
    async def list_recipes(
        request: Request,
        region: str | None = None,
        diet: str | None = None,
        meal_type: str | None = None,
        limit: int = Query(default=50, ge=1, le=200),
    ) -> list[Recipe]:
        recipes = request.app.state.repository.recipes
        if region:
            recipes = [item for item in recipes if normalize(item.region) == normalize(region)]
        if diet:
            requested = normalize(diet)
            recipes = [
                item for item in recipes if requested in {normalize(tag) for tag in item.dietary_tags}
            ]
        if meal_type:
            requested = normalize(meal_type)
            recipes = [
                item for item in recipes if requested in {normalize(meal) for meal in item.meal_types}
            ]
        return recipes[:limit]

    @app.get("/api/v1/recipes/{recipe_id}", response_model=Recipe, tags=["recipes"])
    async def get_recipe(recipe_id: str, request: Request) -> Recipe:
        recipe = request.app.state.repository.get(recipe_id)
        if recipe is None:
            raise HTTPException(status_code=404, detail="Recipe not found")
        return recipe

    @app.post(
        "/api/v1/recommend",
        response_model=RecommendationResponse,
        tags=["recommendations"],
    )
    async def recommend(
        body: RecommendationRequest, request: Request
    ) -> RecommendationResponse:
        preferences, recommendations = request.app.state.recommender.recommend(body)
        message, ai_used = await request.app.state.personalizer.personalize(body, recommendations)
        return RecommendationResponse(
            interpreted_preferences=preferences,
            recommendations=recommendations,
            assistant_message=message,
            ai_used=ai_used,
        )

    @app.post(
        "/api/v1/feedback",
        status_code=status.HTTP_201_CREATED,
        tags=["learning"],
    )
    async def save_feedback(body: FeedbackRequest, request: Request) -> dict[str, str]:
        if request.app.state.repository.get(body.recipe_id) is None:
            raise HTTPException(status_code=404, detail="Recipe not found")
        record = FeedbackRecord(**body.model_dump())
        path: Path = app_settings.feedback_data_path
        async with feedback_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(record.model_dump_json() + "\n")
        return {"status": "saved", "message": "Feedback stored for future ranking improvements."}

    return app


app = create_app()

