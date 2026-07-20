from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


ROOT = Path(__file__).resolve().parents[1]


def make_client(tmp_path: Path) -> TestClient:
    app = create_app(
        Settings(
            recipe_data_path=ROOT / "data" / "recipes.jsonl",
            feedback_data_path=tmp_path / "feedback.jsonl",
            gemini_api_key=None,
            cors_origins=("http://localhost:3000",),
        )
    )
    return TestClient(app)


def test_health_and_recipe_endpoints(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["recipes_loaded"] >= 25
        assert health.json()["gemini_configured"] is False

        recipes = client.get("/api/v1/recipes", params={"region": "Gujarati"})
        assert recipes.status_code == 200
        assert recipes.json()
        assert all(item["region"] == "Gujarati" for item in recipes.json())


def test_recommendation_works_without_gemini_key(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        response = client.post(
            "/api/v1/recommend",
            json={
                "query": "Gujarati vegan fat-loss dinner",
                "allergies": ["dairy"],
                "top_k": 3,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["ai_used"] is False
        assert body["recommendations"]
        assert all(
            "vegan" in item["recipe"]["dietary_tags"] for item in body["recommendations"]
        )


def test_feedback_is_validated_and_saved(tmp_path: Path) -> None:
    feedback_path = tmp_path / "feedback.jsonl"
    with make_client(tmp_path) as client:
        response = client.post(
            "/api/v1/feedback",
            json={"recipe_id": "gujarati-moong-dal-dhokla", "rating": 5},
        )
        assert response.status_code == 201
        assert feedback_path.exists()

        missing = client.post(
            "/api/v1/feedback",
            json={"recipe_id": "does-not-exist", "rating": 5},
        )
        assert missing.status_code == 404

