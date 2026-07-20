from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _csv_env(name: str, default: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in os.getenv(name, default).split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "Fit Rasoi API")
    app_version: str = "1.0.0"
    recipe_data_path: Path = Path(
        os.getenv("RECIPE_DATA_PATH", str(PROJECT_ROOT / "data" / "recipes.jsonl"))
    )
    feedback_data_path: Path = Path(
        os.getenv("FEEDBACK_DATA_PATH", str(PROJECT_ROOT / "data" / "feedback.jsonl"))
    )
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY") or None
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    gemini_timeout_seconds: float = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "25"))
    cors_origins: tuple[str, ...] = _csv_env(
        "CORS_ORIGINS", "http://localhost:3000,http://localhost:5173"
    )


settings = Settings()
