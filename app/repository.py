from __future__ import annotations

import json
from pathlib import Path

from .models import Recipe


class RecipeRepository:
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self._recipes: list[Recipe] = []
        self._by_id: dict[str, Recipe] = {}

    def load(self) -> None:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Recipe dataset not found: {self.data_path}")

        recipes: list[Recipe] = []
        seen: set[str] = set()
        with self.data_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    recipe = Recipe.model_validate(json.loads(line))
                except Exception as exc:
                    raise ValueError(
                        f"Invalid recipe on line {line_number} of {self.data_path}: {exc}"
                    ) from exc
                if recipe.id in seen:
                    raise ValueError(f"Duplicate recipe id: {recipe.id}")
                seen.add(recipe.id)
                recipes.append(recipe)

        if not recipes:
            raise ValueError(f"Recipe dataset is empty: {self.data_path}")
        self._recipes = recipes
        self._by_id = {recipe.id: recipe for recipe in recipes}

    @property
    def recipes(self) -> list[Recipe]:
        return list(self._recipes)

    def get(self, recipe_id: str) -> Recipe | None:
        return self._by_id.get(recipe_id)

