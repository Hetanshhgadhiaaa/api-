# Fit Rasoi Backend API

A ready-to-run Indian recipe recommendation backend. It uses strict database filters and local TF-IDF ranking first, then optionally asks Gemini to turn the verified results into a natural personalized answer.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Hetanshhgadhiaaa/api-)

Click **Deploy to Render** above for the simplest online setup. Gemini is optional; the verified local recommendation engine works without an API key.

The included starter database contains 32 complete recipes across Gujarati, Punjabi, Rajasthani, South Indian, Tamil, Kerala, Andhra, Bengali, Maharashtrian, Goan, Kashmiri, Northeast Indian, Hyderabadi and North Indian food. It covers vegan, vegetarian, Jain, eggetarian and non-vegetarian choices.

## What is included

- `POST /api/v1/recommend`: understands a natural-language request, filters unsafe matches, ranks recipes and optionally personalizes the answer with Gemini.
- `GET /api/v1/recipes`: browses the database with region, diet and meal filters.
- `GET /api/v1/recipes/{id}`: returns one full recipe.
- `POST /api/v1/feedback`: records ratings for future recommendation-model training.
- `GET /health`: reports readiness, recipe count and whether Gemini is configured.
- Swagger API tester at `/docs`.
- A validated JSONL recipe dataset and automated tests.
- Deterministic fallback: recommendations still work when Gemini is unavailable or no key is set.

## Fastest Windows setup

1. Install Python 3.12 or newer.
2. Double-click `start_windows.bat`.
3. Open <http://localhost:8000/docs>.

The first run creates a virtual environment, installs packages and copies `.env.example` to `.env`.

## Add Gemini

1. Create an API key at <https://aistudio.google.com/apikey>.
2. Open the generated `.env` file.
3. Put the key after `GEMINI_API_KEY=`:

```dotenv
GEMINI_API_KEY=your_real_key_here
GEMINI_MODEL=gemini-3.5-flash
```

Never put this key in frontend JavaScript or commit it to Git. Restart the API after changing `.env`. The default model can be changed through `GEMINI_MODEL` without changing code.

## macOS/Linux setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

## Test a recommendation

```bash
curl -X POST http://localhost:8000/api/v1/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "query": "I am Gujarati, vegan and want a flavorful fat-loss dinner",
    "allergies": ["dairy"],
    "exclude_ingredients": ["tofu"],
    "max_calories": 400,
    "top_k": 3
  }'
```

Important request fields:

| Field | Example | Purpose |
| --- | --- | --- |
| `query` | `vegan Gujarati dinner` | Natural-language request |
| `dietary_preferences` | `["vegan"]` | Explicit hard dietary preference |
| `allergies` | `["peanut"]` | Excludes matching recipes |
| `exclude_ingredients` | `["tofu"]` | Ingredients the user dislikes or avoids |
| `available_ingredients` | `["moong dal", "spinach"]` | Boosts recipes using those ingredients |
| `max_calories` | `400` | Hard per-serving calorie limit |
| `min_protein_g` | `15` | Hard estimated protein minimum |
| `max_time_minutes` | `45` | Hard total-time limit |
| `top_k` | `3` | Number of results, from 1 to 10 |
| `use_ai` | `true` | Set false to skip Gemini |

## Call it from your frontend

```javascript
const response = await fetch("http://localhost:8000/api/v1/recommend", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    query: "high-protein South Indian breakfast",
    dietary_preferences: ["vegan"],
    top_k: 3
  })
});

const result = await response.json();
console.log(result.assistant_message);
console.log(result.recommendations);
```

Add the deployed frontend URL to `CORS_ORIGINS` in `.env`. Keep Gemini calls in this backend, never in the browser.

## How the accuracy layer works

1. The API identifies explicit diet, cuisine, goal and meal clues.
2. Allergies, excluded ingredients, diet rules, time and nutrition limits are hard filters.
3. A local TF-IDF model ranks the remaining recipes using the request and recipe metadata.
4. Gemini receives only the top verified recipes and is instructed not to invent nutrition, ingredients or alternatives.
5. If Gemini fails, the ranked results and a deterministic explanation are returned.

The database—not Gemini—is the source of recipe facts. Nutrition values are estimates and should be professionally reviewed before a health-focused production launch.

## Expand the database

Add one valid JSON object per line to `data/recipes.jsonl`. Keep recipe IDs unique. Restart the API to validate and load the new data. The existing entries are copyable examples of the complete schema.

For a large database, move the same fields into PostgreSQL and use an embedding/vector index. The API models and frontend response format can remain the same.

## Run tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Docker

```bash
docker build -t fit-rasoi-api .
docker run --rm -p 8000:8000 --env-file .env fit-rasoi-api
```

## Production checklist

- Human-review every recipe and nutrition estimate.
- Add authentication and request rate limits.
- Use a managed database instead of writing feedback to a local file.
- Allow only your real frontend domain through CORS.
- Store `GEMINI_API_KEY` in the hosting provider's secret manager.
- Display the nutrition and medical disclaimer in the user interface.
