@echo off
setlocal
if not exist .venv (
  python -m venv .venv
)
call .venv\Scripts\activate
python -m pip install -r requirements.txt
if not exist .env copy .env.example .env
echo.
echo Open http://localhost:8000/docs after the server starts.
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

