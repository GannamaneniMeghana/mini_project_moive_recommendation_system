@echo off
echo Starting Movie Recommendation App...
if exist .venv\Scripts\activate (
    call .venv\Scripts\activate
) else (
    echo Virtual environment not found. Please ensure .venv exists.
)
python app.py
pause
