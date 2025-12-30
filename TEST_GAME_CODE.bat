@echo off
REM Test the implementation

call .venv\Scripts\activate
pytest rlcard_custom/tests/games/test_bauernskat_game.py rlcard_custom/tests/envs/test_bauernskat.py
pause