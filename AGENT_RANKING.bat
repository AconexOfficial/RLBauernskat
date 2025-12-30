@echo off

REM Run agent ranking

call .venv\Scripts\activate
python helper/agent_ranking.py
pause