@echo off

REM Run Bauernskat UI

call .venv\Scripts\activate
streamlit run ui/bauernskat_ui.py
pause