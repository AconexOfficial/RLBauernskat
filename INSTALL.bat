@echo off

REM Installation Script for the edited RLCard library

pip install --upgrade uv
uv venv --python 3.11.13

call .venv\Scripts\activate

uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
uv pip install -r requirements.txt
uv pip install -e rlcard_custom
uv pip list

python helper/download_pretrained.py

echo Setup and model download complete!
pause
deactivate