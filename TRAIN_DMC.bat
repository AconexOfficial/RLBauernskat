@echo off

REM Train the Deep Monte Carlo Agent
call .venv\Scripts\activate
start "TensorBoard" tensorboard --logdir="experiments/dmc_agent_result/dmc_agent_bauernskat_v1/tensorboard_logs"
python rlcard_custom\rlcard\agents\bauernskat\dmc_agent\trainer.py
pause