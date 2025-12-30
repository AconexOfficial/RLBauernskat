@echo off

REM Train the Soft Actor-Critic Agent

call .venv\Scripts\activate
start "TensorBoard" tensorboard --logdir="experiments/sac_agent_result/sac_agent_bauernskat_v1/tensorboard_logs"
python rlcard_custom\rlcard\agents\bauernskat\sac_agent\trainer.py
pause