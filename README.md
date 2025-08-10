# Tetris Gym + PPO
Wraps the nuno-faria/tetris-ai engine into a Gymnasium env and trains with PPO (Stable-Baselines3).

## Quickstart
```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3.13 prepare_engine.py
python3.13 ppo_train.py --timesteps 200000
python3.13 ppo_eval.py --model runs/ppo_tetris.zip --episodes 5

## Status
Bootstrap initialized.
