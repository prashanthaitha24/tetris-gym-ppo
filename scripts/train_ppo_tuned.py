import os, warnings, argparse
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from torch.utils.tensorboard import SummaryWriter
from gym_tetris import TetrisEnv

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Protobuf gencode version")

# Custom callback to collect eval metrics
class CustomEvalCallback(EvalCallback):
    def __init__(self, eval_env, log_dir, eval_freq=50000, n_eval_episodes=20):
        super().__init__(
            eval_env,
            best_model_save_path=os.path.join(log_dir, "best"),
            log_path=os.path.join(log_dir, "eval_logs"),
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=False,
            render=False
        )
        self.writer = SummaryWriter(log_dir)
        self.best_mean_reward = -float("inf")
        self.last_mean_ep_length = 0
        self.last_median_steps = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            super()._on_step()
            # Log eval metrics
            if self.last_mean_reward is not None:
                self.writer.add_scalar("eval/mean_reward", self.last_mean_reward, self.num_timesteps)
                self.writer.add_scalar("eval/mean_ep_length", self.last_mean_ep_length, self.num_timesteps)
                self.writer.add_scalar("eval/median_steps", self.last_median_steps, self.num_timesteps)
                if self.last_mean_reward > self.best_mean_reward:
                    self.best_mean_reward = self.last_mean_reward
        # Log training info
        if self.locals.get('infos'):
            for info in self.locals['infos']:
                if 'lines_delta' in info:
                    self.writer.add_scalar('rollout/lines_cleared', info['lines_delta'], self.num_timesteps)
                if 'score_delta' in info:
                    self.writer.add_scalar('rollout/score_delta', info['score_delta'], self.num_timesteps)
        return True

    def _on_training_end(self):
        self.writer.close()

# Reward shaping wrapper
class RewardShapedTetris(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_board = None
        self.writer = None  # To be set by env
        self.step_count = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_board = obs.copy()
        self.step_count = 0
        return obs, info

    def step(self, action):
        obs, _, done, trunc, info = self.env.step(action)
        self.step_count += 1
        score_delta = float(info.get("score_delta", 0))
        shaped = 2.0 * score_delta + 0.05

        if self.last_board is not None:
            holes_before = self._holes(self.last_board)
            holes_after = self._holes(obs)
            shaped -= max(0, holes_after - holes_before) * 0.005
            h_before = self._max_height(self.last_board)
            h_after = self._max_height(obs)
            shaped -= max(0, h_after - h_before) * 0.002
            # Log board metrics
            if self.writer:
                self.writer.add_scalar('board/holes', holes_after, self.step_count)
                self.writer.add_scalar('board/max_height', h_after, self.step_count)

        self.last_board = obs.copy()
        return obs, float(shaped), done, trunc, info

    def _holes(self, board):
        H, W = board.shape
        filled = board != 0
        holes = 0
        for c in range(W):
            seen = False
            for r in range(H):
                if filled[r, c]:
                    seen = True
                elif seen:
                    holes += 1
        return holes

    def _max_height(self, board):
        H, W = board.shape
        filled = board != 0
        h = 0
        for c in range(W):
            col = filled[:, c]
            h = max(h, H - np.argmax(col) if np.any(col) else 0)
        return h

def make_env(max_steps=5000, tb_logdir=None):
    def _thunk():
        env = TetrisEnv()
        env = TimeLimit(env, max_episode_steps=max_steps)
        env = Monitor(env)
        env = RewardShapedTetris(env)
        if tb_logdir:
            env.writer = SummaryWriter(os.path.join(tb_logdir, "env_logs"))
        return env
    return _thunk

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=800_000)
    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--n-steps", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--vf-coef", type=float, default=0.3)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--save-dir", type=str, default="runs_tuned")
    ap.add_argument("--tb-logdir", type=str, default="ppo_tetris_tensorboard_tuned")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.tb_logdir, exist_ok=True)

    train_env = DummyVecEnv([make_env(args.max_steps, args.tb_logdir)])
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, clip_reward=50.0)

    eval_env = DummyVecEnv([make_env(args.max_steps, args.tb_logdir)])
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=True, clip_reward=50.0)
    eval_env.training = False

    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=f"./{args.tb_logdir}",
        verbose=1,
    )

    eval_cb = CustomEvalCallback(
        eval_env,
        log_dir=args.tb_logdir,
        eval_freq=50_000,
        n_eval_episodes=20
    )

    model.learn(total_timesteps=args.timesteps, callback=eval_cb)

    # Saving final model and VecNormalize stats
    model.save(os.path.join(args.save_dir, "ppo_tetris_tuned"))
    train_env.save(os.path.join(args.save_dir, "vecnorm.pkl"))
    print(f"✅ Training complete. Model + stats saved under {args.save_dir}/")

    # Logging final stats
    writer = SummaryWriter(os.path.join(args.tb_logdir, "final"))
    writer.add_scalar("final/best_mean_reward", eval_cb.best_mean_reward, args.timesteps)
    writer.add_scalar("final/mean_episode_length", eval_cb.last_mean_ep_length, args.timesteps)
    writer.add_scalar("final/median_steps_deterministic", eval_cb.last_median_steps, args.timesteps)
    writer.close()

if __name__ == "__main__":
    main()
