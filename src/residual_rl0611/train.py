"""Short training utilities for residual SAC experiments."""

from __future__ import annotations

import argparse

import numpy as np

from .config import DisturbanceConfig, EnvConfig, TrainingConfig
from .data import sine_reference
from .env import VibrationControlEnv
from .sac import ReplayBuffer, SACAgent


def train_short(
    use_disturbance_obs: bool = False,
    disturbance_mode: str = "none",
    episodes: int = 2,
    steps: int = 300,
    seed: int = 7,
) -> dict[str, float]:
    train_cfg = TrainingConfig(seed=seed, episodes=episodes, steps_per_episode=steps, warmup_steps=50)
    env_cfg = EnvConfig(
        use_disturbance_obs=use_disturbance_obs,
        disturbance=DisturbanceConfig(mode=disturbance_mode),
    )
    reference = sine_reference(steps + 1, amplitude=1.0, frequency_hz=3.0, dt=env_cfg.plant.dt)
    env = VibrationControlEnv(env_cfg, reference=reference)
    agent = SACAgent(env.obs_dim, env.action_dim, train_cfg)
    buffer = ReplayBuffer(env.obs_dim, env.action_dim, train_cfg.replay_size, seed=seed)
    rng = np.random.default_rng(seed)
    total_reward = 0.0
    updates = 0

    global_step = 0
    for _ in range(episodes):
        obs = env.reset(reference)
        for _ in range(steps):
            if global_step < train_cfg.warmup_steps:
                action = rng.uniform(-1.0, 1.0, size=env.action_dim).astype(np.float32)
            else:
                action = agent.act(obs)
            next_obs, reward, done, _ = env.step(action)
            buffer.add(obs, action, reward, next_obs, done)
            obs = next_obs
            total_reward += reward
            global_step += 1
            if buffer.size >= train_cfg.batch_size:
                agent.update(buffer.sample(train_cfg.batch_size, agent.device))
                updates += 1
            if done:
                break
    return {"total_reward": float(total_reward), "updates": float(updates)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-disturbance-obs", action="store_true")
    parser.add_argument("--disturbance-mode", default="none")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    result = train_short(
        use_disturbance_obs=args.use_disturbance_obs,
        disturbance_mode=args.disturbance_mode,
        episodes=args.episodes,
        steps=args.steps,
        seed=args.seed,
    )
    print(result)


if __name__ == "__main__":
    main()
