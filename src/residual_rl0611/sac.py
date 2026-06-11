"""Minimal Soft Actor-Critic implementation for continuous residual actions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import TrainingConfig


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class ReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, capacity: int, seed: int = 0) -> None:
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(self, obs, action, reward: float, next_obs, done: bool) -> None:
        idx = self.ptr
        self.obs[idx] = obs
        self.actions[idx] = action
        self.rewards[idx, 0] = reward
        self.next_obs[idx] = next_obs
        self.dones[idx, 0] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        idx = self.rng.integers(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[idx], device=device),
            "actions": torch.as_tensor(self.actions[idx], device=device),
            "rewards": torch.as_tensor(self.rewards[idx], device=device),
            "next_obs": torch.as_tensor(self.next_obs[idx], device=device),
            "dones": torch.as_tensor(self.dones[idx], device=device),
        }


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.backbone = MLP(obs_dim, hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(obs)
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1.0 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self(obs)
        return torch.tanh(mean)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.q = MLP(obs_dim + action_dim, 1, hidden_dim)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q(torch.cat([obs, action], dim=-1))


@dataclass
class SACUpdateInfo:
    critic_loss: float
    actor_loss: float
    alpha_loss: float
    alpha: float


class SACAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: TrainingConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.config = config or TrainingConfig()
        self.device = torch.device(device)
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        self.actor = SquashedGaussianActor(obs_dim, action_dim, self.config.hidden_dim).to(self.device)
        self.q1 = QNetwork(obs_dim, action_dim, self.config.hidden_dim).to(self.device)
        self.q2 = QNetwork(obs_dim, action_dim, self.config.hidden_dim).to(self.device)
        self.q1_target = QNetwork(obs_dim, action_dim, self.config.hidden_dim).to(self.device)
        self.q2_target = QNetwork(obs_dim, action_dim, self.config.hidden_dim).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=self.config.critic_lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=self.config.critic_lr)
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.config.alpha_lr)
        self.target_entropy = -float(action_dim)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                action = self.actor.deterministic(obs_t)
            else:
                action, _ = self.actor.sample(obs_t)
        return action.squeeze(0).cpu().numpy()

    def update(self, batch: dict[str, torch.Tensor]) -> SACUpdateInfo:
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_obs)
            target_q = torch.min(
                self.q1_target(next_obs, next_action),
                self.q2_target(next_obs, next_action),
            )
            backup = rewards + self.config.gamma * (1.0 - dones) * (
                target_q - self.alpha.detach() * next_log_prob
            )

        q1_loss = F.mse_loss(self.q1(obs, actions), backup)
        q2_loss = F.mse_loss(self.q2(obs, actions), backup)
        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()
        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()

        new_action, log_prob = self.actor.sample(obs)
        q_new = torch.min(self.q1(obs, new_action), self.q2(obs, new_action))
        actor_loss = (self.alpha.detach() * log_prob - q_new).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)

        return SACUpdateInfo(
            critic_loss=float((q1_loss + q2_loss).detach().cpu().item()),
            actor_loss=float(actor_loss.detach().cpu().item()),
            alpha_loss=float(alpha_loss.detach().cpu().item()),
            alpha=float(self.alpha.detach().cpu().item()),
        )

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        tau = self.config.tau
        with torch.no_grad():
            for src, dst in zip(source.parameters(), target.parameters()):
                dst.data.mul_(1.0 - tau).add_(tau * src.data)
