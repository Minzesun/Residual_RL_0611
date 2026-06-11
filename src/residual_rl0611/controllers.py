"""PID and residual-action controller components."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import PIDConfig, ResidualConfig


class PIDController:
    def __init__(self, config: PIDConfig | None = None) -> None:
        self.config = config or PIDConfig()
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error: float) -> float:
        cfg = self.config
        self.integral += error * cfg.dt
        self.integral = float(np.clip(self.integral, -cfg.integral_limit, cfg.integral_limit))
        derivative = (error - self.prev_error) / cfg.dt
        self.prev_error = float(error)
        command = cfg.kp * error + cfg.ki * self.integral + cfg.kd * derivative
        return float(np.clip(command, -cfg.output_limit, cfg.output_limit))


@dataclass(frozen=True)
class ControlBreakdown:
    u_base: float
    u_residual: float
    u_total_pre_clip: float
    u_total: float
    raw_action: np.ndarray


class ResidualActionCombiner:
    def __init__(self, config: ResidualConfig | None = None) -> None:
        self.config = config or ResidualConfig()

    @property
    def action_dim(self) -> int:
        return 2 if self.config.mode == "hybrid" else 1

    def combine(self, u_base: float, action: np.ndarray | list[float] | tuple[float, ...]) -> ControlBreakdown:
        cfg = self.config
        raw = np.asarray(action, dtype=np.float64).reshape(-1)
        if raw.size < self.action_dim:
            raw = np.pad(raw, (0, self.action_dim - raw.size))
        raw = np.clip(raw[: self.action_dim], -1.0, 1.0)

        if cfg.mode == "absolute":
            u_residual = cfg.beta_absolute * raw[0]
        elif cfg.mode == "relative":
            u_residual = u_base * cfg.beta_relative * raw[0]
        elif cfg.mode == "hybrid":
            u_residual = u_base * cfg.beta_relative * raw[0] + cfg.beta_absolute * raw[1]
        else:
            raise ValueError(f"Unsupported residual mode: {cfg.mode}")

        u_total_pre_clip = float(u_base + u_residual)
        u_total = float(np.clip(u_total_pre_clip, -cfg.output_limit, cfg.output_limit))
        return ControlBreakdown(
            u_base=float(u_base),
            u_residual=float(u_residual),
            u_total_pre_clip=u_total_pre_clip,
            u_total=u_total,
            raw_action=raw.copy(),
        )
