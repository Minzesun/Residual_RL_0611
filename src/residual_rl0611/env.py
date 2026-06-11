"""Gym-like vibration control environment."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from .config import EnvConfig
from .controllers import PIDController, ResidualActionCombiner
from .data import sine_reference
from .plant import LinearVibrationPlant


class VibrationControlEnv:
    """PID baseline plus residual-action environment.

    The observation can optionally include causal output prediction residual
    features. No future output is used.
    """

    base_obs_dim = 10

    def __init__(self, config: EnvConfig | None = None, reference: np.ndarray | None = None) -> None:
        self.config = config or EnvConfig()
        self.reference = (
            np.asarray(reference, dtype=np.float64).reshape(-1)
            if reference is not None
            else sine_reference(1000, dt=self.config.plant.dt)
        )
        if self.reference.size < 2:
            raise ValueError("reference must contain at least two samples")

        self.plant = LinearVibrationPlant(self.config.plant, self.config.disturbance)
        self.pid = PIDController(self.config.pid)
        self.residual = ResidualActionCombiner(self.config.residual)
        self.action_dim = self.residual.action_dim
        self.action_history = np.zeros(
            (self.config.action_history_len, self.action_dim), dtype=np.float64
        )
        self.error_buffer: deque[float] = deque(maxlen=self.config.error_window)
        self.d_y_buffer: deque[float] = deque(maxlen=self.config.residual_rms_window)
        self.current_step = 0
        self.y_current = 0.0
        self.prev_error = 0.0
        self.prev_command = 0.0
        self.prev_base_command = 0.0
        self.filtered_d_y_hat = 0.0
        self.prev_filtered_d_y_hat = 0.0
        self.last_info: dict[str, Any] = {}

    @property
    def obs_dim(self) -> int:
        dim = (
            self.base_obs_dim
            + self.config.action_history_len * self.action_dim
            + self.config.lookahead_horizon
        )
        if self.config.use_disturbance_obs:
            dim += 3
        return dim

    @property
    def ref_scale(self) -> float:
        return float(max(np.max(np.abs(self.reference)), 1.0))

    def reset(self, reference: np.ndarray | None = None) -> np.ndarray:
        if reference is not None:
            self.reference = np.asarray(reference, dtype=np.float64).reshape(-1)
            if self.reference.size < 2:
                raise ValueError("reference must contain at least two samples")
        self.current_step = 0
        self.y_current = self.plant.reset()
        self.pid.reset()
        self.action_history.fill(0.0)
        self.error_buffer.clear()
        self.d_y_buffer.clear()
        self.prev_error = 0.0
        self.prev_command = 0.0
        self.prev_base_command = 0.0
        self.filtered_d_y_hat = 0.0
        self.prev_filtered_d_y_hat = 0.0
        self.last_info = {}
        return self._make_obs()

    def step(self, residual_action: np.ndarray | list[float] | tuple[float, ...] | None = None):
        if self.current_step >= self.reference.size - 1:
            raise RuntimeError("step called after episode finished")

        raw_action = (
            np.zeros(self.action_dim, dtype=np.float64)
            if residual_action is None
            else np.asarray(residual_action, dtype=np.float64).reshape(-1)
        )
        ref_now = self._reference_at(self.current_step)
        error_now = ref_now - self.y_current
        u_base = self.pid.update(error_now)
        breakdown = self.residual.combine(u_base, raw_action)
        u_total = self._rate_limit(breakdown.u_total)

        plant_info = self.plant.step(u_total, self.current_step)
        self._push_action_history(breakdown.raw_action)
        self._update_disturbance_features(plant_info.d_y_hat)

        self.prev_base_command = u_base
        command_delta = u_total - self.prev_command
        self.prev_command = u_total
        self.y_current = plant_info.y_next
        self.current_step += 1

        ref_next = self._reference_at(self.current_step)
        error_next = ref_next - self.y_current
        self.error_buffer.append(float(error_next))

        norm_error = abs(error_next) / self.ref_scale
        smoothness = (command_delta / self.config.plant.u_max) ** 2
        saturation = abs(u_total) >= 0.98 * self.config.plant.u_max
        reward = -norm_error - self.config.reward_smoothness_weight * smoothness
        if saturation:
            reward -= self.config.reward_saturation_weight

        self.prev_error = float(error_now)
        done = self.current_step >= self.reference.size - 1
        self.last_info = {
            "step": self.current_step,
            "reference": ref_next,
            "output": self.y_current,
            "error": error_next,
            "u_base": u_base,
            "u_residual": breakdown.u_residual,
            "u_total": u_total,
            "raw_action": breakdown.raw_action.copy(),
            "d_y_hat": plant_info.d_y_hat,
            "filtered_d_y_hat": self.filtered_d_y_hat,
            "d_x_hat": plant_info.d_x_hat.copy(),
            "effective_command": plant_info.effective_command,
            "output_disturbance": plant_info.output_disturbance,
            "saturation": saturation,
        }
        return self._make_obs(), float(reward), done, dict(self.last_info)

    def rollout_zero_residual(self, steps: int | None = None) -> dict[str, np.ndarray]:
        obs = self.reset()
        del obs
        limit = min(steps or self.reference.size - 1, self.reference.size - 1)
        rows: dict[str, list[float]] = {
            "reference": [],
            "output": [],
            "error": [],
            "u_base": [],
            "u_residual": [],
            "u_total": [],
            "d_y_hat": [],
            "filtered_d_y_hat": [],
        }
        for _ in range(limit):
            _, _, done, info = self.step(np.zeros(self.action_dim, dtype=np.float64))
            for key in rows:
                rows[key].append(float(info[key]))
            if done:
                break
        return {key: np.asarray(values, dtype=np.float64) for key, values in rows.items()}

    def _reference_at(self, index: int) -> float:
        safe_index = min(max(index, 0), self.reference.size - 1)
        return float(self.reference[safe_index])

    def _rate_limit(self, command: float) -> float:
        delta = float(np.clip(command - self.prev_command, -self.config.max_delta_u, self.config.max_delta_u))
        return float(np.clip(self.prev_command + delta, -self.config.plant.u_max, self.config.plant.u_max))

    def _push_action_history(self, raw_action: np.ndarray) -> None:
        self.action_history[:-1] = self.action_history[1:]
        self.action_history[-1] = raw_action[: self.action_dim]

    def _update_disturbance_features(self, d_y_hat: float) -> None:
        self.prev_filtered_d_y_hat = self.filtered_d_y_hat
        alpha = self.config.residual_lpf_alpha
        self.filtered_d_y_hat = alpha * self.filtered_d_y_hat + (1.0 - alpha) * d_y_hat
        self.d_y_buffer.append(float(self.filtered_d_y_hat))

    def _make_obs(self) -> np.ndarray:
        ref = self._reference_at(self.current_step)
        error = ref - self.y_current
        de = error - self.prev_error
        if self.error_buffer:
            mean_abs_error = float(np.mean(np.abs(np.asarray(self.error_buffer))))
            std_error = float(np.std(np.asarray(self.error_buffer)))
        else:
            mean_abs_error = abs(error)
            std_error = 0.0
        sat_margin = (self.config.plant.u_max - abs(self.prev_command)) / self.config.plant.u_max
        base = np.array(
            [
                error / self.ref_scale,
                de / self.ref_scale,
                mean_abs_error / self.ref_scale,
                std_error / self.ref_scale,
                self.prev_base_command / self.config.plant.u_max,
                self.prev_command / self.config.plant.u_max,
                ref / self.ref_scale,
                self.y_current / self.ref_scale,
                np.tanh(self.pid.integral / 10.0),
                sat_margin,
            ],
            dtype=np.float64,
        )
        parts = [base, self.action_history.reshape(-1), self._reference_preview()]
        if self.config.use_disturbance_obs:
            d_norm = max(self.ref_scale, self.config.disturbance_norm)
            d_arr = np.asarray(self.d_y_buffer, dtype=np.float64)
            d_rms = float(np.sqrt(np.mean(d_arr**2))) if d_arr.size else 0.0
            disturbance_features = np.array(
                [
                    self.filtered_d_y_hat / d_norm,
                    (self.filtered_d_y_hat - self.prev_filtered_d_y_hat) / d_norm,
                    d_rms / d_norm,
                ],
                dtype=np.float64,
            )
            parts.append(disturbance_features)
        obs = np.concatenate(parts).astype(np.float32)
        if obs.size != self.obs_dim:
            raise RuntimeError(f"Observation size mismatch: expected {self.obs_dim}, got {obs.size}")
        return obs

    def _reference_preview(self) -> np.ndarray:
        horizon = self.config.lookahead_horizon
        idx = np.arange(self.current_step, self.current_step + horizon)
        clipped = np.clip(idx, 0, self.reference.size - 1)
        return (self.reference[clipped] / self.ref_scale).astype(np.float64)
