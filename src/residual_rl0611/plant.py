"""Linear plant and causal equivalent-disturbance residuals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DisturbanceConfig, PlantConfig


@dataclass(frozen=True)
class PlantStepInfo:
    step_index: int
    command: float
    effective_command: float
    output_disturbance: float
    y_pred_nominal: float
    y_next: float
    d_y_hat: float
    d_x_hat: np.ndarray
    x_prev: np.ndarray
    x_pred_nominal: np.ndarray
    x_next: np.ndarray


class LinearVibrationPlant:
    """Three-state nominal plant with explicit mismatch/disturbance injection."""

    def __init__(
        self,
        config: PlantConfig | None = None,
        disturbance: DisturbanceConfig | None = None,
    ) -> None:
        self.config = config or PlantConfig()
        self.disturbance = disturbance or DisturbanceConfig()
        self.x = np.zeros(3, dtype=np.float64)

    def reset(self, x0: np.ndarray | None = None) -> float:
        self.x = np.zeros(3, dtype=np.float64) if x0 is None else np.asarray(x0, dtype=np.float64)
        if self.x.shape != (3,):
            raise ValueError(f"x0 must have shape (3,), got {self.x.shape}")
        return self.output

    @property
    def output(self) -> float:
        return float((self.config.c @ self.x.reshape(-1, 1)).item())

    def effective_command(self, command: float, step_index: int) -> float:
        if (
            self.disturbance.mode == "actuator_gain"
            and step_index >= self.disturbance.actuator_gain_start
        ):
            return self.disturbance.actuator_gain * command
        return command

    def output_disturbance(self, step_index: int) -> float:
        mode = self.disturbance.mode
        if mode == "impulse" and step_index == self.disturbance.impulse_step:
            return self.disturbance.impulse_magnitude
        if mode == "sinusoidal":
            t = step_index * self.config.dt
            return self.disturbance.sinusoidal_amplitude * np.sin(
                2.0 * np.pi * self.disturbance.sinusoidal_frequency_hz * t
            )
        return 0.0

    def step(self, command: float, step_index: int) -> PlantStepInfo:
        command = float(np.clip(command, -self.config.u_max, self.config.u_max))
        x_prev = self.x.copy()

        # The nominal predictor uses the issued command, not the hidden effective command.
        x_pred_nominal = self.config.a @ x_prev + self.config.b[:, 0] * command
        y_pred_nominal = float((self.config.c @ x_pred_nominal.reshape(-1, 1)).item())

        effective_command = self.effective_command(command, step_index)
        x_next = self.config.a @ x_prev + self.config.b[:, 0] * effective_command

        disturbance_output = self.output_disturbance(step_index)
        if disturbance_output:
            output_gain = float(self.config.c[0, -1])
            if abs(output_gain) < 1e-12:
                raise ValueError("Cannot inject output-equivalent disturbance with zero output gain")
            x_next[-1] += disturbance_output / output_gain

        y_next = float((self.config.c @ x_next.reshape(-1, 1)).item())
        d_y_hat = y_next - y_pred_nominal
        d_x_hat = x_next - x_pred_nominal
        self.x = x_next

        return PlantStepInfo(
            step_index=step_index,
            command=command,
            effective_command=float(effective_command),
            output_disturbance=float(disturbance_output),
            y_pred_nominal=y_pred_nominal,
            y_next=y_next,
            d_y_hat=float(d_y_hat),
            d_x_hat=d_x_hat,
            x_prev=x_prev,
            x_pred_nominal=x_pred_nominal,
            x_next=x_next.copy(),
        )
