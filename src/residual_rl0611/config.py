"""Configuration objects for the Residual_RL0611 prototype."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


DisturbanceMode = Literal["none", "impulse", "sinusoidal", "actuator_gain"]
ResidualMode = Literal["absolute", "relative", "hybrid"]


@dataclass
class PlantConfig:
    """Nominal linear vibration-table model copied from the 0423 prototype."""

    dt: float = 0.001
    u_max: float = 400.0
    a: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [0.613991991741585, -91.77054541544227, 0.0],
                [0.0016217442129418348, 0.9009155731850262, 0.0],
                [1.7509931433054655e-06, 0.0019315348813782334, 1.0],
            ],
            dtype=np.float64,
        )
    )
    b: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [0.0016217442129418346],
                [1.7509931433054706e-06],
                [1.2098970253536503e-09],
            ],
            dtype=np.float64,
        )
    )
    c: np.ndarray = field(
        default_factory=lambda: np.array([[0.0, 0.0, 433053.63655287167]], dtype=np.float64)
    )


@dataclass
class DisturbanceConfig:
    mode: DisturbanceMode = "none"
    impulse_step: int = 100
    impulse_magnitude: float = 0.25
    sinusoidal_amplitude: float = 0.08
    sinusoidal_frequency_hz: float = 4.0
    actuator_gain: float = 0.8
    actuator_gain_start: int = 0


@dataclass
class PIDConfig:
    kp: float = 9.0
    ki: float = 0.0
    kd: float = 0.0
    dt: float = 0.001
    integral_limit: float = 100.0
    output_limit: float = 400.0


@dataclass
class ResidualConfig:
    mode: ResidualMode = "hybrid"
    beta_relative: float = 0.6
    beta_absolute: float = 8.0
    output_limit: float = 400.0


@dataclass
class EnvConfig:
    plant: PlantConfig = field(default_factory=PlantConfig)
    disturbance: DisturbanceConfig = field(default_factory=DisturbanceConfig)
    pid: PIDConfig = field(default_factory=PIDConfig)
    residual: ResidualConfig = field(default_factory=ResidualConfig)
    use_disturbance_obs: bool = False
    lookahead_horizon: int = 15
    action_history_len: int = 10
    error_window: int = 20
    residual_lpf_alpha: float = 0.9
    residual_rms_window: int = 20
    disturbance_norm: float = 1.0
    max_delta_u: float = 80.0
    reward_smoothness_weight: float = 0.01
    reward_saturation_weight: float = 0.05


@dataclass
class TrainingConfig:
    seed: int = 7
    episodes: int = 10
    steps_per_episode: int = 800
    batch_size: int = 128
    replay_size: int = 100_000
    warmup_steps: int = 500
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    hidden_dim: int = 128
