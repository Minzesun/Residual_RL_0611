"""Evaluation helpers for baseline and residual policies."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import DisturbanceConfig, EnvConfig
from .data import load_reference, sine_reference
from .env import VibrationControlEnv
from .metrics import compute_tracking_metrics, lagged_correlation


def evaluate_zero_residual(
    reference: np.ndarray,
    use_disturbance_obs: bool = False,
    disturbance_mode: str = "none",
    steps: int | None = None,
) -> dict[str, float]:
    env_cfg = EnvConfig(
        use_disturbance_obs=use_disturbance_obs,
        disturbance=DisturbanceConfig(mode=disturbance_mode),
    )
    env = VibrationControlEnv(env_cfg, reference=reference)
    rollout = env.rollout_zero_residual(steps=steps)
    metrics = compute_tracking_metrics(
        rollout["reference"],
        rollout["output"],
        rollout["u_total"],
        env_cfg.plant.u_max,
    )
    lag, corr = lagged_correlation(rollout["d_y_hat"], rollout["u_residual"], max_lag=50)
    return {
        "rmse": metrics.rmse,
        "mae": metrics.mae,
        "max_error": metrics.max_error,
        "control_increment_rms": metrics.control_increment_rms,
        "saturation_count": float(metrics.saturation_count),
        "d_y_u_residual_lag": float(lag),
        "d_y_u_residual_corr": float(corr),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--use-disturbance-obs", action="store_true")
    parser.add_argument("--disturbance-mode", default="none")
    args = parser.parse_args()
    if args.reference:
        reference = load_reference(args.reference, max_points=args.steps + 1)
    else:
        reference = sine_reference(args.steps + 1, amplitude=1.0, frequency_hz=3.0)
    print(
        evaluate_zero_residual(
            reference=reference,
            use_disturbance_obs=args.use_disturbance_obs,
            disturbance_mode=args.disturbance_mode,
            steps=args.steps,
        )
    )


if __name__ == "__main__":
    main()
