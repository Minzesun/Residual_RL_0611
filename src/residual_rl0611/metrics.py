"""Evaluation metrics for vibration tracking and disturbance recovery."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrackingMetrics:
    rmse: float
    mae: float
    max_error: float
    control_increment_rms: float
    saturation_count: int


def compute_tracking_metrics(reference: np.ndarray, output: np.ndarray, control: np.ndarray, u_max: float) -> TrackingMetrics:
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    y = np.asarray(output, dtype=np.float64).reshape(-1)
    u = np.asarray(control, dtype=np.float64).reshape(-1)
    n = min(ref.size, y.size)
    if n == 0:
        raise ValueError("reference and output must be non-empty")
    err = ref[:n] - y[:n]
    du = np.diff(u) if u.size > 1 else np.array([0.0])
    return TrackingMetrics(
        rmse=float(np.sqrt(np.mean(err**2))),
        mae=float(np.mean(np.abs(err))),
        max_error=float(np.max(np.abs(err))),
        control_increment_rms=float(np.sqrt(np.mean(du**2))),
        saturation_count=int(np.sum(np.abs(u) >= 0.98 * u_max)),
    )


def recovery_time(
    error: np.ndarray,
    start_step: int,
    threshold: float,
    dt: float,
    hold_steps: int = 20,
) -> float | None:
    err = np.abs(np.asarray(error, dtype=np.float64).reshape(-1))
    if start_step >= err.size:
        return None
    for idx in range(start_step, err.size):
        window = err[idx : idx + hold_steps]
        if window.size < hold_steps:
            return None
        if np.all(window <= threshold):
            return float((idx - start_step) * dt)
    return None


def lagged_correlation(x: np.ndarray, y: np.ndarray, max_lag: int) -> tuple[int, float]:
    xs = np.asarray(x, dtype=np.float64).reshape(-1)
    ys = np.asarray(y, dtype=np.float64).reshape(-1)
    n = min(xs.size, ys.size)
    if n < 3:
        return 0, 0.0
    xs = xs[:n]
    ys = ys[:n]
    best_lag = 0
    best_corr = 0.0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a, b = xs[-lag:], ys[: n + lag]
        elif lag > 0:
            a, b = xs[: n - lag], ys[lag:]
        else:
            a, b = xs, ys
        if a.size < 3:
            continue
        a_std = float(np.std(a))
        b_std = float(np.std(b))
        if a_std < 1e-12 or b_std < 1e-12:
            corr = 0.0
        else:
            corr = float(np.corrcoef(a, b)[0, 1])
        if abs(corr) > abs(best_corr):
            best_lag = lag
            best_corr = corr
    return best_lag, best_corr
