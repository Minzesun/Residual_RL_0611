"""Reference waveform loading and synthetic reference generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_reference(path: str | Path, max_points: int | None = None, scale: float = 1.0) -> np.ndarray:
    values = np.loadtxt(Path(path), dtype=np.float64)
    values = np.asarray(values, dtype=np.float64).reshape(-1) * scale
    if max_points is not None:
        values = values[:max_points]
    if values.size < 2:
        raise ValueError(f"Reference must contain at least two samples: {path}")
    return values


def sine_reference(
    steps: int,
    amplitude: float = 1.0,
    frequency_hz: float = 2.0,
    dt: float = 0.001,
) -> np.ndarray:
    t = np.arange(steps, dtype=np.float64) * dt
    return amplitude * np.sin(2.0 * np.pi * frequency_hz * t)


def multi_sine_reference(
    steps: int,
    amplitudes: tuple[float, ...] = (0.7, 0.3),
    frequencies_hz: tuple[float, ...] = (2.0, 7.0),
    dt: float = 0.001,
) -> np.ndarray:
    if len(amplitudes) != len(frequencies_hz):
        raise ValueError("amplitudes and frequencies_hz must have the same length")
    t = np.arange(steps, dtype=np.float64) * dt
    out = np.zeros(steps, dtype=np.float64)
    for amp, freq in zip(amplitudes, frequencies_hz):
        out += amp * np.sin(2.0 * np.pi * freq * t)
    return out
