import numpy as np

from residual_rl0611.metrics import compute_tracking_metrics, lagged_correlation, recovery_time


def test_tracking_metrics_are_deterministic():
    reference = np.array([0.0, 1.0, 2.0])
    output = np.array([0.0, 0.5, 1.0])
    control = np.array([0.0, 2.0, 4.0])

    metrics = compute_tracking_metrics(reference, output, control, u_max=4.0)

    assert np.isclose(metrics.rmse, np.sqrt((0.0**2 + 0.5**2 + 1.0**2) / 3.0))
    assert np.isclose(metrics.mae, 0.5)
    assert np.isclose(metrics.max_error, 1.0)
    assert np.isclose(metrics.control_increment_rms, 2.0)
    assert metrics.saturation_count == 1


def test_recovery_time_requires_hold_window():
    error = np.array([1.0, 0.8, 0.2, 0.1, 0.05, 0.04])

    assert recovery_time(error, start_step=0, threshold=0.1, dt=0.01, hold_steps=2) == 0.03
    assert recovery_time(error, start_step=0, threshold=0.01, dt=0.01, hold_steps=2) is None


def test_lagged_correlation_finds_shift():
    x = np.array([0.0, 1.0, 0.0, -1.0, 0.0, 1.0])
    y = np.roll(x, 1)

    lag, corr = lagged_correlation(x, y, max_lag=2)

    assert lag == 1
    assert corr > 0.9
