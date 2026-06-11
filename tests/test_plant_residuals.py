import numpy as np

from residual_rl0611.config import DisturbanceConfig, PlantConfig
from residual_rl0611.plant import LinearVibrationPlant


def test_no_disturbance_has_zero_prediction_residual():
    plant = LinearVibrationPlant(PlantConfig(), DisturbanceConfig(mode="none"))
    plant.reset()

    info = plant.step(command=10.0, step_index=0)

    assert info.d_y_hat == 0.0
    assert np.allclose(info.d_x_hat, np.zeros(3))


def test_impulse_residual_is_reported_on_transition_that_contains_impulse():
    plant = LinearVibrationPlant(
        PlantConfig(),
        DisturbanceConfig(mode="impulse", impulse_step=2, impulse_magnitude=0.25),
    )
    plant.reset()

    before = plant.step(command=0.0, step_index=1)
    impulse = plant.step(command=0.0, step_index=2)

    assert before.d_y_hat == 0.0
    assert np.isclose(impulse.d_y_hat, 0.25)
    assert np.isclose(impulse.output_disturbance, 0.25)


def test_actuator_gain_residual_uses_command_not_hidden_effective_command():
    cfg = PlantConfig()
    gain = 0.5
    command = 100.0
    plant = LinearVibrationPlant(
        cfg,
        DisturbanceConfig(mode="actuator_gain", actuator_gain=gain, actuator_gain_start=0),
    )
    plant.reset()

    info = plant.step(command=command, step_index=0)
    expected = float((cfg.c @ (cfg.b[:, 0] * (gain - 1.0) * command).reshape(-1, 1)).item())

    assert np.isclose(info.effective_command, gain * command)
    assert np.isclose(info.d_y_hat, expected)
