import numpy as np

from residual_rl0611.config import DisturbanceConfig, EnvConfig
from residual_rl0611.data import sine_reference
from residual_rl0611.env import VibrationControlEnv


def test_observation_dimension_without_and_with_disturbance_features():
    reference = sine_reference(50)
    plain_env = VibrationControlEnv(EnvConfig(use_disturbance_obs=False), reference=reference)
    d_env = VibrationControlEnv(EnvConfig(use_disturbance_obs=True), reference=reference)

    assert plain_env.reset().shape == (plain_env.obs_dim,)
    assert d_env.reset().shape == (d_env.obs_dim,)
    assert plain_env.obs_dim == 45
    assert d_env.obs_dim == 48


def test_disturbance_features_change_after_causal_impulse_step():
    cfg = EnvConfig(
        use_disturbance_obs=True,
        disturbance=DisturbanceConfig(mode="impulse", impulse_step=0, impulse_magnitude=0.5),
    )
    env = VibrationControlEnv(cfg, reference=np.zeros(20))
    obs0 = env.reset()
    obs1, _, _, info = env.step(np.zeros(env.action_dim))

    assert info["d_y_hat"] == 0.5
    assert np.allclose(obs0[-3:], np.zeros(3))
    assert obs1[-3] > 0.0
    assert obs1[-1] > 0.0


def test_zero_residual_rollout_returns_aligned_arrays():
    env = VibrationControlEnv(EnvConfig(), reference=sine_reference(30))
    rollout = env.rollout_zero_residual(steps=10)

    assert set(rollout) == {
        "reference",
        "output",
        "error",
        "u_base",
        "u_residual",
        "u_total",
        "d_y_hat",
        "filtered_d_y_hat",
    }
    assert all(values.shape == (10,) for values in rollout.values())
