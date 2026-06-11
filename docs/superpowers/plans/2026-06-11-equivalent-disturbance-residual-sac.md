# Equivalent Disturbance Residual SAC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible P0 project where causal output prediction residuals augment PID residual SAC observations for vibration table tracking under controlled disturbance modes.

**Architecture:** The project is a small Python package with separate modules for plant dynamics, controllers, environment, SAC components, metrics, training, and evaluation. The deployable equivalent disturbance feature is `d_y_hat`, computed from the previous state/action prediction and current output; state residuals are diagnostics only. CLI scripts run smoke tests and short experiments without requiring notebook execution.

**Tech Stack:** Python 3.12, NumPy, PyTorch, Matplotlib, Pytest, plain Git.

---

## File Structure

- `src/residual_rl0611/config.py`: Dataclass configuration for plant, PID, residual action, observation, disturbance, and training.
- `src/residual_rl0611/plant.py`: Linear vibration table plant, disturbance injection, and causal prediction residual calculations.
- `src/residual_rl0611/controllers.py`: PID controller and residual action combiner.
- `src/residual_rl0611/env.py`: Gym-like environment exposing baseline and disturbance-aware observations.
- `src/residual_rl0611/sac.py`: Minimal continuous-action SAC implementation for residual policies.
- `src/residual_rl0611/metrics.py`: Tracking, recovery, saturation, smoothness, and lagged-correlation metrics.
- `src/residual_rl0611/data.py`: Reference waveform loading and synthetic signal generation.
- `src/residual_rl0611/train.py`: Training entry point.
- `src/residual_rl0611/evaluate.py`: Evaluation entry point.
- `scripts/run_smoke.py`: Fast local verification script.
- `tests/`: Unit tests for causality, disturbance behavior, observation dimensions, and metrics.
- `docs/audits/`: Stage audit requests and results.

## Task 1: Project Skeleton

**Files:**
- Create: `README.md`
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/residual_rl0611/__init__.py`

- [x] Create a Python package with editable install support.
- [x] Add README sections for scope, P0 claim, migrated assets, commands, and limits.
- [x] Add `.gitignore` entries for caches, checkpoints, run outputs, and virtual environments.
- [x] Verify with `python -m pytest -q`.

## Task 2: Plant And Causal Residuals

**Files:**
- Create: `src/residual_rl0611/config.py`
- Create: `src/residual_rl0611/plant.py`
- Test: `tests/test_plant_residuals.py`

- [x] Implement the 3-state linear model from the 0423 prototype.
- [x] Implement disturbance modes: none, impulse, sinusoidal, and actuator_gain.
- [x] Implement `d_y_hat[k-1] = y[k] - C(A x[k-1] + B u_cmd[k-1])`.
- [x] Keep `d_x_hat` diagnostic-only.
- [x] Test that `d_y_hat` uses previous transition information and no future output.

## Task 3: Controllers And Environment

**Files:**
- Create: `src/residual_rl0611/controllers.py`
- Create: `src/residual_rl0611/env.py`
- Test: `tests/test_environment.py`

- [x] Implement PID with resettable integral and derivative state.
- [x] Implement residual action modes: absolute, relative, and hybrid.
- [x] Build environment observation with `use_disturbance_obs=False` and `True`.
- [x] Add filtered residual features: `d_y_hat`, `delta_d_y_hat`, and RMS window.
- [x] Test observation dimensions and reset/step behavior.

## Task 4: Metrics And Data

**Files:**
- Create: `src/residual_rl0611/metrics.py`
- Create: `src/residual_rl0611/data.py`
- Test: `tests/test_metrics.py`

- [x] Implement RMSE, MAE, max error, control increment RMS, saturation count, recovery time, and lagged correlation.
- [x] Implement reference text-file loader and synthetic sine helpers.
- [x] Test metrics on deterministic toy sequences.

## Task 5: SAC And CLI

**Files:**
- Create: `src/residual_rl0611/sac.py`
- Create: `src/residual_rl0611/train.py`
- Create: `src/residual_rl0611/evaluate.py`
- Create: `scripts/run_smoke.py`

- [x] Implement minimal SAC agent compatible with the environment action dimension.
- [x] Add a smoke path using random or zero residual action to avoid long training.
- [x] Add short training command for API verification.
- [x] Verify with `python scripts/run_smoke.py --steps 200`.

## Task 6: Migration And Documentation

**Files:**
- Copy selected assets into `data/`.
- Copy latest notebooks into `legacy/0423/`.
- Create: `docs/audits/stage0_codex_self_review.md`

- [x] Copy selected reference files.
- [x] Copy selected earthquake files.
- [x] Copy `crrl_vibration_control_8.ipynb` and `crrl_vibration_control_8_1.ipynb`.
- [x] Document what was copied and what was intentionally excluded.

## Task 7: Stage Audit And Publication

**Files:**
- Update: `docs/audits/`
- Update: `task_plan.md`

- [ ] Initialize Git repository.
- [ ] Run tests.
- [ ] Run Claude Code adversarial review on the stage content if the companion tool is callable.
- [ ] Fix critical and important review issues that are within P0 scope.
- [ ] Commit and push to `https://github.com/Minzesun/Residual_RL_0611.git`.

## Self-Review Checklist

- [ ] The plan maps every P0 requirement to a file and verification step.
- [ ] The plan does not require full DOB/ESO or time-frequency reward in P0.
- [ ] The deployable disturbance feature is output residual `d_y_hat`, not hidden state residual.
- [ ] Comparison groups can be run with the same PID, SAC, reward, and network settings.
- [ ] Tests can run quickly on a normal local machine.
