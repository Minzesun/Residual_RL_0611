# Residual_RL0611

This repository is the clean P0 restart of the earlier `Residual_RL0423` PID + residual SAC vibration-control prototype.

## Research Target

The current target is not "PID plus another RL output" as a generic idea. The P0 target is:

```text
PID baseline control
+ causal equivalent-disturbance representation
+ residual SAC compensation
```

The deployable disturbance representation is the causal output prediction residual:

```text
d_y_hat[k-1] = y[k] - C(A x[k-1] + B u_cmd[k-1])
```

State residuals can be logged for simulation diagnosis, but they are not the main deployable observation feature.

## P0 Evidence Chain

```text
disturbance or model mismatch appears
-> d_y_hat responds causally
-> d_y_hat enters the residual SAC observation
-> residual action changes in relation to d_y_hat
-> tracking and recovery metrics improve under controlled tests
```

## What Was Migrated From Residual_RL0423

- Selected reference waveforms in `data/reference/`.
- Selected earthquake records and `PID.csv` in `data/earthquakes/`.
- Latest notebooks in `legacy/0423/` for provenance:
  - `crrl_vibration_control_8.ipynb`
  - `crrl_vibration_control_8_1.ipynb`
- Research notes and literature artifacts in `docs/research/`.

Large rendered folders, Word reports, old notebook variants, model exports, and temporary comparison files were intentionally excluded.

## Install

```powershell
python -m pip install -e .[dev]
```

## Fast Verification

```powershell
python -m pytest -q
python scripts/run_smoke.py --steps 200
```

## Main Comparison Groups

The intended P0 comparisons are:

1. `PID`
2. `PID + residual SAC`
3. `PID + d_y_hat residual SAC`

The comparison should keep PID gains, SAC implementation, network size, reward, reference signals, and disturbance schedule controlled.

## P0 Disturbance Modes

- `none`
- `impulse`
- `sinusoidal`
- `actuator_gain`

Input delay, full DOB/ESO, PID+DOB, time-frequency reward, recurrent policy, and multi-head residual policy are intentionally outside P0.
