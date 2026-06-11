# Research Question Card: PID-Guided Constrained Residual RL for Vibration Control

## Core Question
Can a bounded residual reinforcement learning policy improve high-frequency vibration-table or seismic-wave tracking when added on top of a fixed PID controller, without sacrificing the PID controller's stabilizing and deployment-friendly properties?

## Target System
- Plant: vibration table / high-frequency servo tracking system.
- Baseline controller: fixed PID, output `u_PID`.
- Learned component: SAC actor or equivalent continuous-control RL policy, output `a_res in [-1, 1]`.
- Final control:
  - absolute residual: `u = sat(u_PID + beta_a * a_res)`
  - relative residual: `u = sat(u_PID * (1 + beta_r * a_res))`

## Hypothesis
A constrained residual policy can compensate for unmodeled dynamics, frequency-dependent tracking errors, actuator effects, and external disturbances while leaving the main stabilizing role to PID. Relative residual constraints should improve safety during exploration, while absolute residual constraints may provide stronger compensation when `u_PID` is small.

## Current Evidence
- Local Zotero collection `残差强化学习` has 57 items, including 26 PDFs.
- Johannink et al. 2019 establishes conventional controller + learned residual superposition.
- Staessens et al. 2022 establishes constrained residual RL and relative residual constraints for mechatronic systems.
- Ishihara et al. 2023 directly studies cascaded PID + residual RL for quadcopter wind resistance.
- Adjacent vibration/seismic literature shows RL is used for active vibration or seismic control, but often as RL tuning or pure RL rather than bounded residual correction around fixed PID.
- Current project already contains notebooks, MATLAB test scripts, fixed PID baseline code, and ONNX actor exports.

## Missing Evidence
- Direct comparison on the user's vibration-table model and reference signals.
- Absolute vs relative residual comparison under actuator saturation.
- Frequency-domain and earthquake-wave hold-out metrics.
- Deployment timing and fallback validation in MATLAB/ONNX.

## Support Criteria
The hypothesis is supported if:
- PID + residual reduces RMSE/MAE or frequency-domain error on held-out references.
- Saturation ratio and control smoothness do not degrade beyond a predefined threshold.
- `beta=0` exactly reproduces PID behavior.
- Actor inference meets sampling-time constraints.
- Results remain stable across at least several seeds or fixed evaluation references.

## Falsification Criteria
The hypothesis is weakened if:
- Residual control improves training references but fails on held-out earthquake references.
- Improvement is only achieved through frequent saturation or high-frequency chattering.
- RL-PID tuning or BO-tuned PID performs equally well with lower risk and lower complexity.
- The residual actor learns near-zero output and does not improve any meaningful metric.
- MATLAB/ONNX deployment cannot meet timing or input-output contract requirements.

## Minimal Next Action
Run a fixed evaluation table with four controllers:
1. fixed PID only
2. fixed PID + zero residual
3. fixed PID + absolute residual SAC
4. fixed PID + relative residual SAC

Use the same reference set and report MAE, RMSE, peak error, saturation ratio, control variation, and PSD/frequency-domain error.
