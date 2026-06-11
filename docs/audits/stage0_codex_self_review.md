# Stage 0 Codex Self-Review

## Scope Reviewed

- Project structure.
- P0 research claim.
- Migration boundary.
- Causal equivalent-disturbance implementation plan.

## Main Strengths

- The project is no longer notebook-only.
- `d_y_hat` is defined as a causal output prediction residual.
- Hidden state residual `d_x_hat` is diagnostic-only.
- Disturbance modes are feature-flagged through configuration.
- Metrics and tests are defined before performance claims.

## Main Risks

1. The current smoke verification does not prove learned policy superiority.
2. Full SAC training is intentionally lightweight at this stage and may require tuning.
3. `d_y_hat` currently depends on the nominal model state in simulation; deployment should replace or estimate state if full state is not measured.
4. Input delay disturbance is excluded from P0 because it needs a separate action-alignment design.

## Required Evidence Before Thesis Claim

- Fixed-seed comparison among `PID`, `PID + residual SAC`, and `PID + d_y_hat residual SAC`.
- At least two disturbance modes with the same reference set.
- Same PID, reward, network, and training budget across residual SAC groups.
- Curves for tracking, error, command decomposition, `d_y_hat`, and `d_y_hat` versus residual action.
- Saturation/chatter checks that do not worsen materially.
