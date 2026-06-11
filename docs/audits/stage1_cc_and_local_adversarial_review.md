# Stage 1 CC And Local Adversarial Review

## Review Target

- Repository: `Residual_RL0611`
- Git target: branch diff against `HEAD~1`
- Main scope: `src/`, `tests/`, `README.md`, `task_plan.md`, `docs/audits/`, and the Superpowers plan.
- Excluded from detailed review: migrated waveform data, legacy notebooks, and copied literature files.

## Claude Code Invocation Evidence

### Attempt 1

- Command kind: `adversarial-review`
- Target: working tree
- Result: failed before review output.
- Error: `spawn ENAMETOOLONG`
- Interpretation: the initial untracked file list was too long on Windows because the new repository contained data files and notebooks.

### Attempt 2

- Command kind: `adversarial-review`
- Target: branch diff against `HEAD~1`
- Job id: `review-mq99bbx0-y0rd6p`
- Status: completed.
- Problem: the job record contains `parseError: Could not parse structured JSON output from Claude Code.`
- Usable findings: none. The captured output only showed Claude Code reading files.

### Attempt 3

- Command kind: `review`
- Target: branch diff against `HEAD~1`
- Job id: `review-mq99zihj-9c6boh`
- Status when checked: running/tool phase after API retry.
- Usable findings at the time of this audit file: none.

## Local Adversarial Findings

### Finding 1: State Availability Is Still A Deployment Risk

The implementation correctly keeps `d_x_hat` diagnostic-only and uses `d_y_hat` as the observation feature. However, the current `d_y_hat` calculation still uses the simulated nominal state `x[k-1]` inside the predictor. This is acceptable for the P0 simulation prototype, but a real vibration table deployment must either measure enough state or add a state estimator.

Action: keep this as an explicit limitation in README and audits. Do not claim hardware deployability yet.

### Finding 2: Observation Dimension Changes Between B And C Groups

The `PID + residual SAC` and `PID + d_y_hat residual SAC` groups cannot have identical input dimensions because the disturbance-aware policy receives three additional features. The fair comparison should state: same PID, same reward, same action space, same hidden width, same training budget, and same disturbance schedule; input dimension differs by design.

Action: document this in future experiment reports.

### Finding 3: Smoke Tests Prove Plumbing, Not Learning Superiority

`pytest` and `scripts/run_smoke.py` verify causality, dimensions, rollout, metrics, and short SAC API integrity. They do not prove that learned `d_y_hat` residual SAC is better than ordinary residual SAC.

Action: require fixed-seed training comparisons before making thesis-level performance claims.

### Finding 4: Windows Line Endings Can Pollute Data And Notebook Diffs

The first Git commit emitted many LF-to-CRLF warnings. This can make future waveform diffs noisy and make notebook provenance harder to audit.

Action taken: added `.gitattributes` to pin text line endings and mark notebook diffs as disabled.

### Finding 5: Full Delay Disturbance Should Stay Out Of P0

Input delay changes action-output alignment and can invalidate the simple `d_y_hat` versus `u_residual` lag interpretation. It should remain P1 until the actuator command history and residual timing are explicitly modeled.

Action: keep delay excluded from P0.

## Verdict

The stage is publishable as a clean P0 project bootstrap, not as a finished experimental result. The repo currently supports the right minimum scientific mechanism: causal output prediction residuals can be computed, filtered, exposed in observation, tested, and evaluated under controlled disturbances.
