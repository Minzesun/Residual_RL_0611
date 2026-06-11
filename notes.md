# Notes: Residual_RL0611 Bootstrap

## Source Project Findings

### Source Folder
- Path: `D:\00研究生阶段\00主要工作\VScode\RL\Residual_RL\Residual_RL0423`
- Git status: not a Git repository.
- Relevant latest notebooks:
  - `crrl_vibration_control_8.ipynb`
  - `crrl_vibration_control_8_1.ipynb`
- Relevant data folders:
  - `reference/`
  - `大连理工测试地震波工况/`
- Relevant research notes:
  - `research_notes.md`
  - `research_question_card.md`
  - `research_task_plan.md`
  - `residual_rl_pid_literature_landscape.md`
  - `residual_rl_pid_literature_matrix.md`
  - `zotero_residual_rl_collection.md`
  - `zotero_residual_rl_collection.json`

### Current Prototype Characteristics
- Current control concept: PID baseline plus residual SAC action.
- Latest inspected environment used a 3-state linear plant with matrices `A`, `B`, and `C`.
- Observation used base error/control features, residual action history, and future reference preview.
- There was no implemented `d_hat`, `d_y_hat`, DOB, ESO, disturbance injection mode, or disturbance-aware observation.
- Existing training results were not consistently superior across sweep and earthquake tests, so the new project must preserve strict baseline comparisons.

## P0 Technical Claim
The new project should support this minimum evidence chain:

```text
disturbance or model mismatch appears
-> causal prediction residual d_y_hat responds
-> d_y_hat enters the residual SAC observation
-> residual action changes in relation to d_y_hat
-> tracking error and disturbance recovery improve under controlled tests
```

## Risk Notes
- If state residual `d_x_hat` is used as the main observation, the method can be criticized as relying on simulation-only hidden state.
- If disturbance modes, reward, network, and observation all change together, the improvement cannot be attributed to disturbance representation.
- If metrics are defined after seeing results, the thesis claim becomes vulnerable to cherry-picking.

## Planned Verification
- Unit tests for causal residual alignment.
- Unit tests for disturbance mode behavior.
- Unit tests for observation dimension toggling.
- Smoke rollout that runs PID and fixed residual policies without long training.
- Optional short SAC smoke training for API integrity, not for thesis-level performance claims.
