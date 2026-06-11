# Task Plan: Residual_RL0611 Project Bootstrap

## Goal
Build a clean `Residual_RL0611` repository that turns the previous PID + residual SAC prototype into a testable minimum closed loop for equivalent-disturbance-informed PID residual SAC control, then publish it to `Minzesun/Residual_RL_0611.git`.

## Scope
- P0 only: causal output prediction residual `d_y_hat`, disturbance-aware observation, controlled disturbance scenarios, baseline comparisons, and reproducible metrics.
- Excluded from P0: full DOB/ESO, PID+DOB baseline, adaptive reward weights, time-frequency reward, LSTM/GRU policy, multi-head residual policy, and multi-algorithm comparison.

## Phases
- [x] Phase 1: Inspect source project, target folder, tools, and Git/GitHub availability.
- [x] Phase 2: Create persistent planning files and implementation roadmap.
- [x] Phase 3: Migrate necessary source assets from `Residual_RL0423`.
- [x] Phase 4: Implement minimal Python project for equivalent-disturbance residual SAC.
- [x] Phase 5: Add tests and smoke verification.
- [ ] Phase 6: Run Claude Code stage audit and incorporate actionable fixes.
- [ ] Phase 7: Initialize Git repository, commit, connect remote, and push to GitHub.

## Key Questions
1. Can `d_y_hat` be computed causally without using future output or hidden future state?
2. Does the disturbance-aware observation change only the observation surface while keeping SAC, PID, and reward controlled for comparison?
3. Can the repository be verified without heavy full training?
4. Can Git push authenticate to `https://github.com/Minzesun/Residual_RL_0611.git` from this machine?

## Decisions Made
- Use `Residual_RL0423` only as reference input; do not copy historical rendered outputs or Word reports into the new repo.
- Migrate selected reference and earthquake waveform files plus the two latest 0423 notebooks under `legacy/0423/` for provenance.
- Implement a Python package rather than continuing notebook-only development, because the P0 claim needs tests, command-line runs, and repeatable metrics.
- Use causal output residual `d_y_hat` as the deployable disturbance feature; keep state residual available only in diagnostics.

## Errors Encountered
- `Residual_RL0423` is not a Git repository, so no source Git history can be preserved directly.
- `gh` is not installed in PATH; GitHub publication must use plain `git` or the GitHub connector.
- Claude Code plugin skills exist in plugin cache, but callable shell execution must be verified after the new repo has a Git diff.

## Status
**Currently in Phase 6** - Initializing Git and running a stage audit before publication.
