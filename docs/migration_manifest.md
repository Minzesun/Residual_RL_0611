# Migration Manifest

## Source

`D:\00研究生阶段\00主要工作\VScode\RL\Residual_RL\Residual_RL0423`

## Migrated Data

### Reference Waveforms

- `reference/0_50Hz_0.4mm.txt` -> `data/reference/0_50Hz_0.4mm.txt`
- `reference/0_50Hz_1mm.txt` -> `data/reference/0_50Hz_1mm.txt`
- `reference/Tohuku_X_SF1_TSF1.txt` -> `data/reference/Tohuku_X_SF1_TSF1.txt`

### Earthquake Cases

- `大连理工测试地震波工况/CHICHI_X_SF1_TSF0.2.txt` -> `data/earthquakes/CHICHI_X_SF1_TSF0.2.txt`
- `大连理工测试地震波工况/KOBE_X_SF1.8_TSF0.2.txt` -> `data/earthquakes/KOBE_X_SF1.8_TSF0.2.txt`
- `大连理工测试地震波工况/TABAS_X_SF1_TSF0.4.txt` -> `data/earthquakes/TABAS_X_SF1_TSF0.4.txt`
- `大连理工测试地震波工况/TORTHR_X_SF1_TSF0.2.txt` -> `data/earthquakes/TORTHR_X_SF1_TSF0.2.txt`
- `大连理工测试地震波工况/PID.csv` -> `data/earthquakes/PID.csv`

### Provenance

- `crrl_vibration_control_8.ipynb` -> `legacy/0423/crrl_vibration_control_8.ipynb`
- `crrl_vibration_control_8_1.ipynb` -> `legacy/0423/crrl_vibration_control_8_1.ipynb`

### Research Notes

- `research_notes.md`
- `research_question_card.md`
- `research_task_plan.md`
- `residual_rl_pid_literature_landscape.md`
- `residual_rl_pid_literature_matrix.md`
- `zotero_residual_rl_collection.md`
- `zotero_residual_rl_collection.json`

All research notes were copied into `docs/research/`.

## Intentionally Excluded

- Historical notebook variants before `crrl_vibration_control_8`.
- Rendered report folders.
- Word reports.
- MATLAB generated result artifacts.
- Exported `.onnx` and `.pt` models from the previous prototype.
- Large raw diff/debug files.

## Rationale

The new repository should preserve the evidence needed to reproduce and explain P0 experiments, while avoiding historical clutter that makes the scientific claim harder to audit.
