# Notes: Residual RL + PID Literature Landscape

## Local Project Context
- Project root: `D:\00研究生阶段\00主要工作\VScode\RL\Residual_RL\Residual_RL0423`.
- Current project already contains a residual RL/PID vibration-control line:
  - `crrl_vibration_control*.ipynb`: multiple notebook versions.
  - `crrl_test.m`: MATLAB test script loading SAC actor ONNX, with 35-dimensional observation and one-dimensional residual action.
  - `Fixed_PID.m`: fixed PID vibration-table baseline.
  - `exports/`: `crrl_actor.pt`, `crrl_actor.onnx`, `crrl_actor_5.onnx`, `crrl_actor_r2019a.onnx`.
- Notebook keyword scan:
  - `crrl_vibration_control_8_1.ipynb`: 30 cells, ~51.8k chars, `residual` 108 hits, `absolute` 27 hits, `relative` 13 hits, `beta` 19 hits.
  - `crrl_vibration_control_6.ipynb` to `_8_1.ipynb` include earthquake-related references in text.
- Local implication:
  - The research framing should not start from scratch.
  - It should treat this as a PID-stabilized vibration tracking system with a learned residual action and MATLAB/ONNX deployment constraints.

## Zotero Evidence
- Zotero database: `D:\Zotero_sunzemin51\zotero.sqlite`.
- Access mode used: SQLite `mode=ro&immutable=1`.
- Target collection: `残差强化学习`, `collectionID=64`.
- Parent collection: collection ID 49.
- Extracted items: 57.
- Items with PDF attachments: 26.
- Derived files:
  - `zotero_residual_rl_collection.json`
  - `zotero_residual_rl_collection.md`
  - `residual_rl_pid_literature_matrix.md`

## Zotero Collection Distribution
- Years:
  - 2019: 1
  - 2022: 5
  - 2023: 11
  - 2024: 12
  - 2025: 19
  - 2026: 9
- Top venues by item count:
  - IEEE Robotics and Automation Letters: 7
  - Expert Systems with Applications: 3
  - IEEE Transactions on Cybernetics: 3
  - IEEE Transactions on Industrial Electronics: 3
  - IEEE Transactions on Automation Science and Engineering: 3
  - IEEE/ASME Transactions on Mechatronics: 1

## First-Pass Literature Themes
- Core: 3
  - Johannink et al., 2019, residual RL for robot control.
  - Staessens et al., 2022, constrained residual RL for mechatronic systems.
  - Ishihara et al., 2023, residual RL on top of cascaded PID quadcopter control.
- Direct: 5
  - PID/mechatronic/cascaded/tracking/control-prior papers.
- Adjacent: 26
  - Residual RL applications in robot manipulation, navigation, active tracking, or bounded policies.
- Background: 23
  - General RL control, robust RL, motion planning, or related control-learning hybrids.

## Core PDF Evidence Extracted
- Johannink et al., 2019:
  - Conventional feedback controller handles structured part.
  - RL learns the residual.
  - Final policy is the superposition of both control signals.
  - Local PDF: `D:\Zotero_sunzemin51\storage\3V6LXQCD\Johannink 等 - 2019 - Residual reinforcement learning for robot control.pdf`.
- Staessens et al., 2022:
  - Conventional controller provides base output.
  - RL learns corrective adaptations.
  - Residual actions are constrained, especially relative to the base controller.
  - Includes Lyapunov-stability-oriented argument and experimental slider-crank validation.
  - Local PDF: `D:\Zotero_sunzemin51\storage\TENTZFCY\Staessens 等 - 2022 - Adaptive control of a mechatronic system using constrained residual reinforcement learning.pdf`.
- Ishihara et al., 2023:
  - Cascaded PID controller remains as base controller.
  - Residual RL learns disturbance compensation for wind.
  - Training in simulator, direct deployment to hardware.
  - Reports about 50% position-deviation reduction under strong wind.
  - Local PDF: `D:\Zotero_sunzemin51\storage\JHDNGSNY\Ishihara 等 - 2023 - Improving wind resistance performance of cascaded PID controlled quadcopters using residual reinforc.pdf`.
- Li et al., 2024:
  - Data-informed residual RL for high-dimensional robotic tracking.
  - Useful for sample efficiency and scalable tracking-control framing.
- Dodeja et al., 2025:
  - Residual RL plus uncertainty estimation.
  - Useful for exploration/sample-efficiency extensions, but not PID-specific.
- Ma et al., 2026:
  - Value-learning bottlenecks in residual RL.
  - Useful for diagnosing critic learning when residual action is small relative to base action.
- Rana et al., 2023:
  - Bayesian controller fusion.
  - Adjacent alternative to additive residual action: uncertainty-aware arbitration between control prior and RL.

## External Search Evidence
- Search path:
  - Academic MCP search was attempted but failed with `asyncio.run() cannot be called from a running event loop`.
  - Web search was used for supplementary verification.
- Key external sources:
  - Johannink et al., "Residual Reinforcement Learning for Robot Control", arXiv:1812.03201.
  - Staessens et al., "Adaptive control of a mechatronic system using constrained residual reinforcement learning", arXiv:2110.02566.
  - Ishihara et al., "Improving Wind Resistance Performance of Cascaded PID Controlled Quadcopters using Residual Reinforcement Learning", arXiv:2308.01648.
  - Staessens et al., "Optimizing Cascaded Control of Mechatronic Systems through Constrained Residual Reinforcement Learning", Machines 2023, DOI: 10.3390/machines11030402.
  - Yang et al., "Prior-Guided Residual Reinforcement Learning for Active Suspension Control", Machines 2025, DOI: 10.3390/machines13110983.
  - Febvre et al., "Deep reinforcement learning for tuning active vibration control on a smart piezoelectric beam", Journal of Intelligent Material Systems and Structures 2024, DOI: 10.1177/1045389X241260976.
  - Khalatbarisoltani et al., "Online control of an active seismic system via reinforcement learning", Structural Control and Health Monitoring 2019, DOI: 10.1002/stc.2298.
  - Zhou et al., "A active vibration control strategy based on reinforcement learning", Journal of Vibration and Shock 2021.

## Synthesized Findings
- The general residual RL idea is well established in robotics.
- The constrained residual RL variant is more directly aligned with industrial/mechatronic safety needs.
- The PID-specific residual RL example exists in quadcopters, but the vibration-table or seismic-tracking version is still underexplored.
- Active vibration control literature has RL and PID baselines, but it often tunes controllers or uses pure RL rather than learning a bounded residual on top of a fixed PID output.
- The strongest research opening is therefore: constrained residual RL as an add-on to fixed PID for high-frequency vibration-table or seismic-reference tracking, with absolute-vs-relative residual constraints, actuator saturation, frequency-domain metrics, and MATLAB/ONNX deployment.
