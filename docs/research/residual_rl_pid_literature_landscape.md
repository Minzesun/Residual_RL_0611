# 残差强化学习 + PID 控制：结构化问题、研究空白与文献图景

## 结论
建议把当前模糊主题收敛为：

> 面向高频振动台/地震波跟踪控制，在固定 PID 控制器提供稳定基准控制量的前提下，训练一个有约束的残差强化学习策略输出额外控制量，并系统比较固定 PID、RL 调参 PID、绝对残差、相对残差在跟踪精度、安全约束、泛化和部署实时性上的差异。

这个选题的核心不是“用 RL 替代 PID”，也不建议写成“RL 自动调 PID 参数”。更有研究价值的表述是：**PID 负责稳定和可解释的基准控制，RL 只学习 PID 不能处理的模型误差、扰动和高频动态残差**。

## 已确认的本地证据
- Zotero collection：`残差强化学习`，`collectionID=64`。
- 抽取条目：57 篇。
- 有本地 PDF 附件：26 篇。
- 派生证据文件：
  - `zotero_residual_rl_collection.json`
  - `zotero_residual_rl_collection.md`
  - `residual_rl_pid_literature_matrix.md`
- 相邻 Zotero collection：
  - `强化学习(振动台)`：26 篇。
  - `PID`：至少 2 个子 collection，覆盖 RL-PID、meta-RL-PID、SAC-PID、PID autotuning。
  - `SafeBOPID`：覆盖振动台、实时混合试验、PID/控制器调参、安全贝叶斯优化。
- 当前仓库已有实验资产：
  - `crrl_vibration_control*.ipynb`
  - `crrl_test.m`
  - `Fixed_PID.m`
  - `exports/crrl_actor*.onnx`
  - `exports/crrl_actor.pt`

## 1. 从模糊主题到结构化问题

### 原始主题
残差强化学习 + PID 控制，学习额外残差控制量。

### 结构化对象
- 被控对象：高频振动台、伺服液压/机电执行系统，或地震波/正弦/混合频率参考跟踪系统。
- 基准控制器：固定 PID 或 PI/PID，输出 `u_PID`。
- 学习策略：RL actor 输出归一化残差动作 `a_res in [-1, 1]`。
- 最终控制量：
  - 绝对残差：`u = sat(u_PID + beta_a * a_res)`
  - 相对残差：`u = sat(u_PID * (1 + beta_r * a_res))`
- 目标：在不破坏 PID 稳定兜底的前提下，降低跟踪误差、控制饱和、相位滞后和波形复现误差。
- 约束：残差幅值、执行器饱和、控制增量、实时推理时延、训练探索安全。

### 推荐研究问题
RQ1：在高频振动/地震波跟踪任务中，固定 PID + 有界残差 RL 是否比固定 PID、RL 调参 PID 和纯 RL 更稳定且更准确？

RQ2：相对残差约束 `u_PID * (1 + beta_r a_res)` 是否比绝对残差约束 `u_PID + beta_a a_res` 更适合振动台控制中的安全探索和控制幅值分配？

RQ3：残差策略的观测量应包含哪些 PID/误差信息，才能让 RL 学到“补偿残差”而不是重新学习完整控制器？

RQ4：残差策略能否跨参考波形泛化，例如从正弦/扫频训练迁移到地震波，或从一种地震波迁移到另一种地震波？

RQ5：在 MATLAB/ONNX 部署和固定采样周期约束下，残差 actor 的推理延迟、输出平滑和安全裁剪是否满足实时控制要求？

## 2. 文献图景

### A. 残差 RL 基础
核心脉络从 Johannink et al. 开始：传统反馈控制解决结构清楚的部分，RL 学习传统控制难以建模的残差，最终策略是两者控制信号叠加。这个思想直接支持你的“PID + 残差控制量”表述。

代表文献：
- Johannink et al., 2019, *Residual Reinforcement Learning for Robot Control*。本地 PDF 已有；外部链接：[arXiv:1812.03201](https://arxiv.org/abs/1812.03201)，[IEEE/ACM DOI 页面](https://dl.acm.org/doi/10.1109/ICRA.2019.8794127)。

### B. 有约束 CRRL 与机电系统
Staessens et al. 将残差 RL 推向工业机电控制场景，强调 conventional controller 的鲁棒性和 RL 的自适应性之间的组合，并提出相对残差约束。该论文是你当前选题的最核心理论和方法基准。

代表文献：
- Staessens et al., 2022, *Adaptive Control of a Mechatronic System Using Constrained Residual Reinforcement Learning*。本地 PDF 已有；外部链接：[arXiv:2110.02566](https://arxiv.org/abs/2110.02566)，[UGent record](https://biblio.ugent.be/publication/8734421)。
- Staessens et al., 2023, *Optimizing Cascaded Control of Mechatronic Systems through Constrained Residual Reinforcement Learning*。外部链接：[Machines 2023](https://www.mdpi.com/2075-1702/11/3/402)。

### C. PID/级联控制 + 残差 RL
Ishihara et al. 是目前最接近“PID + 残差 RL”的直接证据：级联 PID 作为四旋翼基准控制器，RL 学习抗风残差补偿，并在仿真训练后迁移到硬件。它说明你的选题不是凭空拼接，但也留下了振动台/地震跟踪领域的空白。

代表文献：
- Ishihara et al., 2023, *Improving Wind Resistance Performance of Cascaded PID Controlled Quadcopters using Residual Reinforcement Learning*。本地 PDF 已有；外部链接：[arXiv:2308.01648](https://arxiv.org/abs/2308.01648)。

### D. 振动、悬架、地震控制中的 RL
这一支证明“振动控制 + RL”本身是活跃方向，但多数工作是 RL 调参、纯 RL、LQR/TD3 残差、半主动控制，尚未形成“固定 PID + 有界残差动作 + 振动台/地震波复现”的完整路线。

代表文献：
- Yang et al., 2025, *Prior-Guided Residual Reinforcement Learning for Active Suspension Control*。外部链接：[Machines 2025](https://www.mdpi.com/2075-1702/13/11/983)。
- Febvre et al., 2024, *Deep Reinforcement Learning for Tuning Active Vibration Control on a Smart Piezoelectric Beam*。外部链接：[SAGE DOI](https://journals.sagepub.com/doi/10.1177/1045389X241260976)，[HAL record](https://hal-lara.archives-ouvertes.fr/ENTPE/hal-04770230v1)。
- Khalatbarisoltani et al., 2019, *Online Control of an Active Seismic System via Reinforcement Learning*。外部链接：[Wiley DOI](https://onlinelibrary.wiley.com/doi/abs/10.1002/stc.2298)。
- Tang et al., 2025, *Active control technology of strip mill vibration based on deep deterministic policy gradient*。外部链接：[Iron & Steel DOI 页面](https://www.chinamet.cn/gt/en/article/doi/10.13228/j.boyuan.issn0449-749x.20250065)。

### E. RL-PID 调参：重要对照，不是主线
Zotero 的 PID collection 里有大量 RL-PID、SAC-PID、meta-RL-PID 文献。它们适合作为 baseline 或 related work，但不应成为你的主方法，因为“调 PID 参数”和“学习额外残差控制量”是不同问题。

代表文献：
- Dogru et al., 2022, *Reinforcement Learning Approach to Autonomous PID Tuning*。外部链接：[DOI/metadata 页面](https://ouci.dntb.gov.ua/en/works/96ROEoY9/)。
- Wang et al., 2024, *An Adaptive PID Controller for Path Following of Autonomous Underwater Vehicle Based on Soft Actor-Critic*。外部检索确认 DOI：`10.1016/j.oceaneng.2024.118171`。

### F. 算法效率与残差学习诊断
这些文献适合放在方法扩展或失败分析中。它们解释为什么残差 RL 可能比从零训练更高效，也解释当 `u_PID` 很强、`a_res` 很小时 critic 学习会遇到尺度问题。

代表文献：
- Dodeja et al., 2025, *Accelerating Residual Reinforcement Learning with Uncertainty Estimation*。本地 PDF 已有；外部链接：[arXiv HTML](https://arxiv.org/html/2506.17564v1)。
- Ma et al., 2026, *What Makes Value Learning Efficient in Residual Reinforcement Learning?*。本地 PDF 已有；Zotero DOI：`10.48550/arXiv.2602.10539`。
- Rana et al., 2023, *Bayesian Controller Fusion: Leveraging Control Priors in Deep Reinforcement Learning for Robotics*。本地 PDF 已有；Zotero DOI：`10.1177/02783649231167210`。

## 3. 文献分类

| 类别 | 作用 | 代表文献 | 对你课题的用途 |
|---|---|---|---|
| Core | 定义残差 RL 与 CRRL 主框架 | Johannink 2019; Staessens 2022 | 写方法定义、公式、研究动机 |
| Direct | PID/级联/机电控制直接相关 | Ishihara 2023; Staessens 2023 | 写“PID 基准 + 残差动作”的可行性 |
| Vibration Adjacent | 振动、悬架、地震、结构控制 | Yang 2025; Febvre 2024; Khalatbarisoltani 2019 | 写应用空白和实验指标 |
| Contrast | RL-PID 调参、自适应 PID | Dogru 2022; Wang 2024 | 作为 baseline，不作为主贡献 |
| Extension | 不确定性、value learning、controller fusion | Dodeja 2025; Ma 2026; Rana 2023 | 后续提升样本效率和稳定性 |
| Background | 一般 RL control、鲁棒控制、BO/PID | SafeBOPID collection | 支撑实验对照和安全调参讨论 |

完整矩阵见：`residual_rl_pid_literature_matrix.md`。

## 4. 研究空白分析

### Gap 1：振动台/地震波跟踪中的 CRRL-PID 缺口
已有 residual RL 主要集中在机器人装配、车辆、四旋翼、主动悬架和一般机电系统。你的 Zotero 中 `强化学习(振动台)` 有振动台和结构控制文献，但它们多是 RL 调参、DDPG 控制、实时混合试验控制或时滞补偿。固定 PID + 约束残差 RL 的振动台/地震波复现路线还不明显。

可写贡献：
> 提出并验证一种面向振动台波形复现的 PID-constrained residual RL 控制框架。

### Gap 2：RL-PID 调参和残差动作学习的边界不清
许多 PID+RL 文献把 RL 用于调 `Kp/Ki/Kd` 或切换控制器参数。你的方法应明确不同：PID 参数固定，RL 输出的是最终控制量上的补偿项。这能降低学习维度，也保留 PID 的稳定兜底。

可写贡献：
> 将 PID 参数优化问题转化为有界残差控制问题，减少策略对闭环稳定性的破坏。

### Gap 3：相对残差约束在振动/PID 场景缺少系统比较
Staessens 2022 强调 relative residual constraint，但振动控制和 PID residual work 中常见的是直接叠加或未明确安全管束。你的实验可以比较：
- absolute residual
- relative residual
- residual disabled
- RL-PID tuning

可写贡献：
> 给出 absolute vs relative residual 在波形跟踪、饱和率、探索安全和泛化上的经验比较。

### Gap 4：评价指标仍偏时间域，频域和波形复现指标不足
振动台控制不能只看 MAE/RMSE。需要加入：
- peak error
- phase lag
- frequency response / PSD error
- target-response coherence
- actuator saturation ratio
- control smoothness
- earthquake record hold-out generalization

可写贡献：
> 建立面向振动台 residual RL 的综合评价指标，而不是只报告 reward 或平均误差。

### Gap 5：部署约束没有被作为研究对象
当前仓库已经有 ONNX/MATLAB 导出路径。多数论文只展示算法结果，不把采样周期、推理时延、MATLAB 兼容性、输出裁剪和异常回退作为核心验证点。

可写贡献：
> 证明 residual actor 可以作为 PID 外挂模块部署，并定义 actor 输出异常时的回退策略。

### Gap 6：残差尺度小导致学习困难
当 PID 已经较强，残差动作的有效贡献可能很小。Ma 2026 和 Dodeja 2025 指向 residual RL 的 value learning、uncertainty 和 exploration 问题。你可以先不解决理论问题，但要在实验中记录残差幅值、critic 学习稳定性和 beta 参数敏感性。

可写贡献：
> 通过 beta sweep 和残差幅值统计解释残差策略何时有效、何时退化为零输出。

## 5. 初步研究计划

### Phase 0：问题冻结
目标：把研究对象固定为“振动台 PID 基准 + 有界残差动作”。

产出：
- 系统方程或离散状态空间模型。
- PID 控制律。
- residual wrapper 公式。
- 安全约束表。

不建议现在做：
- 多输入多输出扩展。
- 复杂安全证明。
- 端到端纯 RL 对照。

### Phase 1：基准复现
目标：建立可重复的固定 PID baseline。

实验：
- 正弦 14 Hz、34/56 Hz、0-50 Hz。
- 地震波：Kobe、Tohoku、Chichi、Tabas 等已有文件。
- 指标：MAE、RMSE、peak error、phase lag、PSD error、saturation ratio。

产出：
- `PID.csv` 或统一 metrics 表。
- baseline plots。

### Phase 2：残差接口实验
目标：只验证接口，不追求最优 RL。

实验组：
- PID only。
- PID + zero residual。
- PID + random bounded residual。
- PID + trained absolute residual。
- PID + trained relative residual。

验收：
- `beta=0` 时完全复现 PID。
- residual action 不导致执行器持续饱和。
- actor 输出异常时能回退到 PID only。

### Phase 3：SAC 残差训练
目标：训练一维 residual actor。

建议观测：
- 当前误差 `e`
- 误差差分 `de`
- 短窗 MAE / RMS
- `u_PID`
- 上一时刻 `u`
- 参考 `r`
- 输出 `y`
- PID 积分状态
- 饱和裕度
- 频段或波形类型编码，若做多任务训练

建议 reward：
```text
reward =
  - tracking_error
  - lambda_du * control_smoothness
  - lambda_sat * saturation_penalty
  - lambda_res * residual_magnitude
  - crash_penalty
```

### Phase 4：gap 驱动的主实验矩阵
P0 必做：
- fixed PID vs SAC residual。
- absolute vs relative residual。
- seen reference vs held-out reference。
- beta sweep：例如 `beta_r = 0.05, 0.1, 0.2, 0.3`。

P1 建议：
- RL-PID tuning baseline。
- GA/BO tuned PID baseline。
- noise/disturbance robustness test。

P2 之后再做：
- uncertainty-guided residual exploration。
- critic normalization / warmup。
- Lyapunov-style stability analysis。

### Phase 5：部署验证
目标：证明方法能进入你当前 MATLAB/ONNX workflow。

检查：
- ONNX 推理输入维度和输出维度。
- 单步推理时间。
- actor 输出裁剪。
- residual disabled 回退。
- MATLAB `crrl_test.m` 与 Python 测试指标一致。

## 6. 可执行阅读顺序

第一轮，必须精读：
1. Johannink et al. 2019：理解 residual RL 基本分解。
2. Staessens et al. 2022：理解 constrained / relative residual。
3. Ishihara et al. 2023：理解 cascaded PID + residual RL。
4. Staessens et al. 2023：理解 cascaded CRRL。

第二轮，建立应用背景：
5. Yang et al. 2025：active suspension residual RL。
6. Febvre et al. 2024：active vibration control + DRL tuning。
7. Khalatbarisoltani et al. 2019：active seismic system + RL。

第三轮，设置对照组：
8. Dogru et al. 2022：autonomous PID tuning。
9. Wang et al. 2024：SAC adaptive PID。
10. SafeBOPID collection 中的 constrained BO / safe PID tuning。

第四轮，解释训练失败或扩展：
11. Dodeja et al. 2025：uncertainty residual RL。
12. Ma et al. 2026：residual RL value learning。
13. Rana et al. 2023：Bayesian controller fusion。

## 7. 推荐论文标题方向

偏工程实现：
> Constrained Residual Reinforcement Learning for PID-Based High-Frequency Vibration Tracking Control

偏振动台/地震波：
> PID-Guided Residual Reinforcement Learning for Seismic Waveform Tracking on Electro-Hydraulic Shaking Tables

偏方法比较：
> Absolute versus Relative Residual Reinforcement Learning for Safe PID-Augmented Vibration Control

中文表述：
> 基于有约束残差强化学习的 PID 振动台波形跟踪控制方法

## 8. 最小可行研究假设

H1：在同一固定 PID baseline 上，学习有界 residual action 能在不显著增加饱和率的情况下降低 held-out 波形的 RMSE。

H2：相对残差约束比绝对残差约束更不容易在 PID 输出较大时产生危险探索，但可能在 PID 输出接近 0 时补偿能力不足。

H3：残差 RL 的主要收益来自对未建模动态、执行器滞后和频率相关误差的补偿，而不是替代 PID 的主要稳定作用。

H4：当 PID baseline 已足够强时，residual actor 可能学习近零输出；此时需要通过 beta sweep、频段指标和残差幅值统计判断是否真的有贡献。

## 9. 本轮不建议扩大的范围
- 不建议把研究扩展成通用 robot residual RL。
- 不建议把主方法写成 adaptive PID tuning。
- 不建议第一版加入复杂 Lyapunov 证明。
- 不建议同时做多智能体、多轴、多输入多输出。
- 不建议把所有 Zotero 57 篇逐篇精读；先精读 8-12 篇核心/直接相关文献。

## 10. 下一步最小动作
1. 从 `crrl_vibration_control_8_1.ipynb` 和 `Fixed_PID.m` 固定一组 PID baseline 指标。
2. 用 `beta=0` 验证 residual wrapper 可完全退化为 PID。
3. 做 2x2 小实验：
   - absolute residual vs relative residual
   - sine/sweep reference vs held-out earthquake reference
4. 把结果表按 `PID / absolute / relative / RL-PID baseline` 四列整理。
