# PROJECT MASTER AUDIT

独立科学与软件尽调；审查日期：2026-09-05—06。配套执行文档：[CODE_ROADMAP.md](CODE_ROADMAP.md)。

## 审查基准、范围与证据边界

**唯一代码基准是远端 master `0c8a6ca34c6c8b4f505dc153dd0ca8cc7949ce56`**，多次通过 `git ls-remote https://github.com/Endymion2288/alignment_ML.git refs/heads/master` 核实，最后一次在 2026-09-06 交付检查期间。永久入口：[被审查的提交](https://github.com/Endymion2288/alignment_ML/tree/0c8a6ca34c6c8b4f505dc153dd0ca8cc7949ce56)。GitHub 网页缓存曾展示较旧 README，因此不以网页缓存重建现状。

开始时本地 HEAD 是 `9996765`，续审时是 `ba9ef9a`，并有未提交文件；这些更新均不纳入结论。代码、配置、文档和测试从指定提交单独解包到 `/tmp/alignment-master-audit-SavyXR`。本文中的路径与函数均指该提交，不能据本地后来同名文件推断本文已审查其修改。两份交付文档之外，本轮没有修改生产代码、条件或既有产物。

审查包括 README 中英文、docs 与 workbook 的阶段演进、所有顶层代码目录的文件清单，以及与主张相关的核心实现、配置、测试、非 test manifest/summary/closure/contract。采用“全项目目录覆盖 + 科学关键路径逐函数追踪”，**不声称逐行阅读全部历史脚本**。没有重跑 Athena、训练、物理 production 或完整测试套件；没有打开 sealed-test event content，也没有打开历史 sealed-test 输出包。历史 sealed-test 数值仅引用已公开的版本控制文档，证据等级相应降低。

`outputs/` 在这个提交里只有 `.gitkeep`。下述本地非 test 产物是辅助证据，不是 Git 自动认证的结果。核对了关键 JSON、已有非 test NPZ、配置哈希和真实数据冻结包的文件哈希，但尚不能证明每个产物由一个完全干净、可恢复的 Calypso 二进制构建产生。Git HEAD 与配置哈希不能代替外部 C++ 源码、dirty patch、库与 conditions 的指纹。

### 实际完成的短时验证

1. 在隔离代码上运行 11 个指定测试文件，共 **73 passed，22.54 s**；交付前从同一隔离目录重跑为 **73 passed，13.94 s**：`test_physical_jacobian`、`test_tracker_only_identifiable_subspace`、`test_rigid_station_only_identifiability`、`test_cluster_local_observable_cross_run`、`test_route_assignment`、`test_structured_assignment`、`test_anchor_selected_update`、`test_gauge_equivalence`、`test_expanded_control_boundaries`、`test_ntuple_conversion`、`test_field_route_fitter`。均使用测试 fixture，不是实际 sealed 数据。测试通过不验证真实探测器模型。
2. 对保存的 workbook-73 18 源奇异谱复算固定 `0.01` 判据，得到 `5,5,2,3,5,4,5,4,5,5,5,5,5,5,5,5,5,5`，与报告一致。
3. workbook 68/71/73 的报告配置 SHA256 与审查快照对应 YAML 完全一致；Operating Protocol V1 reproducibility manifest 的 **13 项文件哈希全部匹配**，包含 frozen V2 checkpoint。
4. 在临时目录运行小矩阵反例，验证 bootstrap 重复计数、相对 rank、满维 projector、欠定 SVD null basis、非正定 covariance 的问题，具体见下文。
5. 只读重算已有 iteration-01 validation 去重 NPZ 的 WLS，验证目标 residual 对恢复结果的影响；没有产生新物理闭环。

运行环境为 CERN LCG_110_cuda，Python 3.13.11 / NumPy 2.4.4 / SciPy 1.17.1 / Torch 2.11.0 / uproot 5.7.1。pytest 8.4.2 仅安装到 `/tmp`。最初缺少 pytest，随后一次启动覆盖了 LCG 的 PYTHONPATH 导致依赖收集失败；保留 LCG 路径后上述短测通过。这是审计环境准备问题，不计为仓库测试失败，也没有安装到用户环境。

## Executive Summary

**物理目标值得保留；“继续扩大 Transformer 或反复寻找一个通过相对 rank 门的 tracker-only 5DoF 子空间”不值得作为未来半年的主线。** 推荐只保留一个方法主线：**基于真实测量和磁场传播、剖面化轨迹 nuisance、可接入经认证 survey 的受约束 alignment likelihood；冻结 association 作控制，先证明 estimator，再处理 association 不确定性。** 唯一备选是维护现有 residual/DQ monitoring，并完成一份边界明确的负结果研究。

当前最重要的缺口不是网络容量，而是：现有 alignment 控制主要恢复**同事件、不同 geometry refit 的响应差**；用于实际 data 的绝对测量模型、协方差契约和 nuisance 消元尚未验证。最新停止部署的决定是合理的，但部分“物理不可辨识”归因比证据走得更远。

最影响资源决策的发现：

- **响应 closure 不等于数据可用 estimator closure。** 同一 467 条去重 validation 边上，paired-target WLS 增量为 `(0.08974 mm, −0.07480 mm, −0.75854 mrad)`；零目标 WLS 增量为 `(−0.10676 mm, −0.16389 mm, −4.18015 mrad)`。加到原 anchor 后，后一方案剩余约 `(−0.24759 mm, −0.04949 mm, −3.43753 mrad)`，超出原 `dx=0.1 mm / Ry=1 mrad` 门。该差异无需改变 association 或模型即可出现。
- **bootstrap 有确定实现错误。** `alignment/tracker_only_identifiable_subspace.py::bootstrap_subspaces` 将抽样 index 转为布尔 mask，丢掉有放回 multiplicity。影响 workbook 68、69 的 event 分支、73；不是所有 bootstrap 都有此错误。
- **官方 rank 是工程性截断 rank，不是结构性信息秩。** workbook 73 的低 rank 源往往首先表现为最大奇异值异常大；并非其余奇异值都趋零。固定门应保留为历史规则，但不得据此宣称完整 track likelihood 也不可能测量这些参数。
- **cluster-local 路线尚不是 field-aware global alignment。** 它用排除目标站后的三维直线预测、只用 cluster 的 `u` 方差，并以软件扰动 surface 做 FD；`ry` 的符号和 pivot 还与 station payload chart 不同。它没有穷尽 cluster-level alignment 可用的信息。
- **真实数据现状仅支持 monitoring。** self-nulling Station Mode 和 reduced mode 的跨 run 失败支持拒绝写 geometry；不能唯一归因为 detector 拓扑的物理无解。原 covariance、轨迹模型、内部形变与事件选择均未被充分排除。

最先应完成的不是新 campaign，而是本报告路线中的 **统计/坐标/协方差契约修复与 target-reference 消融复核**。这些工作的负结果必须能够终止后续昂贵计算。

## Reconstructed Current State

### 真正的物理问题

目标是把 IFT 与 spectrometer 的局部 tracklets 正确关联为同一条粒子的轨迹，并从真实 tracker 测量约束 detector 的相对位置、姿态与有限内部形变，最终改善 track extrapolation、匹配和物理重建。alignment 不是回归一组 MC displacement 标签，也不是单独把 residual RMS 压低。

必须区分三个量：测量坐标所在的 sensor geometry、重建使用的 alignment conditions，以及粒子轨迹本身。改变重建 geometry 后，cluster local measurement 原则上保持；global cluster position、segment fit、传播 surface 和 prediction 都应一致更新。现有固定 cluster refit 对研究重建 geometry 响应有价值；它没有模拟位移改变 digitization、clustering、接受度或真实材料分布时的全部效应。

### 输入、输出与监督

| 层 | 当前输入 | 当前输出/用途 | truth 的位置 |
| --- | --- | --- | --- |
| cluster/refit | 外部 Calypso、xAOD `SCT_ClusterContainer`、conditions | `SegmentFitRefit` / `SegmentsRefit` | MC matching/audit；生产细节不全在本仓库 |
| canonical tracklets | flat ROOT `tracklets`；明确 station ID、`x,y,z,tx,ty`、4×4 `Cov[x,y,tx,ty]`、fit quality、hit pattern | `EventTracklets` | MC 标签用于监督和计数；data 不应要求 truth |
| Acts pair records | mode-0 source seed、target reference surface、geometry | prediction、4×4 covariance、success、target z、模式字段 | mode 1 truth-q/p 控制不可作为部署输入 |
| candidate graph | 上述真实 prediction 与现有 tracklets | 候选边、四维 residual、pull、chi2 | `build_field_candidates` 不读取 truth 作选择 |
| association | synthetic overlay 后的 physical states/candidates；真实数据 identity corpus | edge/route scores、unit-capacity routes、unmatched | truth/role 仅构造 label 与审计 |
| alignment 控制 | anchor/probe/reference 的同源 residual，truth-selected 或 anchor-selected physical edges | FD Jacobian、WLS 增量、condition/rank、paired closure | truth-selected 固定身份；route-selected 不用 truth 选边，但仍可读取已知 reference geometry residual |
| data operating mode | 当前 official geometry、frozen V2、selected routes | residual/DQ 指标与告警 | 无 alignment truth；禁止把告警解释成机械 correction |

### Canonical pipeline：需要同时说清三条状态

```text
物理 MC 控制：
xAOD clusters → /Tracker/Align 独立 payload → SegmentFitRefit → SegmentsRefit
→ NtupleDumper → mode-0 Acts → convert/audit → source-disjoint physical manifest
→ 同 payload 内 overlay（仅改变组合/缺失/背景）→ ungated physical graph
→ frozen V2 + calibration + contiguous route packing
→ anchor-selected + physical-edge-deduplicated FD response WLS → paired closure

最新研究判定链：
hierarchical FD bank → 68 子空间 → 69 stable core → 70 exact-join repair
→ 71 cluster-local transfer → 72 failure provenance → 73 rigid 5DoF
→ 74 residual-blind coverage inventory（未打开新 FD）

真实数据：
当前 official geometry → export/identity → frozen V2 → residual/DQ monitoring
                                                    geometry_write_allowed=false
```

冻结 V2 checkpoint SHA256 为 `0c85a001677678163512c59bd5ceb6d08bdefaab79c0f4628659a4a8ad766a27`。真实数据操作点记录 threshold `0.001`、unmatched penalty `−1.0`，观察类型 `anchor_selected_field_edge`，统计语义 `physical_edge_deduplicated`。这些是既有 controls，不因本审查重新选择。

“V2”有历史 99-event 版本与 expanded BCE control 两种重要语境：前者 route context 权重选到 0；后者及 Ry 控制有非零 context 贡献。不能把早期 V2 负结果套到全部后续 V2，也不能把后者当作独立最终测试胜出。V1 有保留的 sealed negative result；V3 structured 路线冻结为 validation negative result。

### 历史演进与文档冲突

审查了近期提交内容/统计及相关实现：`77ec84e` 修 legacy 默认 test 访问与 CUDA fallback；`0dc5780` 关闭 mode-3 分支；`c5436ce` 的 station 5DoF + 5 mm dz prior；`a75800f` hierarchy/calibration 模式；`2ca7bf9` 真实数据 monitoring 冻结；`e5d3345` cluster-local/topology；`a1fd01b` survey/IOV；`0c8a6ca` 68–74。

旧 `docs/project_audit_and_next_plan.md` 仍写 iteration-0 aggregation 待做、loop 未闭合、三类 legacy 默认可读 test。这些在该审查基准已不是现状：后续有 paired closure，legacy 默认拒绝/排除 test；旧审核不能直接作为任务队列。`docs/global_alignment_multidof_loop.md` 同时保留“5 track-constrained DoF”“dz survey-constrained”与较后阶段失败史，必须增加明确的 superseded 状态。5 mm prior 是历史设定，不是已认证的 survey measurement。

### Legacy 与可复用组件

- `geometry/propagation.py`、`models/track_fitter.py` 的直线模型：保留作无磁场 toy / diagnostic；不能因名称是 global fitter 就用于跨磁铁的最终 alignment。
- `alignment/closure.py`、`alignment/payload.py` 中 coordinate-shift 控制，以及 `run_payload_coordinate_capture_scan.py` 等：历史单元/接口控制，不是 physical refit 的替代品。
- 早期 chi2/greedy/Hungarian/Sinkhorn/multistation-flow 扫描：历史比较；一般 solver 与简单 physics controls 值得复用，不需删除。
- MLP、V1、V2：冻结比较对象；V2 是当前 association backbone，不是被证实最优的研究终点。
- V3、mode-3 suppression/retraining、联合或顺序 `C_dx+C_rx`、hierarchical rescue、real-data self-nulling：保留负结果，不自动重启。
- 数十个 `alignment/*feasibility*.py` 与 report 脚本多是 campaign-specific policy/report 逻辑，不是一个统一可部署 estimator。

## Scientific Audit

### 参数、gauge 与物理可辨识性

Station payload 为 `[dx,dy,dz,rx,ry,rz]`，mm/rad；报告与 FD 配置旋转用 mrad。物理 convention 为 global left multiplication、`T Rz Ry Rx`，station 旋转中心为 FASER 原点；plane 层级的 conjugation/pivot 不同。IFT 约 `z=−1860 mm`，global `Ry=1 mrad` 因此包含约 `−1.86 mm` 的横移响应。把这个大 response 单独称为“强旋转信息”会混淆可与 dx 补偿的共同移动和真正改变方向的部分。

固定 S1–S3、只动 IFT 不仅是选一个全局坐标 gauge，还把所有 spectrometer 内部相对几何当作完全准确的模型边界。真实 data 必须传播 reference geometry 的不确定性，或在联合 fit 中作为 nuisance/约束；不能把任意真实未知模式固定为零后称其被测量。

真正 gauge 是 likelihood 的对称性；near-null 是有限精度下弱响应；cut-defined null 是保守估计器舍弃的方向。三者不可互换。全局 detector 位姿的 gauge 也依赖外部磁场/材料/beam 坐标如何固定；不能无条件宣布同一个“六个 gauge”适用一切配置。当前 IFT 相对 dz 主要通过轨迹斜率进入，近轴样本下很弱，但在 S1–S3 固定后它通常不是严格的全 detector 平移 gauge。

`C_dx=(dx_L0−dx_L2)/2` 的零共同模 contrast 定义合理；`reference_layer` 在 station 同时补偿时才可能描述同一物理族，station 冻结则改变约束。项目后来已认识并测试这一点。对于有限旋转，需完整 SE(3) composition；不能按六个 Euler 数直接相加代替相同 geometry。绝对 payload 替换与局部 `Δθ` composition 也必须是不同 API。

已有 hierarchy leakage `B=(J_sᵀWJ_s)⁻¹J_sᵀWj_c` 的近共线证据有价值；这里用 B 避免与 weighted Jacobian A 混名。文档给出 `B_dx≈−59.21 mm/mm`、`B_ry≈−31.91 mrad/mm`，及 `R²≈0.99966`，说明在**该 residual、权重、chart** 中忽略微小内部形变会造成很大的 station bias。`1.5–1.7 μm` 隔离预算来自这个模型和既定 dx 门，不是通用机械制造要求，也不是不可被更完整 likelihood 改变的物理常数。

### 问题更可能在哪一层

现有证据支持“当前 estimator/sample 下存在严重弱模式、偏差与 coverage dependence”；没有充分支持“完整 tracker measurement 在统计量足够时也不可能约束任何多 DoF”。特别是：

- FD paired response 消掉了同事件参考 residual 的静态偏差，也消掉一部分公共噪声。
- pair covariance 是 deterministic WLS weight，并不是同 cluster refit 差的采样 covariance；相邻/多目标边共享 tracklet，去除 overlay duplicate 后仍相关。
- cluster-local 软件 FD 固定了轨迹直线，没有 field-aware nuisance response。
- 真实 data 的 self-nulling 可把 field、momentum、材料、重建与 selection bias 吸收到 geometry。

因此最大 scientific risk 是**对错误/不完整 observable 的可辨识性做了非常精细的研究，却把它当成完整 detector 的信息极限**。下一步应检查 measurement likelihood，而不是追加参数族。

## Association Audit

### Candidate recall 的分母决定结论

`evaluation/route_metrics.py` 的 complete truth-chain denominator 是已重建且各站 truth 唯一的四站链。它不包含未重建、未导出、truth ambiguous 或被 physical propagation 丢掉的所有可探测粒子。因此“candidate complete recall=1”是条件 recall，不是 detector tracking efficiency=1。

在 frozen ungated overlay graph 上，raw complete recall 近 1，故该条件问题中 graph retention 不是主要瓶颈。原始小样本 chi2=25 gate 扫描、放宽到 500 的 truth coverage、0→1 exporter/传播损失、synthetic overlay recall 属于不同分母/selection；不能互相否定。部分 physical source 每站至多一条可用 tracklet 时 fake candidate=0，不代表真实多粒子 occupancy 下放宽 gate 无代价。

必须统一报告从 generated/fiducial truth、重建 tracklet、成功传播、raw graph、阈值 graph、selected routes 到 alignment-used edges 的计数瀑布，并区分 event、physical track、edge、overlay replica。禁止只展示其中一个漂亮百分比。

### Assignment 是否正确

`baselines/route_assignment.py` 对给定 hypothesis set 的 endpoint unit-capacity set packing 是合理的；DP/MILP 与暴力最优解的测试有实质价值。dustbin/unmatched **已经存在**，不应在 roadmap 中重复“新增一个 dustbin”。

限制是 hypothesis/utility：

- canonical route 只走相邻连续站；非相邻边可用于 message context，但不能自动形成跨缺失内部站的同一条 route。partial contiguous route 与完整包含 skip edge 的 track 是两种任务。
- 完整 route 的 learned non-additive utility 使问题成为 hyperedge packing；并不一般等价于普通 edge-additive min-cost-flow。
- tracklet 容量为一不保证 raw cluster 容量为一；若同站 ghost/重复 segment 共享 hits，后续 fit 需显式报告/约束。
- calibrated BCE score 不是完整 route 的 likelihood ratio；相邻边相关，直接加 log-odds 加上调出的 dustbin utility，不自动得到联合概率模型。

### Classification 与 alignment 的目标错位

alignment 一阶 contamination bias 约为 `N⁻¹ Σ J_eᵀ W_e b_e`。它取决于错误边方向、杠杆、协方差和弱模投影，不只取决于 fake fraction。即使 purity 提升，留下的少数高影响错误也可能使 dx/Ry 更差。项目已有单个错配边及 overlay multiplicity 放大 bias 的控制；去重后的 validation paired closure 仍有约 `−0.0511 mm` dx error。

Route-level decision 和全局容量必要，但 route Transformer 并不因此必要。ML 最合理的角色是：在 physics candidates 内提供关于 ambiguity/outlier 的辅助信息，或帮助估计 data-supported background/assignment propensity；不得负责补出物理不存在的信息，也不得把绝对 geometry 的答案编码成 source shortcut。

推荐以后用包含 unmatched 的事件级 assignment likelihood，并传播 assignment uncertainty。先用冻结的 V2 与简单 physics route scorer 对照；只有 field-aware estimator 在 truth-association 下通过、unknown association 的偏差确实占主导，才讨论是否需训练轻量 residual score。EM-like 交替是合适的求解组织方式，不是自动一致/全局收敛保证。端到端 differentiable alignment 暂无必要。

## Alignment / Identifiability Audit

### 1. Residual、whitening 与 covariance

物理 pair residual 是 target `[x,y,tx,ty]` 减 mode-0 prediction；`combined_covariance=C_pred+C_target`。`identifiable_subspace.py::covariance_left_sqrt_inverse` 用 `C=LLᵀ, M=L⁻¹`，故 `MᵀM=C⁻¹`；`flatten_physical_jacobian` 构造 `A=MJS` 的方向正确，**没有 Cholesky 转置错误**。

但块对角 W 忽略共享源/目标 segment 的跨边 covariance；MC paired response 还共享完全相同 clusters。`physical_jacobian.py` 注释承认它是 WLS weight，然而输出 `covariance_native`、`response_ndof=4n−rank` 不能据此被当作已经校准的置信区间/chi-square distribution。取 inverse normal 而无 reference/FD/association/model uncertainty，会得到形式精度而非完整误差。

已有 mode-0 pull core 极端各向异性，y/ty robust widths 约 `0.002–0.003` 的记录是重大契约警报。对角 rescale 或 Huber 导致 closure 更差，并不证明原 covariance 正确；也可能说明原 estimator 正利用失真的权重抵消另一偏差。必须从 local/curvilinear/bound/global 状态定义、角度到 slope、q/p 的 MeV/GeV 单位、Jacobian 与 process noise 开始审计。正定性本身不能证明 covariance 的坐标语义正确。

### 2. Fixed relative rank 不是结构性 rank

`physical_jacobian.py` 对 scaled normal matrix 使用 `rcond=1e−10`；最新 `identifiable_subspace.py` 对 A 使用 `0.01`。前者约对应 A 的相对 `1e−5` 数值截断，后者主动舍弃更弱模式。`κ(AᵀA)=κ(A)²`；两类 condition number 不能混表比较。

workbook 68 的 pooled A 奇异值为 `[3729.82,1747.30,292.11,170.81,164.27,28.16,2.52]`。所有七个均非零；“5 identifiable + 2 null”是规定 estimator 的 retained/weak split，不是数学证明两维对观测完全无响应。

workbook 73 已存谱的例子：

| source | 固定 rank | σ1 | σ3 | σ5 |
| --- | ---: | ---: | ---: | ---: |
| 100043_00200_00299 | 5 | 1084.62 | 61.13 | 32.24 |
| 100043_00400_00499 | 2 | 8748.26 | 65.00 | 16.33 |
| 100043_00500_00599 | 3 | 11021.57 | 335.03 | 13.61 |

第二行 σ3 甚至高于第一行，但由于 σ1 大八倍被截断。最弱绝对 sensitivity 也有所降低；这不能简单归为“只有分母效应”，但足够否定“rank 2 表明只有两个物理方向有信息”的解释。

独立反例 `diag(100,2,2)` 与 `diag(1000,2,2)` 在固定 0.01 门下 rank 从 3 变 1，尽管只加强了第一模、其余信息完全不变。重复独立事件使 A 的 singular values 按 `√N` 增长而比例不变，因此该规则也不能回答增加统计量是否能达到一个绝对精度目标。

**处置：不改历史 threshold、不替失败发 pass；保留原门的失败，同时撤销由该门直接作出的结构性无信息解释。** 新 estimator 以预注册的 bias、实际 covariance/coverage、geometry 预测不确定度与线性有效性决定部署。不要换个 chart 或 S 以获得漂亮 rank。

### 3. Bootstrap、LOSO 与“子空间一致”

确定缺陷：`bootstrap_subspaces` 第 391 行起，抽取 `n_events` 个 index 后把 `selected[group]=True`； multiplicity 被丢弃。小检查 1000 draws 只留下 629 unique events。它是随机 thinning/subset stability，不是标准 n-out-of-n event bootstrap。修复须保持同一事件的全部边、保留抽取次数；pool 后 event key 还必须包含 original source UID，不能只凭原 run/event（MC chunk 可重号）。现有报告的 bootstrap pass/fail 都需单独标记待复核。

`cross_source_stable_core.py` 的 source bootstrap 保留重复 source；`cluster_local_observable_cross_run.py::official_event_bootstrap` 用 row list 保留重复事件，不能把缺陷泛化给它们。修正前者也不会消掉 workbook-69 独立 8/11 的失败，因为这是另一条证据。

LOSO 比较来源子空间是有用 stress test，却不是参数预测 transfer 的替代。source file 拆分不等于独立 physics process 或 detector condition；七个定义源与余下十一源来自已反复研究的 corpus，所谓 independent 仅在该次 basis construction 的文件集合意义上成立，不是全项目未见 final test。

完整 3D 参数空间里的 rank-3 projector 必为 I；两 run 3/3 时全空间夹角≈0是恒等事实。5D/5 参数同理。已存 workbook-71 的 `8.5e−7 degree`、persistence≈1 不验证参数误差、弱特征方向或 estimator 的跨 run transfer。该报告另有 weak-plane angle，较有意义；仍应比较 full information tensor 和 held-out prediction。

### 4. FD 与 cluster-local parameter chart

物理 FD 使用独立 payload/refit，central difference 步长 translations 0.5 mm、rotations 10 mrad 等；这是正确的响应实验结构。不过不同 probe 取共同可重建集合会引入 survivor selection。现有 validity 检查主要是完整、有限、非零列、population join；并未证明这些步长的 derivative 已收敛、branch switching 已排除或整体 covariance 正确。需要预注册 `h/2,h,2h` 加无量纲误差，不能根据结果反调 rank tolerance。

cluster-local 的实现细节使“换 observable 后仍失败，故物理无解”不成立：

- `true_cluster_local_residual.py::build_cluster_residuals` 排除目标站后，使用其它站 cluster global xyz 的未加权三维直线（SVD）；失败回退到 tracklet straight-line fit。station-unbiased 表示没使用目标站 hit，不表示跨磁场 prediction 无偏。
- strip local-u 是测量；把条带几何中心 global xyz 当三维测量点，会给未测或弱测的沿条带方向隐含权重。
- `local_u_var_mm2` 不含预测轨迹 covariance；非法方差还 fallback `1e−4 mm²`。共用同一排除站轨迹的多条 residual 相关。
- `rotation_about_y_matching_se3` 采用与 station payload 相反的 Ry 符号；`perturb_surface` 绕 `STATION_Z_MM`，而 station payload 绕 FASER 原点。这可以定义另一种合法 chart，但不能不变换 Jacobian、参数尺度/先验就按同一 `station_ry` 比较。带自由 dx 时适当 chart map 可保留线性列空间；固定相同对角 S 的截断结果不必保持。

另外 `nov22_metrology_provenance_station_ry.py::extract_alpha_beta_gamma` 使用 `asin(R[0,2])`，不能一般反解同文件构造的 `Rz Ry Rx` 混合旋转。在隔离 master 上，输入 `(rx,ry,rz)=(0.01,0.02,0.03) rad`，提取再构建后的矩阵 Frobenius 差约 `9.80e−4`，而非浮点舍入。单轴 Ry smoke 看不出问题。它目前用于诊断/外部映射，不证明历史 physical writer 注入错误；应补混合旋转 round-trip 合约。

### 5. 对 identifiability 路线的八个明确判断

1. **数学构造部分可靠，解释与 bootstrap 不完全可靠。** Whitening、scaling、SVD 本身正确；relative truncation、误差语义和上游 observable 限制决定结论边界。
2. **rank instability 不自动等于模式不可部署。** 它使原准入门失败；是否可部署要看目标 IOV 样本下绝对信息、偏差和 coverage。每个小 slope tertile 都需全 rank 是比 pooled calibration 更强的要求。
3. **observable 设计很可能贡献了失败。** 直线 cluster predictor 与 correlated pair WLS 均不是完整 likelihood。
4. **global fit 可能恢复被压缩/遗漏的信息，但不能创造真正缺失的信息。** 正确剖面化 track nuisance 往往也会减少固定-track 模型的虚假信息，故必须允许更负的结果。
5. **足以停止当前 tracker-only 5DoF 救援/部署 campaign；不足以宣布所有 tracker-only multi-DoF 原理上无解。** 不追加同类扫描；仅给一次受限的 measurement-model 证伪机会。
6. **下一步是 estimator/coordinate/covariance 的最小 falsification。** 不以新的 source family 或删 DoF 作默认出口。
7. **survey 最终应是经认证且相关性完整的 measurement likelihood/prior。** 真 gauge 可用 exact constraint；有限误差 survey 不应硬固定。当前 Nov-22 资料继续 cross-check only，初始化也不能隐藏机械/IOV 不确定性。
8. **应构造完整 global alignment normal equation。** 首版只需 IFT 与少量明确 nuisance，采用 Schur 消元或等价 Kalman/global covariance；无需一上来全 module Millepede 或神经网络。

## ML Audit

### Representation

MLP 具有 residual/state/covariance/hitpattern 多种 feature sets；V1/V2 使用 `NODE_FEATURE_NAMES` 的绝对 `x,y,z,tx,ty`、log sigma、fit quality、hit summary，加 station/pair embedding。绝对坐标与 station 信息是 detector-specific 合理输入，但提供 geometry/source shortcut 的机会；这不是仅凭输入清单就能断言的数据泄漏。未在追踪过的 feature 路径发现直接 truth ID、synthetic role、source UID 被当作数值特征；它们在 label/audit 分支中存在。

以下仍未被充分证伪：固定 z 与 covariance 指纹识别 source/condition；fake 注入方式可被 fit-quality/hitpattern 区分；重复物理事件 overlay 造成有效样本量远低于图数量；模型依赖 simulation support 而不学习几何一致性。

优先做冻结模型的 permutation、合法 coordinate-chart transformation、feature/source predictability 和 condition-held-out controls。合法 transformation 必须同步变换 field、surfaces、states 与 covariance；不能在固定探测器里乱转坐标后要求 physics score 不变。Relative residual、local frame 和规范化 covariance 更值得先验证；SE(3)-equivariant 网络既不消除 weak mode，也不能无条件把 detector/material/field 当旋转对称系统。GNN 不是自动优于已有稀疏图 attention 的新科学问题。

### Objective 与模型比较

| 路线 | 已实现目标 | 审查判断 |
| --- | --- | --- |
| pair MLP | positive-weight BCE、curriculum/negative weighting、validation calibration | 合理控制；AP/F1 不保证 alignment bias 最小 |
| V1 | sparse geometric attention + edge BCE/focal/local head | 局部 decoder 主导、context gain 缺乏稳定证据；不支持扩大网络 |
| 历史 V2 | edge + complete-route BCE、endpoint competition/fake penalty | direct route utility 与 partial utility 失配；w=0 失败记录有效 |
| expanded V2 BCE | 同 backbone、更大 source-disjoint corpus、route query | 可用 frozen control；非零 context 与最终 alignment 因果增益未被独立证明 |
| V3 | exact loss-augmented complete-route hinge；edge/route BCE 主项关闭 | discrete argmax 的次梯度形式合理；当前负结果不能排除所有 structured/probabilistic alignment |

V3 特别存在目标对齐的解释限制：training oracle 在 complete routes 中优化，inference 还允许 partial routes/dustbins，并对 edge score 单独校准与筛阈值。优化 `Σedge logits + route residual` 不唯一确定边分解；complete-route hinge 下学得的 edge component，未必适合校准成 partial-route utility。这不是 solver 最优性错误，而是训练/部署可行集与 score 语义不完全一致。无需为验证这一点再开 V4。

Class weighting/hard-negative enrichment 使 raw sigmoid 不代表 deployment posterior。Platt/temperature 有助于指定 validation mixture 的 calibration；ECE 下降不证明条件变化后仍 calibrated，也不证明 route likelihood 正确。若以后引入 learned likelihood correction，必须防止 residual 信息被 physics likelihood 和 ML 两次计入，并把 occupancy/condition domain 写入 calibration contract。

## Experimental Methodology Audit

### 有效的做法

原 xAOD 文件 split、原事件 UID 检查、overlay origin namespace、物理 state 不变的 pooling、checkpoint/calibration hash、固定负结果、先 validation 后 frozen evaluation、完整 failure source 保留，以及 source-local HTCondor + 单进程 aggregation，都有实质价值。不能因为发现根本问题就把这些工程证据一起否定。

### 仍影响科学外推的问题

1. **Synthetic vs physical 要分层命名。** geometry response 来自 physical refit，但 3 tracks/event、10% missing、easy/hard backgrounds 是 overlay 任务分布；没有真实 hit overlap、shared clusters、ghost reconstruction 和 collision multiplicity 的完整模拟。物理几何真不等于 occupancy 真。
2. **多次看 validation 已发生于设计流程。** 该 corpus 承担 checkpoint、calibration、threshold、architecture/objective、mode、source influence 与研究路线决策。source-disjoint 防止直接事件泄漏，但不能使反复使用的 validation 恢复为 untouched test。新方法需新的、提前冻结的 confirmatory protocol；本轮不创建数据银行。
3. **Source-disjoint 不等于 detector-condition-disjoint。** `sample_linear_regime_points` 明确 train/validation 共用同一 payload set，这是 shared-geometry closure 的合理控制，但不构成 held-out misalignment 泛化。需要另报 source-held-out 与 condition-held-out 两轴。
4. **Truth-selected 的上限只适用于同一 survivor 集。** MC `.99` truth quality 与 complete-chain 条件让实验更干净；不能据此估计 real-data efficiency/purity，也不能忽略 FD 期间身份/接受度变化。
5. **残差减少不等于 unbiased geometry。** data monitoring 的 robust z 是相对冻结 residual scale 的 effect size，不是天然正态标准误下的 Nσ 显著性。尚缺已知漂移的 detection power、false alarm rate 与 IOV mismatch 验证。
6. **Negative result 应区分 hypothesis failure 与普遍否证。** 现有 V3/5DoF 门失败可冻结；不能把测量/likelihood/parameter chart 改正前的失败写成完整物理 no-go theorem。
7. **“normal_physics_or_coverage_loss”并非独立因果诊断。** `rigid_station_only_identifiability.py::diagnose_rank_loss` 在有限的 validity checks 通过且 rank gate 失败时直接赋此分类。它排除一部分坏文件，不排除错误 covariance 坐标、FD 偏差或模型失配。`dropped=true` 指 rank drop，不是实际删除 source；其命名尤其容易误读。

### Physical refit correctness 的分级结论

S3 `+1 mm`、27 对、最大误差 `4.866e−13 mm` 的本地产物与文档一致，证明指定 rigid translation 响应和代码路径一致。机器精度来自共享 cluster 的确定性平移，不是 detector alignment 精度。Ry 的大角单轴响应支持 rotation payload 生效，但不独立验证全 state covariance、所有条件 IOV 或 mixed-DoF global estimator。外部 C++ 不在该 Git tree，故“physical chain globally correct”只能是限定范围的强支持，不能 PROVEN。

## Software Audit

P0 指可能使一类科学主张无效，并不意味着全部结果必须作废。以下均以 `0c8a6ca` 为准；不纳入本地后续修复。

| ID / Priority | 文件/定位 | 问题及影响 | 应采取的动作 |
| --- | --- | --- | --- |
| F01 / P0 | `alignment/tracker_only_identifiable_subspace.py:391`，`bootstrap_subspaces` | 有放回抽样 multiplicity 丢失；68/69/73 event-bootstrap 语义错误 | 保留旧结果，修抽样，原 S/threshold 下写新版本 reconciliation |
| F02 / P0 | `alignment/physical_jacobian.py:413`；`scripts/run_route_selected_multidof_update.py:819` | paired reference target 是 MC oracle；“truth-free selection”易被误读为 data-available estimator | API/claim 分离 response recovery 与 absolute inference；必须通过禁读 reference 证伪 |
| F03 / P0 | `alignment/identifiable_subspace.py:270`；`rigid_station_only_identifiability.py:633` | 相对 rank gate、简单 completeness audit 被用于物理无解归因 | 拆分 numerical rank / operational rank / absolute information / cause unknown |
| F04 / P0 | `alignment/true_cluster_local_residual.py:166,241,504` | straight-line/cluster-only variance、不同 Ry sign/pivot；不能视为完整 field-aware alignment 的复核 | 统一 chart contract，构造 track nuisance 与 measurement covariance |
| F05 / P0（对误差/部署主张） | `physical_jacobian.py`、`models/field_route_fitter.py`、外部 exporter 边界 | 忽略共享边相关与上游 covariance 语义未验证；inverse normal 不等于可信 uncertainty | upstream coordinate Jacobian 契约 + global/Schur fit + coverage tests |
| F06 / P1 | `baselines/field_chi2_matching.py::_valid_covariance`；`geometry/propagation.py::mahalanobis_chi2` | 只测对称/正对角，未测 SPD；可接受负 chi2 | 反例 C 的 xy block 为 [[1,2],[2,1]]，eigenvalue −1，r=(1,−1,0,0) 得 chi2=−2；入口需 Cholesky/有限非负检查。未证明旧 corpus 实际受此污染 |
| F07 / P1 | `alignment/identifiable_subspace.py::identifiable_svd` | `full_matrices=False` 在 m<n 时漏掉额外 null basis | 2×3 rank-2 反例报告 null_dimension=1 而 v_null.shape=(3,0)；当前官方 tall banks 不受影响，稀疏部署有风险 |
| F08 / P1 | `datasets/propagation_loader.py:127`；`datasets/physical_curriculum.py:86` | 缺失 mode 默认为 0；低层 loader 默认全 split；审计 CLI 可选 test | strict schema/version 与显式访问策略；不要把 legacy 空字段当 canonical mode-0 |
| F09 / P1 | `scripts/report_{tracker_only_identifiable_subspace,rigid_station_only_identifiability,cluster_local_observable_cross_run}.py` | `exist_ok=True` 后直接 `write_text`，可覆盖 frozen report；保护并非全局一致 | immutable run ID、独占目录、成功后原子 finalize；旧负结果只 append supersession |
| F10 / P1 | `scripts/setup_environment.sh`；`pyproject.toml`；outputs manifests | 外部 Calypso/Python/PATH/lib 可变；依赖只宽下限且缺 torch/scipy/pytest 声明；只有 outputs.gitkeep | 保存实际环境与外部 source/build/condition 指纹；引入 core/ml/dev extras 与可恢复证据包 |
| F11 / P1 | `training/structured_assignment.py::route_assignment_utilities/structured_margin_loss` 与 route inference | complete-only training 与 partial inference 的 score/可行集不一致 | 记录为 V3 hypothesis limitation；冻结模型，不新增架构“修”负结果 |
| F12 / P1 | `nov22_metrology_provenance_station_ry.py::extract_alpha_beta_gamma` | Euler extraction 不是 T Rz Ry Rx 的一般逆 | 混合旋转 round-trip；当前 survey 禁止 ingest 状态不变 |
| F13 / P2 | report/feasibility modules、硬编码 root/tag/geometry | 研究策略大量写入函数与常量，跨层 import 易让新 estimator 继承历史门；report 与 solver 职责混合 | 小范围抽出 contract、matrix、report 层；不大规模重构历史 campaign |
| F14 / P2 | `scripts/audit_field_global_fit_contract.py` | 任一分支名包含 jacob/transport 即可能标支持 global fit | 用 schema、dimension、frame、state order、reference surface 与数值 test 准入 |
| F15 / P2 | `_RECORD_ROW_CACHE`、route hypothesis enumeration、CUDA scatter | cache 强引用无限保留；高 occupancy combinatorics；种子不保证跨库/设备 bitwise deterministic | bounded lifetime、资源计数、explicit budget、solver status 与 reproducibility tolerance |
| F16 / P2–P3 | 旧 roadmap、`magnitude_mm` 兼容字段、`dropped`、多义 `V1/V2/global/null` | 会让下一 agent 重开已失败任务或混淆单位 | canonical index + evidence type + superseded labels；优先改有科学误导的命名 |

旧审核列的 legacy 默认 test 访问漏洞已由 `77ec84e` 部分修复，73 项短测中的 boundary tests 验证这一点；仍有显式 override 和低层默认，不应报告为“完全未修”。`device=auto` 已在主要 ML 路径 fail loud 无 CUDA；CPU 需显式选择。无自动 CPU fallback 是可重复资源策略，不是 bug。

历史 `xcheng.cc` 曾被跟踪、后移除并 ignore。没有读取凭证内容；当前 Git ignore 不清除旧历史。维护者应另行确认历史凭证失效与仓库安全状态，不在本轮改写历史。

## Claim Audit

PROVEN 只用于严格限定的代码/算术命题；STRONGLY SUPPORTED 指多重现有证据支持的有限经验结论；TENTATIVE 尚需独立验证；UNSUPPORTED 缺直接证据；CONTRADICTED 存反例；NOT IDENTIFIABLE FROM CURRENT DATA 指当前材料不能判定，而非物理永远不能测量。

| Existing claim | Evidence | Your assessment | Confidence | Missing evidence |
| --- | --- | --- | --- | --- |
| S3 +1 mm rigid refit 响应正确 | `...station3_dx1mm_closure/closure.json`；27 对；4.866e−13 mm | STRONGLY SUPPORTED，限该链路/平移控制 | 高 | 独立 geometry/model、covariance coverage |
| 整个 physical refit/covariance 已正确 | source-audited 文档、PD/pull/closure controls | TENTATIVE；不能由 deterministic translation 推广 | 高（对限制） | pinned upstream、状态转换、过程噪声与真实 pull 验证 |
| Candidate recall=1，故 tracking 不丢粒子 | `route_metrics.py` 条件分母 | CONTRADICTED（后一推论）；conditional recall 可支持 | 高 | reconstruction/propagation 全分母瀑布 |
| 当前 ungated overlay 图上 candidate generation 不是主瓶颈 | V1/V2/V3 route counts、近 1 recall | STRONGLY SUPPORTED，限该 graph/task | 中高 | 真实 occupancy、完整 truth 可接受度 |
| V1 Transformer 扩大 capture range | 版本文档 frozen multi-direction 表 | CONTRADICTED，未满足原主门 | 中高 | 未开独立新方法测试；本审查不读 sealed 包 |
| 全局 message passing 是收益来源 | no-context 与全 context 比较、score drift | UNSUPPORTED；已有比较反向 | 中高 | matched objective、source/condition 独立机制验证 |
| 所有 V2 route context 无用 | 早期 w=0，expanded/Ry 非零 context | CONTRADICTED（全称）；因果 alignment gain 仍 TENTATIVE | 中 | 独立 downstream ablation |
| V3 当前 structured hypothesis 失败 | local validation contract；六个 magnitude 全 false | STRONGLY SUPPORTED | 高 | 无需为冻结失败开 test |
| structured/probabilistic alignment 普遍无用 | V3 完整 route hinge 失败 | UNSUPPORTED | 高 | 同一 measurement likelihood/partial-assignment 空间的试验 |
| 未发现直接 label 作为 feature | 追踪的 NODE/EDGE/MLP feature 与 label 分支 | STRONGLY SUPPORTED，非形式化全仓无泄漏证明 | 中 | 全入口 schema whitelist 与 mutation tests |
| source-file disjoint 994/796 corpus | manifest/contract 与构建器 | STRONGLY SUPPORTED | 中高 | 独立 production/condition 泛化、全输入哈希 |
| 3DoF unknown-association iterative closure | iteration-01 去重数组重算；467 边 | STRONGLY SUPPORTED，必须称 paired-reference response closure | 高 | 无 reference 的 estimator + 新 refit 迭代 |
| 该 closure 已证明 data-deployable alignment | 同数组 zero-target 失败 | CONTRADICTED（推论） | 高 | 正确绝对 likelihood、校准误差、data closure |
| 单独 C_dx / C_rx 1D 控制可恢复 | hierarchy docs/controls | STRONGLY SUPPORTED，限其它模式固定 | 中 | 共存 nuisance 与 field-aware transfer |
| 当前 hierarchy 的 leakage 极强 | B、经验 error/C_dx、Schur profiling | STRONGLY SUPPORTED，限当前模型 | 中高 | chart/covariance 修正后的值 |
| workbook 68/73 rank 门失败 | 已保存谱按 0.01 独立复算 | PROVEN（报告算术） | 高 | 不需要改门；需要校正其物理解释 |
| 68/69/73 event bootstrap 证明稳定/不稳定 | 布尔 mask 实现 | CONTRADICTED（bootstrap 语义） | 高 | multiplicity-correct event/source 重采样 |
| tracker-only 5DoF 已被物理证伪 | relative rank、limited validity classifier | UNSUPPORTED（普遍 no-go） | 高 | measurement-level profiled information 与绝对 precision/closure |
| workbook-69 独立 gate 未通过 | 8/11 < 0.80，三源失败 | PROVEN（该规则/计数）；外推为 TENTATIVE | 高 | 真正 untouched confirmatory set |
| cluster-local 在两 run full-span 一致表示 transfer | 3/3 full-space projector≈I | UNSUPPORTED；一致性指标此处平凡 | 高 | 弱模、参数预测、field-aware covariance |
| 真实 data station/reduced self-nulling 不可部署 | saved final evidence、跨 run 失败 | STRONGLY SUPPORTED | 高 | 有物理模型的补充 estimator 才能重新申请 |
| 失败唯一来自 physics_nonidentifiability | rule classifier、同一失配模型 | UNSUPPORTED | 高 | covariance/field/source/selection 因果分解 |
| Nov-22 C_dx≈0.254117 mm 可直接作当前 prior | parser 可复现，但无 measurement covariance/IOV | CONTRADICTED（可 ingest）；数值复述有支持 | 高 | 原始 metrology、相关 covariance、frame/IOV |
| 可得可靠 survey station Ry 与跨年约束 | 当前 summary/PDF/conditions | NOT IDENTIFIABLE FROM CURRENT DATA | 高 | 原始测量与机械稳定性证据 |
| 真实数据可做 residual monitoring | 13 项哈希一致冻结包、七扩展 run nominal | STRONGLY SUPPORTED，限监测一致性 | 中高 | 对已知漂移的效能/误报校准 |
| 已可自动写 official geometry | 各 gate 均 false | CONTRADICTED | 高 | 本路线末尾所有 deployment gates |

## Major Risks 与研究扩张判断

确实存在研究横向扩张：模型 V1→V2→V3 后，DoF 3→6→5、layer contrast、顺序模式、hierarchy、reduced mode、cluster-local、stable core、topology、survey/IOV、再回 rigid 5DoF。部分分支回答了有价值的问题，尤其 dedup、gauge 和 survey provenance；但默认每次失败都追加一个 campaign，会把“不同约束下得到一个可过门的模式”替代“可部署 estimator 是否无偏”的核心问题。

```text
Physics question：相对 geometry 如何改善并可信地支持重建？
       ↓
Observable：测量、磁场、covariance、track nuisance      ← 第一阻塞层
       ↓
Identifiability：结构 gauge / weak information / 精度  ← 现有结论混合三者
       ↓
Association：冻结 route control；条件 recall 高，错配偏差仍重要
       ↓
Alignment estimator：paired-response WLS 已有；absolute likelihood 缺失
       ↓
Iterative closure：paired 控制已闭；无 oracle 全闭环未证实
       ↓
Real-data validation：当前仅 residual/DQ monitoring
```

这是证据依赖顺序；实际系统中 association 与 alignment 会迭代耦合。association 的成熟不能越过 observable/estimator 门。

现在最不应继续的五件事：

1. 训练更大 Transformer/GNN、增加 attention 或超参数来解决 weak mode。
2. 改 S/rank tolerance、删失败 source/DoF，或选择有利 slope bins 来把旧门“修成通过”。
3. 在 covariance/coordinate contract 未解决前追加大规模 5DoF physical banks、topology 搜索或 bootstrap campaign。
4. 以同事件 reference-subtracted residual 的下降宣称绝对 geometry 精度，或用 real-data self-nulling 写 conditions。
5. 用现有 validation 再选模型/阈值后称独立确认，或重开 V1 sealed event content。

## Recommended End-State Architecture

**一个 hybrid 系统：物理 likelihood 是主干，ML 仅作可关闭的 association 辅助。** 不要求必须含 ML。

| 模块 | 输入 → 输出 | 必须物理/适合 ML | uncertainty |
| --- | --- | --- | --- |
| Measurement adapter | local-u/必要 stereo 测量、sensor IDs、surface、IOV → typed measurements | physics 必须；不猜 covariance | sensor covariance 与 calibration nuisance |
| Candidate builder | tracklets、geometry、field、传播合约 → 稀疏候选/允许缺失的 route hypotheses | physics 主导；冻结旧图作控制 | propagation uncertainty、missing acceptance |
| Route track fit | route measurements + geometry → q̂、likelihood、track/global derivatives | physics 必须；q/p 从可用观测 fit 或合理 nuisance | track covariance、material/process noise、shared-hit accounting |
| Assignment | route likelihood、occupancy/background、capacity → MAP 与限定候选上的概率/边际 | 可用冻结 V2 提案；不必新网络 | unmatched、竞争 routes、截断误差、selection dependence |
| Alignment estimator | measurement derivatives、q nuisance、assignment、survey → Δα 与 joint covariance | constrained global/Schur likelihood | data information、prior information、assignment/model 系统项分开 |
| Refit/iteration | Δα、当前 full transforms → shadow geometry、重新传播/拟合/关联 | physics；明确 SE(3) composition | 新 anchor FD validity、step uncertainty、IOV |
| Validation/publish gate | 闭环/held-out/run DQ/provenance → accept/reject/rollback package | 无需 ML | acceptance 必须看 bias、coverage 和 physics observables |

采用测量 residual `r=m−h(q,α)`，在共同 anchor 线性化 `h≈h0+G Δα+H Δq`。在独立测量或正确组合的协方差 R 下，轨迹 nuisance 消元给出

```text
P = R⁻¹ − R⁻¹ H (Hᵀ R⁻¹ H + Nq)⁻¹ Hᵀ R⁻¹
Nα = Σ Gᵀ P G + Ns
bα = Σ Gᵀ P r + bs
```

`Nq` 表示物理上有根据的 local nuisance 信息；`Ns,bs` 是 survey 等外部测量在当前 anchor 的贡献。实际应按完整分块方程处理非零 prior 均值、process-noise/scatter 参数、rank-deficient local blocks 和 gauge；上式是结构示意。建议先实现小规模 Schur 与直接 joint least-squares 的等价测试，再接入 Acts。该 global/local 消元路线有成熟依据，见 [Millepede II 方法手册](https://www.desy.de/~kleinwrt/MP2/doc/html/draftman_page.html)；这里推荐的是其统计结构，不是要求项目立即迁移到特定软件包。

磁场中不可用 straight-line 替代 q/p nuisance。FASER tracker 的设计、metrology 与实际重建已有 detector 文献背景，见 [The tracking detector of the FASER experiment](https://arxiv.org/abs/2112.01116)；论文中的 assembly precision 不能替换 Nov-22 具体数据的 measurement covariance。

联合问题可写为 `p(m,s | α)=Σ_{A feasible} p(A|occupancy) ∫p(m|q,α,A)p(q)dq · p(s|α)`。先实现 hard-assignment physics baseline；若需要软 assignment，只在冻结的小事件候选集合上验证归一化/容量与 approximate-EM 的误差。posterior 权重本身由同一 residual 获得，不能把它当固定外部权重再把 Hessian 逆当完整 posterior；还需 observed-information 或 event/source bootstrap。不要把 V2 分数与含同一 residual 的 likelihood 盲目相乘。

Survey 的最终定位：有可靠 covariance 和 frame/IOV 的独立 measurement 因子；true gauge 用数学约束；初始化用同一可信来源但不丢弃其误差；未认证的 summary 仅 external cross-check。已作为 prior 的数据不能再当独立 validation。移动旋转 pivot 并非让角度不可转换：`p′=c+R(p−c)+t_c=Rp+[t_c+(I−R)c]`，需转换完整 SE(3) 及相关 covariance。当前不 ingest 的根本理由仍是 provenance/covariance/IOV 缺口和内部形变混合。

停止条件不只是 residual 小：两次连续 accepted step 的预测 sensor motion/参数增量小于预注册精度预算，objective 不恶化，assignment 与 held-out predictive checks 稳定，FD 线性误差受控，covariance/coverage 合格。最大迭代数预先固定；触发非线性、coverage 不足或 provenance 错误就返回失败，无自动增加数据/次数。

## Stop / Continue / Start Decisions

**Stop：** 当前 tracker-only 5DoF 的“rank rescue”与无证据 ML 扩容；任何真实 geometry correction。冻结旧门与负结果，但修订其物理解释。

**Continue：** 现有 frozen association controls、physical refit 控制、严格 provenance、完整失败源保留和 residual monitoring。MC paired closure 作为回归资产继续有用。

**Start：** 一次有计算上限的 measurement/estimator 证伪链：统计和 chart 修复 → 上游 covariance 契约 → 无 reference 的 truth-association global estimator → unknown association + uncertainty → 真实 refit closed loop。详见配套 task；任一关键失败可终止其后阶段。外部 survey 获取是独立资料依赖，不能以无休止文件搜索替代物理验证，也不在本轮联系他人。

### 最后十个问题的明确回答

1. **大方向对吗？** 相对 geometry + association 的物理目标对；把网络升级或通过某个 rank 门作为研究主线不对。
2. **最大 scientific risk？** 把不完整 measurement/propagation/weight 模型的失败误判成 detector 的物理不可辨识，或把 paired response recovery 误判成真实 alignment。
3. **最大 software/methodological risk？** 确定的 bootstrap 语义 bug，加上跨仓 covariance/geometry contract 未闭合；反复 validation 决策进一步限制外推。
4. **真正 bottleneck？** 首先是 observable/covariance 与 absolute alignment estimator，其次才是该模型和样本下的 weak information；错配影响已经存在但不是新增模型的先决理由。
5. **还需复杂 Transformer/GNN？** 现在不需要。冻结 V2、MLP 和 physics controls 足以检验下一步科学问题。
6. **tracker-only multi-DoF 值得继续？** 当前 rescue campaign 停止。仅允许一次有界的 field-aware profiled-likelihood 证伪；未通过则停止无约束多 DoF 部署路线，不继续换参数族。
7. **survey 怎么进入？** 经 frame、covariance、IOV 认证的相关 measurement prior；只有 true gauge 用 exact constraint。当前 Nov-22 summary 继续 cross-check only。
8. **下一项最值得实验？** 在同一既有 physical population 上完成 covariance/chart 合约后，冻结 truth-association，比较 paired reference 与无 reference 的 measurement-level global fit；先用已有数组复现本文 oracle-gap，不做训练。该最小对照直接决定昂贵闭环是否值得运行。
9. **只能留一条主线？** 可验证的、受物理与合格外部约束控制的 global alignment likelihood，association 是其中受检验的一层。
10. **最可信/有论文意义的贡献？** FASER 几何与重建契约下，对 association-induced alignment bias、observable compression/weak modes、uncertainty 与闭环部署边界作可复现的系统测量；若新 estimator 成功，给出条件明确的 constrained alignment 方法。不能提前承诺“Transformer 首次解决 alignment”或普遍 tracker-only no-go。负结果加完整反证/边界也有价值，投稿新颖性需另行文献定位。

## 证据索引与复算记录

以下路径均在项目 `outputs/` 下，全部为本轮实际读取的非 sealed-test 证据。SHA256 是审查时复算；恢复证据时须重新核对。它们不是代码或 upstream binary 的 hash。

| Evidence ID | 相对 outputs 路径 | SHA256 |
| --- | --- | --- |
| E01 | `mc24_muon_fasernu_5events_segment_refit_station3_dx1mm_closure/closure.json` | `38d9a6d462a79397824b10fd3d9945576c720b85f0bc34e971b01486304505b4` |
| E02 | `mc24_multidof_ift_iteration01_route_selected_update_validation_physical_edge_deduplicated_v1/route_selected_update.json` | `e16849a2f47a456318730384ee6752095fed5381cc5970560528b7dad587e437` |
| E03 | `mc24_v3_expanded_trainval_v3_route_validation_v1/validation_run_contract.json` | `ed438a2ce2ef5cec7ee131b3f5fd02c0654222dc4e7f6bde2c8459ad35470210` |
| E04 | `mc24_v3_expanded_trainval_v3_route_validation_v1/validation_selected_operating_point.json` | `77fb5e16b90b9b9941f9198ae25ee6425deabde6083abb2447a68c330c3cfe50` |
| E05 | `rigid_station_only_tracker_alignment_identifiability_v1/identifiable_basis.json` | `f8a8a92b02e99c17151b5c7a8918253df823ddd79e385a340fcd8dfae7664229` |
| E06 | `tracker_only_identifiable_subspace_three_arm_closure_v1/identifiable_basis.json` | `367773837eda2d8cb30ee4b0a6f75e3fb49530315e9680e7c964de9bd399e645` |
| E07 | `cluster_local_observable_cross_run_identifiability_v1/cross_run_subspace.json` | `dcabec2805652d6d62631cbf3250bb1e11cf5449672e6ec9e95d25fdd7f7e9fd` |
| E08 | `cross_source_stable_core_identifiable_subspace_v1/independent_validation.json` | `12df6f42209f1069f8e72c6f4d857e108a615dc2e1d6f3dd051f08688a157da0` |
| E09 | `operating_protocol_v1_final_real_data_closure_v1/reproducibility_manifest.json` | `6e93faf9cd32c88979530e2c5df03ab131e880310e92030f6b681aa16465020b` |

E02 同目录 `route_selected_update_arrays.npz` 的 SHA256 为 `322959624c7a5e7cb53007e04ca7a5b9131d348c53b2e9a138c9956e608c3451`。诊断为：从其 J、C 重建 `N=ΣJᵀC⁻¹J`，分别用 RHS `ΣJᵀC⁻¹(r_target−r_anchor)` 与 `ΣJᵀC⁻¹(−r_anchor)` 求解。前者在数值精度内复现保存的三参数增量；两解之差等于 reference residual 的投影 `(0.19650 mm,0.08910 mm,3.42161 mrad)`。这是原 frozen validation 上的**探索性审计反例**，不是新 confirmatory 测试或新算法结果。

本轮没有读取新本地 workbook 75 及以后的实验内容来补齐远端 master；不得将它们的结果暗中并入本报告。后续 agent 若选择从更新代码开发，必须先做明确的基准 reconciliation，再把任务判定为已修复、仍存在或不适用。

### 可复算的关键数值检查

以下是 E02 的只读重算，不需要重建 ROOT/refit，也不打开其它数据。保存的参数增量在 JSON 的 `parameters` 中；不能假定不同年代的 NPZ 都有 `recovered_delta` 字段。先核对上述两个输入哈希，再运行：

```python
import json
from pathlib import Path
import numpy as np

base = Path("outputs/mc24_multidof_ift_iteration01_route_selected_update_"
            "validation_physical_edge_deduplicated_v1")
report = json.loads((base / "route_selected_update.json").read_text())
with np.load(base / "route_selected_update_arrays.npz", allow_pickle=False) as z:
    names = z["parameter_names"].tolist()
    j, c, ra, rt = [z[k] for k in (
        "derivative_native", "covariance", "anchor_residual", "target_residual")]
    normal = np.einsum("nki,nkj->ij", j, np.linalg.solve(c, j))
    def delta(residual):
        wr = np.linalg.solve(c, residual[..., None])[..., 0]
        return np.linalg.solve(normal, np.einsum("nki,nk->i", j, wr))
    by_name = {p["name"]: p for p in report["parameters"]}
    saved = np.array([by_name[n]["recovered_local_delta"] for n in names])
    anchor = np.array([by_name[n]["anchor_value"] for n in names])
    np.testing.assert_allclose(delta(rt-ra), saved, rtol=1e-8, atol=1e-8)
    print(names, len(j))
    print("paired / zero / reference projection", delta(rt-ra), delta(-ra), delta(rt))
    print("remaining zero-target error", anchor + delta(-ra))
```

小矩阵反例不需要任何 event 内容：`C=eye(4); C[0,1]=C[1,0]=2`、`r=(1,−1,0,0)` 可验证非 SPD 却得负 chi2；`rng=default_rng(7); ids=rng.integers(0,1000,1000)` 后 `len(unique(ids))=629` 可复现布尔 mask 丢失 multiplicity。相对 rank 反例和满维 projector 恒等式已在正文给出。修复任务须把这些转成永久 regression tests，而不是仅保留临时审计脚本。

短测复现时，工作目录必须是单独导出的审查提交，不能是有后续代码的本地工作树；仅设置 PYTHONPATH 不足以阻止当前目录优先导入。测试命令为：

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_110_cuda/x86_64-el9-gcc13-opt/setup.sh
export PYTHONPATH=/tmp/alignment-master-audit-SavyXR:/tmp/alignment-audit-testdeps:${PYTHONPATH:-}
cd /tmp/alignment-master-audit-SavyXR
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_physical_jacobian.py tests/test_tracker_only_identifiable_subspace.py \
  tests/test_rigid_station_only_identifiability.py tests/test_cluster_local_observable_cross_run.py \
  tests/test_route_assignment.py tests/test_structured_assignment.py \
  tests/test_anchor_selected_update.py tests/test_gauge_equivalence.py \
  tests/test_expanded_control_boundaries.py tests/test_ntuple_conversion.py \
  tests/test_field_route_fitter.py
```

临时路径只是本次运行位置，可能被系统清理；以后需从固定 SHA 重建隔离快照与所列依赖，不能依赖 `/tmp` 作为永久 evidence store。

### 交付验证与未完成证据

本报告为 **Share with caveats**：足以支持停止当前扩容/救援 campaign 和安排一次有界证伪；不足以认证新 estimator、完整物理 no-go 或 real-data geometry correction。已核对任务字段与依赖、表格数值/限定分母、代码路径及证据哈希。量化证据用精确对照表，依赖关系用文本图；没有从稀疏历史点绘制暗示连续趋势的图。交付为用户指定的 Markdown，检查了标题、表格与代码围栏；未做 GitHub 浏览器渲染验证。

未完成且会阻止部署的证据：历史完整 upstream 构建恢复、独立 covariance/measurement contract、无 oracle 的物理闭环、校准的 coverage、合格 survey/IOV 和独立真实数据验证。它们是路线的显式门，不是本轮暗中完成的工作。技术报告的进一步问题落实在 T10–T12 的假设与分支规则中；本轮不额外生成 notebook、HTML 或公开站点。
