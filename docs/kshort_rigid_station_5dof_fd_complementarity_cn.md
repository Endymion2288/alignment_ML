# 物理不同 K-short 刚体站 5DoF FD 互补可辨识性 V1

Workbook 76 / 2026-09-05。Workbook 59–75 保持冻结。本文档是该战役的
中文预注册；中文 workbook 条目为
`workbook/2026-09-05_76_K-short物理不同轨迹的刚体站5DoF_FD互补可辨识性.md`。
以下全部判据在首个完整 K-short FD HTCondor submission 之前冻结。
禁止在查看任何 K-short FD singular spectrum 之后修改任何判据。

## 科学问题

条目 73 冻结 `rigid_station_five_dof_not_source_or_coverage_portable`：
canonical 18-source rigid-station 5DoF pooled rank 可达 5，但
source-disjoint 与 coverage-disjoint portability 失败。条目 74/75 完成
track-coverage search：唯一通过 residual-blind coverage inventory 的
candidate 是 `mc24_100130_kshort_end_fasernu`——10 个 file-level
independent xAOD source，其 charged-pion daughter population 带有
物理上不同的 angular/topological support。

本战役回答的唯一问题：这种物理上不同的 support 是否为 canonical
population 提供**稳定、可 source-disjoint / coverage-disjoint 搬运的
新增 alignment information**，从而解决条目 73 的 portability 失败。

本战役**不是**“K-short 单独是不是 rank 5”，也**不是**“canonical +
K-short pooled 后是不是 rank 5”。`joint pooled rank == 5` 单独绝不是
成功标准。必须区分三个 information systems：

- **A. canonical-only**——复现条目 73 frozen conclusion
  （regression / negative control，不是重新调参）。
- **B. K-short-candidate-only**——10 个 source 分别与整体的同定义
  spectrum、rank、identifiable subspace、source-disjoint 与
  coverage-disjoint portability。K-short-only **不**要求 rank 5；其
  价值可能只在于提供 canonical 缺少的部分方向。
- **C. joint complementarity**——预注册的 canonical+K-short 组合：
  subspace overlap / principal angles、weakest canonical direction 上
  的信息量、joint source/coverage stability、leave-source-out 与
  bootstrap stability，以及 workbook-69 stable-core independent
  validation（角色预注册：hypothesis = 18 canonical sources，
  independent = 10 K-short sources）。

## 冻结参数化（继承，绝不反调）

- 参数向量 `ift_dx_mm, ift_dy_mm, ift_rx_mrad, ift_ry_mrad,
  ift_rz_mrad`（mm, mm, mrad, mrad, mrad）；`ift_dz_mm` 与 `C_dx`
  不进入 tracker fit。
- `S = (5, 5, 60, 60, 60)`；`rank_tolerance = 0.01`；
  `A = W^{1/2} J S`；W = nominal WLS inverse 4x4 pair covariance。
- physical central FD step `(0.5, 0.5, 10, 10, 10)`，与 hierarchical
  V1 iteration-00 bank（workbook 44）及条目 73 完全一致。
- residual observable：条目 73 的 truth-selected mode-0 pair residual
  （`min_truth_match_fraction = 0.99`，`q_over_p_mode = 0`）。
- source-level rank 定义、event bootstrap（40 replicates、12
  half-splits、seed 20260905 沿用冻结的 workbook-date 约定）、
  slope-tertile coverage（3 bins、min 8 events）、
  complete-four-station truth-route topology，以及全部
  principal-angle / Frobenius 阈值（source/split/disjoint/topology
  15°/1.0；bootstrap 20°/1.2）原样复用条目 68/69/73 实现。
- workbook-73/74/75 config、workbook-75 manifest 与 reinventory、
  canonical iteration manifest 的继承 SHA256 在 config 加载时逐文件
  校验；任何不匹配即 hard failure。

## K-short population contract

条目 75 的 `|pdg| = 211` mask 只用于定义 residual-blind angular
population（中性 `K0S` 不在 tracker 留 charged trajectory；
tracker-visible objects 是衰变 daughters）。FD observable **不**换成
pion-specific residual：原样继承条目 73 的 rigid-station physical-FD
observable 与 WLS contract。truth matching 为 majority-barcode 且
particle-generic（条目 75 exporter-contract 审计：无 PDG=13 强制、
无单 muon parent 假设），其对 K-short decay daughters 的适用性已在
预注册中冻结（`fd_observable_truth_selection:
particle_generic_majority_barcode`，`fd_observable_pdg_mask: none`）。
不得在看 FD 结果后改变 pion/track/event selection。

10 个 K-short source 保持 file-level independent。train/validation
split 按 file-name order 预注册（前 5 train，后 5 validation），永不
修订。不得把多个 source 事后合并为一个 source 来隐藏失败，也不得
看到 singular spectrum 后删 source。

## merged-rec physical-event identity（硬合同）

条目 75 证明 merged MC24 K-short rec 文件中 `(run_id, event_id)` 不是
唯一 physical-event key：每个 rec 文件合并 10 个 generator job，event
number 重启。因此任何 sorted `(run,event)` grouping 都禁止用于
cross-geometry FD matching。

实现（`datasets/physical_event_identity.py`）：

- 每个 FD refit point 写出 `enhanced_tracklets.root` ntuple，每
  processed event 一个 entry（file order）。entry 级 occurrence = 该
  entry 之前相同 `(run, eventID)` 的 entry 数；synthetic unique key
  `uid = event_id + occurrence * 2^20`。
- per-entry tracklet/propagation vector length 把 flat
  `tracklets.root` / `propagations.root` 的 row block 桥接回 ntuple
  entry，因此即使某 event 只对其中一个文件贡献行（tracklet-only
  event），两个文件的 occurrence 指派仍然一致。
- reference 与 ±FD geometry 间的 join 是 deterministic exact join
  （augmented uid + 既有 tracklet/truth identity）。禁止 fuzzy join、
  nearest-neighbour join、residual-proximity join。duplicate
  physical-event identity 是 hard failure。
- 无碰撞文件中 occurrence 恒为 0，uid 等于 raw event id，canonical
  行为 bit-identical。

回归测试故意构造重复 `(run, event)` block、tracklet-only 中间 event、
跨 occurrence misjoin 诱饵，确认：不合并、不错配、sorted loader 对该
类文件 hard-fail、正确配对的 residual 精确等于设计值。

## FD reconstruction（仅 HTCondor）

Interactive 节点只允许 config validation、unit test、tiny smoke、单
point 短 provenance check。完整 K-short physical FD reconstruction 在
HTCondor 上运行，复用既有 physical payload / Calypso FD
infrastructure（`prepare_multisource_multidof_iteration.py` +
`submit_multisource_multidof_iteration_condor.py`）；不另写第二套
reconstruction 链。

- 模板 `configs/physical_refit_mc24_kshort_5dof_fd_iteration.yaml` 逐字
  复制 canonical hierarchical V1 `physical_refit_capture_scan` 块
  （全部 7 个 parameter specs，含 `ift_dz_mm` / `C_dx` 探针，冻结
  step；分析时经 `only_parameters` 取 5 个 station-free 参数，与条目
  73 加载 canonical bank 的方式一致）。`nevents = 10000`（每文件全部
  event）。无 held-out closure point。
- 每个 candidate source × physical FD point 保存：exact input xAOD、
  source id、config SHA、starting git SHA、Calypso revision、
  payload/geometry delta、command、environment、stdout/stderr、exit
  code、tracklet/refit path、join counts、dropped/unmatched counts 及
  明确原因。
- 失败 job 不静默删除。infrastructure failure 可重试；不得为改善
  rank 重跑/选择 physics source。

## 分析顺序：先 physical closure，后 information analysis

1. **Physical closure / provenance**——每个 pre-declared source × FD
   point 完整（payload、tracklets、propagations、enhanced ntuple、
   content audit），per-source Jacobian validity 过冻结 gate，
   exact-join duplicate physical-event identity 计数为 0，无
   geometry/official-conditions write。physical chain 失败则分类为
   physical/provenance failure，不输出任何 “rank”。
2. **Canonical-only regression**——复用 frozen canonical bank（SHA
   校验，不重跑 Athena）；重建的条目 73 分析必须复现 frozen
   per-source rank、singular values（rtol 1e-10）、pooled rank 与
   frozen decision。
3. **K-short-only**——per-source 与 pooled 的 `A = W^{1/2} J S`、
   singular spectrum、rank、identifiable subspace、source stability
   （参照 K-short pooled native rank，不要求 rank 5）、slope-tertile /
   topology coverage stability、bootstrap / half-split stability。
4. **Joint complementarity**——预注册组合（canonical 18 + K-short 10
   pooled，equal-per-source consensus）：canonical↔K-short pooled
   subspace principal angles（report-only；无冻结阈值，禁止发明）、
   weakest canonical direction 上 K-short 信息量（report-only）、
   joint pooled rank（必要非充分）、joint source stability
   （pooled-remainder LOSO：去掉每个 source 的 pooled subspace vs 完整
   pooled subspace，rank 5 / 15° / 1.0；条目 73 的
   remaining-vs-left-out-source 方向仍报告但不作 joint gate，因为
   低-rank canonical source 按构造留在 pool 中）、joint coverage
   stability、joint bootstrap / half-split stability，以及 workbook-69
   stable-core independent validation。

## 预注册判定（FD 前冻结）

`complementarity_pass = true` 当且仅当以下**全部**成立（数值阈值全部
继承条目 68/69/73）：

1. physical closure / provenance 全部通过；
2. canonical-only regression 精确复现条目 73 frozen conclusion；
3. K-short pooled identifiable rank ≥ 1（有信息；不要求 rank 5）；
4. K-short source stability 在其 native pooled rank 上过冻结 gate；
5. K-short coverage stability（tertile + topology）在其 native pooled
   rank 上过冻结 gate；
6. joint pooled identifiable rank == 5（必要非充分）；
7. joint source stability（pooled-remainder LOSO）过冻结 gate；
8. joint coverage stability 过冻结 gate；
9. joint bootstrap / half-split stability 过冻结 gate；
10. workbook-69 independent validation：canonical hypothesis core
    dimension == 5 且 K-short independent set 过冻结 support /
    persistence / angle / missing-Frobenius gate。

## 条件分支

- **成功**：冻结条目 76 与 artifact。真实数据 correction 保持关闭。
  后续 full tracker-only 5DoF controlled closure、three-arm
  synthetic/injected closure、Frozen-V2 unknown-association
  integration 各自单独预注册。
- **失败**：冻结失败。禁止：降 rank tolerance、改 S、删 source、改
  pion selection、改 residual、再找一个相近 MC 连续试、回到 7D /
  cluster-local / stable-core rescue。此时条目 74/75 的
  track-coverage search 已完成，唯一 admitted candidate 未解决 5DoF
  portability，才满足转向单独预注册的 **Gauge-Constrained /
  External-Constraint Alignment Feasibility** 战役的科学条件。
  external constraint 仍须区分 real independent survey/metrology 与
  software gauge/regularization；无 measurement covariance、validated
  frame mapping、IOV provenance 的 Nov-2022 quantity 不得进入
  physical prior。

## 工程

- `configs/physical_curriculum_mc24_kshort_trainval.yaml`——10 source
  corpus 与预注册 split。
- `configs/physical_refit_mc24_kshort_5dof_fd_iteration.yaml`——FD 生产
  模板。
- `configs/kshort_rigid_station_5dof_fd_complementarity_feasibility_v1.yaml`——
  战役 config（本预注册核心）。
- `datasets/physical_event_identity.py`——ntuple-bridged occurrence
  identity；`datasets/root_loader.py` 重构出 `_read_tracklet_columns`
  单一字段读取源（行为不变，全回归通过）。
- `alignment/kshort_rigid_station_5dof_complementarity.py`——分析模块，
  复用条目 68/69/73 原语，无算法复制。
- `scripts/report_kshort_rigid_station_5dof_complementarity.py`——
  `validate-config` / `canonical-regression` / `full` 阶段。
- `tests/test_kshort_rigid_station_5dof_complementarity.py`——19 tests。
- `outputs/kshort_rigid_station_5dof_fd_complementarity_feasibility_v1/`——
  artifact 含 SHA256、config SHA、source SHA、starting git SHA、
  Calypso revision、commands、Condor cluster/job IDs、provenance、
  pass/fail reason、final frozen decision。
