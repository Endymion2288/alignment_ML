# 77. Gauge-Constrained / External-Constraint Alignment Feasibility V1

日期：2026-09-05
状态：**完成并冻结**（预注册 `9ba17c7` → amendment `ab608da` → 结果回填）

## 任务与科学问题

在连续预注册战役否定了全部 tracker-only 路线之后——7D tracker-only
identifiable subspace（条目 68：basis 不稳定）、cross-source stable
core（条目 69：独立验证 8/11 < 0.80 失败）、cluster-local observable
（条目 70-71）、rigid-station-only 5DoF（条目 73：pooled rank 5 但不可
移植）、physically-distinct K-short coverage（条目 74-76：物理闭合失败，
每源 movable-station 对 6-19 < 冻结门 min_pairs=20）——本条目回答两个
严格分开的问题：

**在 tracker information 不完整的现实条件下，什么是合法的
reconstruction gauge representative？现有外部测量是否有资格把其中某些
gauge/null directions 提升为真正的物理约束？**

本阶段只做 constraint-space design / feasibility，**不求真实数据
correction**。

## 两个概念的严格分离（本条目核心纪律）

- **A) software/reconstruction gauge constraint**：线性约束
  `G θ = 0`，只用于在 tracker 不可辨识方向中选择一个确定的数学代表。
  它**不是物理测量**；gauge-fixed 参数永远不得写成"测得为 0"。
- **B) external physical constraint**：只有真正独立、frame mapping 已
  验证、measurement covariance 已知、IOV/mechanical-stability
  provenance 明确的 survey/metrology 才能充当。二者不得混在同一个
  结论里。

## 冻结继承（SHA256 在 config 中逐条钉死）

- 条目 68：7D 参数空间
  `(ift_dx_mm, ift_dy_mm, ift_dz_mm, ift_rx_mrad, ift_ry_mrad,
  ift_rz_mrad, C_dx)`，S = (5, 5, 5, 60, 60, 60, 0.12)，
  `rank_tolerance = 0.01`，`A = W^{1/2} J S`；pooled rank 5 /
  null 2（近零奇异值 28.16 与 2.52，相对最大奇异值均 < 0.01）；
  `solver_restricted_to_identifiable_subspace`；
  "V_null^T u_hat = 0 是 scaled 坐标下的 minimum-norm/gauge 代表，
  不是 null modes 物理上为零的测量"。
- 条目 69/73/76：全部 identifiability 路线的失败结论保持冻结；禁止
  事后降低 min_pairs、禁止把条目 76 的 report-only K-short pooled
  rank=5 重新解释为成功、禁止用条目 76 的 FD 谱设计或调整任何 gauge
  判据、禁止再找相近 MC / 删 source / 改 residual / 改 S / 改
  rank_tolerance / 删 DoF / 重开任何 rescue / 继续堆同类 tracker
  data 追 rank。
- 条目 61-67：现有 survey/metrology 证据全部不是可 ingest 的物理约束。
  `ift_C_dx` / `ift_l0_minus_l2_dx` 保持 `feasibility_only,
  sigma=null`；`ift_station0_ry` 保持 `unavailable`；Nov-2022/cad_survey
  数字、population Sigma、现有 `/Tracker/Align` conditions、Kabsch /
  dz-vs-x tilt 继续不得冒充 physical prior。条目 67 的 Stations ry
  软件合同（FASER 原点左乘 `T·Rz·Ry·Rx`）保持冻结。

## 参数空间与 tracker 信息合同

- 参数空间：冻结 hierarchical V1 7D（同上），不增不减。
- Tracker 信息：从冻结的 hierarchical V1 iteration-00 bank 用冻结
  loader 重建 pooled 7D normal matrix（与条目 68 完全相同的 7 源
  corpus、anchor、truth-match、mode-0），并对冻结的条目 68
  `identifiable_basis.json` 做回归（奇异值 rtol=1e-6、rank/null 精确、
  identifiable/null basis 最大主角 ≤ 1e-6°）。回归失败则整个战役以
  `tracker_information_regression_failed` 终止，不做任何 gauge 使用。
- 不做任何新重建；本阶段无 HTCondor 任务。

## 候选数学 gauge（三类，全部预注册）

每个候选都给出：线性约束形式 `G θ = 0`（native 7 参数顺序显式矩阵）、
剩余自由度（一律 5）、是否改变 tracker-observable prediction（一律
否，由 closure 验证）、与 Calypso alignment convention 的关系、哪些
物理参数因此失去独立解释。

1. **`named_parameter_zero_gauge`**（固定明确 reference rigid mode）：
   `θ_dz = 0` 且 `θ_C_dx = 0`。对应 Calypso 既有软件 gauge 语言
   （条目 61-62：dz 不返回 tracks、dz=0±5mm 是软件 gauge；C_dx 是
   gauge 定义而非测量）。`ift_dz_mm` 与 `C_dx` 失去独立解释；
   `ift_dx_mm` 意味着 "dz=0 且 C_dx=0 下的 dx"。
2. **`minimum_norm_scaled_gauge`**（minimum-norm representative，
   severity-scaled 度规）：`V_null^T S^{-1} θ = 0`。即条目 68 冻结
   假设中已命名的代表。dx/dz/C_dx 混合方向失去独立解释。
3. **`minimum_norm_native_gauge`**（minimum-norm representative，
   native 度规）：`(S V_null)^T θ = 0`。与 scaled 版在 S 非均匀处
   给出不同代表（预期，report-only 对比）。

合法性判据（预注册）：`rank(G) = 2` 且 `det(G · S · V_null) ≠ 0`
（约束与每个 gauge 轨道恰好相交一次）。

## Pure-gauge closure（MC/control 线性代数，预注册 gate）

遵循冻结的 `solver_restricted_to_identifiable_subspace` 原则：solve
使用可辨识限制 `A_eff = A P_id`（近零方向被冻结 rank cut 排除在
solve 之外，gauge 空间因此是 exact null(A_eff)，gauge 不变性是精确
线性代数命题而非近似）。测量模型 `r_w = A_eff u + ε`，
`ε ~ N(0, I)`（加权空间，σ=1.0）。

对每个 replicate：抽取含 identifiable + null 成分的 truth `u*`，构造
仅沿 null 方向不同的 observable-equivalent representatives `u*_k`
（产生完全相同的测量），在每种合法 gauge 下用 KKT 求解（禁止
full-parameter unconstrained Newton）。

预注册 gate（全部必须通过）：

- `gauge_constraint_exactly_satisfied`：`|G û|max ≤ 1e-9`
- `representative_unique_per_gauge`：同一 gauge 下不同等价 truth 给出
  逐位相同代表（≤ 1e-8）——solver 不沿未约束 null direction 漂移
- `observable_prediction_gauge_invariant`：跨 gauge 可观测预测相对差
  ≤ 1e-8
- `identifiable_projection_gauge_invariant`：跨 gauge 可辨识投影绝对差
  ≤ 1e-8
- `held_out_observables_gauge_invariant`：按 pair 对半分（held-out
  seed 冻结），half-1 求解、half-2 预测，跨 gauge 相对差 ≤ 1e-8
- `kkt_solve_residual_within_gate` ≤ 1e-8；`kkt_numerically_stable`
  （满秩且 cond ≤ 1e12）

显式**非** gate：恢复完整注入 θ truth（gauge-equivalent truth 本身非
唯一）。跨 gauge 的 gauge-dependent absolute components 允许且预期不
同（report-only）。被冻结 rank cut 丢弃的 near-null 信息作为诊断量
报告，永不使用。

## External-constraint eligibility table（预注册规则）

对 15 个候选 quantity 逐项审计：measurement value、真实 measurement
uncertainty/covariance、coordinate frame、pivot/rotation convention、
Calypso parameter mapping、measurement date、IOV/mechanical-stability
provenance、独立性（相对于 tracks 与 conditions）。四个门**全部**通过
才标记 `eligible_physical_constraint=true`：

1. `parameter_mapping_validated`
2. `measurement_covariance_has_independent_provenance`
3. `measurement_year_conditions_iov_identified`
4. `independent_of_tracks_and_conditions`

三个官方 slot（`ift_C_dx`、`ift_l0_minus_l2_dx`、`ift_station0_ry`）
额外与冻结的条目 66/67 slot artifact 机械交叉核对（availability、
sigma、value 一致）。审计行见 config `external_constraint_eligibility.
audit_rows`。

**预注册预期**（来自条目 61-67 冻结链）：没有任何现有 quantity 通过
全部四门 → 冻结 `no_ingestable_external_physical_constraint_available`。
这不阻止纯 gauge feasibility。若意外发现 eligible 候选，判定为
`external_physical_constraint_candidate_found_separate_subcampaign_required`
并另开独立子战役（measurement 写成 `y = H θ + ε, ε~N(0,C)`，H 的
frame/pivot mapping 须已有代码/几何验证，C 须来自真实 measurement
covariance；无 covariance 不得做正式 Fisher/posterior，只能
feasibility report）。本阶段**永不**授权 ingest。

## 预注册判定（decision tree，顺序短路）

1. tracker 信息回归失败 → `tracker_information_regression_failed`
2. 任一 gauge 合同非法 → `gauge_constraint_contract_invalid`
3. KKT 求解病态/漂移 → `gauge_fixed_solver_not_well_posed`
4. closure 任一 gate 失败 → `gauge_invariant_observable_closure_failed`
5. 发现 eligible external 候选 →
   `external_physical_constraint_candidate_found_separate_subcampaign_required`
6. 否则 → `gauge_feasibility_closed_no_eligible_external_physical_constraint`

本阶段冻结：

- `gauge_constraint_contract_valid = ?`
- `gauge_fixed_solver_well_posed = ?`
- `gauge_invariant_observable_closure = ?`
- `eligible_external_physical_constraints = [...]`
- `external_constraint_ingest_authorized = false`（恒定）
- `real_data_candidate_alignment_authorized = false`（恒定）
- `geometry_write_allowed = false`（恒定）
- `official_conditions_write_allowed = false`（恒定）

## 条件分支

- 纯 gauge closure 通过：不立即写真实 geometry。下一阶段须独立预注册
  **Gauge-Fixed Real-Data Alignment Diagnostic V1**：输出只能称为
  reconstruction gauge representative / candidate diagnostic，不得解释
  为唯一真实 mechanical station pose；用 calibration/held-out split，
  只在 held-out residual/DQ population 上评价 gauge-invariant
  observable improvement；official COOL/POOL write 保持关闭。
- 发现真正 eligible 的 external physical measurement：另开独立子战役，
  不与 software gauge 混在同一结论；随后才比较 tracker-only 与
  external information 在 null/near-null directions 上的互补性并做
  source/IOV robustness。

## 工程与文档

- `configs/gauge_constraint_external_constraint_feasibility_v1.yaml`
  （本条目核心，含全部继承 SHA256）
- `alignment/gauge_constraint_feasibility.py`（分析模块：tracker 信息
  回归、gauge 构造、KKT solver、closure、eligibility 表、判定）
- `scripts/report_gauge_constraint_external_constraint_feasibility.py`
  （validate-config / all 阶段）
- `tests/test_gauge_constraint_feasibility.py`（16 tests）
- `docs/gauge_constraint_external_constraint_feasibility.md` +
  `_cn.md`（双语同步）
- `outputs/gauge_constraint_external_constraint_feasibility_v1/`

所有 artifact 保存 config SHA、git SHA、输入 provenance、constraint
matrix G、parameter ordering、units、null/identifiable basis
provenance、solver method、condition metrics、gauge-invariant
metrics、external-evidence eligibility 与 unresolved assumptions。

## 结果（2026-09-05 分析运行后回填）

### 执行时间线

1. 预注册提交 `9ba17c7`（config + 模块 + 19 tests + workbook + 双语文档，
   全部 gate 在见结果前冻结）。
2. 首次 `--stage all` 运行：判定 `tracker_information_regression_failed`
   ——阴性对照在 identifiable basis 主角门上失败（观测 1.2074e-6° vs
   预注册 1e-6°）。
3. **根因调查（先取证、后修正）**：逐源 pair 数与冻结 artifact 完全一致
   （232/221/240/231/240/216/223，共 1603）；奇异值逐位相同（相对差
   精确为 0）；identifiable 与 null **projector Frobenius 距离精确为
   0.0**；每个冻结 mode 在重建 span 中的残差 ~1e-16；同进程重跑 SVD
   复现同一 1.2e-6°。结论：子空间逐位相同，差异是
   `arccos(1-ε)` 在 ε≈1 ulp 处的量化台阶（√(2ε) ≈ 2.1e-8 rad ≈
   1.2e-6°）——预注册门设在了所选度量的 float64 分辨极限之下，
   该度量**无法表示**"完全一致"。
4. **修正（amendment 提交 `ab608da`，透明记录、重跑前冻结）**：回归
   改用基无关的 projector Frobenius 距离（条目 68-69 冻结的
   `subspace_distance` 约定，全 float64 分辨），门 1e-8；主角仅作诊断。
   新增 3 个 amendment 回归测试（同子空间通过、子空间内旋转基通过、
   真实旋出子空间失败）。原始失败 artifact 保留于
   `/tmp/wb77_preregate_regression.json` 证据副本。
5. 重跑 `--stage all`：全部通过。

### Tracker 信息回归（阴性对照，通过）

- 重建 pooled 7D tracker 信息（条目 68 同 7 源 corpus，1603 pairs）：
  奇异值与冻结 artifact 逐位一致
  `[3729.82, 1747.30, 292.107, 170.813, 164.271, 28.1569, 2.51578]`；
  rank 5 / null 2 精确；identifiable/null projector Frobenius 距离
  **均为 0.0**（≤ 门 1e-8）。
- 诊断主角（arccos 量化，非 gate）：identifiable 1.2074e-6°、null 0.0°。

### Gauge 合同合法性（三类候选全部 valid）

| gauge | rank(G) | 剩余自由度 | det(G·S·V_null) | cond |
|---|---|---|---|---|
| `named_parameter_zero_gauge` (θ_dz=0, θ_C_dx=0) | 2 | 5 | -0.345 | 72.5 |
| `minimum_norm_scaled_gauge` (V_null^T S^{-1} θ=0) | 2 | 5 | 1.0 | 1.0 |
| `minimum_norm_native_gauge` ((S V_null)^T θ=0) | 2 | 5 | 540.9 | 1.16 |

三者均与 null 空间互补（每 gauge 轨道恰好相交一次），均不改变
tracker-observable prediction（closure 验证）。

### Gauge-invariant observable closure（全部 7 gate 通过）

- `|G û|max = 2.0e-12` ≤ 1e-9（约束精确满足）
- representative 唯一性：**0.0**（≤ 1e-8；同一 gauge 下 4 个
  observable-equivalent truth 给出逐位相同代表，无 null 漂移）
- 跨 gauge 可观测预测相对差 4.2e-15 ≤ 1e-8
- 跨 gauge 可辨识投影绝对差 3.1e-15 ≤ 1e-8
- held-out（pair 对半分，801/802）预测跨 gauge 相对差 2.4e-15 ≤ 1e-8
- KKT 最大条件数 2.0e8 ≤ 1e12，求解残差与满秩检查全部通过
- **预期内的 gauge 依赖性（report-only）**：null 坐标均值
  named=[0.0122, 6.7e-6]、min-norm-scaled=[~0, ~0]、
  min-norm-native=[-0.221, -0.0053]，跨 gauge spread 0.234——
  gauge-dependent absolute components 确实不同，永不作跨 gauge 比较；
  恢复完整注入 truth 未作为 gate。
- 诊断：被冻结 rank cut 丢弃的 near-null 信息最大相对占比 4.06%
  （报告，永不使用）。

### External-constraint eligibility table（15/15 全部不合格）

机械评估四门规则，并与冻结的条目 66/67 slot artifact 交叉核对一致。
无任一候选通过全部四门：

- `ift_C_dx`（+0.2541169 mm）：mapping 已验证、独立，但**无测量
  covariance、无有效 IOV** → `feasibility_only` 保持。
- `ift_l0_minus_l2_dx`（+0.5082339 mm）：与 C_dx 同 DoF 不独立，
  无 covariance/IOV → 不合格。
- `ift_station0_ry`：无值、survey→Stations ry 映射未解决 →
  `unavailable` 保持。
- Kabsch IFT ry（-7.7537 mrad）、dz-vs-x ry（-7.6956 mrad）、layer
  coherent tilt（-8.07 mrad）、support-beam tilt（0.350 mrad）、2021
  I/F normal tilt（5.00 mrad）：映射全部禁止/未协调，无 covariance。
- Nov-2022 station0 mean offset、population Sigma、2021 in-plane yaw、
  conditions station0_ry/planes_ry/C_dx_cond、interface 局部 x：
  或无 covariance、或非独立（conditions 是重建状态）、或无 provenance。

**没有任何现有外部测量有资格把 gauge/null directions 提升为物理约束。**

### 最终冻结判定

`gauge_feasibility_closed_no_eligible_external_physical_constraint`

- `gauge_constraint_contract_valid = true`
- `gauge_fixed_solver_well_posed = true`
- `gauge_invariant_observable_closure = true`
- `eligible_external_physical_constraints = []`
- `external_constraint_ingest_authorized = false`
- `no_ingestable_external_physical_constraint_available = true`
- `real_data_candidate_alignment_authorized = false`
- `geometry_write_allowed = false`
- `official_conditions_write_allowed = false`

### Artifact SHA256（outputs/gauge_constraint_external_constraint_feasibility_v1/）

- `config_validation.json` =
  `3864833dab091dfc370f1efa6d39267f4548582329a0d14c349166f8e8b54865`
- `tracker_information_regression.json` =
  `0a01d1b528878d1695d47c8a423633094263ada83009bd6a7dd49d6245dda5b8`
- `gauge_candidates.json` =
  `7ef11cf8e304dfffbc2ef600737561decfd94d4b3b6d27e61689184f7fe0161d`
- `gauge_invariant_closure.json` =
  `e77a18d99156f7173b4d50a45c0c24eef9176da831aa3721223eb8664651f0e1`
- `external_constraint_eligibility.json` =
  `dd0c00afa6e2f6dc56dc10dcaba5187fdd7a291a2bccb8f5c869546dee392020`
- `next_stage_decision.json` =
  `7b3d3af8cf03c6834882a0e6d773c46521f1fe1498bca09e5aad090049d886fd`
- `campaign_summary.json` =
  `c24660afc1acf5e8be85b14298ad692e314cbbbfd96747a47ad4c4e248a88eca`

### 关键提交

- `9ba17c7` 预注册（gate 冻结先于结果）
- `ab608da` 回归度量 amendment（含完整取证与 3 个回归测试）
- 本条目结果回填 + 全量回归 558 tests 通过后冻结提交

### 后续分支（均需独立预注册）

1. **Gauge-Fixed Real-Data Alignment Diagnostic V1**：calibration subset
   求 gauge-fixed candidate（称为 reconstruction gauge representative /
   candidate diagnostic），只在 held-out residual/DQ population 评价
   gauge-invariant observable improvement；official COOL/POOL write
   保持关闭。
2. 若未来出现真正 eligible 的 external physical measurement（真实
   covariance + 已验证映射 + IOV provenance + 独立性）：另开独立子
   战役，`y = H θ + ε` 形式，与 software gauge 结论严格分离。
