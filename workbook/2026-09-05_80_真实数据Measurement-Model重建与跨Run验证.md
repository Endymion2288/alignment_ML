# Workbook 80: Real-Data Measurement-Model Reconstruction & Cross-Run Validation V1

日期：2026-09-05
状态：**完成并冻结** —— 单一交互会话内按冻结顺序执行；所有验证 gate 在任何 confirmatory 计算前写入 config 并冻结，执行后未修改任何 gate（仅修复两处不影响 gate/科学的代码 bug）。

**最终决策：`measurement_model_multiple_components_not_validated`**

两个独立关键 gate **全部失败**：(A) residual covariance model 未通过 cross-fit 验证；(B) MC→real Jacobian-transfer model 的 real-data 支持不足。因此 `covariance_model_validated=false`、`jacobian_transfer_model_validated=false`，cross-run information 检查按预注册规则**结构性跳过**，`real_data_alignment_v2_preregistration_allowed=false`。真实数据继续 `residual_dq_monitoring_only`；**不开 Workbook 81**。

本战役只重建并验证 measurement model（covariance + Jacobian transfer），**不求最终 alignment correction、不打开 held-out、不写 geometry、不做 nonlinear iteration、不改变 gauge、不重新定义 V_id/V_null、不修改 S/rank_tolerance、不使用 external prior**。整个 Workbook 80：

- `geometry_write_allowed = false`
- `official_conditions_write_allowed = false`
- `real_data_candidate_alignment_authorized = false`
- `external_constraint_ingest_authorized = false`
- `held_out_accessed = false`（held-out 继续完全关闭，代码级 guard）

## 1. 科学问题（唯一）

WB79 已冻结 `real_data_statistical_model_multiple_failures`：frozen combined covariance 不能描述真实 residual 涨落（whitening χ²/ndof=1058.46、99.74% χ² 集中在最小 covariance eigenmode、condition 中位数 2.3×10¹²），且 (0,1) real-vs-MC Jacobian 支持只有 86.34%、14973/14974 informed rank 3/2、γ 方向夹角 88.16°。

**本战役问题：能否从第一性原理重建并验证一个可信的 measurement model——(A) residual covariance model，(B) MC→real Jacobian-transfer model——使得二者各自通过独立的、residual-blind 的 cross-run/MC-source-disjoint 验证？**

只有两者都通过且 cross-run information 稳定，才允许另开 Workbook 81（Gauge-Fixed Real-Data Alignment Diagnostic V2）。本战役**本身不产生任何 deployable candidate**。

### 1.1 禁止错误解释 WB79（逐条继承）

WB79 已核实：full 与 diagonal-marginal covariance 给出**逐位相同**的 alignment information matrix H、其 eigenvalues、β，且两者 `max|γ|` 都是 2.460。因此禁止写"off-diagonal covariance 是 γ=2.46 的唯一原因"或"把 covariance diagonalize 就修好了 alignment"。正确冻结解释：**当前 measurement likelihood 无效，同时 Jacobian transfer 和 cross-run information 也不充分；现阶段无法判断一个可信的 real-data alignment correction。** Workbook 80 的任务不是选择一个让 γ 更小的 W。

## 2. 起始状态审计（STEP 1，已完成）

- Workbook 79 freeze commit：`29362ec41861cdbdd63d079ea2642dd5f8a45f27`（预注册 `0257c9e` → 结果回填冻结 `29362ec`）。
- WB79 最终冻结：`real_data_statistical_model_multiple_failures`，三项 gate 全败。
- WB79 关键 artifact SHA256（本战役 `load_config` 逐项 SHA 验证，不一致即拒绝）：
  - config `8e89f9c449bf78c118b5be3b4d5f632694fd23f817a10cc590869ede1e6d75ea`
  - baseline_reproduction.json `8ae7d181c595150411ccb7518e271c535999e80217b6d00f2b885b7731625ab3`
  - covariance_audit.json `ea4ec27dcb7d8f83b24fa753fac08dccd348d02975191645528703a4960f5c5d`
  - transfer_support.json `b377c259fc2e86f6bbc9e9754729c7069db5f3df9b29708d5ec5dfe02d466483`
  - cross_run.json `a142b001e026671edf817fab7ccfec161e92699e63fb8f22f2c23f62e3b6f983`
  - model_adequacy_decision.json `4d43a4d4057574d213711e33397abfcf150adce4f322cc4f3f1cd458da3fe654`
- WB79 冻结 baseline（本战役 Stage 0 逐项复现，rtol/atol=1e-6）：n_pairs=378、χ²_total=1999613.163574398、whitened χ²/ndof=1058.4585330827601、whitened_ndof=1492、最小 eigenmode χ² 占比 0.9974148966076067、condition 中位数 2346694286586.7583。
- split SHA256 `c9c359793c41b59df1296e3ec2075af652c3f585746ca832a9c2e37f4a4f61d3`；calibration bank SHA256 `6e4eaae08bdc8b48174b2f24d1dc772f4cade7cb1918d59a8f017eb505687114`。
- WB78/WB79 完整测试套件 603 项在 WB79 freeze 时通过。

## 3. 冻结数据角色（不重新选 population）

- **Calibration（唯一允许使用的真实数据）**：run 14973 + run 14974，以及被冻结的 route/event/source identity、Frozen-V2 selected-route 合同与 split SHA。不得增加 run、删 event、重选 route、按 residual/covariance condition/model fit quality 删 pair、重切 calibration population。
- **Held-out（继续完全关闭）**：14975/14976 + 7 个 monitoring run（14971/14972/14980/14981/14985/14989/15007）+ 14977 report-only。Workbook 80 不得读取任何 held-out residual；`assert_no_held_out_access` 代码级 guard 对任何 held-out run id 直接 raise。
- Tracker information / gauge / identifiable basis / S / rank_tolerance 完全继承 WB78（经 WB79 链加载并 SHA 验证，不复制以防漂移）；frozen V_id/V_null 在 `load_config` 时注入，**绝不在真实数据上重新 SVD**。

## 4. 预注册设计

### 4.1 Stage 0 — 精确复现 WB79 baseline（硬停止）

用 WB79 同一确定性链重建 calibration bank + covariance eigensystem/whitening baseline，逐项对照 §2 的 WB79 冻结数值（rtol/atol=1e-6；bank SHA 精确匹配）。任何不一致 → `measurement_model_validation_inconclusive`，立即停止。

### 4.2 Stage 1 — covariance 生成语义完整追踪（`covariance_semantics_audit.json`）

在构造任何 replacement covariance 前，从代码与 reconstruction provenance 逐层追踪当前 combined 4×4 covariance 的产生方式，逐项回答预注册的 12 个问题（formula / 源码位置 / matrix ordering / units / assumption / resolved-or-unresolved verdict），并做经验表征（condition、最小特征向量方向、marginal pull、model vs empirical correlation）。**只读，不删 pair。**

### 4.3 Stage 2 — first-principles covariance hypotheses

区分"物理相关"与"错误相关模型"：长外推 x(z)≈x0+tx·Δz 自然产生 x↔tx 高相关，近奇异本身不是 bug；真正问题是**预测的 correlated uncertainty 是否与真实 residual 涨落一致**。预注册五条假设：H_cov_1（缺 process/noise term）、H_cov_2（source/target independence 假设错误、缺 cross-covariance）、H_cov_3（坐标/传播 Jacobian 或 covariance transformation 错误）、H_cov_4（track-fit covariance 只是 conditional fit uncertainty）、H_cov_5（未建模的 multiple-scattering/detector/association/model discrepancy）。

### 4.4 Stage 3 — candidate covariance model + cross-fit 验证（核心 gate）

候选模型**只能**来自第一性原理传播、明确的缺失-noise 模型、或 residual-independent 统计模型；**绝不**以 γ 更小 / rank 更稳 / condition 更好 / χ² 下降最多为设计依据。预注册四个候选：

- `C0_frozen`：frozen combined covariance（baseline，无 nuisance）。
- `C1_ms_leverarm`：`C = C_combined + θ²·G_MS(L)`，multiple-scattering 随机游走 process-noise 结构 `G_MS(L)` 在 (x,tx)、(y,ty) 上为 `[[L²/3, L/2],[L/2, 1]]`，L 为 source→target lever arm；θ² 为唯一 cross-fit nuisance。
- `C2_scaled_diagonal_floor`：`C = C_combined + s·diag(pair-type marginal floor)`。
- `C3_variance_inflation`：`C = s·C_combined`（对照：纯 inflation 不能修结构）。

**Cross-fitting（禁止同数据自证）**：Fold A 用 14973 推导 nuisance → 14974 验证；Fold B 反之。nuisance 用 Gaussian NLL 在**推导 run 的去均值 residual**上估计（确定性 log-grid + golden refine）。任何 empirical covariance 只用于 derivation/control，绝不作为同一 run 的 confirmatory evidence。

**预注册 confirmatory gate（对 validation run，scale 冻结）**：whitened χ²/ndof ≤ 4.0；经验 Cov(z) 特征值 ∈ [0.25, 4.0]；positive-definite；factorization 稳定（‖C·C⁻¹−I‖max ≤ 1e-6）。coherent residual mean 按预声明 run×pair-type cell diagnostic 去除（**去均值 ≠ 新 residual 定义**，不用于 correction）。`covariance_model_validated=true` 要求**至少一个物理候选（非 C0）在两个 fold 上都通过全部 gate**。

### 4.5 Stage 4 — 数值求逆审计（`numerical_inversion.json`）

对**同一 frozen covariance** 比较 direct `inv(C)` / Cholesky / eigendecomposition / SVD-pseudoinverse 的 χ²。严格区分：(A) 更稳定地计算同一个 C⁻¹（数值实现，允许）与 (B) 修改 C 本身（统计模型改变，需独立预注册，本战役**不做**）。最差 pair condition ~1.6e14、‖C·C⁻¹−I‖max ~5.7e-6 已知，故必须审计。

### 4.6 Stage 5 — MC-only conditional Jacobian model 开发（source-disjoint）

WB79 已确认 MC mean-J dispersion 合理（(0,1) 0.244、(0,2) 0.078、(0,3) 0.111），失败在 real support。本战役只允许 **residual-blind kinematic transfer**：`J = J(pair_type, pred_tx, pred_ty)`，特征与 alignment residual 无关、在看 real solve 前冻结、MC 可算、real 同定义。禁止 residual nearest-neighbour J、禁止用 alignment improvement 选特征、禁止用 held-out 选 binning。

**MC source-disjoint**：3 个 construction source（mc24_100043_00200_00299、mc24_100043_00600_00699、mc24_100044_00300_00399）上拟合，4 个 validation source（mc24_100047_00000_00049、mc24_100047_00150_00199、mc24_100048_00000_00049、mc24_100048_00150_00199）上验证，源文件级互斥。候选：`J0_station_pair_mean`、`J1_kinematic_binned`、`J2_linear_regression`。**预注册 MC gate**：per-pair Frobenius 相对误差、identifiable-direction injection recovery、null leakage 各 ≤ 0.35。

### 4.7 Stage 6 — real-data transfer-support 验证（residual-blind）

MC model 冻结后才应用到 14973/14974 kinematics（**不读 residual**）。real pair 的 (pred_tx, pred_ty) 落在 MC construction 99% Mahalanobis 包络（χ²₂(0.99)=9.21）内记为 within support。**预注册 gate：(0,1) 与 (0,2) 各 ≥90%**；(0,3) 仅 2 pair，report-only。out-of-support pair **绝不**外推进 solve；若仍有大量 out-of-support，冻结 `real_data_jacobian_transfer_model_not_validated`。

### 4.8 Stage 7 — cross-run information sanity check（仅 diagnostic，条件触发）

**只有 covariance + Jacobian 各自 validation 都通过后**才允许做：分别构造 14973、14974 的 `A_s = W_new^{1/2} J_new S` 与 residual-independent information geometry（rank、eigenvalues、subspace projector），再用 calibration residual 做 score direction consistency。预注册 gate 继承 WB79：rank 相等、dominant informed direction ≤15°、score/γ direction ≤15°。**即使一致也不生成 deployable candidate**（`alignment_authorized=false`）。若任一模型未通过，本阶段**结构性跳过**。

### 4.9 决策树（预注册，恰好一个终止字符串）

- baseline 复现失败 → `measurement_model_validation_inconclusive`
- covariance 失败且 Jacobian 失败 → `measurement_model_multiple_components_not_validated`
- 仅 covariance 失败 → `measurement_model_covariance_not_validated`
- 仅 Jacobian 失败 → `measurement_model_jacobian_transfer_not_validated`
- 两者都过但 cross-run 不稳 → `measurement_model_cross_run_information_not_stable`
- 三者都过 → `measurement_model_validated_real_data_alignment_v2_preregistration_allowed`（**只授权另开 Workbook 81**）

## 5. 工程规范

- 新增：`configs/real_data_measurement_model_reconstruction_validation_v1.yaml`、`alignment/real_data_measurement_model_validation.py`、`alignment/residual_covariance_model.py`、`alignment/conditional_jacobian_transfer.py`、`scripts/report_real_data_measurement_model_validation.py`、`tests/test_real_data_measurement_model_validation.py`、双语 docs、`outputs/real_data_measurement_model_reconstruction_validation_v1/`。
- 无需新 reconstruction：读取既有 real-data outputs + MC reference propagations + 线性代数，交互完成。
- Driver 阶段结构性强制 freeze ordering：`validate-config → reproduce → covariance-semantics → numerical-inversion → covariance-model → jacobian-model → jacobian-support → cross-run-info → decide`；cross-run-info 只在 covariance+Jacobian 都过时运行；decide 读取冻结 artifact。
- 所有 diagnostic 输出带 `diagnostic_only=true, alignment_authorized=false`；`held_out_accessed=false` 贯穿。
- 回归测试 31 项覆盖：WB79 baseline 精确复现、held-out access 硬失败、冻结 calibration population、no event dropping、covariance derivation/validation 分离、empirical covariance 不同-run 自证禁止、diagnostic covariance 不进 alignment solver、数值求逆与模型修改严格分离、J model MC source-disjoint、J feature residual-blind、real support 不读 residual、no external prior、no geometry/conditions write、no final candidate、deterministic cross-fit、决策树全部分支。

## 6. 结果（执行记录与冻结）

### 6.1 分阶段执行记录

| 阶段 | 结果 |
| --- | --- |
| `validate-config` | 通过。WB79 config + 5 artifact SHA 全部一致；全部冻结 flag 验证；frozen subspace（rank 5）注入。 |
| `reproduce`（Stage 0） | **通过**。WB79 covariance eigensystem/whitening baseline 逐项精确复现（6/6 checks）：bank SHA256 `6e4eaae0...`、χ²_total=1999613.163574398、whitened χ²/ndof=1058.4585330827601、最小 eigenmode 占比 0.99741、condition 中位数 2.347e12、n_pairs=378。 |
| `covariance-semantics`（Stage 1+2） | 完成（见 §6.2）。 |
| `numerical-inversion`（Stage 4） | 完成（见 §6.3）。 |
| `covariance-model`（Stage 3） | **失败**（见 §6.4）：无物理候选通过 cross-fit。 |
| `jacobian-model`（Stage 5） | MC source-disjoint **通过**（见 §6.5）。 |
| `jacobian-support`（Stage 6） | **失败**（见 §6.6）：real 支持不足。 |
| `cross-run-info`（Stage 7） | **结构性跳过**（covariance 与 Jacobian 均未过）。 |
| `decide` | `measurement_model_multiple_components_not_validated`。 |

### 6.2 Covariance 生成语义追踪（12 问 verdict）

- **residual_covariance_formula**：`C_combined = C_propagated_source + C_target`（plain sum），源码 `evaluation/field_propagation.py:238`。本战役 `load_calibration_pairs_enriched` 逐 pair 验证 `C_combined == C_prop + C_target`，max abs diff = **0.0**（逐位成立）。**resolved**。
- **source_tracklet_state_covariance_origin**：station-0 source tracklet 的局部 fit covariance（`tracklets` tree `cov_*`）。**resolved**。
- **propagated_state_covariance_transformation**：field-aware propagated source covariance，由**外部** Calypso/ACTS 外推（FaserActsExtrapolationTool）算出，从 `propagations` tree `pred_cov_*` 读入；本 repo 只读。经验上近奇异（median condition ~4.4e14，最小特征值 ~1e-10），是 source fit covariance 的确定性 pencil-beam 输运。**该外部工具是否加入 multiple-scattering process noise 从本 repo 无法确认 → unresolved_external_provenance**。
- **target_tracklet_covariance_origin**：station-j target tracklet 的局部 fit covariance（`tracklets` tree `cov_*`）。**resolved**。
- **source_target_independence_assumption**：`C_combined = C_prop + C_target` 即假设 source 与 target 独立（无 cross-covariance 项）。假设存在；其有效性存疑（H_cov_2）。**resolved (assumption present); validity questionable**。
- **shared_track_common_hits_common_fit**：source（station 0，2 层）与 target（station j，2 层）用**不同 hits**，结构性独立 fit；residual common-mode（如 multiple scattering）是 process-noise 问题，不是 shared-fit 问题。**resolved (structurally separate fits)**。
- **missing_cross_covariance_term**：确实存在遗漏（independence 假设），但符号无法解释大 χ²。**resolved (present omission, wrong sign to explain large chi2)**。
- **local_global_surface_jacobian_units_ordering**：4×4 顺序 [x_mm, y_mm, tx, ty]，单位 mm/无量纲，与本 repo `datasets/schema.py` COVARIANCE_FIELDS 一致。**resolved**。
- **x_tx_y_ty_correlation_physical_origin**：长外推 x≈x0+tx·L 的物理来源已识别；但**幅度被错误建模**（见 §6.4）。**resolved (physical origin identified; magnitude mis-modelled)**。
- **long_lever_arm_near_rank1_correlation**：是，长 lever-arm 外推自然产生近 rank-1 相关（(0,1) lever 1907.6mm、(0,2) 3097.6mm、(0,3) 4287.6mm；combined condition 中位数 1.0e12/1.2e13/6.5e13）。**resolved**。
- **inverse_method_direct_inv_vs_stable_factorization**：real-data solve 用 `np.linalg.inv`（`alignment/gauge_fixed_real_data_diagnostic.py:669`），而 MC bank 处理用 Cholesky（`alignment/physical_jacobian.py`）——不一致。**resolved (inconsistent; direct inv used in the real-data solve)**。
- **covariance_type_fit_vs_prediction_vs_mixed**：mixed（propagated prediction covariance + target fit covariance）。**resolved (mixed)**。

**经验表征**：marginal per-pair pull RMS（与 WB79 同定义）x=1.676、y=0.128、tx=0.423、ty=0.117——**marginal 大致校准**（x 略欠估，其余量级 1），问题不在 marginal 而在 off-diagonal 相关结构。最小 covariance 特征向量主要在 slope（tx,ty）子空间（(0,1)：|ty|=0.846、|tx|=0.295）。

### 6.3 数值求逆审计（Stage 4）

四种方法对**同一 frozen covariance** 的总 χ² 一致到 ~2.2e-7（相对）：direct_inv=1999613.6、cholesky=1999613.6、eigendecomposition=1999613.2、svd_pseudoinverse=1999613.2。**零 factorization 失败**。‖C·C⁻¹−I‖max：direct_inv=5.7e-6、cholesky=2.1e-5、eigh=6.4e-4、svd=6.4e-4。

**关键结论**：巨大 χ² **不是数值 artifact**——所有数值方法（包括稳定的 eigendecomposition/SVD）都给出同样的巨大 χ²。问题在统计模型本身，不在求逆数值。direct_inv 对总 χ² 而言与稳定方法一致到 2e-7。本阶段严格是 method-A（同一 C⁻¹ 的稳定计算）比较，**未修改 C 本身**（method-B 需独立预注册，本战役不做）。

### 6.4 Covariance model cross-fit 验证：**失败**

| 候选 | χ²/ndof 范围 | Cov(z) min 特征值 | Cov(z) max 特征值 | 两 fold 都过 |
| --- | --- | --- | --- | --- |
| C0_frozen | 812–1571 | 0.001–0.008 | 3211–6186 | 否（χ²、Cov(z) 全败） |
| C1_ms_leverarm | **0.64–1.09（过）** | **0.001–0.009（败 <0.25）** | 1.08–2.89（过） | **否** |
| C2_scaled_diagonal_floor | **0.65–0.93（过）** | **0.001–0.009（败 <0.25）** | 1.32–2.06（过） | **否** |
| C3_variance_inflation | 0.52–1.95（过） | 0.000（败） | 2.06–7.68（败 >4） | **否** |

- `covariance_model_validated=false`，`validated_physical_candidates=[]`。
- **决定性发现**：C1（MS process noise）与 C2（diagonal floor）都把 χ²/ndof 从 ~1000 降到 ~1（约 1000× 改善，物理上合理），且最大 Cov(z) 特征值健康（1–3）；**但最小 Cov(z) 特征值 ~0.001–0.009，远低于 gate 0.25**。这是**单向过约束方向**的签名：模型沿 pencil 方向（C_prop 的大特征值方向）过估方差 ~100×。
- **机制定位**：比较经验去均值 residual 协方差与 C1 模型的特征结构（以 (0,1) 为例）：经验特征值 [5e-5, 7.6e-4, 166.8, 3602.5]，C1 模型 [9.1e-3, 1.3e-2, 3.5e5, 4.7e5]。**C_prop 沿 pencil 方向（大特征值）过估真实 residual 方差 ~100–2000×**——即 propagated covariance 声称的 x↔tx 强相关（pencil beam）在真实 residual 中并不存在那么强。C1/C2 在 C_combined 之上**叠加**了同种 pencil 结构（MS 的 L/2 x-tx 耦合、或 diagonal floor），无法**去除**这个过相关，故 whiten 后仍留一个近零 Cov(z) 特征值。C3（纯 inflation）如预期无法修结构（两端都败）。
- 这精确指向 **H_cov_3 / H_cov_1 的组合**：问题不只是"缺 process noise"（C1 加了仍败），而是 **C_prop 的相关结构本身过相关**（pencil 方向方差过估）。外部传播工具是否已含 multiple-scattering process noise 从本 repo 不可确认（§6.2 unresolved_external_provenance）。
- C1 nuisance scale 的 cross-run 复现性（report-only）：|log(scale_A/scale_B)| (0,1)=0.297、(0,2)=0.785。
- 按规则：未把 diagonal/capped/unit covariance 提升为 official W；未用 empirical covariance 同-run 自证；未让任何候选进入 alignment solve。

### 6.5 MC-only conditional Jacobian model：**MC 验证通过**

3 construction source 拟合、4 validation source 验证（源文件级互斥，`source_disjoint=true`），特征 residual-blind（仅 pred_tx, pred_ty）。三个候选在两个 validation pair-type 上均通过全部 MC gate（frobenius/injection/null ≤ 0.35）：

| 模型 | Frobenius 相对 RMS | injection recovery 中位 | null leakage 中位 | MC 验证 |
| --- | --- | --- | --- | --- |
| J0_station_pair_mean | 0.097 | 0.053 | 0.012 | 过 |
| J1_kinematic_binned | 0.093 | 0.064 | 0.011 | 过 |
| J2_linear_regression | 0.094 | 0.049 | 0.011 | 过 |

per-pair Frobenius 相对中位误差：(0,1) 0.5–1.0%、(0,2) 0.5–1.6%、(0,3) 1.1–2.9%——conditional J model 在 MC 上是可靠摘要，且 source-disjoint 可移植。

### 6.6 Real-data transfer-support 验证：**失败**

冻结的 MC conditional-J model 应用到 14973/14974 kinematics（residual-blind）：

| station pair | real within MC 99% 包络 | gate | real ty p95 vs MC | real tx p95 vs MC | 判定 |
| --- | --- | --- | --- | --- | --- |
| (0,1) | **0.806** | ≥0.90 | 0.0262 vs 0.0096（~2.7×） | 0.0653 vs 0.0360 | **败** |
| (0,2) | **0.872** | ≥0.90 | 0.0133 vs 0.0102 | 0.0448 vs 0.0375 | **败** |
| (0,3) | 1.0（n_real=2） | report-only | — | — | 不 gate |

- `jacobian_transfer_model_validated=false`。真实 (0,1)/(0,2) pair 的 track-slope 支持显著宽于 MC（ty 尤其），~13–19% 的真实 pair 落在预注册 applicability domain 之外。out-of-support pair **未**外推进任何 solve。
- 与 WB79 一致（WB79 (0,1)=86.34%）；本战役用 conditional-J model 的 construction-source applicability domain，数值略不同（80.6%）但结论相同：real 支持不足。

### 6.7 Cross-run information sanity check：结构性跳过

因 covariance 与 Jacobian 模型均未通过各自 validation，按预注册规则本阶段**不运行**（`skipped=true`，理由：只有在 validated measurement model 下 cross-run information 检查才有意义）。未构造任何 information matrix，未生成任何 candidate。

### 6.8 决策与物理解读

- 冻结决策：`measurement_model_multiple_components_not_validated`（covariance 与 Jacobian-transfer 两个独立 gate 均失败）。
- **回答 §1 科学问题**：在当前 frozen 输入下，**无法**重建并验证一个可信的 measurement model。
  - **Covariance**：MS process-noise / diagonal-floor 能把 χ² 幅度修到 ~1，但都无法复现真实 residual 的相关结构（近零 Cov(z) 特征值）——根因是 **C_prop 的 pencil 方向过相关**（过估真实方差 ~100–2000×），叠加式修正无法去除。在当前 frozen combined-covariance 输入下，没有候选通过 cross-fit。
  - **Jacobian transfer**：conditional-J model 在 MC 上 source-disjoint 可移植，但真实 (0,1)/(0,2) 运动学支持不足（real ty 过宽），不能可信地转移到真实数据。
- `geometry_write_allowed=false`、`official_conditions_write_allowed=false`、`real_data_candidate_alignment_authorized=false`、`external_constraint_ingest_authorized=false`、`held_out_accessed=false` 全部保持。
- 未执行（按规则禁止）：打开 held-out、求最终 alignment correction、nonlinear iteration、改 gauge、重定义 V_id/V_null、调 S/rank_tolerance、用 external prior、把 diagonal/capped/unit covariance 提升为 official W、用 empirical covariance 同-run 自证、外推 out-of-support J、重开 tracker-only identifiability rescue、启动 nonlinear/trust-region response。
- 继续 `residual_dq_monitoring_only`。

### 6.9 后续路径（失败分支）

按 §4.9，本战役失败 → 继续 `residual_dq_monitoring_only`，**不开 Workbook 81**。有科学依据的 measurement-model 研究方向（需另开独立预注册 campaign）：

1. **C_prop 过相关的根源**：外部 Calypso/ACTS 传播工具（FaserActsExtrapolationTool）的 covariance transformation 与 multiple-scattering process-noise 处理需要直接从该工具 provenance 确认（本 repo 不可见）。若确认其 pencil 方向方差系统性过估，应在上游修正 propagated covariance，而非在本 repo 叠加修正。
2. **真实运动学支持**：real ty 分布比 MC 宽 ~2.7×，需要覆盖宽 ty 的真实拓扑的 MC/control 样本，才能把 conditional-J model 的 applicability domain 扩到真实支持。

在这些根源修复前，任何 alignment solve 都不可信。

### 6.10 Artifact 清单与 SHA256

| artifact | SHA256 |
| --- | --- |
| config (`configs/real_data_measurement_model_reconstruction_validation_v1.yaml`) | `8f0f0445741611e2f4ddbe6889eb107eaa18b9a5af56f655aaac120c44fa74d2` |
| baseline_reproduction.json | `3b3c10478fd1210f4ae4a7914fb2be06fd9646554939d72e9582e59fcee64cd0` |
| covariance_semantics_audit.json | `08470aa18f947281dbcaec6caaa2cd0424cf53dcec3ce116fca65755e06f18a8` |
| numerical_inversion.json | `13e0450b21929fc0194b7bc01efe03e2f6b315067afe3911a926dc339d7898bf` |
| covariance_model.json | `e48fd400d23141944668f741602b9bdfd3a0f9193cf088f59626266ceca48845` |
| jacobian_model.json | `c17beaacd7a22fccc7138fccac96f958347899904e832b44bedaae0271547f8c` |
| jacobian_support.json | `b0d27136bf37228db58857d4e20cfa5d698b0e94d6a7a525db863697084b9ab3` |
| cross_run_info.json | `8a6417d4266fbc13d11fb6e5f4a1b088edbaba50d2d1f94830a90a71d8647ae7` |
| measurement_model_decision.json | `c450a0b946961d1663f87bd0b4d5a26ccb8c5e7a21ab634aca8c415e05b52a90` |
| campaign_summary.json | `75c04b987506bd1d4c9bfe64178afe2a9dfd64eed1bd604961345c1afb597421` |

冻结提交见本节末尾 git 记录；完整测试套件（含 31 项 WB80 回归）通过。
