# Workbook 78: Gauge-Fixed Real-Data Alignment Diagnostic V1（预注册）

日期：2026-09-05
状态：**预注册（pre-registered）** —— 本条目在任何 real-data candidate 计算之前冻结设计；结果将在执行后回填并单独冻结提交。

## 1. 科学问题（唯一）

**在不声称恢复唯一 mechanical station pose 的前提下，一个预先冻结的 reconstruction gauge representative 是否能够从 real-data calibration subset 得到稳定 candidate，并在完全 held-out 真实数据上改善 gauge-invariant tracker observables？**

本阶段是 diagnostic，不是 geometry correction deployment。整个 Workbook 78：

- `geometry_write_allowed = false`
- `official_conditions_write_allowed = false`
- `real_data_candidate_alignment_authorized = false`
- `external_constraint_ingest_authorized = false`

## 2. 起始状态审计（STEP 1，已完成）

- Workbook 77 freeze commit：`9996765`（"Freeze workbook 77: gauge feasibility closed, no eligible external physical constraint"），前置 amendment `ab608da`、预注册 `9ba17c7`。
- WB77 七项 artifact SHA256 全部与冻结 workbook 表一致（本 session 开机时逐项重算验证）。
- WB77 config SHA256：`d44876589d6bfed742714b957de80119024d7c93f747e45a1ed9357a5c0527e6`。
- WB77 冻结结论：`gauge_feasibility_closed_no_eligible_external_physical_constraint`；`gauge_constraint_contract_valid = gauge_fixed_solver_well_posed = gauge_invariant_observable_closure = true`；`eligible_external_physical_constraints = []`。
- WB77 amendment 继承：subspace identity 回归使用基无关 projector Frobenius distance（`regression_max_projector_frobenius = 1e-8`），**不**恢复 `principal angle <= 1e-6°` 作为 exact-subspace gate。
- 完整测试套件 558 项在 WB77 freeze 时通过。

## 3. Real-data provenance inventory（STEP 2，只读，已完成）

冻结 population（Operating Protocol V1 / Frozen-V2 selected-route contract；Frozen-V2 checkpoint SHA256 `0c85a001...766a27`，threshold 0.001，unmatched penalty -1.0）：

| run | role（冻结） | selected routes | field edges |
| ---: | --- | ---: | ---: |
| 14973 | calibration | 121 | 204 |
| 14974 | calibration | 109 | 186 |
| 14975 | held_out | 156 | 253 |
| 14976 | held_out | 63 | 108 |
| 14977 | held_out_report_only | 2 | 4 |
| 14971/14972/14980/14981/14985/14989/15007 | held_out_monitoring | 233/193/114/77/74/91/32 | 392/316/187/123/129/167/51 |

- 角色映射来自 WB48–52 冻结的 residual-blind blind-corpus 分工（occupancy 窗口在 association 之前、不看 residual 冻结）。本战役**不新切 split**，直接继承该角色表；split SHA 在 `freeze-split` 阶段由 route metadata（run/event/source ID、station coverage、anchor-pair composition）冻结。
- 真实数据 residual 以 per-edge 4 维（x, y, tx, ty）+ 4×4 combined covariance 存在于冻结 CSV；identity 文件（`synthetic_tracklets.root` + `field_candidates.root`）包含全部 (source tracklet → target station) 传播记录。
- 禁止事项逐条继承：不按 residual/fit improvement 选 event、不新造 wide-angle cut、不重堆 2024 r0022、不生成 FD probe、不跑 Newton、不写 payload。

### 3.1 关键结构发现（MC-only 设计试点，预注册前）

1. **真实数据 selected routes 以 3-station (0,1,2) 为主**：全部 12 run 的 anchor pair 组成 (0,1):1074 / (0,2):763 / **(0,3):18**。calibration 侧 (0,1):227 / (0,2):156 / (0,3):2。
2. **结构 rank 上限**：把冻结 MC bank 裁到真实 composition（(0,1)+(0,2)，精确 per-pair J），frozen identifiable 5D 空间中的信息矩阵在 `rank_tolerance=0.01` 下 rank=2（eigvals 1.5e4, 2.0e4, 6.3e4, 2.3e6, 1.1e7）。缺失的 3 维需要 (0,3) 杠杆，真实数据中基本不存在。**即使 per-pair J 完美，真实 population 也只支持 2D informed subspace。**
3. **Informed 方向物理意义**：两个 informed 方向 ≈ {rx+dy, ry+dx} 组合——正是 track-driven isolation/cross-level 通道；dz/rz/C_dx 主导方向不在其中。
4. **Mean-J transfer 偏差**：station-pair 均值 Jacobian transfer（唯一合法路径：真实数据 FD probe 继续禁止）在 MC half-split 注入试验中 informed-subspace 恢复偏差 8–27%（0.10 scaled 幅度）；null 噪声地板 |γ|≈0.027。据此预注册 MC control gate：`injection_recovery_max_relative = 0.35`、`leakage_max_relative = 0.35`（pilot 观测 0.275 + 余量；MC 导出、在打开 held-out 前冻结）。
5. **真实协方差结构**：real (0,1) median diag cov = (4.2e5, 2.3e5, 0.60, 0.32)，MC = (1.3e3, 2.4e5, 7e-4, 0.33)——x/tx 行在真实数据上权重 ~300×/~800× 更低（collision muon 长外推的冻结属性，不是调参旋钮）。MC control 据此设两个 scenario（`mc_covariance` 与 `real_scale_covariance`）。
6. **Bank builder 正确性**：用冻结 identity 文件 + 冻结 `evaluate_field_propagation` 链（`require_truth_match=False, q_over_p_mode=0`）重建 14973 全部 204 条邻边 residual/covariance，与冻结 CSV **逐位一致（max diff = 0.0）**。该回归已固化为单元测试。

## 4. 预注册设计

### 4.1 Gauge 合同（看真实数据 fit 结果前冻结）

- Primary：`minimum_norm_scaled_gauge`（`V_null^T S^{-1} θ = 0`）。理由：直接继承 WB68/77 的 severity-scaled minimum-norm representative 与 `solver_restricted_to_identifiable_subspace` 合同，**不是**因为 WB77 中 condition number 最低。
- Secondary control：`named_parameter_zero_gauge`（θ_dz=0, θ_Cdx=0），仅用于证明 observable prediction 对合法 gauge 选择不敏感。
- Report-only：`minimum_norm_native_gauge`。
- 看真实数据后不得切换 primary gauge。
- 输出参数一律命名 `reconstruction gauge representative` / `gauge-fixed candidate diagnostic`；禁止 "measured station dx/..."、"true station position"、"physical C_dx = 0" 等物理测量语言。

### 4.2 Tracker information 合同

完全继承 WB77：7D 参数序、S=(5,5,5,60,60,60,0.12)、rank_tolerance=0.01、WB68 冻结 V_id/V_null/P_id；SHA + amended regression（projector Frobenius ≤ 1e-8）验证后才允许任何 solve；不得在真实数据上重新 SVD。

### 4.3 Observable model transfer

- 真实数据 per-pair Jacobian 需要 FD probe —— 继续禁止。唯一合法 transfer：冻结 MC pooled bank 的 station-pair 均值 native Jacobian（unweighted mean）。
- Transfer 模型误差由 MC control 量化（注入恢复 + null ensemble），在触碰真实 calibration residual 之前完成。

### 4.4 Calibration / held-out split

- 规则：冻结 operating-protocol 角色表（residual-blind，WB48–52 继承）。
- Calibration：14973+14974；held-out 主评估：14975+14976 + 7 个 expansion monitoring run + 14977（report-only）。
- `freeze-split` 记录：source/event IDs、route 数、anchor-pair composition、station coverage、split SHA；并硬性验证 MC control composition 与冻结 split 完全一致。
- Held-out 在 candidate 冻结前不得用于：选 gauge、调 damping、调 parameter subset、选 iteration count、调 event selection、定 correction 符号/步长。结构上由 driver 保证：`evaluate` 阶段必须验证 frozen candidate artifact SHA 后才读取 held-out residual。

### 4.5 One-shot diagnostic（禁止迭代）

calibration residuals → 冻结 tracker information（transfer 模型）→ 投影进冻结 identifiable basis → real information rank K（frozen 0.01 tolerance）→ informed-subspace WLS → primary gauge representative → **冻结 candidate artifact + SHA** → 才评估 held-out。

- 若 K=0：冻结 `real_data_identifiable_information_rank_zero`。
- 未-informed 的 identifiable modes：零更新，明确标注 `not_informed_by_real_data`（**不是**测得为 0）。
- 禁止 candidate → held-out → 修 candidate 循环；禁止 full-parameter unconstrained Newton。

### 4.6 Pre-fit applicability / linearity audit

求解前报告（不据此选 event）：weighted residual norm、per-observable 分布、per-run 一致性、W applicability（real vs MC 协方差）、non-finite/损坏检查、bank/CSV 逐位复现。
线性包络门（继承 WB33–35/68 注入设计）：`max |γ_k| ≤ 0.15`（scaled severity）。超出即冻结 `real_data_linearized_model_out_of_support`，不产生 correction、不打开 held-out。

### 4.7 Candidate solver

- 方法：`informed_subspace_wls_plus_gauge_null_space`（WB77 KKT 的等价 null-space 形式）。
- 必须满足：`G θ_hat ≈ 0`（≤1e-9，继承 WB77）、null/gauge 方向不漂移、solver finite、restricted condition ≤ 1e12（继承）、observable prediction finite。
- 三个 gauge 的 absolute θ 允许不同；只比较 `P_id θ_hat`、predicted observables、gauge-invariant 量。

### 4.8 Candidate artifact（先冻结再开 held-out）

保存：calibration provenance（source/route/event 计数、split SHA、input bank SHA）、gauge G 矩阵、V_id/V_null provenance（WB68 SHA）、参数序/单位、candidate θ（三个 gauge）、identifiable projection、informed basis 与信息特征值、predicted calibration observable change、solver condition/residual、bootstrap 报告、git/config SHA、`frozen_before_held_out_access: true`。

### 4.9 Held-out primary gates（预注册）

- **A**：held-out 加权残差范数改善超过 MC null ensemble 的 95 分位虚假改善地板（`mc_control` 阶段先行计算；Δχ² < 0 且 |Δχ²| > floor）。
- **B**：run 级一致性——14975 与 14976 各自 Δχ² < 0；对所有 sufficient run（≥30 anchor pairs）做 leave-one-run-out，剩余 Δχ² 均 < 0（改善不得由单一 run 驱动）。
- **C**：station-pair × observable cell 恶化不得超过 2√(2·n_cell)（χ² 涨落界，公式预注册）；null control：冻结 CSV 的 (1,2)/(2,3) 非 IFT 邻边预测修正必须精确为 0、Δχ² 精确为 0。
- **D**：三个合法 gauge 的 held-out predicted observable correction 一致（相对差 ≤ 1e-8，继承 WB77）。
- **E**：calibration event-level bootstrap（50 次，冻结种子）下 identifiable projection 方向稳定（最大夹角 ≤ 15°，继承 WB68 source-stability 门）且 rank 不变。
- **F**：DQ slices——held-out sufficient run 的 isolation 通道（y_mm=dy、ty=rx）median 不得比修正前更远离零；不得有新 run 越过冻结 entry-52 报警 |robust_z| > 5（参考尺度冻结：dy median −3.80966 mm / scale 6.71221 mm；rx median −0.001780 / scale 0.005244；n=390）。
- **G**：candidate 幅度在线性包络内（同 4.6）。

同时报告（非 gate）：Δχ²、per-source/per-station/per-observable 改善、slice 分析。禁止事后挑"改善的指标"当 primary。

### 4.10 MC control（先于真实数据 solve）

- 两个 scenario：`mc_covariance` 与 `real_scale_covariance`（x×300, tx×800）。
- 注入恢复：沿 informed 方向 0.10 scaled 注入，经精确 per-pair MC J 传播、用 transfer 模型求解；informed 投影恢复相对偏差 ≤ 0.35；uninformed 注入泄漏 ≤ 0.35。
- Null ensemble：200 次/ scenario，按冻结 composition 有放回抽样，θ=0，完整管线 → 虚假 held-out 改善分布 → q95 地板（两 scenario 取 max）。
- Gauge invariance：scenario 信息上三 gauge 预测一致 ≤ 1e-8。
- 任一失败 → 冻结 `real_data_observable_model_transfer_failed`，不触碰真实 calibration residual 的 solve。

### 4.11 决策树（预注册）

终止字符串恰好一个：

- `gauge_fixed_real_data_candidate_diagnostic_pass`
- `tracker_information_regression_failed`
- `real_data_population_provenance_mismatch`
- `real_data_observable_model_transfer_failed`
- `real_data_diagnostic_inconclusive`
- `real_data_identifiable_information_rank_zero`
- `real_data_linearized_model_out_of_support`
- `real_data_gauge_fixed_candidate_not_source_stable`
- `real_data_gauge_invariance_failed`
- `real_data_heldout_observable_not_improved`

通过：冻结 candidate、held-out metrics、全部 SHA；下一阶段可单独预注册 `Gauge-Fixed Temporary Reconstruction Candidate Validation V1`（仍禁止 official conditions write）。
失败：按类型冻结；不得换 gauge / 重切 split / 调 selection / 调 S/rank / 删 source / 回到 identifiability rescue；继续 `residual_dq_monitoring_only`。

### 4.12 External constraint branch 继续关闭

WB77 已机械审计 15/15 不合格。`eligible_external_physical_constraints = []`、`external_constraint_ingest_authorized = false`。Nov-2022 C_dx/L0-L2、Kabsch ry、dz-vs-x、layer/support-beam tilt、conditions values、population Sigma 一律不得用于选 gauge、regularize、Bayesian prior 或判断 candidate 对错。

## 5. 工程规范

- 新增：`configs/gauge_fixed_real_data_alignment_diagnostic_v1.yaml`、`alignment/gauge_fixed_real_data_diagnostic.py`、`scripts/report_gauge_fixed_real_data_alignment_diagnostic.py`、`tests/test_gauge_fixed_real_data_diagnostic.py`、双语 docs、`outputs/gauge_fixed_real_data_alignment_diagnostic_v1/`。
- 无需新 reconstruction：全部读取既有 real-data outputs + 线性代数，交互完成（bank 构建 ~1 min/source）。
- Driver 阶段顺序结构性强制 freeze ordering：`validate-config → freeze-split → mc-control → calibrate → evaluate → decide`；`calibrate` 要求 mc-control 通过；`evaluate` 要求 frozen candidate artifact 且 SHA 匹配。
- 回归测试 20 项覆盖：primary gauge 冻结、split 确定性、held-out 冻结前不可访问、gauge 等价预测、无外部 prior、无 geometry/conditions write、candidate artifact 可复现、错误 split/provenance 硬失败、bank/CSV 逐位复现、MC-control 机制、决策树全部分支。

## 6. 结果（执行后回填）

（待回填）
