# Gauge-Fixed Real-Data Alignment Diagnostic V1（Workbook 78）

状态：**预注册**（设计在任何 real-data candidate 计算之前冻结；结果在执行后回填并单独冻结提交）。

## 科学问题

在不声称恢复唯一 mechanical station pose 的前提下，一个预先冻结的 reconstruction gauge representative 是否能够从 real-data calibration subset 得到稳定 candidate，并在完全 held-out 真实数据上改善 gauge-invariant tracker observables？

本阶段是 diagnostic，不是 geometry correction deployment。整个 Workbook 78：`geometry_write_allowed = false`、`official_conditions_write_allowed = false`、`real_data_candidate_alignment_authorized = false`、`external_constraint_ingest_authorized = false`。

## 冻结继承

- Workbook 77 freeze commit `9996765`；七项 WB77 artifact SHA256 全部验证一致；WB77 config SHA256 `d44876589d6bfed742714b957de80119024d7c93f747e45a1ed9357a5c0527e6`。WB77 以"无合格外部物理约束"关闭 gauge/external-constraint feasibility 战役。
- Tracker information：冻结 WB68 7D 参数序、severity scales S=(5,5,5,60,60,60,0.12)、`rank_tolerance=0.01`、冻结 identifiable/null basis；SHA + amended regression（基无关 projector Frobenius distance ≤ 1e-8；**不**恢复 arccos 量化的 principal-angle 门）。
- Real-data population：冻结 Operating Protocol V1 / Frozen-V2 selected-route 合同（12 个 2024 r0022 run）。residual-blind 角色表继承自 WB48–52 blind corpus：calibration = 14973+14974；held-out = 14975+14976（主）+ 7 个 expansion monitoring run + 14977（report-only）。不重切 split、不选 event、不重堆。

## 关键结构发现（预注册前的 MC-only 设计试点）

1. 冻结 selected routes 以 3-station (0,1,2) 为主：全部 12 run 的 anchor pair 组成 (0,1):1074 / (0,2):763 / (0,3):18。
2. 将冻结 MC bank 裁到真实 composition 后，即使使用精确 per-pair Jacobian，frozen 5D identifiable 空间中的信息矩阵在冻结 rank_tolerance 下也只有 rank 2：缺失的三维需要真实 population 中基本不存在的 (0,3) 杠杆。
3. 两个 informed 方向是 track-driven 的 {rx+dy, ry+dx} 类组合。
4. 唯一合法的 observable-model transfer（冻结 MC pooled bank 的 station-pair 均值 native Jacobian；真实数据 FD probe 继续禁止）在 MC half-split 试点中 informed-subspace 注入恢复偏差 8–27%；MC-control gate 据此加余量预注册为 0.35（在打开 held-out 前冻结）。
5. 真实 (0,1) 协方差 x 行/tx 行比 MC 大约 300×/800×——collision 数据 population 的冻结属性，由 MC-control 的 `real_scale_covariance` scenario 镜像。
6. Real-data bank builder（冻结 identity 文件 + 冻结 `evaluate_field_propagation` 链）逐位复现全部冻结 DQ 邻边 CSV 行；已固化为硬回归测试。

## 预注册设计

- **Gauge**：primary `minimum_norm_scaled_gauge`（继承 WB68/77 severity-scaled minimum-norm representative 与 `solver_restricted_to_identifiable_subspace` 合同——按合同选择，而非 condition number）；secondary control `named_parameter_zero_gauge`；report-only `minimum_norm_native_gauge`。看真实数据后不得切换 primary gauge。输出参数一律命名 *reconstruction gauge representative* / *gauge-fixed candidate diagnostic*，不得使用物理测量语言。
- **Transfer**：冻结 MC pooled bank 的 station-pair 均值 native Jacobian；模型误差由 MC control 在任何真实数据 solve 之前量化。
- **Solve**（one-shot，禁止迭代）：calibration residuals → transfer 模型 → 投影进冻结 identifiable basis → 冻结 tolerance 下的 real-information rank K → informed-subspace WLS → WB77 KKT 合同的 null-space 形式给出 gauge representatives → candidate artifact 冻结 + SHA → 才允许 held-out 评估。未-informed 的 identifiable modes 零更新并明确标注 `not_informed_by_real_data`（不是"测得为 0"）。
- **Applicability/linearity audit**：求解前 report-first（不选 event）；线性包络门 `max |γ_k| ≤ 0.15`（继承 WB33–35/68 注入设计）；超出即冻结 `real_data_linearized_model_out_of_support`。
- **MC control**（先于真实数据 solve）：两个协方差 scenario；注入恢复/泄漏门 0.35；200 次 null ensemble 产生 held-out gate A 的虚假改善地板；gauge-invariance closure 1e-8。
- **Held-out gates**：A 改善超过 null 地板；B run 级一致性（两个 primary run 各自改善；leave-one-run-out 保持为负）；C 每 cell 恶化不超过 2√(2n) χ² 涨落界 + 非 IFT 邻边精确零预测 null control；D gauge 间 held-out 预测一致（1e-8）；E bootstrap 方向稳定（15°，继承 WB68）；F 冻结 DQ slices 无新系统偏移（继承 entry-52 报警）；G 幅度在线性包络内。
- **决策树**（恰好一个终止字符串）：`gauge_fixed_real_data_candidate_diagnostic_pass` 或九个预注册失败字符串之一（`tracker_information_regression_failed`、`real_data_population_provenance_mismatch`、`real_data_observable_model_transfer_failed`、`real_data_diagnostic_inconclusive`、`real_data_identifiable_information_rank_zero`、`real_data_linearized_model_out_of_support`、`real_data_gauge_fixed_candidate_not_source_stable`、`real_data_gauge_invariance_failed`、`real_data_heldout_observable_not_improved`）。失败：按类型冻结，继续 `residual_dq_monitoring_only`，不得换 gauge/split/selection/S/rank，不得回到 identifiability rescue。

## 工程

- 新增：`configs/gauge_fixed_real_data_alignment_diagnostic_v1.yaml`、`alignment/gauge_fixed_real_data_diagnostic.py`、`scripts/report_gauge_fixed_real_data_alignment_diagnostic.py`、`tests/test_gauge_fixed_real_data_diagnostic.py`、双语 docs、`outputs/gauge_fixed_real_data_alignment_diagnostic_v1/`。
- 无需新 reconstruction：既有 real-data outputs + 线性代数，交互完成。
- Driver 阶段结构性强制 freeze ordering：`validate-config → freeze-split → mc-control → calibrate → evaluate → decide`；`calibrate` 要求 MC control 通过；`evaluate` 必须先验证 frozen candidate artifact 的 SHA 再读取任何 held-out residual。
- 20 项回归测试覆盖：primary gauge 冻结、split 确定性、held-out 冻结前不可访问、gauge 等价预测、无外部 prior、无 geometry/conditions write、candidate artifact 可复现、错误 split/provenance 硬失败、bank/CSV 逐位复现、MC-control 机制、决策树全部分支。

## 结果

（执行后回填）
