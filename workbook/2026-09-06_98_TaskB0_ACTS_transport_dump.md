# Workbook 98: Task B0 C++ ACTS transport dump helper

日期：2026-09-06
状态：**完成** —— C++ helper 已物化 `C0/C1/Q_ACTS`；B2 closure 未过冻结 gate。未改 WB87/WB95/WB96/WB97 结论，未写 payload，未进入 alignment / Measurement Model V2，未新建 source campaign，未按 chi2 调 Q。

**最终判定：`FAIL`（情况 B）**

- `decision = acts_process_noise_contract_not_established`
- `mechanism = acts_process_noise_closure_failed`
- `failure_type = measurement_track_model_insufficient`
- `situation = acts_q_materialized_closure_failed`
- `process_noise_visible = true`（不是人为 scale）
- `contract_established_for_measurement_model_v2 = false`
- `measurement_model_v2_entered = false`
- `geometry_write_allowed = false`
- `unconstrained_tracker_only_stopped = true`

冻结保持：

- WB87：`faseracts_transport_covariance_not_validated` / `q_over_p_uncertainty_semantics`
- WB95：`physical_qoverp_semantics_not_established`
- WB96：`ckf_qoverp_covariance_export_established`
- WB97：`acts_process_noise_contract_not_established` / `acts_process_noise_not_materialized` / `acts_process_noise_configuration`

## 起始状态

HEAD（任务开始时）：`32c9044a8543845aaa0767d8089ba6fd731f31a9`（WB97 / Task B FAIL）

相对 WB97：仅新增本任务文件。未重跑 WB87–WB97，未改冻结结论。未覆盖 WB97 dumps。

输入：同一 WB87 construction / validation 9 源，WB96 CKF 5×5 作为 `C_in`，Calypso pin `40892527e9c65409afd2378a2abfc25ddbddac03`，Athena 24.0.41，ACTS 32.0.2。

## 验收标准

| 项 | 要求 | 结果 |
| --- | --- | --- |
| B0 dump | C++ helper 输出 Cin、F、C0、C1、Q、surface、state definition、四 hash | **通过**。独立 plugin，未改 Calypso 源码。 |
| process noise 可见 | `Q_ACTS != 0`，不是人为 scale | **通过**。construction 1471/1485、validation 1189/1195 可见。 |
| C0 Jacobian | 无材料：`C0 ≈ F Cin F^T` | 2-event smoke 相对 Frobenius ≈ `2×10^{-4}`。 |
| B2 Model 1 | construction + validation 过冻结 gate | **未通过**。χ²/ndof 均值、特征值、pencil 均出界。 |
| Highland | 仅诊断，不能替代 ACTS | 保持。未提升。 |
| 禁止项 | 不进 alignment / MM V2，不用 truth q/p，不调 Q | 保持。 |

Task B **整体仍未 PASS**。B0 只把 transport execution contract 从情况 1 推进到情况 B。

## Part B0 — C++ dump helper

Python / cppyy 构造 `BoundTrackParameters` / `PlaneSurface` 会打崩 Athena。本任务只在 `alignment_ML` 增加：

- `alignment/acts_transport_dump/CkfActsTransportDumpAlg.{h,cxx}`
- `scripts/build_ckf_acts_transport_dump.sh`（对着已有 Calypso build 编 `.so` + genconf）
- Athena job 只配置两个 `FaserActsExtrapolationTool`（MS/Eloss 关 / 开），不在 Python 里造 Acts 对象

转换路径复制 NtupleDumper 的 Trk→Acts 数值 Jacobian：**不** suppress q/p，**不** 用 truth 覆盖。导出态 `(x, y, tx, ty, q/p)`，`q/p` 带符号、单位 `1/MeV`。

`Q_ACTS := C1 − C0`。`F` 由无材料 propagate 的中心差分得到，不是从 `C0` 反推。

2-event smoke（login，同一 100043 xAOD）：6 行全部 `model0/model1` 成功，`Q ≠ 0`。

HTCondor cluster `1109765`（bigbird24）：9/9 `ExitCode=0`。未覆盖 WB97 产物。

## Part B2 — Process noise closure

预注册模型不变。Gate 仍是冻结 WB81/WB87 集：χ²/ndof ≤ 4，特征值与 pencil ∈ [0.25, 4]。

Model 1（唯一通过模型）：

| split | pair | n | χ²/ndof | median χ²/ndof | pencil ratio | principal |
| --- | --- | --- | --- | --- | --- | --- |
| construction | (0,1) | 445 | 29.5 | 0.46 | 6.82 | **x** |
| construction | (0,2) | 438 | 12.4 | 0.29 | 11.3 | **x** |
| construction | (0,3) | 426 | 93.8 | 0.37 | 7.04 | **x** |
| validation | (0,1) | 349 | 38.7 | 0.055 | 7.64 | **x** |
| validation | (0,2) | 348 | 6.52 | 0.063 | 6.95 | **x** |
| validation | (0,3) | 346 | 10.6 | 0.063 | 4.65 | **x** |

一致性报告（预注册，未事后改 bin）：

- low / medium / high `|q/p|`：median χ² 大多 < 1，但 low-momentum 有极端 outlier（max 可达 10³–10⁴）。均值 χ² 被 outlier 拉高。
- 主方向是 **x**，不是 y/ty。pencil ratio 全部 > 4。
- process noise 贡献：construction 中位 `‖Q‖/‖C0‖ ≈ 0.40`；validation 中位 ≈ 0.015。不是人为 scale。

Model 0 同样不过 gate。Highland 与 Model 0 几乎重合，不能替代 ACTS。

dummy SegmentFit / truth q/p：0 条。

## 失败分类

**情况 B：Q 已物化，但 closure fail。**

这不再是 `acts_process_noise_configuration`。缺口从 “Python 调不到 ACTS propagate” 变成协方差物理：

- material / scattering 是否足够
- 状态转换 / surface 语义
- 均值 χ² 被 outlier 主导，而中位 χ² 过小（协方差往往偏大）
- 主残差在 x，不在 y/ty

此时才允许讨论 Measurement Model V2 是否需要修改。**本 workbook 没有进入 V2。**

## 工程记录

起始 HEAD：`32c9044a8543845aaa0767d8089ba6fd731f31a9`

Diff（相对 WB97，未提交）：

- `alignment/acts_transport_dump/`
- `configs/acts_transport_dump_v1.yaml`
- `datasets/acts_transport_dump.py`
- `scripts/build_ckf_acts_transport_dump.sh`
- `scripts/dump_ckf_acts_transport_covariance.py`
- `scripts/submit_acts_transport_dump_condor.py`
- `scripts/run_acts_transport_dump_condor.sh`
- `scripts/audit_acts_transport_dump.py`
- `tests/test_acts_transport_dump.py`
- `docs/acts_transport_dump.md`、`docs/acts_transport_dump_cn.md`
- `.gitignore` 增加 `build/`

Config SHA：`1969cde56bd59cb53700420de93dc4d3313b5649b667c4d5cd799ba5855a2f9c`

Run ID：`sbb0_acts_transport_dump_20260906T172303Z_9825e122`

Condor：`1109765`（9 jobs，全部成功）

Plugin SHA：`bbcc8794e8c82a4f87f0d0f4e6b2573d667c15032222943a2973a6c94b908909`

产物（EOS，不入 git）：`outputs/acts_transport_dump_v1/sbb0_acts_transport_dump_20260906T172303Z_9825e122/`

| 产物 | SHA256 |
| --- | --- |
| `acts_transport_dump_configuration.json` | `90202041503567419697e04dc53a4dbaa929d29bdbbb228c1b3d22822a41cd42` |
| `acts_transport_dump_contract.json` | `186218ecf205423bbce7676f71acb1d4637c0a3fb1c81fd082ca68a4bd9d7aa0` |
| `source_inventory.json` | `28f6c99091f7b66eb1ba7d93c6b7d18b55cd1ab54d4ac60eff51665f97786f6b` |
| `inherited_stage.json` | `ea5d29e8831e3ae3ab605717306f4251aa9bf34bbc1488992173edb9ee54018f` |
| `COMPLETE.json` | `acbdd45e8e92cb30bd020a98e3823a06518e46ed287670d3c6de780bf80ddeed` |

geometry_hash：`4e965f3631dd4ff6e04b141ac36efc49603a1dfc0625de5ea7558dd929486b15`  
field_hash：`60432de5864cfba361f91f47a694dc111bbb06dcec44434e9682548680460e9c`  
conditions_hash：`d30733216d6eb392dbdbcf5a252fed01ff56a045ade59731ca8419febce7a63d`  
material_map_hash：与 WB97 同一材料图派生

测试：`tests/test_acts_transport_dump.py` + 既有 WB97 测试，13 passed。

## 下一步

情况 B。可以做物理诊断（material / scattering / 状态转换 / outlier 语义），并讨论 Measurement Model V2 **是否需要改**。

仍然禁止：进入 alignment、训练 ML、调 Q、调 covariance、用 truth q/p、用 dummy SegmentFit、删 q/p、写 `/Tracker/Align`。
