# Workbook 97: Task B ACTS MaterialInteractor / process-noise 契约

日期：2026-09-06
状态：**完成** —— B1 配置契约已建立；B2 closure 未物化。未改 WB87/WB95/WB96，未写 payload，未进入 alignment / Measurement Model V2，未新建 source campaign，未按 chi2 调 Q。

**最终判定：`FAIL`**

- `decision = acts_process_noise_contract_not_established`
- `mechanism = acts_process_noise_not_materialized`
- `failure_type = acts_process_noise_configuration`（情况 1：先修 transport contract，不是情况 2）
- `contract_established_for_measurement_model_v2 = false`
- `measurement_model_v2_entered = false`
- `geometry_write_allowed = false`
- `unconstrained_tracker_only_stopped = true`

冻结保持：

- WB87：`faseracts_transport_covariance_not_validated` / `q_over_p_uncertainty_semantics`
- WB95：`physical_qoverp_semantics_not_established`
- WB96：`ckf_qoverp_covariance_export_established`

## 起始状态

HEAD（任务开始时）：`df863df0278177f6eafbf6886707ee39b0b83c83`（WB96 / Task A2 PASS）

相对 WB96：仅新增本任务文件。未重跑 WB87–WB96，未改冻结结论。

输入：同一 WB87 construction / validation 9 源，WB96 CKF 5×5 作为 `C_in`，Calypso pin `40892527e9c65409afd2378a2abfc25ddbddac03`，Athena 24.0.41，ACTS 32.0.2。

## 验收标准

| 项 | 要求 | 结果 |
| --- | --- | --- |
| B1 契约 | ACTS version、geometry/material/field/conditions hash、transport options、process-noise flags 全部明确 | **通过**。`acts_process_noise_configuration.json` |
| C_prop 来源 | 不是 dummy、不是经验 scale、不是 tuning | 生产 transport **仍无 Q**。未发明 Q。 |
| B2 Model 1 | `C1 = F Cin F^T + Q_ACTS` 在冻结 gate 上 construction/validation 一致通过 | **未物化**。没有可用 C0/C1。 |
| Gate | χ²/ndof ≤ 4，特征值与 pencil ∈ [0.25, 4] | 未评估。没有数据就不调 gate。 |
| Highland | 仅诊断，不能替代 ACTS | 保持。未提升。 |

## Part B1 — 配置审计

`contract_clear = true`。生产外推：

- `MaterialInteractor` 在 ActionList 中
- `InteractionMultiScatering/Eloss/Record` **默认 false**
- `NtupleDumperConfig` **不覆盖**这些开关
- 因此生产 `C_prop` **不写入 process noise**

对照：

- CKF2 fitter：`makeTrackFitterFunction(..., true, true, ...)`，拟合内 MS/Eloss **开**
- 外推工具：开关 **关**
- 材料图存在：`material-maps-alma9.json`，file SHA `0dc70ded…`
- 2-event smoke 已加载该材料图（`Configured to use material input: ...alma9.json`）

这只证明机制存在且生产关掉。这不是 closure。

## Part B2 — Process noise closure

预注册模型：

| 模型 | 公式 | 角色 |
| --- | --- | --- |
| 0 | `C0 = F Cin F^T` | 无 Q |
| 1 | `C1 = F Cin F^T + Q_ACTS` | **唯一通过模型** |
| 2 | `C2 = C0 + Q_Highland`（`x/X0=0.05`/gap，不调） | 诊断 |

`C_in` 必须是 WB96 CKF 5×5。禁止 truth q/p、dummy SegmentFit、rescale、按结果改 Q。

Athena smoke（2 event，同一 100043 xAOD）读出了 CKF 5×5，并创建了两个 `FaserActsExtrapolationTool`（flags 关 / 开）。但是：

- PyAthena **没有** `import Acts`
- cppyy 构造 `PlaneSurface` / `BoundTrackParameters` 会把 Athena 进程打崩
- 因此 `tool.propagate(CKF 5×5 → station plane)` **没有安全的 Python 调用路径**

未提交 9 源 HTCondor 生产作业：smoke 已证明当前 dump 不能物化 C1。

## 失败分类

**情况 1：transport contract 尚未可执行。**

不是情况 2（ACTS Q 正确但 closure 失败）。还没有 Q 可判。

下一步必须是：在 **alignment_ML 内**加 C++ dump helper（不改 Calypso `FaserActsExtrapolationTool` 源码），对同一 9 个已有 xAOD 物化 C0/C1，再重跑 B2。长任务走 HTCondor。

现在 **不进入**：

- Measurement Model V2
- field-aware likelihood
- alignment
- tracker-only / rank rescue / ML estimator

也还不进入「real-data residual/DQ + 可复现负结果」终局——那是情况 2 的出口。

## 禁止项（均保持）

不用 truth q/p；不用 dummy SegmentFit；不 rescale；不调 chi2；不调 Q；不删状态变量；不写 `/Tracker/Align`；不打开 sealed test。

## 工程产物

- 配置：`configs/acts_process_noise_contract_v1.yaml`
- 模块：`datasets/acts_process_noise_contract.py`
- Athena dump：`scripts/dump_ckf_acts_process_noise.py`
- Condor：`scripts/submit_ckf_acts_process_noise_condor.py`、`scripts/run_ckf_acts_process_noise_condor.sh`
- 审计：`scripts/audit_acts_process_noise_contract.py`
- 测试：`tests/test_acts_process_noise_contract.py`（7 个测试）
- 文档：`docs/acts_process_noise_contract.md`、`docs/acts_process_noise_contract_cn.md`

起始 HEAD：`df863df0278177f6eafbf6886707ee39b0b83c83`

Config SHA：`765dfacaa086610eefe0b2c64cbf38325e28e4b52f755b3e3e43c173bb657baa`

Run ID：`sbb_acts_process_noise_20260906T161525Z_54a1cc8d`

产物（EOS，不入 git）：`outputs/acts_process_noise_contract_v1/sbb_acts_process_noise_20260906T161525Z_54a1cc8d/`

| 产物 | SHA256 |
| --- | --- |
| `acts_process_noise_configuration.json` | `1a72166b1063ddcb609c1cea4ae35b8bb37a0be2af95bc217a8e6de1b33b2c50` |
| `acts_process_noise_contract.json` | `5f5c900cad6c7f4484fa588cf016a176622003ef7d3c631bc4f0ca2bf4de6637` |
| `source_inventory.json` | `8740aeae063b378a135808d3d5019d1a34a87ecfa2780f4202a7c9758f8f3360` |
| `inherited_stage.json` | `58fc58cc678c1a3b3cd3d64e30f9e497de65d975184ece8da24c7c61278fd3e8` |
| `COMPLETE.json` | `0b33f1c17c5e0fe51c10bd1aea4e4eb709c9b52c4c4892d5d5c1d71b35e48169` |

geometry_hash：`4e965f3631dd4ff6e04b141ac36efc49603a1dfc0625de5ea7558dd929486b15`  
field_hash：`60432de5864cfba361f91f47a694dc111bbb06dcec44434e9682548680460e9c`  
conditions_hash：`d30733216d6eb392dbdbcf5a252fed01ff56a045ade59731ca8419febce7a63d`  
material_map_hash：`0df37bd621e6caac2e223fda12fec32a6f8c49685c7b831181b956e59996433c`
