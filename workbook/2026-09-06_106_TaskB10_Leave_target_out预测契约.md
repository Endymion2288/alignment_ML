# Workbook 106: Task B10 Leave-Target-Out 预测契约

日期：2026-09-06
状态：**完成** —— 在 WB105 `mixed_or_inconclusive` 之后，审计能否建立排除目标站 measurement 的独立预测状态。未重开 raw CKF，未收紧 WB103 契约，未编造经验 `Cov(pred,target)`，未把 `KalmanFitterTool.fit` 当成 LTO，未删 100043/37，未调 C/Q，未进 Measurement Model V2 / alignment / ML。未改 WB96–WB105 结论。

**最终判定：`NOT_ESTABLISHED`**

- `decision = leave_target_out_prediction_contract_not_established`
- `primary_case = leave_target_out_prediction_contract_not_established`
- `secondary_cases = calypso_has_cluster_unbiased_residual_only, official_fit_is_leave_source_ift_out_not_leave_target_out`
- `next_step = independent_leave_target_out_helper_not_kalmanfittertool_fit`
- `independence_proven = false`
- `lto_states_materialized = false`
- `measurement_model_v2_authorized = false`

冻结保持：WB96 PASS；WB98–WB105 结论不变。WB105 仍是 `mixed_or_inconclusive`。

## 起始状态

HEAD：`32c9044a8543845aaa0767d8089ba6fd731f31a9`

相对 WB105：只新增本任务与并行 B11 / 后续 B12 文件。未覆盖 WB98 dumps。

## 验收标准

| 项 | 要求 | 结果 |
| --- | --- | --- |
| 接口目录 | 确认 Calypso 是否已有 LTO / unbiased 导出 | **完成**。只有逐 cluster unbiased residual，没有整站源面 5×5。 |
| 独立状态 | 排除目标站后的 track state + 5×5 | **未物化**。官方 dump 无 LTO 字段。 |
| 独立性证明 | 目标站未参与该预测状态 | **失败**。官方 pair 目标站在同一 CKF fit 的比例 1.0。 |
| 禁止补偿 | 不编造 `Cov(pred,target)` | 保持。 |

## Calypso 接口（冻结 SHA `40892527e…`）

`KalmanFitterTool.fit` 使用全部 `measurementsOnTrack`，
`cluster_z = -1e6`，把 `z < -100 mm` 的 IFT 标成 outlier。
这是 leave-source-out，不是 leave-target-out。
种子协方差对角 ×100 再 ×10。

`getUnbiasedResidual(cluster_z)` 只在 `cluster_z < -10000` 时
特殊排除。目标站 z 不会被排除。IFT cluster 列表重载会追加
cluster，然后仍加入全部 hit。

不修改冻结 Calypso。不把上述接口包装成已建立的 LTO 契约。

## 样本

1989 契约行 / 1974 official pair。`100043/37` 保留。
资格与 WB103 相同。

## 机制判定

官方 Cin 不是 leave-target-out 预测。目标站 measurement
参与同一全局 CKF / KF refit。没有可传播的独立 LTO 状态。
因此契约未建立，不得用经验交叉协方差补偿。

下一步：若要物化 LTO，必须在 `alignment_ML` 写**独立**
helper，且**不能**原样调用 `KalmanFitterTool.fit`。
长任务走 HTCondor。本任务未声称 helper 已建立契约。

## 工程记录

起始 HEAD：`32c9044a8543845aaa0767d8089ba6fd731f31a9`

Diff：`configs/leave_target_out_prediction_contract_v1.yaml`、
`datasets/leave_target_out_prediction_contract.py`、
`scripts/audit_leave_target_out_prediction_contract.py`、
`tests/test_leave_target_out_prediction_contract.py`、
本 docs / workbook。

Config SHA：`54d6c38fdfeb0897e01385ab2259e2f8fb36bc3e580083620060ec0b8cd66f3b`

WB105 decision SHA：`e29fd93b0dbc9f435934184b93ce550d7dc2e3236177ea47502129c6ec2890f2`  
WB104 decision SHA：`8e620b50a3b5509d39e66e6dd7e1fd7a1a7aed5f53fdf63f53ec2456da266031`  
WB103 contract SHA：`e8c1f927e1ba03d418c2ed9b03084b198d09ba15196297a1a13cb5044ec48e3d`

Run ID：`sbb10_leave_target_out_20260906T211854Z_951d7052`

| 产物 | SHA256 |
| --- | --- |
| `leave_target_out_dependency_contract.json` | `d6b878a3faa782f871f1ffee4577d1b6eeb8678d2c376ce517a01a9d98c10d82` |
| `leave_target_out_state_inventory.json` | `20d4e020cd7f699515306deb4d0f6aa4b019663fce390fcf6c9af00e32933b3c` |
| `leave_target_out_covariance_contract.json` | `d825da05c8a126b4b88f36b2d8b11e90ab67b6d1f98c18a2383a4d9fe8acea00` |
| `target_independence_audit.json` | `2a8306a5c237dcf0247412c8e255226491ceeee17f7b4f662a460a73821cc139` |
| `leave_target_out_prediction_decision.json` | `f8e2140f0c7f057ba8b8c616695bf003b7681625ee5daf75d1b0af23b0fedf67` |
| `inherited_stage.json` | `29e82acd8c95a4d15478beab0691f59ec33e7cd85a3c4aff930a381f9f7648e6` |
| `COMPLETE.json` | `570b5d7a562e49dc08fe7a0de73f7c6613e6d1bc12e959b78e4c7422495e7e1f` |

Provenance 与 WB98 冻结一致。测试 4 passed。只读已有 dump / ntuple / Calypso 源码，login 短跑，未开 HTCondor。

## 下一步

契约未建立。并行 B11 解释 Cin 语义；B12 预注册四臂但 C/D 封锁。
不得进 V2。
