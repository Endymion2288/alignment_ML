# Workbook 109: Task B13 Independent Leave-Target-Out State 物化

日期：2026-09-06
状态：**完成 / PASS** —— cluster `1109785` 的 9 个冻结 source LTO dump 已闭环；全样本 B13 正式验收通过。未进 B15，未进 Measurement Model V2。未改 WB96–WB108 结论。未宣称 transport covariance 已验证。

**当前正式判定：`PASS`**

- `decision = leave_target_out_state_materialization_established`
- `independence_proven = true`
- `lto_states_materialized = true`
- `b14_authorized = true`
- `b15_authorized = false`
- `measurement_model_v2_authorized = false`
- `transport_covariance_validated = false`

## 起始状态

HEAD：`32c9044a8543845aaa0767d8089ba6fd731f31a9`

相对 WB108：新增独立 LTO helper / Condor dump / B13 契约审计。没有重跑 A/B full-track 臂，没有 material / Q / rescale / reconstruction 横向搜索。

继承（以正式 decision JSON 为准）：

| Workbook | SHA256 | 冻结 token |
| --- | --- | --- |
| WB103 | `e8c1f927e1ba03d418c2ed9b03084b198d09ba15196297a1a13cb5044ec48e3d` | `eligible_for_transport_validation` |
| WB105 | `e29fd93b0dbc9f435934184b93ce550d7dc2e3236177ea47502129c6ec2890f2` | `mixed_or_inconclusive` |
| WB106 | `f8e2140f0c7f057ba8b8c616695bf003b7681625ee5daf75d1b0af23b0fedf67` | `leave_target_out_prediction_contract_not_established` |
| WB107 | `d65200270fab5a3c796b03c2b5c67b3280ad6afdea569518650d90ddee3aade6` | `official_cin_is_global_kf_refit_front_state` |
| WB108 | `666859d4a032a93c80f50b36dab8f67267e246b138efe333dc2aa9bb58445deb` | `leave_target_out_arms_unavailable` |

## Helper

不得包装 `KalmanFitterTool.fit`。独立 C++ Athena 算法
`CkfLeaveTargetOutDumpAlg`：

- 读取同一冻结 xAOD 的 `CKFTrackCollection`（无新 source campaign）
- 从 `trackStateOnSurfaces()` 收集 measurement **和** outlier TSOS 上的 `FaserSCT_ClusterOnTrack`
- 原因：官方 KF 把 IFT 标成 outlier，`measurementsOnTrack()` 不含 station 0
- 对每个 target ∈ {1,2,3} 显式去掉该站全部 hits，保留 IFT / 其余站
- 内联 ACTS KalmanFitter：无 outlier finder，MS/Eloss ON
- seed **均值**可来自官方 `front()` 运动学；seed **协方差**是无信息对角，不用官方 Cin
- 不用 truth q/p

2-event 冒烟（`mc24_100043_00400_00499`，event 0/1）：

| 检查 | 结果 |
| --- | --- |
| 行数 | 6（2 event × 3 target） |
| `target_station_measurements_used = 0` | 6/6 |
| `kalman_fitter_tool_fit_called` | false |
| IFT recovered | `all = [6,6,6,6]`，`n_outlier_hits_recovered = 6` |
| source used | 6 |
| 成功拟合 | 6/6，source plane z = −1860.15 mm，5×5 + signed q/p |
| event 0 q/p 跨 target | −8.7e-7 / −7.2e-7 / −6.9e-7（同号、同量级） |

冒烟不是全样本验收。`100043/37` 不在前 2 个 event 里，全样本 dump 必须覆盖。

## Denominator

WB103 冻结范围保持：

- raw 2680
- ineligible 691
- contracted 1989
- official pairs 1974
- `100043/37` 保留

冻结范围在正式 audit 中保持不变：raw 2680 / ineligible 691 / contracted 1989 / official pairs 1974。`100043/37` 保留。

## HTCondor cluster `1109785`

9/9 Normal termination（return value 0）。未重提成功 shard，未覆盖已有 dump，未改 config / helper / source 输入。stderr 空。

| Source | 行 | 事件 | 成功/失败 | dump SHA256 |
| --- | --- | ---: | --- | --- |
| mc24_100043_00200_00299 | 294 | 98 | 280/14 | `b85626b2…139d21c` |
| mc24_100043_00300_00399 | 297 | 99 | 278/19 | `149bb27a…fdbcedbb38` |
| mc24_100043_00400_00499 | 300 | 99 | 292/8 | `55e13b6a…e34215cfc8` |
| mc24_100044_00200_00299 | 297 | 99 | 281/16 | `020ad2c0…bebf59f` |
| mc24_100044_00300_00399 | 297 | 99 | 287/10 | `6bd7499d…c3517b8` |
| mc24_100047_00000_00049 | 306 | 99 | 303/3 | `c95e45e1…7d4aa685` |
| mc24_100047_00050_00099 | 300 | 100 | 281/19 | `628b1a6d…331cdab38b0b1` |
| mc24_100048_00000_00049 | 297 | 99 | 289/8 | `0cb1acf3…4b4daa51ac7b1` |
| mc24_100048_00050_00099 | 297 | 98 | 286/11 | `2e38f9bc…e50963662b` |

dump-wide：2685 行，2577 成功，108 失败，失败原因全部是 `too_few_remaining_measurements`。这些失败都在 WB103 ineligible 集合里，**不**从 contracted denominator 删除。

每一行（含失败行）`target_station_measurements_used = 0`。`kalman_fitter_tool_fit_called = 0`。helper = `CkfLeaveTargetOutDumpAlg`。seed 无信息；官方 Cin 未作 prior。

## 工程记录

Config SHA：`6c4393ee8202a4b641b25599fd0b48eeba08b0930996217a25bda5cbc16cd0d7`

Helper SHA：`46f30d1a8846e11c218cd88ba7dbdbf5ab9d7ec3503af7a533650fe017e4c2ca`

Run ID：`sbb13_leave_target_out_state_20260906T220850Z_80ec187e`

| 产物 | SHA256 |
| --- | --- |
| `leave_target_out_state_materialization_decision.json` | `bfa12f05612946835c514f4240aeb371a3e697f6ab0186f269a06864ae39feba` |
| `leave_target_out_measurement_removal_contract.json` | `e31a30dcab3761079feacb31f244331a8843b174c437027ad545a55cb3a54560` |
| `leave_target_out_state_inventory.json` | `47b47b6005ea09588a9374b6630b04837868a222319803e4c4be703dbae17f30` |
| `leave_target_out_covariance_contract.json` | `6cc388837acd3ecaad2e6d4b99125719aaf142776941f5d78d0186b251b9a74b` |
| `leave_target_out_denominator.json` | `6a2d0729d4e912f6aaf175606cf2bfb7f7be12e85a9a71d0048e89d8fe3a7be2` |
| `inherited_stage.json` | `4b4273db1206d77161255c9dd7169d852b00155196d195392e116893e93e994d` |
| `COMPLETE.json` | `7711cd06b41745a9b6e7bc969154f4b4dcc3997509faa6c75aaefdfb39455bb8` |

测试 5 passed。该 BLOCKED run 未进 B14 / B15 / V2。

Provenance 与 WB98 冻结一致：

- geometry `4e965f3631dd4ff6e04b141ac36efc49603a1dfc0625de5ea7558dd929486b15`
- field `60432de5864cfba361f91f47a694dc111bbb06dcec44434e9682548680460e9c`
- material_map `0df37bd621e6caac2e223fda12fec32a6f8c49685c7b831181b956e59996433c`
- conditions `d30733216d6eb392dbdbcf5a252fed01ff56a045ade59731ca8419febce7a63d`

## 正式全样本验收

第一次 BLOCKED run（dumps 未落地，不得当作 PASS）：

- Run ID：`sbb13_leave_target_out_state_20260906T220850Z_80ec187e`
- `decision = leave_target_out_dumps_not_materialized`

正式 run（9 dump 落地后，新 run ID，不覆盖上一次）：

- Run ID：`sbb13_leave_target_out_state_20260906T223538Z_6d55f3e4`
- Config SHA：`6c4393ee8202a4b641b25599fd0b48eeba08b0930996217a25bda5cbc16cd0d7`
- Helper SHA：`46f30d1a8846e11c218cd88ba7dbdbf5ab9d7ec3503af7a533650fe017e4c2ca`
- Decision SHA：`af7ccd055c2ff3d34f92429e37be19a89ed90a9b0db11ec6770c161c79b6e933`

| 产物 | SHA256 |
| --- | --- |
| `leave_target_out_state_materialization_decision.json` | `af7ccd055c2ff3d34f92429e37be19a89ed90a9b0db11ec6770c161c79b6e933` |
| `leave_target_out_measurement_removal_contract.json` | `7acbae9718f6e038e7cd5d20fe3f6fc667b9d1bf63cea3686dd7f46152907dff` |
| `leave_target_out_state_inventory.json` | `4ca0278ee7179b639c0a6d02a89b3f016f7c5b3ddc7550541248bb8701f6849e` |
| `leave_target_out_covariance_contract.json` | `74897af1b5e02a8695e03807e9525b911b8c2c1ee53f6cbc79149c82b723a0bb` |
| `leave_target_out_denominator.json` | `d305d177f5e5928055e87c65783c85c165a7a16b9153f1774afb3a073dbe2041` |
| `COMPLETE.json` | `6cbe38fdcbd35513abc7be6a6e25bbed352f7290429209bfaf74c11b3d534a27` |

冻结 denominator 成立：2680 / 691 / 1989 / 1974。contracted 1989 行全部匹配且全部拟合成功。

construction：1116/1116。validation：873/873。target 1/2/3 各 663/663。9 个 source 全部成功。

| 契约 | 结果 |
| --- | --- |
| 1. target exclusion 机器可证 | 2685/2685 行 `target_station_measurements_used=0`；成功行排除 hits 中位 6，IFT recovered 中位 6 |
| 2. independent state 物化 | 1989/1989 成功行有 reference surface、native 5-vector、export 5×5、finite signed q/p、SPD |
| 3. covariance 独立 | `official_WB107_Cin_reused=false`（0/1989 bit-identical）；`seed_covariance_used_as_output_covariance=false` |
| 4. q/p contract | 拟合 signed q/p，单位 1/MeV，无 dummy，无 truth |
| 5. 全 denominator | requested=1989，success=1989，failure=0；dump-wide 108 个 `too_few_remaining_measurements` 留在 ineligible |
| 6. `100043/37` | 全样本 dump 中 target 1/2/3 都在 |

### Focus `100043/37`（诊断，未删除）

三个 target 都 `fit_success=true`，`target_station_measurements_used=0`，IFT recovered=5。

| target | excluded | used stations | used hits | q/p (1/MeV) | σ(q/p) | χ² | n_fit | cov valid |
| ---: | ---: | --- | ---: | --- | --- | --- | ---: | --- |
| 1 | 5 | 0,2,3 | 16 | −2.416353439070118e-6 | 0.001 | 3.76e-7 | 2 | true |
| 2 | 5 | 0,1,3 | 16 | −2.416353439070118e-6 | 0.001 | 3.76e-7 | 2 | true |
| 3 | 6 | 0,1,2 | 15 | −2.416353439070118e-6 | 0.001 | 3.76e-7 | 2 | true |

该身份的 q/p 跨 target 相同，σ(q/p)=0.001 仍是无信息 seed 尺度。这是预注册诊断，不是删除或改 fitter 的理由。B13 不因此 FAIL；B14 必须回答它在 LTO 后是否仍在。

冒烟 6/6 不是正式 PASS。正式 PASS 只建立 **LTO prediction state contract**。

## 下一步

B14：LTO state / covariance semantics（propagation 之前）。不得跳到 B15 / V4 C/D / Measurement Model V2。
