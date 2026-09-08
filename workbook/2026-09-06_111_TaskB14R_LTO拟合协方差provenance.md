# Workbook 111: Task B14R LTO 拟合协方差 provenance / 信息量审计

日期：2026-09-07
状态：**完成 / DIAGNOSED** —— 查清 `CkfLeaveTargetOutDumpAlg` 导出的 5×5 来自哪个 ACTS 状态、是否随 seed 尺度改变、以及哪些 5D 方向真正被剩余 measurements 约束。未进 B15，未进 V4 C/D，未进 Measurement Model V2。未按 truth / χ² 挑选 seed 尺度。未 rescale / inflate / clip / PSD。未删 `100043/37`。未改 WB96–WB110 结论。未覆盖 WB109 dumps。

**最终判定：`DIAGNOSED`**

- `decision = mixed_or_inconclusive`
- `primary_case = mixed_or_inconclusive`
- `active_mechanisms = B + D`
- `lto_cin_contract_established = false`
- `b15_authorized = false`
- `transport_covariance_validated = false`
- `measurement_model_v2_authorized = false`
- `seed_scale_selected_from_closure = false`

## 起始状态

HEAD：`32c9044a8543845aaa0767d8089ba6fd731f31a9`

WB109 PASS：`leave_target_out_state_materialization_established`  
Decision SHA：`af7ccd055c2ff3d34f92429e37be19a89ed90a9b0db11ec6770c161c79b6e933`

WB110 FAIL：`lto_input_covariance_shape_not_validated`  
Decision SHA：`eba85c1835e90f85a1ca0aec4ee896cc1b611214f9213c7602e160cbd291857b`

冻结：

```
leave_target_out_state_materialization_established = true
lto_cin_contract_established = false
b15_authorized = false
transport_covariance_validated = false
measurement_model_v2_authorized = false
```

WB103 冻结 denominator 保持：raw 2680 / ineligible 691 / contracted 1989 / official pairs 1974。

本任务只问：helper 输出的 5×5 到底是不是由剩余 measurements 约束得到的、可解释的 fitted posterior uncertainty。不追求 closure PASS。

## 1. 逐事件拟合信息量

`n_measurements_in_fit` 的 ACTS 定义已经沿代码确认，不是凭字段名猜测：

- `fittedTrack.nMeasurements()` = `Acts::calculateTrackQuantities` 对 `TrackStateFlag::MeasurementFlag` 的计数
- `nDoF()` = 这些 measurement states 的 `calibratedSize()` 之和
- 因此 `n_fit = 2` 表示 **两次 measurement update**，不是输入 hit 数的别名

正式 WB109 dump 的 contracted 1989 行：

| 子集 | N | n_fit 中位 (min/p95/max) | used 中位 | n_fit≤2 | χ² 中位 | 站点数中位 | z-span 中位 [mm] |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 全部 | 1989 | 14 (2 / 16 / 24) | 18 | 3 | 10.27 | 3 | 4287.55 |
| construction | 1116 | 14 (2 / 16 / 24) | 18 | 3 | 10.84 | 3 | 4287.55 |
| validation | 873 | 14 (4 / 16 / 22) | 18 | 0 | 9.78 | 3 | 4287.55 |
| target 1 | 663 | 14 | 18 | 1 | — | 3 | 4287.55 |
| target 2 | 663 | 14 | 18 | 1 | — | 3 | 4287.55 |
| target 3 | 663 | 14 | 18 | 1 | — | 3 | 3097.55 |

`n_fit / used` 中位 = 14/18 = 0.78。主体样本不是“只有两个 hit”。

contracted 里 `n_fit≤2` 的 3 行全部是 `100043/37` 的 target 1/2/3。seed-scale `σ(q/p)=0.001` 的 contracted 行共 9 条：focus 37 三条，加上 `100047/19` t1、`100048/31` t1、`100048/86` t1/2/3、`100048/42` t1（后四者 `n_fit=4`）。

WB109 schema 没有 `ndof` / filtered / smoothed / outlier。这些量在 B14R smoke schema 上可用，对正式全样本标记 `unavailable`，不推测。

## 2. Seed-covariance 依赖性

预注册尺度 `0.1× / 1× / 10×`。固定 development 事件 `0, 1, 37`，同一 measurements 与 seed mean，只改 seed covariance 整体尺度。生产尺度保持 1。没有按 truth / χ² 挑选。

9 组比较（3 事件 × 3 target）全部成功。

**拟合均值几乎不随 seed 变。** 典型事件 0/1 的 q/p 相对 1× 变化通常 < 5%；空间均值同样稳定。

**拟合 5×5 / σ 明显随 seed 变。** 例如事件 0 / target 1：

| 尺度 | n_fit | q/p [1/MeV] | σ(q/p) | σ(x) | σ(y) |
| ---: | ---: | --- | --- | --- | --- |
| 0.1 | 16 | −8.74e-7 | 2.30e-7 | 0.336 | 0.067 |
| 1 | 16 | −8.71e-7 | 3.65e-6 | 0.695 | 0.761 |
| 10 | 16 | −8.57e-7 | 8.94e-6 | 1.579 | 9.26 |

`typical_qoverp_seed_sensitive = true`，`typical_spatial_seed_sensitive = true`。三个 target 的 smoke 都敏感。去掉 target 3 后 z-span 从 4287.55 mm 降到 3097.55 mm，但 seed 敏感性并不是只出现在 target 3。

因此：即使典型事件有 ~14 次 measurement update，当前导出的 5×5 **还不能当作已摆脱 seed 的物理 uncertainty contract**。这不是挑一个“更好看”的尺度能解决的。

## 3. 导出状态 provenance

代码路径与 smoke 实证一致：

```
exported_state_type = kalman_fitted_parameters_at_reference_surface
reference_surface = source_station_plane  (z = −1860.15 mm)
referenceSurfaceStrategy = first
```

导出的是 KalmanFitter `fittedParameters`：第一个 smoothed measurement **传输到 source 平面之后** 的状态。不是 `KalmanFitterTool.fit`。

smoke scale=1 的 9 个成功行：

- 与 seed bit-close：0/9
- 与 first predicted bit-close：0/9
- 与 first/last filtered、first/last smoothed 原始矩阵 bit-close：0/9

`back_propagation_used = unavailable`（ACTS API 未直接给出，不推测）。比较在 native bound MeV 下进行，先把 GeV seed / predicted 按 helper 同一规则换成 MeV，避免单位混比。

**Case A 不成立：** 导出的 5×5 不是误取的 seed 或 predicted state。它是 fitter 写在参考面上的 fitted parameters。典型事件的 `σ(q/p)` 也远小于 seed 的 0.001。

## 4. 参数信息 / 可测性

ACTS normal / information matrix：**unavailable**。没有用历史 alignment `rank_tolerance=0.01`。

用 n_fit 与预注册 seed 有限差分代替：

- 去 target 1：中位 n_fit 仍是 14，used 18，z-span 仍 4287.55 mm；q/p 均值可更新，但 σ 仍随 seed 变
- 去 target 2：同样 n_fit 中位 14；事件 0 的 σ(q/p) 在 1× 已到 1.7e-7，10× 却升到 4.4e-6
- 去 target 3：z-span 降到 3097.55 mm（弱 bending lever arm），n_fit 中位仍 14，seed 敏感性仍在

结论：剩余 measurements **足以移动 5D 均值**，但 **不足以把 5×5 做成 seed-independent 的认证 uncertainty**。target-dependent 的 WB110 λ 失败与这种信息几何一致，但不能把数值 rank 当成 alignment gate。

## 5. Pull tail provenance

正式 contracted pulls（truth 仅诊断，未做 selection）：

| 分量 | median | median \|pull\| | p68 | p95 | p99 | RMS | top 1% χ² 份额 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| x | 0.070 | 0.831 | 1.31 | 5.84 | 16.3 | 3.65 | 0.573 |
| y | −0.004 | 0.170 | 0.381 | 1.93 | 6.71 | 24.1 | **0.999** |
| tx | 0.036 | 1.072 | 1.82 | 14.9 | 43.2 | 9.88 | 0.683 |
| ty | −0.002 | 0.617 | 1.11 | 10.6 | 26.2 | 6.04 | 0.547 |
| q/p | −0.009 | 0.565 | 1.54 | 17.3 | 49.4 | 16.6 | 0.825 |

y 的主体（p68=0.38）合理，RMS=24.1 几乎完全由少数灾难拟合贡献。`|y-pull|>10` 只有 11/1989。最大 \|y-pull\|=620 来自 `100048/86` 的三个 target（n_fit=4，seed-scale σ(q/p)，χ²=37.6）。未删除。

其余分量的 p68 都 < 2，所以 **Case C 不成立**（不是“主体 covariance 系统过窄且 seed-independent”）。RMS ≫ 1 主要是尾巴。n_fit 与 \|pull\| 的 Pearson 相关很弱；`n_fit≤2` 箱子的 \|pull\| 反而小，因为 σ 接近 seed、特别宽。

## 6. Frozen focus `100043/37`

三条 target 都保留。scale=1 smoke 与 WB109 正式 dump 的 q/p / χ² / n_fit 一致。

| target | used | n_fit | filtered/smoothed | χ² | q/p [1/MeV] | σ(q/p) |
| ---: | ---: | ---: | ---: | --- | --- | --- |
| 1 | 16 | 2 | 3 / 3 | 3.76e-7 | −2.416353439070118e-6 | 0.001 |
| 2 | 16 | 2 | 3 / 3 | 3.76e-7 | −2.416353439070118e-6 | 0.001 |
| 3 | 15 | 2 | 3 / 3 | 3.76e-7 | −2.416353439070118e-6 | 0.001 |

为什么：

- `n_fit=2` **就是** 两次 MeasurementFlag update。used=15–16，所以不是 dump 把 hit 数写成了 `n_fit`
- q/p 在 0.1 / 1 / 10 三个尺度上 **完全相同**，并等于官方 seed mean
- `σ(q/p)` 严格按 `0.001 × √scale` 走：0.000316 / 0.001 / 0.00316
- native 块 `phi / theta / q/p` 仍接近 seed
- 三个 target 的 LTO q/p 完全相同，是因为共同 seed + 同样的两次 measurement update，不是三条独立拟合碰巧得到同一物理解

未特殊处理、未删除、未降权、未用 truth 替换。

## 分类

| Case | token | 本 run |
| --- | --- | --- |
| A | `lto_exported_covariance_is_seed_or_predicted_state` | 否。导出的是参考面上的 fitted parameters |
| B | `lto_fit_information_insufficient` | **是。** 均值被更新，但 5×5 仍明显依赖 seed |
| C | `lto_fitter_covariance_semantics_mismatch` | 否。协方差并非 seed-independent |
| D | `tail_dominated` | **是。** y RMS 由极少数失败拟合主导；事件保留 |
| E | `mixed_or_inconclusive` | **正式判定。** B 与 D 同时成立，不强行单根因 |

下一步不是缩放 Cin，也不是从三个尺度里挑一个。下一步是重新定义 **物理可支持的 LTO track-state model**，然后再做 B14。B15 / V4 C/D 仍然关闭。

未提交 9-source HTCondor 重 dump：smoke + 冻结 WB109 全样本已经足够完成 A–E 分类。新 helper schema 只写在 `outputs/leave_target_out_dump_v1/b14r_smoke/`。WB109 正式 dump 时间戳仍是 2026-09-07 00:15，未被覆盖。

## 工程记录

Config SHA：`4d2eeb031da5e307c470240aed5114ad3f23a14cc6e9798a4f4d271d7adf32f5`

当前 helper SHA：`89692555792f383006cdd9fe6323ad4867b296fcc2c878f9e71995c2a422f6bd`  
（WB109 正式 dump 仍对应当时的 helper `46f30d1a…`；scale=1 smoke 复现了 focus 37 的正式 q/p / n_fit / χ²。）

WB110 decision SHA：`eba85c1835e90f85a1ca0aec4ee896cc1b611214f9213c7602e160cbd291857b`  
WB109 decision SHA：`af7ccd055c2ff3d34f92429e37be19a89ed90a9b0db11ec6770c161c79b6e933`  
WB103 contract SHA：`e8c1f927e1ba03d418c2ed9b03084b198d09ba15196297a1a13cb5044ec48e3d`

Run ID：`sbb14r_lto_fit_provenance_20260906T235735Z_8ae1d934`

| 产物 | SHA256 |
| --- | --- |
| `lto_fit_covariance_provenance_decision.json` | `c9d35002671327dd401e1ca67bd17cb7dacfadf85efcb53e15af6d94dfc2a644` |
| `lto_fit_information_inventory.json` | `831fdd8b6d6b4a8778ad8edad697f6eaa3de31b181f8d34aa54229e8a15672b1` |
| `lto_seed_covariance_sensitivity.json` | `f90c58f44e357f9fc94d59dfa94ff3f5b8cea82629b34dcb250846cde061fb76` |
| `lto_covariance_state_provenance.json` | `ddabbc2575a8446502129c5bfc339cf5a0dec8fef08f9c50edf6299faad5a238` |
| `lto_parameter_information_audit.json` | `2e261a8499e9fad286631a4731e1bb4b37becd4137a4e9ac096cb088c1553e56` |
| `lto_pull_information_correlation.json` | `3ba3431b158d2e183aaceba5ff7cfedd2aac5d8261b9b353bd3171d5fb7432d7` |
| `focus_identity_lto_fit_provenance.json` | `baa6a2cc0dd0388cb8caf398d4d615a8171f2217ae05c7ebc99c315681f44623` |
| `inherited_stage.json` | `a12523f374fac67d7f328a8475f7ef4305977a8036873d81660b045f5418f382` |
| `COMPLETE.json` | `21d41d2b6ae22a88189f9b988accb6b919d099f25c63c4e683e51a01678a2e70` |

测试 7 passed。无 B15，无 V4 C/D，无 Measurement Model V2，无 alignment，无 ML。

Provenance 与 WB98 冻结一致：

- geometry `4e965f3631dd4ff6e04b141ac36efc49603a1dfc0625de5ea7558dd929486b15`
- field `60432de5864cfba361f91f47a694dc111bbb06dcec44434e9682548680460e9c`
- material_map `0df37bd621e6caac2e223fda12fec32a6f8c49685c7b831181b956e59996433c`
- conditions `d30733216d6eb392dbdbcf5a252fed01ff56a045ade59731ca8419febce7a63d`

## 下一步

保持冻结：

```
b15_authorized = false
lto_cin_contract_established = false
measurement_model_v2_authorized = false
transport_covariance_validated = false
```

主线不变：

```
WB109: LTO state established
WB110: LTO covariance semantics FAIL
    ↓
WB111 / B14R: mixed (B + D)
    ↓
重新定义物理可支持的 LTO track-state model
    ↓
重新做 B14
    ↓
只有 B14 PASS
    ↓
B15 V4 C/D
```

不得用 0.1× / 1× / 10× 里任何一个当作新的生产 seed。不得 rescale LTO Cin。
