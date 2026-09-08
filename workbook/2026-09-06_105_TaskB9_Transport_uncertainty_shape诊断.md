# Workbook 105: Task B9 Transport / Material / State-Uncertainty 形状诊断

日期：2026-09-06
状态：**完成** —— 在 WB104 `transport_covariance_shape_not_validated` 之后，对冻结的 WB103 契约样本做协方差预算、共享测量语义、Cin / material / transport 形状诊断。未重开 raw CKF，未收紧 reconstruction contract，未按残差 / χ² / pull / truth 加新筛选，未删 100043/37，未调 C/Q，未做 PSD 投影，未改冻结闸门，未编造经验 `Cov(pred,target)`，未把过覆盖说成可接受的保守协方差，未进 Measurement Model V2 / alignment / ML。未改 WB87–WB104 结论。

**最终判定：`DIAGNOSED`（主因 Case E）**

- `decision = mixed_or_inconclusive`
- `primary_case = mixed_or_inconclusive`
- `secondary_cases = shared_measurement_covariance_semantics_missing, ckf_input_covariance_overcovered`
- `active_mechanisms` = 上述两个
- `next_step = no_forced_single_root_cause`
- `measurement_model_v2_authorized = false`
- `measurement_model_v2_entered = false`
- `conservative_covariance_declared_acceptable = false`
- `focus_identity_retained = true`
- `raw_ckf_used = false`

冻结保持：

- WB96：`ckf_qoverp_covariance_export_established`
- WB98：`acts_q_materialized_closure_failed`
- WB99：`high_chi2_tail_dominated`
- WB100：`wrong_track_state_dominated`
- WB101：`ckf_reconstruction_failure`
- WB102：`reconstruction_input_contract_missing`
- WB103：`ckf_reconstruction_contract_validated`
- WB104：`transport_covariance_shape_not_validated`

## 起始状态

HEAD（任务开始时）：`32c9044a8543845aaa0767d8089ba6fd731f31a9`

相对 WB104：只新增本任务文件。未覆盖 WB98 dumps。正式输入只能是契约样本。

## 验收标准

| 项 | 要求 | 结果 |
| --- | --- | --- |
| B9.1 协方差预算 | 区分 `C_pred` / `C_target` / 官方公式 | **完成**。官方是 `e = x_prop − x_truth`，`C = C_prop`，不加 `C_target`。 |
| B9.2 共享测量 | 依赖图 + provenance；不编造交叉协方差 | **完成**。共享拟合比例 1.0；滤波/平滑/cluster **unavailable**。 |
| B9.3 Cin 形状 | 传播前 vs 源站经验误差 | **完成**。663 track；3/4 方向过宽，pencil 6.80。 |
| B9.4 material | `λ(C_emp, C0)` 与 `λ(C_emp, C1)`；Q 不作 PSD | **完成**。C0 已经过宽；C1 进一步过宽。 |
| B9.5 线性化尾巴 | 契约样本重报，不重筛 | **完成**。中位数 1.63e-4；>0.05 占 6.7%；非系统性。 |
| B9.6 100043/37 | 保留，只报贡献 | **完成**。占 χ² 80.3%；LOO 后形状仍失败。 |
| 分类 | A–E 择一主因；多机制则 E | **Case E**。A 与 B 同时成立。 |
| 禁止项 | 不进 V2，不调 C/Q，不删 37 | 保持。 |

## 正式残差契约（必须先写清）

官方 V3 残差相对 **truth**，不是相对 target measurement / tracklet：

```
e = x_prop − x_truth
C = C_prop = MODEL1 4×4
```

官方闸门**不**加 `C_target`。因此当前 transport closure **没有**使用

```
C_residual = C_pred + C_target
```

若以后改用测量残差，才需要

```
C_residual = C_pred + C_target − 2 Cov(pred,target)
```

本任务不编造该项。官方 truth 闸门里也没有 `C_target` 重复计数。
共享拟合仍然关键：Cin 来自已包含目标站的全局 CKF fit，不是
leave-target-out 预测。

## B9.1 协方差预算

| 量 | 计数 |
| --- | --- |
| 契约行 | 1989 |
| official pair | 1974 |
| 唯一 CKF track | 663 |

construction (0,1) Frobenius：`C_emp` = 4.13，Cin 4×4 = 2.81，
`F Cin F^T` = 88.0，`C0` = 85.3，`C1` = 85.2。
传播后的 Cin 在解释 material 之前已经远大于 target `C_emp`。
Cin 不在这里与 target `C_emp` 比形状。

## B9.2 共享测量 / 平滑泄漏

| 项 | 结果 |
| --- | --- |
| source 与 target 同一 CKF track | 是（比例 1.0） |
| 目标站在 tracklet 中 | 是（比例 1.0） |
| 目标站有 cluster | 是（比例 1.0） |
| source 状态来源 | `Trk::Track.trackParameters().front()` |
| 滤波 vs 平滑 | **unavailable** |
| 逐 cluster 身份 | **unavailable** |
| `Cov(pred,target)` | 未估计，未编造 |

`prediction_and_target_measurement_independent = false`。
`Track_InStation0` 仍几乎总是 0；站完备性继续用 tracklet。

## B9.3 Cin 形状

663 条唯一契约 track，全部匹配源站 truth（只诊断）。

| Cin 对角 | 中位数 | 均值 |
| --- | --- | --- |
| x | 2.96 | 2.84 |
| y | 0.0135 | 0.0188 |
| tx | 1.94e-6 | 1.63e-5 |
| ty | 1.49e-6 | 7.95e-6 |
| q/p | 4.90e-13 | 1.39e-11 |

条件数中位数 5.8e12。q/p 相关中位数接近 0。
Cin Frobenius 在高 / 中 / 低动量箱中位数 2.94 / 2.93 / 3.00。

相对源站经验误差：λ = 0.027 / 0.054 / 0.153 / 11.35；
pencil = 6.80，主轴 `x`。前三个方向 Cin 过宽，第四个 ty 组合
覆盖不足。过覆盖在**传播前**已经存在，不是传播后才出现。

## B9.4 material-on vs material-off

`Q = C1 − C0` 不作 PSD process-noise，也不投影。

| split | pair | C0 pencil | C0 λ | C1 pencil | C1 λ | C1 更过宽 |
| --- | --- | --- | --- | --- | --- | --- |
| construction | (0,1) | 22.39 | 0.036–0.191 | 21.50 | 0.006–0.070 | 是 |
| construction | (0,2) | 23.16 | 0.018–0.197 | 20.99 | 0.018–0.155 | 是 |
| construction | (0,3) | 22.07 | 0.030–0.191 | 20.17 | 0.030–0.144 | 是 |
| validation | (0,1) | 12.89 | 0.021–0.629 | 12.67 | 0.007–0.111 | 是 |
| validation | (0,2) | 12.96 | 0.014–0.628 | 12.06 | 0.009–0.094 | 是 |
| validation | (0,3) | 11.94 | 0.030–0.628 | 10.85 | 0.027–0.181 | 是 |

C0 已经过宽。material-on 不修复形状，还把最小 λ 压得更小。
物质不足不是主因。Case C 不成立。

## B9.5 Transport 线性化

契约样本重报，不重筛。`‖C0 − F Cin F^T‖_F / ‖C0‖_F`：
中位数 1.63e-4，均值 1.00，p95 0.155，高于 0.05 占 6.7%。
中位数远小于 0.05，尾巴远小于 30%，不升级为系统性
transport 失败。`100043/37` 的该量约 1.6e-4。Case D 不成立。

## B9.6 100043/37

满足契约的真实 CKF momentum-state tail。未删除、未降权、
未膨胀 C、未用 truth q/p。占混合 χ² 的 80.3%。
construction LOO（诊断，不是 cut）：

| pair | LOO pencil | LOO λ |
| --- | --- | --- |
| (0,1) | 21.84 | 0.006–0.070 |
| (0,2) | 21.39 | 0.018–0.155 |
| (0,3) | 20.53 | 0.029–0.144 |

LOO 后形状仍失败。因此：

```
ckf_tail != covariance-shape root cause
```

## 机制判定

| Case | 结论 | 依据 |
| --- | --- | --- |
| A 共享测量协方差语义缺失 | **成立** | 目标站 100% 参与同一 CKF fit；Cin 不是 leave-target-out。官方虽不加 `C_target`，预测与目标测量仍不独立。 |
| B Cin 传播前已过覆盖 | **成立** | 源站 λ 三个 < 0.25，pencil 6.80。 |
| C material / transport 形状失配 | 否 | C0 已经过宽；C1 只是进一步过宽。 |
| D transport 线性化失配 | 否 | Jacobian 尾巴是少数数值行，不是系统性失败。 |
| E 混合 / 不能单选 | **主因** | A 与 B 同时成立，不得强行选一个。 |

下一步（允许路线，尚未执行；不得为进 V2 而把过覆盖写成可接受）：

```
WB104: transport covariance shape NOT validated
  → Task B9（本 workbook）：Case E
       同时成立：
         A leave-target-out prediction contract
         B CKF fit-covariance semantics
  → 修 measurement / transport covariance contract
  → 重新做 Transport Covariance V3
  → 只有 V3 PASS 才授权 Measurement Model V2
```

不得编造经验 `Cov(pred,target)`。不得缩放 C/Q。不得再收紧
reconstruction contract。不得删 37。

## 工程记录

起始 HEAD：`32c9044a8543845aaa0767d8089ba6fd731f31a9`

Diff（相对 WB104，未提交）：

- `configs/transport_uncertainty_shape_diagnosis_v1.yaml`
- `datasets/transport_uncertainty_shape_diagnosis.py`
- `scripts/audit_transport_uncertainty_shape_diagnosis.py`
- `tests/test_transport_uncertainty_shape_diagnosis.py`
- `docs/transport_uncertainty_shape_diagnosis.md`、`docs/transport_uncertainty_shape_diagnosis_cn.md`
- 本 workbook

Config SHA：`ec02fd5117fdd1b91fe58c1bdeb01b30cd921649af8485d4dd88bd85519452bf`

WB103 contract / config SHA：`e8c1f927e1ba03d418c2ed9b03084b198d09ba15196297a1a13cb5044ec48e3d`  
WB103 decision SHA：`a699415777d23770ab63216b92bdef218965264b353a5cee47c8834c264b8ef6`  
WB104 decision SHA：`8e620b50a3b5509d39e66e6dd7e1fd7a1a7aed5f53fdf63f53ec2456da266031`

Run ID：`sbb9_transport_uncertainty_shape_20260906T201529Z_793f63dd`

产物（EOS，不入 git）：`outputs/transport_uncertainty_shape_diagnosis_v1/sbb9_transport_uncertainty_shape_20260906T201529Z_793f63dd/`

| 产物 | SHA256 |
| --- | --- |
| `covariance_budget_decomposition.json` | `55c01ebe2cdee4f2131e50178fef8938d45076ba3b7a1074a95ae5539d01de0f` |
| `shared_measurement_dependency.json` | `70f199ccd9a6d7e817aabaa86321332d04c3bbf04fdd31cf552dc21b22184707` |
| `ckf_input_covariance_shape.json` | `d0f8b02940e87982423e72f072b546337836fb06ca21f32f83a774aed4310b98` |
| `material_shape_contribution.json` | `2efd096bd63fb7dc307627f1a2d4e61f1cdc9551756bc66dca36ce4ac2a0ef6d` |
| `transport_linearization_shape_audit.json` | `a955d5fde9c0dc3c06edfa575edb92152065262ff998ee92f7ae97eb0e13622f` |
| `focus_identity_case_study.json` | `8342e58ea2c094281121574cea2b5dc5e44cacd88a2ceb764ee374e5aa83315c` |
| `transport_uncertainty_shape_decision.json` | `e29fd93b0dbc9f435934184b93ce550d7dc2e3236177ea47502129c6ec2890f2` |
| `inherited_stage.json` | `92762c5683dc583c93ca5b93d6c86c9ae139e58bb2d2521ea43ac979801888e7` |
| `COMPLETE.json` | `c9f006dfd1252a608bd6fe69ca5207eb99f60419e853bb53e8b6998fe6a013f6` |

Provenance（与 WB98 冻结一致）：

- geometry `4e965f3631dd4ff6e04b141ac36efc49603a1dfc0625de5ea7558dd929486b15`
- field `60432de5864cfba361f91f47a694dc111bbb06dcec44434e9682548680460e9c`
- material_map `0df37bd621e6caac2e223fda12fec32a6f8c49685c7b831181b956e59996433c`
- conditions `d30733216d6eb392dbdbcf5a252fed01ff56a045ade59731ca8419febce7a63d`

测试：`tests/test_transport_uncertainty_shape_diagnosis.py`，9 passed。

本诊断只读已有 WB98 dump / ntuple，与 WB99–WB104 同类，在 login 上完成，未另开 HTCondor。先前一次探测 run `sbb9_transport_uncertainty_shape_20260906T201037Z_e8e9ba61` 保留在 EOS，不作正式判定；正式判定是本次补齐 Cin 对角方差后的 run。

## 下一步

主因 Case E：A 与 B 同时成立，不得强行选单一根因。

- 机制 A：建立 leave-target-out 预测契约（不是直接进 MM V2）。
- 机制 B：审计 CKF fit covariance 语义。
- 机制 C / D：本 run 不支持作为主因。

修完 measurement / transport covariance contract 之后，才能重做
Transport Covariance V3。只有将来 V3 PASS 才授权 Measurement Model V2。
