# Workbook 104: Task B8 Transport Covariance Validation V3

日期：2026-09-06
状态：**完成** —— 在 WB103 冻结的 `eligible_for_transport_validation` 上正式重算 Transport Covariance V3。未重开 raw CKF，未按残差 / χ² / pull / truth / 37 / 动量差加新筛选，未删 100043/37，未调 C/Q，未做 PSD 投影，未改冻结闸门，未进 Measurement Model V2 / alignment / ML。未改 WB87–WB103 结论。

**最终判定：`NOT_ESTABLISHED`（主因 Case C）**

- `decision = transport_covariance_shape_not_validated`
- `primary_case = transport_covariance_shape_not_validated`
- `secondary_cases = ckf_fitting_tail_blocks_transport_validation`
- `next_step = transport_material_uncertainty_diagnosis`
- `closure_pass = false`
- `measurement_model_v2_authorized = false`
- `measurement_model_v2_entered = false`
- `focus_identity_retained = true`
- `raw_ckf_used = false`

冻结保持：

- WB96：`ckf_qoverp_covariance_export_established`
- WB98：`acts_q_materialized_closure_failed`
- WB99：`high_chi2_tail_dominated`
- WB100：`wrong_track_state_dominated`
- WB101：`ckf_reconstruction_failure`
- WB102：`ckf_reconstruction_contract_audited` / `reconstruction_input_contract_missing`
- WB103：`ckf_reconstruction_contract_validated` / `input_scope_mismatch_confirmed`

## 起始状态

HEAD（任务开始时）：`32c9044a8543845aaa0767d8089ba6fd731f31a9`

相对 WB103：只新增本任务文件。未覆盖 WB98 dumps。正式输入只能是契约样本。

## 验收标准

| 项 | 要求 | 结果 |
| --- | --- | --- |
| B8.1 冻结输入 | 继承 WB103 资格，不加新筛选 | **完成**。1989 行 / 1974 pair，与 WB103 Sample B 一致。 |
| B8.2 正式重算 | construction / validation × (0,1)/(0,2)/(0,3) | **完成**。见下表。 |
| B8.3 100043/37 | 保留，只作诊断 | **完成**。占混合 χ² 的 80.3%。LOO 不是 cut。 |
| B8.4 形状 / overcoverage | λ(C_emp, C_prop) 映射到 x/y/tx/ty | **完成**。全部 overwide；pencil 主轴是 x。 |
| B8.5 Q / material | 只诊断，非 PSD 不自动 FAIL | **完成**。63.8% Q 非 PSD，未投影。 |
| B8.6 分类 | A / B / C / D 择一 | **Case C**。次因 B。 |
| 禁止项 | 不进 V2，不调 C/Q，不删 37 | 保持。 |

## B8.1 输入

资格与 WB103 完全相同：long-track、四 tracklet 站、四 segment、有限 5×5 Cin、有限带符号 q/p、有效 surface。不用 χ² / pull / 残差 / truth。

| 量 | 计数 |
| --- | --- |
| raw dump 行 | 2680 |
| 不合格 | 691 |
| 契约行 | 1989 |
| official pair | 1974 |

## B8.2 正式 MODEL1

同一 dump、同一 C0/C1/Q、同一闸门。只改输入范围为契约样本。

事件级：均值 25.76，中位数 0.115，q/p pull RMS 0.84。

| split | pair | N | χ² 均值 | χ² 中位 | pencil | λ | χ² 闸门 | 形状 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| construction | (0,1) | 370 | 13.13 | 0.359 | 21.50 | 0.006–0.070 | 失败 | 失败 |
| construction | (0,2) | 370 | 8.24 | 0.261 | 20.99 | 0.018–0.155 | 失败 | 失败 |
| construction | (0,3) | 370 | 111.39 | 0.295 | 20.17 | 0.030–0.144 | 失败 | 失败 |
| validation | (0,1) | 288 | 3.34 | 0.044 | 12.67 | 0.007–0.111 | 通过 | 失败 |
| validation | (0,2) | 288 | 1.59 | 0.046 | 12.06 | 0.009–0.094 | 通过 | 失败 |
| validation | (0,3) | 288 | 1.10 | 0.048 | 10.85 | 0.027–0.181 | 通过 | 失败 |

三层必须分开：

- **主体校准**：中位数 χ² 很低；validation 均值 1.1–3.3，whitened-χ² 通过。
- **尾巴贡献**：`100043/37` 占混合 χ² 总和 80.3%，把 construction (0,3) 拉到 111。
- **形状失配**：validation 没有 event 37，pencil 仍是 10.8–12.7，λ 全部 < 0.25。

source 级中位数都在 0.03–0.33。`mc24_100043_00400_00499` 均值 181.6，中位数 0.33。validation 四个 source 均值 0.99–2.96。

## B8.3 100043/37

满足契约：long track，四站，四 segment，21 测量，CKF χ²/ndof = 1.42。reco ≈ 414 GeV，truth 诊断 ≈ 2.0 TeV，`|q/p pull| ≈ 13.8`。三个 pair 的 transport χ²：2971 / 760 / 37120。未删除、未降权、未膨胀协方差、未用 truth q/p 替换。

construction leave-one-out（诊断，不是 cut）：

- (0,3) 均值 111 → 11.1
- pencil 仍约 21
- 形状闸门仍失败

因此：保留 37 时，V3 **不是**“主体已校准、只被拟合尾巴挡住”。主体上协方差形状本身就不成立；37 另外污染均值。

## B8.4 形状

pencil 主轴都是 `x`。广义本征方向全部 overwide（λ < 0.25），最小 λ 落在 `tx`/`ty`。validation 95% 覆盖 0.92–0.93，同时均值 χ² 已接近合理范围，说明 C_prop 过宽。禁止标量 rescale。

## B8.5 Q

`Q = C1 − C0` 不能当成固定线性化下的 PSD process-noise。63.8% 非 PSD，不因此判 FAIL，也不做投影。Frobenius 中位数：C0 = 27.4，C1 = 18.3，Q = 5.7。material-on 不是简单加宽。Q 随低动量增大。只记录，不调。

## 机制判定

| Case | 结论 | 依据 |
| --- | --- | --- |
| A 契约输入下 transport 已验证 | 否 | construction 与 validation 都未全过闸门。 |
| B CKF 拟合尾巴挡住验证 | **次因** | 37 合法且占 χ² 的 80%；LOO 能降均值，但不能让形状过闸。 |
| C 协方差形状未验证 | **主因** | validation（无 37）pencil / λ 仍系统性失败。 |
| D 样本不足 | 否 | N = 1974。 |

下一步（允许路线，尚未执行）：

```
WB103 contracted reconstruction input
  → Task B8（本 workbook）：Case C
  → transport / material / state-uncertainty 诊断
       （仍用本契约；不删 37；不调 C/Q）
  → 次级：CKF momentum-state audit（37）
  → 只有将来 Case A 才授权 Measurement Model V2
```

## 工程记录

起始 HEAD：`32c9044a8543845aaa0767d8089ba6fd731f31a9`

Diff（相对 WB103，未提交）：

- `configs/transport_covariance_validation_v3.yaml`
- `datasets/transport_covariance_validation_v3.py`
- `scripts/run_transport_covariance_validation_v3.py`
- `scripts/audit_transport_covariance_validation_v3.py`
- `tests/test_transport_covariance_validation_v3.py`
- `docs/transport_covariance_validation_v3.md`、`docs/transport_covariance_validation_v3_cn.md`
- 本 workbook

Config SHA：`9b435e0adb1f24233ad0396b2b5ac46426ab5152f7dd43aa44a583e938bff1e5`

WB103 contract / config SHA：`e8c1f927e1ba03d418c2ed9b03084b198d09ba15196297a1a13cb5044ec48e3d`  
WB103 decision SHA：`a699415777d23770ab63216b92bdef218965264b353a5cee47c8834c264b8ef6`

Run ID：`sbb8_transport_covariance_v3_20260906T195338Z_e6c68984`

产物（EOS，不入 git）：`outputs/transport_covariance_validation_v3/sbb8_transport_covariance_v3_20260906T195338Z_e6c68984/`

| 产物 | SHA256 |
| --- | --- |
| `contracted_input_inventory.json` | `2498d61e762bf9489d2c5f4f121b5f2fb9b3964dd4835ae6561b038072abc6af` |
| `transport_covariance_v3_metrics.json` | `976b3d2f671c4b0b28e846177e80da9dfc8ec58ff560ca57e498357397a61dab` |
| `eigenvalue_pencil_analysis.json` | `6cd3d5e43acac5f7ebcebc5cf63c3517ab5da3cf3c116a7dfdd57abaa3c362dc` |
| `ckf_tail_case_study.json` | `cb454c880f0402c8a98d0d8383afd96d91a100f3ccccf6837a2492565419f98b` |
| `transport_covariance_v3_decision.json` | `8e620b50a3b5509d39e66e6dd7e1fd7a1a7aed5f53fdf63f53ec2456da266031` |
| `inherited_stage.json` | `05876facad9e5192f3defd90ec6e44a4b2feb4f7f9d55879f5ebbdbe3f15fa74` |
| `COMPLETE.json` | `26dc0b933bd184e01c0cdd92495acc2853fada384e8ddd92d70001f1cf2d65e0` |

Provenance（与 WB98 冻结一致）：

- geometry `4e965f3631dd4ff6e04b141ac36efc49603a1dfc0625de5ea7558dd929486b15`
- field `60432de5864cfba361f91f47a694dc111bbb06dcec44434e9682548680460e9c`
- material_map `0df37bd621e6caac2e223fda12fec32a6f8c49685c7b831181b956e59996433c`
- conditions `d30733216d6eb392dbdbcf5a252fed01ff56a045ade59731ca8419febce7a63d`

测试：`tests/test_transport_covariance_validation_v3.py`，8 passed。

本验证只读已有 WB98 dump / ntuple，与 WB99–WB103 同类，在 login 上完成，未另开 HTCondor。

## 下一步

主因 Case C：继续诊断 transport / material / state uncertainty model。不要为了过闸门再收紧 reconstruction contract，也不要缩放协方差。

次因 Case B：`100043/37` 仍留在样本里，作为真实 CKF 拟合尾巴。后续 CKF momentum-state audit 可以单独做，但不能靠删点或 inflate C 把 V3 写成 PASS。
