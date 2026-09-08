# Workbook 99: Task B3 ACTS transport covariance 物理诊断

日期：2026-09-06
状态：**完成** —— 已定位 WB98 closure failure 的机制。未修复 closure，未进入 Measurement Model V2，未进入 alignment，未训练 ML，未调 Q / covariance，未删除 outlier，未改 WB87–WB98 结论。

**最终判定：`DIAGNOSED`（主因 Case C）**

- `decision = acts_transport_covariance_failure_diagnosed`
- `primary_case = high_chi2_tail_dominated`
- `secondary_cases = acts_material_process_noise_mismatch`, `measurement_uncertainty_model_insufficient`
- `next_step = uncertainty_model_analysis_no_covariance_tuning`
- `closure_pass = false`
- `measurement_model_v2_entered = false`
- `geometry_write_allowed = false`
- `unconstrained_tracker_only_stopped = true`

冻结保持：

- WB87：`faseracts_transport_covariance_not_validated` / `q_over_p_uncertainty_semantics`
- WB95：`physical_qoverp_semantics_not_established`
- WB96：`ckf_qoverp_covariance_export_established`
- WB97：`acts_process_noise_contract_not_established` / `acts_process_noise_not_materialized` / `acts_process_noise_configuration`
- WB98：`acts_process_noise_contract_not_established` / `acts_process_noise_closure_failed` / `acts_q_materialized_closure_failed` / `measurement_track_model_insufficient`

## 起始状态

HEAD（任务开始时）：`32c9044a8543845aaa0767d8089ba6fd731f31a9`（WB97 / Task B FAIL；WB98 源码尚未提交）

相对 WB98：只新增本任务文件。未重跑 WB87–WB98，未改冻结结论。未覆盖 WB98 dumps / `sbb0_acts_transport_dump_20260906T172303Z_9825e122`。

输入：只读 WB98 dump `outputs/acts_transport_dump_v1/dumps/*/ckf_acts_transport.jsonl`，同一 9 个 WB87 source，同一 truth join。未新建 MC，未写 geometry payload。

## 验收标准

| 项 | 要求 | 结果 |
| --- | --- | --- |
| B3.1 状态 / surface | `state_surface_contract.json`：输入/输出定义、frame、单位、Jacobian、geometry hash、surface id | **完成**。frame contract 成立。 |
| B3.2 C0 vs `F Cin F^T` | 直接比较 dump 的 F、Cin、C0；不反推 F | **完成**。中位相对 Frobenius `2.05×10^{-4}`。 |
| B3.3 残差分解 | x/y/tx/ty/q/p；construction/validation；冻结 bin | **完成**。未改 bin。 |
| B3.4 高 χ² 溯源 | top 1% / 0.1% 全部保留 | **完成**。0 删除、0 clip。 |
| 机制分类 | Case A/B/C/D，不要求 closure PASS | **Case C 主因**；B、D 为次因。 |
| 禁止项 | 不进 V2 / alignment，不用 truth q/p 作 Cin，不调 Q | 保持。 |

## B3.1 State / surface convention

C++ helper 契约（2680/2680 行一致）：

- 输入：Athena `(loc1, loc2, phi, theta, q/p)`，单位 `mm, mm, rad, rad, 1/MeV`，带符号。`loc1=loc2=0` 是 curvilinear 原点，不是缺 x/y。
- Acts bound：`(loc0, loc1, phi, theta, q/p per GeV, time)`；`q/p_Acts = q/p_native / 1_MeV`。
- 输出：全局 `(x, y, tx=px/pz, ty=py/pz, q/p per MeV)`，来自 `position()` / `momentum()`，**不用 loc 当 x/y**。
- 目标面：`PlaneSurface((0,0,z), normal=(0,0,1))`，z 为冻结 WB87 station z。
- 方向：`target_z >= source_z` 则 Forward；注册 pair 全部 Forward。
- q/p 符号与 charge 2680/2680 一致。
- geometry_hash：`4e965f3631dd4ff6e04b141ac36efc49603a1dfc0625de5ea7558dd929486b15`

`F` 的磁弯轴：`mean |F_{y,q/p}| > mean |F_{x,q/p}|`，与 FASER `B ∥ x̂`、弯折在 y 一致。x 不是被换轴后的弯折方向。`Corr(e_x, charge) ≈ 0.003`。

**Case A 不成立（系统性契约）。** 不是 loc/x 对调，不是 q/p 符号，不是表面法向。

Jacobian 仍有少数行 `‖C0 − F Cin F^T‖_F / ‖C0‖_F` 很大（p95 `0.32`，约 8.3% 行 > 0.05）。这是数值 Jacobian 尾巴，不是整体 frame 错误；见 B3.2。

## B3.2 C0 Jacobian consistency

直接比较 dump 字段，2662 行，0 次与 dump 自带 `c0_jacobian_frobenius_rel` 不一致，**没有从协方差反推 F**。

| 量 | 中位 | p95 | max |
| --- | --- | --- | --- |
| ‖C0 − F Cin F^T‖_F / ‖C0‖_F | `2.05×10^{-4}` | 0.32 | 543 |
| x 对角相对误差 | `7.4×10^{-5}` | 0.17 | 443 |
| y 对角相对误差 | `2.0×10^{-4}` | 0.19 | 783 |
| tx 对角相对误差 | `1.9×10^{-4}` | 0.59 | 6187 |
| ty 对角相对误差 | `6.4×10^{-4}` | 0.63 | 2.8×10⁴ |
| q/p 对角相对误差 | `8.1×10^{-12}` | `1.8×10^{-11}` | `3.7×10^{-11}` |

典型事件：`C0 ≈ F Cin F^T` 成立。尾巴集中在 tx/ty，不是系统性 x 换轴。

`Q_ACTS := C1 − C0`：

- 1549/2662（58%）不是 PSD
- 1354 行 `Q_xx < 0`，即 `C1_xx < C0_xx`

这不是人为 scale。两个独立 `FaserActsExtrapolationTool`（无材料 / 有材料）绕不同轨迹线性化，差值不必是 PSD process-noise。这是 **Case B 次因**，不是 Case A。

## B3.3 Residual decomposition

官方诊断只用冻结 pair `(0,1)/(0,2)/(0,3)`，与 WB98 closure 一致。Dump 里另有 262 条 source station ≠ 0 的短程/跨站 pair，全部保留，不参与主分类。

官方 2352 条（construction 1309 + validation 1043）。truth q/p **只**作第 5 分量诊断残差，从未写入 Cin。材料厚度 dump 中不存在，proxy 为 `‖Q‖_F` 与 `‖Q‖/‖C0‖`。

| 分量 | mean e | RMS e | pull RMS | 中位 χ² 贡献 |
| --- | --- | --- | --- | --- |
| x | 0.48 mm | 5.24 mm | 1.47 | 见整体 χ² |
| y | — | 13.7 mm | 0.93 | — |
| tx | — | 3.4×10⁻³ | 1.55（全体） | — |
| ty | — | 6.4×10⁻³ | 1.85（全体） | — |
| q/p | −4.8×10⁻⁷ /MeV | 2.4×10⁻⁵ /MeV | 4.82 | 诊断 only |

整体官方 χ²/ndof：median `0.150`，mean `33.3`，p95 `31.6`，max `3.71×10⁴`。

x 残差依赖（冻结 bin，未改）：

- **电荷**：construction 负/正 mean e_x = 0.66 / 0.39 mm；validation 0.19 / 0.62 mm。相关 `0.003`。不是电荷换号。
- **动量**：`Corr(|e_x|, |q/p|) = 0.39`。low-momentum 中位 χ² 仍 < 1，但有极端 max。
- **材料 proxy**：`Corr(|e_x|, ‖Q‖_F) = 0.29`。高 Q bin 的 x RMS 更大，但中位 χ² 仍常 < 1。
- **入射**：large slope 样本很少（construction 13，validation 3），x RMS 更大；主样本是 small slope。
- **station pair**：三个官方 pair 中位 χ² 均 < 0.5；mean 被尾巴拉高，尤其 construction `(0,3)` mean `101`。

主体事件 pull_x p95 `0.67`、median χ² `0.15`：典型协方差**偏大**（overcover），与 WB98 pencil 主方向为 x 一致。这是 **Case D 次因**（uncertainty model 不足以描述残差分布），不是 Case A。

## B3.4 High-χ² tail provenance

官方 pair：

- top 1%：24 条，χ²/ndof ≥ 300，占全部 χ² 之和的 **85.3%**
- top 0.1%：3 条，χ²/ndof ≥ 4243，占 **67.5%**
- mean / median = **221**
- 删除 / clip / reject：0

top 0.1%（全部保留）：

| source | run/event | pair | p_reco | p_truth | q/p | ‖Q‖_F | e_x | e_y | Q_xx | C 最小/最大特征值 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100043_00400_00499 | 100043/37 | (0,3) | 414 GeV | 2014 GeV | −2.42×10⁻⁶ | 0.15 | 10.7 | −5.8 | −0.10 | 2.6×10⁻¹⁵ / 0.066 |
| 100048_00050_00099 | 100048/86 | (0,1) | 90 GeV | 2566 GeV | −1.12×10⁻⁵ | 3.09 | 8.5 | −14.4 | +0.20 | 4.1×10⁻¹³ / 3.84 |
| 100044_00200_00299 | 100044/92 | (0,1) | 32 GeV | 20 GeV | +3.15×10⁻⁵ | 19.9 | 9.8 | 8.2 | +14.1 | 5.1×10⁻¹³ / 55.5 |

前两条 reco p 与 truth p 差一个量级，更像错误关联 / 错误种子，不是普通 multiple scattering。同一 event `100043/37` 还出现在 `(0,1)`、`(0,2)` 的 1% 尾巴里。材料厚度字段为 null。

off-contract pair（source station ≠ 0）262 条全部保留；它们的 mean χ² 更高，但不进入 WB98 官方 closure 集合。

## 机制判定

| Case | 结论 | 依据 |
| --- | --- | --- |
| A 修 transport contract | **否（系统性）** | frame / 符号 / 弯折轴 / 中位 `C0≈F Cin F^T` 均成立。Jacobian p95 尾巴单独记录，不把少数数值行当成整体契约失败。 |
| B ACTS material diagnosis | **次因** | `Q=C1−C0` 58% 非 PSD；`C1_xx<C0_xx` 常见；残差 RMS 随 Q、|q/p| 上升。下一步可以做材料诊断，但**先不要调 Q**。 |
| C 少数 tail 主导 | **主因** | 1% 事件贡献 85% 的 χ² 总和；mean/median=221。必须保留这些事件。 |
| D 重新评估 MM V2 输入模型 | **次因** | 主体 median χ²≪1、x 方向 overcover。transport 契约正确后，当前 measurement/uncertainty 模型仍盖不住残差分布。B3 **没有进入 V2**。 |

下一步（允许路线，尚未执行）：

```
WB98
  → Task B3 (本 workbook)：failure mechanism = Case C (+B, +D)
  → uncertainty-model analysis（不调 covariance）
  → 如需：ACTS material diagnosis（不调 Q）
  → 再讨论 Measurement Model V2 的输入模型
  → Transport covariance V3
  → Measurement Model V2
```

禁止跳过上述步骤进入 alignment。

## 工程记录

起始 HEAD：`32c9044a8543845aaa0767d8089ba6fd731f31a9`

Diff（相对 WB98，未提交）：

- `configs/acts_transport_diagnosis_v1.yaml`
- `datasets/acts_transport_diagnosis.py`
- `scripts/audit_acts_transport_diagnosis.py`
- `tests/test_acts_transport_diagnosis.py`
- `docs/acts_transport_diagnosis.md`、`docs/acts_transport_diagnosis_cn.md`
- 本 workbook

WB98 文件保持原样，未覆盖 dumps。

Config SHA：`84c7d96c42ee1b04816b5ea485919b5d139e887f4b5111bd1868334a68e55a62`

正式 Run ID：`sbb3_acts_transport_diagnosis_20260906T180100Z_574d6429`

预备 run `sbb3_acts_transport_diagnosis_20260906T175845Z_00b4be2e` 未覆盖；它把 Jacobian p95 尾巴误判成 Case A。正式分类改为：Case A 只认系统性契约失败（中位 Jacobian + frame）；残差/尾巴按冻结 pair 统计。

产物（EOS，不入 git）：`outputs/acts_transport_diagnosis_v1/sbb3_acts_transport_diagnosis_20260906T180100Z_574d6429/`

| 产物 | SHA256 |
| --- | --- |
| `state_surface_contract.json` | `8044cd27fc0eb74e71d03214a9892ff530a9f91afefaf1ffdb704b2bb966d97d` |
| `jacobian_covariance_consistency.json` | `4e01cefdbc24bf6e24753248dea93809034d681aeb9a2b559582b8fcfee4e012` |
| `residual_decomposition.json` | `5787972f0499c097b690c1c5963e9c5b430e3fed0c31660602a2eea51122f4fe` |
| `tail_provenance.json` | `c6ad96f9b019f0ef1f32ec6d5476ebbb99234aab8e9207a2a314ad89e94627ce` |
| `acts_transport_diagnosis_contract.json` | `3595c6ef3573947f6ed5ceb5cc835a6fa406cb160c0909f121e6f1cad8ec5fa2` |
| `inherited_stage.json` | `7bd02b0d88788da8f290de020fd59e5857c5039a6df70d81c9645baa81883f34` |
| `COMPLETE.json` | `ef4a6976771ed4ee27471fb83b5e0e9050065254c6aed1682b78faab1979b51e` |

geometry / field / conditions hash 与 WB96/WB98 相同。

测试：`tests/test_acts_transport_diagnosis.py`，12 passed。诊断本身是对已有 dump 的短审计，与 WB98 B2 一样在 login 跑；未覆盖 Condor dump。

## 下一步

主因 Case C：先分析 uncertainty model，**不调、不删、不 clip** 协方差或尾巴。

次因允许随后做 ACTS material diagnosis（解释非 PSD 的 `Q=C1−C0`），以及在 transport 契约保持正确的前提下重新评估 Measurement Model V2 的**输入**模型。

仍然禁止：进入 alignment、训练 ML、调 Q、调 covariance、用 truth q/p 作 Cin、用 dummy SegmentFit、删 q/p、写 `/Tracker/Align`、进入 Measurement Model V2。
