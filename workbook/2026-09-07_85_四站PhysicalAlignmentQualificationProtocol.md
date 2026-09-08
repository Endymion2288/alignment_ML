# Workbook 85 — Physical Common-Track Alignment Qualification Protocol Freeze

日期： 2026-09-07
分支： `4station`
任务： 只设计并冻结 physical common-track qualification protocol。不跑 alignment，不看 physical-corpus alignment outcome。

---

## 0. 裁决（当前）

```text
physical_replica_production_complete = true
calypso_physical_corpus_qualified    = true

alignment_oracle_qualified_for_physical_FASER = false
ml_alignment_eval_authorized                  = false
qualification_authorized                      = false
executable                                    = false
```

WB83 / WB84 未改。未生产新 physical replica。未返回 association。未授权 HTCondor qualification。

```text
protocol_audit = PASS
```

---

## 1. 输入 corpus

唯一允许：

```text
provenance = calypso_physical_independent_v1
n_qualified_exports = 2394
```

计数来自 WB84 corpus QA，不是按 WB83 failed cells 重新设计的 n。

| family | survey tag | qualified |
|---|---|---:|
| identity | fixed_dz | 299 |
| identity | finite_survey_prior | 303 |
| identifiable_translation | fixed_dz | 311 |
| identifiable_translation | finite_survey_prior | 298 |
| identifiable_rotation | fixed_dz | 300 |
| identifiable_rotation | finite_survey_prior | 283 |
| weak_jg_diagnostic | fixed_dz | 302 |
| weak_jg_diagnostic | finite_survey_prior | 298 |

```text
n_physical_geometry_families = 4
```

`fixed_dz` 与 `finite_survey_prior` 是 downstream alignment analysis modes，不是不同的 Calypso reconstruction physics。WB84 已给两个 tag 分配不同 independent event ensembles，因此可以分别资格审查，但不得称为 paired event-by-event comparison。

---

## 2. 冻结 geometry（禁止事后重选）

沿用 WB84 已冻结的 4 families：

```text
identity:
  all zero

identifiable_translation:
  S1 dx = +0.30 mm
  S2 dy = -0.20 mm

identifiable_rotation:
  S1 rz = -0.5 mrad
  S3 rz = +0.5 mrad

weak_jg_diagnostic:
  S3 dx = +0.20 mm
  S3 ry = +0.2 mrad
```

禁止在看到 alignment 结果以后换 weak direction、改幅度、删除难 condition、或增加更容易的 condition。

---

## 3. Qualification charts

没有自动恢复旧 15D WLS。每个 family 对应显式 common-track chart。

共同合同：

```text
reference convention     = hold S0 at identity
left-SE(3) update        = true
local nuisance           = ξ = (x, y, tx, ty, q/p), Schur
q/p treatment            = q_over_p_prior, σ = 1e-3
solver                   = common_track_solver WB83-v2
symmetrized normal       = true
strict SPD               = true
absolute-state survey    = true
iterative relinearization = true
max iterations           = 10
automatically pass at max = false
robust IRLS              = false
```

`fixed_dz`：dz held exactly。  
`finite_survey_prior`：free-station dz 进入 chart，absolute-state MAP，σ = 5 mm；dz 不是 projected scientific mode。

| chart | families | free scientific parameters | mode_status | 进入 primary T_LS |
|---|---|---|---|---|
| translation | identity, identifiable_translation | S1/S2/S3 dx, dy | identifiable | yes |
| rotation | identifiable_rotation | S1/S3 rz | identifiable | yes |
| weak-JG diagnostic | weak_jg_diagnostic | S3 dx, S3 ry | weak | no |

Identity 与 translation 的 projected mode 是同一 a priori 单位向量 `(0.30, −0.20)/‖·‖`。Rotation 投影是 `(-0.5, +0.5)/‖·‖`。

若某 physical mode 在固定 field/material 下不可辨识或严重 weak：

```text
mode_status = weak / non-identifiable
```

不得在 pseudoinverse / dropped column 后仍称 full-rank qualified。identifiable chart 上出现 dropped parameter 直接 FAIL。

官方执行必须走真实 Calypso/ACTS chain。`iterate_common_track(..., field_y=...)` 的 toy `lab_transport` 不是 official engine。Newton/Schur/SPD/survey-prior 目标函数冻结，本 workbook 不改 solver objective。

---

## 4. Primary statistical qualification

不复用 WB83 的

```text
every cell independently passes an unadjusted threshold
```

作为 primary global gate。

预注册 **M2 global location-scale GOF**。选择 M2 是因为它对应 “solver 在预注册 identifiable modes 上校准” 这一科学声明，**不是**因为它本来会让 WB83 PASS。WB83 的 `-0.262893` 与 `89/100` 不是调参目标。

对 6 个 identifiable strata：

```text
identity × {fixed_dz, finite_survey_prior}
identifiable_translation × {fixed_dz, finite_survey_prior}
identifiable_rotation × {fixed_dz, finite_survey_prior}
```

```text
z_ir = (θ̂_ir − θ_true_i) / σ̂_ir
```

`σ̂_ir = sqrt(uᵀ Cov u)`。weak-JG 不进入 primary。

```text
H0: E[z] = 0, Var[z] = 1
T_loc  = Σ_i n_i z̄_i²
T_sc   = Σ_i (n_i − 1)/2 (s_i² − 1)²
T_LS   = T_loc + T_sc  ~ χ²_{12}
α      = 0.05
accept if T_LS ≤ χ²_{12, 0.95} ≈ 21.026
```

Strata 独立（不同 event ensembles）。Stacking 预注册，禁止 post-hoc pooling。

---

## 5. Multiplicity / family structure

预注册 diagnostic families：

```text
translation
rotation
weak-JG
survey-prior
```

Primary FAIL 后才输出这些分解。cellwise p-values 做 Holm。这些 diagnostics：

```text
不是重新选择模型的依据
不是改门槛的依据
```

---

## 6. Catastrophic guardrails

Global PASS 不能平均掉明显错误。任一发生都直接 FAIL：

```text
identifiable stratum 非收敛比例 > 5%
NaN / Inf
unresolved non-SPD
geometry sign/frame failure（反号且 |mean z| > 3）
FD relative error > 0.01
identifiable translation/identity |bias| > 0.1 mm
identifiable rotation |rz bias| > 1 mrad
unqualified dropped identifiable mode
coverage grossly below nominal
```

“grossly below nominal” 在看数据前定义为：

```text
empirical coverage < 0.70
或 95% CP 上界 < 0.80
```

0.1 mm / 1 mrad 是 **research screening threshold**，不是 collaboration physics requirement。

weak-JG 不套普通 translation 0.1 mm。改为 standardized weak-mode bias（|mean z| > 5 为 catastrophic）并报告 absolute S3 dx / ry sensitivity。

---

## 7. Coverage

仍报告 95% empirical coverage、CP interval、pull mean、pull width。

不再要求每一个 cell 的 raw 95% CP interval 都必须包含 0.95 作为唯一 global gate。

预注册：

```text
T_cov = Σ_i (K_i − n_i p)² / (n_i p(1−p))  ~ χ²_6
p = 0.95
```

外加 Holm-adjusted cell coverage diagnostics。

---

## 8. Sample size / power

使用 WB84 已生产的 qualified count。不因 WB83 两个 failed cells 重新设计 n。禁止 post-hoc pooling 提升名义 n。

Prospective power，保守按每 identifiable stratum n = 200：

```text
one-stratum persistent mean 0.35 : power ≈ 0.936
one-stratum scale 1.25           : power ≈ 0.983
80% MDE |μ|                      ≈ 0.294
80% MDE scale                    ≈ 1.191 / 0.763
```

`μ = 0.25` 的 power ≈ 0.62，**不足 80%**。该 alternative 若被当作声明，则

```text
qualification_status = UNKNOWN
```

而不是 PASS。它不是 primary claim。

执行时任一 identifiable stratum 的 solvable n < 200 → UNKNOWN，不是 PASS。

---

## 9. Independence

```text
cluster identity = physical_event_uid + input_locator
```

一个 underlying event 只贡献一个 independent statistical unit。以后重复跑 solver **不**创造新 replica。

---

## 10. 未来执行流（本 workbook 不执行）

```text
WB84 physical measurements
    ↓
truth association labels only
    ↓
common-track solve on the frozen chart
    ↓
apply left-SE(3) alignment update
    ↓
physical Calypso refit / ACTS repropagation
    ↓
relinearize
    ↓
independent validation
```

必须是真实 Calypso/ACTS chain。

---

## 11. 输出

`outputs/mc24_four_station_wb85_physical_alignment_protocol_v1/`

```text
wb85_physical_alignment_protocol.json
global_statistical_gate.json
mode_chart_contract.json
power_analysis.json
multiplicity_contract.json
qualification_schema.json
```

全部：

```text
executable = false
qualification_authorized = false
```

---

## 12. 最终问题

> 当前 WB84 physical corpus 是否已经配套一个前瞻、multiplicity-aware、fail-closed 的 physical alignment qualification protocol？

**是。** Protocol freeze PASS。正式 HTCondor qualification 需要下一步单独授权 `qualification_authorized = true`。本 workbook 不执行 qualification。
