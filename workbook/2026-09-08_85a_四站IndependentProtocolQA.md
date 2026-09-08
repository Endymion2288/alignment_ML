# Workbook 85a — Independent Protocol QA and Execution-Readiness Audit

日期： 2026-09-08
分支： `4station`
基线： `4station@fe00c0a9674b861f408e5c7e6c0d00163a42d116`
任务： 在不看 WB84 physical alignment outcomes、不执行 physical qualification 的前提下，证明 WB85 冻结的 global statistical protocol 和 future physical execution contract 是统计校准、实现正确、fail-closed 的。

---

## 0. 结论（先写）

```text
wb85_protocol_qa_pass                      = true
ready_to_authorize_physical_qualification  = true

qualification_authorized                   = false
executable                                 = false
looked_at_physical_alignment_outcomes      = false
alignment_oracle_qualified_for_physical_FASER = false
ml_alignment_eval_authorized               = false
wb86_automatically_authorized              = false
```

| check | 结果 |
|---|---|
| synthetic_null_calibrated | true |
| power_validation_pass | true |
| statistical_unit_contract_pass | true |
| nuisance_covariance_pass | true |
| fail_closed_tests_pass | true |
| weak_jg_contract_pass | true |
| physical_execution_ready | true |
| provenance_complete | true |

未跑 WB84 corpus 上的正式 alignment。未打开 physical alignment outcomes。未改 WB83 / WB84 artifacts。未自动提交 WB86。

下一步若单独授权，只能执行已经冻结并通过 QA 的 WB85 protocol：

```text
WB86 — Physical Common-Track Alignment Qualification v1
```

WB86 不得提出新模型、新门槛或新数据。

---

## 1. Synthetic null calibration

验收规则在跑 Monte-Carlo **之前**冻结，见 `wb85a_acceptance_rule.json`。只用 WB84 identifiable-stratum counts，合成

```text
z_ir ~ N(0,1)
coverage_ir ~ Bernoulli(0.95)
```

不得读取 physical alignment output。`n_mc = 20000`，seed `2026090817`。

| gate | Type-I | 95% CP | target | 裁决 |
|---|---:|---|---:|---|
| T_LS location-scale | 0.0518 ± 0.0016 | [0.0488, 0.0550] | 0.05 | PASS |
| T_cov coverage | 0.04985 ± 0.0015 | [0.0469, 0.0530] | 0.05 | PASS |
| combined final gate | 0.0990 ± 0.0021 | [0.0949, 0.1032] | 0.0975 | PASS |

combined 的名义目标是两个独立 α=0.05 gate 的并集 `1-(1-α)²`，不是 0.05。200 次官方函数交叉核对 mismatches = 0。

```text
synthetic_null_calibrated = true
```

---

## 2. Power injection

对照 WB85 冻结 analytic（保守 n=200）。未按 WB83 remaining failures 调 alternative。`n_mc = 4000`，seed `2026090818`。

| alternative | analytic | MC | 80%? | 裁决 |
|---|---:|---:|---|---|
| mean shift 0.35，一个 identifiable stratum | 0.936 | 0.939 [0.931, 0.946] | yes | PASS |
| scale 1.25，一个 identifiable stratum | 0.983 | 0.937 [0.929, 0.944] | yes | PASS |
| scale deflation 0.763（WB85 80% MDE） | 0.800 | 0.864 [0.853, 0.874] | yes | PASS |
| gross coverage p=0.50 | — | 1.000 | — | PASS |

scale 1.25 的 ncx2 解析功效略乐观（Δ≈0.046），仍在冻结容差 0.08 内，且 MC 与 analytic 都 ≥ 0.80。这不是根据 physical outcome 改门槛。

```text
power_validation_pass = true
```

---

## 3. Statistical-unit contract

正式 result schema：

```text
one underlying physical event
→ one preregistered projected z
per identifiable stratum
```

```text
a_hat = u^T theta_hat
sigma_a^2 = u^T Cov(theta_hat) u
z = (a_hat - a_true) / sigma_a
```

同一 event 的多个 correlated pose coordinates 不得当作独立 z。unit tests 拒绝 duplicate event-stratum 与长度为 6 的 pose-z 向量。

```text
statistical_unit_contract_pass = true
```

---

## 4. Nuisance covariance

`finite_survey_prior` 下 dz + 5 mm survey prior 的目标 projected mode covariance：

```text
joint solve  = Schur/marginal = projected sigma
```

三者一致；无 prior 的 dense joint 与有 prior 的 solver covariance 不同。dz 不是 projected scientific mode。

```text
nuisance_covariance_pass = true
```

---

## 5. Global gate implementation QA

全部 fail-closed：

| 路径 | 状态 |
|---|---|
| missing stratum | FAIL |
| n < 200 | UNKNOWN |
| NaN / Inf | FAIL |
| nonconvergence > 5% | FAIL |
| unresolved non-SPD | FAIL |
| FD > 1% | FAIL |
| large engineering bias | FAIL |
| gross coverage failure | FAIL |
| dropped identifiable mode / column | FAIL |
| weak catastrophic standardized bias | FAIL |

Holm diagnostics 只在 primary FAIL 后输出，`may_reselect_model = false`。

```text
fail_closed_tests_pass = true
```

---

## 6. Weak-JG contract

`weak_jg_diagnostic` 不能进入 primary T_LS。加入 primary 会 fail-closed。catastrophic guardrail（|mean z| > 5）仍然约束 weak mode。禁止在看到 physical result 后把 weak mode 加入或移出 primary。

```text
weak_jg_contract_pass = true
```

---

## 7. Physical execution-readiness

未在 WB84 corpus 上执行正式 alignment。补齐并审计了 future official runner：

```text
WB84 physical measurement
→ truth association
→ common-track solve
→ left-SE(3) update
→ Calypso SegmentFitRefit
→ ACTS mode-0 repropagation
→ relinearization
```

官方 engine = `calypso_segmentfit_acts_mode0`。禁止 fallback 到 `toy_uniform_By`、fixed first-step measurements、lab-only transport。`iterate_common_track(..., field_y=...)` 不是 official engine。WB85a 只规划 Calypso 命令，不执行。

```text
physical_execution_ready = true
execution_ready          = true
executable               = false
```

---

## 8. Provenance freeze

`outputs/mc24_four_station_wb85a_protocol_qa_v1/`

```text
wb85a_acceptance_rule.json
wb85a_protocol_qa.json
wb85a_synthetic_calibration.json
wb85a_power_validation.json
wb85a_execution_readiness.json
wb85a_provenance_freeze.json
```

记录了 git SHA `fe00c0a9674b861f408e5c7e6c0d00163a42d116`、WB85 Python/YAML SHA256、`common_track_solver` SHA256、WB84 corpus manifest / QA / snapshot / Calypso source bundle SHA256、field-map hashes、geometry payload hashes。WB84 snapshot 里 material hashed_files 为空，按原样记录，不重算、不改 WB84。

```text
provenance_complete = true
```

---

## 9. 最终问题

> 不看 physical outcomes，WB85 冻结的 global statistical protocol 和 future physical execution contract 是否已经统计校准、实现正确、fail-closed，并且可以单独授权正式 physical qualification？

**是。** Protocol QA PASS。正式 qualification 仍保持

```text
qualification_authorized = false
```

需要下一步单独创建并授权 WB86。WB86 只能跑这份已冻结并通过 QA 的 protocol。
