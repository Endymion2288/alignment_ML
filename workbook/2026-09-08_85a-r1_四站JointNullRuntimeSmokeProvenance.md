# Workbook 85a-r1 — Joint-Null Calibration, Runtime Smoke Test, and Provenance Closure

日期： 2026-09-08
分支： `4station`
冻结 parent： `4station@fe00c0a9674b861f408e5c7e6c0d00163a42d116`
生成代码 commit： `4station@582316b662fe9f5d7ad1bd1f5957a3258af7e6bf`
任务： 修正 WB85a 把 `z` 与 coverage 当成独立合成量的 joint-null，补上官方 Calypso/ACTS runtime smoke，并冻结实际产生 QA 的 generating-code identity。不执行 WB86，不读取 WB84 physical alignment outcomes，不覆盖历史 WB85a artifact。

---

## 0. 结论（先写）

```text
joint_synthetic_null_calibrated            = true
power_validation_pass                      = true
statistical_unit_contract_pass             = true
nuisance_covariance_pass                   = true
fail_closed_tests_pass                     = true
weak_jg_contract_pass                      = true
physical_execution_contract_pass           = true
physical_execution_runtime_smoke_pass      = true
provenance_complete                        = true

protocol_revision_required                 = true
wb85_protocol_qa_pass                      = false
ready_to_authorize_physical_qualification  = false

qualification_authorized                   = false
executable                                 = false
looked_at_physical_alignment_outcomes      = false
alignment_oracle_qualified_for_physical_FASER = false
ml_alignment_eval_authorized               = false
wb86_automatically_authorized              = false
```

九项 E 门全部为 true，但冻结的 two-gate 并集无法把 overall qualification false-fail 控制在 0.05。这是事先写下的 revision 规则，不是事后调 α。WB85 physical thresholds 未改。未创建、未提交、未运行 WB86。

历史 WB85a 的 `0.0975 = 1-(1-0.05)²` 只保留为 “independent synthetic gates expectation”，**不是** official joint-null target。

---

## 1. 历史 WB85a 保持不动

`outputs/mc24_four_station_wb85a_protocol_qa_v1/` 未覆盖、未删除、未改写。SHA256：

| artifact | sha256 |
|---|---|
| wb85a_acceptance_rule.json | `56da725c8e599f7b790c4cc8d22901a130841b00aa01e23f664cb048ce65634d` |
| wb85a_protocol_qa.json | `c05af1662d4171eed4a9de2036656c03768e5591c03249304d5256d2ca58a4b3` |
| wb85a_synthetic_calibration.json | `7526ec7d035090eb772254abc8cfd803a3a9436d21dbf104b5ec56e7acef999a` |
| wb85a_power_validation.json | `8a1ade10bbaa0f01042fa8fc015078e54d3834a6b06dc5752af41893fcb0f282` |
| wb85a_execution_readiness.json | `f0cf046488179bfec0c3bca1444ee720e82fc91ba94818e373ac249f38945dc5` |
| wb85a_provenance_freeze.json | `1a79d01803d7904e096f7b702ea02d1a3654205da88600f2f202959acbab8fd7` |

WB85a 独立合成 `z ~ N(0,1)` 与 `coverage ~ Bernoulli(0.95)`，并把 combined Type-I 对到 `0.0975`。那一轮可以证明两个 **独立** gate 各自校准，但不能作为正式 combined qualification null。

---

## 2. 验收规则先于任何新 Monte-Carlo 冻结

规则文件：`outputs/mc24_four_station_wb85a_r1_protocol_qa_v1/wb85a_r1_acceptance_rule.json`，在 joint-null 开始前写入。

冻结要点：

```text
counts = 299, 303, 311, 298, 300, 283
coverage = official symmetric Gaussian 95% CI
covered_ir = I(|z_ir| <= norm.ppf(0.975)) = 1.959963984540054
official_joint_target_is_independent_union_0.0975 = false
scientific_overall_qualification_false_fail_cap = 0.05
may_retune_wb85_thresholds_from_this_qa = false
may_silently_change_alpha = false
```

revision 规则（写在看到本次数值之前）：

```text
protocol_revision_required if
  union CP lower bound > 0.05
  or (union point > 0.05 and the 95% CP does not contain 0.05)
```

Independence 未被假设。官方 coverage 是同一 `z` 的函数，因此 `(T_LS, T_cov)` 依构造相关。

---

## 3. Corrected joint-null table

生成器：synthetic `theta_hat` + 对角 SPD `Cov`，再走 official

```text
a_hat = u^T theta_hat
sigma_a^2 = u^T Cov u
z = (a_hat - a_true) / sigma_a
covered_95 = I(|z| <= norm.ppf(0.975))
```

同一 realization 调用 `location_scale_gof`、`coverage_calibration`、`evaluate_future_qualification`（200 次官方三函数交叉核对 mismatches = 0）。`n_mc = 20000`，seed `2026090819`。未读 physical alignment output。

| gate | Type-I | 95% CP | target | 裁决 |
|---|---:|---|---:|---|
| T_LS location-scale | 0.0498 ± 0.0015 | [0.0468, 0.0529] | 0.05 | PASS |
| T_cov coverage | 0.0493 ± 0.0015 | [0.0463, 0.0524] | 0.05 | PASS |
| joint / union false-fail | 0.0874 ± 0.0020 | [0.0835, 0.0914] | **measured; not 0.0975** | 超过 overall cap 0.05 |

```text
corr(T_LS, T_cov)           = 0.405
corr(reject_LS, reject_cov) = 0.196
historical independent-union expectation (WB85a only) = 0.0975
official joint target is not 0.0975
```

并集点估计 0.0874，CP 下界 0.0835 > 0.05。冻结 two-gate 程序不能满足 overall qualification false-fail ≤ 0.05。

```text
joint_synthetic_null_calibrated = true
protocol_revision_required      = true
```

未据此回调 WB85 threshold，也未静默改 α。

---

## 4. Individual gate / power QA（复现，证明 gate 实现未变）

同一冻结函数、同一 WB85a seed。独立 null（`z ~ N(0,1)` 与 `Bernoulli(0.95)`）完全复现历史 WB85a：

| gate | Type-I | 95% CP | target | 裁决 |
|---|---:|---|---:|---|
| T_LS | 0.0518 | [0.0488, 0.0550] | 0.05 | PASS |
| T_cov | 0.04985 | [0.0469, 0.0530] | 0.05 | PASS |
| independent-union（历史期望） | 0.0990 | [0.0949, 0.1032] | 0.0975 | 仅作历史对照 |

Power，`n_mc = 4000`，seed `2026090818`，对照 n=200：

| alternative | analytic | MC | 标记 | 裁决 |
|---|---:|---:|---|---|
| mean 0.35，一个 identifiable stratum | 0.936 | 0.939 [0.931, 0.946] | analytic ≈ approximation | PASS |
| scale 1.25，一个 identifiable stratum | 0.983 | 0.937 [0.929, 0.944] | **discrepancy 原样报告** | PASS |
| scale deflation 0.763 | 0.800 | 0.864 [0.853, 0.874] | MC = empirical QA | PASS |
| gross coverage p=0.50 | — | 1.000 | — | PASS |

scale=1.25：analytic ≈ 0.983，MC ≈ 0.937。ncx2 偏乐观。未回调 alternative 或 tolerance。

```text
power_validation_pass = true
```

---

## 5. Official physical-engine runtime smoke

命令规划只能证明 `physical_execution_contract_pass`。本轮 HTCondor 实际跑了：

```text
input xAOD
→ Calypso SegmentFitRefit
→ ACTS mode-0
→ parsed new measurements
→ common-track relinearization
→ preregistered S1 dx = +0.05 mm geometry update
→ second Calypso/ACTS iteration
```

合法 fixture：`mc24_100047_00100_00149`，`skip_events=0`，`nevents=1`。低于 WB84 `skip_base=2000`，不在 qualification allocation，不是 00350 / 00800 / 100116/117 / Final Blind / sealed。

Condor：cluster `1112282`，schedd `bigbird24.cern.ch`，host `b9g03p3557.cern.ch`，return code 0，execute 571 s。

| 量 | iteration 1 (identity) | iteration 2 (S1 dx = +0.05 mm) |
|---|---|---|
| sqlite sha256 | `538021ba…8f71b6f8` | `54184d8b…6f833138` |
| enhanced sha256 | `fae2d2e7…f2a6a20c` | `cfb5e110…0a46ae18` |
| tracklets sha256 | `75dc6a69…e8fee26d` | `d1a564ae…4bb07314` |
| stacked measurement digest | `45b7cac0…986b81f` | `0487d2e7…67e3028` |
| payload / refit / export rc | 0 / 0 / 0 | 0 / 0 / 0 |
| `/Tracker/Align` sqlite override | true | true |
| FaserActsAlignment | true | true |

iteration 2 的 rerefit command 指向 **更新后的** sqlite，不再指向 iteration 1 sqlite。两次 enhanced / tracklets 路径与哈希都不同：没有 cached first-step measurements，没有 lab-only transport，没有 `toy_uniform_By`，没有 silent fallback。`CalypsoActsPhysicalBackend.execute` 保持 false；smoke 是独立 executor。

Relinearization 只记录 `solver_status=ok`、`n_matched_hits=4`。**未报告 alignment performance。** 几何更新是事先登记的 smoke probe，不是 recovered alignment。这不是 physics qualification。

```text
physical_execution_runtime_smoke_pass = true
```

该 smoke 只证明 runtime plumbing。

---

## 6. Provenance closure

`fe00c0a…` 不含 WB85a-r1 实现。Result JSON 引用的是实际 generating-code SHA256，而不只是 parent SHA。

运行时 worktree dirty（全部为未跟踪源文件；`git diff --binary` 对已跟踪文件为空，SHA256 = `e3b0c442…` 空 blob）。同时记录：

```text
git HEAD              = fe00c0a9674b861f408e5c7e6c0d00163a42d116
git status --porcelain = untracked WB85a / WB85a-r1 sources
git diff --binary      = empty vs tracked tree
source snapshot SHA256 = 1a5e0f3f75e6a9e399930756f9572b3abbb1e35459577c3772cd1180b197a540
```

冻结的 generating sources 包括：WB85a QA、joint-null generator、power validation、statistical-unit / nuisance / fail-closed tests、official planner、runtime smoke、acceptance-rule config、Condor submit/worker、environment manifest。

WB84 snapshot 的 `material_map.hashed_files` 保持历史空字段，不回填、不改 WB84。Future WB86 runtime provenance **必须**在运行时识别实际加载的 geometry / material / conditions / field / Calypso / ACTS，而不能依赖这份空历史字段。

---

## 7. Final gate

E 门全 true，但

```text
protocol_revision_required = true
```

因此

```text
wb85_protocol_qa_pass                     = false
ready_to_authorize_physical_qualification = false
```

即使九项全过，本轮也必须保持：

```text
qualification_authorized                      = false
executable                                    = false
alignment_oracle_qualified_for_physical_FASER = false
wb86_automatically_authorized                 = false
```

下一步若要继续，只能开一份 **新的 prospective protocol**（新 workbook），不得静默改当前 WB85 α 或门槛。本轮停止。未创建 WB86。
