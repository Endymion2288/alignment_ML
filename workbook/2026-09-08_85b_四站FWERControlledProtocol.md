# Workbook 85b — FWER-Controlled Physical Alignment Qualification Protocol v2

日期： 2026-09-08
分支： `4station`
冻结 parent： `4station@fe00c0a9674b861f408e5c7e6c0d00163a42d116`
HEAD（本轮 QA 时）： `4station@fe2a97666d688e31380ff27f8550dd629631daea`
任务： 前瞻冻结固定 Bonferroni 的 qualification 统计协议，并在 synthetic-only joint-null / power 与 official production-runner smoke 上验收。不执行 WB86，不读取 WB84 physical alignment outcomes，不覆盖、不修改历史 WB85 / WB85a / WB85a-r1。

---

## 0. 结论（先写）

```text
wb85b_protocol_qa_pass                     = true
ready_to_authorize_physical_qualification = true

protocol_freeze_pass                       = true
joint_synthetic_null_calibrated            = true
power_validation_pass                      = true
power_requirement_pass                     = true
statistical_unit_contract_pass             = true
nuisance_covariance_pass                   = true
fail_closed_tests_pass                     = true
weak_jg_contract_pass                      = true
calypso_acts_engine_runtime_smoke_pass     = true
official_qualification_runner_e2e_smoke_pass = true
provenance_complete                        = true
implementation_or_null_model_inconsistency = false

qualification_authorized                      = false
executable                                    = false
looked_at_physical_alignment_outcomes         = false
alignment_oracle_qualified_for_physical_FASER = false
ml_alignment_eval_authorized                  = false
wb86_automatically_authorized                 = false
```

`ready_to_authorize_physical_qualification = true` 只表示本前瞻协议与 QA 已闭合。它**不是**授权。本轮停止时 qualification 仍未授权，也未执行。

未创建、未提交、未运行 WB86。

---

## 1. 历史语义：WB85 从未保证 qualification-level FWER = 0.05

不得回写或修改历史 WB85。准确记录：

```text
WB85:
  T_LS alpha = 0.05
  T_cov was an additional global gate
  no explicit qualification-level FWER <= 0.05 claim

WB85a-r1:
  prospectively imposed primary statistical FWER cap = 0.05
  before any physical alignment outcome was observed
  coherent joint-null measured union false-fail
      = 0.0874
      95% CP = [0.0835, 0.0914]
  therefore the old two-alpha=0.05 decision rule is not eligible
  under the new global statistical requirement
```

WB85a-r1 测得的

```text
corr(T_LS, T_cov)           = 0.405
corr(reject_LS, reject_cov) = 0.196
```

只作为历史 dependence 记录。WB85b **没有**据此反推相关性特定 threshold，也没有做 alpha allocation sweep。

历史 artifacts 保持不动：

| 目录 | 关键文件 | sha256 |
|---|---|---|
| `outputs/mc24_four_station_wb85_physical_alignment_protocol_v1/` | `wb85_physical_alignment_protocol.json` | `cfd016625b7573efebbfc86e067b583594d18b7ec21337035dd7ba2d27d0cefa` |
| `outputs/mc24_four_station_wb85a_protocol_qa_v1/` | `wb85a_acceptance_rule.json` | `56da725c8e599f7b790c4cc8d22901a130841b00aa01e23f664cb048ce65634d` |
| | `wb85a_protocol_qa.json` | `c05af1662d4171eed4a9de2036656c03768e5591c03249304d5256d2ca58a4b3` |
| `outputs/mc24_four_station_wb85a_r1_protocol_qa_v1/` | `wb85a_r1_acceptance_rule.json` | `ca86ef52416eb8287b46b28df6d6434daf46098781fd61b3f1a973a25bbc2d47` |
| | `wb85a_r1_protocol_qa.json` | `25ee78b28511b5716c7753c7caa313b70adaebaeaec833e1482c186e96615a0b` |
| `outputs/mc24_four_station_wb85a_r1_runtime_smoke_v1/` | `wb85a_r1_runtime_smoke.json` | `6c2d6a2a7ea6ceabdd0c35e94707ca4f3ac4fa99a007edf08c3e0d7a2f3acaa2` |

---

## 2. 冻结 statistical family：固定 Bonferroni

Primary statistical family 只有：

```text
H_LS  : projected-z location/scale calibration
H_cov : projected 95% interval coverage calibration
```

```text
familywise_alpha = 0.05
alpha_LS         = 0.025
alpha_cov        = 0.025
```

因此在任意 dependence 下：

```text
FWER <= alpha_LS + alpha_cov <= 0.05
```

冻结 critical values（写盘后未改）：

```text
T_LS  ~ chi2(df=12)
critical_LS  = chi2.ppf(0.975, 12) = 23.33666415864534

T_cov ~ chi2(df=6)
critical_cov = chi2.ppf(0.975, 6)  = 14.44937533544792
```

Primary statistical FAIL：

```text
T_LS > critical_LS
OR
T_cov > critical_cov
```

统计量复用冻结的 `location_scale_gof` / `coverage_calibration`，但**忽略**它们内部 α=0.05 的 `accepted` flag。决策函数是 `evaluate_wb85b_qualification`。

协议与验收规则在任何新 Monte-Carlo 之前写盘：

| artifact | sha256 |
|---|---|
| `outputs/mc24_four_station_wb85b_fwer_protocol_v1/wb85b_fwer_protocol.json` | `1bac20ffdd26b7fbc4cd912ce5d75db266683b279e860837eab55f0da519af21` |
| `outputs/mc24_four_station_wb85b_fwer_protocol_v1/wb85b_acceptance_rule.json` | `a3adf56709a474ac963da3d45ff627b562465127e3146d642f99916905e83427` |

这两份哈希在 MC 前后相同。

---

## 3. FWER claim 的精确范围

不得写无条件：

```text
overall qualification false-fail <= 0.05
```

必须写成：

```text
primary_statistical_FWER_cap              = 0.05
conditional_on_valid_analyzable_execution = true
```

含义：在

```text
all required strata exist
solvable n >= 200
finite outputs
converged according to frozen contract
SPD/FD/execution prerequisites satisfied
```

的条件下，对 fully calibrated statistical null：

```text
P(false statistical FAIL from H_LS or H_cov) <= 0.05
```

Engineering / data-quality guardrails 不是这 0.05 family 的成员：

```text
missing stratum
n < 200
nonconvergence
NaN/Inf
non-SPD
FD failure
dropped identifiable mode
runtime/provenance failure
```

它们仍按原 frozen fail-closed / UNKNOWN semantics 独立处理。

---

## 4. 其余 WB85 physical contract 全冻结

未改：WB84 corpus membership、geometry families、survey tags、projected scientific modes、weak-JG status、common-track solver objective、q/p prior、Schur、left-SE(3)、max iterations、Calypso SegmentFitRefit、ACTS mode-0、relinearization、catastrophic engineering thresholds、0.1 mm / 1 mrad research-screening guards、weak `|mean z| > 5` catastrophic guard、truth-only association。

未新增模型、训练、association 方法或 physical dataset。

`CalypsoActsPhysicalBackend.execute` 保持 `false`。

---

## 5. Joint-null QA（implementation verification，不是 FWER 证明）

构造与 WB85a-r1 相同：

```text
z
→ covered_95 = I(|z| <= norm.ppf(0.975))
```

`n_mc = 20000`，seed `2026090820`，stratum counts `299, 303, 311, 298, 300, 283`。前 200 次调用 `evaluate_wb85b_qualification`，mismatches = 0。FWER 的 primary mathematical guarantee 来自 Bonferroni，不依赖模拟相关系数。

预先冻结：若 union 95% CP 下界 > 0.05，则

```text
implementation_or_null_model_inconsistency = true
ready_to_authorize_physical_qualification = false
```

不得调 threshold 让它 PASS。

| gate | Type-I | 95% CP | target | 裁决 |
|---|---:|---|---:|---|
| T_LS | 0.02545 | [0.02331, 0.02773] | 0.025 | PASS |
| T_cov | 0.02655 | [0.02437, 0.02887] | 0.025 | PASS |
| union | 0.04720 | [0.04430, 0.05023] | Bonferroni cap 0.05 | 与控制兼容 |

```text
corr(T_LS, T_cov)           = 0.399
corr(reject_LS, reject_cov) = 0.163
union CP lower bound        = 0.0443  <= 0.05
implementation_or_null_model_inconsistency = false
thresholds_not_retuned_from_this_simulation = true
```

Union 点估计 0.0472 < 0.05。CP 上界略高于 0.05 不构成 inconsistency；primary guarantee 仍是 Bonferroni，不是这组相关系数。

```text
joint_synthetic_null_calibrated = true
```

---

## 6. Power：在协议写盘并 hash 之后重新前瞻验证

未假定旧 α=0.05 的 power 仍成立。对照 n = 200，`n_mc = 4000`，seed `2026090821`。

必须满足 power ≥ 0.80 的 alternatives 只有 mean-shift 0.35 与 scale 1.25。scale deflation 0.763 只报告，不是 80% 要求。

| alternative | analytic approximation | MC empirical | 95% CP | required | 裁决 |
|---|---:|---:|---|---|---|
| one-stratum mean 0.35 | 0.89423 | 0.8955 | [0.8856, 0.9048] | yes | PASS |
| one-stratum scale 1.25 | 0.96723 | 0.90875 | [0.8994, 0.9175] | yes | PASS |
| scale deflation 0.763 | 0.71304 | 0.7685 | [0.7551, 0.7815] | no | 记录；不要求 ≥0.80 |
| gross coverage p=0.50 | — | 1.000 | [0.9991, 1.000] | yes | PASS |

mean-shift analytic sanity check 由冻结实现重新产生：`power_mean_wb85b(0.35, n=200) = 0.8942286683713147`。

scale=1.25：analytic 偏乐观（0.967 vs MC 0.909），与 WB85a-r1 同类 discrepancy。未事后改 alpha allocation、alternative 或 sample-size claim。

```text
power_requirement_pass = true
power_validation_pass  = true
```

---

## 7. WB85a-r1 runtime provenance 修正

WB85a-r1 smoke 实际运行时：

```text
runtime_parent_commit              = fe00c0a9674b861f408e5c7e6c0d00163a42d116
runtime_worktree_was_dirty_untracked = true
runtime_source_snapshot_sha256     = 1a5e0f3f75e6a9e399930756f9572b3abbb1e35459577c3772cd1180b197a540
```

不得把事后 commit `582316b662fe9f5d7ad1bd1f5957a3258af7e6bf` 称为 runtime generating commit。

已证明该 post-run freeze tree 含有与 frozen source snapshot 字节相同的 generating sources：

```text
post_run_source_freeze_commit = 582316b662fe9f5d7ad1bd1f5957a3258af7e6bf
post_run_freeze_contains_byte_identical_generating_sources = true
```

`origin/4station` 仍停在 `fe00c0a…`。`582316b` 与 `fe2a976` 均未 push。

```text
remote_commit_verification = false
```

不能假装远端 provenance 已验证。

---

## 8. Exact production-runner smoke

未来 qualification **不会**复用 `execute_runtime_smoke` 作为 top-level executor。

```text
official_runner = WB85bOfficialCalypsoActsBackend.execute_refit
official_runner_uses_smoke_tested_executor    = false
official_runner_uses_smoke_tested_primitives  = true
therefore_dedicated_official_runner_e2e_smoke_is_required = true
calypso_acts_physical_backend_execute_stays_false = true
```

因此：

```text
calypso_acts_engine_runtime_smoke_pass       = true   # 历史 WB85a-r1
official_qualification_runner_e2e_smoke_pass = true   # 本轮新 smoke
```

本轮在合法 fixture 上跑了 exact production runner：

```text
source_id   = mc24_100047_00100_00149
skip_events = 0
nevents     = 1
below WB84 skip_base=2000
not in qualification allocation
```

Condor cluster `1112325`，schedd `bigbird24.cern.ch`，host `b9g03p3124.cern.ch`，return code 0，execute 961 s。输出目录：`outputs/mc24_four_station_wb85b_official_runner_smoke_v1/`。

`alignment_payload.json` 证实 iteration 2 的 sqlite payload 含事先登记的

```text
S1 dx = +0.05 mm
station:1[0] = 0.05
```

iteration 2 的 rerefit command 消费的是 **更新后的** sqlite，不再指向 iteration 1 sqlite。两次 sqlite / enhanced / tracklets / stacked digest 均不同。没有 cached first-step measurements，没有 lab-only transport，没有 `toy_uniform_By`。

| 量 | iteration 1 (identity) | iteration 2 (S1 dx = +0.05 mm) |
|---|---|---|
| payload S1 dx (mm) | 0.0 | 0.05 |
| sqlite sha256 | `fc0b190a…ae492daf8` | `27558948…64e23987` |
| enhanced sha256 | `33cb6004…accdfbf33` | `89908c29…0783ba30` |
| tracklets sha256 | `deb7ac62…24e35682` | `f5f7b340…d117407e` |
| stacked digest | `45b7cac0…986b81f` | `0487d2e7…67e3028` |
| payload / refit / export rc | 0 / 0 / 0 | 0 / 0 / 0 |
| `/Tracker/Align` sqlite override | true | true |
| FaserActsAlignment | true | true |

Relinearization 只记录 `solver_status=ok`、`n_matched_hits=4`。**未报告 alignment performance。** 这不是 physics qualification。

smoke report sha256：`286c13e4360a546d079cd3d45f073c5800133674413d33f112839bec3b7083d8`

---

## 9. WB85b generating-code identity

本轮 QA 的 generating-code snapshot（不是 parent SHA）：

```text
source_snapshot_sha256 = 3945d7065c11b007292bbcd3253028dc1f435f0eee8c449c6c363137ddaccbde
wb85b_fwer_protocol.py = 757955097f41a5598e01f0e86f2128224b68bf0c3a199161c14936f42edbe756
wb85b_protocol_qa.py   = b67d797ea6f15d95f201955843327c02613e0c8a2112b8b34e21079f7f650616
wb85b_official_runner.py = 6cc919202be3cabf4e61df499231593f4f830bd83116d3ab73db2e5441f0babe
```

运行时：

```text
runtime_parent_commit          = fe2a97666d688e31380ff27f8550dd629631daea
runtime_worktree_was_dirty     = true   # WB85b sources 仍为 untracked
post_run_source_freeze_commit  = (not created in this workbook)
remote_commit_verification     = false
```

`outputs/*` gitignored。WB85b 源文件尚未 commit。不能把 `fe2a976` 或 `582316b` 说成 WB85b 的 runtime generating commit。

---

## 10. 最终状态

前瞻 protocol freeze、synthetic joint-null QA、power QA、provenance、exact production-runner contract 全部 PASS，因此：

```text
wb85b_protocol_qa_pass                     = true
ready_to_authorize_physical_qualification = true
```

即便如此仍保持：

```text
qualification_authorized = false
executable               = false
wb86_automatically_authorized = false
```

本轮停止。未创建 WB86。
