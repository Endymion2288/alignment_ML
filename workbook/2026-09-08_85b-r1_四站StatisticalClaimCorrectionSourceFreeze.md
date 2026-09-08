# Workbook 85b-r1 — Statistical Claim Correction and Source-Freeze Closure

日期： 2026-09-08
分支： `4station`
任务： 纯 closure。修正 WB85b FWER 措辞，并把产生 WB85b QA / power / joint-null / official-runner smoke 的 generating sources 原样冻结。不执行 WB86，不读取 WB84 physical alignment outcomes，不修改 WB85b thresholds、alpha allocation、power alternatives、MC results、solver、runner 或 physical contract。不重跑 Monte-Carlo，不重跑 production smoke。

---

## 0. 结论（先写）

```text
generating_code_identity_complete = true
generating_code_archive_complete  = true
provenance_complete               = true

wb85b_protocol_qa_pass = true

nominal_statistical_protocol_qualified = true
exact_finite_sample_FWER_proven        = false
finite_sample_FWER_mathematically_proven = false

ready_to_authorize_physical_qualification = true

qualification_authorized                      = false
executable                                    = false
alignment_oracle_qualified_for_physical_FASER = false
ml_alignment_eval_authorized                  = false
wb86_automatically_authorized                 = false
looked_at_physical_alignment_outcomes         = false
```

Decision rule 未改。未创建、未提交、未运行 WB86。

---

## 1. FWER claim 修正，不改变 decision rule

保留：

```text
alpha_LS     = 0.025
alpha_cov    = 0.025
critical_LS  = 23.33666415864534
critical_cov = 14.44937533544792
```

明确区分 Bonferroni 不等式与 nominal individual size：

```text
Bonferroni inequality:
P(A union B) <= P(A) + P(B)

nominal individual size = 0.025
```

不得再无条件声称：

```text
finite_sample_FWER_mathematically_proven = true
```

准确写法：

```text
nominal_primary_statistical_FWER_cap = 0.05

FWER <= 0.05 is conditional on
actual individual gate sizes being <= 0.025.

joint-null MC (cited, not regenerated):
T_LS type-I  = 0.02545 [0.02331, 0.02773]
T_cov type-I = 0.02655 [0.02437, 0.02887]
union         = 0.04720 [0.04430, 0.05023]

individual_gate_nominal_calibration_compatible = true
joint_null_nominal_FWER_compatible             = true
exact_finite_sample_FWER_proven                = false
```

这是措辞修正。未据此重新调 alpha、threshold 或重新选择统计量。未产生新的 model-selection information。

WB85b 的 `fwer_bound = FWER <= alpha_LS + alpha_cov <= 0.05 under arbitrary dependence` 应读作 Bonferroni 不等式，而不是“有限样本 FWER 已被无条件数学证明”。

---

## 2. Source-freeze：byte-identical

WB85b 运行时记录：

```text
runtime_parent_commit          = fe2a97666d688e31380ff27f8550dd629631daea
runtime_worktree_was_dirty     = true
runtime_source_snapshot_sha256 = 3945d7065c11b007292bbcd3253028dc1f435f0eee8c449c6c363137ddaccbde
```

不得把事后 freeze commit 称为 runtime generating commit。

已创建 source-freeze commit，并且 `git show <freeze>:<file>` 对全部 14 个 generating sources 与运行时 SHA256 字节相同：

```text
post_run_source_freeze_commit = 6e065aeed01a6bdf353baea0565b59dd86966d90
source_freeze_tree_sha        = fdf4da5de4450afa8d5f48d8560c25efe987fc1f

post_run_freeze_contains_byte_identical_generating_sources = true

frozen_source_snapshot_sha256
==
runtime_source_snapshot_sha256
==
3945d7065c11b007292bbcd3253028dc1f435f0eee8c449c6c363137ddaccbde
```

本地不可变 archive：

```text
outputs/mc24_four_station_wb85b_r1_claim_closure_v1/wb85b_generating_sources.tar
archive_sha256 = 8d19f12fc7536db7194ff21c7e5f4d64719a8b19cf03852c4ecb6245e1f71440
n_members = 14
deterministic_mtime_uid_gid = true
```

`origin/4station` 在本 workbook 写成时仍可能停在旧 tip。若尚未确认 `6e065ae` 是 `origin/4station` 的 ancestor：

```text
remote_commit_verification = false
```

本地 commit object 与 source archive 都可恢复这 14 个文件。

---

## 3. 不重跑 production smoke，不重跑 Monte-Carlo

WB85b official runner smoke 保持原结果与原 hash：

```text
official_runner =
alignment.wb85b_official_runner.WB85bOfficialCalypsoActsBackend.execute_refit

official_qualification_runner_e2e_smoke_pass = true
smoke sha256 = 286c13e4360a546d079cd3d45f073c5800133674413d33f112839bec3b7083d8
alignment_performance_not_reported = true
is_physics_qualification = false
```

WB85b QA artifacts 哈希未改：

| artifact | sha256 |
|---|---|
| `wb85b_fwer_protocol.json` | `1bac20ffdd26b7fbc4cd912ce5d75db266683b279e860837eab55f0da519af21` |
| `wb85b_acceptance_rule.json` | `a3adf56709a474ac963da3d45ff627b562465127e3146d642f99916905e83427` |
| `wb85b_joint_null_calibration.json` | `a39df2d0c1e918a505d745feacc38a907c08e966f06085cf3c1a5aa42243a4b2` |
| `wb85b_power_validation.json` | `00166f65a067475b44261df269a4c7a3237729da77f2ca88363747bfc9f8bcf4` |
| `wb85b_protocol_qa.json` | `14fc6a3da4fbe76f88c803089619712b1e75aaef3aed3e9d1edec62d44914c22` |

历史 WB85 / WB85a / WB85a-r1 目录未覆盖。

---

## 4. 最终 gate

byte-identical source freeze 成功，因此：

```text
generating_code_identity_complete = true
generating_code_archive_complete  = true
provenance_complete               = true
wb85b_protocol_qa_pass            = true
nominal_statistical_protocol_qualified = true
exact_finite_sample_FWER_proven   = false
ready_to_authorize_physical_qualification = true
```

仍必须：

```text
qualification_authorized = false
executable               = false
```

本轮停止。未创建 WB86。
