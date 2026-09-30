# Workbook 86 — Physical Common-Track Alignment Qualification v1

日期： 2026-09-08
分支： `4station`
HEAD： `4station@3bd3073e3ffe66a74aed5dc55a6b551853d7dc98`
冻结 generating sources： `4station@6e065aeed01a6bdf353baea0565b59dd86966d90`
任务： execution-only。在冻结 WB84 语料、冻结 WB85b 判决函数、冻结 common-track solver、以及官方 Calypso/ACTS runner 上，对 truth-associated common-track alignment 做预注册 physical oracle。不发明新模型、新协议、新阈值、新数据集或新统计量。

---

## 0. 结论（先写）

```text
remote_commit_verification = true
qualification_authorized   = true
executable                 = true

SNAPSHOT_VERIFY = PASS
CORPUS_VERIFY   = PASS
n_events        = 2394

looked_at_physical_alignment_outcomes_before_authorization = false
alignment_oracle_qualified_for_physical_FASER              = false
ml_alignment_eval_authorized                               = false
```

`alignment_oracle_qualified_for_physical_FASER` 在本 workbook 写成时仍为 `false`。

第一轮 DAG `1112463` **没有**科学闭合。`SNAPSHOT_VERIFY` 与 `CORPUS_VERIFY` PASS 后，`PHYSICAL_ALIGNMENT_SHARDS` 在 `1112466.0` 因执行层 bug 以 rc=1 失败：`load_replica_record` 误把只读 WB84 `replicas.jsonl` 当成 rewrite 拒绝。DAGMan 随即 abort 其余 shard（仅约 35 个 technical FAIL 落盘，无科学 SUCCESS）。这不是 alignment 结果，不得据此调协议。

已修：只读 ingest 不再调用 `refuse_wb83_wb84_write`；event job 在写出 technical 记录后 exit 0，避免单点 rc=1 杀掉整个 2394-proc node。从 `wb86.dag.rescue001` 重提，不重跑 snapshot/corpus，不改阈值。

本轮**没有**读取任何 sealed 科学字段，也没有计算 pull / bias / coverage 汇总。

判决函数只能是：

```text
alignment.wb85b_fwer_protocol.evaluate_wb85b_qualification
```

未复制 decision logic，未改 α，未改 critical values。

```text
alpha_LS     = 0.025
alpha_cov    = 0.025
critical_LS  = 23.33666415864534
critical_cov = 14.44937533544792
```

---

## 1. 远端 provenance prerequisite 已闭合

用户已确认 freeze commit 已 push。HTTPS fetch 后：

```text
origin/4station = 3bd3073e3ffe66a74aed5dc55a6b551853d7dc98

git merge-base --is-ancestor \
  6e065aeed01a6bdf353baea0565b59dd86966d90 \
  origin/4station
# exit code = 0
```

记录（未改任何 WB85b generating source / protocol / threshold / artifact）：

`outputs/mc24_four_station_wb85b_r1_remote_verification_v1/wb85b_r1_remote_commit_verification.json`

```text
remote_commit_verification = true
qualification_authorized   = false   # 该文件只是 prerequisite
```

WB86 在该 prerequisite PASS 之后才单独授权 execution。

---

## 2. 冻结输入，禁止替换为 latest

SNAPSHOT_VERIFY 对照的 expected SHA256：

| 输入 | sha256 |
|---|---|
| WB84 `physical_replica_manifest.json` | `05cf0c66f97b94ee1948fc5ad8cf00fad1b171832fa5586680eab3c712a8fca0` |
| WB84 `corpus_qa.json` | `1a9a9bc69ef2e02bcecf2a70a8bc184bb10c2e8a8b1c1b03275cb99743c95876` |
| WB84 `geometry_payload_manifest.json` | `df8361fb8760944f98c6b8b231c5c5b563d52680f09e1920583332cf5064ec12` |
| WB84 `production_snapshot.json` | `8706e244c850c5d1d97a7bd03779d1d94e5eb5a4c01681689da9d2e6aff17158` |
| WB84 `calypso_source_bundle.json` | `0ee0dcf948a0f3392a2a19d27fbc5d667632493bbe64c477aa983a5254239a52` |
| WB84 `provenance_freeze.json` | `e25864f486710531a011697fa0da6660cf2a0af76e8b50c41132ce4788f60e17` |
| WB85b `wb85b_fwer_protocol.json` | `1bac20ffdd26b7fbc4cd912ce5d75db266683b279e860837eab55f0da519af21` |
| WB85b `wb85b_acceptance_rule.json` | `a3adf56709a474ac963da3d45ff627b562465127e3146d642f99916905e83427` |
| `alignment/wb85b_fwer_protocol.py` | `757955097f41a5598e01f0e86f2128224b68bf0c3a199161c14936f42edbe756` |
| `alignment/common_track_solver.py` | `b8d2bc31fd736d84fb7a0c18a229d34c8acd6b7a5be774a4ef3bd0b98a793bf4` |
| `alignment/wb85b_official_runner.py` | `6cc919202be3cabf4e61df499231593f4f830bd83116d3ab73db2e5441f0babe` |
| `alignment/physical_common_track_execution.py` | `08770a252f1a0ea8adef9ee120289cb91fec2597d7d377717ffd591436bf8072` |

身份核对：

```text
athena_version = 24.0.41
geometry       = FASERNU-04
global_tag     = OFLCOND-FASER-06
acts_tool      = FaserActsExtrapolationTool
q_over_p_mode  = 0
n_qualified_exports = 2394
material_map.hashed_files 在 WB84 snapshot 中历史上为空；本轮不回填 WB84
```

```text
snapshot_verify_pass = true
corpus_verify_pass   = true
```

未增加、删除或改写任何 WB84 replica。未打开 overlay / Final Blind / sealed_test / 禁区 run range。

---

## 3. 官方执行链（每个 event）

```text
WB84 physical measurement
→ truth association labels only
→ common-track solve on the frozen chart
→ left-SE(3) update
→ WB85bOfficialCalypsoActsBackend.execute_refit
→ Calypso SegmentFitRefit
→ ACTS mode-0 repropagation
→ new physical predictions for relinearization
→ next iteration
```

约束：

```text
official_runner = WB85bOfficialCalypsoActsBackend.execute_refit
CalypsoActsPhysicalBackend.execute stays False
toy_uniform_By / lab_transport / first-step cache reuse = forbidden
WB84 observations stay fixed; they are not treated as identity-payload predictions
same-payload reuse is allowed only for official rerefit products
MAX_ITERATIONS = 10
CONSECUTIVE_REQUIRED = 2
automatically_pass_at_max_iterations = False
```

`OfficialPhysicalIterator` 不实现 consecutive-2，且 planner backend 不能 execute。WB86 **没有**改那个冻结文件；包装器调用冻结 primitives。

---

## 4. HTCondor DAG

```text
SNAPSHOT_VERIFY
→ CORPUS_VERIFY
→ PHYSICAL_ALIGNMENT_SHARDS   # 2394 jobs, one event each
→ RESULT_INTEGRITY            # presence + hashes only
→ GLOBAL_SUMMARY              # first scientific aggregate
→ FROZEN_QUALIFICATION_GATE   # evaluate_wb85b_qualification
→ CLOSURE
```

```text
flavour        = tomorrow
request_memory = 8000 MB
request_cpus   = 1
max_retries    = 1   # infrastructure, byte-identical rerun only
max_materialize = 40  # avoid 800-way Athena cold-start against EOS Calypso build
schedd         = bigbird24.cern.ch
dagman_cluster_failed_r0 = 1112463
dagman_cluster_stuck_r1  = 1112604
shard_cluster_stuck_r1   = 1112606
dagman_cluster           = 1113234
```

输出根：

```text
outputs/mc24_four_station_wb86_physical_qualification_v1/
  events/{index:04d}/technical.json
  sealed/{index:04d}.json
```

technical 只写 success/failure、runtime、hashes、iteration count、error code。在全部 2394 个 shard 完成前，不得发射 pull / bias / coverage 汇总。

---

## 5. 三层判决（预注册，尚未应用）

**A. execution_qualification**  
全部 required strata；每个 identifiable stratum solvable n≥200；无禁区资产；provenance 匹配；finite / SPD / FD / runner contract；无 dropped identifiable mode；nonconvergence ≤ 0.05。  
`n<200` → `UNKNOWN`，不是 FAIL/PASS。

**B. statistical_qualification**  
仅当 A 可分析：调用冻结 `evaluate_wb85b_qualification`。  
FAIL iff `T_LS > 23.33666415864534` OR `T_cov > 14.44937533544792`。  
忽略 `location_scale_gof.accepted` / `coverage_calibration.accepted` 的历史 α=0.05 旗标。  
Holm 只作 diagnostics。

**C. physics_screening**  
冻结 research screening，**不是** collaboration-approved physics requirements：

```text
translation / identity |bias| ≤ 0.1 mm
rotation rz |bias| ≤ 1 mrad
weak-JG |mean z| ≤ 5
plus frozen catastrophic coverage / sign / frame guards
```

Oracle：

```text
alignment_oracle_qualified_for_physical_FASER = true
```

仅当 A∧B∧C 全 PASS；否则 false。UNKNOWN ⇒ oracle false。

FAIL 后只允许只读 attribution。禁止 retune、新 corpus、IRLS、改 association、或 requalify-until-PASS。

PASS 后冻结 `wb86_final_qualification.json` 与 `physical_alignment_oracle_closure.json`。  
`ml_alignment_eval_authorized` 仍为 false。下一阶段必须单独授权。

---

## 6. 本轮明确没有做的事

- 没有改 WB85 / WB85a / WB85a-r1 / WB85b generating sources、thresholds、α、power、MC、solver、runner 或 physical contract
- 没有读取 WB84 alignment outcomes（语料只有 measurements）
- 没有看任何 sealed 科学字段
- 没有跑 GLOBAL_SUMMARY / GATE / CLOSURE
- 没有启动 ML-vs-truth
- 没有 W64 retraining / V5A / route energy / hybrid / calibration / association 变更
- 没有新 replica 或新几何

单问句：在冻结 WB84 语料 + WB85b 统计 + 冻结 common-track solver + 真实 Calypso/ACTS 迭代下，truth-associated common-track alignment 是否通过预注册 physical oracle？答案要等 DAG 跑完。

---

## 7. 闭合（2026-09-29，只读，协议未改）

记录时的仓库状态：

```text
branch                  = 4station
HEAD                    = eea34212265e95060f1facf9bb744e104e5e2958
origin/4station         = eea34212265e95060f1facf9bb744e104e5e2958
working tree            = clean relative to that commit
uncommitted WB86 repair = none
```

未跟踪的 `core.*` 是崩溃转储，不是 WB86 修补。DAG 各 stage 的 stdout 记录的执行代码是授权时的 `3bd3073e3ffe66a74aed5dc55a6b551853d7dc98`。`eea3421` 只提交了当时已经跑完的 execution 代码与本 workbook 的前半部分，没有改阈值、统计量、solver、association 或 corpus。

### 7.1 DAG / Condor

2026-09-29 在 `bigbird24.cern.ch` 上查询：该用户队列为 0（无 RUNNING / IDLE / HELD）。历史作业已不在队列里。本地 DAG 记录：

| 角色 | cluster | 结局 | 分类 |
|---|---|---|---|
| 首轮 DAGMan | 1112463 | `PHYSICAL_ALIGNMENT_SHARDS` 1112466.0 rc=1 后 abort | infrastructure：只读 WB84 ingest 被误判为 rewrite |
| 卡住的 rescue DAGMan | 1112604 | 被后续提交取代 | infrastructure |
| 卡住的 shard cluster | 1112606 | 新 DAG 启动时 abort，DAGMan 记为 unknown-node | infrastructure |
| 完成的 DAGMan | 1113234 | 7/7 nodes done，exit 0，2026-09-13 13:43:17 | execution complete |
| 完成的 shards | 1113379 | 2394 procs，user log 2407 次正常终止且 return value 0，0 hold，0 abort | execution complete |
| RESULT_INTEGRITY | 1116536 | completed successfully | structural |
| GLOBAL_SUMMARY | 1116537 | completed successfully | first scientific aggregate |
| FROZEN_QUALIFICATION_GATE | 1116538 | completed successfully；stdout `overall=FAIL` | frozen decision |
| CLOSURE | 1116539 | completed successfully；stdout `oracle=False` | frozen decision |

`1113379` 的终止次数多于 2394，是 DAGMan 已记录的 “total end count != 1 (2)” 重复终止事件，不是缺 shard，也不是科学失败。首轮 1112466 的 stderr 只属于那次 infrastructure abort，不得拿来改协议。

完成链：

```text
SNAPSHOT_VERIFY 1113235 / 早先 1112464
→ CORPUS_VERIFY 1113378
→ PHYSICAL_ALIGNMENT_SHARDS 1113379
→ RESULT_INTEGRITY 1116536
→ GLOBAL_SUMMARY 1116537
→ FROZEN_QUALIFICATION_GATE 1116538
→ CLOSURE 1116539
```

### 7.2 Shard completion

目录 `events/0000`–`events/2393` 与 `sealed/0000.json`–`sealed/2393.json` 无缺口。全部 2394 个 technical 记录的 `cluster_id` 都是 `1113379`。

```text
n_expected           = 2394
n_technical          = 2394
n_technical_success  = 2392
n_technical_fail     = 2
n_sealed             = 2394
n_missing            = 0
n_corrupt            = 0
n_hash_mismatch      = 0   # RESULT_INTEGRITY
n_duplicate          = 0
```

官方链身份：`engine = calypso_segmentfit_acts_mode0`，`executor = WB85bOfficialCalypsoActsBackend.execute_refit`。SUCCESS runtime 中位数约 4195 s。

两个 technical FAIL 都是官方 rerefit 之后、冻结代码主动拒绝，不是调度器崩溃：

```text
index 0489  identity_finite_survey_prior
  mc24_100043_00300_00399:100043:2100
  missing truth-associated hit after official rerefit
  particle 10001 station 3
  host b9p06p4369.cern.ch  runtime 881 s  iterations 0

index 1745  identifiable_translation_fixed_dz
  mc24_100047_00100_00149:100047:2149
  missing truth-associated hit after official rerefit
  particle 10001 station 1
  host b9p20p6104.cern.ch  runtime 223 s  iterations 0
```

其余 sealed reason：

```text
max_iterations = 2352
converged      = 40
WB86Error      = 2
```

`automatically_passed_at_max_iterations` 保持 false。没有用 toy transport。

### 7.3 RESULT_INTEGRITY

`wb86_result_integrity.json`，utc `2026-09-13T11:27:53Z`：

```text
result_integrity_pass = true
scientific_aggregates_not_computed_here = true
n_expected = 2394
n_technical_success = 2392
n_technical_fail = 2
n_missing = 0
n_hash_mismatch = 0
```

### 7.4 GLOBAL_SUMMARY

`wb86_global_summary.json`，utc `2026-09-13T11:31:07Z`。可分析定义是 converged、solver_ok、`n_dropped = 0` 且 projected 存在。`diagnostics_only = null`，因为不是每个 identifiable stratum 都有至少 2 个 z。

| stratum | attempted | converged / analyzable | nonconvergence | dropped-parameter events | engineering bias | empirical coverage | sign/frame |
|---|---:|---:|---:|---:|---:|---:|---|
| identity_fixed_dz | 299 | 0 | 1.000 | 0 | 0 | 0/0 | false |
| identity_finite_survey_prior | 303 | 0 | 1.000 | 3 | 0 | 0/0 | false |
| identifiable_translation_fixed_dz | 311 | 10 | 0.968 | 1 | +4.92e-14 mm | 10/10 | false |
| identifiable_translation_finite_survey_prior | 298 | 7 | 0.977 | 4 | −8.29e-15 mm | 7/7 | false |
| identifiable_rotation_fixed_dz | 300 | 1 | 0.997 | 0 | −44.625 mrad | 0/1 | true |
| identifiable_rotation_finite_survey_prior | 283 | 0 | 1.000 | 2 | 0 | 0/0 | false |
| weak_jg_diagnostic_fixed_dz | 302 | 13 | 0.957 | 1 | z_dx −2.05e-12, z_ry +2.05e-12 | — | — |
| weak_jg_diagnostic_finite_survey_prior | 298 | 8 | 0.973 | 4 | z_dx +2.75e-12, z_ry −2.75e-12 | — | — |

Finite / SPD：`n_nan = 0`，`n_inf = 0`，`n_non_spd = 0`。记录到的 `fd_rel_max_observed = 0`。

旋转 finite stratum 另有 1 个 event（index 1104）solver 收敛但 `n_dropped = 2`，projected 为空，因此 analyzable = false，不进入上表的 converged 计数。旋转 fixed 的唯一 analyzable event（index 657）`a_hat = −43.918 mrad`，`a_true = 0.707 mrad`，`z = −496.3`，coverage false。另一个未收敛 event（index 239）`sign_frame_failure = true`，`a_hat = 73.63 mrad`。

弱 JG 的 |mean z| 远小于 5。翻译 stratum 里少数收敛事件的 bias 在 1e-13 mm 量级，但 n 远小于 200。

### 7.5 Execution qualification

```text
execution_qualification = FAIL
analyzable               = false
```

FAIL 原因，全部来自冻结 `execution_layer`：

```text
nonconvergence identity_fixed_dz
nonconvergence identity_finite_survey_prior
dropped identifiable mode identity_finite_survey_prior
nonconvergence identifiable_translation_fixed_dz
dropped identifiable mode identifiable_translation_fixed_dz
nonconvergence identifiable_translation_finite_survey_prior
dropped identifiable mode identifiable_translation_finite_survey_prior
nonconvergence identifiable_rotation_fixed_dz
nonconvergence identifiable_rotation_finite_survey_prior
dropped identifiable mode identifiable_rotation_finite_survey_prior
```

同时记录、但不把结论改成 UNKNOWN 的 `n < 200`：

```text
identity_fixed_dz n=0
identity_finite_survey_prior n=0
identifiable_translation_fixed_dz n=10
identifiable_translation_finite_survey_prior n=7
identifiable_rotation_fixed_dz n=1
identifiable_rotation_finite_survey_prior n=0
```

冻结规则是：有 nonconvergence / dropped-mode 等 failure 时 status = FAIL；只有在没有 failure、仅有 `n < 200` 时才是 UNKNOWN。这里不是 UNKNOWN。

### 7.6 Statistical qualification

```text
statistical_qualification = null
T_LS                       = not computed
T_cov                      = not computed
critical_LS                = 23.33666415864534
critical_cov               = 14.44937533544792
```

`frozen_qualification_gate` 只在 `execution.analyzable` 或 `execution.status == UNKNOWN` 时调用 `alignment.wb85b_fwer_protocol.evaluate_wb85b_qualification`。本次 execution 是 FAIL，因此该函数没有被调用，也没有被复制。

WB85b-r1 措辞保持：

```text
nominal_statistical_protocol_qualified = true
exact_finite_sample_FWER_proven        = false
```

本次没有 primary statistical PASS/FAIL，因为样本不可分析。不得把 null 说成统计通过。

### 7.7 Physics screening

研究筛选，不是 collaboration-approved detector requirement。阈值未改：

```text
translation / identity |bias| <= 0.1 mm
rotation rz |bias|             <= 1 mrad
weak-JG |mean z|               <= 5
```

```text
physics_screening = FAIL
```

失败项：

```text
gross coverage identifiable_rotation_finite_survey_prior
rotation rz screening identifiable_rotation_fixed_dz
sign/frame identifiable_rotation_fixed_dz
gross coverage identifiable_rotation_fixed_dz
gross coverage identity_finite_survey_prior
gross coverage identity_fixed_dz
```

弱 JG 没有越过 |mean z| = 5。两个翻译 stratum 的工程 bias 没有越过 0.1 mm，但它们已经在 execution 层因 nonconvergence 失败。空样本的 empirical coverage 被既有实现记为 0，从而触发 gross coverage；这是冻结 screening 对不可分析 stratum 的结果，不是事后加的门槛。

### 7.8 Final oracle

```text
execution_qualification = FAIL
statistical_qualification = null
physics_screening = FAIL

overall_qualification = FAIL
alignment_oracle_qualified_for_physical_FASER = false
ml_alignment_eval_authorized = false
```

UNKNOWN 没有出现。oracle 不是近似成功。

### 7.9 Failure attribution（只读，未 retune）

主因是 **nonconvergence**。2392 个官方链成功事件里，2352 个在 `MAX_ITERATIONS = 10` 处停止，solver status 仍是 `ok`，连续两步未同时满足冻结的 scaled-update 与 validation-relative 容差。六个 identifiable stratum 的 nonconvergence 都大于 0.05，四个 stratum 还有 dropped identifiable parameter。

其次：

```text
physics bias / sign-frame / coverage:
  identifiable_rotation_fixed_dz
  唯一 analyzable event bias = -44.625 mrad，|z| = 496
  另有未进入 analyzable 集合的 sign/frame event

coverage:
  四个 n=0 或经验覆盖为 0 的 stratum 触发 gross coverage

execution refusal, 2/2394, 不主导 FAIL:
  官方 rerefit 后缺少 truth-associated hit
  index 0489 station 3, index 1745 station 1

not observed as the failing gate:
  statistical calibration (T_LS / T_cov not evaluated)
  weak-JG |mean z| > 5
  NaN / Inf / non-SPD
  infrastructure on the completing cluster 1113379
```

没有因此改 corpus、阈值、association、solver、几何、收敛准则，也没有加 IRLS 或删难例。

### 7.10 Artifact hashes

以下是 2026-09-29 对落盘文件的 SHA256。closure 内嵌的 `final_qualification_sha256` 与文件一致。本轮没有重写这些 JSON。

```text
db60a8ab3f95e72034a5127bf0466e8cd2a66eaee43374bcecd37568f9c29423  wb86_snapshot_verify.json
c82cb0bee3a25a687e6d0d7931db24bd3b785ae0efd07b3dc17a5304e892e546  wb86_corpus_verify.json
a929928af016236c8f77cbe2c4b9fc0c1a6663a60ce43bc5474a7863c8dbdd4b  wb86_result_integrity.json
c750b409a2540c291f8126c7add03e5e2df67bc551a5eed9f77e01359bc5aff4  wb86_global_summary.json
f43e385bb5060d5d0f780ca8be44d9da73bfa378a48103deb15057f3da65a910  wb86_execution_qualification.json
7d290eefeb3b48ce5c209aab8f9cbbdb86638ebd803bd6c5376fcf9009ed4e5c  wb86_final_qualification.json
4030b576ddefe5560efcf42514063b1edb4619b4648a4cc596329440bb454981  physical_alignment_oracle_closure.json
```

### 7.11 下一步唯一允许的动作

WB86 physical oracle 已闭合为 false。不要重跑 DAG 来求 PASS，不要改 α、critical value、收敛准则或 WB84 membership，不要启动 ML-vs-truth。

```text
ml_alignment_eval_authorized = false
next_stage_requires_separate_authorization = true
```

若要继续，只能新开一份 prospective workbook，单独授权，并且不得把这次 FAIL 当成调参依据。
