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
