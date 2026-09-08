# Workbook 84 — Calypso Physical Replica Production v1

日期： 2026-09-07
分支： `4station`
任务： 只生产 `calypso_physical_independent_v1` corpus。不做 alignment qualification，不做 ML association。

---

## 0. 裁决（当前）

```text
physical_replica_production_complete = true
calypso_physical_corpus_qualified    = true

alignment_oracle_qualified_for_physical_FASER = false
ml_alignment_eval_authorized                  = false
qualification_authorized                      = false
```

WB83 / WB83b 未改。toy qualification 未重跑。association 未返回。

HTCondor DAG `1110444`：49/49 nodes 成功，`EXITING WITH STATUS 0`。

```text
qualified exports = 2394
all 8 conditions >= 200
independence_violation = false
```

下一阶段 Physical Common-Track Alignment Qualification **尚未授权，未执行**。

---

## 1. 冻结规模（未按 WB83 failed cells 修改）

```text
n_geometry_conditions              = 8
target_replicas_per_condition      = 200
expected_yield                     = 0.50
required_input_independent_events  = 3200
```

8 conditions = 4 个 a priori family × 2 个 survey tag。survey 不是重建差异；事件仍不跨 condition 复用。

| family | 注入（station six-vector） |
|---|---|
| identity | 全零 |
| identifiable_translation | S1 dx = 0.30 mm，S2 dy = −0.20 mm |
| identifiable_rotation | S1 rz = −0.5 mrad，S3 rz = +0.5 mrad |
| weak_jg_diagnostic | S3 dx = 0.20 mm，S3 ry = 0.2 mrad |

不是 WB83 matrix。

---

## 2. 独立性与 allowlist

合并 xAOD 里 `RunNumber` 是 MC channel，`EventNumber` 每 5000 条循环。分配锁在每个文件的 **第一个 production chunk**，`skip_base = 2000`，因此

```text
event_uid = source_id:run_id:event_id
```

在本语料中唯一，没有靠改名掩盖 reuse。

六源授权 train-range 均在 EOS 上。分配 3200 个独立事件，independence audit 在 allocation 时 `n_violations = 0`。未打开 00350 / 00800–00849 / 100116 / 100117 / Final Blind / sealed。

W64 已见过这六源。本语料只服务 truth-only alignment；**不是** source-unseen ML evaluation。

---

## 3. Snapshot / payload / roundtrip

Calypso HEAD `40892527…` 仍是 dirty。正式 identity 是 `calypso_source_bundle.json`（dirty 文件 SHA + diff SHA），不是单独的 dirty commit hash。

4 个 family payload 已写入 `/Tracker/Align` SQLite/POOL。requested == written。

Geometry roundtrip：

```text
input transform → written payload → Calypso sqlite load → SCTAlignmentStore / FaserActsAlignment
geometry_roundtrip = PASS
```

identity 与 identifiable_translation 各 1 个事件的物理 refit 都成功，日志含 sqlite override 与 ACTS alignment context。

---

## 4. 本阶段禁止（保持）

未跑 common-track coverage，未改 statistical gate，未做 ML / W64 / V5A / route-energy / hybrid / calibration，未写 `alignment_oracle_qualified=true`。

---

## 5. 输出根

`outputs/mc24_four_station_calypso_physical_replicas_v1/`

要求的 JSON 均已生成。job logs 在 `condor/logs/`。

---

## 6. DAG 完成后的 corpus QA

`1110444`：48 refit + 1 QA，全部成功。独立复核：48 个 `replicas.jsonl`，2394 unique UID，0 cross-condition reuse，0 same-file entry reuse，0 invalid covariance，0 shard failure。禁止资产字符串只出现在浮点测量值里，不是 00350 / 00800 文件路径。六源均为授权 train-range。

| condition | qualified | yield | 主 reject |
|---|---:|---:|---|
| identity_fixed_dz | 299 | 0.748 | insufficient_stations 94 |
| identity_finite_survey_prior | 303 | 0.758 | insufficient_stations 95 |
| identifiable_translation_fixed_dz | 311 | 0.778 | insufficient_stations 88 |
| identifiable_translation_finite_survey_prior | 298 | 0.745 | insufficient_stations 100 |
| identifiable_rotation_fixed_dz | 300 | 0.750 | insufficient_stations 95 |
| identifiable_rotation_finite_survey_prior | 283 | 0.708 | insufficient_stations 116 |
| weak_jg_diagnostic_fixed_dz | 302 | 0.755 | insufficient_stations 93 |
| weak_jg_diagnostic_finite_survey_prior | 298 | 0.745 | insufficient_stations 96 |

实际 yield 约 0.71–0.78，高于规划 0.50。每个 condition 都达到 200，gap = 0。未降低 n。

```text
calypso_physical_corpus_qualified = true
alignment_oracle_qualified_for_physical_FASER = false
```
