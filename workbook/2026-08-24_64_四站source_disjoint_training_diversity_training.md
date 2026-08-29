# 条目 64 — source-disjoint four-station training diversity（冻结 V2 objective）

日期：2026-08-24  
分支：`4station`  
状态：已冻结。条目 63 授权的六源 diversity control 已按冻结条目 62 objective 训完并在 reserved blind 上开闸。**未通过。** 归类 `source_diversity_blind_failed_other`。`continue_to_15d_relative_wls=false`。不打开 15D WLS，不再增加同类 source，不改 loss / 权重 / OP。  
密封 test：永久关闭。

## 本条目允许做什么

在保持条目 62 整套 objective / operating convention 不变的前提下，把训练源从现有两条扩到授权的六条原始 xAOD，再在从未进入 48–63 训练 / diagnostic / OP / source-audit 的 reserved blind pair 上开一次最终 gate。

不允许：改 loss / 权重 / reduction / OP、扩大 misalignment envelope、加 DoF、加 FD probe、用同一两条 train file 的 synthetic duplication 代替 source diversity、把条目 56 transfer 或条目 53–55 development 当成新模型 gate。

## 冻结 convention

合同：`configs/physical_four_station_diversity_training.yaml`  
控制号：`retrained_v2_source_disjoint_diversity_v1`  
父 checkpoint：条目 62 `a46a35bd28eb294fe307590d4f12595f6d3bfaa8dc64aea0bf7418543605e1ef`

| 量 | 冻结值 |
| --- | --- |
| objective | 条目 62 整体：dustbin-aware route-margin + gauge consistency + `max_over_complete_truth_routes_in_event` |
| aux weights | packing `0.07061055340401011`，dustbin-aware `0.05`，gauge `1.0`（copy workbook 59） |
| Platt / threshold / unmatched_penalty | identity / 0.001 / −1.0 |
| packing margin | 1.0 |
| 物理 curriculum | 条目 53：15D relative + left-SE(3) twin，`≤0.5 mm / ≤5 mrad`，真实 `/Tracker/Align -> SegmentFitRefit -> SegmentsRefit -> NtupleDumper -> Acts(mode 0)` |
| 训练 seed / epoch | `20260822` / 30 / 最后 epoch checkpoint / GPU-only |
| 条目 56 transfer | mechanism / history comparison only |
| 条目 53–55 development | mechanism / history comparison only |
| 预留 blind | `mc24_100047_00350_00399`、`mc24_100048_00350_00399`（训练全程不读） |
| unused reserve | `mc24_100047_00800_00849`、`mc24_100048_00800_00849`（不加载） |

## 训练源

保留：

- `mc24_100043_00200_00299`（μ−）
- `mc24_100044_00300_00399`（μ+）

新增（条目 63 授权，本条目先做 provenance / disjoint / ROOT / content / covariance / 四站覆盖审计）：

- `mc24_100043_00300_00399`
- `mc24_100044_00200_00299`
- `mc24_100047_00100_00149`
- `mc24_100048_00100_00149`

现有两条 train 的 workbook-53 物理 bank 不重做。新增四条用同一 curriculum 模板、同一 seed `20260821` 生产，以便 `common_scan_plan` 对齐后 merge。V3 identity bank 只做生产前 sanity，不能替代四站 15D relative + gauge-twin 物理链。

## 运行

```bash
source scripts/setup_environment.sh ml
bash scripts/run_four_station_source_diversity_training.sh audit-new-sources
bash scripts/run_four_station_source_diversity_training.sh audit-blind-unused
bash scripts/run_four_station_source_diversity_training.sh prepare-train
bash scripts/run_four_station_source_diversity_training.sh submit-train
# 四源物理生产完成后
bash scripts/run_four_station_source_diversity_training.sh assemble-train
bash scripts/run_four_station_source_diversity_training.sh merge-train
bash scripts/run_four_station_source_diversity_training.sh overlay-train
bash scripts/run_four_station_source_diversity_training.sh coverage-sanity
bash scripts/run_four_station_source_diversity_training.sh train
bash scripts/run_four_station_source_diversity_training.sh freeze-checkpoint
# reserved blind 仅在 unused 审计通过且 checkpoint 冻结后打开
bash scripts/run_four_station_source_diversity_training.sh prepare-blind
bash scripts/run_four_station_source_diversity_training.sh submit-blind
bash scripts/run_four_station_source_diversity_training.sh assemble-blind
bash scripts/run_four_station_source_diversity_training.sh overlay-blind
bash scripts/run_four_station_source_diversity_training.sh infer-blind
bash scripts/run_four_station_source_diversity_training.sh mechanism-blind
bash scripts/run_four_station_source_diversity_training.sh assess
```

coverage-sanity 只做 train-only 对照：用冻结条目 62 在六源 train overlay 上打分，确认对条目 63 failure region 的 `x/y/tx/ty`（尤其 S2/S3 `ty`、S3 `tx`）、2→3 logit 与 `Δ` 覆盖是否扩大。**不允许据此重选 source 或改 curriculum。**

## 最终 gate（打开数字后冻结）

只打开 reserved blind pair，对照 **本 bank 自己的 nominal**。闸门保持：

- raw complete-chain / adjacent truth-edge recall ≥ 0.90
- nominal purity ≥ 0.95、fake ≤ 0.05
- 全部 non-nominal 相对 blind nominal：`Δeff≤0.10`、`Δpurity≤0.05`、`Δfake≤0.05`
- 三组 gauge twins 的 efficiency / purity / fake 与 2→3 差满足既有 tolerance
- 报告 0→1 / 1→2 / 2→3、S3 participation、`U_truth`、fragment / dustbin winner、`Δ=U_truth-max(U_fragment,0)`

通过：归类 `training_domain_coverage_limitation_resolved_by_source_diversity`，冻结 checkpoint / SHA / operating contract，第一次授权 route-selected 15D `ΔT_ij` WLS。  
仍以同一 left-SE(3) common transform 下 systematic truth-utility drop 失败（即使 raw ranking 与 candidate recall 仍好）：停止当前 V2 主线，下一步明确转向 architecture-level relative / gauge-equivariant representation。不再靠增加同类 source、reweighting 或 OP 救 V2。

## 阶段性结果

密封 test 未打开。条目 56 transfer / 53–55 development 未进入新模型训练或 checkpoint selection。

### 新四源生产前审计

脚本：`scripts/audit_four_station_source_diversity_new_sources.py`  
产物：`outputs/mc24_four_station_source_diversity_v1/new_source_audit_v1/decision.json`  
结论：`passed=true`，`authorize_physical_production=true`。与当前 train / development / transfer / reserved blind / unused reserve / sealed / 历史 outlier 均 disjoint。V3 identity 只做 sanity，明确不能替代四站 curriculum。

| source | 电荷 | xAOD entries | V3 identity 四站事件 | cov PD | 主 PDG |
| --- | --- | ---: | --- | ---: | --- |
| `mc24_100043_00300_00399` | μ− | 500000 | 92/97/94/92 | 1.0 | 13 |
| `mc24_100044_00200_00299` | μ+ | 500000 | 89/98/93/93 | 1.0 | −13 |
| `mc24_100047_00100_00149` | μ− | 250000 | 90/99/92/94 | 1.0 | 13 |
| `mc24_100048_00100_00149` | μ+ | 250000 | 89/95/94/94 | 1.0 | −13 |

ROOT `CollectionTree` 均有 `SCT_ClusterContainer` / `SegmentFit` / `Segments` / `TruthParticlesAux.`。

### Reserved blind unused 审计

脚本：`scripts/audit_four_station_reserved_blind_unused.py`  
产物：`outputs/mc24_four_station_source_diversity_v1/blind_unused_audit.json`  
`mc24_100047_00350_00399` 与 `mc24_100048_00350_00399` 未出现在 48–63 的 physical/synthetic/iteration manifest、`identity_summaries` 或 `truth_routes.jsonl` 的已加载 source 字段中。配置与 workbook 中的 reserved / do-not-load 声明不算使用。xAOD 文件存在。本步未打开 blind 内容做训练或 OP。

### 物理生产

新增四源 train：`outputs/mc24_four_station_source_diversity_new_train_v1/`  
- curriculum seed `20260821`，与条目 53 `common_scan_plan` 合同字段一致（已核对 `draw_00` 相同）
- Condor cluster `1001451`（bigbird24 / eossubmit），4 jobs，flavour `tomorrow`

Reserved blind：`outputs/mc24_four_station_source_diversity_blind_v1/`  
- 新 seed `20260824`，7 个同名 payload，随机 draw 与 train 不同
- Condor cluster `1001452`，2 jobs
- 训练全程未读这些 source；checkpoint 冻结之后才做 artifact 审计与 assemble

### 1001451 物理产物审计（不只看 return code）

四个 job 均 `Normal termination (return value 0)`，但正式接受条件是 artifact 审计：`scripts/audit_four_station_physical_production_artifacts.py`  
产物：`outputs/mc24_four_station_source_diversity_v1/new_train_production_audit.json`

- 4 source × 7 payload = **28/28** 通过
- 无 `failure.json`
- ROOT tracklets 可读，四站均有命中，content audit covariance 全 PD
- payload / scan_plan 与 iteration `common_scan_plan` 一致
- 与 workbook-53 原 train bank 的 `common_scan_plan`、payload 命名、`15d_gauge_then_left_se3` / mode-0 语义一致
- 新四源 reference 共 200 个 source-event UID（每源 50 events）

随后 `assemble-train` → `merge-train`（六源 train-only，`common_scan_plan_verified_identical=true`）→ `overlay-train`（7 个 train payload，constituents 恰好是授权六源）。

### Train-only coverage-sanity（冻结条目 62，不改 source / curriculum）

用冻结条目 62 在六源 train overlay 上打分，对照条目 63 已写入的 transfer failure core（405 条，history only）。**没有据此重选 source 或改 curriculum / loss / OP。**

| 量 | 旧两源 train | 扩展六源 train |
| --- | ---: | ---: |
| complete truth | 3331 | 9893 |
| kinematic mean inside failure core | 0.821 | 0.810 |
| S2 `ty` inside | 0.595 | 0.568 |
| S3 `ty` inside | 0.827 | 0.827 |
| S3 `tx` inside | 0.716 | 0.716 |
| failure-core 2→3 logit 落入 train 5–95% | 0.257 | **0.798** |
| failure-core `Δ` 落入 train 5–95% | 0.356 | **0.867** |
| `draw_01+common` hard rate | 0.234 | 0.598 |
| `draw_01+common` fragment-winner rate | 0.010 | 0.095 |
| near-boundary complete routes | 618 | 3788 |
| short-station hard/near | 325 | 3568 |

`coverage_expanded=true`，主要来自 2→3 logit 与 `Δ` 支撑变宽；S2/S3 `ty`、S3 `tx` 的 5–95% inside **没有**变好。这只是 sanity，合同不改。

### 训练与 checkpoint 冻结

lxplus901 Tesla T4，seed `20260822`，30/30 epoch 完成，最后 epoch checkpoint。复制条目 59 aux 权重，训练中未改。训练 overlay 只有 train、六源齐全、reserved blind 未加载。

Python 训练正常结束。bash wrapper 随后因脚本在训练中被编辑而报 `unexpected EOF while looking for matching '"'`（exit 2）。这是 wrapper 语法错误，不是训练失败。checkpoint 文件完整：

- `outputs/mc24_four_station_source_diversity_v1/checkpoint/route_aware_transformer_v2.pt`
- sha256 `0c3a28704cc01151fab7ac943e41338e859b5dde1502c0b10e2f12ada04e6236`
- 冻结 JSON：`outputs/mc24_four_station_source_diversity_v1/checkpoint_freeze.json`
- `blind_not_used_for_training=true`
- `reserved_blind_used_for_checkpoint_selection=false`
- `early_stopping=false`，`objective_reestimated=false`
- 加载源恰好是授权六源

### 1001452 物理产物审计（冻结之后才打开）

两个 job 均 `Normal termination (return value 0)`，正式接受条件仍是 artifact 审计。  
产物：`outputs/mc24_four_station_source_diversity_v1/blind_production_audit.json`

- 2 source × 7 payload = **14/14** 通过
- split mode `reserved_blind_validation`
- 源恰好是 `mc24_100047_00350_00399`、`mc24_100048_00350_00399`
- 无 `failure.json`，ROOT 可读，四站 + PD cov
- 未把 train bank（seed `20260821`）当作 blind 的 reference scan plan

随后 `assemble-blind`（validation 99 events，train=0）→ `overlay-blind`（7 payload，constituents 恰好 reserved pair）。密封 test 未打开。

### Reserved-blind association（对照本 bank nominal）

checkpoint：`0c3a28704cc01151fab7ac943e41338e859b5dde1502c0b10e2f12ada04e6236`  
诊断：`outputs/mc24_four_station_source_diversity_v1/blind_association/`

| payload | raw chain | 0→1 | 1→2 | 2→3 | S3 | eff | purity | fake | Δeff | Δpurity | Δfake |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| reference | 1.000 | 0.936 | 0.928 | 0.934 | 564 | 0.933 | 0.961 | 0.038 | — | — | — |
| hard_s3_ry | 1.000 | 0.932 | 0.930 | 0.933 | 569 | 0.929 | 0.947 | 0.052 | 0.004 | 0.014 | 0.014 |
| hard_s3_ry+common | 1.000 | 0.910 | 0.894 | 0.904 | 553 | 0.892 | 0.941 | 0.061 | 0.042 | 0.021 | 0.022 |
| draw_00 | 1.000 | 0.898 | 0.894 | **0.838** | 522 | **0.817** | 0.918 | 0.073 | **0.117** | 0.043 | 0.034 |
| draw_00+common | 1.000 | 0.907 | 0.915 | **0.833** | 517 | 0.847 | 0.943 | 0.061 | 0.087 | 0.018 | 0.022 |
| draw_01 | 1.000 | 0.924 | 0.918 | 0.943 | 573 | 0.911 | 0.961 | 0.036 | 0.022 | 0.001 | −0.002 |
| draw_01+common | 1.000 | 0.914 | 0.928 | 0.924 | 561 | 0.905 | 0.967 | 0.038 | 0.028 | −0.006 | 0.000 |

Layer 1：raw complete-chain / adjacent truth-edge recall 全部 1.0，≥ 0.90。  
Nominal：purity 0.961 ≥ 0.95，fake 0.038 ≤ 0.05。  
Layer 2 失败点：

- `draw_00` 完整-track `Δeff=0.117 > 0.10`
- `draw_00+common` 的 2→3 `Δeff=0.101 > 0.10`（`adjacent_23_and_s3.ok=false`）

其余 payload 的完整-track Δeff / Δpurity / Δfake 在闸门内。三个 plus-common 的完整-track vs-nominal 均通过。

### Gauge twins（layer 3）

| family | \|Δeff\| | \|Δpurity\| | \|Δfake\| | \|Δ 2→3\| | 闸门 |
| --- | ---: | ---: | ---: | ---: | --- |
| hard_s3_ry | 0.037 | 0.006 | 0.009 | 0.029 | 0.05 / 0.05 / 0.05 / 0.08 |
| draw_00 | 0.030 | 0.025 | 0.012 | 0.005 | 通过 |
| draw_01 | 0.006 | 0.006 | 0.002 | 0.019 | 通过 |

`gauge.ok=true`。这不是 twin 指标破裂。

### Mechanism / `U_truth`

产物：`outputs/mc24_four_station_source_diversity_v1/blind_mechanism.json`  
`Δ = U_truth − max(U_fragment, 0)`。pooled：`U_truth=1.750`，`U_fragment=0.704`，`Δ=1.045`。production winner：truth 3008 / fragment 281 / dustbin 89；admitted fragment wins 272。

| payload | `U_truth` | `U_fragment` | `Δ` | `U_truth≤0` | truth&lt;fragment |
| --- | ---: | ---: | ---: | ---: | ---: |
| reference | 1.901 | 0.756 | 1.145 | 0.029 | 42 |
| hard_s3_ry | 1.781 | 0.714 | 1.067 | 0.033 | 39 |
| hard_s3_ry+common | 1.821 | 0.740 | 1.081 | 0.067 | 61 |
| draw_00 | 1.566 | 0.666 | 0.900 | 0.102 | 90 |
| draw_00+common | 1.608 | 0.688 | 0.919 | 0.080 | 83 |
| draw_01 | 1.794 | 0.680 | 1.114 | 0.060 | 46 |
| draw_01+common | 1.820 | 0.683 | 1.137 | 0.043 | 54 |

三组 left-SE(3) twin 的 `U_truth` **都没有下降**（chart → twin：1.781→1.821，1.566→1.608，1.794→1.820）。`twin_truth_utility.systematic=false`，`n_drops=0`。因此**不能**归为条目 62/63 那种 common-SE(3) systematic truth-utility drop，也就不是 `source_diversity_insufficient_for_gauge_transfer`。

失败是 `draw_00` 这张硬 15D 随机图上的 association efficiency，以及其 twin 上 2→3 刚过 0.10。raw ranking / candidate recall 全程正常。

## 决策

| 冻结条件 | 本条目 |
| --- | --- |
| 是否改 OP / 权重 / reduction / 架构 | **否** |
| checkpoint SHA | `0c3a28704cc01151fab7ac943e41338e859b5dde1502c0b10e2f12ada04e6236` |
| layer 1 raw candidate | **通过** |
| layer 2 vs 本 bank nominal | **失败**（`draw_00` Δeff；`draw_00+common` 2→3） |
| layer 3 gauge twins | **通过** |
| common-SE(3) `U_truth` 系统下降 | **否**（0/3） |
| 失败归类 | **`source_diversity_blind_failed_other`** |
| `continue_to_15d_relative_wls` | **false** |
| 打开 15D WLS | **否** |
| 再增加同类 source / reweight / 改 OP | **否** |
| 密封 test | 未打开 |

数字打开后合同冻结。没有因为 `draw_00` 差一点而改 threshold、margin、loss 或再扩源。条目 56 transfer 与 53–55 development 不是本 gate。

授权的 source-diversity control 已经做完且未通过最终 blind。下一步不再继续现有 V2 的扩源 / reweighting / OP rescue；也不把这次失败误写成 common-SE(3) gauge-transfer 模式。15D `ΔT_ij` WLS 保持关闭。
