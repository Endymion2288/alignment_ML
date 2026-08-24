# 条目 64 — source-disjoint four-station training diversity（冻结 V2 objective）

日期：2026-08-24  
分支：`4station`  
状态：进行中。条目 63 已冻结结论为 `source_phase_space_undercoverage`，并授权扩源。本条目只验证 **source-disjoint 训练源多样性** 能否覆盖现有 V2 在新 source 上的 phase-space / source-characteristic undercoverage。不改 objective、loss weight、reduction、margin、threshold、`unmatched_penalty`、Platt、candidate builder、unit-capacity solver 或 Transformer 架构。不打开 15D WLS，除非新 reserved blind 完整通过。  
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
# reserved blind 仅在 unused 审计通过后生产；训练不读这些文件
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

### 物理生产（进行中）

新增四源 train：`outputs/mc24_four_station_source_diversity_new_train_v1/`  
- curriculum seed `20260821`，与条目 53 `common_scan_plan` 合同字段一致（已核对 `draw_00` 相同）
- Condor cluster `1001451`（bigbird24 / eossubmit），4 jobs，flavour `tomorrow`

Reserved blind：`outputs/mc24_four_station_source_diversity_blind_v1/`  
- 新 seed `20260824`，7 个同名 payload，随机 draw 与 train 不同
- Condor cluster `1001452`，2 jobs
- 训练脚本禁止读取这些 source

下一步（jobs 完成后，不要改合同）：

```bash
bash scripts/run_four_station_source_diversity_training.sh assemble-train
bash scripts/run_four_station_source_diversity_training.sh merge-train
bash scripts/run_four_station_source_diversity_training.sh overlay-train
bash scripts/run_four_station_source_diversity_training.sh coverage-sanity
bash scripts/run_four_station_source_diversity_training.sh train
bash scripts/run_four_station_source_diversity_training.sh assemble-blind
bash scripts/run_four_station_source_diversity_training.sh overlay-blind
bash scripts/run_four_station_source_diversity_training.sh infer-blind
bash scripts/run_four_station_source_diversity_training.sh mechanism-blind
bash scripts/run_four_station_source_diversity_training.sh assess
```

coverage-sanity、训练 SHA、blind gate 数字待 jobs 完成后写入。15D WLS 仍关闭。
