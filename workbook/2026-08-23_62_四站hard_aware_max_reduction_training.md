# 条目 62 — hard-aware max-reduction control

日期：2026-08-23  
分支：`4station`  
状态：条目 61 预注册的唯一 control 已训练，并在同一冻结 transfer 上只评估一次。layer 1 通过；layer 2 / 3 未全过。`continue_to_15d_relative_wls: false`。失败归类 **`objective_reduction_failure`**。停止继续用 weighting / reduction / OP 救模型。不冻结为新的 production V2，不打开 15D WLS。密封 test 未打开。

## 本条目允许做什么

只训练并评估条目 61 已经冻结的 `max_over_complete_truth_routes_in_event`。不再做 feasibility scan，不比较 mean / max / top-k，不改任何 inference convention 或 aux 权重。

## 冻结 convention

合同：`configs/physical_four_station_hard_aware_reduction_training.yaml`  
架构拷贝：`configs/geometry_aware_transformer_v2_four_station_hard_aware_reduction.yaml`  
控制号：`retrained_v2_hard_aware_max_reduction_v1`  
父 checkpoint：条目 59 `6565d028e01e561105d4ce467d10d5c82e5a2435daad6a69c47a07920defb5dc`

| 量 | 冻结值 |
| --- | --- |
| 架构 / feature / candidate builder | 与条目 54/56/59 相同 |
| score stream | raw sigmoid + identity Platt |
| threshold | 0.001 / 0.001 / 0.001 |
| unmatched_penalty | −1.0 |
| dustbin | 0 |
| packing margin | 1.0 |
| solver | `adjacent_contiguous_unit_capacity_set_packing` |
| 统计语义 | `physical_edge_deduplicated` |
| packing / dustbin-aware reduction | **max**（唯一允许改动） |
| packing 权重 | 0.07061055340401011（复制，不重估） |
| dustbin-aware 权重 | 0.05（复制，不重估） |
| gauge 权重 | 1.0（复制，不重估） |
| seed | 20260822 |
| checkpoint | 固定 30 epoch 的最后一个 |
| transfer | 条目 56/59 已打开的 source-disjoint bank，只评一次 |

## 闸

1. raw complete-chain recall ≥ 0.90，邻边 truth-edge recall ≥ 0.90。
2. nominal purity ≥ 0.95、fake ≤ 0.05；全部 non-nominal 相对 nominal：`Δeff≤0.10`、`Δpurity≤0.05`、`Δfake≤0.05`。报告 0→1 / 1→2 / 2→3 / S3。
3. 三组 gauge twin 的 efficiency / purity / fake 以及 2→3 efficiency 差过既有 tolerance。
4. score-scale 相对条目 54 不恶化。

全过：冻结 SHA256，第一次授权 route-selected 15D `ΔT_ij`（条目 49/50 判据不变）。  
任一失败：停止 weighting / reduction / OP rescue，按 train 上 584 个 event-max short case 是否被推到 `Δ≥1` 归类为 `objective_reduction_failure` 或 `training_domain_coverage_limitation`。`continue_to_15d_relative_wls=false`。

## 运行

```bash
source scripts/setup_environment.sh ml
bash scripts/run_four_station_hard_aware_reduction_training.sh train
bash scripts/run_four_station_hard_aware_reduction_training.sh infer-transfer
bash scripts/run_four_station_hard_aware_reduction_training.sh score-scale
bash scripts/run_four_station_hard_aware_reduction_training.sh mechanism
bash scripts/run_four_station_hard_aware_reduction_training.sh assess
bash scripts/run_four_station_hard_aware_reduction_training.sh compare
```

单测：`tests/test_hard_aware_reduction_eval.py` 4 项在评估前通过。

## 结果

合同未改。数字只记录。15 维 WLS 未打开。条目 59 checkpoint `6565d028…cd80` 未被覆盖。

### 权重合同（第一次 optimizer step 之前）

`outputs/mc24_four_station_hard_aware_reduction_v1/checkpoint/aux_loss_weight_contract.json`

| 项 | 权重 |
| --- | ---: |
| packing-route competition | 0.07061055340401011 |
| dustbin-aware route margin | 0.05 |
| gauge-twin consistency | 1.0 |

`algorithm: copy_frozen_workbook59_aux_weights`。`scale_estimation_skipped: true`。`route_competition_reduction: max`。`development_validation_used: false`。没有因为 max 变大而重新归一化。

### 训练

- 节点：lxplus901 Tesla T4；seed `20260822`；固定 30 epoch 的最后一个
- 交互式 GPU，未提交 Condor，训练过程未加载 transfer
- checkpoint：`outputs/mc24_four_station_hard_aware_reduction_v1/checkpoint/route_aware_transformer_v2.pt`
- sha256：`a46a35bd28eb294fe307590d4f12595f6d3bfaa8dc64aea0bf7418543605e1ef`
- identity Platt 与 0.001 / −1.0 未改

| epoch | edge | packing | dustbin-aware | gauge twin |
| ---: | ---: | ---: | ---: | ---: |
| 0（nominal） | 0.241 | 3.403 | 5.174 | 0.000 |
| 7（nominal 末） | 0.294 | 1.443 | 2.541 | 0.000 |
| 8（stage 2 首，1680 graphs / 720 twins） | 0.281 | 1.459 | 2.483 | 0.040 |
| 29（最后） | 0.048 | 0.126 | 0.126 | 0.030 |

packing / dustbin-aware 在 max reduction 下高于条目 59 的 epoch-29 mean（0.096 / 0.097），这是 reduction 本身，不是权重被重估。

### Transfer 三层闸（只评一次）

决策：`outputs/mc24_four_station_hard_aware_reduction_v1/transfer_gate_decision.json`

| 层 | 结果 |
| --- | --- |
| 1 raw complete-chain / 邻边 truth-edge recall | **过**：7/7 payload = 1.0 |
| 2 association | **不过**：nominal 绝对闸过（purity 0.9575、fake 0.040、efficiency 0.793、selected 555）；`draw_01_plus_common` efficiency drop **0.104 > 0.10**；2→3 drop 0.121 |
| 3 score-scale vs 条目 54 | **过且更好**：median \|Δ raw logit\| 0.092 / 0.630 = 0.147；median \|Δ truth-route utility\| 0.176 / 1.349 = 0.131 |
| 3 twin route-metric | **不过**：`draw_01` family efficiency \|Δ\| **0.070 > 0.05**；2→3 \|Δ\| 0.074 < 0.08（这项比条目 59 的 0.102 已过）；另外两族通过 |

`continue_to_15d_relative_wls: false`。`next_step: stop_weighting_and_operating_point_rescue`。没有因为某个 payload 差一点而改 loss 或 OP。

#### 逐 payload association

| payload | eff | purity | fake | Δeff / Δpurity / Δfake | 0→1 / 1→2 / 2→3 | S3 | selected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| reference | 0.793 | 0.958 | 0.040 | — | 0.792 / 0.801 / 0.786 | 466 | 555 |
| hard_s3_ry | 0.799 | 0.951 | 0.042 | −0.006 / 0.007 / 0.002 | 0.786 / 0.798 / 0.795 | 471 | 550 |
| hard_s3_ry + common | 0.783 | 0.943 | 0.054 | 0.010 / 0.015 / 0.014 | 0.776 / 0.789 / 0.772 | 461 | 540 |
| draw_00 | 0.731 | 0.917 | 0.072 | 0.062 / 0.041 / 0.032 | 0.777 / 0.717 / 0.693 | 428 | 527 |
| draw_00 + common | 0.747 | 0.930 | 0.063 | 0.046 / 0.027 / 0.023 | 0.792 / 0.724 / 0.722 | 444 | 543 |
| draw_01 | 0.760 | 0.931 | 0.060 | 0.033 / 0.026 / 0.020 | 0.774 / 0.783 / 0.740 | 449 | 533 |
| draw_01 + common | 0.689 | 0.933 | 0.056 | **0.104** / 0.025 / 0.017 | 0.729 / 0.705 / **0.666** | 408 | 497 |

相对条目 59：`draw_01 + common` 的 Δeff 从 0.143 降到 0.104，仍超 0.10；twin efficiency 差从 0.108 降到 0.070，仍超 0.05。

### 与条目 59 的 mechanism 对照

| 量 | 条目 59 | 本 control |
| --- | ---: | ---: |
| transfer `U_truth` 中位 | 0.870 | 0.968 |
| `U_truth ≤ 0` 比例 | — | 0.189 |
| `U_best_fragment` 中位 | — | 0.367 |
| production margin 满足 | 2384 | 2492 |
| `truth < fragment` | 907 | 810 |
| dustbin winner（miner 未见） | 433 | 329 |
| production fragment winner | 513 | **490** |
| selected complete truth | 2432 | 2561 |

Transfer 上 2/3/4-station competitor 的 `Δ` 中位：2 站 +1.059、3 站 +0.910、4 站 +0.281。2-station hard 323、3-station hard 251、4-station hard 291。短 fragment 在 transfer 上仍然大量位于 boundary 内侧。

### Train：740 / 584 有没有被推离 boundary

对照把条目 59 与本 checkpoint 在同一 train overlay 上按 `payload + run + event + truth_id` 对齐（3331 / 3331）。

条目 61 冻结的 740 条 short-boundary / 584 条 event-max short：

| 584 个 event-max short | 计数 |
| --- | ---: |
| 现在 easy（`Δ ≥ 1`） | 280 |
| 仍 near-boundary | 287 |
| 变成 hard | 17 |
| 新 `Δ` 中位 | **0.992 < 1** |

740 条 short-boundary：377 条变成 easy，345 条仍 near，18 条变 hard；中位 `Δ` 1.002，刚擦过 margin。Train 全体：hard 39→28，near 989→618，short near 740→325，event-max short 584→274。4-station 上原来的 hard route **没有**被单独过拟合到 easy（对齐到的 10 条非短 hard 仍全部不是 easy）。

按预注册规则：event-max short 的新 `Δ` 中位仍 `< 1`，判 **`short_event_max_still_on_boundary`**，因此失败属于 **`objective_reduction_failure`**，不是 training-domain coverage。max reduction 移动了一部分 near-boundary，但没有把最危险的 per-event max 稳定推到冻结 margin 之外。

### `draw_01` / `draw_01_plus_common` twin

对齐键仍是 `origin_signature + truth_id`。配对 269 / 483（与条目 60 相同的覆盖限制）。

| 量 | `draw_01` | `+ common` | twin−chart |
| --- | ---: | ---: | ---: |
| `U_truth` 中位 | 1.159 | 0.816 | **−0.343** |
| selected | 367 | 333 | −34 |
| fragment winner | 66 | 82 | +16 |

配对丢失的 14 条：truth utility 下降 14 / 14，fragment 上升 5，truth drop 主导 **13**。attribution 仍是 **`truth_utility_drop`**。0.070 efficiency 差没有因为 max reduction 被消除。

## 决策

| 冻结条件 | 本条目 |
| --- | --- |
| 是否改 OP / 权重 / 架构 | **否** |
| 是否在 transfer 上比较 mean / max / top-k | **否** |
| layer 1 | **通过** |
| layer 2 / 3 | **不通过**（同一 `draw_01` family） |
| 584 event-max short 是否离开 `Δ≥1` | **否**（中位 0.992） |
| 失败归类 | **`objective_reduction_failure`** |
| 是否再训第二个 max / top-k / reweight 模型 | **否** |
| 冻结为新 production V2 | **否** |
| `continue_to_15d_relative_wls` | **false** |
| 打开 15D WLS | **否** |

下一步禁止继续 weighting / reduction / OP rescue。若以后重新打开四站 association，应讨论 **source-disjoint four-station training diversity** 或其它超出本 control 合同的问题，而不是再调这个已经冻结的 max reduction。
