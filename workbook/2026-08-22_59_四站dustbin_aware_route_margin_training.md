# 条目 59 — dustbin-aware route-margin objective control

日期：2026-08-22  
分支：`4station`  
状态：预注册 objective 已训练并打开 transfer 闸。`continue_to_15d_relative_wls: false`。layer 1 通过；layer 2 / 3 未全过。`U_truth` 已整体跨过 dustbin，剩余失败由 fragment 竞争与 `draw_01` twin 主导。不冻结为新的 production V2，不打开 15D WLS。objective / margin / 权重 / threshold / penalty / calibration 保持冻结。密封 test 未打开。

## 本条目允许做什么

只做一个预注册的 dustbin-aware route-margin objective control。不改 inference convention、threshold、unmatched penalty、Platt、Transformer 架构、candidate builder、unit-capacity solver、`physical_edge_deduplicated`。不在 development / transfer validation 上选 loss weight、margin 或 operating point。不扩大物理问题，不打开 test。

目标是让训练目标与生产 packing 决策边界一致：

\[
U_{\mathrm{truth}} > \max(U_{\mathrm{fragment}}, U_{\mathrm{dustbin}}=0)+m
\]

## 冻结 convention

合同：`configs/physical_four_station_dustbin_aware_route_training.yaml`  
架构拷贝：`configs/geometry_aware_transformer_v2_four_station_dustbin_aware.yaml`  
控制号：`retrained_v2_dustbin_aware_route_margin_v1`

| 量 | 冻结值 |
| --- | --- |
| 架构 / feature / candidate builder | 与条目 54/56 相同 |
| score stream | raw sigmoid + identity Platt |
| threshold | 0.001 / 0.001 / 0.001 |
| unmatched_penalty | −1.0 |
| dustbin | 0 |
| packing margin \(m\) | 1.0 |
| solver | `adjacent_contiguous_unit_capacity_set_packing` |
| 统计语义 | `physical_edge_deduplicated` |
| train overlay | 条目 56 已有 train bank |
| transfer overlay | 条目 56 已有 source-disjoint transfer，不新建 Athena |
| checkpoint | 固定 30 epoch 的最后一个 |

## Objective（第一次 step 前冻结）

保留历史 V2 四项和条目 56 的 fragment packing loss，**不删除**。只增加一个低容量辅助项，直接作用在 route utility margin，不重定义 inference score。

1. **Dustbin-aware route margin**：`relu(max(U_fragment, 0) + m − U_truth)`。
2. **原 packing competition**：仍用 `max(U_rival)`，rival 为空才代入 0。
3. **Gauge-twin consistency**：保留。零均值（第一阶段 nominal-only 没有 twin）使用预注册 fallback **1.0**，禁止再把权重顶到 clip 上限 20。这是条目 56/58 已记录的 scale compression 机制，不是 validation 调参。

权重算法：第一次 optimizer step 之前，对 train-only 做一次 no-grad 前向，按与 edge loss 同量级归一化，再 clip 到 `[0.05, 20]`；零均值项走 fallback。立刻写入 `outputs/mc24_four_station_dustbin_aware_route_v1/checkpoint/aux_loss_weight_contract.json`。之后禁止改。

## Transfer 闸（结果打开后禁止调参）

同一 source-disjoint transfer overlay：

1. raw complete-chain recall ≥ 0.90。
2. association：nominal purity ≥ 0.95、fake ≤ 0.05；非 nominal 相对 nominal：efficiency drop ≤ 0.10、purity drop ≤ 0.05、fake increase ≤ 0.05。报告 0→1 / 1→2 / 2→3 与 S3。
3. gauge twin：同一 `ΔT_ij` 下 route metric 一致；相对条目 54，raw logit drift 与 route-utility drift **不恶化**（不要求条目 56 那种 0.50 倍收缩）。

Mechanism audit：`U_truth` 是否跨过 0、fragment 竞争、dustbin winner 比例、selected 数量、2→3 / S3。

- 三层闸全过：冻结新 V2，第一次打开 route-selected 15D `ΔT_ij` WLS。
- association 仍失败，但 `U_truth` 已过 dustbin、主要输给 fragment：下一步只诊断 hard-negative mining。
- utility margin 仍失败：route objective 仍未覆盖 production decision boundary。

## 运行

```bash
source scripts/setup_environment.sh ml
bash scripts/run_four_station_dustbin_aware_training.sh train
bash scripts/run_four_station_dustbin_aware_training.sh infer-transfer
bash scripts/run_four_station_dustbin_aware_training.sh score-scale
bash scripts/run_four_station_dustbin_aware_training.sh mechanism
bash scripts/run_four_station_dustbin_aware_training.sh assess
```

批量 GPU 也可：`python scripts/submit_four_station_dustbin_aware_condor.py --submit`（`request_gpus = 1`）。

单测：dustbin-aware / packing / identifiability 共 26 项在训练前通过。

## 训练与闸结果

合同未改。数字只记录。15 维 WLS 未打开。条目 56 checkpoint `0e2ffe…cd80` 未被覆盖。

### 权重合同（第一次 optimizer step 之前）

`outputs/mc24_four_station_dustbin_aware_route_v1/checkpoint/aux_loss_weight_contract.json`

| 项 | 均值 | 权重 |
| --- | ---: | ---: |
| edge | 0.2430 | 1.0（历史主体，不归一化） |
| packing-route competition | 3.4415 | 0.07061055340401011 |
| dustbin-aware route margin | 5.2637 | **0.05**（clip 下限；原始比值 0.243/5.264 ≈ 0.046） |
| gauge-twin consistency | 0.0 | **1.0**（零均值 fallback，不是 clip 上限 20） |

`written_before_optimizer_steps: true`。`development_validation_used: false`。原 packing loss 保留；dustbin-aware 是第三项辅助，不重定义 inference score。

### 训练

- 节点：lxplus901 Tesla T4；seed `20260822`；固定 30 epoch 的最后一个
- 交互式 GPU，未提交 Condor
- checkpoint：`outputs/mc24_four_station_dustbin_aware_route_v1/checkpoint/route_aware_transformer_v2.pt`
- sha256：`6565d028e01e561105d4ce467d10d5c82e5a2435daad6a69c47a07920defb5dc`
- identity Platt 与 0.001 / −1.0 未改

| epoch | edge | packing | dustbin-aware | gauge twin |
| ---: | ---: | ---: | ---: | ---: |
| 0（nominal） | 0.241 | 3.267 | 5.032 | 0.000 |
| 7（nominal 末） | 0.292 | 1.365 | 2.409 | 0.000 |
| 8（stage 2 首） | 0.271 | 1.361 | 2.318 | 0.047 |
| 29（最后） | 0.048 | 0.096 | 0.097 | 0.028 |

gauge-twin 第二阶段 720 pairs，train loss 稳定在 0.03–0.05，没有条目 56 那种把绝对尺度压扁的迹象。

### Transfer 三层闸

决策：`outputs/mc24_four_station_dustbin_aware_route_v1/transfer_gate_decision.json`

| 层 | 结果 |
| --- | --- |
| 1 raw complete-chain recall | **过**：7/7 payload = 1.0；邻边 truth-edge recall 均为 1.0 |
| 2 association | **不过**：nominal 绝对闸过（purity 0.961、fake 0.037、efficiency 0.762、selected 537）；`draw_01_plus_common` efficiency drop 0.143 > 0.10 |
| 3 score-scale vs 条目 54 | **过且更好**：median \|Δ raw logit\| 0.093 / 0.630 = 0.148；median \|Δ truth-route utility\| 0.186 / 1.349 = 0.138 |
| 3 twin route-metric | **不过**：`draw_01` family efficiency \|Δ\| 0.108 > 0.05，2→3 \|Δ\| 0.102 > 0.08；另外两族通过 |

同一 overlay 上的条目 54 对照自己也未过 vs-nominal association 稳定闸。条目 56 在同一 packing 约定下 7/7 `selected_routes = 0`。本控制已经选出轨道，失败收缩到一个 gauge twin。

#### Association 分 payload

| payload | efficiency | purity | fake | vs-nom Δeff / Δpur / Δfake | 2→3 | S3 endpoints | selected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| reference | 0.762 | 0.961 | 0.037 | — | 0.759 | 451 | 537 |
| hard_s3_ry | 0.764 | 0.961 | 0.036 | −0.002 / 0.000 / −0.002 | 0.764 | 453 | 534 |
| hard_s3_ry + common | 0.743 | 0.940 | 0.060 | 0.019 / 0.021 / 0.022 | 0.736 | 443 | 519 |
| draw_00 | 0.702 | 0.919 | 0.071 | 0.060 / 0.042 / 0.034 | 0.667 | 410 | 506 |
| draw_00 + common | 0.718 | 0.933 | 0.058 | 0.043 / 0.028 / 0.021 | 0.690 | 422 | 516 |
| draw_01 | 0.727 | 0.934 | 0.057 | 0.035 / 0.027 / 0.020 | 0.703 | 428 | 509 |
| draw_01 + common | 0.619 | 0.923 | 0.067 | **0.143** / 0.038 / 0.030 | 0.602 | 370 | 445 |

2→3 与 S3 全部非零。只有 `draw_01 + common` 掉出 vs-nominal efficiency 与 2→3 recovery。

#### Ranking AUC（不用于选点）

nominal：0→1 / 1→2 / 2→3 = 0.987 / 0.990 / 0.979。七个 payload 的邻边 AUC 都在 0.966–0.991。条目 56 同 overlay nominal 为 0.974 / 0.984 / 0.964；ranking 没有被 route-margin 目标毁掉。

### Mechanism audit

`outputs/mc24_four_station_dustbin_aware_route_v1/transfer_mechanism.json`  
3381 条完整 truth route，全部 score-retained。

| 量 | 条目 56 transfer | 本控制 |
| --- | ---: | ---: |
| `U_truth` 中位 | −0.962 | **+0.870** |
| `U_truth ≤ 0` 比例 | 1.000 | **0.241** |
| selected truth | 0 | **2432** |
| production dustbin winner | 3381 | 433 |
| production fragment winner | 0（被 dustbin 掩盖） | 516 |
| `production_margin_satisfied` | 0 | 2384 |
| `truth_gt_fragment_lt_dustbin` | 1290 | 90 |
| `truth_lt_fragment` | 2091 | 907 |
| miner-satisfied / inference-fails | 0 | 0 |
| `U_best_fragment` 中位 | < 0 | +0.303 |
| required margin to dustbin 中位 | ~1.96 | 0.130 |

`U_truth` 已经整体跨过 dustbin = 0。fragment 竞争仍在：被 admit 的 fragment 赢 513–516 条，2-station / 3-station 仍进入 packing。dustbin winner 433/3381，且全部被旧 miner 记为 `dustbin_winner_missed_by_training_miner`（原 packing 项仍不把 0 放进 `max(rival)`；新辅助项才看 `max(rival, 0)`）。solver 实现一致：`margin_satisfied_solver_does_not_select = 0`（pooled 仅 3 条）。

## 决策

| 冻结条件 | 本条目 |
| --- | --- |
| layer 1 raw recall | 通过 |
| layer 2 association | **不通过**（一个 twin payload） |
| layer 3 score-scale vs 条目 54 | 通过，且不恶化 |
| layer 3 twin route-metric | **不通过**（同一 `draw_01` family） |
| `U_truth` 是否跨过 0 | **是**（中位 +0.870，75.9% > 0） |
| 剩余失败主导 | fragment，不是 dustbin |
| `continue_to_15d_relative_wls` | **false** |
| 冻结为新 production V2 | **否** |
| 打开 15D WLS | **否** |

机器闸输出的 `next_step = diagnose_association_after_dustbin_aware_objective`，因为自动规则用了 `admitted_fragment > selected`（513 ≯ 2432），在已经大量选中轨道时不会触发。按本条目预注册的机制规则：association 未全过，但 `U_truth` 已过 dustbin、未选中的 truth 主要输给 fragment（907 vs 90 条 dustbin-scale；production winner fragment 516 vs dustbin 433）。下一步只诊断 **solver-generated hard-negative mining**，不改 architecture，不改 operating point，不打开 15D。

本控制已经证明：在固定 `dustbin=0`、`unmatched_penalty=-1.0` 下，把训练目标改成 `max(U_fragment, 0)` 可以同时保持 edge ranking、把 gauge score-scale 做到不差于条目 54，并把绝大多数 complete truth route 送过 production admit boundary。剩下的是更硬的 fragment / common-SE(3) 竞争，不是绝对尺度再对不齐。
