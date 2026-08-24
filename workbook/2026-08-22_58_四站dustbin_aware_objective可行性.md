# 条目 58 — dustbin-aware route-margin 离线可行性

日期：2026-08-22  
分支：`4station`  
状态：train-only objective feasibility 已完成。`continue_to_15d_relative_wls: false`。`new_checkpoint_authorized: false`。主导失败族 = `truth_gt_fragment_lt_dustbin`；下一步若打开新 control = `design_dustbin_aware_route_margin_objective`。当前仍停留在 route competition objective 诊断。未训练、未改 checkpoint、未改 inference convention、未提交 Condor、未打开 15 维 WLS。密封 test 未打开。  
密封 test：永久关闭。

## 本条目允许做什么

只做 objective-level diagnosis 和离线 feasibility。不修改 inference convention，不调 threshold / unmatched penalty / Platt，不改 Transformer 架构、candidate builder、unit-capacity solver、`physical_edge_deduplicated`。不训练，不写新 checkpoint，不生产 Athena/Condor，不打开 test。

条目 57 已在 transfer bank 上证明生产失败是 dustbin。本条目回答训练目标能不能看见这条边界，以及 truth route 要抬多少 utility 才能满足 `dustbin = 0`。

## 冻结 convention

合同：`configs/physical_four_station_dustbin_aware_objective_feasibility.yaml`  
控制号：`dustbin_aware_objective_feasibility_v1`

| 量 | 冻结值 |
| --- | --- |
| candidate | `0e2ffe171e7cbbfd8ced426c9ce34759ffd6d0a2d6b26216df5814d48337cd80` |
| control | 条目 54 retrained V2 |
| score stream | raw sigmoid + identity Platt |
| threshold | 0.001 / 0.001 / 0.001 |
| unmatched_penalty | −1.0 |
| dustbin | 0 |
| packing margin | 1.0（条目 56 注册值，只用于诊断） |
| solver | `adjacent_contiguous_unit_capacity_set_packing` |
| 统计语义 | `physical_edge_deduplicated` |
| 数据 | 条目 56 train overlay，只加载 `train` |
| 源 | `mc24_100043_00200_00299`、`mc24_100044_00300_00399` |

诊断定义（不进入 optimizer）：

- `required_margin_to_dustbin = -U_truth + margin`
- `required_margin_to_fragment = U_best_fragment - U_truth + margin`
- 条目 56 miner strongest = `max(feasible rival)`，rival 为空才用 0
- dustbin-aware strongest = `max(rival, 0)`

## 运行

节点：lxplus901 Tesla T4。本条目是 scoring / audit，不是训练，因此不用 Condor。后续若打开新 objective control，批量 GPU 训练按 CERN Batch `request_gpus = 1` 提交，见 https://batchdocs.web.cern.ch/gpu/index.html。本条目没有提交任何 job。

```bash
source scripts/setup_environment.sh ml
bash scripts/run_four_station_dustbin_aware_objective_feasibility.sh
```

单测：`tests/test_dustbin_aware_route_margin.py` 与既有 packing / identifiability 测试共 24 项通过。checkpoint sha256 审计后仍为 `0e2ffe…cd80`。

产物：`outputs/mc24_four_station_gauge_consistent_route_v1/dustbin_aware_objective_feasibility_v1/`

- `audit_contract.json`
- `packing_loss_audit.json`
- `feasibility_report.json`
- `candidate_summary.json` / `control_summary.json`
- `candidate_truth_routes.jsonl` / `control_truth_routes.jsonl`
- `model_comparison.json`
- `decision.json`

## 条目 56 packing loss 审计

`packing_route_competition_loss` 源码与行为一致：

| 检查 | 结果 |
| --- | --- |
| 使用 `max(feasible rival utilities)` | 是 |
| 只在 rival 集为空或全被 mask 时代入 dustbin 0 | 是 |
| 显式 `max(rival, dustbin)` | **否** |
| 负效用 fragment 存在时遗漏 dustbin winner | **是** |

训练 enumerator **覆盖** 2-station、3-station prefix/suffix、mixed 4-station、共享 endpoint 冲突。它遗漏的不是这些 fragment 拓扑，而是生产里真正赢了的 **dustbin = 0**。

## Train-only 可行性（gauge-consistent V2）

7 payload × 240 event，完整 truth chain **3331**。全部 score-retained。Identity Platt + 0.001 / −1.0。

| 量 | 值 |
| --- | --- |
| `U_truth` 中位 | −0.729 |
| `U_truth ≤ 0` | **0.971** |
| `U_best_fragment` 中位 | −0.811 |
| selected | 97 |
| 生产 admit 的 fragment 赢 | **1** |
| `training_margin_satisfied_inference_fails` | **0** |
| `required_margin_to_dustbin` 中位 | **1.729** |
| `required_margin_to_fragment` 中位 | 0.922 |
| 条目 56 miner gap 中位 | 0.922 |
| dustbin-aware gap 中位 | **1.729** |
| miner 低估生产 gap | **3282 / 3331** |
| miner 已满足但生产失败 | 0 |

失败族：

| family | 条数 | 含义 |
| --- | ---: | --- |
| `truth_gt_fragment_lt_dustbin` | **2276** | ranking 对，输给 dustbin |
| `truth_lt_fragment` | 958 | 也输给未 admit 的更短片段 |
| `production_margin_satisfied` | 97 | `U_truth > max(fragment, 0)` |
| solver 公式不一致 | 0 | 不必查 clipped log-odds / normalization |

主导失败是 **`U_truth < 0`，不是错误 route 赢了 solver**。958 条 `truth < fragment` 的 fragment 几乎都 `U < 0`，生产同样选 dustbin。p95 `U_truth` 仍是 −0.289，所以这是尺度整体下移，不是尾巴。

要让中位 truth 满足 `U >= 0 + margin`，需要把 route utility 再抬约 **1.73**。当前 miner 只要求相对 fragment 再抬约 0.92，因此即使 packing 完全收敛，目标也比生产 decision boundary 低约 0.8。权重 0.0706 使这项实际也没收敛。

### 分 payload

| payload | n | selected | `U_truth` 中位 | `>frag,<dustbin` | `<frag` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `iteration_00_reference` | 473 | 11 | −0.758 | 307 | 155 |
| `iteration_00_hard_s3_ry` | 473 | 8 | −0.740 | 331 | 134 |
| `iteration_00_hard_s3_ry_plus_common` | 478 | 13 | −0.710 | 347 | 118 |
| `iteration_00_draw_00` | 473 | 15 | −0.725 | 320 | 138 |
| `iteration_00_draw_00_plus_common` | 478 | 17 | −0.718 | 315 | 146 |
| `iteration_00_draw_01` | 478 | 16 | −0.729 | 329 | 133 |
| `iteration_00_draw_01_plus_common` | 478 | 17 | −0.720 | 327 | 134 |
| 合计 | 3331 | 97 | −0.729 | **2276** | 958 |

## 为什么 ranking 在、admit 几乎全失败

邻边 median logit = 1.084（p ≈ 0.75）。四站 utility = `Σ logit − 4`，中位 −0.73，过不了 dustbin 0。source-level ranking 不依赖绝对零点，所以边仍能排对；unit-capacity 的 admit 条件是 `U > 0`，绝对尺度不够就全部进 dustbin。

同一 train overlay 上的条目 54 V2（同样 identity Platt + 0.001 / −1.0）：

| 量 | gauge-consistent V2 | retrained V2 |
| --- | ---: | ---: |
| selected | 97 | 3099 |
| `U_truth` 中位 | −0.729 | **+3.769** |
| `U_truth ≤ 0` | 0.971 | 0.012 |
| 边 logit 中位 | 1.084 | **2.606** |
| `required_margin_to_dustbin` 中位 | +1.729 | −2.769 |
| miner gap 中位 | 0.922 | 0 |

origin-aligned：median signed logit **−1.522**，median `U_truth` **−4.50**。`absolute_scale_compressed_vs_control: true`。gauge-twin consistency 把绝对 score scale 压低了约 1.5 logit / 4.5 utility，ranking 保留，生产 admit 塌缩。

## 决策

`outputs/mc24_four_station_gauge_consistent_route_v1/dustbin_aware_objective_feasibility_v1/decision.json`

| 条件 | 本条目 |
| --- | --- |
| 生产失败主因 | `U_truth < 0`（dustbin），不是错误 route |
| 训练目标已满足但推理不满足 | 否（0 条） |
| 下一 objective（若打开新 control） | **只设计 dustbin-aware route margin**，在固定 `dustbin=0` 下学绝对 utility |
| 现在就训练 / 新 checkpoint | **否** |
| 15D `ΔT_ij` WLS | **否** |
| 改 threshold / penalty / Platt / 架构 | **否** |

新 objective control 只有在完全 source-disjoint transfer 上同时通过 raw recall、association gate 和 gauge twin consistency 之后，才允许冻结新的 V2 并继续 route-selected 15D closure。当前继续停留在 route competition objective 诊断，不实施该 objective。
