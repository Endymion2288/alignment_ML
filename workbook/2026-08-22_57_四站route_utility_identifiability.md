# 条目 57 — 四站 route-utility identifiability audit

日期：2026-08-22  
分支：`4station`  
状态：冻结 checkpoint 下的 route-utility 可辨识性审计已完成。`continue_to_15d_relative_wls: false`。主导失败类 = `truth_beats_wrong_routes_but_loses_to_dustbin`；下一步 = `stay_on_route_competition_objective_diagnosis`。未训练、未改 checkpoint、未改 threshold / unmatched penalty / Platt、未打开 15 维 WLS。密封 test 未打开。  
密封 test：永久关闭。

## 本条目允许做什么

只对条目 56 已冻结的 gauge-consistent V2 做 production-utility 可辨识性审计。不训练，不改 checkpoint，不调 threshold / unmatched penalty / Platt，不打开 15D WLS，不改模型结构，不改 production contract。

目标是判断 `selected_routes = 0` 来自哪一层：

1. **absolute utility scale / dustbin**：truth route 胜过错误 route，但 `U_truth < 0`，被 dustbin 吞掉。
2. **packing competition 覆盖不足**：truth route 在 utility 上输给 fragment。
3. **训练目标与 solver 约定不一致**：negative mining 没覆盖生产 hard competitor，或 margin 已满足但 solver 仍不选。

## 冻结 convention（审计前预注册）

合同：`configs/physical_four_station_route_utility_identifiability.yaml`  
控制号：`route_utility_identifiability_v1`

| 量 | 冻结值 |
| --- | --- |
| candidate checkpoint | `outputs/mc24_four_station_gauge_consistent_route_v1/checkpoint/route_aware_transformer_v2.pt` |
| sha256 | `0e2ffe171e7cbbfd8ced426c9ce34759ffd6d0a2d6b26216df5814d48337cd80` |
| control | 条目 54 retrained V2，`ec3d40ea43a533e4883becf256d32a7e1966f1bd226a01988f42cced24c1f9a4` |
| score stream | raw sigmoid + identity Platt（两个模型都强制 identity，不用条目 54 存盘 Platt） |
| `0→1 / 1→2 / 2→3` threshold | 0.001 |
| unmatched_penalty | −1.0 |
| dustbin utility | 0 |
| complete-route query 注入 packing | 否 |
| solver | `adjacent_contiguous_unit_capacity_set_packing` |
| transfer overlay | `outputs/mc24_four_station_relative_transfer_validation_v1/overlay_synthetic_v1/` |
| 源 | `mc24_100047_00300_00349`（μ−）、`mc24_100048_00300_00349`（μ+），validation-only |

utility 定义与生产 `_route_hypotheses` 相同：`Σ clipped log-odds + n_stations × unmatched_penalty`。假设只在 `utility > 0` 时被 admit；空分配的 dustbin 效用恒为 0。本审计额外枚举 `U ≤ 0` 的 threshold-feasible competitor，否则无法看见被拒绝的 fragment。

## 运行

节点：lxplus901 Tesla T4。不改权重。

```bash
source scripts/setup_environment.sh ml
bash scripts/run_four_station_route_utility_identifiability.sh
```

单测：`tests/test_route_utility_identifiability.py` 与既有 operating / gauge 测试共 22 项通过。checkpoint sha256 审计后仍为 `0e2ffe…cd80`。

产物：`outputs/mc24_four_station_gauge_consistent_route_v1/route_utility_identifiability_v1/`

- `audit_contract.json`
- `candidate_summary.json` / `control_summary.json`
- `candidate_truth_routes.jsonl` / `control_truth_routes.jsonl`
- `model_comparison.json`
- `mining_coverage.json`
- `decision.json`

## 生产 winner：全部是 dustbin

7 payload × 483 条 complete truth chain = **3381**。candidate 上：

| 量 | 值 |
| --- | --- |
| `score_retained` | 3381 / 3381 |
| `below_threshold_or_missing_candidate` | 0 |
| `selected` | 0 |
| `truth_loses_to_fragment`（生产 admit 的 fragment 赢） | 0 |
| `margin_satisfied_solver_does_not_select` | 0 |
| `truth_beats_wrong_routes_but_loses_to_dustbin` | **3381** |
| `production_winner` | dustbin 3381 |
| `U_truth` 中位数 | −0.962 |
| `U_truth ≤ 0` | **1.0** |
| `ΔU = U_truth − max(U_competitor, 0)` 中位数 | −0.962 |
| `U_truth` p05 / p95 | −2.317 / **−0.609** |

`selected_routes = 0` 不是 candidate 丢边，也不是 threshold mask，也不是 solver 实现和公式对不上。p95 仍是 −0.609：连最好的 truth route 也低于 dustbin。要让 4-station 假设被 admit，需要 `Σ logit > 4`，即平均每边 logit ≳ 1.333（p ≳ 0.79）。candidate 三条邻边的 median p 只有 0.746 / 0.725 / 0.745（logit 1.078 / 0.969 / 1.073），和 −1.0 的 4 个 endpoint 罚刚好差约 1。

source-rank-1 仍然高（边级 0.950；完整链三条都 rank-1 约 0.88）。ranking 在，scale 不够。

### 分 payload（candidate）

每个 payload 都是 240 event、483 条 truth、`selected = 0`、`U ≤ 0` 全员成立。

| payload | `U_truth` 中位 | 胜过所有重叠 competitor | 输给未 admit fragment |
| --- | ---: | ---: | ---: |
| `iteration_00_reference` | −0.909 | 212 | 271 |
| `iteration_00_hard_s3_ry` | −0.935 | 214 | 269 |
| `iteration_00_hard_s3_ry_plus_common` | −0.990 | 177 | 306 |
| `iteration_00_draw_00` | −0.989 | 157 | 326 |
| `iteration_00_draw_00_plus_common` | −0.965 | 162 | 321 |
| `iteration_00_draw_01` | −0.941 | 198 | 285 |
| `iteration_00_draw_01_plus_common` | −1.024 | 170 | 313 |
| 合计 | −0.962 | **1290** | **2091** |

## 在 dustbin 之下，truth 有没有赢过 fragment

生产层答案已经确定：没有任何 fragment 被 admit，所以不存在“solver 选了错误 route”。  
utility 层还要看 `U_truth` 对未 admit competitor 的排序：

| 子集 | 条数 | 含义 |
| --- | ---: | --- |
| ranking 正确但输给 dustbin | 1290 | `U_truth >` 所有重叠 competitor，但仍 `< 0` |
| 输给未 admit fragment，同时 dustbin 赢 | 2091 | 主要是更短的 truth-consistent 片段 |
| `U_truth − U_best_competitor` 中位 | −0.091 | 几乎打平，略输 |

最强 overlapping competitor 族：

| family | 条数 |
| --- | ---: |
| `two_station_fragment` | 1993 |
| truth-consistent 3-station prefix | 564 |
| truth-consistent 3-station suffix | 573 |
| mixed 4-station | 242 |
| 非 truth-consistent 3-station | 9 |

负 logit + `unmatched_penalty × n_stations` 会偏爱更短的片段：同一条高分边做成 2-station 的罚是 −2，做成 4-station 的罚是 −4。当前 median 边分刚好落在“自己的 2-station 片段略高于完整 route、两者都低于 0”的区间。这是 combinatorial length bias，不是另一条错误粒子赢了。

## 训练 packing miner 覆盖了什么

条目 56 的 `packing_route_competition_loss`：

- **有覆盖**：2-station、3-station prefix `(0,1,2)`、suffix `(1,2,3)`、mixed 4-station、共享 endpoint 的冲突 route。生产里真正出现的 hard competitor 拓扑都在 enumerator 里。
- **没覆盖 dustbin winner**：miner 取 `max(feasible rival utilities)`，只有 rival 集为空才代入 dustbin 0。它**从不**计算 `max(rival, dustbin)`。
- 生产只要 `U ≤ 0` 就不 admit。因此只要存在任何负效用 fragment，训练负例就看不到真正赢了的 dustbin。

本审计：candidate 上 **3381 / 3381** 条 truth 都是 `dustbin_winner_missed_by_training_miner`。这些事件里都有 threshold-feasible fragment，所以 miner 的 strongest rival 是负的 2/3-station，不是 0。

若 packing 对 2-station 真收敛到注册 margin 1.0，而典型 `U_2 ≈ −0.9`，则 `U_4 > 0.1`，会顺带胜过 dustbin。末 epoch packing train loss ≈ 0.935、权重只有 0.0706，说明这项既弱也没收敛。当前诊断只记录这个错位，不改权重、不改 miner。

## 对照：同一 convention 下的条目 54 V2

两个模型都用 identity Platt + 0.001 / −1.0，不是条目 54 存盘的 0.5 / 0.5 / `unmatched_penalty = +0.5`。

| 量 | gauge-consistent V2 | retrained V2 |
| --- | ---: | ---: |
| selected truth routes | 0 | 1379 |
| `U_truth` 中位 | −0.962 | +0.784 |
| `U_truth ≤ 0` | 1.000 | 0.420 |
| `truth_loses_to_fragment` | 0 | 1276 |
| production dustbin winner | 3381 | 699 |
| 边级 source-rank-1 | 0.950 | 0.950 |
| 邻边 median p | 0.75 / 0.72 / 0.75 | 0.84 / 0.87 / 0.83 |

origin-aligned 比较（3381 条同一 truth route）：

| 量 | 值 |
| --- | ---: |
| median \|Δ edge logit\| | 1.170 |
| median signed logit（candidate − control） | −0.729 |
| median `U_truth`（candidate − control） | **−1.830** |
| median `ΔU`（candidate − control） | −0.500 |
| decision class 改变 | 2716 |
| control 选中但 candidate margin 未改善 | 1371 |
| control 选中且 candidate `U_truth ≤ 0` | **1379 / 1379** |

条目 56 已证明 gauge-twin 的 raw-logit / route-utility **一致性**相对条目 54 收缩（median \|Δ\| 比 0.033 / 0.034）。本条目证明同一改进**没有**抬高 solver decision margin：绝对 utility 反而掉了约 1.83，把全部 truth route 推到 dustbin 以下。gauge loss 改善的是 score consistency，不是 packing 决策。

条目 54 在同一冻结 convention 下仍然会选中约 41% 的 truth route，同时有 1276 条输给 **已被 admit 的 fragment**。那才是“真实 combinatorial competition 已进入 solver”的对照。gauge-consistent V2 把所有 utility 压到 0 以下之后，fragment 竞争被 dustbin 掩盖了。

条目 54 另有 10 条 `margin_satisfied_solver_does_not_select`。candidate 上这一类是 0，因此**不**进入 clipped log-odds / threshold mask / unmatched-penalty 计数的 solver 实现审计。

## 决策

`outputs/mc24_four_station_gauge_consistent_route_v1/route_utility_identifiability_v1/decision.json`

| 冻结条件 | 本条目 |
| --- | --- |
| truth route utility margin | **不满足**（`ΔU ≤ 0` 全员） |
| association gate | **不满足**（`selected = 0`） |
| gauge twin | 条目 56 已过 score-scale，route-metric 仍无法定义 |
| `continue_to_15d_relative_wls` | **false** |
| `open_15d_wls_authorized_by_this_audit` | **false** |

下一步仍停留在 **route competition objective 诊断**，不改 architecture，不打开 15D `ΔT_ij` WLS。

生产失败的直接原因是 **absolute utility scale 对不上 solver 的 dustbin = 0 约定**。次要机制是同一尺度下完整 4-station 常常略输给自己的 2-station / 3-station fragment。训练 miner 枚举了这些 fragment，但没有把 dustbin 放进 `max(rival, 0)`。三者里没有一条授权改 threshold、unmatched penalty、Platt 或模型结构。

若后续条目要动 objective，只允许改 packing negative mining 或 route-competition target（显式包含 dustbin），并继续冻结 checkpoint 结构与 operating point。本条目不实施该修改。
