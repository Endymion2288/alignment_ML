# 条目 61 — weighting / reduction feasibility

日期：2026-08-23  
分支：`4station`  
状态：objective / 推理约定冻结。本条目只做 **train-only** margin / reduction / route-length audit，并按预注册规则写下至多一个同架构 control。不在本条目训练、不写新 checkpoint、不打开 15D WLS、不加载 transfer、不用 transfer 选权重或 reduction。密封 test 未打开。  
密封 test：永久关闭。

## 本条目允许做什么

在条目 60 已经排除 miner 漏挖之后，只问两件事：train 上 2/3-station near-boundary competition 是否存在；以及当前 per-event mean 有没有把这些（或其它）危险 route 稀释掉。不改 threshold / `unmatched_penalty` / Platt / solver / margin / dustbin-aware 定义 / 架构。

## 冻结 convention

合同：`configs/physical_four_station_weighting_reduction_feasibility.yaml`  
控制号：`weighting_reduction_feasibility_v1`  
checkpoint：条目 59 `6565d028e01e561105d4ce467d10d5c82e5a2435daad6a69c47a07920defb5dc`

| 量 | 冻结值 |
| --- | --- |
| `Δ` | `U_truth - max(U_fragment, 0)`，`U_fragment` 来自生产 `_route_hypotheses` |
| hard | `Δ ≤ 0` |
| near-boundary | `0 < Δ < 1` |
| easy | `Δ ≥ 1` |
| short sparse | train 上 2/3-station hard+near `< 1%` 完整 truth route（预注册，不是 transfer 比例） |
| 若打开 control | 只改 reduction 为 per-event **max**，不改权重 |

## 决策规则（train-only，一次冻结）

- 2/3-station hard+near 为 0，或不足 1% route：判 **curriculum/domain-coverage**，不预注册 reduction control。
- short boundary 存在，但从来不是该 event 的 max dustbin-aware loss：max reduction 也训练不到它们，同样判 domain-coverage。
- short boundary 存在、会成为 event-max，且 mean 相对 max 稀释 boundary mass：预注册 **一个** `max_over_complete_truth_routes_in_event` control。
- 不得在 transfer 上比较 mean / max / top-k。

## 运行

```bash
source scripts/setup_environment.sh ml
bash scripts/run_four_station_weighting_reduction_audit.sh
```

单测：`tests/test_route_reduction_audit.py`、`tests/test_gauge_consistent_route.py`、`tests/test_dustbin_aware_route_margin.py`、`tests/test_solver_hard_negative_audit.py`。  
产物：`outputs/mc24_four_station_dustbin_aware_route_v1/weighting_reduction_feasibility_v1/`

## 结果

数字只记录，不回改合同。统计对象是 **全部 3331 条 train complete truth route**，不是条目 60 已经赢了的 14 条 production winner。`Δ` 用生产 convention，不把 `U ≤ 0` 的重枚举 rival 算进去。Transfer 未加载。

### 冻结 checkpoint 上的 train margin bins

| 量 | 计数 |
| --- | ---: |
| complete truth / events | 3331 / 1624 |
| easy / near-boundary / hard | 2303 / **989** / **39** |
| hard production winners | 14（与条目 60 一致） |
| 2/3-station hard | **0** |
| 2/3-station near-boundary | **740（22.2%）** |
| 4-station hard+near | 397（39 hard + 358 near） |

`Δ` 中位：全体 +1.088。按 competitor 长度：2 站 +1.385（easy），3 站 +1.111（刚过 near 上沿），4 站 +0.982（near / hard）。

### 所有 truth route 都先分箱，不只看 14 个 winner

条目 60 的 14 条 train winner 全部是 4-station mixed，且已经 `Δ ≤ 0`。本条目把其余尚未反超 truth 的短 fragment 也算进去：

| competitor 长度 | absent | easy | near-boundary | hard |
| --- | ---: | ---: | ---: | ---: |
| 2-station | 148 | 3064 | **119** | **0** |
| 3-station | 27 | 2593 | **711** | **0** |
| 4-station | 2554 | 380 | 358 | 39 |

2-station near-boundary 119 条里，`0→1` 49、`2→3` 70。3-station near-boundary 711 条里，prefix 336、suffix 375。  
namespaced origin（`9000000000→100043`，`9000000100→100044`）：short-boundary 260 / 480。  
按 family：`draw_01` 246、`draw_00` 166、`hard_s3_ry` 165、`reference` 163。短 near-boundary **不是**某一个 payload 或某一个 source 独有。

结论：transfer 上大量 2/3-station winner 对应的 train 样本 **存在**，只是还停在 `0 < Δ < 1`，没有变成 production winner。不能把“train 只有 14 个 winner”读成“train 没有短 fragment 竞争”。

### mean-over-truth-routes 之后的梯度 / loss 质量

当前 packing 与 dustbin-aware 都对每个 event 的完整 truth route 做 **mean**。easy 在 margin=1 下 loss 恒为 0，所以稀释不是“easy 也出 loss”，而是 **0 与非 0 平均**。

冻结条目 59 权重：packing `0.07061055340401011`，dustbin-aware **0.05（clip 下限）**，gauge 1.0。

| 量 | 值 |
| --- | ---: |
| mean dustbin-aware / packing loss | 0.0802 / 0.0801 |
| 加权后 packing 贡献 | 0.00565 |
| 加权后 dustbin-aware 贡献 | 0.00401 |
| hard+near 占 mean dustbin mass | **1.000** |
| hard 占 mean dustbin mass | **0.138** |
| near-boundary 占 mean dustbin mass | **0.862** |
| hard 条数占比 | 39 / 3331 = 0.012 |
| short-boundary 占 mean dustbin mass | 0.515 |
| 4-station dustbin loss 总和 | 195.1 |
| 3-station / 2-station 总和 | 79.5 / 11.1 |
| boundary event 完整 route 中位条数 | **2.0** |

Hard route 在 mean 之后仍只占 dustbin-aware 梯度质量的 **13.8%**（原始 per-route loss 质量 16.4%）。Near-boundary 占 **86.2%**。4-station competitor 的 per-route mean loss（0.251）远大于 3-station（0.024）和 2-station（0.0035），所以长度 bias 仍然把总质量推向 4-station mixed。

### 谁是 event-max（max reduction 实际会训练谁）

loss > 0 时的 event-max competitor 长度：

| 长度 | events |
| --- | ---: |
| 3-station | **458** |
| 4-station | 289 |
| 2-station | 30 |
| none | 8 |

short-boundary 成为该 event max dustbin-aware loss：**584**。  
`mean_dilutes_boundary = true`：boundary event 中位 multiplicity 2.0，event-max 会被另外 1–2 条 easy truth route 以 1/2 或 1/3 稀释。

因此问题 **不是** “train 物理分布里几乎没有短-fragment near-boundary”，而是 **objective reduction 没有强调已经存在的 near-boundary**。2-station 会成为 event-max 的只有 30 个 event，max reduction 主要放大的是 **3-station near-boundary** 和 **4-station mixed hard/near**。这条阅读不回改已经冻结的决策规则。

### Route-length bias（不改 solver utility）

在冻结 production utility 下：

- 2-station `Δ` 中位 +1.385，绝大多数 easy；near 119，hard 0。
- 3-station `Δ` 中位 +1.111，near 711，hard 0。
- 4-station `Δ` 中位 +0.982，是唯一出现 hard 的长度类。

这与 2-station `1·logit + 2·(−1)` 更容易 `U>0`、也更容易被 train 压到 `Δ≥1` 一致。Train **不是**缺少短 fragment 覆盖，而是缺少已经反超 truth 的短 hard case。按预注册 1% 规则，740 / 3331 = 22.2% 远高于 sparse 阈值，**禁止**用极端 reweighting 去硬学一个不存在的 2/3-station hard 分布，也 **禁止**因此改判为 domain-coverage 而停掉 reduction control。

## 决策

`decision.json`：

| 字段 | 值 |
| --- | --- |
| `train_short_boundary_is_absent` | false |
| `train_short_boundary_is_sparse` | false |
| `train_short_boundary_is_event_max` | true |
| `train_mean_dilutes_boundary` | true |
| `pre_register_hard_aware_reduction_control` | **true** |
| `reduction_if_opened` | `max_over_complete_truth_routes_in_event` |
| `reason` | `short_boundary_is_present_and_mean_dilutes_the_event_max_dangerous_route` |
| `next_step` | `design_hard_aware_max_reduction_control` |
| `do_not_increase_dustbin_weight` | true |
| `transfer_used_to_pick_reduction_or_weight` | false |
| `continue_to_15d_relative_wls` | **false** |
| `new_checkpoint_authorized` | **false** |

本条目 **预注册** 唯一同架构 control，**不训练**：

- 合同：`configs/physical_four_station_hard_aware_reduction_training.yaml`
- 架构 yaml：`configs/geometry_aware_transformer_v2_four_station_hard_aware_reduction.yaml`
- 控制号：`retrained_v2_hard_aware_max_reduction_v1`
- 只改 packing / dustbin-aware 的 per-event reduction：`mean → max`
- 条目 59 checkpoint architecture、identity Platt、0.001 / −1.0、margin=1、dustbin-aware target、gauge consistency、aux 权重全部原样复制
- 权重算法是 `copy_frozen_workbook59_aux_weights`，不因 max 变大而重新 scale
- 固定 30 epoch，GPU-only
- 以后只允许在 **同一个已经冻结的 transfer set** 上评估一次
- 不得在 transfer 上再比较 mean / max / top-k

闸：nominal purity≥0.95、fake≤0.05、全部 non-nominal `Δeff≤0.10`、gauge twin route metrics 全过、且 2→3 / S3 稳定，才允许冻结新 checkpoint 并第一次打开 route-selected 15D `ΔT_ij` closure。否则停止继续用 weighting / OP 救模型，并记录失败属于 objective reduction 还是 training-domain coverage。

| 冻结条件 | 本条目 |
| --- | --- |
| train 是否缺少 2/3-station near-boundary | **否**（740 条，22.2%） |
| 是否缺少已经反超的短 hard | 是（0 条），但不触发 sparse 规则 |
| mean 是否稀释 event-max | **是** |
| 是否预注册 reduction control | **是**（只改 max） |
| 是否在本条目训练 | **否** |
| 是否增加 dustbin 权重 / 改 OP / 改架构 | **否** |
| `continue_to_15d_relative_wls` | **false** |
| 打开 15D WLS | **否** |

下一步若执行该 control：

```bash
bash scripts/run_four_station_hard_aware_reduction_training.sh train
bash scripts/run_four_station_hard_aware_reduction_training.sh infer-transfer
bash scripts/run_four_station_hard_aware_reduction_training.sh score-scale
bash scripts/run_four_station_hard_aware_reduction_training.sh mechanism
bash scripts/run_four_station_hard_aware_reduction_training.sh assess
```
