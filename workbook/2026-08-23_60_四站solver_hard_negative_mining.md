# 条目 60 — solver-generated hard-negative mining 审计

日期：2026-08-23  
分支：`4station`  
状态：objective / 推理约定冻结。本条目只做 production solver hypothesis audit 与 train-only miner coverage / feasibility。不训练、不写新 checkpoint、不打开 15D WLS、不用 transfer 调参。密封 test 未打开。  
密封 test：永久关闭。

## 本条目允许做什么

只诊断条目 59 之后剩下的 fragment 竞争：生产 unit-capacity solver 的真实 hypothesis set 里，究竟是谁赢了 unselected truth route，以及条目 59 的 `max(rival)` miner 有没有把这个 winner 选成 strongest negative。不改 absolute utility scale，不增加 dustbin loss 权重，不改 margin / threshold / `unmatched_penalty` / Platt / solver / Transformer 架构。

## 冻结 convention

合同：`configs/physical_four_station_solver_hard_negative_feasibility.yaml`  
控制号：`solver_hard_negative_mining_feasibility_v1`  
checkpoint：条目 59 `6565d028e01e561105d4ce467d10d5c82e5a2435daad6a69c47a07920defb5dc`

| 量 | 冻结值 |
| --- | --- |
| score stream | raw sigmoid + identity Platt |
| threshold | 0.001 / 0.001 / 0.001 |
| unmatched_penalty | −1.0 |
| dustbin | 0 |
| packing margin | 1.0 |
| solver | `adjacent_contiguous_unit_capacity_set_packing` |
| production hypothesis | `baselines.route_assignment._route_hypotheses`（只含 `U > 0`） |
| transfer overlay | 条目 56/59 已有 source-disjoint bank |
| train overlay | 条目 56/59 已有 train bank |

生产 competitor **禁止**用 `enumerate_threshold_feasible_routes` 这种含 `U ≤ 0` 的重枚举代替。miner 覆盖率按同一 event 的 mined-negative set 计算，不把“拓扑被 enumerator 支持”当成覆盖。

## 决策规则（train-only）

- production hard winner 大量存在但未被 miner 选为 strongest，或 miner loss = 0 而 oracle loss > 0：只允许预注册一个同架构 **solver-in-the-loop hard-negative control**。
- production winner 已被 miner 充分覆盖但仍压过 truth：不立刻训练第二个模型，先检查 loss weighting、per-event reduction、route-length bias。
- 无论哪种，15D WLS 仍必须等下一次 control 在同一冻结 transfer 上过完 layer 1/2/3。

## 运行

```bash
source scripts/setup_environment.sh ml
bash scripts/run_four_station_solver_hard_negative_audit.sh transfer
bash scripts/run_four_station_solver_hard_negative_audit.sh train
```

单测：`tests/test_solver_hard_negative_audit.py` 9 项通过。产物：

`outputs/mc24_four_station_dustbin_aware_route_v1/solver_hard_negative_mining_v1/`

## 结果

数字只记录，不回改合同。生产 competitor 全部来自 `_route_hypotheses`（`U > 0`）。条目 59 miner 不调用 packer。

### Transfer（source-disjoint）

3381 条完整 truth route；selected 2432；`U_truth` 中位 +0.870（与条目 59 一致）。

| 量 | 计数 |
| --- | ---: |
| production fragment winner | **513**（条目 59 记 516；差 3 条来自本条目要求 `U_fragment > 0` 且未选中） |
| `truth < enumerated fragment` | **907**（与条目 59 相同） |
| 其中无生产 competitor、双方 `U ≤ 0` | 343 |
| 其中生产 fragment winner | 513 |
| 其中 truth 仍被选中 | 51 |
| winner 进入 mined-negative set | **513 / 513** |
| winner 被选为 strongest miner | **513 / 513** |
| missed training signal（oracle>0 且 miner dustbin-aware≤0） | **0** |
| winner 上 dustbin-aware loss 均值 | 2.038 |
| winner 上 oracle loss 均值 | 2.038 |

`max(rival)` **没有**被容易的 fragment 占住而漏掉生产 winner。生产 winner 就是 miner 选出的 strongest，并且已经产生非零梯度。

#### 513 个 production winner 的拓扑 / 成分

| 拓扑 | 计数 |
| --- | ---: |
| 4-station mixed | 204 |
| 2-station `0→1` | 119 |
| 3-station prefix | 101 |
| 3-station suffix | 68 |
| 2-station `2→3` | 21 |
| 2-station `1→2` | 0 |
| 另一条完整 truth | 0 |

| 成分 | 计数 |
| --- | ---: |
| own-truth partial | 300 |
| mixed route | 164 |
| fake-endpoint conflict | 43 |
| cross-truth endpoint conflict | 6 |
| 另一条 truth-consistent partial（单独类） | 0 |

按 competitor 站数：2 站 140、3 站 169、4 站 204。  
`2→3` / S3：293 / 513。  
event multiplicity：1 轨 47、2 轨 214、3 轨 252。  
namespaced source（`origin_run` 9000000000→100047、9000000100→100048）：**100048 = 381，100047 = 132**。失败集中在 μ− source，不是某一个 payload 独有。

`plus_common` 三族合计 235 个 winner，四个 chart payload 合计 278。common-SE(3) twin 会增加 winner，但不是唯一来源。

907 个 `truth<fragment`：343 条是条目 59 把 `U ≤ 0` 的重枚举 rival 算进去的 dustbin-scale 比较，**不是**生产 solver 的 admitted fragment。本条目不再用它们代替 `_route_hypotheses`。

### Train-only miner coverage（唯一决策面）

3331 条完整 truth；selected 3315；`U_truth` 中位 +1.601，全部 `U_truth > 0`。

| 量 | 计数 |
| --- | ---: |
| production fragment winner | **14** |
| winner 进入 mined-negative set | **14 / 14** |
| winner 被选为 strongest | **14 / 14** |
| missed training signal | **0** |
| winner 拓扑 | 14 条全部 `four_station_mixed` |
| winner 成分 | 13 fake-endpoint + 1 mixed |
| winner 涉及 `2→3` | 14 / 14 |

`decision.json`：

| 字段 | 值 |
| --- | --- |
| `production_hard_winners_fully_covered_as_strongest` | true |
| `pre_register_solver_in_the_loop_control` | **false** |
| `next_step` | `diagnose_loss_weighting_per_event_reduction_and_route_length_bias` |
| `continue_to_15d_relative_wls` | false |
| `new_checkpoint_authorized` | false |

### `draw_01` / `draw_01_plus_common` 的 0.108 efficiency 差

对齐键是 `origin_signature + truth_id`。overlay 里 `origin_tracklet_id` 几乎总是 `(0,1,2,3)`，只标识原始 `(run, event)`；`truth_id` 是按 synthetic event/slot 重新编号的，chart/twin 打包不一致时不能全部配对。`draw_01` 配对 269/483（未匹配各 214）；`hard_s3_ry` 几乎对齐（480/483）。

**配对子集（269 条，丢失 20 条 chart-selected / twin-lost）：**

| 量 | 值 |
| --- | --- |
| `ΔU_truth` 中位（twin−chart） | −0.497 |
| `ΔU_fragment` 中位 | +0.025 |
| truth utility 下降 | 20 / 20 |
| fragment utility 上升 | 7 / 20 |
| truth drop 主导 | **19** |
| fragment rise 主导 | 1 |
| attribution | **truth_utility_drop** |

**全量 payload（不依赖配对）：**

| 量 | `draw_01` | `+ common` | twin−chart |
| --- | ---: | ---: | ---: |
| `U_truth` 中位 | 1.057 | 0.669 | **−0.388** |
| `U_truth ≤ 0` 比例 | 0.238 | 0.335 | +0.097 |
| selected | 351 | 299 | −52 |
| fragment winner | 68 | 87 | +19 |
| winner `U_fragment` 中位 | 0.443 | 0.339 | **−0.104** |

common transform 之后 truth utility 明显下移；fragment utility 中位不升反降。0.108 efficiency 差来自 **truth utility 下移**，不是 fragment 在 common SE(3) 后相对抬升。`draw_00` 的 12 条配对丢失反而更像 fragment rise，但那不是本条目的 0.108 闸。

### 为什么 covered winner 仍在 transfer 上压过 truth

条目 59 的 miner 已经看见并选中了生产 winner，所以 **禁止** 打开 solver-in-the-loop control，也 **禁止** 再加 dustbin 权重 / 改 margin。

剩下的是训练信号稀释和 route-length / 域偏移：

1. **Train 几乎没有 transfer 上那些短 fragment。** train 的 14 个 winner 全是 4-station mixed fake-endpoint；transfer 的 140 个 2-station + 169 个 3-station winner 在 train 上并不赢 truth，因此 enumerator“支持该拓扑”没有变成梯度。
2. **Per-event `mean`。** `packing_route_competition_loss` / dustbin-aware 对每个 graph 的完整 truth route 做平均。train winner 所在 event 有 2 或 3 条 complete route，单条 hard case 只占 1/2 或 1/3（14 条的平均 share 0.381）。
3. **Hard route 占比。** train 14/3331 = 0.42%；transfer 513/3381 = 15.2%。14 条 winner 对 train 均值 dustbin-aware loss 的贡献大约 `14 × 1.14 / 3331 ≈ 0.005`，而 epoch 29 该项是 0.097——均值被大量“赢了但不够 margin=1”的 easy route 占据。
4. **权重在 clip 下限。** dustbin-aware 权重被写成 **0.05**（clip floor）；packing 0.071；edge body 仍是主项。epoch 29：`0.097 × 0.05` 对总 loss 0.102 只贡献 ~0.005。
5. **Route-length bias。** 2-station utility 是 1 个 hop + `2 × (−1)`，高置信单边很容易 `U > 0`。train 拟合把这类短 rival 压了下去，transfer 上 `0→1` 仍有 119 个 production winner。这不是 miner 没选中它们，而是 train 上它们几乎不产生 hard loss。

诊断写入 `solver_hard_negative_mining_v1/weighting_reduction_diagnosis.json`。

## 决策

| 冻结条件 | 本条目 |
| --- | --- |
| 生产 winner 是否被 miner 选为 strongest | **是**（train 14/14，transfer 513/513） |
| 是否预注册 solver-in-the-loop | **否** |
| 是否训练第二个模型 | **否** |
| 是否增加 dustbin 权重 / 改 OP / 改架构 | **否** |
| `draw_01` 0.108 差的来源 | truth utility 下移，不是 fragment 抬升 |
| `continue_to_15d_relative_wls` | **false** |
| 冻结为新 production V2 | **否** |
| 打开 15D WLS | **否** |

下一步只允许做 **loss weighting / per-event reduction / route-length bias** 诊断与至多一个同架构 control。不得把本条目的 transfer 数字拿去调权重。新的 control 仍必须在完全冻结的 source-disjoint transfer 上同时通过 nominal purity/fake、所有 non-nominal `Δeff≤0.10`、gauge-twin metric 和 2→3/S3 stability，才允许冻结 checkpoint 并第一次打开 route-selected 15D `ΔT_ij` closure。
