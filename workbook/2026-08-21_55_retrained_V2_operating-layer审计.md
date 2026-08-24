# 2026-08-21 (55) retrained V2 operating-layer 机制审计与一次预注册控制

## 任务边界

条目 54 把问题从「历史 V2 不懂四站相对几何」收缩到 retrained V2 的
route score / calibration / packing operating-point 不够稳健。本条目
**只**做 validation 机制审计，并在诊断成立后评估 **一个**预先冻结的低容量
控制。保持不变：mode-0、candidate graph、retrained V2 checkpoint、
unit-capacity solver、`physical_edge_deduplicated`、预注册 association
闸、条目 52 两个 held-out 与密封 test。

没有打开 15 维未知关联 WLS，没有扩物理生产，没有改 Transformer 深度或
solver，没有扫描 unmatched penalty 去越过 0.10，也没有用单个
`draw_00_plus_common` 调参。

## 审计范围

同一本轮 train/validation curriculum。Focus 是唯一越界的
`iteration_00_draw_00`（S0 采样图）与
`iteration_00_draw_00_plus_common`（同一 `ΔT_ij` 的 left-SE(3) twin），
并对照 nominal `iteration_00_reference` 解释 fake≈0.10。

Overlay `alignment_iteration_shared_across_payloads`，462 条完整 truth
chain 按 `(run, event, origin signature)` 全部对齐。Packing 仍只用相邻边
校准 log-odds 之和加 `n_stations × unmatched_penalty`；V2 complete-route
query 只作诊断，不进 solver。Dustbin utility 恒为 0。

产物：`outputs/mc24_four_station_relative_association_retrain_v1/operating_layer_audit_v1/`。

## 机制分解（validation-selected OP：0.001 / 0.5 / 0.5，penalty +0.5）

462 条完整 truth chain：

| payload | 选出 | 缺 candidate | 低于 pair 阈值 | utility≤0 | packing 竞争 |
| --- | ---: | ---: | ---: | ---: | ---: |
| reference | 345 | 0 | 64 | 0 | 53 |
| draw_00 | 315 | 0 | 87 | 0 | 60 |
| draw_00_plus_common | 297 | 0 | 110 | 0 | 55 |

相对 nominal，twin 多损失 48 条完整 truth：阈值多丢 46，packing 多丢 2。
越界 0.104 几乎全部来自 **1→2 / 2→3 阈值 0.5**，不是 candidate graph，也
不是 complete-route utility 累加把 truth 送进 dustbin（utility_nonpositive
= 0）。

Gauge twin 逐 event：chart 选出而 twin 丢失 45 条（阈值 26、packing 19），
反向 27 条，净损失 18，与 315→297 一致。Twin 上新失败的 1→2 / 2→3 仍多为
**source rank-1**，但校准概率大幅下跌（1→2 中位 0.75→0.25，2→3 中位
0.76→0.42）。Platt 在 462/462 条链上保持 pair 内排序。因此 **边排名本身
稳定，绝对分数尺度 / gauge 一致性不稳定**。

低于阈值的 truth 边多数不是卡在 0.49：twin 上失败 2→3 的中位校准概率
0.17，74 条中 53 条 <0.30。这不是微调 0.5 能救的近阈噪声。

Packing 竞争：nominal 53 条里 35 条最强冲突假设是 **同一事件里另一条
truth-consistent 的 3 站片段**。0→1 阈值 0.001 放进弱边后，完整四站
utility `Σlogit + 2.0` 可以输给 1→2→3 后缀。定义了 raw 对比的完整四站
假/混合竞争者中，twin 上 18/19 在 **raw 概率**上就已经压过 truth；这是
少数，不是系统性 raw-ranking 失败。

## 为什么 2→3 回到 0.8–0.86，nominal fake 却到 ~0.10

Matched retraining 把 2→3 truth 的校准分数抬过 0.5，association efficiency
从冻结对照 ~0.5 回到 0.81–0.89。同一 validation 网格却选出
`unmatched_penalty = +0.5`（历史 −1.0）。四站完整 route 的正 utility 条件
从 `Σlogit > 4` 变成 `Σlogit > −2`；两站片段只需 `logit > −1`（p≳0.27）。

Nominal 754 条选中 route 中 390 条是 2/3 站片段。78 条 fake/mixed 里 59 条
是片段，52 条在历史 penalty −1.0 的 **反事实** utility 上将非正——这只是
诊断，不是新 OP。因此 fake≈0.10 与 2→3 恢复来自 **同一套更松的 packing
常数**，不是 2→3 表示又坏了。

## 预注册的唯一新控制

诊断属于「排名稳定、尺度/utility 不一致」，因此 **不**回到加深度或改
solver 的重训。预先冻结（在看该控制的 validation 数字之前写入
`configs/physical_four_station_operating_layer_control.yaml`）：

1. 冻结 retrained V2 **edge logits**；
2. 只在 **train** 相邻边上拟合 3 对 Platt（6 参数），`fit_split=train_only`；
3. Packing 常数复制历史冻结 V2：阈值全 0.001，`unmatched_penalty=-1.0`。
   这不是 penalty 网格，也不是用 `draw_00_plus_common` 选阈。

一次冻结评估，必须同时：nominal fake≤0.05 且 purity≥0.95；六个非 nominal
payload efficiency 下降 ≤0.10；gauge twin 继续通过预注册差限。

## 该控制的 validation 结果

`outputs/mc24_four_station_relative_association_retrain_v1/operating_layer_control_v1/`
`continue_to_15d_relative_wls=false`。

| payload | efficiency | 相对 nominal 下降 | purity | fake |
| --- | ---: | ---: | ---: | ---: |
| reference | 0.455 | — | 0.968 | 0.037 |
| hard_s3_ry | 0.446 | 0.009 | 0.954 | 0.065 |
| hard_s3_ry_plus_common | 0.491 | −0.037 | 0.958 | 0.070 |
| draw_00 | 0.481 | −0.026 | 0.945 | 0.066 |
| draw_00_plus_common | 0.435 | 0.019 | 0.935 | 0.086 |
| draw_01 | 0.374 | 0.080 | 0.940 | 0.057 |
| draw_01_plus_common | 0.338 | **0.117** | 0.934 | 0.068 |

- Nominal fake/purity **回到**原 capture 质量区间（0.037 / 0.968）。
- 原越界点 `draw_00_plus_common` 的 efficiency 下降降到 0.019，已 ≤0.10。
- Gauge twin **全部通过**。
- `draw_01_plus_common` efficiency 下降 **0.117 > 0.10**；draw_01 族 2→3
  vs-nominal 也超过 0.10。失败类 `association_domain_shift`。
- Nominal complete-track efficiency 从 0.747 掉到 0.455：train Platt
  （例如 0→1 slope 2.75、intercept −1.68）把中等 logit 压到低概率，再叠加
  历史 −1.0 后许多完整 truth 的 packing utility 变负。Station-pair Platt
  **不能**抹平 gauge twin 上同一 origin 边的绝对分数差。

没有做第二次控制，也没有放宽 0.10。

## 记录

**表示基本足够，association operating layer 尚未达到 production gate。**
不把 0.004 的边缘超限写成闸门失败的理由去放宽判据；唯一预注册控制在
source-disjoint validation 上没有完整通过，因此 **不**冻结该 V2，**不**
打开 15 维相对 WLS。

若以后继续，应回到 **同一 V2 架构** 的训练目标：显式 gauge-twin
consistency 与 route-competition loss，而不是再扫 threshold / penalty，
也不是加深度或改 solver。条目 52 held-out 与密封 test 仍封存。

## 产物

- 审计：`outputs/mc24_four_station_relative_association_retrain_v1/operating_layer_audit_v1/`
- 控制：`.../operating_layer_control_v1/`（含 `calibration.json`、
  `operating_point.json`、`control_gate_decision.json`）
- 配置：`configs/physical_four_station_operating_layer_control.yaml`
- 代码：`training/route_operating_audit.py`、
  `scripts/audit_four_station_route_operating_layer.py`、
  `scripts/run_four_station_operating_layer_control.py`
