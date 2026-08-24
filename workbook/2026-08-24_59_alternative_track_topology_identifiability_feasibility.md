# 2026-08-24 Alternative Track-Topology Identifiability Feasibility V1

## 任务

条目 57–58 说明 true cluster-local `r_u` 恢复了第三条 leakage 维，但
`ry/C_dx` 是否可分离取决于 track slope。本阶段把问题从 measurement 粒度
转成 **residual-blind track topology**：在看 residual / cosine 之前冻结
物理类别，再对每个预声明类别重算同一套 `{dx, ry, C_dx}` Jacobian。
不训练、不进 full module map、不写 geometry、不按 cosine 挑事件。

## 预声明类别（冻结于 Jacobian 之前）

Slope 定义是 spectrometer `Δx/Δz`，由 selected-route endpoint 的 tracklet
`(x,y,z)` 计算，**不是** 单站 SegmentFit `tx/ty`（后者中位数约 0.017，
会把几乎全部 ≥3-station 路线误标成 wide-angle）。

| id | 物理 cut | 可否解锁 module map |
|---|---|---|
| `ge3_station_with_ift` | ≥3 stations 含 IFT | 否（当前混合样本） |
| `ip_like_forward` | global slope < 0.5 mrad（~2.4× IP acceptance） | 否（control） |
| `intermediate_angle` | 0.5–2 mrad | 否（control） |
| `wide_incident_angle` | slope ≥ stereo/10 = 2 mrad | 是 |
| `complete_four_station` | 4-station selected route | 是 |
| `long_lever_arm` | 含 station 0 和 3 | 是 |
| `full_ift_three_layers` | IFT L0+L1+L2 | 否（常见覆盖） |

Cut 不是条目 58 的数据 tertile（0.00068 / 0.00154）。

## 问题

是否存在一种独立、可重复获得的真实 track topology，能补上普通 r0022
selected routes 缺失的 angular lever arm，把 `ry/C_dx` 变成可搬运的
alignment information？

## 答案

**No。** `no_portable_alternative_topology_in_current_r0022`

`go_to_full_module_identifiability_map=false`。ML 继续冻结。下一步不是
再堆同类 collision-like 事件，而是独立 external survey，以及当前 rec
树里尚未作为 named stream 存在的 non-collision 样本（cosmic-like /
beam-halo / TestBeam / 其它年份）。

## Inventory（12 个 frozen-V2 窗口，residual-blind）

1265 selected routes，811 条 Jacobian-eligible（≥3-station + IFT）。
完整 4-station selected route **一共 17 条**（14973 为 0，14974 为 2）。

14973/14974 的 ≥3-station+IFT 分解：

| run | ge3+IFT | ip-like | intermediate | wide (≥2 mrad) | 4-station |
|---|---:|---:|---:|---:|---:|
| 14973 | 82 | 3 | 36 | 43 | 0 |
| 14974 | 74 | 7 | 40 | 27 | 2 |

Occupancy 100-event 窗口里 four-station **tracklet coincidence** 约 8–10%；
parent tracklets 里四站同现可达 ~38%（14973：9210/24182）。Frozen V2
几乎不把它们收成 complete 4-station selected routes。扩大同类 r0022
统计不会自动得到长 lever-arm 样本。

EOS `/eos/experiment/faser/rec` 没有名为 cosmic/halo/calib 的 stream。
可见 alternatives：2022 r0022、2023 r0019–r0021、2025 r0023、TestBeam2021。
独立 survey 仍不可用。MC 5 mrad particle-gun 存在，但不是真实 topology，
未用于类别或阈值。

## 预声明类别的 Jacobian（同一套 true cluster-local `r_u`）

| 类别 | 14973 `\|cos(ry,C_dx)\|` / rank | 14974 | bootstrap 集中？ | 跨 run？ |
|---|---|---|---|---|
| ge3 混合 | 0.531 / 3 | 0.571 / 3 | 否（95% 上沿 0.82） | 点估计可，14975/76 漂到 0.75/0.88 |
| ip-like | 0.987 / **2** | 0.984 / **2** | N<8 | 稳定共线 |
| intermediate | **0.981 / 2** | **0.981 / 2** | 是，但共线 | 四 run 都是 rank 2、~0.98 |
| wide ≥2 mrad | 0.416 / 3 | 0.382 / 3 | **否**（14973 95% [0.285, 0.712]） | 14974 点估计一致；14975/76 漂到 0.686/0.798 |
| 4-station / long lever | 不足 | 2 条 | — | 不可用 |

物理 control 成立：slope < 2 mrad 的真实轨道在四次独立 run 上把
`ry/C_dx` **锁在 rank-2 共线**。wide-angle 打开 rank-3 且 14973/14974
点估计方向一致（0.42/0.38），但 bootstrap 不集中，monitoring run 不复现，
因此还不是可搬运 alignment basis。

## 决策含义

True cluster-local observable 仍然有效；缺的是 **可重复获得的高 lever-arm
真实拓扑**。当前 r0022 selected routes 里：

1. 约一半是 stereo/10 以下的碰撞类轨道，Jacobian 已证明稳定共线；
2. 过 2 mrad 的一半打开了第三维，但统计和跨 run 覆盖仍不够；
3. 4-station selected route 极稀有，不能靠再跑同类窗口堆积出来。

不允许打开 full module-level identifiability map。
