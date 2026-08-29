# 2026-08-24 Non-Collision / Cross-Year Track-Topology Inventory V1

## 任务

条目 59 表明 true cluster-local `r_u` 能打开第三条 leakage 维，但普通
2024 r0022 collision-like selected routes 在 `slope<2 mrad` 上稳定
`ry↔C_dx` rank-2 共线，wide-angle 虽可 rank-3 却统计不足、跨 run 不可搬运。
本阶段不再堆同类 r0022、不训练、不进 full module map、不写 geometry、
不按 cosine 挑事件，而是问：

**FASER 现有真实数据中是否存在一种与普通 2024 r0022 collision-like tracks
不同、可重复获得并具有足够 angular / lever-arm information 的 topology，
可以稳定解除 `ry↔C_dx` 退化？**

## 答案

**No。** 正式冻结 `real_track_topology_insufficient_for_ry_cdx_separation`。

`go_to_full_module_identifiability_map=false`。ML 继续冻结。后续主线是
**external survey / 机械约束** 与 **year/IOV-dependent alignment framework**，
不是继续增加 ML complexity。

报告：`outputs/noncollision_crossyear_topology_identifiability_v1/`。

## 方法约束

- 类别在看 residual / Jacobian / cosine **之前**按 run metadata 与
  reconstruction flags 预声明（`2024_cos`=`--cosmics` beam mode，
  `2023_cos`=named no-stable stream，`*_back`=`--backward`，
  `*_alps`=物理 filter，`*_ift`=`--useIFT` collision GRL，
  TestBeam2021=`FASER-TB00`）。
- Slope 用 **单条 CKF track** 的 `(x0,y0,z0)→(x1,y1,z1)`，不用本地
  SegmentFit `tx/ty`，也不用同一事件里未关联 TrackSegment 的端点连线。
- 不同年份只作 topology source；禁止把 residual 或 alignment constants
  与 2024 混合。
- Jacobian 只在预声明门槛通过后才复用条目 57–59 的 true cluster-local
  `r_u`：cosmic-like 需 IFT、xAOD、同 IOV 至少两 run，并且
  CKF-wide≥90% **或** 四站符合率 ≥3× r0022 occupancy（~9%）。
  Collision-like 年份/stream 即使 mix 从 55% 漂到 70% 也不准入。

## Provenance / IOV

`faser_reco.py --noIFT` 只关闭 4-station CKF，**不**关闭 SegmentFit。
IFT cluster 仍可存在。`C_dx` 仍然要求 station 0。

| 来源 | geometry | conditions | 探测器 / rec | xAOD |
|---|---|---|---|---|
| 2022 r0021 | FASERNU-04 | OFLCOND-FASER-04 | `--noIFT` 三站 CKF | clusters + SegmentFit |
| 2022 r0022 | FASERNU-04 | OFLCOND-FASER-05 | 4-station CKF 开（无 `--noIFT`） | 同上，含 `CKFTrackCollection` |
| 2023 r0021 | FASERNU-04 | OFLCOND-FASER-04 | `--noIFT` | 有 |
| 2023_cos rec log | **FASERNU-03** | OFLCOND-FASER-04 | TI12Data03 | **无保留 xAOD**（仅 `r0019_log`） |
| 2023_ift ntuple | FASERNU-04 | OFLCOND-FASER-05 | `--useIFT` + 2023 GRL | PHYS 有；`rec/2023/dev` 空 |
| 2024 r0022 / r0023 | FASERNU-04 | OFLCOND-FASER-05 | `--noIFT`；当前协议读 **OFLCOND-FASER-06** | 有 |
| 2025 r0023 / p0014 | FASERNU-04 | OFLCOND-FASER-05 | `--noIFT --triggerMask 27` | r0023 仅 3 个 run |
| TestBeam2021 r0010 | **FASER-TB00** | **OFLCOND-FASER-TB00** | 无 LHCData，无 Segments | clusters + SegmentFit |

PHYS 有 named `2023_cos`（17 run）和 `2024_cos`（覆盖全部 276 个 r0022 run）。
Rec 树里没有名为 cosmic/halo 的 reconstruction stream。

## Residual-blind topology

相对冻结的 2024 r0022 基线：occupancy 四站符合 ~9%，V2 完整四站 selected
route 17/1265，Jacobian-eligible wide ~50%，PHYS `2024_ift` CKF-wide ~55%。

| 预声明类 | 代表样本 | IFT / 四站 / CKF-wide | 结论 |
|---|---|---|---|
| `2024_cos` `--cosmics` | 14587: 59 ev；14973: 164 ev | 四站 0%；几乎无 CKF | beam-mode 过滤几乎没有 tracker 轨迹 |
| `2023_cos` | 11780 / 11792 ~5–6k ev | IFT cluster ~14%，ge3+IFT ~24 ev，四站 ~0%，无多站 segment | 不是高 lever-arm 样本；xAOD 未保留 |
| `*_back` | 2024_back 14973 | 四站 94%，CKF-wide **56%** | 与 collision 同分布的反向 CKF |
| `*_alps` | 2024_alps 16064 | CKF-wide **58%** | 物理 filter，不是大角度拓扑 |
| `*_ift` collision | 2022_ift / 2023_ift / 2024_ift | 四站 93–96%，CKF-wide 70 / 75 / 55% | 仍是 collision-like，且跨年 IOV 不同 |
| 2025 r0023 | 20012 | 四站 19%，CKF-wide 73% | 仍是 TI12 collision；不能与 2024 混 residual |
| TestBeam2021 | xAOD 71038 ev | 无 IFT/`C_dx` 几何 | 不能解 `ry↔C_dx` |

未关联的事件内 TrackSegment 端点连线会把 2024_ift 误标成 ~64% wide，
**不用**它做门槛。

## Jacobian

准入列表为空，因此 **没有** 对任何跨年/cosmic 源复用条目 57–59 Jacobian。
跨年只比较 identifiability structure 的步骤未启动；更禁止比较 correction
数值。`portable_high_lever_arm_topology_found` 未置位。

## 下一步（允许）

1. 独立 external survey / 机械约束，给 `ry` 或 IFT 层间 `C_dx` 一个
   Jacobian 之外的先验。
2. Year / IOV-dependent alignment framework：2022/2023/2024/2025 生产
   条件分别是 OFLCOND-FASER-04 或 -05，当前协议读 -06，TestBeam 是 TB00。
3. 不要再堆 2024 r0022 collision-like 统计，不要为 `ry↔C_dx` 训练更复杂网络。
