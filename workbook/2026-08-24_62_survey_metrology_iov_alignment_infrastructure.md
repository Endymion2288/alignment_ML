# 2026-08-24 Survey/Metrology Ingestion + IOV-Aware Alignment Infrastructure V1

## 任务

条目 61 已证明轨迹 Jacobian 无法稳定解除 `ry↔C_dx`。本阶段不再从
2024 r0022 collision-like tracks 榨取信息、不训练、不写 geometry、不产生
alignment payload。目标是把软件和统计框架准备到：

**一旦拿到真实、独立、带不确定度的 survey/metrology，就可以在对应 IOV 内
立即测试它是否足以解除 `ry↔C_dx`，而无需再改变 ML、track selection 或
alignment parameterization。**

## 答案

冻结 `survey_metrology_ingestion_ready_awaiting_independent_measurements`。

- External-constraint schema 已就位；当前全部 `value=null / sigma=null /
  unavailable`。设计尺寸与软件 gauge **不能**充当测量。
- `θ^(y)=θ_static+Δθ_IOV^(y)`：stereo / strip pitch / LAYERPITCH / 名义 z
  为 immutable static；station 刚性运动默认 IOV-specific；`C_dx` 为
  common-static candidate，可被 external metrology 覆盖；`dz` 不交回
  tracks。
- 2024 生产 `OFLCOND-FASER-05` 与协议 `OFLCOND-FASER-06` 是不同
  conditions provenance，禁止跨 IOV 平均 residual 或 correction。
- Track + external prior Fisher 组合器已用条目 61 冻结阈值做
  `feasibility_only` 单元测试：synthetic 20 mrad `ry` 或 0.315 mm `C_dx`
  在协方差匹配的 collinear toy 上恢复 rank-3，且不能进入 geometry
  candidate。
- 未重建 2024 r0022 Jacobian。

报告：`outputs/survey_metrology_iov_alignment_infrastructure_v1/`。

## 向硬件/alignment 团队索取的测量

不要把 `20 mrad / 0.315 mm` 写成最终目标。那只是 **degeneracy-breaking**。

| 优先级 | 测量什么 | 打破 degeneracy | 有物理用处 | stretch |
|---|---|---|---|---|
| 1 | IFT/station-0 相对 FASER global 的 `ry`（不是 module stereo） | ≲ 20 mrad | ≲ **0.5 mrad** | 0.208 mrad (IP) |
| 1 | IFT L0 相对 L2 的横向 `dx`；`C_dx=(dx_L0-dx_L2)/2` | `σ(C_dx)≲0.315 mm` 即相对位移 ≲ 0.63 mm | `σ(C_dx)≲` **0.080 mm**（strip pitch），相对位移 ≲ 0.16 mm | 0.010 mm / 0.020 mm |

任一 degeneracy-breaking 约束即可开始测试 ingest。当前 collinear tracks
在把 `ry` 收死后 `σ(C_dx)` 仍饱和在 0.345 mm，因此有用的 `C_dx` 必须是
**直接层间 metrology**。GeomDB LAYERPITCH、STEREOANGLE=40 mrad、
WriteAlignment 中性 POOL、软件 `dz=5 mm` **不要**当测量交来。

## Calypso hierarchy（schema 坐标系）

来自 `SCT_DetectorFactory` / `FaserSCT_GeoModelConfig` / `faser_reco.py` /
`geomDB.sql`：

- `/Tracker/Align/Stations` level 3 global；`Planes` level 2 global；
  station 0/1/2/3 模块槽为 Interface/Upstream/Central/Downstream，local。
- `STEREOANGLE=40.0` milliradian，两侧 ±20 mrad；`LAYERPITCH=31.5 mm`。
  二者都是设计，不是 survey。
- `TI12Data04` → `FASERNU-04`；当前 `faser_reco` 默认 conditions
  `OFLCOND-FASER-06`。`--noIFT` 只关 4-station CKF。
- `TrackerAlignDBTool::dirkey` 的 level 编号与 DetectorFactory 相反；
  schema frame 以 DetectorFactory `addChannel` 为准。

## IOV manifest

GRL v10 只提供**年**级 run/fill 范围，不是把不同 rec/conditions 合成一套
constants。

| IOV | geometry | conditions | IFT/`C_dx` | GRL run 范围（年） |
|---|---|---|---|---|
| 2022 r0021 | FASERNU-04 | OFLCOND-FASER-04 | 无 `C_dx` | 7733–9188 |
| 2022 r0022 | FASERNU-04 | OFLCOND-FASER-05 | 有 | 同上年 |
| 2023 r0021 | FASERNU-04 | OFLCOND-FASER-04 | 无 | 10419–11703 |
| 2023 cos | **FASERNU-03** | OFLCOND-FASER-04 | 无；xAOD 未保留 | 同上年 |
| 2023 ift | FASERNU-04 | OFLCOND-FASER-05 | 有 | 同上年 |
| **2024 r0022 production** | FASERNU-04 | **OFLCOND-FASER-05** | 有 | 14593–17230 |
| **2024 protocol** | FASERNU-04 | **OFLCOND-FASER-06** | 有 | 同上年，**不同 IOV** |
| 2025 r0023 | FASERNU-04 | OFLCOND-FASER-05（pool 文件名 FASER-06_2025） | 有 | 19093–20677 |

Fill/run 级自由度一开始不放开。

## 下一阶段

允许：把真实独立 survey 填进 schema 的 `measured` 槽，绑定到匹配的
year/conditions IOV，跑 track+prior Fisher，看是否解除 `ry↔C_dx`。

不允许：再堆 r0022 tracks、改 ML、改 route policy、把 synthetic prior
写成 geometry candidate、跨 IOV 平均 residual/correction。
