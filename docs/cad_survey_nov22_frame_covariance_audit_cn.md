# cad_survey_nov22 坐标系与协方差审计 V1

Workbook 66 / 2026-09-02。条目 65 已经把 survey-prior interface 准备好，
但没有验证过的 Calypso global `ry` mapping，也没有真正的 measurement
covariance。本阶段把 `docs/cad_survey_nov22.txt` 作为不可修改的证据接入，
只回答一个问题：该文件是否提供独立、frame-correct、带可辩护 uncertainty
的外部测量？

不重新训练 V2/V3/Transformer，不改冻结的 pairwise/route policy，不增加
2024 r0022 collision-like 统计，不发明 cosine cut，不进入 full-module
identifiability map，不运行 Newton，不写 geometry，不产生 alignment
payload。

## 冻结合同

- 真实数据仍是 `residual_dq_monitoring_only`。
- `geometry_write_allowed=false`。
- 冻结 V2 checkpoint SHA256
  `0c85a001677678163512c59bd5ceb6d08bdefaab79c0f4628659a4a8ad766a27`。
- 条目 61 解锁网格不重调：`σ(ry)≲20 mrad` 或 `σ(C_dx)≲0.315 mm`；
  收紧 `ry` 后剩余 `σ(C_dx)≈0.345 mm`；有用精度仍是
  `0.5 mrad / 0.080 mm`。
- Population scatter 不是 Gaussian prior `sigma`。
- 现有 `/Tracker/Align` constants 是 reconstruction alignment state，
  不是独立 survey。
- 没有机械稳定性证据时，2022 survey 不能当 2024/2025 station 刚体修正。

## 源文件

`docs/cad_survey_nov22.txt` 原样保存，未修改。

| 字段 | 值 |
| --- | --- |
| SHA256 | `1ea2e62d5340bc6242cdd887954cd92761d3472cf736e1831a5094ff83c1c8f6` |
| 大小 | 21284 bytes |
| sensors | 192 个不重复 `(station, layer, phi_module, eta_module, side)` |
| Delta FASER | `(0.006016, 268.964, 1228.694) mm` |

该 dump 的产生脚本不在 Calypso 或本仓库中。2022-11-04 Casper 幻灯片写明：
GeoModelTest 在未加 alignment 的情况下 dump as-built sensor 位置，把
FASER identifier 赋给 wafer，并用 Stations 1–3 front/pigtail 平均固定
support-beam 平移。本文件是那次比较的输出，不是原始仪器表。

## 标识 tuple

五维索引是 `FaserSCT_ID::wafer_id(station, layer, phi_module,
eta_module, side)`。物理含义来自 Calypso 源码，不是从图上猜的：

| 维 | 名称 | 含义 | 证据 |
| ---: | --- | --- | --- |
| 0 | `station` | 0=Interface/IFT，1=Upstream，2=Central，3=Downstream | `FaserSCT_ID.h`、`IdDictInterface.xml`、`SCT_DetectorFactory.cxx` |
| 1 | `layer` | 每站三层，站内由上游到下游 | `FaserSCT_ID.h`、`SCT_Station.cxx`、`NUMLAYERS=3` |
| 2 | `phi_module` | 竖直行：Bottom=0 … Top=3 | `IdDictInterface.xml`、`SCT_Frame.cxx` |
| 3 | `eta_module` | Starboard=−1，Port=+1，面向下游 | `FaserSCT_ID.h`、`IdDictInterface.xml` |
| 4 | `side` | 0=Upper/pigtail/front，1=Lower | `IdDictInterface.xml`、`SCTBRLMODULE SIDEUPPER=0` |

文件的 station/layer summary 只用 `side=0`，与 2022 幻灯片一致
（CAD wafer z 间距不是 GeoModel 的 0.6 mm gap）。

## Parser 回归

逐 sensor 行打印到三位小数。Station/layer summary 保留更高精度。精确复现
的对象是 quoted summary 块；从 printed sensors 重算只做舍入一致性检查
（`atol=5e-4 mm`）。

Quoted IFT layer means 给出

```text
C_dx = (x_L0 − x_L2) / 2 = +0.25411693479855924 mm
Δx_L0−L2 = +0.5082338695971185 mm
```

与 workbook 64（`+0.2541169 mm`、`+0.5082339 mm`）一致。Station 0 quoted
mean 在引用精度内匹配幻灯片 `(0.727, −0.982, −27.772) mm`。Layer 0/1/2
的平均精确等于 Station 0，因此它们就是 IFT。三位小数 sensor 重算得到
`C_dx≈0.2540625 mm`，不得用于条目 64 回归。

文件里的 `Sigma` 更接近 side=0 offset 的总体标准差（`ddof=0`），而不是
样本标准差。它是 sensor-to-sensor 几何散布，不是测量误差。

## 坐标系清单

本 dump 已经处于 2022 survey-adjusted FASER 坐标系：

- 文件头：四个站都平移 Delta FASER，以近似 FASER 原点。
- Casper 2022-11-04：三次平移来自 Stations 1–3 front sensor 的平均
  nominal 与平均 survey 位置匹配；IFT 不参与该平移。
- 文件轴顺序：`x` 水平，`y` 竖直，`z` 束流。

2021 native CAD → Calypso 旋转仍是
`unresolved_assumption_not_required_for_this_file_C_dx`。没有任何符号
因为“看起来差不多”而被选定。

Calypso station constants 仍是毫米/弧度的
`T(dx,dy,dz)*Rz(rz)*Ry(ry)*Rx(rx)`，提取为 `ry=asin(R_xz)`、
`rx=atan2(-R_yz,R_zz)`、`rz=atan2(-R_xy,R_xx)`。

## 几何诊断（不是 prior）

只用 side=0 sensors：

- Station 平移就是 quoted station means。
- IFT 相对位移来自 quoted layer means。
- IFT sensors 的 Kabsch 刚体拟合给出 `ry=-7.7537 mrad`
  （`rx=+4.085 mrad`、`rz=+3.048 mrad`，RMS `0.219 mm`）。该拟合会把与
  `C_dx` 相同的 L0/L2 `x` contrast 吸进转动，因此不是独立的 `ry`。其平移
  `(-13.70, -8.58, -27.84) mm` 也不是 quoted station mean
  `(0.727, -0.982, -27.772) mm`。
- `δz ≈ −ry (x−x0)` 与 layer-`x` slope 独立，给出 `ry=-7.6956 mrad`，
  分层 `-7.831 / -7.683 / -7.573 mrad`（span `0.258 mrad`）。

这些数字都不是 Calypso `/Tracker/Align/Stations` 的 `ry`。在转动原点、
唯一 CAD/FASER→Calypso 约定、measurement covariance 和 IOV 被证明之前，
禁止建立参数映射。2021 CAD normal tilt、2022 layer-`x` slope、module
stereo、`LAYERPITCH` 以及 `/Tracker/Align/Planes` 转动都不得映射成
station `ry`。

## 协方差与 IOV

文件和相邻 PDF 中都没有 per-point uncertainty、重复 survey uncertainty、
仪器精度或 fit covariance。Population `Sigma` 只用于诊断，不写入
Gaussian prior。`constructed_measurement_covariance` 保持 `null`。

Talk date 是 2022-11-04。Station 刚性运动默认 IOV-specific。`C_dx` 仍是
common-static candidate，等待 opening/thermal/metrology 证据。禁止跨
2022/2023/2024/2025 或跨 conditions tag 平均 residual、correction 或
constants。

## 官方约束槽

条目 62 的三条 ingest gate 没有同时满足：

1. Calypso station `ry` 的 parameter mapping 已验证 — **否**
2. 有独立 provenance 的 measurement covariance — **否**
3. 可用于 2024/2025 的 year/conditions IOV 已明确 — **否**

因此官方槽保持

| 槽 | availability | value | sigma |
| --- | --- | --- | --- |
| `ift_C_dx` | `feasibility_only` | `+0.2541169 mm` | `null` |
| `ift_l0_minus_l2_dx` | `feasibility_only` | `+0.5082339 mm` | `null` |
| `ift_station0_ry` | `unavailable` | `null` | `null` |

不填 `measured`。不进入 Fisher。不运行 Newton。

## 判定

`cad_survey_nov22_reproduces_slide_C_dx_but_lacks_validated_ry_mapping_measurement_covariance_and_cross_year_iov`

该文件是 2022 幻灯片 `C_dx` 中心值在 survey-adjusted FASER 坐标系下的
逐 sensor 来源。这还不足以写成 alignment prior 或 geometry candidate。
本阶段的成功标准是可复现的否定结论加上明确的数据需求。

## 向 alignment / hardware / survey 团队索取的最小字段

1. Nov 2022 wafer 表的 per-point 或刚体拟合 covariance：sensor 标识、
   测量全局 `(x,y,z)` 或相对命名 nominal 的 offset、该坐标系下 3×3
   （至少对角）测量协方差、仪器、日期、操作者、重复次数。不要把
   station/module population 标准差当 `sigma` 交来。
2. 该表对应的唯一 FASER/Calypso station-`ry` 定义：转动原点（station
   GeoModel origin vs sensor centroid vs support beam）、`T*Rz*Ry*Rx`
   之后的轴，以及若意图是 IFT in-plane yaw 则给出量化数字。不要交
   `LAYERPITCH` slope、`STEREOANGLE` 或 `/Tracker/Align/Planes` `ry`。
3. IOV 声明：2022 survey 可以约束哪一年 / 哪个 conditions tag；若用于
   2024/2025，需要机械稳定性证据；`C_dx` 是否在 opening/thermal 检查后
   被主张为 common-static。

## 报告

`outputs/cad_survey_nov22_frame_covariance_audit_v1/`

Workbook 67 从本冻结继续：
[Nov-2022 原始 metrology provenance 与 Calypso station-ry 合同](nov22_metrology_provenance_station_ry_contract_cn.md)。
Stations `ry` 的软件合同现已唯一（全局左乘，绕 FASER 原点）。原始
协方差和 2024/2025 IOV 仍未解除，因此官方槽位保持
`feasibility_only` / `unavailable`。
