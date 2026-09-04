# Nov-2022 原始 Metrology Provenance 与 Calypso station-ry 合同 V1

Workbook 67 / 2026-09-02。条目 66 已冻结
`cad_survey_nov22_reproduces_slide_C_dx_but_lacks_validated_ry_mapping_measurement_covariance_and_cross_year_iov`。
本阶段不再从现有数字里榨一个 `ry`，而是系统寻找能够解除三条 ingest
gate 的独立证据，并从 Calypso 源码加上 geometry-only 有限差分写出
`/Tracker/Align/Stations` `ry` 的数学合同。

不重新训练 V2/V3/Transformer，不改冻结的 pairwise/route policy，不增加
2024 r0022 collision-like 统计，不发明 cosine cut，不进入 full-module
identifiability map，不运行 Newton，不写 geometry，不产生 alignment
payload。Kabsch `−7.75 mrad`、layer-`x` slope、`STEREOANGLE`、
`LAYERPITCH`、`/Tracker/Align/Planes` 转动和现有 conditions constants
都不得冒充 station `ry`。Population scatter 不是 measurement covariance。

## 冻结合同

- 真实数据仍是 `residual_dq_monitoring_only`。
- `geometry_write_allowed=false`。
- 冻结 V2 checkpoint SHA256
  `0c85a001677678163512c59bd5ceb6d08bdefaab79c0f4628659a4a8ad766a27`。
- 条目 61 解锁网格不重调：`σ(ry)≲20 mrad` 或 `σ(C_dx)≲0.315 mm`；
  收紧 `ry` 后剩余 `σ(C_dx)≈0.345 mm`；有用精度仍是
  `0.5 mrad / 0.080 mm`。
- 没有机械稳定性证据时，2022 survey 不能当 2024/2025 station 刚体修正。
- `C_dx` 仍是 common-static *候选*，等待 opening/thermal 证据。
  Station 刚性运动保持 IOV-specific。

## 判定

`nov22_raw_covariance_and_iov_unresolved_station_ry_is_global_left_multiply_about_faser_origin`

| blocker | 状态 | 解除所需的精确证据 |
| --- | --- | --- |
| Nov-2022 原始测量协方差 | **未解除** | 原始 wafer/sensor 表，带 per-point 或 fit covariance、仪器、日期、操作者、重复次数、SHA256 |
| Calypso `/Tracker/Align/Stations` `ry` 软件合同 | **已解除** | 全局左乘 `r′ = g·r`，绕 FASER 原点；`∂x/∂ry = +z`，`∂z/∂ry = −x` |
| survey 估计量 → Stations `ry` 映射 | **未解除** | 绕 FASER 原点的 survey 刚体转动，不是 sensor centroid / station GeoModel origin / support beam |
| IOV | **未解除，仅 2022** | 具名机械 / opening / closing / thermal 状态；若用于 2024/2025 还需 2022→目标年的独立稳定性证据 |

官方槽位保持 `feasibility_only` / `unavailable`。不进入 Fisher。不填
`measured`。

## Blocker 1 — 原始 metrology provenance（未解除）

只读 inventory，不凭文件名推断物理含义：

| 位置 | 结果 |
| --- | --- |
| 本仓库 | 只有 `docs/cad_survey_nov22.txt` 和两份相邻 PDF |
| Calypso 树 | 没有产生 `Delta FASER` / `cad_survey_nov22` / `04Nov2022` 的脚本 |
| `/eos/experiment/faser` 深度 2 | 没有 `survey/` / `metrology/` / `cad/` / `geo/`；README（2024-04-30）写明 reconstruction 在 `data0` |
| CVMFS `poolcond` | 只有逐年 `FASER-0X_YYYY_Align.pool.root`；那是 reconstruction conditions，不是 survey |
| `/eos/home-f/fcadoux`、`/eos/user/f/fcadoux`、`/eos/home-d/dcasper`、`/eos/user/d/dcasper` | 目录存在，权限 `drwx------`，`PermissionError`；本账号不可读 |
| `/eos/user/c/casper`、`/eos/home-c/casper` | 不存在 |

`docs/04Nov2022_Survey.pdf`（D. Casper，survey 数据来自 Franck/Cadoux，
2022-11-04）写明：

- Cadoux CAD，support beam 放在名义位置（平移待定）。
- wafer ID 手工赋给。
- GeoModelTest dump as-built sensor 位置，**未加 alignment**。
- Stations 1–3 front/pigtail 平均固定三个未知平移。
- IFT `z` 是他们“可能想写进名义几何”的唯一平移。
- Stations 1–3 “consistent with 350 µrad horizontal misalignment of the support beam”。

该 PDF 没有给出源数据文件名、仪器、操作者、重复次数或协方差。文献中的
UNIGE CMM 精度（NIMA 1034 (2022) 166825 / arXiv:2112.01116）是平面装配
计量（面内约 5 µm、面外 10–15 µm），**不得**当作本 dump 的 Nov-2022
协方差。

没有建立 machine-readable ingestion 层，因为那会凭空发明文件里不存在的
仪器 / 操作者 / 不确定度字段。`cad_survey_nov22` 的 quoted means 仍以
条目 66 的 parser 为准。

## Blocker 2 — Calypso station-`ry` 合同（软件：已解除；survey 映射：未解除）

### 源码路径

1. `TrackerAlignDBTool::stationAlignment` 把
   `[dx, dy, dz, rx, ry, rz]`（mm/rad）写成
   `T(dx,dy,dz) * Rz(rz) * Ry(ry) * Rx(rx)`
   （`TrackerAlignDBTool.cxx`）。
2. `/Tracker/Align/Stations` 在 `SCT_DetectorFactory.cxx` 注册为
   `addChannel(..., 3, TrackerDD::global)`。`dirkey` 的 level 标签是反的
   （它的 “level 1” = Stations）；folder 是同一套。
3. `SiDetectorManager::setAlignableTransformGlobalDelta` 做 conjugation
   `c = T.inverse() * g * T`，其中
   `T = child fullPhysVol getDefAbsoluteTransform`。
   因此 `T*c = g*T`，全局点变换为 **`r′ = g·r`**。
4. `FaserSCT_AlignCondAlg` 把 container 写入 `SCTAlignmentStore`。
   `FaserSCT_DetectorElementCondAlg` 把绑定该 store 的
   `SiDetectorElement` 拷进 `SCT_DetectorElementCollection`。
   `SiDetectorElement::center()` 从 store 读 `transformHit()`。
5. `GeoModelTestAlg` 当前 dump 的是
   `SCT_DetectorManager::getDetectorElementCollection()`，即
   **未对齐**的 DetStore collection。对 GeoModelTest 注入 `±ry` 只会重印
   as-built 位置，不能验证 Stations `ry`。以后若做 aligned dump，必须读
   `SCT_DetectorElementCollection`。

FASERNU-04 `SCTTopLevel-02`（HVS 107793）：SCT parent `z = 1237.4 mm`，
Interface `POSZ = −3097.55 mm`，所以 station 0 的 GeoModel origin 是
`z = −1860.15 mm`。该 origin **不是** 全局 `g` 的转动支点。

Planes（level 2）使用不同的 conjugation
`Translate(0,0, element z) * alignment * inverse`，因此 Planes `ry`
不是 Stations `ry`。

### 有限差分结果（仅软件灵敏度）

点云：`cad_survey_nov22` 中 24 个 IFT side=0 的 implied GeoModel
nominal（`corrected − offset`）。注入 `ry = ±0.0001 rad`。
这不是 alignment solve，也不是 geometry candidate。

| 估计量 | RMS(数值 − 模型) mm/rad |
| --- | ---: |
| `r′ = Ry(ry)·r`，绕 FASER 原点 `(0,0,0)` | `3.1e-6` |
| 绕 sensor centroid 转动 | `1860.15` |
| 绕 station GeoModel origin `(0,0,−1860.15)` | `1860.15` |
| 绕 Casper Delta FASER / support beam 转动 | `1228.69` |

`ry=0` 处解析 Jacobian：`∂x/∂ry = +z`，`∂y/∂ry = 0`，
`∂z/∂ry = −x`。IFT 平均 `z = −1860.15 mm`，所以
`⟨∂x/∂ry⟩ = −1860.15 mm/rad`。数值中心差分与有限 `Ry` 的 `sinc(ry)`
因子一致。

对 IFT 而言，sensor centroid 和 station GeoModel origin 几乎是同一点
（`z ≈ −1860.15 mm`，`x,y ≈ 0`）。二者都与 Stations `ry` **不同**：
绕这两个支点转动时 IFT 的 `⟨∂x/∂ry⟩ ≈ 0`，而 Stations `ry` 会把整个
IFT 平移 `z·ry`。

条目 17 物理 refit smoke（IFT `+10 mrad`，truth-matched track 状态）
给出 `Δx = −18.5981 mm` ⇒ `Δx/ry = −1859.81 mm/rad`。符号和量级与
IFT 平均 `z` 一致。那是 track-state 响应，不是 sensor-center dump，
只作为物理链一致性记录，不是 survey prior。

### 本阶段明确不做的事

不把 Kabsch `ry = −7.75 mrad` 或 `dz`-vs-`x` `ry = −7.70 mrad` 映射成
Stations `ry`。那些估计量绕 sensor centroid / 平均 `x` 转，有限差分
已经排除。软件合同是唯一的；survey 映射仍未建立。

## Blocker 3 — IOV（未解除，仅 2022）

Talk date 是 2022-11-04。PDF 和 `cad_survey_nov22` 都没有记录
opening / closing / thermal / 机械状态。CERN backbone survey 和 UNIGE
平面 CMM 早于 Nov-22，不能证明 2022→2024/2025 的 station 刚体稳定性。
逐年 Align POOL 文件是 reconstruction conditions，不是 survey。

- Station 刚性运动保持 IOV-specific。
- `C_dx` 仍是 common-static 候选，**不升级**。
- 2022 不能当 2024/2025 station 刚体修正。

若要把该 survey 用于 2024/2025，需要：

1. Nov-2022 survey 的具名机械 / 安装 / opening / closing / thermal 状态；
2. 覆盖 2022 到目标年的独立稳定性测量；
3. 明确它可以约束哪一个 conditions tag。

## 官方槽位

| 槽位 | availability | value | sigma |
| --- | --- | ---: | --- |
| `ift_C_dx` | `feasibility_only` | `+0.2541169 mm` | `null` |
| `ift_l0_minus_l2_dx` | `feasibility_only` | `+0.5082339 mm` | `null` |
| `ift_station0_ry` | `unavailable` | `null` | `null` |

条目 62 的三条 ingest gate 没有同时满足，因此不进入 Fisher。
条目 61 的 rank / sigma / scan 网格保持冻结、未使用。

## 未解除的假设（记录，不猜测）

- 2021 native CAD → Calypso 轴符号仍未唯一确定，本阶段不用。
- Cadoux/Casper 的 EOS 目录存在但本账号不可读；不能当成“里面没有数据”。
- 没有跑 live aligned sensor-center dump，因为 GeoModelTest dump 的是
  未对齐 manager collection。合同由源码加上 `T*c = g*T` 数值验证。
  以后的 dump 必须读 `SCT_DetectorElementCollection`。
- `cad_survey_nov22` 的 implied nominal 是 Casper 的 GeoModelTest
  as-built 重印，这里只当 FD 点云，不当测量。

## 向 alignment / hardware / survey 团队索取的最小字段

1. Nov-2022 原始 wafer/sensor 表：标识、测量 `(x,y,z)` 或相对命名
   nominal 的 offset、该坐标系下 3×3（至少对角）协方差、仪器、日期、
   操作者、重复次数、SHA256。不要把 population 标准差当 `sigma` 交来。
2. 绕 FASER 原点的 survey 刚体转动，并明确它就是
   `/Tracker/Align/Stations` `ry`。不要交 Kabsch、`dz`-vs-`x`、
   `LAYERPITCH`、`STEREOANGLE` 或 Planes `ry`。
3. 上面列出的 IOV / 机械稳定性声明。

## 报告

`outputs/nov22_metrology_provenance_station_ry_contract_v1/`

源 SHA256 `1ea2e62d5340bc6242cdd887954cd92761d3472cf736e1831a5094ff83c1c8f6`。
配置 SHA256 `446d87eb322b9d2b60ae5bda7c87ee361662e354845b79bc4aa752553d3c3008`。
HEAD `a1fd01bb5c2c5daf432c13db3b021feb42bdf6d1`。
