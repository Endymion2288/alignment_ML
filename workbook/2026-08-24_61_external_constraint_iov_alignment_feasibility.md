# 2026-08-24 External-Constraint + Year/IOV Alignment Feasibility V1

## 任务

条目 57–60 已经定位：true cluster-local `r_u` 能恢复额外 alignment
information，但现有真实 track topology 无法稳定、跨 run 地把 `ry` 与 IFT
内部 `C_dx` 分开。本阶段不再从 2024 r0022 collision-like tracks 单独解除
`ry↔C_dx`，不训练更复杂网络，不进 full module map，不写 geometry。问两件事：

1. **需要什么类型、什么精度的独立机械/测量约束才能真正打破 `ry↔C_dx`？**
2. **在这些约束存在时，2022/2023/2024/2025 是否应采用 common static
   geometry + year/IOV-specific corrections，而不是假设所有年份共享同一套
   alignment constants？**

## 答案

**第一，打破 degeneracy 需要独立 station-`ry` 或 IFT 层间 `C_dx` 约束；现有
corpus 里没有 as-built 数值。** 信息需求（不是 alignment payload）：

| 样本 | 解除 `|ρ|≤0.90` rank-3 的最弱 prior | 最紧 prior 后剩余误差 |
|---|---|---|
| 2024 r0022 `intermediate_angle` 14973（要求样本） | `σ(ry)≲20 mrad` **或** `σ(C_dx)≲0.315 mm` | `ry` 固定后 `σ(C_dx)→0.345 mm`（粗于 strip pitch 0.080 mm）；`C_dx` 固定后 `σ(ry)→10.8 mrad` |
| 同 IOV 14974 转移 | 同一量级：`20 mrad` / `0.315 mm` | 同形 plateau |
| 完全共线 analytic toy | `σ(ry)≲0.5 mrad` **或** `σ(C_dx)≲0.01 mm` | `ry` 固定后 `σ(C_dx)→8.5 µm` |

20 mrad / 0.315 mm 是 **degeneracy-breaking**，不是 strip-pitch 级 `C_dx`
测量。LAYERPITCH=31.5 mm 只给出单位映射 `C_dx ≈ ry × 31.5 mm`，不是测量。
GeomDB、WriteAlignment 中性 POOL、软件 `dz=5 mm` gauge 都不是 as-built
survey。

**第二，是。** 用 `θ^(y)=θ_static+Δθ_year^(y)`。官方
`FASER-0X_YYYY_Align.pool.root` 已经按年拆分；2024 生产
`OFLCOND-FASER-05` 与当前协议 `OFLCOND-FASER-06` 还是年内 IOV 差。禁止把
不同年份 residual 或 correction 直接合并。Fill/run 级自由度一开始不要放开。

冻结
`external_survey_or_metrology_required_and_year_iov_parameterization_required`。
`go_to_full_module_identifiability_map=false`。不产生 alignment payload。

报告：`outputs/external_constraint_iov_alignment_feasibility_v1/`。

## 方法约束

- 不发明 survey 数值。没有表就做 **prior-strength feasibility scan**。
- Prior 网格在看 cosine / residual **之前**按探测器尺度预声明：`σ(ry)` =
  100…0.01 mrad，`σ(C_dx)` = 10…0.001 mm。测量 σ 取 strip pitch 0.080 mm，
  不是 residual RMS。
- Combined Fisher 在 `(dx mm, ry mrad, C_dx mm)` 上堆高斯约束行
  `Cθ=b`；rank / `σ3/σ1` 用 **column-normalized** SVD，避免强 prior 抬高
  `σ1` 后假性降秩。
- 信息需求读 **rank-2 collinear** `intermediate_angle` Jacobian，不用已经
  有限样本 rank-3 的 mixed `ge3_station_with_ift`。
- 只 Jacobian 2024 r0022 dump 14973/14974。其它年份只做 provenance /
  track-sample manifest。

## External-constraint inventory

浅层检索 Calypso / CVMFS DBRelease / EOS `/eos/experiment/faser` 顶层。
EOS 无 `survey/`、`metrology/`、`condb/`、`alignment/` 目录。文件名匹配
survey/metrology/as-built 为空。不虚构数值。

| 来源 | 坐标系 / frame | 年份/IOV | 精度 | 独立于 track residual？ | 结论 |
|---|---|---|---|---|---|
| GeomDB `LAYERPITCH=31.5 mm` | GeoModel local z / cassette 设计 | geometry tag | 无 σ | 是 | 设计间距，映射 `ry↔C_dx`，不测任一者 |
| 名义 station z | FASER global z | FASERNU-04 | 无 σ | 是 | 设计位置 |
| stereo 20 mrad | module local | 全部 | 无 σ | 是 | 读出轴，不是 station `ry` survey |
| `/Tracker/Align` Stations/Planes/Interface* | global / local AlignableTransform | conditions tag | 无 | 是 | 软件槽位 |
| `FASER-*_YYYY_Align.pool.root` | `/Tracker/Align` | 文件名含年 | 无 | 是 | WriteAlignment **中性** constants，不是 metrology |
| `C_dx=(dx_L0-dx_L2)/2` | IFT outer contrast | 定义 | 无 | 是 | gauge，不是测量 |
| Station Mode `dz=0±5 mm` | global z | 分析约定 | 5 mm | 是 | **软件 gauge**，不约束 `ry`/`C_dx` |
| IFT cassette 刚性 | cassette | 静态 | 无文档 σ | 是 | 定性；不发明 σ |
| 光学/激光 as-built 表 | — | — | — | — | **未找到** |

可转换成先验的形式已预留：`θ_ry = θ_ry^ext ± σ_ry`，
`C_dx = C_dx^ext ± σ_Cdx`，或约束矩阵 `Cθ=b`。今天没有可填的中心值或 σ。

## Year / IOV parameterization

`θ^(y)=θ_static+Δθ_year^(y)`。长期制造/安装项为 common component；
year-specific 只允许在同一 IOV 内由数据约束。

| mode | 跨年 | 必须独立 | survey 若存在则固定 |
|---|---|---|---|
| stereo / strip pitch | common static | 否 | 否 |
| IFT LAYERPITCH / cassette drawing | common static | 否 | 是（as-built `C_dx`） |
| `C_dx` 内部变形 | 直到有拆装/热变形证据之前 common | 否 | 是 |
| station `ry` | 否 | 是，`Δθ_year^(y)` | 是 |
| station `dx/dy/rx/rz` | 否 | 是，年内数据项 | 否 |
| station `dz` | 否 | 否 | 是（gauge-like） |
| 2022 r0021 pre-IFT | 否 | 是；无 `C_dx` | 否 |
| TestBeam2021 `FASER-TB00` | 否 | 完全独立于 TI12 | 否 |

Manifest 要点：2022 r0021 `OFLCOND-FASER-04` `--noIFT`；2022 r0022 IFT-era；
2023 含 `FASERNU-03` cosmic rec（xAOD 未保留）与 `FASERNU-04` collision；
2024 生产 `-05` / 协议 `-06`；2025 少量 r0023；TB00 无 IFT。本阶段不对
非 2024 r0022 样本做 Jacobian，也不混 residual。

## Combined identifiability

Track-only 14973 intermediate：rank 2，`|cos(ry,C_dx)|≈0.981`，
`σ3/σ1≈0.0058`（191 行）。14974：rank 2，`|cos|≈0.981`。Mixed 14973：
有限样本 rank 3，`|cos|≈0.531`，**不作为** requirement 样本。

加入预声明 Gaussian prior 后，collinear 子空间在 `σ(ry)≤20 mrad` 或
`σ(C_dx)≤0.315 mm` 起稳定 rank-3，且对更紧的网格点保持解锁（plateau）。
Analytic toy 验证同一机制：完全共线时阈值更严（0.5 mrad / 0.01 mm）。

## 下一阶段

允许：拿到独立 station-`ry` 或 IFT 层间 metrology 之后，在 **每个 IOV 内**
拟合 `θ_static+Δθ_year^(y)`。

不允许：再堆 r0022 collision-like、训练更复杂 cluster 网络、full module
map、写 geometry、把跨年 residual/constants 合成一套 alignment constants。
