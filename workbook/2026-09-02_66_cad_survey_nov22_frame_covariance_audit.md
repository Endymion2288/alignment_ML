# 2026-09-02 cad_survey_nov22 Frame / Covariance 审计 V1

## 任务

条目 65 已冻结
`survey_prior_interface_ready_candidates_lack_validated_ry_mapping_and_measurement_covariance`。
现有 association policy / V2 保持不变；真实数据仍是
`residual_dq_monitoring_only`；`geometry_write_allowed=false`。

本阶段接手新的 `docs/cad_survey_nov22.txt`，判断它是否真正补上条目 65
的缺口：独立、frame-correct、带可信 uncertainty 的 external measurement。
不训练、不改 pairwise/route policy、不增加 2024 r0022 collision-like
统计、不 Newton、不写 geometry / alignment payload。先完成
repository/status + 条目 60–65 合同审计、parser/provenance 和
frame-definition inventory，再做数值诊断。

## 仓库状态

```text
branch: master
HEAD:   a1fd01b Add survey/metrology IOV provenance and prior-interface feasibility chain (60-65).
ancestor check: a1fd01bb5c2c5daf432c13db3b021feb42bdf6d1 ⊂ HEAD
```

原始文件作为 immutable evidence 保存，SHA256
`1ea2e62d5340bc6242cdd887954cd92761d3472cf736e1831a5094ff83c1c8f6`，
21284 bytes / 192 unique sensors。未修改原始内容。

## 答案

冻结
`cad_survey_nov22_reproduces_slide_C_dx_but_lacks_validated_ry_mapping_measurement_covariance_and_cross_year_iov`。

`cad_survey_nov22` **不能**作为独立、frame-correct、带可信 uncertainty
的 alignment measurement。可复现的否定结论如下。

- **Parser / provenance：通过。** 索引 tuple 是
  `FaserSCT_ID` 的 `(station, layer, phi_module, eta_module, side)`，
  证据来自 `FaserSCT_ID.h`、`IdDictInterface.xml`、`SCT_Identifier.cxx`、
  `SCT_DetectorFactory.cxx`、`GeoModelTestAlg.cxx`，没有凭经验猜测。
  Delta FASER `(0.006016, 268.964, 1228.694)`、192 unique sensors、
  四个 station / 十二个 layer 的 quoted summary 全部解析。文件里的
  逐 sensor 行是三位小数重印；station/layer summary 才是全精度。
- **C_dx 中心值：复现条目 64。** 必须用 quoted layer means：
  `C_dx=(x_L0-x_L2)/2=+0.25411693479855924 mm`，与条目 64 幻灯片
  `+0.2541169 mm` 一致；`Δx_L0-L2=+0.5082338695971185 mm`。
  Station 0 quoted mean 在 5e-4 mm 内匹配幻灯片
  `(0.727, -0.982, -27.772) mm`。Layer 0/1/2 的平均精确等于 Station 0，
  因此 Layer 0–2 就是 IFT。三位小数 sensor 重算得到
  `C_dx≈0.2540625 mm`，不能用于条目 64 回归。
- **Frame：本文件已经是 2022 survey-adjusted FASER。** 文件头与
  Casper 2022-11-04 幻灯片写明：用 Stations 1–3 front/pigtail sensor
  平均把四个站平移到近似 FASER 原点；IFT 不参与该平移。轴顺序
  `(x 水平, y 竖直, z 束流)` 由文件头 + 幻灯片 + 条目 64 共同确定。
  **2021 native CAD → Calypso 旋转仍未唯一确定**，标为 unresolved
  assumption，不用于本文件的 C_dx。没有“看起来差不多”的符号选择。
- **Station ry：没有验证过的 Calypso mapping。** 诊断数字（禁止当 prior）：
  IFT Kabsch `ry=-7.7537 mrad`（同时 `rx=+4.085 mrad`、`rz=+3.048 mrad`，
  RMS 0.219 mm）会把 `C_dx` 吸进转动，其平移
  `(-13.70, -8.58, -27.84) mm` 也不是 quoted station mean
  `(0.727, -0.982, -27.772) mm`。`dz-vs-x` 诊断 `ry=-7.6956 mrad`，
  分层 `-7.831 / -7.683 / -7.573 mrad`（span 0.258 mrad < 0.5 mrad），
  与 layer-x slope 独立，但仍不是 `/Tracker/Align/Stations` 的 `ry`：
  缺唯一转动原点、缺 CAD/FASER→Calypso 约定、缺 measurement
  covariance、缺 IOV。禁止把 2021 CAD normal tilt、2022 layer-x slope、
  STEREOANGLE、LAYERPITCH、`/Tracker/Align/Planes` 非零转动自动映射成
  station `ry`。
- **Sigma：population spread，不是 Gaussian prior σ。** 文件 Sigma
  更接近 side=0 offset 的总体标准差（ddof=0），不是样本标准差，
  更不是测量误差。文件和相邻 PDF 都没有 per-point uncertainty、
  重复 survey、仪器精度或 fit covariance。因此
  `constructed_measurement_covariance=null`，availability 保持
  `feasibility_only`。
- **IOV：2022 survey 不能当 2024/2025 station 刚体修正。** Talk date
  是 2022-11-04。Station 刚性运动默认 IOV-specific；`C_dx` 仍是
  common-static candidate，等待 opening/thermal/metrology 证据。
  禁止跨年或跨 conditions tag 平均。
- **官方槽位：不填 `measured`。** 三条 ingest gate
  （validated parameter mapping / independent covariance / year-IOV）
  没有同时满足。`ift_C_dx` 与 `ift_l0_minus_l2_dx` 保持
  `feasibility_only`、`sigma=null`；`ift_station0_ry` 保持
  `unavailable`。因此 **不会进入 Fisher**，也不 Newton。

成功标准按合同执行：确定该文件是否提供独立、frame-correct、带可信
uncertainty 的 external measurement。答案是否定的。否定结论本身算完成。

## 输入 / 命令

- 配置：`configs/cad_survey_nov22_frame_covariance_audit_v1.yaml`
- 源：`docs/cad_survey_nov22.txt`（immutable）
- 相邻：`docs/04Nov2022_Survey.pdf`、`docs/FAS_IFT_Dec10-2021.pdf`
- Calypso 证据：`FaserSCT_ID.h`、`IdDictInterface.xml`、
  `SCT_Identifier.cxx`、`SCT_DetectorFactory.cxx`、`SCT_Station.cxx`、
  `SCT_Frame.cxx`、`SCT_Module.cxx`、`GeoModelTestAlg.cxx`、
  `TrackerAlignDBTool.cxx`、`geomDB.sql`
- 环境：`source scripts/setup_environment.sh ml`（`LCG_110_cuda`）

```bash
source scripts/setup_environment.sh ml
python -c "import torch; print(torch.cuda.is_available())"
pytest -q tests/test_cad_survey_nov22.py
python scripts/report_cad_survey_nov22_frame_covariance_audit.py
```

## Blockers（向硬件 / alignment / survey 团队索取的最小字段）

1. Nov 2022 wafer 表的 per-point 或刚体拟合 covariance：sensor 标识、
   测量全局 `(x,y,z)` 或相对命名 nominal 的 offset、该 frame 下 3×3
   （至少对角）测量协方差、仪器 / 日期 / 操作者 / 重复次数。
   **不要**把 station/module population SD 当 sigma 交来。
2. 该表对应的唯一 FASER/Calypso station-`ry` 定义：转动原点
   （station GeoModel origin vs sensor centroid vs support beam）、
   轴（`T*Rz*Ry*Rx` 之后的 Calypso global y）、若意图是 IFT in-plane
   yaw 则给出量化数字。**不要**交 LAYERPITCH slope、STEREOANGLE 或
   `/Tracker/Align/Planes` `ry`。
3. IOV 声明：2022 survey 可以约束哪一年 / 哪个 conditions tag；若用于
   2024/2025，需要机械稳定性证据；`C_dx` 是否在 opening/thermal 检查后
   被主张为 common-static。

## 冻结结论

- Frozen V2 / association policy 不变。
- 真实数据继续 `residual_dq_monitoring_only`。
- `geometry_write_allowed=false`；无 alignment payload。
- `cad_survey_nov22` 提供条目 64 的 per-sensor 来源和 C_dx 中心值，
  但不提供 validated Calypso station `ry`、不提供 measurement
  covariance、不提供 2024/2025 IOV。
- 下一步：保持 DQ monitoring；向 alignment/hardware/survey 团队索取
  上述三组字段。拿到之后再复用现有 track+external-prior Fisher，
  阈值和 scan grid 继承条目 61，不得看结果后调整。
  条目 67 已把 Stations `ry` 软件合同推进为“绕 FASER 原点的全局左乘”，
  但原始协方差和 2024/2025 IOV 仍未解除，因此仍不进入 Fisher。

## 数值结果（artifact）

| 量 | 值 | 用途 |
| --- | ---: | --- |
| quoted `C_dx` | `+0.25411693479855924 mm` | 复现条目 64；`feasibility_only` 中心值 |
| printed-sensor `C_dx` | `+0.2540625 mm` | 三位小数重印，**不**用于回归 |
| `Δx_L0-L2` | `+0.5082338695971185 mm` | 与 `C_dx` 同一 DoF |
| Station 0 quoted mean | `(0.726775, -0.982135, -27.772185) mm` | 文件 frame 平移 |
| IFT Kabsch `ry` | `-7.7537 mrad` | 诊断，`not_calypso_station_ry` |
| IFT `dz-vs-x` `ry` | `-7.6956 mrad` | 诊断，`not_calypso_station_ry` |
| file Sigma | population std `ddof=0` | 禁止当 Gaussian prior σ |
| measurement covariance | `null` | 不填 `measured` |

源 SHA256 `1ea2e62d5…1c8f6`；config SHA256 `a392f36d…34e1b`；
HEAD `a1fd01bb5c2c5daf432c13db3b021feb42bdf6d1`。

报告：`outputs/cad_survey_nov22_frame_covariance_audit_v1/`。
