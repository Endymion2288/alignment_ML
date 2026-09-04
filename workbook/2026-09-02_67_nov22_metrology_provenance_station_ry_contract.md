# 2026-09-02 Nov22 原始 metrology provenance + Calypso station-ry transform contract

## 任务

条目 66 已冻结
`cad_survey_nov22_reproduces_slide_C_dx_but_lacks_validated_ry_mapping_measurement_covariance_and_cross_year_iov`。
现有 association policy / V2 保持不变；真实数据仍是
`residual_dq_monitoring_only`；`geometry_write_allowed=false`。

本阶段不再从现有数字里榨一个 `ry`。按合同推进三个 blocker：

1. 只读 provenance inventory：工作区、Calypso tree、FASER 文档、
   survey/metrology 文件、ROOT metadata、conditions provenance、
   `/eos/experiment/faser` 已有 reconstruction/survey artifacts。寻找
   Nov-2022 原始输入、产生 `cad_survey_nov22.txt` 的脚本、原始
   wafer/sensor 表、laser tracker / theodolite / UNIGE / CERN
   metrology、重复 survey、仪器精度或 fit covariance。不凭文件名推断
   物理含义。
2. 正式定义 Calypso station `ry` 的数学合同，并用代码 / 最小数值
   perturbation 验证，而不是靠读变量名猜。
3. IOV：Nov-2022 发生在什么机械状态下；它能直接约束哪些年份和
   conditions tags。没有独立稳定性证据时，只记录为 2022 measurement。

不训练、不改 pairwise/route policy、不增加 2024 r0022 collision-like
统计、不 Newton、不写 geometry / alignment payload。三条 gate 没有
同时满足时不进入 Fisher。

## 仓库状态

```text
branch: master
HEAD:   a1fd01b Add survey/metrology IOV provenance and prior-interface feasibility chain (60-65).
ancestor check: a1fd01bb5c2c5daf432c13db3b021feb42bdf6d1 ⊂ HEAD
host:   lxplus908.cern.ch
CUDA:   True (Tesla T4)
```

`docs/cad_survey_nov22.txt` 仍是 immutable evidence，SHA256
`1ea2e62d5340bc6242cdd887954cd92761d3472cf736e1831a5094ff83c1c8f6`。
未修改原始内容。

## 答案

冻结
`nov22_raw_covariance_and_iov_unresolved_station_ry_is_global_left_multiply_about_faser_origin`。

三个 blocker 分别推进到明确状态。成功标准按合同执行：不是必须得到
alignment correction，而是把三个 blocker 写成 resolved / unresolved
with exact required evidence。最终仍不能建立可信 external prior，因此
冻结否定结论，不人为补全缺失信息。

- **Blocker 1 原始协方差：unresolved。** 工作区、Calypso、
  `/eos/experiment/faser` 深度 2、CVMFS `poolcond` 都没有 Nov-2022
  原始 wafer/sensor 表、产生 `cad_survey_nov22.txt` 的脚本、laser
  tracker / theodolite / UNIGE / CERN 原始文件、重复 survey 或 fit
  covariance。Cadoux/Casper EOS 目录存在但权限 `drwx------`，本账号
  `PermissionError`，不能当成空。NIMA 1034 / arXiv:2112.01116 的 UNIGE
  CMM 精度是平面装配计量，禁止当作本 dump 的协方差。没有建立
  ingestion 层，因为那会发明文件里不存在的字段。
- **Blocker 2 软件合同：resolved；survey 映射：unresolved。**
  `TrackerAlignDBTool::stationAlignment` 是
  `T*Rz*Ry*Rx`；`setAlignableTransformGlobalDelta` 给出
  `c = T^{-1} g T`，因此全局点 `r′ = g·r`，转动支点是 FASER 原点
  `(0,0,0)`，不是 station GeoModel origin `z=−1860.15 mm`，也不是
  sensor centroid，也不是 Casper support-beam / Delta FASER。
  小角度 `∂x/∂ry = +z`，`∂z/∂ry = −x`。在 cad_survey 的 24 个 IFT
  side=0 implied nominal 上注入 `ry=±0.0001 rad`：绕 FASER 原点的
  RMS 是 `3.1e-6 mm/rad`，绕 centroid / station origin 是
  `1860.15 mm/rad`，绕 Delta FASER 是 `1228.69 mm/rad`。
  `⟨∂x/∂ry⟩ = −1860.15 mm/rad`，等于 IFT 平均 `z`。
  GeoModelTest 读的是未对齐 manager collection，不能当 aligned
  sensor-center dump；本阶段用源码合同 + `T*c=g*T` 数值验证代替。
  Kabsch / `dz`-vs-`x` 绕 centroid 转，有限差分已排除，禁止映射成
  Stations `ry`。条目 17 物理 refit `Δx/ry ≈ −1859.81 mm/rad` 只作为
  物理链一致性，不是 survey prior。
- **Blocker 3 IOV：unresolved，仅 2022。** Talk date 2022-11-04。
  PDF 和 dump 都没有 opening / closing / thermal / 机械状态。没有
  2022→2024/2025 独立稳定性证据。逐年 Align POOL 是 reconstruction
  conditions，不是 survey。Station 刚性运动保持 IOV-specific；`C_dx`
  仍是 common-static 候选，不升级。
- **官方槽位：不填 `measured`。** 三条 ingest gate 没有同时满足。
  `ift_C_dx` / `ift_l0_minus_l2_dx` 保持 `feasibility_only`、
  `sigma=null`；`ift_station0_ry` 保持 `unavailable`。因此 **不会进入
  Fisher**，也不 Newton。条目 61 的阈值和 scan grid 原样冻结、未使用。

## 输入 / 命令

- 配置：`configs/nov22_metrology_provenance_station_ry_contract_v1.yaml`
- 源：`docs/cad_survey_nov22.txt`（immutable）
- 相邻：`docs/04Nov2022_Survey.pdf`、`docs/FAS_IFT_Dec10-2021.pdf`
- Calypso 证据：`TrackerAlignDBTool.cxx`、`SiDetectorManager.cxx`、
  `SCT_DetectorManager.cxx`、`SCT_DetectorFactory.cxx`、
  `FaserSCT_AlignCondAlg.cxx`、`FaserSCT_DetectorElementCondAlg.cxx`、
  `SiDetectorElement.cxx`、`GeoModelTestAlg.cxx`、`geomDB.sql`、
  `TopLevelPlacements.cxx`
- 环境：`source scripts/setup_environment.sh ml`（`LCG_110_cuda`）

```bash
source scripts/setup_environment.sh ml
python -c "import torch; print(torch.cuda.is_available())"
pytest -q tests/test_nov22_metrology_provenance_station_ry.py tests/test_cad_survey_nov22.py
python scripts/report_nov22_metrology_provenance_station_ry.py
```

CUDA `True`（Tesla T4）。9 passed。没有提交 Condor：本阶段是
provenance / geometry audit 和短数值 FD，交互节点足够。

## 数值结果（artifact）

| 量 | 值 | 用途 |
| --- | ---: | --- |
| Stations `ry` 支点 | FASER 原点 `(0,0,0)` | 软件合同，已验证 |
| `⟨∂x/∂ry⟩` analytic | `−1860.15 mm/rad` | 等于 IFT 平均 `z` |
| `⟨∂x/∂ry⟩` numeric (`ry=±0.1 mrad`) | `−1860.1499969 mm/rad` | 中心差分，sinc 修正后与解析一致 |
| RMS vs FASER origin | `3.1e-6 mm/rad` | 最近支点 |
| RMS vs centroid / station origin | `1860.15 mm/rad` | 排除 |
| RMS vs Delta FASER | `1228.69 mm/rad` | 排除 |
| 条目 17 `Δx/ry`（track 状态） | `−1859.81 mm/rad` | 物理链一致性，不是 prior |
| quoted `C_dx` | `+0.2541169 mm` | 条目 66 冻结中心值，`feasibility_only` |
| measurement covariance | `null` | 不填 `measured` |
| 2024/2025 IOV | `false` | 仅 2022 measurement |

源 SHA256 `1ea2e62d5…1c8f6`；config SHA256 `446d87eb…3c3008`；
HEAD `a1fd01bb5c2c5daf432c13db3b021feb42bdf6d1`。

报告：`outputs/nov22_metrology_provenance_station_ry_contract_v1/`。

## 未解除的假设

- 2021 native CAD → Calypso 轴符号仍未唯一确定，本阶段不用。
- Cadoux/Casper EOS 目录存在但不可读，不能当成空。
- GeoModelTest 不能当 aligned sensor dump；以后必须读
  `SCT_DetectorElementCollection`。
- `cad_survey_nov22` implied nominal 只当 FD 点云，不当测量。

## 向硬件 / alignment / survey 团队索取的最小字段

1. Nov 2022 原始 wafer 表：sensor 标识、测量全局 `(x,y,z)` 或相对命名
   nominal 的 offset、该 frame 下 3×3（至少对角）测量协方差、仪器 /
   日期 / 操作者 / 重复次数 / SHA256。**不要**把 station/module
   population SD 当 sigma 交来。
2. 绕 FASER 原点的 survey 刚体转动，并明确它就是
   `/Tracker/Align/Stations` `ry`。**不要**交 Kabsch、`dz`-vs-`x`、
   LAYERPITCH、STEREOANGLE 或 `/Tracker/Align/Planes` `ry`。
3. IOV 声明：2022 survey 的机械 / opening / closing / thermal 状态；
   若用于 2024/2025，需要独立稳定性证据；`C_dx` 是否在
   opening/thermal 检查后被主张为 common-static。

## 冻结结论

- Frozen V2 / association policy 不变。
- 真实数据继续 `residual_dq_monitoring_only`。
- `geometry_write_allowed=false`；无 alignment payload；未进入 Fisher。
- Stations `ry` 的软件合同已经唯一：全局左乘，绕 FASER 原点，
  `∂x/∂ry=+z`。这还不足以把任何现有 survey 数字填进 `ift_station0_ry`。
- Nov-2022 仍是 2022 measurement。`C_dx` 仍是 common-static 候选。
- 下一步：保持 DQ monitoring；向 Casper/Cadoux 索取上述三组字段。
  拿到之后再复用现有 track+external-prior Fisher，阈值和 scan grid
  继承条目 61，不得看结果后调整。先做 dry-run/config audit，再运行
  真实 Fisher。若仍缺任一 gate，继续冻结否定结论。
