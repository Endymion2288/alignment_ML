# 2026-08-21 (47) Calibration-mode large-statistics MC transfer V1

## 任务

正式执行条目 46 已冻结的两个**互斥** calibration mode 在新增、source-disjoint、
非 sealed MC 上的无调参迁移。不进入真实 FASER data。

冻结：mode-0；V2 checkpoint/calibration/route policy；
`physical_edge_deduplicated`；Station Mode 5-DoF + `dz=5 mm` survey prior；
IFT-Internal Mode 1-D `C_dx`；全部 capture criteria；mode-validity contract；
A 的 10% stability gate。禁止重训、重注册阈值、联合 station+`C_dx` 注入、
互相迭代、nuisance/Schur production、新增 layer/module DoF。残差下降只是
DQ observable，不是正确 alignment 的证据。

## 语料

第一波 `configs/calibration_modes_large_stats_transfer_v1.yaml`：

- 新增 train 10：`mc24_100043_{00000,00100,00700,00800,00900}_*` 与
  `mc24_100044` 对应区间
- 新增 validation 4：`mc24_100047/100048_{00200,00250}_*`
- 源配置：`configs/physical_curriculum_calibration_modes_large_stats_sources.yaml`
- 与当前 10+8 语料 source-disjoint；密封 test `100116/100117` 未打开

新增 train 只做统计稳定性审计，不优化模型或判据。新增 validation 只做冻结
transfer。

## 三个物理 bank（真实链）

每个点：`/Tracker/Align → SCT_ClusterContainer → SegmentFitRefit →
SegmentsRefit → NtupleDumper → Acts(mode 0)`。

| bank | 根目录 | 点/源 | 注入 |
| --- | --- | ---: | --- |
| Station Mode | `outputs/mc24_ift_station_mode_large_stats_transfer_physical_v1/` | 17 | 随机 5-DoF（severity 0.12），`dz≡0`，layer/`C_dx≡0`，`cdx_fixed_by=isolation_zero` |
| IFT-Internal Mode | `outputs/mc24_ift_internal_cdx_large_stats_transfer_physical_v1/` | 5 | station 六矢名义几何；只注入 `C_dx=±0.12 mm`，FD `±0.10 mm` |
| A 分析（非 mode） | `outputs/mc24_ift_leakage_operator_large_stats_transfer_physical_v1/` | 15 | 全零线性化的轴向 FD（6 station + `C_dx`）；**无**联合 operating-point 注入，不写 geometry |

Station 随机起点（seed `20260822`，与当前语料 `20260819` 不同）：

`(dx, dy, rx, ry, rz) = (+0.204 mm, −0.088 mm, −0.222 mrad, +6.665 mrad, +0.490 mrad)`，`dz=0`。

Condor（eossubmit / `tomorrow` / 6000 MB，每源一作业）：

| bank | cluster | 作业 |
| --- | ---: | ---: |
| Station | **1000337** | 14 |
| C_dx | **1000338** | 14 |
| A Jacobian | **1000339** | 14 |

## Condor 完成与本地修复

三 cluster 均已离开队列。Station（17×14）与 C_dx（5×14）全部
`completion_status=accepted`。A Jacobian bank 有 2 个 truncated `TBasket`
（worker 浅层 `num_entries` 误报完成）：

- `mc24_100043_00000_00099 / iteration_00_fd_ift_ry_mrad_p/propagations.root`
- `mc24_100044_00900_00999 / iteration_00_fd_ift_dx_mm_p/propagations.root`

Athena `enhanced_tracklets.root` 完好（100 event）。本地
`convert_ntuple_tracklet_propagations.py` 重写（2100 / 2048 records），坏文件
保留为 `propagations.root.corrupt_20260821T113156Z`。**未**重提 cluster。
修复后三 bank 全部 complete。

之后每次提交 Condor：**每 15 分钟检查一次完成情况，完成后再继续分析**。

## 冻结分析（无重训）

关联锚：Station `iteration_00_anchor`；C_dx 与 A `iteration_00_reference`。
观测：`anchor_selected_field_edge` + `physical_edge_deduplicated`。V2
checkpoint SHA256 仍为
`0c85a001677678163512c59bd5ceb6d08bdefaab79c0f4628659a4a8ad766a27`。

1-D C_dx route-selected 在只有 `C_dx` 的 plan 上缺 `C_rx` 键；call site
把缺省对比度填 0（与 isolation `C_rx≡0` 一致），不改
`remaining_after_block_step` 的严格接口。

### Station Mode — 生产观测（route-selected）通过

| split | unique edges | rank | cond | framework | engineering | geometry write |
| --- | ---: | ---: | ---: | --- | --- | --- |
| train | 551 | 6 | 1.38e6 | **true** | false（dy 0.170 mm > 0.1） | **true** |
| validation | 211 | 6 | 5.94e6 | **true** | false（dy −0.194 mm） | **true** |

Train recovered Δ（期望 dx/dy/rx/ry/rz = −0.204 / +0.088 / +0.222 / −6.665 / −0.490；dz 期望 0）：

| 参数 | recΔ | err | σ | pull_fit | eng | stat |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| dx | −0.165 mm | +0.039 | 0.047 | +0.84 | true | true |
| dy | +0.258 mm | +0.170 | 0.096 | +1.77 | false | true |
| dz | +0.211 mm | +0.211 | 3.24 | +0.07 | n/a | true |
| rx | +0.053 mrad | −0.169 | 0.118 | −1.43 | true | true |
| ry | −6.644 mrad | +0.021 | 0.067 | +0.31 | true | true |
| rz | −0.495 mrad | −0.005 | 0.517 | −0.01 | true | true |

Validation 同样 framework+statistical 通过；dy 工程容差仍失败（与当前 10+8
语料同一模式，**未**重注册）。`--require-mode-valid` 允许 geometry write
（`cdx_fixed_by=isolation_zero`，unmodeled `|C_dx|=0`）。

Station train V2（anchor）：AP 0.965；pair ε/purity/fake
0.731/0.990/0.010，0.746/0.996/0.004，0.646/0.999/0.0005；complete-track
ε/purity 0.699/0.977；track fake 0.026。Validation AP 0.955；track purity
0.966。anchor-selected overlap train 0.866 / val 0.926。残差 χ² 下降只记
DQ（train `response_chi2=19.6`，ndof 2198），**不是** alignment 成功证据。

### Station Mode — truth-selected 诊断失败（不调参）

Train 五自由参数中 **rz 工程+统计都失败**（recΔ −2.45 mrad vs 期望 −0.49，
err −1.96 mrad，pull −15.2）。ry 工程过但 coverage/track 失败（pull 7.9，σ
过小）。dz 被数据拉到 +9.6 mm（survey prior 信息份额仅 0.6%）。
`capture_success=false`。独立 validation 工程过、framework 不过。

这是 **truth-selected 诊断的 transfer limitation**，不是生产观测失败。
禁止据此重训 V2、重注册 capture、改 A 或加 DoF。残差 χ² 从 1.74e6 降到
1021 **仍不算成功**。

### IFT-Internal Mode — truth 与 route 都通过

只评冻结 2-D capture JSON 的 **C_dx 行**。station 六矢保持 0。

Truth-selected（σ 与冻结 B RSS 0.715 µm 分开报，不吸入 fit σ）：

| split / target | err (µm) | σ_stat (µm) | pull | capture |
| --- | ---: | ---: | ---: | --- |
| train start | −0.140 | 0.371 | −0.38 | true |
| train heldout | +0.002 | 0.371 | +0.01 | true |
| val start | −0.409 | 0.989 | −0.41 | true |
| val heldout | +0.287 | 0.989 | +0.29 | true |

Route-selected start（生产观测）：

| split | unique edges | rec C_dx (mm) | remaining (µm) | σ_stat (µm) | B RSS (µm) | capture | geometry write |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| train | 642 | 0.11997 | −0.034 | 0.857 | 0.715 | **true** | **true** |
| validation | 247 | 0.12122 | +1.22 | 5.78 | 0.715 | **true** | **true** |

Held-out route 同样 capture true（train remaining +0.031 µm；val −1.60 µm）。
Validation 统计 σ 大于 B；最终不确定度 = 统计 ⊕ 冻结 station-geometry
systematic，**不**把 B 并进 fit σ，也**不**把 station 当 nuisance。

### A 重测（分析列，不写 geometry）

全零线性化 Jacobian bank。冻结
`A_dx=−59.213`、`A_ry=−31.908`（µm / µm C_dx）。10% gate **全部稳定，不更新 A**。

| 观测 | split | A_dx | rel | A_ry | rel |
| --- | --- | ---: | ---: | ---: | ---: |
| truth | train | −54.819 | 7.4% | −29.822 | 6.5% |
| truth | val | −55.350 | 6.5% | −29.726 | 6.8% |
| route | train | −56.269 | 5.0% | −31.868 | 0.12% |
| route | val | −57.822 | 2.3% | −31.148 | 2.4% |

## Transfer matrix 与 MC transfer gate

`outputs/mc24_ift_calibration_mode_transfer_large_stats_v1/`：

- `transfer_report.json`：两种 mode 的 train **与** 独立 validation 都是
  `independent_closure=true`
- `transfer_matrix.json`：相对当前 10+8 语料；`failed_cells=[]`
- `A_stability.json`：四角全部 `stable`
- `--require-mode-valid`：Station 与 IFT-Internal 均为 `geometry_write_allowed`

**MC transfer gate 通过。** 生产判据是冻结 route-selected capture +
mode-validity contract + A 10% gate。未发现新的 source-dependent
route-selected 失败。truth-selected Station rz 失败已定位为诊断限制，
禁止回头调参。

下一步才是真实 FASER data 的 Operating Protocol V1 dry-run。本条目**没有**
打开真实数据或密封 test。

## 本条目到此为止

物理链、冻结 V2 推断、两种互斥 mode 的独立闭合、A 稳定性与 transfer
matrix 已完成。gate 通过后的真实数据 dry-run 另开条目；在用户明确要求之前
不进入。
