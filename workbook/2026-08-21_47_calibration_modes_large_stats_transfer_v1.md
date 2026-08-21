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

## 物理链完成后的冻结分析（尚未跑）

不重训 V2。关联锚：Station 用 `iteration_00_anchor`；C_dx 与 A 用
`iteration_00_reference`。观测语义保持 `anchor_selected_field_edge` +
`physical_edge_deduplicated`。

- Station：truth-selected 与 frozen-V2 route-selected 五参数 bias/σ/pull、
  source spread、法矩阵 rank/condition、association ε/purity/fake、
  unique-edge、post/pre residual；`evaluate_calibration_mode_validity.py
  --require-mode-valid`
- 在 A Jacobian bank 上**重新测量** `A_dx`/`A_ry`，与冻结合同比 10% band；
  超门记 **contract instability**，不更新 A
- C_dx：独立 1-D closure；最终不确定度显式区分 statistical 与冻结
  station-geometry systematic（`B` RSS），后者不得吸入 fit σ
- 产出 `current-corpus vs large-statistics transfer matrix`

通过标准：新增 train **与** 独立 validation 上两种 mode 都保持冻结判据，
且无新的 source-dependent failure → **MC transfer gate 通过**，下一阶段才是
真实 FASER data 的 Operating Protocol V1 dry-run。任一 mode 在新增
validation 上失败：只定位该 mode 的 transfer limitation，回头调 V2、
capture、A 或加 DoF 都被禁止。

完成审计：

```
python scripts/audit_calibration_mode_transfer_completion.py
python scripts/run_calibration_modes_large_stats_postrefit.py --stage complete
```

物理链全部 `accepted` 后：

```
python scripts/run_calibration_modes_large_stats_postrefit.py --stage corpus
python scripts/run_calibration_modes_large_stats_postrefit.py --stage commands
```

## 本条目到此为止

物理 transfer bank 已制备并提交。闭合、A 稳定性、transfer matrix 与
MC transfer gate 判定等待 Condor 完成；在此之前**不**进入真实数据 dry-run。
