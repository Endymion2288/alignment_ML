# FASER Alignment Operating Protocol V1

条目 46 / 2026-08-21。Hierarchical alignment V1 作为联合层级已关闭。生产路径
只有两个**互斥** calibration mode。

## 冻结不变量

本协议不重调：

- 传播：**mode-0**
- association：冻结 V2 checkpoint / calibration / route policy
  （`outputs/mc24_v3_expanded_trainval_v2_bce_control_v1`，ungated）
- 观测：`anchor_selected_field_edge` + `physical_edge_deduplicated`
- station capture：条目 36
- `C_dx` capture：条目 40 的 `C_dx` 行
- 泄漏算子 `A` 与本 mode-validity contract：条目 45/46
- sealed test：**永不打开**
- 真实链：`/Tracker/Align → SCT_ClusterContainer → SegmentFitRefit → SegmentsRefit → NtupleDumper → Acts(mode 0)`

禁止：Schur 作为生产估计器；更多层级迭代；新的 layer/module DoF；联合
station+`C_dx` Newton；把另一层当 nuisance；把 residual 下降当成正确
alignment（条目 44）。

机器合同：

`outputs/mc24_ift_calibration_mode_validity_contract_v1/mode_validity_contract.json`

## 何时允许跑 Station Mode

只拟合 `ift_dx_mm ift_dy_mm ift_dz_mm ift_rx_mrad ift_ry_mrad ift_rz_mrad`，
`--prior-sigma ift_dz_mm:5.0`。写入的 `dz` 保持 **0**。不写 layer。

前提：IFT internal `C_dx` 已由**外部几何**或 **dedicated calibration** 固定，
不是在本数据阶段估出来的。Isolation MC（layer 恒为 0）使用
`cdx_fixed_by=isolation_zero`。

用 `|A_dx|≈59.213` 的门：

| 门 | 允许的未建模 `|C_dx|` |
| --- | ---: |
| 统计（绑定，3σ_dx） | **1.462 µm** |
| 工程（0.1 mm dx） | 1.689 µm |
| 对外 operating band | **1.5–1.7 µm** |

超过统计门的结果必须标 **`cross_level_contaminated`**，不得作为 station
几何写入。当前重建上已注册 `σ(C_dx)=6.91 µm` **不满足**该预算：在同一样本
上跑完 dedicated `C_dx` 并不能给 Station Mode 开绿灯。

## 何时允许跑 dedicated `C_dx` Mode

只拟合 `C_dx`。Payload `L0=+C_dx`，`L1=0`，`L2=-C_dx`。不写 station 六矢。

前提：station 六矢已经通过条目 **36** 的冻结 framework capture，而且不是
同一数据阶段的 station 步 leftover。Isolation MC（station 恒为 nominal）
可以引用条目 36 的已闭合结果。必须记录 station 不确定度对 `C_dx` 的传播
系统项（自由 station 1σ RSS **0.715 µm**；3σ RSS 2.15 µm；survey `dz` 1σ
**0.47 µm**）。不把 station 当 nuisance，不重调 `σ(C_dx)`。

## 哪些指标触发拒绝

出现任一条即不得写入 geometry/conditions：

- Station Mode 未建模 `|C_dx|` 超过统计预算
- Station Mode 未声明 `C_dx` 已固定
- `C_dx` Mode 前 station framework capture 未通过
- 同一数据阶段两种 mode 互迭代
- 把另一层当 nuisance、联合 Newton、或 Schur 生产
- 新样本上泄漏算子 `A` 相对冻结点偏差 `>10%`（**不重调 `A`**，也不把本
  合同的 `C_dx` 预算套到该样本）
- 用 residual 下降当作成功（条目 44）

由 station `dx` 反推的 `|C_dx|=|dx|/|A_dx|` 超过同一预算，也是拒绝指标。

## 两种结果如何分别写入

两条互斥写路径。禁止把未浮动的那一层用 remaining chart 清零。

| mode | 写入 | 不写 |
| --- | --- | --- |
| Station | IFT station `dx/dy/rx/ry/rz`；`dz=0` | layer / `C_dx` |
| IFT-Internal | `L0=+C_dx`，`L1=0`，`L2=-C_dx` | station 六矢 |

`--require-mode-valid` 强制 `geometry_write_allowed`。写入需要 capture
成功 **且** mode 有效；residual RMS 下降单独不够。

## Large-statistics MC transfer（冻结推理）

不重新训练。仍然 source-disjoint、非 sealed。只做 isolation 注入。

当前 10+8 V3 语料上的独立 isolation closure 已在本合同下记为独立闭合
（`outputs/mc24_ift_calibration_mode_transfer_current_corpus_v1/`）。
`A` 在 8 个 iteration-00 train/val × truth/route 角上稳定（相对散布 ~7%，
低于 10% 门）。

第一波额外文件（须先走完物理链）见
`configs/calibration_modes_large_stats_transfer_v1.yaml`：10 个额外 train
source、4 个额外 validation source。两种 mode **分别**验证 bias、pull、
source/run 稳定性、association efficiency/purity/fake、正规矩阵条件数、
独立 closure，以及 `A` 相对本冻结点的稳定性。不再重开 hierarchical V1。

## 真实 FASER data（MC transfer 稳定之后）

不再使用 truth closure。按 mode 建立 data-quality 观测量：

- unbiased residual 宽度与均值
- source/run 一致性
- 拟合前后 residual 下降（**只记录，不等于正确 alignment**）
- 参数稳定性
- route multiplicity
- 重拟合后 geometry consistency

条目 03 对 2022 data0 IFT 重导出 provenance 仍未放行。真实数据准入仍取决于
该物理链。

## 结束方法开发

若 large-statistics MC transfer **与** 真实数据 DQ transfer 都在本协议下
稳定，就停止增加自由度。下一阶段是两种互斥几何的 physics-production
validation，而不是新的 alignment 估计器。
