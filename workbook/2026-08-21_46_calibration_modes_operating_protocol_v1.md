# 2026-08-21 (46) 两个互斥 calibration mode、mode-validity contract 与 Operating Protocol V1

## 任务

条目 45 已关闭 hierarchical V1 rescue：station 5-DoF + survey `dz` 与 IFT
1-D `C_dx` 各自可测，但不是可同时求解的层级参数。本条目把该结论冻结成
**两个互斥 calibration mode**，把泄漏预算写成机器可读的 mode-validity
contract，并给出可执行的 **FASER alignment operating protocol V1**。

随后只做冻结推理的 transfer：不重新训练 V2，不重调 capture / `A` / route /
χ²，不打开 sealed test，不增加 layer/module DoF，不用 Schur 当生产估计器，
不在同一数据阶段把两种 mode 互迭代或把对方当 nuisance。

## 两个互斥 mode

**Station Mode**

- 只拟合 `dx/dy/rx/ry/rz`，`dz` 保留 5 mm survey prior，写入 `dz=0`。
- 前提：IFT internal `C_dx` 已由外部几何或 dedicated calibration 固定。
- Isolation MC（layer 恒 0）声明 `cdx_fixed_by=isolation_zero`。

**IFT-Internal Mode（dedicated `C_dx`）**

- 只拟合 1-D `C_dx=(dx_L0-dx_L2)/2`，payload `L0=+C_dx`，`L1=0`，`L2=-C_dx`。
- 前提：station 六矢已通过条目 36 冻结 capture，且不是同一数据阶段。
- 记录 station 不确定度对 `C_dx` 的传播系统项，不把 station 当 nuisance。

二者禁止写进同一 Newton，禁止同一阶段互为 leftover，禁止 mixed remaining
chart 把未浮动层清零。

## Mode-validity contract

产物：`outputs/mc24_ift_calibration_mode_validity_contract_v1/mode_validity_contract.json`

schema `faser-ift-calibration-mode-validity-v1`。validation 未参与注册，
test 未打开。`A` 取条目 45 的生产可比算子（train route-selected、
`iteration_00_start`、`A_6` + 5 mm `dz` prior）：

| 量 | 冻结值 |
| --- | ---: |
| `A_dx` | **−59.213**（µm station dx / µm `C_dx`） |
| `A_ry` | **−31.908** |
| 统计预算（绑定） | **1.462 µm** |
| 工程预算 | 1.689 µm |
| 对外 band | **1.5–1.7 µm** |
| 已注册 `σ(C_dx)` | 6.91 µm（**不满足** station 预算） |
| 反向 `B_dx` | −0.01650 mm `C_dx` / mm leftover station dx |
| station→`C_dx` 1σ RSS | **0.715 µm** |
| 同上 3σ RSS | 2.15 µm |
| survey `dz` 1σ 系统项 | 0.47 µm |
| `A` transfer 相对偏差门 | 10%（观测 8 角散布 ~7%，不重调） |

Station Mode：未建模 `|C_dx|` 超过 1.462 µm 必须标
`cross_level_contaminated`，`geometry_write_allowed=false`。当前重建上的
dedicated `C_dx` 估计器达不到该预算，因此不能用「先在同一样本上估
`C_dx` 再跑 station」当作前提。

`C_dx` Mode：必须先有条目 36 framework capture；合同写入上述 RSS 作为
`C_dx` 系统项。

拒绝指标还包括：未声明 `C_dx` 已固定、同阶段互迭代、nuisance / 联合
Newton / Schur 生产、`A` 不稳定、把 residual 下降当成功（条目 44）、
由 station `dx` 反推的 implied `|C_dx|` 超预算。

## 当前语料上的冻结推理 baseline

没有新 Condor。只评分已经闭合的 isolation banks：

| mode | 样本 | unique edges | 独立 closure |
| --- | --- | ---: | --- |
| Station train | 5-DoF iteration-00 route-selected | 574 | 通过 |
| Station validation | 同上，source-disjoint | 460 | 通过 |
| `C_dx` train | 1-D relative-dx route-selected | 193 | 通过 |

`outputs/mc24_ift_calibration_mode_transfer_current_corpus_v1/`。

泄漏算子 `A` 在 iteration-00 的 route-selected 与 truth-selected 各 4 个
角上全部落在 10% 稳定性门内（
`outputs/mc24_ift_calibration_mode_validity_contract_v1/A_stability_iteration00_*.json`）。

这是**当前 10+8 source 语料**的冻结推理确认，不是 large-statistics 完成。
更大、仍 source-disjoint、非 sealed 的第一波额外文件已列入
`configs/calibration_modes_large_stats_transfer_v1.yaml`（+10 train、
+4 validation），须先走完物理链再做 per-mode isolation transfer。不提交
联合 hierarchical 注入。

## 真实数据阶段（尚未开始）

MC transfer 稳定之后才进入。不用 truth closure。每个 mode 的 DQ：unbiased
residual 宽/均值、source/run 一致性、拟合前后 residual 下降（只记录）、
参数稳定性、route multiplicity、重拟合后 geometry consistency。条目 03
的 data0 provenance 门仍然有效。**禁止把 residual 下降解释为正确
alignment。**

## Operating Protocol V1

可执行文本：

- `docs/faser_alignment_operating_protocol_v1_cn.md`
- `docs/faser_alignment_operating_protocol_v1.md`

它回答：何时允许 Station Mode、何时允许 dedicated `C_dx` Mode、哪些
cross-level 指标拒绝写入、两种结果如何分别写入 geometry/conditions。
若 large-statistics MC 与真实数据 transfer 都稳定，结束方法开发，转向
physics-production validation，不再增加自由度。

## 代码

- `alignment/calibration_modes.py`：互斥 mode、合同校验、validity 评估、
  `A` 稳定性门。
- `alignment/hierarchical_v1_leakage.py`：反向算子 `B`、station→`C_dx`
  系统项。
- `scripts/register_calibration_mode_validity_contract.py`
- `scripts/evaluate_calibration_mode_validity.py`
- `scripts/audit_leakage_operator_stability.py`
- `scripts/score_calibration_mode_transfer.py`
- `scripts/run_route_selected_multidof_update.py` 与
  `scripts/run_hierarchical_v1_truth_closure.py`：`--calibration-mode`、
  `--mode-validity-contract`、`--require-mode-valid`。
- `tests/test_calibration_modes.py`（与泄漏 / V1 remaining 测试一并
  通过）。

冻结不变：mode-0、V2、`physical_edge_deduplicated`、全部 capture、sealed
test。无新物理 refit。test 未打开。
