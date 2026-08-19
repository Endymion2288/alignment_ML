# 2026-08-18 (25) mode-0 协方差校准 + candidate coverage 修复阶段：完整诊断与冻结判定

## 任务边界（用户指令）

冻结 V2 association backbone、route policy、multi-DoF update 定义与 historical test；不训练新模型、不扩 Rx/Rz/dz、不提交新 alignment iteration；校准因子只允许从 train physical source 导出，禁止用 validation 调 scale；candidate endpoint 定义不变。本阶段全部操作为既有物理 bank 的只读再分析，无任何新物理生产。

## 一、train-only 分量级 pull 校准统计

新工具 `scripts/audit_propagation_pull_calibration.py`：对 iteration-1 bank 的 `iteration_01_reference`（nominal）与 `iteration_01_anchor` 两点、10 个 train source，逐 truth-matched mode-0 传播记录计算残差与 combined（propagated+tracklet）协方差的 pull，按 station pair × 分量汇总。

**Pooled train 结果**（每 pair 约 1630–1800 条 truth 边；`outputs/mc24_multidof_ift_pull_calibration_train_v1/`）：

| pair | 分量 | robust σ(pull) | q95 | 物理解读 |
|------|------|------|------|------|
| 0->1 | x | 0.427 | 1.74 | 核心高估 ~5×（方差），重尾 |
| 0->1 | y | 0.003 | 0.03 | 核心高估 ~10⁵（方差）|
| 0->1 | tx | 0.765 | 1.91 | 轻度高估，重尾 |
| 0->1 | ty | 0.003 | 0.03 | 核心高估 ~10⁵ |
| 1->2 | x | 0.383 | 1.88 | 同上 |
| 2->3 | x | 0.304 | 1.81 | 同上 |

物理残差尺度：x 残差 robust σ ≈ 10–16 mm（q95 41–73 mm），y 残差 robust σ ≈ 0.73–1.5 mm；而 exporter 的 propagated σ_y ≈ 225–490 mm。x/tx 的大残差来自 mode-0 默认 q/p 种子的动量模糊（tx 残差 robust σ ≈ 24 mrad）。Mahalanobis χ² 中位数 5.6–12（χ²₄ 中位数应为 3.36），q99 高达 2×10⁴ —— 极重尾。

**结论修正**：此前"x/tx 欠估 6–7×"是 RMS/尾部口径；核心（robust）口径下 x/tx 也为高估。真实图像是**核心高估 + 重尾欠覆盖**的非高斯失校，y/ty 核心高估约 5 个数量级。

**Validation 只读确认**（`outputs/mc24_multidof_ift_pull_calibration_validation_readonly_v1/`）：8 个 validation source 的 pull 宽度与 train 几乎逐点一致（如 0->1 x：train 0.427 / val 0.650；tx：0.765/0.777）——失校是 source-independent 的，train 导出的因子在分布层面可移植。

## 二、两个 train-冻结 control 的实现

新模块 `alignment/covariance_calibration.py`（6 个单元测试）：

1. **Diagonal covariance rescaling**：因子 = pooled train robust pull 宽度²，按 (station pair, 分量) 冻结于 `mode0_covariance_calibration_train_frozen.json`：x ×0.09–0.18，y ×~1×10⁻⁵，tx ×0.59–0.77，ty ×~6–9×10⁻⁶。应用为 C' = D½·C·D½（保持相关结构），只作用于 propagated 协方差。
2. **Robust χ² / Huber 加权**：在 route-selected update 的 WLS 前对 anchor 边做一步 M-估计，w = min(1, k/√χ²)，k=2.5 先验固定（不对任何 split 调参），通过 C→C/w 实现，不改 solver。

`run_route_selected_multidof_update.py` 新增 `--covariance-calibration`、`--huber-k`、`--anchor-payload-sample` 三个开关；`read_anchor_selected_field_edge_observations` 新增可选校准入口。

**实现中发现并修复的关键管线问题**：WLS 权重只取 anchor bank 的协方差，而 anchor bank 此前从冻结 backbone 的 CSV 读入（未校准），导致第一轮 calibrated 运行结果与原始逐位相同（校准只作用于 target/probe bank，未进 solver）。新增 `--anchor-payload-sample` 使 anchor bank 也从候选图重测（route 集合仍严格为冻结 backbone 的 anchor 选择），校准才真正进入 WLS。该路径有专门单元测试（`test_anchor_selected_reader_applies_covariance_calibration`）。

## 三、candidate gate / coverage 重扫

新工具 `scripts/scan_candidate_gate_coverage.py`：在 nominal / iteration-0 anchor / iteration-1 anchor 三个几何点，对 10 个 train source + 离群 validation source `mc24_100047_00150_00199`（只读跟踪），扫描 χ² gate ∈ {5,10,25,50,100,200,500} × {原始, 校准} 两种协方差模型。

**Pooled train complete truth-chain recall**（`outputs/mc24_multidof_ift_gate_coverage_scan_v1/`）：

| 几何点 | 模型 | g=10 | g=25 | g=50 | g=100 | g=200 | g=500 |
|------|------|------|------|------|------|------|------|
| nominal | 原始 | 0.346 | 0.503 | 0.609 | 0.702 | 0.775 | 0.859 |
| nominal | 校准 | 0.111 | 0.255 | 0.421 | 0.576 | 0.668 | 0.753 |
| iter0 anchor | 原始 | 0.244 | 0.443 | 0.528 | 0.612 | 0.683 | 0.772 |
| iter0 anchor | 校准 | 0.032 | 0.161 | 0.348 | 0.512 | 0.629 | 0.716 |
| iter1 anchor | 原始 | 0.239 | 0.396 | 0.519 | 0.627 | 0.719 | 0.829 |
| iter1 anchor | 校准 | 0.104 | 0.252 | 0.424 | 0.573 | 0.668 | 0.751 |

0->1 truth-edge recall 同趋势（nominal 原始 g25=0.663 → g500=0.917）。

**关键发现**：

1. **fake-candidate growth 在所有 gate 下恒为零**。直接验证：物理 bank 每个事件每 station 至多 1 条 tracklet（单径迹 MC），同一事件内根本不存在错误 target 候选。raw 物理层的 χ² gate 因此**只拒 truth、不拒 fake**——coverage 的唯一有效杠杆是 gate 本身，放宽 gate 无 fake 代价。
2. 校准在固定 gate 下**降低** recall（核心变紧 → 重尾被拒更多）；g500 时两模型趋于接近（0.72–0.75 vs 0.77–0.86），剩余损失由 propagation failure（~2–3%）与超重尾构成。
3. 离群源 `mc24_100047_00150_00199` 的 coverage 曲线与 train 池无系统差异——它的 closure 离群**不是** candidate coverage 效应。

## 四、校准后 route-selected closure 对比（冻结判定）

6 个变体（train/validation × 原始 / 校准 / 校准+Huber / 仅Huber），其余全部冻结：

| split | 变体 | dx err (mm) | dy err (mm) | Ry err (mrad) | 通过 | cond |
|------|------|------|------|------|------|------|
| train | v1 冻结 | −0.0008 | −0.0005 | −0.0003 | ✓ | 8.0e1 |
| train | 校准 | +0.0668 | +0.0007 | +0.0653 | ✓ | 3.4e3 |
| train | 校准+Huber | +0.0563 | +0.0011 | +0.0471 | ✓ | 2.7e2 |
| train | 仅 Huber | −0.0001 | −0.0007 | +0.0039 | ✓ | 1.1e1 |
| validation | v1 冻结 | −0.0945 | +0.0575 | −0.0280 | ✓ | 8.6e1 |
| validation | 校准 | **−1.1944** | +0.0059 | **−1.1733** | ✗ | 2.5e2 |
| validation | 校准+Huber | −1.1405 | +0.0027 | −1.1303 | ✗ | 8.0e1 |
| validation | 仅 Huber | −0.1764 | +0.0758 | −0.0236 | ✗ | 3.5e1 |

**逐 source 分解**（`outputs/mc24_multidof_ift_iteration01_route_selected_calibrated_comparison_v1/`）：校准后 dy 误差在全部 18 个 source 上 |err|<0.01 mm（y/ty 权重修正的直接收益）；但 dx/Ry 逐源散布爆炸——validation 校准后 dx 误差从 −1.23 mm（离群源）到 +0.32 mm，pooled −1.19 mm 由离群源主导。

**机制诊断（candidate selection 与 WLS weighting 已分离）**：

- **candidate selection 层**：coverage 是 gate-受限而非协方差-受限；校准只移动曲线位置，且单径迹事件使 fake 恒零。
- **WLS weighting 层**：对角核心宽度校准把 y/ty 块权重抬高 ~10⁵，而 y/ty 只约束 dy；dx/Ry 只剩 x/tx 块且相对权重被压垮，dx–Ry 近简并方向失去约束（条件数 80→3350），逐源系统学经简并方向放大。原始（失校的）协方差反而意外地抑制了无信息分量、保护了 dx/Ry 解。

## 五、冻结判定（按用户决策树严格执行）

- 校准**未**显著提高 raw complete-chain coverage（固定 gate 下反而降低；coverage 提升来自 gate 放宽，与协方差模型无关）；
- 校准在 validation closure 中把 dx 误差从 0.0945 mm 推向 −1.19 mm（远离容差边界的错误方向），Huber 控制同样使 validation 变差（−0.176 mm）。

**判定：两个 control 均不采纳。** canonical physical candidate/WLS covariance model 保持为 exporter 原始协方差；校准因子与全部变体结果作为文档化负结果保留。association 架构、route policy、update 定义、test 封存状态不变。

**对下一阶段（Rx/Rz/dz sensitivity 与 station-level 6-DoF）的前置约束**：基础层尚未"稳定"——dx–Ry 简并方向的逐源系统学（离群源）仍是 closure 精度的主导项；任何未来的协方差修正必须保持分量间信息权重平衡（不能只做核心宽度归一），且需要针对重尾的逐边稳健化同时不破坏 dx 约束方向。

## 产物清单

- `scripts/audit_propagation_pull_calibration.py`、`scripts/scan_candidate_gate_coverage.py`、`alignment/covariance_calibration.py`（新）
- `scripts/run_route_selected_multidof_update.py`、`alignment/route_selected_update.py`（校准/Huber/anchor 重测开关）
- `scripts/run_calibrated_closure_variants.sh`（6 变体驱动）
- `tests/test_covariance_calibration.py`（6 测试）、`tests/test_anchor_selected_update.py`（+1 测试）；全套 187 测试通过
- `outputs/mc24_multidof_ift_pull_calibration_train_v1/`（含冻结因子 JSON）
- `outputs/mc24_multidof_ift_pull_calibration_validation_readonly_v1/`
- `outputs/mc24_multidof_ift_gate_coverage_scan_v1/`
- `outputs/mc24_multidof_ift_iteration01_route_selected_update_{train,validation}_{calibrated,calibrated_huber}_v2/`、`_huber_only_v1/`
- `outputs/mc24_multidof_ift_iteration01_route_selected_calibrated_comparison_v1/`
