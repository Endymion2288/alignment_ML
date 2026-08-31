# 条目 66 — 四站下一代 Relative Route Transformer V4 架构预注册（合同修正版）

日期：2026-09-01  
分支：`4station`  
状态：**预注册完成（经证据对账修正，只读设计，禁止训练）**  
物理闭环合同：`continue_to_15d_relative_wls = false` 保持冻结  
密封 Test：永久封存，`sealed_test_accessed = false`  
训练授权状态：`training_authorized = false`（本条目不启动任何模型训练）

---

## 1. 背景与前置冻结结论

在 Workbook 64（六源 source-disjoint 训练）关闭 reserved-blind gate 后，Workbook 65 完成了对冻结 V2 检查点（SHA256: `0c3a28704cc01151fab7ac943e41338e859b5dde1502c0b10e2f12ada04e6236`）在 Reserved-Blind 数据集上的逐 event / 逐 route / 逐 station-pair 失败机制定位审计。

### Workbook 65 冻结的定量事实
1. **Candidate Missing (A) = 0 条（0.0%）**：物理 Candidate Builder 保持 100% 真实边 recall，无候选遗漏。
2. **Edge Thresholding (B) = 0 条（0.0%）**：所有真实边预测概率均通过 0.001 站对阈值，无单边硬截断。
3. **未解释损失 (E) = 0 条（0.0%）**：无任何未在预注册损失链内的未知损失。
4. **$U_{\text{truth}} \le 0$ (C) = 49/88 条（净增 35 条，占 0.117 效率 drop 的 62.5%）**：主因。三边 logit 因错位下移导致 $U_{\text{truth}} = \sum_{e} \text{logit}(p_e) - 4.0 \le 0$，直接被 solver 因 dustbin 惩罚放弃。
5. **Packing Competition (D) = 39/88 条（净增 21 条，占 0.117 效率 drop 的 37.5%）**：次因。在 $U_{\text{truth}} \in (0, 1)$ 区间内，Production Margin 恶化，当 $\text{logit}(p_{2\to3}) < 1.0$ 时，$U_{\text{complete}} < U_{\text{frag3}}$，完整 4 站 route 被 3 站 fragment（0-1-2）或同端点 blocker 挤占。
6. **物理退化集中在 S3 相关站对**：0→1 与 1→2 站对平均 logit 仅变化 $\sim 0.03$，而 2→3 站对平均 logit 从 2.180 骤降至 1.828；未选中 route 中 51.1% 存在直接 S3 参与度。
7. **实现 Bug 已排除**：代码、求解器、阈值定义完全吻合合同。

本预注册条目的唯一任务是：**针对 C（62.5%）和 D（37.5%）这两个已被证据锁定的根本机制，设计最小但足够的下一代架构改进，并在训练前完全冻结全部实验合同。**

---

## 2. 生产代码级事实与当前路径审查

经对代码库（`training/curriculum_mlp.py`、`baselines/field_chi2_matching.py`、`models/transformer.py`、`models/route_transformer.py`、`baselines/route_assignment.py`、`training/route_reduction_audit.py`）的深入审计，确认当前 V2 路径存在以下代码级事实与结构脱节：

```text
[物理输入]
Event Tracklets (state, cov, chi2, hits) + Mode-0 Acts Field Propagation
  ↓
[Candidate Builder] (100% recall)
CandidateSet with residual_v1 features (15 dim physical + 6 dim pair one-hot)
  ↓
[Node & Edge Graph Construction] (training/geometry_aware_transformer.py)
Nodes: Absolute global coordinates (x_mm, y_mm, tx, ty, z_mm, log_sigma, ...)
Edges: Pair-relative observables (residual, pull, log1p_chi2, logdet, delta_z)
  ↓
[V2 Backbone Encoding & Route Query] (models/route_transformer.py)
4-layer Sparse Transformer -> node_states -> Route-Query Head -> route_logits
Route Edge Correction -> Additive edge correction: edge_logits = base_edge_logits + correction
  ↓
[生产推理与求解脱节] (scripts/run_frozen_association_backbone.py & baselines/route_assignment.py)
推理脚本仅提取 edge_scores = sigmoid(edge_logits)，传递给 assign_adjacent_route_sets
complete_route_scores_by_event 被置为 None！
  ↓
[生产求解器 Route Utility] (baselines/route_assignment.py: _route_hypotheses)
完整 4 站 route 的 utility 计算为独立边 log-odds 的无正规化求和：
L_edge = logit(p_01) + logit(p_12) + logit(p_23)
U_complete = L_edge + 4 * unmatched_penalty = L_01 + L_12 + L_23 - 4.0
2 站 / 3 站 fragment utility 为：
U_frag2 = L_01 - 2.0
U_frag3 = L_01 + L_12 - 3.0
  ↓
[Unit-Capacity Set Packing Solver]
当 logit(p_23) < 1.0 (即 p_23 < 0.731) 时，U_complete - U_frag3 = logit(p_23) - 1.0 < 0，求解器必然丢弃 4 站整轨而选择 3 站 fragment！
当 Σ logit(p_edge) <= 4.0 时，U_complete <= 0，求解器必然放弃该整轨！
```

### 关键代码级发现与历史澄清
1. **Route Scoring 接口早已存在但未接通生产**：`baselines/route_assignment.py` 的 `_route_hypotheses` 原生支持 `complete_route_scores` 字典与 `complete_route_score_composition = "replace" | "residual"`。在历史 Workbook 15 的早期探索中，直接替换因缺乏联合表示和合适校准而未胜出，随后冻结在 `complete_route_scores = None`（即边求和主线）。
2. **已有损失历史澄清**：Workbook 59 / 62 / 64 的 V2 训练目标已包含 `packing route competition`、`dustbin-aware route margin`、`gauge consistency` 与 `hard-aware max reduction`。但此前这些损失主要用于约束独立边 logits 及其中间聚合，未能提供一个被生产求解器直接消费的显式整轨评分（explicit route score）。
3. **Node 绝对坐标输入引入全局坐标依赖**：`NODE_FEATURE_NAMES` 直接输入了 global 坐标系下的 `x_mm, y_mm, tx, ty, z_mm`。在 Station 0 相对锚定或 pair-relative 表征下，可以大幅提升几何表征的局部稳定度。

---

## 3. 下一代唯一主假设：Relative Route Transformer V4

为避免与历史负结果（Geometry-Aware Transformer V3 结构化指派）命名混淆，下一代架构正式命名为：

**Relative Route Transformer V4 (RelativeRoute V4)**

### 唯一 primary architecture 假设
> **在严格保持现有物理 Candidate Graph、15D Relative Curriculum、Left-SE(3) Gauge Twin 与 Unit-Capacity Set-Packing Solver 完全不变的前提下，构建基于 3 条相邻物理边联合表征与 Station-0 相对几何状态的显式整轨评分头（Explicit Complete-Route Scorer），并将该整轨评分直接通过现有 `complete_route_scores` 接口接入求解器，与已有的 solver-aware margin 目标实现端到端闭环。**

### 结构修改与失败机制的因果对应

| 失败机制 (Workbook 65 证据) | 根本原因 | RelativeRoute V4 的对应解决方案 |
| :--- | :--- | :--- |
| **C. $U_{\text{truth}} \le 0$ (62.5% 损失)** | 3 边独立边预测中单边概率受错位扰动下移，导致 $\sum \text{logit}(p_e) \le 4.0$。 | **Explicit Complete-Route Scorer**：整轨 logit 由 4 站联合上下文一致性直接给出，避免单边乘积崩溃导致整体 utility 下凹。 |
| **D. Packing Competition (37.5% 损失)** | 单边 $\text{logit}(p_{2\to3}) < 1.0$ 导致 $U_{\text{complete}} < U_{\text{frag3}}$，被 3 站 fragment 击败。 | **端到端 Route Utility 注入 + Margin 耦合**：整轨评分直接决定 $U_{\text{complete}}$，在求解器中形成对局部 fragment 的明确裕度。 |

---

## 4. 冻结与可变范围矩阵（Frozen / Mutable Matrix）

| 模块 / 合同 | 状态 | 严格范围与约束 |
| :--- | :---: | :--- |
| **Physical Reconstruction Chain** | **FROZEN** | `/Tracker/Align -> SegmentFitRefit -> SegmentsRefit -> NtupleDumper -> FaserActsExtrapolationTool(mode 0)` |
| **Momentum Parameterization** | **FROZEN** | `q_over_p_mode = 0`（真实模式） |
| **Physical Candidate Builder** | **FROZEN** | `build_candidate_sets`，`residual_v1` 特征提取，`chi2_gate = None`（保持 100% recall） |
| **Geometry & Alignment DoF** | **FROZEN** | S0, S1, S2, S3 各自 5 个自由度 `dx, dy, rx, ry, rz`；`dz` 保持 5 mm survey prior |
| **Alignment Solution Space** | **FROZEN** | 15D Relative Subspace $\Delta T_{ij} = T_i^{-1} T_j$；`continue_to_15d_relative_wls = false` |
| **Physical Curriculum** | **FROZEN** | `relative_curriculum = 15d_gauge_then_left_se3`，`translation_max_mm = 0.50`，`rotation_max_mrad = 5.0` |
| **Left-SE(3) Gauge Twin Contract** | **FROZEN** | 包含 common-mode rigid transformation 的 twin 验证对（`common_bounds: trans 0.3mm, rot 3.0mrad`） |
| **Training Corpus Sources** | **FROZEN** | 严格继承 `configs/physical_curriculum_four_station_diversity_train_sources.yaml` 冻结的六个 source-disjoint training xAOD（不增不减） |
| **Route Solver Algorithm** | **FROZEN** | `adjacent_contiguous_unit_capacity_set_packing`（保留 unit-capacity 与 tie-break） |
| **Sealed Test Policy** | **FROZEN** | Sealed test 永久封存，`sealed_test_accessed = false` |
| **Route Representation** | **MUTABLE** | 引入以 3 条相邻物理边联合特征及 Station-0 相对几何状态构建的 route 表征 |
| **Route Scoring Head** | **MUTABLE** | 产生供生产求解器直接消费的 `complete_route_scores` |
| **Production Solver Utility Feed** | **MUTABLE** | 在 `assign_adjacent_route_sets` 中传递 `complete_route_scores` |

---

## 5. Feature Contract 与坐标变换规范

根据 FASER 与 Calypso 坐标语义，所有输入特征划分为四类：

```text
A. 现有 Pair-Relative 物理量 (已严格相对化):
   - residual_x_mm, residual_y_mm, residual_tx, residual_ty
   - pull_x, pull_y, pull_tx, pull_ty
   - log1p_chi2, combined_covariance_logdet, delta_z_mm
   - source_chi2_per_ndof, target_chi2_per_ndof, source_n_hit, target_n_hit
   - hit_layer_l_side_s (6-bit hit pattern per station)
   - log_sigma_x, log_sigma_y, log_sigma_tx, log_sigma_ty

B. 现有绝对节点量 (保留用于 Backbone 兼容):
   - x_mm, y_mm, tx, ty, z_mm

C. 专用于新 Route Scorer 的 Station-0 相对锚定量 (Gauge-Stabilized):
   - Δx_i = x_i - x_0, Δy_i = y_i - y_0, Δtx_i = tx_i - tx_0, Δty_i = ty_i - ty_0
   - Δz_i = z_i - z_0
   (注意：平移差分在纯平移下严格不变；但在空间旋转下仅作为局部对齐稳定表征，不宣称数学严格 SE(3) 旋转等变)

D. 严格禁止引入的量:
   - 禁止输入未经检验的全局坐标非线性变换
   - 禁止输入任何含有 MC truth 或 synthetic provenance 的标签
```

---

## 6. 数据角色划分与 Final Blind 未使用审计

### 数据角色划分
1. **Train Split (训练集)**：
   - 严格继承 `configs/physical_curriculum_four_station_diversity_train_sources.yaml` 的 6 个 Source-Disjoint xAOD 文件：
     - `mc24_100043_00200_00299`
     - `mc24_100044_00300_00399`
     - `mc24_100043_00300_00399`
     - `mc24_100044_00200_00299`
     - `mc24_100047_00100_00149`
     - `mc24_100048_00100_00149`
2. **Development Split (开发与机制验证集)**：
   - Workbook 64 已打开的 Reserved Blind（转为 Development 诊断用途）：
     - `mc24_100047_00350_00399`
     - `mc24_100048_00350_00399`
   - 历史 Transfer 集合（Workbook 53–56）：`00300_00349`
3. **New Final Blind (下一代最终封存盲集)**：
   - 选定此前从未进入任何训练、选择或特征检查的全新 Reserve Sources：
     - `mc24_100047_00800_00849`
     - `mc24_100048_00800_00849`

### Final Blind 未使用 Provenance 审计
经对全仓库历史 commit、config、manifest 与 outputs 扫描，确认：
- `mc24_100047_00800_00849` 与 `mc24_100048_00800_00849` 仅作为字符串声明于 `configs/physical_curriculum_four_station_diversity_sources.yaml` 与 `training/source_diversity_audit.py` 的 `UNUSED_RESERVE_SOURCES` 列表中；
- 没有任何 output、checkpoint、model selection 或 manifest 文件读取过这两个源文件；
- **本条目中未打开这两个 blind 的 ROOT 内容，未进行任何特征分布检查或推理打分。**
- **Provenance 审计结论：Pristine & Untouched，准入为下一代 Final Blind。**

---

## 7. 求解器 Route Composition 语义与数学定义

在 `baselines/route_assignment.py` 中，求解器支持的 `complete_route_score_composition` 数学定义为：

$$
L_{\text{final}} = (1 - \alpha) L_{\text{edge}} + \alpha L_{\text{route}}
$$

其中：
- $L_{\text{edge}} = \sum_{i=0}^2 \text{logit}(p_{i \to i+1})$
- $L_{\text{route}} = \text{logit}(p_{\text{route}})$
- $\alpha = 0$：完全退化为 edge-only 求解；
- $\alpha = 1$：完全等价于 direct replacement（$L_{\text{final}} = L_{\text{route}}$）；
- $0 < \alpha < 1$：残差融合。

**关于 $\alpha$ 参数的冻结合同**：
本架构实现必须完全支持并单元测试 `replace` 与 `residual` 两种模式。但在训练被授权之前，**严禁使用 development 数据扫参选择 $\alpha$**。当前状态标记为：
`route_composition_training_value_not_yet_authorized`。

---

## 8. 最小 Attribution Control 设计

为确保科学归因清晰，必须能够回答：
> **改善究竟来自 Relative Complete-Route Representation 与直接求解器注入，还是仅仅因为增加了参数容量？**

### 预注册 Control 模型：Ablated Control (Capacity-Matched V2 Control)
- **Control 结构**：
  - 具有与 Relative Route Transformer V4 完全相同大小的参数量（匹配 hidden dim 与 MLP 宽度）；
  - 使用相同的数据集、优化器、epoch 预算与学习率调度；
  - **唯一削除（Ablation）**：
    1. 移除 Station-0 相对节点锚定（回退至绝对坐标输入）；
    2. 求解器断开 `complete_route_scores` 注入（回退至单边独立求和）。
- **Attribution 判定线**：
  - 若新架构在 `draw_00` 上的 efficiency 显著高于 Control（$\Delta \text{eff} > 0.05$）且 C 类损失显著下降，则证明改善源于 Relative Route 机制；
  - 反之，若两者无差异，则证明纯粹增加容量无法解决 C/D 失败。

---

## 9. 评估体系与三层 Gate 冻结

### Layer 1: Representation & Candidate Preservation
- Physical Candidate Recall 保持 $1.000$（0 丢失）；
- Candidate Graph 与冻结 baseline 精确一致；
- `complete_route_scores = None` 时与现有生产求解器完全数值等价。

### Layer 2: Development Mechanism Targets (针对 C & D)
在 Development 数据集（含 `draw_00`、`draw_00_plus_common`、`reference` 等）上评估：
- **Target 1 (针对 C)**：`draw_00` 上的 $U_{\text{truth}} \le 0$ 比例从 10.2% 降至 $< 4.0\%$；
- **Target 2 (针对 D)**：`draw_00` 上的 Fragment Winner 比例从 8.1% 降至 $< 4.0\%$；
- **Target 3 (效率恢复)**：`draw_00` Complete-Track Efficiency 从 0.817 恢复至 $\ge 0.900$（$\Delta \text{eff} \le 0.035$ vs nominal）；
- **Target 4 (Nominal 保护)**：`reference` Efficiency $\ge 0.930$，Fake Rate $\le 0.030$，Purity $\ge 0.985$；
- **Target 5 (Twin 一致性)**：所有 3 组 Gauge Twins 的效率差 $|\Delta \text{eff}| \le 0.020$。

### Layer 3: Final Source-Disjoint Blind Gate (在全新盲集上执行)
在打开新 Final Blind（`00800_00849`）前，Gate 标准冻结如下：
1. `complete_track_efficiency >= 0.900` across all 7 physical payloads (including `draw_00` and `draw_01`);
2. `complete_track_purity >= 0.980` across all payloads;
3. `track_fake_rate <= 0.035` across all payloads;
4. `left_se3_twin_max_delta_efficiency <= 0.025`;
5. `u_truth_fraction_nonpositive <= 0.050` in worst-case payload.

---

## 10. 总结与授权状态

| 声明项 | 状态 / 取值 |
| :--- | :--- |
| **主架构选择** | **Relative Route Transformer V4 (Relative Route Head + Direct Production Route Scoring)** |
| **针对机制** | **C. Score Scale ($U_{\text{truth}} \le 0$) 占 62.5% + D. Packing Competition 占 37.5%** |
| **物理 Candidate Graph** | **严格冻结保持不变** |
| **15D WLS Alignment Gate** | **continue_to_15d_relative_wls = false (保持冻结)** |
| **Sealed Test 访问状态** | **sealed_test_accessed = false (严密封存)** |
| **New Final Blind 状态** | **mc24_100047_00800_00849 & mc24_100048_00800_00849 (未打开，已审计)** |
| **训练授权状态** | **training_authorized = false (本条目严禁训练，待实现与验证完成后另行授权)** |
