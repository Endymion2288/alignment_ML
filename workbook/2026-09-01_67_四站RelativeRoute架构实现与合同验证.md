# 条目 67 — 四站 RelativeRoute 架构实现与合同验证

日期：2026-09-01  
分支：`4station`  
状态：**架构实现与确定性验证完成（禁止训练）**  
物理闭环合同：`continue_to_15d_relative_wls = false` 保持冻结  
密封 Test：永久封存，`sealed_test_accessed = false`  
新 Final Blind 内容访问：`new_final_blind_content_accessed = false`  
训练授权状态：`training_authorized = false`（本条目不启动任何模型训练）

---

## 1. Workbook 66 合同硬错误修正记录

在实施代码前，对 Workbook 66 进行了全面的证据对账与合同纠偏，修正了以下硬错误：

1. **六源训练集修正**：修正了误写的 source 列表，严格恢复为 `configs/physical_curriculum_four_station_diversity_train_sources.yaml` 中声明的 6 个 source-disjoint xAOD：
   - `mc24_100043_00200_00299` ($\mu^-$)
   - `mc24_100044_00300_00399` ($\mu^+$)
   - `mc24_100043_00300_00399` ($\mu^-$)
   - `mc24_100044_00200_00299` ($\mu^+$)
   - `mc24_100047_00100_00149` ($\mu^-$)
   - `mc24_100048_00100_00149` ($\mu^+$)
2. **物理课程恢复**：删除了非法的 0.0/0.1/0.5/1.0 mm 描述，严格恢复 Workbook 64 冻结的四站 15D 相对课程合同：
   - `relative_curriculum = 15d_gauge_then_left_se3`
   - `translation_max_mm = 0.50`
   - `rotation_max_mrad = 5.0`
   - `q_over_p_mode = 0`
   - `physical geometry repropagation = true`
3. **已有损失历史澄清**：删除了“V2 未包含 dustbin-aware margin”的错误表述。确认 Workbook 59/62/64 已包含 `packing route competition`、`dustbin-aware route margin`、`gauge consistency` 与 `hard-aware max reduction`。新架构的科学贡献是**将显式整轨联合表示与求解器直接消费打通，形成端到端闭环**，而非首次发明 margin 损失。
4. **求解器 Route Utility 数学定义纠偏**：修正了以概率求和表达 utility 的笔误，恢复代码级真实定义：
   $$L_{\text{edge}} = \sum_{i=0}^2 \text{logit}(p_{i \to i+1}), \quad U_{\text{route}} = L_{\text{edge}} + N_{\text{station}} \times \text{unmatched\_penalty}$$
   在 $\text{unmatched\_penalty} = -1.0$ 下：
   $$U_{\text{frag2}} = L_{01} - 2.0, \quad U_{\text{frag3}} = L_{01} + L_{12} - 3.0, \quad U_{\text{complete}} = L_{01} + L_{12} + L_{23} - 4.0$$
   $$U_{\text{complete}} - U_{\text{frag3}} = L_{23} - 1.0$$
5. **Route Composition 语义澄清**：明确 `residual` 组合公式为 $L_{\text{final}} = (1 - \alpha) L_{\text{edge}} + \alpha L_{\text{route}}$。$\alpha = 1.0$ 数学上严格等价于 `replace`；$\alpha = 0.0$ 严格等价于 `edge-only`。未在开发集上擅自选择 $\alpha$，标记为 `route_composition_training_value_not_yet_authorized`。
6. **架构命名防冲突**：为避免与历史负结果（Geometry-Aware Transformer V3 结构化指派）混淆，新模型正式命名为 **Relative Route Transformer V4 (RelativeRoute V4)**。
7. **几何等变声称纠偏**：澄清 Station-0 差分坐标为 Station-0-anchored latent route representation（参考条件相对表示），不具备数学严格平移不变或 SE(3) 等变性；其能否提升 gauge/common-mode 稳定性属于待测科学假设。

---

## 2. 架构设计与实现修改

### 核心修改文件
1. **`models/route_transformer.py`**：
   - 增加 `RelativeRouteTransformerConfig`（继承 `RouteAwareTransformerConfig`，显式设置 `use_relative_route_representation = True`）；
   - 增加 `RelativeRouteSparseTransformer`（RelativeRoute V4 主类）；
   - 在 `forward` 中，当 `use_relative_route_representation = True` 时，将 4 站节点特征在 latent 空间以 Station 0 为参考基准进行差分投影：
     $$\mathbf{s}_{0}, \quad \mathbf{s}_{1} - \mathbf{s}_{0}, \quad \mathbf{s}_{2} - \mathbf{s}_{0}, \quad \mathbf{s}_{3} - \mathbf{s}_{0}$$
     与 3 条相邻物理边特征（来自 `residual_v1` 边投影）、3 个 station-pair embeddings 以及 3 个相邻边 base logits 拼接后输入 `route_encoder`，输出标量 `route_logits`；
   - 保持向后兼容：当配置为 `use_relative_route_representation = False` 时，完全复现历史 V2 行为。
2. **`training/route_aware_transformer.py`**：
   - 暴露 `RelativeRouteTransformerConfig` 与 `RelativeRouteSparseTransformer` 导出接口；
   - 确保 `predict_route_aware_scores` 与 `route_query_score_maps_by_event` 准确提取 `(idx_0, idx_1, idx_2, idx_3) -> float(p_route)` 映射。
3. **`tests/test_relative_route_v4.py`**：
   - 新增针对 RelativeRoute V4 的 9 项专项单元测试套件。

### 数据流与求解器管道对接（Exact Route-Score Plumbing）

```text
Event Graph Bundle
  ↓
RelativeRouteSparseTransformer.forward()
  ↓
route_logits [routes]  ──(sigmoid)──> route_scores [routes]
  ↓
route_query_score_maps_by_event()
  ↓
complete_route_scores_by_event[event_key][(idx0, idx1, idx2, idx3)] = p_route
  ↓
assign_adjacent_route_sets(..., complete_route_scores_by_event)
  ↓
adjacent_route_assignment(..., complete_route_scores)
  ↓
_route_hypotheses(..., complete_route_scores, composition, context_weight)
  ↓
4 站整轨直接获得联合 Log-odds: L_final = (1 - α) * L_edge + α * logit(p_route)
2 站 / 3 站局部 fragment 保持独立单边 Log-odds: L_frag = Σ logit(p_edge)
  ↓
solve_unit_capacity_route_packing() -> 最终物理整轨与 Dustbin 决策
```

---

## 3. Feature Contract 与几何变换属性

```text
A. 现有 Pair-Relative 物理量 (严格相对量):
   - residual_x_mm, residual_y_mm, residual_tx, residual_ty
   - pull_x, pull_y, pull_tx, pull_ty
   - log1p_chi2, combined_covariance_logdet, delta_z_mm
   - source_chi2_per_ndof, target_chi2_per_ndof, source_n_hit, target_n_hit
   - hit_layer_l_side_s (6-bit hit pattern per station)
   - log_sigma_x, log_sigma_y, log_sigma_tx, log_sigma_ty

B. 现有绝对节点量 (用于 Backbone 兼容输入):
   - x_mm, y_mm, tx, ty, z_mm

C. 专用于新 Route Scorer 的 Station-0 相对锚定表示 (Station-0-Anchored Latent Representation):
   - Δs_1 = s_1 - s_0, Δs_2 = s_2 - s_0, Δs_3 = s_3 - s_0 （及参考节点 s_0）
   - 语义：Reference-conditioned relative route representation。通过显式提供跨站 latent differences 降低 route scorer 对独立 absolute node states 的依赖，并提供相对多站上下文；是否提高 gauge/common-mode stability 是待实验验证的 hypothesis，而不是架构保证。不宣称数学严格平移不变或 SE(3) 等变。

D. 严格禁止引入的量:
   - 禁止输入未经检验的全局坐标非线性变换
   - 禁止输入任何含有 MC truth 或 synthetic provenance 的标签
```

---

## 4. 确定性单元测试验证结果

执行针对 RelativeRoute V4 与相关模块的完整 pytest 套件：

```bash
source scripts/setup_environment.sh ml && pytest tests/test_blind_failure_localization.py tests/test_relative_route_v4.py tests/test_route_aware_transformer.py tests/test_route_assignment.py
```

### 验证矩阵（31/31 全部通过）

| 测试项 | 测试文件 | 验证内容 | 结果 |
| :--- | :--- | :--- | :---: |
| **Test 1: Legacy Preservation** | `test_relative_route_v4.py` | `complete_route_scores = None` 时与现有生产求解器数值 100% 一致 | **PASS** |
| **Test 2: Key Alignment & Exception** | `test_relative_route_v4.py` | 验证 4 站索引元组对齐；缺失 route 评分时严格抛出 `RuntimeError` | **PASS** |
| **Test 3: Numerical Composition** | `test_relative_route_v4.py` | 手算验证 `replace`、`residual(α=1.0)` 与 `residual(α=0.5)` 数值精度 | **PASS** |
| **Test 4: Fragment Preservation** | `test_relative_route_v4.py` | 注入整轨评分后，2/3 站 fragment 枚举、边分和 utility 保持 100% 不变 | **PASS** |
| **Test 5: Candidate Graph Invariance** | `test_relative_route_v4.py` | 物理候选集生成与 baseline 逐 candidate 严格恒等 | **PASS** |
| **Test 6: CPU & GPU Finite Forward** | `test_relative_route_v4.py` | CPU 及 Tesla T4 GPU 前向传播无 NaN/Inf，概率严格在 $[0, 1]$ 之间 | **PASS** |
| **Test 7: Relative Semantics** | `test_relative_route_v4.py` | `use_relative_route_representation=True` 前向及梯度通路正常 | **PASS** |
| **Test 8: Final-Blind Guard** | `test_relative_route_v4.py` | 验证测试过程未加载 `00800_00849` 与 sealed test | **PASS** |
| **Test 9: Deterministic Inference** | `test_relative_route_v4.py` | 固定参数下两次推理输出逐 bit 恒等 | **PASS** |
| **Historical & Module Suites** | `test_blind_failure_localization.py` 等 | 包含 localization audit、V2 backbone 与 assignment 在内的 22 项历史测试 | **PASS** |

**最终测试统计**：`31 passed in 11.07s`。

---

## 5. Final Blind 与 Sealed Test 封存状态审计

- **新 Final Blind 源**（`mc24_100047_00800_00849` 与 `mc24_100048_00800_00849`）：
  - 仅作为字符串配置存在；
  - 本任务期间未执行任何 ROOT open、事件读取、特征提取或推理；
  - `new_final_blind_content_accessed = false`（Pristine）。
- **Sealed Test**：
  - 永久封存，`sealed_test_accessed = false`。

---

## 6. 科学与实现未决事项检查

- **是否存在阻止进入下一阶段训练预注册的物理或实现问题？**
  - **不存在**。RelativeRoute V4 的前向图构建、Station-0 相对表示、直接求解器接入管道与单元测试已全部闭环。
- **关于训练授权**：
  - 本条目只完成架构实现与代码级确定性验证；
  - **严禁在当前条目下启动训练**；
  - 训练参数、epoch budget、loss 权重与 gate 必须在后续独立条目中正式获得授权后方可执行。

---

## 7. 冻结声明与授权状态

```text
training_authorized = false
continue_to_15d_relative_wls = false
sealed_test_accessed = false
new_final_blind_content_accessed = false
route_composition_training_value_not_yet_authorized
```
