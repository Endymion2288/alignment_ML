# 条目 68A — 四站 RelativeRoute 训练合同勘误与 Objective 梯度审计

日期：2026-09-01  
分支：`4station`  
状态：**训练合同勘误完成，C/D 求解器感知损失梯度审计通过，首轮 Head-Only 训练正式授权**  
物理闭环合同：`continue_to_15d_relative_wls = false` 保持冻结  
密封 Test：永久封存，`sealed_test_accessed = false`  
新 Final Blind 内容访问：`new_final_blind_content_accessed = false`（Pristine，完全未打开）  
训练授权状态：`training_authorized = true_for_workbook69_head_only_primary_and_control_training`

---

## 1. 求解器 Tie-break 与 Utility 勘误

### 1.1 代码真实值 (`_CONTINUATION_TIE_BREAK = 1e-9`)
在 `baselines/route_assignment.py` 中，确定性连续性打破平局常数（tie-break）的真实代码定义为：
```python
_CONTINUATION_TIE_BREAK = 1.0e-9
```
因此各站长路线的确定性存储 Utility 为：
- 2 站路线：$U_2 = (L_{01} - 2.0) + 1 	imes 10^{-9}$
- 3 站路线：$U_3 = (L_{01} + L_{12} - 3.0) + 2 	imes 10^{-9}$
- 4 站路线：$U_4 = (L_{01} + L_{12} + L_{23} + \Delta L_{\text{route}} - 4.0) + 3 	imes 10^{-9}$

### 1.2 物理主 Utility 与 Tie-break 的语义分离
- **物理主项 (Physical Utility)**：
  - $U_2^{\text{phys}} = L_{01} - 2.0$
  - $U_3^{\text{phys}} = L_{01} + L_{12} - 3.0$
  - $U_4^{\text{phys}} = L_{01} + L_{12} + L_{23} + \Delta L_{\text{route}} - 4.0$
- **Tie-break 项**：仅用于生产求解器在浮点数完全相等时确定性偏好更长路线（Deterministic exact-tie preference），**严禁**作为物理 Margin 进入损失函数。
- **勘误说明**：Workbook 68 早期文本中出现的 $1\times 10^{-6}, 2\times 10^{-6}, 3\times 10^{-6}$ 表述系笔误，现已全面修正并由 `Test 16` 验证与代码实现 100% 吻合。

---

## 2. 训练超参数 Provenance 完整对账表

明确区分 **Historical V2 Route-Head Convention** 与 **Workbook-64 Frozen Solver-Aware Objective**，杜绝将历史沿用参数伪称为新增项：

| 超参数名称 | 预注册数值 | 来源文件 / 历史阶段 | 历史作用 | 在 V4 Head-Only 中的复用原因 |
| :--- | :---: | :--- | :--- | :--- |
| `optimizer` | `AdamW` | `configs/geometry_aware_transformer_v2.yaml` | Historical V2 route-head 引擎 | 保持优化器基础行为稳定 |
| `learning_rate` | `2e-4` | `configs/geometry_aware_transformer_v2.yaml` | Historical V2 route-head 引擎 | 经 V2 验证的稳定收敛学习率 |
| `weight_decay` | `1e-4` | `configs/geometry_aware_transformer_v2.yaml` | Historical V2 正则化 | 防止打分头过拟合 |
| `batch_size` | `32` | `configs/geometry_aware_transformer_v2.yaml` | Historical V2 图 batch 大小 | 图级别采样与内存平衡 |
| `seed` | `20260822` | Workbook 64 多样性训练配置 | Workbook 64 冻结随机种子 | 继承已有可复现基准 |
| `epoch_budget` | `30` | `configs/physical_four_station_diversity_training.yaml` | Workbook 64 冻结周期预算 | 8 nominal + 22 15D relative |
| `early_stopping` | `False` | Workbook 64 冻结科学选择合同 | 严禁早停，消除选择偏差 | 确保两臂经历相同优化预算 |
| `checkpoint_rule`| `last_completed_epoch` | Workbook 64 冻结科学选择合同 | 固定取 epoch 30 | 杜绝开发集或测试集偷窥 |
| `focal_gamma` | `1.5` | `training/route_aware_transformer.py` | Historical V2 Focal BCE | 抑制简单易分样本 |
| `hard_negative_weight` | `2.0` | `training/route_aware_transformer.py` | Historical V2 难负例加权 | 强化混淆边辨别 |
| `edge_loss_weight` | `1.0` | `training/route_aware_transformer.py` | Historical V2 边损失 | Backbone 冻结，对 head 为 0 梯度 |
| `route_consistency_weight` | `1.0` | `configs/geometry_aware_transformer_v2.yaml` | Historical V2 整轨 BCE 监督 | 提供正样本 4 站联合打分牵引 |
| `one_to_one_competition_weight` | `0.25` | `configs/geometry_aware_transformer_v2.yaml` | Historical V2 端点 Softmax 竞争 | 惩罚共享端点的多余整轨候选 |
| `fake_route_penalty_weight` | `0.25` | `configs/geometry_aware_transformer_v2.yaml` | Historical V2 Fake 路线 Softplus | 压低含假边路线打分 |
| `packing_route_competition_weight` | `0.07061055340401011` | Workbook 59/62/64 冻结辅助损失 | **Workbook-64 Solver-Aware (Problem D)** | 显式拉大真整轨与局部片断的 Margin |
| `dustbin_aware_route_margin_weight` | `0.05` | Workbook 59/62/64 冻结辅助损失 | **Workbook-64 Solver-Aware (Problem C)** | 显式拉大真整轨与 Dustbin (0) 的 Margin |
| `gauge_twin_consistency_weight` | `1.0` | Workbook 59/62/64 冻结辅助损失 | **Workbook-64 Solver-Aware (Gauge)** | 约束 Chart 与 Twin 采样的打分一致性 |
| `route_competition_reduction` | `max` | `configs/physical_four_station_diversity_training.yaml` | Workbook 64 难例聚焦 | 对 Event 内最强竞争者做 Max Reduction |

---

## 3. Workbook 69 总损失函数与梯度流动审计

### 3.1 完整复合目标函数 (Total Composite Objective)

$$L_{\text{total}} = L_{\text{body}} + 0.07061055340401011 \times L_{\text{packing}} + 0.05 \times L_{\text{dustbin}} + 1.0 \times L_{\text{gauge}}$$

其中：
$$L_{\text{body}} = 1.0 \cdot L_{\text{edge}} + 1.0 \cdot L_{\text{route\_consistency}} + 0.25 \cdot L_{\text{competition}} + 0.25 \cdot L_{\text{fake\_penalty}}$$

### 3.2 损失梯度流动审计表 (Gradient Routing Matrix)

| 损失组件 | 历史来源 | 数学输入形式 | 对 `delta_route_head` 的梯度 | 梯度物理作用 |
| :--- | :--- | :--- | :---: | :--- |
| `edge_loss` | Historical V2 | `output.edge_logits` | **0.0 (Backbone 冻结)** | 保护基础边打分不偏移 |
| `route_consistency` | Historical V2 | `output.route_logits` | **非零 (BCE 梯度)** | 将真整轨 log-odds 向上拉升 |
| `one_to_one_competition` | Historical V2 | `output.route_logits` | **非零 (LogSumExp 梯度)** | 压制共享端点的伪整轨候选 |
| `fake_route_penalty` | Historical V2 | `output.route_logits` | **非零 (Softplus 梯度)** | 压制含假边的整轨打分 |
| `packing_route_competition` | **Workbook 64 (Problem D)** | $U_{\text{truth}}(L_{\text{corrected}}) \text{ vs } U_{\text{fragment}}$ | **非零 (ReLU 梯度)** | **直接解决 Problem D**：击败局部竞争片断 |
| `dustbin_aware_route_margin` | **Workbook 64 (Problem C)** | $U_{\text{truth}}(L_{\text{corrected}}) \text{ vs } \max(U_{\text{rival}}, 0)$ | **非零 (ReLU 梯度)** | **直接解决 Problem C**：使真整轨 Utility 跃过 0 门槛 |
| `gauge_twin_consistency` | **Workbook 64 (Gauge)** | $(U_{\text{chart}} - U_{\text{twin}})^2$ | **非零 (MSE 梯度)** | 抑制规范变换下的整轨打分扰动 |

---

## 4. 强制 C/D 梯度审计数值实测证据 (Mandatory C/D Gradient Evidence)

在 `lxplus-gpu`（Tesla T4 GPU, CUDA 12.1, PyTorch 2.11.0）上执行 `scripts/audit_route_head_solver_gradients.py`，分别对各损失分量进行独立反向传播并测量 `delta_route_head` 的梯度范数：

```text
================ Gradient Audit on CUDA ================
[route_consistency (V2 BCE)]
  loss value: 0.276283
  grad_norm(trainable_head): 0.088939
  frozen_param_leaks: 0
[one_to_one_competition (V2 LogSumExp)]
  loss value: 2.206181
  grad_norm(trainable_head): 0.176445
  frozen_param_leaks: 0
[fake_route_penalty (V2 Softplus)]
  loss value: 0.000000
  grad_norm(trainable_head): 0.000000
  frozen_param_leaks: 0
[packing_route_competition (WB64 / Problem D)]
  loss value: 3.353992
  grad_norm(trainable_head): 1.318633
  frozen_param_leaks: 0
[dustbin_aware_route_margin (WB64 / Problem C)]
  loss value: 5.218324
  grad_norm(trainable_head): 1.318633
  frozen_param_leaks: 0
[gauge_twin_consistency (WB64 Gauge)]
  loss value: 0.000000
  grad_norm(trainable_head): 0.000000
  frozen_param_leaks: 0

[Total Composite Objective (WB68A Contract)]
  total loss value: 1.587742
  grad_norm(trainable_head): 0.260271
  frozen_param_leaks: 0
```

### 审计结论
1. **Problem C 对应项**（`dustbin_aware_route_margin`）对可训练打分头产生 **1.318633** 的显著非零梯度，梯度直接作用于 $\Delta L_{\text{route}}$；
2. **Problem D 对应项**（`packing_route_competition`）对可训练打分头产生 **1.318633** 的显著非零梯度；
3. **复合总目标函数**对可训练打分头产生 **0.260271** 的非零梯度；
4. **冻结参数隔离审计**：在所有单独及复合反向传播中，614,947 个冻结参数的梯度泄漏数均为 **0**。

---

## 5. 单元测试矩阵验证 (39/39 全部通过)

```bash
source scripts/setup_environment.sh ml && pytest tests/test_blind_failure_localization.py tests/test_relative_route_v4.py tests/test_route_aware_transformer.py tests/test_route_assignment.py
```

**测试结果统计**：`39 passed in 12.74s`。
- `Test 16` 验证：生产 Utility 主项与训练 Packing Utility 100% 恒等，且严格满足 $+ (N-1) \times 10^{-9}$ tie-break；
- `Test 17` 验证：Problem C 与 Problem D 求解器感知损失对可训练路线头具有非零梯度。

---

## 6. 最终授权决议与科学纪律

所有 Phase A 训练前置审计与勘误条件已 100% 完成：
- [x] Tie-break 勘误为 $10^{-9}$，并通过训练/推理 Utility 恒等性测试；
- [x] 超参数 Provenance 明确分类，区别历史 V2 body 与 Workbook-64 solver-aware 项；
- [x] 完整复用 Workbook-64 冻结的 `dustbin_aware_route_margin` 与 `packing_route_competition`；
- [x] GPU 梯度审计证实 Problem C 与 Problem D 损失对 `delta_route_head` 具有强非零梯度；
- [x] 冻结参数梯度隔离 100% 通过（0 泄漏）；
- [x] 39 项全套单元测试全部通过；
- [x] Development 数据未打开；
- [x] Final Blind（`00800_00849`）保持封存（Pristine）；
- [x] Sealed Test 永久封存。

正式批准进入 **Phase B (Workbook 69)** 执行预注册的 Head-Only 训练：

```text
training_authorized = true_for_workbook69_head_only_primary_and_control_training
continue_to_15d_relative_wls = false
sealed_test_accessed = false
new_final_blind_content_accessed = false
```

---

> **Erratum: see Workbook 68B** (`workbook/2026-09-02_68B_四站RelativeRoute冻结边接线与Head-Only保存合同勘误.md`)
>
> 1. 68A 的 C/D 梯度审计成立，但当时的单模块 additive 前向把可训练 `delta_route_logit` 送进了 `route_edge_correction`，因此生产边不是冻结的 Workbook-64 边。
> 2. Condor cluster `9254670` / `9254671` 两臂都跑完 30 epoch，但 `RouteAwareTransformerArtifact` 保存接口写错，checkpoint 未落盘。这两次作业的权重不存在，禁止当作 Workbook 69 结果，也禁止从它们 resume。
> 3. 68A 的 `training_authorized=true` 被 68B 冻结边接线合同取代；在 68B 通过前不得再训。
