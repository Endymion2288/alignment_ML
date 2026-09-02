# 条目 68 — 四站 RelativeRoute 首轮训练合同预注册

日期：2026-09-01  
分支：`4station`  
状态：**首轮训练合同与三臂归因预注册完成（通过全部预训练 Dry-Run 与测试审计）**  
物理闭环合同：`continue_to_15d_relative_wls = false` 保持冻结  
密封 Test：永久封存，`sealed_test_accessed = false`  
新 Final Blind 内容访问：`new_final_blind_content_accessed = false`（Pristine，完全未打开）  
训练授权状态：`training_authorized = true_for_workbook69_head_only_primary_and_control_training`（仅授权下一条目 Workbook 69 执行 Primary 与 Control 的 Head-Only 训练；本任务未启动训练，未产生任何 checkpoint）

---

## 1. 科学假设与首轮最小实验合同

### 1.1 核心科学问题
> 在**完全冻结 Workbook-64 edge/backbone representation 和 edge scores**的条件下，Workbook-65 确立的 **C. absolute score-scale / route-utility threshold** 和 **D. route aggregation / set-packing competition** 失败机制是否能够仅通过一个 explicit complete-route correction head 得到恢复？

### 1.2 为什么坚持 Head-Only 训练？
1. **严格隔离变量**：若在引入新架构时同时进行 end-to-end backbone 微调或修改 candidate graph，将无法归因改善究竟来自整轨上下文建模还是基础边打分的偏移；
2. **保护高精度边证据**：Workbook-64 训练的 V2 backbone 已在 15D 课程下学习了高质量的几何表征与两两匹配特征，仅需在其之上学习跨四站的上下文打分修正项 $\Delta L_{\text{route}}$；
3. **零初始化平滑过渡**：Head-Only 架构支持严格数学上的零初始化，使得初始模型与生产 edge-only 解算器 100% 恒等。

---

## 2. 几何表征语义纠偏 (Representation Semantics)

根据 Phase 1 的理论对账，Workbook 67 中关于 $[s_0, s_1 - s_0, s_2 - s_0, s_3 - s_0]$ 表征的描述已在代码和文档中正式纠偏：

- **禁止声称**：`strict translation invariant`, `strict gauge invariant`, `SE(3)-equivariant`；
- **准确定义**：**Station-0-anchored latent route representation（参考条件相对路线表征）**；
- **科学内涵**：由于 $s_i$ 是经过含绝对节点坐标的非线性 backbone 提取后的潜在状态，$f(x_i + c) - f(x_0 + c) \neq f(x_i) - f(x_0)$，且表征中显式包含了参考节点 $s_0$。显式提供跨站潜在差分旨在降低整轨打分头对孤立绝对节点状态的依赖，并引入多站全局上下文；其能否提升 gauge/common-mode 稳定性属于待实验检验的假设，而非数学架构保证。

---

## 3. 残差整轨打分与零初始化数学合同

### 3.1 累加修正公式 (Additive Route Logit Correction)
RelativeRoute V4 的整轨头输出标量修正项 $\Delta L_{\text{route}}$（`delta_route_logit`），整轨联合 Log-odds 定义为：

$$L_{\text{edge}} = \text{logit}(p_{01}) + \text{logit}(p_{12}) + \text{logit}(p_{23})$$
$$L_{\text{corrected}} = L_{\text{edge}} + \Delta L_{\text{route}}$$
$$p_{\text{complete}} = \sigma(L_{\text{corrected}})$$

生产解算器直接消费：
$$\text{complete\_route\_score\_composition} = \text{"replace"}$$
$$\text{complete\_route\_scores} = p_{\text{complete}}$$

在生产解算器内部计算得到的完整 4 站物理整轨 Utility 为：
$$U_{\text{complete}} = L_{\text{edge}} + \Delta L_{\text{route}} + 4 \times \text{unmatched\_penalty} + 3 \times 10^{-6}$$

局部 2 站与 3 站 fragment 严格保持纯边 Log-odds：
$$U_{\text{frag2}} = L_{01} - 2.0 + 10^{-6}$$
$$U_{\text{frag3}} = L_{01} + L_{12} - 3.0 + 2 \times 10^{-6}$$

### 3.2 零初始化合同 (Zero-Initialization Guarantee)
整轨打分输出层 `route_score = nn.Linear(route_hidden_dim, 1)` 在初始化时显式置零：
```python
nn.init.zeros_(self.route_score.weight)
nn.init.zeros_(self.route_score.bias)
```
这保证在训练开始前 $\Delta L_{\text{route}} \equiv 0$。因此：
$$\text{RelativeRouteV4@init} \equiv \text{Workbook64 Edge-Only Production Solver}$$
测试验证证明两者在整轨 utility、选中路线集合、未匹配端点上 100% 逐 bit 恒等。

---

## 4. 三臂归因实验设计 (Three-Arm Study)

为了避免 Workbook 66 早期草案中多因素同时变更导致的混淆，首轮实验确立严格的单因素三臂设计：

```text
Arm 0: Frozen Workbook-64 Baseline (Edge-only Production Assignment)
Arm 1: Absolute-Route Control (Frozen Backbone + Trainable Absolute Route Head)
Arm 2: RelativeRoute V4 Primary (Frozen Backbone + Trainable Station-0 Anchored Route Head)
```

### 4.1 各臂配置与参数对比

| 实验臂 | 基础 Backbone | 整轨节点表征 | $\Delta L_{\text{route}}$ 语义 | Trainable 参数量 | Frozen 参数量 |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Arm 0 (Baseline)** | Workbook-64 Checkpoint | 无整轨打分（Edge-only） | 0 | 0 | 787,460 |
| **Arm 1 (Control)** | Workbook-64 Checkpoint | 绝对节点 $[s_0, s_1, s_2, s_3]$ | Additive (Zero-Init) | **172,513** | **614,947** |
| **Arm 2 (Primary)** | Workbook-64 Checkpoint | Station-0 锚定 $[s_0, s_1-s_0, s_2-s_0, s_3-s_0]$ | Additive (Zero-Init) | **172,513** | **614,947** |

> **关键控制**：Arm 1 与 Arm 2 的可训练参数量严格一致（均为 172,513 参数），输入维度严格一致（563 维），优化器、学习率、种子、batch、损失函数与训练数据 100% 相同。唯一的物理与架构差别是**是否采用 Station-0 锚定差分表征**。

### 4.2 两阶段科学归因逻辑 (Two-Stage Attribution)
1. **Question A（显式整轨联合打分是否有效？）**：
   - 比较 **Arm 1 vs Arm 0**；
   - 若 Arm 1 显著改善 C/D 失败机制，证明收益来自求解器对显式整轨联合打分的消费。
2. **Question B（Station-0 锚定相对表征是否带来额外增益？）**：
   - 比较 **Arm 2 vs Arm 1**；
   - 仅当 Arm 2 的 C/D 减少量显著优于 Arm 1 且纯度/fake/twin 保持在容差内时，方可支持“相对表征带来独立物理增益”的科学结论。
   - **禁止**通过 Arm 2 vs Arm 0 将所有改善笼统归因于相对表征。

---

## 5. 训练数据、损失函数与超参数继承

### 5.1 冻结六源训练集
严格继承 `configs/physical_curriculum_four_station_diversity_train_sources.yaml` 中经过多样性审计的 6 个 source-disjoint xAOD：
1. `mc24_100043_00200_00299` ($\mu^-$)
2. `mc24_100044_00300_00399` ($\mu^+$)
3. `mc24_100043_00300_00399` ($\mu^-$)
4. `mc24_100044_00200_00299` ($\mu^+$)
5. `mc24_100047_00100_00149` ($\mu^-$)
6. `mc24_100048_00100_00149` ($\mu^+$)

### 5.2 物理课程与超参数（全部来自 Workbook 64 冻结配置）
- **物理课程**：`15d_gauge_then_left_se3`，`translation_max_mm = 0.50`，`rotation_max_mrad = 5.0`，`q_over_p_mode = 0`；
- **优化器**：`AdamW(lr=2e-4, weight_decay=1e-4)`；
- **训练周期**：固定 30 epochs（Nominal 8 epochs + Relative/Gauge 22 epochs），无 Early Stopping，采用 `last_completed_epoch` 作为唯一 checkpoint；
- **随机种子**：`20260822`；
- **Batch Size**：`32`。

### 5.3 损失函数梯度流动审计 (Loss Gradient Contract)

| 损失组件 | 作用对象 | 权重 | 权重来源 | 是否向 Trainable Route Head 提供梯度 |
| :--- | :--- | :---: | :--- | :---: |
| `edge_loss` | `output.edge_logits` | 1.0 | Workbook 64 | **否 (0 梯度，Backbone 冻结)** |
| `route_consistency` | `output.route_logits` | 1.0 | Workbook 64 | **是 (BCE 正样本监督)** |
| `one_to_one_competition` | `output.route_logits` | 0.25 | Workbook 64 | **是 (端点竞争 LogSumExp)** |
| `fake_route_penalty` | `output.route_logits` | 0.25 | Workbook 64 | **是 (Fake 端点 Softplus 惩罚)** |

---

## 6. Development 判定门槛与机制指标

### 6.1 生产级开发集门槛 (Workbook-64 Production Gate)
在开发集（`100047_00350` 与 `100048_00350`）上评估时，模型必须首先满足历史生产质量底线：
- Nominal Purity $\ge 0.95$，Nominal Fake $\le 0.05$；
- 非 Nominal 扰动下：$\Delta \text{efficiency} \le 0.10$, $\Delta \text{purity} \le 0.05$, $\Delta \text{fake} \le 0.05$；
- Gauge-Twin 对称性：$|\Delta \text{efficiency}| \le 0.05$, $|\Delta \text{purity}| \le 0.05$, $|\Delta 2\to 3 \text{ eff}| \le 0.08$。

### 6.2 C+D 机制诊断指标 (Workbook-65 Diagnostics)
针对 `reference`、`draw_00`、`draw_00_plus_common` 重点报告：
1. $U_{\text{truth}} \le 0$ 事件数与比例（Problem C）；
2. 真实整轨被局部 fragment 挤占胜出的事件数与比例（Problem D）；
3. $U_{\text{truth}}$ 中位数与最佳竞争者 Utility 中位数；
4. 2→3 边 Log-odds 稳定性（验证冻结 Backbone 的不变性）。

---

## 7. 预训练验证执行记录 (Pre-training Verification)

### 7.1 GPU 前向与反向传播 Dry-Run
在 `lxplus-gpu`（LCG 110 CUDA, Tesla T4, PyTorch 2.11.0）上执行 `scripts/dry_run_relative_route_v4.py`：

```bash
source scripts/setup_environment.sh ml && python scripts/dry_run_relative_route_v4.py
```

**执行结果**：
- Target Device: `cuda` (`Tesla T4`, CUDA available: True)
- Arm 1 (Control) 参数：Trainable = 172,513, Frozen = 614,947
- Arm 2 (Primary) 参数：Trainable = 172,513, Frozen = 614,947（严格匹配）
- Arm 2 Loss: `total = 1.0861`, `route_consistency = 0.2835`, `competition = 2.1731`
- 反向传播后梯度审计：
  - 冻结参数中具有梯度的数量：**0**
  - 可训练参数中缺失梯度的数量：**0**
- **状态**：`PASS`（未保存任何 checkpoint，无训练副产物）。

### 7.2 完整单元测试套件矩阵 (37/37 全部通过)

```bash
source scripts/setup_environment.sh ml && pytest tests/test_blind_failure_localization.py tests/test_relative_route_v4.py tests/test_route_aware_transformer.py tests/test_route_assignment.py
```

**测试结果统计**：`37 passed in 12.30s`。

| 测试项编号 | 验证内容 | 结果 |
| :---: | :--- | :---: |
| **Test 1** | $\Delta L = 0$ 零初始化严格复现 Workbook-64 edge-only 解算器 | **PASS** |
| **Test 2** | 4 站整轨索引元组对齐与缺失 key 严格异常抛出 | **PASS** |
| **Test 3** | 累加修正 $\text{logit}(p_{\text{complete}}) = L_{\text{edge}} + \Delta L$ 数学合同验证 | **PASS** |
| **Test 4** | 局部 2/3 站 fragment 拓扑与 utility 保持 100% 冻结 | **PASS** |
| **Test 5** | 物理候选图（Candidate Graph）完全保持不变 | **PASS** |
| **Test 6** | 参数冻结与反向传播梯度隔离审计（0 冻结参数泄露梯度） | **PASS** |
| **Test 7** | Arm 1 (Control) 与 Arm 2 (Primary) 可训练参数量精确一致 (172,513) | **PASS** |
| **Test 8** | Arm 0/1/2 边打分一致性验证（Backbone 恒等） | **PASS** |
| **Test 9** | CPU 与 Tesla T4 GPU 前向有限值与无 NaN/Inf 验证 | **PASS** |
| **Test 10** | 固定随机种子确定性推理验证 | **PASS** |
| **Test 11** | Final Blind 与 Sealed Test 访问保护审计 | **PASS** |
| **Test 12** | 六源训练集 Manifest 完整性审计（精确包含授权 6 源） | **PASS** |
| **Test 13** | 开发集源（`00350_00399`）绝对排除在训练集之外 | **PASS** |
| **Test 14** | Final Blind 源（`00800_00849`）绝对排除在所有训练配置之外 | **PASS** |
| **Test 15** | Primary 与 Control 预注册 YAML 配置严格单因素对齐 | **PASS** |
| **16–37** | 历史定位审计、V2 backbone 与 assignment 专项测试（22 项） | **PASS** |

---

## 8. 预注册配置文件

1. **Primary (Arm 2)**：[`configs/relative_route_v4_head_only_train.yaml`](file:///eos/user/x/xcheng/FASER/alignment_ML_4station_branch/configs/relative_route_v4_head_only_train.yaml)
2. **Control (Arm 1)**：[`configs/absolute_route_control_head_only_train.yaml`](file:///eos/user/x/xcheng/FASER/alignment_ML_4station_branch/configs/absolute_route_control_head_only_train.yaml)
3. **Dry-run 审计脚本**：[`scripts/dry_run_relative_route_v4.py`](file:///eos/user/x/xcheng/FASER/alignment_ML_4station_branch/scripts/dry_run_relative_route_v4.py)

---

## 9. 授权决议与科学纪律声明

鉴于以下全部预训练前置条件已 100% 满足：
- [x] 所有 37 项针对性单元测试全部通过；
- [x] GPU 反向传播与梯度隔离 Dry-Run 全部通过；
- [x] $\Delta L = 0$ 精确复现基线解算器数学证明与数值测试通过；
- [x] 冻结参数与可训练参数审计通过（无未声明参数）；
- [x] 损失函数梯度流动审计通过；
- [x] 训练与推理整轨 Utility 严格一致；
- [x] 六源训练集清单通过；
- [x] 验证开发数据严格隔离于训练之外；
- [x] Final Blind（`00800_00849`）保持未访问（Pristine）；
- [x] Sealed Test 保持封存。

正式更新授权状态：

```text
training_authorized = true_for_workbook69_head_only_primary_and_control_training
continue_to_15d_relative_wls = false
sealed_test_accessed = false
new_final_blind_content_accessed = false
```

> **科学纪律**：本任务已完成所有训练前合同冻结与代码审计，**未启动任何实际训练循环，未生成任何 checkpoint**。训练将严格在下一条目 Workbook 69 中依据本合同正式执行。

---

> **Erratum: see Workbook 68A** (`workbook/2026-09-01_68A_四站RelativeRoute训练合同勘误与Objective梯度审计.md`)
> 1. Continuation tie-break production contract is strictly `1e-9` (not `1e-6`).
> 2. Training hyperparameters are explicitly categorized into historical V2 route-head convention vs Workbook-64 solver-aware objective (including dustbin-aware route margin, packing route competition, and gauge consistency).
>
> **Erratum: see Workbook 68B** (`workbook/2026-09-02_68B_四站RelativeRoute冻结边接线与Head-Only保存合同勘误.md`)
> Head-only additive 必须使用冻结 Workbook-64 生产边；Condor `9254670`/`9254671` 未保存 checkpoint，不计入 Workbook 69。
