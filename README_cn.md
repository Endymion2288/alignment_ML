# FASER Tracklet Alignment ML（四站分支 4station）

本仓库研发用于 FASER 径迹谱仪的机器学习辅助对准（alignment）与局部径迹元（local tracklet）关联算法。本分支（`4station`）专门面向四站系统配置（IFT/Station 0、Station 1、Station 2、Station 3），在完整无近似的物理链路上求解 15 维相对子空间对准与长径迹关联：

```text
/Tracker/Align → SCT_ClusterContainer → SegmentFitRefit → SegmentsRefit → NtupleDumper
  → FaserActsExtrapolationTool (mode 0)
```

规范输入为名为 `tracklets` 的 flat ROOT tree，每行对应一条 local tracklet。它要求显式的 station ID、全局 `(x, y, z, tx, ty)`、状态 `[x, y, tx, ty]` 的协方差、拟合质量、hit 摘要，以及监督阶段所需的 MC truth 标签。

## 快速开始

### 环境配置

```bash
cd /eos/home-x/xcheng/FASER
# 初始化 LCG Python user site 环境（每个 LCG 发行版只需执行一次）
alignment_ML_4station_branch/scripts/bootstrap_ml_environment.sh

# 在每个新 shell 中激活 ML 环境
source alignment_ML_4station_branch/scripts/setup_environment.sh ml
cd alignment_ML_4station_branch
```

*注：* Calypso 物理链操作需在独立 shell 中执行：在完成 Calypso 编译后运行 `source .../setup_environment.sh calypso`。

### 运行基线与测试

```bash
# 运行单元测试（覆盖 RelativeRoute V4、求解器感知梯度与零初始化验证）
pytest -q

# 生成合成 tracklet 并运行纯几何卡方匹配基线
python -m scripts.make_synthetic_tracklets --output data/synthetic_tracklets.root --seed 7
python -m scripts.run_chi2_baseline \
  --input data/synthetic_tracklets.root \
  --config configs/baseline_chi2.yaml \
  --output-dir outputs/synthetic_chi2
```

### RelativeRoute V4 与 Head-Only 配对训练指令

```bash
# 求解器感知损失梯度审计（机制 C & D、Left-SE(3) 规范一致性）
python scripts/audit_route_head_solver_gradients.py

# 模型 dry-run 与确定性零初始化验证
python scripts/dry_run_relative_route_v4.py

# 提交 Head-Only 配对训练作业到 HTCondor（Arm 1 Control 对照臂 vs Arm 2 Primary 主实验臂）
python scripts/submit_relative_route_v4_head_only_condor.py --submit

# 冻结 Arm 0/1/2 reserved-blind development 评估（条目 70；禁止见结果后改 OP）
python scripts/submit_relative_route_v4_development_eval_condor.py --submit
```

## 四站对准与研究演进

### 1. 四站参数化与 15 维相对子空间（阶段一，条目 48–50）
- **参数化**：探测四站各 5 个受径迹约束的自由度（每站 `dx, dy, rx, ry, rz`，共 20 维），测量测量值 `dz` 施加 5 mm 先验约束。不假设 Station 0 为绝对固定基准。
- **可辨识度与 SVD**：在物理链路上执行中心有限差分（FD）分析表明，未约束的 20 维空间条件数高达 ~8.5×10⁵，存在 5 个全局不可辨识规范度（3 个平移 + 2 个转动）。
- **准入 15 维相对子空间**：选定一站作为参考基准消除 5 个规范度后，得到满秩的 15 维相对子空间（条件数降至 ~2×10⁴）。通过规范不变量 $\Delta T_{ij} = T_i^{-1} T_j$ 严格评估，truth 选择的 15 维相对 WLS 闭环以数值极限精度（$4.9\times 10^{-13}\text{ mm}$）恢复注入错位。

### 2. 关联控制与六源训练多样性（条目 51–64）
- **相对课程匹配重训**：在 15 维相对错位课程下对 matched V2 进行重训，引入规范一致性目标、dustbin 感知 route margin 与 hard-aware max reduction。
- **六源多样性训练（条目 64）**：在 6 个源文件互斥的 xAOD 上训练冻结的条目 62 目标。
- **Reserved-Blind 闸门评估**：保留盲样对（`100047_00350` / `100048_00350`）出现整轨效率衰减（$\Delta\text{eff} = 0.117$，2→3 边 $\Delta\text{eff} = 0.101$），归类为 `source_diversity_blind_failed_other`。按预注册合同，15 维相对 WLS 求解器保持严格关闭（`continue_to_15d_relative_wls = false`）。

### 3. Reserved-Blind 失败机制逐 Event 定位审计（条目 65）
在保留盲样上开展逐 event、逐 route、逐 station-pair 的定位审计：
- **Candidate Builder Recall = 100%（A = 0）**：几何/Acts 传播无候选真实边遗漏。
- **边阈值截断完好（B = 0）**：所有真实边打分均通过 0.001 阈值，无硬截断。
- **代码与求解器实现 Bug 已排除**：解算器、阈值定义与损失逻辑确定性核验无误。
- **锁定根本机制**：
  1. **机制 C（错位下打分尺度漂移 / 跌落 Dustbin，占 62.5%，净增 35 条）**：错位导致边 logit 下移，使得 $U_{\text{truth}} = \sum_e \text{logit}(p_e) - 4.0 \le 0$，直接被求解器当做背景丢弃。
  2. **机制 D（集合打包竞争 / 碎片挤占，占 37.5%，净增 21 条）**：在 $U_{\text{truth}} \in (0, 1)$ 区间内 production margin 崩溃，当 $\text{logit}(p_{2\to3}) < 1.0$ 时出现 $U_{\text{complete}} < U_{\text{frag3}}$，整轨被 3 站碎片（0-1-2）或同端点 blocker 挤占。
  3. **空间高度集中**：退化显著集中在 Station 3 相关的 2→3 边（平均 logit 从 2.180 骤降至 1.828）。

### 4. 下一代 RelativeRoute Transformer V4（条目 66–68B）
专门针对机制 C 与 D 进行最小且充分的架构设计，避免 backbone 表征混淆：
- **严格加性整轨修正 Head**：完全冻结条目 64 的 backbone 与边打分，不做 end-to-end 微调：
  $$L_{\text{corrected}} = L_{\text{edge, W64}} + \Delta L_{\text{route}}, \quad \text{score} = \sigma(L_{\text{corrected}})$$
- **零初始化合同**：修正 Head 在数学上严格零初始化（第 0 步 $\Delta L_{\text{route}} \equiv 0$），保证与生产 edge-only 解算器 100% 恒等。
- **双臂归因实验**：
  - **Arm 1（Control 对照臂）**：绝对整轨坐标表征 $P_4$。
  - **Arm 2（Primary 主实验臂）**：相对跨站坐标与残差表征 $R_4$。
- **求解器感知梯度审计**：平局打破常数（tie-break）确定为 $1.0\times 10^{-9}$；验证了机制 C（margin/dustbin）、机制 D（packing margin）及 Left-SE(3) 规范一致性梯度路由（`split_graph_route_logits`）。
- **合同勘误（条目 68B）**：冻结条目 64 生产边接线、加性 $L_{\text{corrected}}$ 与 checkpoint 保存/加载接口。

### 5. 当前状态：Head-Only Development 评估已关闭（门失败，条目 69–70）
- 条目 69 Arm 1/2 last-epoch checkpoint 已落盘（Condor `1104860`/`1104861`，return 0）。Canonical freeze 文件为 `checkpoint_last.pt`。
- Arm 1 SHA256 `e8a6d6c6e26545de91d9ded59c7b4f9b7840de2e9c94b88d56e1857d6aebd3c8`；Arm 2 SHA256 `a52037ead07555ef937a7e02b65adc9526b509e765341e8af31992552f1fdd3a`。
- 条目 70 reserved-blind development 评估（`1108310`，return 0）**三臂 Workbook-64 门均失败**。Arm 0 C/D 复放仍为 C=200、D=170。Arm 1/2 相对 Arm 0 未恢复整轨效率（reference 0.933 → 0.917 → 0.888）。机制 C 从 200 → 327 → 441。
- reserved-blind overlay `00350_00399` 现已作为 development 见过。未用 final-blind `00800_00849` 与密封测试仍关闭。禁止见过 development 后再调 OP。
- 15 维相对 WLS 仍严格冻结（`continue_to_15d_relative_wls = false`）。未授权下一条训练或 final-blind 评估。

## 核心操作不变量

依据 [FASER 对准运行规程 V1](docs/faser_alignment_operating_protocol_v1_cn.md)：
- **标准传播**：严格采用 mode-0 真实磁场 Acts 外推。
- **物理链路唯一性**：仅由 `/Tracker/Align` 驱动持久化 cluster 重拟合链路；严禁坐标级或残差级代理。
- **密封测试集保护**：最终测试集在算法研发与调优期间永久封存。
- **可辨识度准入**：几何自由度必须基于物理有限差分与 SVD 严格准入，严禁将残差下降作为对准成功的单一判定。

## 文档索引

- [输入 schema 与 exporter 契约（英文）](docs/tracklet_export_contract.md)
- [输入 schema 与 exporter 契约（中文）](docs/tracklet_export_contract_cn.md)
- [当前数据审计（英文）](docs/data_audit.md)
- [当前数据审计（中文）](docs/data_audit_cn.md)
- [baseline 验证（英文）](docs/baseline_validation.md)
- [baseline 验证（中文）](docs/baseline_validation_cn.md)
- [含磁场 propagation 验证（英文）](docs/field_aware_propagation.md)
- [含磁场 propagation 验证（中文）](docs/field_aware_propagation_cn.md)
- [固定 truth alignment closure（英文）](docs/alignment_closure.md)
- [固定 truth alignment closure（中文）](docs/alignment_closure_cn.md)
- [conditions payload 与 coordinate-level closure（英文）](docs/condition_payload_alignment.md)
- [conditions payload 与 coordinate-level closure（中文）](docs/condition_payload_alignment_cn.md)
- [偏移几何 Segment refit（英文）](docs/displaced_geometry_refit.md)
- [偏移几何 Segment refit（中文）](docs/displaced_geometry_refit_cn.md)
- [物理 refit capture-range scan（英文）](docs/physical_capture_scan.md)
- [物理 refit capture-range scan（中文）](docs/physical_capture_scan_cn.md)
- [synthetic multi-track overlay（英文）](docs/synthetic_multitrack.md)
- [synthetic multi-track overlay（中文）](docs/synthetic_multitrack_cn.md)
- [真实 payload synthetic unknown-association baseline（英文）](docs/synthetic_unknown_association.md)
- [真实 payload synthetic unknown-association baseline（中文）](docs/synthetic_unknown_association_cn.md)
- [物理错位增强 curriculum MLP baseline（英文）](docs/curriculum_mlp_baseline.md)
- [物理错位增强 curriculum MLP baseline（中文）](docs/curriculum_mlp_baseline_cn.md)
- [pairwise MLP 与全局指派 baseline（英文）](docs/global_assignment_mlp_baseline.md)
- [pairwise MLP 与全局指派 baseline（中文）](docs/global_assignment_mlp_baseline_cn.md)
- [Geometry-Aware Sparse Transformer V1（英文）](docs/geometry_aware_transformer_v1.md)
- [Geometry-Aware Sparse Transformer V1（中文）](docs/geometry_aware_transformer_v1_cn.md)
- [Geometry-Aware Transformer V2 机制诊断（英文）](docs/geometry_aware_transformer_v2_diagnostics.md)
- [Geometry-Aware Transformer V2 机制诊断（中文）](docs/geometry_aware_transformer_v2_diagnostics_cn.md)
- [Geometry-Aware Transformer V2 route-aware 验证研究（英文）](docs/geometry_aware_transformer_v2.md)
- [Geometry-Aware Transformer V2 route-aware 验证研究（中文）](docs/geometry_aware_transformer_v2_cn.md)
- [Geometry-Aware Transformer V3 结构化全局指派研究（英文）](docs/structured_assignment_v3.md)
- [Geometry-Aware Transformer V3 结构化全局指派研究（中文）](docs/structured_assignment_v3_cn.md)
- [多方向 route-level 物理扫描（英文）](docs/multidirection_route_level_physical_scan.md)
- [多方向 route-level 物理扫描（中文）](docs/multidirection_route_level_physical_scan_cn.md)
- [IFT R_y 真实转动研究（英文）](docs/ift_ry_physical_rotation.md)
- [IFT R_y 真实转动研究（中文）](docs/ift_ry_physical_rotation_cn.md)
- [multi-DoF 全局 alignment 闭环（英文）](docs/global_alignment_multidof_loop.md)
- [multi-DoF 全局 alignment 闭环（中文）](docs/global_alignment_multidof_loop_cn.md)
- [项目审查与下一阶段计划（英文）](docs/project_audit_and_next_plan.md)
- [项目审查与下一阶段计划（中文）](docs/project_audit_and_next_plan_cn.md)
- [FASER 对准运行规程 V1（英文）](docs/faser_alignment_operating_protocol_v1.md)
- [FASER 对准运行规程 V1（中文）](docs/faser_alignment_operating_protocol_v1_cn.md)
- [四站 alignment（英文）](docs/four_station_alignment.md)
- [四站 alignment（中文）](docs/four_station_alignment_cn.md)
- [英文 README](README.md)
