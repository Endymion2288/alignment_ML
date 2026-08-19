# 2026-08-18 (26) candidate generation 与 dx–Ry source 依赖性的分离诊断

## 任务边界（用户指令）

不扩 Rx/Rz/dz、不训练新模型、不改 V2 route policy、不再尝试简单 covariance rescaling。两条线：(1) alignment-side 只读分析 iteration-1 train/validation route-selected observations，逐 source leverage/influence、Jacobian SVD/eigenmode、dx–Ry 最弱约束方向、x/tx residual 尾部贡献、LOSO update shift，解释 `mc24_100047_00150_00199` 的反号/大幅更新（禁止剔除或重加权该 source）；(2) 回到 multi-track unknown-association synthetic overlay（单径迹 physical bank fake 恒零，无法决定真实 candidate gate），保持 exporter 原始 covariance、冻结 V2 checkpoint/calibration/threshold，扫描 candidate χ² gate 25/50/100/200/500/ungated，train 选择、validation 只做冻结评估。

## 线 1：逐 source influence 诊断（alignment-side 只读）

新工具 `scripts/audit_route_selected_source_influence.py`（2 个单元测试）：从 update NPZ 重建 pooled normal equation（与 summary 的 recovered delta 逐位一致），按 source 分解 N=ΣN_s、b=Σb_s，计算 leverage tr(N⁻¹N_s)、scaled 本征系统、anchor χ² 尾部对弱方向 rhs 的贡献、逐源 Jacobian 弱方向与 pooled 的夹角、精确 LOSO 重解。

产物：`outputs/mc24_multidof_ift_iteration01_source_influence_{train,validation}_v1/`。

### 层面一：iteration-1 的正常矩阵并不简并

scaled 本征值比 = 条件数 80（train）/86（validation）；本征方向几乎轴对齐（最弱 = dy，其次 dx，最强 Ry）；dx–Ry 相关仅 0.08（train）/0.27（validation）。iteration-0 的 dx–Ry −0.97 强相关是 35 mrad 大失配锚点的现象，收敛后消失。

### 层面二：离群源的 LOSO 与逐源解

| source (validation) | leverage | χ² q99 | Jacobian 弱方向对齐 | LOSO Δdx (mm) | 逐源解 dx err (mm) |
|---|---|---|---|---|---|
| **mc24_100047_00150_00199** | 0.415 | 1168 | 0.920 | **+0.0929** | **−0.438** |
| 其余 7 源 | 0.21–0.59 | 547–3188 | 0.30–0.97 | ≤0.022 | ≤0.012 |

去掉离群源后 validation pooled dx delta = 0.1393 vs 期望 0.1408（误差 −0.0015 mm）——**validation 的全部 dx 误差来自这一个 source**。但它的 leverage、χ² 尾部、Jacobian 方向、选中边运动学（<x>、<|x|>、tx/ty 分布）全部处于正常范围。用户列出的三个假设（高 leverage / 重尾 residual / 局部 Jacobian 异常）**均被排除**。

### 层面三：微观机制——单条错配边 × 6 路重复

逐边响应偏差（response − J·δ_true）分析：离群源 90% 的边 |bias_x|≤0.022 mm（完全正常），但 2.5%（6 条观测）bias_x > 50 mm、最大 362 mm，全部来自**同一条物理边**（origin run 9000000300, event 12, tracklet 0 → target tracklet 1，0->1）：

- 该物理事件是**多径迹事件**（station 0 一条 tracklet，station 1 三条 tracklet）。truth-free backbone 为同一 source 选中了三条候选边：到 target 2/3 的边 χ²=0.2/0.0（正确），到 target 1 的边 χ²=12.8、pull_x=−3.08（**错配**，residual −150 mm）——在虚高协方差下仅 3σ，低于 gate 25。
- 该错配边在 anchor residual −150 mm、target +148 mm（错误 target 不随源径迹的几何响应移动 → 响应 +298 mm，物理上不可能）。
- 同一物理边被 overlay 复用到 6 个 synthetic event、形成 6 条 route 副本进入解（validation 全体：1756 条观测 = 736 条唯一物理边，重复度最高 9×，平均 2.4×）——**overlay 复用把单条坏边的权重放大 6 倍**。
- WLS 无法降权：anchor χ²=12.8 对应 Huber k=2.5 权重 0.70（几乎不降）；这解释了前一阶段 Huber 控制为何无效，也解释了校准（缩小 x 方差 ×0.18）为何把该边的 rhs 贡献放大 5.6 倍导致 validation dx 误差 −1.19 mm。

**因果链**：多径迹事件 + 虚高协方差使错配边以 3σ 通过 → overlay 复用 ×6 放大 → WLS 无法识别 → 单源解 −0.44 mm → pooled −0.0945 mm。这不是 leverage/尾部/Jacobian 问题，而是 **association 错配边在现有协方差模型下不可识别**的问题。

## 线 2：candidate χ² gate 扫描（multi-track overlay，冻结 V2）

实现：backbone 脚本新增 `--candidate-chi2-gate`（默认 None = 现行冻结行为，只改候选图构建，不动 checkpoint/calibration/threshold/solver），anchor payload 上 5 gate × 2 split 推理；ungated 复用现有输出。随后每个 gate 的 anchor 选路接入同一个冻结 route-selected update（原始协方差、无 Huber）。

产物：`outputs/mc24_multidof_ift_iteration01_v2_backbone_gate{25,50,100,200,500}_{train,validation}_v1/`、`outputs/mc24_multidof_ift_iteration01_route_selected_update_{train,validation}_gate*_v1/`、驱动 `scripts/run_gate_scan_backbones.sh` / `run_gate_scan_closures.sh`。

### Route 级指标（train / validation）

| gate | cand chain recall | route retention | track eff | purity | fake rate |
|---|---|---|---|---|---|
| 25 | 0.41 / 0.28 | 0.003 / 0.000 | 0.003 / 0.000 | 1.000 | 0.000 |
| 50 | 0.54 / 0.45 | 0.008 / 0.001 | 0.008 / 0.001 | 1.000 / 0.933 | 0.000 / 0.067 |
| 100 | 0.65 / 0.57 | 0.022 / 0.010 | 0.022 / 0.010 | 1.000 / 0.973 | 0.000 / 0.027 |
| 200 | 0.74 / 0.71 | 0.047 / 0.034 | 0.047 / 0.034 | 0.992 / 0.966 | 0.008 / 0.034 |
| 500 | 0.87 / 0.85 | 0.109 / 0.101 | 0.109 / 0.101 | 0.988 / 0.966 | 0.012 / 0.034 |
| ungated | **1.00 / 1.00** | **0.854 / 0.743** | **0.854 / 0.743** | 0.975 / 0.962 | 0.025 / 0.038 |

### 各 gate 接入冻结 update 的 closure 误差（dx/dy/Ry，mm/mm/mrad）

| split | gate | dx err | dy err | Ry err | 公共边 | 判定 |
|---|---|---|---|---|---|---|
| train | 25–500 | ≤0.002 | ≤0.001 | ≤0.057 | 23–413 | 全过 |
| train | ungated | −0.0008 | −0.0005 | −0.0003 | 2309 | ✓ |
| validation | 25 | +0.0046 | −0.0010 | +0.0002 | 17 | ✓（脆弱）|
| validation | 50 | +0.0054 | −0.0008 | +0.0161 | 24 | ✓（脆弱）|
| validation | 100 | +0.0013 | −0.0020 | +0.0159 | 66 | ✓（脆弱）|
| validation | 200 | **−0.4041** | +0.1111 | −0.1061 | 158 | ✗ |
| validation | 500 | **−0.1864** | +0.0548 | −0.0507 | 350 | ✗ |
| validation | ungated | −0.0945 | +0.0575 | −0.0280 | 1756 | ✓ |

直接验证：离群源错配边（χ²=12.8）在 gate 25/100 的稀疏图上未成活（0 份副本），从 gate 200 起进入解（1 份），ungated 时 6 份——gate 25–100 的"通过"是低统计下偶然排除了污染边，不是可用操作区。

## 判定（按用户决策分支）

**不存在可用的更优 operating region；当前瓶颈是 propagation/candidate model 本身，不是阈值。** 依据：

1. 现行 candidate-generation policy 已是 ungated（覆盖最大端：候选层 complete truth-chain recall 0.998/1.000）；gate 轴上没有任何方向可"放宽以提高 coverage"。
2. 收紧 gate 只毁效率不赚纯度：purity 0.975→1.000 的代价是 track efficiency 0.854→0.003；中间点（200/500）把低统计与污染浓缩结合在一起，validation closure 反而失控（−0.40/−0.19 mm）。
3. 唯一需要排除的错配边 χ²=12.8，低于任何可用 gate（gate 25 已把 truth-chain recall 打到 0.41）——truth 的重尾与 fake 的中等 χ² 在现有协方差模型下不可分离。**候选/传播模型（mode-0 协方差给出的 χ² 判别力）才是瓶颈**。
4. candidate-generation policy 因此保持现状（ungated + 原始协方差），不冻结任何新阈值；validation closure 维持上一阶段的通过结果（−0.0945/+0.0575/−0.0280）。

### 对后续扩展的前置约束（更新）

- dx–Ry"简并"在 iteration-1 已消失；剩余风险是**单点错配边经 overlay 重复放大**。未来做 Rx/Rz/dz 与 station-level 6-DoF 前，应优先考虑：(a) 在 route-selected update 的观测构建层对同一物理边的多 route 副本做去重或权重归一（统计上更诚实，且直接消除 6× 放大器）；(b) 在 candidate 模型内提升 χ² 判别力（正确的 mode-0 协方差），而不是调 gate。
- 离群源 `mc24_100047_00150_00199` 未被剔除或重加权；以上全部为只读诊断。

## 测试

- 新增 `tests/test_route_selected_source_influence.py`（2 测试）；backbone 的 `--candidate-chi2-gate` 默认行为不变（既有回归测试覆盖）。
- 全套测试 189 通过。
