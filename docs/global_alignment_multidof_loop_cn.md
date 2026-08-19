# 多自由度全局 Alignment 闭环

## 范围

这是 FASER 全局 association 与 alignment 的物理控制链，不新增 Transformer 架构。
association backbone 保持冻结，当前可用的严格封存控制为 V2 BCE route-query。
永久封存的 test bank 不会被读取。

第一批激活的是 IFT/station-0 的 `dx`、`dy`、`Rx`、`Ry`、`Rz`，`dz` 以 survey prior
约束；下游 S1--S3 固定为参考坐标系。
Calypso payload 为 `[dx, dy, dz, Rx, Ry, Rz]`，单位为 mm/rad；报告中转动使用 mrad。

## 物理约束

每个点都独立执行：

```text
/Tracker/Align SQLite/POOL payload
  -> SCT_ClusterContainer
  -> SegmentFitRefit
  -> SegmentsRefit
  -> NtupleDumper
  -> FaserActsExtrapolationTool (mode 0)
```

不允许坐标平移替代、residual-level 注入、缓存 propagation 或 local `q/p` 替代。
local segment `q/p` 仍只是不可依赖的 seed，因此物理 V1 一律使用 mode 0。

## 可重复的 Iteration

生成 source-disjoint 的 anchor/probe bank：

```bash
source scripts/setup_environment.sh ml
python scripts/prepare_multisource_multidof_iteration.py \
  --source-config configs/physical_curriculum_v3_expanded_trainval.yaml \
  --iteration-template configs/physical_refit_multidof_smoke_mc24_100043.yaml \
  --output-root outputs/mc24_multidof_ift_iteration00_anchor_trainval_physical_v1 \
  --iteration 0 \
  --current ift_dx_mm:2.0 --current ift_dy_mm:-1.5 --current ift_ry_mrad:35.0 \
  --nevents 100
```

该 bank 含 10 个 train 与 8 个 validation 原始 xAOD file。每个 source 有 8 个独立
refit 的真实物理点：reference、anchor，以及三个激活参数的正负 probe。每 source 一个 job：

```bash
python scripts/submit_multisource_multidof_iteration_condor.py \
  --iteration-manifest outputs/mc24_multidof_ift_iteration00_anchor_trainval_physical_v1/iteration_manifest.json \
  --submit-dir outputs/condor_mc24_multidof_ift_iteration00_anchor_trainval_physical_v1 \
  --schedd-mode eossubmit --submit
```

全部点通过 completion check 后，聚合真实有限差分：

```bash
python scripts/run_multisource_refit_multidof_local_step.py \
  --iteration-manifest outputs/mc24_multidof_ift_iteration00_anchor_trainval_physical_v1/iteration_manifest.json \
  --anchor-point iteration_00_anchor --target-point iteration_00_reference \
  --fit-split train --held-out-split validation \
  --capture-tolerance ift_dx_mm:0.1 \
  --capture-tolerance ift_dy_mm:0.1 \
  --capture-tolerance ift_ry_mrad:1.0 --require-full-rank \
  --output-dir outputs/mc24_multidof_ift_iteration00_anchor_trainval_closure_v1
```

train 只用于确定 update；validation 只评估该冻结 update，并输出独立 fit 诊断。
结果包括每个点的 raw candidate truth-chain retention、有限差分曲率、rank、无量纲
condition number、covariance/correlation，以及 source/station-pair 稳定性。只有
`capture_success=true` 才可推进下一轮。

## Association 闭环

完成的 bank 可以只读地交给既有 pooled synthetic 工具：

```bash
python scripts/assemble_multisource_multidof_iteration_manifest.py \
  --iteration-manifest outputs/mc24_multidof_ift_iteration00_anchor_trainval_physical_v1/iteration_manifest.json \
  --materialization-config configs/physical_alignment_iteration_trainval.yaml \
  --output outputs/mc24_multidof_ift_iteration00_anchor_trainval_physical_v1/physical_corpus_manifest.json
```

`alignment_iteration_shared_across_payloads` 只共享确定性的 overlay 选择，以便跨真实
payload 求 selected-route provenance 的交集。真实 tracklet state、covariance 与 Acts 输出
仍各自对应 payload。冻结 V2 通过 `--payload-id` 对每个物理点单独推理，再把 selected 的
精确 mode-0 Acts edge 交给 `run_route_selected_multidof_update.py`。其中 straight-line fit
只可作诊断，不能作为磁场下的 alignment objective。

`audit_field_global_fit_contract.py` 已审计当前 propagation product：tree 具有全部
pairwise residual/covariance 字段，但没有导出的 transport Jacobian 或 source-state
transition representation。因此当前 field-aware update 必须准确称为“route consistency +
WLS”，不能称为独立全局 likelihood。真正的 field-aware global fitter 要等 Calypso 导出
该 Jacobian，或提供可验证的共同状态 Acts repropagation API 后再实现。

用已验证 update 的 `--update-json` 生成下一轮真实 payload bank。该选项默认拒绝未通过
closure 的 update，诊断性 override 必须显式指定。

## 自由度准入

payload/Jacobian 代码已支持完整 station rigid components。6-DoF identifiability
pilot 准入 station-0 的 `dx`、`dy`、`Rx`、`Ry`、`Rz` 作为径迹约束自由参数，并拒绝
`dz`（对当前近平行样本呈 gauge-like）。`dz` 只能以冻结 survey prior 进入法方程；
prior 主导的后验恢复不得解释为径迹测量。单步 Newton 更新保持在已验证线性域
（5-DoF 归一化 severity `<= 0.15`，另保留少量 `~0.2` 应力点）。禁止把
severity `0.5–1.0` 当作单步 closure 目标。station、layer、module 层级仍预留；
不得把 layer/module conditions 静默映射成 station payload。

## 当前状态

station-level **5 个径迹约束 DoF + 1 个 survey 约束 DoF** 已冻结（条目 36）。
从 severity 0.12 的联合随机 5-DoF 起点出发，未知关联 Newton 步在 source-disjoint
train 与 validation 上单轮闭合（`framework_capture_success=true`，剩余 severity
0.01–0.02）。`dz` 不进入径迹 capture，不得解释为测量。下一步是 station/layer
层级，不是新的 Transformer。

**3-DoF 迭代环此前已闭合。** iteration-0（anchor dx/dy/Ry = 2.0 mm/−1.5 mm/35 mrad）在 held-out
validation 上将偏移恢复到 −0.26/−0.22 mm/−0.05 mrad，并提出 iteration-1 anchor
（−0.14 mm/+0.11 mm/+0.74 mrad）。iteration-1 通过全部冻结 held-out 容差
（dx −0.014 mm、dy +0.019 mm、Ry −0.0003 mrad；逐源散布塌缩 20–40 倍），随后无真值的
route-selected update 用冻结 V2 backbone 在两个 source-disjoint 划分上都闭合：
train dx/dy/Ry 误差 −0.0008/−0.0005 mm/−0.0003 mrad（2309 条公共边，条件数 80），
validation −0.0945/+0.0575 mm/−0.0280 mrad（1756 条公共边，条件数 85）——全部在冻结容差
（0.1 mm/0.1 mm/1 mrad）以内。这是项目第一个 unknown-association + source-disjoint +
真实 payload/refit/Acts + multi-DoF 迭代对齐 closure。validation 的 dx 余量主要由一个
离群 source（`mc24_100047_00150_00199`）主导；详见
`workbook/2026-08-18_24_validation独立closure与第一阶段结论.md`。

过程中修复两个实现缺陷并有回归测试覆盖：alignment-iteration 物化命名空间改为 payload 稳定
（`materialize_pooled_curriculum_synthetics.py`）；update 改为 anchor-selected 模式
（`--observation-kind anchor_selected_field_edge`，在 anchor 固定选路、于每个 payload 的
候选图中按 origin 重测），因为逐 payload 独立选路在 FD probe 间没有交集。

association 架构就此冻结。下一阶段转向两个已知基础瓶颈：mode-0 传播协方差的严重各向异性
失校，以及 0->1 raw candidate coverage。

该阶段已完成第一轮审查循环（train-only pull 校准、冻结对角重标定 + 先验固定 Huber 控制、
nominal 与两个 iteration anchor 上的 gate/coverage 重扫、冻结 iteration-1 bank 上的校准 closure
变体）。结论记录于 `workbook/2026-08-18_25_mode0协方差校准与candidate覆盖修复.md`：

- mode-0 协方差失校的本质是"核心高估 + 重尾欠覆盖"的非高斯畸变：robust pull 宽度
  x 0.30–0.43、y ~0.003、tx 0.72–0.88、ty ~0.002–0.003，|pull| q99 高达 4；
  validation 的 pull 宽度与 train 几乎逐点一致，说明畸变是 source-independent 的。
- raw candidate coverage 是 gate-受限而非协方差-受限：物理事件每 station 至多一条
  tracklet（单径迹拓扑），fake candidate 在所有 gate 下恒为零；把 χ² gate 从 25 放宽到
  500 可将 complete truth-chain recall 从 ~0.40–0.50 提升到 ~0.72–0.86，且无 fake 代价。
- 两个 train-冻结 control 均不采纳：对角核心宽度重标定使 dy 在所有 source 上
  |误差|<0.01 mm，但摧毁 dx/Ry 解（validation dx 误差 −1.19 mm）——重标定后 y/ty 块
  权重超出 x/tx 约 10⁵ 倍而只约束 dy，近简并的 dx–Ry 方向被噪声主导；仅 Huber 控制
  同样使 validation 变差（dx 误差 −0.176 mm）。canonical physical candidate/WLS 协方差
  模型保持为 exporter 原始协方差，校准产物作为文档化负结果保留
  （`outputs/mc24_multidof_ift_pull_calibration_train_v1/`、
  `outputs/mc24_multidof_ift_gate_coverage_scan_v1/`）。

随后的只读诊断把 candidate generation 与 dx–Ry source 依赖性彻底拆开
（`workbook/2026-08-18_26_离群源影响机制与candidate_gate操作区.md`）：

- 收敛后的 iteration-1 锚点上正常矩阵条件数 80–86、本征方向几乎轴对齐——
  iteration-0 的 dx–Ry −0.97 强相关是大失配锚点现象，已消失。
- validation 的 dx 误差完全是单源效应：剔除 `mc24_100047_00150_00199` 后 pooled
  dx 落到真值上（误差 −0.0015 mm）。该源不是高 leverage、非重尾、逐边 Jacobian 与
  运动学均正常。根因是一条多径迹物理事件中的 0->1 错配边（错误 station-1 候选，
  χ²=12.8，两个 payload 上 residual ±150 mm），经 overlay 复用形成 6 条 route 副本
  放大权重；虚高的 mode-0 协方差使该边对 candidate gate 和 WLS 都不可识别。
- 冻结 candidate χ² gate 扫描（25/50/100/200/500/ungated，冻结 V2
  checkpoint/calibration/threshold，train 选择、validation 冻结评估）未找到可用
  operating region：现行 policy 已是 ungated、候选层 complete truth-chain recall
  ~1.0；收紧 gate 使 track efficiency 从 0.854 塌缩到 0.003（gate 25）而 purity 仅从
  0.975 升到 1.000；中间 gate（200/500）把低统计与污染浓缩结合，validation dx 误差
  −0.40/−0.19 mm。candidate-generation policy 保持 ungated + 原始协方差；瓶颈是
  propagation/candidate model 的 χ² 判别力而非任何阈值。Rx/Rz/dz 与 station-level
  6-DoF 扩展前，应先在 route-selected update 的观测构建层对同一物理边的多 route
  副本去重或权重归一。

### 观测统计语义：物理边权重归一（2026-08-18，已冻结为 canonical）

synthetic overlay 把同一条真实物理边嵌入多个 synthetic event，冻结选路因此把同一
物理边的副本观测最多 11 次放进法方程。在冻结的 iteration-1 train/validation closure 上
比较了三种预先定义的统计语义（`scripts/run_observation_statistics_variants.sh`，
`alignment/route_selected_update.py::apply_observation_statistics`；分组只用端点
provenance——原始 source 文件/事件 UID、source/target station 与 tracklet——绝不用 truth）：

| 语义 | split | 观测数 | 唯一物理边 | dx err (mm) | dy err (mm) | Ry err (mrad) |
|---|---|---|---|---|---|---|
| replica_weighted（control） | train | 2309 | 623 | -0.0008 | -0.0005 | -0.0003 |
| physical_edge_deduplicated | train | 623 | 623 | -0.0009 | -0.0003 | -0.0008 |
| physical_edge_inverse_multiplicity_weighted | train | 2309 | 623 | -0.0009 | -0.0003 | -0.0008 |
| replica_weighted（control） | validation | 1756 | 467 | -0.0945 | +0.0575 | -0.0280 |
| physical_edge_deduplicated | validation | 467 | 467 | **-0.0511** | +0.0396 | -0.0159 |
| physical_edge_inverse_multiplicity_weighted | validation | 1756 | 467 | **-0.0511** | +0.0396 | -0.0159 |

发现（`outputs/mc24_multidof_ift_iteration01_obsstat_{train,validation}_v1`，
`..._source_influence_{train,validation}_dedup_v1`）：

- 副本残差逐 bit 相同（最大 spread 0.0），因此去重与逆多重性加权在数值上完全等价，
  与理论预期一致；两者都把每条物理边的总统计权重归一为 1。
- `mc24_100047_00150_00199` 那条已知错配边在 replica_weighted update 中进入 6 次、
  在 dedup update 中恰好进入 1 次（已对观测键显式验证）。
- validation dx 误差减半（-0.0945 → -0.0511 mm），稳定远离 0.1 mm 容差边界；
  dy/Ry 误差同步缩小；train closure 不变（约 1 µm）。两个 split 均通过冻结容差。
- 其余 7 个 validation source 未被牺牲：逐 source dx 解移动最多 0.004 mm。
  离群源的 LOSO 影响减半（+0.0929 → +0.0488 mm）；validation 逐 source dx spread
  从 0.1448 缩到 0.0855 mm。rank 保持 3；条件数仅温和上升（80–86 → 108–135）；
  |corr(dx,Ry)| 保持在 0.3 以下。

**判定：`physical_edge_deduplicated` 冻结为新的 canonical alignment-observation
语义**（确定性的 sorted-first 代表元；`replica_weighted` 经 `--observation-statistics`
保留为严格 control）。统计语义修正完成后，下一个独立层次是 mode-0 propagation
covariance / χ² 判别力研究；只有该层被理解之后，才允许进入 Rx/Rz/dz sensitivity 与
station-level 6-DoF 扩展。

### Mode-0 propagation χ² 判别力机制（2026-08-18，负结果，χ² 层关闭）

专门的边级研究（`scripts/audit_propagation_discrimination.py`、
`scripts/evaluate_propagation_compatibility.py`、`alignment/propagation_compatibility.py`；
train 拟合、validation 仅一次冻结迁移；candidate endpoint、V2 分数与 WLS 协方差均不动）
对 anchor 与 reference 两个 payload 上全部 ungated 相邻 station-pair candidate
（train 84,770 / validation 67,251 条边）按残差、完整联合协方差、pull、χ²、track
state、station pair 和 source 分别建模。

已知错配边（|r_x| = 150 mm 而 χ² = 12.8）的机制：

- 0->1 的 mode-0 联合协方差巨大（σ_x ≈ 49 mm、σ_y ≈ 493 mm），150 mm 残差在 x 方向
  仅为 3.1σ 边际 pull。
- χ² = 12.8 位于 train truth 边 χ² 分布的**中位数**（分位 0.507）：truth 分布本身
  极端重尾——中位数 15.6（校准 χ²_4 应为 3.4），29% 的 truth 边 χ²>100，
  q99 ≈ 1.3×10⁵，最大 4.8×10⁷。
- fake 边 χ² 中位数约 5×10³，但其 1% 下尾（≈6–10）落入 truth 核心，而 truth 上尾
  又越过 fake 上尾——两个分布在每个分位上都重叠。

train 冻结模型比较（按 station pair 的 Student-t 网格 MLE ν、core+tail 缩放高斯 EM
mixture、Huber/Tukey（尺度取 train χ² 分位），均含 log|C| 项；阈值冻结在 train truth
99.5% 保留率）：

- AUC(χ²) = 0.86–0.92；Student-t / mixture 最多提升 +0.01；Huber/Tukey 是 χ² 的
  单调变换，不改变排序；分量 pull 更差（AUC 0.55–0.59）。
- 冻结 veto 下 fake 拒绝率仅 0.3–6%；错配边不被任何模型 veto（因 truth 重尾，
  阈值高达 ~10⁵）。
- 依赖性弱且不可利用：逐 source truth χ² 结构均匀（离群源完全典型：中位数 15.7、
  P(χ²>100) = 0.196）；anchor 与 reference 几乎相同；仅大 |y| 边尾部更重
  （|y|>100 mm 时 P(χ²>100) = 0.59）。

**结论（按冻结判定树）：没有任何 train 冻结似然能可靠分离 truth 重尾与错配边——
mode-0 propagation 在 χ² 层缺少足够判别信息，χ² 层调参到此为止。** 不再做
compatibility-veto closure 变体：错配边位于 truth 中位数，任何能 veto 它的阈值都会
同时拒绝约一半 truth 边。瓶颈在统计层之上游：必须先理解 mode-0
propagation/covariance 模型本身（σ_y ≈ 493 mm、truth χ² 尾到 10⁷），之后才允许
Rx/Rz/dz sensitivity 与 station-level 6-DoF 扩展。

### Mode-0 propagation 根因审计（2026-08-18，根因已定位）

χ² 层关闭后，对 truth-matched propagation uncertainty 做了只读来源分解
（`scripts/audit_propagation_uncertainty_budget.py`；train/validation 物理 bank 的
anchor 与 reference payload）。现有 bank 已为每条 truth 边存了三个同边 q/p control
（除非注明，重建 position/direction 相同）：mode 0 = 重建状态 + segment 拟合 q/p；
mode 1 = 重建状态 + truth q/p 且**抑制 q/p 协方差**；mode 2 = 全 truth 状态 + truth q/p。

不确定度预算（train，iteration-1 anchor，中位数；生产配置中 material/process noise
关闭，transport 为纯 J·C·Jᵀ）：

| pair | mode | src σ_y (mm) | prop σ_y (mm) | tgt σ_y (mm) | prop σ_x (mm) | res σ_x (mm) |
|---|---|---|---|---|---|---|
| 0->1 | 0（固定 100 GeV） | 0.50 | **492** | 0.50 | 32.5 | 16.0 |
| 0->1 | 1（truth q/p） | 0.50 | **16.1** | 0.50 | 31.9 | 15.2 |
| 1->2 | 0 | 0.50 | **228** | 0.50 | 22.9 | 11.9 |
| 1->2 | 1 | 0.50 | **0.73** | 0.50 | 22.9 | 11.4 |
| 2->3 | 0 | 0.50 | **225** | 0.50 | 22.9 | 10.1 |
| 2->3 | 1 | 0.50 | **0.73** | 0.50 | 22.9 | 10.1 |

根因（按主导程度排序）：

1. **病态 σ_y = segment 拟合 dummy q/p 方差经磁场 transport。** 每条 tracklet 携带
   q/p = 1e-5/MeV（直线 segment 拟合的 100 GeV 占位值），方差 5e-6/MeV²，即
   σ(q/p) 是中心值的 224 倍——动量实际无约束。mode 0 把该 dummy 方差经磁场
   transport 进 (y, ty) 方向。同边 mode-1 control（truth q/p seed + 抑制 q/p 协方差）
   使 σ_y 塌缩 30–300×，而残差基本不变（0->1：16.0 → 15.2 mm）——纯协方差膨胀，
   无残差退化。这是实现问题，不是物理。
2. **σ_x ≈ 23–32 mm = 方向不确定度的正常杠杆臂 transport**（两种 mode、所有 pair
   均存在）；在 0->1 相对残差高估约 2×。
3. **truth χ² 重尾（10⁵–10⁷）= combined 协方差结构性近奇异 + 真实残差离群子群。**
   杠杆臂 transport 使 position-direction 块近退化：core 边 cond(C) 中位数已达
   3.5×10¹⁰（tail 6.9×10¹¹），近零特征方向 = 0.73·tx + 0.46·ty。tail 边中 64%
   是中等残差被病态条件数放大，36% 是真实残差离群（边际 pull > 5）。重尾与 q/p
   mode 无关（>1e4 比例：mode 0 1.5% vs mode 1 3.5%，且是同一批边）。
4. **material/process noise 贡献严格为零**——生产配置未开启
   InteractionMultiScatering/InteractionEloss（默认 False），transport 不含过程噪声。
5. 未发现单位、frame 或 FD-Jacobian 异常：native→global 与 bound→bound 变换步长合理
   （1e-4 mm、1e-6 rad），core pull ≤ 1。mode-1 在 0->1 的残余 σ_y（anchor 16 mm vs
   reference 0.9 mm）是跨磁铁间隙的 misalignment 相关场耦合；validation 复现 train
   数字（12 mm）。

**修复方向已被 train-only 同边 control 证明**：固定 seed（mode-0）协方差 transport
应抑制 dummy q/p 方差——mode-1 control 显示 σ_y 塌缩 30–300× 且残差不变，mode-1 在
0->1 的 x/tx pull 已接近校准（robust pull σ = 1.01/1.07）。

### Mode-3 抑制 pilot（2026-08-18，判定：仅诊断保留，mode-0 维持 canonical）

已证明的修复方向完成了最小真实 production 验证：NtupleDumper 新增独立 variant
**mode 3**（重建 position/direction 与固定 q/p seed 同 mode-0 完全一致，仅
`suppressQOverPCovariance=true`；mode 0/1/2 不变），3 个 train source × iteration-1
全部 8 个 payload 点完整走 payload → SegmentFitRefit → NtupleDumper → Acts
propagation 真实链（`outputs/mc24_mode3_suppression_pilot_physical_v1/`，24/24 点，
0 失败），随后构建 pilot overlay（参数与生产逐项一致、两 variant 共享同一 synthetic
events）、冻结 V2 backbone 推理、physical_edge_deduplicated 语义下的 route-selected
dx/dy/Ry closure（`scripts/run_mode3_suppression_pilot.sh`）。

pilot 冻结判据（5 项全部成立才升级 canonical）：

| 判据 | 结果 | 判定 |
|---|---|---|
| residual 不变 | 预测与 mode-0 逐比特一致（523/523 记录） | 通过 |
| q/p 诱导协方差膨胀消失 | σ_y 塌缩 30–537×（最小 29.8×） | 通过 |
| separation 或 coverage 改善 | V2 AUC 0.9978 → 0.9817（anchor）；模型无关 χ² AUC 0.955 → 0.941；coverage 完全相同（0.9913） | **失败** |
| route metrics 不退化 | complete-track efficiency 0.88 → 0.14（anchor） | **失败** |
| closure 通过冻结容差 | 两 mode 均 capture；mode-3 dx/dy/ry 误差 ≤ mode-0 | 通过 |

mode-3 协方差本身接近校准（reference 0->1 y pull robust σ = 1.62，mode-0 为
0.0034）：抑制机制严格按设计工作。关联崩塌是冻结栈的特征分布漂移效应而非新物理：
V2 checkpoint/calibration/route 阈值在 mode-0 过覆盖 χ² 特征上训练，诚实协方差下
truth 边 χ² 中位数升至 ~48–92，落入模型的 fake 样区域并在 0->1 阈值处断裂。
**按冻结决策规则：mode-3 保留为诊断 variant，mode-0 维持 canonical propagation
mode。** 要利用校准后的协方差需在 mode-3（或校准协方差）候选图上重训关联模型，
明确超出本阶段范围。在此之前不进入 Rx/Rz/dz sensitivity 与 station-level 6-DoF，
也不再尝试统计层补丁。
