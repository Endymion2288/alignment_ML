# 四站 Alignment（S0/IFT + S1 + S2 + S3）

## 科学问题

当 IFT/S0、S1、S2、S3 都带有真实 `/Tracker/Align` misalignment 时，现有
source-disjoint 物理链

```text
/Tracker/Align → SegmentFitRefit → SegmentsRefit → NtupleDumper
  → FaserActsExtrapolationTool(mode 0)
```

能否通过轨迹 association + alignment 恢复**相对**四站几何？必须显式处理
global common mode / gauge 以及 survey `dz` prior。任何一站都不被假定正确，
station 0 也不是默认 reference。

本分支**不**复制历史上「固定 S0、只拟合 IFT」的假设。旧主线的 IFT-only
5-DoF + survey-`dz` 结果保持冻结；本分支建立新的 contract。

## 第一版状态

| 角色 | 分量 | 个数 |
| --- | --- | ---: |
| 自由、径迹约束 | 每站 `dx, dy, rx, ry, rz` | 20 |
| survey 约束 | 每站 `dz`，5 mm prior | 4 |

`dz` 不升为自由 Newton 坐标。20 维空间**不被假定可解**；由物理 Jacobian
的 SVD 决定准入的相对子空间。

## Schema 与注入

机器可读参数名为 `s{station}_{component}`，例如 `s2_ry_mrad`。每个物理点
都写出**全部四站**六矢量。Calypso 合成仍是 `T(dx,dy,dz) * Rz * Ry * Rx`
（mm、rad）。GeoModel 以左乘 `T_new = g * T_nominal` 应用 station delta。
禁止坐标或残差 surrogate。

`alignment_formulation: four_station_v1` 必须显式打开。省略该字段的配置
保持历史行为：非空 reference，且 reference 站必须停在单位变换。

## Gauge 与物理约束

Jacobian 建立之后的求解坐标：

1. **`reference_station`**：改写 `T_i' = T_ref^{-1} T_i`，使一站为单位变换。
   reference 是坐标选择，不是真实几何。
2. **`common_mode_constraint`**：四站都可写，用左作用去掉 common mode
   `T_i' = G^{-1} T_i`，`G` 由四站平均六矢量构成。对平均六矢量做分量相减
   在有限转动下**不是**同一个映射，只作为线性化 Jacobian 诊断。

FD / identifiability 使用 **`unconstrained_full`**：四站都做扰动，没有任何
一站被钉在单位变换。

结果只在转换成

```text
ΔT_ij = T_i^{-1} T_j
```

之后比较。左乘同一个 SE(3) 元使 `ΔT_ij` 不变（真 gauge）。给所有站加上
同一个平移六矢量也是真 gauge。有限的**加性**公共转动并不自动等于左乘
SE(3) 转动；那是 transform semantics，不能叫做 gauge。把一站强行写成单位
变换却不改写其他站，是另一种物理约束。

## 第一阶段 — 只做 identifiability，不训练 Transformer

1. 在 1–3 个 train source、nominal 附近，对 20 个自由分量与 4 个 survey
   `dz` 做独立 central-FD。
2. 输出逐参数灵敏度、站对残差响应、rank、scaled condition、SVD 向量、
   协方差/相关、source spread。
3. 显式检查整体平移/转动 common mode、相邻站相对模、长基线弱模。
4. 两种求解坐标的 gauge 不变量 `ΔT_ij` 比较。

identifiability map 出来之前，不重训 V2，不生产 20 维 curriculum。若冻结
V2 在四站同时错位下仍然稳定，则继续使用；只有「truth 边还在，但冻结
score/route 系统性退化」时才启动四站感知重训。train/validation 按原始
xAOD 隔离。密封 test 保持关闭。若训练模型，只允许 GPU。

## 第一阶段结果 — 已准入的相对子空间

两个 train source（各 50 事件，μ−/μ+）完成 51 点物理 FD bank。Condor
cluster 1000360：每源 51/51，零 `failure.json`，content audit 齐全。

无约束 24 维数值满秩但不可用（scaled condition ~1.6×10⁹）。survey `dz`
数据 σ 约 43–49 mm。去掉 `dz` 后的 20 维自由图 condition ~8.5×10⁵，仍藏
着五个整体 common mode（最后奇异值 ~6–12，相对 `rx`/`ry` ~10⁶；站间
`dx`–`dx` 相关 0.96–0.98）。**24 维与 20 维都不准入为 Newton 坐标。**

参考站 15 维图（求解时丢掉一站的五个径迹约束坐标）满秩，scaled
condition ~2×10⁴。pooled native σ 对 `dx,dy,rx,ry` 约 0.2 mm / 0.15 mrad，
`rz` 约 1.2 mrad。再加相对 `dz` 后 condition 回到 ~1.5×10⁸。`dz` 保持
5 mm survey prior。若 15 维相对 closure 失败，首先丢掉 `rz`。

在单位变换处线性化的 Jacobian 上删列是求解期坐标，并不表示该站物理
正确，也不等于有限转动后的 `T_i' = T_ref^{-1} T_i` 再线性化。几何只通过
`ΔT_ij = T_i^{-1} T_j` 比较。左乘 SE(3) common mode 保持该表；分量平均
相减不保持。

下一驱动是对现有 held-out 跑
`scripts/run_four_station_relative_closure.py`，capture 已预注册在
`configs/physical_refit_four_station_relative_closure.yaml`。除非 truth
边仍在而冻结 score/route 退化，否则不重训 V2。

本 identifiability bank 上的 truth-selected 15 维 closure 已按该预注册
`ΔT_ij` 合同通过（两个 held-out、两种参考站图；见条目 50）。公共 dx
不进入相对几何。S3 删列图在 5 mrad 有限转动上大约用掉一半 `rz` 容差；
那是线性化/坐标图失配，不是 S0 物理正确。

未知关联闭环仍用同一 bank、同一 `ΔT_ij` 合同：不重训 V2、不扩 Athena
生产、不用两个 held-out 调阈。观测过滤必须保持
`movable_station_ids = [0, 1, 2, 3]`；缩成 station 0 会静默丢掉 1→2 / 2→3。
15 维删列只是求解期坐标，不是某一站物理真值。无约束 20 维仍不准入。
Association 直接打在 identity 物理 ROOT 上（无 overlay）。闸门预注册在
`configs/physical_refit_four_station_unknown_association.yaml`（条目 51）。
残差下降只作为 DQ。

本 bank 上的冻结 V2 未知关联诊断（条目 52）**未过**预注册 vs-nominal
association 闸。Identity 物理图仍保留 truth chain 与冻结 score，但冻结
operating point 的域是 overlay。Overlay 上 raw chain recall 为 1.0，选中边
含 1→2 / 2→3，但 complete-track efficiency 从 nominal 0.90 降到两个
held-out 的 0.71，集中在注入 5 mrad `ry` 的 2→3 / S3。Score 阈值仍留住
全部完整 truth chain（473/473）。未打开 15 维相对 WLS。失败层级是
association domain shift；现在可以规划四站感知 GPU 重训，但禁止用这两个
held-out 调阈。

条目 53 开始该 matched retraining pilot。**不**重设计 Transformer。相对
错位在已准入的 15 维 S0 图中采样，再左乘公共 SE(3) 作为 gauge-control。
训练仍用 identifiability pilot 已完成的两个 xAOD；validation 用从未进入
该 pilot 的两个 expanded-contract source
（`mc24_100047_00050_00099`、`mc24_100048_00050_00099`）。架构、
`residual_v1`、unit-capacity packing 与 30 epoch 预算保持历史 V2。禁止用
条目 52 的 held-out overlay 选 threshold / unmatched penalty /
calibration。在 source-disjoint association 闸通过之前，15 维未知关联
WLS 保持关闭。

条目 54 跑完该 pilot。Condor cluster 1000434 完成 28/28 物理点。新的
source-disjoint validation overlay 上，冻结历史 V2 仍未过 vs-nominal
（hard `s3_ry` efficiency 下降 0.13；同一 `ΔT_ij` 的 gauge twin 不一致）。
Matched retraining 恢复了 2→3 / S3（hard 2→3 0.51 → 0.86），并通过
gauge-invariance audit。预注册 complete-track efficiency 下降 ≤0.10 仍在
一个 payload 上失败（`draw_00_plus_common`，下降 0.104）。15 维未知关联
WLS 保持关闭。2→3 崩塌主要是历史分布失配，不是完全缺少相对几何归纳偏置；
该 checkpoint 不冻结进 WLS。

条目 55 是对该 retrained V2 的 validation-only operating-layer 审计，外加
一个预先冻结的低容量控制。逐 event 对齐的 `draw_00` gauge twin 显示：边
排名稳定（Platt 保持 pair 内排序；未过 0.5 的 1→2 / 2→3 仍多为 source
rank-1），但 left-SE(3) twin 上绝对分数下跌。0.104 的 efficiency 越界几乎
全部来自 0.5 的 station-pair 阈值，不是 candidate graph 缺失。Nominal
fake≈0.10 来自 validation 选出的 `unmatched_penalty=+0.5`，它放进大量短
片段。唯一控制（train-only 三对 Platt、冻结 logits、历史 packing 0.001 /
−1.0）把 nominal fake/purity 拉回原质量区间，也清掉了
`draw_00_plus_common`，但把 nominal efficiency 打到 0.45，并在
`draw_01_plus_common` 上以 0.117 失败。记录：表示基本足够，association
operating layer 尚未达到 production gate。不放宽 0.10。不打开 15 维 WLS。

条目 56 是同一架构 V2 的单一预注册训练目标控制：gauge-twin consistency
加上局部 packing-utility margin，保留原 V2 edge BCE / route-query 主体。
训练前冻结 operating convention（identity Platt、threshold 0.001、
unmatched_penalty −1.0）。条目 53–55 的 validation 源降为
`development_validation_only`，不能再宣称 production gate。最终闸改到全新
source-disjoint transfer bank（`mc24_100047_00300_00349`、
`mc24_100048_00300_00349`）。密封 test 保持关闭。只有 transfer 完整通过
才打开已 truth-selected 的 15 维 WLS，capture 继续用条目 49/50 冻结合同。

## 命令

```bash
source scripts/setup_environment.sh ml
python scripts/prepare_four_station_identifiability_pilot.py \
  --source-config configs/physical_curriculum_four_station_identifiability_sources.yaml \
  --iteration-template configs/physical_refit_four_station_identifiability_pilot.yaml \
  --output-root outputs/mc24_four_station_identifiability_pilot_v1 \
  --iteration 0 \
  --nevents 50 \
  --source-id mc24_100043_00200_00299 \
  --source-id mc24_100044_00300_00399
```

物理生产沿用现有 Condor 封装。完成后必须检查 `failure.json`、ROOT
可读性、content audit 与 manifest completion，不能只看 Condor 退出码。然后：

```bash
python scripts/audit_four_station_identifiability.py \
  --iteration-manifest outputs/mc24_four_station_identifiability_pilot_v1/iteration_manifest.json \
  --output-json outputs/mc24_four_station_identifiability_pilot_v1/identifiability_audit.json \
  --split train
```

准入相对子空间（只读 train）。capture 预注册之后，再在同一 bank 上做
truth-selected 15 维 closure：

```bash
python scripts/admit_four_station_relative_subspace.py \
  --iteration-manifest outputs/mc24_four_station_identifiability_pilot_v1/iteration_manifest.json \
  --output-json outputs/mc24_four_station_identifiability_pilot_v1/admitted_subspace.json \
  --split train

python scripts/run_four_station_relative_closure.py \
  --iteration-manifest outputs/mc24_four_station_identifiability_pilot_v1/iteration_manifest.json \
  --observed-point iteration_00_closure_relative \
  --output-json outputs/mc24_four_station_identifiability_pilot_v1/relative_closure_relative.json \
  --operating-point configs/physical_refit_four_station_relative_closure.yaml \
  --split train
```

未知关联（冻结 V2、identity 物理 bank、15 维相对 WLS）：

```bash
python scripts/prepare_four_station_identity_association_manifest.py \
  --iteration-manifest outputs/mc24_four_station_identifiability_pilot_v1/iteration_manifest.json \
  --output-json outputs/mc24_four_station_identifiability_pilot_v1/identity_association_manifest.json

python scripts/run_frozen_association_backbone.py --backbone v2 --q-over-p-mode 0 \
  --split train --device auto \
  --synthetic-manifest outputs/mc24_four_station_identifiability_pilot_v1/identity_association_manifest.json \
  --frozen-output /eos/home-x/xcheng/FASER/alignment_ML/outputs/mc24_v3_expanded_trainval_v2_bce_control_v1 \
  --payload-id iteration_00_reference \
  --output-dir outputs/mc24_four_station_identifiability_pilot_v1/frozen_v2/iteration_00_reference

python scripts/run_four_station_route_selected_relative_closure.py \
  --iteration-manifest outputs/mc24_four_station_identifiability_pilot_v1/iteration_manifest.json \
  --observed-point iteration_00_closure_relative \
  --anchor-association-output outputs/mc24_four_station_identifiability_pilot_v1/frozen_v2/iteration_00_reference \
  --output-json outputs/mc24_four_station_identifiability_pilot_v1/unknown_association_relative.json \
  --operating-point configs/physical_refit_four_station_unknown_association.yaml \
  --split train
```

Matched association retraining（条目 53）。在 validation association 闸通过
之前，不要从这条路径打开 15 维 WLS：

```bash
bash scripts/run_four_station_association_retraining.sh prepare
bash scripts/run_four_station_association_retraining.sh submit
```

条目 55 的 operating-layer 审计与唯一预注册控制（不打开 15 维 WLS）：

```bash
python scripts/audit_four_station_route_operating_layer.py \
  --synthetic-manifest outputs/mc24_four_station_relative_association_retrain_v1/overlay_synthetic_v1/synthetic_corpus_manifest.json \
  --frozen-output outputs/mc24_four_station_relative_association_retrain_v1/retrained_v2 \
  --output-dir outputs/mc24_four_station_relative_association_retrain_v1/operating_layer_audit_v1 \
  --split validation --device auto

python scripts/run_four_station_operating_layer_control.py \
  --control-config configs/physical_four_station_operating_layer_control.yaml \
  --synthetic-manifest outputs/mc24_four_station_relative_association_retrain_v1/overlay_synthetic_v1/synthetic_corpus_manifest.json \
  --frozen-output outputs/mc24_four_station_relative_association_retrain_v1/retrained_v2 \
  --iteration-manifest outputs/mc24_four_station_relative_association_retrain_v1/iteration_manifest.json \
  --output-dir outputs/mc24_four_station_relative_association_retrain_v1/operating_layer_control_v1 \
  --device auto
```

条目 56 已在新 transfer bank 上打开闸：raw-chain recall 通过，相对条目 54
的 origin-matched score-scale 已收缩，但 packing 选出 0 条 route。条目
57–58 证明缺的是 dustbin 0，不是 fragment 拓扑。条目 59 在同一冻结
packing 约定下训练了一个预注册的 dustbin-aware route-margin 辅助项。
layer 1 通过，多数 payload 已选出 route（`U_truth` 中位 +0.87），但
`draw_01` 加 common SE(3) 未过 vs-nominal efficiency 与 twin route-metric。
15 维 WLS 不打开。新 checkpoint 只是 control，不是冻结的 production V2。
条目 60 在该冻结 checkpoint 上直接调用生产 `_route_hypotheses`：所有
production fragment winner 都已被条目 59 的 `max(rival)` 选中（train
14/14，transfer 513/513）。不要打开 solver-in-the-loop mining。条目 61 统计了全部 train
完整 truth route，而不是只看 14 个 winner：740 / 3331 条在冻结
margin 内已有 2/3-station production competitor，其中 584 条是该
event 的 max dustbin-aware loss，mean reduction 会按 2 或 3 条
完整 route 把这个 max 稀释掉。Train 并非缺少短 near-boundary
覆盖，因此不要判成 curriculum/domain-coverage，也不要用极端
reweighting 去硬学不存在的短 hard 分布。该 control 已在条目 62 按冻结合同训练
（`a46a35bd28eb294fe307590d4f12595f6d3bfaa8dc64aea0bf7418543605e1ef`）。
layer 1 通过，score-scale 相对条目 54 收缩，但 `draw_01` 加
common SE(3) 仍未过 vs-nominal efficiency（`Δeff=0.104`）和 twin
efficiency（`|Δ|=0.070`）。Train 上 584 个 event-max short case
的新 `Δ` 中位仍是 0.992，没有离开冻结 margin。停止继续用
weighting / reduction / OP 救模型。失败归类
`objective_reduction_failure`。15 维 WLS 不打开。

条目 63 是 source-disjoint training-diversity 审计。不再改条目 59/62
的 weighting、reduction、margin 或 operating point。已经打开的条目 56
transfer 从现在起只做 development diagnostic，不再承担下一模型的最终
独立 gate。冻结条目 62 在 train / development / transfer 上的计分，
加上未密封候选 identity bank，显示 occupancy、charge 与 route
multiplicity 已经重叠，但 `draw_01+common` failure core 在 `ty` /
S3 `tx` 上落在当前两条 train source 之外（kinematic inside 0.821 /
0.809 < 0.90）。同一 overlay recipe 还有很大的 source-characteristic
偏移：transfer hard rate 0.799 vs train 0.234，fragment-winner rate
0.170 vs 0.010，`Δ` 中位 0.413 vs 1.104。development 的
`100047/100048` 00050–00099 已经重复这一图案。覆盖归类
`source_phase_space_undercoverage`。授权新增 source-disjoint 训练源
（保留当前 μ± 对，另加 `100043_00300`、`100044_00200`、
`100047_00100`、`100048_00100`）。条目 62 objective 整体冻结。新的
最终 gate 是从未进入 48–62、本条目也未加载的 reserved blind
`100047_00350` / `100048_00350`。闸门仍是 nominal purity ≥ 0.95、
fake ≤ 0.05、全部 non-nominal `Δeff≤0.10`、gauge-twin 与 2→3/S3
stability。15 维 WLS 要等这个新 blind 通过才打开。若冻结模型在那里
仍以相同 common-SE(3) truth-utility drop 失败，下一步才讨论
architecture-level relative / gauge-equivariant representation。

条目 56 的 gauge-consistent route training（唯一预注册 objective；
从条目 63 起，已打开的 transfer 只做 development diagnostic）：

```bash
bash scripts/run_four_station_gauge_consistent_training.sh audit-sources
bash scripts/run_four_station_gauge_consistent_training.sh prepare-transfer
bash scripts/run_four_station_gauge_consistent_training.sh submit-transfer
bash scripts/run_four_station_gauge_consistent_training.sh train
```

条目 59 的 dustbin-aware route-margin control（推理约定冻结；打开
transfer 结果后禁止调参）：

```bash
bash scripts/run_four_station_dustbin_aware_training.sh train
bash scripts/run_four_station_dustbin_aware_training.sh infer-transfer
bash scripts/run_four_station_dustbin_aware_training.sh score-scale
bash scripts/run_four_station_dustbin_aware_training.sh mechanism
bash scripts/run_four_station_dustbin_aware_training.sh assess
```

条目 60 的 solver-generated hard-negative mining 审计（冻结条目 59
checkpoint；只按 train 决策；禁止用 transfer 调参）：

```bash
bash scripts/run_four_station_solver_hard_negative_audit.sh all
```

条目 61 的 train-only weighting / reduction 可行性（冻结条目 59
checkpoint；禁止用 transfer 选 reduction）：

```bash
bash scripts/run_four_station_weighting_reduction_audit.sh
```

条目 62 的 hard-aware max-reduction control（条目 61 冻结合同；
在已打开的 transfer 上只评估一次）：

```bash
bash scripts/run_four_station_hard_aware_reduction_training.sh train
bash scripts/run_four_station_hard_aware_reduction_training.sh infer-transfer
bash scripts/run_four_station_hard_aware_reduction_training.sh score-scale
bash scripts/run_four_station_hard_aware_reduction_training.sh mechanism
bash scripts/run_four_station_hard_aware_reduction_training.sh assess
bash scripts/run_four_station_hard_aware_reduction_training.sh compare
```

条目 63 的 source-disjoint training-diversity 审计（冻结条目 62
checkpoint；禁止调参；不加载 reserved blind；条目 56 transfer 只做
diagnostic）：

```bash
bash scripts/run_four_station_training_diversity_audit.sh
```
