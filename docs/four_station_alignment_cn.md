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
