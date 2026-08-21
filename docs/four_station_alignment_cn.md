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
