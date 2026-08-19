# 2026-08-19 (37) IFT station→layer identifiability pilot：显式 gauge 与最小可闭合集合

## 任务

station-level 5-DoF + survey-dz 框架已冻结（条目 36）。本阶段**只做** IFT
layer-level identifiability pilot，**不**生产完整 layer curriculum，**不**打开
sealed test，**不**调 mode-0 / V2 backbone / route policy /
`physical_edge_deduplicated` / station 五自由参数 / 5 mm dz prior。

目标是一张可信的 **identifiability map + gauge 定义 + 最小可闭合 layer 参数集合**。
只有这一层稳定后，才决定哪些 layer DoF 进入 train/validation hierarchy
curriculum，以及是否向 module 扩展。

## 物理约定

- 线性化点：已收敛 station geometry，即全零 station 5-DoF（剩余 severity 0.01
  相对 FD 步长可忽略）。`dz` 恒为 0。
- 联合 Jacobian：station 公共模 `dx/dy/rx/ry/rz` **加上** IFT 三层各自的
  `dx/dy/rx/ry/rz`（20 参数）。禁止把 layer 条件写入 station payload 槽。
- Calypso 映射：L1 `station:<id>`；L2 平面键 `f"{station}{layer}"`（`00/01/02`），
  `TrackerAlignDBTool` 在层 z 处做共轭。平移共轭不变；转动的转动中心与
  station 转动不同，**是否近简并由 Jacobian 测量，不由加法约定事先断言**。
- 两种显式 gauge（比较物理解，禁止重复拟合同一刚体自由度）：
  1. **sum_to_zero**：station 承担 common mode；layer 校正覆盖加权和为零
     （默认丢掉 layer 2）。
  2. **reference_layer**：station 承担 common mode；固定 IFT layer 0。
- 读出约定：`station + weighted_mean(layer)` 为 common-mode 报表；
  `layer_i - mean` 为 internal deformation。这是比较两种 gauge 的语言，
  不是把 layer 转动等同于 station 转动。

## FD 设计

- 3 个 train source（与 6-DoF identifiability 相同）：
  `mc24_100043_00200_00299`、`mc24_100043_00600_00699`、`mc24_100044_00300_00399`。
- 每源 43 点：nominal + 20×2 central FD + 2 个 held-out。
  - station FD：dx/dy 0.5 mm，rx/ry/rz 10 mrad（沿用已验证步长）
  - layer FD：dx/dy 0.2 mm，rx/ry/rz 2.0 mrad（小幅线性）
- Held-out：
  - `closure_internal`：layer0/layer2 反向 dx/dy/ry，station=0，等权 sum-to-zero
  - `closure_mixed`：同上 + station dx = 0.15 mm（检验 internal 不被吸进 station）
- 真实链：`/Tracker/Align → SCT_ClusterContainer → SegmentFitRefit →
  SegmentsRefit → NtupleDumper → Acts(mode 0)`。

## 准入与后续（生产完成后）

`scripts/audit_layer_identifiability.py` 将报告：

- 无规范 20 参数 SVD / rank / condition / 列余弦（station dx↔layer dx、
  dy↔dy、rx/ry↔station 转动、layer 间反相关）
- 两种 gauge 下的 covariance、source spread、覆盖加权
- 只有 gauge 后仍独立、逐源稳定、子块满秩的 layer DoF 才进入
  小 severity truth-selected joint closure，再用冻结 V2 做
  unknown-association route-selected closure
- 关闭条件：恢复 layer internal **且不破坏** 已闭合 station 解

本条目启动生产；identifiability map 的数值结论记在后续条目。

## 产物

- 模板：`configs/physical_refit_ift_layer_identifiability_pilot.yaml`
- 源声明：`configs/physical_curriculum_ift_layer_identifiability_pilot.yaml`
- 物理 bank：`outputs/mc24_ift_layer_identifiability_pilot_v1/`
- Condor：cluster **998508** on bigbird24，3 个 worker，flavour `testmatch`（43 点/源）
