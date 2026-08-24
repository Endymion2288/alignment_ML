# 2026-08-23 (56) Module-Level Residual & Identifiability PoC V1

## 任务

路线图 `workbook/2026-08-23_56_module_level路线` 的 Phase 1。不训练新模型，
不改冻结 V2，不重开 Station / reduced Station / `C_dx` Mode，不写
geometry，不求 alignment correction。唯一问题：station-level
non-identifiability 是否主要来自把 cluster 压成 station tracklet。

## 数据链

继续用 14973 full-segment 上冻结 V2 已经 selected 的 121 条 route。从现有
`enhanced_tracklets.root` 的 `TrackletHit_*` 回取

`route → tracklet → cluster identifier → layer → module`。

Calypso/Acts 没有导出 unbiased cluster residual，也没有 cluster 全局坐标。
测量位置用 tracklet 在名义 layer `z` 上的截距；预测用已有的
leave-one-station-out 全局直线拟合投到该 layer。该量明确标为
**module-proxy unbiased residual**，不能包装成完整 Acts cluster residual。

## FD / Jacobian

只在 IFT station 0、layer 0 上取 2–4 个命中最多的 module，对 module 六轴和
layer antisymmetric `C_dx` 做 software FD。确认 payload 只作用到目标
module、`dx`/`C_dx` 符号正确、两档步长在线性区。Jacobian 只做 rank /
spectrum / Fisher / correlation / null-space，禁止求解。

## Go / No-Go

比较同一批 selected tracks 上 `station dx ↔ C_dx`、`station ry ↔ C_dx`
在 **station-recompressed**（压缩 tracklet 传到其他 station `z`）与
module-proxy residual 空间的 |cosine|，并检查
`{station dx, station ry, C_dx}` leakage 子空间秩。

- 两组 cosine 都显著下降且子空间可分辨 →
  `module_level_observable_restores_identifiability_candidate`，允许进入
  full module-level identifiability map，仍不训练新网络。
- 仅 `dx↔C_dx` 下降、`ry↔C_dx` 仍共线 → measurement compression 解释了
  translational degeneracy，但 `ry/C_dx` 仍受三平面近直轨 topology 限制；
  本阶段总体 No-Go。
- 两组仍共线、leakage rank `< 2`，或更严重 null-space →
  `track_topology_limited`，停止向更复杂 ML 下沉。

唯一问题：module-level observable 是否真正解除目前 station `dx/ry` 与
internal `C_dx` 的关键退化，从而值得进入 full-detector module-level
alignment basis study。

## 产出

配置：`configs/module_level_residual_identifiability_poc_v1.yaml`

代码：`alignment/module_level_residual_poc.py`，
`scripts/report_module_level_residual_identifiability_poc.py`，
`tests/test_module_level_residual_identifiability_poc.py`

报告：`outputs/module_level_residual_identifiability_poc_v1/` 下六份 JSON。
