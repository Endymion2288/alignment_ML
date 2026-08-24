# Module-Level Residual 与 Identifiability Proof-of-Concept V1

## 范围

本阶段不训练新关联模型，不改冻结 V2 的 checkpoint、route policy、threshold
或 unmatched penalty，也不重开 Station Mode、reduced Station Mode 或
`C_dx` Mode。不写官方 geometry，不求 alignment correction。

唯一问题：station 级 `dx/ry` 与内部 `C_dx` 的退化，是否主要来自 measurement
被压成 station tracklet。

## 输入

真实 run 14973 上冻结 V2 已选出的 route
（`data24_r14973_00007_skip49500_n84988`），以及已有 enhanced ntuple 的
`TrackletHit_*` 标识。

## Mapping 与 residual

每条 selected route 回溯到 SCT cluster / module：

```text
V2 selected route → station tracklet → SCT cluster → layer → module
```

当前导出没有 Calypso/Acts unbiased cluster residual，也没有 cluster 全局
坐标。因此用 tracklet 在名义 layer `z` 上的截距，再用
leave-one-station-out 直线拟合做预测。该量是 **module-proxy unbiased
residual**，不能包装成完整 Acts cluster residual。禁止把 in-sample
biased residual 当 alignment observable。任何 residual 宽度或下降只标
**DQ observable**。

## 有限差分烟测

一个代表 station（IFT，station 0）、一个 layer（layer 0）、2–4 个 module，
对 module `dx/dy/dz/rx/ry/rz` 和层反对称 `C_dx` 做 software 小扰动。检查
payload 是否只作用到目标 module、`dx`/`C_dx` 导数符号、两档步长是否在
线性区。这不是 Calypso conditions 写入。

## Identifiability 与泄漏

对 `J_module = ∂r_x/∂θ` 只做 rank、奇异谱、条件数、Fisher、参数相关和
弱/零空间分析，不求解 correction。

定量比较 **station-recompressed** 观测（压缩后的 tracklet 传播到其他
station 的 `z`，即 `cluster → tracklet` 压缩的 field-edge 等价物）与
module-proxy residual 空间中

```text
|cos(J_station_dx, J_C_dx)|
|cos(J_station_ry, J_C_dx)|
```

并检查 `{station dx, station ry, C_dx}` leakage 子空间的秩。

## Go / No-Go

唯一问题：

> module-level observable 是否真正解除目前 station `dx/ry` 与 internal
> `C_dx` 的关键退化，从而值得进入 full-detector module-level alignment
> basis study？

**Yes**（`module_level_observable_restores_identifiability_candidate`）：
两组 cosine 都显著下降，且 leakage 子空间可分辨（rank 3）。下一步才允许
full-detector module-level identifiability map，仍然不训练新网络。

**No，mixed**：只有 `dx↔C_dx` 下降。measurement compression 解释了
translational degeneracy，但 `ry/C_dx` 仍受三平面近直轨 topology 限制。
本阶段总体仍为 No-Go。

**No，`track_topology_limited`**：两组仍共线、leakage rank `< 2`，或
module Jacobian 出现更严重 null-space。停止向更复杂 ML 下沉，优先
external survey 或新的 track topology。

Implied `|C_dx|` 不是测量值。
