# 2026-08-23 (57) True Cluster-Local Residual Feasibility V1

## 任务

Stage 1.5。不训练新网络，不扩展 full-detector module alignment，不写
geometry，不重开 Station / reduced Station / `C_dx` Mode。唯一问题：上一轮
module-proxy 里剩下的 `station ry ↔ C_dx` 共线，究竟是三平面近直轨
topology，还是因为 residual 仍是 nominal-layer-z tracklet intercept。

## 数据链

同一批 14973 full-segment frozen-V2 selected routes。当前 enhanced ntuple
没有 cluster local `u` / covariance / surface。最小导出：

`SCT_ClusterContainer` 中的 `Tracker::FaserSCT_Cluster` →
`localPosition(Trk::locX)`、`localCovariance`、`globalPosition`、
`SiDetectorElement` 的 center / phiAxis / etaAxis / normal / transform。

禁止把 nominal-layer-z tracklet intercept 当作最终 cluster residual。
Unbiased 预测：其它 station 的真实 cluster 全局点拟合直线，投到该
detector surface，`r_u = u_cluster - u_track^unbiased`。

## 判据

同一 IFT 代表区域重建 `J_station_dx`、`J_station_ry`、`J_Cdx`。
`dx/C_dx` 是上一轮复现检查。核心是 `|cos(ry, C_dx)|`：

- 显著下降并形成独立方向 →
  `true_cluster_local_observable_restores_remaining_identifiability`
- 仍 `> 0.9` 且两档 FD / subset 稳定 →
  `ry_cdx_track_topology_degeneracy`，停止为 identifiability 把 ML
  下沉到更复杂的 cluster-level architecture

## 结果

真实 `FaserSCT_Cluster` local `u` 与 `SiDetectorElement` surface 已从
`SCT_ClusterContainer` 导出：121/121 selected events，2473 个 selected
cluster identifier 全部 join 上，PRD 与 ClusterOnTrack 的 local `u` 完全一致，
surface 重建到 1e-14 mm。454 个 IFT unbiased `r_u` 全部来自其它 station 的
cluster 全局点投到真实 detector surface，没有再用 nominal-layer-z intercept。

| 观察空间 | `|cos(dx,C_dx)|` | `|cos(ry,C_dx)|` | leakage rank |
|---|---:|---:|---:|
| station-recompressed | 0.973 | 1.000 | 1 |
| module-proxy intercept | 0.022 | 0.995 | 2 |
| true cluster-local `r_u` | **0.021** | **0.531** | **3** |

两档 FD 线性、符号、C_dx 的 L1 locality 全部通过，`dx/C_dx` 复现了上一轮
分离。`ry/C_dx` 从 0.995 降到 0.531，不是 `>0.9` 的 topology lock；但
even/odd subset 为 0.421 vs 0.782，方向不稳定。因此既不能写成
`true_cluster_local_observable_restores_remaining_identifiability`，也不能写成
`ry_cdx_track_topology_degeneracy`。

正式标签：`true_cluster_local_ry_cdx_inconclusive`。下一步仍不训练新网络，
不进入 full module-level identifiability map，也不为 identifiability 再下沉
更复杂的 cluster-level architecture。

## 产出

`outputs/true_cluster_local_residual_feasibility_v1/` 下五份 JSON。
