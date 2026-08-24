# 2026-08-24 True Cluster-Local ry↔C_dx Stability & Transfer Audit V1

## 任务

条目 57 把 `|cos(ry,C_dx)|` 从 module-proxy `0.995` 降到 `0.531`，leakage
rank 从 2 升到 3，但 even/odd 为 `0.421/0.782`。本阶段只做 stability
characterization：同一套 true cluster-local `r_u`、surface、leave-one-station-out、
FD step 和 frozen V2 selected routes。不训练新网络，不进 full module map，
不写 geometry，不发明新的 cosine cut。

## 两个问题

1. 14973 的 `0.531` 在 event bootstrap / coverage-matched subsets 下是否集中稳定。
2. 独立 calibration run 14974 是否方向一致、量级相近。14975/14976 只读，
   不参与方法或阈值选择。

稳定且 14974 复现 →
`true_cluster_local_observable_restores_remaining_identifiability_candidate`。
持续漂移或重新接近共线 → `cluster_local_identifiability_not_transferable`。

## 冻结条件（未改）

- `real_data_operating_mode=residual_dq_monitoring_only`
- Frozen V2 SHA256 `0c85a001677678163512c59bd5ceb6d08bdefaab79c0f4628659a4a8ad766a27`
- `r_u = u_cluster - u_track^unbiased` on `SiDetectorElement`
- leave-one-station-out，用其它 station 的 cluster globals 投影到 detector surface
- FD：translations 10 µm，rotations 0.05 mrad，`C_dx` 10 µm
- 代表 station：IFT 0
- 14975/14976 不参与 threshold 或方法选择

## 输出

`alignment_ML/outputs/true_cluster_local_ry_cdx_stability_transfer_v1/`

- `bootstrap_stability_report.json`
- `coverage_dependence_report.json`
- `cross_run_cluster_local_jacobian_report.json`
- `leakage_subspace_transfer_report.json`
- `next_stage_decision.json`

Dump：14974/14975/14976 selected events 全部命中（109/109、156/156、63/63）。
14973 复用条目 57 dump。`reproduces_entry_57_ry_cdx=true`。

## 点估计（相同 Jacobian 构造，无调参）

| run | 角色 | IFT routes | n_obs | `|cos(dx,C_dx)|` | `|cos(ry,C_dx)|` | rank | σ3/σ1 |
|---|---|---:|---:|---:|---:|---:|---:|
| 14973 | reference | 82 | 454 | 0.021 | **0.531** | 3 | 0.0144 |
| 14974 | transfer | 74 | 403 | 0.013 | **0.571** | 3 | 0.0149 |
| 14975 | read-only | 88 | 507 | 0.021 | 0.752 | 3 | 0.0156 |
| 14976 | read-only | 44 | 250 | 0.017 | 0.876 | 3 | 0.0133 |

`dx` 与 `C_dx` 在所有 run 上保持近正交。`ry`/`C_dx` 的点估计在 14973/14974
方向一致、量级相近，但只读 14975/14976 向 module-proxy 共线回漂。

## 问题 1：14973 的 0.531 是否集中稳定？ **否**

Event bootstrap（400 次，按 event 重抽 Jacobian 行，不重跑 FD）：

| 量 | median | 68% | 95% | width 95 |
|---|---:|---|---|---:|
| `|cos(ry,C_dx)|` | 0.543 | [0.448, 0.697] | [0.387, 0.786] | **0.399** |
| `|cos(dx,C_dx)|` | 0.021 | [0.007, 0.039] | [0.001, 0.055] | 0.054 |
| rank | 3 | [3, 3] | [3, 3] | 0 |
| σ3/σ1 | 0.0146 | [0.0128, 0.0158] | [0.0115, 0.0160] | 0.0046 |

- 中位数收回点估计（0.543 vs 0.531）。
- rank 在 400 次里始终为 3：第三条 leakage 维不是一次抽样偶然。
- 95% 宽度 0.399 **略小于** 相对 module-proxy 的声称下降 0.464，因此未触发
  `sampling_fluctuation`（宽度 ≥ 声称下降）。
- 95% 上沿 0.786 **更靠近** module-proxy 0.995 而不是 0.531
  （`upper_95_closer_to_proxy_than_to_point=true`）。
- Random half-split（200 次）：半样本 `|cos(ry,C_dx)|` 95% 为 [0.418, 0.820]，
  两半差的中位数 0.232，最大 0.537。与条目 57 的 even/odd 0.421/0.782 同量级。

Coverage：结果**不是**少数 module family 单独撑起来的
（最大 leave-one-module `|Δcos|=0.124`，未达声称下降的一半），
但**强烈依赖 track slope**：

| 14973 slope tertile | n_routes | `|cos(ry,C_dx)|` | rank |
|---|---:|---:|---:|
| 浅（\|t\| ≲ 6.8e-4） | 27 | **0.986** | **2** |
| 中 | 27 | **0.978** | **2** |
| 陡（\|t\| ≳ 1.5e-3） | 28 | **0.348** | **3** |

全样本 `0.531` 是浅轨几乎共线与陡轨可分离的**混合**。单 module 子集
`only_s0_l*_e1_p3` 回到 `0.93–0.94`。Layer topology 以 `L0+L1+L2` 为主
（68/82），该拓扑单独给出 0.532，与全样本一致；缺层拓扑样本过小。

因此：bootstrap 中位数稳定、rank 稳定，但 `|cos(ry,C_dx)|` **不集中**，
会被 track-angle mix 重新拉向共线。`14973_bootstrap_concentrated=false`。

## 问题 2：14974 是否方向一致、量级相近？ **点估计是；覆盖匹配下不能当作可搬运 basis**

- 14974 点估计 0.571 落在 14973 bootstrap 95% 内；两 run 68% 区间重叠
  （14974：[0.503, 0.697]）。
- 14974 自身 bootstrap 中位数 0.582，95% [0.444, 0.789]，宽度仍有 0.345。
- 参数空间 weak `{ry,C_dx}` principal angle 仅 **0.077°**；第三条维在
  几何上跨 run 对齐，但列余弦仍随样本组成漂移。
- **Coverage-matched 重采样无法执行**：14974 填不满 14973 的 slope 直方图
  （需要 27/27/28，实际 26/24/22）。
- 14974 自己的 slope tertile 与 14973 **同构**：0.984 / 0.979（rank 2）
  vs 0.369（rank 3）。独立 run 复现的是“陡轨才分离、浅轨仍共线”，
  不是一个与 coverage 无关的稳定 cosine。
- 只读 14975/14976（未参与决策）点估计 0.752 / 0.876，朝共线回漂。

## 决策

**`cluster_local_identifiability_not_transferable`**

- 答案：**No** — 条目 57 的 `0.531` 不是可重复、可跨 run 的 alignment basis。
- 不升级到 `true_cluster_local_observable_restores_remaining_identifiability_candidate`。
- `go_to_full_module_identifiability_map=false`
- `next_allowed_step=keep_ml_frozen_no_cluster_architecture_descent`
- 不发明新 cosine cut，不写 geometry，不进 Station / C_dx Mode。

物理含义不变：true cluster-local `r_u` **确实**打开了 module-proxy 锁住的
第三条 leakage 维（全样本与 bootstrap 的 rank 均为 3）。当前真实 track
样本（~80 条 ≥3-station IFT 路线）的 angle/module coverage 仍不足以把
`|cos(ry,C_dx)|≈0.53` 变成可搬运的 identifiability 度量。
