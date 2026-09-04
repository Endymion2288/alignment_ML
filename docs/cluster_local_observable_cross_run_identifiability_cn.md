# Cluster-local observable 跨 run identifiability 与 transfer 验证 V1

Workbook 71 / 2026-09-03。条目 68 / 69 继续冻结 tracklet-level
失败。条目 70 只恢复了 cluster-local transfer exact join。条目
58 的 `cluster_local_identifiability_not_transferable` 继续冻结，
直到本战役在统一 exact-join 接线下降该失败分类。

核心问题只有一个：在严格 exact-join、相同 population、相同参数
定义和预注册 weighting / scaling 下，true cluster-local residual
是否比此前 tracklet-level `[rx, ry, rtx, rty]` 提供更稳定、跨
run 可搬运的 alignment information。

第一阶段只用 r14973 reference 和 r14974 transfer。不做
correction，不做三臂，不打开 Frozen-V2 alignment loop。

不重新训练 V2/V3/Transformer，不改冻结的 pairwise/route policy，
不继续堆 2024 r0022 collision-like tracks，不发明 cosine cut，不
根据 residual/cosine 调 selection，不运行 full-parameter Newton，
不写 geometry，不产生 alignment payload。Survey/metrology 仍只作
external cross-check。真实数据仍是
`residual_dq_monitoring_only`。禁止拯救 tracklet-level
observable。

## 冻结合同

- 真实数据仍是 `residual_dq_monitoring_only`。
- `geometry_write_allowed=false`。
- 冻结 V2 checkpoint SHA256
  `0c85a001677678163512c59bd5ceb6d08bdefaab79c0f4628659a4a8ad766a27`。
- 继承条目 58 / 68 / 69 / 70 的冻结判定。
- 官方 SVD 是 `A = W^{1/2} J S`，`J` 为 native mm/mrad，
  `S = (5, 60, 0.12)`，`rank_tolerance = 0.01`。禁止把裸混合
  单位 Jacobian 当作官方 rank。
- `parameter_columns` 的 `ry` 列按 radian；形成 `A` 前换成
  1/mrad。
- join key 保持
  `run + event + TrackletHit_cluster_identifier =
  FaserSCT_Cluster.identify().get_compact()`。
- residual 是 `r_u = u_cluster - u_track^unbiased`，在
  `SiDetectorElement` 上。Tracklet 在名义 layer z 的截距不是
  observable。
- run 是统计单位。不得先 pool measurement 再用 pooled rank
  代替 transferability。
- 若某 slope tertile 有至少 8 条 routes 且 rank 低于全样本，
  则多出来的维是 coverage-conditioned，不可搬运。
- 条目 58 裸 SVD / `|cos(ry, C_dx)|` 只作负控制。
- 条目 70 的 join 修复不自动推翻条目 58。
- 四门全部通过之前：不开三臂、不开 Frozen-V2 alignment
  loop、不对真实数据求 correction、不写 geometry、不扩展更多
  run。
- 若失败，冻结否定结论。不得调 scale、rank、selection 或
  population 追结果。

## 判定

`cluster_local_observable_not_cross_run_portable`

| 门 | 值 |
| --- | --- |
| `cluster_local_reference_jacobian_valid` | **true** |
| `cluster_local_transfer_jacobian_valid` | **true** |
| `cross_run_rank_consistent` | **true**（两端官方 rank 都是 3） |
| `cross_run_subspace_transferable` | **false**（slope tertile rank drop） |
| 条目 58 负控制 | 仍为 `cluster_local_identifiability_not_transferable` |
| 条目 70 wrong-dump 仍为 0 | **true** |
| Frozen-V2 unknown-association | **未授权** |
| 三臂 | **未授权** |
| 真实数据 alignment correction | **未授权** |
| geometry write | **false** |

Exact join 完整，两端 Jacobian 有效。全样本官方子空间几乎重合。
失败是预注册 coverage 门：slope tertile（≥8 routes）把官方 rank
从 3 降到 2。第三条维是 coverage-conditioned，不是可搬运
identifiable subspace。

## 官方谱

配置
`configs/cluster_local_observable_cross_run_identifiability_v1.yaml`，
SHA256
`dff4d132aed358dc37c0d76d95f67823de4844491147e3efec6bc598ff8d5577`。

| 量 | r14973 | r14974 |
| --- | ---: | ---: |
| exact joined keys | 2473 / 2473 | 2123 / 2123 |
| measurements | 454 | 403 |
| IFT routes | 82 | 74 |
| 官方 rank | 3 | 3 |
| 奇异值 | 100.85, 55.72, 1.667 | 97.30, 55.12, 1.604 |
| native \|cos(ry, C_dx)\| | 0.531 | 0.571 |
| slope tertile ranks | **2 / 2 / 3** | **2 / 2 / 3** |

全样本最大 principal angle `8.5e-7` deg；projector Frobenius
`2.5e-16`；min mode persistence 1.0。模式是 reconstruction-
observable 线性组合，不得改名为测得的 `ry` 或 `C_dx`。

同一 exact join 下条目 58 仍然失败：14973 的 `|cos(ry, C_dx)|`
bootstrap 不集中，coverage-matched 填不满 27 / 27 / 28（实际
26 / 24 / 22），判定仍为
`cluster_local_identifiability_not_transferable`。官方 rank 与裸
rank 都是 3，所以失败不是 Jacobian 构造 artifact。

## 命令

```bash
source scripts/setup_environment.sh ml
pytest -q tests/test_cluster_local_observable_cross_run.py
python scripts/report_cluster_local_observable_cross_run.py
```

CUDA `True`（Tesla T4）。相关测试 40 passed。没有 Condor。

报告：
`outputs/cluster_local_observable_cross_run_identifiability_v1/`。
记录 HEAD `a1fd01bb5c2c5daf432c13db3b021feb42bdf6d1`。

## 下一阶段

保持 residual DQ monitoring。不得在 tracker-only cluster-local
或 tracklet observable 上继续调阈值。并行的条目 69 失败源审计
（条目 72）只读，不是 confirmatory set。
