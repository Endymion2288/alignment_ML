# 2026-09-03 Cluster-local observable 跨 run identifiability 与 transfer 验证 V1

## 任务

条目 68 冻结 `tracker_only_identifiable_basis_unstable_solve_stopped`。
条目 69 冻结 `cross_source_stable_core_independent_validation_fail`
（独立确认 8/11 = 0.727 < 0.80）。**禁止继续拯救 tracklet-level
observable**：不得改 `rank_tolerance=0.01`，不得反调 `S`，不得删
失败 source，不得改 topology / selection / route / Frozen V2，也
不得重新定义 tracklet stable-core criterion。

条目 70 恢复了 cluster-local transfer **exact join**（r14974
routes × r14973 dump 是 run-id 错配）。那只是 provenance。条目
58 的 `cluster_local_identifiability_not_transferable` 继续冻结，
不能因为 join 修好了就宣布 cluster-local 可搬运。

本阶段新开独立预注册战役 **Cluster-Local Observable Cross-Run
Identifiability & Transfer Validation V1**。核心问题只有一个：在
严格 exact-join、相同 population、相同参数定义和预注册
weighting / scaling 下，true cluster-local residual 是否比此前
tracklet-level `[rx, ry, rtx, rty]` 提供更稳定、跨 run 可搬运的
alignment information。

第一阶段只允许 r14973 reference 和 r14974 transfer。不做
correction，不做三臂，不打开 Frozen-V2 alignment loop。

合同：

1. 官方 rank 必须来自 `A = W^{1/2} J S`。`J` 的 `ry` 列先从
   1/rad 换成 1/mrad。禁止对混合单位裸 Jacobian 做官方 SVD。
2. 条目 58 的裸 SVD / `|cos(ry,C_dx)|` 只作负控制复现，不是
   官方门。
3. 四个预注册门全部通过之前：`three_arm_authorized=false`、
   `frozen_v2_alignment_loop_authorized=false`、
   `real_data_correction_authorized=false`、
   `geometry_write_allowed=false`。
4. 若官方门失败，冻结否定结论。不得调阈值、重选 population
   或改 observable 追结果。
5. 条目 69 失败 source 不是本战役 confirmatory 样本。

不训练、不改 pairwise/route、不增加 2024 r0022 collision-like
统计、不 Newton、不写 geometry / alignment payload。真实数据
仍是 `residual_dq_monitoring_only`。

成功标准不是得到 alignment correction，而是回答：exact
provenance 下，cluster-local observable 是否能产生跨 run 可搬运
的 identifiable subspace；如果不能，失败来自真实 track /
topology 信息不足，还是可修复的数据 / 构造问题。

## 仓库状态

```text
branch: master
HEAD:   a1fd01b Add survey/metrology IOV provenance and prior-interface feasibility chain (60-65).
ancestor check: a1fd01bb5c2c5daf432c13db3b021feb42bdf6d1 ⊂ HEAD
host:   lxplus909.cern.ch
CUDA:   True (Tesla T4)
```

Population 仍是条目 57–58/70 的 Frozen-V2 selected-route 集合，
未重选。Reconstruction 仍是 2024 r0022。未提交 Condor：短
exact-join、software FD Jacobian、SVD、event bootstrap 在交互
节点完成。未访问封存 test。未重新训练 V2/V3/Transformer。

## 答案

冻结 `cluster_local_observable_not_cross_run_portable`。

| 门 | 冻结值 |
| --- | --- |
| `cluster_local_reference_jacobian_valid` | **true** |
| `cluster_local_transfer_jacobian_valid` | **true** |
| `cross_run_rank_consistent` | **true**（两端官方 rank 都是 3） |
| `cross_run_subspace_transferable` | **false**（slope tertile rank drop） |
| 条目 58 负控制复现 | **仍为** `cluster_local_identifiability_not_transferable` |
| 条目 70 wrong-dump 仍为 0 | **true** |
| Frozen-V2 unknown-association | **未授权** |
| 三臂 | **未授权** |
| 真实数据 alignment correction | **未授权** |
| geometry write | **false** |
| 扩展更多 run | **false** |

Exact join 完整，Jacobian validity 通过，全样本官方子空间几乎
重合。失败不是 join wiring，也不是裸 SVD 与 `A = W^{1/2} J S`
的构造差。失败是预注册 coverage 门：slope tertile（每档 ≥8
routes）把官方 rank 从 3 降到 2。第三条维是 coverage-conditioned
extra dimension，不是跨 run 可搬运的 identifiable subspace。

按合同停止。不得在 tracker-only cluster-local observable 上继续
调阈值追结果。

## 预注册合同（先冻结，后看 SVD）

配置
`configs/cluster_local_observable_cross_run_identifiability_v1.yaml`，
SHA256
`dff4d132aed358dc37c0d76d95f67823de4844491147e3efec6bc598ff8d5577`。

继承：

- Frozen V2 SHA256
  `0c85a001677678163512c59bd5ceb6d08bdefaab79c0f4628659a4a8ad766a27`
- `rank_tolerance = 0.01`
- cluster-local `S = (5, 60, 0.12)` mm / mrad，对应
  `(station_dx_mm, station_ry_mrad, C_dx_mm)`
- join key
  `run + event + TrackletHit_cluster_identifier =
  FaserSCT_Cluster.identify().get_compact()`
- residual `r_u = u_cluster - u_track^unbiased` on
  `SiDetectorElement` / `Trk::locX`
- leave-one-station-out，用其它 station 的 cluster globals
  投影到 detector surface
- software FD：translations 10 µm，rotations 0.05 mrad，
  `C_dx` 10 µm；代表 IFT station 0
- `W = 1 / var(u)`，对角

四个官方门：

- Jacobian validity：exact join 完整、≥50 measurements / ≥8
  routes、残差与 Jacobian 有限、`+/-` software probe 存在、
  FD smoke 通过、无相对范数 ≤ 1e-6 的近零列
- 两端官方 identifiable rank 相同
- 最大 principal angle ≤ 15 deg，projector Frobenius ≤ 1.0，
  min mode persistence ≥ 0.85，弱平面角 ≤ 15 deg
- 若某 slope tertile 有 ≥8 routes 且 rank 低于全样本，则判
  coverage-conditioned，子空间不可搬运

## Jacobian validity

| 量 | r14973 | r14974 |
| --- | ---: | ---: |
| exact joined keys | 2473 / 2473 | 2123 / 2123 |
| measurements | 454 | 403 |
| IFT routes | 82 | 74 |
| missing key rate | 0 | 0 |
| finite residual / Jacobian | true | true |
| FD smoke / 线性区间 | true | true |
| near-zero columns | 无 | 无 |
| native column norms | 0.436 / 0.021 / 0.356 | 0.411 / 0.019 / 0.340 |
| 判定 | **valid** | **valid** |

负控制：r14974 routes × r14973 dump 仍为 0，
`mismatch_reason = run_id_disjoint_dump_does_not_contain_route_run`。
条目 70 的 wiring 修复被复现，不是本战役的 identifiability
成功。

`parameter_columns` 的 `ry` 列按 radian 求导；官方 `J` 在形成
`A` 前把该列乘 `1e-3`，单位变成 1/mrad。这解释了 native
`station_ry` 列范数偏小，不是近零列。

## 官方 `A = W^{1/2} J S`

比较的是子空间，不是 signed vector 元素。模式是
reconstruction-observable 线性组合，**不得**改名为“测得
station `ry` / `C_dx`”。

| 量 | r14973 | r14974 |
| --- | ---: | ---: |
| 官方 rank | **3** | **3** |
| σ | 100.85, 55.72, **1.667** | 97.30, 55.12, **1.604** |
| σ3 / σ1 | 0.0165 | 0.0165 |
| normal-matrix condition | 3661 | 3681 |
| native \|cos(dx, C_dx)\| | 0.021 | 0.013 |
| native \|cos(ry, C_dx)\| | **0.531** | **0.571** |
| weighted \|cos(ry, C_dx)\| | 0.542 | 0.549 |

全样本跨 run：

- 最大 principal angle **8.5e-7 deg**
- projector Frobenius **2.5e-16**
- min mode persistence **1.0**
- 弱平面角 **2.61 deg**（低于 15 deg 门）

Event bootstrap（40 次，按 event 重抽，官方 `A`）：r14973 全部
rank 3；r14974 中位数 3，有一次掉到 2。全样本两端子空间对齐，
**不能**据此宣布可搬运。

## Slope tertile（失败门）

预注册：每档 ≥8 routes 时，rank 不得低于全样本。

| run | 浅 | 中 | 陡 |
| --- | ---: | ---: | ---: |
| r14973 routes | 27 | 27 | 28 |
| r14973 官方 rank | **2** | **2** | **3** |
| r14974 routes | 25 | 24 | 25 |
| r14974 官方 rank | **2** | **2** | **3** |

这与条目 58 的裸 SVD 覆盖结论相同：全样本 rank 3 / `|cos|≈0.53`
是浅轨几乎共线与陡轨可分离的**混合**。第三条维依赖 track-slope
coverage，不是跨 run 稳定的 portable mode。

## 条目 58 负控制（统一 exact join）

裸混合单位 SVD 复现：

- r14973 `|cos(ry, C_dx)| = 0.531`，rank 3
- r14974 `|cos(ry, C_dx)| = 0.571`，落在 r14973 bootstrap 95%
  内
- r14973 bootstrap **不集中**（95% 上沿更靠近 module-proxy
  0.995 而不是 0.531）
- coverage-matched 仍无法执行：14974 填不满 27 / 27 / 28，
  实际 26 / 24 / 22
- 判定仍为 `cluster_local_identifiability_not_transferable`

官方 rank 与裸 rank 都是 3，**不是** Jacobian 构造差异。条目
70 修好 join 后，旧失败在统一 exact-join implementation 下仍
复现。冻结为真正的 cross-run physics / coverage
non-transferability。旧条目 58 结果保留为负控制，不被覆盖。

## 失败原因分类

| 候选原因 | 本战役 |
| --- | --- |
| 旧 provenance / data wiring | wrong-dump 仍为 0，已修复路径不再解释失败 |
| observable / Jacobian 构造差异 | 官方 rank 与裸 rank 同为 3，否 |
| source / topology coverage | **是**：slope tertile rank 3→2 |
| 真实 physics / coverage 不可搬运 | **是**：统一 exact join 下旧失败复现 |

## 输入 / 命令

```bash
source scripts/setup_environment.sh ml
python -c "import torch; print(torch.cuda.is_available())"
pytest -q tests/test_cluster_local_observable_cross_run.py \
  tests/test_tracklet_independent_failure_provenance.py \
  tests/test_cluster_local_jacobian_transfer_repair.py \
  tests/test_cross_source_stable_core.py \
  tests/test_tracker_only_identifiable_subspace.py
python scripts/report_cluster_local_observable_cross_run.py
```

CUDA `True`（Tesla T4）。相关 40 passed。没有 Condor。

## 数值结果（artifact）

| 量 | 值 | 用途 |
| --- | ---: | --- |
| 官方 r14973 / r14974 rank | 3 / 3 | rank 一致 |
| 全样本 max principal angle | 8.5e-7 deg | 子空间对齐 |
| slope tertile min rank | **2** | **失败门** |
| 条目 58 复现 | not transferable | 负控制 |
| 三臂 / Frozen-V2 / correction | false | 合同 |
| geometry write | false | 合同 |

报告：`outputs/cluster_local_observable_cross_run_identifiability_v1/`。
记录 HEAD `a1fd01bb5c2c5daf432c13db3b021feb42bdf6d1`。

## 未解除 / 明确不做

- 不得把全样本 3 维官方子空间叫成“已确认可搬运的
  cluster-local identifiable basis”。
- 不得把 mode 0/1/2 叫成“测得 `dx` / `ry` / `C_dx`”。
- 不得为了让浅轨 tertile 升到 rank 3 而改 `rank_tolerance`、
  `S`、selection 或 population。
- 不得把条目 69 失败 source 当成本战役 confirmatory 样本。
- 不得打开三臂、Frozen-V2 alignment loop、真实数据
  correction，不得写 geometry。
- 若答案仍为否定，停止在 tracker-only observable 上调阈值。

## 冻结结论

- Frozen V2 / association policy 不变。
- 真实数据继续 `residual_dq_monitoring_only`。
- `geometry_write_allowed=false`；无 alignment payload。
- Survey 继续只做外部交叉检验。
- Exact provenance 下，cluster-local observable **不能**产生
  跨 run 可搬运的 identifiable subspace。失败来自真实
  track-slope / topology coverage，不是可修复的 join 或
  Jacobian 构造问题。
- 条目 58 否定结论在统一 exact join 下成立，予以保留。
- 下一步：停止继续在 tracker-only cluster-local / tracklet
  observable 上追阈值。并行的条目 69 失败源只做只读
  provenance audit（条目 72），不重开 stable-core。
