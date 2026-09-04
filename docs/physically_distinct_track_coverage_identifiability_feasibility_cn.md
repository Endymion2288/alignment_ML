# Physically-Distinct Track-Coverage Identifiability Feasibility V1

Workbook 74 / 2026-09-03。条目 59–73 继续冻结。本战役不拯救
7D tracklet-level observable、cluster-local observable、
cross-source stable core 或 rigid-station 5DoF。不改
`rank_tolerance=0.01`，不根据谱反调 `S`，不删失败 source，不删
`rz` 或其他 DoF，不更换 residual，不重定义 stable-core，不重新
训练 V2/V3/Transformer，不改 pairwise/route policy，也不增加更多
与现有 2024 r0022 collision-like population 同分布的统计量。

核心问题不是“更多 tracks 是否提高 rank”，而是：物理上不同的
轨迹角度、起源或拓扑 population 是否能提供当前 collision-like
与 canonical 5 mrad FLUKA-E tracks 缺失的独立 alignment
information，使 rigid-station 5DoF 在 source-disjoint 与
coverage-disjoint 条件下稳定可辨识。

第一阶段是完全 residual-blind 的 coverage inventory。候选只允许
根据预先存在的物理 metadata 定义：provenance、run type、
generator/process、detector configuration、track angular phase
space、station coverage。不得根据 residual、Jacobian singular
value、cosine、alignment response 或最终 rank 反向挑样本。
branch/run meaning 不确定时必须从 Calypso、ROOT metadata、
GRL/run documentation 和已有 reconstruction 输出确认，绝不能凭
文件名猜（`5mrad` 不是 phase-space proof）。本地 SegmentFit
`hypot(tx, ty)` 不是 spectrometer wide-angle 门槛。observed
distinctness 只按与 canonical `(tx, ty)` envelope 的 overlap
判定。统计门在看 coverage 数字前冻结；不满足的 candidate 直接
`insufficient`，不得降低门。

本阶段不构造 `A = W^{1/2} J S`，不做 SVD，不看 rank。只有
residual-blind inventory 证明某个 candidate 确实 metadata-distinct、
observed-`(tx, ty)`-distinct 且统计充分后，才允许另开对应的原生
5DoF FD campaign。混合 pooled rank 不是 complementarity。Gauge /
external-constraint 分支不得提前跳入，直到 inventory 严格证明
现有可用 reconstruction 没有统计充分、物理不同的 5DoF coverage
population。

不重新训练 V2/V3/Transformer，不改 Frozen V2，不 Newton，不写
geometry，不产生 alignment payload。真实数据仍是
`residual_dq_monitoring_only`。identifiability 通过前，三臂、
Frozen-V2 unknown-association 和真实数据 correction 保持关闭。

## 冻结合同

- 真实数据仍是 `residual_dq_monitoring_only`。
- `geometry_write_allowed=false`。
- 冻结 V2 checkpoint SHA256
  `0c85a001677678163512c59bd5ceb6d08bdefaab79c0f4628659a4a8ad766a27`。
- 继承条目 59
  `no_portable_alternative_topology_in_current_r0022`；条目 60
  `real_track_topology_insufficient_for_ry_cdx_separation`；条目 68
  `tracker_only_identifiable_basis_unstable_solve_stopped`；条目 69
  `cross_source_stable_core_independent_validation_fail`；条目 70
  `cluster_local_transfer_exact_join_restored_not_an_identifiability_campaign`；
  条目 71 `cluster_local_observable_not_cross_run_portable`；条目 72
  `tracklet_independent_failure_provenance_audit_complete_sources_not_dropped`；
  条目 73 `rigid_station_five_dof_not_source_or_coverage_portable`。
- Rigid-station 5DoF（本阶段不重建）：
  `ift_dx_mm, ift_dy_mm, ift_rx_mrad, ift_ry_mrad, ift_rz_mrad`；
  单位 `mm, mm, mrad, mrad, mrad`；`S = (5, 5, 60, 60, 60)`；
  `rank_tolerance = 0.01`；`A = W^{1/2} J S`。
- 若报告 slope，定义是锚点 source-station tracklet 局部状态的
  `hypot(source_tx, source_ty)`，不是 leftover residual
  `(rtx, rty)`，也不是 spectrometer `Δx/Δz`。它不是 admission 门槛。
- Observed distinctness：`outside_canonical_quantile_box_fraction ≥
  0.20` **或** histogram intersection `≤ 0.80`。
- 统计门：`min_events=200`，`min_ift_events=200`，
  `min_complete_four_station_events=80`，
  `min_independent_sources_or_runs=2`，必须有 IFT 与 stations 0–3。
- Collision-like metadata、与 canonical 同 production、sealed test、
  coverage unmeasured 均不得准入 FD。
- Unused 100043/044/047/048 同 production 文件不是新 population。
- 封存 `mc24_100116_00030_00039` 与 `mc24_100117_00030_00039`
  保持关闭。
- 本战役不是条目 36、59 或 60 的 rank 救援。

## 决策

`residual_blind_export_authorized_fd_not_opened`

| 门 | 冻结值 |
| --- | --- |
| residual-blind Stage 1 | **true** |
| SVD / rank 本阶段计算 | **false** |
| FD identifiability 本阶段打开 | **false** |
| 准入独立 FD campaign 的 population | **0** |
| 授权 residual-blind HTCondor export | **2**（`mc24_100120_muon_floor`，`mc24_100130_kshort_end_fasernu`） |
| 统计门被降低 | **false** |
| 本地 `hypot(tx,ty)` 当作 spectrometer wide-angle 门槛 | **false** |
| 按文件名 `5mrad` 猜 phase space | **false** |
| 重堆 2024 r0022 collision-like | **false** |
| 重堆 unused 5 mrad 文件 | **false** |
| 打开 sealed test | **false** |
| 冻结 `current_track_coverage_insufficient_for_unconstrained_rigid_station_5dof` | **false**（inventory 尚未关闭） |
| 打开 gauge / external-constraint 分支 | **false** |
| 三臂 / Frozen-V2 / 真实数据 correction | **未授权** |
| geometry write | **false** |

没有任何已导出 candidate 同时满足 metadata-distinct、observed
`(tx, ty)`-distinct、统计门达标、且不被 metadata 禁止。因此不得
打开后续原生 5DoF FD campaign。两个尚无 tracklets 的
metadata-distinct xAOD 只被授权做 residual-blind HTCondor export；
那不是 rank 救援，也不是 FD。Inventory 尚未证明现有 reconstruction
没有统计充分的物理不同 population，因此不得跳到 gauge /
external constraint。条目 61 对旧 `ry↔C_dx` 问题仍然冻结。

## Canonical envelope

Canonical 是 hierarchical V1 iteration-00 的 train/validation
FD-anchor tracklets，不是真实 2024 r0022。Generator logs 确认
SingleParticle、`z = -3990 mm`、`theta = None`、`phi = 0..2π`、
energy=`TH2Sampler`、FASERNU-04 / TI12MC04。

Pooled residual-blind truth-matched SegmentFit：

| 量 | 值 |
| --- | ---: |
| sources | 18 |
| events | 1790 |
| IFT events | 1615 |
| complete four-station events | 1433 |
| tracklets | 6690 |
| angular tracklets | 1695 |
| station events 0/1/2/3 | 1615 / 1705 / 1696 / 1674 |
| quantile box `tx` | `[-0.05600, 0.05311]` |
| quantile box `ty` | `[-0.01911, 0.01874]` |
| observed extrema `tx` | `[-0.19049, 0.16983]` |
| observed extrema `ty` | `[-0.09908, 0.16064]` |
| self histogram intersection | 1.0 |

## 候选 admission

配置
`configs/physically_distinct_track_coverage_identifiability_feasibility_v1.yaml`，
SHA256
`2a8b17a362fa38c4fb9ad16fdb379bf4c7f8046fb38f1bc1f345776e21936caf`。
schema
`faser-physically-distinct-track-coverage-identifiability-feasibility-v1`。
created_utc `2026-09-03T19:12:10.950820+00:00`。记录 HEAD
`a1fd01bb5c2c5daf432c13db3b021feb42bdf6d1`。

| id | hypothesis | observed distinct | events / IFT / complete4 / sources | hist ∩ | outside box | verdict |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `mc24_100012_100gev_gaussian_theta` | true | true（∩=0.764） | 20 / 18 / 17 / 4 | 0.764 | 0.050 | **insufficient**（统计不足；不降门） |
| `mc24_100116_117_2d_fluka_nonsealed` | true | **false**（∩=0.860） | 78 / 75 / 58 / 8 | 0.860 | 0.107 | **insufficient**（重叠 + 统计不足） |
| `mc24_100049_050_calonu_5mrad` | **false** | true（∩=0.674，outside=0） | 5 / 5 / 5 / 1 | 0.674 | 0.0 | **insufficient**（hypothesis false；CaloNu 未授权 export） |
| `mc24_100120_muon_floor` | true | unmeasured | 0 / 0 / 0 / 0 | — | — | **authorize residual-blind export**（不是 FD） |
| `mc24_100123_124_year_labeled_muon` | **false** | unmeasured | 0 / 0 / 0 / 0 | — | — | **insufficient**（year tag 不是 phase-space proof；不授权 export） |
| `mc24_100130_kshort_end_fasernu` | true | unmeasured | 0 / 0 / 0 / 0 | — | — | **authorize residual-blind export**（不是 FD） |
| `mc24_fluka_210010_full_shower` | true | unmeasured | 0 / 0 / 0 / 1 | — | — | **insufficient**（已知空 SegmentFit；禁止 export） |
| `real_2024_cosmic_beam_mode` | true | inherit 60 | — | — | — | **insufficient**（occupancy-empty；不重堆） |
| `real_2023_cosmic_like` | true | inherit 60 | — | — | — | **insufficient**（不是高 lever-arm；不重堆） |
| `real_backward_alps_ift_collision` | **false** | inherit 60 | — | — | — | **insufficient**（仍 collision-like） |
| `real_testbeam_2021` | true | inherit 60 | — | — | — | **insufficient**（无 IFT） |
| `real_2024_r0022_wide_angle_selected_routes` | **false** | inherit 59 | complete4 selected routes = 17 | — | — | **insufficient**（`do_not_restack`） |

100012 是最接近的已导出 population：ConstSampler[100 GeV] +
GaussianSampler theta，histogram intersection 0.764 ≤ 0.80，因此
observed-distinct。但 20/18/17 远低于 200/200/80。wide local slope
fraction 0.70 **不是**准入理由。不得降低门。

100116/117 非封存 mag_0 train/val 的 hypothesis 是 2D FLUKA sampler，
但观测 `(tx, ty)` 仍与 canonical envelope 重叠（∩=0.860）。封存
`00030_00039` 从未打开。

100049 `--geom TI12MCCaloNu` 改变的是 detector configuration
hypothesis，预注册即为 `physically_distinct_hypothesis: false`。
文件名 `5mrad` 不是更宽 phase space 的证据。剩余 rec 文件存在，
但 hypothesis=false，不授权 export。

100120 起源 `x=[-100,100], y=-120, z=[-2000,1000] mm`，相对
IP-like `z=-3990 mm` 是 metadata-distinct floor/edge topology。
100130 是 `pid=310`、`z=[-2000,-1951] mm` 的中性 kaon gun。两者
现有 reconstruction 无 tracklets，coverage unmeasured，因此不能
准入 FD，但满足 residual-blind export 合同。

100123/124 的 year 标签不是 phase-space proof；generator 仍是
`z=-3990 mm`、`theta=None`、TH2Sampler，hypothesis=false，不授权
export。FLUKA 210010 在 10-event 和 100-event smoke 中
SegmentFit/Segments 为空（`known_empty_segmentfit: true`），禁止
export。

## 命令

```bash
source scripts/setup_environment.sh ml
pytest -q tests/test_physically_distinct_track_coverage.py
python scripts/report_physically_distinct_track_coverage.py
```

CUDA `True`（Tesla T4）。本战役测试 7 passed。没有 Condor。本阶段
没有构造 `A = W^{1/2} J S`。

报告：
`outputs/physically_distinct_track_coverage_identifiability_feasibility_v1/`。
记录 HEAD `a1fd01bb5c2c5daf432c13db3b021feb42bdf6d1`。

## 下一阶段

不得打开 FD。不得跳到 gauge / external constraint。不得降低门。
不得重堆 r0022 或 unused 5 mrad 文件。下一步允许的技术步骤：对
`mc24_100120_muon_floor` 和 `mc24_100130_kshort_end_fasernu` 做
HTCondor residual-blind tracklet export，然后重复 coverage
inventory。若重复 inventory 后仍无统计充分、物理不同的
population，再冻结
`current_track_coverage_insufficient_for_unconstrained_rigid_station_5dof`
并打开 Gauge-Constrained / External-Constraint Alignment
Feasibility。现在还不到那一步。
