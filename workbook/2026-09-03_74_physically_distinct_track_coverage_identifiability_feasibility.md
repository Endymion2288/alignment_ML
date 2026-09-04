# 2026-09-03 Physically-Distinct Track-Coverage Identifiability Feasibility V1

## 任务

条目 59 冻结 `no_portable_alternative_topology_in_current_r0022`。
条目 60 冻结 `real_track_topology_insufficient_for_ry_cdx_separation`。
条目 68 冻结 `tracker_only_identifiable_basis_unstable_solve_stopped`。
条目 69 冻结 `cross_source_stable_core_independent_validation_fail`。
条目 70 只修复 cluster-local exact join provenance。条目 71 冻结
`cluster_local_observable_not_cross_run_portable`。条目 72 把条目
69 的 rank 2/3/4 源分类为正常 physics / coverage loss。条目 73 冻结
`rigid_station_five_dof_not_source_or_coverage_portable`：固定内部
geometry 后，即使只拟合 station `dx, dy, rx, ry, rz`，18 个源中仍有
4 个原生 rank&lt;5，9 个源的 slope tertile 掉 rank。

**停止用算法或参数化救 rank。** 不得改 `rank_tolerance=0.01`，不得
根据谱反调 `S`，不得删低-rank source，不得再删 `rz` 或其他 DoF，
不得换 residual、重定义 stable-core、重新训练 V2/V3/Transformer、
调整 pairwise/route policy，也不得增加更多与现有 2024 r0022
collision-like population 同分布的统计量。不得重开 7D /
cluster-local / stable-core / rigid-station-5DoF 救援。

本阶段新开独立预注册战役 **Physically-Distinct Track-Coverage
Identifiability Feasibility V1**。核心问题不是“更多 tracks 是否提高
rank”，而是：物理上不同的轨迹角度、起源或拓扑 population 是否能
提供当前 collision-like / canonical 5 mrad FLUKA-E tracks 缺失的
独立 alignment information，使 rigid-station 5DoF 在 source-disjoint
与 coverage-disjoint 条件下稳定可辨识。

第一阶段只做 **完全 residual-blind 的 coverage inventory**。候选只
允许根据预先存在的物理 metadata 定义：data provenance、run type、
generator/process、detector configuration、track angular phase space、
station coverage。不得根据 residual、Jacobian singular value、cosine、
alignment response 或最终 rank 反向挑样本。branch/run meaning 不确定
时必须从 Calypso 代码、ROOT metadata、GRL/run documentation 和已有
reconstruction 输出确认，绝不能凭文件名猜（`5mrad` 不是 phase-space
proof）。本地 SegmentFit `hypot(tx, ty)` **不能**当作 spectrometer
wide-angle 门槛；observed distinctness 只按与 canonical `(tx, ty)`
envelope 的 overlap 判定。统计门在看 coverage 数字前冻结；不满足
的 candidate 直接 `insufficient`，不得降低 event/route gate。

本阶段 **不构造** `A = W^{1/2} J S`，不做 SVD，不看 rank。只有
residual-blind inventory 证明某个 candidate 确实提供显著不同且有
足够统计的 angular/topological support 后，才允许另开对应的
new-observable-independent physical FD identifiability campaign。
Gauge / external-constraint 分支也不得提前跳入：条目 61 对旧
`ry↔C_dx` 问题仍然冻结，但本战役必须先严格证明现有可用数据没有
统计充分、物理不同的 5DoF coverage population。

当前阶段 `three_arm_authorized=false`、
`frozen_v2_alignment_loop_authorized=false`、
`real_data_correction_authorized=false`、
`geometry_write_allowed=false`、
`official_conditions_write_allowed=false`。真实数据仍是
`residual_dq_monitoring_only`。封存 test 继续禁止访问。

成功标准只有两个：找到并独立验证一个真正提供新 alignment
information 的物理 track population，或者严格证明现有可用数据没有
这样的 population，从而有依据地转向 gauge / external constraint，
而不是继续修改算法。本阶段只完成前者的 coverage 前提，尚未关闭
后者。

## 仓库状态

```text
branch: master
HEAD:   a1fd01b Add survey/metrology IOV provenance and prior-interface feasibility chain (60-65).
ancestor check: a1fd01bb5c2c5daf432c13db3b021feb42bdf6d1 ⊂ HEAD
host:   lxplus909.cern.ch
CUDA:   True (Tesla T4)
```

未重跑 Athena。未提交 Condor。短 metadata inventory、ROOT audit 和
unit tests 在交互节点完成；live report 约 11 s。未访问封存 test
`mc24_100116_00030_00039` / `mc24_100117_00030_00039`。未重新训练
V2/V3/Transformer。未构造 5DoF Jacobian，未做 SVD。

## 答案

本阶段决策 `residual_blind_export_authorized_fd_not_opened`。

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
| 重堆 unused 5 mrad 100043/044/047/048 文件 | **false** |
| 打开 sealed test | **false** |
| 冻结 `current_track_coverage_insufficient_for_unconstrained_rigid_station_5dof` | **false**（inventory 尚未关闭） |
| 打开 gauge / external-constraint 分支 | **false** |
| 三臂 / Frozen-V2 / 真实数据 correction | **未授权** |
| geometry write | **false** |

没有任何已导出 candidate 同时满足：metadata hypothesis true、observed
`(tx, ty)` 与 canonical envelope 显著不同、统计门达标、且不被
collision-like / same-production / sealed / inherit-59-60 禁止。
因此 **不得**打开独立 5DoF FD campaign。两个 metadata-distinct、
尚未导出 tracklets 的 xAOD 只被授权做 residual-blind HTCondor
export，然后重复 coverage inventory；那不是 rank 救援，也不是 FD。

尚未证明“现有可用数据没有这样的 population”，因此不得跳到
gauge / external constraint。条目 61 对旧 `ry↔C_dx` 问题仍然冻结。

## 预注册合同

配置
`configs/physically_distinct_track_coverage_identifiability_feasibility_v1.yaml`，
SHA256
`2a8b17a362fa38c4fb9ad16fdb379bf4c7f8046fb38f1bc1f345776e21936caf`。
schema
`faser-physically-distinct-track-coverage-identifiability-feasibility-v1`。
created_utc `2026-09-03T19:12:10.950820+00:00`。记录 HEAD
`a1fd01bb5c2c5daf432c13db3b021feb42bdf6d1`。

允许的选择变量：data provenance、run type、generator/process、
detector configuration、canonical tracklet 的 `(tx, ty)` phase space、
station coverage。禁止：alignment residual、Jacobian singular value、
cosine、alignment response、final rank、把文件名 `5mrad` 当
phase-space proof。

统计门（看数字前冻结，不得降低）：

- `min_events = 200`
- `min_ift_events = 200`
- `min_complete_four_station_events = 80`
- `min_independent_sources_or_runs = 2`
- 必须有 IFT（station 0）以及 stations 1–3
- observed distinctness：`outside_canonical_quantile_box_fraction ≥ 0.20`
  **或** histogram intersection `≤ 0.80`
- YAML 保留 `min_wide_local_slope_fraction = 0.30`，但 **不作为
  admission 门槛**。本地 SegmentFit `hypot(tx, ty)` 不是 spectrometer
  wide-angle 定义。
- collision-like metadata、与 canonical 同 production、sealed test、
  coverage unmeasured 均不得准入 FD
- export 另有 `min_xaod_files_for_export = 2`、
  `min_xaod_events_per_file_for_export = 1000`、必须有 SegmentFit 和
  IFT geometry；export 仍 residual-blind，不降低 tracklet 统计门

若未来打开 FD，必须原样复用冻结 rigid-station 5DoF：参数
`ift_dx_mm, ift_dy_mm, ift_rx_mrad, ift_ry_mrad, ift_rz_mrad`，单位
`mm, mm, mrad, mrad, mrad`，`S = (5, 5, 60, 60, 60)`，
`rank_tolerance = 0.01`，`A = W^{1/2} J S`。混合 pooled rank 不是
complementarity。

## Canonical envelope

Canonical 是 hierarchical V1 iteration-00 的 18 个 train/validation
FD-anchor tracklets，不是真实 2024 r0022。Generator logs 确认
SingleParticle、`z = -3990 mm`、`theta = None`、`phi = 0..2π`、
energy=`TH2Sampler`、FASERNU-04 / TI12MC04。Unused 100043/044/047/048
文件是同一 5 mrad FLUKA-E production，不是新 population。

Pooled（residual-blind，truth-matched SegmentFit）：

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

100012 是最接近的已导出 population：generator 是 ConstSampler[100 GeV]
+ GaussianSampler theta，histogram intersection 0.764 ≤ 0.80，因此
observed-distinct。但 20/18/17 远低于 200/200/80。wide local slope
fraction 0.70 **不是**准入理由。不得降低门。

100116/117 非封存 mag_0 train/val 的 hypothesis 是 2D FLUKA sampler，
但观测 `(tx, ty)` 仍与 canonical envelope 重叠（∩=0.860）。封存
`00030_00039` 从未打开。

100049 `--geom TI12MCCaloNu` 改变的是 detector configuration
hypothesis，预注册即为 `physically_distinct_hypothesis: false`。
文件名 `5mrad` 不是更宽 phase space 的证据。剩余 99/100 rec 文件
存在，但 hypothesis=false，不授权 export。

100120 起源 `x=[-100,100], y=-120, z=[-2000,1000] mm`，相对
IP-like `z=-3990 mm` 是 metadata-distinct floor/edge topology。
100130 是 `pid=310`、`z=[-2000,-1951] mm` 的中性 kaon gun。两者
现有 reconstruction 无 tracklets，coverage unmeasured，因此不能
准入 FD，但满足 residual-blind export 合同。

100123/124 的 year 标签不是 phase-space proof；generator 仍是
`z=-3990 mm`、`theta=None`、TH2Sampler，hypothesis=false，不授权
export。FLUKA 210010 在 10-event 和 100-event smoke 中
SegmentFit/Segments 为空，YAML `known_empty_segmentfit: true`，
禁止 export。

## 输入 / 命令

```bash
source scripts/setup_environment.sh ml
pytest -q tests/test_physically_distinct_track_coverage.py
python scripts/report_physically_distinct_track_coverage.py
```

CUDA `True`（Tesla T4）。本战役单元测试 7 passed。没有 Condor。
没有构造 `A = W^{1/2} J S`。

报告：
`outputs/physically_distinct_track_coverage_identifiability_feasibility_v1/`
（`selection_contract.json`、`canonical_coverage.json`、
`candidate_coverage.json`、`inherited_real_data.json`、
`eos_layout.json`、`admission.json`、`next_stage_decision.json`、
复制的 `config.yaml`）。

实现：`alignment/physically_distinct_track_coverage.py`。复用条目
60 的 `inventory_eos_layout` / `parse_job_log` / `probe_xaod`。
未测量 coverage 的 xAOD 只做 metadata probe，不做 residual/SVD。

## 未解除 / 明确不做

- 不得把 100012 的 observed-distinct 写成“已准入 FD”。统计门未过。
- 不得把 100012 的 wide local slope 0.70 当成 spectrometer
  wide-angle 新 population。
- 不得降低 200/200/80 门来纳入 100012 或 100116。
- 不得打开 100116/117 sealed `00030_00039`。
- 不得把 CaloNu / year-labeled muon 当新 angular topology。
- 不得重堆 2024 r0022 collision-like 或 unused 5 mrad 同 production
  文件。
- 不得在本阶段构造 5DoF Jacobian、SVD 或 mixed pooled rank。
- 不得跳到 gauge / external constraint；inventory 尚未关闭。
- 不得把 population spread 或 existing conditions 冒充 prior。
- 不得打开三臂、Frozen-V2 unknown-association、真实数据
  correction 或 geometry write。
- 不得重开 7D / cluster-local / stable-core / rigid-station-5DoF
  救援。

## 冻结结论

- Frozen V2 / association policy 不变。
- 真实数据继续 `residual_dq_monitoring_only`。
- `geometry_write_allowed=false`。
- Stage 1 决策：`residual_blind_export_authorized_fd_not_opened`。
- 下一步允许的技术步骤：对 `mc24_100120_muon_floor` 和
  `mc24_100130_kshort_end_fasernu` 做 HTCondor residual-blind
  tracklet export，然后重复 coverage inventory。那不是 FD，也不是
  rank 救援。
- 若重复 inventory 后仍无统计充分、物理不同的 population，再冻结
  `current_track_coverage_insufficient_for_unconstrained_rigid_station_5dof`
  并另开 Gauge-Constrained / External-Constraint Alignment
  Feasibility。现在还不到那一步。
