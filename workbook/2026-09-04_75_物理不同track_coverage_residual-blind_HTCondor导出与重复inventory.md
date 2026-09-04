# 2026-09-04 Physically-Distinct Track-Coverage Residual-Blind HTCondor Export & Reinventory V1

## 任务

条目 74 冻结 `residual_blind_export_authorized_fd_not_opened`：
Stage 1 residual-blind coverage inventory 没有任何 candidate 获准
进入新的 rigid-station 5DoF FD campaign；唯一授权继续处理的是两个
metadata-distinct candidate：

1. `mc24_100120_muon_floor`（floor-origin muon gun，pid={-13,13}，
   x=[-100,100] mm，y=-120 mm，z=[-2000,1000] mm）
2. `mc24_100130_kshort_end_fasernu`（中性 kaon gun，pid=310，
   z=[-2000,-1951] mm）

它们只获得 **HTCondor residual-blind tracklet export** 授权，不是
FD、SVD、Jacobian 或 alignment 授权。本条目执行该授权并重复
coverage inventory，从而闭合“是否还有物理上不同且统计充分的
track coverage”这一最后未决问题。

严格继承条目 59–74 的全部冻结结论。禁止算法“救 rank”：不得修改
`rank_tolerance=0.01`，不得反调 `S`，不得删低-rank source，不得删
`rz` 或任何 DoF，不得换 residual，不得重定义 stable-core，不得重训
V2/V3/Transformer，不得改 Frozen-V2 pairwise/route policy，不得重堆
2024 r0022 collision-like population，不得使用 sealed test。
Survey/metrology 只作 external cross-check；真实数据仍为
`residual_dq_monitoring_only`；`geometry_write_allowed=false`、
`official_conditions_write_allowed=false`、
`real_data_correction_authorized=false`。

本阶段 **不构造** `A = W^{1/2} J S`，不做 SVD，不看 rank，不注入
alignment payload，不构造 physical FD point，不做 residual-based
selection。不处理/扩展 `100012`、`100049/050`、`100123/124`、FLUKA
210010、普通 r0022、unused 100043/044/047/048，不打开 sealed
`100116/117 00030_00039`。

## 冻结继承与 config

新 config：
`configs/physically_distinct_track_coverage_residual_blind_export_reinventory_v1.yaml`。
加载时逐条校验继承自 workbook-74 config（SHA256
`2a8b17a362fa38c4fb9ad16fdb379bf4c7f8046fb38f1bc1f345776e21936caf`）
的 candidate 定义、canonical envelope 定义、observed-distinctness
定义与全部 coverage gate。冻结门不变：`min_events=200`、
`min_ift_events=200`、`min_complete_four_station_events=80`、
`min_independent_sources_or_runs=2`，要求 IFT + stations 0/1/2/3；
observed distinctness 仍为
`outside_canonical_quantile_box_fraction >= 0.20` 或
`histogram_intersection <= 0.80`。不把 local `hypot(tx,ty)` 解释为
spectrometer wide-angle gate。

## Immutable input manifest（仅 EOS/xAOD metadata）

`input_manifest.json`（SHA256
`17ea99b5f56c1741f3facd1086f6bf92221812fdc80a0b61571a8a0eb2b24c85`）
记录每个输入文件的完整 EOS path、source/production ID、event 数、
generator/reconstruction provenance、geometry/reco tag、文件大小与
EOS adler32 checksum：

- `mc24_100120_muon_floor`：10 个 xAOD（`rec/s0013-r0020`），每文件
  100,000 events，共 1,000,000 events，约 9.51 GB；全部文件
  SegmentFit 存在、adler32 可取。
- `mc24_100130_kshort_end_fasernu`：10 个 xAOD（`rec/s0013-r0020`），
  每文件 10,000 events，共 100,000 events，约 25.21 GB；全部文件
  SegmentFit 存在、adler32 可取。
- 两者 reco tag `s0013-r0020`、geometry `FASERNU-04` / `TI12MC04`、
  conditions `OFLCOND-FASER-04`。

Workbook-74 导出门校验（`export_gate.json`）：两 candidate 均满足
（≥2 个 xAOD、每文件 ≥1000 events、SegmentFit 存在、IFT geometry
确认）。一个 xAOD 文件在既有 file-level production-slice schema 下是
一个 source（10 个独立 source/candidate）；不把单文件切成伪造
“独立 source”；两 candidate 绝不 pool。

## Exporter-contract audit（大规模提交前预注册）

`exporter_contract_audit.json`。代码级审计结论：tracklet exporter
（`NtupleDumperAlg::appendDetailedTracklet`，输入为 persisted
SegmentFit 经 GhostBusters 的 segment）、truth matching
（`TrackTruthMatchingTool::getTruthParticle`，SCT SDO majority
barcode，无 PDG 过滤）、SegmentFit population 定义与 MC event
selection 均为 particle-generic——无单 muon 假设、无 `PDG=13`
要求、无单一 truth parent 要求。唯一 muon-specific 元素是
workbook-74 reporter 的 angular mask `|pdg|=13`。

- `mc24_100120_muon_floor`：gun 粒子即 muon，angular population
  沿用 workbook-74 定义（`|pdg|=13`），不变。
- `mc24_100130_kshort_end_fasernu`：中性 kaon 不在 tracker 留径迹，
  tracker 可见的是带电衰变末态（`K0S → π+ π−`）。Angular
  population 在完整导出前预注册为 truth-matched 带电 pion
  `|pdg|=211`（truth-match 语义不变：match fraction ≥ 0.99、
  `truth_particle_id ≥ 0`、preferred station 一个代表 tracklet）。
  严格 `|pdg|=13` 变体仍作为带标签 cross-check 计算（中性 gun 下
  按构造为空；500-event smoke：0 条 muon-tagged tracklet），绝不
  作为 admission 输入。不改变 truth matching、particle selection
  或 route definition。

Smoke 验证（各 500 events，完整链）：100120 得 415 个含 tracklet
event、1248 条 tracklet、31 IFT events、13 complete-4-station
events；100130 得 231 个含 tracklet event、996 条 tracklet、19 IFT
events、0 complete-4-station。pion-masked angular population 的
`ty` 分布（[-0.348, 0.744]）远宽于 canonical envelope，提示物理
不同的 phase space。floor-origin muon 出现非致命
`FaserActsExtrapolation SurfaceError:1` 警告；100130 出现预期的
truth-matching unmatched 警告。两者均不修改链。

## Merged rec event-number 冲突与 physical-event 身份

Merged MC24 rec 文件复用 generator-job event number（100120：每文件
10 × 0–9999；100130：10 × 0–999），单 xAOD 内 `(run_id, event_id)`
不唯一。`datasets.root_loader.load_events` 的排序分组会把 10 个不同
physical event 错误合并。Reinventory 使用新
`load_events_physical_order`：文件序中每个极大连续相等
`(run_id, event_id)` 块是一个 physical event，与 exporter 逐 entry
顺序严格一致；块内重复 `tracklet_id` 或非有限 state 为硬
provenance failure。不使用 fuzzy join / 最近邻匹配。这是对
provenance 正确性的修复，不是对链的修改。

## HTCondor 导出

完整导出（每文件全部 events，`--nevents -1`）统一使用 HTCondor。
Worker：`scripts/run_residual_blind_export_condor.sh`；提交器：
`scripts/submit_residual_blind_export_condor.py`。每输入 xAOD 一个
作业（每 candidate 10 个，共 20 个；cluster 1108798 / 1108799，
flavour `tomorrow`，8 GB）。链为 nominal：persisted SegmentFit →
GhostBusters → NtupleDumperAlg detailed tracklets → canonical
converter → schema 检查 → content audit；不跑 chi-square baseline。
每作业写 `job_provenance.json`（输入 provenance、命令、
environment/Calypso revision、git SHA、各步 exit code、输出 ROOT
路径）；失败作业非零退出并由 report 显式分类，不静默跳过。

## Deterministic provenance validation（导出完成后）

每个导出 source 必须：`metadata/source_file` 与声明输入精确相等、
run id 等于 production DSID、station z 在冻结容差内映射到
IFT/0/1/2/3 z 表、enhanced-ntuple entry 数等于作业 processed-event
数、tracklet `(tx,ty)` 有限。负对照：wrong-source 必须检测为不匹配、
missing-output 必须分类为 `missing_output`、empty-SegmentFit 必须
分类为 `segmentfit_empty`。重复 inventory 必须给出一致统计。

## 重复 inventory 与逐 candidate verdict

用 workbook-74 reporter 与 canonical `(tx, ty)` envelope（重建并对
workbook-74 `canonical_coverage.json` 回归校验：n_events=1790、
n_angular=1695、quantile box tx=[-0.0560005, 0.0531091]、
ty=[-0.0191054, 0.0187434]）分别重算两 candidate 的 events、IFT
events、complete four-station events、station 0/1/2/3 coverage、
tracklet count、source/run count、`tx/ty` quantile/extrema、
histogram intersection、outside canonical box fraction、
slope/azimuth/quadrant。禁止读取 residual、Jacobian、singular
value、cosine、rank。

最终 verdict 逐 candidate 独立给出
（`admitted_for_separate_5dof_fd_preregistration` /
`insufficient_statistics` / `not_observed_phase_space_distinct` /
`segmentfit_empty` / `exporter_contract_incompatible` /
`provenance_failure` / `missing_output` / `export_job_failed`），
不合并 candidate 重新判定。Stage 75 自身不做 FD/SVD。

**（导出与 verdict 结果待 HTCondor 作业完成后回填。）**

## 条件分支（预注册）

- 若至少一个 candidate 通过全部冻结门：先冻结本条目、config SHA、
  export manifest/hash 与全部 coverage artifact，然后另开单独预注册
  的 `Physically-Distinct <candidate> Rigid-Station 5DoF FD
  Complementarity Feasibility V1` config 与 workbook 条目，继承
  `ift_dx_mm, ift_dy_mm, ift_rx_mrad, ift_ry_mrad, ift_rz_mrad`、
  `S=(5,5,60,60,60)`、`rank_tolerance=0.01` 与现有 WLS 定义；必须
  分别分析 canonical-only、candidate-only 与预注册
  joint-information 子空间，pooled rank=5 不算成功；长时 physical
  FD reconstruction 用 HTCondor。三臂、Frozen-V2
  unknown-association、真实数据 correction 在 identifiability gate
  通过前仍关闭。
- 若两 candidate 均不通过：冻结
  `current_track_coverage_insufficient_for_unconstrained_rigid_station_5dof`，
  不降低 200/200/80 门，不重开任何 7D / cluster-local /
  stable-core / rigid-5DoF threshold rescue；只有在此之后才允许预
  注册 `Gauge-Constrained / External-Constraint Alignment
  Feasibility` 新战役，且严格区分 reconstruction gauge 约定与真实
  survey/metrology 测量（缺 validated frame + measurement covariance
  + IOV 的 survey 数字不能冒充 physical prior）。

## 仓库状态

```text
branch: master
HEAD:   0c8a6ca Add physically-distinct track-coverage identifiability feasibility (74).
```

## 代码与 artifact

- `configs/physically_distinct_track_coverage_residual_blind_export_reinventory_v1.yaml`
- `alignment/physically_distinct_track_coverage_export.py`
- `scripts/run_residual_blind_export_condor.sh`
- `scripts/submit_residual_blind_export_condor.py`
- `scripts/report_residual_blind_export_reinventory.py`
- `tests/test_physically_distinct_track_coverage_export.py`
- `docs/physically_distinct_track_coverage_residual_blind_export_reinventory.md`
  / `_cn.md`
- `outputs/physically_distinct_track_coverage_residual_blind_export_reinventory_v1/`：
  `config.yaml`、`input_manifest.json`、`export_gate.json`、
  `exporter_contract_audit.json`、`condor_submit/`、
  `exports/<candidate>/<source_id>/`、`provenance_validation.json`、
  `reinventory.json`、`next_stage_decision.json`

测试：`tests/test_physically_distinct_track_coverage_export.py`（11
项）与既有 508 项全量回归全部通过。
