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

`input_manifest.json`（SHA256 见 sidecar `input_manifest.sha256`：
`d7048fadec1ce03b5eed9bf5388e42d2f1b67a0722835f2771807fe0cc384ab6`；
manifest 一旦写出即不可变，hash 不内嵌自身）
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

**失败分类与显式重试**：20/20 作业的 Calypso 导出、转换与 schema
检查全部成功（100120 每文件约 9 分钟/100k events），但 content
audit 步骤全部失败——`scripts/audit_tracklets.py` 使用排序分组
`load_events`，把 merged-rec 文件内复用 event number 的 10 个不同
physical event 错误合并并触发 duplicate tracklet_id 拒绝。分类为
`audit_loader_event_id_collision`（下游诊断工具与 merged-rec event
编号不兼容；导出与转换本身成功）。修复：`datasets/root_loader.py`
新增 `preserve_file_order=True` 分组模式（默认行为不变），
`audit_tracklets.py` 新增 `--physical-order`，Stage 75 loader 改为
委托该实现。随后用 `scripts/retry_residual_blind_export_audit.py`
对 20 个 source 仅重跑 audit 步骤（Calypso 导出不重跑），20/20 重试
成功，每 source 写 `content_audit_retry.json`，汇总于
`audit_retry_summary.json`。原始 `job_provenance.json` 保持不修改。

## Deterministic provenance validation（导出完成后）

每个导出 source 必须满足精确两跳反查（无 fuzzy join）：
`tracklets.root` 的 `metadata/source_file` 等于同目录
`enhanced_tracklets.root`（converter 记录的精确输入），且
`job_provenance.json` 的 `input_xaod` 等于 config 声明的输入 xAOD；
run id 等于 production DSID；station z 在冻结容差内映射到
IFT/0/1/2/3 z 表；enhanced-ntuple entry 数与 physical-event 数一致
（100120：100,000 entries/文件，约 84k 含 tracklet 的 physical
events；100130：10,000 entries/文件，约 4.7k）；tracklet `(tx,ty)`
有限。负对照全部通过：wrong-source 检测为不匹配、missing-output
分类为 `missing_output`、empty-SegmentFit 分类为
`segmentfit_empty`。**结果：20/20 source provenance valid，全部负
对照通过**（`provenance_validation.json`）。

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

最终 verdict 逐 candidate 独立给出，不合并重新判定。Stage 75 自身
不做 FD/SVD。Canonical envelope 重建与 workbook-74
`canonical_coverage.json` 逐位一致（回归通过）。

### 重复 inventory 结果（`reinventory.json`）

**`mc24_100120_muon_floor`** → **`not_observed_phase_space_distinct`**

- 840,967 physical events（10 独立 source，每 source 约 84k）、
  IFT events 43,751、complete-four-station 19,962、tracklets
  2,367,368、angular（|pdg|=13）818,571 —— 统计门全部远超冻结值。
- 但 observed distinctness 不满足：
  `outside_canonical_quantile_box_fraction = 0.0972 < 0.20` 且
  `histogram_intersection = 0.8167 > 0.80`。floor-origin muon 在
  station 0 的 `(tx,ty)` 与 canonical envelope 显著重叠
  （tx p01/p99 = ∓0.0635/±0.0635 vs canonical ∓0.0560/±0.0531；
  ty p01/p99 = -0.0152/+0.0237 vs canonical ∓0.0191/±0.0187）。
- metadata 假设不同，但观测到的 tracklet `(tx,ty)` 仍与 canonical
  envelope 重叠 → 按冻结定义判定
  `not_observed_phase_space_distinct`。不降低门、不改定义。

**`mc24_100130_kshort_end_fasernu`** →
**`admitted_for_separate_5dof_fd_preregistration`**

- 47,075 physical events（10 独立 source，每 source 约 4.7k）、IFT
  events 4,451、complete-four-station 278（≥80）、tracklets
  206,654、angular（预注册 |pdg|=211 pion daughters）33,120。
- Observed distinctness 满足：
  `outside_canonical_quantile_box_fraction = 0.4986 ≥ 0.20` 且
  `histogram_intersection = 0.6894 ≤ 0.80`。pion daughter 的 `ty`
  分布（p01=-0.241、p99=+0.260、extrema [-1.48, +3.36]）远宽于
  canonical（±0.019），是 canonical 缺失的物理不同 phase space。
- 严格 `|pdg|=13` cross-check（非 admission 输入）：256 条 angular
  tracklet（全统计下少量次级 muon），与预注册一致。
- 全部冻结门满足：metadata-distinct、observed-(tx,ty)-distinct、
  47,075 ≥ 200 events、4,451 ≥ 200 IFT events、278 ≥ 80
  complete-four-station、10 ≥ 2 独立 file-level source、IFT 与
  stations 0–3 coverage 齐全、无 metadata prohibition。

### Stage 75 冻结决策

`next_stage_decision.json`：
**`candidate_admitted_for_separate_5dof_fd_preregistration`**
（admitted: `mc24_100130_kshort_end_fasernu`）。Stage 75 不打开 FD；
`fd_identifiability_executed=false`、`svd_or_rank_computed=false`。
`current_track_coverage_insufficient_for_unconstrained_rigid_station_5dof`
**不**冻结（存在一个通过全部冻结门的 candidate）。

## 条件分支（按冻结 gate 落入第一分支）

两 candidate 中 `mc24_100130_kshort_end_fasernu` 通过全部冻结门，
因此落入第一分支：

- **本条目、config SHA、export manifest/hash 与全部 coverage
  artifact 在本条目提交时冻结**（artifact SHA256 见下节）。
- 下一步允许（且仅允许）另开单独预注册的连续新战役
  `Physically-Distinct mc24_100130_kshort_end_fasernu Rigid-Station
  5DoF FD Complementarity Feasibility V1`（新 config + 新 workbook
  条目），继承 `ift_dx_mm, ift_dy_mm, ift_rx_mrad, ift_ry_mrad,
  ift_rz_mrad`、`S=(5,5,60,60,60)`、`rank_tolerance=0.01` 与现有
  WLS 定义；必须分别分析 canonical-only、candidate-only 与预注册
  joint-information 的 singular/information 子空间，验证 candidate
  是否真正补充 canonical 缺失方向，并继续做 source/coverage
  portability；不能只 pool 后看 rank=5 就宣布成功；长时 physical
  FD reconstruction 必须用 HTCondor。三臂、Frozen-V2
  unknown-association、真实数据 correction 在该 identifiability
  gate 通过前仍关闭。**该 FD 战役的预注册不属于本条目范围。**
- 第二分支（冻结
  `current_track_coverage_insufficient_for_unconstrained_rigid_station_5dof`
  并转向 gauge/external constraint）本次不触发；若后续 FD
  complementarity 战役失败，gauge 分支的预注册条件不变。

## 冻结 artifact SHA256

```text
config.yaml                 52bb9c36eeb2c92b178f0319dab90bb96a3ce67d30b55ff29a08c6cedfbf76ff
input_manifest.json         d7048fadec1ce03b5eed9bf5388e42d2f1b67a0722835f2771807fe0cc384ab6
export_gate.json            91eb18317617996e93f5f4c0a3f7fac93bd24831be9107c70b3e32f0f42a8ddb
exporter_contract_audit.json 64beca9a74a13f4e6a14b96227f83b8702958601931c39517ffd3cef76a04a84
provenance_validation.json  a4f2baa454292c6840d5c3ebf3681f518e35b854e17c656a8a048d38a80933b7
reinventory.json            75c7a649df01197059edc03b2d44996b60e0b5161327b73084df380742f2b50d
next_stage_decision.json    9a477d1f25821609fd9ef8738d9f9bdc724eb840a3c57188bff98cc260790176
audit_retry_summary.json    e605cc4ce01dfe9108db94b6bc51208a4e33d37a64524b99e302195f25f37db7
```

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
- `scripts/retry_residual_blind_export_audit.py`
- `datasets/root_loader.py`（新增 `preserve_file_order` 分组模式，
  默认行为不变）
- `scripts/audit_tracklets.py`（新增 `--physical-order`）
- `tests/test_physically_distinct_track_coverage_export.py`
- `docs/physically_distinct_track_coverage_residual_blind_export_reinventory.md`
  / `_cn.md`
- `outputs/physically_distinct_track_coverage_residual_blind_export_reinventory_v1/`：
  `config.yaml`、`input_manifest.json` + `input_manifest.sha256`、
  `export_gate.json`、`exporter_contract_audit.json`、
  `condor_submit/`、`exports/<candidate>/<source_id>/`、
  `audit_retry_summary.json`、`provenance_validation.json`、
  `reinventory.json`、`next_stage_decision.json`

测试：`tests/test_physically_distinct_track_coverage_export.py`（11
项）与既有全量回归（508 项）全部通过。
