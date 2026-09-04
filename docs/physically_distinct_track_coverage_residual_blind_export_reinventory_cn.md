# 物理不同 Track-Coverage Residual-Blind HTCondor 导出与重复 Inventory V1

Workbook 75 / 2026-09-04。Workbook 59–74 保持冻结。本战役执行
workbook 74 唯一授权的步骤
（`residual_blind_export_authorized_fd_not_opened`）：对两个
metadata-distinct candidate `mc24_100120_muon_floor` 与
`mc24_100130_kshort_end_fasernu` 做 residual-blind HTCondor
tracklet 导出，随后按冻结的 workbook-74 门与 canonical `(tx, ty)`
envelope 重复 residual-blind coverage inventory。

本阶段不构造 `A = W^{1/2} J S`，不做 SVD，不查看 rank，不注入任何
alignment payload，不构造 physical FD point，不用 residual、
Jacobian、singular value、cosine 或 alignment response 选择事件。
任何 coverage gate 均不修改。导出授权不是 FD 授权。

## 冻结继承

- Workbook-74 config SHA256
  `2a8b17a362fa38c4fb9ad16fdb379bf4c7f8046fb38f1bc1f345776e21936caf`；
  所有 candidate 定义、canonical envelope 定义、
  observed-distinctness 定义与全部 coverage gate 原样继承，并在
  config 加载时逐条校验。
- 统计门：`min_events=200`、`min_ift_events=200`、
  `min_complete_four_station_events=80`、
  `min_independent_sources_or_runs=2`，要求 IFT + stations 0–3。
- Observed distinctness：
  `outside_canonical_quantile_box_fraction ≥ 0.20` **或**
  histogram intersection `≤ 0.80`。
- 导出门：至少 2 个满足要求的 xAOD 文件、每文件至少 1000 events、
  SegmentFit 存在、IFT-capable geometry。
- Rigid-station 5DoF 合同（本阶段不重建）：
  `ift_dx_mm, ift_dy_mm, ift_rx_mrad, ift_ry_mrad, ift_rz_mrad`；
  `S = (5, 5, 60, 60, 60)`；`rank_tolerance = 0.01`。
- 真实数据保持 `residual_dq_monitoring_only`；
  `geometry_write_allowed=false`；
  `official_conditions_write_allowed=false`；
  `real_data_correction_authorized=false`。Sealed
  `100116/117 00030_00039` 保持关闭。同 production 未使用的
  100043/044/047/048 文件不是新 population。不重堆 2024 r0022
  collision-like 数据。

## 导出链

完全复用已验证的 nominal 链：persisted SegmentFit → GhostBusters →
`NtupleDumperAlg` detailed tracklets
（`faser_ntuple_maker.py --isMC --useIFT --export-tracklets
--nevents -1`）→ canonical converter
（`scripts.convert_ntuple_tracklets --include-truth`）→ schema 检查
→ content audit。不注入 alignment payload，不用
`--refit-segments`，不构造 physical FD point，不做 residual-based
selection，不跑 chi-square baseline。每个输入 xAOD 一个 HTCondor
作业（每 candidate 10 个，共 20 个）。每个作业写
`job_provenance.json`，记录输入路径、source id、精确命令、
environment/Calypso revision、git SHA、各步骤 exit code 与输出
ROOT 路径。失败作业以非零状态退出，由 report 步骤显式分类，绝不
静默跳过。

## Immutable input manifest

`input_manifest.json` 对每个声明的输入文件记录：完整 EOS 路径、
文件/production/source 标识、xAOD event 数、generator 与
reconstruction provenance（generator log 的 `Sim.Gun` 参数、reco
log 的 geometry/conditions tag）、geometry/reco tag、文件大小与
EOS 存储的 adler32 checksum。manifest 仅由 EOS/xAOD metadata 构建。
一个 xAOD 文件在既有 file-level production-slice schema 下是一个
source；不把任何文件切成伪造的“独立 source”，两个 candidate 绝不
pool。

## Exporter-contract audit

在大规模提交前预注册。Tracklet exporter
（`NtupleDumperAlg::appendDetailedTracklet`）、truth matching
（`TrackTruthMatchingTool::getTruthParticle`，基于 SCT SDO deposit
的 majority barcode，无 PDG 过滤）、SegmentFit population 定义与
MC event selection 均为 particle-generic：无单 muon 假设、无
`PDG=13` 要求、无单一 truth parent 要求。唯一 muon-specific 的元素
是 workbook-74 reporter 的 angular mask `|pdg|=13`。

- `mc24_100120_muon_floor`：muon gun `pid={-13,13}`；angular
  population 即 workbook-74 truth-matched muon population，不变。
- `mc24_100130_kshort_end_fasernu`：中性 kaon gun `pid=310`；gun
  粒子不在 tracker 留径迹。tracker 可见的生成过程是带电衰变末态
  （`K0S → π+ π−`）。Angular population 预注册为 truth-matched 带电
  pion `|pdg|=211`，沿用不变的 workbook-74 truth-match 语义
  （match fraction ≥ 0.99、`truth_particle_id ≥ 0`、preferred
  station 一个代表 tracklet）。严格的 workbook-74 `|pdg|=13` 变体
  仍作为带标签的 cross-check 计算——对中性 gun 它按构造为空
  （smoke：500 events 中 0 条 muon-tagged tracklet）——且绝不作为
  admission 输入。Truth matching、particle selection 与 route
  definition 不做事后修改。

## Merged rec 文件中的 physical-event 身份

Merged MC24 rec 文件复用 generator-job event number（100120：每文件
10 × 0–9999；100130：10 × 0–999），因此单个 xAOD 内
`(run_id, event_id)` 不唯一。`datasets.root_loader.load_events` 的
排序分组会把不同 physical event 错误合并为一个逻辑 event。
Reinventory 因此使用 `load_events_physical_order` 加载导出
tracklet：文件序中每个极大连续相等 `(run_id, event_id)` 块是一个
physical event，与 exporter 的逐 entry 顺序严格一致。块内重复
`tracklet_id` 与非有限 tracklet state 均为硬 provenance failure。
任何环节不使用 fuzzy join 或最近邻匹配。

## Deterministic provenance validation

在计算任何 coverage 数字之前，每个导出 source 必须能精确反查到其
声明的输入 xAOD（`metadata/source_file` 字符串相等）、run id 必须
等于 production DSID、station z 必须在冻结容差内映射到冻结的
IFT/0/1/2/3 z 表、enhanced-ntuple entry 数必须等于作业的
processed-event 数、tracklet `(tx, ty)` 必须有限。显式保留负对照：
wrong-source 校验必须被检测为不匹配、缺失输出必须分类为
`missing_output`、空导出必须分类为 `segmentfit_empty`。对相同输出
重复 inventory 必须给出一致统计。

## 逐 candidate verdict

每个 candidate 用 workbook-74 reporter 与 canonical `(tx, ty)`
envelope（重建并对 workbook-74 `canonical_coverage.json` 做回归
校验）独立对照冻结门判定。Verdict 标签：
`admitted_for_separate_5dof_fd_preregistration`、
`insufficient_statistics`、`not_observed_phase_space_distinct`、
`segmentfit_empty`、`exporter_contract_incompatible`、
`provenance_failure`、`missing_output`、`export_job_failed`。
Candidate 绝不合并后重新判定。

只有同时满足 metadata-distinct、observed-`(tx,ty)`-distinct、全部
统计门、至少 2 个 file-level 独立 source、IFT + stations 0–3
coverage、且无 metadata prohibition 的 candidate 才能标记为
`admitted_for_separate_5dof_fd_preregistration=true`。Stage 75 自身
不打开 FD。

- 若至少一个 candidate 获准，先冻结 workbook 75、config SHA、
  export manifest/hash 与全部 coverage artifact，然后才可另开单独
  预注册的 native 5DoF FD campaign config 与 workbook 条目，并沿用
  冻结的 rigid-station 参数、`S=(5,5,60,60,60)`、
  `rank_tolerance=0.01` 与同一 WLS 定义。必须分别分析
  canonical-only、candidate-only 与预注册 joint-information 子空间；
  pooled rank=5 不是 complementarity。
- 若两个 candidate 都不能通过冻结门，本阶段冻结
  `current_track_coverage_insufficient_for_unconstrained_rigid_station_5dof`；
  不降低 200/200/80 门，也不重开任何 7D / cluster-local /
  stable-core / rigid-5DoF threshold rescue。只有在此之后才允许预
  注册 `Gauge-Constrained / External-Constraint Alignment
  Feasibility` 分支，且必须把 reconstruction gauge 约定与真实
  survey/metrology 测量严格区分。

## Artifacts

`outputs/physically_distinct_track_coverage_residual_blind_export_reinventory_v1/`：

- `config.yaml` — stage-75 config 的冻结副本。
- `input_manifest.json` — immutable input manifest（仅 EOS/xAOD
  metadata），带自身 SHA256。
- `export_gate.json` — 冻结导出门校验。
- `exporter_contract_audit.json` — 预注册链审计。
- `condor_submit/` — submit 文件、逐 candidate 作业清单、提交记录
  与作业日志。
- `exports/<candidate>/<source_id>/` — `enhanced_tracklets.root`、
  `tracklets.root`、`content_audit.json`、`job_provenance.json`。
- `provenance_validation.json` — deterministic 校验与负对照。
- `reinventory.json` — 逐 candidate coverage、与 canonical envelope
  的 overlap、admission、verdict 与 canonical envelope 回归校验。
- `next_stage_decision.json` — 冻结的 stage-75 决策。
