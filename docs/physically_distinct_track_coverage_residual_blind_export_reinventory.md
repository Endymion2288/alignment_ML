# Physically-Distinct Track-Coverage Residual-Blind HTCondor Export & Reinventory V1

Workbook 75 / 2026-09-04. Workbooks 59–74 remain frozen. This
campaign executes the single step authorized by workbook 74
(`residual_blind_export_authorized_fd_not_opened`): the residual-blind
HTCondor tracklet export of the two metadata-distinct candidates
`mc24_100120_muon_floor` and `mc24_100130_kshort_end_fasernu`,
followed by a repeated residual-blind coverage inventory against the
frozen workbook-74 gates and the canonical `(tx, ty)` envelope.

This stage does not construct `A = W^{1/2} J S`, does not SVD, does
not inspect rank, does not inject an alignment payload, does not
build a physical FD point, and does not select events from residuals,
Jacobian, singular values, cosine, or alignment response. No coverage
gate is modified. The export authorization is not an FD
authorization.

## Frozen inheritance

- Workbook-74 config SHA256
  `2a8b17a362fa38c4fb9ad16fdb379bf4c7f8046fb38f1bc1f345776e21936caf`;
  all candidate definitions, the canonical envelope definition, the
  observed-distinctness definition, and every coverage gate are
  inherited unchanged and verified at config load.
- Statistical gates: `min_events=200`, `min_ift_events=200`,
  `min_complete_four_station_events=80`,
  `min_independent_sources_or_runs=2`, IFT + stations 0–3 required.
- Observed distinctness: `outside_canonical_quantile_box_fraction ≥
  0.20` **or** histogram intersection `≤ 0.80`.
- Export gate: at least 2 suitable xAOD files, at least 1000 events
  per file, SegmentFit present, IFT-capable geometry.
- Rigid-station 5DoF contract (not reconstructed this stage):
  `ift_dx_mm, ift_dy_mm, ift_rx_mrad, ift_ry_mrad, ift_rz_mrad`;
  `S = (5, 5, 60, 60, 60)`; `rank_tolerance = 0.01`.
- Real data remains `residual_dq_monitoring_only`;
  `geometry_write_allowed=false`;
  `official_conditions_write_allowed=false`;
  `real_data_correction_authorized=false`. Sealed
  `100116/117 00030_00039` stays closed. Unused same-production
  100043/044/047/048 files are not a new population. 2024 r0022
  collision-like data is not restacked.

## Export chain

The validated nominal chain is used unchanged: persisted SegmentFit →
GhostBusters → `NtupleDumperAlg` detailed tracklets
(`faser_ntuple_maker.py --isMC --useIFT --export-tracklets
--nevents -1`) → canonical converter
(`scripts.convert_ntuple_tracklets --include-truth`) → schema check →
content audit. No alignment payload is injected, no
`--refit-segments`, no physical FD point, no residual-based
selection, no chi-square baseline. One HTCondor job per input xAOD
(10 jobs per candidate, 20 total). Every job writes
`job_provenance.json` with the input path, source id, exact command,
environment/Calypso revision, git SHA, per-step exit codes, and
output ROOT paths. Failed jobs exit nonzero and are explicitly
classified by the report step; they are never silently skipped.

## Immutable input manifest

`input_manifest.json` records, for every declared input file, the
full EOS path, file/production/source identifiers, xAOD event count,
generator and reconstruction provenance (generator log `Sim.Gun`
parameters, reco log geometry/conditions tags), geometry/reco tags,
file size, and the EOS-stored adler32 checksum. It is built from
EOS/xAOD metadata only. One xAOD file is one source under the
existing file-level production-slice schema; no file is split into
fake independent sources, and the two candidates are never pooled.

## Exporter-contract audit

Pre-registered before large-scale submission. The tracklet exporter
(`NtupleDumperAlg::appendDetailedTracklet`), the truth matching
(`TrackTruthMatchingTool::getTruthParticle`, majority barcode from
SCT SDO deposits, no PDG filter), the SegmentFit population
definition, and the MC event selection are all particle-generic: no
single-muon assumption, no `PDG=13` requirement, no single-truth-
parent requirement. The only muon-specific element is the
workbook-74 reporter's angular mask `|pdg|=13`.

- `mc24_100120_muon_floor`: muon gun `pid={-13,13}`; the angular
  population is the workbook-74 truth-matched muon population,
  unchanged.
- `mc24_100130_kshort_end_fasernu`: neutral-kaon gun `pid=310`; the
  gun particle leaves no tracker track. The tracker-visible generated
  process is the charged decay daughters (`K0S → π+ π−`). The angular
  population is pre-registered as truth-matched charged pions
  `|pdg|=211` with the unchanged workbook-74 truth-match semantics
  (match fraction ≥ 0.99, `truth_particle_id ≥ 0`, one representative
  tracklet at the preferred station). The strict workbook-74
  `|pdg|=13` variant is still computed as a labelled cross-check — it
  is empty by construction for a neutral gun (smoke: 0 muon-tagged
  tracklets in 500 events) — and is never an admission input. Truth
  matching, particle selection, and route definition are not modified
  post hoc.

## Physical-event identity in merged rec files

Merged MC24 rec files reuse generator-job event numbers (100120:
10 × 0–9999 per file; 100130: 10 × 0–999 per file), so
`(run_id, event_id)` is not unique inside one xAOD. The sorted
grouping of `datasets.root_loader.load_events` would merge distinct
physical events into one logical event. The reinventory therefore
loads exported tracklets with
`load_events_physical_order`: each maximal consecutive block of equal
`(run_id, event_id)` rows in exporter file order is one physical
event, exactly matching the exporter's per-entry order. Duplicate
`tracklet_id` inside a block and non-finite tracklet state are hard
provenance failures. No fuzzy join and no nearest-neighbour matching
is used anywhere.

## Deterministic provenance validation

Before any coverage number is computed, every exported source must
back-reference its exact declared input xAOD (`metadata/source_file`
string equality), the run id must equal the production DSID, station
z positions must map to the frozen IFT/0/1/2/3 z table within the
frozen tolerance, the enhanced-ntuple entry count must equal the
job's processed-event count, and tracklet `(tx, ty)` must be finite.
Explicit negative controls are retained: wrong-source validation must
be detected as a mismatch, a missing output must classify as
`missing_output`, and an empty export must classify as
`segmentfit_empty`. Repeating the inventory on the same outputs gives
identical statistics.

## Per-candidate verdicts

Each candidate is judged independently against the frozen workbook-74
gates with the workbook-74 reporter and the canonical `(tx, ty)`
envelope (rebuilt and regression-checked against the workbook-74
`canonical_coverage.json`). Verdict labels:
`admitted_for_separate_5dof_fd_preregistration`,
`insufficient_statistics`, `not_observed_phase_space_distinct`,
`segmentfit_empty`, `exporter_contract_incompatible`,
`provenance_failure`, `missing_output`, `export_job_failed`.
Candidates are never pooled for re-evaluation.

A candidate is `admitted_for_separate_5dof_fd_preregistration=true`
only if it is metadata-distinct, observed-`(tx, ty)`-distinct, above
all statistical gates, has at least 2 independent file-level sources,
covers IFT + stations 0–3, and has no metadata prohibition. Stage 75
itself does not open FD.

- If at least one candidate is admitted, workbook 75, the config SHA,
  the export manifest/hash, and all coverage artifacts are frozen
  first; only then may a separate pre-registered native 5DoF FD
  campaign config and workbook entry be opened, reusing the frozen
  rigid-station parameters, `S=(5,5,60,60,60)`,
  `rank_tolerance=0.01`, and the same WLS definition. Canonical-only,
  candidate-only, and pre-registered joint-information subspaces must
  be analyzed separately; pooled rank=5 is not complementarity.
- If neither candidate passes the frozen gates, this stage freezes
  `current_track_coverage_insufficient_for_unconstrained_rigid_station_5dof`;
  the 200/200/80 gates are not lowered, and no 7D / cluster-local /
  stable-core / rigid-5DoF threshold rescue is reopened. Only then
  may a `Gauge-Constrained / External-Constraint Alignment
  Feasibility` branch be pre-registered, keeping reconstruction gauge
  conventions strictly distinct from real survey/metrology
  measurements.

## Artifacts

`outputs/physically_distinct_track_coverage_residual_blind_export_reinventory_v1/`:

- `config.yaml` — frozen copy of the stage-75 config.
- `input_manifest.json` — immutable input manifest (EOS/xAOD
  metadata only) with its own SHA256.
- `export_gate.json` — frozen export-gate verification.
- `exporter_contract_audit.json` — pre-registered chain audit.
- `condor_submit/` — submit files, per-candidate job lists,
  submission record, and job logs.
- `exports/<candidate>/<source_id>/` — `enhanced_tracklets.root`,
  `tracklets.root`, `content_audit.json`, `job_provenance.json`.
- `provenance_validation.json` — deterministic validation plus
  negative controls.
- `reinventory.json` — per-candidate coverage, overlap with the
  canonical envelope, admissions, verdicts, and the canonical
  envelope regression check.
- `next_stage_decision.json` — the frozen stage-75 decision.
