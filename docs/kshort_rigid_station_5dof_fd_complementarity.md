# Physically-Distinct K-short Rigid-Station 5DoF FD Complementarity Feasibility V1

Workbook 76 / 2026-09-05. Workbooks 59–75 remain frozen. This
document is the English pre-registration of the campaign; the Chinese
workbook entry is
`workbook/2026-09-05_76_K-short物理不同轨迹的刚体站5DoF_FD互补可辨识性.md`.
Every criterion below is frozen before the first full K-short FD
HTCondor submission. No criterion may be revised after inspecting any
K-short FD singular spectrum.

## Scientific question

Workbook 73 froze
`rigid_station_five_dof_not_source_or_coverage_portable`: the
canonical 18-source rigid-station 5DoF pooled rank can reach 5, but
source-disjoint and coverage-disjoint portability fail. Workbooks
74/75 completed the track-coverage search: the only candidate admitted
by the residual-blind coverage inventory is
`mc24_100130_kshort_end_fasernu` — ten file-level independent xAOD
sources whose charged-pion daughter population carries
physically-distinct angular/topological support.

The only question this campaign answers is whether that
physically-distinct support provides **stable, source-disjoint and
coverage-disjoint NEW alignment information** complementary to the
canonical population, resolving the workbook-73 portability failure.

The campaign is NOT "is K-short alone rank 5" and NOT "is pooled
canonical+K-short rank 5". `joint pooled rank == 5` alone is never a
success criterion. Three information systems are distinguished:

- **A. canonical-only** — reproduces the workbook-73 frozen conclusion
  (regression / negative control, not a retune).
- **B. K-short-candidate-only** — same-definition spectrum, rank,
  identifiable subspace, source-disjoint and coverage-disjoint
  portability for each of the 10 sources and pooled. K-short-only is
  NOT required to reach rank 5; its value may be a subset of
  directions canonical lacks.
- **C. joint complementarity** — the pre-registered canonical+K-short
  combination: subspace overlap / principal angles, information along
  the weakest canonical direction, joint source/coverage stability,
  leave-source-out and bootstrap stability, and the workbook-69
  stable-core independent validation with pre-registered roles
  (hypothesis = 18 canonical sources, independent = 10 K-short
  sources).

## Frozen parameterization (inherited, never retuned)

- Parameter vector `ift_dx_mm, ift_dy_mm, ift_rx_mrad, ift_ry_mrad,
  ift_rz_mrad` (mm, mm, mrad, mrad, mrad); `ift_dz_mm` and `C_dx`
  stay out of the tracker fit.
- `S = (5, 5, 60, 60, 60)`; `rank_tolerance = 0.01`;
  `A = W^{1/2} J S`; W = nominal WLS inverse 4x4 pair covariance.
- Physical central FD steps `(0.5, 0.5, 10, 10, 10)`, exactly as the
  hierarchical V1 iteration-00 bank (workbook 44) and workbook 73.
- Residual observable: the workbook-73 truth-selected mode-0 pair
  residual (`min_truth_match_fraction = 0.99`, `q_over_p_mode = 0`).
- Source-level rank definition, event bootstrap (40 replicates, 12
  half-splits, seed 20260905 per the frozen workbook-date convention),
  slope-tertile coverage (3 bins, min 8 events), complete-four-station
  truth-route topology, and all principal-angle / Frobenius thresholds
  (source/split/disjoint/topology 15 deg / 1.0; bootstrap 20 deg /
  1.2) are reused from the frozen workbook 68/69/73 implementations.
- Inheritance SHA256 of the workbook-73/74/75 configs, the workbook-75
  manifest and reinventory, and the canonical iteration manifest are
  verified at config load; any mismatch is a hard failure.

## K-short population contract

The workbook-75 `|pdg| = 211` mask defined only the residual-blind
angular population for admission (the neutral `K0S` leaves no charged
trajectory; the tracker-visible objects are the decay daughters). The
FD observable is NOT replaced by a pion-specific residual: the
workbook-73 rigid-station physical-FD observable and WLS contract are
inherited unchanged. Truth matching is majority-barcode and
particle-generic (workbook-75 exporter-contract audit: no PDG=13
enforcement, no single-muon-parent assumption); its applicability to
K-short decay daughters is frozen in the pre-registration
(`fd_observable_truth_selection: particle_generic_majority_barcode`,
`fd_observable_pdg_mask: none`). No pion/track/event selection may
change after FD results are seen.

The ten K-short sources stay file-level independent. The
train/validation split is pre-registered by file-name order (first
five train, last five validation) and is never revised. Sources are
never merged into one to hide a failure, and no source is dropped
after seeing the singular spectrum.

## Merged-rec physical-event identity (hard contract)

Workbook 75 proved `(run_id, event_id)` is not a unique physical-event
key in the merged MC24 K-short rec files: each rec file merges ten
generator jobs that restart event numbering. Any sorted `(run, event)`
grouping is therefore forbidden for cross-geometry FD matching.

Implementation (`datasets/physical_event_identity.py`):

- Every FD refit point writes an `enhanced_tracklets.root` ntuple with
  one entry per processed event in file order. The entry-level
  occurrence of each entry among earlier entries with the same
  `(run, eventID)` is the absolute physical occurrence; the synthetic
  unique key is `uid = event_id + occurrence * 2^20`.
- Per-entry tracklet/propagation vector lengths bridge each flat
  `tracklets.root` / `propagations.root` row block back to its ntuple
  entry, so the two converted files stay consistent even when an event
  contributes rows to only one of them (e.g. a tracklet-only event).
- The reference and ±FD geometries are joined by deterministic exact
  join on the augmented uid plus the existing tracklet/truth identity.
  No fuzzy join, no nearest-neighbour join, no residual-proximity
  join. A duplicate physical-event identity is a hard failure.
- For files without collisions every occurrence is zero and the
  augmented id equals the raw event id, leaving canonical behaviour
  bit-identical.

Regression tests deliberately construct duplicate `(run, event)`
blocks, a tracklet-only middle event, and cross-occurrence misjoin
decoys, and confirm: no merging, no misjoining, the sorted loader
hard-fails on such files, and correctly paired residuals equal the
designed values exactly.

## FD reconstruction (HTCondor only)

Interactive nodes are restricted to config validation, unit tests, a
tiny smoke test, and a short single-point provenance check. The full
K-short physical FD reconstruction runs on HTCondor and reuses the
existing physical payload / Calypso FD infrastructure
(`prepare_multisource_multidof_iteration.py` +
`submit_multisource_multidof_iteration_condor.py`); no second
reconstruction chain is written.

- Template `configs/physical_refit_mc24_kshort_5dof_fd_iteration.yaml`
  copies the canonical hierarchical V1 `physical_refit_capture_scan`
  block verbatim (all seven parameter specs including the `ift_dz_mm`
  / `C_dx` probes with frozen steps; the analysis selects the five
  station-free parameters via `only_parameters`, exactly as workbook
  73 loaded the canonical bank). `nevents = 10000` (every event of
  every file). No held-out closure points.
- Every candidate source x physical FD point persists: exact input
  xAOD, source id, config SHA, starting git SHA, Calypso revision,
  payload/geometry delta, command, environment, stdout/stderr, exit
  codes, resulting tracklet/refit paths, join counts, and
  dropped/unmatched counts with explicit reasons.
- Failed jobs are never silently deleted. Infrastructure failures may
  be retried; physics sources are never rerun or selected to improve
  rank.

## Analysis order: physical closure before information

1. **Physical closure / provenance** — every pre-declared source x FD
   point complete (payload, tracklets, propagations, enhanced ntuple,
   content audit), per-source Jacobian validity passes the frozen
   gates, exact-join duplicate physical-event identity count is zero,
   no geometry or official-conditions write. If the physical chain
   fails, the result is classified as physical/provenance failure and
   no "rank" is reported.
2. **Canonical-only regression** — the frozen canonical bank is reused
   (SHA-verified, no Athena rerun); the rebuilt workbook-73 analysis
   must reproduce the frozen per-source ranks, singular values
   (rtol 1e-10), pooled rank, and frozen decision.
3. **K-short-only** — per-source and pooled `A = W^{1/2} J S`,
   singular spectrum, rank, identifiable subspace, source stability
   referenced to the K-short pooled native rank (rank 5 not required),
   slope-tertile / topology coverage stability, bootstrap / half-split
   stability.
4. **Joint complementarity** — pre-registered combination (canonical
   18 + K-short 10 pooled, equal-per-source consensus):
   canonical↔K-short pooled subspace principal angles (report-only; no
   frozen threshold exists and inventing one is forbidden), K-short
   information along the weakest canonical direction (report-only),
   joint pooled rank (necessary, never sufficient), joint source
   stability via pooled-remainder LOSO (the pooled subspace rebuilt
   without each source vs the full pooled subspace at rank 5 / 15 deg
   / 1.0; the workbook-73 remaining-vs-left-out-source direction is
   still reported but is not the joint gate because low-rank canonical
   sources stay in the pool by construction), joint coverage
   stability, joint bootstrap / half-split stability, and the
   workbook-69 stable-core independent validation.

## Pre-registered decision (frozen before FD)

`complementarity_pass = true` if and only if ALL of the following hold
(every numeric threshold inherited from workbooks 68/69/73):

1. physical closure / provenance fully pass;
2. canonical-only regression reproduces the workbook-73 frozen
   conclusion exactly;
3. K-short pooled identifiable rank >= 1 (information present; rank 5
   not required);
4. K-short source stability passes the frozen gates at its native
   pooled rank;
5. K-short coverage stability (tertile + topology) passes the frozen
   gates at its native pooled rank;
6. joint pooled identifiable rank == 5 (necessary, not sufficient);
7. joint source stability (pooled-remainder LOSO) passes the frozen
   gates;
8. joint coverage stability passes the frozen gates;
9. joint bootstrap / half-split stability passes the frozen gates;
10. workbook-69 independent validation: the canonical hypothesis core
    has dimension 5 and the K-short independent set passes the frozen
    support / persistence / angle / missing-Frobenius gates.

## Conditional branches

- **Pass**: freeze workbook 76 and its artifacts. Real-data correction
  stays closed. Full tracker-only 5DoF controlled closure, three-arm
  synthetic/injected closure, and Frozen-V2 unknown-association
  integration each require a separate pre-registration.
- **Fail**: freeze the failure. Forbidden: lowering rank tolerance,
  changing S, dropping sources, changing pion selection, changing the
  residual, trying another nearby MC in sequence, or returning to 7D /
  cluster-local / stable-core rescue. With the workbook-74/75
  track-coverage search complete and the only admitted candidate not
  resolving 5DoF portability, the scientific condition for turning to
  a separately pre-registered **Gauge-Constrained /
  External-Constraint Alignment Feasibility** campaign is met.
  External constraints must still distinguish real independent
  survey/metrology from software gauge/regularization; no Nov-2022
  quantity without measurement covariance, validated frame mapping,
  and IOV provenance may enter a physical prior.

## Engineering

- `configs/physical_curriculum_mc24_kshort_trainval.yaml` — 10-source
  corpus with the pre-registered split.
- `configs/physical_refit_mc24_kshort_5dof_fd_iteration.yaml` — FD
  production template.
- `configs/kshort_rigid_station_5dof_fd_complementarity_feasibility_v1.yaml`
  — the campaign config (core of this pre-registration).
- `datasets/physical_event_identity.py` — ntuple-bridged occurrence
  identity; `datasets/root_loader.py` refactored to a single
  `_read_tracklet_columns` field-reading source (behaviour unchanged,
  full regression green).
- `alignment/kshort_rigid_station_5dof_complementarity.py` — analysis
  module reusing the workbook 68/69/73 primitives without algorithm
  copies.
- `scripts/report_kshort_rigid_station_5dof_complementarity.py` —
  `validate-config` / `canonical-regression` / `full` stages.
- `tests/test_kshort_rigid_station_5dof_complementarity.py` — 19
  tests.
- `outputs/kshort_rigid_station_5dof_fd_complementarity_feasibility_v1/`
  — artifacts with SHA256, config SHA, source SHA, starting git SHA,
  Calypso revision, commands, Condor cluster/job IDs, provenance,
  pass/fail reasons, and the final frozen decision.

## Results (backfilled 2026-09-04 after FD completion)

### Execution timeline

- First HTCondor submission (cluster 1108926, 10 sources x 15 points):
  the Athena refit and both converters succeeded on all 150 points
  (10000 events, ~8500 tracklets, ~5100 propagation records per point),
  but the content audit `audit_tracklets.py` hard-failed on the
  merged-rec `(run, event)` collisions under sorted grouping - a pure
  infrastructure defect (the driver never passed `--physical-order`).
- Fix: new scan-config key `merged_rec_physical_order: true` (commit
  `f7cf1f1`; canonical templates leave it unset and are bit-identical,
  539 tests green), the ten per-source configs patched in place, and a
  resume resubmission (cluster 1108940): 150/150 points complete with
  all refit artifacts reused.
- The first analysis run exposed a JSON serialization leak (the
  `core_space` object returned by `build_core_from_rows` was not
  stripped from public keys); fixed with a serialization regression
  test, and the full analysis completed at 2026-09-04 21:57 UTC.

### Physical closure: FAILED

- All 10 sources fail only on `too_few_pairs`: per-source
  movable-station pair counts 6/8/8/9/10/11/12/16/19/19 against the
  frozen gate `min_pairs=20` (inherited verbatim from workbook 73,
  pre-registered, post-hoc modification forbidden).
- Every other Jacobian validity check passed: observed FD steps match
  the declared steps exactly (|Delta| < 1e-9), plus/minus probes
  present, populations aligned, residuals/covariances/Jacobians all
  finite, no near-zero columns, no held-out mixing, no test access.
- Diagnosis (source mc24_100130_00000_00009): the reference point
  alone has only 18 movable-station pairs (out of 1194 total), and the
  15-point intersection keeps 14 - the bottleneck is the intrinsic
  acceptance of "one pion in station 0, sibling pion in stations 1-3"
  for Ks daughters (~0.15%/event), not cross-point reconstruction
  instability (intersection retention ~78%).
- Physical-event identity: 9000 of 10000 `(run, event)` collisions per
  source uniquely resolved by the ntuple-bridged occurrence-augmented
  uid; zero duplicate-identity hard failures bank-wide; exact join
  only (no fuzzy/nearest-neighbour).

### Canonical-only regression (negative control): PASSED

- The canonical-only analysis rebuilt from the frozen hierarchical V1
  bank exactly reproduces the workbook-73 frozen artifacts: identical
  source set, identical per-source ranks, singular values within
  rtol=1e-6, pooled rank 5, identical decision (both frozen artifact
  SHA256 checks passed).

### K-short-only (report-only, not claimable)

- Pooled singular values 536.5 / 458.3 / 48.02 / 37.25 / 16.25, pooled
  rank 5.
- Per-source ranks: 7 sources rank 5, 3 sources rank 4 (with only 6-19
  pairs each).
- Source stability at the native rank: FAILED; coverage stability:
  passed.

### Joint complementarity (report-only)

- Joint pooled rank 5 PASS; joint source stability
  (`pooled_remainder` LOSO) PASS.
- The canonical and K-short pooled subspaces are nearly identical:
  maximum identifiable principal angle 1.48e-6 deg, projector
  Frobenius distance 1.48e-15 - with thousands of canonical pairs per
  source against ~a dozen K-short pairs, the pooled matrix is
  absolutely dominated by canonical information and the K-short weight
  is invisible at the pooled level.
- Canonical hypothesis stable-core dimension 4 (gate requires 5:
  FAIL); the K-short workbook-69 independent validation of that core
  passes 10/10 sources.
- Weakest-canonical-direction information ratio K-short/canonical =
  0.175 (report-only, no frozen threshold); joint bootstrap/half-split
  and coverage stability FAILED.

### Final frozen decision

- `kshort_rigid_station_five_dof_physical_or_provenance_failure`,
  `no_rank_claimed = true`.
- The decision chain follows the pre-registration exactly: physical
  closure short-circuits first; `joint pooled rank == 5` alone is not
  success; all report-only observables are archived, never promoted to
  conclusions.
- Failure nature: intrinsic movable-station pair acceptance sparsity
  of the K-short population (physics), not an infrastructure,
  event-identity, or reconstruction-chain defect.

### Artifact SHA256

| artifact | SHA256 |
|---|---|
| physical_closure.json | 800817ce9edeeea5436ed53fa845ebe0b6c72df919d832aa56a260916a922c97 |
| canonical_regression.json | b1ca95d0ec901fa4ba8d8fe967fb40c40569fd206fe7091cc7d9e7b8a5529427 |
| canonical_only.json | a94d6b82496ec002c1a6ccfa97fb2394b1c97f8d692be5109672378fc11f3805 |
| kshort_only.json | 1325403f0d1f1d42506c67a11d520b3fd4e4c4dbe5243589148d9e1d777f3767 |
| joint.json | 2d39d72efc11f3d5181c74953ac2e62dbda90df976c6db2db361997d908e9567 |
| complementarity.json | 8e8d96248d35b61d3d05598bf0800a96558031adc8b6edef376228b2c93044ad |
| next_stage_decision.json | 7894195c36cbb24f88fd4d0117ddf5010bfa0328d6ce760904fb1ba8ee52aefd |
| iteration_manifest.json | 315ba0a3916e80d9c89b2744524516584089592f0ac364a3854f324f8bb99bda |
| smoke_reference_rz_probe.json | 4233224444b9abea1b2f22854b0d495a619ee26c825fcd5623a9d0b97a0fb46b |

Condor: cluster 1108926 (infrastructure failure), cluster 1108940
(150/150 complete). Key commits: `5329687` (pre-registration),
`b21ea23` (bridge matched to the converter drop mask), `f7cf1f1`
(merged-rec audit switch).

### Follow-up (pre-registered branch)

- The failure is frozen, unlocking the `Gauge-Constrained /
  External-Constraint Alignment Feasibility` campaign (independently
  pre-registered config + workbook; it must not reuse this entry's FD
  spectra to design its gates).
- Observation (not a conclusion): the K-short pooled spectrum reaches
  rank 5 and the independent validation passes; any future use of
  K-short information requires a new campaign whose statistics gates
  are pre-registered before seeing any new FD spectrum (e.g. a
  min_pairs designed from per-source pair-yield priors, or observables
  that do not require movable-station pairs). Post-hoc relaxation of
  `min_pairs=20` inside this entry is forbidden.
