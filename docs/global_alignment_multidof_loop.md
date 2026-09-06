# Source-Disjoint Multi-DoF Global Alignment Loop

> **Superseded as the execution plan (2026-09-06).** This document remains a
> historical physical-control description. Current tasks are
> `docs/CODE_ROADMAP.md`. Do not reopen rank-rescue or Frozen-V2 deployment
> from this page.

## Scope

This is the physical control path for FASER global association and alignment.
It adds no new Transformer architecture. The association backbone remains
frozen, with the sealed V2 BCE route-query artifact as the usable control.
The permanently sealed test bank is never read.

The initial active block is IFT/station-0 `dx`, `dy`, `Rx`, `Ry`, and `Rz`,
with `dz` survey-constrained; stations S1--S3 define the reference frame.
Calypso payloads use `[dx, dy, dz, Rx, Ry, Rz]` in mm/rad, while reports use
mrad for rotations.

## Physical Contract

Every point independently executes:

```text
/Tracker/Align SQLite/POOL payload
  -> SCT_ClusterContainer
  -> SegmentFitRefit
  -> SegmentsRefit
  -> NtupleDumper
  -> FaserActsExtrapolationTool (mode 0)
```

No coordinate shift, residual-level injection, cached propagation, or local
`q/p` substitute is allowed. Local segment `q/p` remains an unreliable seed,
so physical V1 always uses mode 0.

## Reproducible Iteration

Prepare the source-disjoint anchor/probe bank:

```bash
source scripts/setup_environment.sh ml
python scripts/prepare_multisource_multidof_iteration.py \
  --source-config configs/physical_curriculum_v3_expanded_trainval.yaml \
  --iteration-template configs/physical_refit_multidof_smoke_mc24_100043.yaml \
  --output-root outputs/mc24_multidof_ift_iteration00_anchor_trainval_physical_v1 \
  --iteration 0 \
  --current ift_dx_mm:2.0 --current ift_dy_mm:-1.5 --current ift_ry_mrad:35.0 \
  --nevents 100
```

The bank contains 10 train and 8 validation original xAOD files. Each source
has eight independently refitted points: reference, anchor, and plus/minus
probes for the three active parameters. Submit one source per job:

```bash
python scripts/submit_multisource_multidof_iteration_condor.py \
  --iteration-manifest outputs/mc24_multidof_ift_iteration00_anchor_trainval_physical_v1/iteration_manifest.json \
  --submit-dir outputs/condor_mc24_multidof_ift_iteration00_anchor_trainval_physical_v1 \
  --schedd-mode eossubmit --submit
```

After every point passes completion checks, aggregate the real finite
differences:

```bash
python scripts/run_multisource_refit_multidof_local_step.py \
  --iteration-manifest outputs/mc24_multidof_ift_iteration00_anchor_trainval_physical_v1/iteration_manifest.json \
  --anchor-point iteration_00_anchor --target-point iteration_00_reference \
  --fit-split train --held-out-split validation \
  --capture-tolerance ift_dx_mm:0.1 \
  --capture-tolerance ift_dy_mm:0.1 \
  --capture-tolerance ift_ry_mrad:1.0 --require-full-rank \
  --output-dir outputs/mc24_multidof_ift_iteration00_anchor_trainval_closure_v1
```

The train split determines the update. Validation only evaluates that frozen
update and reports an independent diagnostic fit. Outputs include raw
candidate truth-chain retention at every point, finite-difference curvature,
rank, scaled condition number, covariance/correlation, and source/station-pair
stability. An update advances only with `capture_success=true`.

## Association Loop

The completed bank is assembled read-only for the existing pooled synthetic
tools:

```bash
python scripts/assemble_multisource_multidof_iteration_manifest.py \
  --iteration-manifest outputs/mc24_multidof_ift_iteration00_anchor_trainval_physical_v1/iteration_manifest.json \
  --materialization-config configs/physical_alignment_iteration_trainval.yaml \
  --output outputs/mc24_multidof_ift_iteration00_anchor_trainval_physical_v1/physical_corpus_manifest.json
```

`alignment_iteration_shared_across_payloads` shares only deterministic overlay
choices to intersect route provenance across physical payloads. Physical
tracklet states, covariances, and Acts output stay payload-specific. Run frozen
V2 one physical point at a time with `--payload-id`, then feed selected exact
mode-0 Acts edges to `run_route_selected_multidof_update.py`. Its straight-line
fit is diagnostic only; it is not an alignment objective in magnetic field.

`audit_field_global_fit_contract.py` audited the current propagation product:
the tree has all pairwise residual/covariance fields but no exported transport
Jacobian or source-state transition representation. Therefore the current
field-aware update is correctly labelled *route consistency plus WLS*, not a
global independent likelihood. A true field-aware global fitter is deferred
until Calypso exports such a Jacobian or exposes a validated common-state Acts
repropagation API.

Use a verified update to create the next real payload bank with `--update-json`.
That CLI rejects an unverified update unless an explicit diagnostic override is
requested.

## DoF Admission

The payload/Jacobian code supports all station rigid components. The 6-DoF
identifiability pilot admitted station-0 `dx`, `dy`, `Rx`, `Ry`, and `Rz` as
track-constrained free parameters and rejected `dz` as gauge-like for the
current near-parallel sample. `dz` may enter the normal equation only with a
frozen survey prior; prior-dominated posterior recovery is not a track-based
measurement. Single-step Newton updates stay in the validated linear regime
(normalized 5-DoF severity `<= 0.15`, with sparse `~0.2` stress points).
Severity `0.5–1.0` is not a single-step closure target. The IFT
station/layer hierarchy is now the active identifiability step: station
common-mode `dx/dy/rx/ry/rz` stay frozen, `dz` stays survey-constrained, and
layer internals are admitted only in the zero-common-mode contrast basis
(`outer_contrast`, with equal-weight `sum_to_zero` as the same family).
Frozen-station `reference_layer` is a negative control of a different
physical constraint, not a gauge cross-check. A layer condition is written as a Calypso L2 `{station}{layer}`
key and is never copied onto a station payload. Module level remains reserved.

## Current Status

Truth-selected IFT **layer-internal relative dx** closed (workbook 39): both
gauges recover the outer-antisymmetric `[+0.12, 0, −0.12]` mm injection as
the same `layer_i − mean` internals (outer relative 0.241 mm vs 0.240 mm),
full rank, condition 16–37, residual RMS ×0.035, no station-slot leakage.
Frozen-V2 route-selected closure of the same relative dx also passed: 193
deduplicated physical edges, rank 2/2, condition 28–54, outer relative
0.240 mm (`sum_to_zero`) and 0.236 mm (`reference_layer`), residual RMS
×0.005, station six-vector unchanged. The admitted physical internals are now written as the explicit contrast
basis `C_dx=(dx_L0-dx_L2)/2` (workbook 40): layer common mode is fixed at
zero and layer 1 is not floated. Re-solving the workbook-39 relative-dx
bank in this basis recovers `C_dx=0.121` mm (outer 0.243 mm vs 0.240 mm)
in agreement with both older gauges; `C_dx` is frozen and not retuned.
Workbook 41 showed that frozen-station `reference_layer` is not a
coordinate of the same physical subspace: the additive map of
`L=[+C,0,-C]` onto layer 0 fixed at 0 is `L'=[0,-C,-2C]` **plus** a
compensating station `S=+C`. With the station frozen, `reference_layer`
fits a different family. Calypso composition then shows that even the
compensated chart is SE(3)-equivalent for `dx` but not for `rx`, because
station `rx` rotates about the global origin while layer `rx` is
conjugated to plane *z*. Outer relative rx is admitted in the contrast
basis: truth-selected `C_rx=0.702` mrad recovers 1.404 vs 1.400 mrad
(condition 1, residual RMS ×0.015), and frozen-V2 route-selected
recovers 1.399 mrad on 195 deduplicated edges (rank 1, condition 1,
residual RMS ×0.005), station six-vector unchanged. `C_dx` is not mixed
into that solve. The first 2-D `C_dx+C_rx` source-disjoint
mini-curriculum closed and was **not admitted** (workbook 42): the 10/8
joint bank (cluster 999274, 180 real refits) has train posterior
correlation **0.969**, truth-selected source-dependent sign flips on
`C_rx`, and source-disjoint validation route-selected residuals that
barely drop (post/pre 0.92–0.97). Workbook 43 then treats the two closed
1-D modes as block coordinates rather than retrying a joint Newton step.
On the same `iteration_00_start` / `heldout_00` injections, 1-D `C_dx`
(with `C_rx` held at the current geometry) is statistically inside the
frozen 3σ window with a ~0.009 mm leakage bias from the unmodelled
rotation; 1-D `C_rx` (with `C_dx` held) sign-flips and leaves remaining
`|C_rx|≈1.7` mrad, outside the verified 0.70 mrad envelope, so the
`C_rx→C_dx` order is not physically produced. The `C_dx→C_rx` remaining
payloads were written and refit (cluster 999291, 36 real refits); the
second physical `C_rx` step then recovers only ~7% of leftover rotation
on train and leaves source-disjoint validation residuals unchanged
(post/pre 0.996–1.00). Sequential contrast alignment is **not admitted**.
`C_rx` is dropped from the hierarchy mainline; the only reliable IFT
internal DoF remains the already-frozen 1-D `C_dx`. Hierarchical alignment
V1 is **closed as a joint hierarchy** (workbook 45): station 5-DoF+survey
`dz` and IFT 1-D `C_dx` remain valid **independent calibration modes**, but
the leakage operator `A_dx≈−59`, `A_ry≈−32` is a stable Jacobian geometry,
so a station `dx` gate of 0.1 mm would require leftover `|C_dx|≲1.7 µm`
while registered `σ(C_dx)=6.91 µm`. Nuisance projection of `C_dx` leaves
station 5-DoF formally full rank but inflates `σ(dx)` by ~30×. They must
not be solved together, sequentially or jointly. Do not add layer
parameters or pull in relative ry to compensate. Workbook 46 freezes them
as two exclusive calibration modes with a machine-readable mode-validity
contract (`|A_dx|≈59.213` requires unmodeled `|C_dx|≲1.5–1.7 µm` before
Station Mode; `C_dx` Mode requires workbook-36 station capture and records
a 0.715 µm station→`C_dx` systematic). See
`docs/faser_alignment_operating_protocol_v1.md`.
Outer relative ry,
layer **dy/rz**, and module level stay out. Station 5-DoF stays
fixed. Station-level **5 track-constrained DoF + 1 survey-constrained DoF**
remains frozen (workbook 36). `dz` stays out of the track-based capture.

**The 3-DoF loop had already closed.** Iteration-0 (anchor dx/dy/Ry = 2.0 mm/−1.5 mm/35 mrad)
recovered the offset to −0.26/−0.22 mm/−0.05 mrad on held-out validation and
proposed the iteration-1 anchor (−0.14 mm/+0.11 mm/+0.74 mrad). Iteration-1
passed every frozen held-out tolerance (dx −0.014 mm, dy +0.019 mm, Ry
−0.0003 mrad; source-to-source spread collapsed 20–40×), and the truth-free
route-selected update then closed on both source-disjoint splits with the
frozen V2 backbone: train dx/dy/Ry errors −0.0008/−0.0005 mm/−0.0003 mrad
(2309 common edges, condition number 80), validation −0.0945/+0.0575 mm/
−0.0280 mrad (1756 common edges, condition number 85) — all within the frozen
tolerances (0.1 mm/0.1 mm/1 mrad). This is the first unknown-association,
source-disjoint, real-payload/refit/Acts, multi-DoF iterative alignment
closure of the project. The validation dx margin is dominated by one outlier
source (`mc24_100047_00150_00199`); see
`workbook/2026-08-18_24_validation独立closure与第一阶段结论.md`.

Two implementation fixes were required and are now covered by regression
tests: the alignment-iteration materialization namespace is payload-stable
(`materialize_pooled_curriculum_synthetics.py`), and the update consumes an
anchor-selected route set re-measured in every payload's candidate graph
(`--observation-kind anchor_selected_field_edge`) because per-payload route
selections do not intersect across finite-difference probes.

The association architecture is now frozen. The next phase targets the two
known foundational bottlenecks: the severely anisotropic mode-0 propagation
covariance mis-calibration and the 0->1 raw candidate coverage.

That phase has completed its first audit cycle (train-only pull calibration,
frozen diagonal rescaling + a priori Huber control, gate/coverage rescan at
nominal and both iteration anchors, and calibrated closure variants on the
frozen iteration-1 banks). The findings, recorded in
`workbook/2026-08-18_25_mode0协方差校准与candidate覆盖修复.md`:

- The mode-0 covariance mis-calibration is a core-overestimated,
  heavy-tailed, non-Gaussian distortion: robust pull widths are 0.30-0.43
  (x), ~0.003 (y), 0.72-0.88 (tx), ~0.002-0.003 (ty), with |pull| q99 up to
  4; validation pull widths match train almost exactly, so the distortion is
  source-independent.
- Raw candidate coverage is gate-limited, not covariance-limited: the
  physical events carry at most one tracklet per station, so fake candidates
  are identically zero at every gate, and raising the chi2 gate from 25 to
  500 lifts complete truth-chain recall from ~0.40-0.50 to ~0.72-0.86 at no
  fake cost.
- Neither train-frozen control is adopted. The diagonal core-width rescaling
  improves dy everywhere (|error| < 0.01 mm per source) but collapses the
  dx/Ry solve (validation dx error -1.19 mm): the rescaled y/ty block
  outweighs x/tx by ~1e5 while constraining only dy, leaving the
  near-degenerate dx-Ry direction to noise. The Huber control alone also
  degrades validation (dx error -0.176 mm). The canonical physical
  candidate/WLS covariance model therefore remains the exporter's original
  covariance, and the calibration artifacts are kept as documented negative
  results (`outputs/mc24_multidof_ift_pull_calibration_train_v1/`,
  `outputs/mc24_multidof_ift_gate_coverage_scan_v1/`).

A follow-up read-only diagnosis separated candidate generation from the
dx-Ry source dependence (`workbook/2026-08-18_26_离群源影响机制与candidate_gate操作区.md`):

- At the converged iteration-1 anchor the normal matrix is well conditioned
  (80-86) with nearly axis-aligned eigenmodes; the dx-Ry -0.97 correlation
  was an iteration-0 large-misalignment phenomenon and is gone.
- The validation dx error is entirely a single-source effect: removing
  `mc24_100047_00150_00199` moves the pooled dx delta onto the truth
  (error -0.0015 mm).  The source is not high-leverage, not heavy-tailed,
  and its per-edge Jacobian/kinematics are typical.  The cause is one
  mis-associated 0->1 edge in a multi-track physical event (wrong station-1
  candidate at chi2 12.8, residuals +/-150 mm at the two payloads), embedded
  by the overlay into six route copies that multiply its weight; the
  inflated mode-0 covariance makes the edge statistically unrecognisable to
  both the candidate gate and the WLS solve.
- A frozen candidate chi2 gate scan (25/50/100/200/500/ungated, frozen V2
  checkpoint/calibration/thresholds, train-selected and validation
  frozen-evaluated) found no usable operating region: the current policy is
  already ungated with candidate-level complete truth-chain recall ~1.0,
  tightening the gate collapses track efficiency (0.854 -> 0.003 at gate 25)
  for a purity gain of only 0.975 -> 1.000, and intermediate gates combine
  low statistics with concentrated contamination (validation dx error
  -0.40/-0.19 mm at gates 200/500).  The candidate-generation policy
  therefore remains ungated with the original covariance; the bottleneck is
  the propagation/candidate model's chi2 discrimination, not any threshold.
  Before Rx/Rz/dz or station-level 6-DoF expansion, the route-selected
  update should deduplicate or reweight the multiple route copies of the
  same physical edge created by overlay reuse.

### Observation statistics: physical-edge weight normalization (2026-08-18, canonical)

The overlay embeds one real physical edge into several synthetic events, so the
frozen route selection placed replica observations of the same physical edge
into the normal equation up to 11 times.  Three pre-defined statistical
semantics were compared on the frozen iteration-1 train/validation closures
(`scripts/run_observation_statistics_variants.sh`,
`alignment/route_selected_update.py::apply_observation_statistics`; grouping
uses only endpoint provenance — original source file/event UID, source/target
station and tracklet — never truth):

| semantics | split | observations | unique edges | dx err (mm) | dy err (mm) | Ry err (mrad) |
|---|---|---|---|---|---|---|
| replica_weighted (control) | train | 2309 | 623 | -0.0008 | -0.0005 | -0.0003 |
| physical_edge_deduplicated | train | 623 | 623 | -0.0009 | -0.0003 | -0.0008 |
| physical_edge_inverse_multiplicity_weighted | train | 2309 | 623 | -0.0009 | -0.0003 | -0.0008 |
| replica_weighted (control) | validation | 1756 | 467 | -0.0945 | +0.0575 | -0.0280 |
| physical_edge_deduplicated | validation | 467 | 467 | **-0.0511** | +0.0396 | -0.0159 |
| physical_edge_inverse_multiplicity_weighted | validation | 1756 | 467 | **-0.0511** | +0.0396 | -0.0159 |

Findings (`outputs/mc24_multidof_ift_iteration01_obsstat_{train,validation}_v1`,
`..._source_influence_{train,validation}_dedup_v1`):

- Replica residuals are bit-identical across copies (max spread 0.0), so
  deduplication and inverse-multiplicity weighting are numerically identical,
  as predicted; either normalizes each physical edge's total weight to one.
- The known mis-associated edge of `mc24_100047_00150_00199` enters the
  replica-weighted update 6 times and the deduplicated update exactly once
  (verified explicitly against the observation keys).
- Validation dx error halves (-0.0945 -> -0.0511 mm), moving stably away from
  the 0.1 mm tolerance boundary; dy/Ry errors also shrink.  Train closure is
  unchanged (~1 um).  Both splits pass the frozen tolerances.
- The other seven validation sources are not sacrificed: their per-source dx
  solves move by at most 0.004 mm.  The outlier source's leave-one-source-out
  influence halves (+0.0929 -> +0.0488 mm); the validation per-source dx
  spread shrinks 0.1448 -> 0.0855 mm.  Rank stays 3; the condition number
  rises only mildly (80-86 -> 108-135); |corr(dx,Ry)| stays below 0.3.

**Decision: `physical_edge_deduplicated` is frozen as the canonical
alignment-observation semantics** (deterministic sorted-first representative;
`replica_weighted` retained as a strict control via
`--observation-statistics`).  With the statistical semantics fixed, the next
independent layer is the mode-0 propagation covariance / chi2 discrimination
study; only after that layer is understood may Rx/Rz/dz sensitivity and
station-level 6-DoF expansion begin.

### Mode-0 propagation chi2 discrimination mechanism (2026-08-18, negative result, chi2 layer closed)

A dedicated edge-level study (`scripts/audit_propagation_discrimination.py`,
`scripts/evaluate_propagation_compatibility.py`,
`alignment/propagation_compatibility.py`; train-fit, single frozen validation
transfer; candidate endpoints, V2 scores and the WLS covariance untouched)
modelled every ungated adjacent-pair candidate — 84,770 train / 67,251
validation edges at the anchor and reference payloads — by residual, full
combined covariance, pulls, chi2, track state, station pair and source.

Mechanism of the known mis-associated edge (chi2 = 12.8 at |r_x| = 150 mm):

- The mode-0 combined covariance at 0->1 is enormous (sigma_x ~ 49 mm,
  sigma_y ~ 493 mm), so the 150 mm residual is only a 3.1-sigma marginal in x.
- chi2 = 12.8 sits at the **median** of the train truth-edge chi2
  distribution (quantile 0.507): the truth distribution itself is
  catastrophically heavy — median 15.6 (vs 3.4 for a calibrated chi2_4),
  29% of truth edges above chi2 = 100, q99 ~ 1.3e5, max 4.8e7.
- Fake edges have median chi2 ~ 5e3 but their 1% lower tail (~6-10) overlaps
  the truth core, while the truth tail extends past the fake upper tail.
  The two distributions overlap at every quantile.

Train-frozen model comparison (per-station-pair Student-t grid-MLE nu,
core+tail scaled-Gaussian EM mixture, Huber/Tukey at train chi2 quantiles,
all with the log|C| term; thresholds frozen at 99.5% train truth retention):

- AUC(chi2) = 0.86-0.92 per pair; Student-t / mixture improve it by at most
  +0.01; Huber/Tukey are monotone in chi2 and cannot change ranking.
  Component-pull scores are worse (AUC 0.55-0.59).
- At the frozen veto, fake rejection is 0.3-6%; the bad edge is vetoed by no
  model (thresholds ~1e5 because of the truth tail).
- Dependences are weak and cannot be exploited: per-source truth chi2
  structure is uniform (the outlier source is typical: median 15.7,
  P(chi2>100) = 0.196), anchor vs reference payloads are near-identical,
  only large-|y| edges are heavier-tailed (P(chi2>100) = 0.59 for |y|>100 mm).

**Conclusion (frozen decision tree): no train-frozen likelihood reliably
separates the truth heavy tail from wrong edges — mode-0 propagation carries
insufficient discriminating information at the chi2 layer, and chi2-layer
tuning stops here.**  No compatibility-veto closure variant is warranted: the
bad edge is at the truth median, so any veto catching it would reject about
half of all truth edges.  The bottleneck is upstream of statistics: the
mode-0 propagation/covariance model itself (sigma_y ~ 493 mm, truth chi2 tail
to 1e7) must be understood before Rx/Rz/dz sensitivity or station-level
6-DoF expansion.

### Mode-0 propagation root-cause audit (2026-08-18, root cause identified)

Following the chi2-layer closure, a read-only source decomposition of the
truth-matched propagation uncertainty was performed
(`scripts/audit_propagation_uncertainty_budget.py`; train and validation
physical banks at anchor and reference payloads).  The existing bank already
stores three same-edge q/p controls per truth edge (identical reconstructed
position/direction unless noted): mode 0 = reconstructed state with the
segment fit's q/p, mode 1 = reconstructed state with truth q/p and the q/p
covariance SUPPRESSED, mode 2 = full truth state with truth q/p.

Uncertainty budget (train, iteration-1 anchor, medians; material/process
noise is DISABLED in the production config, so transport is pure J·C·Jᵀ):

| pair | mode | src σ_y (mm) | prop σ_y (mm) | tgt σ_y (mm) | prop σ_x (mm) | res σ_x (mm) |
|---|---|---|---|---|---|---|
| 0->1 | 0 (fixed 100 GeV) | 0.50 | **492** | 0.50 | 32.5 | 16.0 |
| 0->1 | 1 (truth q/p) | 0.50 | **16.1** | 0.50 | 31.9 | 15.2 |
| 1->2 | 0 | 0.50 | **228** | 0.50 | 22.9 | 11.9 |
| 1->2 | 1 | 0.50 | **0.73** | 0.50 | 22.9 | 11.4 |
| 2->3 | 0 | 0.50 | **225** | 0.50 | 22.9 | 10.1 |
| 2->3 | 1 | 0.50 | **0.73** | 0.50 | 22.9 | 10.1 |

Root causes, in order of dominance:

1. **Pathological σ_y = transport of the segment fit's dummy q/p variance.**
   Every tracklet carries q/p = 1e-5/MeV (fixed 100 GeV placeholder from the
   straight-line segment fit) with variance 5e-6/MeV², i.e. σ(q/p) is 224×
   the central value — momentum is effectively unconstrained.  Mode 0
   transports this dummy variance through the magnetic field into (y, ty).
   The same-edge mode-1 control (truth q/p seed + suppressed q/p covariance)
   collapses σ_y by 30-300× while the residual is essentially unchanged
   (16.0 -> 15.2 mm at 0->1): pure covariance inflation, no residual
   degradation.  This is an implementation issue, not physics.
2. **σ_x ~ 23-32 mm = legitimate lever-arm transport** of the direction
   uncertainty (both modes, all pairs); it overestimates the residual by
   ~2× at 0->1.
3. **The truth chi2 heavy tail (1e5-1e7) = structural near-singularity of
   the combined covariance plus a genuine residual-outlier sub-population.**
   The lever-arm transport makes the position-direction blocks nearly
   degenerate: median cond(C) = 3.5e10 even for core edges (6.9e11 in the
   tail), with the near-null direction at 0.73·tx + 0.46·ty.  64% of tail
   edges are modest residuals amplified by the ill-conditioning; 36% are
   genuine residual outliers (marginal pull > 5).  The tail is q/p-mode
   independent (1.5% mode 0 vs 3.5% mode 1 above 1e4; the same edges).
4. **Material/process noise contributes exactly zero** — the production
   config leaves InteractionMultiScatering/InteractionEloss at their default
   False, so no process noise enters the transport.
5. No unit, frame, or FD-Jacobian anomalies: the native->global and
   bound->bound transforms use sound steps (1e-4 mm, 1e-6 rad) and core
   pulls are <= 1.  The residual mode-1 σ_y at 0->1 (16 mm at the anchor vs
   0.9 mm at the reference) is misalignment-dependent field coupling across
   the magnet gap; validation reproduces the train numbers (12 mm).

**Fix direction proven by the train-only same-edge control:** suppress the
dummy q/p variance in fixed-seed (mode-0) covariance transport — the mode-1
control shows σ_y collapses 30-300× at unchanged residual, and mode-1 pulls
at 0->1 are nearly calibrated in x/tx (robust pull sigma 1.01/1.07).

### Mode-3 suppression pilot (2026-08-18, verdict: diagnostic only, mode-0 stays canonical)

The proven fix direction was validated in a minimal real production pilot: a
new independent dumper variant **mode 3** (identical reconstructed
position/direction and the same fixed q/p seed as mode 0, with
`suppressQOverPCovariance=true`; modes 0/1/2 unchanged) was run through the
full payload -> SegmentFitRefit -> NtupleDumper -> Acts propagation chain for
three train sources x all eight iteration-1 payload points
(`outputs/mc24_mode3_suppression_pilot_physical_v1/`, 24/24 points, 0
failures), followed by a pilot overlay (production-identical parameters, same
synthetic events for both candidate variants), frozen V2 backbone inference,
and the route-selected dx/dy/Ry closure under physical_edge_deduplicated
semantics (`scripts/run_mode3_suppression_pilot.sh`).

Pilot gate results (all five criteria must hold for canonical promotion):

| criterion | result | verdict |
|---|---|---|
| residuals unchanged | predictions bit-identical to mode 0 (523/523 records) | PASS |
| q/p-induced covariance inflation removed | σ_y collapse 30-537× (min 29.8×) | PASS |
| separation or coverage improved | V2 AUC 0.9978 -> 0.9817 (anchor); model-free chi2 AUC 0.955 -> 0.941; coverage identical (0.9913) | **FAIL** |
| route metrics not degraded | complete-track efficiency 0.88 -> 0.14 (anchor) | **FAIL** |
| closure within frozen tolerances | both modes capture; mode-3 dx/dy/ry errors <= mode-0 | PASS |

The mode-3 covariance itself is nearly calibrated (0->1 y pull robust sigma
1.62 at the reference, vs 0.0034 for mode 0): the suppression works exactly
as designed.  The association collapse is a frozen-stack feature-shift
effect, not new physics: the V2 checkpoint, calibration, and route thresholds
were trained on mode-0's over-covered chi2 features, so truth edges with
honest covariances (chi2 median ~48-92) land in the model's fake-like region
and die at the 0->1 threshold.  Per the frozen decision rule, **mode 3 is
retained as a diagnostic variant and mode 0 remains the canonical propagation
mode**.  Exploiting the calibrated covariance would require retraining the
association model on mode-3 (or calibrated-covariance) candidate graphs,
which is explicitly out of scope for this phase.  Until then, no Rx/Rz/dz
sensitivity or station-level 6-DoF work starts, and no further
statistical-layer patches are attempted.
