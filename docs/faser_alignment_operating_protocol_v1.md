# FASER Alignment Operating Protocol V1

Workbook 46 / 2026-08-21. Hierarchical alignment V1 is closed as a joint
hierarchy. Production uses two mutually exclusive calibration modes.

## Frozen invariants

These are not retuned by this protocol:

- Canonical propagation: **mode-0**
- Association: frozen V2 checkpoint, calibration, and route policy
  (`outputs/mc24_v3_expanded_trainval_v2_bce_control_v1`, ungated)
- Observation: `anchor_selected_field_edge` + `physical_edge_deduplicated`
- Station capture: workbook 36
  (`outputs/mc24_ift_5dof_survey_dz_capture_criteria_train_v1/`)
- `C_dx` capture: `C_dx` row of workbook 40
  (`outputs/mc24_ift_layer_contrast_2d_capture_criteria_train_v1/`)
- Leakage operator `A` and this mode-validity contract: workbook 45/46
- Sealed test: **never opened**
- Real chain only: `/Tracker/Align → SCT_ClusterContainer → SegmentFitRefit → SegmentsRefit → NtupleDumper → Acts(mode 0)`

Forbidden: Schur projection as a production estimator; more hierarchical
iterations; new layer/module DoF (`C_rx`, relative ry, layer dy/rz, layer1,
module); joint station+`C_dx` Newton; treating the other level as a
nuisance; interpreting residual reduction as correct alignment.

Machine contract:

`outputs/mc24_ift_calibration_mode_validity_contract_v1/mode_validity_contract.json`

## When Station Mode is allowed

Float only `ift_dx_mm ift_dy_mm ift_dz_mm ift_rx_mrad ift_ry_mrad ift_rz_mrad`
with `--prior-sigma ift_dz_mm:5.0`. Written `dz` stays **0**. Layers are not
written.

Prerequisite: IFT internal `C_dx` is already fixed by **external geometry**
or a **dedicated calibration**, not by iterating on this data stage.
Isolation MC banks with identically zero layer internals use
`cdx_fixed_by=isolation_zero`.

Mode-validity gate, from `|A_dx|≈59.213`:

| gate | max unmodeled `|C_dx|` |
| --- | ---: |
| statistical (binding, 3σ_dx) | **1.462 µm** |
| engineering (0.1 mm dx) | 1.689 µm |
| published operating band | **1.5–1.7 µm** |

If potential unmodeled `|C_dx|` exceeds the statistical gate, the result is
**`cross_level_contaminated`**. Do not write it as a station geometry.
Registered `σ(C_dx)=6.91 µm` on the current reconstruction **does not**
satisfy this budget: a dedicated `C_dx` fit on this sample is not a valid
Station Mode prerequisite.

## When IFT-Internal (`C_dx`) Mode is allowed

Float only `C_dx`. Payload `L0=+C_dx`, `L1=0`, `L2=-C_dx`. Station
six-vector is not written.

Prerequisite: the station six-vector has already passed workbook **36**
framework capture, on a **different** data stage or by construction
(isolation MC with station identically nominal). Record the frozen
station-uncertainty systematic on `C_dx` (RSS **0.715 µm** at 1σ of the
free station sigmas; 3σ RSS 2.15 µm; survey `dz` 1σ **0.47 µm**). Do not
float station as a nuisance and do not retune `σ(C_dx)`.

## Cross-level contamination: reject and do not write

Reject geometry write if any of:

- unmodeled `|C_dx|` above the statistical budget (Station Mode)
- `C_dx` not declared fixed before Station Mode
- station framework capture not passed before `C_dx` Mode
- same-data-stage iteration between the two modes
- other level floated as a nuisance, joint Newton, or Schur projection
- leakage operator `A` unstable on the new sample (relative `|A_dx|` or
  `|A_ry|` deviation `>10%` vs freeze). **Do not retune `A`**; do not
  transfer this contract's budget onto that sample
- residual reduction used as the success bit (workbook 44)

Implied `|C_dx| = |dx| / |A_dx|` from a station `dx` shift is an additional
reject indicator when it exceeds the same budget.

## How results are written to geometry / conditions

Two separate write paths. Never a mixed remaining chart that zeros the
unfloated level.

| mode | writes | does not write |
| --- | --- | --- |
| Station | IFT station `dx/dy/rx/ry/rz`; `dz=0` | layer internals / `C_dx` |
| IFT-Internal | `L0=+C_dx`, `L1=0`, `L2=-C_dx` | station six-vector |

`--require-mode-valid` enforces `geometry_write_allowed`. Capture success
and mode validity are both required; neither is residual RMS drop.

## Large-statistics MC transfer (frozen inference)

No retraining. Source-disjoint, non-sealed. Isolation injections only.

Current 10+8 V3 corpus isolation closures already score as independent
closures under this contract
(`outputs/mc24_ift_calibration_mode_transfer_current_corpus_v1/`).
`A` is stable on the eight iteration-00 train/val × truth/route corners
(relative spread ~7% `<` 10% gate).

First-wave extra files (pending the physical chain): 10 extra train
sources and 4 extra validation sources, listed in
`configs/calibration_modes_large_stats_transfer_v1.yaml`. Validate each
mode separately: bias, pull, source/run stability, association
efficiency/purity/fake, normal-matrix condition, independent closure, and
`A` stability vs this freeze. Do not reopen hierarchical V1.

## Real FASER data (after MC transfer is stable)

No truth closure. Per-mode data-quality observables:

- unbiased residual width and mean
- source/run consistency
- pre/post residual reduction (**recorded, never success**)
- parameter stability
- route multiplicity
- post-refit geometry consistency

Workbook 03 still blocks unverified 2022 data0 IFT re-export provenance.
The V1 dry-run admits 2024 r0022 reconstructed xAOD only, with a frozen
calibration / holdout / held-out-DQ split. See
[the real-data dry-run](operating_protocol_v1_real_data_dryrun.md).
Real-data admission remains conditional on that chain.  This dry-run does
not write the official conditions database.

## Close method development

If large-statistics MC transfer **and** real-data DQ transfer are both
stable under this protocol, stop adding degrees of freedom. Next stage is
physics-production validation of the two exclusive geometries, not a new
alignment estimator.
