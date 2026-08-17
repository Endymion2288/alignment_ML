# Source-Disjoint Multi-DoF Global Alignment Loop

## Scope

This is the physical control path for FASER global association and alignment.
It adds no new Transformer architecture. The association backbone remains
frozen, with the sealed V2 BCE route-query artifact as the usable control.
The permanently sealed test bank is never read.

The initial active block is IFT/station-0 `dx`, `dy`, and `Ry`; stations
S1--S3 define the reference frame. Calypso payloads use
`[dx, dy, dz, Rx, Ry, Rz]` in mm/rad, while reports use mrad for rotations.

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

The payload/Jacobian code supports all station rigid components, but `dz`,
`Rx`, and `Rz` remain inactive. Admit one only after a real finite-difference
bank demonstrates full rank, acceptable scaled conditioning/correlation,
source and station-pair stability, and held-out physical closure. Station,
layer, and module hierarchy levels are reserved; a layer/module condition must
not be silently mapped onto a station payload.

## Current Status

The earlier 10-event joint anchor test had rank three but unstable `dx`, so it
does not update geometry. The multi-source iteration-0 physical bank has been
submitted to Condor; conclusions wait for every real refit and mode-0 Acts
export to pass its completion gate.
