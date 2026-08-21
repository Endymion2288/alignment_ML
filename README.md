# FASER Tracklet Alignment ML

This repository starts with deliberately small, reproducible baselines for
multi-station local-tracklet association. The ROOT input contract, physical
refit/Acts chain, and geometry-only chi-square reference were validated before
the Geometry-Aware Sparse Transformer V1 was trained. Its first sealed test is
also preserved, including its negative capture-range result.

The canonical input is a flat ROOT tree named `tracklets`, one row per local
tracklet. It requires explicit station IDs, global `(x, y, z, tx, ty)`,
the covariance of `[x, y, tx, ty]`, fit quality, hit summary, and MC truth
labels for supervised stages. Existing FASER PHYS ntuples are useful for
auditing but do not yet meet this contract.

## Quick start

```bash
cd /eos/home-x/xcheng/FASER
alignment_ML/scripts/bootstrap_ml_environment.sh  # once per LCG Python environment
source alignment_ML/scripts/setup_environment.sh ml
cd alignment_ML
python -m scripts.make_synthetic_tracklets --output data/synthetic_tracklets.root --seed 7
python -m scripts.run_chi2_baseline \
  --input data/synthetic_tracklets.root \
  --config configs/baseline_chi2.yaml \
  --output-dir outputs/synthetic_chi2
pytest -q
```

The baseline writes the resolved config, metrics JSON, match CSV, and a
candidate-chi-square plot. MC metrics distinguish raw predictions from the
subset that can be scored with unique truth labels. Use
`source .../setup_environment.sh calypso` only in a separate shell after
building Calypso. The bootstrap installs this project and its declared Python
dependencies into the LCG user site; subsequent shells only need the `source`
command above.

## Current Four-Station Control

MC24 100 GeV FASERnu muons are the current IFT+1+2+3 control sample. The
repository preserves optional q/p fields, converts field-aware
`FaserActsExtrapolationTool` pair records, validates truth-fixed residual/pull/
chi-square diagnostics, and runs station-level x/y closure. The V4 truth audit
finds that reconstructed local q/p is not usable as a V1 local measurement;
mode 1 is therefore an MC truth-q/p propagation control, not a deployable
input. Geometry-Aware Sparse Transformer V1 now consumes only the existing
mode-0 physical candidate graph and retains the existing route assignment
backend. Its validation-selected, sealed multi-direction test does not yet
demonstrate a capture-range expansion; see the dedicated V1 document before
interpreting it as an alignment result.

The validation-only Geometry-Aware Transformer V2 route-aware study likewise
does not pass its fixed primary point. Its current route context falls back to
the exact edge-only control, while raw physical truth chains remain available.
The V1 test source remains sealed and no new final test bank has been made;
see the V2 study document for the mechanism diagnosis and reproducible
validation contract.

Geometry-Aware Transformer V3 was trained and validated as a
train/validation-only structured global-assignment hypothesis on the completed
expanded physical corpus (994 train / 796 validation events, source-file
disjoint). The structured loss-augmented objective did not beat the frozen
pairwise route control on validation, so the V3 line is suspended as a
documented negative result; no new test bank was opened.

The IFT R_y physical-rotation study closes the rotational leg of the chain:
real `/Tracker/Align` rotation payloads refit through the same physical chain,
and a local alignment step recovers injected +/-60 mrad IFT rotations to better
than 1 mrad. The frozen MLP/V1/V2 route controls keep their validation primary
gates out to 60 mrad, so R_y alone is not the association bottleneck at that
scale. The current mainline is a multi-DoF (dx, dy, R_y) global alignment loop
on a multi-source iteration-0 physical bank; read the multi-DoF loop document
and the project audit before starting new work.

```bash
cd /eos/home-x/xcheng/FASER
source alignment_ML/scripts/setup_environment.sh ml
cd alignment_ML

python scripts/audit_tracklet_qoverp.py \
  outputs/mc24_muon_fasernu_5events_fieldaware_v4_truthcontrol/tracklets.root \
  --output outputs/mc24_muon_fasernu_5events_fieldaware_v4_truthcontrol/qoverp_audit_rerun.json

python scripts/evaluate_field_propagation.py \
  --tracklets outputs/mc24_muon_fasernu_5events_fieldaware_v4_truthcontrol/tracklets.root \
  --propagations outputs/mc24_muon_fasernu_5events_fieldaware_v4_truthcontrol/propagations.root \
  --q-over-p-mode 0 --min-truth-match-fraction 0.99 \
  --output-dir outputs/mc24_muon_fasernu_5events_fieldaware_v4_truthcontrol/field_propagation_mode0_rerun
```

MC22 electrons remain the stations-1-to-3 exporter/covariance/loader smoke
test. For the MC24 four-station control, a real Calypso `/Tracker/Align`
payload now drives a refit from persistent `SCT_ClusterContainer` through
`SegmentFitRefit`, `SegmentsRefit`, the tracklet exporter, and
`FaserActsExtrapolationTool`. The station-3 `+1 mm` physical closure recovers
the injected offset with a maximum error of `4.9e-13 mm` over 27 truth-matched
pairs. A completed 25-point, three-direction physical capture scan now reruns
that chain for every payload point. Under its strict `0.01 mm` recovery
criterion, all `0.1 mm` trials capture and all `1 mm` trials fail; this is a
configuration-specific control result, not a universal detector tolerance.
See the physical-capture document and saved point-level diagnostics before
interpreting the historical coordinate-level scan.

For a complete MC export, conversion, content audit, and baseline run:

```bash
cd /eos/home-x/xcheng/FASER
alignment_ML/scripts/export_mc_tracklets.sh \
  --input INPUT-xAOD.root \
  --output-dir alignment_ML/outputs/EXPERIMENT_NAME \
  --nevents 10 \
  --source-station SOURCE --target-station TARGET \
  --chi2-gate GATE_FROM_AUDIT
```

Select station IDs from `content_audit.json`; never infer them from z. Omit
the three baseline override options to use the YAML configuration. The script
uses `faser_ntuple_maker.py --export-tracklets`, writes the event-wise enhanced
ROOT file, converts it, and refuses to overwrite existing artifacts. Use
`--include-truth` only when directly invoking the converter on MC. The detailed
contract and validation limits are linked below.

## Documents

- [Input schema and exporter contract](docs/tracklet_export_contract.md)
- [Current data audit](docs/data_audit.md)
- [Baseline validation](docs/baseline_validation.md)
- [Field-aware propagation validation](docs/field_aware_propagation.md)
- [Truth-fixed alignment closure](docs/alignment_closure.md)
- [Conditions payload and coordinate-level closure](docs/condition_payload_alignment.md)
- [Displaced-geometry segment refit](docs/displaced_geometry_refit.md)
- [Physical refit capture-range scan](docs/physical_capture_scan.md)
- [Synthetic multi-track overlay](docs/synthetic_multitrack.md)
- [Physical-payload synthetic unknown-association baselines](docs/synthetic_unknown_association.md)
- [Physical misalignment-augmentation curriculum MLP baseline](docs/curriculum_mlp_baseline.md)
- [Pairwise MLP with global assignment baseline](docs/global_assignment_mlp_baseline.md)
- [Geometry-Aware Sparse Transformer V1](docs/geometry_aware_transformer_v1.md)
- [Geometry-Aware Transformer V2 mechanism diagnosis](docs/geometry_aware_transformer_v2_diagnostics.md)
- [Geometry-Aware Transformer V2 route-aware validation study](docs/geometry_aware_transformer_v2.md)
- [Geometry-Aware Transformer V3 structured global-assignment study](docs/structured_assignment_v3.md)
- [Multi-direction route-level physical scan](docs/multidirection_route_level_physical_scan.md)
- [IFT R_y physical rotation study](docs/ift_ry_physical_rotation.md)
- [Multi-DoF global alignment loop](docs/global_alignment_multidof_loop.md)
- [Project audit and next-stage plan](docs/project_audit_and_next_plan.md)
- [Four-station alignment](docs/four_station_alignment.md)
- [Chinese input schema and exporter contract](docs/tracklet_export_contract_cn.md)
- [Chinese data audit](docs/data_audit_cn.md)
- [Chinese baseline validation](docs/baseline_validation_cn.md)
- [Chinese field-aware propagation validation](docs/field_aware_propagation_cn.md)
- [Chinese truth-fixed alignment closure](docs/alignment_closure_cn.md)
- [Chinese conditions payload and coordinate-level closure](docs/condition_payload_alignment_cn.md)
- [Chinese displaced-geometry segment refit](docs/displaced_geometry_refit_cn.md)
- [Chinese physical refit capture-range scan](docs/physical_capture_scan_cn.md)
- [Chinese synthetic multi-track overlay](docs/synthetic_multitrack_cn.md)
- [Chinese physical-payload synthetic unknown-association baselines](docs/synthetic_unknown_association_cn.md)
- [Chinese physical misalignment-augmentation curriculum MLP baseline](docs/curriculum_mlp_baseline_cn.md)
- [Chinese pairwise MLP with global assignment baseline](docs/global_assignment_mlp_baseline_cn.md)
- [Chinese Geometry-Aware Sparse Transformer V1](docs/geometry_aware_transformer_v1_cn.md)
- [Chinese Geometry-Aware Transformer V2 mechanism diagnosis](docs/geometry_aware_transformer_v2_diagnostics_cn.md)
- [Chinese Geometry-Aware Transformer V2 route-aware validation study](docs/geometry_aware_transformer_v2_cn.md)
- [Chinese Geometry-Aware Transformer V3 structured global-assignment study](docs/structured_assignment_v3_cn.md)
- [Chinese multi-direction route-level physical scan](docs/multidirection_route_level_physical_scan_cn.md)
- [Chinese IFT R_y physical rotation study](docs/ift_ry_physical_rotation_cn.md)
- [Chinese multi-DoF global alignment loop](docs/global_alignment_multidof_loop_cn.md)
- [Chinese project audit and next-stage plan](docs/project_audit_and_next_plan_cn.md)
- [Chinese four-station alignment](docs/four_station_alignment_cn.md)
- [Chinese README](README_cn.md)
