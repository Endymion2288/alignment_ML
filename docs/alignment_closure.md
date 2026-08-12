# Truth-Fixed Station-x/y Alignment Closure

## Current Physical Closure

The required displaced-geometry local-segment refit is now implemented from
the persistent `SCT_ClusterContainer`. For the MC24 station-3 `+1 mm` payload,
the refit plus field-aware propagation recovers the injected offset with a
maximum movable-station error of `4.9e-13 mm` over 27 truth-matched pairs.
The physical chain uses all station pairs, not the historical
reference-source-only restriction. See
[Displaced-geometry segment refit](displaced_geometry_refit.md) for its
reproduction commands and limits.

## Status and Boundary

No learned association model is used here. All studies keep truth association
fixed and solve only station-level `delta_x, delta_y`, with IFT (station 0)
fixed as the reference.

The historical development has two controlled validation layers:

1. The original residual-level solver verifies the linear weighted least-squares
   objective with the explicit convention
   `r_injected = r_nominal + delta_target - delta_source`.
2. The current payload-calibrated coordinate-level study anchors the same
   convention to a real Calypso `/Tracker/Align` SQLite/POOL payload, then
   applies its verified translation to a canonical tracklet file.

Neither historical layer is a raw-hit deformed-geometry reconstruction. The
new cluster-to-segment refit resolves that limitation for the V1 rigid
translation study without returning to RDO. See
[the conditions-payload validation](condition_payload_alignment.md) for the
historical control and its implications.

## Current Payload-Calibrated Closure

The retained control writes station 3 as global `delta_x = +1.0 mm`,
`delta_y = 0.0 mm`. Its condition chain is verified in Calypso. The separate
canonical coordinate surrogate then applies:

```text
x' = x + delta_x(station)
y' = y + delta_y(station)
```

Mode-1 (MC truth q/p) propagation records are evaluated with a truth-match
fraction at least `0.99`. To keep source states consistent with the nominal
propagation records, only pairs sourced at reference station 0 are used.

| Diagnostic | Result |
| --- | ---: |
| Truth-matched pairs before reference filter | 30 |
| Reference-source pairs | 15 |
| Coordinate-increment maximum error | `8.9e-16 mm` |
| Normal-matrix rank | 6 |
| Recovered station-3 offset | `[+1.0, 0.0] mm` |
| Largest movable-station recovery error | Numerical zero |

This validates the V1 coordinate-level objective and transform sign. It does
not quantify a detector-level alignment closure.

## Current Coordinate-Level Capture Scan

With 100 random directions per magnitude, three refinement iterations, a
chi-square gate of 25, zero added noise, and `0.01 mm` tolerance, the
reference-source surrogate gives:

| Translation per movable station [mm] | Capture fraction | Mean active-pair fraction |
| ---: | ---: | ---: |
| 0, 0.1, 0.5, 1, 2, 5 | 1.00 | 1.00 |
| 10 | 0.39 | 0.624 |
| 20 | 0.05 | 0.214 |
| 50 | 0.01 | 0.082 |

These values are not a physical detector capture range. They are sensitive to
the present pair covariance and to the reference-source restriction. They only
decide whether the controlled V1 optimizer is ready for the next data-product
validation.

## Reproduction

```bash
cd /eos/home-x/xcheng/FASER
source alignment_ML/scripts/setup_environment.sh ml
cd alignment_ML

python scripts/run_payload_alignment_closure.py \
  --nominal-tracklets outputs/mc24_muon_fasernu_5events_fieldaware_v4_truthcontrol/tracklets.root \
  --observed-tracklets outputs/mc24_muon_fasernu_5events_physical_station3_dx1mm/coordinate_injected_tracklets.root \
  --propagations outputs/mc24_muon_fasernu_5events_fieldaware_v4_truthcontrol/propagations.root \
  --payload-manifest outputs/mc24_muon_fasernu_5events_physical_station3_dx1mm/payload/alignment_payload.json \
  --output-dir outputs/mc24_muon_fasernu_5events_physical_station3_dx1mm/payload_coordinate_closure_rerun

python scripts/run_payload_coordinate_capture_scan.py \
  --tracklets outputs/mc24_muon_fasernu_5events_fieldaware_v4_truthcontrol/tracklets.root \
  --propagations outputs/mc24_muon_fasernu_5events_fieldaware_v4_truthcontrol/propagations.root \
  --payload-manifest outputs/mc24_muon_fasernu_5events_physical_station3_dx1mm/payload/alignment_payload.json \
  --output-dir outputs/mc24_muon_fasernu_5events_physical_station3_dx1mm/payload_coordinate_capture_scan_rerun
```

Each command saves its resolved configuration, JSON metrics, plot, and pair or
scan diagnostics. Existing artifact paths are never overwritten.

## Next Physical Validation

The single-payload physical closure is complete. Before comparing a sequential
pipeline with a joint alignment model, repeat this cluster-to-segment refit and
truth-fixed closure over a multi-payload dx/dy capture-range scan. Only then
introduce association ambiguity, synthetic overlays, an MLP, or a Transformer.
