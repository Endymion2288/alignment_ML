# Field-Aware Muon Propagation and q/p Audit

## Scope

The current four-station control sample is MC24 100 GeV FASERnu muons:

`/eos/experiment/faser/sim/mc24/particle_gun/100012/rec/dev/FaserMC-MC24_PG_muon_fasernu_100GeV-100012-00000-00004-xAOD.root`

It contains IFT plus stations 1, 2, and 3. MC22 100 GeV electrons remain a
separate stations-1-to-3 exporter, covariance, and loader smoke test. They are
not used to establish this four-station control.

The V4 export uses `FaserActsExtrapolationTool` with the field map
`FaserFieldTable_v2.root`, loaded from the configured FASER ReleaseData area.
The state convention for the audit is global `[x, y, tx, ty]` at the stored
reference z plane; residual means target state minus propagated source state.

## Local q/p Audit

The five-event canonical file contains 24 local tracklets, with station counts
IFT/1/2/3 = 5/6/8/5. All 24 exported reconstructed q/p values are finite and
self-consistent with charge divided by the reconstructed momentum magnitude.
That is only a serialization check because both quantities originate from the
same `TrackParameters` object.

The independent MC comparison is decisive: 22 tracklets have finite truth q/p,
only 14/22 signs agree (`0.6364`), and the median local q/p uncertainty is
`223.61` times the truth q/p magnitude. The current reconstructed local q/p is
therefore **not usable as a V1 local measurement**. It may be retained as a
global-track latent parameter or for MC auditing, but it must not be used as a
deployment q/p input or as an uncertainty calibration.

## Propagation Controls

The exporter writes three explicitly labeled variants for every truth-matched
ordered station pair.

| Mode | Source state | Intended use |
| --- | --- | --- |
| 0 | Reconstructed local state and reconstructed q/p | Current deployable-shape control; its q/p covariance is not calibrated. |
| 1 | Reconstructed position/direction with MC truth q/p; q/p seed covariance suppressed | MC-only field-propagation control used by the historical coordinate-level study, not by the physical refit closure. |
| 2 | MC truth position, direction, and q/p | Separate truth-state transport diagnostic; not a local-tracklet fit. |

With truth-match fraction at least `0.99`, modes 0 and 1 each retain 30
covariance-carrying pairs. The mode-0 median 4D chi-square is `60.05`, compared
with `432.03` for a straight-line propagation. This is a useful field-aware
relative comparison, but the broad q/p covariance makes its chi-square scale
unsuitable as a calibrated physical gate. In the truth-q/p control (mode 1),
the median is `427.55`; the x residual RMS is `43.97 mm`, y residual RMS is
`1.513 mm`, and the pull RMS values for `[x, y, tx, ty]` are
`[1.035, 0.553, 1.002, 0.639]`.

Mode 2 yields 36 successful truth-state records and is useful only as a
transport sanity check: its RMS residuals are `1.859 mm`, `1.349 mm`,
`9.03e-4`, and `8.56e-4` for `[x, y, tx, ty]`. It does not turn the persisted
local tracklets into a truth-calibrated measurement.

These results validate the export and field-aware propagation data path. They
do not establish a calibrated covariance model, nominal detector alignment, or
a real-data q/p input.

## Reproduction

The retained V4 artifacts are under
`outputs/mc24_muon_fasernu_5events_fieldaware_v4_truthcontrol/`:

```bash
cd /eos/home-x/xcheng/FASER
source alignment_ML/scripts/setup_environment.sh ml
cd alignment_ML

python scripts/audit_tracklet_qoverp.py \
  outputs/mc24_muon_fasernu_5events_fieldaware_v4_truthcontrol/tracklets.root \
  --output outputs/mc24_muon_fasernu_5events_fieldaware_v4_truthcontrol/qoverp_audit.json

python scripts/evaluate_field_propagation.py \
  --tracklets outputs/mc24_muon_fasernu_5events_fieldaware_v4_truthcontrol/tracklets.root \
  --propagations outputs/mc24_muon_fasernu_5events_fieldaware_v4_truthcontrol/propagations.root \
  --q-over-p-mode 0 \
  --min-truth-match-fraction 0.99 \
  --output-dir outputs/mc24_muon_fasernu_5events_fieldaware_v4_truthcontrol/field_propagation_mode0_rerun
```

Use `--q-over-p-mode 1` only for the MC truth-q/p control. The evaluator writes
JSON metrics, per-station-pair CSV, residual/pull/chi-square plots, and accepted
pair tensors.

## Required Next Validation

Before using a chi-square gate in an association study, audit the propagated
covariance and Jacobian conventions on a larger independent muon sample. A
four-station electron MC remains valuable for particle-type and magnetic-field
generalization, but it is not a blocker for the controlled station-x/y
alignment work.
