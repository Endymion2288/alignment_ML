# Tracklet Independent-Failure Provenance Audit V1

Workbook 72 / 2026-09-03. Workbook 69 remains frozen as
`cross_source_stable_core_independent_validation_fail`. This audit
does not reopen that campaign, retune `S` or `rank_tolerance=0.01`,
drop the rank 2/3/4 sources, force rank 5, or treat those sources as
cluster-local confirmation.

It is a read-only classification of coverage loss versus pipeline or
artifact corruption for:

- `mc24_100043_00400_00499` (rank 2)
- `mc24_100043_00500_00599` (rank 3)
- `mc24_100044_00200_00299` (rank 4)

Pre-registered sibling controls from the same independent-validation
list that already passed workbook 69, used only as a read-only
comparison:

- `mc24_100043_00300_00399`
- `mc24_100044_00400_00499`

It does not retrain V2/V3/Transformer, change the frozen
pairwise/route policy, run Newton, write geometry, or emit an
alignment payload. Real data stays `residual_dq_monitoring_only`.
Sealed test is not opened. Athena is not rerun.

## Frozen contract

- Real data remains `residual_dq_monitoring_only`.
- `geometry_write_allowed=false`.
- Frozen V2 checkpoint SHA256
  `0c85a001677678163512c59bd5ceb6d08bdefaab79c0f4628659a4a8ad766a27`.
- Inherited workbook-69 decision
  `cross_source_stable_core_independent_validation_fail`.
- Form `A = W^{1/2} J S` with frozen
  `S = (5, 5, 5, 60, 60, 60, 0.12)` and frozen
  `rank_tolerance = 0.01`. Native frozen-rank SVD; do not truncate
  to 5-D.
- Severe artifact flags are missing FD probes, plus/minus population
  misalignment, non-finite residuals, sealed-test access, and
  held-out points mixed into the SVD. Those classify as pipeline
  corruption.
- Ill-conditioned pair covariance is a warning. It is not by itself
  corruption if sibling rank-5 sources share it.
- Sources are not dropped. Rank is not forced to 5. The stable-core
  criterion is not redefined. These sources are not a cluster-local
  confirmatory set.

## Decision

`tracklet_independent_failure_provenance_audit_complete_sources_not_dropped`

| gate | value |
| --- | --- |
| overall classification | **normal_physics_or_coverage_loss** |
| sources dropped | **false** |
| rank forced to five | **false** |
| stable-core criterion redefined | **false** |
| used as cluster-local confirmation | **false** |
| workbook-69 decision | **still frozen** |
| geometry write | **false** |

Plus/minus probes are complete, populations align, residuals are
finite, no Jacobian column is near zero, held-out points were not
loaded, and sealed test was not opened. Native ranks 2 / 3 / 4
reproduce. The shared `ill_conditioned_pair_covariance` flag also
appears on both rank-5 siblings, so it does not mark pipeline
corruption.

## Results

Config
`configs/tracklet_independent_failure_provenance_audit_v1.yaml`,
SHA256
`b73e35148e7ee8b31a579db741d1a9d36a0571f2df80215e3b16f10f432ee410`.

| source | pairs | official rank | classification |
| --- | ---: | ---: | --- |
| `mc24_100043_00400_00499` | 240 | **2** | coverage loss |
| `mc24_100043_00500_00599` | 234 | **3** | coverage loss |
| `mc24_100044_00200_00299` | 234 | **4** | coverage loss |
| sibling `mc24_100043_00300_00399` | 237 | 5 | read-only control |
| sibling `mc24_100044_00400_00499` | 235 | 5 | read-only control |

The low ranks come from the frozen relative cut on `A`, not from
missing probes. Station coverage remains IFT→1/2/3.

## Commands

```bash
source scripts/setup_environment.sh ml
pytest -q tests/test_tracklet_independent_failure_provenance.py
python scripts/report_tracklet_independent_failure_provenance.py
```

CUDA `True` (Tesla T4). Related tests included in the 40 passed. No
Condor job.

Reports:
`outputs/tracklet_independent_failure_provenance_audit_v1/`.
Recorded HEAD `a1fd01bb5c2c5daf432c13db3b021feb42bdf6d1`.

## Next stage

Keep residual DQ monitoring. Do not recover rank 5 by dropping
sources or retuning thresholds. Do not merge these sources into the
cluster-local campaign.
