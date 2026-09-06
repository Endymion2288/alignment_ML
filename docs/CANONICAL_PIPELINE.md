# Canonical Pipeline

Dated navigation for the current scientific mainline.  Historical V1/MLP/V2/V3,
rank-rescue, and Frozen-V2 association remain controls.  They are not the
future estimator path.

Authoritative audit and task list:

- [PROJECT_MASTER_AUDIT.md](PROJECT_MASTER_AUDIT.md) — frozen 2026-09-05/06 review of remote master `0c8a6ca34c6c8b4f505dc153dd0ca8cc7949ce56`
- [CODE_ROADMAP.md](CODE_ROADMAP.md) — T00–T41 execution plan

Do not backfill later workbooks into that audit.  Workbooks 75–87 are
post-audit lineage recorded by T00.

## Three entries that must not be mixed

| Entry | What it consumes | What it may claim | What it must not claim |
| --- | --- | --- | --- |
| **inference** | frozen V2 / MLP / route packing on a candidate graph | association scores, route occupancy, residual/DQ after a frozen policy | a deployable alignment correction |
| **paired-response analysis** | same-event FD residual banks, WLS, rank | recovery of a known geometry difference on the same events | absolute data alignment, structural null of the detector |
| **real-data monitoring** | official geometry + frozen association | residual/DQ alerts | geometry write, mechanical correction |

T00 registers E01–E09 under these three kinds only.

## Current scientific question

```text
real measurement → field-aware global track likelihood
                 → track nuisance profiling
                 → alignment information / covariance / coverage
```

The question is how much alignment information sits in the real measurement,
not whether a larger network can fit a paired residual.

Stopped routes: V4/GNN/Transformer tuning, parameterization rescue,
tracker-only 5DoF rank-subspace search, rank-threshold rescue, sealed-test
reopen, rewriting frozen negatives.

## Provenance required on every new claim

code SHA, dirty-tree SHA, resolved-config SHA, input GUID/SHA, output SHA,
schema, seed, source/condition split, UTC, argv, job ID, closure type,
negative/supersession flag.

External Calypso/Athena/ACTS pins are recorded as provenance.  If the live
binary cannot be proven equal to the historical production, mark
`historical_runtime_unverified`.  Do not invent a commit.

## Frozen permissions

`geometry_write_allowed=false`, `real_data_alignment_authorized=false`,
`measurement_model_validated=false`, `held_out_accessed=false`.
