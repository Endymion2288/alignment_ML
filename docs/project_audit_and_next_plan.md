# Project Audit and Next-Stage Plan

Date: 2026-08-18
Auditor: AI agent (read-only audit; no new production, training, test-bank generation, or
architecture development was performed; sealed test event data was never opened — only
historical summary/contract JSONs were read for provenance).

## Executive summary

The repository is **healthy and trustworthy at its canonical core**: the full test suite
passes (157/157 in 40.5 s), the physical refit/Acts chain is real and verified, all
headline numerical claims in the workbook were reproduced exactly from the on-disk
artifacts, and the sealed-test boundary is correctly enforced by every canonical
V1/V2/Ry/multi-DoF entry point. The problems are concentrated at the edges: a committed
Kerberos credential cache (urgent), a stale README, legacy baseline entry points that can
still open sealed test files, a silent CPU fallback behind `device="auto"`, and one stale
multi-DoF corpus manifest. Scientifically, the project has proven the physical chain,
single-parameter dx/dy and Ry identifiability with truth-fixed closure, and has established
that association — not candidate loss — fails first on large translations. It has **not**
yet proven multi-DoF joint recovery, unknown-association alignment, or the iterative loop.
The immediate next step is not more ML: it is aggregating the already-completed
multi-source multi-DoF iteration-0 bank and testing whether dx stabilizes at
100 events/source.

## Environment

- LXPLUS GPU node; LCG view `/cvmfs/sft.cern.ch/lcg/views/LCG_110_cuda/x86_64-el9-gcc13-opt/setup.sh`
  (note: the actual CVMFS bundle is `x86_64-el9-gcc13-opt`, not the centos7-gcc11 pattern
  from older instructions).
- Python 3.13.11, PyTorch 2.11.0, CUDA available (Tesla T4), uproot 5.7.1.
- `pytest -q`: **157 passed, 0 failed, 0 skipped, 40.47 s** (2026-08-18).
- Git: branch `master`, clean tree, synced with origin. Only **two commits** total
  (`268d091` 2026-08-12, `eea834c` 2026-08-17); development provenance lives in
  `workbook/` and `outputs/`, not in git history.

## Repository status

| Area | Verdict |
| --- | --- |
| `alignment/` | Canonical. Payload IO, physical finite-difference Jacobian, rotation closure, route-selected update. |
| `baselines/route_assignment.py` | **Canonical route solver** (unit-capacity route packing; MILP + DP fast path). No duplicate implementation. |
| `baselines/global_assignment.py` | Legacy assignment methods (greedy/Hungarian/dustbin/Sinkhorn); `ScoreMatrix` remains shared infrastructure. |
| `baselines/multistation_assignment.py` | Legacy, superseded by `route_assignment.py`. |
| `evaluation/pairwise_metrics.py` | **Canonical calibration** (Platt/temperature). No duplicate implementation. |
| `models/`, `training/geometry_aware_transformer.py`, `training/route_aware_transformer.py` | Canonical V1/V2 stack. |
| `training/structured_assignment.py` + V3 scripts | Experimental, **suspended** (negative result). |
| `training/station_pair_thresholds.py`, `scripts/materialize_curriculum_synthetics.py`, pre-refit alignment scripts, synthetic-MC-era scripts | Legacy; keep for provenance, do not extend. |
| `tests/` | Healthy: 157 tests, all pass. |
| `docs/` | 19/19 EN-CN pairs complete; spot-checked pairs numerically in sync. |
| `README.md` / `README_cn.md` | **Stale** (see below). |
| `xcheng.cc` | **Urgent: Kerberos credential cache committed to git.** |

## Data inventory

| Corpus | Status | Verified facts |
| --- | --- | --- |
| Expanded dx/dy physical (`mc24_v3_expanded_trainval_physical_v1`) | COMPLETE | 18 sources (10 train + 8 validation); 108/108 points accepted; magnitudes {0, 0.1, 1, 5, 10, 50 mm}; 994 train / 796 validation source events; UID intersection 0; test 0; `q_over_p_mode=0`; no `failure.json`; all 450 referenced paths exist. |
| Expanded dx/dy synthetic (`mc24_v3_expanded_trainval_synthetic_v1`) | COMPLETE | 12 samples = 6 payload groups per split; UIDs 994/796, intersection 0. |
| IFT Ry physical bank (`mc24_ift_ry_expanded_trainval_physical_v1`) | COMPLETE | 234/234 points accepted; 13 points/source (9 pure Ry: 0, ±10, ±25, ±40, ±60 mrad; 4 joint ±40 mrad ⊕ ±1 mm dx/dy); single condition axis `ift_ry_mrad`; 994/796; intersection 0. |
| IFT Ry synthetic (`mc24_ift_ry_expanded_trainval_synthetic_v1`) | COMPLETE | 26 samples = 13 train + 13 validation. |
| Multi-DoF iteration-00 anchor bank (`mc24_multidof_ift_iteration00_anchor_trainval_physical_v1`) | COMPLETE on disk | 18 sources × 8 points = 144/144 refit quartets; Condor cluster 991650 all normal termination; no failures. **Aggregation closure and pooled manifest assembly not yet run.** |
| Multi-DoF joint curriculum bank (`mc24_multidof_ift_joint_curriculum_trainval_physical_v1`) | COMPLETE on disk, **manifest STALE** | 234/234 on disk (cluster 991651); `physical_corpus_manifest.json` was written at submit time and marks every point `completed=false`; the declared post-job single-process refresh never ran. **Refresh before any use.** |
| Historical sealed multidirection test (100116/100117, 19 events) | SEALED, intact | Frozen contracts complete (SHA-256-pinned checkpoint/calibration/operating point, validation-only selection, `no_test_time_calibration=true`); strictly source-disjoint from train/validation. |
| Early small corpus (`mc24_muon_2dfluka_curriculum_physical_v1`) | COMPLETE (historical) | Actually 10 sources / **97** events (59/19/19 train/val/test); the "99" in older notes was rounding. |

Event-count footnote: six sources hold 99 events and two hold 98 instead of the nominal
100 (uniform across all points of a source; totals 994/796 everywhere). This is an
input-xAOD property, not job loss.

## Experiment inventory

All numbers below were re-read from the artifacts during this audit and match the
workbook claims exactly (to the quoted precision) unless marked otherwise.

1. **Physical chain smoke (station-3 +1 mm)**: recovered [+1.0, 0.0] mm, max error
   4.87e-13 mm over 27 truth-matched pairs.
   `outputs/mc24_muon_fasernu_5events_segment_refit_station3_dx1mm_closure/closure.json`
2. **Physical capture scan (25 points, 9 magnitudes × 3 directions, strict 0.01 mm
   criterion)**: 0.1 mm captures 3/3; 1 mm and above capture 0/3; ≥500 mm rank collapse.
   `outputs/mc24_muon_fasernu_physical_capture_scan_v1/capture_scan_summary.json`
3. **Historical sealed multidirection test** (frozen MLP + unit-capacity route solver;
   recorded, not re-evaluated): complete-track efficiency 0.773/0.797/0.820/0.553/0.264/0.0
   at 0/0.1/1/5/10/50 mm; raw truth-chain recall 1.0 everywhere. This formally established
   "candidate graph retains truth, frozen association fails" at ≥5 mm.
   `outputs/mc24_muon_2dfluka_multidirection_test_route_level_v1/`
4. **V1 sealed test comparison** (recorded): all Transformer variants 0/3 capture at
   5/10/50 mm; geometry-aware improved 5/10 mm efficiency vs MLP but fake rate 0.065–0.070
   exceeded the 0.05 gate; no-context ablation beat full context.
   `outputs/geometry_aware_transformer_v1f_final_multidirection_test_v1/`
5. **Expanded-corpus controls (validation-only, 994/796 events)**:
   - MLP+route: nominal 0.9094/0.9724/fake 0.06286 (fails fake gate); 5/10/50 mm
     efficiency 0.0; candidate truth-chain recall 1.0 at all magnitudes.
     `outputs/mc24_v3_expanded_trainval_mlp_route_validation_v1/`
   - V1 four controls: all fail the gate at 5 mm; full-event context does not widen the
     capture range; no-context 10 mm efficiency 0.5760 > full-context 0.2802. Caveat found
     by this audit: the no-context capture set is {0, 1 mm} (0.1 mm purity 0.94568 < 0.95),
     not {0, 0.1, 1 mm} as the workbook table implies.
     `outputs/mc24_v3_expanded_trainval_v1_*/`
   - V2 BCE route-query: nominal 0.9163/0.9667/0.0443 (passes), 5 mm
     0.7559/0.9462/0.0725 (fails purity+fake); capture {0, 0.1, 1 mm}; selected context
     weight 0.05, unmatched penalty −0.5. **Path correction**: these numbers live in
     `outputs/mc24_v3_expanded_trainval_v2_direct_route_validation_v1` (stream
     `v2_direct_route_query`), not in `mc24_v3_expanded_trainval_v2_bce_control_v1` as
     workbook 16 states (the latter is a different run: nominal 0.7646/0.9798/0.0238).
   - V3 exact structured-margin: nominal 0.7988/0.8639/0.2523; 5 mm 0.5450/0.8294/0.2638;
     10 mm 0.2188/0.5953/0.3218; capture=false at all six magnitudes; best epoch 2. Soft
     control worse (nominal 0.7335/0.8509/0.2779). Edge-only control nominal efficiency
     0.4460. **V3 is a confirmed negative result.**
     `outputs/mc24_v3_expanded_trainval_v3_route_validation_v1/`
6. **IFT Ry physical study**:
   - State response: ±60 mrad → Δx ∓111 mm, Δtx ±0.060 — rotation genuinely enters
     refit/propagation.
   - One-shot linearization from nominal fails: +54.2246/−54.8091 mrad recovered for
     ±60 mrad (errors 5.775/5.191 mrad).
   - Local physical iteration closes: anchor ±50 mrad, probes ±45/±55, held-out ±60
     recovered as **+60.068979 / −60.412497 mrad** (errors 0.069/0.413 mrad, rank 1,
     condition 1). No truth leakage: derivatives use probe refits only; the held-out point
     never enters the fit; stations 1–3 are held at identity (atol 1e-15). Design note:
     the common truth-pair set is intersected across nominal/probe/observed evaluations
     (pair membership depends on the held-out refit; values do not).
     `outputs/mc24_ift_ry_refinement_smoke_v1/local_step_to_p60/closure.json`, `.../local_step_to_m60/closure.json`
   - Raw physical candidate complete truth-chain recall ≈ 0.95 (validation: nominal
     0.9493, −60 mrad 0.9542, +60 mrad 0.9678); 0→1 is the dominant loss pair. The
     synthetic-overlay recall of 1.0 uses a complete-truth-track denominator and must not
     be conflated with raw coverage.
     `outputs/mc24_ift_ry_expanded_trainval_physical_audit_v1/summary.json`
   - Validation controls (nominal / |Ry|=60 mrad; all capture 5/5):
     MLP+route 0.9489/0.9370 eff, 0.9888/0.9878 purity, 0.0312/0.0270 fake;
     V1 full 0.9651/0.9626, 0.9819/0.9859, 0.0295/0.0253;
     V1 no-context 0.9382/0.9429, 0.9848/0.9870, 0.0210/0.0192;
     V2 BCE 0.9355/0.9381, 0.9797/0.9811, 0.0404/0.0368.
     V2 same-checkpoint base-edge control fails (nominal fake 0.0628), so the route query
     is load-bearing for V2 here; V1 full-context has the best efficiency.
     `outputs/ift_ry_validation_control_assessment_v2/`
7. **Multi-DoF (dx+dy+Ry) 10-event joint anchor**: rank 3/3, scaled condition 511.79, but
   recovered increment (+4.683 mm, +1.330 mm, −29.931 mrad) vs target (−2.0, +1.5, −35.0);
   dx leave-one-event-out range −2.80…+9.55 mm; capture_success=false; candidate retention
   1.0. One-shot closures (a/b) show sign-flipped dx. **Multi-DoF closure is not achieved
   at 10 events.**
   `outputs/mc24_multidof_ift_iteration00_joint_a_local_step_v2/local_step.json`
8. **Contract sweep**: 85 contract/summary/closure JSONs scanned; no
   `test_events_loaded`/`test_opened`/`test_artifacts_opened=true` anywhere outside the
   documented historical frozen-test directories.

## Model inventory

| Model | Canonical config | Status |
| --- | --- | --- |
| Pairwise MLP + route | `configs/pairwise_mlp_route_ift_ry_validation.yaml` (Ry); expanded-corpus control `configs/pairwise_mlp_route_expanded_control.yaml` | Canonical baseline. Passes gate on Ry; fails on dx/dy ≥5 mm. |
| V1 geometry-aware Transformer (full / no-context / no-chi2 / ordinary sparse) | `configs/geometry_aware_transformer_v1_ift_ry_validation.yaml`; expanded controls `configs/geometry_aware_transformer_v1_expanded*_control.yaml` | Canonical Transformer. Best efficiency on Ry; no capture-range expansion shown on dx/dy. |
| V2 route-aware (BCE route-query) | `configs/geometry_aware_transformer_v2_ift_ry_validation.yaml`; `configs/geometry_aware_transformer_v2_expanded_bce_control.yaml` | Canonical route-aware model; the frozen association backbone for the alignment loop. Historical (99-event) V2 selected residual weight w=0; on expanded/Ry corpora the route query contributes (context weight 0.05). |
| V3 structured-margin | `configs/geometry_aware_transformer_v3.yaml` | **Suspended negative result.** Do not resume without a new hypothesis. |
| Chi2 / Hungarian / dustbin / Sinkhorn / multistation-flow baselines | various `physical_global_assignment_*.yaml` | Legacy. |

## Alignment inventory

- Payload semantics verified against Calypso source: per-station
  `[dx_mm, dy_mm, dz_mm, rx_rad, ry_rad, rz_rad]`, global `T·Rz·Ry·Rx`; both segment refit
  and ACTS surfaces follow the payload.
- `q_over_p_mode=0` is enforced at ~25 checkpoints across corpus build, materialization,
  training, evaluation, frozen-test, and multi-DoF entry points. Two loader-level gaps:
  the synthetic-manifest loader does not validate manifest-level `q_over_p_mode`, and
  `datasets/propagation_loader.py:127-131` silently zero-fills a missing q/p branch
  (legacy MC22 compatibility), making "absent" indistinguishable from "mode 0".
- Truth-fixed single-parameter closure: dx/dy at 5e-13 mm (small offsets); Ry via local
  iteration to <1 mrad at ±60 mrad.
- Multi-DoF (dx+dy+Ry): full rank but statistically unstable at 10 events; the
  multi-source iteration-0 bank (100 events/source) is complete and awaiting aggregation.
- Field-aware global track fit: **interface missing** — the propagation tree has no
  transport Jacobian (`supports_field_aware_global_track_fit=false`); the current update
  is correctly labelled route-consistency + WLS, not an independent global likelihood.
- Iterative loop (associate → fit → align → new payload → refit → re-associate):
  components exist (`run_frozen_association_backbone.py --payload-id`,
  `run_route_selected_multidof_update.py`, payload writer with `--update-json` guard),
  but the loop has never closed on real refits.

## Test boundary audit

- Canonical entry points (V1/V2/V3 training, refreeze, route evaluators, frozen backbone,
  Ry controls, multi-DoF chain) all pass `allowed_splits=(train, validation)`; test paths
  are never resolved. Verified in code and by contract JSONs.
- Sanctioned sealed-test readers are only the two historical frozen scans
  (`run_frozen_route_level_scan.py`, `run_frozen_geometry_aware_transformer_v1_scan.py`),
  which verify sealed sources via `_audit_sealed_samples`.
- **Gap (major)**: three legacy baseline entry points can still open sealed test ROOT
  files: `scripts/run_curriculum_mlp_baseline.py:499+577` (unconditionally),
  `scripts/run_station_pair_threshold_baseline.py:391+666-669` (`--evaluate-test`),
  `scripts/run_global_assignment_mlp_baseline.py:118+1221-1223` (default mode without
  `--validation-only`). The loader resolves all splits by default
  (`datasets/physical_curriculum.py:111-121`); protection is opt-in per call site.
  `scripts/audit_field_candidate_coverage.py` accepts `--split test` without a seal check.
- Recommendation: add a global seal guard (e.g. loader-level refusal to resolve test
  paths unless an explicit `i_am_the_frozen_test_evaluation` token is passed) and patch
  the three legacy entry points.

## Completed work

1. Real physical chain: `/Tracker/Align` payload → `SCT_ClusterContainer` →
   `SegmentFitRefit` → `SegmentsRefit` → `NtupleDumper` → mode-0 Acts — DONE and verified.
2. Tracklet export contract + ROOT schema (`faser-tracklets-v1`) — DONE.
3. dx/dy physical scan corpus (expanded, source-disjoint, 994/796) — DONE.
4. Pairwise MLP + global route assignment (unit-capacity packing) — DONE.
5. V1 Transformer + mechanism diagnostics (local decoder dominates; over-smoothing;
   no capture-range expansion) — DONE (negative/limits documented).
6. V2 route-aware objective (solver-consistent residual form) — DONE; expanded/Ry BCE
   route-query is a strong control.
7. V3 structured margin — DONE (negative result, suspended).
8. IFT Ry physical bank + candidate audit + validation controls + local-iteration
   closure — DONE.
9. Multi-DoF physical banks (anchor + joint curriculum, train/validation only) — DONE on
   disk; aggregation pending.
10. Frozen sealed multidirection test (historical, one-shot) — DONE and sealed.

## Incomplete work

1. Multi-DoF closure aggregation on the completed iteration-0 bank (next immediate step).
2. Unknown-association alignment update (truth-free route-selected update on real refits).
3. Iterative association/alignment loop closure.
4. Raw candidate coverage improvement (0→1 pair losses; ~95% raw recall).
5. Field-aware global track fit (needs Calypso transport-Jacobian export or validated
   common-state Acts repropagation API).
6. Rx/Rz/dz admission (requires per-DoF finite-difference banks + rank/condition/closure
   evidence).
7. Layer/module hierarchical alignment (reserved; no Calypso mapping yet).
8. Real-data strategy (data0 2022 IFT re-export failed provenance check in workbook 03;
   unresolved).
9. Final independent test strategy (new source-disjoint multi-DoF test bank — only after
   methodology freeze).

## Obsolete work

- Coordinate-shift / residual-shift surrogates (superseded by the physical refit chain).
- `baselines/multistation_assignment.py`, `training/station_pair_thresholds.py`,
  `scripts/materialize_curriculum_synthetics.py`, pre-refit alignment scripts,
  synthetic-MC-era scripts — retained for provenance; do not extend.
- V3 structured-margin stack — suspended negative result.
- Historical 99-event (actually 97) corpus conclusions — superseded by the expanded
  corpus where noted in the workbook.

## Known negative results

1. V3 exact structured-margin and soft-assignment control fail the gate at every
   magnitude on the expanded corpus (fake rate 0.25–0.32 at nominal).
2. Full-event Transformer context does not widen the dx/dy capture range; the no-context
   ablation is better at 10 mm.
3. One-shot large-range linearization fails for Ry at ±60 mrad (5–6 mrad error); local
   iteration is required.
4. Multi-DoF joint closure at 10 events is statistically unstable (dx sign flips).
5. Historical direct route-probability replacement is solver-inconsistent (no complete
   route ever selected); the residual form `L = L_edge + w(L_route − L_edge)` fixed it.

## Current scientific conclusions

**Established:**
- The physical misalignment chain is real end-to-end (no surrogates anywhere in the
  canonical path).
- Single-station dx/dy is identifiable and closable to numerical precision at small
  offsets with truth-fixed association.
- IFT Ry is identifiable up to at least ±60 mrad and recoverable to <1 mrad by local
  physical iteration (truth-fixed, 10 events).
- On large translations (≥5 mm) the physical candidate graph retains truth chains while
  all frozen association models fail the primary gate: **association, not candidate
  generation, is the first failure on dx/dy**.
- On Ry up to ±60 mrad, all reasonable association models pass the gate on validation:
  **Ry is not the association bottleneck**, and no Transformer capture-range expansion can
  be claimed there.
- V3 structured-margin is a confirmed dead end at current scale.

**Not established:**
- Multi-DoF joint identifiability/recovery (rank is full; statistics fail at 10 events).
- Unknown-association alignment recovery.
- Iterative convergence of the full loop.
- Raw candidate coverage sufficiency (~95%, 0→1 losses).
- Anything about real data, 6-DoF, or layer/module alignment.
- The historical sealed-test numbers are a single frozen sample; they must not be
  extrapolated.

## Current bottlenecks (ranked)

1. **Multi-DoF statistical stability / identifiability** — dx unstable at 10 events
   (condition ~512, sign flips). The completed 100-events/source iteration-0 bank is the
   immediate test.
2. **Raw physical candidate coverage (~95%, 0→1 pair)** — the ceiling for any
   unknown-association alignment.
3. **Association capture range on translations** — failure boundary between 1 and 5 mm
   on dx/dy; all current models fail ≥5 mm.
4. **Missing field-aware global-fit interface** — no transport Jacobian export; current
   update is route-consistency + WLS only.

Not bottlenecks right now: data volume for single-DoF studies (994/796 suffices at the
current gate), route-solver correctness, calibration infrastructure.

## Risk list

| # | Risk | Severity | Action |
| --- | --- | --- | --- |
| 1 | `xcheng.cc` Kerberos credential cache committed to git | **Critical (security)** | `git rm --cached xcheng.cc`, add to `.gitignore`, treat the ticket as compromised (kdestroy/re-new). |
| 2 | Legacy baseline scripts can open sealed test ROOT files | Major | Add global seal guard; patch the three entry points. |
| 3 | Silent CPU fallback behind `device="auto"` (2 helpers + ~15 defaults) | Major | Make `auto` fail loudly when CUDA is unavailable for training/inference entry points; keep explicit `cpu` for unit tests. |
| 4 | Stale joint-curriculum `physical_corpus_manifest.json` (marks complete points missing) | Major | Run the declared post-job single-process refresh before any use. |
| 5 | README stale (V3 "in production", no Ry/multi-DoF, 3 doc pairs unindexed); V3 doc header self-contradictory | Minor | Update README + V3 doc header (EN+CN). |
| 6 | Loader prefers legacy `magnitude_mm` over `condition_magnitude` when both present | Minor | Flip precedence or reject divergence. |
| 7 | `propagation_loader.py` zero-fills missing q/p branch silently | Minor | Log/flag the fill; keep legacy path explicit. |
| 8 | Git history has only 2 commits; provenance depends on workbook/outputs | Minor | Commit per stage going forward. |
| 9 | `faser_tracklet_alignment.egg-info/` tracked | Minor | Untrack; add `*.egg-info/` to `.gitignore`. |
| 10 | V2 expanded-control numbers misattributed to the wrong output directory in workbook 16 | Minor | Correct provenance (this audit records the right path). |

## Recommended canonical pipeline

```text
configs/physical_curriculum_v3_expanded_trainval.yaml        # source lists (train/val only)
  -> scripts/build_physical_curriculum_corpus.py             # physical corpus manifest
  -> scripts/materialize_pooled_curriculum_synthetics.py     # synthetic overlay manifest
  -> training: train_geometry_aware_transformer_v1.py / train_route_aware_transformer_v2.py
       (validation-only calibration + operating point; forbidden_splits=[test])
  -> frozen inference: scripts/run_frozen_association_backbone.py --payload-id ...
  -> alignment update: scripts/run_route_selected_multidof_update.py
  -> next payload: scripts/write_station_alignment_payload.py --update-json ...
  -> physical banks: prepare_multisource_multidof_iteration.py
       + submit_multisource_multidof_iteration_condor.py
       + run_multisource_refit_multidof_local_step.py (train fits, validation held-out)
```

Canonical artifacts: solver `baselines/route_assignment.py`; calibration
`evaluation/pairwise_metrics.py`; route operating point
`training/transformer_route_selection.py`; association backbone for the loop = frozen V2
BCE route-query artifact.

## Next implementation plan (strictly dependency-ordered)

**Phase 1 — Audit cleanup (this audit's follow-ups, ~small)**
1. Remove `xcheng.cc` from git tracking + `.gitignore` (`*.cc` credential pattern,
   `*.egg-info/`); untrack egg-info.
2. Refresh the stale joint-curriculum manifest (single-process, read-only refresh).
3. Patch the three legacy entry points to refuse test splits (or add the loader-level
   seal guard).
4. Make `device="auto"` fail loudly without CUDA in the two device helpers; keep
   explicit `cpu` for tests.
5. Fix README (EN+CN) and the V3 doc status header; add the three missing doc pairs to
   the index; correct the V2-control path in workbook 16's record (point to
   `mc24_v3_expanded_trainval_v2_direct_route_validation_v1`).

**Phase 2 — Multi-DoF pilot (dx+dy+Ry, already-produced data)**
6. Run `run_multisource_refit_multidof_local_step.py` on the completed iteration-0
   anchor bank (train determines the update; validation is held-out closure). Record
   rank, scaled condition number, parameter covariance/correlation, per-source and
   per-station-pair stability, candidate retention.
7. If dx is still unstable: diagnose before any new production (per-source Jacobians,
   event-count scaling, anchor choice). Do not advance the payload on a failed closure.
8. If closure passes: frozen V2 inference per physical point (`--payload-id`), then
   `run_route_selected_multidof_update.py` (truth-free), then write the iteration-1
   payload with `--update-json` and refit — closing the first real iteration.

**Phase 3 — Multi-DoF physical curriculum**
9. Only after a passing pilot: refresh the joint-curriculum manifest, audit it
   (`audit_multidof_physical_corpus.py`), materialize train/validation synthetics, and
   run the held-out joint closure (`run_multisource_multidof_physical_closure.py`).
   Misalignments stay randomly jointly sampled (already configured in
   `configs/physical_curriculum_multidof_ift_trainval.yaml`).

**Phase 4 — Association backbone comparison on multi-DoF**
10. Compare frozen MLP+route, V1 full/no-context, V2 BCE on the multi-DoF corpus with
    the existing validation-only contracts. V3 remains a negative control. No new
    architecture unless a frozen gate fails with candidate truth chains retained.

**Phase 5 — Alignment recovery**
11. Truth-association multi-parameter closure first; then unknown association.

**Phase 6 — Iterative loop**
12. Close associate → fit → align → payload → refit → re-associate; track efficiency,
    purity, fake rate, chi2, unbiased residuals, parameter error, normal matrix,
    condition number, and iteration-to-iteration convergence together.

**Phase 7 — Additional DoF**
13. Admit Rx/Rz/dz one at a time, only on finite-difference rank/condition/correlation
    and held-out closure evidence.

**Phase 8 — Hierarchy**
14. Station → layer → module only after the Calypso conditions mapping is established.

**Phase 9 — Final test**
15. Only after methodology freeze: build a new source-disjoint multi-DoF physical test
    bank and evaluate once. The historical sealed test remains untouched.
