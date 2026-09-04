# FASER Tracklet Alignment ML (4station branch)

This repository develops machine-learning-assisted alignment and local tracklet
association for the FASER tracking spectrometer. This branch (`4station`) is
dedicated to the four-station configuration (IFT/Station 0, Station 1,
Station 2, Station 3), solving relative geometry alignment across an admitted
15-DoF relative subspace under the complete, unapproximated physical chain:

```text
/Tracker/Align → SCT_ClusterContainer → SegmentFitRefit → SegmentsRefit → NtupleDumper
  → FaserActsExtrapolationTool (mode 0)
```

The canonical input is a flat ROOT tree named `tracklets`, one row per local
tracklet. It requires explicit station IDs, global `(x, y, z, tx, ty)`,
the covariance of `[x, y, tx, ty]`, fit quality, hit summary, and MC truth
labels for supervised stages.

## Quick Start

### Environment Setup

```bash
cd /eos/home-x/xcheng/FASER
# Initialize LCG Python user site environment (run once per LCG release)
alignment_ML_4station_branch/scripts/bootstrap_ml_environment.sh

# Activate ML environment in every new shell
source alignment_ML_4station_branch/scripts/setup_environment.sh ml
cd alignment_ML_4station_branch
```

*Note:* Calypso operations require a separate shell:
`source .../setup_environment.sh calypso` after building Calypso.

### Running Baselines and Tests

```bash
# Run unit tests (including RelativeRoute V4, solver gradients, and zero-init checks)
pytest -q

# Synthetic tracklet generator and baseline chi-square matching
python -m scripts.make_synthetic_tracklets --output data/synthetic_tracklets.root --seed 7
python -m scripts.run_chi2_baseline \
  --input data/synthetic_tracklets.root \
  --config configs/baseline_chi2.yaml \
  --output-dir outputs/synthetic_chi2
```

### RelativeRoute V4 & Head-Only Paired Training Drivers

```bash
# Audit route head solver-aware loss gradients (Mechanisms C & D, left-SE(3) gauge)
python scripts/audit_route_head_solver_gradients.py

# Model dry-run and deterministic zero-initialization verification
python scripts/dry_run_relative_route_v4.py

# Submit Head-Only paired training to HTCondor (Arm 1 Control vs Arm 2 Primary)
python scripts/submit_relative_route_v4_head_only_condor.py --submit

# Frozen Arm 0/1/2 reserved-blind development eval (Workbook 70; do not retune)
python scripts/submit_relative_route_v4_development_eval_condor.py --submit
```

## Four-Station Alignment and Research Progression

### 1. Four-Station Parameterization and 15-DoF Relative Subspace (Phase 1, Workbooks 48–50)
- **Parameterization**: Probes 20 free track-constrained DoFs (`dx, dy, rx, ry, rz` on each of S0–S3) with survey `dz` constrained by a 5 mm prior. Station 0 is not assumed to be fixed or privileged.
- **Identifiability & SVD**: Central finite-difference (FD) analysis on the physical chain reveals that the unconstrained 20-D chart has a condition number of ~8.5×10⁵ due to 5 global unconstrained gauge modes (3 translations + 2 rotations).
- **Admitted Relative Subspace**: Gauging one reference station drops the 5 gauge modes, yielding an admitted 15-DoF relative subspace (condition number ~2×10⁴). Evaluated strictly through gauge-invariant transformations $\Delta T_{ij} = T_i^{-1} T_j$, truth-selected 15-DoF relative WLS closure recovers injected misalignments with numerical precision ($4.9\times 10^{-13}\text{ mm}$).

### 2. Association Controls and Six-Source Training Diversity (Workbooks 51–64)
- **Matched Retraining on Relative Curriculum**: Matched V2 association was retrained on a 15-DoF relative curriculum, incorporating gauge-consistent route objectives, dustbin-aware route margins, and hard-aware max reduction.
- **Six-Source Diversity Training (Workbook 64)**: Trained the frozen Workbook-62 objective on six source-disjoint xAOD files.
- **Reserved-Blind Gate Evaluation**: The reserved-blind pair (`100047_00350` / `100048_00350`) failed on complete-track efficiency degradation ($\Delta\text{eff} = 0.117$, with 2→3 edge $\Delta\text{eff} = 0.101$). Classified as `source_diversity_blind_failed_other`. Under the pre-registered contract, the 15-DoF relative WLS solver remains strictly closed (`continue_to_15d_relative_wls = false`).

### 3. Reserved-Blind Failure Mechanism Localization Audit (Workbook 65)
An event-by-event, route-by-route audit on the reserved-blind dataset established:
- **Candidate Builder Recall = 100% (A = 0)**: No true edges missed by geometry/Acts propagation.
- **Edge Thresholding Intact (B = 0)**: All true edges passed the 0.001 station-pair threshold.
- **Implementation Bugs Excluded**: Solvers, thresholding, and loss chains verified deterministic.
- **Locked Root Causes**:
  1. **Mechanism C (Score Degradation / Dustbin Drop, 62.5% of loss, net 35 routes)**: Under misalignment, edge logit degradation causes $U_{\text{truth}} = \sum_e \text{logit}(p_e) - 4.0 \le 0$, causing the solver to discard true complete tracks.
  2. **Mechanism D (Set Packing Competition / Fragment Preemption, 37.5% of loss, net 21 routes)**: When $U_{\text{truth}} \in (0, 1)$, production margin collapses; when $\text{logit}(p_{2\to3}) < 1.0$, $U_{\text{complete}} < U_{\text{frag3}}$, causing the 4-station track to be preempted by 3-station fragments (0-1-2) or endpoint blockers.
  3. **Spatial Concentration**: Degradation is heavily concentrated at Station 3 (2→3 edge average logit dropped from 2.180 to 1.828).

### 4. Next-Generation RelativeRoute Transformer V4 (Workbooks 66–68B)
Designed specifically to address Mechanisms C and D without confounding backbone representation:
- **Strictly Additive Complete-Route Head**: Frozen Workbook-64 backbone and edge representation are preserved without fine-tuning:
  $$L_{\text{corrected}} = L_{\text{edge, W64}} + \Delta L_{\text{route}}, \quad \text{score} = \sigma(L_{\text{corrected}})$$
- **Zero-Initialization Contract**: The correction head is mathematically initialized to zero ($\Delta L_{\text{route}} \equiv 0$ at step 0), ensuring 100% identity with the production edge solver at initialization.
- **Two-Arm Attribution**:
  - **Arm 1 (Control)**: Absolute route representation $P_4$.
  - **Arm 2 (Primary)**: Relative coordinate and residual representation $R_4$.
- **Solver-Aware Gradient Audit**: Tie-break audited to $1.0\times 10^{-9}$. Solver-aware gradients (Mechanism C margin/dustbin, Mechanism D packing margin, and left-SE(3) gauge invariance) verified through `split_graph_route_logits`.
- **Corrigendum (Workbook 68B)**: Fixed frozen production edge wiring, additive $L_{\text{corrected}}$, and checkpoint artifact saving interface.

### 5. Current State: Head-Only Development Eval Closed as Gate Failure (Workbooks 69–70)
- Workbook 69 Arm 1/2 last-epoch checkpoints exist (`1104860`/`1104861`, return 0). Canonical freeze files are `checkpoint_last.pt`.
- Arm 1 SHA256 `e8a6d6c6e26545de91d9ded59c7b4f9b7840de2e9c94b88d56e1857d6aebd3c8`; Arm 2 SHA256 `a52037ead07555ef937a7e02b65adc9526b509e765341e8af31992552f1fdd3a`.
- Workbook 70 reserved-blind development eval (`1108310`, return 0) **failed the Workbook-64 gates on all three arms**. Arm 0 C/D replay remains C=200, D=170. Arm 1/2 did not recover complete-track efficiency versus Arm 0 (reference 0.933 → 0.917 → 0.888). Mechanism C rose 200 → 327 → 441.
- The reserved-blind overlay `00350_00399` is now development-seen. The unused final-blind pair `00800_00849` and sealed test remain closed. Do not retune after seeing development.
- 15-DoF relative WLS remains strictly held (`continue_to_15d_relative_wls = false`). No next training or final-blind eval is authorized.

## Core Operational Invariants

In accordance with [FASER Alignment Operating Protocol V1](docs/faser_alignment_operating_protocol_v1.md):
- **Canonical Propagation**: Exclusively mode-0 field-aware Acts propagation.
- **Physical Chain Integrity**: Geometry is updated exclusively via `/Tracker/Align` payloads driving persistent cluster refits; coordinate or residual surrogates are forbidden.
- **Sealed Test Isolation**: Final test banks are never opened during development or tuning.
- **Identifiability Discipline**: Alignment parameters must be admitted via physical sensitivity and SVD; residual reduction alone is never accepted as proof of alignment success.

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
- [FASER alignment operating protocol V1](docs/faser_alignment_operating_protocol_v1.md)
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
- [Chinese FASER alignment operating protocol V1](docs/faser_alignment_operating_protocol_v1_cn.md)
- [Chinese four-station alignment](docs/four_station_alignment_cn.md)
- [Chinese README](README_cn.md)
