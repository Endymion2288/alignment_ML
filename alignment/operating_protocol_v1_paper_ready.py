"""Publication analysis of the frozen Operating Protocol V1 closure.

Read-only.  Does not reopen Station, reduced Station, or C_dx Mode.
Does not retune V2 or alarms.  Residual or chi-squared decrease is a
DQ observable only.  Implied C_dx is a rejection diagnostic, not a
measurement.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from alignment.operating_protocol_v1_final_closure import (
    FROZEN_V2_CHECKPOINT_SHA256,
    OPERATING_MODE,
    RESIDUAL_DECREASE_LABEL,
    evaluate_unlock,
    project_root,
    unlock_criteria,
    walk_forbidden,
)

SCHEMA_VERSION = "faser-operating-protocol-v1-paper-ready-validation"
DEFAULT_CONFIG_RELATIVE = (
    "configs/operating_protocol_v1_paper_ready_validation_v1.yaml"
)
DECISION_FLOW_TEXT = (
    "MC transfer PASS → real-data association PASS → "
    "Station calibration REJECT → reduced calibration REJECT → "
    "residual/DQ monitoring PASS"
)

FORBIDDEN_POSITIVE_CLAIMS = (
    "is alignment closure",
    "as alignment closure",
    "alignment success",
    "measured c_dx",
    "measured |c_dx|",
    "c_dx measurement",
    "geometry write is unfinished",
    "incomplete because no geometry write",
)

CONFIG_MUST_BE_FALSE = (
    "geometry_write_allowed",
    "station_calibration_mode_available",
    "cdx_mode_allowed",
    "alignment_payload_from_self_nulling_residuals",
)
CONFIG_MUST_BE_TRUE = (
    "frozen",
    "do_not_construct_station_calibration_mode",
    "do_not_construct_reduced_station_mode",
    "do_not_enter_cdx_mode",
    "do_not_retrain_v2",
    "do_not_reestimate_alarm_thresholds",
    "do_not_search_geometry_write_subset",
    "do_not_generate_fd_probes",
    "do_not_run_newton",
    "do_not_write_payload",
    "residual_reduction_is_not_alignment_success",
    "implied_cdx_is_not_a_measurement",
)


def load_paper_config(path: str | None = None) -> dict[str, Any]:
    source = Path(path).expanduser().resolve() if path is not None else (
        project_root() / DEFAULT_CONFIG_RELATIVE
    )
    with source.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"paper-ready config must be a mapping: {source}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unexpected paper-ready schema: {source}")
    if str(payload.get("residual_decrease_label")) != RESIDUAL_DECREASE_LABEL:
        raise ValueError("residual decrease must stay labeled 'DQ observable'")
    for key in CONFIG_MUST_BE_FALSE:
        if payload.get(key) is not False:
            raise ValueError(f"paper-ready config must set {key}=false")
    for key in CONFIG_MUST_BE_TRUE:
        if payload.get(key) is not True:
            raise ValueError(f"paper-ready config must set {key}=true")
    return {"path": str(source), **dict(payload)}


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON mapping: {path}")
    return dict(payload)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_frozen(config: Mapping[str, Any], key: str) -> Path:
    path = Path(str(config[key]))
    if not path.is_absolute():
        path = project_root() / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_frozen_package(config: Mapping[str, Any]) -> dict[str, Any]:
    chart_root = resolve_frozen(config, "chart_data").parent
    return {
        "final_report": read_json(resolve_frozen(config, "final_report")),
        "evidence_matrix": read_json(resolve_frozen(config, "evidence_matrix")),
        "manifest": read_json(resolve_frozen(config, "reproducibility_manifest")),
        "unlock": read_json(resolve_frozen(config, "unlock_criteria")),
        "chart_data": read_json(resolve_frozen(config, "chart_data")),
        "scaling_rows": read_csv(chart_root / "selected_route_scaling.csv"),
        "composition_rows": read_csv(chart_root / "route_composition.csv"),
        "timeseries_rows": read_csv(chart_root / "dy_rx_robust_z_timeseries.csv"),
        "spectrum_rows": read_csv(chart_root / "station_jacobian_singular_spectrum.csv"),
        "reduced_rows": read_csv(chart_root / "reduced_mode_cross_run_inconsistency.csv"),
        "a_rows": read_csv(chart_root / "frozen_A_versus_station_weak_direction.csv"),
    }


def campaign_allows_geometry_write(_decision: str = "") -> bool:
    return False


def three_layer_claims() -> list[dict[str, Any]]:
    return [
        {
            "layer": 1,
            "id": "frozen_v2_association_transfer",
            "title": "Frozen V2 association transfers from MC to real data",
            "status": "PASS",
            "proves": (
                "Frozen V2 association is transferable from MC to 2024 r0022 "
                "data.  The empty 100-event selected graph is statistics-limited, "
                "and full-segment route acceptance recovers."
            ),
            "does_not_prove": (
                "That a recoverable association graph is a solvable station "
                "alignment problem, or that residual decrease is closure."
            ),
            "decision_flow_node": "real-data association PASS",
        },
        {
            "layer": 2,
            "id": "association_is_not_alignment",
            "title": "Association usable is not alignment solvable",
            "status": "REJECT_GEOMETRY_WRITE",
            "proves": (
                "The real-data Station Jacobian has a gauge-like dz weak "
                "direction.  The 14974 5-DoF weak direction coincides with the "
                "frozen C_dx→station leakage direction A.  Reduced {dy,rx,rz} "
                "is Fisher-identifiable but does not transfer between independent "
                "calibration runs.  Geometry write is rejected by identifiability "
                "and cross-level ambiguity, not by a conservative threshold."
            ),
            "does_not_prove": (
                "A numerical value of C_dx, a transferable station correction, "
                "or that a chi-squared drop proves the geometry has been corrected."
            ),
            "decision_flow_node": "Station calibration REJECT / reduced calibration REJECT",
        },
        {
            "layer": 3,
            "id": "residual_dq_monitoring",
            "title": "Current-geometry frozen-V2 residual/DQ monitoring is stable",
            "status": "PASS",
            "proves": (
                "Under current official geometry, frozen V2 residual/DQ "
                "monitoring is stable on independent r0022 runs.  The seven "
                "new runs are nominal.  There is no persistent dy/rx drift.  "
                "Run 14977 is insufficient statistics, not an alignment anomaly."
            ),
            "does_not_prove": (
                "A geometry update, a survey trigger, or that monitoring "
                "residuals can be inverted into an alignment payload."
            ),
            "decision_flow_node": "residual/DQ monitoring PASS",
        },
    ]


def supported_claims() -> list[dict[str, Any]]:
    return [
        {
            "id": "mc_transfer_pass",
            "layer": 1,
            "claim": (
                "On source-disjoint MC, exclusive Station and IFT-Internal "
                "modes both independently close.  Frozen A is not updated."
            ),
        },
        {
            "id": "real_data_association_statistics_limited_then_recovered",
            "layer": 1,
            "claim": (
                "At 100 events the selected graph is empty while the all-pairs "
                "candidate graph is not.  Full remaining segments recover "
                "selected-route counts 121/109/156/63/2 on 14973–14977, and "
                "233/193/114/77/74/91/32 on the seven independent expansion runs."
            ),
        },
        {
            "id": "station_jacobian_dz_gauge",
            "layer": 2,
            "claim": (
                "The six-DoF Station Jacobian has a near-null direction that is "
                "almost pure dz (gauge-like).  More events of the same topology "
                "do not lift that direction."
            ),
        },
        {
            "id": "weak_direction_aligns_with_A",
            "layer": 2,
            "claim": (
                "After dropping survey dz, the 14974 five-DoF weak direction is "
                "almost pure dx and has cosine 0.994 with frozen A.  This is "
                "cross-level leakage geometry, not a C_dx measurement."
            ),
        },
        {
            "id": "reduced_mode_not_transferable",
            "layer": 2,
            "claim": (
                "Reduced {dy,rx,rz} is full rank (cond. 73/86) and nearly "
                "orthogonal to A, but the self-nulling correction changes sign "
                "and magnitude between 14973 and 14974 (max ~258σ).  "
                "Linearized 14974→14973 chi-squared ratio ~16 is a DQ "
                "observable only."
            ),
        },
        {
            "id": "geometry_write_rejected_by_physics",
            "layer": 2,
            "claim": (
                "Not writing geometry is a validated scientific conclusion: "
                "the observed real-data topology cannot isolate a transferable "
                "station update from C_dx leakage.  It is not an unfinished "
                "analysis and is not caused by an overly tight alarm threshold."
            ),
        },
        {
            "id": "monitoring_nominal_on_independent_runs",
            "layer": 3,
            "claim": (
                "Seven independent r0022 runs are all nominal_monitoring under "
                "the frozen protocol.  Isolation |robust z| stays well below 3.  "
                "alignment_drift_candidate remains false."
            ),
        },
        {
            "id": "low_stats_is_insufficient_not_anomaly",
            "layer": 3,
            "claim": (
                "Run 14977 has two selected routes and is classified "
                "insufficient_statistics_for_alignment_dq, not an alignment "
                "anomaly and not a detector-condition change."
            ),
        },
    ]


def unsupported_claims() -> list[dict[str, Any]]:
    return [
        {
            "id": "no_real_data_station_geometry_update",
            "claim": "The present real-data sample supports a station geometry write.",
        },
        {
            "id": "no_cdx_measurement",
            "claim": (
                "Implied |C_dx| from a self-nulling station solve can be quoted "
                "as the true layer contrast."
            ),
        },
        {
            "id": "no_residual_as_closure",
            "claim": (
                "A residual or linearized chi-squared decrease can be read as "
                "proof that the station geometry has been corrected."
            ),
        },
        {
            "id": "no_reduced_mode_payload",
            "claim": (
                "Because {dy,rx,rz} is Fisher-identifiable it may be written "
                "as a reduced station correction."
            ),
        },
        {
            "id": "no_drift_to_geometry",
            "claim": (
                "A future alignment_drift_candidate may be inverted into a "
                "geometry payload without new external constraints."
            ),
        },
        {
            "id": "no_v2_retune_needed",
            "claim": (
                "The 100-event empty selected graph requires retuning V2 "
                "thresholds or unmatched penalty."
            ),
        },
        {
            "id": "no_subset_rescue",
            "claim": (
                "A quieter subset of the same r0022 topology can rescue "
                "geometry_write_allowed."
            ),
        },
    ]


def limitations() -> list[str]:
    return [
        "Complete four-station routes remain rare; the Station solve is dominated by 2- and 3-station routes.",
        "No independent survey or external station constraint is available in this corpus to fix dx/ry/dz.",
        "True |C_dx| on these real runs is not measured; only the MC isolation budget 1.5–1.7 µm is known.",
        "Time order uses LHC fill, then run, then skip.  This corpus has no unix timestamp in occupancy.",
        "14977 is too short for alignment DQ and is kept only as a statistics-control example.",
        "The sealed test split 100116/100117 is never opened.",
        "Workbook 03 still blocks unverified 2022 data0 IFT PHYS/xAOD re-export.",
    ]


def why_not_writing_geometry_is_a_result() -> dict[str, Any]:
    return {
        "statement": (
            "Not writing geometry is a real-data scientific result, not an "
            "unfinished state."
        ),
        "reasons": [
            "Association recovery at full-segment scale shows the graph is usable, so the failure is not empty data or a broken V2.",
            "The six-DoF Jacobian is empirically degenerate along dz.",
            "The remaining five-DoF weak direction on 14974 coincides with frozen A, so a track-driven station update is cross-level contaminated.",
            "The only Fisher-safe reduced subspace {dy,rx,rz} does not transfer between independent calibration runs.",
            "Modes that still float dx or ry exceed the 1.5–1.7 µm isolation budget; more events cannot rescue that leakage.",
            "Independent monitoring runs stay nominal at current official geometry, so there is no empirical need to force a write.",
        ],
        "not_caused_by": [
            "an overly conservative alarm threshold",
            "an unfinished software path",
            "failure to search a quieter event subset",
        ],
        "geometry_write_allowed": False,
        "real_data_operating_mode": OPERATING_MODE,
    }


def assert_no_forbidden_positive_claims(text: str, *, where: str) -> None:
    lowered = text.lower()
    negations = ("not ", "never ", "cannot ", "mustn't ", "do not ", "don't ", "不是")
    for phrase in FORBIDDEN_POSITIVE_CLAIMS:
        start = 0
        while True:
            index = lowered.find(phrase, start)
            if index < 0:
                break
            window = lowered[max(0, index - 28) : index]
            if not any(token in window for token in negations):
                raise ValueError(f"{where} contains forbidden positive claim {phrase!r}")
            start = index + 1


def build_main_claims(frozen: Mapping[str, Any]) -> dict[str, Any]:
    final = frozen["final_report"]
    unlock = frozen["unlock"]
    if final.get("frozen_v2_checkpoint_sha256") != FROZEN_V2_CHECKPOINT_SHA256:
        raise ValueError("paper package must keep the frozen V2 SHA256")
    if final.get("geometry_write_allowed") is not False:
        raise ValueError("paper package must not reopen geometry write")
    if unlock.get("currently_met") is not False:
        raise ValueError("paper package must keep V2 unlock unmet")
    payload = {
        "schema_version": f"{SCHEMA_VERSION}-claims",
        "protocol_version": "operating_protocol_v1",
        "real_data_operating_mode": OPERATING_MODE,
        "geometry_write_allowed": False,
        "station_calibration_mode_available": False,
        "cdx_mode_allowed": False,
        "alignment_drift_candidate": False,
        "alignment_payload_from_self_nulling_residuals": False,
        "residual_decrease_label": RESIDUAL_DECREASE_LABEL,
        "implied_cdx_is_not_a_measurement": True,
        "residual_reduction_is_not_alignment_success": True,
        "decision_flow": DECISION_FLOW_TEXT,
        "layers": three_layer_claims(),
        "supported_claims": supported_claims(),
        "unsupported_claims": unsupported_claims(),
        "limitations": limitations(),
        "why_not_writing_geometry_is_a_result": why_not_writing_geometry_is_a_result(),
        "unlock_criteria": unlock_criteria(),
        "unlock_evaluation": evaluate_unlock(),
        "frozen_v2_checkpoint_sha256": FROZEN_V2_CHECKPOINT_SHA256,
        "source_closure_root": "outputs/operating_protocol_v1_final_real_data_closure_v1",
        "read_only": True,
        "do_not_search_geometry_write_subset": True,
    }
    walk_forbidden(payload, where="main_claims_and_limitations")
    return payload


def figure_specs() -> list[dict[str, str]]:
    return [
        {
            "id": "fig00_decision_flow",
            "file_stem": "fig00_decision_flow",
            "title": "Operating Protocol V1 decision flow",
        },
        {
            "id": "fig01_route_acceptance_scaling",
            "file_stem": "fig01_route_acceptance_scaling",
            "title": "Selected-route acceptance versus event statistics",
        },
        {
            "id": "fig02_route_composition",
            "file_stem": "fig02_route_composition",
            "title": "2/3/4-station selected-route composition",
        },
        {
            "id": "fig03_dy_rx_robust_z",
            "file_stem": "fig03_dy_rx_robust_z_timeseries",
            "title": "Isolation dy/rx robust-z versus LHC fill",
        },
        {
            "id": "fig04_jacobian_spectrum",
            "file_stem": "fig04_station_jacobian_singular_spectrum",
            "title": "Station Jacobian singular spectrum",
        },
        {
            "id": "fig05_reduced_inconsistency",
            "file_stem": "fig05_reduced_mode_cross_run_inconsistency",
            "title": "Reduced-mode cross-run inconsistency",
        },
        {
            "id": "fig06_a_versus_weak",
            "file_stem": "fig06_a_versus_station_weak_direction",
            "title": "Frozen A versus station weak direction",
        },
    ]


def figure_captions() -> list[dict[str, str]]:
    return [
        {
            "id": "fig00_decision_flow",
            "en": (
                "Figure 0. Decision flow of Operating Protocol V1 on 2024 r0022 "
                "data.  Frozen V2 association transfers from MC and recovers at "
                "full-segment statistics (PASS).  Usable association is not a "
                "solvable station alignment: full-segment Station calibration is "
                "rejected by physical non-identifiability and cross-level "
                "contamination, and the reduced {dy,rx,rz} mode is rejected by "
                "cross-run non-transferability.  The surviving operation is "
                "current-geometry residual/DQ monitoring, which passes on "
                "independent runs.  Residual or χ² decrease is never treated as "
                "alignment closure."
            ),
            "cn": (
                "图 0. Operating Protocol V1 在 2024 r0022 上的决策流。冻结 V2 "
                "关联从 MC 迁移到真实数据，并在 full-segment 统计下恢复（PASS）。"
                "关联可用并不等于 station alignment 可解：全段 Station calibration "
                "因物理不可识别与跨层污染被否决，reduced {dy,rx,rz} 因跨 run "
                "不可搬运被否决。幸存路径是当前官方 geometry 下的 residual/DQ "
                "monitoring，并在独立 run 上通过。Residual / χ² 下降从不作为 "
                "alignment closure。"
            ),
        },
        {
            "id": "fig01_route_acceptance_scaling",
            "en": (
                "Figure 1. Frozen-V2 selected-route count versus reconstructed "
                "event statistics.  Every 100-event window has a nonempty "
                "all-pairs candidate graph but zero selected routes "
                "(statistics-limited).  Acceptance recovers on the full remaining "
                "segment for 14973–14976 and on seven independent expansion runs.  "
                "14977 saturates at two selected routes because the remaining "
                "segment is only 1687 events.  This is association transfer, not "
                "an alignment update."
            ),
            "cn": (
                "图 1. 冻结 V2 selected-route 数随重建事例统计的标度。所有 "
                "100-event 窗口的 all-pairs 候选图非空，但 selected 图为空"
                "（statistics-limited）。14973–14976 的 full remaining segment "
                "以及 7 个独立 expansion run 上 acceptance 恢复。14977 停在 2 "
                "条 selected route，因为剩余 segment 只有 1687 个事例。这是关联 "
                "迁移，不是 alignment 更新。"
            ),
        },
        {
            "id": "fig02_route_composition",
            "en": (
                "Figure 2. Composition of frozen-V2 selected routes by endpoint "
                "length.  The sample is dominated by 2- and 3-station routes; "
                "complete 4-station routes remain rare.  14977 (two 3-station "
                "routes) is insufficient statistics for alignment DQ, not an "
                "alignment anomaly.  Route composition is a data-quality "
                "observable."
            ),
            "cn": (
                "图 2. 冻结 V2 selected route 按端点站数的组成。样本以 2 站和 3 "
                "站 route 为主，完整 4 站 route 仍然很少。14977（两条 3 站 "
                "route）属于 alignment DQ 统计不足，不是 alignment 异常。Route "
                "组成是数据质量观测量。"
            ),
        },
        {
            "id": "fig03_dy_rx_robust_z",
            "en": (
                "Figure 3. Isolation residual robust-z for dy (residual_y) and "
                "rx (residual_ty) versus LHC fill, using the frozen 14973/14974 "
                "reference scale.  All statistically sufficient runs stay well "
                "inside |z|<3.  There is no persistent same-sign drift and no "
                "alignment_drift_candidate.  14977 is drawn hollow because it is "
                "insufficient statistics, not a detector-condition change.  "
                "These residuals are DQ observables only and are not inverted "
                "into a geometry correction."
            ),
            "cn": (
                "图 3. Isolation residual 的 dy（residual_y）和 rx（residual_ty） "
                "robust-z 随 LHC fill 的变化，参考尺度冻结自 14973/14974。所有"
                "统计充分的 run 都远低于 |z|<3。没有持续同号漂移，也没有 "
                "alignment_drift_candidate。14977 用空心点表示，因为它是统计不足，"
                "不是探测器工况变化。这些 residual 只是 DQ observable，不会被反演"
                "成 geometry 改正。"
            ),
        },
        {
            "id": "fig04_jacobian_spectrum",
            "en": (
                "Figure 4. Singular spectrum of the real-data Station Jacobian "
                "rebuilt from the already-captured finite-difference probes on "
                "14973 and 14974.  The six-DoF spectrum has a near-null value "
                "whose direction is almost pure dz (gauge-like).  Dropping dz "
                "lifts the 14973 condition number to 114, but the 14974 five-DoF "
                "spectrum remains weak along a direction that is almost pure dx.  "
                "This is an identifiability diagnostic, not a closure test.  "
                "Implied C_dx is not plotted as a measurement."
            ),
            "cn": (
                "图 4. 由 14973/14974 已捕获有限差分探针重建的真实数据 Station "
                "Jacobian 奇异谱。六自由度谱存在接近零的奇异值，方向几乎是纯 "
                "dz（规范型）。去掉 dz 后 14973 条件数降到 114，但 14974 的五自由"
                "度谱仍沿几乎纯 dx 的方向偏弱。这是可识别性诊断，不是 closure。"
                "Implied C_dx 不作为测量值绘制。"
            ),
        },
        {
            "id": "fig05_reduced_inconsistency",
            "en": (
                "Figure 5. Self-nulling parameter updates of the reduced "
                "{dy,rx,rz} mode on the two independent calibration runs.  The "
                "mode is full rank and nearly orthogonal to frozen A, but the "
                "updates disagree in sign and magnitude (dy +1.08 mm versus "
                "−8.93 mm; rz −4.3 mrad versus +95 mrad; maximum separation "
                "~258σ).  The linearized 14974→14973 χ² ratio is ~16.  Any χ² "
                "decrease in the opposite direction is a DQ observable only and "
                "is not alignment closure.  The mode is therefore not a "
                "writable geometry correction."
            ),
            "cn": (
                "图 5. reduced {dy,rx,rz} 模式在两个独立 calibration run 上的 "
                "self-nulling 参数更新。该模式满秩且几乎与冻结 A 正交，但更新的"
                "符号和量级不一致（dy +1.08 mm 对 −8.93 mm；rz −4.3 mrad 对 "
                "+95 mrad；最大约 258σ）。14974→14973 线性化 χ² 比约 16。相反"
                "方向上的任何 χ² 下降都只是 DQ observable，不是 alignment "
                "closure。因此该模式不能写成 geometry 改正。"
            ),
        },
        {
            "id": "fig06_a_versus_weak",
            "en": (
                "Figure 6. Geometry of the frozen C_dx→station leakage operator "
                "A and the real-data Station weak directions.  Left: after "
                "dropping dz, the 14974 five-DoF weakest right-singular vector "
                "is almost pure dx and has cosine 0.994 with unit-normalized A.  "
                "Right: the six-DoF weakest direction on both calibration runs "
                "is almost pure dz.  The alignment of the remaining weak "
                "direction with A is the empirical reason a track-driven station "
                "update is cross-level contaminated.  A is not retuned, and the "
                "figure is not a C_dx measurement."
            ),
            "cn": (
                "图 6. 冻结 C_dx→station 泄漏算子 A 与真实数据 Station 弱方向的"
                "几何关系。左：去掉 dz 后，14974 五自由度最弱右奇异向量几乎是纯 "
                "dx，与单位化 A 的余弦为 0.994。右：两个 calibration run 的六自由"
                "度最弱方向几乎都是纯 dz。剩余弱方向与 A 对齐，是 track-driven "
                "station 更新被跨层污染的实证原因。A 不重调，本图也不是 C_dx "
                "测量。"
            ),
        },
    ]


def reviewer_risks() -> list[dict[str, str]]:
    return [
        {
            "id": "empty_100_event_looks_like_v2_failure",
            "risk": "A reviewer may read the empty 100-event selected graph as a V2 failure.",
            "response": (
                "The all-pairs candidate graph is already nonempty at 100 events.  "
                "Selected routes appear as soon as the same frozen protocol sees "
                "O(10^3)–O(10^4) events and recover at full segment.  Thresholds "
                "were not retuned."
            ),
        },
        {
            "id": "residual_drop_as_success",
            "risk": "A reviewer may treat a residual or χ² decrease as proof that the geometry has been corrected.",
            "response": (
                "Every residual/χ² change in this package is labeled DQ observable.  "
                "The reduced-mode 14974→14973 transfer worsens χ² by ×16, so even "
                "the same estimator is not a transferable correction."
            ),
        },
        {
            "id": "implied_cdx_as_measurement",
            "risk": "A reviewer may quote implied |C_dx| as a measured layer contrast.",
            "response": (
                "Implied |C_dx| is only a contamination diagnostic against the MC "
                "1.5–1.7 µm isolation budget.  No real-data C_dx Mode is opened."
            ),
        },
        {
            "id": "fisher_identifiable_should_be_written",
            "risk": "A reviewer may argue that identifiable {dy,rx,rz} should be written.",
            "response": (
                "Fisher identifiability is necessary, not sufficient.  The two "
                "independent calibration runs produce opposite-sign, hundreds-of-σ "
                "inconsistent updates.  A non-transferable correction is not a "
                "geometry write."
            ),
        },
        {
            "id": "not_writing_looks_unfinished",
            "risk": "A reviewer may call geometry_write_allowed=false an incomplete result.",
            "response": (
                "The paper result is the negative: the observed topology cannot "
                "isolate a transferable station update from C_dx leakage.  "
                "Monitoring on seven later runs stays nominal at current geometry, "
                "so there is no empirical mandate to force a write."
            ),
        },
        {
            "id": "quieter_subset_could_rescue",
            "risk": "A reviewer may ask for a quieter subset that would allow a write.",
            "response": (
                "Same-topology scaling leaves the six-DoF condition number unchanged.  "
                "dx/ry leakage cannot be rescued by more events or looser thresholds.  "
                "Searching a residual-selected subset would break the residual-blind "
                "occupancy contract."
            ),
        },
        {
            "id": "14977_looks_like_anomaly",
            "risk": "A reviewer may read 14977 as an alignment excursion.",
            "response": (
                "The frozen alarm table classifies 0<selected<10 as insufficient "
                "statistics, not an alignment anomaly.  14977 has two selected "
                "routes on a 1687-event remainder."
            ),
        },
        {
            "id": "monitoring_equals_alignment",
            "risk": "A reviewer may treat stable dy/rx monitoring as a successful alignment.",
            "response": (
                "Stability at current official geometry supports DQ monitoring only.  "
                "It does not validate a new station payload and is not a C_dx Mode."
            ),
        },
    ]


def _md_bullets(items: Sequence[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def render_paper_results_summary(claims: Mapping[str, Any]) -> str:
    layers = "\n\n".join(
        f"### Layer {item['layer']}. {item['title']}\n\n"
        f"**Status:** {item['status']}\n\n"
        f"**Supports.** {item['proves']}\n\n"
        f"**Does not support.** {item['does_not_prove']}"
        for item in claims["layers"]
    )
    supported = _md_bullets(f"`{item['id']}`: {item['claim']}" for item in claims["supported_claims"])
    unsupported = _md_bullets(f"`{item['id']}`: {item['claim']}" for item in claims["unsupported_claims"])
    limits = _md_bullets(claims["limitations"])
    why = claims["why_not_writing_geometry_is_a_result"]
    unlock = claims["unlock_criteria"]
    text = f"""# Operating Protocol V1 paper results summary

Read-only publication analysis of the frozen entry-54 evidence package.
No new alignment mode is opened.  V2, occupancy, alarms, and `A` are not
retuned.  Residual or χ² decrease is a **DQ observable** only.  Implied
`C_dx` is not a measurement.

**Decision flow.** `{claims["decision_flow"]}`

**Operating state.** `real_data_operating_mode={claims["real_data_operating_mode"]}`,
`geometry_write_allowed=false`, `station_calibration_mode_available=false`,
`cdx_mode_allowed=false`, `alignment_drift_candidate=false`.

## Three layers

{layers}

## Claims the paper can support

{supported}

## Claims the paper cannot support

{unsupported}

## Why not writing geometry is the result

{why["statement"]}

Reasons:

{_md_bullets(why["reasons"])}

This is not caused by:

{_md_bullets(why["not_caused_by"])}

## Unlock criteria for Operating Protocol V2

Currently unmet.  Either of the following may open V2; neither writes
geometry inside this package:

1. Independent survey or external station constraints fix `dx/ry/dz`, and
   an independent measurement proves that true `|C_dx|` lies in the
   1.5–1.7 µm isolation budget.
2. A new independent real-data track topology empirically demonstrates a
   calibration subspace that transfers across runs.

Until then the only code path is current official geometry → frozen V2 →
residual/DQ monitoring.  {unlock["until_then"]["text"]}

## Limitations

{limits}

Frozen V2 SHA256: `{claims["frozen_v2_checkpoint_sha256"]}`.
"""
    assert_no_forbidden_positive_claims(text, where="paper_results_summary.md")
    return text


def render_paper_results_summary_cn(claims: Mapping[str, Any]) -> str:
    layers = "\n\n".join(
        f"### 第 {item['layer']} 层. {item['title']}\n\n"
        f"**状态：** {item['status']}\n\n"
        f"**能够证明。** {item['proves']}\n\n"
        f"**不能证明。** {item['does_not_prove']}"
        for item in claims["layers"]
    )
    supported = _md_bullets(f"`{item['id']}`: {item['claim']}" for item in claims["supported_claims"])
    unsupported = _md_bullets(f"`{item['id']}`: {item['claim']}" for item in claims["unsupported_claims"])
    limits = _md_bullets(claims["limitations"])
    why = claims["why_not_writing_geometry_is_a_result"]
    text = f"""# Operating Protocol V1 论文结果摘要

这是对条目 54 冻结证据包的只读论文分析。不再打开新的 alignment mode，
不重训 V2，不改 occupancy / 报警 / `A`。Residual 或 χ² 下降只是
**DQ observable**。Implied `C_dx` 不是测量值。

**决策流。** `{claims["decision_flow"]}`

**运行状态。** `real_data_operating_mode={claims["real_data_operating_mode"]}`，
`geometry_write_allowed=false`，`station_calibration_mode_available=false`，
`cdx_mode_allowed=false`，`alignment_drift_candidate=false`。

## 三个层次

{layers}

## 本文可以支持的 claim

{supported}

## 本文不能支持的 claim

{unsupported}

## 为什么“不写 geometry”本身就是结论

{why["statement"]}

原因：

{_md_bullets(why["reasons"])}

这不是因为：

{_md_bullets(why["not_caused_by"])}

## Operating Protocol V2 的解锁条件

当前未满足。以下任一条件可以打开 V2；本包仍然不写 geometry：

1. 独立 survey / 外部约束固定 `dx/ry/dz`，并且独立测量证明真实
   `|C_dx|` 落在 1.5–1.7 µm isolation budget 内。
2. 新的独立真实 track topology 实证得到跨 run 可搬运的 calibration
   subspace。

在此之前唯一代码路径是：当前官方 geometry → 冻结 V2 → residual/DQ
monitoring。禁止从真实数据 self-nulling residual 生成 alignment payload。

## 限制

{limits}

冻结 V2 SHA256：`{claims["frozen_v2_checkpoint_sha256"]}`。
"""
    assert_no_forbidden_positive_claims(text, where="paper_results_summary_cn.md")
    return text


def render_figure_caption_drafts() -> str:
    blocks = []
    for item in figure_captions():
        blocks.append(f"## {item['id']}\n\n**EN.** {item['en']}\n\n**CN.** {item['cn']}")
    text = (
        "# Operating Protocol V1 figure caption drafts\n\n"
        "Captions for the paper-ready figures generated from the frozen "
        "entry-54 `chart_data`.  Residual or χ² decrease is a DQ observable "
        "only.  Implied `C_dx` is not a measurement.\n\n"
        + "\n\n".join(blocks)
        + "\n"
    )
    assert_no_forbidden_positive_claims(text, where="figure_caption_drafts.md")
    return text


def render_reviewer_risk_checklist() -> str:
    blocks = []
    for item in reviewer_risks():
        blocks.append(
            f"## {item['id']}\n\n**Risk.** {item['risk']}\n\n**Response.** {item['response']}"
        )
    text = (
        "# Operating Protocol V1 reviewer-risk checklist\n\n"
        "Anticipated misreadings of the frozen real-data result.  The "
        "scientific conclusion is that geometry must not be written from the "
        "present self-nulling residuals.\n\n"
        + "\n\n".join(blocks)
        + "\n"
    )
    assert_no_forbidden_positive_claims(text, where="reviewer_risk_checklist.md")
    return text


def render_all_documents(claims: Mapping[str, Any]) -> dict[str, str]:
    return {
        "paper_results_summary.md": render_paper_results_summary(claims),
        "paper_results_summary_cn.md": render_paper_results_summary_cn(claims),
        "figure_caption_drafts.md": render_figure_caption_drafts(),
        "reviewer_risk_checklist.md": render_reviewer_risk_checklist(),
    }
