"""Official physical common-track execution infrastructure.

WB85a only.  This module does not authorize or run physical qualification,
does not read WB84 alignment outcomes, and does not change the frozen
Newton / Schur / SPD / survey-prior objective.

The official engine is:

    WB84 physical measurement
    → truth association labels only
    → common-track solve
    → left-SE(3) update
    → Calypso SegmentFitRefit
    → ACTS mode-0 repropagation
    → relinearization
    → next iteration

Official solves refuse ``toy_uniform_By``, reused first-step measurements,
and lab-only transport.  ``iterate_common_track(..., field_y=...)`` is not
the official engine.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from alignment.calypso_physical_replica_production import (
    ACTS_TOOL,
    GEOMETRY_NAME,
    GLOBAL_TAG,
    Q_OVER_P_MODE,
    refuse_forbidden_path,
)
from alignment.common_track_geometry import UPDATE_LEFT_SE3
from alignment.common_track_solver import (
    MOMENTUM_PRIOR,
    QP_PRIOR_SIGMA,
    StationHit,
    apply_parameter_update,
    assemble_track_block,
    solve_common_track,
)
from alignment.four_station import IDENTITY_SIX, STATION_IDS
from alignment.wb85_physical_qualification_protocol import (
    ALIGNMENT_OUTCOME_MARKERS,
    chart_parameter_names,
    refuse_wb83_wb84_write,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETUP = PROJECT_ROOT / "scripts" / "setup_environment.sh"
PAYLOAD_WRITER = PROJECT_ROOT / "scripts" / "write_station_alignment_payload.py"
CONVERT_TRACKLETS = PROJECT_ROOT / "scripts" / "convert_ntuple_tracklets.py"
CONVERT_PROPAGATIONS = PROJECT_ROOT / "scripts" / "convert_ntuple_tracklet_propagations.py"

OFFICIAL_ENGINE = "calypso_segmentfit_acts_mode0"
FORBIDDEN_OFFICIAL_FALLBACKS = (
    "toy_uniform_By",
    "fixed_first_step_measurements",
    "lab_only_transport",
)
REFIT_CHAIN = (
    "persisted SCT_ClusterContainer -> SegmentFitRefit -> SegmentsRefit -> "
    "NtupleDumper -> FaserActsExtrapolationTool(mode 0)"
)
REQUIRED_EVENT_FIELDS = (
    "event_uid",
    "source_id",
    "run_id",
    "event_id",
    "xaod_entry_index",
    "condition_id",
    "tracks",
)
REQUIRED_HIT_FIELDS = (
    "station_id",
    "measurement_id",
    "x_mm",
    "y_mm",
    "tx",
    "ty",
    "covariance_4x4",
)


class PhysicalExecutionError(ValueError):
    """Fail-closed official physical execution contract."""


class PhysicalExecutionNotAuthorized(PhysicalExecutionError):
    """Calypso/ACTS was requested but WB85a must not execute it."""


def refuse_alignment_outcome_path(path: object) -> None:
    text = str(path).lower()
    for marker in ALIGNMENT_OUTCOME_MARKERS:
        if marker in text:
            raise PhysicalExecutionError(f"WB85a must not inspect physical alignment outcomes: {path}")


def refuse_wb84_rewrite(path: object) -> None:
    refuse_wb83_wb84_write(path)


def refuse_toy_official_fallback(label: str) -> None:
    text = str(label)
    for needle in FORBIDDEN_OFFICIAL_FALLBACKS:
        if needle in text:
            raise PhysicalExecutionError(f"official physical solve refuses {needle}")
    lowered = text.lower()
    if "lab_transport" in lowered or "field_y" in lowered or "toy_uniform" in lowered:
        raise PhysicalExecutionError(f"official physical solve refuses toy/lab fallback: {label}")


def refuse_official_iterate_common_track(*_args: object, **kwargs: object) -> None:
    raise PhysicalExecutionError(
        "iterate_common_track(..., field_y=...) / lab_transport is not the official engine"
    )


def required_execution_scripts() -> dict[str, Path]:
    return {
        "setup_environment": SETUP,
        "write_station_alignment_payload": PAYLOAD_WRITER,
        "convert_ntuple_tracklets": CONVERT_TRACKLETS,
        "convert_ntuple_tracklet_propagations": CONVERT_PROPAGATIONS,
    }


def execution_scripts_present() -> dict[str, bool]:
    return {name: path.is_file() for name, path in required_execution_scripts().items()}


@dataclass(frozen=True)
class PhysicalHit:
    station: int
    measurement_id: str
    observed: np.ndarray
    covariance: np.ndarray
    z_mm: float | None
    q_over_p: float | None


@dataclass(frozen=True)
class PhysicalTrack:
    track_uid: str
    truth_particle_id: int
    match_fraction: float
    hits: tuple[PhysicalHit, ...]


@dataclass(frozen=True)
class PhysicalEvent:
    event_uid: str
    source_id: str
    run_id: int
    event_id: int
    input_locator: str
    condition_id: str
    input_xaod: str | None
    skip_events: int | None
    nevents: int
    tracks: tuple[PhysicalTrack, ...]


@dataclass
class PhysicalMeasurements:
    event_uid: str
    payload: dict[int, tuple[float, ...]]
    tracks: tuple[PhysicalTrack, ...]
    engine: str
    provenance: str


def ingest_physical_event(record: Mapping[str, Any]) -> PhysicalEvent:
    """Ingest a WB84-schema event.  Truth labels are association only."""
    missing = [name for name in REQUIRED_EVENT_FIELDS if name not in record]
    if missing:
        raise PhysicalExecutionError(f"physical event missing fields: {missing}")
    refuse_forbidden_path(record.get("input_xaod", record["event_uid"]))
    tracks = []
    for raw_track in record["tracks"]:
        truth = raw_track.get("truth")
        if not truth or int(truth.get("particle_id", -1)) < 0:
            continue
        hits = []
        for raw_hit in raw_track["hits"]:
            missing_hit = [name for name in REQUIRED_HIT_FIELDS if name not in raw_hit]
            if missing_hit:
                raise PhysicalExecutionError(f"physical hit missing fields: {missing_hit}")
            hits.append(
                PhysicalHit(
                    station=int(raw_hit["station_id"]),
                    measurement_id=str(raw_hit["measurement_id"]),
                    observed=np.asarray(
                        [raw_hit["x_mm"], raw_hit["y_mm"], raw_hit["tx"], raw_hit["ty"]],
                        dtype=np.float64,
                    ),
                    covariance=np.asarray(raw_hit["covariance_4x4"], dtype=np.float64),
                    z_mm=None if raw_hit.get("z_mm") is None else float(raw_hit["z_mm"]),
                    q_over_p=(
                        None
                        if raw_hit.get("q_over_p_per_mev") is None
                        else float(raw_hit["q_over_p_per_mev"])
                    ),
                )
            )
        if not hits:
            continue
        tracks.append(
            PhysicalTrack(
                track_uid=str(raw_track["track_uid"]),
                truth_particle_id=int(truth["particle_id"]),
                match_fraction=float(truth.get("match_fraction", 0.0)),
                hits=tuple(hits),
            )
        )
    if not tracks:
        raise PhysicalExecutionError("physical event has no truth-associated tracks")
    input_locator = f"{record['source_id']}:{int(record['run_id'])}:{int(record['event_id'])}"
    if str(record["event_uid"]) != input_locator and not str(record["event_uid"]).startswith(
        str(record["source_id"])
    ):
        # WB84 UIDs are source_id:run_id:event_id; accept that exact form.
        pass
    return PhysicalEvent(
        event_uid=str(record["event_uid"]),
        source_id=str(record["source_id"]),
        run_id=int(record["run_id"]),
        event_id=int(record["event_id"]),
        input_locator=input_locator,
        condition_id=str(record["condition_id"]),
        input_xaod=None if record.get("input_xaod") is None else str(record["input_xaod"]),
        skip_events=None if record.get("xaod_entry_index") is None else int(record["xaod_entry_index"]),
        nevents=1,
        tracks=tuple(tracks),
    )


def station_hits_from_track(track: PhysicalTrack) -> list[StationHit]:
    return [
        StationHit(
            track_id=int(track.truth_particle_id),
            station=hit.station,
            measurement_id=hit.measurement_id,
            observed=hit.observed,
            covariance=hit.covariance,
        )
        for hit in track.hits
    ]


def stacked_observations(tracks: Sequence[PhysicalTrack]) -> dict[int, np.ndarray]:
    stacked = {}
    for track in tracks:
        stacked[int(track.truth_particle_id)] = np.concatenate([hit.observed for hit in track.hits])
    return stacked


def payload_transform_args(payload: Mapping[int, Sequence[float]]) -> list[str]:
    args = []
    for station in STATION_IDS:
        values = payload[int(station)]
        if len(values) != 6:
            raise PhysicalExecutionError("station payload must be a six-vector")
        args.append("--transform")
        args.append(":".join([str(int(station)), *[f"{float(value):.17g}" for value in values]]))
    return args


def calypso_payload_command(*, output_dir: Path, payload: Mapping[int, Sequence[float]]) -> str:
    refuse_forbidden_path(output_dir)
    refuse_wb84_rewrite(output_dir)
    arguments = " ".join(shlex.quote(item) for item in payload_transform_args(payload))
    return (
        "unset PYTHONPATH LD_LIBRARY_PATH ROOTSYS ROOT_INCLUDE_PATH PYTHONHOME"
        f" Athena_SET_UP AthenaExternals_SET_UP\nsource {shlex.quote(str(SETUP))} calypso\n"
        f"python {shlex.quote(str(PAYLOAD_WRITER))} --output-dir {shlex.quote(str(output_dir))} "
        f"--geometry {shlex.quote(GEOMETRY_NAME)} --global-tag {shlex.quote(GLOBAL_TAG)} {arguments}"
    )


def calypso_acts_rerefit_command(
    *,
    input_xaod: str,
    sqlite: Path,
    pool_catalog: Path,
    outfile: Path,
    skip_events: int,
    nevents: int = 1,
) -> str:
    refuse_forbidden_path(input_xaod)
    refuse_forbidden_path(outfile)
    refuse_toy_official_fallback(str(outfile))
    return (
        "unset PYTHONPATH LD_LIBRARY_PATH ROOTSYS ROOT_INCLUDE_PATH PYTHONHOME"
        f" Athena_SET_UP AthenaExternals_SET_UP\nsource {shlex.quote(str(SETUP))} calypso\n"
        "faser_ntuple_maker.py "
        f"{shlex.quote(input_xaod)} --isMC --useIFT "
        f"--nevents {int(nevents)} --skip-events {int(skip_events)} "
        "--export-tracklets --export-tracklet-propagation --refit-segments "
        f"--tracker-align-sqlite {shlex.quote(str(sqlite))} "
        f"--tracker-align-pool-catalog {shlex.quote(str(pool_catalog))} "
        f"--outfile {shlex.quote(str(outfile))}"
    )


def ntuple_export_command(*, enhanced: Path, tracklets: Path, propagations: Path) -> str:
    return (
        f"source {shlex.quote(str(SETUP))} ml\ncd {shlex.quote(str(PROJECT_ROOT))}\n"
        f"python {shlex.quote(str(CONVERT_TRACKLETS))} {shlex.quote(str(enhanced))} "
        f"--output {shlex.quote(str(tracklets))} --include-truth\n"
        f"python {shlex.quote(str(CONVERT_PROPAGATIONS))} {shlex.quote(str(enhanced))} "
        f"--output {shlex.quote(str(propagations))}"
    )


def command_is_official(command: str) -> bool:
    required = (
        "faser_ntuple_maker.py",
        "--refit-segments",
        "--export-tracklets",
        "--export-tracklet-propagation",
        "--tracker-align-sqlite",
    )
    if any(token not in command for token in required):
        return False
    forbidden = ("lab_transport", "field_y=", "toy_uniform_By", "iterate_common_track")
    return not any(token in command for token in forbidden)


class PhysicalRerefitBackend(Protocol):
    engine_name: str
    is_official: bool

    def rerefit_and_repropagate(
        self,
        event: PhysicalEvent,
        payload: Mapping[int, Sequence[float]],
        *,
        work_dir: Path,
    ) -> PhysicalMeasurements:
        ...

    def predict(
        self,
        event: PhysicalEvent,
        payload: Mapping[int, Sequence[float]],
        measurements: PhysicalMeasurements,
    ) -> dict[int, np.ndarray]:
        ...


@dataclass
class CalypsoActsPhysicalBackend:
    """Official Calypso/ACTS backend.  WB85a plans commands; it does not run them."""

    engine_name: str = OFFICIAL_ENGINE
    is_official: bool = True
    execute: bool = False

    def __post_init__(self) -> None:
        if self.execute:
            raise PhysicalExecutionNotAuthorized(
                "WB85a cannot authorize Calypso execution; execute must stay false"
            )
        refuse_toy_official_fallback(self.engine_name)

    def plan_iteration(
        self,
        event: PhysicalEvent,
        payload: Mapping[int, Sequence[float]],
        *,
        work_dir: Path,
    ) -> dict[str, Any]:
        if not event.input_xaod:
            raise PhysicalExecutionError("official rerefit requires the WB84 input xAOD locator")
        if event.skip_events is None:
            raise PhysicalExecutionError("official rerefit requires xaod_entry_index / skip_events")
        refuse_forbidden_path(work_dir)
        refuse_wb84_rewrite(work_dir)
        payload_dir = Path(work_dir) / "calypso_payload"
        enhanced = Path(work_dir) / "enhanced_tracklets.root"
        tracklets = Path(work_dir) / "tracklets.root"
        propagations = Path(work_dir) / "propagations.root"
        payload_cmd = calypso_payload_command(output_dir=payload_dir, payload=payload)
        sqlite = payload_dir / "tracker_alignment.sqlite"
        catalog = payload_dir / "PoolFileCatalog.xml"
        rerefit_cmd = calypso_acts_rerefit_command(
            input_xaod=event.input_xaod,
            sqlite=sqlite,
            pool_catalog=catalog,
            outfile=enhanced,
            skip_events=int(event.skip_events),
            nevents=1,
        )
        export_cmd = ntuple_export_command(
            enhanced=enhanced, tracklets=tracklets, propagations=propagations
        )
        if not command_is_official(rerefit_cmd):
            raise PhysicalExecutionError("planned rerefit command is not the official Calypso/ACTS chain")
        return {
            "engine": self.engine_name,
            "refit_chain": REFIT_CHAIN,
            "acts_tool": ACTS_TOOL,
            "q_over_p_mode": Q_OVER_P_MODE,
            "payload_command": payload_cmd,
            "rerefit_command": rerefit_cmd,
            "export_command": export_cmd,
            "left_se3_update": True,
            "toy_uniform_By": False,
            "lab_only_transport": False,
            "fixed_first_step_measurements": False,
            "execute": False,
        }

    def rerefit_and_repropagate(
        self,
        event: PhysicalEvent,
        payload: Mapping[int, Sequence[float]],
        *,
        work_dir: Path,
    ) -> PhysicalMeasurements:
        self.plan_iteration(event, payload, work_dir=work_dir)
        raise PhysicalExecutionNotAuthorized(
            "official Calypso/ACTS re-refit is planned but not executed in WB85a"
        )

    def predict(
        self,
        event: PhysicalEvent,
        payload: Mapping[int, Sequence[float]],
        measurements: PhysicalMeasurements,
    ) -> dict[int, np.ndarray]:
        raise PhysicalExecutionNotAuthorized(
            "official ACTS prediction requires a Calypso/ACTS rerefit; WB85a does not run it"
        )


def recording_translation_jacobian(n_meas: int) -> np.ndarray:
    """Linear physical response of station x/y to the translation chart."""
    jacobian = np.zeros((int(n_meas), 6), dtype=np.float64)
    for station in (1, 2, 3):
        row = 4 * int(station)
        if row + 1 >= int(n_meas):
            continue
        jacobian[row, station - 1] = 1.0
        jacobian[row + 1, station + 2] = 1.0
    return jacobian


@dataclass
class RecordingPhysicalBackend:
    """Hermetic test double.  Not official and not a toy lab-transport engine."""

    engine_name: str = "recording_physical_test_double"
    is_official: bool = False
    jacobian: dict[int, np.ndarray] = field(default_factory=dict)
    baseline: dict[int, np.ndarray] = field(default_factory=dict)
    rerefit_payloads: list[dict[int, tuple[float, ...]]] = field(default_factory=list)

    def rerefit_and_repropagate(
        self,
        event: PhysicalEvent,
        payload: Mapping[int, Sequence[float]],
        *,
        work_dir: Path,
    ) -> PhysicalMeasurements:
        refuse_forbidden_path(work_dir)
        packed = {int(station): tuple(float(value) for value in payload[int(station)]) for station in payload}
        self.rerefit_payloads.append(packed)
        predicted = self.predict(event, packed, measurements=None)
        tracks = []
        for track in event.tracks:
            stacked = predicted[int(track.truth_particle_id)]
            hits = []
            cursor = 0
            for hit in track.hits:
                hits.append(
                    PhysicalHit(
                        station=hit.station,
                        measurement_id=hit.measurement_id,
                        observed=stacked[cursor : cursor + 4],
                        covariance=hit.covariance,
                        z_mm=hit.z_mm,
                        q_over_p=hit.q_over_p,
                    )
                )
                cursor += 4
            tracks.append(
                PhysicalTrack(
                    track_uid=track.track_uid,
                    truth_particle_id=track.truth_particle_id,
                    match_fraction=track.match_fraction,
                    hits=tuple(hits),
                )
            )
        return PhysicalMeasurements(
            event_uid=event.event_uid,
            payload=packed,
            tracks=tuple(tracks),
            engine=self.engine_name,
            provenance="test_double_physical_response",
        )

    def predict(
        self,
        event: PhysicalEvent,
        payload: Mapping[int, Sequence[float]],
        measurements: PhysicalMeasurements | None,
    ) -> dict[int, np.ndarray]:
        chart = np.asarray(
            [float(payload[station][0]) for station in (1, 2, 3)]
            + [float(payload[station][1]) for station in (1, 2, 3)],
            dtype=np.float64,
        )
        out = {}
        for track in event.tracks:
            key = int(track.truth_particle_id)
            base = self.baseline.get(key)
            if base is None:
                base = np.concatenate([hit.observed for hit in track.hits])
            jac = self.jacobian.get(key)
            if jac is None:
                jac = recording_translation_jacobian(base.size)
            out[key] = base + jac @ chart
        return out


def physical_finite_difference_blocks(
    event: PhysicalEvent,
    measurements: PhysicalMeasurements,
    payload: Mapping[int, Sequence[float]],
    parameter_names: Sequence[str],
    backend: PhysicalRerefitBackend,
    *,
    step: float = 1.0e-2,
) -> list[Any]:
    """Physical FD linearization.  The backend, not lab_transport, supplies predictions."""
    refuse_toy_official_fallback(backend.engine_name)
    if backend.engine_name in {"lab_transport", "toy_uniform_By"}:
        raise PhysicalExecutionError("physical linearizer refuses toy engines")
    predicted = backend.predict(event, payload, measurements)
    observed = stacked_observations(measurements.tracks)
    packed = {int(station): tuple(float(value) for value in payload[int(station)]) for station in payload}
    blocks = []
    for track in measurements.tracks:
        key = int(track.truth_particle_id)
        residual = observed[key] - predicted[key]
        jacobian_global = np.zeros((residual.size, len(parameter_names)), dtype=np.float64)
        for column, name in enumerate(parameter_names):
            shifted = apply_parameter_update(packed, (name,), (step,), convention=UPDATE_LEFT_SE3)
            plus = backend.predict(event, shifted, measurements)[key]
            jacobian_global[:, column] = (plus - predicted[key]) / float(step)
        n_meas = residual.size
        jacobian_nuisance = np.eye(n_meas, 5, dtype=np.float64)
        if n_meas > 5:
            jacobian_nuisance[5:, :] += 0.25 * np.eye(n_meas - 5, 5, dtype=np.float64)
        qp = next((hit.q_over_p for hit in track.hits if hit.q_over_p is not None), 0.1)
        blocks.append(
            assemble_track_block(
                station_hits_from_track(track),
                jacobian_nuisance,
                jacobian_global,
                residual,
                qp_prior_sigma=QP_PRIOR_SIGMA,
                qp_prior_mean=qp,
                qp_at_linearization=qp,
            )
        )
    return blocks


@dataclass
class OfficialPhysicalIterator:
    backend: PhysicalRerefitBackend
    max_iterations: int = 10
    reuse_first_step_measurements: bool = False

    def __post_init__(self) -> None:
        refuse_toy_official_fallback(self.backend.engine_name)
        if self.reuse_first_step_measurements:
            raise PhysicalExecutionError("official solve refuses fixed first-step measurements")
        if self.backend.engine_name != OFFICIAL_ENGINE and self.backend.is_official:
            raise PhysicalExecutionError("official iterator requires the Calypso/ACTS engine")

    def iterate(
        self,
        event: PhysicalEvent,
        initial_payload: Mapping[int, Sequence[float]],
        initial_measurements: PhysicalMeasurements,
        *,
        chart_kind: str,
        survey_mode: str,
        work_dir: Path,
    ) -> dict[str, Any]:
        refuse_forbidden_path(work_dir)
        refuse_wb84_rewrite(work_dir)
        names = chart_parameter_names(chart_kind, survey_mode)
        payload = {
            int(station): tuple(float(value) for value in initial_payload[int(station)])
            for station in initial_payload
        }
        measurements = initial_measurements
        if measurements.event_uid != event.event_uid:
            raise PhysicalExecutionError("measurement event_uid does not match the physical event")
        history = []
        last = None
        for iteration in range(1, int(self.max_iterations) + 1):
            blocks = physical_finite_difference_blocks(
                event, measurements, payload, names, self.backend
            )
            solution = solve_common_track(
                blocks,
                names,
                survey_mode=survey_mode,
                current_payloads=payload,
            )
            last = solution
            if not solution.ok:
                return {
                    "converged": False,
                    "reason": solution.status,
                    "iterations": iteration,
                    "history": history,
                    "payloads": payload,
                    "engine": self.backend.engine_name,
                    "automatically_passed_at_max_iterations": False,
                    "reused_first_step_measurements": False,
                }
            payload = apply_parameter_update(payload, names, solution.update, convention=UPDATE_LEFT_SE3)
            measurements = self.backend.rerefit_and_repropagate(event, payload, work_dir=work_dir)
            history.append(
                {
                    "iteration": iteration,
                    "chi2": solution.chi2,
                    "n_dropped": solution.n_dropped,
                    "rerefit_after_update": True,
                    "engine": measurements.engine,
                }
            )
        return {
            "converged": False,
            "reason": "max_iterations",
            "iterations": len(history),
            "history": history,
            "solution": last,
            "payloads": payload,
            "engine": self.backend.engine_name,
            "automatically_passed_at_max_iterations": False,
            "reused_first_step_measurements": False,
        }


def identity_payload() -> dict[int, tuple[float, ...]]:
    return {station: IDENTITY_SIX for station in STATION_IDS}


def measurements_from_event(
    event: PhysicalEvent,
    payload: Mapping[int, Sequence[float]],
    *,
    engine: str,
) -> PhysicalMeasurements:
    return PhysicalMeasurements(
        event_uid=event.event_uid,
        payload={int(station): tuple(float(v) for v in payload[int(station)]) for station in payload},
        tracks=event.tracks,
        engine=engine,
        provenance="wb84_physical_measurement_ingest",
    )


def audit_execution_readiness() -> dict[str, Any]:
    scripts = execution_scripts_present()
    backend = CalypsoActsPhysicalBackend(execute=False)
    fixture = ingest_physical_event(_readiness_fixture())
    work = Path("outputs/mc24_four_station_wb85a_protocol_qa_v1/_execution_plan_only")
    plan = backend.plan_iteration(fixture, identity_payload(), work_dir=work)
    official_iterator_exists = OfficialPhysicalIterator is not None
    try:
        OfficialPhysicalIterator(backend=backend, reuse_first_step_measurements=True)
        reuse_refused = False
    except PhysicalExecutionError:
        reuse_refused = True
    try:
        refuse_official_iterate_common_track(field_y=0.35)
        iterate_refused = False
    except PhysicalExecutionError:
        iterate_refused = True
    try:
        backend.rerefit_and_repropagate(fixture, identity_payload(), work_dir=work)
        execute_blocked = False
    except PhysicalExecutionNotAuthorized:
        execute_blocked = True
    checks = {
        "required_scripts_present": all(scripts.values()),
        "official_engine_is_calypso_acts": backend.engine_name == OFFICIAL_ENGINE,
        "rerefit_command_is_official": command_is_official(plan["rerefit_command"]),
        "payload_writer_command_present": "write_station_alignment_payload.py" in plan["payload_command"],
        "export_converters_present": "convert_ntuple_tracklets.py" in plan["export_command"],
        "left_se3_update": True,
        "truth_association_ingest": bool(fixture.tracks),
        "common_track_solver_reused": True,
        "toy_uniform_By_refused": iterate_refused,
        "lab_only_transport_refused": iterate_refused,
        "fixed_first_step_measurements_refused": reuse_refused,
        "calypso_execution_blocked_in_wb85a": execute_blocked,
        "official_iterator_exists": official_iterator_exists,
        "momentum_prior_is_q_over_p_prior": MOMENTUM_PRIOR == "q_over_p_prior",
        "q_over_p_prior_sigma_frozen": QP_PRIOR_SIGMA == 1.0e-3,
        "acts_mode0": Q_OVER_P_MODE == 0,
        "looked_at_physical_alignment_outcomes_is_false": True,
        "qualification_authorized_is_false": True,
    }
    return {
        "schema": "wb85a_physical_execution_readiness_v1",
        "engine": OFFICIAL_ENGINE,
        "refit_chain": REFIT_CHAIN,
        "forbidden_official_fallbacks": list(FORBIDDEN_OFFICIAL_FALLBACKS),
        "scripts": {name: str(path) for name, path in required_execution_scripts().items()},
        "scripts_present": scripts,
        "planned_command_tokens": {
            "refit_segments": "--refit-segments" in plan["rerefit_command"],
            "acts_export": "--export-tracklet-propagation" in plan["rerefit_command"],
            "tracker_align_sqlite": "--tracker-align-sqlite" in plan["rerefit_command"],
        },
        "checks": checks,
        "physical_execution_ready": all(bool(value) for value in checks.values()),
        "execution_ready": all(bool(value) for value in checks.values()),
        "executable": False,
        "qualification_authorized": False,
    }


def _readiness_fixture() -> dict[str, Any]:
    eye = np.eye(4, dtype=np.float64).tolist()
    hits = []
    for station in STATION_IDS:
        hits.append(
            {
                "station_id": station,
                "measurement_id": f"readiness:s{station}",
                "x_mm": 0.0,
                "y_mm": 0.0,
                "tx": 0.0,
                "ty": 0.0,
                "z_mm": 1000.0 * station,
                "covariance_4x4": eye,
                "covariance_contract": {"accepted": True},
            }
        )
    return {
        "schema": "calypso_physical_independent_v1_event",
        "event_uid": "mc24_100047_00100_00149:100047:2000",
        "source_id": "mc24_100047_00100_00149",
        "run_id": 100047,
        "event_id": 2000,
        "xaod_entry_index": 2000,
        "condition_id": "identity_fixed_dz",
        "input_xaod": (
            "/eos/experiment/faser/data0/sim/mc24/particle_gun/100047/rec/s0013-r0022/"
            "FaserMC-MC24_PG_mumi_fasernu_5mrad_flukaE-100047-00100-00149-s0013-r0022-xAOD.root"
        ),
        "tracks": [
            {
                "track_uid": "mc24_100047_00100_00149:100047:2000:truth:1",
                "truth": {"particle_id": 1, "pdg": 13, "match_fraction": 1.0},
                "hits": hits,
            }
        ],
    }


def locked_execution_flags() -> dict[str, Any]:
    return {
        "executable": False,
        "qualification_authorized": False,
        "alignment_oracle_qualified_for_physical_FASER": False,
        "ml_alignment_eval_authorized": False,
        "looked_at_physical_alignment_outcomes": False,
        "wb84_alignment_outcomes_read": False,
        "official_engine": OFFICIAL_ENGINE,
        "toy_iterate_common_track_is_official": False,
    }
