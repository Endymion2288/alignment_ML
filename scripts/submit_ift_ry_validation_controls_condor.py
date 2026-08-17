#!/usr/bin/env python3
"""Submit frozen IFT-R_y MLP, route, V1, and V2 validation controls to Condor GPUs.

The submitter accepts only the three predeclared control kinds and validates
that the synthetic corpus contains train/validation data only.  Worker
commands never invoke a test evaluator or test-time calibration.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from datasets.physical_curriculum import load_synthetic_curriculum_manifest
from scripts.config_loader import load_yaml_with_base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKER = PROJECT_ROOT / "scripts" / "run_ift_ry_validation_control_condor.sh"
ALLOWED_KINDS = {"mlp", "mlp_route", "v1", "v2"}
# LCG_110_cuda's PyTorch build has kernels beginning at CUDA capability 7.5.
# CERN H100 MIG slots currently expose their device name but not
# ``GPUs_Capability``, so retain them explicitly while excluding V100 (7.0).
GPU_COMPATIBILITY_REQUIREMENT = (
    '(TARGET.GPUs_Capability >= 7.5) || '
    'regexp("H100|A100|L40|RTX", TARGET.GPUs_DeviceName)'
)


def _require_eos(path: Path, label: str) -> None:
    if not str(path).startswith("/eos/"):
        raise ValueError(f"EOSSubmit requires {label} under /eos: {path}")


def _validate_synthetic_manifest(path: Path) -> None:
    _, samples, _ = load_synthetic_curriculum_manifest(
        path,
        require_all_splits=False,
        allowed_splits=("train", "validation"),
    )
    splits = {sample.split for sample in samples}
    if splits != {"train", "validation"}:
        raise ValueError("validation control requires exactly train and validation synthetic samples")
    if any(sample.condition_axis != "ift_ry_mrad" for sample in samples):
        raise ValueError("validation control requires the explicit ift_ry_mrad condition axis")


def _validate_config(kind: str, path: Path) -> None:
    config = load_yaml_with_base(path)
    root = config.get("physical_curriculum_mlp")
    if not isinstance(root, Mapping):
        raise ValueError(f"{kind} config lacks physical_curriculum_mlp")
    if tuple(root.get("allowed_splits", ())) != ("train", "validation"):
        raise ValueError(f"{kind} config must allow exactly train and validation")
    if "test" not in {str(value) for value in root.get("forbidden_splits", ())}:
        raise ValueError(f"{kind} config must explicitly forbid test")
    if kind in {"v2", "mlp_route"}:
        contract = config.get("route_aware_transformer_v2", {}).get("input_contract", {})
        if "test" not in {str(value) for value in contract.get("forbidden_splits", ())}:
            raise ValueError("V2 input contract must explicitly forbid test")


def _parse_control(value: str) -> tuple[str, Path, Path]:
    fields = value.split(":", 2)
    if len(fields) != 3:
        raise ValueError("--control must be KIND:CONFIG:OUTPUT_DIR")
    kind, config, output = fields
    if kind not in ALLOWED_KINDS:
        raise ValueError(f"unsupported control kind: {kind}")
    return kind, Path(config).expanduser().resolve(), Path(output).expanduser().resolve()


def _write_submit(
    target: Path,
    *,
    controls_path: Path,
    synthetic_manifest: Path,
    log_dir: Path,
    request_memory_mb: int,
    request_cpus: int,
    job_flavour: str,
    mlp_checkpoint: Path | None,
    mlp_calibration: Path | None,
) -> None:
    checkpoint_argument = "__none__" if mlp_checkpoint is None else str(mlp_checkpoint)
    calibration_argument = "__none__" if mlp_calibration is None else str(mlp_calibration)
    target.write_text(
        "\n".join(
            (
                "universe = vanilla",
                f"executable = {WORKER}",
                (
                    "arguments = $(kind) $(config) "
                    f"{synthetic_manifest} $(output_dir) {PROJECT_ROOT} "
                    f"{checkpoint_argument} {calibration_argument}"
                ),
                f"output = {log_dir}/$(kind).$(ClusterId).$(ProcId).out",
                f"error = {log_dir}/$(kind).$(ClusterId).$(ProcId).err",
                f"log = {log_dir}/ift_ry_controls.$(ClusterId).log",
                f"request_memory = {request_memory_mb}",
                f"request_cpus = {request_cpus}",
                "request_gpus = 1",
                f"requirements = {GPU_COMPATIBILITY_REQUIREMENT}",
                f'+JobFlavour = "{job_flavour}"',
                "getenv = True",
                f"queue kind, config, output_dir from {controls_path}",
                "",
            )
        ),
        encoding="utf-8",
    )


def _submit(path: Path) -> subprocess.CompletedProcess[str]:
    command = [
        "bash",
        "-lc",
        "source /usr/share/Modules/init/bash "
        "&& module load lxbatch/eossubmit "
        "&& myschedd out "
        f"&& condor_submit {shlex.quote(str(path))}",
    ]
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-manifest", required=True)
    parser.add_argument("--submit-dir", required=True)
    parser.add_argument("--control", action="append", required=True)
    parser.add_argument("--request-memory-mb", type=int, default=16000)
    parser.add_argument("--request-cpus", type=int, default=4)
    parser.add_argument("--job-flavour", default="tomorrow")
    parser.add_argument(
        "--mlp-checkpoint",
        default=None,
        help="Completed shared MLP checkpoint to reuse for an MLP-family control.",
    )
    parser.add_argument(
        "--mlp-calibration",
        default=None,
        help="Frozen station-pair MLP calibration required by an mlp_route control.",
    )
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if args.request_memory_mb < 1 or args.request_cpus < 1:
        parser.error("resource requests must be positive")
    if not WORKER.is_file() or not WORKER.stat().st_mode & 0o111:
        raise PermissionError(f"Condor worker must be executable: {WORKER}")
    manifest = Path(args.synthetic_manifest).expanduser().resolve()
    _validate_synthetic_manifest(manifest)
    controls = [_parse_control(value) for value in args.control]
    kinds = [kind for kind, _, _ in controls]
    if len(kinds) != len(set(kinds)):
        raise ValueError("each validation control kind may be submitted only once")
    for kind, config, output in controls:
        if not config.is_file():
            raise FileNotFoundError(config)
        _validate_config(kind, config)
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"refusing to overwrite existing control output: {output}")

    mlp_checkpoint = (
        None
        if args.mlp_checkpoint is None
        else Path(args.mlp_checkpoint).expanduser().resolve()
    )
    mlp_calibration = (
        None
        if args.mlp_calibration is None
        else Path(args.mlp_calibration).expanduser().resolve()
    )
    if mlp_checkpoint is not None or mlp_calibration is not None:
        if len(kinds) != 1 or kinds[0] not in {"mlp", "mlp_route"}:
            raise ValueError("MLP artifacts are valid only for one MLP-family control")
    if len(kinds) == 1 and kinds[0] == "mlp_route" and (
        mlp_checkpoint is None or mlp_calibration is None
    ):
        raise ValueError("mlp_route requires --mlp-checkpoint and --mlp-calibration")
    if len(kinds) == 1 and kinds[0] == "mlp" and mlp_calibration is not None:
        raise ValueError("--mlp-calibration is only valid for mlp_route")
    if mlp_checkpoint is not None:
        if not mlp_checkpoint.is_file():
            raise FileNotFoundError(f"MLP checkpoint does not exist: {mlp_checkpoint}")
    if mlp_calibration is not None and not mlp_calibration.is_file():
        raise FileNotFoundError(f"MLP calibration does not exist: {mlp_calibration}")

    submit_dir = Path(args.submit_dir).expanduser().resolve()
    if submit_dir.exists() and any(submit_dir.iterdir()):
        raise FileExistsError("refusing to overwrite a non-empty Condor submit directory")
    log_dir = submit_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=False)
    controls_path = submit_dir / "controls.tsv"
    controls_path.write_text(
        "\n".join(f"{kind}\t{config}\t{output}" for kind, config, output in controls) + "\n",
        encoding="utf-8",
    )
    submit_file = submit_dir / "ift_ry_validation_controls.sub"
    for label, path in (
        ("worker", WORKER),
        ("synthetic manifest", manifest),
        ("submit directory", submit_dir),
        ("log directory", log_dir),
        ("control list", controls_path),
    ):
        _require_eos(path, label)
    for _, config, output in controls:
        _require_eos(config, "control config")
        _require_eos(output, "control output")
    if mlp_checkpoint is not None:
        _require_eos(mlp_checkpoint, "MLP checkpoint")
    if mlp_calibration is not None:
        _require_eos(mlp_calibration, "MLP calibration")
    _write_submit(
        submit_file,
        controls_path=controls_path,
        synthetic_manifest=manifest,
        log_dir=log_dir,
        request_memory_mb=args.request_memory_mb,
        request_cpus=args.request_cpus,
        job_flavour=args.job_flavour,
        mlp_checkpoint=mlp_checkpoint,
        mlp_calibration=mlp_calibration,
    )
    report: dict[str, Any] = {
        "schema_version": "faser-ift-ry-validation-controls-condor-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "synthetic_manifest": str(manifest),
        "controls": [
            {"kind": kind, "config": str(config), "output_dir": str(output)}
            for kind, config, output in controls
        ],
        "test_contract": "train_validation_only_no_test_evaluator_or_calibration",
        "request_gpus": 1,
        "gpu_compatibility_requirement": GPU_COMPATIBILITY_REQUIREMENT,
        "request_memory_mb": args.request_memory_mb,
        "request_cpus": args.request_cpus,
        "mlp_checkpoint": None if mlp_checkpoint is None else str(mlp_checkpoint),
        "mlp_calibration": None if mlp_calibration is None else str(mlp_calibration),
        "submit_file": str(submit_file),
        "submitted": False,
    }
    if args.submit:
        result = _submit(submit_file)
        report["condor_submit_output"] = result.stdout
        report["condor_submit_returncode"] = result.returncode
        report["submitted"] = result.returncode == 0
        if result.returncode:
            (submit_dir / "submission.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            raise RuntimeError("condor_submit failed:\n" + result.stdout)
    (submit_dir / "submission.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "submit_dir": str(submit_dir),
                "controls": kinds,
                "submitted": bool(report["submitted"]),
                "condor_submit_output": report.get("condor_submit_output"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
