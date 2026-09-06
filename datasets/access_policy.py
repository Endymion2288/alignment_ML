"""T01 access boundary.  Default fail-closed; sealed test is never a normal flag."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping


class AccessScope(str, Enum):
    TRAIN = "train"
    DEVELOPMENT_VALIDATION = "development_validation"
    REAL_DATA_MONITORING = "real_data_monitoring"
    FROZEN_EVALUATION = "frozen_evaluation"


class AccessPolicyError(PermissionError):
    """Raised when a caller is not allowed to resolve or open a dataset."""


@dataclass(frozen=True)
class FrozenEvaluationCapability:
    """Locked plan for a one-time confirmatory unsealing.  Not a CLI flag."""

    plan_sha256: str
    checkpoint_sha256: str
    calibration_sha256: str
    source_set_sha256: str
    unseal_allowed: bool = False

    def is_complete(self) -> bool:
        fields = (
            self.plan_sha256,
            self.checkpoint_sha256,
            self.calibration_sha256,
            self.source_set_sha256,
        )
        return all(len(item) == 64 for item in fields)


SCOPE_SPLITS: dict[AccessScope, frozenset[str]] = {
    AccessScope.TRAIN: frozenset({"train"}),
    AccessScope.DEVELOPMENT_VALIDATION: frozenset({"train", "validation"}),
    AccessScope.REAL_DATA_MONITORING: frozenset({"train", "validation"}),
    AccessScope.FROZEN_EVALUATION: frozenset({"train", "validation"}),
}

_DATASET_SUFFIXES = {".root", ".npz", ".h5", ".hdf5"}
_DATASET_MARKERS = ("tracklets", "propagations", "ntuple", "events", "clusters")


def refuse_allow_sealed_test_flag() -> None:
    raise AccessPolicyError(
        "--allow-sealed-test is not a development license; "
        "frozen_evaluation capability is required and is not granted in T00-T12"
    )


def is_restricted_dataset_path(
    path: str | Path | None, *, split: str | None = None
) -> bool:
    if str(split or "") == "test":
        return True
    if path is None:
        return False
    lowered = str(path).replace("\\", "/").lower()
    if "sealed_test" in lowered or "sealed-test" in lowered:
        return True
    dataset_like = Path(lowered).suffix in _DATASET_SUFFIXES or any(
        marker in lowered for marker in _DATASET_MARKERS
    )
    if not dataset_like:
        return False
    return "test" in Path(lowered).parts


def authorize_split(
    split: str,
    scope: AccessScope,
    *,
    capability: FrozenEvaluationCapability | None = None,
) -> None:
    del capability
    if str(split) == "test":
        raise AccessPolicyError(
            f"split 'test' is sealed under {scope.value}; "
            "T00-T12 do not grant frozen_evaluation unsealing"
        )
    allowed = SCOPE_SPLITS[AccessScope(scope)]
    if str(split) not in allowed:
        raise AccessPolicyError(f"split '{split}' is outside {scope.value}")


def authorize_path(
    path: str | Path,
    scope: AccessScope,
    *,
    split: str | None = None,
    capability: FrozenEvaluationCapability | None = None,
    opener: Callable[[str | Path], Any] | None = None,
    resolver: Callable[[str | Path], Any] | None = None,
) -> Path:
    """Return a Path only after the access check.  Denied calls never open."""
    del capability
    if is_restricted_dataset_path(path, split=split) or str(split or "") == "test":
        raise AccessPolicyError(
            f"refusing sealed/test dataset under {scope.value}: {path}"
        )
    if str(split or "") not in ("", "train", "validation"):
        authorize_split(str(split), scope)
    if resolver is not None:
        resolved = resolver(path)
    else:
        resolved = Path(path)
    if opener is not None:
        opener(resolved)
    return Path(resolved)


def load_curriculum_for_scope(
    manifest_path: str | Path,
    scope: AccessScope,
    *,
    capability: FrozenEvaluationCapability | None = None,
):
    """Shared curriculum entry.  Test paths are never resolved in T00-T12."""
    from datasets.physical_curriculum import load_synthetic_curriculum_manifest

    authorize_split("train", scope, capability=capability)
    return load_synthetic_curriculum_manifest(
        manifest_path,
        require_all_splits=False,
        allowed_splits=tuple(sorted(SCOPE_SPLITS[AccessScope(scope)])),
        access_scope=scope,
        access_capability=capability,
    )


def policy_audit_record() -> dict[str, Any]:
    return {
        "kind": "access_policy_audit",
        "schema_version": "access-boundary-v1",
        "task": "T01",
        "fail_closed": True,
        "scopes": [item.value for item in AccessScope],
        "allow_sealed_test_is_not_a_license": True,
        "frozen_evaluation_unseal_granted": False,
        "geometry_write_allowed": False,
        "held_out_accessed": False,
    }
