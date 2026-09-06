"""Fail-closed gate evaluation for versioned association metrics.

A gate never passes when a required metric is missing, ``None``, NaN, or
non-finite.  An empty specification is a failure: vacuous success is not
allowed.  Historical V5A / Workbook-74/75 artifacts are not rewritten.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence


ALLOWED_OPS = frozenset({"<=", ">=", "<", ">"})


@dataclass(frozen=True)
class GateSpec:
    name: str
    metric: str
    op: str
    bound: float
    required: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("gate name must be non-empty")
        if not self.metric:
            raise ValueError("gate metric must be non-empty")
        if self.op not in ALLOWED_OPS:
            raise ValueError(f"gate op must be one of {sorted(ALLOWED_OPS)}")
        bound = float(self.bound)
        if not math.isfinite(bound):
            raise ValueError(f"gate bound for {self.name!r} must be finite")
        object.__setattr__(self, "bound", bound)


@dataclass(frozen=True)
class GateResult:
    name: str
    metric: str
    op: str
    bound: float
    value: float | None
    passed: bool
    reason: str


def _as_finite(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def _compare(value: float, op: str, bound: float) -> bool:
    if op == "<=":
        return value <= bound
    if op == ">=":
        return value >= bound
    if op == "<":
        return value < bound
    if op == ">":
        return value > bound
    raise ValueError(f"unsupported gate op {op!r}")


def evaluate_gates(
    metrics: Mapping[str, object],
    specs: Sequence[GateSpec],
    *,
    required_metric_version: str | None = None,
) -> dict[str, object]:
    """Evaluate ``specs`` against ``metrics`` and refuse to pass on missing data."""
    if required_metric_version is not None:
        observed = metrics.get("metric_version")
        if observed != required_metric_version:
            return {
                "gate_pass": False,
                "fail_closed": True,
                "reason": (
                    f"required metric_version {required_metric_version!r}, "
                    f"got {observed!r}"
                ),
                "results": [],
                "n_specs": int(len(specs)),
                "n_passed": 0,
                "n_failed": int(len(specs)),
            }
    if not specs:
        return {
            "gate_pass": False,
            "fail_closed": True,
            "reason": "empty gate specification cannot pass",
            "results": [],
            "n_specs": 0,
            "n_passed": 0,
            "n_failed": 0,
        }

    results: list[GateResult] = []
    for spec in specs:
        if spec.metric not in metrics:
            results.append(
                GateResult(
                    name=spec.name,
                    metric=spec.metric,
                    op=spec.op,
                    bound=spec.bound,
                    value=None,
                    passed=False,
                    reason="missing metric",
                )
            )
            continue
        value = _as_finite(metrics[spec.metric])
        if value is None:
            results.append(
                GateResult(
                    name=spec.name,
                    metric=spec.metric,
                    op=spec.op,
                    bound=spec.bound,
                    value=None,
                    passed=False,
                    reason="non-finite or missing value",
                )
            )
            continue
        passed = _compare(value, spec.op, spec.bound)
        results.append(
            GateResult(
                name=spec.name,
                metric=spec.metric,
                op=spec.op,
                bound=spec.bound,
                value=value,
                passed=passed,
                reason="ok" if passed else "threshold failed",
            )
        )

    n_passed = sum(int(item.passed) for item in results)
    n_failed = len(results) - n_passed
    required_failed = [
        item for item, spec in zip(results, specs) if spec.required and not item.passed
    ]
    gate_pass = not required_failed and n_failed == 0 and len(results) == len(specs)
    return {
        "gate_pass": bool(gate_pass),
        "fail_closed": True,
        "reason": "all gates passed" if gate_pass else "one or more gates failed closed",
        "results": [
            {
                "name": item.name,
                "metric": item.metric,
                "op": item.op,
                "bound": item.bound,
                "value": item.value,
                "passed": item.passed,
                "reason": item.reason,
            }
            for item in results
        ],
        "n_specs": int(len(specs)),
        "n_passed": int(n_passed),
        "n_failed": int(n_failed),
    }
