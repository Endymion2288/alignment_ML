"""Small YAML configuration loader with explicit relative inheritance."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge mappings recursively; scalar and sequence overrides replace values."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(dict(result[key]), value)
        else:
            result[key] = value
    return result


def load_yaml_with_base(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    """Load a mapping and recursively resolve an optional relative ``base_config``.

    A child replaces lists such as ``sources`` in full.  This prevents a pilot
    corpus from silently retaining a source from its parent configuration.
    """
    resolved = path.expanduser().resolve()
    chain = set() if seen is None else set(seen)
    if resolved in chain:
        raise ValueError(f"cyclic base_config reference at {resolved}")
    chain.add(resolved)
    with resolved.open(encoding="utf-8") as handle:
        supplied = yaml.safe_load(handle)
    if not isinstance(supplied, Mapping):
        raise ValueError(f"configuration must be a YAML mapping: {resolved}")
    payload = dict(supplied)
    base_reference = payload.pop("base_config", None)
    if base_reference is None:
        return payload
    base_path = Path(str(base_reference)).expanduser()
    if not base_path.is_absolute():
        base_path = resolved.parent / base_path
    return deep_merge(load_yaml_with_base(base_path, chain), payload)
