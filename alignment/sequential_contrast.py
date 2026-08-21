"""Sequential 1-D outer-contrast remaining payloads.

C_dx and C_rx are individually identifiable but are not jointly solved in
one Newton step.  Each block-coordinate update floats exactly one contrast;
the other stays at its current geometry value.  Remaining after a 1-D
recovery of the floated contrast is written as a real layer payload and
refit; the next 1-D residual is measured on that new geometry.
"""

from __future__ import annotations

from typing import Mapping

from alignment.contrast_sampling import CONTRAST_PARAMETERS, CONTRAST_UNITS


def remaining_after_block_step(
    *,
    injected: Mapping[str, float],
    recovered_floated: float,
    floated: str,
) -> dict[str, float]:
    """Return remaining contrast after correcting one floated coordinate.

    The unfloated contrast is copied from the current geometry.  It is never
    algebraically subtracted with a 2-D Jacobian or treated as a nuisance.
    """
    if floated not in CONTRAST_PARAMETERS:
        raise ValueError(f"floated contrast must be one of {CONTRAST_PARAMETERS}, got {floated!r}")
    missing = [name for name in CONTRAST_PARAMETERS if name not in injected]
    if missing:
        raise ValueError("injected contrast is missing " + ", ".join(missing))
    remaining = {name: float(injected[name]) for name in CONTRAST_PARAMETERS}
    remaining[floated] = float(injected[floated]) - float(recovered_floated)
    return remaining


def contrast_unit(name: str) -> str:
    return CONTRAST_UNITS[name]
