"""Four-valued entailment states and fail-closed decision helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Final


class EntailmentState(StrEnum):
    """Outcome of testing lesion-present and lesion-absent feasibility."""

    ENTAILED_PRESENT = "ENTAILED_PRESENT"
    ENTAILED_ABSENT = "ENTAILED_ABSENT"
    RAW_UNIDENTIFIABLE = "RAW_UNIDENTIFIABLE"
    OUT_OF_MODEL = "OUT_OF_MODEL"


_TRUTH_TABLE: Final[dict[tuple[bool, bool], EntailmentState]] = {
    (True, False): EntailmentState.ENTAILED_PRESENT,
    (False, True): EntailmentState.ENTAILED_ABSENT,
    (True, True): EntailmentState.RAW_UNIDENTIFIABLE,
    (False, False): EntailmentState.OUT_OF_MODEL,
}

# Public and immutable so tests, reports, and downstream implementations can audit
# the complete semantics without duplicating it.
FEASIBILITY_TRUTH_TABLE: Final[Mapping[tuple[bool, bool], EntailmentState]] = MappingProxyType(
    _TRUTH_TABLE
)

SUCCESSFUL_SOLVER_STATUSES: Final[frozenset[str]] = frozenset(
    {"completed", "converged", "feasible", "ok", "optimal", "success", "toy_baseline"}
)


def state_from_feasibility(positive_feasible: bool, negative_feasible: bool) -> EntailmentState:
    """Map the two hypothesis feasibility flags through the complete truth table.

    Exact booleans are required.  In particular, integers are rejected even though
    ``bool`` is an ``int`` subclass; silently interpreting a numeric status as a
    clinical proposition would violate the fail-closed contract.
    """

    if type(positive_feasible) is not bool:  # noqa: E721 - exact bool is intentional
        raise TypeError("positive_feasible must be a bool")
    if type(negative_feasible) is not bool:  # noqa: E721 - exact bool is intentional
        raise TypeError("negative_feasible must be a bool")
    return FEASIBILITY_TRUTH_TABLE[(positive_feasible, negative_feasible)]


def solver_status_succeeded(status: str | None) -> bool:
    """Return whether a solver status is explicitly recognized as successful."""

    return isinstance(status, str) and status.strip().lower() in SUCCESSFUL_SOLVER_STATUSES


def fail_closed_state(
    *,
    positive_feasible: bool | None,
    negative_feasible: bool | None,
    positive_score: float | None,
    negative_score: float | None,
    positive_threshold: float | None,
    negative_threshold: float | None,
    positive_solver_status: str | None,
    negative_solver_status: str | None,
    solver_status: str | None = "success",
    degradation_known: bool = True,
) -> EntailmentState:
    """Decide only when every prerequisite is explicit and valid.

    Missing/non-finite scores or thresholds, unknown degradations, failed or
    unrecognized solver statuses, and non-boolean feasibility values all map to
    ``OUT_OF_MODEL``.  This helper never raises for untrusted decision inputs.
    """

    if type(degradation_known) is not bool or not degradation_known:  # noqa: E721
        return EntailmentState.OUT_OF_MODEL
    if type(positive_feasible) is not bool or type(negative_feasible) is not bool:  # noqa: E721
        return EntailmentState.OUT_OF_MODEL

    numeric_inputs = (
        positive_score,
        negative_score,
        positive_threshold,
        negative_threshold,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        for value in numeric_inputs
    ):
        return EntailmentState.OUT_OF_MODEL

    statuses = (positive_solver_status, negative_solver_status, solver_status)
    if not all(solver_status_succeeded(status) for status in statuses):
        return EntailmentState.OUT_OF_MODEL
    return state_from_feasibility(positive_feasible, negative_feasible)


# Readable aliases for callers that naturally phrase this operation as a decision.
decide_entailment_state = state_from_feasibility
decide_state_fail_closed = fail_closed_state


__all__ = [
    "EntailmentState",
    "FEASIBILITY_TRUTH_TABLE",
    "SUCCESSFUL_SOLVER_STATUSES",
    "decide_entailment_state",
    "decide_state_fail_closed",
    "fail_closed_state",
    "solver_status_succeeded",
    "state_from_feasibility",
]
