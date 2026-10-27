from __future__ import annotations

import math

import pytest

from retinaentail.entailment.states import (
    FEASIBILITY_TRUTH_TABLE,
    EntailmentState,
    fail_closed_state,
    state_from_feasibility,
)


@pytest.mark.parametrize(
    ("positive", "negative", "expected"),
    [
        (True, False, EntailmentState.ENTAILED_PRESENT),
        (False, True, EntailmentState.ENTAILED_ABSENT),
        (True, True, EntailmentState.RAW_UNIDENTIFIABLE),
        (False, False, EntailmentState.OUT_OF_MODEL),
    ],
)
def test_complete_feasibility_truth_table(
    positive: bool, negative: bool, expected: EntailmentState
) -> None:
    assert state_from_feasibility(positive, negative) is expected
    assert FEASIBILITY_TRUTH_TABLE[(positive, negative)] is expected


@pytest.mark.parametrize("invalid", [None, 0, 1, "true"])
def test_truth_table_rejects_non_boolean_inputs(invalid: object) -> None:
    with pytest.raises(TypeError):
        state_from_feasibility(invalid, False)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "override",
    [
        {"positive_score": math.nan},
        {"negative_score": math.inf},
        {"positive_threshold": None},
        {"negative_threshold": math.nan},
        {"positive_solver_status": "failed"},
        {"negative_solver_status": "timeout"},
        {"solver_status": "unknown"},
        {"degradation_known": False},
        {"positive_feasible": None},
    ],
)
def test_invalid_decision_inputs_fail_closed(override: dict[str, object]) -> None:
    inputs: dict[str, object] = {
        "positive_feasible": True,
        "negative_feasible": False,
        "positive_score": 0.1,
        "negative_score": 0.9,
        "positive_threshold": 0.2,
        "negative_threshold": 0.2,
        "positive_solver_status": "success",
        "negative_solver_status": "success",
        "solver_status": "success",
        "degradation_known": True,
    }
    inputs.update(override)
    assert fail_closed_state(**inputs) is EntailmentState.OUT_OF_MODEL  # type: ignore[arg-type]


def test_valid_decision_uses_truth_table() -> None:
    assert (
        fail_closed_state(
            positive_feasible=True,
            negative_feasible=True,
            positive_score=0.1,
            negative_score=0.1,
            positive_threshold=0.2,
            negative_threshold=0.2,
            positive_solver_status="optimal",
            negative_solver_status="optimal",
            solver_status="completed",
        )
        is EntailmentState.RAW_UNIDENTIFIABLE
    )
