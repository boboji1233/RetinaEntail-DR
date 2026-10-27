from __future__ import annotations

import json
from pathlib import Path

from retinaentail.calibration.protocol import CalibrationLabel, EvaluationFailureReason
from retinaentail.entailment.states import EntailmentState
from retinaentail.evaluation.metrics import (
    STAGE4_EVALUATION_REPORT_SCHEMA_VERSION,
    STAGE4_INTERVAL_METHOD,
    LockedStage4Contract,
    Stage4AggregationFailureReason,
    Stage4EvaluationScope,
    Stage4ReportStatus,
    aggregate_locked_test_evaluations,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = PROJECT_ROOT / "data_registry" / "schemas" / "stage4-evaluation-report-v4.schema.json"


def _mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    assert all(isinstance(key, str) for key in value)
    return value


def _sequence(value: object) -> list[object]:
    assert isinstance(value, list)
    return value


def _string_set(value: object) -> set[str]:
    items = _sequence(value)
    assert all(isinstance(item, str) for item in items)
    return {item for item in items if isinstance(item, str)}


def _assert_typed_objects_closed(value: object) -> None:
    if isinstance(value, dict):
        if value.get("type") == "object":
            assert value.get("additionalProperties") is False
        for child in value.values():
            _assert_typed_objects_closed(child)
    elif isinstance(value, list):
        for child in value:
            _assert_typed_objects_closed(child)


def test_stage4_report_schema_is_closed_offline_draft_202012() -> None:
    schema = _mapping(json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["additionalProperties"] is False
    assert "http://" not in json.dumps(schema, sort_keys=True)
    _assert_typed_objects_closed(schema)


def test_stage4_schema_versions_scopes_states_labels_and_reasons_match_python() -> None:
    schema = _mapping(json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))
    properties = _mapping(schema["properties"])
    definitions = _mapping(schema["$defs"])

    assert _mapping(properties["schema_version"])["const"] == (
        STAGE4_EVALUATION_REPORT_SCHEMA_VERSION
    )
    assert _mapping(properties["interval_method"])["const"] == STAGE4_INTERVAL_METHOD
    assert _string_set(_mapping(properties["scope"])["enum"]) == {
        item.value for item in Stage4EvaluationScope
    }
    assert _string_set(_mapping(properties["status"])["enum"]) == {
        item.value for item in Stage4ReportStatus
    }
    assert _mapping(properties["split"])["const"] == "test"
    assert _mapping(properties["clinical_claim"])["const"] is False

    state_counts = _mapping(definitions["stateCounts"])
    assert set(_mapping(state_counts["properties"])) == {item.value for item in EntailmentState}
    labels = _mapping(definitions["referenceLabelCounts"])
    assert set(_mapping(labels["properties"])) == {item.value for item in CalibrationLabel}
    assert _string_set(_mapping(definitions["stage4FailureReason"])["enum"]) == {
        item.value for item in Stage4AggregationFailureReason
    }
    failure_count_keys = set(_mapping(_mapping(definitions["failureReasonCounts"])["properties"]))
    assert failure_count_keys == {
        *(item.value for item in EvaluationFailureReason),
        *(item.value for item in Stage4AggregationFailureReason if item.value != "EMPTY_TEST_SET"),
    }


def test_schema_top_level_and_nested_contracts_are_exact_and_strict() -> None:
    schema = _mapping(json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))
    properties = _mapping(schema["properties"])
    required = _string_set(schema["required"])
    assert set(properties) == required
    definitions = _mapping(schema["$defs"])

    locked = _mapping(definitions["lockedContract"])
    locked_properties = _mapping(locked["properties"])
    assert set(locked_properties) == _string_set(locked["required"])
    assert set(locked_properties) == {
        "alpha",
        "positive_threshold",
        "negative_threshold",
        "positive_sample_count",
        "negative_sample_count",
        "total_sample_count",
        "calibration_artifact_hash",
        "config_hash",
        "data_hash",
        "model_hash",
    }
    sha256 = _mapping(definitions["sha256"])
    assert sha256 == {"type": "string", "pattern": "^[0-9a-f]{64}$"}
    for name in (
        "calibration_artifact_hash",
        "config_hash",
        "data_hash",
        "model_hash",
    ):
        assert _mapping(locked_properties[name]) == {"$ref": "#/$defs/sha256"}

    proportion = _mapping(definitions["proportion"])
    assert set(_mapping(proportion["properties"])) == _string_set(proportion["required"])
    assert len(_sequence(proportion["allOf"])) == 1


def test_schema_fields_exactly_match_runtime_payload() -> None:
    schema = _mapping(json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))
    report = aggregate_locked_test_evaluations(
        (),
        locked_contract=LockedStage4Contract(
            alpha=0.25,
            positive_threshold=0.5,
            negative_threshold=0.5,
            positive_sample_count=4,
            negative_sample_count=4,
            total_sample_count=8,
            calibration_artifact_hash="4" * 64,
            config_hash="1" * 64,
            data_hash="2" * 64,
            model_hash="3" * 64,
        ),
        scope=Stage4EvaluationScope.SYNTHETIC_ONLY,
    ).to_dict()
    assert set(report) == set(_mapping(schema["properties"])) == _string_set(schema["required"])
    locked_payload = _mapping(report["locked_contract"])
    locked_schema = _mapping(_mapping(schema["$defs"])["lockedContract"])
    assert set(locked_payload) == set(_mapping(locked_schema["properties"]))
    proportion_payload = _mapping(report["evaluable_proportion"])
    proportion_schema = _mapping(_mapping(schema["$defs"])["proportion"])
    assert set(proportion_payload) == set(_mapping(proportion_schema["properties"]))
