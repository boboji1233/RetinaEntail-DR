from __future__ import annotations

import json
from pathlib import Path

from retinaentail.pilot.gate import (
    STAGE5_PREFLIGHT_SCHEMA_VERSION,
    STAGE5_PREREQUISITE_SCHEMA_VERSION,
    Stage5PreflightReason,
    Stage5PrerequisiteReason,
    Stage5VerificationContext,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = PROJECT_ROOT / "data_registry" / "schemas"


def _load(name: str) -> dict[str, object]:
    value = json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return value


def _sequence(value: object) -> list[object]:
    assert isinstance(value, list)
    return value


def _assert_closed_objects(value: object) -> None:
    if isinstance(value, dict):
        if value.get("type") == "object":
            assert value.get("additionalProperties") is False
        for child in value.values():
            _assert_closed_objects(child)
    elif isinstance(value, list):
        for child in value:
            _assert_closed_objects(child)


def test_stage5_schemas_decode_and_close_objects() -> None:
    for name in ("stage5-preflight-v5.schema.json", "stage5-prerequisite-v5.schema.json"):
        schema = _load(name)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["additionalProperties"] is False
        _assert_closed_objects(schema)


def test_stage5_schema_versions_contexts_and_reason_enums_match_runtime() -> None:
    preflight = _load("stage5-preflight-v5.schema.json")
    prerequisite = _load("stage5-prerequisite-v5.schema.json")
    preflight_properties = _mapping(preflight["properties"])
    prerequisite_properties = _mapping(prerequisite["properties"])
    preflight_definitions = _mapping(preflight["$defs"])
    prerequisite_definitions = _mapping(prerequisite["$defs"])

    assert _mapping(preflight_properties["schema_version"])["const"] == (
        STAGE5_PREFLIGHT_SCHEMA_VERSION
    )
    assert _mapping(prerequisite_properties["schema_version"])["const"] == (
        STAGE5_PREREQUISITE_SCHEMA_VERSION
    )
    assert set(_sequence(_mapping(preflight_definitions["verificationContext"])["enum"])) == {
        item.value for item in Stage5VerificationContext
    }
    assert set(_sequence(_mapping(preflight_definitions["preflightReason"])["enum"])) == {
        item.value for item in Stage5PreflightReason
    }
    assert set(_sequence(_mapping(prerequisite_definitions["prerequisiteReason"])["enum"])) == {
        item.value for item in Stage5PrerequisiteReason
    }


def test_stage5_schema_zero_io_and_no_serialized_capability_constants() -> None:
    preflight = _mapping(_load("stage5-preflight-v5.schema.json")["properties"])
    prerequisite = _mapping(_load("stage5-prerequisite-v5.schema.json")["properties"])

    for properties in (preflight, prerequisite):
        assert _mapping(properties["authorizes_data_access"])["const"] is False
        assert _mapping(properties["dataset_root_probed"])["const"] is False
        assert _mapping(properties["real_data_read"])["const"] is False
        assert _mapping(properties["sample_count"])["const"] == 0
    assert "dataset_root" not in prerequisite
    assert "authorization" not in prerequisite
