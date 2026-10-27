from __future__ import annotations

import json
from pathlib import Path

from retinaentail.evaluation.toy_fixture import (
    STAGE4_TOY_CONFIG_VERSION,
    load_stage4_toy_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "stage4_toy_locked_test.yaml"
SCHEMA_PATH = PROJECT_ROOT / "data_registry" / "schemas" / "stage4-toy-config-v4.schema.json"


def _mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    assert all(isinstance(key, str) for key in value)
    return value


def _sequence(value: object) -> list[object]:
    assert isinstance(value, list)
    return value


def _assert_typed_objects_closed(value: object) -> None:
    if isinstance(value, dict):
        if value.get("type") == "object":
            assert value.get("additionalProperties") is False
        for child in value.values():
            _assert_typed_objects_closed(child)
    elif isinstance(value, list):
        for child in value:
            _assert_typed_objects_closed(child)


def test_stage4_toy_config_schema_is_closed_offline_draft_202012() -> None:
    schema = _mapping(json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["additionalProperties"] is False
    assert "$data" not in json.dumps(schema, sort_keys=True)
    assert "http://" not in json.dumps(schema, sort_keys=True)
    _assert_typed_objects_closed(schema)


def test_stage4_toy_schema_pins_lock_and_ordered_case_plan() -> None:
    schema = _mapping(json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))
    properties = _mapping(schema["properties"])
    definitions = _mapping(schema["$defs"])
    assert _mapping(properties["schema_version"])["const"] == STAGE4_TOY_CONFIG_VERSION
    assert _mapping(properties["run_id"])["const"] == "stage4-toy-locked-test-001"

    lock = _mapping(definitions["stage3Lock"])
    lock_properties = _mapping(lock["properties"])
    assert set(lock_properties) == set(_sequence(lock["required"]))
    assert _mapping(lock_properties["alpha"])["const"] == 0.25
    assert _mapping(lock_properties["positive_sample_count"])["const"] == 4
    assert _mapping(lock_properties["negative_sample_count"])["const"] == 4
    assert _mapping(lock_properties["total_sample_count"])["const"] == 8

    test_cases = _mapping(properties["test_cases"])
    prefix = _sequence(test_cases["prefixItems"])
    assert test_cases["minItems"] == test_cases["maxItems"] == len(prefix) == 7
    assert test_cases["items"] is False
    assert prefix == [
        {"$ref": "#/$defs/entailedPresent"},
        {"$ref": "#/$defs/entailedAbsent"},
        {"$ref": "#/$defs/rawUnidentifiable"},
        {"$ref": "#/$defs/infeasibleOutOfModel"},
        {"$ref": "#/$defs/domainOutOfModel"},
        {"$ref": "#/$defs/unknownDegradation"},
        {"$ref": "#/$defs/invalidWitness"},
    ]


def test_checked_in_stage4_config_roundtrips_exactly() -> None:
    config = load_stage4_toy_config(CONFIG_PATH)
    assert config.schema_version == STAGE4_TOY_CONFIG_VERSION
    assert config.stage3_lock.total_sample_count == 8
    assert len(config.test_cases) == 7
    assert [case.case_id for case in config.test_cases] == [
        "stage4-toy-entailed-present",
        "stage4-toy-entailed-absent",
        "stage4-toy-raw-unidentifiable",
        "stage4-toy-infeasible-out-of-model",
        "stage4-toy-domain-out-of-model",
        "stage4-toy-unknown-degradation",
        "stage4-toy-invalid-witness",
    ]
