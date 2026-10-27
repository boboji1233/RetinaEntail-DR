from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from retinaentail.pilot.stage9_fixture import (
    STAGE9_PACKAGE_VERSION,
    STAGE9_TOY_CONFIG_VERSION,
    _stage8_file_diff,
    load_stage9_toy_config,
)
from retinaentail.pilot.stage9_gate import (
    STAGE9_GATE_SCHEMA_VERSION,
    Stage9PreflightStatus,
    Stage9Reason,
    assess_stage9_preflight,
)
from retinaentail.reproducibility.stage9 import (
    STAGE9_SCHEMA_VERSION,
    Stage9DifferenceKind,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = PROJECT_ROOT / "data_registry" / "schemas"
CONFIG_PATH = PROJECT_ROOT / "configs" / "stage9_toy_independent_session_diff.yaml"
SCHEMAS = (
    "stage9-preflight-v9.schema.json",
    "stage9-toy-config-v9.schema.json",
    "stage9-diff-v9.schema.json",
    "stage9-package-diff-v9.schema.json",
)


def _load(name: str) -> dict[str, object]:
    value: object = json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def _mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def _sequence(value: object) -> list[object]:
    assert isinstance(value, list)
    return value


def _closed_objects(value: object) -> None:
    if isinstance(value, dict):
        if value.get("type") == "object":
            assert value.get("additionalProperties") is False
        for child in value.values():
            _closed_objects(child)
    elif isinstance(value, list):
        for child in value:
            _closed_objects(child)


def _offline_refs(value: object) -> None:
    if isinstance(value, dict):
        reference = value.get("$ref")
        if reference is not None:
            assert isinstance(reference, str)
            assert reference.startswith("#")
        for child in value.values():
            _offline_refs(child)
    elif isinstance(value, list):
        for child in value:
            _offline_refs(child)


def test_stage9_schemas_are_closed_draft_202012_and_offline() -> None:
    for name in SCHEMAS:
        schema = _load(name)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["$id"] == f"https://retinaentail.local/schemas/{name}"
        assert schema["additionalProperties"] is False
        _closed_objects(schema)
        _offline_refs(schema)


def test_runtime_versions_status_reasons_and_differences_match_schemas() -> None:
    preflight = _load(SCHEMAS[0])
    config = _load(SCHEMAS[1])
    diff = _load(SCHEMAS[2])
    package_diff = _load(SCHEMAS[3])
    assert _mapping(_mapping(preflight["properties"])["schema_version"])["const"] == (
        STAGE9_GATE_SCHEMA_VERSION
    )
    assert _mapping(_mapping(config["properties"])["schema_version"])["const"] == (
        STAGE9_TOY_CONFIG_VERSION
    )
    assert _mapping(_mapping(diff["properties"])["schema_version"])["const"] == (
        STAGE9_SCHEMA_VERSION
    )
    assert _mapping(_mapping(package_diff["properties"])["schema_version"])["const"] == (
        STAGE9_PACKAGE_VERSION
    )
    status_values = set(_sequence(_mapping(_mapping(preflight["properties"])["status"])["enum"]))
    assert status_values == {item.value for item in Stage9PreflightStatus}
    reason_values = set(
        _sequence(_mapping(_mapping(_mapping(preflight["properties"])["reasons"])["items"])["enum"])
    )
    assert reason_values == {item.value for item in Stage9Reason}
    difference_values = set(_sequence(_mapping(_mapping(diff["$defs"])["difference"])["enum"]))
    assert difference_values == {item.value for item in Stage9DifferenceKind}


def test_runtime_preflight_config_and_package_diff_have_exact_required_fields() -> None:
    preflight_schema = _load(SCHEMAS[0])
    config_schema = _load(SCHEMAS[1])
    package_diff_schema = _load(SCHEMAS[3])
    report = assess_stage9_preflight(
        dataset_id=None,
        dataset_version=None,
        complete_stage8_package_a_capability_provided=False,
        complete_stage8_package_b_capability_provided=False,
        fresh_session_a_capability_provided=False,
        fresh_session_b_capability_provided=False,
        environment_lock_preregistered=False,
        case_seed_replay_preregistered=False,
        locked_hashes_thresholds_preregistered=False,
    ).to_dict()
    config = load_stage9_toy_config(CONFIG_PATH)
    file_diff = _stage8_file_diff({"a.txt": b"same"}, {"a.txt": b"same"})
    assert set(report) == set(_sequence(preflight_schema["required"]))
    assert set(config.raw) == set(_sequence(config_schema["required"]))
    assert set(file_diff) == set(_sequence(package_diff_schema["required"]))
    assert file_diff["differing_file_count"] == 0
    assert file_diff["matching_file_count"] == 1
