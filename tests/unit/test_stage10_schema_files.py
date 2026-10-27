from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from retinaentail.pilot.stage10_fixture import (
    STAGE10_TOY_CONFIG_VERSION,
    _artifact_diff,
    load_stage10_toy_config,
)
from retinaentail.pilot.stage10_gate import (
    STAGE10_PREFLIGHT_SCHEMA_VERSION,
    Stage10PreflightStatus,
    Stage10Reason,
    assess_stage10_preflight,
)
from retinaentail.reproducibility.stage10 import (
    STAGE10_SCHEMA_VERSION,
    Stage10DifferenceKind,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = PROJECT_ROOT / "data_registry" / "schemas"
CONFIG_PATH = PROJECT_ROOT / "configs" / "stage10_toy_runner_portability.yaml"
SCHEMAS = (
    "stage10-preflight-v10.schema.json",
    "stage10-toy-config-v10.schema.json",
    "stage10-signed-envelope-v10.schema.json",
    "stage10-artifact-diff-v10.schema.json",
    "stage10-diff-v10.schema.json",
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


def test_stage10_schemas_are_closed_draft_202012_and_offline() -> None:
    for name in SCHEMAS:
        schema = _load(name)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["$id"] == f"https://retinaentail.local/schemas/{name}"
        assert schema["additionalProperties"] is False
        _closed_objects(schema)
        _offline_refs(schema)


def test_runtime_versions_enums_and_required_fields_match_schemas() -> None:
    preflight = _load(SCHEMAS[0])
    config_schema = _load(SCHEMAS[1])
    diff = _load(SCHEMAS[4])
    assert _mapping(_mapping(preflight["properties"])["schema_version"])["const"] == (
        STAGE10_PREFLIGHT_SCHEMA_VERSION
    )
    assert _mapping(_mapping(config_schema["properties"])["schema_version"])["const"] == (
        STAGE10_TOY_CONFIG_VERSION
    )
    assert _mapping(_mapping(diff["properties"])["schema_version"])["const"] == (
        STAGE10_SCHEMA_VERSION
    )
    statuses = set(_sequence(_mapping(_mapping(preflight["properties"])["status"])["enum"]))
    assert statuses == {item.value for item in Stage10PreflightStatus}
    reasons = set(
        _sequence(_mapping(_mapping(_mapping(preflight["properties"])["reasons"])["items"])["enum"])
    )
    assert reasons == {item.value for item in Stage10Reason}
    differences = set(_sequence(_mapping(_mapping(diff["$defs"])["difference"])["enum"]))
    assert differences == {item.value for item in Stage10DifferenceKind}


def test_runtime_preflight_config_and_artifact_diff_have_exact_required_fields() -> None:
    report = assess_stage10_preflight(
        dataset_id=None,
        dataset_version=None,
        complete_real_stage9_package_a_capability_present=False,
        complete_real_stage9_package_b_capability_present=False,
        fresh_runner_a_capability_present=False,
        fresh_runner_b_capability_present=False,
        two_approved_runners_confirmed=False,
        environment_lock_preregistered=False,
        case_seed_replay_preregistered=False,
        locked_hashes_thresholds_preregistered=False,
        signer_trust_policy_preregistered=False,
    ).to_dict()
    config = load_stage10_toy_config(CONFIG_PATH)
    artifact_diff = _artifact_diff(
        {"same": b"x", "audit.json": b"a"},
        {"same": b"x", "audit.json": b"b"},
        allowed_paths=config.allowed_stage9_provenance_paths,
    )
    assert set(report) == set(_sequence(_load(SCHEMAS[0])["required"]))
    assert set(config.raw) == set(_sequence(_load(SCHEMAS[1])["required"]))
    assert set(artifact_diff) == set(_sequence(_load(SCHEMAS[3])["required"]))
    assert artifact_diff["allowed_provenance_difference_count"] == 1
    assert artifact_diff["deterministic_file_mismatch_count"] == 0
