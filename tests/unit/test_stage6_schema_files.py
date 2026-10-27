from __future__ import annotations

import json
from pathlib import Path

from retinaentail.pilot.stage6_fixture import (
    STAGE6_TOY_CONFIG_VERSION,
    load_stage6_toy_config,
)
from retinaentail.pilot.stage6_gate import (
    STAGE6_GATE_SCHEMA_VERSION,
    Stage6PrerequisiteStatus,
    Stage6Reason,
    Stage6VerificationContext,
)
from retinaentail.reporting.stage6 import (
    STAGE6_AUTHORITY_BOUNDARY,
    STAGE6_REPORT_SCHEMA_VERSION,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = PROJECT_ROOT / "data_registry" / "schemas"
CONFIG_PATH = PROJECT_ROOT / "configs" / "stage6_toy_verifier_pilot.yaml"


def _load(name: str) -> dict[str, object]:
    value: object = json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return value


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


def test_stage6_schemas_are_versioned_closed_and_offline() -> None:
    for name in (
        "stage6-preflight-v6.schema.json",
        "stage6-prerequisite-v6.schema.json",
        "stage6-toy-config-v6.schema.json",
        "stage6-pilot-report-v6.schema.json",
    ):
        schema = _load(name)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["additionalProperties"] is False
        assert "http://" not in json.dumps(schema, sort_keys=True)
        _closed_objects(schema)


def test_stage6_runtime_versions_and_reason_enums_match_schemas() -> None:
    preflight = _load("stage6-preflight-v6.schema.json")
    prerequisite = _load("stage6-prerequisite-v6.schema.json")
    config = _load("stage6-toy-config-v6.schema.json")
    report = _load("stage6-pilot-report-v6.schema.json")
    preflight_properties = _mapping(preflight["properties"])
    prerequisite_properties = _mapping(prerequisite["properties"])
    config_properties = _mapping(config["properties"])
    report_properties = _mapping(report["properties"])
    assert _mapping(preflight_properties["schema_version"])["const"] == (STAGE6_GATE_SCHEMA_VERSION)
    assert _mapping(prerequisite_properties["schema_version"])["const"] == (
        STAGE6_GATE_SCHEMA_VERSION
    )
    assert _mapping(config_properties["schema_version"])["const"] == (STAGE6_TOY_CONFIG_VERSION)
    assert _mapping(report_properties["schema_version"])["const"] == (STAGE6_REPORT_SCHEMA_VERSION)
    reason_schema = _mapping(_mapping(preflight["$defs"])["reason"])
    assert set(_sequence(reason_schema["enum"])) == {item.value for item in Stage6Reason}
    prerequisite_reason_schema = _mapping(_mapping(prerequisite["$defs"])["reason"])
    assert set(_sequence(prerequisite_reason_schema["enum"])) == {
        item.value for item in Stage6Reason
    }
    assert set(_sequence(_mapping(prerequisite_properties["status"])["enum"])) == {
        item.value for item in Stage6PrerequisiteStatus
    }
    assert set(_sequence(_mapping(preflight_properties["verification_context"])["enum"])) == {
        item.value for item in Stage6VerificationContext
    }


def test_stage6_checked_in_config_and_report_safety_constants() -> None:
    config = load_stage6_toy_config(CONFIG_PATH)
    assert config.process_bound_capabilities_required is True
    assert config.capabilities_serialized is False
    assert config.detached_summaries_authoritative is False
    assert config.test_tuning_prohibited is True
    assert config.real_data_access is False

    report = _mapping(_load("stage6-pilot-report-v6.schema.json")["properties"])
    assert _mapping(report["authority_boundary"])["const"] == STAGE6_AUTHORITY_BOUNDARY
    for name, expected in {
        "thresholds_reestimated": False,
        "test_split_used_for_calibration": False,
        "test_tuning_prohibited": True,
        "capabilities_serialized": False,
        "detached_summaries_authoritative": False,
        "synthetic_capabilities_authorize_real_data": False,
        "synthetic_components_same_dataset_provenance": False,
        "real_authorization_issued": False,
        "dataset_root_probed": False,
        "real_data_used": False,
        "gpu_used": False,
        "network_used": False,
        "clinical_claim": False,
    }.items():
        assert _mapping(report[name])["const"] is expected
