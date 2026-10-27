from __future__ import annotations

import json
from pathlib import Path

from retinaentail.pilot.toy_fixture import (
    STAGE5_TOY_CONFIG_VERSION,
    load_stage5_toy_config,
)
from retinaentail.reporting.stage5 import (
    STAGE5_AUTHORITY_BOUNDARY,
    STAGE5_PILOT_REPORT_SCHEMA_VERSION,
    STAGE5_SOURCE_DIGEST_METHOD,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "stage5_toy_locked_pilot.yaml"
SCHEMA_DIR = PROJECT_ROOT / "data_registry" / "schemas"
CONFIG_SCHEMA = SCHEMA_DIR / "stage5-toy-config-v5.schema.json"
REPORT_SCHEMA = SCHEMA_DIR / "stage5-pilot-report-v5.schema.json"


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


def test_stage5_schemas_are_versioned_closed_and_offline() -> None:
    for path in (CONFIG_SCHEMA, REPORT_SCHEMA):
        schema = _mapping(json.loads(path.read_text(encoding="utf-8")))
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["additionalProperties"] is False
        assert "$data" not in json.dumps(schema, sort_keys=True)
        assert "http://" not in json.dumps(schema, sort_keys=True)
        _assert_typed_objects_closed(schema)


def test_stage5_config_schema_pins_source_digest_lock_and_case_order() -> None:
    schema = _mapping(json.loads(CONFIG_SCHEMA.read_text(encoding="utf-8")))
    properties = _mapping(schema["properties"])
    definitions = _mapping(schema["$defs"])
    assert set(properties) == _string_set(schema["required"])
    assert _mapping(properties["schema_version"])["const"] == STAGE5_TOY_CONFIG_VERSION
    source = _mapping(definitions["stage4Source"])
    source_properties = _mapping(source["properties"])
    assert set(source_properties) == _string_set(source["required"])
    digest = _mapping(source_properties["package_sha256"])
    digest_conditions = _sequence(digest["allOf"])
    assert digest_conditions[-1] == {
        "const": "bfb7b22393bf6bafd6944a6ef091fa2464d70b95a344ad5d78cbd0fb3d1713df"
    }
    cases = _mapping(properties["pilot_case_ids"])
    prefix = _sequence(cases["prefixItems"])
    assert cases["minItems"] == cases["maxItems"] == len(prefix) == 7
    assert cases["items"] is False


def test_checked_in_stage5_config_roundtrips_exactly() -> None:
    config = load_stage5_toy_config(CONFIG_PATH)
    payload = config.to_dict()
    assert payload["schema_version"] == STAGE5_TOY_CONFIG_VERSION
    assert payload["threshold_policy"] == "LOCKED_STAGE3_NO_REESTIMATION"
    assert payload["test_tuning_prohibited"] is True
    assert payload["real_data_access"] is False
    source = _mapping(payload["source_stage4"])
    assert source["package_file_count"] == 88
    assert source["alpha"] == 0.25
    assert source["total_sample_count"] == 8


def test_stage5_report_schema_matches_runtime_authority_and_safety_constants() -> None:
    schema = _mapping(json.loads(REPORT_SCHEMA.read_text(encoding="utf-8")))
    properties = _mapping(schema["properties"])
    assert set(properties) == _string_set(schema["required"])
    assert _mapping(properties["schema_version"])["const"] == (STAGE5_PILOT_REPORT_SCHEMA_VERSION)
    assert _mapping(properties["source_digest_method"])["const"] == (STAGE5_SOURCE_DIGEST_METHOD)
    assert _mapping(properties["authority_boundary"])["const"] == STAGE5_AUTHORITY_BOUNDARY
    for name, expected in {
        "thresholds_locked_before_pilot": True,
        "thresholds_reestimated": False,
        "test_split_used_for_calibration": False,
        "test_tuning_prohibited": True,
        "gpu_used": False,
        "network_used": False,
        "real_data_used": False,
        "clinical_claim": False,
        "individual_report_authoritative": False,
    }.items():
        assert _mapping(properties[name])["const"] is expected
    assert _mapping(properties["evaluation"]) == {"$ref": "stage4-evaluation-report-v4.schema.json"}
