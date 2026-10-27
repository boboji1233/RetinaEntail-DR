from __future__ import annotations

import json
from pathlib import Path

from retinaentail.evaluation.gate import (
    REQUIRED_STAGE4_REVIEW_PURPOSES,
    STAGE4_LOCKED_EVIDENCE_SCHEMA_VERSION,
    STAGE4_PREFLIGHT_SCHEMA_VERSION,
    Stage4LockedEvidence,
    Stage4PreflightReason,
    Stage4PreflightStatus,
    Stage4ReviewPurpose,
    assess_stage4_preflight,
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


def _locked_evidence() -> Stage4LockedEvidence:
    return Stage4LockedEvidence(
        dataset_id="idrid",
        dataset_version="release-v1",
        license_review_sha256="1" * 64,
        adapter_spec_sha256="2" * 64,
        stage2_audit_sha256="3" * 64,
        frozen_data_card_sha256="4" * 64,
        stage3_calibration_package_sha256="5" * 64,
        manual_confirmation_sha256="6" * 64,
        calibration_artifact_sha256="7" * 64,
        config_sha256="8" * 64,
        data_sha256="9" * 64,
        model_sha256="a" * 64,
        alpha=0.05,
        positive_threshold=0.1,
        negative_threshold=0.2,
        positive_calibration_sample_count=5,
        negative_calibration_sample_count=7,
        total_calibration_sample_count=12,
    )


def _assert_closed_objects(value: object) -> None:
    if isinstance(value, dict):
        if value.get("type") == "object":
            assert value.get("additionalProperties") is False
        for child in value.values():
            _assert_closed_objects(child)
    elif isinstance(value, list):
        for child in value:
            _assert_closed_objects(child)


def test_stage4_schemas_decode_and_close_every_object() -> None:
    for name in (
        "stage4-locked-evidence-v4.schema.json",
        "stage4-preflight-v4.schema.json",
    ):
        schema = _load(name)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["additionalProperties"] is False
        _assert_closed_objects(schema)


def test_schema_properties_exactly_match_runtime_payload_contracts() -> None:
    evidence = _locked_evidence()
    report = assess_stage4_preflight(
        dataset_id="idrid",
        dataset_version="release-v1",
        approved_review_provided=True,
        approved_review_purposes=REQUIRED_STAGE4_REVIEW_PURPOSES,
        dataset_root_provided=True,
        adapter_spec_provided=True,
        stage2_pass_audit_provided=True,
        frozen_data_card_provided=True,
        stage3_calibration_package_provided=True,
        manual_confirmation_record_provided=True,
        locked_evidence_provided=True,
        locked_evidence=evidence,
    )
    evidence_schema = _load("stage4-locked-evidence-v4.schema.json")
    preflight_schema = _load("stage4-preflight-v4.schema.json")

    assert set(_mapping(evidence_schema["properties"])) == set(evidence.to_dict())
    assert set(_sequence(evidence_schema["required"])) == set(evidence.to_dict())
    assert set(_mapping(preflight_schema["properties"])) == set(report.to_dict())
    assert set(_sequence(preflight_schema["required"])) == set(report.to_dict())

    definitions = _mapping(preflight_schema["$defs"])
    nested = _mapping(definitions["lockedEvidence"])
    assert set(_mapping(nested["properties"])) == set(evidence.to_dict())
    assert set(_sequence(nested["required"])) == set(evidence.to_dict())


def test_schema_versions_enums_required_purposes_and_zero_io_constants_match_runtime() -> None:
    evidence = _load("stage4-locked-evidence-v4.schema.json")
    preflight = _load("stage4-preflight-v4.schema.json")
    evidence_properties = _mapping(evidence["properties"])
    preflight_properties = _mapping(preflight["properties"])
    definitions = _mapping(preflight["$defs"])

    assert _mapping(evidence_properties["schema_version"])["const"] == (
        STAGE4_LOCKED_EVIDENCE_SCHEMA_VERSION
    )
    assert _mapping(preflight_properties["schema_version"])["const"] == (
        STAGE4_PREFLIGHT_SCHEMA_VERSION
    )
    assert set(_sequence(_mapping(preflight_properties["status"])["enum"])) == {
        item.value for item in Stage4PreflightStatus
    }
    assert set(_sequence(_mapping(definitions["reviewPurpose"])["enum"])) == {
        item.value for item in Stage4ReviewPurpose
    }
    assert set(_sequence(_mapping(definitions["preflightReason"])["enum"])) == {
        item.value for item in Stage4PreflightReason
    }
    for field_name, expected in (
        ("test_tuning_prohibited", True),
        ("prerequisites_verified", False),
        ("authorizes_data_access", False),
        ("real_data_read", False),
        ("sample_count", 0),
    ):
        assert _mapping(preflight_properties[field_name])["const"] == expected

    ready_rule = _mapping(_sequence(preflight["allOf"])[1])
    ready_properties = _mapping(_mapping(ready_rule["then"])["properties"])
    assert _mapping(ready_properties["locked_evidence_provided"])["const"] is True
    assert "locked_evidence" not in ready_properties
    assert "locked_evidence_binding_sha256" not in ready_properties
    purposes = _sequence(_mapping(ready_properties["approved_review_purposes"])["allOf"])
    required_by_schema = {_mapping(_mapping(rule)["contains"])["const"] for rule in purposes}
    assert required_by_schema == {purpose.value for purpose in REQUIRED_STAGE4_REVIEW_PURPOSES}


def test_all_stage4_digest_fields_use_strict_lowercase_sha256() -> None:
    evidence = _load("stage4-locked-evidence-v4.schema.json")
    properties = _mapping(evidence["properties"])
    definitions = _mapping(evidence["$defs"])
    assert _mapping(definitions["sha256"]) == {
        "type": "string",
        "pattern": "^[0-9a-f]{64}$",
    }
    digest_fields = {
        "license_review_sha256",
        "adapter_spec_sha256",
        "stage2_audit_sha256",
        "frozen_data_card_sha256",
        "stage3_calibration_package_sha256",
        "manual_confirmation_sha256",
        "calibration_artifact_sha256",
        "config_sha256",
        "data_sha256",
        "model_sha256",
    }
    for field_name in digest_fields:
        assert _mapping(properties[field_name]) == {"$ref": "#/$defs/sha256"}
