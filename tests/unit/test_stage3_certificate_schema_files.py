from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

from retinaentail.calibration.certificate import (
    STAGE3_CERTIFICATE_SCHEMA_VERSION,
    Stage3CertificateScope,
    Stage3SyntheticCertificate,
    Stage3WitnessEvidence,
    WitnessSide,
)
from retinaentail.schemas import KNOWN_DEGRADATIONS, LesionRegion, WitnessRecord
from retinaentail.witnesses.stage3_replay import (
    STAGE3_REPLAY_SCHEMA_VERSION,
    PairReplayFailureReason,
    ReplayFailureReason,
    Stage3DualReplayResult,
    Stage3ReplayResult,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = PROJECT_ROOT / "data_registry" / "schemas"
REPLAY_SCHEMA = "stage3-replay-v3.schema.json"
CERTIFICATE_SCHEMA = "stage3-certificate-v3.schema.json"


def _load(name: str) -> dict[str, object]:
    decoded = json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))
    assert isinstance(decoded, dict)
    return decoded


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


def _class_fields(value: type[Any]) -> set[str]:
    return {item.name for item in fields(value)}


def _assert_exact_object_contract(schema: dict[str, object], expected: set[str]) -> None:
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(_mapping(schema["properties"])) == expected
    assert set(_sequence(schema["required"])) == expected


def _assert_typed_objects_are_closed(value: object) -> None:
    if isinstance(value, dict):
        if value.get("type") == "object":
            assert value.get("additionalProperties") is False
        for child in value.values():
            _assert_typed_objects_are_closed(child)
    elif isinstance(value, list):
        for child in value:
            _assert_typed_objects_are_closed(child)


def test_stage3_replay_and_certificate_schemas_are_offline_closed_draft_202012() -> None:
    for name in (REPLAY_SCHEMA, CERTIFICATE_SCHEMA):
        schema = _load(name)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["additionalProperties"] is False
        _assert_typed_objects_are_closed(schema)

    certificate = _load(CERTIFICATE_SCHEMA)
    serialized = json.dumps(certificate, sort_keys=True)
    assert "evaluation-result-v3.schema.json" in serialized
    assert "stage3-replay-v3.schema.json" in serialized
    assert "http://" not in serialized


def test_schema_fields_exactly_match_python_serialized_dataclasses() -> None:
    replay = _load(REPLAY_SCHEMA)
    replay_defs = _mapping(replay["$defs"])
    _assert_exact_object_contract(replay, _class_fields(Stage3DualReplayResult))
    _assert_exact_object_contract(
        _mapping(replay_defs["replayResult"]),
        _class_fields(Stage3ReplayResult),
    )

    certificate = _load(CERTIFICATE_SCHEMA)
    certificate_defs = _mapping(certificate["$defs"])
    _assert_exact_object_contract(certificate, _class_fields(Stage3SyntheticCertificate))
    _assert_exact_object_contract(
        _mapping(certificate_defs["evidence"]),
        _class_fields(Stage3WitnessEvidence),
    )
    _assert_exact_object_contract(
        _mapping(certificate_defs["witness"]),
        _class_fields(WitnessRecord),
    )
    _assert_exact_object_contract(
        _mapping(certificate_defs["region"]),
        _class_fields(LesionRegion),
    )


def test_versions_scopes_sides_and_failure_reason_enums_match_python() -> None:
    replay = _load(REPLAY_SCHEMA)
    replay_properties = _mapping(replay["properties"])
    replay_defs = _mapping(replay["$defs"])
    replay_result_properties = _mapping(_mapping(replay_defs["replayResult"])["properties"])
    assert _mapping(replay_properties["schema_version"])["const"] == (STAGE3_REPLAY_SCHEMA_VERSION)
    assert _mapping(replay_result_properties["schema_version"])["const"] == (
        STAGE3_REPLAY_SCHEMA_VERSION
    )
    assert _string_set(_mapping(replay_defs["replayFailureReason"])["enum"]) == {
        item.value for item in ReplayFailureReason
    }
    assert _string_set(_mapping(replay_defs["pairFailureReason"])["enum"]) == {
        item.value for item in PairReplayFailureReason
    }
    assert _string_set(_mapping(replay_result_properties["side"])["enum"]) == {
        item.value for item in WitnessSide
    }

    certificate = _load(CERTIFICATE_SCHEMA)
    properties = _mapping(certificate["properties"])
    definitions = _mapping(certificate["$defs"])
    evidence_properties = _mapping(_mapping(definitions["evidence"])["properties"])
    assert _mapping(properties["schema_version"])["const"] == (STAGE3_CERTIFICATE_SCHEMA_VERSION)
    assert _mapping(properties["scope"])["const"] == Stage3CertificateScope.SYNTHETIC_TOY
    assert _string_set(_mapping(evidence_properties["side"])["enum"]) == {
        item.value for item in WitnessSide
    }
    operation_properties = _mapping(_mapping(definitions["operation"])["properties"])
    assert _string_set(_mapping(operation_properties["name"])["enum"]) == set(KNOWN_DEGRADATIONS)


def test_all_digest_and_nullable_fields_are_explicit() -> None:
    replay = _load(REPLAY_SCHEMA)
    replay_defs = _mapping(replay["$defs"])
    sha256 = _mapping(replay_defs["sha256"])
    assert sha256 == {"type": "string", "pattern": "^[0-9a-f]{64}$"}
    replay_properties = _mapping(_mapping(replay_defs["replayResult"])["properties"])
    for name in (
        "region_sha256",
        "expected_observation_sha256",
        "expected_trace_sha256",
    ):
        assert _mapping(replay_properties[name]) == {"$ref": "#/$defs/nullableSha256"}
    for name in (
        "expected_latent_sha256",
        "actual_latent_sha256",
        "actual_observation_sha256",
        "actual_trace_sha256",
    ):
        assert _mapping(replay_properties[name]) == {"$ref": "#/$defs/nullableSha256"}
    assert _mapping(replay_properties["replay_error"]) == {"$ref": "#/$defs/nullableReplayError"}

    certificate = _load(CERTIFICATE_SCHEMA)
    certificate_defs = _mapping(certificate["$defs"])
    certificate_properties = _mapping(certificate["properties"])
    assert _mapping(certificate_defs["sha256"]) == sha256
    for name in (
        "calibration_artifact_hash",
        "observation_sha256",
        "degradation_trace_sha256",
    ):
        assert _mapping(certificate_properties[name]) == {"$ref": "#/$defs/sha256"}
    assert "data_hash" not in certificate_properties


def test_success_and_failure_conditionals_are_explicit_and_separate() -> None:
    replay = _load(REPLAY_SCHEMA)
    pair_conditions = _sequence(replay["allOf"])
    assert len(pair_conditions) == 2
    replay_defs = _mapping(replay["$defs"])
    result_conditions = _sequence(_mapping(replay_defs["replayResult"])["allOf"])
    assert len(result_conditions) == 2
    success_then = _mapping(_mapping(result_conditions[0])["then"])
    success_properties = _mapping(success_then["properties"])
    assert _mapping(success_properties["predicate_valid"])["const"] is True
    assert _mapping(success_properties["failure_reasons"])["maxItems"] == 0
    failure_then = _mapping(_mapping(result_conditions[1])["then"])
    assert _mapping(_mapping(failure_then["properties"])["failure_reasons"])["minItems"] == 1

    certificate = _load(CERTIFICATE_SCHEMA)
    certificate_conditions = _sequence(certificate["allOf"])
    assert len(certificate_conditions) == 2
    serialized_success = json.dumps(_mapping(certificate_conditions[0]), sort_keys=True)
    serialized_failure = json.dumps(_mapping(certificate_conditions[1]), sort_keys=True)
    assert '"SUCCESS"' in serialized_success
    assert '"passed": {"const": true}' in serialized_success
    assert '"FAIL_CLOSED"' in serialized_failure
    assert '"OUT_OF_MODEL"' in serialized_failure


def test_degradation_parameter_objects_are_operator_specific_and_closed() -> None:
    certificate = _load(CERTIFICATE_SCHEMA)
    definitions = _mapping(certificate["$defs"])
    operation = _mapping(definitions["operation"])
    conditions = _sequence(operation["allOf"])
    assert len(conditions) == len(KNOWN_DEGRADATIONS)
    parameter_definitions = {
        "identityParameters",
        "gaussianParameters",
        "defocusParameters",
        "motionParameters",
        "downsampleParameters",
        "jpegParameters",
        "noiseParameters",
        "gammaParameters",
        "vignettingParameters",
        "illuminationParameters",
    }
    for name in parameter_definitions:
        parameters = _mapping(definitions[name])
        assert parameters["type"] == "object"
        assert parameters["additionalProperties"] is False
