from __future__ import annotations

import json
from pathlib import Path

import retinaentail.calibration.data_card as data_card_module
from retinaentail.calibration.data_card import (
    DATA_CARD_SCHEMA_VERSION,
    DatasetCardScope,
    FrozenDatasetCard,
    IdentityConfirmations,
    SplitSampleCounts,
)
from retinaentail.calibration.protocol import (
    CALIBRATION_METHOD_VERSION,
    CALIBRATION_SCHEMA_VERSION,
    CalibrationFailureReason,
    CalibrationLabel,
    CalibrationRunStatus,
    CalibrationSample,
    EvaluationFailureReason,
    EvaluationStatus,
    HypothesisEvaluation,
    ProtocolScope,
    SampleSplit,
    calibrate_order_statistic,
    evaluate_hypotheses,
)
from retinaentail.calibration.stage3_gate import (
    STAGE3_PREFLIGHT_SCHEMA_VERSION,
    Stage3PreflightReason,
    Stage3PreflightStatus,
    assess_stage3_preflight,
)
from retinaentail.calibration.toy_fixture import (
    STAGE3_TOY_CONFIG_VERSION,
    load_stage3_toy_config,
)
from retinaentail.data.audit import AUDIT_SCHEMA_VERSION
from retinaentail.data.stage2_models import LabelSemantic
from retinaentail.entailment.states import EntailmentState
from retinaentail.schemas import KNOWN_DEGRADATIONS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = PROJECT_ROOT / "data_registry" / "schemas"
SCHEMA_NAMES = (
    "stage3-preflight-v3.schema.json",
    "calibration-artifact-v3.schema.json",
    "evaluation-result-v3.schema.json",
    "frozen-data-card-v3.schema.json",
    "stage3-toy-config-v3.schema.json",
)
CONFIG_HASH = "1" * 64
DATA_HASH = "2" * 64
MODEL_HASH = "3" * 64


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


def _properties(schema: dict[str, object]) -> dict[str, object]:
    return _mapping(schema["properties"])


def _assert_exact_payload_contract(schema: dict[str, object], payload: dict[str, object]) -> None:
    properties = _properties(schema)
    required = _sequence(schema["required"])
    assert set(properties) == set(required) == set(payload)


def _assert_object_schemas_are_closed(value: object) -> None:
    if isinstance(value, dict):
        if value.get("type") == "object":
            assert value.get("additionalProperties") is False
        for child in value.values():
            _assert_object_schemas_are_closed(child)
    elif isinstance(value, list):
        for child in value:
            _assert_object_schemas_are_closed(child)


def _successful_calibration_payloads() -> tuple[dict[str, object], dict[str, object]]:
    samples = tuple(
        CalibrationSample(
            sample_id=f"present-{index}",
            split=SampleSplit.CALIBRATION,
            label=CalibrationLabel.LESION_PRESENT,
            positive_score=score,
            negative_score=0.9,
        )
        for index, score in enumerate((0.1, 0.2, 0.3), start=1)
    ) + tuple(
        CalibrationSample(
            sample_id=f"absent-{index}",
            split=SampleSplit.CALIBRATION,
            label=CalibrationLabel.LESION_ABSENT,
            positive_score=0.9,
            negative_score=score,
        )
        for index, score in enumerate((0.4, 0.5, 0.6), start=1)
    )
    artifact = calibrate_order_statistic(
        samples,
        alpha=0.25,
        config_hash=CONFIG_HASH,
        data_hash=DATA_HASH,
        model_hash=MODEL_HASH,
    )
    assert artifact.status is CalibrationRunStatus.SUCCESS
    evaluation = evaluate_hypotheses(
        HypothesisEvaluation(
            case_id="synthetic-case",
            lesion_class="toy_dot",
            positive_score=0.2,
            negative_score=0.7,
            config_hash=CONFIG_HASH,
            data_hash=DATA_HASH,
            model_hash=MODEL_HASH,
        ),
        artifact,
    )
    assert evaluation.status is EvaluationStatus.SUCCESS
    return artifact.to_dict(), evaluation.to_dict()


def _frozen_card_payload() -> dict[str, object]:
    card = FrozenDatasetCard(
        scope=DatasetCardScope.APPROVED_REAL_DATA,
        dataset_id="idrid",
        dataset_version="1.0",
        manifest_sha256=DATA_HASH,
        registry_sha256="4" * 64,
        adapter_spec_sha256="5" * 64,
        audit_sha256="6" * 64,
        audit_schema_version=AUDIT_SCHEMA_VERSION,
        license_review_sha256="7" * 64,
        license_review_schema_version="1.0.0",
        license_terms_sha256="8" * 64,
        license_evidence_sha256="9" * 64,
        reviewer_fingerprint="a" * 64,
        reviewed_at="2026-08-23T12:00:00+08:00",
        label_semantics=(LabelSemantic.MICROANEURYSM_MASK,),
        label_semantics_evidence_sha256="b" * 64,
        confirmations=IdentityConfirmations(
            source_image_id=True,
            patient_id=True,
            eye_id=True,
            followup_id=True,
            intervention_pair_id=True,
            split_isolation=True,
        ),
        identity_evidence_sha256="c" * 64,
        split_counts=SplitSampleCounts(train=1, calibration=1, test=1),
        sample_count=3,
        config_sha256=CONFIG_HASH,
        data_sha256=DATA_HASH,
        model_sha256=MODEL_HASH,
        _token=data_card_module._CARD_TOKEN,
    )
    return card.to_dict()


def test_all_stage3_json_schemas_decode_and_close_every_object() -> None:
    for name in SCHEMA_NAMES:
        schema = _load(name)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["additionalProperties"] is False
        _assert_object_schemas_are_closed(schema)


def test_schema_properties_exactly_match_current_to_dict_contracts() -> None:
    blocked = assess_stage3_preflight(
        dataset_id=None,
        dataset_version=None,
        review_record_provided=False,
        dataset_root_provided=False,
        adapter_spec_provided=False,
    ).to_dict()
    artifact, evaluation = _successful_calibration_payloads()
    card = _frozen_card_payload()
    toy_config = load_stage3_toy_config(
        PROJECT_ROOT / "configs" / "stage3_toy_calibration.yaml"
    ).to_dict()

    _assert_exact_payload_contract(_load("stage3-preflight-v3.schema.json"), blocked)
    _assert_exact_payload_contract(_load("calibration-artifact-v3.schema.json"), artifact)
    _assert_exact_payload_contract(_load("evaluation-result-v3.schema.json"), evaluation)
    _assert_exact_payload_contract(_load("frozen-data-card-v3.schema.json"), card)
    _assert_exact_payload_contract(_load("stage3-toy-config-v3.schema.json"), toy_config)

    card_schema = _load("frozen-data-card-v3.schema.json")
    card_defs = _mapping(card_schema["$defs"])
    confirmations = _mapping(card["identity_confirmations"])
    counts = _mapping(card["split_counts"])
    _assert_exact_payload_contract(_mapping(card_defs["identityConfirmations"]), confirmations)
    _assert_exact_payload_contract(_mapping(card_defs["splitCounts"]), counts)


def test_controlled_enums_match_python_contracts() -> None:
    preflight = _load("stage3-preflight-v3.schema.json")
    preflight_properties = _properties(preflight)
    assert _string_set(_mapping(preflight_properties["status"])["enum"]) == {
        item.value for item in Stage3PreflightStatus
    }
    reason_items = _mapping(_mapping(preflight_properties["reasons"])["items"])
    assert _string_set(reason_items["enum"]) == {item.value for item in Stage3PreflightReason}

    artifact = _load("calibration-artifact-v3.schema.json")
    artifact_properties = _properties(artifact)
    artifact_defs = _mapping(artifact["$defs"])
    assert _string_set(_mapping(artifact_properties["protocol_scope"])["enum"]) == {
        item.value for item in ProtocolScope
    }
    assert _string_set(_mapping(artifact_properties["status"])["enum"]) == {
        item.value for item in CalibrationRunStatus
    }
    assert _string_set(_mapping(artifact_defs["failureReason"])["enum"]) == {
        item.value for item in CalibrationFailureReason
    }

    evaluation = _load("evaluation-result-v3.schema.json")
    evaluation_properties = _properties(evaluation)
    evaluation_defs = _mapping(evaluation["$defs"])
    assert _string_set(_mapping(evaluation_properties["state"])["enum"]) == {
        item.value for item in EntailmentState
    }
    assert _string_set(_mapping(evaluation_properties["status"])["enum"]) == {
        item.value for item in EvaluationStatus
    }
    assert _string_set(_mapping(evaluation_defs["failureReason"])["enum"]) == {
        item.value for item in EvaluationFailureReason
    }

    card = _load("frozen-data-card-v3.schema.json")
    card_properties = _properties(card)
    assert _string_set(_mapping(card_properties["scope"])["enum"]) == {
        item.value for item in DatasetCardScope
    }
    lesion_semantics = {
        LabelSemantic.MICROANEURYSM_MASK,
        LabelSemantic.HEMORRHAGE_MASK,
        LabelSemantic.HARD_EXUDATE_MASK,
        LabelSemantic.SOFT_EXUDATE_MASK,
        LabelSemantic.LESION_DETECTION,
    }
    label_items = _mapping(_mapping(card_properties["label_semantics"])["items"])
    assert _string_set(label_items["enum"]) == {item.value for item in lesion_semantics}

    toy_config = _load("stage3-toy-config-v3.schema.json")
    toy_defs = _mapping(toy_config["$defs"])
    assert _string_set(_mapping(toy_defs["knownDegradation"])["enum"]) == set(KNOWN_DEGRADATIONS)


def test_versions_are_constants_bound_to_python_contracts() -> None:
    preflight = _properties(_load("stage3-preflight-v3.schema.json"))
    artifact = _properties(_load("calibration-artifact-v3.schema.json"))
    evaluation = _properties(_load("evaluation-result-v3.schema.json"))
    card = _properties(_load("frozen-data-card-v3.schema.json"))
    toy_config = _properties(_load("stage3-toy-config-v3.schema.json"))

    assert _mapping(preflight["schema_version"])["const"] == STAGE3_PREFLIGHT_SCHEMA_VERSION
    assert _mapping(artifact["schema_version"])["const"] == CALIBRATION_SCHEMA_VERSION
    assert _mapping(evaluation["schema_version"])["const"] == CALIBRATION_SCHEMA_VERSION
    assert _mapping(artifact["method_version"])["const"] == CALIBRATION_METHOD_VERSION
    assert _mapping(evaluation["method_version"])["const"] == CALIBRATION_METHOD_VERSION
    assert _mapping(card["schema_version"])["const"] == DATA_CARD_SCHEMA_VERSION
    assert _mapping(card["audit_schema_version"])["const"] == AUDIT_SCHEMA_VERSION
    assert _mapping(toy_config["schema_version"])["const"] == STAGE3_TOY_CONFIG_VERSION


def test_toy_config_ranges_and_operator_parameter_objects_match_runtime_contract() -> None:
    schema = _load("stage3-toy-config-v3.schema.json")
    properties = _properties(schema)
    definitions = _mapping(schema["$defs"])

    assert _mapping(properties["seed"]) == {
        "type": "integer",
        "minimum": 0,
        "maximum": 2_000_000_000,
    }
    assert _mapping(properties["image_size"]) == {
        "type": "integer",
        "minimum": 48,
        "maximum": 512,
    }
    assert _mapping(properties["lesion_radius"]) == {
        "type": "integer",
        "minimum": 2,
        "maximum": 64,
    }
    assert _mapping(properties["alpha"]) == {
        "type": "number",
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 1,
    }
    levels = _mapping(properties["calibration_residual_levels"])
    assert levels["minItems"] == 3
    assert levels["uniqueItems"] is True
    assert _mapping(levels["items"]) == {
        "type": "integer",
        "minimum": 1,
        "maximum": 32,
    }

    degradations = _mapping(properties["degradations"])
    assert degradations["minItems"] == 1
    branches = _sequence(_mapping(degradations["items"])["oneOf"])
    assert len(branches) == len(KNOWN_DEGRADATIONS)
    branch_names: set[str] = set()
    for branch_reference in branches:
        reference = _mapping(branch_reference)["$ref"]
        assert isinstance(reference, str)
        definition_name = reference.removeprefix("#/$defs/")
        branch = _mapping(definitions[definition_name])
        assert branch["additionalProperties"] is False
        assert set(_sequence(branch["required"])) == {"name", "severity"}
        branch_properties = _mapping(branch["properties"])
        name_conditions = _sequence(_mapping(branch_properties["name"])["allOf"])
        name = _mapping(name_conditions[1])["const"]
        assert isinstance(name, str)
        branch_names.add(name)
        parameter_schema = _mapping(definitions["noParameters"])
        if name != "identity":
            parameter_schema = _mapping(branch_properties["parameters"])
        assert parameter_schema["type"] == "object"
        assert parameter_schema["additionalProperties"] is False
    assert branch_names == set(KNOWN_DEGRADATIONS)


def test_all_digest_fields_use_strict_lowercase_sha256_definition() -> None:
    strict_fields = {
        "manifest_sha256",
        "registry_sha256",
        "adapter_spec_sha256",
        "audit_sha256",
        "license_review_sha256",
        "license_terms_sha256",
        "license_evidence_sha256",
        "reviewer_fingerprint",
        "label_semantics_evidence_sha256",
        "identity_evidence_sha256",
        "config_sha256",
        "data_sha256",
        "model_sha256",
    }
    schema = _load("frozen-data-card-v3.schema.json")
    definitions = _mapping(schema["$defs"])
    sha256 = _mapping(definitions["sha256"])
    assert sha256 == {"type": "string", "pattern": "^[0-9a-f]{64}$"}
    properties = _properties(schema)
    for field_name in strict_fields:
        assert _mapping(properties[field_name]) == {"$ref": "#/$defs/sha256"}

    for name, digest_fields in (
        ("calibration-artifact-v3.schema.json", {"config_hash", "data_hash", "model_hash"}),
        (
            "evaluation-result-v3.schema.json",
            {
                "config_hash",
                "data_hash",
                "model_hash",
                "input_config_hash",
                "input_data_hash",
                "input_model_hash",
            },
        ),
    ):
        schema = _load(name)
        definitions = _mapping(schema["$defs"])
        sha256 = _mapping(definitions["sha256"])
        assert sha256 == {"type": "string", "pattern": "^[0-9a-f]{64}$"}
        nullable = _mapping(definitions["nullableSha256"])
        assert _sequence(nullable["anyOf"]) == [
            {"$ref": "#/$defs/sha256"},
            {"type": "null"},
        ]
        properties = _properties(schema)
        for field_name in digest_fields:
            assert _mapping(properties[field_name]) == {"$ref": "#/$defs/nullableSha256"}
        success = _mapping(_mapping(_sequence(schema["allOf"])[0])["then"])
        success_properties = _mapping(success["properties"])
        for field_name in digest_fields:
            assert _mapping(success_properties[field_name]) == {"$ref": "#/$defs/sha256"}


def test_nullable_fields_are_explicit_and_success_failure_rules_are_separate() -> None:
    preflight = _properties(_load("stage3-preflight-v3.schema.json"))
    assert _string_set(_mapping(preflight["dataset_id"])["type"]) == {"string", "null"}
    assert _string_set(_mapping(preflight["dataset_version"])["type"]) == {"string", "null"}

    artifact_schema = _load("calibration-artifact-v3.schema.json")
    artifact = _properties(artifact_schema)
    assert _mapping(artifact["positive_threshold"])["$ref"] == ("#/$defs/nullableNonnegativeNumber")
    assert _mapping(artifact["negative_threshold"])["$ref"] == ("#/$defs/nullableNonnegativeNumber")
    assert len(_sequence(artifact_schema["allOf"])) == 2

    evaluation_schema = _load("evaluation-result-v3.schema.json")
    evaluation = _properties(evaluation_schema)
    for field_name in (
        "positive_score",
        "negative_score",
        "positive_threshold",
        "negative_threshold",
    ):
        assert _mapping(evaluation[field_name])["$ref"] == ("#/$defs/nullableNonnegativeNumber")
    for field_name in ("positive_feasible", "negative_feasible"):
        assert _mapping(evaluation[field_name])["$ref"] == "#/$defs/nullableBoolean"
    assert len(_sequence(evaluation_schema["allOf"])) == 2


def test_frozen_card_has_no_nullable_payload_field() -> None:
    card = _load("frozen-data-card-v3.schema.json")

    def assert_no_null(value: object) -> None:
        if isinstance(value, dict):
            type_value = value.get("type")
            if isinstance(type_value, list):
                assert "null" not in type_value
            for child in value.values():
                assert_no_null(child)
        elif isinstance(value, list):
            for child in value:
                assert_no_null(child)

    assert_no_null(card)
