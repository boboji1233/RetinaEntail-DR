from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest

from retinaentail.data.audit import AuditResult, AuditStatus, AuditSummary
from retinaentail.data.stage2_models import (
    DatasetManifestV2,
    LabelAvailability,
    LabelRecordV2,
    LabelSemantic,
    ManifestRecordV2,
)
from retinaentail.evaluation.metrics import wilson_score_proportion
from retinaentail.pilot.identity import (
    Stage5IdentityRecord,
    audit_five_identity_split_isolation,
)
from retinaentail.pilot.stage6_gate import Stage6VerificationContext
from retinaentail.pilot.stage7_fixture import (
    STAGE7_TOY_CONFIG_VERSION,
    load_stage7_toy_config,
)
from retinaentail.pilot.stage7_gate import (
    STAGE7_GATE_SCHEMA_VERSION,
    Stage7PreflightStatus,
    Stage7Reason,
    assess_stage7_preflight,
)
from retinaentail.pilot.stage7_rebuild import (
    STAGE7_REBUILD_SCHEMA_VERSION,
    Stage7DifferenceKind,
    Stage7RebuildReason,
    Stage7RebuildStatus,
    verify_stage7_rebuild,
)
from retinaentail.reporting.stage7 import (
    STAGE7_AUTHORITY_BOUNDARY,
    STAGE7_METHOD,
    STAGE7_REPORT_SCHEMA_VERSION,
    issue_stage7_pilot_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = PROJECT_ROOT / "data_registry" / "schemas"
CONFIG_PATH = PROJECT_ROOT / "configs" / "stage7_toy_consumption_pilot.yaml"
STAGE7_SCHEMAS = (
    "stage7-preflight-v7.schema.json",
    "stage7-rebuild-v7.schema.json",
    "stage7-toy-config-v7.schema.json",
    "stage7-pilot-report-v7.schema.json",
)


def _load(name: str) -> dict[str, object]:
    value: object = json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def _mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    assert all(isinstance(key, str) for key in value)
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


def _json_equal(left: object, right: object) -> bool:
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def _schema_type_matches(value: object, expected: str) -> bool:
    if expected == "null":
        return value is None
    if expected == "boolean":
        return type(value) is bool  # noqa: E721
    if expected == "integer":
        return type(value) is int  # noqa: E721
    if expected == "number":
        return type(value) in {int, float} and math.isfinite(float(cast(int | float, value)))
    if expected == "string":
        return isinstance(value, str)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    raise AssertionError(f"unsupported schema type in lightweight test validator: {expected}")


def _resolve_ref(
    reference: str,
    root: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    if reference.startswith("#/"):
        target: object = root
        for segment in reference[2:].split("/"):
            target = _mapping(target)[segment.replace("~1", "/").replace("~0", "~")]
        return _mapping(target), root
    name, separator, fragment = reference.partition("#")
    external = _load(name)
    if not separator:
        return external, external
    target = cast(object, external)
    for segment in fragment.removeprefix("/").split("/"):
        target = _mapping(target)[segment.replace("~1", "/").replace("~0", "~")]
    return _mapping(target), external


def _matches_schema(
    value: object,
    schema: dict[str, object],
    *,
    root: dict[str, object],
) -> bool:
    reference = schema.get("$ref")
    if reference is not None:
        assert isinstance(reference, str)
        target, target_root = _resolve_ref(reference, root)
        if not _matches_schema(value, target, root=target_root):
            return False

    all_of = schema.get("allOf")
    if all_of is not None and not all(
        _matches_schema(value, _mapping(item), root=root) for item in _sequence(all_of)
    ):
        return False
    any_of = schema.get("anyOf")
    if any_of is not None and not any(
        _matches_schema(value, _mapping(item), root=root) for item in _sequence(any_of)
    ):
        return False
    one_of = schema.get("oneOf")
    if (
        one_of is not None
        and sum(_matches_schema(value, _mapping(item), root=root) for item in _sequence(one_of))
        != 1
    ):
        return False

    conditional = schema.get("if")
    if conditional is not None:
        branch = "then" if _matches_schema(value, _mapping(conditional), root=root) else "else"
        branch_schema = schema.get(branch)
        if branch_schema is not None and not _matches_schema(
            value,
            _mapping(branch_schema),
            root=root,
        ):
            return False

    if "const" in schema and not _json_equal(value, schema["const"]):
        return False
    enum = schema.get("enum")
    if enum is not None and not any(_json_equal(value, item) for item in _sequence(enum)):
        return False
    raw_type = schema.get("type")
    if raw_type is not None:
        expected_types = [raw_type] if isinstance(raw_type, str) else _sequence(raw_type)
        if not any(
            isinstance(expected, str) and _schema_type_matches(value, expected)
            for expected in expected_types
        ):
            return False

    if isinstance(value, str):
        minimum_length = schema.get("minLength")
        if isinstance(minimum_length, int) and len(value) < minimum_length:
            return False
        pattern = schema.get("pattern")
        if isinstance(pattern, str) and re.search(pattern, value) is None:
            return False
    if type(value) in {int, float}:  # noqa: E721
        numeric = float(cast(int | float, value))
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        exclusive_minimum = schema.get("exclusiveMinimum")
        exclusive_maximum = schema.get("exclusiveMaximum")
        if isinstance(minimum, int | float) and numeric < float(minimum):
            return False
        if isinstance(maximum, int | float) and numeric > float(maximum):
            return False
        if isinstance(exclusive_minimum, int | float) and numeric <= float(exclusive_minimum):
            return False
        if isinstance(exclusive_maximum, int | float) and numeric >= float(exclusive_maximum):
            return False
    if isinstance(value, list):
        minimum_items = schema.get("minItems")
        maximum_items = schema.get("maxItems")
        if isinstance(minimum_items, int) and len(value) < minimum_items:
            return False
        if isinstance(maximum_items, int) and len(value) > maximum_items:
            return False
        if schema.get("uniqueItems") is True:
            serialized = [json.dumps(item, sort_keys=True) for item in value]
            if len(serialized) != len(set(serialized)):
                return False
        item_schema = schema.get("items")
        if item_schema is not None and not all(
            _matches_schema(item, _mapping(item_schema), root=root) for item in value
        ):
            return False
    if isinstance(value, dict):
        required = schema.get("required")
        if required is not None and not set(cast(list[str], _sequence(required))).issubset(value):
            return False
        properties = {} if schema.get("properties") is None else _mapping(schema["properties"])
        if schema.get("additionalProperties") is False and not set(value).issubset(properties):
            return False
        for key, child in value.items():
            child_schema = properties.get(key)
            if child_schema is not None and not _matches_schema(
                child,
                _mapping(child_schema),
                root=root,
            ):
                return False
    return True


def _validate(name: str, instance: object) -> None:
    schema = _load(name)
    assert _matches_schema(instance, schema, root=schema)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _label(index: int) -> LabelRecordV2:
    digit = format(index + 4, "x")
    return LabelRecordV2(
        semantic=LabelSemantic.MICROANEURYSM_MASK,
        availability=LabelAvailability.AVAILABLE,
        relative_path=f"labels/case-{index}.png",
        sha256=digit * 64,
        byte_size=50 + index,
    )


def _record(index: int, split: str) -> ManifestRecordV2:
    digit = format(index + 1, "x")
    return ManifestRecordV2(
        dataset_id="idrid",
        relative_path=f"images/case-{index}.png",
        sha256=digit * 64,
        byte_size=100 + index,
        width=32,
        height=24,
        channels=3,
        image_format="PNG",
        source_image_id=f"source-{index}",
        patient_id=f"patient-{index}",
        eye_id=f"eye-{index}",
        followup_id=f"followup-{index}",
        intervention_pair_id=f"pair-{index}",
        split=split,
        labels=(_label(index),),
    )


def _manifest() -> DatasetManifestV2:
    return DatasetManifestV2(
        dataset_id="idrid",
        dataset_version="stage7-schema-fixture-v1",
        records=(
            _record(0, "train"),
            _record(1, "calibration"),
            _record(2, "test"),
        ),
    )


def _audit(manifest: DatasetManifestV2) -> AuditResult:
    return AuditResult(
        status=AuditStatus.PASS,
        dataset_id=manifest.dataset_id,
        dataset_version=manifest.dataset_version,
        manifest_sha256=_canonical_sha256(manifest.to_dict()),
        summary=AuditSummary(
            record_count=3,
            image_files_checked=3,
            label_files_checked=3,
            mask_files_decoded=3,
            unknown_label_count=0,
            issue_counts=(),
        ),
    )


def _identity_audit() -> dict[str, object]:
    records = tuple(
        Stage5IdentityRecord(
            dataset_id="retinaentail_stage5_toy",
            relative_path=f"images/{split}.png",
            split=split,
            source_image_id=f"source-{split}",
            patient_id=f"patient-{split}",
            eye_id=f"eye-{split}",
            followup_id=f"followup-{split}",
            intervention_pair_id=f"pair-{split}",
        )
        for split in ("train", "calibration", "test")
    )
    return cast(dict[str, object], audit_five_identity_split_isolation(records).to_dict())


def _locked_contract() -> dict[str, object]:
    return {
        "alpha": 0.25,
        "positive_threshold": 0.5,
        "negative_threshold": 0.5,
        "positive_sample_count": 4,
        "negative_sample_count": 4,
        "total_sample_count": 8,
        "calibration_artifact_hash": "4" * 64,
        "config_hash": "1" * 64,
        "data_hash": "2" * 64,
        "model_hash": "3" * 64,
    }


def _proportion(numerator: int, denominator: int = 7) -> dict[str, object]:
    return wilson_score_proportion(
        numerator,
        denominator,
        confidence_level=0.95,
    ).to_dict()


def _case_index() -> list[dict[str, object]]:
    return [
        {
            "pilot_index": index,
            "case_id": f"case-{index}",
            "reference_label": "LESION_PRESENT",
            "state": "ENTAILED_PRESENT",
            "status": "SUCCESS",
            "failure_reasons": [],
            "dual_witness_replay_passed": True,
            "positive_witness_replay_passed": True,
            "negative_witness_replay_passed": True,
            "evaluation_sha256": "5" * 64,
            "certificate_sha256": "6" * 64,
            "dual_replay_sha256": "7" * 64,
            "positive_witness_sha256": "8" * 64,
            "negative_witness_sha256": "9" * 64,
        }
        for index in range(7)
    ]


def _evaluation(locked: dict[str, object]) -> dict[str, object]:
    state_counts = {
        "ENTAILED_PRESENT": 7,
        "ENTAILED_ABSENT": 0,
        "RAW_UNIDENTIFIABLE": 0,
        "OUT_OF_MODEL": 0,
    }
    return {
        "schema_version": "4.0.0",
        "scope": "SYNTHETIC_ONLY",
        "split": "test",
        "clinical_claim": False,
        "status": "SUCCESS",
        "locked_contract": locked,
        "confidence_level": 0.95,
        "interval_method": "wilson_score",
        "input_record_count": 7,
        "test_case_count": 7,
        "rejected_record_count": 0,
        "evaluable_count": 7,
        "unevaluable_count": 0,
        "fail_closed_count": 0,
        "out_of_distribution_count": 0,
        "both_witness_replays_passed_count": 7,
        "one_or_more_witness_replays_failed_count": 0,
        "labeled_reference_count": 7,
        "state_counts": state_counts,
        "reference_label_counts": {
            "LESION_PRESENT": 7,
            "LESION_ABSENT": 0,
            "UNKNOWN": 0,
        },
        "failure_reason_counts": {},
        "rejected_reason_counts": {},
        "aggregation_failure_reasons": [],
        "state_proportions": {state: _proportion(count) for state, count in state_counts.items()},
        "evaluable_proportion": _proportion(7),
        "unevaluable_proportion": _proportion(0),
        "fail_closed_proportion": _proportion(0),
        "out_of_distribution_proportion": _proportion(0),
        "both_witness_replays_passed_proportion": _proportion(7),
        "one_or_more_witness_replays_failed_proportion": _proportion(0),
    }


def _pilot_report() -> dict[str, object]:
    locked = _locked_contract()
    cases = _case_index()
    report = issue_stage7_pilot_report(
        run_id="stage7-toy-authorization-consumption-001",
        stage6_source={
            "scope": "SYNTHETIC_ONLY",
            "run_id": "stage6-toy-verifier-pilot-001",
            "authority_boundary": "COMPLETE_VERIFIED_STAGE6_PACKAGE",
            "package_sha256": "a" * 64,
            "package_file_count": 122,
            "envelope_sha256": "b" * 64,
            "complete_byte_replay_verified": True,
        },
        authorization_consumption={
            "scope": "SYNTHETIC_ONLY",
            "ttl_seconds": 5,
            "process_bound": True,
            "non_serializable": True,
            "one_shot": True,
            "issued": True,
            "consumed": True,
            "second_consumption_rejected": True,
            "authorizes_embedded_fixture_replay": True,
            "authorizes_real_data": False,
            "authorizes_physical_dataset_root": False,
            "detached_receipt_authoritative": False,
        },
        rebuild_equivalence={
            "mode": "EMBEDDED_STAGE2_TYPED_ROUNDTRIP",
            "dataset_id": "synthetic_stage2",
            "dataset_version": "synthetic-fixture-v1",
            "record_count": 1,
            "original_manifest_sha256": "c" * 64,
            "rebuilt_manifest_sha256": "c" * 64,
            "record_fingerprints_sha256": "d" * 64,
            "canonical_equivalent": True,
            "byte_equivalent": True,
            "mismatch_count": 0,
            "physical_dataset_root_used": False,
            "missing_label_policy": "UNKNOWN_NEVER_NEGATIVE",
        },
        identity_isolation=_identity_audit(),
        locked_contract=locked,
        evaluation=_evaluation(locked),
        case_index=cases,
        stage7_config_sha256="e" * 64,
        stage7_source_payload_sha256="f" * 64,
        stage7_model_provenance_sha256="0" * 64,
    )
    return report.to_dict()


def test_stage7_schemas_are_valid_closed_draft_202012_and_offline() -> None:
    for name in STAGE7_SCHEMAS:
        schema = _load(name)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["$id"] == f"https://retinaentail.local/schemas/{name}"
        assert schema["additionalProperties"] is False
        assert "http://" not in json.dumps(schema, sort_keys=True)
        _closed_objects(schema)


def test_stage7_runtime_versions_statuses_reasons_and_difference_kinds_match() -> None:
    preflight = _load("stage7-preflight-v7.schema.json")
    rebuild = _load("stage7-rebuild-v7.schema.json")
    config = _load("stage7-toy-config-v7.schema.json")
    report = _load("stage7-pilot-report-v7.schema.json")
    preflight_properties = _mapping(preflight["properties"])
    rebuild_properties = _mapping(rebuild["properties"])
    config_properties = _mapping(config["properties"])
    report_properties = _mapping(report["properties"])

    assert _mapping(preflight_properties["schema_version"])["const"] == (STAGE7_GATE_SCHEMA_VERSION)
    assert _mapping(rebuild_properties["schema_version"])["const"] == (
        STAGE7_REBUILD_SCHEMA_VERSION
    )
    assert _mapping(config_properties["schema_version"])["const"] == (STAGE7_TOY_CONFIG_VERSION)
    assert _mapping(report_properties["schema_version"])["const"] == (STAGE7_REPORT_SCHEMA_VERSION)
    assert _mapping(report_properties["method"])["const"] == STAGE7_METHOD
    assert _mapping(report_properties["authority_boundary"])["const"] == (STAGE7_AUTHORITY_BOUNDARY)
    assert set(_sequence(_mapping(preflight_properties["status"])["enum"])) == {
        item.value for item in Stage7PreflightStatus
    }
    assert set(_sequence(_mapping(_mapping(preflight["$defs"])["reason"])["enum"])) == {
        item.value for item in Stage7Reason
    }
    assert set(_sequence(_mapping(rebuild_properties["status"])["enum"])) == {
        item.value for item in Stage7RebuildStatus
    }
    rebuild_defs = _mapping(rebuild["$defs"])
    assert set(_sequence(_mapping(rebuild_defs["reason"])["enum"])) == {
        item.value for item in Stage7RebuildReason
    }
    difference = _mapping(rebuild_defs["difference"])
    assert set(_sequence(_mapping(_mapping(difference["properties"])["kind"])["enum"])) == {
        item.value for item in Stage7DifferenceKind
    }


def test_runtime_preflight_rebuild_config_and_report_payloads_validate() -> None:
    preflight = assess_stage7_preflight(
        verification_context=Stage6VerificationContext.REAL_PREREQUISITE,
        dataset_id=None,
        dataset_version=None,
        three_purpose_review_provided=False,
        physical_root_provided=False,
        adapter_spec_provided=False,
        stage2_pass_package_provided=False,
        approved_real_data_card_provided=False,
        manual_confirmation_provided=False,
        stage3_calibration_package_provided=False,
        stage4_locked_evidence_provided=False,
        stage6_bound_root_provided=False,
    )
    manifest = _manifest()
    rebuild = verify_stage7_rebuild(
        frozen_manifest=manifest,
        frozen_audit=_audit(manifest),
        rebuilt_manifest=DatasetManifestV2.from_dict(manifest.to_dict()),
        rebuilt_audit=_audit(manifest),
    )
    fail_closed_rebuild = verify_stage7_rebuild(
        frozen_manifest=None,
        frozen_audit=None,
        rebuilt_manifest=None,
        rebuilt_audit=None,
    )
    assert rebuild.status is Stage7RebuildStatus.PASS
    assert fail_closed_rebuild.status is Stage7RebuildStatus.FAIL_CLOSED

    _validate("stage7-preflight-v7.schema.json", preflight.to_dict())
    _validate("stage7-rebuild-v7.schema.json", rebuild.to_dict())
    _validate("stage7-rebuild-v7.schema.json", fail_closed_rebuild.to_dict())
    _validate(
        "stage7-toy-config-v7.schema.json",
        load_stage7_toy_config(CONFIG_PATH).to_dict(),
    )
    _validate("stage7-pilot-report-v7.schema.json", _pilot_report())


@pytest.mark.parametrize(
    ("schema_name", "payload"),
    [
        (
            "stage7-preflight-v7.schema.json",
            {
                **assess_stage7_preflight(
                    verification_context=Stage6VerificationContext.SYNTHETIC_FIXTURE,
                    dataset_id="synthetic_stage2",
                    dataset_version="synthetic-fixture-v1",
                    three_purpose_review_provided=True,
                    physical_root_provided=True,
                    adapter_spec_provided=True,
                    stage2_pass_package_provided=True,
                    approved_real_data_card_provided=True,
                    manual_confirmation_provided=True,
                    stage3_calibration_package_provided=True,
                    stage4_locked_evidence_provided=True,
                    stage6_bound_root_provided=True,
                ).to_dict(),
                "reasons": ["INPUT_INVALID"],
            },
        ),
        (
            "stage7-toy-config-v7.schema.json",
            {
                **load_stage7_toy_config(CONFIG_PATH).to_dict(),
                "authorization_ttl_seconds": 6,
            },
        ),
    ],
)
def test_status_or_fixed_config_contract_drift_is_rejected(
    schema_name: str,
    payload: dict[str, object],
) -> None:
    schema = _load(schema_name)
    assert not _matches_schema(payload, schema, root=schema)


def test_rebuild_digest_unknown_label_and_extra_fields_are_rejected() -> None:
    manifest = _manifest()
    payload = verify_stage7_rebuild(
        frozen_manifest=manifest,
        frozen_audit=_audit(manifest),
        rebuilt_manifest=DatasetManifestV2.from_dict(manifest.to_dict()),
        rebuilt_audit=_audit(manifest),
    ).to_dict()
    for name, replacement in (
        ("verification_sha256", "A" * 64),
        ("frozen_unknown_label_count", 1),
    ):
        invalid = deepcopy(payload)
        invalid[name] = replacement
        schema = _load("stage7-rebuild-v7.schema.json")
        assert not _matches_schema(invalid, schema, root=schema)
    extra = deepcopy(payload)
    extra["physical_root"] = "forbidden"
    schema = _load("stage7-rebuild-v7.schema.json")
    assert not _matches_schema(extra, schema, root=schema)


def test_pilot_report_real_authority_bad_hash_and_case_semantics_are_rejected() -> None:
    payload = _pilot_report()

    real_authority = deepcopy(payload)
    real_authority["synthetic_capability_authorizes_real_data"] = True
    schema = _load("stage7-pilot-report-v7.schema.json")
    assert not _matches_schema(real_authority, schema, root=schema)

    bad_hash = deepcopy(payload)
    bad_hash["stage7_config_sha256"] = "E" * 64
    assert not _matches_schema(bad_hash, schema, root=schema)

    inconsistent_case = deepcopy(payload)
    cases = cast(list[dict[str, object]], inconsistent_case["case_index"])
    cases[0]["status"] = "FAIL_CLOSED"
    cases[0]["failure_reasons"] = ["UNKNOWN_DEGRADATION"]
    assert not _matches_schema(inconsistent_case, schema, root=schema)

    unknown = deepcopy(payload)
    unknown["physical_root"] = "forbidden"
    assert not _matches_schema(unknown, schema, root=schema)
