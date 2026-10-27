from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest

from retinaentail.entailment.states import EntailmentState
from retinaentail.pilot.stage6_gate import Stage6VerificationContext
from retinaentail.pilot.stage8_fixture import (
    STAGE8_TOY_CONFIG_VERSION,
    load_stage8_toy_config,
)
from retinaentail.pilot.stage8_gate import (
    STAGE8_GATE_SCHEMA_VERSION,
    STAGE8_NO_PHYSICAL_ROOT_SHA256,
    Stage8PreflightStatus,
    Stage8Reason,
    assess_stage8_preflight,
)
from retinaentail.reporting.stage8 import (
    STAGE8_REPORT_AUTHORITY_BOUNDARY,
    STAGE8_REPORT_METHOD,
    STAGE8_REPORT_SCHEMA_VERSION,
    issue_stage8_pilot_report,
)
from retinaentail.stability.stage8 import (
    STAGE8_SCHEMA_VERSION,
    Stage8AuditStatus,
    Stage8CaseObservation,
    Stage8EventMetric,
    Stage8FailureReason,
    Stage8Preregistration,
    Stage8ReplaySpec,
    Stage8Scope,
    Stage8StabilityMetric,
    audit_stage8_repeatability,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = PROJECT_ROOT / "data_registry" / "schemas"
CONFIG_PATH = PROJECT_ROOT / "configs" / "stage8_toy_repeatability_audit.yaml"
STAGE8_SCHEMAS = (
    "stage8-preflight-v8.schema.json",
    "stage8-toy-config-v8.schema.json",
    "stage8-stability-v8.schema.json",
    "stage8-pilot-report-v8.schema.json",
)

CASES = (
    "stage4-toy-entailed-present",
    "stage4-toy-entailed-absent",
    "stage4-toy-raw-unidentifiable",
    "stage4-toy-infeasible-out-of-model",
    "stage4-toy-domain-out-of-model",
    "stage4-toy-unknown-degradation",
    "stage4-toy-invalid-witness",
)
STAGE7_PACKAGE_SHA256 = "232506ae1514b8ff52b60621b73dcc3218c315277bd360efdacdf030c39eb61a"
STAGE7_TREE_SHA256 = "6aa58bdfbd304b2a1109ea35733e1580a1f8c7b549efb2d35fc75d1a2d503964"
STAGE7_AUDIT_SHA256 = "ada988dd84d27b99a9c78e593197521cd0a7feab090de6ac31991a6b9c99d9ad"
DEGRADATION_SET_SHA256 = "3ecb4261c82250937253dea330ba898317ed7903416d88529fe10ef8fd6a9e76"
CONFIG_HASH = "0ea540363d0a768bc1964c007f6a37a393f1b29b3c9a312ed92bd59a76bbab52"
DATA_HASH = "ab4fa68cb4228df6a1db23caaeacb8d1c35cca6ec7da220e600a8bdcbc067ecd"
MODEL_HASH = "ca2fa473841f6b79d8b327525fb14de0e099ac8d342538a5031d9a54783b81c1"
STAGE8_CONFIG_SHA256 = "726b8a70c4f296e3ad5c6a93c7dfe94195bb311d1a11e9b8d374566c2111feb6"


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


def _refs_are_offline(value: object) -> None:
    if isinstance(value, dict):
        reference = value.get("$ref")
        if reference is not None:
            assert isinstance(reference, str)
            assert not reference.startswith(("http://", "https://"))
            if not reference.startswith("#"):
                name = reference.partition("#")[0]
                assert (SCHEMA_DIR / name).is_file()
        for child in value.values():
            _refs_are_offline(child)
    elif isinstance(value, list):
        for child in value:
            _refs_are_offline(child)


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
    raise AssertionError(f"unsupported JSON Schema type: {expected}")


def _resolve_ref(
    reference: str,
    root: dict[str, object],
) -> tuple[object, dict[str, object]]:
    if reference.startswith("#"):
        target: object = root
        fragment = reference.removeprefix("#").removeprefix("/")
        if not fragment:
            return target, root
        for segment in fragment.split("/"):
            target = _mapping(target)[segment.replace("~1", "/").replace("~0", "~")]
        return target, root
    name, _, fragment = reference.partition("#")
    external = _load(name)
    target = cast(object, external)
    for segment in fragment.removeprefix("/").split("/") if fragment else ():
        target = _mapping(target)[segment.replace("~1", "/").replace("~0", "~")]
    return target, external


def _matches_schema(value: object, raw_schema: object, *, root: dict[str, object]) -> bool:
    if raw_schema is True:
        return True
    if raw_schema is False:
        return False
    schema = _mapping(raw_schema)
    reference = schema.get("$ref")
    if reference is not None:
        assert isinstance(reference, str)
        target, target_root = _resolve_ref(reference, root)
        if not _matches_schema(value, target, root=target_root):
            return False

    for keyword, expected in (("allOf", all), ("anyOf", any)):
        variants = schema.get(keyword)
        if variants is not None and not expected(
            _matches_schema(value, item, root=root) for item in _sequence(variants)
        ):
            return False
    one_of = schema.get("oneOf")
    if (
        one_of is not None
        and sum(_matches_schema(value, item, root=root) for item in _sequence(one_of)) != 1
    ):
        return False
    conditional = schema.get("if")
    if conditional is not None:
        branch = "then" if _matches_schema(value, conditional, root=root) else "else"
        if branch in schema and not _matches_schema(value, schema[branch], root=root):
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
        prefix = _sequence(schema["prefixItems"]) if "prefixItems" in schema else []
        for index, item in enumerate(value[: len(prefix)]):
            if not _matches_schema(item, prefix[index], root=root):
                return False
        item_schema = schema.get("items")
        start = len(prefix) if prefix else 0
        if item_schema is not None and any(
            not _matches_schema(item, item_schema, root=root) for item in value[start:]
        ):
            return False
        contains = schema.get("contains")
        if contains is not None:
            match_count = sum(_matches_schema(item, contains, root=root) for item in value)
            minimum_contains = schema.get("minContains", 1)
            maximum_contains = schema.get("maxContains")
            if isinstance(minimum_contains, int) and match_count < minimum_contains:
                return False
            if isinstance(maximum_contains, int) and match_count > maximum_contains:
                return False
    if isinstance(value, dict):
        required = schema.get("required")
        if required is not None and not set(cast(list[str], _sequence(required))).issubset(value):
            return False
        properties = {} if schema.get("properties") is None else _mapping(schema["properties"])
        for key, child in value.items():
            child_schema = properties.get(key)
            if child_schema is not None and not _matches_schema(child, child_schema, root=root):
                return False
        extras = set(value) - set(properties)
        additional = schema.get("additionalProperties")
        if additional is False and extras:
            return False
        if additional not in (None, True, False) and any(
            not _matches_schema(value[key], additional, root=root) for key in extras
        ):
            return False
    return True


def _validate(name: str, payload: object) -> None:
    schema = _load(name)
    assert _matches_schema(payload, schema, root=schema)


def _preflight(*, ready: bool) -> dict[str, object]:
    return assess_stage8_preflight(
        dataset_id="synthetic_stage2" if ready else None,
        dataset_version="synthetic-fixture-v1" if ready else None,
        complete_stage7_package_capability_provided=ready,
        fresh_root_run_capability_provided=ready,
        fixed_seed_preregistered=ready,
        degradation_replay_set_preregistered=ready,
        inclusion_rules_preregistered=ready,
        stability_metrics_preregistered=ready,
        locked_contract_provided=ready,
        verification_context=Stage6VerificationContext.SYNTHETIC_FIXTURE,
    ).to_dict()


def _preregistration() -> Stage8Preregistration:
    specs = tuple(
        Stage8ReplaySpec(
            run_id=f"stage8-repeat-00{index}",
            seed=20260824,
            degradation_replay_id="frozen_stage7_test_trace_set_v1",
            degradation_replay_set_sha256=DEGRADATION_SET_SHA256,
        )
        for index in (1, 2)
    )
    return Stage8Preregistration(
        dataset_id="synthetic_stage2",
        dataset_version="synthetic-fixture-v1",
        scope=Stage8Scope.SYNTHETIC_ONLY,
        inclusion_case_ids=CASES,
        replay_specs=specs,
        alpha=0.25,
        positive_threshold=0.000012299400515471258,
        negative_threshold=0.000012299400515471254,
        positive_sample_count=4,
        negative_sample_count=4,
        config_hash=CONFIG_HASH,
        data_hash=DATA_HASH,
        model_hash=MODEL_HASH,
        stage7_package_sha256=STAGE7_PACKAGE_SHA256,
        provenance_sha256=STAGE7_AUDIT_SHA256,
    )


def _observation(
    case_id: str,
    *,
    run_id: str,
    seed: int,
    state: EntailmentState,
    positive_score: float,
    negative_score: float,
    fail_closed: bool = False,
    out_of_distribution: bool = False,
    degradation_known: bool = True,
    positive_replay: bool = True,
    negative_replay: bool = True,
) -> Stage8CaseObservation:
    index = CASES.index(case_id)
    digit = format(index + 1, "x")
    return Stage8CaseObservation(
        case_id=case_id,
        run_id=run_id,
        dataset_id="synthetic_stage2",
        dataset_version="synthetic-fixture-v1",
        split="test",
        scope=Stage8Scope.SYNTHETIC_ONLY,
        seed=seed,
        degradation_replay_id="frozen_stage7_test_trace_set_v1",
        degradation_replay_set_sha256=DEGRADATION_SET_SHA256,
        case_degradation_trace_sha256=digit * 64,
        state=state,
        fail_closed=fail_closed,
        out_of_distribution=out_of_distribution,
        alpha=0.25,
        positive_score=positive_score,
        negative_score=negative_score,
        positive_threshold=0.000012299400515471258,
        negative_threshold=0.000012299400515471254,
        positive_sample_count=4,
        negative_sample_count=4,
        config_hash=CONFIG_HASH,
        data_hash=DATA_HASH,
        model_hash=MODEL_HASH,
        stage7_package_sha256=STAGE7_PACKAGE_SHA256,
        provenance_sha256=STAGE7_AUDIT_SHA256,
        degradation_known=degradation_known,
        solver_succeeded=True,
        positive_witness_sha256="a" * 64,
        negative_witness_sha256="b" * 64,
        positive_witness_replay_valid=positive_replay,
        negative_witness_replay_valid=negative_replay,
    )


def _observations() -> tuple[list[Stage8CaseObservation], list[Stage8CaseObservation]]:
    specifications = (
        (EntailmentState.ENTAILED_PRESENT, 0.0, 1.0, False, False, True, True, True),
        (EntailmentState.ENTAILED_ABSENT, 1.0, 0.0, False, False, True, True, True),
        (EntailmentState.RAW_UNIDENTIFIABLE, 0.0, 0.0, False, False, True, True, True),
        (EntailmentState.OUT_OF_MODEL, 1.0, 1.0, False, False, True, True, True),
        (EntailmentState.OUT_OF_MODEL, 1.0, 1.0, True, True, True, True, True),
        (EntailmentState.OUT_OF_MODEL, 1.0, 1.0, True, False, False, True, True),
        (EntailmentState.OUT_OF_MODEL, 1.0, 1.0, True, False, True, True, False),
    )
    baseline: list[Stage8CaseObservation] = []
    repeats: list[Stage8CaseObservation] = []
    for case_id, values in zip(CASES, specifications, strict=True):
        state, positive, negative, fail_closed, ood, known, pos_replay, neg_replay = values
        baseline.append(
            _observation(
                case_id,
                run_id="stage7_locked_baseline",
                seed=0,
                state=state,
                positive_score=positive,
                negative_score=negative,
                fail_closed=fail_closed,
                out_of_distribution=ood,
                degradation_known=known,
                positive_replay=pos_replay,
                negative_replay=neg_replay,
            )
        )
        for run_id in ("stage8-repeat-001", "stage8-repeat-002"):
            repeats.append(
                _observation(
                    case_id,
                    run_id=run_id,
                    seed=20260824,
                    state=state,
                    positive_score=positive,
                    negative_score=negative,
                    fail_closed=fail_closed,
                    out_of_distribution=ood,
                    degradation_known=known,
                    positive_replay=pos_replay,
                    negative_replay=neg_replay,
                )
            )
    return baseline, repeats


def _stability_payload() -> dict[str, object]:
    baseline, repeats = _observations()
    report = audit_stage8_repeatability(
        preregistration=_preregistration(),
        baseline_records=baseline,
        repeat_records=repeats,
    )
    assert report.status is Stage8AuditStatus.SUCCESS
    assert report.failure_reason_counts
    assert set(report.failure_reason_counts).issubset(
        {
            Stage8FailureReason.OUT_OF_DISTRIBUTION,
            Stage8FailureReason.UNKNOWN_DEGRADATION,
            Stage8FailureReason.INVALID_WITNESS_REPLAY,
        }
    )
    return report.to_dict()


def _fail_closed_stability_payload() -> dict[str, object]:
    baseline, repeats = _observations()
    report = audit_stage8_repeatability(
        preregistration=_preregistration(),
        baseline_records=baseline,
        repeat_records=repeats[1:],
    )
    assert report.status is Stage8AuditStatus.FAIL_CLOSED
    return report.to_dict()


def _pilot_payload() -> dict[str, object]:
    baseline, repeats = _observations()
    audit = audit_stage8_repeatability(
        preregistration=_preregistration(),
        baseline_records=baseline,
        repeat_records=repeats,
    )
    frozen_bindings = {
        "schema_version": "8.0.0",
        "split": "test",
        "fixed_seed": 20260824,
        "degradation_replay_set_sha256": DEGRADATION_SET_SHA256,
        "inclusion_rules_sha256": "1" * 64,
        "stability_metrics_sha256": "2" * 64,
        "locked_contract_sha256": "3" * 64,
        "test_split_sha256": "4" * 64,
        "config_hash": CONFIG_HASH,
        "data_hash": DATA_HASH,
        "model_hash": MODEL_HASH,
        "provenance_sha256": STAGE7_AUDIT_SHA256,
        "preregistration_sha256": STAGE8_CONFIG_SHA256,
    }
    per_run = [
        {
            "run_id": spec.run_id,
            "seed": spec.seed,
            "scope": "SYNTHETIC_FIXTURE",
            "dataset_id": "synthetic_stage2",
            "dataset_version": "synthetic-fixture-v1",
            "ttl_seconds": 5,
            "process_bound": True,
            "non_serializable": True,
            "one_shot": True,
            "issued": True,
            "consumed": True,
            "complete_stage7_package_replayed": True,
            "second_consumption_rejected": True,
            "authorizes_embedded_package_replay": True,
            "authorizes_real_data": False,
            "authorizes_physical_dataset_root": False,
            "detached_receipt_authoritative": False,
            "authorizes_file_open": False,
        }
        for spec in audit.preregistration.replay_specs
    ]
    receipts = [
        {
            "schema_version": "8.0.0",
            "run_id": spec.run_id,
            "seed": spec.seed,
            "degradation_replay_id": spec.degradation_replay_id,
            "degradation_replay_set_sha256": spec.degradation_replay_set_sha256,
            "stage7_package_sha256": STAGE7_PACKAGE_SHA256,
            "stage7_tree_sha256": STAGE7_TREE_SHA256,
            "case_observations_sha256": "5" * 64,
            "dual_witness_replay_set_sha256": "6" * 64,
            "complete_stage7_package_replay_verified": True,
            "locked_baseline_reexecuted": True,
            "thresholds_reestimated": False,
            "test_tuning_performed": False,
            "sample_replacement_performed": False,
        }
        for spec in audit.preregistration.replay_specs
    ]
    report = issue_stage8_pilot_report(
        run_id="stage8-toy-repeatability-stability-001",
        source_package={
            "schema_version": "8.0.0",
            "scope": "SYNTHETIC_FIXTURE",
            "dataset_id": "synthetic_stage2",
            "dataset_version": "synthetic-fixture-v1",
            "package_sha256": STAGE7_PACKAGE_SHA256,
            "package_file_count": 130,
            "stage2_package_sha256": "7" * 64,
            "stage3_package_sha256": "8" * 64,
            "physical_root_identity_sha256": STAGE8_NO_PHYSICAL_ROOT_SHA256,
            "frozen_bindings": frozen_bindings,
            "fixture_only": True,
        },
        capability_lifecycle={
            "scope": "SYNTHETIC_FIXTURE",
            "issued_count": 2,
            "consumed_count": 2,
            "second_use_rejected_count": 2,
            "complete_stage7_replay_count": 2,
            "process_bound": True,
            "non_serializable": True,
            "one_shot": True,
            "all_capabilities_fresh": True,
            "authorizes_embedded_package_replay": True,
            "authorizes_real_data": False,
            "authorizes_physical_dataset_root": False,
            "detached_receipt_authoritative": False,
            "authorizes_file_open": False,
            "per_run": per_run,
        },
        stability_audit=audit,
        replay_receipts=receipts,
        stage8_config_sha256=STAGE8_CONFIG_SHA256,
        stage8_observations_sha256="9" * 64,
        stage8_model_provenance_sha256="a" * 64,
    )
    return report.to_dict()


def test_stage8_schemas_are_closed_draft_202012_and_use_only_offline_refs() -> None:
    for name in STAGE8_SCHEMAS:
        schema = _load(name)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["$id"] == f"https://retinaentail.local/schemas/{name}"
        assert schema["additionalProperties"] is False
        _closed_objects(schema)
        _refs_are_offline(schema)


def test_stage8_runtime_versions_and_enums_match_schemas() -> None:
    preflight = _load("stage8-preflight-v8.schema.json")
    stability = _load("stage8-stability-v8.schema.json")
    config = _load("stage8-toy-config-v8.schema.json")
    report = _load("stage8-pilot-report-v8.schema.json")
    preflight_properties = _mapping(preflight["properties"])
    stability_properties = _mapping(stability["properties"])
    config_properties = _mapping(config["properties"])
    report_properties = _mapping(report["properties"])
    stability_defs = _mapping(stability["$defs"])

    assert _mapping(preflight_properties["schema_version"])["const"] == (STAGE8_GATE_SCHEMA_VERSION)
    assert _mapping(stability_properties["schema_version"])["const"] == STAGE8_SCHEMA_VERSION
    assert _mapping(config_properties["schema_version"])["const"] == (STAGE8_TOY_CONFIG_VERSION)
    assert _mapping(report_properties["schema_version"])["const"] == (STAGE8_REPORT_SCHEMA_VERSION)
    assert _mapping(report_properties["method"])["const"] == STAGE8_REPORT_METHOD
    assert _mapping(report_properties["authority_boundary"])["const"] == (
        STAGE8_REPORT_AUTHORITY_BOUNDARY
    )
    assert set(_sequence(_mapping(preflight_properties["status"])["enum"])) == {
        item.value for item in Stage8PreflightStatus
    }
    assert set(_sequence(_mapping(_mapping(preflight["$defs"])["reason"])["enum"])) == {
        item.value for item in Stage8Reason
    }
    assert set(_sequence(_mapping(stability_properties["status"])["enum"])) == {
        item.value for item in Stage8AuditStatus
    }
    assert set(_sequence(_mapping(stability_defs["stabilityMetric"])["enum"])) == {
        item.value for item in Stage8StabilityMetric
    }
    assert set(_sequence(_mapping(stability_defs["eventMetric"])["enum"])) == {
        item.value for item in Stage8EventMetric
    }
    assert set(_sequence(_mapping(stability_defs["failureReason"])["enum"])) == {
        item.value for item in Stage8FailureReason
    }
    runtime_reasons = {
        Stage8FailureReason.UNKNOWN_DEGRADATION.value,
        Stage8FailureReason.SOLVER_FAILED.value,
        Stage8FailureReason.INVALID_WITNESS_REPLAY.value,
        Stage8FailureReason.OUT_OF_DISTRIBUTION.value,
    }
    assert (
        set(_sequence(_mapping(stability_defs["contractFailureReason"])["enum"]))
        == {item.value for item in Stage8FailureReason} - runtime_reasons
    )


def test_runtime_preflight_config_stability_and_pilot_payloads_validate() -> None:
    _validate("stage8-preflight-v8.schema.json", _preflight(ready=False))
    _validate("stage8-preflight-v8.schema.json", _preflight(ready=True))
    _validate(
        "stage8-toy-config-v8.schema.json",
        load_stage8_toy_config(CONFIG_PATH).to_dict(),
    )
    _validate("stage8-stability-v8.schema.json", _stability_payload())
    _validate("stage8-stability-v8.schema.json", _fail_closed_stability_payload())
    _validate("stage8-pilot-report-v8.schema.json", _pilot_payload())


def test_success_stability_allows_controlled_runtime_fail_closed_strata() -> None:
    payload = _stability_payload()
    assert payload["status"] == "SUCCESS"
    failures = _mapping(payload["failure_reason_counts"])
    assert failures
    assert set(failures).issubset(
        {"OUT_OF_DISTRIBUTION", "UNKNOWN_DEGRADATION", "INVALID_WITNESS_REPLAY"}
    )
    _validate("stage8-stability-v8.schema.json", payload)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("source_package", "scope"), "APPROVED_REAL_DATA"),
        (("source_package", "package_sha256"), "A" * 64),
        (("source_package", "frozen_bindings", "model_hash"), "0" * 64),
        (("capability_lifecycle", "all_capabilities_fresh"), False),
        (("capability_lifecycle", "detached_receipt_authoritative"), True),
        (("capability_lifecycle", "per_run", 0, "seed"), 20260825),
        (("replay_receipts", 0, "thresholds_reestimated"), True),
        (("preregistration", "inclusion_case_ids", 0), "replacement-case"),
        (("stability_audit", "preregistration", "model_hash"), "0" * 64),
        (("synthetic_authorizes_real_data",), True),
        (("process_capability_serialized",), True),
        (("authority_boundary",), "DETACHED_REPORT"),
    ],
)
def test_pilot_report_rejects_hash_case_run_authority_and_safety_drift(
    path: tuple[str | int, ...], replacement: object
) -> None:
    payload = deepcopy(_pilot_payload())
    target: object = payload
    for segment in path[:-1]:
        target = target[segment]  # type: ignore[index]
    target[path[-1]] = replacement  # type: ignore[index]
    schema = _load("stage8-pilot-report-v8.schema.json")
    assert not _matches_schema(payload, schema, root=schema)


def test_transition_hash_counts_and_unknown_fields_are_rejected() -> None:
    payload = _stability_payload()
    schema = _load("stage8-stability-v8.schema.json")

    bad_transition = deepcopy(payload)
    transitions = cast(list[dict[str, object]], bad_transition["case_transitions"])
    transitions[0]["transition"] = "ENTAILED_PRESENT->ENTAILED_ABSENT"
    assert not _matches_schema(bad_transition, schema, root=schema)

    bad_hash = deepcopy(payload)
    preregistration = _mapping(bad_hash["preregistration"])
    preregistration["data_hash"] = "A" * 64
    assert not _matches_schema(bad_hash, schema, root=schema)

    bad_reason = deepcopy(payload)
    counts = _mapping(bad_reason["failure_reason_counts"])
    counts["UNKNOWN_REASON"] = 1
    assert not _matches_schema(bad_reason, schema, root=schema)

    bad_witness_hash = deepcopy(payload)
    witness_transitions = cast(list[dict[str, object]], bad_witness_hash["case_transitions"])
    witness_transitions[0]["baseline_positive_witness_sha256"] = "A" * 64
    assert not _matches_schema(bad_witness_hash, schema, root=schema)

    missing_witness_hash = deepcopy(payload)
    missing_transitions = cast(list[dict[str, object]], missing_witness_hash["case_transitions"])
    del missing_transitions[0]["repeat_negative_witness_sha256"]
    assert not _matches_schema(missing_witness_hash, schema, root=schema)

    extra = deepcopy(payload)
    extra["physical_root"] = "forbidden"
    assert not _matches_schema(extra, schema, root=schema)


def test_preflight_and_fixed_config_contract_drift_are_rejected() -> None:
    preflight = _preflight(ready=True)
    preflight["dataset_id"] = None
    preflight_schema = _load("stage8-preflight-v8.schema.json")
    assert not _matches_schema(preflight, preflight_schema, root=preflight_schema)

    config = load_stage8_toy_config(CONFIG_PATH).to_dict()
    config_schema = _load("stage8-toy-config-v8.schema.json")
    bad_seed = deepcopy(config)
    runs = cast(list[dict[str, object]], bad_seed["replay_runs"])
    runs[0]["seed"] = 20260825
    assert not _matches_schema(bad_seed, config_schema, root=config_schema)
    bad_source = deepcopy(config)
    source = _mapping(bad_source["source_contract"])
    source["stage7_package_sha256"] = "0" * 64
    assert not _matches_schema(bad_source, config_schema, root=config_schema)
