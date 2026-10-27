from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from retinaentail.pilot.stage12_fixture import STAGE12_CONFIG_VERSION, load_stage12_config
from retinaentail.pilot.stage12_gate import (
    STAGE12_PREFLIGHT_SCHEMA_VERSION,
    Stage12PreflightStatus,
    Stage12Reason,
    assess_stage12_preflight,
)
from retinaentail.reproducibility.stage12 import STAGE12_SCHEMA_VERSION

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = PROJECT_ROOT / "data_registry" / "schemas"
CONFIG_PATH = PROJECT_ROOT / "configs" / "stage12_toy_signer_recovery.yaml"
SCHEMAS = (
    "stage12-preflight-v12.schema.json",
    "stage12-toy-config-v12.schema.json",
    "stage12-signer-request-v12.schema.json",
    "stage12-runner-bundle-v12.schema.json",
    "stage12-preregistration-v12.schema.json",
    "stage12-tamper-audit-v12.schema.json",
    "stage12-recovery-audit-v12.schema.json",
    "stage12-package-summary-v12.schema.json",
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


def _blocked_payload() -> dict[str, object]:
    return assess_stage12_preflight(
        dataset_id=None,
        dataset_version=None,
        complete_real_stage11_package_a_capability_present=False,
        complete_real_stage11_package_b_capability_present=False,
        fresh_runner_a_process_capability_present=False,
        fresh_runner_b_process_capability_present=False,
        two_governance_approved_hardware_runners_verified=False,
        hsm_or_controlled_signer_boundary_verified=False,
        trusted_ca_chains_verified=False,
        trusted_timestamp_policy_verified=False,
        current_ocsp_status_verified=False,
        current_crl_status_verified=False,
        complete_wheel_lock_hashes_verified=False,
        hardware_identity_verified=False,
        frozen_baseline_contract_preregistered=False,
        compromise_recovery_drill_preregistered=False,
    ).to_dict()


def test_stage12_schemas_are_closed_draft_202012_and_offline() -> None:
    for name in SCHEMAS:
        schema = _load(name)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["$id"] == f"https://retinaentail.local/schemas/{name}"
        _closed_objects(schema)
        _offline_refs(schema)


def test_runtime_versions_enums_and_required_fields_match_schemas() -> None:
    preflight = _load(SCHEMAS[0])
    config_schema = _load(SCHEMAS[1])
    assert _mapping(_mapping(preflight["properties"])["schema_version"])["const"] == (
        STAGE12_PREFLIGHT_SCHEMA_VERSION
    )
    assert _mapping(_mapping(config_schema["properties"])["schema_version"])["const"] == (
        STAGE12_CONFIG_VERSION
    )
    assert STAGE12_SCHEMA_VERSION == STAGE12_CONFIG_VERSION
    statuses = set(_sequence(_mapping(_mapping(preflight["properties"])["status"])["enum"]))
    assert statuses == {item.value for item in Stage12PreflightStatus}
    reasons = set(
        _sequence(_mapping(_mapping(_mapping(preflight["properties"])["reasons"])["items"])["enum"])
    )
    assert reasons == {item.value for item in Stage12Reason}
    assert set(_blocked_payload()) == set(_sequence(preflight["required"]))
    config = load_stage12_config(CONFIG_PATH)
    assert set(config.raw) == set(_sequence(config_schema["required"]))
