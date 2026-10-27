from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from retinaentail.pilot.stage11_fixture import (
    STAGE11_TOY_CONFIG_VERSION,
    load_stage11_toy_config,
)
from retinaentail.pilot.stage11_gate import (
    STAGE11_PREFLIGHT_SCHEMA_VERSION,
    Stage11PreflightStatus,
    Stage11Reason,
    assess_stage11_preflight,
)
from retinaentail.reproducibility.stage11 import STAGE11_SCHEMA_VERSION

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = PROJECT_ROOT / "data_registry" / "schemas"
CONFIG_PATH = PROJECT_ROOT / "configs" / "stage11_toy_asymmetric_exchange.yaml"
SCHEMAS = (
    "stage11-preflight-v11.schema.json",
    "stage11-trust-anchor-v11.schema.json",
    "stage11-certificate-v11.schema.json",
    "stage11-rotation-proof-v11.schema.json",
    "stage11-revocation-list-v11.schema.json",
    "stage11-signed-envelope-v11.schema.json",
    "stage11-dependency-lock-v11.schema.json",
    "stage11-runtime-lock-v11.schema.json",
    "stage11-tamper-audit-v11.schema.json",
    "stage11-toy-config-v11.schema.json",
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
    return assess_stage11_preflight(
        dataset_id=None,
        dataset_version=None,
        complete_real_stage10_package_a_capability_present=False,
        complete_real_stage10_package_b_capability_present=False,
        fresh_host_a_process_capability_present=False,
        fresh_host_b_process_capability_present=False,
        two_approved_physical_hosts_confirmed=False,
        production_asymmetric_trust_anchor_verified=False,
        unique_runner_leaf_keys_verified=False,
        certificate_chains_verified=False,
        revocation_status_current=False,
        trusted_time_policy_verified=False,
        complete_wheel_lock_hashes_verified=False,
        host_environment_policy_preregistered=False,
        case_seed_replay_preregistered=False,
        locked_hashes_thresholds_preregistered=False,
    ).to_dict()


def test_stage11_schemas_are_closed_draft_202012_and_offline() -> None:
    for name in SCHEMAS:
        schema = _load(name)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["$id"] == f"https://retinaentail.local/schemas/{name}"
        _closed_objects(schema)
        _offline_refs(schema)


def test_runtime_versions_enums_and_required_fields_match_schemas() -> None:
    preflight = _load(SCHEMAS[0])
    config_schema = _load(SCHEMAS[-1])
    assert _mapping(_mapping(preflight["properties"])["schema_version"])["const"] == (
        STAGE11_PREFLIGHT_SCHEMA_VERSION
    )
    assert _mapping(_mapping(config_schema["properties"])["schema_version"])["const"] == (
        STAGE11_TOY_CONFIG_VERSION
    )
    assert STAGE11_SCHEMA_VERSION == STAGE11_TOY_CONFIG_VERSION
    statuses = set(_sequence(_mapping(_mapping(preflight["properties"])["status"])["enum"]))
    assert statuses == {item.value for item in Stage11PreflightStatus}
    reasons = set(
        _sequence(_mapping(_mapping(_mapping(preflight["properties"])["reasons"])["items"])["enum"])
    )
    assert reasons == {item.value for item in Stage11Reason}
    assert set(_blocked_payload()) == set(_sequence(preflight["required"]))
    config = load_stage11_toy_config(CONFIG_PATH)
    assert set(config.raw) == set(_sequence(config_schema["required"]))
