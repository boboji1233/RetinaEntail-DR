"""Offline Stage 7 authorization-consumption and rebuild transaction fixture.

This module accepts only a complete synthetic Stage 6 package.  It never accepts
or probes a dataset root.  The process-bound authorization below grants exactly
one in-process replay of embedded synthetic evidence and can never authorize real
data or a physical dataset root.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import stat
import time
from collections.abc import Mapping, Sequence
from copy import copy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Final, cast

import yaml

from retinaentail.data.audit_reporting import AuditWriteBoundary
from retinaentail.data.stage2_models import DatasetManifestV2
from retinaentail.entailment.states import EntailmentState
from retinaentail.pilot.identity import (
    Stage5IdentityAuditStatus,
    Stage5IdentityRecord,
    audit_five_identity_split_isolation,
)
from retinaentail.pilot.stage6_fixture import verify_stage6_toy_package
from retinaentail.reporting.stage3 import write_stage3_bundle
from retinaentail.reporting.stage7 import (
    STAGE7_AUTHORITY_BOUNDARY,
    STAGE7_DIGEST_METHOD,
    STAGE7_METHOD,
    issue_stage7_pilot_report,
    render_stage7_pilot_html,
    render_stage7_pilot_json,
)
from retinaentail.witnesses.replay import sha256_bytes

STAGE7_TOY_CONFIG_VERSION: Final = "7.0"
STAGE7_PACKAGE_VERSION: Final = "7.0.0"
_MAX_FILE_BYTES: Final = 8 * 1024 * 1024
_MAX_PACKAGE_BYTES: Final = 160 * 1024 * 1024
_MAX_PACKAGE_FILES: Final = 512
_MAX_DEPTH: Final = 20
_ROOT_FILES: Final = frozenset(
    {
        "audit.html",
        "audit.json",
        "config_snapshot.json",
        "model_provenance.json",
        "source_package.json",
        "summary.json",
        "transaction.json",
    }
)
_AUTHORIZATION_TOKEN: Final = object()
_AUTHORIZATION_REGISTRY: dict[int, object] = {}


class Stage7ToyConfigurationError(ValueError):
    """Raised when the fixed Stage 7 synthetic declaration is unsafe."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage7SourceContract:
    stage6_scope: str
    stage6_run_id: str
    stage6_authority_boundary: str
    stage6_package_file_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "stage6_scope": self.stage6_scope,
            "stage6_run_id": self.stage6_run_id,
            "stage6_authority_boundary": self.stage6_authority_boundary,
            "stage6_package_file_count": self.stage6_package_file_count,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage7ToyConfig:
    run_id: str
    source_contract: Stage7SourceContract
    authorization_ttl_seconds: int
    confidence_level: float
    rebuild_mode: str
    identity_policy: str
    threshold_policy: str
    schema_version: str = STAGE7_TOY_CONFIG_VERSION
    method: str = STAGE7_METHOD
    authorization_scope: str = "SYNTHETIC_ONLY"
    one_shot_consumption: bool = True
    test_tuning_prohibited: bool = True
    real_data_access: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "method": self.method,
            "source_contract": self.source_contract.to_dict(),
            "authorization_scope": self.authorization_scope,
            "authorization_ttl_seconds": self.authorization_ttl_seconds,
            "one_shot_consumption": self.one_shot_consumption,
            "confidence_level": self.confidence_level,
            "rebuild_mode": self.rebuild_mode,
            "identity_policy": self.identity_policy,
            "threshold_policy": self.threshold_policy,
            "test_tuning_prohibited": self.test_tuning_prohibited,
            "real_data_access": self.real_data_access,
        }


@dataclass(frozen=True, slots=True)
class Stage7ToyRunResult:
    """Non-authoritative receipt from one full Stage 7 package replay."""

    output_dir: Path
    package_sha256: str
    package_file_count: int
    stage6_package_sha256: str
    stage6_envelope_sha256: str
    rebuilt_record_count: int
    identity_record_count: int
    case_count: int
    states: tuple[EntailmentState, ...]
    verified: bool


@dataclass(frozen=True, slots=True)
class _Prepared:
    config: Stage7ToyConfig
    root_artifacts: Mapping[str, bytes]
    stage6_descriptor: Mapping[str, object]
    rebuild: Mapping[str, object]
    identity: Mapping[str, object]
    states: tuple[EntailmentState, ...]


class _SyntheticReplayAuthorization:
    """One-shot, process-bound authorization for embedded fixture replay only."""

    __slots__ = ("_binding", "_consumed", "_deadline", "_pid")

    def __init__(
        self,
        *,
        binding: str,
        ttl_seconds: int,
        _token: object,
    ) -> None:
        if _token is not _AUTHORIZATION_TOKEN:
            raise TypeError("synthetic authorization must be issued internally")
        self._binding = binding
        self._pid = os.getpid()
        self._deadline = time.monotonic() + ttl_seconds
        self._consumed = False
        _AUTHORIZATION_REGISTRY[id(self)] = self

    def consume(self, *, expected_binding: str) -> None:
        if _AUTHORIZATION_REGISTRY.get(id(self)) is not self:
            raise RuntimeError("synthetic authorization issuance is invalid")
        if os.getpid() != self._pid:
            raise RuntimeError("synthetic authorization is bound to its issuing process")
        if time.monotonic() > self._deadline:
            raise RuntimeError("synthetic authorization expired")
        if self._consumed:
            raise RuntimeError("synthetic authorization was already consumed")
        if expected_binding != self._binding:
            raise RuntimeError("synthetic authorization binding changed")
        self._consumed = True
        _AUTHORIZATION_REGISTRY.pop(id(self), None)

    def __copy__(self) -> object:
        raise TypeError("synthetic authorization cannot be copied")

    def __deepcopy__(self, memo: object) -> object:
        del memo
        raise TypeError("synthetic authorization cannot be copied")

    def __reduce__(self) -> str | tuple[object, ...]:
        raise TypeError("synthetic authorization cannot be serialized")


def parse_stage7_toy_config(value: object) -> Stage7ToyConfig:
    data = _strict_mapping(
        value,
        "Stage 7 config",
        {
            "schema_version",
            "run_id",
            "method",
            "source_contract",
            "authorization_scope",
            "authorization_ttl_seconds",
            "one_shot_consumption",
            "confidence_level",
            "rebuild_mode",
            "identity_policy",
            "threshold_policy",
            "test_tuning_prohibited",
            "real_data_access",
        },
    )
    if data["schema_version"] != STAGE7_TOY_CONFIG_VERSION or data["method"] != STAGE7_METHOD:
        raise Stage7ToyConfigurationError("unsupported Stage 7 config version or method")
    raw_source = _strict_mapping(
        data["source_contract"],
        "source_contract",
        {
            "stage6_scope",
            "stage6_run_id",
            "stage6_authority_boundary",
            "stage6_package_file_count",
        },
    )
    source = Stage7SourceContract(
        stage6_scope=_text(raw_source["stage6_scope"], "stage6_scope"),
        stage6_run_id=_text(raw_source["stage6_run_id"], "stage6_run_id"),
        stage6_authority_boundary=_text(
            raw_source["stage6_authority_boundary"], "stage6_authority_boundary"
        ),
        stage6_package_file_count=_positive_int(
            raw_source["stage6_package_file_count"], "stage6_package_file_count"
        ),
    )
    config = Stage7ToyConfig(
        run_id=_text(data["run_id"], "run_id"),
        source_contract=source,
        authorization_scope=_text(data["authorization_scope"], "authorization_scope"),
        authorization_ttl_seconds=_positive_int(
            data["authorization_ttl_seconds"], "authorization_ttl_seconds"
        ),
        one_shot_consumption=_boolean(data["one_shot_consumption"], "one_shot_consumption"),
        confidence_level=_number(data["confidence_level"], "confidence_level"),
        rebuild_mode=_text(data["rebuild_mode"], "rebuild_mode"),
        identity_policy=_text(data["identity_policy"], "identity_policy"),
        threshold_policy=_text(data["threshold_policy"], "threshold_policy"),
        test_tuning_prohibited=_boolean(data["test_tuning_prohibited"], "test_tuning_prohibited"),
        real_data_access=_boolean(data["real_data_access"], "real_data_access"),
    )
    expected_source = {
        "stage6_scope": "SYNTHETIC_ONLY",
        "stage6_run_id": "stage6-toy-verifier-pilot-001",
        "stage6_authority_boundary": "COMPLETE_VERIFIED_STAGE6_PACKAGE",
        "stage6_package_file_count": 122,
    }
    if source.to_dict() != expected_source:
        raise Stage7ToyConfigurationError("Stage 7 source contract is not the fixed fixture")
    if (
        config.run_id != "stage7-toy-authorization-consumption-001"
        or config.authorization_scope != "SYNTHETIC_ONLY"
        or config.authorization_ttl_seconds != 5
        or config.one_shot_consumption is not True
        or config.confidence_level != 0.95
        or config.rebuild_mode != "EMBEDDED_STAGE2_TYPED_ROUNDTRIP"
        or config.identity_policy != "FIVE_IDENTITY_STRICT_REPLAY"
        or config.threshold_policy != "LOCKED_STAGE3_NO_REESTIMATION"
        or config.test_tuning_prohibited is not True
        or config.real_data_access is not False
    ):
        raise Stage7ToyConfigurationError("Stage 7 fixture safety declarations are invalid")
    return config


def load_stage7_toy_config(path: Path) -> Stage7ToyConfig:
    if not isinstance(path, Path) or path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    payload = path.read_bytes()
    if len(payload) > 64 * 1024:
        raise Stage7ToyConfigurationError("Stage 7 config exceeds its size limit")
    try:
        value = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise Stage7ToyConfigurationError("Stage 7 config is not strict UTF-8 YAML") from error
    return parse_stage7_toy_config(value)


def _prepare(
    config: Stage7ToyConfig,
    *,
    stage6_root: Path,
    stage6_files: Mapping[str, bytes],
) -> _Prepared:
    verified_stage6 = verify_stage6_toy_package(stage6_root)
    after_replay = _read_tree(stage6_root, label="Stage 6")
    if dict(stage6_files) != after_replay:
        raise ValueError("Stage 6 package bytes changed during complete replay")
    stage6_report = _json_mapping(stage6_files["audit.json"], "Stage 6 audit")
    stage6_summary = _json_mapping(stage6_files["summary.json"], "Stage 6 summary")
    _validate_stage6_safety(stage6_report, stage6_summary)

    complete_descriptor = _file_descriptor(stage6_files)
    stage6_descriptor = {
        "scope": stage6_summary.get("scope"),
        "run_id": stage6_summary.get("run_id"),
        "authority_boundary": stage6_summary.get("authority_boundary"),
        "package_sha256": verified_stage6.package_sha256,
        "package_file_count": len(stage6_files),
        "envelope_sha256": complete_descriptor["package_sha256"],
        "complete_byte_replay_verified": verified_stage6.verified,
    }
    _validate_config_source(config, stage6_descriptor)

    binding = sha256_bytes(
        _canonical_compact_bytes(
            {
                "schema_version": STAGE7_PACKAGE_VERSION,
                "scope": "SYNTHETIC_ONLY",
                "stage6_package_sha256": verified_stage6.package_sha256,
                "stage6_envelope_sha256": complete_descriptor["package_sha256"],
                "run_id": config.run_id,
            }
        )
    )
    capability = _SyntheticReplayAuthorization(
        binding=binding,
        ttl_seconds=config.authorization_ttl_seconds,
        _token=_AUTHORIZATION_TOKEN,
    )
    copy_rejected = False
    serialization_rejected = False
    try:
        copy(capability)
    except TypeError:
        copy_rejected = True
    try:
        pickle.dumps(capability)
    except (TypeError, pickle.PicklingError):
        serialization_rejected = True
    if not copy_rejected or not serialization_rejected:
        raise AssertionError("synthetic authorization became detachable")
    capability.consume(expected_binding=binding)
    second_consumption_rejected = False
    try:
        capability.consume(expected_binding=binding)
    except RuntimeError:
        second_consumption_rejected = True
    if not second_consumption_rejected:
        raise AssertionError("one-shot synthetic authorization was reusable")
    authorization = {
        "scope": "SYNTHETIC_ONLY",
        "ttl_seconds": config.authorization_ttl_seconds,
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
    }

    rebuild = _rebuild_stage2_manifest(stage6_files)
    identity = _rerun_five_identity_audit(stage6_files)
    pilot_audit = _mapping(stage6_report.get("pilot_audit"), "Stage 6 pilot_audit")
    locked = _mapping(pilot_audit.get("locked_contract"), "locked_contract")
    evaluation = _mapping(pilot_audit.get("evaluation"), "evaluation")
    raw_cases = pilot_audit.get("case_index")
    if not isinstance(raw_cases, list):
        raise ValueError("Stage 6 pilot case_index is missing")
    cases = tuple(_mapping(item, "case_index item") for item in raw_cases)
    states = tuple(EntailmentState(cast(str, item["state"])) for item in cases)

    config_bytes = _canonical_json_bytes(config.to_dict())
    config_sha256 = sha256_bytes(config_bytes)
    source_payload = {
        "schema_version": STAGE7_PACKAGE_VERSION,
        "digest_method": STAGE7_DIGEST_METHOD,
        "verified_stage6": dict(stage6_descriptor),
        "complete_stage6_envelope": complete_descriptor,
    }
    source_bytes = _canonical_json_bytes(source_payload)
    source_sha256 = sha256_bytes(source_bytes)
    model_provenance = {
        "schema_version": STAGE7_PACKAGE_VERSION,
        "component": "stage7_synthetic_transaction_orchestrator",
        "method": STAGE7_METHOD,
        "model_kind": "none_orchestration_only",
        "trained": False,
        "pretrained_weights": False,
        "threshold_source": "embedded_complete_stage6_locked_contract",
        "test_threshold_reestimation": False,
        "clinical_claim": False,
    }
    model_bytes = _canonical_json_bytes(model_provenance)
    model_sha256 = sha256_bytes(model_bytes)
    report = issue_stage7_pilot_report(
        run_id=config.run_id,
        stage6_source=stage6_descriptor,
        authorization_consumption=authorization,
        rebuild_equivalence=rebuild,
        identity_isolation=identity,
        locked_contract=locked,
        evaluation=evaluation,
        case_index=cases,
        stage7_config_sha256=config_sha256,
        stage7_source_payload_sha256=source_sha256,
        stage7_model_provenance_sha256=model_sha256,
    )
    transaction = {
        "schema_version": STAGE7_PACKAGE_VERSION,
        "scope": "SYNTHETIC_ONLY",
        "run_id": config.run_id,
        "authorization_consumption": authorization,
        "rebuild_equivalence": rebuild,
        "identity_isolation": identity,
        "locked_contract": locked,
        "case_index_sha256": sha256_bytes(_canonical_compact_bytes(list(cases))),
        "thresholds_reestimated": False,
        "test_tuning_prohibited": True,
        "synthetic_capability_authorizes_real_data": False,
        "dataset_root_probed": False,
        "real_data_read_count": 0,
    }
    summary = {
        "schema_version": STAGE7_PACKAGE_VERSION,
        "scope": "SYNTHETIC_ONLY",
        "run_id": config.run_id,
        "method": STAGE7_METHOD,
        "stage6_package_sha256": verified_stage6.package_sha256,
        "stage6_envelope_sha256": complete_descriptor["package_sha256"],
        "stage6_package_file_count": len(stage6_files),
        "authorization_consumed_once": True,
        "rebuild_record_count": rebuild["record_count"],
        "rebuild_equivalent": True,
        "identity_isolation_status": identity["status"],
        "identity_record_count": identity["record_count"],
        "case_count": len(cases),
        "state_counts": evaluation["state_counts"],
        "state_proportions": evaluation["state_proportions"],
        "fail_closed_count": evaluation["fail_closed_count"],
        "fail_closed_proportion": evaluation["fail_closed_proportion"],
        "out_of_distribution_count": evaluation["out_of_distribution_count"],
        "out_of_distribution_proportion": evaluation["out_of_distribution_proportion"],
        "both_witness_replays_passed_count": evaluation["both_witness_replays_passed_count"],
        "both_witness_replays_passed_proportion": evaluation[
            "both_witness_replays_passed_proportion"
        ],
        "alpha": locked["alpha"],
        "positive_threshold": locked["positive_threshold"],
        "negative_threshold": locked["negative_threshold"],
        "calibration_artifact_hash": locked["calibration_artifact_hash"],
        "config_hash": locked["config_hash"],
        "data_hash": locked["data_hash"],
        "model_hash": locked["model_hash"],
        "thresholds_reestimated": False,
        "test_split_used_for_calibration": False,
        "test_tuning_prohibited": True,
        "synthetic_capability_authorizes_real_data": False,
        "real_authorization_issued": False,
        "dataset_root_probed": False,
        "real_data_read_count": 0,
        "authority_boundary": STAGE7_AUTHORITY_BOUNDARY,
        "clinical_claim": False,
    }
    artifacts = {
        "audit.html": render_stage7_pilot_html(report).encode("utf-8"),
        "audit.json": render_stage7_pilot_json(report).encode("utf-8"),
        "config_snapshot.json": config_bytes,
        "model_provenance.json": model_bytes,
        "source_package.json": source_bytes,
        "summary.json": _canonical_json_bytes(summary),
        "transaction.json": _canonical_json_bytes(transaction),
    }
    return _Prepared(
        config=config,
        root_artifacts=artifacts,
        stage6_descriptor=stage6_descriptor,
        rebuild=rebuild,
        identity=identity,
        states=states,
    )


def run_stage7_toy_fixture(
    config_path: Path,
    stage6_package: Path,
    output_dir: Path,
) -> Stage7ToyRunResult:
    """Create Stage 7 output only after complete Stage 6 byte replay succeeds."""

    paths = (config_path, stage6_package, output_dir)
    if any(not isinstance(path, Path) for path in paths):
        raise TypeError("all Stage 7 paths must be pathlib.Path values")
    source_roots = tuple(path.resolve(strict=True) for path in paths[:2])
    target = output_dir.resolve(strict=False)
    _require_nonoverlap((*source_roots, target))
    config = load_stage7_toy_config(source_roots[0])
    stage6_root = source_roots[1]
    stage6_files = _read_tree(stage6_root, label="Stage 6")
    prepared = _prepare(config, stage6_root=stage6_root, stage6_files=stage6_files)
    payloads = dict(prepared.root_artifacts)
    payloads.update({f"stage6/package/{name}": data for name, data in stage6_files.items()})
    payloads["package_manifest.json"] = _canonical_json_bytes(_package_manifest(payloads))
    write_stage3_bundle(
        output_dir,
        payloads,
        boundary=AuditWriteBoundary(allowed_root=output_dir),
    )
    return verify_stage7_toy_package(output_dir)


def verify_stage7_toy_package(output_dir: Path) -> Stage7ToyRunResult:
    """Replay embedded Stage 6 and regenerate every Stage 7 artifact byte-for-byte."""

    if not isinstance(output_dir, Path):
        raise TypeError("Stage 7 package root must be a pathlib.Path")
    package_root = output_dir.resolve(strict=True)
    actual = _read_tree(package_root, label="Stage 7")
    manifest_bytes = actual.get("package_manifest.json")
    if manifest_bytes is None:
        raise ValueError("Stage 7 package manifest is missing")
    manifest = _json_mapping(manifest_bytes, "Stage 7 package manifest")
    _verify_package_manifest(manifest, actual)
    root_files = {name: actual[name] for name in _ROOT_FILES}
    stage6_files = _prefixed(actual, "stage6/package/")
    expected_paths = {
        "package_manifest.json",
        *_ROOT_FILES,
        *(f"stage6/package/{name}" for name in stage6_files),
    }
    if set(actual) != expected_paths:
        raise ValueError("Stage 7 package contains an unknown path")
    config = parse_stage7_toy_config(
        _decode_json(root_files["config_snapshot.json"], "Stage 7 config snapshot")
    )
    prepared = _prepare(
        config,
        stage6_root=package_root / "stage6" / "package",
        stage6_files=stage6_files,
    )
    if dict(prepared.root_artifacts) != root_files:
        raise ValueError("Stage 7 root artifacts do not regenerate byte-for-byte")
    expected_payloads = dict(prepared.root_artifacts)
    expected_payloads.update(
        {f"stage6/package/{name}": data for name, data in stage6_files.items()}
    )
    expected_manifest = _package_manifest(expected_payloads)
    if manifest != expected_manifest:
        raise ValueError("Stage 7 package manifest does not match regenerated artifacts")
    return Stage7ToyRunResult(
        output_dir=output_dir,
        package_sha256=cast(str, expected_manifest["package_sha256"]),
        package_file_count=len(actual),
        stage6_package_sha256=cast(str, prepared.stage6_descriptor["package_sha256"]),
        stage6_envelope_sha256=cast(str, prepared.stage6_descriptor["envelope_sha256"]),
        rebuilt_record_count=cast(int, prepared.rebuild["record_count"]),
        identity_record_count=cast(int, prepared.identity["record_count"]),
        case_count=len(prepared.states),
        states=prepared.states,
        verified=True,
    )


def _validate_stage6_safety(audit: Mapping[str, object], summary: Mapping[str, object]) -> None:
    expected_false = (
        "synthetic_capabilities_authorize_real_data",
        "real_authorization_issued",
        "dataset_root_probed",
        "real_data_used",
        "thresholds_reestimated",
        "test_split_used_for_calibration",
    )
    if (
        audit.get("schema_version") != "6.0.0"
        or audit.get("scope") != "SYNTHETIC_ONLY"
        or audit.get("authority_boundary") != "COMPLETE_VERIFIED_STAGE6_PACKAGE"
        or audit.get("test_tuning_prohibited") is not True
        or any(audit.get(name) is not False for name in expected_false)
        or audit.get("real_data_read_count") != 0
    ):
        raise ValueError("Stage 6 source violates its synthetic safety contract")
    if (
        summary.get("schema_version") != "6.0.0"
        or summary.get("scope") != "SYNTHETIC_ONLY"
        or summary.get("authority_boundary") != "COMPLETE_VERIFIED_STAGE6_PACKAGE"
        or summary.get("dataset_root_probed") is not False
        or summary.get("real_data_read_count") != 0
        or summary.get("thresholds_reestimated") is not False
        or summary.get("test_tuning_prohibited") is not True
    ):
        raise ValueError("Stage 6 summary violates its synthetic safety contract")


def _validate_config_source(config: Stage7ToyConfig, descriptor: Mapping[str, object]) -> None:
    source = config.source_contract
    observed = (
        descriptor["scope"],
        descriptor["run_id"],
        descriptor["authority_boundary"],
        descriptor["package_file_count"],
    )
    expected = (
        source.stage6_scope,
        source.stage6_run_id,
        source.stage6_authority_boundary,
        source.stage6_package_file_count,
    )
    if observed != expected or descriptor["complete_byte_replay_verified"] is not True:
        raise ValueError("complete Stage 6 package does not match the Stage 7 config")


def _rebuild_stage2_manifest(files: Mapping[str, bytes]) -> dict[str, object]:
    manifest_path = "stage2/package/manifest-v2.json"
    audit_path = "stage2/package/audit.json"
    manifest_bytes = files.get(manifest_path)
    audit_bytes = files.get(audit_path)
    if manifest_bytes is None or audit_bytes is None:
        raise ValueError("embedded Stage 2 manifest or audit is missing")
    manifest_mapping = _json_mapping(manifest_bytes, "embedded Stage 2 manifest")
    original = DatasetManifestV2.from_dict(manifest_mapping)
    rebuilt = DatasetManifestV2.from_dict(original.to_dict())
    original_dict = original.to_dict()
    rebuilt_dict = rebuilt.to_dict()
    record_bindings: list[dict[str, object]] = []
    mismatches = 0
    for index, (source, reconstructed) in enumerate(
        zip(original.records, rebuilt.records, strict=True)
    ):
        source_sha256 = sha256_bytes(_canonical_compact_bytes(source.to_dict()))
        rebuilt_sha256 = sha256_bytes(_canonical_compact_bytes(reconstructed.to_dict()))
        if source.to_dict() != reconstructed.to_dict() or source_sha256 != rebuilt_sha256:
            mismatches += 1
        record_bindings.append(
            {
                "index": index,
                "source_image_id": source.source_image_id,
                "relative_path": source.relative_path,
                "source_record_sha256": source_sha256,
                "rebuilt_record_sha256": rebuilt_sha256,
            }
        )
    original_sha256 = sha256_bytes(_canonical_compact_bytes(original_dict))
    rebuilt_sha256 = sha256_bytes(_canonical_compact_bytes(rebuilt_dict))
    rebuilt_bytes = _canonical_json_bytes(rebuilt_dict)
    audit = _json_mapping(audit_bytes, "embedded Stage 2 audit")
    if audit.get("status") != "PASS" or audit.get("manifest_sha256") != original_sha256:
        raise ValueError("embedded Stage 2 audit is not bound to the manifest")
    if original_dict != rebuilt_dict or manifest_bytes != rebuilt_bytes or mismatches:
        raise ValueError("typed Stage 2 manifest rebuild is not byte-equivalent")
    return {
        "mode": "EMBEDDED_STAGE2_TYPED_ROUNDTRIP",
        "dataset_id": original.dataset_id,
        "dataset_version": original.dataset_version,
        "record_count": len(original.records),
        "original_manifest_sha256": original_sha256,
        "rebuilt_manifest_sha256": rebuilt_sha256,
        "record_fingerprints_sha256": sha256_bytes(_canonical_compact_bytes(record_bindings)),
        "canonical_equivalent": True,
        "byte_equivalent": True,
        "mismatch_count": 0,
        "physical_dataset_root_used": False,
        "missing_label_policy": "UNKNOWN_NEVER_NEGATIVE",
    }


def _rerun_five_identity_audit(files: Mapping[str, bytes]) -> dict[str, object]:
    pilot_bytes = files.get("stage5/package/pilot_data.json")
    audit_bytes = files.get("stage5/package/identity_isolation.json")
    if pilot_bytes is None or audit_bytes is None:
        raise ValueError("embedded Stage 5 identity evidence is missing")
    pilot = _json_mapping(pilot_bytes, "embedded Stage 5 pilot data")
    raw_records = pilot.get("identity_records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("embedded Stage 5 identity records are missing")
    records: list[Stage5IdentityRecord] = []
    required = {
        "dataset_id",
        "relative_path",
        "split",
        "source_image_id",
        "patient_id",
        "eye_id",
        "followup_id",
        "intervention_pair_id",
    }
    for index, value in enumerate(raw_records):
        data = _strict_mapping(value, f"identity_records[{index}]", required)
        records.append(
            Stage5IdentityRecord(
                dataset_id=_text(data["dataset_id"], "dataset_id"),
                relative_path=_text(data["relative_path"], "relative_path"),
                split=_text(data["split"], "split"),
                source_image_id=_text(data["source_image_id"], "source_image_id"),
                patient_id=_optional_text(data["patient_id"], "patient_id"),
                eye_id=_optional_text(data["eye_id"], "eye_id"),
                followup_id=_optional_text(data["followup_id"], "followup_id"),
                intervention_pair_id=_optional_text(
                    data["intervention_pair_id"], "intervention_pair_id"
                ),
            )
        )
    recomputed = audit_five_identity_split_isolation(records)
    frozen = _json_mapping(audit_bytes, "embedded Stage 5 identity audit")
    if (
        recomputed.status is not Stage5IdentityAuditStatus.PASS
        or not recomputed.isolation_proven
        or recomputed.to_dict() != frozen
    ):
        raise ValueError("five-identity split-isolation replay did not match")
    return cast(dict[str, object], recomputed.to_dict())


def _file_descriptor(files: Mapping[str, bytes]) -> dict[str, object]:
    entries = [
        {"relative_path": name, "size_bytes": len(payload), "sha256": sha256_bytes(payload)}
        for name, payload in sorted(files.items())
    ]
    base = {"digest_method": STAGE7_DIGEST_METHOD, "file_count": len(entries), "files": entries}
    return {**base, "package_sha256": sha256_bytes(_canonical_compact_bytes(base))}


def _package_manifest(payloads: Mapping[str, bytes]) -> dict[str, object]:
    descriptor = _file_descriptor(payloads)
    return {
        "schema_version": STAGE7_PACKAGE_VERSION,
        "digest_method": descriptor["digest_method"],
        "file_count": descriptor["file_count"],
        "files": descriptor["files"],
        "package_sha256": descriptor["package_sha256"],
    }


def _verify_package_manifest(value: Mapping[str, object], actual: Mapping[str, bytes]) -> None:
    data = _strict_mapping(
        value,
        "Stage 7 package manifest",
        {"schema_version", "digest_method", "file_count", "files", "package_sha256"},
    )
    if (
        data["schema_version"] != STAGE7_PACKAGE_VERSION
        or data["digest_method"] != STAGE7_DIGEST_METHOD
    ):
        raise ValueError("Stage 7 package manifest version or method is invalid")
    payloads = {
        name: payload for name, payload in actual.items() if name != "package_manifest.json"
    }
    if dict(value) != _package_manifest(payloads):
        raise ValueError("Stage 7 package manifest does not match package bytes")


def _read_tree(root: Path, *, label: str) -> dict[str, bytes]:
    if not isinstance(root, Path):
        raise TypeError(f"{label} package root must be a pathlib.Path")
    _reject_link_components(root)
    root_stat = os.lstat(root)
    if not stat.S_ISDIR(root_stat.st_mode) or _is_link_or_reparse(root_stat):
        raise ValueError(f"{label} package root must be a physical directory")
    files: dict[str, bytes] = {}
    total = 0

    def visit(directory: Path, relative: PurePosixPath, depth: int) -> None:
        nonlocal total
        if depth > _MAX_DEPTH:
            raise ValueError(f"{label} package exceeds its depth limit")
        with os.scandir(directory) as entries:
            ordered = sorted(entries, key=lambda item: item.name)
        for entry in ordered:
            metadata = os.lstat(entry.path)
            if _is_link_or_reparse(metadata):
                raise ValueError(f"{label} package contains a link or reparse point")
            child_relative = relative / entry.name
            name = child_relative.as_posix()
            _relative_name(name)
            if entry.is_dir(follow_symlinks=False):
                before_count = len(files)
                visit(Path(entry.path), child_relative, depth + 1)
                if len(files) == before_count:
                    raise ValueError(f"{label} package contains an empty directory")
                continue
            if not entry.is_file(follow_symlinks=False):
                raise ValueError(f"{label} package contains a non-regular entry")
            if metadata.st_size > _MAX_FILE_BYTES:
                raise ValueError(f"{label} package artifact exceeds its size limit")
            payload = _read_regular_bounded(Path(entry.path), metadata)
            after = os.lstat(entry.path)
            if _identity(metadata) != _identity(after) or len(payload) != metadata.st_size:
                raise ValueError(f"{label} package artifact changed while being read")
            total += len(payload)
            if total > _MAX_PACKAGE_BYTES or len(files) >= _MAX_PACKAGE_FILES:
                raise ValueError(f"{label} package exceeds its aggregate limit")
            files[name] = payload

    visit(root, PurePosixPath(), 0)
    if not files:
        raise ValueError(f"{label} package must not be empty")
    return files


def _read_regular_bounded(path: Path, expected: os.stat_result) -> bytes:
    flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0)) | int(getattr(os, "O_NOFOLLOW", 0))
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if _identity(opened) != _identity(expected) or _is_link_or_reparse(opened):
            raise ValueError("package artifact identity changed before reading")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, _MAX_FILE_BYTES + 1 - size))
            if not chunk:
                break
            size += len(chunk)
            if size > _MAX_FILE_BYTES:
                raise ValueError("package artifact exceeds its size limit")
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _identity(after) != _identity(expected):
        raise ValueError("package artifact changed while being read")
    return b"".join(chunks)


def _reject_link_components(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        metadata = os.lstat(current)
        if _is_link_or_reparse(metadata):
            raise ValueError("package path contains a link or reparse point")


def _prefixed(files: Mapping[str, bytes], prefix: str) -> dict[str, bytes]:
    result = {
        name.removeprefix(prefix): payload
        for name, payload in files.items()
        if name.startswith(prefix)
    }
    if not result:
        raise ValueError(f"Stage 7 package is missing {prefix}")
    return result


def _is_link_or_reparse(value: os.stat_result) -> bool:
    attributes = int(getattr(value, "st_file_attributes", 0))
    reparse = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return stat.S_ISLNK(value.st_mode) or bool(attributes & reparse)


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _require_nonoverlap(paths: Sequence[Path]) -> None:
    for index, first in enumerate(paths):
        for second in paths[index + 1 :]:
            if first == second or first in second.parents or second in first.parents:
                raise ValueError("Stage 7 source and output paths must not overlap")


def _relative_name(value: str) -> str:
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        not value
        or "\\" in value
        or posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or posix.as_posix() != value
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise ValueError("package paths must be normalized relative POSIX paths")
    return value


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _canonical_compact_bytes(value: object) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _decode_json(payload: bytes, name: str) -> object:
    try:
        return json.loads(
            payload.decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"{name} contains non-finite constant {value}")
            ),
            object_pairs_hook=_unique_json_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{name} is not strict UTF-8 JSON") from error


def _unique_json_object(pairs: list[tuple[object, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if not isinstance(key, str):
            raise ValueError("JSON object keys must be strings")
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _json_mapping(payload: bytes, name: str) -> dict[str, object]:
    return _mapping(_decode_json(payload, name), name)


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a string-keyed mapping")
    return dict(cast(Mapping[str, object], value))


def _strict_mapping(value: object, name: str, required: set[str]) -> dict[str, object]:
    result = _mapping(value, name)
    if set(result) != required:
        raise Stage7ToyConfigurationError(f"{name} fields are incomplete or unknown")
    return result


def _text(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(character in value for character in "\r\n\x00")
    ):
        raise Stage7ToyConfigurationError(f"{name} must be safe non-empty text")
    return value


def _optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise Stage7ToyConfigurationError(f"{name} must be a positive integer")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise Stage7ToyConfigurationError(f"{name} must be finite")
    return float(value)


def _boolean(value: object, name: str) -> bool:
    if type(value) is not bool:  # noqa: E721
        raise Stage7ToyConfigurationError(f"{name} must be an exact boolean")
    return value


__all__ = [
    "STAGE7_PACKAGE_VERSION",
    "STAGE7_TOY_CONFIG_VERSION",
    "Stage7SourceContract",
    "Stage7ToyConfig",
    "Stage7ToyConfigurationError",
    "Stage7ToyRunResult",
    "load_stage7_toy_config",
    "parse_stage7_toy_config",
    "run_stage7_toy_fixture",
    "verify_stage7_toy_package",
]
