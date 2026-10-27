"""Local Stage 11 Ed25519 rotation/tamper fixture and complete offline verifier."""

from __future__ import annotations

import base64
import copy
import json
import os
import stat
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import yaml
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from retinaentail.pilot.stage10_fixture import verify_stage10_toy_package
from retinaentail.pilot.stage11_gate import (
    Stage11CapabilityError,
    Stage11RunnerAuthority,
    consume_stage11_host_pair,
    host_capability_snapshot,
    issue_synthetic_stage10_host_capability,
)
from retinaentail.reporting.stage11 import render_stage11_audit_html, render_stage11_audit_json
from retinaentail.reproducibility.stage11 import (
    STAGE11_AUTHENTICATION_SCOPE,
    STAGE11_DEPENDENCY_LOCK_KIND,
    STAGE11_RUNTIME_LOCK_KIND,
    STAGE11_SCHEMA_VERSION,
    STAGE11_SIGNATURE_ALGORITHM,
    Stage11VerificationCode,
    Stage11VerificationError,
    build_fixture_certificate,
    build_fixture_revocation_list,
    build_fixture_rotation_proof,
    build_fixture_trust_anchor,
    build_signed_envelope,
    canonical_json_bytes,
    canonical_sha256,
    capture_stage11_dependency_lock,
    capture_stage11_runtime_lock,
    compare_verified_envelope_payloads,
    ensure_independent_runner_keys,
    fixture_private_key,
    key_id,
    sha256_bytes,
    verify_runner_bundle,
)
from retinaentail.witnesses.replay import write_bytes_fail_safe

STAGE11_TOY_CONFIG_VERSION: Final = "11.0.0"
STAGE11_METHOD: Final = "local_ed25519_rotation_tamper_fixture_v1"
STAGE11_SCOPE: Final = "SYNTHETIC_FIXTURE"
STAGE11_AUTHORITY_BOUNDARY: Final = "COMPLETE_VERIFIED_STAGE11_SYNTHETIC_PACKAGE"
STAGE11_DIGEST_METHOD: Final = "sha256-canonical-file-manifest-v1"
_MAX_CONFIG_BYTES: Final = 128 * 1024
_MAX_FILE_BYTES: Final = 32 * 1024 * 1024
_MAX_PACKAGE_BYTES: Final = 256 * 1024 * 1024
_MAX_PACKAGE_FILES: Final = 2_000
_ROOT_FILES: Final = {
    "audit.html",
    "audit.json",
    "baseline_comparison.json",
    "config_snapshot.json",
    "dependency_lock.json",
    "runtime_lock.json",
    "source_authority.json",
    "summary.json",
    "tamper_audit.json",
}
_RUNNER_FILES: Final = {
    "current_certificate.json",
    "envelope.json",
    "old_certificate.json",
    "rotation_proof.json",
}
_TAMPER_SCENARIOS: Final = (
    "payload_mutation",
    "signature_mutation",
    "untrusted_anchor",
    "current_leaf_revoked",
    "expired_certificate",
    "rotation_proof_mutation",
    "stale_revocation_status",
    "revocation_signature_mutation",
    "session_nonce_replay",
    "shared_runner_leaf_key",
)


class Stage11ToyConfigurationError(ValueError):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage11ToyConfig:
    run_id: str
    dataset_id: str
    dataset_version: str
    runner_ids: tuple[str, str]
    trusted_time_epoch: int
    source_stage10_package_sha256: str
    source_stage10_package_file_count: int
    source_stage10_manifest_sha256: str
    source_stage10_audit_sha256: str
    source_stage10_summary_sha256: str
    source_stage10_environment_lock_sha256: str
    trust_anchor_sha256: str
    dependency_distributions: tuple[str, ...]
    alpha: float
    positive_threshold: float
    negative_threshold: float
    positive_sample_count: int
    negative_sample_count: int
    config_hash: str
    data_hash: str
    model_hash: str
    tamper_scenarios: tuple[str, ...]
    raw: Mapping[str, object]


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage11ToyRunResult:
    output_dir: Path
    package_sha256: str
    package_file_count: int
    source_stage10_package_sha256: str
    source_stage10_package_file_count: int
    trust_anchor_sha256: str
    runner_key_ids: tuple[str, str]
    tamper_scenario_count: int
    tamper_rejection_count: int
    baseline_status: str
    status: str
    verified: bool


def _strict_mapping(value: object, name: str, fields: set[str]) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise Stage11ToyConfigurationError(f"{name} fields are incomplete or unknown")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or any(c in value for c in "\r\n\x00"):
        raise Stage11ToyConfigurationError(f"{name} is invalid")
    return value


def _digest(value: object, name: str) -> str:
    result = _text(value, name)
    if len(result) != 64 or any(c not in "0123456789abcdef" for c in result):
        raise Stage11ToyConfigurationError(f"{name} is not a lowercase SHA-256")
    return result


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Stage11ToyConfigurationError(f"{name} must be a positive integer")
    return value


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise Stage11ToyConfigurationError(f"{name} must be numeric")
    result = float(value)
    if not result == result or result in (float("inf"), float("-inf")):
        raise Stage11ToyConfigurationError(f"{name} must be finite")
    return result


def parse_stage11_toy_config(value: object) -> Stage11ToyConfig:
    root = _strict_mapping(
        value,
        "Stage 11 config",
        {
            "schema_version",
            "run_id",
            "method",
            "scope",
            "dataset_id",
            "dataset_version",
            "split",
            "runner_ids",
            "physical_host_count",
            "cross_physical_host_claim",
            "trusted_time_epoch",
            "source_contract",
            "trust_policy",
            "environment_policy",
            "locked_contract",
            "tamper_scenarios",
            "locked_policy",
        },
    )
    if (
        root["schema_version"] != STAGE11_TOY_CONFIG_VERSION
        or root["method"] != STAGE11_METHOD
        or root["scope"] != STAGE11_SCOPE
        or root["split"] != "test"
        or root["physical_host_count"] != 1
        or root["cross_physical_host_claim"] is not False
    ):
        raise Stage11ToyConfigurationError("Stage 11 config exceeds local fixture scope")
    raw_runner_ids = root["runner_ids"]
    if not isinstance(raw_runner_ids, list) or len(raw_runner_ids) != 2:
        raise Stage11ToyConfigurationError("Stage 11 requires exactly two logical runners")
    runner_ids = tuple(_text(item, "runner id") for item in raw_runner_ids)
    if len(set(runner_ids)) != 2:
        raise Stage11ToyConfigurationError("Stage 11 runner IDs must be unique")
    source = _strict_mapping(
        root["source_contract"],
        "source contract",
        {
            "stage10_package_sha256",
            "stage10_package_file_count",
            "stage10_manifest_sha256",
            "stage10_audit_sha256",
            "stage10_summary_sha256",
            "stage10_environment_lock_sha256",
        },
    )
    trust = _strict_mapping(
        root["trust_policy"],
        "trust policy",
        {
            "algorithm",
            "authentication_scope",
            "trust_anchor_sha256",
            "fixture_private_derivation_public",
            "production_authentication",
            "unique_runner_leaf_keys_required",
            "rotation_required",
            "revocation_status_required",
            "trusted_time_required",
            "private_keys_written_to_artifacts",
        },
    )
    if (
        trust["algorithm"] != STAGE11_SIGNATURE_ALGORITHM
        or trust["authentication_scope"] != STAGE11_AUTHENTICATION_SCOPE
        or trust["fixture_private_derivation_public"] is not True
        or trust["production_authentication"] is not False
        or trust["unique_runner_leaf_keys_required"] is not True
        or trust["rotation_required"] is not True
        or trust["revocation_status_required"] is not True
        or trust["trusted_time_required"] is not True
        or trust["private_keys_written_to_artifacts"] is not False
    ):
        raise Stage11ToyConfigurationError("Stage 11 trust policy is unsafe")
    environment = _strict_mapping(
        root["environment_policy"],
        "environment policy",
        {
            "dependency_distributions",
            "complete_wheel_archive_hashes",
            "production_hardware_attestation",
            "cpu_instruction_policy",
            "exact_stage10_environment_lock_required",
        },
    )
    raw_distributions = environment["dependency_distributions"]
    if not isinstance(raw_distributions, list) or not raw_distributions:
        raise Stage11ToyConfigurationError("dependency distribution list is empty")
    distributions = tuple(sorted(_text(item, "distribution") for item in raw_distributions))
    if (
        len(distributions) != len(set(distributions))
        or environment["complete_wheel_archive_hashes"] is not False
        or environment["production_hardware_attestation"] is not False
        or environment["cpu_instruction_policy"] != "RECORDED_NOT_HARDWARE_ATTESTED"
        or environment["exact_stage10_environment_lock_required"] is not True
    ):
        raise Stage11ToyConfigurationError("local dependency/host evidence scope is invalid")
    locked = _strict_mapping(
        root["locked_contract"],
        "locked contract",
        {
            "alpha",
            "positive_threshold",
            "negative_threshold",
            "positive_sample_count",
            "negative_sample_count",
            "config_hash",
            "data_hash",
            "model_hash",
        },
    )
    raw_scenarios = root["tamper_scenarios"]
    if not isinstance(raw_scenarios, list):
        raise Stage11ToyConfigurationError("tamper scenarios must be a list")
    scenarios = tuple(_text(item, "tamper scenario") for item in raw_scenarios)
    if scenarios != _TAMPER_SCENARIOS:
        raise Stage11ToyConfigurationError("tamper scenarios differ from preregistration")
    policy = _strict_mapping(
        root["locked_policy"],
        "locked policy",
        {
            "test_tuning_prohibited",
            "threshold_reestimation_prohibited",
            "sample_replacement_prohibited",
            "network_used",
            "gpu_used",
            "real_data_access",
            "deterministic_difference_fail_closed",
            "large_model_training_prohibited",
            "weights_download_prohibited",
        },
    )
    required_true = {
        "test_tuning_prohibited",
        "threshold_reestimation_prohibited",
        "sample_replacement_prohibited",
        "deterministic_difference_fail_closed",
        "large_model_training_prohibited",
        "weights_download_prohibited",
    }
    required_false = {"network_used", "gpu_used", "real_data_access"}
    if any(policy[name] is not True for name in required_true) or any(
        policy[name] is not False for name in required_false
    ):
        raise Stage11ToyConfigurationError("Stage 11 locked policy is unsafe")
    return Stage11ToyConfig(
        run_id=_text(root["run_id"], "run id"),
        dataset_id=_text(root["dataset_id"], "dataset id"),
        dataset_version=_text(root["dataset_version"], "dataset version"),
        runner_ids=cast(tuple[str, str], runner_ids),
        trusted_time_epoch=_positive_int(root["trusted_time_epoch"], "trusted time"),
        source_stage10_package_sha256=_digest(source["stage10_package_sha256"], "Stage 10 package"),
        source_stage10_package_file_count=_positive_int(
            source["stage10_package_file_count"], "Stage 10 file count"
        ),
        source_stage10_manifest_sha256=_digest(
            source["stage10_manifest_sha256"], "Stage 10 manifest"
        ),
        source_stage10_audit_sha256=_digest(source["stage10_audit_sha256"], "Stage 10 audit"),
        source_stage10_summary_sha256=_digest(source["stage10_summary_sha256"], "Stage 10 summary"),
        source_stage10_environment_lock_sha256=_digest(
            source["stage10_environment_lock_sha256"], "Stage 10 environment lock"
        ),
        trust_anchor_sha256=_digest(trust["trust_anchor_sha256"], "trust anchor"),
        dependency_distributions=distributions,
        alpha=_finite_number(locked["alpha"], "alpha"),
        positive_threshold=_finite_number(locked["positive_threshold"], "positive threshold"),
        negative_threshold=_finite_number(locked["negative_threshold"], "negative threshold"),
        positive_sample_count=_positive_int(
            locked["positive_sample_count"], "positive sample count"
        ),
        negative_sample_count=_positive_int(
            locked["negative_sample_count"], "negative sample count"
        ),
        config_hash=_digest(locked["config_hash"], "config hash"),
        data_hash=_digest(locked["data_hash"], "data hash"),
        model_hash=_digest(locked["model_hash"], "model hash"),
        tamper_scenarios=scenarios,
        raw=root,
    )


def load_stage11_toy_config(path: Path) -> Stage11ToyConfig:
    if not isinstance(path, Path):
        raise TypeError("Stage 11 config path must be pathlib.Path")
    resolved = path.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise FileNotFoundError(path)
    payload = resolved.read_bytes()
    if len(payload) > _MAX_CONFIG_BYTES:
        raise Stage11ToyConfigurationError("Stage 11 config exceeds its size limit")
    try:
        value = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise Stage11ToyConfigurationError("Stage 11 config is not strict UTF-8 YAML") from error
    return parse_stage11_toy_config(value)


def _pretty_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _unique_object(pairs: list[tuple[object, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if not isinstance(key, str) or key in result:
            raise ValueError("JSON object has a non-string or duplicate key")
        result[key] = value
    return result


def _decode_json(payload: bytes, name: str) -> object:
    try:
        return json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not strict UTF-8 JSON") from error


def _json_mapping(payload: bytes, name: str) -> dict[str, object]:
    value = _decode_json(payload, name)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain an object")
    return value


def _read_tree(root: Path, *, label: str) -> dict[str, bytes]:
    package_root = root.resolve(strict=True)
    if package_root.is_symlink() or not package_root.is_dir():
        raise ValueError(f"{label} root must be a regular directory")
    result: dict[str, bytes] = {}
    total = 0
    for path in sorted(package_root.rglob("*")):
        info = path.stat(follow_symlinks=False)
        if path.is_symlink() or bool(
            getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
        ):
            raise ValueError(f"{label} contains a link or reparse point")
        if path.is_dir():
            continue
        if not path.is_file() or info.st_size > _MAX_FILE_BYTES:
            raise ValueError(f"{label} contains a non-regular or oversized file")
        relative = path.relative_to(package_root).as_posix()
        if not relative or relative.startswith("/") or "\\" in relative or "\x00" in relative:
            raise ValueError(f"{label} contains an unsafe path")
        payload = path.read_bytes()
        after = path.stat(follow_symlinks=False)
        if (info.st_size, info.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or len(
            payload
        ) != info.st_size:
            raise ValueError(f"{label} changed while read")
        total += len(payload)
        if total > _MAX_PACKAGE_BYTES or len(result) >= _MAX_PACKAGE_FILES:
            raise ValueError(f"{label} exceeds package bounds")
        result[relative] = payload
    if not result:
        raise ValueError(f"{label} package is empty")
    return result


def _prefixed(files: Mapping[str, bytes], prefix: str) -> dict[str, bytes]:
    return {
        name[len(prefix) :]: payload for name, payload in files.items() if name.startswith(prefix)
    }


def _file_manifest(payloads: Mapping[str, bytes]) -> dict[str, object]:
    files = [
        {"relative_path": name, "sha256": sha256_bytes(payload), "size_bytes": len(payload)}
        for name, payload in sorted(payloads.items())
    ]
    return {
        "schema_version": STAGE11_SCHEMA_VERSION,
        "digest_method": STAGE11_DIGEST_METHOD,
        "file_count": len(files),
        "files": files,
        "package_sha256": sha256_bytes(canonical_json_bytes(files)),
    }


def _verify_manifest(manifest: Mapping[str, object], actual: Mapping[str, bytes]) -> None:
    payloads = {
        name: payload for name, payload in actual.items() if name != "package_manifest.json"
    }
    if dict(manifest) != _file_manifest(payloads):
        raise ValueError("Stage 11 package manifest does not match complete package bytes")


def _paths_nonoverlapping(values: Sequence[Path]) -> None:
    resolved = tuple(item.resolve(strict=False) for item in values)
    for index, first in enumerate(resolved):
        for second in resolved[index + 1 :]:
            if first == second or first in second.parents or second in first.parents:
                raise ValueError("Stage 11 config, source, and output paths must not overlap")


def _stage10_material(package_root: Path, config: Stage11ToyConfig) -> dict[str, object]:
    verified = verify_stage10_toy_package(package_root)
    files = _read_tree(package_root, label="source Stage 10 package")
    audit = _json_mapping(files["audit.json"], "Stage 10 audit")
    summary = _json_mapping(files["summary.json"], "Stage 10 summary")
    config_snapshot = _json_mapping(files["config_snapshot.json"], "Stage 10 config")
    environment_lock = _json_mapping(files["environment_lock.json"], "Stage 10 environment lock")
    manifest_sha = sha256_bytes(files["package_manifest.json"])
    audit_sha = sha256_bytes(files["audit.json"])
    summary_sha = sha256_bytes(files["summary.json"])
    environment_sha = canonical_sha256(environment_lock)
    if (
        verified.status != "SUCCESS"
        or verified.package_sha256 != config.source_stage10_package_sha256
        or verified.package_file_count != config.source_stage10_package_file_count
        or manifest_sha != config.source_stage10_manifest_sha256
        or audit_sha != config.source_stage10_audit_sha256
        or summary_sha != config.source_stage10_summary_sha256
        or environment_sha != config.source_stage10_environment_lock_sha256
    ):
        raise ValueError("source Stage 10 package differs from the frozen Stage 11 contract")
    locked = config_snapshot.get("locked_contract")
    if not isinstance(locked, dict) or any(
        locked.get(name) != value
        for name, value in {
            "alpha": config.alpha,
            "positive_threshold": config.positive_threshold,
            "negative_threshold": config.negative_threshold,
            "positive_sample_count": config.positive_sample_count,
            "negative_sample_count": config.negative_sample_count,
            "config_hash": config.config_hash,
            "data_hash": config.data_hash,
            "model_hash": config.model_hash,
        }.items()
    ):
        raise ValueError("source Stage 10 locked contract differs from Stage 11")
    diff = audit.get("diff")
    if not isinstance(diff, dict):
        raise ValueError("source Stage 10 diff is missing")
    baseline_projection = {
        "source_stage10_package_sha256": verified.package_sha256,
        "stage10_status": audit.get("status"),
        "stage10_scope": audit.get("scope"),
        "stage10_transition_matrix": diff.get("transition_matrix"),
        "stage10_metric_intervals": diff.get("metric_intervals"),
        "stage10_case_comparisons": diff.get("case_comparisons"),
        "stage10_artifact_diff_summary": audit.get("artifact_diff_summary"),
        "locked_contract": dict(locked),
    }
    return {
        "verified": verified,
        "files": files,
        "audit": audit,
        "summary": summary,
        "environment_lock": environment_lock,
        "manifest_sha256": manifest_sha,
        "audit_sha256": audit_sha,
        "summary_sha256": summary_sha,
        "environment_lock_sha256": environment_sha,
        "baseline_projection": baseline_projection,
        "baseline_projection_sha256": canonical_sha256(baseline_projection),
    }


def _certificate_payload(certificate: Mapping[str, object]) -> Mapping[str, object]:
    payload = certificate.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("certificate payload is missing")
    return payload


def _envelope_payload(
    *,
    config: Stage11ToyConfig,
    runner_id: str,
    source: Mapping[str, object],
    runtime_lock: Mapping[str, object],
    dependency_lock: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": STAGE11_SCHEMA_VERSION,
        "runner_id": runner_id,
        "logical_host_id": f"{runner_id}-local-slot",
        "producer_process_id": os.getpid(),
        "session_nonce_sha256": canonical_sha256(
            {"run_id": config.run_id, "runner_id": runner_id, "kind": "stage11-session"}
        ),
        "dataset_id": config.dataset_id,
        "dataset_version": config.dataset_version,
        "scope": "SYNTHETIC_ONLY",
        "split": "test",
        "source_stage10_package_sha256": config.source_stage10_package_sha256,
        "source_stage10_package_file_count": config.source_stage10_package_file_count,
        "stage10_manifest_sha256": config.source_stage10_manifest_sha256,
        "stage10_audit_sha256": config.source_stage10_audit_sha256,
        "stage10_summary_sha256": config.source_stage10_summary_sha256,
        "stage10_environment_lock_sha256": config.source_stage10_environment_lock_sha256,
        "baseline_projection_sha256": source["baseline_projection_sha256"],
        "runtime_lock_sha256": runtime_lock["runtime_lock_sha256"],
        "dependency_lock_sha256": dependency_lock["lock_sha256"],
        "trust_anchor_sha256": config.trust_anchor_sha256,
        "host_profile_sha256": runtime_lock["runtime_lock_sha256"],
        "alpha": config.alpha,
        "positive_threshold": config.positive_threshold,
        "negative_threshold": config.negative_threshold,
        "positive_sample_count": config.positive_sample_count,
        "negative_sample_count": config.negative_sample_count,
        "config_hash": config.config_hash,
        "data_hash": config.data_hash,
        "model_hash": config.model_hash,
        "payload_contains_data": False,
        "payload_contains_root": False,
        "payload_contains_private_key": False,
        "authorizes_real_data": False,
        "authorizes_physical_root": False,
        "detached_envelope_authoritative": False,
        "physical_host_count": 1,
        "cross_physical_host_claim": False,
        "production_authentication": False,
    }


def _runner_artifacts(
    *,
    config: Stage11ToyConfig,
    runner_id: str,
    source: Mapping[str, object],
    runtime_lock: Mapping[str, object],
    dependency_lock: Mapping[str, object],
) -> tuple[dict[str, object], Ed25519PrivateKey, Ed25519PrivateKey]:
    old_private = fixture_private_key(f"{runner_id}:old")
    current_private = fixture_private_key(f"{runner_id}:current")
    old_certificate = build_fixture_certificate(
        subject=runner_id,
        generation=1,
        public_key=old_private.public_key(),
        previous_key_id=None,
        trusted_time_epoch=config.trusted_time_epoch,
    )
    current_certificate = build_fixture_certificate(
        subject=runner_id,
        generation=2,
        public_key=current_private.public_key(),
        previous_key_id=key_id(old_private.public_key()),
        trusted_time_epoch=config.trusted_time_epoch,
    )
    rotation = build_fixture_rotation_proof(
        runner_id=runner_id,
        old_certificate=old_certificate,
        current_certificate=current_certificate,
        old_private_key=old_private,
        source_stage10_package_sha256=config.source_stage10_package_sha256,
        trusted_time_epoch=config.trusted_time_epoch,
    )
    payload = _envelope_payload(
        config=config,
        runner_id=runner_id,
        source=source,
        runtime_lock=runtime_lock,
        dependency_lock=dependency_lock,
    )
    envelope = build_signed_envelope(
        payload,
        current_certificate=current_certificate,
        private_key=current_private,
    )
    return (
        {
            "old_certificate": old_certificate,
            "current_certificate": current_certificate,
            "rotation_proof": rotation,
            "envelope": envelope,
            "expected_payload": payload,
        },
        old_private,
        current_private,
    )


def _verify_runner(
    *,
    config: Stage11ToyConfig,
    runner_id: str,
    trust_anchor: Mapping[str, object],
    revocation_list: Mapping[str, object],
    artifacts: Mapping[str, object],
) -> dict[str, object]:
    expected = artifacts.get("expected_payload")
    expected_payload = expected if isinstance(expected, Mapping) else None
    return verify_runner_bundle(
        runner_id=runner_id,
        trust_anchor=trust_anchor,
        expected_trust_anchor_sha256=config.trust_anchor_sha256,
        revocation_list=revocation_list,
        old_certificate=artifacts["old_certificate"],
        current_certificate=artifacts["current_certificate"],
        rotation_proof=artifacts["rotation_proof"],
        envelope=artifacts["envelope"],
        trusted_time_epoch=config.trusted_time_epoch,
        expected_source_stage10_package_sha256=config.source_stage10_package_sha256,
        expected_payload=expected_payload,
    )


def _authority(
    payload: Mapping[str, object],
    current_certificate: Mapping[str, object],
) -> Stage11RunnerAuthority:
    certificate = _certificate_payload(current_certificate)
    return Stage11RunnerAuthority(
        runner_id=cast(str, payload["runner_id"]),
        logical_host_id=cast(str, payload["logical_host_id"]),
        producer_process_id=cast(int, payload["producer_process_id"]),
        session_nonce_sha256=cast(str, payload["session_nonce_sha256"]),
        dataset_id=cast(str, payload["dataset_id"]),
        dataset_version=cast(str, payload["dataset_version"]),
        scope=cast(str, payload["scope"]),
        split=cast(str, payload["split"]),
        source_stage10_package_sha256=cast(str, payload["source_stage10_package_sha256"]),
        stage10_environment_lock_sha256=cast(str, payload["stage10_environment_lock_sha256"]),
        runtime_lock_sha256=cast(str, payload["runtime_lock_sha256"]),
        dependency_lock_sha256=cast(str, payload["dependency_lock_sha256"]),
        trust_anchor_sha256=cast(str, payload["trust_anchor_sha256"]),
        current_key_id=cast(str, certificate["key_id"]),
        baseline_projection_sha256=cast(str, payload["baseline_projection_sha256"]),
        signature_verified=True,
    )


def _observed_code(callable_: Callable[[], object]) -> str:
    try:
        callable_()
    except Stage11VerificationError as error:
        return error.code.value
    return "NOT_REJECTED"


def _tamper_audit(
    *,
    config: Stage11ToyConfig,
    trust_anchor: Mapping[str, object],
    revocation_list: Mapping[str, object],
    runners: Mapping[str, Mapping[str, object]],
    private_keys: Mapping[str, tuple[Ed25519PrivateKey, Ed25519PrivateKey]],
) -> dict[str, object]:
    runner_a = config.runner_ids[0]
    runner_b = config.runner_ids[1]
    artifacts_a = runners[runner_a]
    artifacts_b = runners[runner_b]

    def verify_a(
        *,
        anchor: object = trust_anchor,
        expected_anchor: str = config.trust_anchor_sha256,
        revocations: object = revocation_list,
        current: object = artifacts_a["current_certificate"],
        rotation: object = artifacts_a["rotation_proof"],
        envelope: object = artifacts_a["envelope"],
        trusted_time: int = config.trusted_time_epoch,
    ) -> dict[str, object]:
        return verify_runner_bundle(
            runner_id=runner_a,
            trust_anchor=anchor,
            expected_trust_anchor_sha256=expected_anchor,
            revocation_list=revocations,
            old_certificate=artifacts_a["old_certificate"],
            current_certificate=current,
            rotation_proof=rotation,
            envelope=envelope,
            trusted_time_epoch=trusted_time,
            expected_source_stage10_package_sha256=config.source_stage10_package_sha256,
        )

    results: list[dict[str, object]] = []

    def record(scenario: str, expected: Stage11VerificationCode, observed: str) -> None:
        results.append(
            {
                "scenario": scenario,
                "expected_code": expected.value,
                "observed_code": observed,
                "rejected": observed == expected.value,
            }
        )

    mutated_payload = cast(dict[str, object], copy.deepcopy(artifacts_a["envelope"]))
    mutated_payload_body = cast(dict[str, object], mutated_payload["payload"])
    mutated_payload_body["baseline_projection_sha256"] = "0" * 64
    record(
        "payload_mutation",
        Stage11VerificationCode.ENVELOPE_SIGNATURE_INVALID,
        _observed_code(lambda: verify_a(envelope=mutated_payload)),
    )

    mutated_signature = cast(dict[str, object], copy.deepcopy(artifacts_a["envelope"]))
    mutated_signature["signature_b64"] = base64.b64encode(bytes(64)).decode("ascii")
    record(
        "signature_mutation",
        Stage11VerificationCode.ENVELOPE_SIGNATURE_INVALID,
        _observed_code(lambda: verify_a(envelope=mutated_signature)),
    )

    record(
        "untrusted_anchor",
        Stage11VerificationCode.UNTRUSTED_ROOT,
        _observed_code(lambda: verify_a(expected_anchor="0" * 64)),
    )

    current_serial_a = cast(
        str,
        _certificate_payload(cast(Mapping[str, object], artifacts_a["current_certificate"]))[
            "serial"
        ],
    )
    old_serials = [
        cast(
            str,
            _certificate_payload(cast(Mapping[str, object], runners[name]["old_certificate"]))[
                "serial"
            ],
        )
        for name in config.runner_ids
    ]
    revoked_current = build_fixture_revocation_list(
        revoked_certificates=[
            *((serial, "SUPERSEDED", config.trusted_time_epoch - 900) for serial in old_serials),
            (current_serial_a, "KEY_COMPROMISE", config.trusted_time_epoch - 60),
        ],
        trusted_time_epoch=config.trusted_time_epoch,
    )
    record(
        "current_leaf_revoked",
        Stage11VerificationCode.CERTIFICATE_REVOKED,
        _observed_code(lambda: verify_a(revocations=revoked_current)),
    )

    record(
        "expired_certificate",
        Stage11VerificationCode.CERTIFICATE_TIME_INVALID,
        _observed_code(lambda: verify_a(trusted_time=config.trusted_time_epoch + 90_000)),
    )

    mutated_rotation = cast(dict[str, object], copy.deepcopy(artifacts_a["rotation_proof"]))
    mutated_rotation["root_signature_b64"] = base64.b64encode(bytes(64)).decode("ascii")
    record(
        "rotation_proof_mutation",
        Stage11VerificationCode.ROTATION_SIGNATURE_INVALID,
        _observed_code(lambda: verify_a(rotation=mutated_rotation)),
    )

    record(
        "stale_revocation_status",
        Stage11VerificationCode.REVOCATION_STATUS_STALE,
        _observed_code(lambda: verify_a(trusted_time=config.trusted_time_epoch + 4_000)),
    )

    mutated_revocations = cast(dict[str, object], copy.deepcopy(revocation_list))
    mutated_revocations["signature_b64"] = base64.b64encode(bytes(64)).decode("ascii")
    record(
        "revocation_signature_mutation",
        Stage11VerificationCode.REVOCATION_SIGNATURE_INVALID,
        _observed_code(lambda: verify_a(revocations=mutated_revocations)),
    )

    replay_payload_b = cast(dict[str, object], copy.deepcopy(artifacts_b["expected_payload"]))
    replay_payload_b["session_nonce_sha256"] = cast(
        Mapping[str, object], artifacts_a["expected_payload"]
    )["session_nonce_sha256"]
    replay_envelope_b = build_signed_envelope(
        replay_payload_b,
        current_certificate=cast(Mapping[str, object], artifacts_b["current_certificate"]),
        private_key=private_keys[runner_b][1],
    )
    replay_bundle_b = dict(artifacts_b)
    replay_bundle_b["envelope"] = replay_envelope_b
    replay_bundle_b["expected_payload"] = replay_payload_b
    replay_verified_b = _verify_runner(
        config=config,
        runner_id=runner_b,
        trust_anchor=trust_anchor,
        revocation_list=revocation_list,
        artifacts=replay_bundle_b,
    )
    replay_comparison = compare_verified_envelope_payloads(
        cast(Mapping[str, object], artifacts_a["expected_payload"]), replay_verified_b
    )
    replay_observed = (
        Stage11VerificationCode.SESSION_NONCE_REPLAYED.value
        if Stage11VerificationCode.SESSION_NONCE_REPLAYED.value
        in cast(list[object], replay_comparison["differences"])
        else "NOT_REJECTED"
    )
    record(
        "session_nonce_replay",
        Stage11VerificationCode.SESSION_NONCE_REPLAYED,
        replay_observed,
    )

    old_private_b = private_keys[runner_b][0]
    shared_current_b = build_fixture_certificate(
        subject=runner_b,
        generation=2,
        public_key=private_keys[runner_a][1].public_key(),
        previous_key_id=key_id(old_private_b.public_key()),
        trusted_time_epoch=config.trusted_time_epoch,
    )
    shared_observed = _observed_code(
        lambda: ensure_independent_runner_keys(
            cast(Mapping[str, object], artifacts_a["current_certificate"]), shared_current_b
        )
    )
    record(
        "shared_runner_leaf_key",
        Stage11VerificationCode.SHARED_RUNNER_KEY_REJECTED,
        shared_observed,
    )

    return {
        "schema_version": STAGE11_SCHEMA_VERSION,
        "status": "SUCCESS" if all(item["rejected"] is True for item in results) else "FAIL_CLOSED",
        "scenario_count": len(results),
        "rejection_count": sum(item["rejected"] is True for item in results),
        "results": results,
        "failure_state": "OUT_OF_MODEL",
        "synthetic_only": True,
    }


def _source_authority(config: Stage11ToyConfig, source: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": STAGE11_SCHEMA_VERSION,
        "source_stage10_package_sha256": config.source_stage10_package_sha256,
        "source_stage10_package_file_count": config.source_stage10_package_file_count,
        "source_stage10_manifest_sha256": config.source_stage10_manifest_sha256,
        "source_stage10_audit_sha256": config.source_stage10_audit_sha256,
        "source_stage10_summary_sha256": config.source_stage10_summary_sha256,
        "source_stage10_environment_lock_sha256": config.source_stage10_environment_lock_sha256,
        "baseline_projection_sha256": source["baseline_projection_sha256"],
        "complete_source_stage10_replay_verified": True,
        "detached_source_summary_authoritative": False,
        "authorizes_real_data": False,
        "authorizes_physical_root": False,
    }


def _summary(
    *,
    config: Stage11ToyConfig,
    baseline_comparison: Mapping[str, object],
    tamper_audit: Mapping[str, object],
    trust_anchor_sha256: str,
) -> dict[str, object]:
    success = (
        baseline_comparison.get("status") == "SUCCESS" and tamper_audit.get("status") == "SUCCESS"
    )
    return {
        "schema_version": STAGE11_SCHEMA_VERSION,
        "run_id": config.run_id,
        "method": STAGE11_METHOD,
        "scope": STAGE11_SCOPE,
        "status": "SUCCESS" if success else "FAIL_CLOSED",
        "physical_host_count": 1,
        "logical_runner_identity_count": 2,
        "independent_runner_process_count": 0,
        "cross_physical_host_claim": False,
        "cross_physical_host_tested": False,
        "asymmetric_signature_algorithm": STAGE11_SIGNATURE_ALGORITHM,
        "authentication_scope": STAGE11_AUTHENTICATION_SCOPE,
        "trust_anchor_sha256": trust_anchor_sha256,
        "runner_leaf_keys_unique": True,
        "key_rotations_verified": 2,
        "revocation_status_verified": True,
        "trusted_time_policy_verified": True,
        "tamper_scenario_count": tamper_audit["scenario_count"],
        "tamper_rejection_count": tamper_audit["rejection_count"],
        "complete_wheel_archive_hashes": False,
        "production_hardware_attestation": False,
        "production_authentication": False,
        "private_keys_written_to_artifacts": False,
        "thresholds_reestimated": False,
        "test_split_used_for_tuning": False,
        "sample_replacement_performed": False,
        "network_used_during_fixture": False,
        "gpu_used": False,
        "weights_downloaded": False,
        "large_model_trained": False,
        "dataset_root_probed": False,
        "real_data_read_count": 0,
        "clinical_claim": False,
        "synthetic_authorizes_real_data": False,
    }


def _audit(
    *,
    config: Stage11ToyConfig,
    source_authority: Mapping[str, object],
    runtime_lock: Mapping[str, object],
    dependency_lock: Mapping[str, object],
    trust_anchor_sha256: str,
    runner_key_ids: tuple[str, str],
    baseline_comparison: Mapping[str, object],
    tamper_audit: Mapping[str, object],
    capability_lifecycle: Mapping[str, object],
    stage10_baseline_evidence: Mapping[str, object],
) -> dict[str, object]:
    summary = _summary(
        config=config,
        baseline_comparison=baseline_comparison,
        tamper_audit=tamper_audit,
        trust_anchor_sha256=trust_anchor_sha256,
    )
    return {
        **summary,
        "authority_boundary": STAGE11_AUTHORITY_BOUNDARY,
        "source_authority": dict(source_authority),
        "runtime_lock": dict(runtime_lock),
        "dependency_lock": dict(dependency_lock),
        "trust_policy": {
            "algorithm": STAGE11_SIGNATURE_ALGORITHM,
            "authentication_scope": STAGE11_AUTHENTICATION_SCOPE,
            "trust_anchor_sha256": trust_anchor_sha256,
            "runner_leaf_key_ids": list(runner_key_ids),
            "fixture_private_derivation_public": True,
            "production_authentication": False,
            "private_keys_written_to_artifacts": False,
            "offline_verification_completed": True,
        },
        "capability_lifecycle": dict(capability_lifecycle),
        "baseline_comparison": dict(baseline_comparison),
        "stage10_baseline_evidence": dict(stage10_baseline_evidence),
        "tamper_audit": dict(tamper_audit),
        "detached_pass_json_authoritative": False,
        "historical_receipt_authoritative": False,
        "deterministic_difference_fail_closed": True,
        "failure_state": "OUT_OF_MODEL",
    }


def _expected_paths(actual: Mapping[str, bytes], config: Stage11ToyConfig) -> set[str]:
    source_files = _prefixed(actual, "source_stage10/")
    return {
        "package_manifest.json",
        *_ROOT_FILES,
        "trust/revocation_list.json",
        "trust/trust_anchor.json",
        *(f"{runner_id}/{name}" for runner_id in config.runner_ids for name in _RUNNER_FILES),
        *(f"source_stage10/{name}" for name in source_files),
    }


def run_stage11_toy_fixture(
    config_path: Path,
    source_stage10_package: Path,
    output_dir: Path,
) -> Stage11ToyRunResult:
    """Build a local two-identity Ed25519 exchange package; never claim two physical hosts."""

    if any(
        not isinstance(item, Path) for item in (config_path, source_stage10_package, output_dir)
    ):
        raise TypeError("all Stage 11 paths must be pathlib.Path values")
    config_root = config_path.resolve(strict=True)
    source_root = source_stage10_package.resolve(strict=True)
    target = output_dir.resolve(strict=False)
    _paths_nonoverlapping((config_root, source_root, target))
    if target.exists():
        raise FileExistsError(f"refusing to reuse a Stage 11 output directory: {target}")
    config = load_stage11_toy_config(config_root)
    source = _stage10_material(source_root, config)
    trust_anchor = build_fixture_trust_anchor(config.trusted_time_epoch)
    if canonical_sha256(trust_anchor) != config.trust_anchor_sha256:
        raise ValueError("fixture trust anchor differs from the preregistered Stage 11 anchor")
    dependency_lock = capture_stage11_dependency_lock(config.dependency_distributions)
    environment_lock = cast(Mapping[str, object], source["environment_lock"])
    runtime_lock = capture_stage11_runtime_lock(
        stage10_environment_lock=environment_lock,
        dependency_lock_sha256=cast(str, dependency_lock["lock_sha256"]),
    )
    runners: dict[str, dict[str, object]] = {}
    private_keys: dict[str, tuple[Ed25519PrivateKey, Ed25519PrivateKey]] = {}
    for runner_id in config.runner_ids:
        artifacts, old_private, current_private = _runner_artifacts(
            config=config,
            runner_id=runner_id,
            source=source,
            runtime_lock=runtime_lock,
            dependency_lock=dependency_lock,
        )
        runners[runner_id] = artifacts
        private_keys[runner_id] = (old_private, current_private)
    old_serials = [
        cast(
            str,
            _certificate_payload(cast(Mapping[str, object], runners[name]["old_certificate"]))[
                "serial"
            ],
        )
        for name in config.runner_ids
    ]
    revocation_list = build_fixture_revocation_list(
        revoked_certificates=[
            (serial, "SUPERSEDED", config.trusted_time_epoch - 900) for serial in old_serials
        ],
        trusted_time_epoch=config.trusted_time_epoch,
    )
    ensure_independent_runner_keys(
        cast(Mapping[str, object], runners[config.runner_ids[0]]["current_certificate"]),
        cast(Mapping[str, object], runners[config.runner_ids[1]]["current_certificate"]),
    )
    replay_counts = {name: 0 for name in config.runner_ids}

    def replay(runner_id: str) -> Stage11RunnerAuthority:
        replay_counts[runner_id] += 1
        payload = _verify_runner(
            config=config,
            runner_id=runner_id,
            trust_anchor=trust_anchor,
            revocation_list=revocation_list,
            artifacts=runners[runner_id],
        )
        return _authority(
            payload, cast(Mapping[str, object], runners[runner_id]["current_certificate"])
        )

    capability_a = issue_synthetic_stage10_host_capability(
        lambda: replay(config.runner_ids[0]), ttl_seconds=60
    )
    capability_b = issue_synthetic_stage10_host_capability(
        lambda: replay(config.runner_ids[1]), ttl_seconds=60
    )
    baseline_comparison = consume_stage11_host_pair(
        capability_a,
        capability_b,
        lambda left, right: compare_verified_envelope_payloads(
            cast(Mapping[str, object], runners[left.runner_id]["expected_payload"]),
            cast(Mapping[str, object], runners[right.runner_id]["expected_payload"]),
        ),
    )
    if baseline_comparison.get("status") != "SUCCESS":
        raise ValueError("Stage 11 frozen baseline comparison failed closed")
    reuse_rejected = False
    try:
        consume_stage11_host_pair(capability_a, capability_b, lambda left, right: (left, right))
    except Stage11CapabilityError:
        reuse_rejected = True
    if not reuse_rejected:
        raise AssertionError("Stage 11 host capabilities became reusable")
    tamper_audit = _tamper_audit(
        config=config,
        trust_anchor=trust_anchor,
        revocation_list=revocation_list,
        runners=runners,
        private_keys=private_keys,
    )
    if tamper_audit["status"] != "SUCCESS":
        raise ValueError("Stage 11 tamper audit did not reject every preregistered scenario")
    source_authority = _source_authority(config, source)
    runner_key_ids = cast(
        tuple[str, str],
        tuple(
            cast(
                str,
                _certificate_payload(
                    cast(Mapping[str, object], runners[name]["current_certificate"])
                )["key_id"],
            )
            for name in config.runner_ids
        ),
    )
    lifecycle = {
        "schema_version": STAGE11_SCHEMA_VERSION,
        "issued_count": 2,
        "consumed_count": 2,
        "second_pair_consumption_rejected": reuse_rejected,
        "verifier_replay_count": sum(replay_counts.values()),
        "process_bound": True,
        "one_shot": True,
        "non_serializable": True,
        "logical_runner_identity_count": 2,
        "physical_host_count": 1,
        "independent_runner_process_count": 0,
        "authorizes_real_data": False,
        "authorizes_physical_root": False,
        "runner_a": host_capability_snapshot(capability_a),
        "runner_b": host_capability_snapshot(capability_b),
    }
    stage10_baseline = cast(Mapping[str, object], source["baseline_projection"])
    audit = _audit(
        config=config,
        source_authority=source_authority,
        runtime_lock=runtime_lock,
        dependency_lock=dependency_lock,
        trust_anchor_sha256=config.trust_anchor_sha256,
        runner_key_ids=runner_key_ids,
        baseline_comparison=baseline_comparison,
        tamper_audit=tamper_audit,
        capability_lifecycle=lifecycle,
        stage10_baseline_evidence=stage10_baseline,
    )
    summary = _summary(
        config=config,
        baseline_comparison=baseline_comparison,
        tamper_audit=tamper_audit,
        trust_anchor_sha256=config.trust_anchor_sha256,
    )
    payloads = {
        "audit.html": render_stage11_audit_html(audit).encode("utf-8"),
        "audit.json": render_stage11_audit_json(audit).encode("utf-8"),
        "baseline_comparison.json": _pretty_json_bytes(baseline_comparison),
        "config_snapshot.json": _pretty_json_bytes(dict(config.raw)),
        "dependency_lock.json": _pretty_json_bytes(dependency_lock),
        "runtime_lock.json": _pretty_json_bytes(runtime_lock),
        "source_authority.json": _pretty_json_bytes(source_authority),
        "summary.json": _pretty_json_bytes(summary),
        "tamper_audit.json": _pretty_json_bytes(tamper_audit),
        "trust/revocation_list.json": _pretty_json_bytes(revocation_list),
        "trust/trust_anchor.json": _pretty_json_bytes(trust_anchor),
    }
    for runner_id in config.runner_ids:
        for artifact_name in _RUNNER_FILES:
            key = artifact_name.removesuffix(".json")
            payloads[f"{runner_id}/{artifact_name}"] = _pretty_json_bytes(runners[runner_id][key])
    for name, payload in cast(Mapping[str, bytes], source["files"]).items():
        payloads[f"source_stage10/{name}"] = payload
    for name, payload in payloads.items():
        write_bytes_fail_safe(target / name, payload)
    generated = _read_tree(target, label="Stage 11 generated package")
    manifest = _file_manifest(generated)
    write_bytes_fail_safe(target / "package_manifest.json", _pretty_json_bytes(manifest))
    actual = _read_tree(target, label="Stage 11 generated package")
    _verify_manifest(manifest, actual)
    result = verify_stage11_toy_package(target)
    if not result.verified:
        raise ValueError("Stage 11 generated package did not verify")
    return result


def verify_stage11_toy_package(output_dir: Path) -> Stage11ToyRunResult:
    """Verify the embedded Stage 10 package, Ed25519 chain, rotation, tamper audit, and manifest."""

    if not isinstance(output_dir, Path):
        raise TypeError("Stage 11 package root must be pathlib.Path")
    root = output_dir.resolve(strict=True)
    actual = _read_tree(root, label="Stage 11 package")
    manifest = _json_mapping(actual.get("package_manifest.json", b""), "Stage 11 manifest")
    _verify_manifest(manifest, actual)
    config = parse_stage11_toy_config(
        _decode_json(actual["config_snapshot.json"], "Stage 11 config")
    )
    if set(actual) != _expected_paths(actual, config):
        raise ValueError("Stage 11 package contains an unknown or missing path")
    source = _stage10_material(root / "source_stage10", config)
    trust_anchor = _json_mapping(actual["trust/trust_anchor.json"], "trust anchor")
    revocation_list = _json_mapping(actual["trust/revocation_list.json"], "revocation list")
    if canonical_sha256(trust_anchor) != config.trust_anchor_sha256:
        raise ValueError("Stage 11 trust anchor differs from preregistration")
    dependency_lock = _json_mapping(actual["dependency_lock.json"], "dependency lock")
    runtime_lock = _json_mapping(actual["runtime_lock.json"], "runtime lock")
    if (
        dependency_lock.get("lock_kind") != STAGE11_DEPENDENCY_LOCK_KIND
        or dependency_lock.get("complete_wheel_archive_hashes") is not False
        or dependency_lock.get("production_cross_host_ready") is not False
        or dependency_lock.get("lock_sha256")
        != canonical_sha256(
            {name: value for name, value in dependency_lock.items() if name != "lock_sha256"}
        )
        or runtime_lock.get("lock_kind") != STAGE11_RUNTIME_LOCK_KIND
        or runtime_lock.get("production_hardware_attestation") is not False
        or runtime_lock.get("runtime_lock_sha256")
        != canonical_sha256(
            {name: value for name, value in runtime_lock.items() if name != "runtime_lock_sha256"}
        )
    ):
        raise ValueError("Stage 11 dependency/runtime lock does not self-verify")
    runners: dict[str, dict[str, object]] = {}
    for runner_id in config.runner_ids:
        artifacts = {
            artifact_name.removesuffix(".json"): _decode_json(
                actual[f"{runner_id}/{artifact_name}"], f"{runner_id} {artifact_name}"
            )
            for artifact_name in _RUNNER_FILES
        }
        expected_payload = _envelope_payload(
            config=config,
            runner_id=runner_id,
            source=source,
            runtime_lock=runtime_lock,
            dependency_lock=dependency_lock,
        )
        envelope = artifacts["envelope"]
        if not isinstance(envelope, Mapping):
            raise ValueError("Stage 11 runner envelope is not an object")
        stored_payload = envelope.get("payload")
        if not isinstance(stored_payload, Mapping):
            raise ValueError("Stage 11 runner payload is missing")
        expected_payload["producer_process_id"] = stored_payload.get("producer_process_id")
        artifacts["expected_payload"] = expected_payload
        runners[runner_id] = artifacts
    ensure_independent_runner_keys(
        cast(Mapping[str, object], runners[config.runner_ids[0]]["current_certificate"]),
        cast(Mapping[str, object], runners[config.runner_ids[1]]["current_certificate"]),
    )
    verified_payloads = [
        _verify_runner(
            config=config,
            runner_id=runner_id,
            trust_anchor=trust_anchor,
            revocation_list=revocation_list,
            artifacts=runners[runner_id],
        )
        for runner_id in config.runner_ids
    ]
    baseline_comparison = compare_verified_envelope_payloads(
        verified_payloads[0], verified_payloads[1]
    )
    if baseline_comparison.get("status") != "SUCCESS":
        raise ValueError("Stage 11 frozen baseline comparison fails closed")
    if actual["baseline_comparison.json"] != _pretty_json_bytes(baseline_comparison):
        raise ValueError("Stage 11 baseline comparison does not regenerate")
    private_keys = {
        runner_id: (
            fixture_private_key(f"{runner_id}:old"),
            fixture_private_key(f"{runner_id}:current"),
        )
        for runner_id in config.runner_ids
    }
    tamper_audit = _tamper_audit(
        config=config,
        trust_anchor=trust_anchor,
        revocation_list=revocation_list,
        runners=runners,
        private_keys=private_keys,
    )
    if actual["tamper_audit.json"] != _pretty_json_bytes(tamper_audit):
        raise ValueError("Stage 11 tamper audit does not regenerate")
    source_authority = _source_authority(config, source)
    if actual["source_authority.json"] != _pretty_json_bytes(source_authority):
        raise ValueError("Stage 11 source authority does not regenerate")
    runner_key_ids = cast(
        tuple[str, str],
        tuple(
            cast(
                str,
                _certificate_payload(
                    cast(Mapping[str, object], runners[name]["current_certificate"])
                )["key_id"],
            )
            for name in config.runner_ids
        ),
    )
    summary = _summary(
        config=config,
        baseline_comparison=baseline_comparison,
        tamper_audit=tamper_audit,
        trust_anchor_sha256=config.trust_anchor_sha256,
    )
    if actual["summary.json"] != _pretty_json_bytes(summary):
        raise ValueError("Stage 11 summary does not regenerate")
    audit = _json_mapping(actual["audit.json"], "Stage 11 audit")
    for name, expected in {
        "status": "SUCCESS",
        "physical_host_count": 1,
        "cross_physical_host_tested": False,
        "production_authentication": False,
        "private_keys_written_to_artifacts": False,
        "dataset_root_probed": False,
        "real_data_read_count": 0,
        "baseline_comparison": baseline_comparison,
        "tamper_audit": tamper_audit,
        "source_authority": source_authority,
        "runtime_lock": runtime_lock,
        "dependency_lock": dependency_lock,
        "stage10_baseline_evidence": source["baseline_projection"],
    }.items():
        if audit.get(name) != expected:
            raise ValueError(f"Stage 11 audit field does not regenerate: {name}")
    trust_policy = audit.get("trust_policy")
    if not isinstance(trust_policy, dict) or trust_policy.get("runner_leaf_key_ids") != list(
        runner_key_ids
    ):
        raise ValueError("Stage 11 audit trust policy does not bind runner keys")
    lifecycle = audit.get("capability_lifecycle")
    if not isinstance(lifecycle, dict) or any(
        lifecycle.get(name) != value
        for name, value in {
            "issued_count": 2,
            "consumed_count": 2,
            "second_pair_consumption_rejected": True,
            "process_bound": True,
            "one_shot": True,
            "non_serializable": True,
            "physical_host_count": 1,
            "independent_runner_process_count": 0,
            "authorizes_real_data": False,
            "authorizes_physical_root": False,
        }.items()
    ):
        raise ValueError("Stage 11 capability lifecycle is invalid")
    if actual["audit.html"] != render_stage11_audit_html(audit).encode("utf-8"):
        raise ValueError("Stage 11 HTML audit does not regenerate")
    package_sha = manifest.get("package_sha256")
    if not isinstance(package_sha, str):
        raise ValueError("Stage 11 package hash is missing")
    return Stage11ToyRunResult(
        output_dir=output_dir,
        package_sha256=package_sha,
        package_file_count=len(actual),
        source_stage10_package_sha256=config.source_stage10_package_sha256,
        source_stage10_package_file_count=config.source_stage10_package_file_count,
        trust_anchor_sha256=config.trust_anchor_sha256,
        runner_key_ids=runner_key_ids,
        tamper_scenario_count=cast(int, tamper_audit["scenario_count"]),
        tamper_rejection_count=cast(int, tamper_audit["rejection_count"]),
        baseline_status=cast(str, baseline_comparison["status"]),
        status="SUCCESS",
        verified=True,
    )
