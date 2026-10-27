"""Stage 11 Ed25519 trust, rotation, revocation, and frozen-host primitives."""

from __future__ import annotations

import base64
import hashlib
import importlib.metadata as metadata
import json
import locale
import os
import platform
import sys
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Final, cast

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.backends.openssl.backend import backend as openssl_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

STAGE11_SCHEMA_VERSION: Final = "11.0.0"
STAGE11_SIGNATURE_ALGORITHM: Final = "Ed25519"
STAGE11_AUTHENTICATION_SCOPE: Final = "LOCAL_ASYMMETRIC_FIXTURE_ONLY"
STAGE11_TRUST_SCOPE: Final = "SYNTHETIC_FIXTURE_TRUST_ANCHOR"
STAGE11_RUNTIME_LOCK_KIND: Final = "installed-runtime-evidence-v1"
STAGE11_DEPENDENCY_LOCK_KIND: Final = "installed-distribution-metadata-v1"
_FIXTURE_KEY_NAMESPACE: Final = b"retinaentail-stage11-public-fixture-key-v1"
_SHA256_HEX = frozenset("0123456789abcdef")


class Stage11VerificationCode(StrEnum):
    MALFORMED_ARTIFACT = "MALFORMED_ARTIFACT"
    UNTRUSTED_ROOT = "UNTRUSTED_ROOT"
    TRUST_ANCHOR_SIGNATURE_INVALID = "TRUST_ANCHOR_SIGNATURE_INVALID"
    TRUST_ANCHOR_TIME_INVALID = "TRUST_ANCHOR_TIME_INVALID"
    CERTIFICATE_SIGNATURE_INVALID = "CERTIFICATE_SIGNATURE_INVALID"
    CERTIFICATE_TIME_INVALID = "CERTIFICATE_TIME_INVALID"
    CERTIFICATE_REVOKED = "CERTIFICATE_REVOKED"
    ROTATION_SIGNATURE_INVALID = "ROTATION_SIGNATURE_INVALID"
    ROTATION_INVALID = "ROTATION_INVALID"
    REVOCATION_SIGNATURE_INVALID = "REVOCATION_SIGNATURE_INVALID"
    REVOCATION_STATUS_STALE = "REVOCATION_STATUS_STALE"
    ENVELOPE_SIGNATURE_INVALID = "ENVELOPE_SIGNATURE_INVALID"
    ENVELOPE_POLICY_INVALID = "ENVELOPE_POLICY_INVALID"
    SHARED_RUNNER_KEY_REJECTED = "SHARED_RUNNER_KEY_REJECTED"
    SESSION_NONCE_REPLAYED = "SESSION_NONCE_REPLAYED"
    FROZEN_CONTRACT_MISMATCH = "FROZEN_CONTRACT_MISMATCH"


class Stage11VerificationError(ValueError):
    def __init__(self, code: Stage11VerificationCode, message: str) -> None:
        self.code = code
        super().__init__(f"{code.value}: {message}")


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strict_mapping(value: object, *, fields: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise Stage11VerificationError(
            Stage11VerificationCode.MALFORMED_ARTIFACT,
            f"{label} fields are incomplete or unknown",
        )
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or any(c in value for c in "\r\n\x00"):
        raise Stage11VerificationError(
            Stage11VerificationCode.MALFORMED_ARTIFACT, f"{label} is invalid"
        )
    return value


def _digest(value: object, label: str) -> str:
    text = _text(value, label)
    if len(text) != 64 or any(character not in _SHA256_HEX for character in text):
        raise Stage11VerificationError(
            Stage11VerificationCode.MALFORMED_ARTIFACT, f"{label} is not SHA-256"
        )
    return text


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Stage11VerificationError(
            Stage11VerificationCode.MALFORMED_ARTIFACT, f"{label} is not an integer"
        )
    return value


def _positive_integer(value: object, label: str) -> int:
    result = _integer(value, label)
    if result <= 0:
        raise Stage11VerificationError(
            Stage11VerificationCode.MALFORMED_ARTIFACT, f"{label} must be positive"
        )
    return result


def _b64(payload: bytes) -> str:
    return base64.b64encode(payload).decode("ascii")


def _unb64(value: object, label: str) -> bytes:
    text = _text(value, label)
    try:
        return base64.b64decode(text.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as error:
        raise Stage11VerificationError(
            Stage11VerificationCode.MALFORMED_ARTIFACT, f"{label} is not canonical base64"
        ) from error


def fixture_private_key(label: str) -> Ed25519PrivateKey:
    """Derive a public, deterministic fixture key; never use this for production trust."""

    checked = _text(label, "fixture key label")
    seed = hashlib.sha256(_FIXTURE_KEY_NAMESPACE + b":" + checked.encode("utf-8")).digest()
    return Ed25519PrivateKey.from_private_bytes(seed)


def public_key_raw(key: Ed25519PublicKey) -> bytes:
    return key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def key_id(key: Ed25519PublicKey) -> str:
    return sha256_bytes(public_key_raw(key))


def _public_from_b64(value: object, label: str) -> Ed25519PublicKey:
    payload = _unb64(value, label)
    if len(payload) != 32:
        raise Stage11VerificationError(
            Stage11VerificationCode.MALFORMED_ARTIFACT,
            f"{label} must contain one raw Ed25519 public key",
        )
    return Ed25519PublicKey.from_public_bytes(payload)


def _signed_object(payload: Mapping[str, object], key: Ed25519PrivateKey) -> dict[str, object]:
    return {
        "payload": dict(payload),
        "signature_b64": _b64(key.sign(canonical_json_bytes(payload))),
    }


def _verify_signature(
    public_key: Ed25519PublicKey,
    signature: object,
    payload: Mapping[str, object],
    *,
    code: Stage11VerificationCode,
    label: str,
) -> None:
    try:
        public_key.verify(_unb64(signature, f"{label} signature"), canonical_json_bytes(payload))
    except InvalidSignature as error:
        raise Stage11VerificationError(code, f"{label} signature does not verify") from error


def build_fixture_trust_anchor(trusted_time_epoch: int) -> dict[str, object]:
    trusted_time = _positive_integer(trusted_time_epoch, "trusted time")
    private = fixture_private_key("root")
    public = private.public_key()
    payload = {
        "schema_version": STAGE11_SCHEMA_VERSION,
        "artifact_type": "TRUST_ANCHOR",
        "algorithm": STAGE11_SIGNATURE_ALGORITHM,
        "key_id": key_id(public),
        "public_key_b64": _b64(public_key_raw(public)),
        "trust_scope": STAGE11_TRUST_SCOPE,
        "not_before_epoch": trusted_time - 86_400,
        "not_after_epoch": trusted_time + 315_360_000,
        "fixture_private_derivation_public": True,
        "production_authentication": False,
    }
    return _signed_object(payload, private)


def verify_trust_anchor(
    anchor: object,
    *,
    expected_anchor_sha256: str,
    trusted_time_epoch: int,
) -> Ed25519PublicKey:
    envelope = _strict_mapping(
        anchor,
        fields={"payload", "signature_b64"},
        label="trust anchor",
    )
    if canonical_sha256(envelope) != _digest(expected_anchor_sha256, "expected trust anchor"):
        raise Stage11VerificationError(
            Stage11VerificationCode.UNTRUSTED_ROOT,
            "trust anchor is not the preregistered pinned object",
        )
    payload = _strict_mapping(
        envelope["payload"],
        fields={
            "schema_version",
            "artifact_type",
            "algorithm",
            "key_id",
            "public_key_b64",
            "trust_scope",
            "not_before_epoch",
            "not_after_epoch",
            "fixture_private_derivation_public",
            "production_authentication",
        },
        label="trust anchor payload",
    )
    public = _public_from_b64(payload["public_key_b64"], "trust anchor public key")
    if (
        payload["schema_version"] != STAGE11_SCHEMA_VERSION
        or payload["artifact_type"] != "TRUST_ANCHOR"
        or payload["algorithm"] != STAGE11_SIGNATURE_ALGORITHM
        or payload["trust_scope"] != STAGE11_TRUST_SCOPE
        or payload["key_id"] != key_id(public)
        or payload["fixture_private_derivation_public"] is not True
        or payload["production_authentication"] is not False
    ):
        raise Stage11VerificationError(
            Stage11VerificationCode.UNTRUSTED_ROOT, "trust anchor policy is invalid"
        )
    _verify_signature(
        public,
        envelope["signature_b64"],
        payload,
        code=Stage11VerificationCode.TRUST_ANCHOR_SIGNATURE_INVALID,
        label="trust anchor",
    )
    now = _positive_integer(trusted_time_epoch, "trusted time")
    if (
        not _integer(payload["not_before_epoch"], "root not-before")
        <= now
        <= _integer(payload["not_after_epoch"], "root not-after")
    ):
        raise Stage11VerificationError(
            Stage11VerificationCode.TRUST_ANCHOR_TIME_INVALID,
            "trust anchor is outside its validity interval",
        )
    return public


def build_fixture_certificate(
    *,
    subject: str,
    generation: int,
    public_key: Ed25519PublicKey,
    previous_key_id: str | None,
    trusted_time_epoch: int,
) -> dict[str, object]:
    checked_subject = _text(subject, "certificate subject")
    checked_generation = _positive_integer(generation, "certificate generation")
    now = _positive_integer(trusted_time_epoch, "trusted time")
    current_key_id = key_id(public_key)
    previous = None if previous_key_id is None else _digest(previous_key_id, "previous key")
    if (checked_generation == 1) != (previous is None):
        raise ValueError("generation one has no predecessor; later generations require one")
    serial = canonical_sha256(
        {
            "subject": checked_subject,
            "generation": checked_generation,
            "key_id": current_key_id,
        }
    )
    payload = {
        "schema_version": STAGE11_SCHEMA_VERSION,
        "artifact_type": "RUNNER_CERTIFICATE",
        "algorithm": STAGE11_SIGNATURE_ALGORITHM,
        "serial": serial,
        "subject": checked_subject,
        "issuer_key_id": key_id(fixture_private_key("root").public_key()),
        "public_key_b64": _b64(public_key_raw(public_key)),
        "key_id": current_key_id,
        "generation": checked_generation,
        "previous_key_id": previous,
        "usage": "RUNNER_ARTIFACT_SIGNING",
        "scope": "SYNTHETIC_ONLY",
        "not_before_epoch": now - (7_200 if checked_generation == 1 else 1_800),
        "not_after_epoch": now + (7_200 if checked_generation == 1 else 86_400),
        "production_authentication": False,
    }
    return _signed_object(payload, fixture_private_key("root"))


def verify_certificate(
    certificate: object,
    *,
    root_public_key: Ed25519PublicKey,
    expected_subject: str,
    trusted_time_epoch: int,
) -> dict[str, object]:
    envelope = _strict_mapping(
        certificate, fields={"payload", "signature_b64"}, label="runner certificate"
    )
    payload = _strict_mapping(
        envelope["payload"],
        fields={
            "schema_version",
            "artifact_type",
            "algorithm",
            "serial",
            "subject",
            "issuer_key_id",
            "public_key_b64",
            "key_id",
            "generation",
            "previous_key_id",
            "usage",
            "scope",
            "not_before_epoch",
            "not_after_epoch",
            "production_authentication",
        },
        label="runner certificate payload",
    )
    public = _public_from_b64(payload["public_key_b64"], "runner certificate public key")
    if (
        payload["schema_version"] != STAGE11_SCHEMA_VERSION
        or payload["artifact_type"] != "RUNNER_CERTIFICATE"
        or payload["algorithm"] != STAGE11_SIGNATURE_ALGORITHM
        or payload["subject"] != _text(expected_subject, "expected certificate subject")
        or payload["issuer_key_id"] != key_id(root_public_key)
        or payload["key_id"] != key_id(public)
        or payload["usage"] != "RUNNER_ARTIFACT_SIGNING"
        or payload["scope"] != "SYNTHETIC_ONLY"
        or payload["production_authentication"] is not False
    ):
        raise Stage11VerificationError(
            Stage11VerificationCode.MALFORMED_ARTIFACT, "runner certificate policy is invalid"
        )
    _digest(payload["serial"], "certificate serial")
    _positive_integer(payload["generation"], "certificate generation")
    previous = payload["previous_key_id"]
    if previous is not None:
        _digest(previous, "certificate previous key")
    _verify_signature(
        root_public_key,
        envelope["signature_b64"],
        payload,
        code=Stage11VerificationCode.CERTIFICATE_SIGNATURE_INVALID,
        label="runner certificate",
    )
    now = _positive_integer(trusted_time_epoch, "trusted time")
    if (
        not _integer(payload["not_before_epoch"], "certificate not-before")
        <= now
        <= _integer(payload["not_after_epoch"], "certificate not-after")
    ):
        raise Stage11VerificationError(
            Stage11VerificationCode.CERTIFICATE_TIME_INVALID,
            "runner certificate is outside its validity interval",
        )
    return payload


def build_fixture_rotation_proof(
    *,
    runner_id: str,
    old_certificate: Mapping[str, object],
    current_certificate: Mapping[str, object],
    old_private_key: Ed25519PrivateKey,
    source_stage10_package_sha256: str,
    trusted_time_epoch: int,
) -> dict[str, object]:
    old_payload = cast(Mapping[str, object], old_certificate["payload"])
    current_payload = cast(Mapping[str, object], current_certificate["payload"])
    payload = {
        "schema_version": STAGE11_SCHEMA_VERSION,
        "artifact_type": "KEY_ROTATION_PROOF",
        "runner_id": _text(runner_id, "runner id"),
        "old_certificate_serial": _digest(old_payload["serial"], "old certificate serial"),
        "new_certificate_serial": _digest(current_payload["serial"], "new certificate serial"),
        "old_key_id": _digest(old_payload["key_id"], "old key id"),
        "new_key_id": _digest(current_payload["key_id"], "new key id"),
        "old_generation": _positive_integer(old_payload["generation"], "old generation"),
        "new_generation": _positive_integer(current_payload["generation"], "new generation"),
        "effective_epoch": _positive_integer(trusted_time_epoch, "trusted time") - 900,
        "rotation_nonce_sha256": canonical_sha256(
            {"runner_id": runner_id, "source": source_stage10_package_sha256, "kind": "rotation"}
        ),
        "source_stage10_package_sha256": _digest(
            source_stage10_package_sha256, "source Stage 10 package"
        ),
    }
    return {
        "payload": payload,
        "old_key_signature_b64": _b64(old_private_key.sign(canonical_json_bytes(payload))),
        "root_signature_b64": _b64(fixture_private_key("root").sign(canonical_json_bytes(payload))),
    }


def verify_rotation_proof(
    rotation: object,
    *,
    old_certificate_payload: Mapping[str, object],
    current_certificate_payload: Mapping[str, object],
    root_public_key: Ed25519PublicKey,
    expected_runner_id: str,
    expected_source_stage10_package_sha256: str,
    trusted_time_epoch: int,
) -> dict[str, object]:
    proof = _strict_mapping(
        rotation,
        fields={"payload", "old_key_signature_b64", "root_signature_b64"},
        label="rotation proof",
    )
    payload = _strict_mapping(
        proof["payload"],
        fields={
            "schema_version",
            "artifact_type",
            "runner_id",
            "old_certificate_serial",
            "new_certificate_serial",
            "old_key_id",
            "new_key_id",
            "old_generation",
            "new_generation",
            "effective_epoch",
            "rotation_nonce_sha256",
            "source_stage10_package_sha256",
        },
        label="rotation proof payload",
    )
    expected = {
        "schema_version": STAGE11_SCHEMA_VERSION,
        "artifact_type": "KEY_ROTATION_PROOF",
        "runner_id": expected_runner_id,
        "old_certificate_serial": old_certificate_payload["serial"],
        "new_certificate_serial": current_certificate_payload["serial"],
        "old_key_id": old_certificate_payload["key_id"],
        "new_key_id": current_certificate_payload["key_id"],
        "old_generation": old_certificate_payload["generation"],
        "new_generation": current_certificate_payload["generation"],
        "source_stage10_package_sha256": expected_source_stage10_package_sha256,
    }
    if any(payload.get(name) != value for name, value in expected.items()):
        raise Stage11VerificationError(
            Stage11VerificationCode.ROTATION_INVALID, "rotation proof conflicts with certificates"
        )
    if (
        _positive_integer(payload["new_generation"], "new generation")
        != _positive_integer(payload["old_generation"], "old generation") + 1
        or current_certificate_payload["previous_key_id"] != old_certificate_payload["key_id"]
        or _integer(payload["effective_epoch"], "rotation effective time")
        > _positive_integer(trusted_time_epoch, "trusted time")
    ):
        raise Stage11VerificationError(
            Stage11VerificationCode.ROTATION_INVALID, "rotation sequence or time is invalid"
        )
    _digest(payload["rotation_nonce_sha256"], "rotation nonce")
    old_public = _public_from_b64(
        old_certificate_payload["public_key_b64"], "old certificate public key"
    )
    _verify_signature(
        old_public,
        proof["old_key_signature_b64"],
        payload,
        code=Stage11VerificationCode.ROTATION_SIGNATURE_INVALID,
        label="old-key rotation proof",
    )
    _verify_signature(
        root_public_key,
        proof["root_signature_b64"],
        payload,
        code=Stage11VerificationCode.ROTATION_SIGNATURE_INVALID,
        label="root rotation proof",
    )
    return payload


def build_fixture_revocation_list(
    *,
    revoked_certificates: Sequence[tuple[str, str, int]],
    trusted_time_epoch: int,
) -> dict[str, object]:
    now = _positive_integer(trusted_time_epoch, "trusted time")
    entries = [
        {
            "certificate_serial": _digest(serial, "revoked certificate serial"),
            "reason": _text(reason, "revocation reason"),
            "revoked_at_epoch": _positive_integer(revoked_at, "revocation time"),
        }
        for serial, reason, revoked_at in sorted(revoked_certificates)
    ]
    payload = {
        "schema_version": STAGE11_SCHEMA_VERSION,
        "artifact_type": "REVOCATION_LIST",
        "issuer_key_id": key_id(fixture_private_key("root").public_key()),
        "issued_at_epoch": now - 600,
        "next_update_epoch": now + 3_600,
        "entries": entries,
        "scope": "SYNTHETIC_ONLY",
        "production_revocation_status": False,
    }
    return _signed_object(payload, fixture_private_key("root"))


def verify_revocation_list(
    revocation_list: object,
    *,
    root_public_key: Ed25519PublicKey,
    trusted_time_epoch: int,
) -> dict[str, tuple[str, int]]:
    envelope = _strict_mapping(
        revocation_list, fields={"payload", "signature_b64"}, label="revocation list"
    )
    payload = _strict_mapping(
        envelope["payload"],
        fields={
            "schema_version",
            "artifact_type",
            "issuer_key_id",
            "issued_at_epoch",
            "next_update_epoch",
            "entries",
            "scope",
            "production_revocation_status",
        },
        label="revocation list payload",
    )
    if (
        payload["schema_version"] != STAGE11_SCHEMA_VERSION
        or payload["artifact_type"] != "REVOCATION_LIST"
        or payload["issuer_key_id"] != key_id(root_public_key)
        or payload["scope"] != "SYNTHETIC_ONLY"
        or payload["production_revocation_status"] is not False
    ):
        raise Stage11VerificationError(
            Stage11VerificationCode.MALFORMED_ARTIFACT, "revocation policy is invalid"
        )
    _verify_signature(
        root_public_key,
        envelope["signature_b64"],
        payload,
        code=Stage11VerificationCode.REVOCATION_SIGNATURE_INVALID,
        label="revocation list",
    )
    now = _positive_integer(trusted_time_epoch, "trusted time")
    if (
        not _integer(payload["issued_at_epoch"], "revocation issued-at")
        <= now
        <= _integer(payload["next_update_epoch"], "revocation next-update")
    ):
        raise Stage11VerificationError(
            Stage11VerificationCode.REVOCATION_STATUS_STALE,
            "revocation status is not current at trusted time",
        )
    entries = payload["entries"]
    if not isinstance(entries, list):
        raise Stage11VerificationError(
            Stage11VerificationCode.MALFORMED_ARTIFACT, "revocation entries are not a list"
        )
    result: dict[str, tuple[str, int]] = {}
    canonical_entries: list[dict[str, object]] = []
    for raw in entries:
        entry = _strict_mapping(
            raw,
            fields={"certificate_serial", "reason", "revoked_at_epoch"},
            label="revocation entry",
        )
        serial = _digest(entry["certificate_serial"], "revoked certificate serial")
        if serial in result:
            raise Stage11VerificationError(
                Stage11VerificationCode.MALFORMED_ARTIFACT,
                "revocation list contains duplicate serials",
            )
        reason = _text(entry["reason"], "revocation reason")
        revoked_at = _positive_integer(entry["revoked_at_epoch"], "revocation time")
        result[serial] = (reason, revoked_at)
        canonical_entries.append(dict(entry))
    if canonical_entries != sorted(
        canonical_entries, key=lambda item: str(item["certificate_serial"])
    ):
        raise Stage11VerificationError(
            Stage11VerificationCode.MALFORMED_ARTIFACT, "revocation entries are not sorted"
        )
    return result


_ENVELOPE_PAYLOAD_FIELDS: Final = {
    "schema_version",
    "runner_id",
    "logical_host_id",
    "producer_process_id",
    "session_nonce_sha256",
    "dataset_id",
    "dataset_version",
    "scope",
    "split",
    "source_stage10_package_sha256",
    "source_stage10_package_file_count",
    "stage10_manifest_sha256",
    "stage10_audit_sha256",
    "stage10_summary_sha256",
    "stage10_environment_lock_sha256",
    "baseline_projection_sha256",
    "runtime_lock_sha256",
    "dependency_lock_sha256",
    "trust_anchor_sha256",
    "host_profile_sha256",
    "alpha",
    "positive_threshold",
    "negative_threshold",
    "positive_sample_count",
    "negative_sample_count",
    "config_hash",
    "data_hash",
    "model_hash",
    "payload_contains_data",
    "payload_contains_root",
    "payload_contains_private_key",
    "authorizes_real_data",
    "authorizes_physical_root",
    "detached_envelope_authoritative",
    "physical_host_count",
    "cross_physical_host_claim",
    "production_authentication",
}


def build_signed_envelope(
    payload: Mapping[str, object],
    *,
    current_certificate: Mapping[str, object],
    private_key: Ed25519PrivateKey,
) -> dict[str, object]:
    checked_payload = _strict_mapping(
        dict(payload), fields=_ENVELOPE_PAYLOAD_FIELDS, label="envelope"
    )
    certificate_payload = cast(Mapping[str, object], current_certificate["payload"])
    envelope = {
        "schema_version": STAGE11_SCHEMA_VERSION,
        "algorithm": STAGE11_SIGNATURE_ALGORITHM,
        "authentication_scope": STAGE11_AUTHENTICATION_SCOPE,
        "certificate_serial": certificate_payload["serial"],
        "key_id": certificate_payload["key_id"],
        "payload": checked_payload,
        "signature_b64": _b64(private_key.sign(canonical_json_bytes(checked_payload))),
    }
    return envelope


def verify_signed_envelope(
    envelope: object,
    *,
    current_certificate_payload: Mapping[str, object],
    expected_payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    signed = _strict_mapping(
        envelope,
        fields={
            "schema_version",
            "algorithm",
            "authentication_scope",
            "certificate_serial",
            "key_id",
            "payload",
            "signature_b64",
        },
        label="signed envelope",
    )
    payload = _strict_mapping(
        signed["payload"], fields=_ENVELOPE_PAYLOAD_FIELDS, label="signed envelope payload"
    )
    if (
        signed["schema_version"] != STAGE11_SCHEMA_VERSION
        or signed["algorithm"] != STAGE11_SIGNATURE_ALGORITHM
        or signed["authentication_scope"] != STAGE11_AUTHENTICATION_SCOPE
        or signed["certificate_serial"] != current_certificate_payload["serial"]
        or signed["key_id"] != current_certificate_payload["key_id"]
    ):
        raise Stage11VerificationError(
            Stage11VerificationCode.ENVELOPE_POLICY_INVALID,
            "envelope signer binding or algorithm is invalid",
        )
    if (
        payload["schema_version"] != STAGE11_SCHEMA_VERSION
        or payload["scope"] != "SYNTHETIC_ONLY"
        or payload["split"] != "test"
        or payload["payload_contains_data"] is not False
        or payload["payload_contains_root"] is not False
        or payload["payload_contains_private_key"] is not False
        or payload["authorizes_real_data"] is not False
        or payload["authorizes_physical_root"] is not False
        or payload["detached_envelope_authoritative"] is not False
        or payload["physical_host_count"] != 1
        or payload["cross_physical_host_claim"] is not False
        or payload["production_authentication"] is not False
    ):
        raise Stage11VerificationError(
            Stage11VerificationCode.ENVELOPE_POLICY_INVALID,
            "local fixture envelope exceeds its authority",
        )
    for name in (
        "session_nonce_sha256",
        "source_stage10_package_sha256",
        "stage10_manifest_sha256",
        "stage10_audit_sha256",
        "stage10_summary_sha256",
        "stage10_environment_lock_sha256",
        "baseline_projection_sha256",
        "runtime_lock_sha256",
        "dependency_lock_sha256",
        "trust_anchor_sha256",
        "host_profile_sha256",
        "config_hash",
        "data_hash",
        "model_hash",
    ):
        _digest(payload[name], name)
    _positive_integer(payload["producer_process_id"], "producer process id")
    _positive_integer(payload["source_stage10_package_file_count"], "Stage 10 file count")
    _positive_integer(payload["positive_sample_count"], "positive sample count")
    _positive_integer(payload["negative_sample_count"], "negative sample count")
    if expected_payload is not None and payload != dict(expected_payload):
        raise Stage11VerificationError(
            Stage11VerificationCode.FROZEN_CONTRACT_MISMATCH,
            "envelope payload differs from the reconstructed frozen contract",
        )
    public = _public_from_b64(
        current_certificate_payload["public_key_b64"], "current runner public key"
    )
    _verify_signature(
        public,
        signed["signature_b64"],
        payload,
        code=Stage11VerificationCode.ENVELOPE_SIGNATURE_INVALID,
        label="runner envelope",
    )
    return payload


def verify_runner_bundle(
    *,
    runner_id: str,
    trust_anchor: object,
    expected_trust_anchor_sha256: str,
    revocation_list: object,
    old_certificate: object,
    current_certificate: object,
    rotation_proof: object,
    envelope: object,
    trusted_time_epoch: int,
    expected_source_stage10_package_sha256: str,
    expected_payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    root_public = verify_trust_anchor(
        trust_anchor,
        expected_anchor_sha256=expected_trust_anchor_sha256,
        trusted_time_epoch=trusted_time_epoch,
    )
    old_payload = verify_certificate(
        old_certificate,
        root_public_key=root_public,
        expected_subject=runner_id,
        trusted_time_epoch=trusted_time_epoch,
    )
    current_payload = verify_certificate(
        current_certificate,
        root_public_key=root_public,
        expected_subject=runner_id,
        trusted_time_epoch=trusted_time_epoch,
    )
    if old_payload["generation"] != 1 or current_payload["generation"] != 2:
        raise Stage11VerificationError(
            Stage11VerificationCode.ROTATION_INVALID, "fixture requires generation 1 to 2"
        )
    rotation_payload = verify_rotation_proof(
        rotation_proof,
        old_certificate_payload=old_payload,
        current_certificate_payload=current_payload,
        root_public_key=root_public,
        expected_runner_id=runner_id,
        expected_source_stage10_package_sha256=expected_source_stage10_package_sha256,
        trusted_time_epoch=trusted_time_epoch,
    )
    revoked = verify_revocation_list(
        revocation_list,
        root_public_key=root_public,
        trusted_time_epoch=trusted_time_epoch,
    )
    old_serial = cast(str, old_payload["serial"])
    current_serial = cast(str, current_payload["serial"])
    old_status = revoked.get(old_serial)
    if (
        old_status is None
        or old_status[0] != "SUPERSEDED"
        or old_status[1] > _integer(rotation_payload["effective_epoch"], "rotation effective time")
    ):
        raise Stage11VerificationError(
            Stage11VerificationCode.ROTATION_INVALID,
            "old certificate is not superseded by the current rotation",
        )
    if current_serial in revoked:
        raise Stage11VerificationError(
            Stage11VerificationCode.CERTIFICATE_REVOKED,
            "current runner signing certificate is revoked",
        )
    return verify_signed_envelope(
        envelope,
        current_certificate_payload=current_payload,
        expected_payload=expected_payload,
    )


def compare_verified_envelope_payloads(
    left: Mapping[str, object], right: Mapping[str, object]
) -> dict[str, object]:
    differences: list[str] = []
    if left.get("runner_id") == right.get("runner_id"):
        differences.append("RUNNER_ID_NOT_INDEPENDENT")
    if left.get("session_nonce_sha256") == right.get("session_nonce_sha256"):
        differences.append(Stage11VerificationCode.SESSION_NONCE_REPLAYED.value)
    dynamic = {"runner_id", "logical_host_id", "session_nonce_sha256"}
    for name in sorted(_ENVELOPE_PAYLOAD_FIELDS - dynamic):
        if left.get(name) != right.get(name):
            differences.append(f"FROZEN_FIELD_MISMATCH:{name}")
    return {
        "schema_version": STAGE11_SCHEMA_VERSION,
        "status": "SUCCESS" if not differences else "FAIL_CLOSED",
        "differences": differences,
        "deterministic_difference_fail_closed": True,
        "effective_state_on_difference": "OUT_OF_MODEL",
        "compared_frozen_field_count": len(_ENVELOPE_PAYLOAD_FIELDS - dynamic),
    }


def ensure_independent_runner_keys(
    left_current_certificate: Mapping[str, object],
    right_current_certificate: Mapping[str, object],
) -> None:
    left = cast(Mapping[str, object], left_current_certificate["payload"])
    right = cast(Mapping[str, object], right_current_certificate["payload"])
    if left.get("key_id") == right.get("key_id"):
        raise Stage11VerificationError(
            Stage11VerificationCode.SHARED_RUNNER_KEY_REJECTED,
            "runner leaf signing keys must be distinct",
        )


def _distribution_file_sha256(distribution: metadata.Distribution, suffix: str) -> str | None:
    matches = [
        item for item in (distribution.files or ()) if str(item).replace("\\", "/").endswith(suffix)
    ]
    if len(matches) != 1:
        return None
    path = Path(str(distribution.locate_file(matches[0])))
    try:
        payload = path.read_bytes()
    except OSError:
        return None
    return sha256_bytes(payload)


def capture_stage11_dependency_lock(distribution_names: Sequence[str]) -> dict[str, object]:
    names = tuple(sorted({_text(name, "distribution name") for name in distribution_names}))
    entries: list[dict[str, object]] = []
    for name in names:
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError as error:
            raise ValueError(f"required Stage 11 distribution is missing: {name}") from error
        entries.append(
            {
                "name": name,
                "version": distribution.version,
                "metadata_sha256": _distribution_file_sha256(distribution, ".dist-info/METADATA"),
                "record_sha256": _distribution_file_sha256(distribution, ".dist-info/RECORD"),
                "installed_file_count": len(distribution.files or ()),
                "wheel_archive_sha256": None,
            }
        )
    lock = {
        "schema_version": STAGE11_SCHEMA_VERSION,
        "lock_kind": STAGE11_DEPENDENCY_LOCK_KIND,
        "entries": entries,
        "complete_wheel_archive_hashes": False,
        "production_cross_host_ready": False,
        "contains_install_paths": False,
        "contains_credentials": False,
    }
    return {**lock, "lock_sha256": canonical_sha256(lock)}


def capture_stage11_runtime_lock(
    *,
    stage10_environment_lock: Mapping[str, object],
    dependency_lock_sha256: str,
) -> dict[str, object]:
    cpu_material = {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "processor_identifier": os.environ.get("PROCESSOR_IDENTIFIER", ""),
        "processor_architecture": os.environ.get("PROCESSOR_ARCHITECTURE", ""),
        "processor_level": os.environ.get("PROCESSOR_LEVEL", ""),
        "processor_revision": os.environ.get("PROCESSOR_REVISION", ""),
    }
    runtime = {
        "schema_version": STAGE11_SCHEMA_VERSION,
        "lock_kind": STAGE11_RUNTIME_LOCK_KIND,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "cpu_profile_sha256": canonical_sha256(cpu_material),
        "cpu_instruction_set_attested": False,
        "cpu_instruction_policy": "RECORDED_NOT_HARDWARE_ATTESTED",
        "cryptography_version": metadata.version("cryptography"),
        "openssl_version": openssl_backend.openssl_version_text(),
        "preferred_encoding": locale.getpreferredencoding(False),
        "filesystem_encoding": sys.getfilesystemencoding(),
        "stage10_environment_lock_sha256": canonical_sha256(stage10_environment_lock),
        "stage10_blas_configuration_sha256": stage10_environment_lock.get(
            "blas_configuration_sha256"
        ),
        "stage10_thread_policy": stage10_environment_lock.get("thread_policy"),
        "stage10_locale_name": stage10_environment_lock.get("locale_name"),
        "dependency_lock_sha256": _digest(dependency_lock_sha256, "dependency lock"),
        "contains_host_name": False,
        "contains_install_paths": False,
        "production_hardware_attestation": False,
    }
    return {**runtime, "runtime_lock_sha256": canonical_sha256(runtime)}
