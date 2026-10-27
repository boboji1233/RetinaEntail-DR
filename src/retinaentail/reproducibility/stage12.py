"""Stage 12 process-local random Ed25519 signer and recovery verification."""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping
from enum import StrEnum
from typing import Final, Protocol, cast

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

STAGE12_SCHEMA_VERSION: Final = "12.0.0"
STAGE12_ALGORITHM: Final = "Ed25519"
STAGE12_SCOPE: Final = "LOCAL_RANDOM_EPHEMERAL_PROCESS_ONLY"
_HEX = frozenset("0123456789abcdef")


class Stage12VerificationCode(StrEnum):
    MALFORMED_ARTIFACT = "MALFORMED_ARTIFACT"
    TRUST_ANCHOR_INVALID = "TRUST_ANCHOR_INVALID"
    CERTIFICATE_SIGNATURE_INVALID = "CERTIFICATE_SIGNATURE_INVALID"
    CERTIFICATE_TIME_INVALID = "CERTIFICATE_TIME_INVALID"
    CERTIFICATE_REVOKED = "CERTIFICATE_REVOKED"
    STATUS_SIGNATURE_INVALID = "STATUS_SIGNATURE_INVALID"
    STATUS_STALE = "STATUS_STALE"
    RECOVERY_PROOF_INVALID = "RECOVERY_PROOF_INVALID"
    ENVELOPE_SIGNATURE_INVALID = "ENVELOPE_SIGNATURE_INVALID"
    ENVELOPE_POLICY_INVALID = "ENVELOPE_POLICY_INVALID"
    REVOKED_SIGNER_REJECTED = "REVOKED_SIGNER_REJECTED"
    SHARED_RECOVERY_KEY_REJECTED = "SHARED_RECOVERY_KEY_REJECTED"
    PROCESS_ID_NOT_INDEPENDENT = "PROCESS_ID_NOT_INDEPENDENT"
    SESSION_NONCE_REPLAYED = "SESSION_NONCE_REPLAYED"
    FROZEN_CONTRACT_MISMATCH = "FROZEN_CONTRACT_MISMATCH"


class Stage12VerificationError(ValueError):
    def __init__(self, code: Stage12VerificationCode, message: str) -> None:
        self.code = code
        super().__init__(f"{code.value}: {message}")


class SignerAdapter(Protocol):
    @property
    def key_id(self) -> str: ...

    @property
    def public_key_b64(self) -> str: ...

    def sign(self, payload: bytes) -> str: ...


class InMemoryEd25519Signer:
    """Random process-local fixture signer; private key has no export method."""

    __slots__ = ("__key",)

    def __init__(self) -> None:
        self.__key = Ed25519PrivateKey.generate()

    @property
    def key_id(self) -> str:
        return sha256_bytes(_public_raw(self.__key.public_key()))

    @property
    def public_key_b64(self) -> str:
        return _b64(_public_raw(self.__key.public_key()))

    def sign(self, payload: bytes) -> str:
        return _b64(self.__key.sign(payload))


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _unb64(value: object, label: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise Stage12VerificationError(Stage12VerificationCode.MALFORMED_ARTIFACT, label)
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as error:
        raise Stage12VerificationError(
            Stage12VerificationCode.MALFORMED_ARTIFACT, f"{label} is not base64"
        ) from error


def _public_raw(key: Ed25519PublicKey) -> bytes:
    return key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def _public(value: object, label: str) -> Ed25519PublicKey:
    raw = _unb64(value, label)
    if len(raw) != 32:
        raise Stage12VerificationError(
            Stage12VerificationCode.MALFORMED_ARTIFACT, f"{label} is not an Ed25519 key"
        )
    return Ed25519PublicKey.from_public_bytes(raw)


def _mapping(value: object, fields: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise Stage12VerificationError(
            Stage12VerificationCode.MALFORMED_ARTIFACT, f"{label} fields differ"
        )
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or any(c in value for c in "\r\n\x00"):
        raise Stage12VerificationError(
            Stage12VerificationCode.MALFORMED_ARTIFACT, f"{label} is invalid"
        )
    return value


def _digest(value: object, label: str) -> str:
    result = _text(value, label)
    if len(result) != 64 or any(character not in _HEX for character in result):
        raise Stage12VerificationError(
            Stage12VerificationCode.MALFORMED_ARTIFACT, f"{label} is not SHA-256"
        )
    return result


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Stage12VerificationError(
            Stage12VerificationCode.MALFORMED_ARTIFACT, f"{label} is not an integer"
        )
    return value


def _verify_signature(
    key: Ed25519PublicKey,
    signature: object,
    payload: Mapping[str, object],
    code: Stage12VerificationCode,
) -> None:
    try:
        key.verify(_unb64(signature, "signature"), canonical_json_bytes(payload))
    except InvalidSignature as error:
        raise Stage12VerificationError(code, "signature verification failed") from error


def _signed(payload: Mapping[str, object], signer: SignerAdapter) -> dict[str, object]:
    return {"payload": dict(payload), "signature_b64": signer.sign(canonical_json_bytes(payload))}


def build_ephemeral_trust_anchor(
    *, runner_id: str, trusted_time_epoch: int, root_signer: SignerAdapter
) -> dict[str, object]:
    payload = {
        "schema_version": STAGE12_SCHEMA_VERSION,
        "artifact_type": "EPHEMERAL_TRUST_ANCHOR",
        "algorithm": STAGE12_ALGORITHM,
        "runner_id": _text(runner_id, "runner id"),
        "key_id": root_signer.key_id,
        "public_key_b64": root_signer.public_key_b64,
        "trust_scope": STAGE12_SCOPE,
        "not_before_epoch": _integer(trusted_time_epoch, "trusted time") - 3_600,
        "not_after_epoch": trusted_time_epoch + 86_400,
        "random_process_local_private_key": True,
        "production_authentication": False,
    }
    return _signed(payload, root_signer)


def build_ephemeral_certificate(
    *,
    runner_id: str,
    generation: int,
    signer: SignerAdapter,
    issuer: SignerAdapter,
    previous_key_id: str | None,
    trusted_time_epoch: int,
) -> dict[str, object]:
    if generation not in (1, 2) or (generation == 1) != (previous_key_id is None):
        raise ValueError("Stage 12 fixture certificate generation is invalid")
    payload = {
        "schema_version": STAGE12_SCHEMA_VERSION,
        "artifact_type": "EPHEMERAL_RUNNER_CERTIFICATE",
        "algorithm": STAGE12_ALGORITHM,
        "runner_id": _text(runner_id, "runner id"),
        "generation": generation,
        "serial": canonical_sha256(
            {"runner_id": runner_id, "generation": generation, "key_id": signer.key_id}
        ),
        "issuer_key_id": issuer.key_id,
        "key_id": signer.key_id,
        "public_key_b64": signer.public_key_b64,
        "previous_key_id": previous_key_id,
        "usage": "LOCAL_ARTIFACT_SIGNING",
        "scope": STAGE12_SCOPE,
        "not_before_epoch": trusted_time_epoch - 3_600,
        "not_after_epoch": trusted_time_epoch + 86_400,
        "production_authentication": False,
    }
    return _signed(payload, issuer)


def build_recovery_proof(
    *,
    runner_id: str,
    old_certificate: Mapping[str, object],
    new_certificate: Mapping[str, object],
    root_signer: SignerAdapter,
    recovered_signer: SignerAdapter,
    compromise_epoch: int,
    recovery_epoch: int,
    source_stage11_package_sha256: str,
    frozen_contract_sha256: str,
) -> dict[str, object]:
    old = cast(Mapping[str, object], old_certificate["payload"])
    new = cast(Mapping[str, object], new_certificate["payload"])
    payload = {
        "schema_version": STAGE12_SCHEMA_VERSION,
        "artifact_type": "KEY_COMPROMISE_RECOVERY_PROOF",
        "runner_id": runner_id,
        "old_certificate_serial": old["serial"],
        "new_certificate_serial": new["serial"],
        "old_key_id": old["key_id"],
        "new_key_id": new["key_id"],
        "old_generation": 1,
        "new_generation": 2,
        "compromise_epoch": compromise_epoch,
        "recovery_epoch": recovery_epoch,
        "source_stage11_package_sha256": _digest(
            source_stage11_package_sha256, "source Stage 11 package"
        ),
        "frozen_contract_sha256": _digest(frozen_contract_sha256, "frozen contract"),
        "old_key_authorized_after_compromise": False,
    }
    encoded = canonical_json_bytes(payload)
    return {
        "payload": payload,
        "root_signature_b64": root_signer.sign(encoded),
        "new_key_proof_of_possession_b64": recovered_signer.sign(encoded),
    }


def build_local_status_staple(
    *,
    runner_id: str,
    old_certificate: Mapping[str, object],
    new_certificate: Mapping[str, object],
    root_signer: SignerAdapter,
    compromise_epoch: int,
    trusted_time_epoch: int,
) -> dict[str, object]:
    old = cast(Mapping[str, object], old_certificate["payload"])
    new = cast(Mapping[str, object], new_certificate["payload"])
    payload = {
        "schema_version": STAGE12_SCHEMA_VERSION,
        "artifact_type": "LOCAL_SIGNED_STATUS_STAPLE",
        "runner_id": runner_id,
        "issuer_key_id": root_signer.key_id,
        "produced_at_epoch": trusted_time_epoch - 60,
        "next_update_epoch": trusted_time_epoch + 3_600,
        "statuses": [
            {
                "certificate_serial": old["serial"],
                "generation": 1,
                "status": "REVOKED",
                "reason": "KEY_COMPROMISE",
                "effective_epoch": compromise_epoch,
            },
            {
                "certificate_serial": new["serial"],
                "generation": 2,
                "status": "GOOD",
                "reason": None,
                "effective_epoch": trusted_time_epoch - 60,
            },
        ],
        "scope": STAGE12_SCOPE,
        "rfc6960_ocsp_response": False,
        "production_status": False,
    }
    return _signed(payload, root_signer)


_ENVELOPE_FIELDS: Final = {
    "schema_version",
    "runner_id",
    "producer_process_id",
    "session_nonce_sha256",
    "phase",
    "signing_generation",
    "source_stage11_package_sha256",
    "source_stage11_package_file_count",
    "source_stage11_manifest_sha256",
    "source_stage11_audit_sha256",
    "source_stage11_summary_sha256",
    "frozen_contract_sha256",
    "alpha",
    "positive_threshold",
    "negative_threshold",
    "positive_sample_count",
    "negative_sample_count",
    "config_hash",
    "data_hash",
    "model_hash",
    "scope",
    "split",
    "payload_contains_data",
    "payload_contains_root",
    "payload_contains_private_key",
    "authorizes_real_data",
    "authorizes_physical_root",
    "production_authentication",
}


def build_signed_envelope(
    payload: Mapping[str, object], *, certificate: Mapping[str, object], signer: SignerAdapter
) -> dict[str, object]:
    checked = _mapping(dict(payload), _ENVELOPE_FIELDS, "envelope payload")
    cert = cast(Mapping[str, object], certificate["payload"])
    return {
        "schema_version": STAGE12_SCHEMA_VERSION,
        "algorithm": STAGE12_ALGORITHM,
        "authentication_scope": STAGE12_SCOPE,
        "certificate_serial": cert["serial"],
        "key_id": cert["key_id"],
        "payload": checked,
        "signature_b64": signer.sign(canonical_json_bytes(checked)),
    }


def verify_runner_recovery_bundle(
    bundle: object,
    *,
    expected_runner_id: str,
    expected_frozen_payload: Mapping[str, object],
    trusted_time_epoch: int,
) -> dict[str, object]:
    artifacts = _mapping(
        bundle,
        {
            "trust_anchor",
            "generation1_certificate",
            "generation2_certificate",
            "recovery_proof",
            "status_staple",
            "pre_compromise_envelope",
            "recovered_envelope",
            "receipt",
        },
        "runner bundle",
    )
    anchor = _mapping(artifacts["trust_anchor"], {"payload", "signature_b64"}, "anchor")
    anchor_payload = _mapping(
        anchor["payload"],
        {
            "schema_version",
            "artifact_type",
            "algorithm",
            "runner_id",
            "key_id",
            "public_key_b64",
            "trust_scope",
            "not_before_epoch",
            "not_after_epoch",
            "random_process_local_private_key",
            "production_authentication",
        },
        "anchor payload",
    )
    root = _public(anchor_payload.get("public_key_b64"), "anchor public key")
    if (
        anchor_payload.get("schema_version") != STAGE12_SCHEMA_VERSION
        or anchor_payload.get("artifact_type") != "EPHEMERAL_TRUST_ANCHOR"
        or anchor_payload.get("algorithm") != STAGE12_ALGORITHM
        or anchor_payload.get("runner_id") != expected_runner_id
        or anchor_payload.get("key_id") != sha256_bytes(_public_raw(root))
        or anchor_payload.get("trust_scope") != STAGE12_SCOPE
        or anchor_payload.get("random_process_local_private_key") is not True
        or anchor_payload.get("production_authentication") is not False
    ):
        raise Stage12VerificationError(
            Stage12VerificationCode.TRUST_ANCHOR_INVALID, "anchor policy is invalid"
        )
    _verify_signature(
        root,
        anchor["signature_b64"],
        anchor_payload,
        Stage12VerificationCode.TRUST_ANCHOR_INVALID,
    )
    now = _integer(trusted_time_epoch, "trusted time")
    if (
        not _integer(anchor_payload["not_before_epoch"], "anchor not-before")
        <= now
        <= _integer(anchor_payload["not_after_epoch"], "anchor not-after")
    ):
        raise Stage12VerificationError(
            Stage12VerificationCode.CERTIFICATE_TIME_INVALID, "anchor time is invalid"
        )

    certificates: list[dict[str, object]] = []
    for generation, name in ((1, "generation1_certificate"), (2, "generation2_certificate")):
        cert = _mapping(artifacts[name], {"payload", "signature_b64"}, "certificate")
        payload = _mapping(
            cert["payload"],
            {
                "schema_version",
                "artifact_type",
                "algorithm",
                "runner_id",
                "generation",
                "serial",
                "issuer_key_id",
                "key_id",
                "public_key_b64",
                "previous_key_id",
                "usage",
                "scope",
                "not_before_epoch",
                "not_after_epoch",
                "production_authentication",
            },
            "certificate payload",
        )
        public = _public(payload.get("public_key_b64"), "certificate public key")
        if (
            payload.get("schema_version") != STAGE12_SCHEMA_VERSION
            or payload.get("artifact_type") != "EPHEMERAL_RUNNER_CERTIFICATE"
            or payload.get("algorithm") != STAGE12_ALGORITHM
            or payload.get("runner_id") != expected_runner_id
            or payload.get("generation") != generation
            or payload.get("issuer_key_id") != anchor_payload["key_id"]
            or payload.get("key_id") != sha256_bytes(_public_raw(public))
            or payload.get("serial")
            != canonical_sha256(
                {
                    "runner_id": expected_runner_id,
                    "generation": generation,
                    "key_id": payload.get("key_id"),
                }
            )
            or payload.get("usage") != "LOCAL_ARTIFACT_SIGNING"
            or payload.get("scope") != STAGE12_SCOPE
            or payload.get("production_authentication") is not False
        ):
            raise Stage12VerificationError(
                Stage12VerificationCode.MALFORMED_ARTIFACT, "certificate policy is invalid"
            )
        _verify_signature(
            root,
            cert["signature_b64"],
            payload,
            Stage12VerificationCode.CERTIFICATE_SIGNATURE_INVALID,
        )
        if (
            not _integer(payload["not_before_epoch"], "cert not-before")
            <= now
            <= _integer(payload["not_after_epoch"], "cert not-after")
        ):
            raise Stage12VerificationError(
                Stage12VerificationCode.CERTIFICATE_TIME_INVALID, "certificate time is invalid"
            )
        certificates.append(payload)
    old, new = certificates
    if new.get("previous_key_id") != old.get("key_id") or old.get("previous_key_id") is not None:
        raise Stage12VerificationError(
            Stage12VerificationCode.RECOVERY_PROOF_INVALID, "certificate generations do not link"
        )

    status = _mapping(artifacts["status_staple"], {"payload", "signature_b64"}, "status")
    status_payload = _mapping(
        status["payload"],
        {
            "schema_version",
            "artifact_type",
            "runner_id",
            "issuer_key_id",
            "produced_at_epoch",
            "next_update_epoch",
            "statuses",
            "scope",
            "rfc6960_ocsp_response",
            "production_status",
        },
        "status payload",
    )
    _verify_signature(
        root,
        status["signature_b64"],
        status_payload,
        Stage12VerificationCode.STATUS_SIGNATURE_INVALID,
    )
    if (
        status_payload.get("schema_version") != STAGE12_SCHEMA_VERSION
        or status_payload.get("artifact_type") != "LOCAL_SIGNED_STATUS_STAPLE"
        or status_payload.get("runner_id") != expected_runner_id
        or status_payload.get("issuer_key_id") != anchor_payload["key_id"]
        or status_payload.get("scope") != STAGE12_SCOPE
        or status_payload.get("rfc6960_ocsp_response") is not False
        or status_payload.get("production_status") is not False
        or not _integer(status_payload["produced_at_epoch"], "status produced")
        <= now
        <= _integer(status_payload["next_update_epoch"], "status next update")
    ):
        raise Stage12VerificationError(Stage12VerificationCode.STATUS_STALE, "status is invalid")
    statuses = status_payload.get("statuses")
    if not isinstance(statuses, list) or len(statuses) != 2:
        raise Stage12VerificationError(
            Stage12VerificationCode.MALFORMED_ARTIFACT, "status entries are invalid"
        )
    checked_statuses = [
        _mapping(
            item,
            {"certificate_serial", "generation", "status", "reason", "effective_epoch"},
            "certificate status",
        )
        for item in statuses
    ]
    by_serial = {item["certificate_serial"]: item for item in checked_statuses}
    if (
        len(by_serial) != 2
        or by_serial.get(old["serial"], {}).get("status") != "REVOKED"
        or by_serial.get(old["serial"], {}).get("reason") != "KEY_COMPROMISE"
        or by_serial.get(old["serial"], {}).get("generation") != 1
        or by_serial.get(new["serial"], {}).get("status") != "GOOD"
        or by_serial.get(new["serial"], {}).get("reason") is not None
        or by_serial.get(new["serial"], {}).get("generation") != 2
    ):
        raise Stage12VerificationError(
            Stage12VerificationCode.CERTIFICATE_REVOKED, "recovery status is unsafe"
        )

    proof = _mapping(
        artifacts["recovery_proof"],
        {"payload", "root_signature_b64", "new_key_proof_of_possession_b64"},
        "recovery proof",
    )
    proof_payload = _mapping(
        proof["payload"],
        {
            "schema_version",
            "artifact_type",
            "runner_id",
            "old_certificate_serial",
            "new_certificate_serial",
            "old_key_id",
            "new_key_id",
            "old_generation",
            "new_generation",
            "compromise_epoch",
            "recovery_epoch",
            "source_stage11_package_sha256",
            "frozen_contract_sha256",
            "old_key_authorized_after_compromise",
        },
        "recovery proof payload",
    )
    new_public = _public(new["public_key_b64"], "new public key")
    _verify_signature(
        root,
        proof["root_signature_b64"],
        proof_payload,
        Stage12VerificationCode.RECOVERY_PROOF_INVALID,
    )
    _verify_signature(
        new_public,
        proof["new_key_proof_of_possession_b64"],
        proof_payload,
        Stage12VerificationCode.RECOVERY_PROOF_INVALID,
    )
    if (
        proof_payload.get("schema_version") != STAGE12_SCHEMA_VERSION
        or proof_payload.get("artifact_type") != "KEY_COMPROMISE_RECOVERY_PROOF"
        or proof_payload.get("runner_id") != expected_runner_id
        or proof_payload.get("old_certificate_serial") != old["serial"]
        or proof_payload.get("new_certificate_serial") != new["serial"]
        or proof_payload.get("old_key_id") != old["key_id"]
        or proof_payload.get("new_key_id") != new["key_id"]
        or proof_payload.get("old_generation") != 1
        or proof_payload.get("new_generation") != 2
        or proof_payload.get("old_key_authorized_after_compromise") is not False
        or proof_payload.get("source_stage11_package_sha256")
        != expected_frozen_payload["source_stage11_package_sha256"]
        or proof_payload.get("frozen_contract_sha256")
        != expected_frozen_payload["frozen_contract_sha256"]
        or not _integer(proof_payload["compromise_epoch"], "compromise time")
        < _integer(proof_payload["recovery_epoch"], "recovery time")
        <= now
        or by_serial[old["serial"]]["effective_epoch"] != proof_payload["compromise_epoch"]
        or not _integer(proof_payload["recovery_epoch"], "recovery time")
        <= _integer(by_serial[new["serial"]]["effective_epoch"], "good status time")
        <= now
    ):
        raise Stage12VerificationError(
            Stage12VerificationCode.RECOVERY_PROOF_INVALID, "recovery proof binding differs"
        )

    pre = _verify_envelope(
        artifacts["pre_compromise_envelope"],
        old,
        expected_frozen_payload,
        expected_runner_id,
        "PRE_COMPROMISE",
    )
    recovered = _verify_envelope(
        artifacts["recovered_envelope"],
        new,
        expected_frozen_payload,
        expected_runner_id,
        "RECOVERED",
    )
    if (
        pre["signing_generation"] != 1
        or recovered["signing_generation"] != 2
        or pre["producer_process_id"] != recovered["producer_process_id"]
    ):
        raise Stage12VerificationError(
            Stage12VerificationCode.ENVELOPE_POLICY_INVALID, "envelope generation differs"
        )
    receipt = _mapping(
        artifacts["receipt"],
        {
            "schema_version",
            "runner_id",
            "producer_process_id",
            "signer_adapter",
            "random_private_keys_generated",
            "private_key_exported",
            "private_key_serialized",
            "private_key_written_to_artifact",
            "hsm_backed",
            "production_authentication",
            "physical_host_attested",
            "network_used",
            "gpu_used",
        },
        "worker receipt",
    )
    if (
        receipt.get("schema_version") != STAGE12_SCHEMA_VERSION
        or receipt.get("runner_id") != expected_runner_id
        or receipt.get("producer_process_id") != recovered["producer_process_id"]
        or receipt.get("signer_adapter") != "InMemoryEd25519Signer"
        or receipt.get("random_private_keys_generated") != 3
        or any(
            receipt.get(name) is not False
            for name in (
                "private_key_exported",
                "private_key_serialized",
                "private_key_written_to_artifact",
                "hsm_backed",
                "production_authentication",
                "physical_host_attested",
                "network_used",
                "gpu_used",
            )
        )
    ):
        raise Stage12VerificationError(
            Stage12VerificationCode.MALFORMED_ARTIFACT, "worker receipt is invalid"
        )
    return {
        "runner_id": expected_runner_id,
        "producer_process_id": recovered["producer_process_id"],
        "session_nonce_sha256": recovered["session_nonce_sha256"],
        "trust_anchor_sha256": canonical_sha256(anchor),
        "compromised_key_id": old["key_id"],
        "recovered_key_id": new["key_id"],
        "pre_compromise_signature_cryptographically_valid": True,
        "pre_compromise_authority_after_revocation": "REJECTED",
        "recovered_authority": "ACCEPTED_LOCAL_FIXTURE_ONLY",
        "frozen_payload": recovered,
    }


def _verify_envelope(
    value: object,
    certificate: Mapping[str, object],
    expected_frozen: Mapping[str, object],
    expected_runner_id: str,
    phase: str,
) -> dict[str, object]:
    envelope = _mapping(
        value,
        {
            "schema_version",
            "algorithm",
            "authentication_scope",
            "certificate_serial",
            "key_id",
            "payload",
            "signature_b64",
        },
        "envelope",
    )
    payload = _mapping(envelope["payload"], _ENVELOPE_FIELDS, "envelope payload")
    if (
        envelope["schema_version"] != STAGE12_SCHEMA_VERSION
        or envelope["algorithm"] != STAGE12_ALGORITHM
        or envelope["authentication_scope"] != STAGE12_SCOPE
        or envelope["certificate_serial"] != certificate["serial"]
        or envelope["key_id"] != certificate["key_id"]
        or payload["schema_version"] != STAGE12_SCHEMA_VERSION
        or payload["runner_id"] != expected_runner_id
        or payload["phase"] != phase
        or payload["scope"] != STAGE12_SCOPE
        or payload["split"] != "test"
        or any(
            payload[name] is not False
            for name in (
                "payload_contains_data",
                "payload_contains_root",
                "payload_contains_private_key",
                "authorizes_real_data",
                "authorizes_physical_root",
                "production_authentication",
            )
        )
    ):
        raise Stage12VerificationError(
            Stage12VerificationCode.ENVELOPE_POLICY_INVALID, "envelope policy is invalid"
        )
    if _integer(payload["producer_process_id"], "producer process id") <= 0:
        raise Stage12VerificationError(
            Stage12VerificationCode.ENVELOPE_POLICY_INVALID, "producer process id is invalid"
        )
    _digest(payload["session_nonce_sha256"], "session nonce")
    dynamic = {
        "runner_id",
        "producer_process_id",
        "session_nonce_sha256",
        "phase",
        "signing_generation",
    }
    for name, expected in expected_frozen.items():
        if name not in dynamic and payload.get(name) != expected:
            raise Stage12VerificationError(
                Stage12VerificationCode.FROZEN_CONTRACT_MISMATCH,
                f"frozen field differs: {name}",
            )
    _verify_signature(
        _public(certificate["public_key_b64"], "certificate public key"),
        envelope["signature_b64"],
        payload,
        Stage12VerificationCode.ENVELOPE_SIGNATURE_INVALID,
    )
    return payload


def compare_recovered_runners(
    left: Mapping[str, object], right: Mapping[str, object]
) -> dict[str, object]:
    differences: list[str] = []
    if left.get("producer_process_id") == right.get("producer_process_id"):
        differences.append(Stage12VerificationCode.PROCESS_ID_NOT_INDEPENDENT.value)
    if left.get("session_nonce_sha256") == right.get("session_nonce_sha256"):
        differences.append(Stage12VerificationCode.SESSION_NONCE_REPLAYED.value)
    if left.get("recovered_key_id") == right.get("recovered_key_id"):
        differences.append(Stage12VerificationCode.SHARED_RECOVERY_KEY_REJECTED.value)
    left_payload = left.get("frozen_payload")
    right_payload = right.get("frozen_payload")
    if not isinstance(left_payload, Mapping) or not isinstance(right_payload, Mapping):
        differences.append(Stage12VerificationCode.MALFORMED_ARTIFACT.value)
    else:
        dynamic = {"runner_id", "producer_process_id", "session_nonce_sha256"}
        for name in sorted(_ENVELOPE_FIELDS - dynamic):
            if left_payload.get(name) != right_payload.get(name):
                differences.append(f"FROZEN_FIELD_MISMATCH:{name}")
    return {
        "schema_version": STAGE12_SCHEMA_VERSION,
        "status": "SUCCESS" if not differences else "FAIL_CLOSED",
        "differences": differences,
        "effective_state_on_difference": "OUT_OF_MODEL",
        "deterministic_difference_fail_closed": True,
    }
