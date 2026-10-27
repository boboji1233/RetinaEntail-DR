from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import cast

import pytest

from retinaentail.reproducibility.stage11 import (
    Stage11VerificationCode,
    Stage11VerificationError,
    build_fixture_certificate,
    build_fixture_revocation_list,
    build_fixture_rotation_proof,
    build_fixture_trust_anchor,
    build_signed_envelope,
    canonical_sha256,
    ensure_independent_runner_keys,
    fixture_private_key,
    key_id,
    verify_runner_bundle,
)

NOW = 1_787_529_600
SOURCE = "1" * 64


def _payload(runner_id: str) -> dict[str, object]:
    return {
        "schema_version": "11.0.0",
        "runner_id": runner_id,
        "logical_host_id": f"{runner_id}-local-slot",
        "producer_process_id": 101,
        "session_nonce_sha256": ("2" if runner_id == "runner-a" else "3") * 64,
        "dataset_id": "synthetic_stage2",
        "dataset_version": "synthetic-fixture-v1",
        "scope": "SYNTHETIC_ONLY",
        "split": "test",
        "source_stage10_package_sha256": SOURCE,
        "source_stage10_package_file_count": 588,
        "stage10_manifest_sha256": "4" * 64,
        "stage10_audit_sha256": "5" * 64,
        "stage10_summary_sha256": "6" * 64,
        "stage10_environment_lock_sha256": "7" * 64,
        "baseline_projection_sha256": "8" * 64,
        "runtime_lock_sha256": "9" * 64,
        "dependency_lock_sha256": "a" * 64,
        "trust_anchor_sha256": "b" * 64,
        "host_profile_sha256": "c" * 64,
        "alpha": 0.25,
        "positive_threshold": 0.1,
        "negative_threshold": 0.2,
        "positive_sample_count": 4,
        "negative_sample_count": 4,
        "config_hash": "d" * 64,
        "data_hash": "e" * 64,
        "model_hash": "f" * 64,
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


def _bundle(runner_id: str) -> dict[str, object]:
    anchor = build_fixture_trust_anchor(NOW)
    old_private = fixture_private_key(f"{runner_id}:old")
    current_private = fixture_private_key(f"{runner_id}:current")
    old_certificate = build_fixture_certificate(
        subject=runner_id,
        generation=1,
        public_key=old_private.public_key(),
        previous_key_id=None,
        trusted_time_epoch=NOW,
    )
    current_certificate = build_fixture_certificate(
        subject=runner_id,
        generation=2,
        public_key=current_private.public_key(),
        previous_key_id=key_id(old_private.public_key()),
        trusted_time_epoch=NOW,
    )
    old_payload = cast(Mapping[str, object], old_certificate["payload"])
    rotation = build_fixture_rotation_proof(
        runner_id=runner_id,
        old_certificate=old_certificate,
        current_certificate=current_certificate,
        old_private_key=old_private,
        source_stage10_package_sha256=SOURCE,
        trusted_time_epoch=NOW,
    )
    revocation = build_fixture_revocation_list(
        revoked_certificates=[(cast(str, old_payload["serial"]), "SUPERSEDED", NOW - 1_000)],
        trusted_time_epoch=NOW,
    )
    payload = _payload(runner_id)
    return {
        "anchor": anchor,
        "anchor_sha256": canonical_sha256(anchor),
        "old_certificate": old_certificate,
        "current_certificate": current_certificate,
        "rotation": rotation,
        "revocation": revocation,
        "payload": payload,
        "envelope": build_signed_envelope(
            payload, current_certificate=current_certificate, private_key=current_private
        ),
    }


def _verify(bundle: Mapping[str, object]) -> dict[str, object]:
    payload = cast(Mapping[str, object], bundle["payload"])
    return verify_runner_bundle(
        runner_id=cast(str, payload["runner_id"]),
        trust_anchor=bundle["anchor"],
        expected_trust_anchor_sha256=cast(str, bundle["anchor_sha256"]),
        revocation_list=bundle["revocation"],
        old_certificate=bundle["old_certificate"],
        current_certificate=bundle["current_certificate"],
        rotation_proof=bundle["rotation"],
        envelope=bundle["envelope"],
        trusted_time_epoch=NOW,
        expected_source_stage10_package_sha256=SOURCE,
        expected_payload=payload,
    )


def test_valid_ed25519_chain_rotation_revocation_and_envelope_replay() -> None:
    bundle = _bundle("runner-a")
    assert _verify(bundle) == bundle["payload"]


def test_payload_signature_rotation_and_revocation_tampering_fail_closed() -> None:
    signature_tamper = copy.deepcopy(_bundle("runner-a"))
    envelope = cast(dict[str, object], signature_tamper["envelope"])
    envelope["signature_b64"] = "A" * 88
    with pytest.raises(Stage11VerificationError) as signature_error:
        _verify(signature_tamper)
    assert signature_error.value.code is Stage11VerificationCode.ENVELOPE_SIGNATURE_INVALID

    rotation_tamper = copy.deepcopy(_bundle("runner-a"))
    rotation = cast(dict[str, object], rotation_tamper["rotation"])
    rotation["old_key_signature_b64"] = "A" * 88
    with pytest.raises(Stage11VerificationError) as rotation_error:
        _verify(rotation_tamper)
    assert rotation_error.value.code is Stage11VerificationCode.ROTATION_SIGNATURE_INVALID

    revoked = copy.deepcopy(_bundle("runner-a"))
    old = cast(Mapping[str, object], revoked["old_certificate"])
    old_payload = cast(Mapping[str, object], old["payload"])
    current = cast(Mapping[str, object], revoked["current_certificate"])
    current_payload = cast(Mapping[str, object], current["payload"])
    revoked["revocation"] = build_fixture_revocation_list(
        revoked_certificates=[
            (cast(str, old_payload["serial"]), "SUPERSEDED", NOW - 1_000),
            (cast(str, current_payload["serial"]), "KEY_COMPROMISE", NOW - 100),
        ],
        trusted_time_epoch=NOW,
    )
    with pytest.raises(Stage11VerificationError) as revocation_error:
        _verify(revoked)
    assert revocation_error.value.code is Stage11VerificationCode.CERTIFICATE_REVOKED


def test_distinct_runner_leaf_keys_are_required() -> None:
    left = _bundle("runner-a")
    right = _bundle("runner-b")
    ensure_independent_runner_keys(
        cast(Mapping[str, object], left["current_certificate"]),
        cast(Mapping[str, object], right["current_certificate"]),
    )
    with pytest.raises(Stage11VerificationError) as error:
        ensure_independent_runner_keys(
            cast(Mapping[str, object], left["current_certificate"]),
            cast(Mapping[str, object], left["current_certificate"]),
        )
    assert error.value.code is Stage11VerificationCode.SHARED_RUNNER_KEY_REJECTED
