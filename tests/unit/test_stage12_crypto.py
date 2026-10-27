from __future__ import annotations

import copy
import json
import pickle
from pathlib import Path
from typing import cast

import pytest

from retinaentail.pilot.stage12_worker import run_worker
from retinaentail.reproducibility.stage12 import (
    InMemoryEd25519Signer,
    Stage12VerificationCode,
    Stage12VerificationError,
    canonical_sha256,
    verify_runner_recovery_bundle,
)

NOW = 1_787_529_600


def _frozen() -> dict[str, object]:
    source: dict[str, object] = {
        "source_stage11_package_sha256": "1" * 64,
        "source_stage11_package_file_count": 608,
        "source_stage11_manifest_sha256": "2" * 64,
        "source_stage11_audit_sha256": "3" * 64,
        "source_stage11_summary_sha256": "4" * 64,
        "alpha": 0.25,
        "positive_threshold": 0.1,
        "negative_threshold": 0.2,
        "positive_sample_count": 4,
        "negative_sample_count": 4,
        "config_hash": "5" * 64,
        "data_hash": "6" * 64,
        "model_hash": "7" * 64,
    }
    return {**source, "frozen_contract_sha256": canonical_sha256(source)}


def _bundle(tmp_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    frozen = _frozen()
    request = {
        "schema_version": "12.0.0",
        "runner_id": "runner-a",
        "trusted_time_epoch": NOW,
        "compromise_epoch": NOW - 600,
        "recovery_epoch": NOW - 300,
        "frozen_payload": frozen,
    }
    request_path = tmp_path / "request.json"
    output_path = tmp_path / "bundle.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    run_worker(request_path, output_path)
    value: object = json.loads(output_path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return cast(dict[str, object], value), frozen


def test_random_signer_has_no_private_export_api_and_is_not_serializable() -> None:
    first = InMemoryEd25519Signer()
    second = InMemoryEd25519Signer()
    assert first.key_id != second.key_id
    assert not hasattr(first, "private_bytes")
    assert not hasattr(first, "export_private_key")
    with pytest.raises(TypeError):
        pickle.dumps(first)


def test_recovery_bundle_verifies_and_compromised_generation_is_rejected(tmp_path: Path) -> None:
    bundle, frozen = _bundle(tmp_path)
    verified = verify_runner_recovery_bundle(
        bundle,
        expected_runner_id="runner-a",
        expected_frozen_payload=frozen,
        trusted_time_epoch=NOW,
    )
    assert verified["pre_compromise_signature_cryptographically_valid"] is True
    assert verified["pre_compromise_authority_after_revocation"] == "REJECTED"
    assert verified["recovered_authority"] == "ACCEPTED_LOCAL_FIXTURE_ONLY"
    assert verified["compromised_key_id"] != verified["recovered_key_id"]


def test_signature_and_receipt_tampering_fail_closed(tmp_path: Path) -> None:
    bundle, frozen = _bundle(tmp_path)
    signature_tamper = copy.deepcopy(bundle)
    cast(dict[str, object], signature_tamper["recovered_envelope"])["signature_b64"] = "A" * 88
    with pytest.raises(Stage12VerificationError) as signature_error:
        verify_runner_recovery_bundle(
            signature_tamper,
            expected_runner_id="runner-a",
            expected_frozen_payload=frozen,
            trusted_time_epoch=NOW,
        )
    assert signature_error.value.code is Stage12VerificationCode.ENVELOPE_SIGNATURE_INVALID

    receipt_tamper = copy.deepcopy(bundle)
    cast(dict[str, object], receipt_tamper["receipt"])["private_key_serialized"] = True
    with pytest.raises(Stage12VerificationError) as receipt_error:
        verify_runner_recovery_bundle(
            receipt_tamper,
            expected_runner_id="runner-a",
            expected_frozen_payload=frozen,
            trusted_time_epoch=NOW,
        )
    assert receipt_error.value.code is Stage12VerificationCode.MALFORMED_ARTIFACT

    runner_tamper = copy.deepcopy(bundle)
    recovered = cast(dict[str, object], runner_tamper["recovered_envelope"])
    cast(dict[str, object], recovered["payload"])["runner_id"] = "runner-b"
    with pytest.raises(Stage12VerificationError) as runner_error:
        verify_runner_recovery_bundle(
            runner_tamper,
            expected_runner_id="runner-a",
            expected_frozen_payload=frozen,
            trusted_time_epoch=NOW,
        )
    assert runner_error.value.code is Stage12VerificationCode.ENVELOPE_POLICY_INVALID
