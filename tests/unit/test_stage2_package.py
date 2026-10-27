from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import time
from dataclasses import replace
from pathlib import Path

import pytest

import retinaentail.data.stage2_package as package_module
from retinaentail.data.audit import AuditResult, AuditStatus, AuditSummary
from retinaentail.data.license_gate import (
    DatasetAccessPermit,
    DataUsePurpose,
    LicenseReviewDecision,
    LicenseReviewRecord,
    RegistryLicenseMetadata,
    authorize_dataset_access,
)
from retinaentail.data.stage2_fixture import _run_stage2_fixture_at
from retinaentail.data.stage2_models import (
    DATASET_LABEL_SEMANTICS,
    DatasetAdapterSpec,
    DatasetEntrySpec,
    DatasetManifestV2,
    LabelAvailability,
    LabelInputSpec,
    LabelRecordV2,
    LabelSemantic,
    ManifestRecordV2,
)
from retinaentail.data.stage2_package import (
    Stage2PackageEvidence,
    Stage2PackageScope,
    Stage2PackageVerificationError,
    consume_stage2_package,
    replay_stage2_package,
    replay_stage2_package_evidence,
    replay_stage2_package_manifest,
    stage2_package_capability_snapshot,
    stage2_package_capability_validation_reasons,
    verify_and_issue_stage2_package,
)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _synthetic_package(tmp_path: Path, name: str = "fixture") -> Path:
    return _run_stage2_fixture_at(tmp_path / name).package_dir


def _rewrite_json(path: Path, update: object) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    if not callable(update):
        raise TypeError("update must be callable")
    update(value)
    path.write_text(
        json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_complete_synthetic_package_issues_replays_and_consumes_once(tmp_path: Path) -> None:
    package_dir = _synthetic_package(tmp_path)

    capability = verify_and_issue_stage2_package(
        package_dir,
        scope=Stage2PackageScope.SYNTHETIC_FIXTURE,
        ttl_seconds=30,
    )

    assert capability.authorizes_dataset_root_access is False
    assert stage2_package_capability_validation_reasons(capability) == ()
    descriptor = stage2_package_capability_snapshot(capability)
    assert descriptor.dataset_id == "synthetic_stage2"
    assert descriptor.dataset_version == "synthetic-fixture-v1"
    assert descriptor.record_count == 1
    assert tuple(item.relative_path for item in descriptor.files) == (
        "audit.json",
        "manifest-v2.json",
    )
    assert descriptor.registry_sha256 is None
    assert descriptor.package_sha256 == descriptor.to_dict()["package_sha256"]

    replay = replay_stage2_package(capability)
    assert replay.valid is True
    assert replay.reasons == ()
    assert replay.descriptor == descriptor
    assert replay.authorizes_consumption is False
    assert replay.to_dict()["authorizes_consumption"] is False

    replayed_manifest = replay_stage2_package_manifest(capability)
    assert replayed_manifest.dataset_id == "synthetic_stage2"
    assert replay_stage2_package_manifest(capability) == replayed_manifest
    assert stage2_package_capability_validation_reasons(capability) == ()
    evidence = replay_stage2_package_evidence(capability)
    assert isinstance(evidence, Stage2PackageEvidence)
    assert evidence.manifest == replayed_manifest
    assert evidence.audit.status is AuditStatus.PASS
    assert evidence.descriptor == descriptor
    assert evidence.authorizes_dataset_root_access is False
    assert evidence.to_dict()["authorizes_dataset_root_access"] is False
    assert stage2_package_capability_validation_reasons(capability) == ()

    manifest = consume_stage2_package(capability)
    assert manifest.dataset_id == "synthetic_stage2"
    assert stage2_package_capability_validation_reasons(capability) == (
        "capability has already been consumed",
    )
    with pytest.raises(Stage2PackageVerificationError, match="already been consumed"):
        consume_stage2_package(capability)


def test_detached_pass_audit_and_unknown_files_never_issue(tmp_path: Path) -> None:
    source = _synthetic_package(tmp_path, "source")
    detached = tmp_path / "detached"
    detached.mkdir()
    (detached / "audit.json").write_bytes((source / "audit.json").read_bytes())

    with pytest.raises(Stage2PackageVerificationError, match="missing required"):
        verify_and_issue_stage2_package(
            detached,
            scope=Stage2PackageScope.SYNTHETIC_FIXTURE,
        )

    complete = _synthetic_package(tmp_path, "unknown")
    (complete / "audit.html").write_text("not part of the evidence package", encoding="utf-8")
    with pytest.raises(Stage2PackageVerificationError, match="unknown files"):
        verify_and_issue_stage2_package(
            complete,
            scope=Stage2PackageScope.SYNTHETIC_FIXTURE,
        )


@pytest.mark.parametrize(
    ("filename", "mutation", "message"),
    (
        (
            "audit.json",
            lambda value: value["summary"].__setitem__("record_count", 2),
            "evidence contract",
        ),
        (
            "audit.json",
            lambda value: value.__setitem__("manifest_sha256", "0" * 64),
            "manifest_sha256",
        ),
        (
            "manifest-v2.json",
            lambda value: value["records"][0].__setitem__("source_image_id", "changed"),
            "manifest_sha256",
        ),
    ),
)
def test_manifest_audit_cross_bindings_fail_closed(
    tmp_path: Path,
    filename: str,
    mutation: object,
    message: str,
) -> None:
    package_dir = _synthetic_package(tmp_path)
    _rewrite_json(package_dir / filename, mutation)

    with pytest.raises(Stage2PackageVerificationError, match=message):
        verify_and_issue_stage2_package(
            package_dir,
            scope=Stage2PackageScope.SYNTHETIC_FIXTURE,
        )


def test_replay_rejects_package_byte_tampering(tmp_path: Path) -> None:
    package_dir = _synthetic_package(tmp_path)
    capability = verify_and_issue_stage2_package(
        package_dir,
        scope=Stage2PackageScope.SYNTHETIC_FIXTURE,
    )

    _rewrite_json(
        package_dir / "manifest-v2.json",
        lambda value: value["records"][0].__setitem__("source_image_id", "tampered"),
    )

    replay = replay_stage2_package(capability)
    assert replay.valid is False
    assert replay.descriptor is None
    with pytest.raises(Stage2PackageVerificationError):
        replay_stage2_package_manifest(capability)
    with pytest.raises(Stage2PackageVerificationError):
        replay_stage2_package_evidence(capability)
    with pytest.raises(Stage2PackageVerificationError):
        consume_stage2_package(capability)


def test_capability_resists_copy_replace_serialization_and_nested_mutation(
    tmp_path: Path,
) -> None:
    package_dir = _synthetic_package(tmp_path)
    capability = verify_and_issue_stage2_package(
        package_dir,
        scope=Stage2PackageScope.SYNTHETIC_FIXTURE,
    )

    with pytest.raises(TypeError, match="cannot be copied"):
        copy.copy(capability)
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.deepcopy(capability)
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(capability)
    with pytest.raises(TypeError):
        json.dumps(capability)

    replacement = replace(capability)
    assert stage2_package_capability_validation_reasons(replacement) == (
        "capability was not issued by the Stage 2 package verifier",
    )

    object.__setattr__(capability.descriptor, "record_count", 99)
    assert stage2_package_capability_validation_reasons(capability) == (
        "capability bindings changed after issuance",
    )

    retargeted = verify_and_issue_stage2_package(
        _synthetic_package(tmp_path, "retargeted"),
        scope=Stage2PackageScope.SYNTHETIC_FIXTURE,
    )
    object.__setattr__(retargeted, "_package_dir", tmp_path / "different-package")
    assert stage2_package_capability_validation_reasons(retargeted) == (
        "capability bindings changed after issuance",
    )


def test_expired_capability_cannot_replay_or_consume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [1_000_000_000]
    monkeypatch.setattr(time, "monotonic_ns", lambda: clock[0])
    capability = verify_and_issue_stage2_package(
        _synthetic_package(tmp_path),
        scope=Stage2PackageScope.SYNTHETIC_FIXTURE,
        ttl_seconds=1,
    )
    original_expiry = capability.expires_monotonic_ns
    clock[0] += 500_000_000
    assert replay_stage2_package_manifest(capability).dataset_id == "synthetic_stage2"
    assert capability.expires_monotonic_ns == original_expiry
    clock[0] = original_expiry

    assert "capability has expired" in stage2_package_capability_validation_reasons(capability)
    assert replay_stage2_package(capability).valid is False
    with pytest.raises(Stage2PackageVerificationError, match="expired"):
        replay_stage2_package_manifest(capability)
    with pytest.raises(Stage2PackageVerificationError, match="expired"):
        replay_stage2_package_evidence(capability)
    with pytest.raises(Stage2PackageVerificationError, match="expired"):
        consume_stage2_package(capability)


def test_package_link_is_rejected_when_platform_allows_symlinks(tmp_path: Path) -> None:
    source = _synthetic_package(tmp_path, "source")
    linked = tmp_path / "linked"
    linked.mkdir()
    try:
        (linked / "audit.json").symlink_to(source / "audit.json")
        (linked / "manifest-v2.json").write_bytes((source / "manifest-v2.json").read_bytes())
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {type(error).__name__}")

    with pytest.raises(Stage2PackageVerificationError, match="link or reparse"):
        verify_and_issue_stage2_package(
            linked,
            scope=Stage2PackageScope.SYNTHETIC_FIXTURE,
        )


def test_oversized_manifest_is_rejected_before_json_decode(tmp_path: Path) -> None:
    source = _synthetic_package(tmp_path, "source")
    oversized = tmp_path / "oversized"
    oversized.mkdir()
    (oversized / "audit.json").write_bytes((source / "audit.json").read_bytes())
    with (oversized / "manifest-v2.json").open("wb") as stream:
        stream.truncate(package_module.MAX_STAGE2_MANIFEST_BYTES + 1)

    with pytest.raises(Stage2PackageVerificationError, match="bounded size"):
        verify_and_issue_stage2_package(
            oversized,
            scope=Stage2PackageScope.SYNTHETIC_FIXTURE,
        )


def _simulated_real_package_and_permit(
    tmp_path: Path,
) -> tuple[Path, DatasetAccessPermit, Path]:
    dataset_version = "simulated-contract-v1"
    image_relative = "images/case.png"
    mask_relative = "labels/microaneurysm.png"
    labels = tuple(
        LabelRecordV2(
            semantic=semantic,
            availability=(
                LabelAvailability.AVAILABLE
                if semantic is LabelSemantic.MICROANEURYSM_MASK
                else LabelAvailability.UNKNOWN
            ),
            relative_path=(mask_relative if semantic is LabelSemantic.MICROANEURYSM_MASK else None),
            sha256=("c" * 64 if semantic is LabelSemantic.MICROANEURYSM_MASK else None),
            byte_size=(123 if semantic is LabelSemantic.MICROANEURYSM_MASK else None),
        )
        for semantic in sorted(DATASET_LABEL_SEMANTICS["idrid"], key=lambda item: item.value)
    )
    manifest = DatasetManifestV2(
        dataset_id="idrid",
        dataset_version=dataset_version,
        records=(
            ManifestRecordV2(
                dataset_id="idrid",
                relative_path=image_relative,
                sha256="d" * 64,
                byte_size=456,
                width=16,
                height=12,
                channels=3,
                image_format="PNG",
                source_image_id="simulated-source-1",
                patient_id=None,
                eye_id="right",
                followup_id=None,
                intervention_pair_id=None,
                split="test",
                labels=labels,
            ),
        ),
    )
    audit = AuditResult(
        status=AuditStatus.PASS,
        dataset_id="idrid",
        dataset_version=dataset_version,
        manifest_sha256=_canonical_sha256(manifest.to_dict()),
        summary=AuditSummary(
            record_count=1,
            image_files_checked=1,
            label_files_checked=1,
            mask_files_decoded=1,
            unknown_label_count=len(labels) - 1,
            issue_counts=(),
        ),
    )
    package_dir = tmp_path / "simulated-real-evidence"
    package_dir.mkdir()
    (package_dir / "manifest-v2.json").write_text(
        json.dumps(manifest.to_dict(), allow_nan=False, sort_keys=True),
        encoding="utf-8",
    )
    (package_dir / "audit.json").write_text(
        json.dumps(audit.to_dict(), allow_nan=False, sort_keys=True),
        encoding="utf-8",
    )
    registry = RegistryLicenseMetadata(
        schema_version="1.0.0",
        dataset_id="idrid",
        dataset_version=dataset_version,
        license_status="verified",
        license_review_status="reviewed",
        label_verification_status="verified",
    )
    review = LicenseReviewRecord(
        dataset_id="idrid",
        dataset_version=dataset_version,
        decision=LicenseReviewDecision.APPROVED,
        permitted_purposes=(DataUsePurpose.READ_ONLY_AUDIT,),
        terms_sha256="a" * 64,
        evidence_sha256="b" * 64,
        reviewer_id="synthetic-unit-test-reviewer",
        reviewed_at="2026-08-24T00:00:00+00:00",
    )
    spec = DatasetAdapterSpec(
        dataset_id="idrid",
        entries=(
            DatasetEntrySpec(
                image_path=Path(image_relative),
                source_image_id="simulated-source-1",
                patient_id=None,
                eye_id="right",
                followup_id=None,
                intervention_pair_id=None,
                split="test",
                labels=(
                    LabelInputSpec(
                        semantic=LabelSemantic.MICROANEURYSM_MASK,
                        relative_path=Path(mask_relative),
                    ),
                ),
            ),
        ),
    )
    never_probed_root = (tmp_path / "nonexistent-simulated-dataset-root").absolute()
    permit = authorize_dataset_access(
        registry,
        review=review,
        purpose=DataUsePurpose.READ_ONLY_AUDIT,
        root=never_probed_root,
        spec=spec,
    )
    return package_dir, permit, never_probed_root


def test_real_scope_requires_live_exact_permit_without_probing_dataset_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_dir, permit, never_probed_root = _simulated_real_package_and_permit(tmp_path)
    real_lstat = os.lstat

    def guarded_lstat(path: os.PathLike[str] | str) -> os.stat_result:
        candidate = Path(path)
        assert candidate != never_probed_root
        assert never_probed_root not in candidate.parents
        return real_lstat(path)

    monkeypatch.setattr(os, "lstat", guarded_lstat)
    capability = verify_and_issue_stage2_package(
        package_dir,
        scope=Stage2PackageScope.APPROVED_REAL_DATA,
        permit=permit,
    )

    descriptor = stage2_package_capability_snapshot(capability)
    assert descriptor.scope is Stage2PackageScope.APPROVED_REAL_DATA
    assert descriptor.registry_sha256 is not None
    assert descriptor.license_review_sha256 is not None
    assert descriptor.adapter_spec_sha256 is not None
    assert descriptor.root_binding_sha256 is not None
    assert descriptor.permit_binding_sha256 is not None
    assert replay_stage2_package(capability).valid is True

    object.__setattr__(permit.registry, "license_status", "unknown")
    assert stage2_package_capability_validation_reasons(capability) == (
        "capability bindings changed after issuance",
    )


def test_real_scope_without_permit_blocks_before_package_file_reads(tmp_path: Path) -> None:
    package_dir = _synthetic_package(tmp_path)

    with pytest.raises(Stage2PackageVerificationError, match="requires a process-issued"):
        verify_and_issue_stage2_package(
            package_dir,
            scope=Stage2PackageScope.APPROVED_REAL_DATA,
        )


def test_real_evidence_path_overlapping_root_blocks_before_any_root_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, permit, never_probed_root = _simulated_real_package_and_permit(tmp_path)

    def unexpected_lstat(*args: object, **kwargs: object) -> os.stat_result:
        del args, kwargs
        raise AssertionError("lstat must not run for evidence inside the dataset root")

    monkeypatch.setattr(os, "lstat", unexpected_lstat)
    with pytest.raises(Stage2PackageVerificationError, match="must not overlap"):
        verify_and_issue_stage2_package(
            never_probed_root / "evidence-package",
            scope=Stage2PackageScope.APPROVED_REAL_DATA,
            permit=permit,
        )
