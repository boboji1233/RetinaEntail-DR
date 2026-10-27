from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from PIL import Image

import retinaentail.data.audit as audit_module
from retinaentail.data.audit import (
    AuditIssue,
    AuditStatus,
    IssueSeverity,
    _audit_synthetic_fixture,
    audit_manifest,
    blocked_audit_result,
)
from retinaentail.data.audit_reporting import (
    AuditWriteBoundary,
    render_audit_html,
    render_audit_json,
    render_manifest_csv,
    write_audit_html,
    write_audit_json,
    write_audit_reports,
)
from retinaentail.data.license_gate import (
    DatasetAccessPermit,
    DataUsePurpose,
    LicenseReviewDecision,
    LicenseReviewRecord,
    RegistryLicenseMetadata,
    authorize_dataset_access,
)
from retinaentail.data.manifest import sha256_file
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


def _permit(
    root: Path,
    *,
    purpose: DataUsePurpose = DataUsePurpose.READ_ONLY_AUDIT,
    label: LabelInputSpec | None = None,
    label_semantics: tuple[LabelSemantic, ...] | None = None,
) -> DatasetAccessPermit:
    registry = RegistryLicenseMetadata(
        schema_version="1.0.0",
        dataset_id="idrid",
        dataset_version="fixture-v1",
        license_status="verified",
        license_review_status="reviewed",
        label_verification_status="verified",
    )
    review = LicenseReviewRecord(
        dataset_id="idrid",
        dataset_version="fixture-v1",
        decision=LicenseReviewDecision.APPROVED,
        permitted_purposes=(purpose,),
        terms_sha256="a" * 64,
        evidence_sha256="b" * 64,
        reviewer_id="synthetic-audit-test-reviewer",
        reviewed_at="2026-08-23T12:00:00+00:00",
    )
    spec = DatasetAdapterSpec(
        dataset_id="idrid",
        label_semantics=label_semantics,
        entries=(
            DatasetEntrySpec(
                image_path=Path("images/case.png"),
                source_image_id="source-1",
                split="train",
                labels=() if label is None else (label,),
            ),
        ),
    )
    return authorize_dataset_access(
        registry,
        review=review,
        purpose=purpose,
        root=root,
        spec=spec,
    )


def test_real_audit_accepts_complete_explicit_task_catalog(tmp_path: Path) -> None:
    _save_image(tmp_path, "images/case.png")
    mask_path = _save_image(
        tmp_path,
        "labels/case-he.png",
        mode="L",
        color=0,
    )
    with Image.open(mask_path) as mask:
        editable = mask.copy()
    editable.putpixel((2, 2), 255)
    editable.save(mask_path)
    semantic = LabelSemantic.HEMORRHAGE_MASK
    manifest = DatasetManifestV2(
        dataset_id="idrid",
        dataset_version="fixture-v1",
        records=(
            _record(
                tmp_path,
                "images/case.png",
                labels=(_available_label(tmp_path, "labels/case-he.png", semantic),),
                source_image_id="source-1",
                split="train",
                dataset_id="idrid",
            ),
        ),
    )
    permit = _permit(
        tmp_path,
        label=LabelInputSpec(
            semantic=semantic,
            relative_path=Path("labels/case-he.png"),
        ),
        label_semantics=(semantic,),
    )

    result = audit_manifest(manifest, permit=permit)

    assert result.status is AuditStatus.PASS
    assert result.summary.unknown_label_count == 0


@pytest.mark.parametrize("mode", ["P", "RGBA"])
def test_idrid_exact_black_red_mask_encodings_are_controlled(
    tmp_path: Path,
    mode: str,
) -> None:
    _save_image(tmp_path, "images/case.png")
    mask_path = tmp_path / "labels" / "case-he.tif"
    mask_path.parent.mkdir(parents=True)
    if mode == "P":
        mask = Image.new("P", (8, 6), color=0)
        palette = [0] * 768
        palette[3:6] = [255, 0, 0]
        mask.putpalette(palette)
        mask.putpixel((2, 2), 1)
    else:
        mask = Image.new("RGBA", (8, 6), color=(0, 0, 0, 255))
        mask.putpixel((2, 2), (255, 0, 0, 255))
    mask.save(mask_path)
    semantic = LabelSemantic.HEMORRHAGE_MASK
    manifest = DatasetManifestV2(
        dataset_id="idrid",
        dataset_version="fixture-v1",
        records=(
            _record(
                tmp_path,
                "images/case.png",
                labels=(_available_label(tmp_path, "labels/case-he.tif", semantic),),
                source_image_id="source-1",
                dataset_id="idrid",
            ),
        ),
    )
    permit = _permit(
        tmp_path,
        label=LabelInputSpec(
            semantic=semantic,
            relative_path=Path("labels/case-he.tif"),
        ),
        label_semantics=(semantic,),
    )

    result = audit_manifest(manifest, permit=permit)

    assert result.status is AuditStatus.PASS
    assert result.issues == ()


def test_palette_mask_outside_idrid_remains_fail_closed(tmp_path: Path) -> None:
    _save_image(tmp_path, "images/case.png")
    mask_path = tmp_path / "labels" / "case-he.png"
    mask_path.parent.mkdir(parents=True)
    mask = Image.new("P", (8, 6), color=0)
    palette = [0] * 768
    palette[3:6] = [255, 0, 0]
    mask.putpalette(palette)
    mask.putpixel((2, 2), 1)
    mask.save(mask_path)
    record = _record(
        tmp_path,
        "images/case.png",
        labels=(
            _available_label(
                tmp_path,
                "labels/case-he.png",
                LabelSemantic.HEMORRHAGE_MASK,
            ),
        ),
        source_image_id="source-1",
    )

    result = _audit_synthetic_fixture((record,), fixture_root=tmp_path)

    assert result.status is AuditStatus.FAIL_CLOSED
    assert "MASK_MODE_PALETTE" in {issue.code for issue in result.issues}


def _save_image(
    root: Path,
    relative_path: str,
    *,
    mode: str = "RGB",
    size: tuple[int, int] = (8, 6),
    color: int | tuple[int, int, int] = (20, 40, 60),
) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, size, color=color).save(path)
    return path


def _available_label(
    root: Path,
    relative_path: str,
    semantic: LabelSemantic,
) -> LabelRecordV2:
    path = root / relative_path
    return LabelRecordV2(
        semantic=semantic,
        availability=LabelAvailability.AVAILABLE,
        relative_path=relative_path,
        sha256=sha256_file(path),
        byte_size=path.stat().st_size,
    )


def _unknown_label(
    semantic: LabelSemantic = LabelSemantic.DR_GRADE,
    *,
    relative_path: str | None = None,
) -> LabelRecordV2:
    return LabelRecordV2(
        semantic=semantic,
        availability=LabelAvailability.UNKNOWN,
        relative_path=relative_path,
    )


def _record(
    root: Path,
    relative_path: str,
    *,
    labels: tuple[LabelRecordV2, ...],
    source_image_id: str,
    split: str = "train",
    patient_id: str | None = None,
    intervention_pair_id: str | None = None,
    dataset_id: str = "toy_stage2",
) -> ManifestRecordV2:
    path = root / relative_path
    with Image.open(path) as image:
        image.load()
        width, height = image.size
        channels = len(image.getbands())
        image_format = image.format or "UNKNOWN"
    return ManifestRecordV2(
        dataset_id=dataset_id,
        relative_path=relative_path,
        sha256=sha256_file(path),
        byte_size=path.stat().st_size,
        width=width,
        height=height,
        channels=channels,
        image_format=image_format,
        source_image_id=source_image_id,
        split=split,
        patient_id=patient_id,
        intervention_pair_id=intervention_pair_id,
        labels=labels,
    )


def test_valid_local_fixture_passes_without_root_disclosure(tmp_path: Path) -> None:
    _save_image(tmp_path, "images/case.png")
    mask_path = _save_image(
        tmp_path,
        "labels/case-ma.png",
        mode="L",
        color=0,
    )
    with Image.open(mask_path) as mask:
        editable = mask.copy()
    editable.putpixel((2, 2), 255)
    editable.save(mask_path)
    label = _available_label(
        tmp_path,
        "labels/case-ma.png",
        LabelSemantic.MICROANEURYSM_MASK,
    )
    record = _record(
        tmp_path,
        "images/case.png",
        labels=(label,),
        source_image_id="source-1",
    )

    manifest = DatasetManifestV2(
        dataset_id="toy_stage2",
        dataset_version="fixture-v1",
        records=(record,),
    )
    result = _audit_synthetic_fixture(manifest, fixture_root=tmp_path)
    json_report = render_audit_json(result)
    html_report = render_audit_html(result)

    assert result.status is AuditStatus.PASS
    assert result.issues == ()
    assert result.summary.image_files_checked == 1
    assert result.summary.label_files_checked == 1
    assert result.summary.mask_files_decoded == 1
    assert str(tmp_path) not in json_report
    assert str(tmp_path) not in html_report
    assert tmp_path.resolve().as_posix() not in json_report
    assert tmp_path.resolve().as_posix() not in html_report
    assert json.loads(json_report)["dataset_id"] == "toy_stage2"

    csv_report = render_manifest_csv(manifest)
    assert "label_semantic" in csv_report.splitlines()[0]
    assert "MICROANEURYSM_MASK" in csv_report
    assert str(tmp_path) not in csv_report


def test_versioned_manifest_keeps_repeated_source_id_for_leakage_audit(
    tmp_path: Path,
) -> None:
    first = _save_image(tmp_path, "images/train.png", color=(10, 20, 30))
    second = tmp_path / "images/test.png"
    second.write_bytes(first.read_bytes())
    train_record = _record(
        tmp_path,
        "images/train.png",
        labels=(_unknown_label(),),
        source_image_id="shared-source",
        split="train",
    )
    test_record = _record(
        tmp_path,
        "images/test.png",
        labels=(_unknown_label(),),
        source_image_id="shared-source",
        split="test",
    )
    manifest = DatasetManifestV2(
        dataset_id="toy_stage2",
        dataset_version="fixture-v1",
        records=(train_record, test_record),
    )

    result = _audit_synthetic_fixture(manifest, fixture_root=tmp_path)

    assert any(
        issue.code == "SPLIT_LEAKAGE"
        and dict(issue.details)["identifier_type"] == "source_image_id"
        for issue in result.issues
    )


def test_unknown_and_missing_labels_remain_unknown_not_negative(tmp_path: Path) -> None:
    _save_image(tmp_path, "images/case.png")
    unknown = _unknown_label(LabelSemantic.DR_GRADE)
    missing_available = LabelRecordV2(
        semantic=LabelSemantic.MICROANEURYSM_MASK,
        availability=LabelAvailability.AVAILABLE,
        relative_path="labels/missing.png",
        sha256="0" * 64,
        byte_size=1,
    )
    record = _record(
        tmp_path,
        "images/case.png",
        labels=(unknown, missing_available),
        source_image_id="source-1",
    )

    result = _audit_synthetic_fixture((record,), fixture_root=tmp_path)
    codes = {issue.code for issue in result.issues}
    serialized = render_audit_json(result)

    assert result.status is AuditStatus.FAIL_CLOSED
    assert result.summary.unknown_label_count == 2
    assert "LABEL_AVAILABILITY_UNKNOWN" in codes
    assert "LABEL_FILE_MISSING" in codes
    assert "ENTAILED_ABSENT" not in serialized
    assert "confirmed negative" not in serialized


def test_non_mask_detection_annotation_is_hashed_without_pillow_decode(tmp_path: Path) -> None:
    _save_image(tmp_path, "images/case.png")
    annotation_path = tmp_path / "labels/detections.json"
    annotation_path.parent.mkdir(parents=True)
    annotation_path.write_text('{"boxes": []}\n', encoding="utf-8")
    detection_label = _available_label(
        tmp_path,
        "labels/detections.json",
        LabelSemantic.LESION_DETECTION,
    )
    record = _record(
        tmp_path,
        "images/case.png",
        labels=(detection_label,),
        source_image_id="source-1",
    )

    result = _audit_synthetic_fixture((record,), fixture_root=tmp_path)

    assert result.status is AuditStatus.PASS
    assert result.summary.label_files_checked == 1
    assert result.summary.mask_files_decoded == 0
    assert "MASK_DECODE_FAILED" not in {issue.code for issue in result.issues}


def test_real_audit_requires_read_only_capability_before_root_access(tmp_path: Path) -> None:
    _save_image(tmp_path, "images/case.png")
    annotation_path = tmp_path / "labels/dr-grade.json"
    annotation_path.parent.mkdir(parents=True)
    annotation_path.write_text('{"grade": 1}\n', encoding="utf-8")
    label = _available_label(
        tmp_path,
        "labels/dr-grade.json",
        LabelSemantic.DR_GRADE,
    )
    placeholders = tuple(
        _unknown_label(semantic)
        for semantic in sorted(DATASET_LABEL_SEMANTICS["idrid"], key=lambda item: item.value)
        if semantic is not LabelSemantic.DR_GRADE
    )
    record = _record(
        tmp_path,
        "images/case.png",
        labels=(label, *placeholders),
        source_image_id="source-1",
        dataset_id="idrid",
    )

    manifest = DatasetManifestV2(
        dataset_id="idrid",
        dataset_version="fixture-v1",
        records=(record,),
    )
    permit = _permit(
        tmp_path,
        label=LabelInputSpec(
            semantic=LabelSemantic.DR_GRADE,
            relative_path=Path("labels/dr-grade.json"),
        ),
    )
    permitted = audit_manifest(manifest, permit=permit)
    incomplete_catalog_manifest = DatasetManifestV2(
        dataset_id="idrid",
        dataset_version="fixture-v1",
        records=(replace(record, labels=record.labels[:-1]),),
    )
    incomplete_catalog = audit_manifest(incomplete_catalog_manifest, permit=permit)
    undeclared_labels = tuple(
        (
            _unknown_label(
                LabelSemantic.DME_GRADE,
                relative_path="labels/not-authorized.json",
            )
            if item.semantic is LabelSemantic.DME_GRADE
            else item
        )
        for item in record.labels
    )
    undeclared_path_manifest = DatasetManifestV2(
        dataset_id="idrid",
        dataset_version="fixture-v1",
        records=(replace(record, labels=undeclared_labels),),
    )
    undeclared_path = audit_manifest(undeclared_path_manifest, permit=permit)
    downgraded_labels = tuple(
        (
            _unknown_label(
                LabelSemantic.DR_GRADE,
                relative_path="labels/dr-grade.json",
            )
            if item.semantic is LabelSemantic.DR_GRADE
            else item
        )
        for item in record.labels
    )
    downgraded_declared_label = audit_manifest(
        DatasetManifestV2(
            dataset_id="idrid",
            dataset_version="fixture-v1",
            records=(replace(record, labels=downgraded_labels),),
        ),
        permit=permit,
    )
    missing_declared_label = audit_manifest(
        DatasetManifestV2(
            dataset_id="idrid",
            dataset_version="fixture-v1",
            records=(
                replace(
                    record,
                    labels=tuple(
                        item
                        for item in record.labels
                        if item.semantic is not LabelSemantic.DR_GRADE
                    ),
                ),
            ),
        ),
        permit=permit,
    )
    unauthorized_manifest = DatasetManifestV2(
        dataset_id="idrid",
        dataset_version="fixture-v1",
        records=(replace(record, relative_path="images/not-authorized.png"),),
    )
    wrong_version_manifest = DatasetManifestV2(
        dataset_id="idrid",
        dataset_version="wrong-version",
        records=(record,),
    )
    unauthorized = audit_manifest(unauthorized_manifest, permit=permit)
    wrong_version = audit_manifest(wrong_version_manifest, permit=permit)
    wrong_manifest_type = audit_manifest((), permit=permit)  # type: ignore[arg-type]
    relative_permit = _permit(
        Path("relative-root-that-does-not-exist"),
        label=LabelInputSpec(
            semantic=LabelSemantic.DR_GRADE,
            relative_path=Path("labels/dr-grade.json"),
        ),
    )
    relative_root = audit_manifest(manifest, permit=relative_permit)
    cloned_permit = replace(permit, root=(tmp_path / "retargeted-root").absolute())
    cloned_capability = audit_manifest(manifest, permit=cloned_permit)
    wrong_purpose = _permit(
        Path("root-that-must-not-be-resolved"),
        purpose=DataUsePurpose.METADATA_VALIDATION,
    )
    blocked = audit_manifest(cast(DatasetManifestV2, ()), permit=wrong_purpose)

    assert permitted.status is AuditStatus.FAIL_CLOSED
    assert permitted.dataset_version == "fixture-v1"
    assert permitted.manifest_sha256 is not None
    assert permitted.summary.unknown_label_count == len(DATASET_LABEL_SEMANTICS["idrid"]) - 1
    assert {issue.code for issue in permitted.issues} == {"LABEL_AVAILABILITY_UNKNOWN"}
    assert incomplete_catalog.status is AuditStatus.BLOCKED
    assert undeclared_path.status is AuditStatus.BLOCKED
    assert downgraded_declared_label.status is AuditStatus.BLOCKED
    assert missing_declared_label.status is AuditStatus.BLOCKED
    assert unauthorized.status is AuditStatus.BLOCKED
    assert wrong_version.status is AuditStatus.BLOCKED
    assert wrong_manifest_type.status is AuditStatus.BLOCKED
    assert relative_permit.root.is_absolute()
    assert relative_root.status is AuditStatus.FAIL_CLOSED
    assert {issue.code for issue in relative_root.issues} == {"DATASET_ROOT_INVALID"}
    assert cloned_capability.status is AuditStatus.BLOCKED
    assert cloned_capability.summary.image_files_checked == 0
    assert {issue.code for issue in cloned_capability.issues} == {"ACCESS_GATE_BLOCKED"}
    assert blocked.status is AuditStatus.BLOCKED
    assert blocked.summary.image_files_checked == 0
    assert {issue.code for issue in blocked.issues} == {"ACCESS_GATE_BLOCKED"}


def test_audit_detects_corruption_duplicates_conflicts_metadata_and_leakage(
    tmp_path: Path,
) -> None:
    first = _save_image(tmp_path, "images/a/original.png", color=(10, 20, 30))
    duplicate = tmp_path / "images/b/copy.png"
    duplicate.parent.mkdir(parents=True)
    duplicate.write_bytes(first.read_bytes())
    _save_image(tmp_path, "images/left/same.png", color=(1, 2, 3))
    _save_image(tmp_path, "images/right/same.png", color=(200, 100, 50))
    corrupt = tmp_path / "images/corrupt.png"
    corrupt.write_bytes(b"not-a-decodable-image")

    first_record = _record(
        tmp_path,
        "images/a/original.png",
        labels=(_unknown_label(),),
        source_image_id="shared-source",
        split="train",
        patient_id="shared-patient",
        intervention_pair_id="shared-pair",
    )
    duplicate_record = _record(
        tmp_path,
        "images/b/copy.png",
        labels=(_unknown_label(),),
        source_image_id="shared-source",
        split="test",
        patient_id="shared-patient",
        intervention_pair_id="shared-pair",
    )
    left_record = _record(
        tmp_path,
        "images/left/same.png",
        labels=(_unknown_label(),),
        source_image_id="left",
    )
    right_record = _record(
        tmp_path,
        "images/right/same.png",
        labels=(_unknown_label(),),
        source_image_id="right",
    )
    wrong_metadata = replace(
        left_record,
        width=99,
        channels=1,
        image_format="JPEG",
        sha256="0" * 64,
        byte_size=1,
    )
    corrupt_record = ManifestRecordV2(
        dataset_id="toy_stage2",
        relative_path="images/corrupt.png",
        sha256=sha256_file(corrupt),
        byte_size=corrupt.stat().st_size,
        width=8,
        height=6,
        channels=3,
        image_format="PNG",
        source_image_id="corrupt",
        split="train",
        labels=(_unknown_label(),),
    )
    records = (
        first_record,
        duplicate_record,
        left_record,
        right_record,
        wrong_metadata,
        wrong_metadata,
        corrupt_record,
    )

    first_result = _audit_synthetic_fixture(records, fixture_root=tmp_path)
    second_result = _audit_synthetic_fixture(records, fixture_root=tmp_path)
    codes = {issue.code for issue in first_result.issues}

    assert first_result == second_result
    assert first_result.status is AuditStatus.FAIL_CLOSED
    assert {
        "IMAGE_CONTENT_DUPLICATE",
        "IMAGE_BASENAME_CONFLICT",
        "RELATIVE_PATH_DUPLICATE",
        "IMAGE_DECODE_FAILED",
        "IMAGE_DIMENSION_MISMATCH",
        "IMAGE_CHANNEL_MISMATCH",
        "IMAGE_FORMAT_MISMATCH",
        "IMAGE_HASH_MISMATCH",
        "IMAGE_SIZE_MISMATCH",
        "SPLIT_LEAKAGE",
    }.issubset(codes)
    assert sum(issue.code == "SPLIT_LEAKAGE" for issue in first_result.issues) == 3
    assert str(tmp_path) not in render_audit_json(first_result)


def test_mask_audit_detects_dimensions_channels_values_and_all_zero(tmp_path: Path) -> None:
    _save_image(tmp_path, "images/case.png", size=(8, 6))
    _save_image(tmp_path, "labels/wrong-size.png", mode="L", size=(7, 6), color=17)
    _save_image(tmp_path, "labels/rgb.png", mode="RGB", size=(8, 6), color=(0, 0, 0))
    mixed_path = _save_image(
        tmp_path,
        "labels/unexpected.png",
        mode="L",
        size=(8, 6),
        color=0,
    )
    with Image.open(mixed_path) as mixed:
        editable = mixed.copy()
    editable.putpixel((1, 1), 1)
    editable.putpixel((2, 2), 255)
    editable.save(mixed_path)
    _save_image(tmp_path, "labels/zero.png", mode="L", size=(8, 6), color=0)
    labels = (
        _available_label(
            tmp_path,
            "labels/wrong-size.png",
            LabelSemantic.MICROANEURYSM_MASK,
        ),
        _available_label(tmp_path, "labels/rgb.png", LabelSemantic.HEMORRHAGE_MASK),
        _available_label(
            tmp_path,
            "labels/unexpected.png",
            LabelSemantic.HARD_EXUDATE_MASK,
        ),
        _available_label(tmp_path, "labels/zero.png", LabelSemantic.SOFT_EXUDATE_MASK),
    )
    record = _record(
        tmp_path,
        "images/case.png",
        labels=labels,
        source_image_id="source-1",
    )

    result = _audit_synthetic_fixture((record,), fixture_root=tmp_path)
    codes = {issue.code for issue in result.issues}

    assert result.status is AuditStatus.FAIL_CLOSED
    assert {
        "MASK_DIMENSION_MISMATCH",
        "MASK_CHANNEL_MISMATCH",
        "MASK_ENCODING_MIXED",
        "MASK_MODE_UNEXPECTED",
        "MASK_VALUE_UNEXPECTED",
        "MASK_ALL_ZERO",
    }.issubset(codes)


def test_palette_mask_is_rejected_instead_of_auditing_palette_indices(tmp_path: Path) -> None:
    _save_image(tmp_path, "images/case.png")
    palette_path = tmp_path / "labels/palette.png"
    palette_path.parent.mkdir(parents=True)
    palette = Image.new("P", (8, 6), color=0)
    palette.putpalette([255, 255, 255, 255, 0, 0] + [0, 0, 0] * 254)
    palette.putpixel((1, 1), 1)
    palette.save(palette_path)
    label = _available_label(
        tmp_path,
        "labels/palette.png",
        LabelSemantic.MICROANEURYSM_MASK,
    )
    record = _record(
        tmp_path,
        "images/case.png",
        labels=(label,),
        source_image_id="source-1",
    )

    result = _audit_synthetic_fixture((record,), fixture_root=tmp_path)

    assert result.status is AuditStatus.FAIL_CLOSED
    assert "MASK_MODE_PALETTE" in {issue.code for issue in result.issues}


def test_files_changed_between_hash_and_decode_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = _save_image(tmp_path, "images/case.png", color=(10, 20, 30))
    mask_path = _save_image(
        tmp_path,
        "labels/mask.png",
        mode="L",
        color=255,
    )
    record = _record(
        tmp_path,
        "images/case.png",
        labels=(
            _available_label(
                tmp_path,
                "labels/mask.png",
                LabelSemantic.MICROANEURYSM_MASK,
            ),
        ),
        source_image_id="source-1",
    )
    original_decode = audit_module._decode_image
    changed: set[Path] = set()

    def decode_then_change(path: Path, *, collect_colors: bool) -> object:
        decoded = original_decode(path, collect_colors=collect_colors)
        if path == image_path and path not in changed:
            Image.new("RGB", (8, 6), color=(200, 100, 50)).save(path)
            changed.add(path)
        if path == mask_path and path not in changed:
            Image.new("L", (8, 6), color=1).save(path)
            changed.add(path)
        return decoded

    monkeypatch.setattr(audit_module, "_decode_image", decode_then_change)
    result = _audit_synthetic_fixture((record,), fixture_root=tmp_path)

    assert result.status is AuditStatus.FAIL_CLOSED
    assert {
        "IMAGE_CHANGED_DURING_AUDIT",
        "LABEL_CHANGED_DURING_AUDIT",
    }.issubset({issue.code for issue in result.issues})


def test_reports_are_escaped_deterministic_and_never_clobber(tmp_path: Path) -> None:
    result = blocked_audit_result(
        (
            "license_status must be verified",
            "<img src=x onerror=alert(1)>",
        ),
        dataset_id="ddr",
    )
    json_path = tmp_path / "reports/audit.json"
    html_path = tmp_path / "reports/audit.html"
    boundary = AuditWriteBoundary(allowed_root=tmp_path / "reports")

    write_audit_json(result, json_path, boundary=boundary)
    write_audit_html(result, html_path, boundary=boundary)
    write_audit_json(result, json_path, boundary=boundary)
    write_audit_html(result, html_path, boundary=boundary)
    html = html_path.read_text(encoding="utf-8")

    assert result.status is AuditStatus.BLOCKED
    assert result.summary.record_count == 0
    assert result.summary.image_files_checked == 0
    assert "<img src=x" not in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html
    assert json.loads(json_path.read_text(encoding="utf-8"))["status"] == "BLOCKED"

    json_path.write_text("different", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_audit_json(result, json_path, boundary=boundary)


def test_report_writes_require_an_explicit_non_data_boundary(tmp_path: Path) -> None:
    result = blocked_audit_result(("license review is required",), dataset_id="idrid")
    data_root = tmp_path / "dataset"
    boundary = AuditWriteBoundary(
        allowed_root=tmp_path,
        forbidden_roots=(data_root,),
    )
    target = data_root / "audit.json"

    with pytest.raises(ValueError, match="overlaps a forbidden data root"):
        write_audit_json(result, target, boundary=boundary)
    assert not target.exists()


def test_paired_report_preflight_does_not_publish_half_a_bundle(tmp_path: Path) -> None:
    result = blocked_audit_result(("license review is required",), dataset_id="idrid")
    output_dir = tmp_path / "reports"
    output_dir.mkdir()
    (output_dir / "audit.html").write_text("different", encoding="utf-8")
    boundary = AuditWriteBoundary(allowed_root=output_dir)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_audit_reports(result, output_dir, boundary=boundary)
    assert not (output_dir / "audit.json").exists()


def test_blocked_factory_rejects_path_bearing_reason() -> None:
    with pytest.raises(ValueError, match="must not contain paths"):
        blocked_audit_result((r"license file C:\private\license.txt is missing",))


def test_issue_detail_name_cannot_smuggle_an_absolute_path() -> None:
    with pytest.raises(ValueError, match="normalized identifiers"):
        AuditIssue(
            severity=IssueSeverity.ERROR,
            code="TEST_ISSUE",
            message="Synthetic test issue.",
            details=((r"C:\private\dataset", "redacted"),),
        )
