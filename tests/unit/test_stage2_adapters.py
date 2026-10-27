from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from retinaentail.data.adapters import (
    AdapterContractError,
    DDRAdapter,
    DeepDRiDAdapter,
    IDRiDAdapter,
)
from retinaentail.data.adapters.base import ExplicitDatasetAdapter
from retinaentail.data.license_gate import (
    DatasetAccessPermit,
    DataUsePurpose,
    LicenseGateError,
    LicenseReviewDecision,
    LicenseReviewRecord,
    RegistryLicenseMetadata,
    assess_dataset_access,
    authorize_dataset_access,
    load_license_review_record,
    load_registry_license_metadata,
    permit_validation_reasons,
)
from retinaentail.data.stage2_models import (
    DATASET_LABEL_SEMANTICS,
    DatasetAdapterSpec,
    DatasetEntrySpec,
    DatasetManifestV2,
    LabelAvailability,
    LabelInputSpec,
    LabelSemantic,
    Stage2ContractError,
    load_dataset_adapter_spec,
)


def _verified_registry(dataset_id: str) -> RegistryLicenseMetadata:
    return RegistryLicenseMetadata(
        schema_version="1.0.0",
        dataset_id=dataset_id,
        dataset_version="synthetic-layout-v1",
        license_status="verified",
        license_review_status="reviewed",
        label_verification_status="verified",
    )


def _approved_review(
    dataset_id: str,
    *,
    dataset_version: str = "synthetic-layout-v1",
    decision: LicenseReviewDecision = LicenseReviewDecision.APPROVED,
    purposes: tuple[DataUsePurpose, ...] = (DataUsePurpose.READ_ONLY_AUDIT,),
) -> LicenseReviewRecord:
    return LicenseReviewRecord(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        decision=decision,
        permitted_purposes=purposes,
        terms_sha256="a" * 64,
        evidence_sha256="b" * 64,
        reviewer_id="synthetic-test-reviewer",
        reviewed_at="2026-08-23T12:00:00+00:00",
    )


def _spec(
    dataset_id: str,
    *,
    image_path: Path = Path("images/sample.png"),
    label: LabelInputSpec | None = None,
) -> DatasetAdapterSpec:
    return DatasetAdapterSpec(
        dataset_id=dataset_id,
        entries=(
            DatasetEntrySpec(
                image_path=image_path,
                source_image_id="explicit-source-id",
                patient_id=None,
                eye_id="right-eye-explicit",
                followup_id=None,
                intervention_pair_id="explicit-pair-id",
                split="train",
                labels=(() if label is None else (label,)),
            ),
        ),
    )


def _write_fixture(root: Path) -> tuple[bytes, bytes]:
    image_path = root / "images" / "sample.png"
    label_path = root / "labels" / "label.png"
    image_path.parent.mkdir(parents=True)
    label_path.parent.mkdir(parents=True)
    Image.new("RGB", (11, 7), color=(10, 20, 30)).save(image_path)
    Image.new("L", (11, 7), color=255).save(label_path)
    return image_path.read_bytes(), label_path.read_bytes()


def test_all_current_registry_entries_are_blocked_pre_io() -> None:
    dataset_ids = (
        "idrid",
        "ddr",
        "deepdrid",
        "eyeq",
        "real_fundus",
        "fgadr",
        "maples_dr",
        "fives",
        "hrf_quality",
    )
    for dataset_id in dataset_ids:
        metadata = load_registry_license_metadata(
            Path("data_registry/datasets") / f"{dataset_id}.yaml"
        )
        decision = assess_dataset_access(
            metadata,
            review=None,
            root_provided=True,
            spec_provided=True,
        )
        assert decision.permitted is False
        assert decision.reasons


def test_denied_deepdrid_review_never_touches_dataset_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = load_registry_license_metadata(Path("data_registry/datasets/deepdrid.yaml"))
    spec = _spec("deepdrid")
    touched: list[str] = []

    def trap_stat(self: Path, *args: object, **kwargs: object) -> object:
        touched.append(str(self))
        raise AssertionError("blocked gate touched dataset root")

    def trap_glob(self: Path, pattern: str) -> object:
        touched.append(f"{self}:{pattern}")
        raise AssertionError("blocked gate globbed dataset root")

    monkeypatch.setattr(Path, "stat", trap_stat)
    monkeypatch.setattr(Path, "glob", trap_glob)
    with pytest.raises(LicenseGateError, match="license review decision must be approved"):
        authorize_dataset_access(
            metadata,
            review=_approved_review(
                "deepdrid",
                dataset_version=metadata.dataset_version,
                decision=LicenseReviewDecision.DENIED,
            ),
            root=Path("must-not-be-touched"),
            spec=spec,
        )
    assert touched == []


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("license_status", "ambiguous", "license_status"),
        ("license_review_status", "pending_review", "license_review_status"),
        ("label_verification_status", "pending_review", "label_verification_status"),
    ],
)
def test_each_registry_status_is_independently_fail_closed(
    field: str, value: str, reason: str
) -> None:
    values = {
        "schema_version": "1.0.0",
        "dataset_id": "idrid",
        "dataset_version": "synthetic-layout-v1",
        "license_status": "verified",
        "license_review_status": "reviewed",
        "label_verification_status": "verified",
    }
    values[field] = value
    metadata = RegistryLicenseMetadata(**values)
    decision = assess_dataset_access(
        metadata,
        review=_approved_review("idrid"),
        root_provided=True,
        spec_provided=True,
    )
    assert decision.permitted is False
    assert any(reason in item for item in decision.reasons)


def test_explicit_root_and_spec_are_both_required() -> None:
    metadata = _verified_registry("idrid")
    review = _approved_review("idrid")
    spec = _spec("idrid")
    with pytest.raises(LicenseGateError, match="Path root"):
        authorize_dataset_access(metadata, review=review, root=None, spec=spec)
    with pytest.raises(LicenseGateError, match="DatasetAdapterSpec"):
        authorize_dataset_access(metadata, review=review, root=Path("unused"), spec=None)


def test_registry_status_alone_cannot_issue_access() -> None:
    metadata = _verified_registry("idrid")
    with pytest.raises(LicenseGateError, match="LicenseReviewRecord"):
        authorize_dataset_access(
            metadata,
            review=None,
            root=Path("unused"),
            spec=_spec("idrid"),
        )


@pytest.mark.parametrize(
    ("review", "reason"),
    [
        (_approved_review("ddr"), "review dataset_id"),
        (
            _approved_review("idrid", dataset_version="different-version"),
            "dataset_version",
        ),
        (
            _approved_review("idrid", decision=LicenseReviewDecision.DENIED),
            "decision must be approved",
        ),
        (
            _approved_review(
                "idrid",
                purposes=(DataUsePurpose.METADATA_VALIDATION,),
            ),
            "must permit purpose read_only_audit",
        ),
    ],
)
def test_review_record_must_match_dataset_version_decision_and_purpose(
    review: LicenseReviewRecord,
    reason: str,
) -> None:
    with pytest.raises(LicenseGateError, match=reason):
        authorize_dataset_access(
            _verified_registry("idrid"),
            review=review,
            root=Path("unused"),
            spec=_spec("idrid"),
        )


def test_review_record_requires_strict_hashes_and_timezone() -> None:
    with pytest.raises(ValueError, match="terms_sha256"):
        LicenseReviewRecord(
            dataset_id="idrid",
            dataset_version="synthetic-layout-v1",
            decision=LicenseReviewDecision.APPROVED,
            permitted_purposes=(DataUsePurpose.READ_ONLY_AUDIT,),
            terms_sha256="not-a-hash",
            evidence_sha256="b" * 64,
            reviewer_id="reviewer",
            reviewed_at="2026-08-23T12:00:00+00:00",
        )
    with pytest.raises(ValueError, match="RFC 3339"):
        LicenseReviewRecord(
            dataset_id="idrid",
            dataset_version="synthetic-layout-v1",
            decision=LicenseReviewDecision.APPROVED,
            permitted_purposes=(DataUsePurpose.READ_ONLY_AUDIT,),
            terms_sha256="a" * 64,
            evidence_sha256="b" * 64,
            reviewer_id="reviewer",
            reviewed_at="2026-08-23T12:00:00",
        )
    with pytest.raises(ValueError, match="RFC 3339"):
        LicenseReviewRecord(
            dataset_id="idrid",
            dataset_version="synthetic-layout-v1",
            decision=LicenseReviewDecision.APPROVED,
            permitted_purposes=(DataUsePurpose.READ_ONLY_AUDIT,),
            terms_sha256="a" * 64,
            evidence_sha256="b" * 64,
            reviewer_id="reviewer",
            reviewed_at="2026-08-23 12:00:00+00:00",
        )


def test_dataset_versions_must_be_report_safe() -> None:
    with pytest.raises(ValueError, match="report-safe"):
        RegistryLicenseMetadata(
            schema_version="1.0.0",
            dataset_id="idrid",
            dataset_version="release/1",
            license_status="verified",
            license_review_status="reviewed",
            label_verification_status="verified",
        )
    with pytest.raises(ValueError, match="report-safe"):
        _approved_review("idrid", dataset_version="release/1")


def test_versioned_review_record_loader_is_strict(tmp_path: Path) -> None:
    review = _approved_review("idrid")
    path = tmp_path / "review.json"
    path.write_text(json.dumps(review.to_dict()), encoding="utf-8")
    assert load_license_review_record(path) == review

    invalid = review.to_dict()
    del invalid["evidence_sha256"]
    path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="missing field"):
        load_license_review_record(path)


def test_access_permit_cannot_be_constructed_without_gate_token() -> None:
    with pytest.raises(TypeError, match="only be issued"):
        DatasetAccessPermit(
            registry=_verified_registry("idrid"),
            review=_approved_review("idrid"),
            purpose=DataUsePurpose.READ_ONLY_AUDIT,
            root=Path("unused"),
            spec=_spec("idrid"),
            _token=object(),
        )


@pytest.mark.parametrize(
    ("adapter_type", "dataset_id", "available_semantic"),
    [
        (IDRiDAdapter, "idrid", LabelSemantic.MICROANEURYSM_MASK),
        (DDRAdapter, "ddr", LabelSemantic.LESION_DETECTION),
        (DeepDRiDAdapter, "deepdrid", LabelSemantic.IMAGE_QUALITY_GRADE),
    ],
)
def test_dataset_adapters_build_complete_read_only_versioned_manifests(
    tmp_path: Path,
    adapter_type: type[ExplicitDatasetAdapter],
    dataset_id: str,
    available_semantic: LabelSemantic,
) -> None:
    image_before, label_before = _write_fixture(tmp_path)
    spec = _spec(
        dataset_id,
        label=LabelInputSpec(
            semantic=available_semantic,
            relative_path=Path("labels/label.png"),
        ),
    )
    permit = authorize_dataset_access(
        _verified_registry(dataset_id),
        review=_approved_review(dataset_id),
        root=tmp_path,
        spec=spec,
    )
    adapter = adapter_type(permit)
    manifest = adapter.build_versioned_manifest()

    assert manifest.schema_version == "2.0.0"
    assert manifest.dataset_id == dataset_id
    assert manifest.dataset_version == "synthetic-layout-v1"
    assert len(manifest.records) == 1
    record = manifest.records[0]
    assert record.relative_path == "images/sample.png"
    assert record.byte_size == len(image_before)
    assert (record.width, record.height, record.channels, record.image_format) == (
        11,
        7,
        3,
        "PNG",
    )
    assert record.source_image_id == "explicit-source-id"
    assert record.patient_id is None
    assert record.eye_id == "right-eye-explicit"
    assert record.followup_id is None
    assert record.intervention_pair_id == "explicit-pair-id"

    labels = {label.semantic: label for label in record.labels}
    assert set(labels) == adapter.supported_label_semantics
    assert adapter.supported_label_semantics == DATASET_LABEL_SEMANTICS[dataset_id]
    available = labels[available_semantic]
    assert available.availability is LabelAvailability.AVAILABLE
    assert available.relative_path == "labels/label.png"
    assert available.byte_size == len(label_before)
    for semantic, label in labels.items():
        if semantic is not available_semantic:
            assert label.availability is LabelAvailability.UNKNOWN
            assert label.relative_path is None
            assert label.sha256 is None
            assert label.byte_size is None

    payload = json.loads(json.dumps(manifest.to_dict(), sort_keys=True))
    assert DatasetManifestV2.from_dict(payload) == manifest
    assert (tmp_path / "images/sample.png").read_bytes() == image_before
    assert (tmp_path / "labels/label.png").read_bytes() == label_before


def test_unsupported_semantic_is_rejected_before_root_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(
        "idrid",
        label=LabelInputSpec(
            semantic=LabelSemantic.IMAGE_QUALITY_GRADE,
            relative_path=Path("labels/quality.csv"),
        ),
    )
    permit = authorize_dataset_access(
        _verified_registry("idrid"),
        review=_approved_review("idrid"),
        root=Path("must-not-be-touched"),
        spec=spec,
    )
    touched: list[str] = []

    def trap_stat(self: Path, *args: object, **kwargs: object) -> object:
        touched.append(str(self))
        raise AssertionError("adapter validation touched root")

    monkeypatch.setattr(Path, "stat", trap_stat)
    with pytest.raises(AdapterContractError, match="unsupported label semantic"):
        IDRiDAdapter(permit)
    assert touched == []


def test_calibration_only_permit_cannot_instantiate_adapter_or_touch_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    permit = authorize_dataset_access(
        _verified_registry("idrid"),
        review=_approved_review(
            "idrid",
            purposes=(DataUsePurpose.LESION_CALIBRATION,),
        ),
        purpose=DataUsePurpose.LESION_CALIBRATION,
        root=Path("must-not-be-touched"),
        spec=_spec("idrid"),
    )
    touched: list[str] = []

    def trap_stat(self: Path, *args: object, **kwargs: object) -> object:
        touched.append(str(self))
        raise AssertionError("adapter validation touched root")

    monkeypatch.setattr(Path, "stat", trap_stat)
    with pytest.raises(AdapterContractError, match="read_only_audit"):
        IDRiDAdapter(permit)
    assert touched == []


def test_declared_missing_label_file_fails_closed(tmp_path: Path) -> None:
    image_path = tmp_path / "images" / "sample.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (4, 4)).save(image_path)
    spec = _spec(
        "idrid",
        label=LabelInputSpec(
            semantic=LabelSemantic.DR_GRADE,
            relative_path=Path("labels/missing.csv"),
        ),
    )
    adapter = IDRiDAdapter(
        authorize_dataset_access(
            _verified_registry("idrid"),
            review=_approved_review("idrid"),
            root=tmp_path,
            spec=spec,
        )
    )
    with pytest.raises(AdapterContractError, match="does not resolve"):
        adapter.build_manifest()


def test_paths_cannot_escape_explicit_dataset_root() -> None:
    with pytest.raises(Stage2ContractError, match="relative"):
        DatasetEntrySpec(
            image_path=Path("../outside.png"),
            source_image_id="explicit",
            split="train",
        )


@pytest.mark.parametrize(
    "invalid_path",
    [r"images\case.png", "./images/case.png", "images//case.png", "images/case.png/"],
)
def test_spec_paths_must_be_normalized_posix_strings(invalid_path: str) -> None:
    with pytest.raises(Stage2ContractError, match="normalized|POSIX"):
        DatasetEntrySpec(
            image_path=invalid_path,  # type: ignore[arg-type]
            source_image_id="explicit",
            split="train",
        )


def test_adapter_spec_rejects_unknown_dataset_and_duplicate_image_paths() -> None:
    entry = DatasetEntrySpec(
        image_path=Path("images/case.png"),
        source_image_id="explicit",
        split="train",
    )
    with pytest.raises(Stage2ContractError, match="no registered Stage 2 adapter"):
        DatasetAdapterSpec(dataset_id="toy_stage2", entries=(entry,))
    with pytest.raises(Stage2ContractError, match="duplicate image_path"):
        DatasetAdapterSpec(dataset_id="idrid", entries=(entry, entry))


def test_explicit_task_label_scope_is_strict_and_roundtrips() -> None:
    label = LabelInputSpec(
        semantic=LabelSemantic.HEMORRHAGE_MASK,
        relative_path=Path("labels/he.tif"),
    )
    spec = DatasetAdapterSpec(
        dataset_id="idrid",
        label_semantics=(
            LabelSemantic.HARD_EXUDATE_MASK,
            LabelSemantic.HEMORRHAGE_MASK,
        ),
        entries=(
            DatasetEntrySpec(
                image_path=Path("images/case.jpg"),
                source_image_id="published-case-key",
                split="TRAIN",
                labels=(label,),
            ),
        ),
    )
    assert spec.effective_label_semantics == {
        LabelSemantic.HEMORRHAGE_MASK,
        LabelSemantic.HARD_EXUDATE_MASK,
    }
    assert DatasetAdapterSpec.from_dict(spec.to_dict()) == spec

    with pytest.raises(Stage2ContractError, match="duplicates"):
        DatasetAdapterSpec(
            dataset_id="idrid",
            label_semantics=(
                LabelSemantic.HEMORRHAGE_MASK,
                LabelSemantic.HEMORRHAGE_MASK,
            ),
            entries=spec.entries,
        )
    with pytest.raises(Stage2ContractError, match="unsupported"):
        DatasetAdapterSpec(
            dataset_id="idrid",
            label_semantics=(LabelSemantic.IMAGE_QUALITY_GRADE,),
            entries=spec.entries,
        )


def test_idrid_adapter_emits_only_authorized_task_label_catalog(tmp_path: Path) -> None:
    image_path = tmp_path / "images" / "case.png"
    he_path = tmp_path / "labels" / "he.png"
    image_path.parent.mkdir(parents=True)
    he_path.parent.mkdir(parents=True)
    Image.new("RGB", (8, 6)).save(image_path)
    Image.new("L", (8, 6), color=255).save(he_path)
    spec = DatasetAdapterSpec(
        dataset_id="idrid",
        label_semantics=(LabelSemantic.HEMORRHAGE_MASK,),
        entries=(
            DatasetEntrySpec(
                image_path=Path("images/case.png"),
                source_image_id="published-case-key",
                split="TRAIN",
                labels=(
                    LabelInputSpec(
                        semantic=LabelSemantic.HEMORRHAGE_MASK,
                        relative_path=Path("labels/he.png"),
                    ),
                ),
            ),
        ),
    )
    adapter = IDRiDAdapter(
        authorize_dataset_access(
            _verified_registry("idrid"),
            review=_approved_review("idrid"),
            root=tmp_path,
            spec=spec,
        )
    )
    manifest = adapter.build_versioned_manifest()
    assert adapter.supported_label_semantics == {LabelSemantic.HEMORRHAGE_MASK}
    assert [label.semantic for label in manifest.records[0].labels] == [
        LabelSemantic.HEMORRHAGE_MASK
    ]


def test_authorized_relative_root_is_frozen_as_absolute_without_stat() -> None:
    permit = authorize_dataset_access(
        _verified_registry("idrid"),
        review=_approved_review("idrid"),
        root=Path("relative-root-not-read-at-gate"),
        spec=_spec("idrid"),
    )
    assert permit.root.is_absolute()


def test_dataclass_replace_cannot_clone_or_retarget_an_issued_capability(
    tmp_path: Path,
) -> None:
    permit = authorize_dataset_access(
        _verified_registry("idrid"),
        review=_approved_review("idrid"),
        root=tmp_path,
        spec=_spec("idrid"),
    )
    cloned = replace(permit, root=(tmp_path / "different-root").absolute())

    with pytest.raises(AdapterContractError, match="was not issued by the license gate"):
        IDRiDAdapter(cloned)


def _assert_mutated_permit_is_rejected(permit: DatasetAccessPermit) -> None:
    assert permit_validation_reasons(permit) == (
        "access capability bindings changed after issuance",
    )
    with pytest.raises(AdapterContractError, match="bindings changed after issuance"):
        IDRiDAdapter(permit)


def test_access_permit_detects_nested_registry_mutation(tmp_path: Path) -> None:
    permit = authorize_dataset_access(
        _verified_registry("idrid"),
        review=_approved_review("idrid"),
        root=tmp_path,
        spec=_spec("idrid"),
    )

    object.__setattr__(permit.registry, "license_status", "restricted")

    _assert_mutated_permit_is_rejected(permit)


def test_access_permit_detects_nested_review_mutation(tmp_path: Path) -> None:
    permit = authorize_dataset_access(
        _verified_registry("idrid"),
        review=_approved_review("idrid"),
        root=tmp_path,
        spec=_spec("idrid"),
    )

    object.__setattr__(permit.review, "terms_sha256", "0" * 64)

    _assert_mutated_permit_is_rejected(permit)


def test_access_permit_detects_deep_adapter_spec_mutation(tmp_path: Path) -> None:
    permit = authorize_dataset_access(
        _verified_registry("idrid"),
        review=_approved_review("idrid"),
        root=tmp_path,
        spec=_spec("idrid"),
    )

    object.__setattr__(permit.spec.entries[0], "source_image_id", "tampered-source-id")

    _assert_mutated_permit_is_rejected(permit)


@pytest.mark.parametrize("field_name", ["root", "purpose"])
def test_access_permit_detects_top_level_binding_mutation(
    tmp_path: Path,
    field_name: str,
) -> None:
    permit = authorize_dataset_access(
        _verified_registry("idrid"),
        review=_approved_review("idrid"),
        root=tmp_path,
        spec=_spec("idrid"),
    )
    replacement: object = (
        (tmp_path / "retargeted-root").absolute()
        if field_name == "root"
        else DataUsePurpose.METADATA_VALIDATION
    )

    object.__setattr__(permit, field_name, replacement)

    _assert_mutated_permit_is_rejected(permit)


def test_json_spec_loader_is_strict(tmp_path: Path) -> None:
    spec = _spec("deepdrid")
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec.to_dict()), encoding="utf-8")
    assert load_dataset_adapter_spec(path) == spec

    invalid = spec.to_dict()
    invalid["unexpected"] = True
    path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(Stage2ContractError, match="unknown field"):
        load_dataset_adapter_spec(path)
