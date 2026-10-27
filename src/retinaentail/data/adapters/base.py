"""Shared explicit-entry implementation for read-only Stage 2 dataset adapters."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

from PIL import Image

from retinaentail.data.license_gate import (
    DatasetAccessPermit,
    DataUsePurpose,
    permit_validation_reasons,
)
from retinaentail.data.manifest import sha256_file
from retinaentail.data.stage2_models import (
    DatasetEntrySpec,
    DatasetManifestV2,
    LabelAvailability,
    LabelRecordV2,
    LabelSemantic,
    ManifestRecordV2,
)


class AdapterContractError(ValueError):
    """Raised when an explicit adapter specification cannot be verified safely."""


def _resolve_declared_file(root: Path, relative_path: Path, description: str) -> Path:
    try:
        candidate = (root / relative_path).resolve(strict=True)
    except OSError as error:
        raise AdapterContractError(
            f"{description} does not resolve to a declared file: {relative_path}"
        ) from error
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise AdapterContractError(f"{description} escapes the authorized dataset root") from error
    if not candidate.is_file():
        raise AdapterContractError(f"{description} is not a regular file: {relative_path}")
    return candidate


class ExplicitDatasetAdapter:
    """Base adapter that reads only paths enumerated by an authorized explicit spec."""

    DATASET_ID: ClassVar[str]
    LABEL_SEMANTICS: ClassVar[frozenset[LabelSemantic]]

    def __init__(self, permit: DatasetAccessPermit) -> None:
        if not isinstance(permit, DatasetAccessPermit):
            raise TypeError("permit must be a DatasetAccessPermit")
        permit_failures = permit_validation_reasons(permit)
        if permit_failures:
            raise AdapterContractError(f"invalid access capability: {permit_failures[0]}")
        if permit.purpose is not DataUsePurpose.READ_ONLY_AUDIT:
            raise AdapterContractError(
                "dataset adapters require a read_only_audit access capability"
            )
        if (
            permit.registry.dataset_id != self.DATASET_ID
            or permit.spec.dataset_id != self.DATASET_ID
        ):
            raise AdapterContractError(
                f"{type(self).__name__} requires dataset_id {self.DATASET_ID!r}"
            )
        self._effective_label_semantics = permit.spec.effective_label_semantics
        for entry in permit.spec.entries:
            unsupported = {label.semantic for label in entry.labels}.difference(
                self._effective_label_semantics
            )
            if unsupported:
                names = ", ".join(sorted(semantic.value for semantic in unsupported))
                raise AdapterContractError(
                    f"unsupported label semantic(s) for {self.DATASET_ID}: {names}"
                )
        self._permit = permit

    @property
    def dataset_id(self) -> str:
        return self.DATASET_ID

    @property
    def supported_label_semantics(self) -> frozenset[LabelSemantic]:
        return self._effective_label_semantics

    def _authorized_root(self) -> Path:
        try:
            root = self._permit.root.resolve(strict=True)
        except OSError as error:
            raise AdapterContractError(
                f"authorized root does not exist: {self._permit.root}"
            ) from error
        if not root.is_dir():
            raise AdapterContractError(f"authorized root is not a directory: {self._permit.root}")
        return root

    def _labels(self, root: Path, entry: DatasetEntrySpec) -> tuple[LabelRecordV2, ...]:
        supplied = {label.semantic: label for label in entry.labels}
        records: list[LabelRecordV2] = []
        for semantic in sorted(self._effective_label_semantics, key=lambda item: item.value):
            label_spec = supplied.get(semantic)
            if label_spec is None or label_spec.relative_path is None:
                records.append(
                    LabelRecordV2(
                        semantic=semantic,
                        availability=LabelAvailability.UNKNOWN,
                    )
                )
                continue
            label_path = _resolve_declared_file(
                root,
                label_spec.relative_path,
                f"{semantic.value} label",
            )
            records.append(
                LabelRecordV2(
                    semantic=semantic,
                    availability=LabelAvailability.AVAILABLE,
                    relative_path=label_spec.relative_path.as_posix(),
                    sha256=sha256_file(label_path),
                    byte_size=label_path.stat().st_size,
                )
            )
        return tuple(records)

    def _record(self, root: Path, entry: DatasetEntrySpec) -> ManifestRecordV2:
        image_path = _resolve_declared_file(root, entry.image_path, "image")
        with Image.open(image_path) as image:
            width, height = image.size
            channels = len(image.getbands())
            image_format = image.format
        if image_format is None:
            raise AdapterContractError(f"image format could not be determined: {entry.image_path}")
        return ManifestRecordV2(
            dataset_id=self.DATASET_ID,
            relative_path=entry.image_path.as_posix(),
            sha256=sha256_file(image_path),
            byte_size=image_path.stat().st_size,
            width=width,
            height=height,
            channels=channels,
            image_format=image_format,
            source_image_id=entry.source_image_id,
            patient_id=entry.patient_id,
            eye_id=entry.eye_id,
            followup_id=entry.followup_id,
            intervention_pair_id=entry.intervention_pair_id,
            split=entry.split,
            labels=self._labels(root, entry),
        )

    def iter_manifest(self) -> Iterator[ManifestRecordV2]:
        """Yield records in spec order; never scan or infer dataset contents."""

        root = self._authorized_root()
        for entry in self._permit.spec.entries:
            yield self._record(root, entry)

    def build_manifest(self) -> tuple[ManifestRecordV2, ...]:
        return tuple(self.iter_manifest())

    def build_versioned_manifest(self) -> DatasetManifestV2:
        """Build a deterministic versioned envelope for this registry release."""

        return DatasetManifestV2(
            dataset_id=self.DATASET_ID,
            dataset_version=self._permit.registry.dataset_version,
            records=self.build_manifest(),
        )


__all__ = ["AdapterContractError", "ExplicitDatasetAdapter"]
