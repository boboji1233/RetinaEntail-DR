from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from PIL import Image

from retinaentail.data.manifest import (
    ManifestIndexer,
    SplitLeakageError,
    assert_no_split_leakage,
    detect_split_leakage,
    sha256_file,
)
from retinaentail.schemas import ImageRecord

HASH = "0" * 64


def _record(
    *,
    relative_path: str,
    source_image_id: str,
    split: str,
    patient_id: str | None = None,
    intervention_pair_id: str | None = None,
) -> ImageRecord:
    return ImageRecord(
        relative_path=relative_path,
        sha256=HASH,
        width=8,
        height=8,
        channels=3,
        source_image_id=source_image_id,
        patient_id=patient_id,
        split=split,
        intervention_pair_id=intervention_pair_id,
    )


def test_sha256_file_streams_correctly(tmp_path: Path) -> None:
    payload = bytes(range(256)) * 8193
    path = tmp_path / "payload.bin"
    path.write_bytes(payload)
    assert sha256_file(path, chunk_size=257) == hashlib.sha256(payload).hexdigest()


def test_manifest_indexer_records_relative_metadata(tmp_path: Path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    image_path = images / "case.png"
    label_path = labels / "case-mask.png"
    Image.new("RGB", (13, 7), color=(20, 40, 60)).save(image_path)
    Image.new("L", (13, 7), color=0).save(label_path)

    record = ManifestIndexer(tmp_path, "toy").index(
        Path("images/case.png"),
        source_image_id="source-1",
        patient_id=None,
        label_path=Path("labels/case-mask.png"),
        split="train",
        intervention_pair_id="pair-1",
    )

    assert record.relative_path == "images/case.png"
    assert record.label_path == "labels/case-mask.png"
    assert (record.width, record.height, record.channels) == (13, 7, 3)
    assert record.sha256 == sha256_file(image_path)
    assert record.dataset_id == "toy"


def test_indexer_rejects_paths_outside_dataset_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.png"
    Image.new("RGB", (2, 2)).save(outside)
    with pytest.raises(ValueError, match="outside manifest root"):
        ManifestIndexer(root).index(outside, source_image_id="source")


def test_all_protected_identifiers_are_checked_for_split_leakage() -> None:
    records = [
        _record(
            relative_path="a.png",
            source_image_id="same-source",
            patient_id="same-patient",
            intervention_pair_id="same-pair",
            split="train",
        ),
        _record(
            relative_path="b.png",
            source_image_id="same-source",
            patient_id="same-patient",
            intervention_pair_id="same-pair",
            split="test",
        ),
    ]

    leakages = detect_split_leakage(records)
    assert {leakage.identifier_type for leakage in leakages} == {
        "source_image_id",
        "patient_id",
        "intervention_pair_id",
    }
    assert all(leakage.splits == ("test", "train") for leakage in leakages)
    with pytest.raises(SplitLeakageError) as error:
        assert_no_split_leakage(records)
    assert error.value.leakages == leakages


def test_records_in_one_split_pass_isolation_check() -> None:
    records = [
        _record(relative_path="a.png", source_image_id="a", split="train"),
        _record(relative_path="b.png", source_image_id="b", split="train"),
    ]
    assert detect_split_leakage(records) == ()
    assert_no_split_leakage(records)
