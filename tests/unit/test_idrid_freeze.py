from __future__ import annotations

from pathlib import Path

from retinaentail.data.idrid_freeze import freeze_idrid_he_ex_spec
from retinaentail.data.stage2_models import LabelSemantic


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture")


def test_freeze_idrid_he_ex_spec_keeps_missing_mask_unknown_and_excluded(
    tmp_path: Path,
) -> None:
    for index in range(1, 82):
        training = index <= 54
        split_name = "a. Training Set" if training else "b. Testing Set"
        image = (
            tmp_path
            / "A. Segmentation"
            / "1. Original Images"
            / split_name
            / f"IDRiD_{index:02d}.jpg"
        )
        masks = tmp_path / "A. Segmentation" / "2. All Segmentation Groundtruths" / split_name
        _touch(image)
        if index != 4:
            _touch(masks / "2. Haemorrhages" / f"IDRiD_{index:02d}_HE.tif")
        _touch(masks / "3. Hard Exudates" / f"IDRiD_{index:02d}_EX.tif")

    summary = freeze_idrid_he_ex_spec(tmp_path)

    assert summary.published_image_count == 81
    assert summary.included_count == 80
    assert summary.excluded_source_image_ids == ("idrid-segmentation-IDRiD_04",)
    assert dict(summary.split_counts)["LOCKED_TEST"] == 27
    assert set(summary.spec.effective_label_semantics) == {
        LabelSemantic.HEMORRHAGE_MASK,
        LabelSemantic.HARD_EXUDATE_MASK,
    }
    assert all(
        entry.patient_id is None
        and entry.eye_id is None
        and entry.followup_id is None
        and entry.intervention_pair_id is None
        for entry in summary.spec.entries
    )
    assert all(len(entry.labels) == 2 for entry in summary.spec.entries)
