"""Deterministic synthetic local-data fixture for the Stage 2 audit path."""

from __future__ import annotations

import io
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from retinaentail.data.audit import AuditResult, AuditStatus, _audit_synthetic_fixture
from retinaentail.data.audit_reporting import AuditWriteBoundary, write_audit_reports
from retinaentail.data.manifest import sha256_file
from retinaentail.data.stage2_models import (
    DatasetManifestV2,
    LabelAvailability,
    LabelRecordV2,
    LabelSemantic,
    ManifestRecordV2,
)
from retinaentail.interventions.toy import generate_toy_fundus
from retinaentail.witnesses.replay import (
    canonical_json_bytes,
    rgb_png_bytes,
    write_bytes_fail_safe,
)

_FIXTURE_MARKER_NAME = ".retinaentail-stage2-synthetic-fixture"
_FIXTURE_MARKER_BYTES = b"RetinaEntail-DR synthetic Stage 2 fixture v1\n"


@dataclass(frozen=True, slots=True)
class Stage2FixtureRun:
    """Paths and verified result from one deterministic synthetic audit run."""

    output_dir: Path
    manifest_path: Path
    audit_json_path: Path
    audit_html_path: Path
    package_dir: Path
    audit_result: AuditResult


def _mask_png_bytes(*, size: int, x: int, y: int, width: int, height: int) -> bytes:
    mask = Image.new("L", (size, size), color=0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((x, y, x + width - 1, y + height - 1), fill=255)
    buffer = io.BytesIO()
    mask.save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must contain a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} keys must be strings")
    return value


def _prepare_fixture_output(output_dir: Path) -> None:
    marker = output_dir / _FIXTURE_MARKER_NAME
    if output_dir.exists():
        if not output_dir.is_dir():
            raise NotADirectoryError(output_dir)
        entries = tuple(output_dir.iterdir())
        if entries and not marker.is_file():
            raise FileExistsError(
                "refusing to use a non-empty directory without the synthetic fixture marker"
            )
    write_bytes_fail_safe(marker, _FIXTURE_MARKER_BYTES)


def _run_stage2_fixture_at(output_dir: Path) -> Stage2FixtureRun:
    """Private test boundary for a preselected synthetic-only output location."""

    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a pathlib.Path")

    _prepare_fixture_output(output_dir)

    fixture_root = output_dir / "fixture_data"
    image_path = fixture_root / "images" / "toy_fundus.png"
    mask_path = fixture_root / "labels" / "toy_microaneurysm_mask.png"
    report_dir = output_dir / "reports"
    manifest_path = report_dir / "manifest-v2.json"
    audit_json_path = report_dir / "audit.json"
    audit_html_path = report_dir / "audit.html"
    package_dir = output_dir / "stage2_package"

    twin = generate_toy_fundus(size=64, lesion_radius=3, seed=20260823)
    write_bytes_fail_safe(image_path, rgb_png_bytes(twin.lesion_present))
    write_bytes_fail_safe(
        mask_path,
        _mask_png_bytes(
            size=64,
            x=twin.region.x,
            y=twin.region.y,
            width=twin.region.width,
            height=twin.region.height,
        ),
    )

    label = LabelRecordV2(
        semantic=LabelSemantic.MICROANEURYSM_MASK,
        availability=LabelAvailability.AVAILABLE,
        relative_path="labels/toy_microaneurysm_mask.png",
        sha256=sha256_file(mask_path),
        byte_size=mask_path.stat().st_size,
    )
    record = ManifestRecordV2(
        dataset_id="synthetic_stage2",
        relative_path="images/toy_fundus.png",
        sha256=sha256_file(image_path),
        byte_size=image_path.stat().st_size,
        width=64,
        height=64,
        channels=3,
        image_format="PNG",
        source_image_id="synthetic-stage2-source-001",
        patient_id=None,
        eye_id=None,
        followup_id=None,
        intervention_pair_id=None,
        split="fixture",
        labels=(label,),
    )
    manifest = DatasetManifestV2(
        dataset_id="synthetic_stage2",
        dataset_version="synthetic-fixture-v1",
        records=(record,),
    )
    write_bytes_fail_safe(manifest_path, canonical_json_bytes(manifest.to_dict()))

    decoded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reloaded_manifest = DatasetManifestV2.from_dict(_mapping(decoded_manifest, "fixture manifest"))
    if reloaded_manifest != manifest:
        raise RuntimeError("serialized fixture manifest failed its round-trip check")

    result = _audit_synthetic_fixture(reloaded_manifest, fixture_root=fixture_root)
    if result.status is not AuditStatus.PASS:
        raise RuntimeError("the clean synthetic Stage 2 fixture failed closed")
    write_audit_reports(
        result,
        report_dir,
        boundary=AuditWriteBoundary(
            allowed_root=output_dir,
            forbidden_roots=(fixture_root,),
        ),
        title="RetinaEntail-DR synthetic Stage 2 data audit",
    )

    decoded_audit = json.loads(audit_json_path.read_text(encoding="utf-8"))
    if _mapping(decoded_audit, "fixture audit").get("status") != AuditStatus.PASS.value:
        raise RuntimeError("serialized fixture audit did not preserve PASS status")
    write_bytes_fail_safe(package_dir / "manifest-v2.json", manifest_path.read_bytes())
    write_bytes_fail_safe(package_dir / "audit.json", audit_json_path.read_bytes())
    return Stage2FixtureRun(
        output_dir=output_dir,
        manifest_path=manifest_path,
        audit_json_path=audit_json_path,
        audit_html_path=audit_html_path,
        package_dir=package_dir,
        audit_result=result,
    )


def _trusted_outputs_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "configs").is_dir():
            return (candidate / "outputs").absolute()
    raise RuntimeError("could not locate the trusted RetinaEntail-DR outputs directory")


def run_stage2_fixture(output_dir: Path) -> Stage2FixtureRun:
    """Generate a fixture only inside this project's ignored ``outputs`` tree."""

    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a pathlib.Path")
    base = _trusted_outputs_root()
    candidate = output_dir.absolute()
    trusted_project = base.parent.resolve(strict=True)
    resolved_base = base.resolve(strict=False)
    try:
        resolved_base.relative_to(trusted_project)
        candidate.relative_to(base)
        candidate.resolve(strict=False).relative_to(resolved_base)
    except ValueError as error:
        raise ValueError(f"fixture output must stay under {base}") from error
    return _run_stage2_fixture_at(candidate)


__all__ = ["Stage2FixtureRun", "run_stage2_fixture"]
