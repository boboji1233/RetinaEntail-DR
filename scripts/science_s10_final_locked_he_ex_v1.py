"""Run the sole preregistered final IDRiD locked HE/EX controlled-twin evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import platform
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

import numpy as np
import science_s7_lite_v7 as core
import science_s7_lite_v8 as v8
import science_s8_controlled_truth_v1 as s8
import science_s10_metric_recovery_v1 as recovery
import torch
import yaml
from numpy.typing import NDArray

import retinaentail.science_s4_twins as s4
from retinaentail.data.lesion_catalog import (
    FOVGeometry,
    SparseComponent,
    _component_touches_fov,
    _ellipse_boundary_distance,
    _load_rgb_strict,
    _resolve_under_root,
    connected_components,
    decode_idrid_mask,
    detect_fov,
    patch_geometry,
)
from retinaentail.data.manifest import sha256_file
from retinaentail.data.stage2_models import DatasetManifestV2, LabelRecordV2, ManifestRecordV2
from retinaentail.interventions.controlled_twin_pilot import QCCell, TwinRejected, _contact_sheet
from retinaentail.science_s7_lite import _cluster_bootstrap_fcr
from retinaentail.science_s8_controlled_truth import wilson_interval
from retinaentail.witnesses.replay import canonical_json_bytes, write_bytes_fail_safe

SCHEMA_VERSION = "1.0.0"
EXPERIMENT_ID = "S10_FINAL_LOCKED_HE_EX_V1"
DETERMINATE = {"ENTAILED_PRESENT", "ENTAILED_ABSENT"}
LESION_CLASSES = ("HE", "EX")
LABEL_SEMANTICS = {"HE": "HEMORRHAGE_MASK", "EX": "HARD_EXUDATE_MASK"}
MODEL_BINDINGS = tuple(f"model_seed_{seed}" for seed in recovery.DEFAULT_SEEDS)


class S10FinalError(RuntimeError):
    """Raised when the final locked evaluation must fail closed."""


@dataclass(frozen=True, slots=True)
class FinalConfig:
    path: Path
    config_sha256: str
    project_root: Path
    expected_baseline_commit: str
    artifact_root: str
    seed: int
    margin_factor: float
    bootstrap_replicates: int
    memory_fraction: float
    maximum_reserved_bytes: int
    bindings: dict[str, Path]

    @classmethod
    def load(cls, path: Path) -> FinalConfig:
        resolved = path.resolve(strict=True)
        project_root = resolved.parents[2]
        raw: object = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise S10FinalError("final config is not a mapping")
        root = dict(raw)
        if (
            root.get("schema_version") != SCHEMA_VERSION
            or root.get("experiment_id") != EXPERIMENT_ID
            or root.get("track") != "SCIENCE"
            or root.get("status") != "PREREGISTERED_NOT_RUN"
        ):
            raise S10FinalError("unsupported final config")
        scope = _mapping(root.get("scope"), "scope")
        selection = _mapping(root.get("target_selection"), "target selection")
        method = _mapping(root.get("method"), "method")
        development = _mapping(root.get("development_gate"), "development gate")
        statistics = _mapping(root.get("statistics"), "statistics")
        expected_scope = {
            "split": scope.get("split") == "LOCKED_TEST",
            "lesions": scope.get("lesions") == ["HE", "EX"],
            "sources": scope.get("source_count") == 27,
            "openings": scope.get("final_openings_allowed") == 1,
            "class_order": selection.get("class_order") == ["HE", "EX"],
            "connectivity": selection.get("connectivity") == 8,
            "candidates": selection.get("maximum_candidates_per_source_class") == 10,
            "one_target": selection.get("maximum_accepted_per_source_class") == 1,
            "conditions": method.get("condition_count") == 19,
            "policy": method.get("policy")
            == "FOUR_VALUED_DUAL_FEASIBILITY_WITH_SELECTIVE_MARGIN_GATE",
            "rows": development.get("expected_total_rows") == 17442,
            "determinate": development.get("expected_determinate_rows") == 1999,
            "false": development.get("expected_false_claims") == 71,
        }
        if not all(expected_scope.values()):
            raise S10FinalError("final scope or development contract changed")
        bindings: dict[str, Path] = {}
        for name, value in _mapping(root.get("bindings"), "bindings").items():
            binding = _mapping(value, f"binding {name}")
            relative = _string(binding.get("path"), f"{name} path")
            expected = _sha256(binding.get("sha256"), f"{name} hash")
            target = project_root.joinpath(*PurePosixPath(relative).parts)
            if not target.is_file() or sha256_file(target) != expected:
                raise S10FinalError(f"binding mismatch: {name}")
            bindings[name] = target
        required = {
            "s1_manifest",
            "s8_config",
            "s8_results",
            "s8_twins",
            "recovery_summary",
            "recovery_rows",
            "recovery_calibration",
            "recovery_code",
            "s4_twin_config",
            *MODEL_BINDINGS,
        }
        if set(bindings) != required:
            raise S10FinalError("binding set changed")
        return cls(
            path=resolved,
            config_sha256=sha256_file(resolved),
            project_root=project_root,
            expected_baseline_commit=_string(
                root.get("expected_baseline_commit"), "baseline commit"
            ),
            artifact_root=_string(root.get("artifact_root"), "artifact root"),
            seed=_integer(root.get("seed"), "seed"),
            margin_factor=_number(method.get("selective_margin_factor"), "margin factor"),
            bootstrap_replicates=_integer(
                statistics.get("source_cluster_bootstrap_replicates"), "bootstrap replicates"
            ),
            memory_fraction=_number(method.get("memory_fraction_limit"), "memory fraction"),
            maximum_reserved_bytes=_integer(
                method.get("maximum_gpu_reserved_bytes"), "GPU boundary"
            ),
            bindings=bindings,
        )


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise S10FinalError(f"{name} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise S10FinalError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise S10FinalError(f"{name} must be a positive integer")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise S10FinalError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise S10FinalError(f"{name} must be finite and positive")
    return result


def _sha256(value: object, name: str) -> str:
    text = _string(value, name).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise S10FinalError(f"{name} is not SHA-256")
    return text


def _false_claim(state: str, truth: str) -> bool:
    return (state == "ENTAILED_PRESENT" and truth == "ABSENT") or (
        state == "ENTAILED_ABSENT" and truth == "PRESENT"
    )


def apply_hybrid_gate(row: Mapping[str, object], factor: float) -> dict[str, object]:
    """Retain a four-valued claim only when the frozen selective margin clears factor."""

    state = _string(row.get("state_four_valued"), "four-valued state")
    margin_value = row.get("selective_margin")
    if isinstance(margin_value, bool) or not isinstance(margin_value, int | float):
        raise S10FinalError("selective margin must be numeric")
    margin = float(margin_value)
    if not math.isfinite(margin) or margin < 0.0:
        raise S10FinalError("selective margin must be finite and non-negative")
    threshold_value = row.get("selective_margin_threshold")
    if isinstance(threshold_value, bool) or not isinstance(threshold_value, int | float):
        raise S10FinalError("selective margin threshold must be numeric")
    threshold = float(threshold_value)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise S10FinalError("selective margin threshold must be finite and non-negative")
    hybrid = state
    if state in DETERMINATE and not margin > factor * threshold:
        hybrid = "RAW_UNIDENTIFIABLE"
    return {**dict(row), "state_hybrid_factor_1_5": hybrid}


def metric_summary(rows: Sequence[Mapping[str, object]], state_key: str) -> dict[str, object]:
    """Return observed coverage/FCR counts with Wilson intervals."""

    determinate = [row for row in rows if row.get(state_key) in DETERMINATE]
    false = sum(
        _false_claim(
            _string(row.get(state_key), "state"), _string(row.get("expected_truth"), "truth")
        )
        for row in determinate
    )
    total = len(rows)
    count = len(determinate)
    return {
        "total": total,
        "determinate": count,
        "coverage": count / total if total else 0.0,
        "false": false,
        "fcr": false / count if count else None,
        "coverage_wilson95": list(wilson_interval(count, total)) if total else None,
        "fcr_wilson95": list(wilson_interval(false, count)) if count else None,
    }


def _development_gate(config: FinalConfig) -> dict[str, object]:
    payload = _mapping(
        json.loads(config.bindings["recovery_rows"].read_text(encoding="utf-8")),
        "recovery rows",
    )
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list):
        raise S10FinalError("recovery rows are invalid")
    rows = [
        apply_hybrid_gate(_mapping(row, "recovery row"), config.margin_factor) for row in raw_rows
    ]
    overall = metric_summary(rows, "state_hybrid_factor_1_5")
    by_lesion = {
        lesion: metric_summary(
            [row for row in rows if row.get("lesion_class") == lesion],
            "state_hybrid_factor_1_5",
        )
        for lesion in ("HE", "EX", "MA", "SE")
    }
    identity = metric_summary(
        [row for row in rows if row.get("severity_level") == "IDENTITY"],
        "state_hybrid_factor_1_5",
    )
    expected = (17442, 1999, 71)
    observed = (
        overall["total"],
        overall["determinate"],
        overall["false"],
    )
    fcr = cast(float, overall["fcr"])
    coverage = cast(float, overall["coverage"])
    passed = (
        observed == expected
        and fcr <= 0.05
        and coverage >= 0.10
        and cast(float, identity["fcr"]) <= 0.05
        and cast(float, by_lesion["HE"]["fcr"]) <= 0.05
        and cast(float, by_lesion["EX"]["fcr"]) <= 0.05
    )
    if not passed:
        raise S10FinalError("frozen S8 development-open gate failed")
    return {"status": "PASS", "overall": overall, "identity": identity, "by_lesion": by_lesion}


def _verify_preregistration_committed(config: FinalConfig) -> str:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=config.project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", config.expected_baseline_commit, head],
        cwd=config.project_root,
        check=False,
    )
    if ancestor.returncode:
        raise S10FinalError("expected S9 baseline is not an ancestor of HEAD")
    tracked = (
        config.path,
        config.project_root / "docs/science_s10_final_locked_he_ex_protocol_v1.md",
        config.project_root / "scripts/science_s10_final_locked_he_ex_v1.py",
        config.project_root / "data_registry/schemas/science-s10-final-locked-v1.schema.json",
        config.project_root / "tests/unit/test_science_s10_final_locked.py",
    )
    for path in tracked:
        relative = path.relative_to(config.project_root).as_posix()
        committed = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", relative],
            cwd=config.project_root,
            check=False,
        )
        listed = subprocess.run(
            ["git", "ls-files", "--error-unmatch", relative],
            cwd=config.project_root,
            check=False,
            capture_output=True,
        )
        if committed.returncode or listed.returncode:
            raise S10FinalError(f"preregistration is not committed: {relative}")
    return head


def _manifest(config: FinalConfig) -> DatasetManifestV2:
    raw = _mapping(
        json.loads(config.bindings["s1_manifest"].read_text(encoding="utf-8")),
        "S1 manifest",
    )
    manifest = DatasetManifestV2.from_dict(raw)
    locked = tuple(record for record in manifest.records if record.split == "LOCKED_TEST")
    if len(locked) != 27 or len(manifest.records) != 80:
        raise S10FinalError("S1 source membership changed")
    return DatasetManifestV2(
        dataset_id=manifest.dataset_id,
        dataset_version=manifest.dataset_version,
        records=locked,
    )


def _label(record: ManifestRecordV2, lesion: str) -> LabelRecordV2:
    matches = [label for label in record.labels if label.semantic.value == LABEL_SEMANTICS[lesion]]
    if len(matches) != 1 or matches[0].availability.value != "AVAILABLE":
        raise S10FinalError(f"{record.source_image_id} lacks one available {lesion} mask")
    return matches[0]


def _component_record(
    record: ManifestRecordV2,
    lesion: str,
    index: int,
    component: SparseComponent,
    fov: FOVGeometry,
    label: LabelRecordV2,
) -> dict[str, object]:
    geometry = patch_geometry(
        image_size=(record.width, record.height),
        centroid=component.centroid,
        patch_size=(256, 256),
    )
    return {
        "schema_version": "1.0.0",
        "dataset_id": record.dataset_id,
        "dataset_version": "idrid-ieee-h25w98-2026-01-12-zenodo17219542-v1",
        "source_image_id": record.source_image_id,
        "split": "LOCKED_TEST",
        "lesion_class": lesion,
        "component_id": f"s10-{record.source_image_id}-{lesion.lower()}-{index}",
        "image_relative_path": record.relative_path,
        "image_sha256": record.sha256,
        "mask_relative_path": label.relative_path,
        "mask_sha256": label.sha256,
        "component_mask_sha256": component.component_mask_sha256,
        "image_size_wh": [record.width, record.height],
        "fov_bbox_xyxy": list(fov.bbox),
        "area_px": component.area,
        "bbox_xyxy": list(component.bbox),
        "centroid_xy": list(component.centroid),
        "eccentricity": component.eccentricity,
        "distance_to_fov_boundary_px": _ellipse_boundary_distance(component.centroid, fov.bbox),
        "touches_image_boundary": (
            component.bbox[0] == 0
            or component.bbox[1] == 0
            or component.bbox[2] == record.width
            or component.bbox[3] == record.height
        ),
        "touches_fov_boundary": _component_touches_fov(component, fov),
        "patch": geometry.to_dict(),
        "inclusion_status": "INCLUDED",
    }


def _eligibility(row: Mapping[str, object]) -> str:
    area = cast(int, row["area_px"])
    bbox = cast(list[int], row["bbox_xyxy"])
    if area < 4:
        return "AREA_BELOW_FROZEN_MINIMUM"
    if area > 1024:
        return "AREA_ABOVE_FROZEN_MAXIMUM"
    if bbox[2] - bbox[0] > 48 or bbox[3] - bbox[1] > 48:
        return "BBOX_ABOVE_FROZEN_MAXIMUM"
    if row["touches_image_boundary"]:
        return "IMAGE_BOUNDARY_TOUCH"
    if row["touches_fov_boundary"]:
        return "FOV_BOUNDARY_TOUCH"
    if cast(float, row["distance_to_fov_boundary_px"]) < 16.0:
        return "FOV_MARGIN_BELOW_FROZEN_MINIMUM"
    patch = cast(Mapping[str, object], row["patch"])
    if patch["padding_ltrb"] != [0, 0, 0, 0]:
        return "PATCH_PADDING_PRESENT"
    return "ELIGIBLE"


def _selection_key(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        cast(int, row["area_px"]),
        *cast(list[int], row["bbox_xyxy"]),
        *cast(list[float], row["centroid_xy"]),
        _string(row["component_mask_sha256"], "component hash"),
    )


def _write_selection_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = (
        "source_image_id",
        "lesion_class",
        "component_id",
        "candidate_rank",
        "eligibility_reason",
        "constructibility_reason",
        "outcome",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field) for field in fields})
    write_bytes_fail_safe(path, stream.getvalue().encode("utf-8"))


def _construct(
    config: FinalConfig, dataset_root: Path, output: Path
) -> tuple[list[dict[str, object]], list[dict[str, object]], Counter[str]]:
    manifest = _manifest(config)
    base = s4.S4TwinConfig.load(config.bindings["s4_twin_config"])
    root = dataset_root.resolve(strict=True)
    accepted: list[dict[str, object]] = []
    selection: list[dict[str, object]] = []
    cells: dict[str, list[QCCell]] = {"HE": [], "EX": []}
    reads: Counter[str] = Counter()
    source_order = sorted(
        manifest.records,
        key=lambda item: hashlib.sha256(item.source_image_id.encode()).hexdigest(),
    )
    for source in source_order:
        image_path = _resolve_under_root(root, source.relative_path)
        if sha256_file(image_path) != source.sha256:
            raise S10FinalError(f"image hash mismatch: {source.source_image_id}")
        image = _load_rgb_strict(image_path, source)
        reads["LOCKED_IMAGE"] += 1
        fov = detect_fov(image, foreground_max_rgb_min=9, minimum_axis_coverage_fraction=0.01)
        masks: dict[str, NDArray[np.bool_]] = {}
        components: dict[str, tuple[SparseComponent, ...]] = {}
        provenance: dict[str, dict[str, str]] = {}
        for lesion in LESION_CLASSES:
            label = _label(source, lesion)
            mask_path = _resolve_under_root(root, cast(str, label.relative_path))
            if sha256_file(mask_path) != label.sha256:
                raise S10FinalError(f"mask hash mismatch: {source.source_image_id}/{lesion}")
            mask = decode_idrid_mask(mask_path, expected_size=(source.width, source.height))
            reads[f"LOCKED_{lesion}_MASK"] += 1
            masks[lesion] = mask
            components[lesion] = connected_components(mask, connectivity=8)
            provenance[lesion] = {
                "relative_path": cast(str, label.relative_path),
                "sha256": cast(str, label.sha256),
            }
        context = s4.SourceContext(source, image, masks, components, fov.mask, provenance)
        for lesion in LESION_CLASSES:
            label = _label(source, lesion)
            candidates = sorted(
                [
                    _component_record(source, lesion, index, component, fov, label)
                    for index, component in enumerate(components[lesion], start=1)
                ],
                key=_selection_key,
            )[:10]
            selected = False
            if not candidates:
                selection.append(
                    {
                        "source_image_id": source.source_image_id,
                        "lesion_class": lesion,
                        "component_id": None,
                        "candidate_rank": None,
                        "eligibility_reason": "EMPTY_ANNOTATION_NOT_NEGATIVE",
                        "constructibility_reason": "NOT_ATTEMPTED",
                        "outcome": "REJECTED_SOURCE_CLASS",
                    }
                )
            for rank, candidate in enumerate(candidates, start=1):
                reason = _eligibility(candidate)
                row = {
                    "source_image_id": source.source_image_id,
                    "lesion_class": lesion,
                    "component_id": candidate["component_id"],
                    "candidate_rank": rank,
                    "eligibility_reason": reason,
                    "constructibility_reason": "NOT_ATTEMPTED",
                    "outcome": "REJECTED",
                }
                if selected:
                    row["constructibility_reason"] = "NOT_REACHED_AFTER_ACCEPTANCE"
                    selection.append(row)
                    continue
                if reason != "ELIGIBLE":
                    selection.append(row)
                    continue
                try:
                    built = s4._build_one(context=context, record=candidate, config=base)
                except TwinRejected as error:
                    row["constructibility_reason"] = str(error)
                    selection.append(row)
                    continue
                record = cast(dict[str, object], built.record)
                source_payload = cast(dict[str, object], record["source"])
                source_payload["lesion_mask_provenance"] = source_payload.pop(
                    "four_mask_provenance"
                )
                for relative, content in built.artifacts.items():
                    write_bytes_fail_safe(output / relative, content)
                accepted.append(record)
                cells[lesion].append(built.qc_cell)
                row["constructibility_reason"] = "CONSTRUCTIBLE"
                row["outcome"] = "ACCEPTED"
                selection.append(row)
                selected = True
    counts = Counter(_string(row["lesion_class"], "lesion") for row in accepted)
    if counts["HE"] < 10 or counts["EX"] < 10:
        raise S10FinalError(f"insufficient constructible targets: {dict(counts)}")
    for lesion in LESION_CLASSES:
        write_bytes_fail_safe(
            output / f"qc-{lesion.lower()}-contact-sheet.png",
            _contact_sheet(cells[lesion], columns=2, panel_size=128),
        )
    twin_manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "scope": "FINAL_LOCKED_TEST_HE_EX_CONTROLLED_TWINS",
        "accepted": accepted,
    }
    write_bytes_fail_safe(output / "twin-manifest-v1.json", canonical_json_bytes(twin_manifest))
    write_bytes_fail_safe(
        output / "target-selection-v1.json",
        canonical_json_bytes({"schema_version": SCHEMA_VERSION, "records": selection}),
    )
    _write_selection_csv(output / "accepted-rejected-v1.csv", selection)
    return accepted, selection, reads


def _load_model(config: FinalConfig, name: str, device: torch.device) -> v8.FrozenResNetAdapter:
    weight_path = recovery._pretrained_weight_path(s8.S8Config.load(config.bindings["s8_config"]))
    model = v8.FrozenResNetAdapter(32, weight_path)
    checkpoint = torch.load(config.bindings[name], map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise S10FinalError(f"invalid checkpoint: {name}")
    model.backbone.layer4.load_state_dict(cast(Mapping[str, object], checkpoint["layer4_state"]))
    incompatible = model.load_state_dict(
        cast(Mapping[str, object], checkpoint["adapter_state"]), strict=False
    )
    if incompatible.unexpected_keys or any(
        not key.startswith("backbone.") for key in incompatible.missing_keys
    ):
        raise S10FinalError(f"checkpoint state mismatch: {name}")
    model.eval()
    return model.to(device)


def _test_items(output: Path, accepted: Sequence[Mapping[str, object]]) -> list[core.TwinItem]:
    return [
        core.TwinItem(
            package_id="S10_IDRID_LOCKED_V1",
            root=output,
            raw=row,
            source_key=f"IDRID_LOCKED:{_string(row['source_image_id'], 'source ID')}",
            source_image_id=_string(row["source_image_id"], "source ID"),
            lesion_class=_string(row["lesion_class"], "lesion class"),
            twin_id=_string(row["twin_id"], "twin ID"),
        )
        for row in accepted
    ]


def _group_metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    state = "state_hybrid_factor_1_5"

    def grouped(field: str) -> dict[str, object]:
        values = sorted({_string(row[field], field) for row in rows})
        return {
            value: metric_summary([row for row in rows if row.get(field) == value], state)
            for value in values
        }

    return {
        "overall": metric_summary(rows, state),
        "by_lesion": grouped("lesion_class"),
        "by_truth": grouped("expected_truth"),
        "by_severity": grouped("severity_level"),
        "by_condition": grouped("condition_id"),
    }


def _artifact_manifest(output: Path) -> dict[str, object]:
    entries = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        if path.name == "artifact-manifest-v1.json":
            continue
        entries.append(
            {
                "relative_path": path.relative_to(output).as_posix(),
                "byte_size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {"schema_version": SCHEMA_VERSION, "experiment_id": EXPERIMENT_ID, "entries": entries}


def run(config: FinalConfig, dataset_root: Path, output_dir: Path) -> dict[str, object]:
    """Open the frozen IDRiD test root once, construct twins and evaluate the fixed method."""

    head = _verify_preregistration_committed(config)
    development = _development_gate(config)
    output = output_dir.resolve(strict=False)
    expected = (config.project_root / config.artifact_root).resolve(strict=False)
    if output != expected or output.exists():
        raise S10FinalError("output path changed or exists")
    output.mkdir(parents=True, exist_ok=False)
    accepted, selection, reads = _construct(config, dataset_root, output)
    test_items = _test_items(output, accepted)
    s8_config = s8.S8Config.load(config.bindings["s8_config"])
    train_items, calibration_items = s8._v8_training_items(s8_config)
    role_sets = {
        "train": {item.source_key for item in train_items},
        "conformal": {item.source_key for item in calibration_items},
        "final": {item.source_key for item in test_items},
    }
    overlap = sum(
        len(role_sets[first] & role_sets[second])
        for first, second in (("train", "conformal"), ("train", "final"), ("conformal", "final"))
    )
    if overlap:
        raise S10FinalError("source-role overlap")
    if not torch.cuda.is_available():
        raise S10FinalError("CUDA is required for the bounded final run")
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(config.memory_fraction, device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(config.seed)
    io_audit = core.IOAudit()
    all_items = [*train_items, *calibration_items, *test_items]
    cache = s8._raw_cache(all_items, io_audit)
    calibration_by_seed: list[list[dict[str, object]]] = []
    test_by_seed: list[list[dict[str, object]]] = []
    replay_counts: Counter[str] = Counter()
    for model_name in MODEL_BINDINGS:
        model = _load_model(config, model_name, device)
        calibration_rows: list[dict[str, object]] = []
        final_rows: list[dict[str, object]] = []
        for condition in s8.frozen_conditions():
            calibration_rows.extend(
                recovery._score_condition(
                    model=model,
                    condition=condition,
                    train=train_items,
                    queries=calibration_items,
                    role="CONFORMAL_FIT_REPLAY",
                    cache=cache,
                    config=s8_config,
                    replay=replay_counts,
                    device=device,
                )
            )
            final_rows.extend(
                recovery._score_condition(
                    model=model,
                    condition=condition,
                    train=train_items,
                    queries=test_items,
                    role="FINAL_LOCKED_IDRID",
                    cache=cache,
                    config=s8_config,
                    replay=replay_counts,
                    device=device,
                )
            )
        calibration_by_seed.append(calibration_rows)
        test_by_seed.append(final_rows)
        model.cpu()
        torch.cuda.empty_cache()
    ensemble_calibration = recovery._average_rows(calibration_by_seed)
    ensemble_raw = recovery._average_rows(test_by_seed)
    thresholds = s8.fit_source_max_thresholds(ensemble_calibration, alpha=s8_config.alpha)
    selective = s8.fit_selective_margin_thresholds(ensemble_calibration, alpha=s8_config.alpha)
    replayed_calibration = canonical_json_bytes(
        {
            "schema_version": SCHEMA_VERSION,
            "method": "MEAN_SCORE_ENSEMBLE_SOURCE_MAX_CLASS_CONDITION_HYPOTHESIS_SPLIT_CONFORMAL",
            "thresholds": thresholds,
            "selective_margin_thresholds": selective,
        }
    )
    if hashlib.sha256(replayed_calibration).hexdigest() != sha256_file(
        config.bindings["recovery_calibration"]
    ):
        raise S10FinalError("conformal calibration replay changed")
    rows = [
        apply_hybrid_gate(s8.apply_policies(row, thresholds, selective), config.margin_factor)
        for row in ensemble_raw
    ]
    if any(
        not math.isfinite(cast(float, row[key]))
        for row in rows
        for key in ("positive_score", "negative_score")
    ):
        raise S10FinalError("non-finite final score")
    write_bytes_fail_safe(
        output / "final-rows-v1.json",
        canonical_json_bytes({"schema_version": SCHEMA_VERSION, "rows": rows}),
    )
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    if peak_reserved > config.maximum_reserved_bytes:
        raise S10FinalError("GPU memory boundary exceeded")
    counts = Counter(_string(row["lesion_class"], "lesion") for row in accepted)
    result = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETE_FINAL_LOCKED_RESULT",
        "bindings": {
            "config_sha256": config.config_sha256,
            "preregistration_commit": head,
            **{name: sha256_file(path) for name, path in config.bindings.items()},
        },
        "construction": {
            "locked_sources": 27,
            "accepted_twins": len(accepted),
            "accepted_by_lesion": dict(sorted(counts.items())),
            "selection_records": len(selection),
        },
        "source_roles": {name: len(values) for name, values in role_sets.items()}
        | {"overlap": overlap},
        "condition_count": 19,
        "row_count": len(rows),
        "primary_policy": "state_hybrid_factor_1_5",
        "metrics": _group_metrics(rows),
        "bootstrap": _cluster_bootstrap_fcr(
            rows,
            "state_hybrid_factor_1_5",
            config.seed,
            config.bootstrap_replicates,
        ),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "memory_fraction_limit": config.memory_fraction,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
        },
        "audit": {
            "development_gate": development,
            "locked_root_reads": dict(sorted(reads.items())),
            "final_openings": 1,
            "twin_png_decodes": io_audit.twin_png_decodes,
            "twin_png_hash_mismatches": io_audit.twin_png_hash_mismatches,
            "degradation_replay_checks": replay_counts["checks"],
            "degradation_replay_mismatches": replay_counts["mismatches"],
            "threshold_refits_using_final_rows": 0,
            "model_training_steps_using_final_rows": 0,
            "target_reselection_after_scores": 0,
        },
        "evidence_boundary": [
            "The final evidence is a controlled-twin proposition-entailment evaluation, "
            "not clinical diagnostic performance.",
            "S8 informed recovery design but no final IDRiD score informed model, target, "
            "threshold or policy selection.",
            "The final locked source root was opened exactly once after the protocol and "
            "executable were committed.",
        ],
    }
    write_bytes_fail_safe(output / "final-results-v1.json", canonical_json_bytes(result))
    audit = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS",
        "config_sha256": config.config_sha256,
        "result_sha256": sha256_file(output / "final-results-v1.json"),
        "locked_root_reads": dict(sorted(reads.items())),
        "source_role_overlap": overlap,
        "replay_mismatches": replay_counts["mismatches"],
        "gpu_peak_reserved_bytes": peak_reserved,
    }
    write_bytes_fail_safe(output / "audit.json", canonical_json_bytes(audit))
    write_bytes_fail_safe(
        output / "artifact-manifest-v1.json",
        canonical_json_bytes(_artifact_manifest(output)),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "construction": result["construction"],
                "overall": cast(Mapping[str, object], result["metrics"])["overall"],
                "bootstrap": result["bootstrap"],
                "runtime": result["runtime"],
            },
            indent=2,
        )
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/science/s10_final_locked_he_ex_v1.yaml"),
    )
    parser.add_argument("--preflight-development", action="store_true")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    config = FinalConfig.load(args.config)
    if args.preflight_development:
        print(json.dumps(_development_gate(config), indent=2))
        return 0
    if args.dataset_root is None or args.output_dir is None:
        parser.error("--dataset-root and --output-dir are required for the final run")
    run(config, args.dataset_root, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
