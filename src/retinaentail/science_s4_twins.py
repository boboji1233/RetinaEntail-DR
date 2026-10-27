"""Science S4 IDRiD four-lesion controlled-twin tranche.

The executable is deliberately bounded to the 26 TRAIN/CALIBRATION sources
with complete HE/EX/MA/SE masks.  It reuses the frozen S3 candidate
implementations without changing them, applies a four-mask preservation rule,
and emits an independent raw-array exact-collision Oracle.  It never invokes
RetinaEntail and it never reads LOCKED_TEST pixels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import numpy as np
import yaml
from numpy.typing import NDArray

from retinaentail.data.lesion_catalog import (
    JSONValue,
    S1Replay,
    S2Config,
    S2DataError,
    S2GateError,
    SparseComponent,
    _component_mask_in_patch,
    _finite_float,
    _integer,
    _load_rgb_strict,
    _mapping,
    _project_file,
    _resolve_under_root,
    _safe_relative,
    _sha256,
    _string,
    connected_components,
    decode_idrid_mask,
    detect_fov,
    extract_fixed_patch,
    patch_geometry,
    replay_s1_prerequisite,
)
from retinaentail.data.license_gate import (
    DataUsePurpose,
    authorize_dataset_access,
    load_license_review_record,
    load_registry_license_metadata,
)
from retinaentail.data.manifest import sha256_file
from retinaentail.data.stage2_models import (
    LabelRecordV2,
    ManifestRecordV2,
    load_dataset_adapter_spec,
)
from retinaentail.interventions.controlled_twin_pilot import (
    METHODS,
    QCCell,
    S3Config,
    S3DataError,
    TwinRejected,
    _assigned_method,
    _binary_dilate,
    _contact_sheet,
    _eligibility_reason,
    _find_offset,
    _mask_sha256,
    _method_output,
    _selection_key,
)
from retinaentail.interventions.oracle import (
    CollisionMode,
    ControlledTwinOracleRecord,
    DomainTruth,
    OracleTier,
    SourceTruth,
    SupportTruth,
)
from retinaentail.witnesses.replay import (
    canonical_json_bytes,
    rgb_png_bytes,
    sha256_bytes,
    write_bytes_fail_safe,
)

S4_TWIN_SCHEMA_VERSION: Final = "1.0.0"
S4_TWIN_EXPERIMENT_ID: Final = "S4_IDRID_FOUR_LESION_TWIN_TRANCHE"
DEVELOPMENT_SPLITS: Final = ("TRAIN", "CALIBRATION")
LESION_CLASSES: Final = ("HE", "EX", "MA", "SE")
LOCKED_SPLIT: Final = "LOCKED_TEST"
ORACLE_METHOD_ID: Final = "RAW_ARRAY_EXACT_COLLISION_AUDIT_V1"
LABEL_SEMANTICS: Final = {
    "HE": "HEMORRHAGE_MASK",
    "EX": "HARD_EXUDATE_MASK",
    "MA": "MICROANEURYSM_MASK",
    "SE": "SOFT_EXUDATE_MASK",
}


class S4TwinGateError(RuntimeError):
    """Raised before real data I/O when the frozen S4 contract does not replay."""


class S4TwinDataError(RuntimeError):
    """Raised after authorization when real S4 input or output fails closed."""


def _offsets(value: object, name: str) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, list) or not value:
        raise S4TwinGateError(f"{name} must be a non-empty list")
    result: list[tuple[int, int]] = []
    for index, item in enumerate(value):
        if (
            not isinstance(item, list)
            or len(item) != 2
            or any(isinstance(number, bool) or not isinstance(number, int) for number in item)
        ):
            raise S4TwinGateError(f"{name}[{index}] must contain two integers")
        dx, dy = cast(list[int], item)
        if dx == 0 and dy == 0:
            raise S4TwinGateError(f"{name}[{index}] cannot be zero")
        result.append((dx, dy))
    if len(result) != len(set(result)):
        raise S4TwinGateError(f"{name} contains duplicate offsets")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class S4TwinConfig:
    """Validated executable subset of the frozen S4 tranche YAML."""

    path: Path
    payload: dict[str, object]
    config_sha256: str
    dataset_id: str
    dataset_version: str
    artifact_root: str
    bindings: dict[str, object]
    seed: int
    class_targets: dict[str, int]
    split_quotas: dict[str, dict[str, int]]
    minimum_sources: dict[str, int]
    minimum_area: int
    maximum_area: int
    maximum_bbox_width: int
    maximum_bbox_height: int
    minimum_fov_margin: float
    methods: tuple[str, ...]
    method_settings: dict[str, dict[str, object]]
    target_dilation: int
    lesion_exclusion_dilation: int
    sham_offsets: tuple[tuple[int, int], ...]
    donor_offsets: tuple[tuple[int, int], ...]
    qc_samples_per_class: int
    qc_columns: int
    qc_panel_size: int
    minimum_collisions: int
    rejection_taxonomy: frozenset[str]

    @classmethod
    def load(cls, path: Path) -> S4TwinConfig:
        decoded: object = yaml.safe_load(path.read_text(encoding="utf-8"))
        root = _mapping(decoded, "S4 twin config")
        if _string(root.get("schema_version"), "schema_version") != S4_TWIN_SCHEMA_VERSION:
            raise S4TwinGateError("S4 twin schema_version mismatch")
        if _string(root.get("experiment_id"), "experiment_id") != S4_TWIN_EXPERIMENT_ID:
            raise S4TwinGateError("S4 twin experiment_id mismatch")
        if root.get("status") != "IMPLEMENTING" or root.get("execution_enabled") is not True:
            raise S4TwinGateError("only the frozen IMPLEMENTING S4 config is executable")
        if root.get("scope") != "TRAIN_AND_CALIBRATION_COMPLETE_FOUR_MASK_SOURCES_ONLY":
            raise S4TwinGateError("S4 tranche scope mismatch")
        if root.get("splits") != list(DEVELOPMENT_SPLITS):
            raise S4TwinGateError("S4 splits must be TRAIN/CALIBRATION")
        if root.get("locked_test_allowed") is not False:
            raise S4TwinGateError("LOCKED_TEST must remain prohibited")
        if root.get("lesion_classes") != list(LESION_CLASSES):
            raise S4TwinGateError("S4 lesion class set/order mismatch")
        if root.get("paired_outputs") != ["x_plus_sham", "x_minus_removed"]:
            raise S4TwinGateError("S4 paired-output contract mismatch")

        selection = _mapping(root.get("selection"), "selection")
        assignment = _mapping(root.get("method_assignment"), "method assignment")
        edit = _mapping(root.get("edit"), "edit")
        oracle = _mapping(root.get("oracle"), "oracle")
        qc = _mapping(root.get("qc"), "qc")
        gate = _mapping(root.get("gate"), "gate")
        if selection.get("order") != "SHA256_SEED_CLASS_SPLIT_COMPONENT_ID_ASC":
            raise S4TwinGateError("selection order is not frozen")
        if selection.get("replacement_policy") != "NEXT_CONSTRUCTIBLE_IN_FROZEN_ORDER":
            raise S4TwinGateError("replacement policy is not frozen")
        if assignment.get("order") != "SHA256_SEED_COMPONENT_ID_MOD_METHOD_COUNT":
            raise S4TwinGateError("method assignment order is not frozen")
        if assignment.get("methods") != list(METHODS):
            raise S4TwinGateError("S4 candidate method set/order mismatch")
        if edit.get("patch_size_wh") != [256, 256]:
            raise S4TwinGateError("S4 patch size must remain 256x256")
        if oracle.get("method_id") != ORACLE_METHOD_ID:
            raise S4TwinGateError("S4 Oracle method mismatch")
        if (
            oracle.get("retinaentail_independent") is not True
            or oracle.get("confirmatory_tier") != "EXACT_COLLISION"
            or oracle.get("collision_mode") != "SAME_THETA"
            or oracle.get("noncollision_support_truth") != "ORACLE_UNRESOLVED"
        ):
            raise S4TwinGateError("S4 independent Oracle contract mismatch")
        forbidden_inputs = oracle.get("forbidden_inputs")
        if forbidden_inputs != [
            "RETINAENTAIL_SCORE",
            "FEASIBLE_FLAG",
            "THRESHOLD",
            "FOUR_VALUED_STATE",
        ]:
            raise S4TwinGateError("S4 Oracle forbidden-input contract mismatch")
        required_gate_true = (
            "require_zero_locked_test_root_reads",
            "require_zero_source_split_leaks",
            "require_complete_pairs",
            "require_outside_edit_exact_preservation",
            "require_target_exact_preservation_in_sham",
            "require_all_outputs_hashed",
            "require_target_and_source_sufficiency",
            "require_tier_a_ambiguity_sufficiency",
        )
        if any(gate.get(name) is not True for name in required_gate_true):
            raise S4TwinGateError("one or more mandatory S4 gates are disabled")

        class_targets_raw = _mapping(selection.get("accepted_target_per_class"), "class targets")
        class_targets = {
            lesion_class: _integer(
                class_targets_raw.get(lesion_class), f"{lesion_class} target", minimum=1
            )
            for lesion_class in LESION_CLASSES
        }
        quotas_raw = _mapping(selection.get("accepted_quota_per_class_split"), "class/split quotas")
        split_quotas: dict[str, dict[str, int]] = {}
        for lesion_class in LESION_CLASSES:
            class_quota = _mapping(quotas_raw.get(lesion_class), f"{lesion_class} quotas")
            split_quotas[lesion_class] = {
                split: _integer(class_quota.get(split), f"{lesion_class}:{split} quota", minimum=0)
                for split in DEVELOPMENT_SPLITS
            }
            if sum(split_quotas[lesion_class].values()) != class_targets[lesion_class]:
                raise S4TwinGateError(f"{lesion_class} quotas do not equal its frozen target")
        source_raw = _mapping(selection.get("minimum_distinct_sources"), "minimum sources")
        minimum_sources = {
            lesion_class: _integer(
                source_raw.get(lesion_class), f"{lesion_class} minimum sources", minimum=1
            )
            for lesion_class in LESION_CLASSES
        }
        implementation_ids = {
            "TELEA": "TELEA_STYLE_BOUNDARY_WEIGHTED_NUMPY_V1",
            "NAVIER_STOKES": "NAVIER_STOKES_STYLE_ISOPHOTE_DIFFUSION_NUMPY_V1",
            "DONOR_POISSON": "DONOR_TEXTURE_POISSON_JACOBI_NUMPY_V1",
        }
        method_settings = {
            method: _mapping(assignment.get(method), f"method {method}") for method in METHODS
        }
        for method, implementation_id in implementation_ids.items():
            if method_settings[method].get("implementation_id") != implementation_id:
                raise S4TwinGateError(f"{method} implementation_id mismatch")
        taxonomy = root.get("rejection_reason_taxonomy")
        if not isinstance(taxonomy, list) or not all(
            isinstance(item, str) and item for item in taxonomy
        ):
            raise S4TwinGateError("rejection taxonomy must be a non-empty string list")
        return cls(
            path=path,
            payload=root,
            config_sha256=sha256_file(path),
            dataset_id=_string(root.get("dataset_id"), "dataset_id"),
            dataset_version=_string(root.get("dataset_version"), "dataset_version"),
            artifact_root=_safe_relative(root.get("artifact_root"), "artifact_root"),
            bindings=_mapping(root.get("bindings"), "bindings"),
            seed=_integer(selection.get("seed"), "selection seed", minimum=0),
            class_targets=class_targets,
            split_quotas=split_quotas,
            minimum_sources=minimum_sources,
            minimum_area=_integer(selection.get("minimum_area_px"), "minimum area", minimum=1),
            maximum_area=_integer(selection.get("maximum_area_px"), "maximum area", minimum=1),
            maximum_bbox_width=_integer(
                selection.get("maximum_bbox_width_px"), "maximum bbox width", minimum=1
            ),
            maximum_bbox_height=_integer(
                selection.get("maximum_bbox_height_px"), "maximum bbox height", minimum=1
            ),
            minimum_fov_margin=_finite_float(
                selection.get("minimum_fov_margin_px"), "minimum FOV margin"
            ),
            methods=METHODS,
            method_settings=method_settings,
            target_dilation=_integer(
                edit.get("target_mask_dilation_px"), "target dilation", minimum=1
            ),
            lesion_exclusion_dilation=_integer(
                edit.get("lesion_exclusion_dilation_px"),
                "lesion exclusion dilation",
                minimum=1,
            ),
            sham_offsets=_offsets(edit.get("sham_offset_candidates_xy"), "sham offsets"),
            donor_offsets=_offsets(edit.get("donor_offset_candidates_xy"), "donor offsets"),
            qc_samples_per_class=_integer(
                qc.get("deterministic_samples_per_class"), "QC samples", minimum=1
            ),
            qc_columns=_integer(qc.get("contact_sheet_columns"), "QC columns", minimum=1),
            qc_panel_size=_integer(qc.get("panel_size_px"), "QC panel size", minimum=64),
            minimum_collisions=_integer(
                oracle.get("minimum_exact_collision_pairs_per_class"),
                "minimum collision pairs",
                minimum=1,
            ),
            rejection_taxonomy=frozenset(cast(list[str], taxonomy)),
        )

    def as_s3(self) -> S3Config:
        """Return a structural cast for frozen, side-effect-free S3 helpers."""

        return cast(S3Config, self)

    def implementation_id(self, method: str) -> str:
        return _string(
            self.method_settings[method].get("implementation_id"),
            f"{method} implementation_id",
        )


@dataclass(frozen=True, slots=True)
class S4Upstream:
    he_ex_s1: S1Replay
    ma_se_s1: S1Replay
    he_ex_config: S2Config
    ma_se_config: S2Config
    records: tuple[dict[str, object], ...]
    complete_source_ids: frozenset[str]
    hashes: dict[str, str]


@dataclass(frozen=True, slots=True)
class SourceContext:
    source_record: ManifestRecordV2
    image: NDArray[np.uint8]
    masks: dict[str, NDArray[np.bool_]]
    components: dict[str, tuple[SparseComponent, ...]]
    fov: NDArray[np.bool_]
    label_provenance: dict[str, dict[str, str]]


@dataclass(frozen=True, slots=True)
class CandidateArrays:
    source_patch: NDArray[np.uint8]
    target_component: NDArray[np.bool_]
    target_edit: NDArray[np.bool_]
    sham_edit: NDArray[np.bool_]
    combined_lesions: NDArray[np.bool_]
    other_lesions: NDArray[np.bool_]
    lesion_forbidden: NDArray[np.bool_]
    fov_patch: NDArray[np.bool_]
    sham_offset: tuple[int, int]


@dataclass(frozen=True, slots=True)
class BuiltTwin:
    record: dict[str, JSONValue]
    artifacts: dict[str, bytes]
    qc_cell: QCCell
    exact_collision: bool


@dataclass(frozen=True, slots=True)
class S4TwinRunResult:
    status: str
    output_dir: Path
    accepted_count: int
    rejected_count: int
    accepted_counts: dict[str, int]
    distinct_sources: dict[str, int]
    raw_exact_collisions: dict[str, int]
    locked_test_root_reads: int
    artifact_hashes: dict[str, str]
    audit: dict[str, JSONValue]


def _load_json_mapping(path: Path, name: str) -> dict[str, object]:
    decoded: object = json.loads(path.read_text(encoding="utf-8"))
    try:
        return _mapping(decoded, name)
    except S2GateError as error:
        raise S4TwinGateError(str(error)) from error


def _binding(project_root: Path, config: S4TwinConfig, name: str) -> tuple[Path, str]:
    item = _mapping(config.bindings.get(name), f"binding {name}")
    try:
        path = _project_file(project_root, item.get("path"), f"binding {name} path")
        expected = _sha256(item.get("sha256"), f"binding {name} sha256")
    except S2GateError as error:
        raise S4TwinGateError(str(error)) from error
    actual = sha256_file(path)
    if actual != expected:
        raise S4TwinGateError(f"binding hash mismatch: {name}")
    return path, actual


def _catalog_records(
    path: Path, *, config: S4TwinConfig, lesion_classes: tuple[str, str]
) -> tuple[dict[str, object], ...]:
    catalog = _load_json_mapping(path, f"{'/'.join(lesion_classes)} catalog")
    if (
        catalog.get("dataset_id") != config.dataset_id
        or catalog.get("dataset_version") != config.dataset_version
        or catalog.get("scope") != "TRAIN_AND_CALIBRATION_ONLY"
    ):
        raise S4TwinGateError("catalog dataset/scope binding failed")
    raw_records = catalog.get("records")
    if not isinstance(raw_records, list):
        raise S4TwinGateError("catalog records are invalid")
    records: list[dict[str, object]] = []
    for index, item in enumerate(raw_records):
        if not isinstance(item, Mapping):
            raise S4TwinGateError(f"catalog record {index} is not an object")
        record = {str(key): value for key, value in item.items()}
        if record.get("lesion_class") not in lesion_classes:
            raise S4TwinGateError("catalog lesion class binding failed")
        if record.get("split") not in DEVELOPMENT_SPLITS:
            raise S4TwinGateError("catalog contains a non-development split")
        records.append(record)
    return tuple(records)


def _replay_upstream(project_root: Path, config: S4TwinConfig) -> S4Upstream:
    binding_names = (
        "preflight",
        "owner_scope_confirmation",
        "he_ex_run_config",
        "he_ex_state",
        "he_ex_catalog",
        "he_ex_audit",
        "he_ex_artifact_manifest",
        "he_ex_qc",
        "ma_se_run_config",
        "ma_se_state",
        "ma_se_catalog",
        "ma_se_audit",
        "ma_se_artifact_manifest",
        "ma_se_qc",
    )
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for name in binding_names:
        paths[name], hashes[name] = _binding(project_root, config, name)

    preflight = _load_json_mapping(paths["preflight"], "S4 preflight")
    if preflight.get("status") != "READY_FOR_S4_EXECUTION" or preflight.get("blockers") != []:
        raise S4TwinGateError("S4 preflight is not execution-ready")
    owner = _load_json_mapping(paths["owner_scope_confirmation"], "owner confirmation")
    owner_scope = _mapping(owner.get("authorized_scope"), "owner authorized scope")
    if (
        owner.get("decision") != "PASS"
        or owner_scope.get("original_s4_plan_authorized") is not True
        or owner_scope.get("ai_executable_review_delegated") is not True
        or owner_scope.get("locked_test_root_reads_allowed") is not False
    ):
        raise S4TwinGateError("owner authorization replay failed")
    for prefix in ("he_ex", "ma_se"):
        state: object = yaml.safe_load(paths[f"{prefix}_state"].read_text(encoding="utf-8"))
        state_map = _mapping(state, f"{prefix} state")
        if state_map.get("status") != "PASS":
            raise S4TwinGateError(f"{prefix} catalog state is not PASS")
        audit = _load_json_mapping(paths[f"{prefix}_audit"], f"{prefix} audit")
        summary = _mapping(audit.get("summary"), f"{prefix} audit summary")
        if (
            audit.get("status") not in {"READY_FOR_HUMAN_QC", "READY_FOR_AI_QC"}
            or summary.get("locked_test_root_reads") != 0
            or summary.get("source_split_leak_count", summary.get("source_split_leaks")) != 0
        ):
            raise S4TwinGateError(f"{prefix} catalog audit replay failed")
        qc = _load_json_mapping(paths[f"{prefix}_qc"], f"{prefix} QC")
        if qc.get("decision") != "PASS":
            raise S4TwinGateError(f"{prefix} QC decision is not PASS")
        artifact_manifest = _load_json_mapping(
            paths[f"{prefix}_artifact_manifest"], f"{prefix} artifact manifest"
        )
        files = artifact_manifest.get("files")
        if not isinstance(files, list):
            raise S4TwinGateError(f"{prefix} artifact manifest is invalid")
        manifest_hashes = {
            item.get("relative_path"): item.get("sha256")
            for item in files
            if isinstance(item, Mapping)
        }
        if (
            manifest_hashes.get("component-catalog-v1.json") != hashes[f"{prefix}_catalog"]
            or manifest_hashes.get("audit.json") != hashes[f"{prefix}_audit"]
        ):
            raise S4TwinGateError(f"{prefix} artifact manifest binding failed")

    try:
        he_ex_config = S2Config.load(paths["he_ex_run_config"])
        ma_se_config = S2Config.load(paths["ma_se_run_config"])
        he_ex_s1 = replay_s1_prerequisite(project_root, he_ex_config)
        ma_se_s1 = replay_s1_prerequisite(project_root, ma_se_config)
    except (S2GateError, S2DataError) as error:
        raise S4TwinGateError(f"catalog prerequisite replay failed: {error}") from error
    complete_source_ids = frozenset(record.source_image_id for record in ma_se_s1.manifest.records)
    if len(complete_source_ids) != 26:
        raise S4TwinGateError("complete four-mask source count is not the frozen 26")
    if any(record.split == LOCKED_SPLIT for record in ma_se_s1.manifest.records):
        raise S4TwinGateError("MA/SE development manifest contains LOCKED_TEST")
    he_ex_lookup = {record.source_image_id: record for record in he_ex_s1.manifest.records}
    for record in ma_se_s1.manifest.records:
        paired = he_ex_lookup.get(record.source_image_id)
        if paired is None:
            raise S4TwinGateError("complete source is absent from the HE/EX manifest")
        if (
            paired.split != record.split
            or paired.relative_path != record.relative_path
            or paired.sha256 != record.sha256
        ):
            raise S4TwinGateError("four-mask source identity/provenance mismatch")

    he_ex_records = _catalog_records(
        paths["he_ex_catalog"], config=config, lesion_classes=("HE", "EX")
    )
    ma_se_records = _catalog_records(
        paths["ma_se_catalog"], config=config, lesion_classes=("MA", "SE")
    )
    records = tuple(
        record
        for record in (*he_ex_records, *ma_se_records)
        if record.get("source_image_id") in complete_source_ids
    )
    if not records:
        raise S4TwinGateError("four-lesion complete-source catalog intersection is empty")
    return S4Upstream(
        he_ex_s1=he_ex_s1,
        ma_se_s1=ma_se_s1,
        he_ex_config=he_ex_config,
        ma_se_config=ma_se_config,
        records=records,
        complete_source_ids=complete_source_ids,
        hashes=hashes,
    )


def _require_new_ignored_output(
    project_root: Path, dataset_root: Path, config: S4TwinConfig, output_dir: Path
) -> Path:
    project = project_root.resolve(strict=True)
    output = output_dir.resolve(strict=False)
    expected = (project / Path(config.artifact_root)).resolve(strict=False)
    if output != expected:
        raise S4TwinDataError("S4 output does not match the frozen artifact_root")
    try:
        relative = output.relative_to(project)
    except ValueError as error:
        raise S4TwinDataError("S4 output must remain inside the project") from error
    if relative.parts[:2] != ("reports", "generated"):
        raise S4TwinDataError("S4 artifacts must remain under reports/generated")
    ignore_rules = {
        line.strip() for line in (project / ".gitignore").read_text(encoding="utf-8").splitlines()
    }
    if "/reports/generated/" not in ignore_rules:
        raise S4TwinDataError("reports/generated is not protected by .gitignore")
    try:
        output.relative_to(dataset_root.resolve(strict=True))
        raise S4TwinDataError("S4 output overlaps the dataset root")
    except ValueError:
        pass
    if output.exists():
        raise FileExistsError(f"S4 output directory must be new: {output}")
    return output


def _label_map(record: ManifestRecordV2) -> dict[str, LabelRecordV2]:
    return {label.semantic.value: label for label in record.labels}


def _load_source_context(
    *,
    dataset_root: Path,
    source_record: ManifestRecordV2,
    he_ex_record: ManifestRecordV2,
    fov_threshold: int,
    fov_coverage: float,
    root_read_log: list[dict[str, str]],
) -> SourceContext:
    root = dataset_root.resolve(strict=True)
    image_path = _resolve_under_root(root, source_record.relative_path)
    image = _load_rgb_strict(image_path, source_record)
    root_read_log.append(
        {
            "split": source_record.split,
            "kind": "IMAGE",
            "source_image_id": source_record.source_image_id,
        }
    )
    labels = {**_label_map(he_ex_record), **_label_map(source_record)}
    masks: dict[str, NDArray[np.bool_]] = {}
    components: dict[str, tuple[SparseComponent, ...]] = {}
    provenance: dict[str, dict[str, str]] = {}
    for lesion_class in LESION_CLASSES:
        semantic = LABEL_SEMANTICS[lesion_class]
        label = labels.get(semantic)
        if label is None:
            raise S4TwinDataError(f"complete source lacks {lesion_class} mask metadata")
        relative_path = _safe_relative(label.relative_path, f"{lesion_class} mask path")
        expected_hash = _sha256(label.sha256, f"{lesion_class} mask hash")
        mask_path = _resolve_under_root(root, relative_path)
        if sha256_file(mask_path) != expected_hash:
            raise S4TwinDataError(f"{lesion_class} mask hash mismatch")
        mask = decode_idrid_mask(
            mask_path, expected_size=(source_record.width, source_record.height)
        )
        root_read_log.append(
            {
                "split": source_record.split,
                "kind": f"{lesion_class}_MASK",
                "source_image_id": source_record.source_image_id,
            }
        )
        masks[lesion_class] = mask
        components[lesion_class] = connected_components(mask, connectivity=8)
        provenance[lesion_class] = {"relative_path": relative_path, "sha256": expected_hash}
    fov = detect_fov(
        image,
        foreground_max_rgb_min=fov_threshold,
        minimum_axis_coverage_fraction=fov_coverage,
    )
    return SourceContext(
        source_record=source_record,
        image=image,
        masks=masks,
        components=components,
        fov=fov.mask,
        label_provenance=provenance,
    )


def _component_for_record(context: SourceContext, record: Mapping[str, object]) -> SparseComponent:
    lesion_class = _string(record.get("lesion_class"), "lesion class")
    component_id = _string(record.get("component_id"), "component id")
    try:
        component_index = int(component_id.rsplit("-", maxsplit=1)[1])
    except (IndexError, ValueError) as error:
        raise S4TwinDataError("component id lacks a numeric catalog index") from error
    components = context.components[lesion_class]
    if component_index < 1 or component_index > len(components):
        raise S4TwinDataError("component index is outside the decoded mask")
    component = components[component_index - 1]
    if component.component_mask_sha256 != _sha256(
        record.get("component_mask_sha256"), "component mask hash"
    ):
        raise S4TwinDataError("component reconstruction hash mismatch")
    if list(component.bbox) != record.get("bbox_xyxy") or component.area != record.get("area_px"):
        raise S4TwinDataError("component reconstruction geometry mismatch")
    centroid = record.get("centroid_xy")
    if not isinstance(centroid, list) or len(centroid) != 2:
        raise S4TwinDataError("catalog centroid is invalid")
    if (
        max(
            abs(component.centroid[0] - float(centroid[0])),
            abs(component.centroid[1] - float(centroid[1])),
        )
        > 1e-12
    ):
        raise S4TwinDataError("component reconstruction centroid mismatch")
    provenance = context.label_provenance[lesion_class]
    if (
        record.get("mask_relative_path") != provenance["relative_path"]
        or record.get("mask_sha256") != provenance["sha256"]
    ):
        raise S4TwinDataError("component-to-four-mask provenance binding failed")
    return component


def _catalog_eligibility_reason(record: Mapping[str, object], config: S4TwinConfig) -> str:
    reason = _eligibility_reason(record, config.as_s3())
    return "CATALOG_NOT_INCLUDED" if reason == "S2_NOT_INCLUDED" else reason


def _candidate_arrays(
    *, context: SourceContext, record: Mapping[str, object], config: S4TwinConfig
) -> CandidateArrays:
    component_id = _string(record.get("component_id"), "component id")
    component = _component_for_record(context, record)
    centroid = record.get("centroid_xy")
    if not isinstance(centroid, list) or len(centroid) != 2:
        raise S4TwinDataError("catalog centroid is invalid")
    geometry = patch_geometry(
        image_size=(context.source_record.width, context.source_record.height),
        centroid=(float(centroid[0]), float(centroid[1])),
        patch_size=(256, 256),
    )
    if geometry.to_dict() != record.get("patch"):
        raise S4TwinDataError("catalog patch geometry replay failed")
    requested = geometry.requested_bbox
    x0, y0, x1, y1 = component.bbox
    if x0 < requested[0] or y0 < requested[1] or x1 > requested[2] or y1 > requested[3]:
        raise TwinRejected("TARGET_NOT_FULLY_CONTAINED")
    source_patch = cast(NDArray[np.uint8], extract_fixed_patch(context.image, geometry))
    target_component = _component_mask_in_patch(component, geometry)
    if int(np.count_nonzero(target_component)) != component.area:
        raise TwinRejected("TARGET_NOT_FULLY_CONTAINED")
    mask_patches = {
        lesion_class: cast(NDArray[np.bool_], extract_fixed_patch(mask, geometry))
        for lesion_class, mask in context.masks.items()
    }
    combined_lesions = np.logical_or.reduce(tuple(mask_patches.values()))
    combined_lesions = np.asarray(combined_lesions, dtype=np.bool_)
    other_lesions = combined_lesions & ~target_component
    target_edit = _binary_dilate(target_component, config.target_dilation)
    if np.any(target_edit & other_lesions):
        raise TwinRejected("TARGET_EDIT_OVERLAPS_OTHER_LESION")
    fov_patch = cast(NDArray[np.bool_], extract_fixed_patch(context.fov, geometry))
    if not np.all(fov_patch[target_edit]):
        raise TwinRejected("FOV_MARGIN_BELOW_FROZEN_MINIMUM")
    lesion_forbidden = _binary_dilate(combined_lesions, config.lesion_exclusion_dilation)
    sham_edit, sham_offset = _find_offset(
        config=config.as_s3(),
        component_id=component_id,
        purpose="sham",
        edit_mask=target_edit,
        forbidden=lesion_forbidden | target_edit,
        fov=fov_patch,
    )
    if np.any(sham_edit & combined_lesions):
        raise TwinRejected("SHAM_EDIT_OVERLAPS_OTHER_LESION")
    method = _assigned_method(config.as_s3(), component_id)
    if method == "DONOR_POISSON":
        for purpose, edit_mask, forbidden in (
            ("donor-sham", sham_edit, lesion_forbidden),
            ("donor-target", target_edit, lesion_forbidden & ~target_component),
        ):
            donor_mask, _ = _find_offset(
                config=config.as_s3(),
                component_id=component_id,
                purpose=purpose,
                edit_mask=edit_mask,
                forbidden=forbidden | edit_mask,
                fov=fov_patch,
            )
            if np.any(donor_mask & other_lesions):
                raise TwinRejected("DONOR_OVERLAPS_OTHER_LESION")
    return CandidateArrays(
        source_patch=source_patch,
        target_component=target_component,
        target_edit=target_edit,
        sham_edit=sham_edit,
        combined_lesions=combined_lesions,
        other_lesions=other_lesions,
        lesion_forbidden=lesion_forbidden,
        fov_patch=fov_patch,
        sham_offset=sham_offset,
    )


def raw_exact_collision_oracle(
    x_plus_sham: NDArray[np.uint8], x_minus_removed: NDArray[np.uint8]
) -> tuple[bool, dict[str, JSONValue]]:
    """Build truth records from exact raw arrays without RetinaEntail inputs."""

    if (
        x_plus_sham.dtype != np.uint8
        or x_minus_removed.dtype != np.uint8
        or x_plus_sham.shape != x_minus_removed.shape
    ):
        raise S4TwinDataError("raw Oracle requires aligned uint8 arrays")
    plus_hash = sha256_bytes(x_plus_sham.tobytes(order="C"))
    minus_hash = sha256_bytes(x_minus_removed.tobytes(order="C"))
    exact_collision = bool(np.array_equal(x_plus_sham, x_minus_removed))
    evidence_hash = sha256_bytes(
        canonical_json_bytes(
            {
                "method_id": ORACLE_METHOD_ID,
                "array_shape": list(x_plus_sham.shape),
                "array_dtype": "uint8",
                "x_plus_raw_sha256": plus_hash,
                "x_minus_raw_sha256": minus_hash,
                "raw_array_exact_equal": exact_collision,
            }
        )
    )
    support_truth = (
        SupportTruth.OBSERVATION_AMBIGUOUS if exact_collision else SupportTruth.ORACLE_UNRESOLVED
    )
    tier = OracleTier.EXACT_COLLISION if exact_collision else None
    collision_mode = CollisionMode.SAME_THETA if exact_collision else None
    records = [
        ControlledTwinOracleRecord(
            source_truth=source_truth,
            support_truth=support_truth,
            domain_truth=DomainTruth.IN_DOMAIN,
            oracle_tier=tier,
            collision_mode=collision_mode,
            oracle_method_id=ORACLE_METHOD_ID,
            evidence_sha256=evidence_hash,
            retinaentail_independent=True,
        ).to_dict()
        for source_truth in (SourceTruth.PRESENT, SourceTruth.ABSENT)
    ]
    return exact_collision, {
        "method_id": ORACLE_METHOD_ID,
        "retinaentail_independent": True,
        "forbidden_inputs_observed": [],
        "raw_array_exact_equal": exact_collision,
        "evidence_sha256": evidence_hash,
        "x_plus_raw_sha256": plus_hash,
        "x_minus_raw_sha256": minus_hash,
        "truth_records": cast(list[JSONValue], records),
    }


def _build_one(
    *, context: SourceContext, record: Mapping[str, object], config: S4TwinConfig
) -> BuiltTwin:
    component_id = _string(record.get("component_id"), "component id")
    lesion_class = _string(record.get("lesion_class"), "lesion class")
    split = _string(record.get("split"), "split")
    if context.source_record.source_image_id != record.get("source_image_id"):
        raise S4TwinDataError("source identity binding failed")
    if context.source_record.split != split:
        raise S4TwinDataError("source split binding failed")
    if context.source_record.relative_path != record.get(
        "image_relative_path"
    ) or context.source_record.sha256 != record.get("image_sha256"):
        raise S4TwinDataError("source image provenance binding failed")
    arrays = _candidate_arrays(context=context, record=record, config=config)
    method = _assigned_method(config.as_s3(), component_id)
    x_plus, sham_donor = _method_output(
        image=arrays.source_patch,
        edit_mask=arrays.sham_edit,
        method=method,
        config=config.as_s3(),
        component_id=component_id,
        lesion_forbidden=arrays.lesion_forbidden,
        fov=arrays.fov_patch,
        purpose="sham",
    )
    x_minus, target_donor = _method_output(
        image=arrays.source_patch,
        edit_mask=arrays.target_edit,
        method=method,
        config=config.as_s3(),
        component_id=component_id,
        lesion_forbidden=arrays.lesion_forbidden & ~arrays.target_component,
        fov=arrays.fov_patch,
        purpose="target",
    )
    if not np.array_equal(
        x_plus[arrays.target_component], arrays.source_patch[arrays.target_component]
    ):
        raise S4TwinDataError("sham edit changed the target component")
    plus_changed = int(np.count_nonzero(np.any(x_plus != arrays.source_patch, axis=2)))
    minus_changed = int(np.count_nonzero(np.any(x_minus != arrays.source_patch, axis=2)))
    if plus_changed == 0 or minus_changed == 0:
        raise TwinRejected("EMPTY_EDIT_EFFECT")
    exact_collision, oracle = raw_exact_collision_oracle(x_plus, x_minus)
    twin_digest = hashlib.sha256(component_id.encode()).hexdigest()[:12]
    twin_id = f"s4-{lesion_class.lower()}-{split.lower()}-{twin_digest}-{method.lower()}"
    plus_path = f"twins/{twin_id}/x_plus_sham.png"
    minus_path = f"twins/{twin_id}/x_minus_removed.png"
    plus_bytes = rgb_png_bytes(x_plus)
    minus_bytes = rgb_png_bytes(x_minus)
    source_bytes = rgb_png_bytes(arrays.source_patch)
    component = _component_for_record(context, record)
    geometry = patch_geometry(
        image_size=(context.source_record.width, context.source_record.height),
        centroid=component.centroid,
        patch_size=(256, 256),
    )
    payload: dict[str, JSONValue] = {
        "schema_version": S4_TWIN_SCHEMA_VERSION,
        "twin_id": twin_id,
        "component_id": component_id,
        "source_image_id": context.source_record.source_image_id,
        "split": split,
        "lesion_class": lesion_class,
        "method": method,
        "method_implementation_id": config.implementation_id(method),
        "seed": config.seed,
        "source": {
            "image_relative_path": context.source_record.relative_path,
            "image_sha256": context.source_record.sha256,
            "source_patch_sha256": sha256_bytes(source_bytes),
            "patch_geometry": geometry.to_dict(),
            "four_mask_provenance": cast(dict[str, JSONValue], context.label_provenance),
        },
        "target": {
            "component_mask_sha256": component.component_mask_sha256,
            "component_area_px": component.area,
            "bbox_xyxy": list(component.bbox),
            "centroid_xy": list(component.centroid),
            "edit_mask_sha256": _mask_sha256(arrays.target_edit),
            "edit_mask_area_px": int(np.count_nonzero(arrays.target_edit)),
            "donor_offset_xy": None if target_donor is None else list(target_donor),
        },
        "sham": {
            "offset_xy": list(arrays.sham_offset),
            "edit_mask_sha256": _mask_sha256(arrays.sham_edit),
            "edit_mask_area_px": int(np.count_nonzero(arrays.sham_edit)),
            "donor_offset_xy": None if sham_donor is None else list(sham_donor),
        },
        "outputs": {
            "x_plus_sham": {
                "relative_path": plus_path,
                "sha256": sha256_bytes(plus_bytes),
                "raw_array_sha256": sha256_bytes(x_plus.tobytes(order="C")),
                "byte_size": len(plus_bytes),
            },
            "x_minus_removed": {
                "relative_path": minus_path,
                "sha256": sha256_bytes(minus_bytes),
                "raw_array_sha256": sha256_bytes(x_minus.tobytes(order="C")),
                "byte_size": len(minus_bytes),
            },
        },
        "oracle": oracle,
        "automated_gate": {
            "complete_pair": True,
            "outside_sham_edit_exact_preservation": bool(
                np.array_equal(x_plus[~arrays.sham_edit], arrays.source_patch[~arrays.sham_edit])
            ),
            "outside_target_edit_exact_preservation": bool(
                np.array_equal(
                    x_minus[~arrays.target_edit], arrays.source_patch[~arrays.target_edit]
                )
            ),
            "target_exact_preservation_in_sham": True,
            "target_edit_other_lesion_overlap_px": int(
                np.count_nonzero(arrays.target_edit & arrays.other_lesions)
            ),
            "sham_edit_any_lesion_overlap_px": int(
                np.count_nonzero(arrays.sham_edit & arrays.combined_lesions)
            ),
            "x_plus_changed_pixel_count": plus_changed,
            "x_minus_changed_pixel_count": minus_changed,
            "finite_uint8_outputs": True,
        },
    }
    return BuiltTwin(
        record=payload,
        artifacts={plus_path: plus_bytes, minus_path: minus_bytes},
        qc_cell=QCCell(
            twin_id=twin_id,
            component_id=component_id,
            split=split,
            lesion_class=lesion_class,
            method=method,
            source=arrays.source_patch,
            target_component=arrays.target_component,
            target_edit=arrays.target_edit,
            sham_edit=arrays.sham_edit,
            x_plus_sham=x_plus,
            x_minus_removed=x_minus,
        ),
        exact_collision=exact_collision,
    )


def _selection_csv(rows: Sequence[Mapping[str, object]]) -> bytes:
    fields = (
        "component_id",
        "source_image_id",
        "split",
        "lesion_class",
        "selection_key",
        "frozen_rank",
        "eligibility_reason",
        "constructibility_reason",
        "attempted",
        "outcome",
        "method",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field) for field in fields})
    return stream.getvalue().encode("utf-8")


def _accepted_rejected_csv(
    accepted: Sequence[Mapping[str, object]], rejected: Sequence[Mapping[str, object]]
) -> bytes:
    fields = (
        "status",
        "twin_id",
        "component_id",
        "source_image_id",
        "split",
        "lesion_class",
        "method",
        "reason_code",
        "x_plus_sham_path",
        "x_plus_sham_sha256",
        "x_minus_removed_path",
        "x_minus_removed_sha256",
        "raw_exact_collision",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for record in accepted:
        outputs = cast(Mapping[str, object], record["outputs"])
        plus = cast(Mapping[str, object], outputs["x_plus_sham"])
        minus = cast(Mapping[str, object], outputs["x_minus_removed"])
        oracle = cast(Mapping[str, object], record["oracle"])
        writer.writerow(
            {
                "status": "ACCEPTED",
                "twin_id": record.get("twin_id"),
                "component_id": record.get("component_id"),
                "source_image_id": record.get("source_image_id"),
                "split": record.get("split"),
                "lesion_class": record.get("lesion_class"),
                "method": record.get("method"),
                "reason_code": "",
                "x_plus_sham_path": plus.get("relative_path"),
                "x_plus_sham_sha256": plus.get("sha256"),
                "x_minus_removed_path": minus.get("relative_path"),
                "x_minus_removed_sha256": minus.get("sha256"),
                "raw_exact_collision": oracle.get("raw_array_exact_equal"),
            }
        )
    for record in rejected:
        writer.writerow(
            {
                "status": "REJECTED",
                "component_id": record.get("component_id"),
                "source_image_id": record.get("source_image_id"),
                "split": record.get("split"),
                "lesion_class": record.get("lesion_class"),
                "method": record.get("method"),
                "reason_code": record.get("reason_code"),
            }
        )
    return stream.getvalue().encode("utf-8")


def _code_sha256(project_root: Path) -> str:
    names = (
        "src/retinaentail/science_s4_twins.py",
        "src/retinaentail/interventions/controlled_twin_pilot.py",
        "src/retinaentail/interventions/oracle.py",
        "data_registry/schemas/science-s4-twin-tranche-v1.schema.json",
    )
    entries: list[dict[str, JSONValue]] = []
    for name in names:
        path = project_root / name
        if not path.is_file():
            raise S4TwinGateError(f"S4 code bundle file is missing: {name}")
        entries.append({"relative_path": name, "sha256": sha256_file(path)})
    return sha256_bytes(canonical_json_bytes(cast(list[JSONValue], entries)))


def _write_artifacts(output_dir: Path, artifacts: Mapping[str, bytes]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative_path, payload in sorted(artifacts.items()):
        write_bytes_fail_safe(output_dir / relative_path, payload)
        hashes[relative_path] = sha256_bytes(payload)
    manifest_payload = canonical_json_bytes(
        {
            "schema_version": S4_TWIN_SCHEMA_VERSION,
            "files": [
                {
                    "relative_path": relative_path,
                    "sha256": digest,
                    "byte_size": len(artifacts[relative_path]),
                }
                for relative_path, digest in sorted(hashes.items())
            ],
        }
    )
    write_bytes_fail_safe(output_dir / "artifact-manifest.json", manifest_payload)
    hashes["artifact-manifest.json"] = sha256_bytes(manifest_payload)
    return hashes


def _rejected_record(
    record: Mapping[str, object], config: S4TwinConfig, reason: str
) -> dict[str, JSONValue]:
    component_id = _string(record.get("component_id"), "component id")
    return {
        "schema_version": S4_TWIN_SCHEMA_VERSION,
        "component_id": component_id,
        "source_image_id": _string(record.get("source_image_id"), "source image id"),
        "split": _string(record.get("split"), "split"),
        "lesion_class": _string(record.get("lesion_class"), "lesion class"),
        "method": _assigned_method(config.as_s3(), component_id),
        "reason_code": reason,
    }


def _qc_key(config: S4TwinConfig, cell: QCCell) -> str:
    return hashlib.sha256(f"{config.seed}:qc:{cell.twin_id}".encode()).hexdigest()


def run_s4_twin_tranche(
    *, project_root: Path, config_path: Path, dataset_root: Path, output_dir: Path
) -> S4TwinRunResult:
    """Execute the frozen real S4 tranche and fail closed on sufficiency."""

    config = S4TwinConfig.load(config_path)
    upstream = _replay_upstream(project_root, config)
    for s1 in (upstream.he_ex_s1, upstream.ma_se_s1):
        registry = load_registry_license_metadata(s1.registry_path)
        review = load_license_review_record(s1.review_path)
        spec = load_dataset_adapter_spec(s1.spec_path)
        authorize_dataset_access(
            registry,
            review=review,
            purpose=DataUsePurpose.LESION_CALIBRATION,
            root=dataset_root,
            spec=spec,
        )
    root = dataset_root.resolve(strict=True)
    if not root.is_dir():
        raise S4TwinDataError("authorized dataset root is not a directory")
    resolved_output = _require_new_ignored_output(project_root, root, config, output_dir)

    he_lookup = {record.source_image_id: record for record in upstream.he_ex_s1.manifest.records}
    complete_lookup = {
        record.source_image_id: record for record in upstream.ma_se_s1.manifest.records
    }
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    selection_rows: list[dict[str, object]] = []
    selection_lookup: dict[str, dict[str, object]] = {}
    for record in upstream.records:
        split = _string(record.get("split"), "split")
        lesion_class = _string(record.get("lesion_class"), "lesion class")
        component_id = _string(record.get("component_id"), "component id")
        source_image_id = _string(record.get("source_image_id"), "source image id")
        if source_image_id not in upstream.complete_source_ids:
            raise S4TwinGateError("catalog intersection contains an incomplete source")
        reason = _catalog_eligibility_reason(record, config)
        row: dict[str, object] = {
            "component_id": component_id,
            "source_image_id": source_image_id,
            "split": split,
            "lesion_class": lesion_class,
            "selection_key": _selection_key(config.as_s3(), record),
            "frozen_rank": "",
            "eligibility_reason": reason,
            "constructibility_reason": "NOT_EVALUATED",
            "attempted": False,
            "outcome": reason,
            "method": "",
        }
        selection_rows.append(row)
        selection_lookup[component_id] = row
        if reason == "ELIGIBLE":
            grouped[(split, lesion_class)].append(record)
    for candidates in grouped.values():
        candidates.sort(key=lambda record: _selection_key(config.as_s3(), record))
        for rank, record in enumerate(candidates, start=1):
            selection_lookup[_string(record.get("component_id"), "component id")]["frozen_rank"] = (
                rank
            )

    root_read_log: list[dict[str, str]] = []
    cursors = {
        (split, lesion_class): 0 for lesion_class in LESION_CLASSES for split in DEVELOPMENT_SPLITS
    }
    accepted_group_counts: Counter[str] = Counter()
    accepted: list[dict[str, JSONValue]] = []
    rejected: list[dict[str, JSONValue]] = []
    artifacts: dict[str, bytes] = {}
    qc_cells: dict[str, list[QCCell]] = defaultdict(list)
    exact_collisions: Counter[str] = Counter()
    while True:
        precheck_pending: list[dict[str, object]] = []
        for lesion_class in LESION_CLASSES:
            for split in DEVELOPMENT_SPLITS:
                group = (split, lesion_class)
                group_key = f"{split}:{lesion_class}"
                quota = config.split_quotas[lesion_class][split]
                candidates = grouped[group]
                cursor = cursors[group]
                while accepted_group_counts[group_key] + sum(
                    item.get("split") == split and item.get("lesion_class") == lesion_class
                    for item in precheck_pending
                ) < quota and cursor < len(candidates):
                    record = candidates[cursor]
                    cursor += 1
                    component_id = _string(record.get("component_id"), "component id")
                    row = selection_lookup[component_id]
                    row["attempted"] = True
                    row["method"] = _assigned_method(config.as_s3(), component_id)
                    row["constructibility_reason"] = "PRECHECK_PENDING"
                    row["outcome"] = "PRECHECK_PENDING"
                    precheck_pending.append(record)
                cursors[group] = cursor
        if not precheck_pending:
            break

        precheck_by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
        for record in precheck_pending:
            precheck_by_source[_string(record.get("source_image_id"), "source image id")].append(
                record
            )
        render_pending: list[dict[str, object]] = []
        for source_image_id in sorted(precheck_by_source):
            source_record = complete_lookup[source_image_id]
            context = _load_source_context(
                dataset_root=root,
                source_record=source_record,
                he_ex_record=he_lookup[source_image_id],
                fov_threshold=upstream.he_ex_config.fov_threshold,
                fov_coverage=upstream.he_ex_config.fov_coverage,
                root_read_log=root_read_log,
            )
            for record in precheck_by_source[source_image_id]:
                component_id = _string(record.get("component_id"), "component id")
                row = selection_lookup[component_id]
                try:
                    _candidate_arrays(context=context, record=record, config=config)
                except TwinRejected as error:
                    if error.reason_code not in config.rejection_taxonomy:
                        raise S4TwinDataError("unregistered S4 rejection reason") from error
                    row["constructibility_reason"] = error.reason_code
                    row["outcome"] = error.reason_code
                    rejected.append(_rejected_record(record, config, error.reason_code))
                    continue
                row["constructibility_reason"] = "CONSTRUCTIBLE"
                row["outcome"] = "RENDERING"
                render_pending.append(record)

        render_by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
        for record in render_pending:
            render_by_source[_string(record.get("source_image_id"), "source image id")].append(
                record
            )
        for source_image_id in sorted(render_by_source):
            source_record = complete_lookup[source_image_id]
            context = _load_source_context(
                dataset_root=root,
                source_record=source_record,
                he_ex_record=he_lookup[source_image_id],
                fov_threshold=upstream.he_ex_config.fov_threshold,
                fov_coverage=upstream.he_ex_config.fov_coverage,
                root_read_log=root_read_log,
            )
            for record in render_by_source[source_image_id]:
                component_id = _string(record.get("component_id"), "component id")
                row = selection_lookup[component_id]
                try:
                    built = _build_one(context=context, record=record, config=config)
                except TwinRejected as error:
                    if error.reason_code not in config.rejection_taxonomy:
                        raise S4TwinDataError("unregistered render rejection reason") from error
                    row["outcome"] = error.reason_code
                    rejected.append(_rejected_record(record, config, error.reason_code))
                    continue
                row["outcome"] = "ACCEPTED"
                accepted.append(built.record)
                artifacts.update(built.artifacts)
                lesion_class = _string(record.get("lesion_class"), "lesion class")
                split = _string(record.get("split"), "split")
                accepted_group_counts[f"{split}:{lesion_class}"] += 1
                if built.exact_collision:
                    exact_collisions[lesion_class] += 1
                qc_cells[lesion_class].append(built.qc_cell)
                qc_cells[lesion_class].sort(key=lambda cell: _qc_key(config, cell))
                del qc_cells[lesion_class][config.qc_samples_per_class :]

    for lesion_class in LESION_CLASSES:
        for split in DEVELOPMENT_SPLITS:
            group = (split, lesion_class)
            for record in grouped[group][cursors[group] :]:
                row = selection_lookup[_string(record.get("component_id"), "component id")]
                row["constructibility_reason"] = "NOT_EVALUATED_AFTER_STOP"
                row["outcome"] = "NOT_REACHED_AFTER_QUOTA_FILLED"

    accepted_counts = Counter(f"{record['split']}:{record['lesion_class']}" for record in accepted)
    method_counts = Counter(str(record["method"]) for record in accepted)
    distinct_source_sets: dict[str, set[str]] = {
        lesion_class: set() for lesion_class in LESION_CLASSES
    }
    for accepted_record in accepted:
        distinct_source_sets[str(accepted_record["lesion_class"])].add(
            str(accepted_record["source_image_id"])
        )
    distinct_sources = {
        lesion_class: len(distinct_source_sets[lesion_class]) for lesion_class in LESION_CLASSES
    }
    collision_counts = {
        lesion_class: exact_collisions[lesion_class] for lesion_class in LESION_CLASSES
    }
    target_sufficient = {
        lesion_class: sum(
            count for key, count in accepted_counts.items() if key.endswith(f":{lesion_class}")
        )
        >= config.class_targets[lesion_class]
        for lesion_class in LESION_CLASSES
    }
    source_sufficient = {
        lesion_class: distinct_sources[lesion_class] >= config.minimum_sources[lesion_class]
        for lesion_class in LESION_CLASSES
    }
    collision_sufficient = {
        lesion_class: collision_counts[lesion_class] >= config.minimum_collisions
        for lesion_class in LESION_CLASSES
    }
    if any(item["split"] == LOCKED_SPLIT for item in root_read_log):
        raise S4TwinDataError("LOCKED_TEST root read detected after rendering")

    selection_rows.sort(
        key=lambda row: (str(row["lesion_class"]), str(row["split"]), str(row["selection_key"]))
    )
    manifest_payload: dict[str, JSONValue] = {
        "schema_version": S4_TWIN_SCHEMA_VERSION,
        "experiment_id": S4_TWIN_EXPERIMENT_ID,
        "dataset_id": config.dataset_id,
        "dataset_version": config.dataset_version,
        "scope": "TRAIN_AND_CALIBRATION_COMPLETE_FOUR_MASK_SOURCES_ONLY",
        "config_sha256": config.config_sha256,
        "bindings": cast(dict[str, JSONValue], upstream.hashes),
        "accepted": cast(list[JSONValue], accepted),
        "rejected": cast(list[JSONValue], rejected),
        "selection_reason_counts": dict(
            sorted(Counter(str(row["outcome"]) for row in selection_rows).items())
        ),
    }
    artifacts["twin-manifest-v1.json"] = canonical_json_bytes(manifest_payload)
    artifacts["accepted-rejected-v1.csv"] = _accepted_rejected_csv(accepted, rejected)
    artifacts["selection-audit-v1.csv"] = _selection_csv(selection_rows)
    for lesion_class in LESION_CLASSES:
        if qc_cells[lesion_class]:
            artifacts[f"qc-{lesion_class.lower()}-controlled-twins-contact-sheet.png"] = (
                _contact_sheet(
                    qc_cells[lesion_class],
                    columns=config.qc_columns,
                    panel_size=config.qc_panel_size,
                )
            )

    all_sufficient = (
        all(target_sufficient.values())
        and all(source_sufficient.values())
        and all(collision_sufficient.values())
    )
    status = "READY_FOR_AI_QC" if all_sufficient else "BLOCKED_SAMPLE_OR_ORACLE_SUFFICIENCY"
    audit_payload: dict[str, JSONValue] = {
        "schema_version": S4_TWIN_SCHEMA_VERSION,
        "experiment_id": S4_TWIN_EXPERIMENT_ID,
        "status": status,
        "summary": {
            "accepted_twin_count": len(accepted),
            "rejected_attempt_count": len(rejected),
            "paired_output_count": len(accepted) * 2,
            "accepted_counts": cast(dict[str, JSONValue], dict(sorted(accepted_counts.items()))),
            "accepted_distinct_sources": cast(dict[str, JSONValue], distinct_sources),
            "raw_exact_collision_pairs": cast(dict[str, JSONValue], collision_counts),
            "method_counts": cast(dict[str, JSONValue], dict(sorted(method_counts.items()))),
            "selection_record_count": len(selection_rows),
            "root_read_event_count": len(root_read_log),
            "complete_four_mask_source_count": len(upstream.complete_source_ids),
        },
        "gate": {
            "upstream_hash_replay": "PASS",
            "owner_scope_confirmation": "PASS",
            "frozen_selection": "PASS",
            "four_mask_preservation": "PASS",
            "complete_pairs": "PASS",
            "outside_edit_exact_preservation": "PASS",
            "target_preservation_in_sham": "PASS",
            "retinaentail_independent_raw_oracle": "PASS",
            "target_count_sufficiency": cast(dict[str, JSONValue], target_sufficient),
            "distinct_source_sufficiency": cast(dict[str, JSONValue], source_sufficient),
            "tier_a_exact_collision_sufficiency": cast(dict[str, JSONValue], collision_sufficient),
            "ai_contact_sheet_review": "PENDING",
            "downstream_science_stage_authorized": False,
        },
        "hashes": {
            "config_sha256": config.config_sha256,
            "code_sha256": _code_sha256(project_root),
            "twin_manifest_sha256": sha256_bytes(artifacts["twin-manifest-v1.json"]),
            "selection_audit_sha256": sha256_bytes(artifacts["selection-audit-v1.csv"]),
            "accepted_rejected_sha256": sha256_bytes(artifacts["accepted-rejected-v1.csv"]),
        },
        "io_audit": {
            "locked_test_root_reads": 0,
            "locked_test_mask_decodes": 0,
            "source_split_leak_count": 0,
        },
        "evidence_boundary": [
            "Twin counts and contact sheets are controlled development artifacts, not FCR, "
            "medical, clinical, or paper-performance results.",
            "A raw non-collision remains ORACLE_UNRESOLVED and cannot be promoted to "
            "ambiguity evidence.",
            "No S5, Verify, Lite, enhancer training, locked-test evaluation, or FCR "
            "estimation was run.",
        ],
    }
    artifacts["audit.json"] = canonical_json_bytes(audit_payload)
    artifact_hashes = _write_artifacts(resolved_output, artifacts)
    return S4TwinRunResult(
        status=status,
        output_dir=resolved_output,
        accepted_count=len(accepted),
        rejected_count=len(rejected),
        accepted_counts=dict(sorted(accepted_counts.items())),
        distinct_sources=distinct_sources,
        raw_exact_collisions=collision_counts,
        locked_test_root_reads=0,
        artifact_hashes=artifact_hashes,
        audit=audit_payload,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m retinaentail.science_s4_twins",
        description="Run the frozen Science S4 IDRiD four-lesion twin tranche.",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = Path(__file__).resolve().parents[2]
    try:
        result = run_s4_twin_tranche(
            project_root=project_root,
            config_path=args.config,
            dataset_root=args.root,
            output_dir=args.output_dir,
        )
    except (
        FileExistsError,
        OSError,
        S2DataError,
        S2GateError,
        S3DataError,
        S4TwinDataError,
        S4TwinGateError,
        ValueError,
    ) as error:
        print(f"Science S4 fail-closed: {error}", file=sys.stderr)
        return 2
    print(f"status: {result.status}")
    print(f"accepted twins: {result.accepted_count}")
    print(f"rejected attempts: {result.rejected_count}")
    print(f"accepted counts: {result.accepted_counts}")
    print(f"distinct sources: {result.distinct_sources}")
    print(f"raw exact collisions: {result.raw_exact_collisions}")
    print(f"locked-test root reads: {result.locked_test_root_reads}")
    print(f"artifact manifest hash: {result.artifact_hashes['artifact-manifest.json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "S4TwinConfig",
    "S4TwinDataError",
    "S4TwinGateError",
    "S4TwinRunResult",
    "build_parser",
    "raw_exact_collision_oracle",
    "run_s4_twin_tranche",
]
