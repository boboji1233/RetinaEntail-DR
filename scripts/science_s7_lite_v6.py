"""Train and evaluate the frozen Science S7 v6 small paired encoder."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import subprocess
import sys
import tracemalloc
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, cast

import numpy as np
import torch
import torch.nn.functional as functional
import yaml
from numpy.typing import NDArray
from PIL import Image
from torch import Tensor, nn

from retinaentail.data.manifest import sha256_file
from retinaentail.entailment.states import EntailmentState
from retinaentail.science_s5_degradation_calibration import (
    _canonical,
    _mapping,
    _safe_relative_string,
    _sequence,
    _string,
    _write_artifacts,
)
from retinaentail.science_s6_verify import HYPOTHESES, LESION_CLASSES
from retinaentail.science_s7_lite import (
    CALIBRATION_HIERARCHIES,
    DETERMINATE_STATES,
    S7Config,
    ThresholdResolution,
    _apply_multi_region,
    _cluster_bootstrap_fcr,
    _p_value,
    _policy_metrics,
)
from retinaentail.science_s7_lite_v2 import _audit_manifest, _v1_probe_hash_audit
from retinaentail.science_s7_lite_v4 import _build_calibration_rows, _state, _threshold
from retinaentail.witnesses.replay import sha256_bytes

SCHEMA_VERSION: Final = "6.0.0"
EXPERIMENT_ID: Final = "S7_RETINAENTAIL_LITE_V6"
READY_STATUS: Final = "READY_FOR_AI_QC"
BLOCKED_STATUS: Final = "BLOCKED_AUTOMATED_GATE"
PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]


class S7LiteV6Error(ValueError):
    """Raised when the frozen v6 contract cannot execute safely."""


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise S7LiteV6Error(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: object, name: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise S7LiteV6Error(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise S7LiteV6Error(f"{name} must be finite and >= {minimum}")
    return result


def _sha256(value: object, name: str) -> str:
    text = _string(value, name).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise S7LiteV6Error(f"{name} is not SHA-256")
    return text


def _verify_binding(raw: object, name: str) -> Path:
    binding = _mapping(raw, f"binding {name}")
    relative = _safe_relative_string(binding.get("path"), f"{name} path")
    expected = _sha256(binding.get("sha256"), f"{name} SHA-256")
    path = PROJECT_ROOT.joinpath(*PurePosixPath(relative).parts)
    if not path.is_file() or sha256_file(path) != expected:
        raise S7LiteV6Error(f"binding mismatch: {name}")
    return path


def _verify_baseline(expected: object) -> None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stdout.strip() != _string(expected, "expected baseline commit"):
        raise S7LiteV6Error("v6 baseline drift")


@dataclass(frozen=True, slots=True)
class V6Config:
    path: Path
    config_sha256: str
    seed: int
    train_fraction: float
    epochs: int
    batch_size: int
    learning_rate: float
    embedding_dim: int
    head_penalty_weight: float
    bootstrap_replicates: int
    maximum_fcr: float
    minimum_moderate_coverage: float
    minimum_verify_agreement: float
    bindings: dict[str, Path]

    @classmethod
    def load(cls, path: Path) -> V6Config:
        resolved = path.resolve(strict=True)
        decoded: object = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        root = _mapping(decoded, "S7 v6 config")
        if (
            root.get("schema_version") != SCHEMA_VERSION
            or root.get("experiment_id") != EXPERIMENT_ID
            or root.get("track") != "SCIENCE"
            or root.get("status") != "EXECUTABLE_RUN_INFORMED_SMALL_ENCODER_AMENDMENT"
        ):
            raise S7LiteV6Error("unsupported S7 v6 config")
        _verify_baseline(root.get("expected_baseline_commit"))
        disclosure = _mapping(root.get("development_disclosure"), "disclosure")
        if disclosure != {
            "v1_to_v5_observed": True,
            "independent_replication": False,
            "final_test_opened": False,
        }:
            raise S7LiteV6Error("v6 disclosure changed")
        method = _mapping(root.get("method"), "method")
        if method.get("runtime") != "PYTHON_3_11_TORCH_2_5_1_CPU":
            raise S7LiteV6Error("v6 runtime changed")
        if method.get("architecture") != (
            "CONV_3_16_CONV_16_24S2_CONV_24_32S2_GAP_LINEAR_32_16_FOUR_HEADS"
        ):
            raise S7LiteV6Error("v6 architecture changed")
        if method.get("witness_policy") != "REUSE_V1_POSITIVE_NEGATIVE_BEST_CANDIDATES":
            raise S7LiteV6Error("v6 witness policy changed")
        if method.get("distance") != "EMBEDDING_MSE_PLUS_AUXILIARY_HEAD_PENALTY":
            raise S7LiteV6Error("v6 score changed")
        split = _mapping(root.get("source_split"), "source split")
        if split.get("validation_role") != "REUSE_V4_DEVELOPMENT_VALIDATION":
            raise S7LiteV6Error("v6 validation role changed")
        if split.get("nested_hash") != "SHA256_SEED_SOURCE_KEY_FIRST64":
            raise S7LiteV6Error("v6 split hash changed")
        bindings_raw = _mapping(root.get("bindings"), "bindings")
        bindings = {name: _verify_binding(raw, name) for name, raw in bindings_raw.items()}
        required = {
            "v6_prompt",
            "v6_code",
            "v6_audit_schema",
            "v1_config",
            "v1_audit",
            "v1_artifact_manifest",
            "v4_source_split",
            "v5_audit",
            "v5_artifact_manifest",
            "v5_results",
        }
        if set(bindings) != required:
            raise S7LiteV6Error("v6 binding set changed")
        training = _mapping(root.get("training"), "training")
        gate = _mapping(root.get("gate"), "gate")
        train_fraction = _number(split.get("model_train_fraction"), "train fraction")
        epochs = _integer(training.get("epochs"), "epochs", 1)
        batch_size = _integer(training.get("batch_size"), "batch size", 1)
        learning_rate = _number(training.get("learning_rate"), "learning rate")
        embedding_dim = _integer(method.get("embedding_dim"), "embedding dim", 1)
        head_penalty_weight = _number(method.get("head_penalty_weight"), "head penalty weight")
        if (
            train_fraction != 0.70
            or epochs != 100
            or batch_size != 32
            or learning_rate != 0.001
            or embedding_dim != 16
            or head_penalty_weight != 0.10
            or training.get("optimizer") != "ADAM"
        ):
            raise S7LiteV6Error("v6 frozen optimization contract changed")
        return cls(
            path=resolved,
            config_sha256=sha256_file(resolved),
            seed=_integer(root.get("seed"), "seed"),
            train_fraction=train_fraction,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            embedding_dim=embedding_dim,
            head_penalty_weight=head_penalty_weight,
            bootstrap_replicates=_integer(
                _mapping(root.get("statistics"), "statistics").get("bootstrap_replicates"),
                "bootstrap replicates",
                1,
            ),
            maximum_fcr=_number(gate.get("maximum_validation_fcr"), "FCR gate"),
            minimum_moderate_coverage=_number(
                gate.get("minimum_validation_moderate_coverage_per_class"), "coverage gate"
            ),
            minimum_verify_agreement=_number(
                gate.get("minimum_validation_lite_verify_agreement"), "agreement gate"
            ),
            bindings=bindings,
        )


class SmallPairedEncoder(nn.Module):
    """Tiny shared encoder with class-conditional auxiliary heads."""

    def __init__(self, embedding_dim: int = 16) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 24, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.embedding = nn.Linear(32, embedding_dim)
        self.heads = nn.Linear(embedding_dim, 4)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        pooled = self.features(inputs).flatten(1)
        embedding = functional.normalize(self.embedding(pooled), dim=1)
        return embedding, self.heads(embedding)


def nested_role(seed: int, source_key: str, original_role: str, train_fraction: float) -> str:
    if original_role == "DEVELOPMENT_VALIDATION":
        return original_role
    digest = hashlib.sha256(f"{seed}|{source_key}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return "MODEL_TRAIN" if value < train_fraction else "CONFORMAL_FIT"


def _decode(path: Path, expected: object) -> NDArray[np.uint8]:
    if not path.is_file() or sha256_file(path) != _sha256(expected, "PNG SHA-256"):
        raise S7LiteV6Error("bound PNG changed")
    with Image.open(path) as source:
        source.load()
        image = np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
    if image.shape != (32, 32, 3):
        raise S7LiteV6Error("bound PNG shape changed")
    return image


def _tensor(images: Sequence[NDArray[np.uint8]]) -> Tensor:
    array = np.stack(images).astype(np.float32) / 255.0
    return torch.from_numpy(array.transpose(0, 3, 1, 2))


def _train_model(
    model: SmallPairedEncoder,
    images: Sequence[NDArray[np.uint8]],
    classes: Sequence[int],
    labels: Sequence[float],
    config: V6Config,
) -> list[float]:
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    inputs = _tensor(images)
    class_tensor = torch.tensor(classes, dtype=torch.long)
    label_tensor = torch.tensor(labels, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(config.seed)
    losses: list[float] = []
    model.train()
    for _epoch in range(config.epochs):
        permutation = torch.randperm(len(images), generator=generator)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, len(images), config.batch_size):
            indices = permutation[start : start + config.batch_size]
            _embedding, logits = model(inputs[indices])
            selected = logits.gather(1, class_tensor[indices, None]).squeeze(1)
            loss = functional.binary_cross_entropy_with_logits(selected, label_tensor[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach())
            batches += 1
        losses.append(epoch_loss / batches)
    return losses


def run_v6(config: V6Config, *, output_dir: Path) -> dict[str, object]:
    output = output_dir.resolve()
    if output.exists():
        raise S7LiteV6Error("S7 v6 output exists; use a new versioned path")
    output.mkdir(parents=True, exist_ok=False)
    if torch.__version__ != "2.5.1+cpu" or torch.cuda.is_available():
        raise S7LiteV6Error("v6 requires the frozen CPU-only torch runtime")
    torch.set_num_threads(4)
    tracemalloc.start()
    v1_config = S7Config.load(config.bindings["v1_config"])
    split_decoded: object = json.loads(
        config.bindings["v4_source_split"].read_text(encoding="utf-8")
    )
    original_roles_raw = _mapping(_mapping(split_decoded, "v4 split").get("roles"), "v4 roles")
    original_roles = {key: _string(value, "role") for key, value in original_roles_raw.items()}
    roles = {
        key: nested_role(config.seed, key, role, config.train_fraction)
        for key, role in original_roles.items()
    }
    v5_decoded: object = json.loads(config.bindings["v5_results"].read_text(encoding="utf-8"))
    v5_rows = [
        _mapping(raw, "v5 row")
        for raw in _sequence(_mapping(v5_decoded, "v5 results").get("records"), "records")
    ]
    v1_root = config.bindings["v1_audit"].parent
    train_images: list[NDArray[np.uint8]] = []
    train_classes: list[int] = []
    train_labels: list[float] = []
    class_index = {lesion_class: index for index, lesion_class in enumerate(LESION_CLASSES)}
    train_sources: set[str] = set()
    for row in v5_rows:
        source_key = _string(row.get("source_key"), "source key")
        truth = row.get("expected_truth")
        if roles[source_key] != "MODEL_TRAIN" or truth not in {"PRESENT", "ABSENT"}:
            continue
        relative = _safe_relative_string(row.get("observation_png_path"), "observation path")
        train_images.append(
            _decode(
                v1_root.joinpath(*PurePosixPath(relative).parts),
                row.get("observation_png_sha256"),
            )
        )
        train_classes.append(class_index[_string(row.get("lesion_class"), "lesion class")])
        train_labels.append(1.0 if truth == "PRESENT" else 0.0)
        train_sources.add(source_key)
    if len(train_images) < 40 or set(train_classes) != set(range(4)):
        raise S7LiteV6Error("v6 model-train split is insufficient")
    torch.manual_seed(config.seed)
    model = SmallPairedEncoder(config.embedding_dim)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    losses = _train_model(model, train_images, train_classes, train_labels, config)
    if not losses or not all(math.isfinite(value) for value in losses):
        raise S7LiteV6Error("v6 training loss is invalid")
    model.eval()
    result_rows: list[dict[str, object]] = []
    png_decodes = len(train_images)
    with torch.no_grad():
        for raw in v5_rows:
            row = dict(raw)
            source_key = _string(raw.get("source_key"), "source key")
            row["v6_role"] = roles[source_key]
            row["v5_record_sha256"] = sha256_bytes(_canonical(raw))
            if raw.get("expected_truth") == "OOD":
                for hierarchy in CALIBRATION_HIERARCHIES:
                    row[f"state_{hierarchy.lower()}"] = EntailmentState.OUT_OF_MODEL.value
                row["state_region_primary"] = EntailmentState.OUT_OF_MODEL.value
                row["state_source_max"] = EntailmentState.OUT_OF_MODEL.value
                result_rows.append(row)
                continue
            observation_relative = _safe_relative_string(
                raw.get("observation_png_path"), "observation path"
            )
            observation = _decode(
                v1_root.joinpath(*PurePosixPath(observation_relative).parts),
                raw.get("observation_png_sha256"),
            )
            png_decodes += 1
            observation_embedding, logits = model(_tensor([observation]))
            lesion_index = class_index[_string(raw.get("lesion_class"), "lesion class")]
            lesion_logit = logits[0, lesion_index]
            scores: dict[str, float] = {}
            for hypothesis in HYPOTHESES:
                search = _mapping(raw.get(f"{hypothesis.lower()}_search"), "search")
                relative = _safe_relative_string(search.get("candidate_png_path"), "candidate path")
                candidate = _decode(
                    v1_root.joinpath(*PurePosixPath(relative).parts),
                    search.get("candidate_png_sha256"),
                )
                png_decodes += 1
                candidate_embedding, _candidate_logits = model(_tensor([candidate]))
                distance = functional.mse_loss(observation_embedding, candidate_embedding)
                penalty = (
                    functional.softplus(-lesion_logit)
                    if hypothesis == "PRESENT"
                    else functional.softplus(lesion_logit)
                )
                score = float(distance + config.head_penalty_weight * penalty)
                if not math.isfinite(score):
                    raise S7LiteV6Error("v6 witness score is non-finite")
                scores[hypothesis] = score
            row["positive_score"] = scores["PRESENT"]
            row["negative_score"] = scores["ABSENT"]
            row["auxiliary_lesion_logit"] = float(lesion_logit)
            result_rows.append(row)
    helper_roles = {
        key: "CALIBRATION_FIT" if role == "CONFORMAL_FIT" else "DEVELOPMENT_VALIDATION"
        for key, role in roles.items()
    }
    calibration_rows = _build_calibration_rows(result_rows, helper_roles)
    for row in result_rows:
        if row.get("expected_truth") == "OOD":
            continue
        lesion_class = _string(row.get("lesion_class"), "lesion class")
        degradation = _string(row.get("observation_condition"), "condition")
        quality = _string(row.get("quality_bin"), "quality")
        present_score = cast(float, row["positive_score"])
        absent_score = cast(float, row["negative_score"])
        resolutions: dict[str, dict[str, ThresholdResolution]] = {
            hypothesis: {} for hypothesis in HYPOTHESES
        }
        for hypothesis in HYPOTHESES:
            for hierarchy in CALIBRATION_HIERARCHIES:
                resolutions[hypothesis][hierarchy] = _threshold(
                    calibration_rows,
                    v1_config,
                    hierarchy,
                    lesion_class=lesion_class,
                    hypothesis=hypothesis,
                    degradation=degradation,
                    quality=quality,
                    source_max=False,
                )
        for hierarchy in CALIBRATION_HIERARCHIES:
            present = resolutions["PRESENT"][hierarchy]
            absent = resolutions["ABSENT"][hierarchy]
            row[f"state_{hierarchy.lower()}"] = _state(
                present_score, absent_score, present.threshold, absent.threshold
            )
        primary_present = resolutions["PRESENT"]["LESION_DEGRADATION_QUALITY"]
        primary_absent = resolutions["ABSENT"]["LESION_DEGRADATION_QUALITY"]
        group_present = _threshold(
            calibration_rows,
            v1_config,
            "LESION_DEGRADATION_QUALITY",
            lesion_class=lesion_class,
            hypothesis="PRESENT",
            degradation=degradation,
            quality=quality,
            source_max=True,
        )
        group_absent = _threshold(
            calibration_rows,
            v1_config,
            "LESION_DEGRADATION_QUALITY",
            lesion_class=lesion_class,
            hypothesis="ABSENT",
            degradation=degradation,
            quality=quality,
            source_max=True,
        )
        row.update(
            {
                "positive_threshold": primary_present.threshold,
                "negative_threshold": primary_absent.threshold,
                "positive_threshold_hierarchy": primary_present.used_hierarchy,
                "negative_threshold_hierarchy": primary_absent.used_hierarchy,
                "positive_p_value": _p_value(present_score, primary_present.values),
                "negative_p_value": _p_value(absent_score, primary_absent.values),
                "state_region_primary": row["state_lesion_degradation_quality"],
                "state_source_max": _state(
                    present_score,
                    absent_score,
                    group_present.threshold,
                    group_absent.threshold,
                ),
            }
        )
    _apply_multi_region(result_rows, v1_config)
    validation_rows = [
        row
        for row in result_rows
        if row["v6_role"] == "DEVELOPMENT_VALIDATION" and row["expected_truth"] != "OOD"
    ]
    verify_decoded: object = json.loads(
        v1_config.bindings["s6_v2_results"].read_text(encoding="utf-8")
    )
    verify_states = {
        _string(_mapping(raw, "S6 row").get("case_id"), "case ID"): _string(
            _mapping(raw, "S6 row").get("state"), "state"
        )
        for raw in _sequence(_mapping(verify_decoded, "S6 results").get("records"), "S6 rows")
    }
    shared = [
        row
        for row in validation_rows
        if row.get("case_id") in verify_states
        and row.get("stratum") in {"PRESENT_ID", "ABSENT_ID", "S5_TIER_A_AMBIGUOUS_ID"}
    ]
    agreements = sum(
        row["state_region_primary"] == verify_states[cast(str, row["case_id"])] for row in shared
    )
    agreement_rate = agreements / len(shared) if shared else 0.0
    moderate_coverage: dict[str, float] = {}
    for lesion_class in LESION_CLASSES:
        moderate = [
            row
            for row in validation_rows
            if row.get("lesion_class") == lesion_class
            and row.get("stratum") in {"PRESENT_GLARE_MODERATE_ID", "ABSENT_GLARE_MODERATE_ID"}
        ]
        moderate_coverage[lesion_class] = (
            sum(row["state_region_primary"] in DETERMINATE_STATES for row in moderate)
            / len(moderate)
            if moderate
            else 0.0
        )
    state_keys = {
        "UNADJUSTED_REGION": "state_region_primary",
        "BONFERRONI": "state_bonferroni",
        "HOLM": "state_holm",
        "SOURCE_MAX": "state_source_max",
    }
    policy_metrics = {
        policy: _policy_metrics(validation_rows, state_key)
        for policy, state_key in state_keys.items()
    }
    fcr_raw = policy_metrics["UNADJUSTED_REGION"]["development_fcr"]
    validation_fcr = float(fcr_raw) if isinstance(fcr_raw, int | float) else math.inf
    ood = [row for row in result_rows if row.get("expected_truth") == "OOD"]
    ood_pass = sum(row["state_region_primary"] == EntailmentState.OUT_OF_MODEL.value for row in ood)
    role_sets = {
        role: {key for key, value in roles.items() if value == role}
        for role in ("MODEL_TRAIN", "CONFORMAL_FIT", "DEVELOPMENT_VALIDATION")
    }
    role_overlap = sum(
        len(role_sets[first].intersection(role_sets[second]))
        for first, second in (
            ("MODEL_TRAIN", "CONFORMAL_FIT"),
            ("MODEL_TRAIN", "DEVELOPMENT_VALIDATION"),
            ("CONFORMAL_FIT", "DEVELOPMENT_VALIDATION"),
        )
    )
    manifests = {
        "v1": _audit_manifest(v1_root, config.bindings["v1_artifact_manifest"]),
        "v5": _audit_manifest(
            config.bindings["v5_audit"].parent, config.bindings["v5_artifact_manifest"]
        ),
    }
    v1_hashes = _v1_probe_hash_audit(v5_rows, v1_root)
    upstream_pass = (
        all(
            audit["mismatches"] == 0 and audit["unlisted"] == 0 and audit["missing"] == 0
            for audit in manifests.values()
        )
        and v1_hashes["mismatches"] == 0
    )
    gate_pass = (
        upstream_pass
        and role_overlap == 0
        and ood_pass == len(ood) == 160
        and validation_fcr <= config.maximum_fcr
        and all(value >= config.minimum_moderate_coverage for value in moderate_coverage.values())
        and agreement_rate >= config.minimum_verify_agreement
    )
    model_stream = io.BytesIO()
    torch.save(model.state_dict(), model_stream)
    model_bytes = model_stream.getvalue()
    results_payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "records": result_rows,
    }
    artifacts: dict[str, bytes] = {
        "small-paired-encoder-v6.pt": model_bytes,
        "lite-results-v6.json": _canonical(results_payload),
        "source-roles-v6.json": _canonical({"seed": config.seed, "roles": roles}),
    }
    contact_sheet_hashes: dict[str, str] = {}
    for lesion_class in LESION_CLASSES:
        source_name = f"qc-{lesion_class.lower()}-lite-contact-sheet.png"
        target_name = f"qc-{lesion_class.lower()}-lite-v6-contact-sheet.png"
        source = v1_root / source_name
        if not source.is_file():
            raise S7LiteV6Error(f"missing bound v1 contact sheet: {source_name}")
        content = source.read_bytes()
        artifacts[target_name] = content
        contact_sheet_hashes[lesion_class] = sha256_bytes(content)
    _current, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    audit: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": READY_STATUS if gate_pass else BLOCKED_STATUS,
        "config_sha256": config.config_sha256,
        "development_disclosure": {
            "v1_to_v5_observed": True,
            "independent_replication": False,
            "final_test_opened": False,
        },
        "method": {
            "runtime": "PYTHON_3_11_TORCH_2_5_1_CPU",
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "torch_version": torch.__version__,
            "parameter_count": parameter_count,
            "embedding_dim": config.embedding_dim,
            "head_penalty_weight": config.head_penalty_weight,
            "pretrained_weights": False,
            "witness_policy": "REUSE_V1_POSITIVE_NEGATIVE_BEST_CANDIDATES",
        },
        "source_split": {
            "role_counts": {role: len(values) for role, values in role_sets.items()},
            "role_overlap": role_overlap,
            "train_sources_used": len(train_sources),
            "train_observations": len(train_images),
            "conformal_rows": len(calibration_rows),
            "validation_rows": len(validation_rows),
        },
        "training": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "minimum_loss": min(losses),
            "finite": True,
        },
        "preservation": {
            "manifests": manifests,
            "v1_probe_hashes": v1_hashes,
            "probe_searches_rerun": 0,
        },
        "multi_region": {"validation_policy_metrics": policy_metrics},
        "development_gate_metrics": {
            "validation_fcr": None if not math.isfinite(validation_fcr) else validation_fcr,
            "validation_cluster_bootstrap": _cluster_bootstrap_fcr(
                validation_rows, "state_region_primary", config.seed, config.bootstrap_replicates
            ),
            "validation_moderate_glare_determinate_coverage_per_class": moderate_coverage,
            "validation_lite_verify_shared_in_domain_records": len(shared),
            "validation_lite_verify_agreement_count": agreements,
            "validation_lite_verify_agreement_rate": agreement_rate,
            "all_ood_records": len(ood),
            "all_ood_out_of_model": ood_pass,
        },
        "resource": {
            "device": "CPU",
            "gpu_used": False,
            "peak_python_tracemalloc_bytes": peak_bytes,
            "silent_resolution_change": False,
        },
        "io_audit": {
            "bound_png_decodes": png_decodes,
            "probe_searches_rerun": 0,
            "locked_test_root_reads": 0,
            "online_evaluation_reads": 0,
            "source_split_leaks": role_overlap,
        },
        "gate": {
            "maximum_validation_fcr": config.maximum_fcr,
            "minimum_validation_moderate_coverage_per_class": config.minimum_moderate_coverage,
            "minimum_validation_lite_verify_agreement": config.minimum_verify_agreement,
            "automated": "PASS" if gate_pass else "FAIL",
            "ai_visual_qc": "PENDING",
            "s8_authorized": False,
        },
        "hashes": {
            "code_sha256": sha256_file(Path(__file__)),
            "model_sha256": sha256_bytes(model_bytes),
            "results_sha256": sha256_bytes(artifacts["lite-results-v6.json"]),
            "roles_sha256": sha256_bytes(artifacts["source-roles-v6.json"]),
            "contact_sheet_sha256": contact_sheet_hashes,
        },
        "evidence_boundary": [
            "V6 is run-informed development work and not an independent replication.",
            (
                "The learned head is auxiliary; both witness-relative hypothesis scores remain "
                "required."
            ),
            "Model, conformal and validation sources are disjoint.",
            "No final test, clinical inference, enhancer training, or paper experiment ran.",
        ],
    }
    artifacts["audit.json"] = _canonical(audit)
    _write_artifacts(output, artifacts)
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        audit = run_v6(V6Config.load(args.config), output_dir=args.output_dir)
    except (OSError, ValueError, subprocess.SubprocessError, RuntimeError) as error:
        print(f"S7 Lite v6 failed closed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["status"] == READY_STATUS else 1


if __name__ == "__main__":
    raise SystemExit(main())
