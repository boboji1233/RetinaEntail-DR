"""Develop a last-block metric recovery without opening the final locked test."""

from __future__ import annotations

import argparse
import io
import json
import math
import platform
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

import numpy as np
import science_s7_lite_v7 as core
import science_s7_lite_v8 as v8
import science_s8_controlled_truth_v1 as s8
import torch
import yaml
from numpy.typing import NDArray
from torch import Tensor
from torch.nn import functional

from retinaentail.data.manifest import sha256_file
from retinaentail.witnesses.replay import (
    canonical_json_bytes,
    sha256_bytes,
    write_bytes_fail_safe,
)

EXPECTED_S8_RESULTS = "82ca0ff513cbc0b969895009468dcb0be283ed3d9df1d297103b98f7c85a7a3b"
EXPECTED_S8_TWINS = "88f94771d963f3b143f329a361502343f34d453452e9de66486fe6ed34eb9f5c"
CLASS_INDEX = {"HE": 0, "EX": 1, "MA": 2, "SE": 3}
DETERMINATE = {"ENTAILED_PRESENT", "ENTAILED_ABSENT"}
DEFAULT_SEEDS = (2026082631, 2026082632, 2026082633, 2026082634, 2026082635)
TRAIN_CONDITION_IDS = {
    "IDENTITY",
    "GLARE_MODERATE",
    "DEFOCUS_MODERATE",
    "JPEG_MODERATE",
    "HAZE_MODERATE",
    "ILLUMINATION_MODERATE",
    "HAZE_GLARE_MODERATE",
}


class S10MetricRecoveryError(ValueError):
    """Raised when a frozen input, memory boundary, or metric contract changes."""


@dataclass(frozen=True, slots=True)
class RecoverySettings:
    epochs: int
    pair_batch_size: int
    layer4_learning_rate: float
    adapter_learning_rate: float
    weight_decay: float
    prototype_weight: float
    head_weight: float
    invariance_weight: float
    separation_weight: float
    separation_margin: float
    memory_fraction: float


def _false_claim(state: str, truth: str) -> bool:
    return (state == "ENTAILED_PRESENT" and truth == "ABSENT") or (
        state == "ENTAILED_ABSENT" and truth == "PRESENT"
    )


def _metrics(rows: Sequence[Mapping[str, object]], state_key: str) -> dict[str, object]:
    determinate = [row for row in rows if row.get(state_key) in DETERMINATE]
    false = sum(
        _false_claim(str(row[state_key]), str(row["expected_truth"])) for row in determinate
    )
    return {
        "total": len(rows),
        "determinate": len(determinate),
        "coverage": len(determinate) / len(rows) if rows else 0.0,
        "false": false,
        "fcr": false / len(determinate) if determinate else None,
        "state_counts": dict(sorted(Counter(str(row[state_key]) for row in rows).items())),
    }


def _load_s8_test_items(path: Path) -> list[core.TwinItem]:
    if sha256_file(path) != EXPECTED_S8_TWINS:
        raise S10MetricRecoveryError("S8 twin manifest hash changed")
    payload = core._load_json(path, "S8 twin manifest")
    items = []
    for raw in core._sequence(payload.get("accepted"), "S8 accepted twins"):
        record = core._mapping(raw, "S8 twin record")
        source_id = core._string(record.get("source_image_id"), "source ID")
        items.append(
            core.TwinItem(
                package_id="S8_DDR_V1",
                root=path.parent,
                raw=record,
                source_key=f"S4_DDR_V5:{source_id}",
                source_image_id=source_id,
                lesion_class=core._string(record.get("lesion_class"), "lesion class"),
                twin_id=core._string(record.get("twin_id"), "twin ID"),
            )
        )
    if len(items) != 459:
        raise S10MetricRecoveryError("S8 test membership changed")
    return items


def _pretrained_weight_path(config: s8.S8Config) -> Path:
    root = core._mapping(
        yaml.safe_load(config.bindings["v8_config"].read_text(encoding="utf-8")),
        "v8 config",
    )
    method = core._mapping(root.get("method"), "v8 method")
    binding = core._mapping(method.get("weight_binding"), "v8 weight binding")
    relative = core._string(binding.get("home_relative_path"), "weight path")
    path = Path.home().joinpath(*PurePosixPath(relative).parts)
    if not path.is_file() or sha256_file(path) != core._string(
        binding.get("sha256"), "weight hash"
    ):
        raise S10MetricRecoveryError("pretrained ResNet-18 weight binding changed")
    return path


def _frozen_trunk(model: v8.FrozenResNetAdapter, inputs: Tensor) -> Tensor:
    backbone = model.backbone
    values = model._imagenet_normalize(inputs)  # noqa: SLF001
    values = backbone.conv1(values)
    values = backbone.bn1(values)
    values = backbone.relu(values)
    values = backbone.maxpool(values)
    values = backbone.layer1(values)
    values = backbone.layer2(values)
    return cast(Tensor, backbone.layer3(values))


def _tail_features(model: v8.FrozenResNetAdapter, trunk: Tensor) -> Tensor:
    values = model.backbone.layer4(trunk)
    values = model.backbone.avgpool(values)
    return torch.flatten(values, 1)


def _training_arrays(
    items: Sequence[core.TwinItem],
    config: s8.S8Config,
    cache: Mapping[tuple[str, str, str], NDArray[np.uint8]],
) -> tuple[NDArray[np.uint8], tuple[s8.ConditionSpec, ...], dict[str, int]]:
    conditions = tuple(
        condition
        for condition in s8.frozen_conditions()
        if condition.condition_id in TRAIN_CONDITION_IDS
    )
    if len(conditions) != 7:
        raise S10MetricRecoveryError("training condition subset changed")
    replay: Counter[str] = Counter()
    arrays: list[NDArray[np.uint8]] = []
    for item in items:
        for hypothesis in ("PRESENT", "ABSENT"):
            raw = cache[(item.package_id, item.twin_id, hypothesis)]
            arrays.extend(
                s8._condition_view(raw, item, hypothesis, condition, config, replay).array
                for condition in conditions
            )
    expected = len(items) * 2 * len(conditions)
    if len(arrays) != expected or replay["mismatches"]:
        raise S10MetricRecoveryError("training view construction changed")
    return np.stack(arrays), conditions, dict(replay)


def _precompute_trunk(
    arrays: NDArray[np.uint8],
    weight_path: Path,
    device: torch.device,
) -> Tensor:
    model = v8.FrozenResNetAdapter(32, weight_path).to(device)
    model.eval()
    chunks: list[Tensor] = []
    with torch.no_grad():
        for start in range(0, len(arrays), 64):
            inputs = core._tensor(arrays[start : start + 64]).to(device)
            chunks.append(_frozen_trunk(model, inputs).cpu())
    return torch.cat(chunks)


def _class_balanced_batch_hard(
    embeddings: Tensor,
    classes: Tensor,
    margin: float,
) -> Tensor:
    """Separate opposite hypotheses from different sources within each class."""

    present = embeddings[:, 0].mean(dim=1)
    absent = embeddings[:, 1].mean(dim=1)
    losses = []
    for lesion_index in range(4):
        selected = torch.nonzero(classes == lesion_index, as_tuple=False).flatten()
        if len(selected) < 2:
            continue
        positive = present[selected]
        negative = absent[selected]
        cross = 1.0 - positive @ negative.T
        identity = torch.eye(len(selected), dtype=torch.bool, device=cross.device)
        hardest = cross.masked_fill(identity, math.inf).min(dim=1).values
        losses.append(functional.relu(margin - hardest).mean())
    if not losses:
        return embeddings.sum() * 0.0
    return torch.stack(losses).mean()


def _train_seed(
    *,
    seed: int,
    trunk_features: Tensor,
    train_items: Sequence[core.TwinItem],
    condition_count: int,
    weight_path: Path,
    device: torch.device,
    settings: RecoverySettings,
) -> tuple[v8.FrozenResNetAdapter, dict[str, object]]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = v8.FrozenResNetAdapter(32, weight_path).to(device)
    for parameter in model.backbone.layer4.parameters():
        parameter.requires_grad_(True)
    model.backbone.eval()
    parameters = [
        {
            "params": list(model.backbone.layer4.parameters()),
            "lr": settings.layer4_learning_rate,
        },
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if not name.startswith("backbone.")
            ],
            "lr": settings.adapter_learning_rate,
        },
    ]
    optimizer = torch.optim.AdamW(parameters, weight_decay=settings.weight_decay, foreach=False)
    classes = torch.tensor(
        [CLASS_INDEX[item.lesion_class] for item in train_items], dtype=torch.long
    )
    class_counts = Counter(item.lesion_class for item in train_items)
    weights = torch.tensor(
        [1.0 / class_counts[item.lesion_class] for item in train_items], dtype=torch.float32
    )
    weights = weights / weights.mean()
    generator = torch.Generator().manual_seed(seed)
    losses = []
    started = time.perf_counter()
    for _epoch in range(settings.epochs):
        permutation = torch.randperm(len(train_items), generator=generator)
        epoch_loss = 0.0
        batch_count = 0
        for start in range(0, len(train_items), settings.pair_batch_size):
            indices = permutation[start : start + settings.pair_batch_size]
            flat_indices = (
                indices[:, None] * (2 * condition_count)
                + torch.arange(2 * condition_count)[None, :]
            ).reshape(-1)
            trunk = trunk_features[flat_indices].to(device)
            batch_classes = classes[indices].to(device)
            pair_weights = weights[indices].to(device)
            feature_vectors = _tail_features(model, trunk)
            repeated_classes = batch_classes.repeat_interleave(2 * condition_count)
            projected, logits = model.project(feature_vectors, repeated_classes)
            embeddings = projected.reshape(len(indices), 2, condition_count, 32)
            logits = logits.reshape(len(indices), 2, condition_count)
            normalized_prototypes = functional.normalize(model.prototypes, dim=2)
            truth_indices = torch.tensor([1, 0], device=device)[None, :, None].expand(
                len(indices), -1, condition_count
            )
            selected_prototypes = normalized_prototypes[batch_classes[:, None, None], truth_indices]
            pull = 1.0 - functional.cosine_similarity(embeddings, selected_prototypes, dim=3)
            pull_loss = core._weighted_mean(pull.mean(dim=(1, 2)), pair_weights)
            labels = torch.tensor([1.0, 0.0], device=device)[None, :, None].expand_as(logits)
            head = functional.binary_cross_entropy_with_logits(
                logits, labels, reduction="none"
            ).mean(dim=(1, 2))
            head_loss = core._weighted_mean(head, pair_weights)
            identity = embeddings[:, :, :1]
            invariance = 1.0 - functional.cosine_similarity(
                embeddings[:, :, 1:], identity.expand_as(embeddings[:, :, 1:]), dim=3
            )
            invariance_loss = core._weighted_mean(invariance.mean(dim=(1, 2)), pair_weights)
            separation = 1.0 - functional.cosine_similarity(
                embeddings[:, 0], embeddings[:, 1], dim=2
            )
            within_twin = functional.relu(settings.separation_margin - separation).mean(dim=1)
            separation_loss = core._weighted_mean(within_twin, pair_weights)
            separation_loss = separation_loss + _class_balanced_batch_hard(
                embeddings, batch_classes, settings.separation_margin
            )
            prototype_distance = 1.0 - functional.cosine_similarity(
                normalized_prototypes[:, 0], normalized_prototypes[:, 1], dim=1
            )
            separation_loss = (
                separation_loss
                + functional.relu(settings.separation_margin - prototype_distance).mean()
            )
            loss = (
                settings.prototype_weight * pull_loss
                + settings.head_weight * head_loss
                + settings.invariance_weight * invariance_loss
                + settings.separation_weight * separation_loss
            )
            if not bool(torch.isfinite(loss)):
                raise S10MetricRecoveryError("metric recovery optimization became non-finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            batch_count += 1
        losses.append(epoch_loss / batch_count)
    elapsed = time.perf_counter() - started
    model.eval()
    return model, {
        "seed": seed,
        "epochs": settings.epochs,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "minimum_loss": min(losses),
        "losses": losses,
        "elapsed_seconds": elapsed,
    }


def _encode_device(
    model: v8.FrozenResNetAdapter,
    images: Sequence[NDArray[np.uint8]],
    lesions: Sequence[str],
    device: torch.device,
) -> NDArray[np.float32]:
    chunks: list[NDArray[np.float32]] = []
    with torch.no_grad():
        for start in range(0, len(images), 96):
            classes = torch.tensor(
                [CLASS_INDEX[value] for value in lesions[start : start + 96]],
                dtype=torch.long,
                device=device,
            )
            inputs = core._tensor(np.stack(images[start : start + 96])).to(device)
            embedding, _ = model(inputs, classes)
            chunks.append(embedding.cpu().numpy().astype(np.float32))
    return np.concatenate(chunks)


def _score_condition(
    *,
    model: v8.FrozenResNetAdapter,
    condition: s8.ConditionSpec,
    train: Sequence[core.TwinItem],
    queries: Sequence[core.TwinItem],
    role: str,
    cache: Mapping[tuple[str, str, str], NDArray[np.uint8]],
    config: s8.S8Config,
    replay: Counter[str],
    device: torch.device,
) -> list[dict[str, object]]:
    descriptors = [
        (item, hypothesis) for item in (*train, *queries) for hypothesis in ("PRESENT", "ABSENT")
    ]
    views = [
        s8._condition_view(
            cache[(item.package_id, item.twin_id, hypothesis)],
            item,
            hypothesis,
            condition,
            config,
            replay,
        )
        for item, hypothesis in descriptors
    ]
    embeddings = _encode_device(
        model,
        [view.array for view in views],
        [item.lesion_class for item, _hypothesis in descriptors],
        device,
    )
    train_count = len(train) * 2
    bank: dict[tuple[str, str], list[tuple[core.TwinItem, NDArray[np.float32]]]] = defaultdict(list)
    for (item, hypothesis), embedding in zip(
        descriptors[:train_count], embeddings[:train_count], strict=True
    ):
        bank[(item.lesion_class, hypothesis)].append((item, embedding))
    rows = []
    for (item, truth), query, view in zip(
        descriptors[train_count:], embeddings[train_count:], views[train_count:], strict=True
    ):
        scores: dict[str, float] = {}
        witnesses: dict[str, dict[str, str]] = {}
        for hypothesis in ("PRESENT", "ABSENT"):
            eligible = [
                entry
                for entry in bank[(item.lesion_class, hypothesis)]
                if entry[0].source_key != item.source_key
            ]
            distances = np.asarray(
                [1.0 - float(np.dot(query, entry[1])) for entry in eligible],
                dtype=np.float64,
            )
            selected = min(
                range(len(eligible)),
                key=lambda index: (float(distances[index]), eligible[index][0].twin_id),
            )
            witness = eligible[selected][0]
            scores[hypothesis] = float(distances[selected])
            witnesses[hypothesis] = {
                "source_key": witness.source_key,
                "twin_id": witness.twin_id,
            }
        rows.append(
            {
                "row_id": f"{role}:{condition.condition_id}:{item.twin_id}:{truth}",
                "role": role,
                "source_key": item.source_key,
                "source_image_id": item.source_image_id,
                "twin_id": item.twin_id,
                "lesion_class": item.lesion_class,
                "expected_truth": truth,
                "condition_id": condition.condition_id,
                "degradation_family": condition.family,
                "severity_level": condition.severity_level,
                "severity": condition.severity,
                "positive_score": scores["PRESENT"],
                "negative_score": scores["ABSENT"],
                "positive_witness": witnesses["PRESENT"],
                "negative_witness": witnesses["ABSENT"],
                "observation_sha256": sha256_bytes(view.array.tobytes()),
                "trace_sha256": view.trace_sha256,
                "degradation_seeds": list(view.seeds),
            }
        )
    return rows


def _evaluate(
    *,
    model: v8.FrozenResNetAdapter,
    train: Sequence[core.TwinItem],
    calibration: Sequence[core.TwinItem],
    test: Sequence[core.TwinItem],
    cache: Mapping[tuple[str, str, str], NDArray[np.uint8]],
    config: s8.S8Config,
    device: torch.device,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    replay: Counter[str] = Counter()
    calibration_rows = []
    test_raw = []
    for condition in s8.frozen_conditions():
        calibration_rows.extend(
            _score_condition(
                model=model,
                condition=condition,
                train=train,
                queries=calibration,
                role="CONFORMAL_FIT",
                cache=cache,
                config=config,
                replay=replay,
                device=device,
            )
        )
        test_raw.extend(
            _score_condition(
                model=model,
                condition=condition,
                train=train,
                queries=test,
                role="S8_POSTHOC_RECOVERY_DEVELOPMENT",
                cache=cache,
                config=config,
                replay=replay,
                device=device,
            )
        )
    thresholds = s8.fit_source_max_thresholds(calibration_rows, alpha=config.alpha)
    selective = s8.fit_selective_margin_thresholds(calibration_rows, alpha=config.alpha)
    test_rows = [s8.apply_policies(row, thresholds, selective) for row in test_raw]
    metrics = {
        "four_valued": _metrics(test_rows, "state_four_valued"),
        "forced_nearest": _metrics(test_rows, "state_forced_nearest"),
        "selective_margin": _metrics(test_rows, "state_selective_margin"),
        "by_severity": {
            severity: _metrics(
                [row for row in test_rows if row["severity_level"] == severity],
                "state_four_valued",
            )
            for severity in ("IDENTITY", "LIGHT", "MODERATE", "HEAVY")
        },
        "by_lesion": {
            lesion: _metrics(
                [row for row in test_rows if row["lesion_class"] == lesion],
                "state_four_valued",
            )
            for lesion in CLASS_INDEX
        },
        "replay": dict(replay),
    }
    return calibration_rows, test_raw, metrics


def _average_rows(seed_rows: Sequence[Sequence[Mapping[str, object]]]) -> list[dict[str, object]]:
    if not seed_rows:
        raise S10MetricRecoveryError("ensemble has no seed rows")
    by_seed = [{str(row["row_id"]): row for row in rows} for rows in seed_rows]
    keys = set(by_seed[0])
    if any(set(rows) != keys for rows in by_seed[1:]):
        raise S10MetricRecoveryError("seed evaluation memberships differ")
    averaged = []
    for key in sorted(keys):
        templates = [rows[key] for rows in by_seed]
        first = templates[0]
        if any(
            row[field] != first[field]
            for row in templates[1:]
            for field in ("source_key", "twin_id", "lesion_class", "expected_truth", "condition_id")
        ):
            raise S10MetricRecoveryError("seed row identity differs")
        averaged.append(
            {
                **first,
                "positive_score": sum(float(row["positive_score"]) for row in templates)
                / len(templates),
                "negative_score": sum(float(row["negative_score"]) for row in templates)
                / len(templates),
                "seed_witnesses": [
                    {
                        "positive": row["positive_witness"],
                        "negative": row["negative_witness"],
                    }
                    for row in templates
                ],
            }
        )
    return averaged


def run(
    *,
    config_path: Path,
    s8_root: Path,
    output_dir: Path,
    seeds: Sequence[int],
    settings: RecoverySettings,
) -> dict[str, object]:
    if output_dir.exists():
        raise S10MetricRecoveryError("output directory exists; use a new versioned path")
    output_dir.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise S10MetricRecoveryError("CUDA is required for the bounded recovery run")
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(settings.memory_fraction, device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.cuda.reset_peak_memory_stats(device)
    config = s8.S8Config.load(config_path)
    results_path = s8_root / "controlled-truth-results-v1.json"
    twin_path = s8_root / "twin-manifest-v1.json"
    if sha256_file(results_path) != EXPECTED_S8_RESULTS:
        raise S10MetricRecoveryError("S8 result binding changed")
    train_items, calibration_items = s8._v8_training_items(config)
    test_items = _load_s8_test_items(twin_path)
    role_sets = [
        {item.source_key for item in items}
        for items in (train_items, calibration_items, test_items)
    ]
    if any(role_sets[i].intersection(role_sets[j]) for i in range(3) for j in range(i + 1, 3)):
        raise S10MetricRecoveryError("source role overlap")
    io_audit = core.IOAudit()
    all_items = [*train_items, *calibration_items, *test_items]
    cache = s8._raw_cache(all_items, io_audit)
    training_arrays, train_conditions, train_replay = _training_arrays(train_items, config, cache)
    weight_path = _pretrained_weight_path(config)
    trunk_features = _precompute_trunk(training_arrays, weight_path, device)
    if tuple(trunk_features.shape) != (len(train_items) * 14, 256, 8, 8):
        raise S10MetricRecoveryError("frozen trunk feature shape changed")
    seed_results = []
    calibration_by_seed = []
    test_by_seed = []
    model_hashes = []
    for seed in seeds:
        model, training = _train_seed(
            seed=seed,
            trunk_features=trunk_features,
            train_items=train_items,
            condition_count=len(train_conditions),
            weight_path=weight_path,
            device=device,
            settings=settings,
        )
        calibration_rows, test_rows, metrics = _evaluate(
            model=model,
            train=train_items,
            calibration=calibration_items,
            test=test_items,
            cache=cache,
            config=config,
            device=device,
        )
        stream = io.BytesIO()
        torch.save(
            {
                "architecture": "_".join(
                    (
                        "RESNET18_FROZEN_TRUNK_THROUGH_LAYER3",
                        "FINETUNED_LAYER4_DUAL_WITNESS_V1",
                    )
                ),
                "seed": seed,
                "pretrained_weight_sha256": sha256_file(weight_path),
                "layer4_state": {
                    name: value.detach().cpu()
                    for name, value in model.backbone.layer4.state_dict().items()
                },
                "adapter_state": model.adapter_state(),
            },
            stream,
        )
        model_bytes = stream.getvalue()
        model_name = f"metric-recovery-seed-{seed}.pt"
        write_bytes_fail_safe(output_dir / model_name, model_bytes)
        model_hashes.append({"seed": seed, "file": model_name, "sha256": sha256_bytes(model_bytes)})
        seed_results.append({"training": training, "metrics": metrics})
        calibration_by_seed.append(calibration_rows)
        test_by_seed.append(test_rows)
        model.cpu()
        torch.cuda.empty_cache()
    ensemble_calibration = _average_rows(calibration_by_seed)
    ensemble_raw = _average_rows(test_by_seed)
    thresholds = s8.fit_source_max_thresholds(ensemble_calibration, alpha=config.alpha)
    selective = s8.fit_selective_margin_thresholds(ensemble_calibration, alpha=config.alpha)
    ensemble_rows = [s8.apply_policies(row, thresholds, selective) for row in ensemble_raw]
    ensemble_metrics = {
        "four_valued": _metrics(ensemble_rows, "state_four_valued"),
        "forced_nearest": _metrics(ensemble_rows, "state_forced_nearest"),
        "selective_margin": _metrics(ensemble_rows, "state_selective_margin"),
        "by_severity": {
            severity: _metrics(
                [row for row in ensemble_rows if row["severity_level"] == severity],
                "state_four_valued",
            )
            for severity in ("IDENTITY", "LIGHT", "MODERATE", "HEAVY")
        },
        "by_lesion": {
            lesion: _metrics(
                [row for row in ensemble_rows if row["lesion_class"] == lesion],
                "state_four_valued",
            )
            for lesion in CLASS_INDEX
        },
    }
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    if peak_reserved > 3_000_000_000:
        raise S10MetricRecoveryError("3 GB GPU memory boundary exceeded")
    payload = {
        "schema_version": "1.0.0",
        "experiment_id": "S10_METRIC_RECOVERY_V1",
        "status": "COMPLETE_POST_S8_DEVELOPMENT_NOT_CONFIRMATORY",
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "memory_fraction_limit": settings.memory_fraction,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
        },
        "bindings": {
            "s8_results_sha256": EXPECTED_S8_RESULTS,
            "s8_twin_manifest_sha256": EXPECTED_S8_TWINS,
            "pretrained_weight_sha256": sha256_file(weight_path),
        },
        "source_counts": {
            "model_train": len(role_sets[0]),
            "conformal_fit": len(role_sets[1]),
            "s8_development": len(role_sets[2]),
            "overlap": 0,
        },
        "training": {
            "seeds": list(seeds),
            "settings": {name: getattr(settings, name) for name in settings.__dataclass_fields__},
            "condition_ids": [condition.condition_id for condition in train_conditions],
            "view_count": len(training_arrays),
            "degradation_replay": train_replay,
        },
        "io_audit": {
            "twin_png_decodes": io_audit.twin_png_decodes,
            "twin_png_hash_mismatches": io_audit.twin_png_hash_mismatches,
            "locked_test_root_reads": 0,
        },
        "model_artifacts": model_hashes,
        "seed_results": seed_results,
        "ensemble_metrics": ensemble_metrics,
        "evidence_boundary": [
            "S8 labels were already observed before this recovery design.",
            "All S8 recovery numbers are post-hoc development evidence.",
            "The final IDRiD locked-test root remained unread.",
        ],
    }
    write_bytes_fail_safe(
        output_dir / "metric-recovery-summary-v1.json", canonical_json_bytes(payload)
    )
    write_bytes_fail_safe(
        output_dir / "metric-recovery-ensemble-rows-v1.json",
        canonical_json_bytes({"schema_version": "1.0.0", "rows": ensemble_rows}),
    )
    write_bytes_fail_safe(
        output_dir / "metric-recovery-calibration-v1.json",
        canonical_json_bytes(
            {
                "schema_version": "1.0.0",
                "method": "_".join(
                    (
                        "MEAN_SCORE_ENSEMBLE_SOURCE_MAX_CLASS_CONDITION",
                        "HYPOTHESIS_SPLIT_CONFORMAL",
                    )
                ),
                "thresholds": thresholds,
                "selective_margin_thresholds": selective,
            }
        ),
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "ensemble": ensemble_metrics,
                "runtime": payload["runtime"],
            },
            indent=2,
        )
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--s8-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=24)
    args = parser.parse_args(argv)
    settings = RecoverySettings(
        epochs=args.epochs,
        pair_batch_size=4,
        layer4_learning_rate=0.00003,
        adapter_learning_rate=0.0005,
        weight_decay=0.0001,
        prototype_weight=1.0,
        head_weight=0.5,
        invariance_weight=0.25,
        separation_weight=0.75,
        separation_margin=0.60,
        memory_fraction=0.16,
    )
    run(
        config_path=args.config,
        s8_root=args.s8_root,
        output_dir=args.output_dir,
        seeds=args.seeds,
        settings=settings,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
