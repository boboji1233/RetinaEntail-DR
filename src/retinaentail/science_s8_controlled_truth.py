"""Pure contracts and statistics for Science S8 controlled-truth evaluation."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from retinaentail.entailment.states import EntailmentState
from retinaentail.science_s5_degradation_calibration import _string
from retinaentail.science_s6_verify import LESION_CLASSES
from retinaentail.science_s7_lite_v4 import _state


class S8ControlledTruthError(ValueError):
    """Raised when the frozen S8 statistical contract cannot be replayed."""


@dataclass(frozen=True, slots=True)
class ConditionSpec:
    """One frozen registered observation condition."""

    condition_id: str
    family: str
    severity_level: str
    severity: float
    operations: tuple[str, ...]


def frozen_conditions() -> tuple[ConditionSpec, ...]:
    """Return the prospectively frozen 19-condition S8 matrix."""

    conditions = [ConditionSpec("IDENTITY", "identity", "IDENTITY", 0.0, ())]
    families = (
        ("GLARE", "glare_reflection"),
        ("DEFOCUS", "defocus_blur"),
        ("JPEG", "jpeg_compression"),
        ("HAZE", "haze_veil"),
        ("ILLUMINATION", "gamma_exposure"),
    )
    levels = (("LIGHT", 0.2), ("MODERATE", 0.6), ("HEAVY", 1.0))
    for label, operation in families:
        for level, severity in levels:
            conditions.append(
                ConditionSpec(
                    f"{label}_{level}",
                    label,
                    level,
                    severity,
                    (operation,),
                )
            )
    for level, severity in levels:
        conditions.append(
            ConditionSpec(
                f"HAZE_GLARE_{level}",
                "HAZE_GLARE",
                level,
                severity,
                ("haze_veil", "glare_reflection"),
            )
        )
    result = tuple(conditions)
    if len(result) != 19 or len({item.condition_id for item in result}) != 19:
        raise S8ControlledTruthError("S8 condition matrix is not unique and complete")
    return result


def fit_source_max_thresholds(
    rows: Sequence[Mapping[str, object]], *, alpha: float
) -> dict[str, dict[str, dict[str, object]]]:
    """Fit source-max class/condition/hypothesis split-conformal thresholds."""

    if not 0.0 < alpha < 1.0:
        raise S8ControlledTruthError("alpha must be in (0, 1)")
    condition_ids = tuple(condition.condition_id for condition in frozen_conditions())
    thresholds: dict[str, dict[str, dict[str, object]]] = {}
    for lesion in LESION_CLASSES:
        thresholds[lesion] = {}
        for condition_id in condition_ids:
            thresholds[lesion][condition_id] = {}
            for hypothesis, score_key in (
                ("PRESENT", "positive_score"),
                ("ABSENT", "negative_score"),
            ):
                grouped: dict[str, list[float]] = defaultdict(list)
                for row in rows:
                    if (
                        row.get("lesion_class") == lesion
                        and row.get("condition_id") == condition_id
                        and row.get("expected_truth") == hypothesis
                    ):
                        value = row.get(score_key)
                        if isinstance(value, bool) or not isinstance(value, int | float):
                            raise S8ControlledTruthError("calibration score is missing or invalid")
                        score = float(value)
                        if not math.isfinite(score):
                            raise S8ControlledTruthError("calibration score is non-finite")
                        grouped[_string(row.get("source_key"), "source key")].append(score)
                values = sorted(max(source_values) for source_values in grouped.values())
                if len(values) < 15:
                    raise S8ControlledTruthError(
                        f"insufficient source groups for {lesion}/{condition_id}/{hypothesis}"
                    )
                rank = math.ceil((len(values) + 1) * (1.0 - alpha))
                thresholds[lesion][condition_id][hypothesis] = {
                    "source_groups": len(values),
                    "rank": rank,
                    "threshold": values[min(rank, len(values)) - 1],
                }
    return thresholds


def fit_selective_margin_thresholds(
    rows: Sequence[Mapping[str, object]], *, alpha: float
) -> dict[str, dict[str, float]]:
    """Fit a conservative error-margin threshold for a selective binary baseline."""

    if not 0.0 < alpha < 1.0:
        raise S8ControlledTruthError("alpha must be in (0, 1)")
    thresholds: dict[str, dict[str, float]] = {}
    for lesion in LESION_CLASSES:
        thresholds[lesion] = {}
        for condition in frozen_conditions():
            wrong_by_source: dict[str, list[float]] = defaultdict(list)
            all_sources: set[str] = set()
            for row in rows:
                if (
                    row.get("lesion_class") != lesion
                    or row.get("condition_id") != condition.condition_id
                ):
                    continue
                source = _string(row.get("source_key"), "source key")
                all_sources.add(source)
                positive = float(cast(float, row["positive_score"]))
                negative = float(cast(float, row["negative_score"]))
                predicted = "PRESENT" if positive < negative else "ABSENT"
                if predicted != row.get("expected_truth"):
                    wrong_by_source[source].append(abs(positive - negative))
            if len(all_sources) < 15:
                raise S8ControlledTruthError("selective baseline lacks calibration sources")
            wrong_source_max = sorted(max(values) for values in wrong_by_source.values() if values)
            thresholds[lesion][condition.condition_id] = (
                wrong_source_max[-1] if wrong_source_max else 0.0
            )
    return thresholds


def apply_policies(
    row: Mapping[str, object],
    thresholds: Mapping[str, Mapping[str, Mapping[str, object]]],
    selective_thresholds: Mapping[str, Mapping[str, float]],
) -> dict[str, object]:
    """Apply the four frozen S8 decision policies to one scored observation."""

    lesion = _string(row.get("lesion_class"), "lesion class")
    condition = _string(row.get("condition_id"), "condition")
    positive = float(cast(float, row["positive_score"]))
    negative = float(cast(float, row["negative_score"]))
    positive_threshold = _threshold_value(
        thresholds[lesion][condition]["PRESENT"], "present threshold"
    )
    negative_threshold = _threshold_value(
        thresholds[lesion][condition]["ABSENT"], "absent threshold"
    )
    main = _state(positive, negative, positive_threshold, negative_threshold)
    forced = (
        EntailmentState.ENTAILED_PRESENT.value
        if positive < negative
        else EntailmentState.ENTAILED_ABSENT.value
    )
    margin = abs(positive - negative)
    selective = (
        forced
        if margin > selective_thresholds[lesion][condition]
        else EntailmentState.RAW_UNIDENTIFIABLE.value
    )
    present_only = (
        EntailmentState.ENTAILED_PRESENT.value
        if positive <= positive_threshold
        else EntailmentState.OUT_OF_MODEL.value
    )
    absent_only = (
        EntailmentState.ENTAILED_ABSENT.value
        if negative <= negative_threshold
        else EntailmentState.OUT_OF_MODEL.value
    )
    return {
        **dict(row),
        "positive_threshold": positive_threshold,
        "negative_threshold": negative_threshold,
        "state_four_valued": main,
        "state_forced_nearest": forced,
        "state_selective_margin": selective,
        "state_present_only": present_only,
        "state_absent_only": absent_only,
        "selective_margin": margin,
        "selective_margin_threshold": selective_thresholds[lesion][condition],
    }


def _threshold_value(value: object, field: str) -> float:
    """Read either a fitted threshold record or a scalar test fixture."""

    if isinstance(value, Mapping):
        value = value.get("threshold")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise S8ControlledTruthError(f"{field} is missing or invalid")
    result = float(value)
    if not math.isfinite(result):
        raise S8ControlledTruthError(f"{field} is non-finite")
    return result


def wilson_interval(
    successes: int, total: int, *, z: float = 1.959963984540054
) -> tuple[float, float]:
    """Return a two-sided Wilson interval for a binomial proportion."""

    if total <= 0 or successes < 0 or successes > total:
        raise S8ControlledTruthError("invalid binomial counts")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return max(0.0, center - half), min(1.0, center + half)


def one_sided_zero_error_upper(total: int, *, alpha: float = 0.05) -> float | None:
    """Return the exact one-sided upper bound when zero errors are observed."""

    if total < 0 or not 0.0 < alpha < 1.0:
        raise S8ControlledTruthError("invalid zero-error bound inputs")
    return None if total == 0 else 1.0 - alpha ** (1.0 / total)
