"""Transparent pixel-replay baseline used only by the synthetic smoke pipeline."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from retinaentail.degradations import validate_image
from retinaentail.schemas import HypothesisScore


def normalized_mse(left: NDArray[np.uint8], right: NDArray[np.uint8]) -> float:
    """Mean squared RGB error on [0, 1], with strict shape/type validation."""

    valid_left = validate_image(left)
    valid_right = validate_image(right)
    if valid_left.shape != valid_right.shape:
        raise ValueError("images must have identical shapes")
    difference = valid_left.astype(np.float64) / 255.0 - valid_right.astype(np.float64) / 255.0
    result = float(np.mean(np.square(difference), dtype=np.float64))
    if not math.isfinite(result):
        raise ValueError("normalized MSE is not finite")
    return result


def score_toy_hypotheses(
    observation: NDArray[np.uint8],
    positive_replay: NDArray[np.uint8],
    negative_replay: NDArray[np.uint8],
    *,
    threshold: float,
) -> tuple[HypothesisScore, HypothesisScore]:
    """Score both explicit hypotheses; this is not the planned research solver."""

    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TypeError("threshold must be a real number")
    calibrated_threshold = float(threshold)
    if not math.isfinite(calibrated_threshold) or calibrated_threshold < 0.0:
        raise ValueError("threshold must be finite and non-negative")
    positive_score = normalized_mse(positive_replay, observation)
    negative_score = normalized_mse(negative_replay, observation)
    return (
        HypothesisScore(
            score=positive_score,
            threshold=calibrated_threshold,
            feasible=positive_score <= calibrated_threshold,
            solver_status="toy_baseline",
        ),
        HypothesisScore(
            score=negative_score,
            threshold=calibrated_threshold,
            feasible=negative_score <= calibrated_threshold,
            solver_status="toy_baseline",
        ),
    )


__all__ = ["normalized_mse", "score_toy_hypotheses"]
