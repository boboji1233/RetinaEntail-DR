import numpy as np
import pytest

from retinaentail.entailment.toy_baseline import normalized_mse, score_toy_hypotheses
from retinaentail.interventions.toy import generate_toy_fundus, toy_lesion_predicate


def test_toy_twins_change_only_controlled_lesion() -> None:
    twin = generate_toy_fundus(size=96, lesion_radius=4, seed=7)
    changed = np.any(twin.lesion_present != twin.lesion_absent, axis=2)
    assert np.array_equal(changed, twin.lesion_mask)
    assert toy_lesion_predicate(twin.lesion_present, twin.region)
    assert not toy_lesion_predicate(twin.lesion_absent, twin.region)


def test_toy_generation_is_seed_deterministic() -> None:
    first = generate_toy_fundus(seed=91)
    second = generate_toy_fundus(seed=91)
    assert np.array_equal(first.lesion_present, second.lesion_present)
    assert np.array_equal(first.lesion_absent, second.lesion_absent)


def test_toy_pixel_baseline_is_explicitly_symmetric() -> None:
    twin = generate_toy_fundus(seed=2)
    observation = twin.lesion_present
    positive, negative = score_toy_hypotheses(
        observation,
        twin.lesion_present,
        twin.lesion_absent,
        threshold=0.02,
    )
    assert positive.score == 0.0
    assert positive.feasible
    assert negative.feasible
    assert normalized_mse(twin.lesion_absent, observation) == negative.score


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -0.1])
def test_toy_pixel_baseline_rejects_invalid_threshold(threshold: float) -> None:
    twin = generate_toy_fundus(seed=3)
    with pytest.raises(ValueError):
        score_toy_hypotheses(
            twin.lesion_present,
            twin.lesion_present,
            twin.lesion_absent,
            threshold=threshold,
        )
