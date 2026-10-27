from __future__ import annotations

import json

import pytest

from retinaentail.interventions.oracle import (
    CollisionMode,
    ControlledTwinOracleRecord,
    DomainTruth,
    OracleTier,
    OracleValidationError,
    SourceTruth,
    SupportTruth,
)

EVIDENCE_HASH = "a" * 64


def _record(**overrides: object) -> ControlledTwinOracleRecord:
    values: dict[str, object] = {
        "source_truth": SourceTruth.PRESENT,
        "support_truth": SupportTruth.OBSERVATION_AMBIGUOUS,
        "domain_truth": DomainTruth.IN_DOMAIN,
        "oracle_tier": OracleTier.EXACT_COLLISION,
        "collision_mode": CollisionMode.SAME_THETA,
        "oracle_method_id": "normalized-array-exact-v1",
        "evidence_sha256": EVIDENCE_HASH,
        "retinaentail_independent": True,
    }
    values.update(overrides)
    return ControlledTwinOracleRecord(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("tier", "collision_mode", "expected"),
    [
        (OracleTier.EXACT_COLLISION, CollisionMode.SAME_THETA, True),
        (OracleTier.EXACT_COLLISION, CollisionMode.TWO_THETA, True),
        (OracleTier.CALIBRATED_SET_OVERLAP, None, True),
        (OracleTier.EMPIRICAL_EQUIVALENCE, None, False),
    ],
)
def test_confirmatory_eligibility_is_limited_to_independent_tier_a_b(
    tier: OracleTier, collision_mode: CollisionMode | None, expected: bool
) -> None:
    assert (
        _record(oracle_tier=tier, collision_mode=collision_mode).confirmatory_ambiguity_eligible
        is expected
    )


def test_out_of_domain_ambiguity_fails_closed_from_confirmatory_analysis() -> None:
    assert not _record(domain_truth=DomainTruth.OUT_OF_DOMAIN).confirmatory_ambiguity_eligible


def test_oracle_record_round_trip_is_closed_and_score_free() -> None:
    record = _record()
    payload = json.loads(json.dumps(record.to_dict(), sort_keys=True))
    assert "score" not in payload
    assert "feasible" not in payload
    assert ControlledTwinOracleRecord.from_dict(payload) == record


@pytest.mark.parametrize(
    "overrides",
    [
        {"oracle_tier": None, "collision_mode": None},
        {"retinaentail_independent": False},
        {"oracle_tier": OracleTier.EXACT_COLLISION, "collision_mode": None},
        {
            "oracle_tier": OracleTier.CALIBRATED_SET_OVERLAP,
            "collision_mode": CollisionMode.SAME_THETA,
        },
    ],
)
def test_invalid_ambiguity_evidence_is_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(OracleValidationError):
        _record(**overrides)


def test_unresolved_record_cannot_carry_oracle_tier() -> None:
    with pytest.raises(OracleValidationError, match="only valid"):
        _record(support_truth=SupportTruth.ORACLE_UNRESOLVED)


def test_unknown_serialized_field_is_rejected() -> None:
    payload: dict[str, object] = _record().to_dict()
    payload["positive_score"] = 0.0
    with pytest.raises(OracleValidationError, match="unknown"):
        ControlledTwinOracleRecord.from_dict(payload)
