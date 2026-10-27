"""Independent controlled-twin oracle contracts for Science Track analyses."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, TypeVar

ORACLE_SCHEMA_VERSION: Final = "1.0.0"
_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_EnumT = TypeVar("_EnumT", bound=StrEnum)


class OracleValidationError(ValueError):
    """Raised when an oracle record violates the independent-truth contract."""


class SourceTruth(StrEnum):
    """Lesion proposition truth in the controlled source."""

    PRESENT = "PRESENT"
    ABSENT = "ABSENT"


class SupportTruth(StrEnum):
    """Whether the normalized observation independently supports a proposition."""

    IDENTIFIABLE_PRESENT = "IDENTIFIABLE_PRESENT"
    IDENTIFIABLE_ABSENT = "IDENTIFIABLE_ABSENT"
    OBSERVATION_AMBIGUOUS = "OBSERVATION_AMBIGUOUS"
    ORACLE_UNRESOLVED = "ORACLE_UNRESOLVED"


class DomainTruth(StrEnum):
    """Membership in the preregistered data/degradation/lesion domain."""

    IN_DOMAIN = "IN_DOMAIN"
    OUT_OF_DOMAIN = "OUT_OF_DOMAIN"


class OracleTier(StrEnum):
    """Strength and construction of independent ambiguity evidence."""

    EXACT_COLLISION = "EXACT_COLLISION"
    CALIBRATED_SET_OVERLAP = "CALIBRATED_SET_OVERLAP"
    EMPIRICAL_EQUIVALENCE = "EMPIRICAL_EQUIVALENCE"


class CollisionMode(StrEnum):
    """Parameter relationship for a Tier-A exact collision."""

    SAME_THETA = "SAME_THETA"
    TWO_THETA = "TWO_THETA"


def _enum_value(value: _EnumT | str, enum_type: type[_EnumT], name: str) -> _EnumT:
    try:
        return value if isinstance(value, enum_type) else enum_type(value)
    except (TypeError, ValueError) as error:
        raise OracleValidationError(f"{name} is invalid") from error


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OracleValidationError(f"{name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True, slots=True, kw_only=True)
class ControlledTwinOracleRecord:
    """Versioned truth record produced independently of RetinaEntail scores."""

    source_truth: SourceTruth
    support_truth: SupportTruth
    domain_truth: DomainTruth
    oracle_method_id: str
    evidence_sha256: str
    retinaentail_independent: bool
    oracle_tier: OracleTier | None = None
    collision_mode: CollisionMode | None = None
    schema_version: str = ORACLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ORACLE_SCHEMA_VERSION:
            raise OracleValidationError(
                f"unsupported schema_version {self.schema_version!r}; "
                f"expected {ORACLE_SCHEMA_VERSION!r}"
            )
        object.__setattr__(
            self, "source_truth", _enum_value(self.source_truth, SourceTruth, "source_truth")
        )
        object.__setattr__(
            self, "support_truth", _enum_value(self.support_truth, SupportTruth, "support_truth")
        )
        object.__setattr__(
            self, "domain_truth", _enum_value(self.domain_truth, DomainTruth, "domain_truth")
        )
        object.__setattr__(
            self, "oracle_method_id", _nonempty(self.oracle_method_id, "oracle_method_id")
        )
        if (
            not isinstance(self.evidence_sha256, str)
            or _SHA256_PATTERN.fullmatch(self.evidence_sha256) is None
        ):
            raise OracleValidationError("evidence_sha256 must be a lowercase SHA-256 digest")
        if type(self.retinaentail_independent) is not bool:  # noqa: E721
            raise OracleValidationError("retinaentail_independent must be a bool")

        tier = self.oracle_tier
        if tier is not None:
            tier = _enum_value(tier, OracleTier, "oracle_tier")
            object.__setattr__(self, "oracle_tier", tier)
        collision_mode = self.collision_mode
        if collision_mode is not None:
            collision_mode = _enum_value(collision_mode, CollisionMode, "collision_mode")
            object.__setattr__(self, "collision_mode", collision_mode)

        if self.support_truth is SupportTruth.OBSERVATION_AMBIGUOUS:
            if tier is None:
                raise OracleValidationError("ambiguous support truth requires an oracle tier")
            if not self.retinaentail_independent:
                raise OracleValidationError("ambiguity oracle must be independent of RetinaEntail")
        elif tier is not None or collision_mode is not None:
            raise OracleValidationError("oracle tier is only valid for ambiguous support truth")

        if tier is OracleTier.EXACT_COLLISION and collision_mode is None:
            raise OracleValidationError("Tier-A exact collision requires collision_mode")
        if tier is not OracleTier.EXACT_COLLISION and collision_mode is not None:
            raise OracleValidationError("collision_mode is only valid for Tier-A exact collision")

    @property
    def confirmatory_ambiguity_eligible(self) -> bool:
        """Return confirmatory eligibility; every unresolved condition fails closed."""

        return (
            self.domain_truth is DomainTruth.IN_DOMAIN
            and self.support_truth is SupportTruth.OBSERVATION_AMBIGUOUS
            and self.oracle_tier in {OracleTier.EXACT_COLLISION, OracleTier.CALIBRATED_SET_OVERLAP}
            and self.retinaentail_independent
        )

    def to_dict(self) -> dict[str, str | bool | None]:
        """Serialize without RetinaEntail scores or feasibility outputs."""

        return {
            "schema_version": self.schema_version,
            "source_truth": self.source_truth.value,
            "support_truth": self.support_truth.value,
            "domain_truth": self.domain_truth.value,
            "oracle_tier": None if self.oracle_tier is None else self.oracle_tier.value,
            "collision_mode": (None if self.collision_mode is None else self.collision_mode.value),
            "oracle_method_id": self.oracle_method_id,
            "evidence_sha256": self.evidence_sha256,
            "retinaentail_independent": self.retinaentail_independent,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ControlledTwinOracleRecord:
        """Strictly deserialize a closed oracle record."""

        expected = {
            "schema_version",
            "source_truth",
            "support_truth",
            "domain_truth",
            "oracle_tier",
            "collision_mode",
            "oracle_method_id",
            "evidence_sha256",
            "retinaentail_independent",
        }
        missing = expected.difference(value)
        unknown = set(value).difference(expected)
        if missing:
            raise OracleValidationError(f"oracle record is missing: {', '.join(sorted(missing))}")
        if unknown:
            raise OracleValidationError(
                f"oracle record has unknown fields: {', '.join(sorted(unknown))}"
            )
        raw_tier = value["oracle_tier"]
        raw_collision = value["collision_mode"]
        return cls(
            schema_version=_nonempty(value["schema_version"], "schema_version"),
            source_truth=_enum_value(
                _nonempty(value["source_truth"], "source_truth"), SourceTruth, "source_truth"
            ),
            support_truth=_enum_value(
                _nonempty(value["support_truth"], "support_truth"),
                SupportTruth,
                "support_truth",
            ),
            domain_truth=_enum_value(
                _nonempty(value["domain_truth"], "domain_truth"), DomainTruth, "domain_truth"
            ),
            oracle_tier=(
                None
                if raw_tier is None
                else _enum_value(_nonempty(raw_tier, "oracle_tier"), OracleTier, "oracle_tier")
            ),
            collision_mode=(
                None
                if raw_collision is None
                else _enum_value(
                    _nonempty(raw_collision, "collision_mode"),
                    CollisionMode,
                    "collision_mode",
                )
            ),
            oracle_method_id=_nonempty(value["oracle_method_id"], "oracle_method_id"),
            evidence_sha256=_nonempty(value["evidence_sha256"], "evidence_sha256"),
            retinaentail_independent=value["retinaentail_independent"],  # type: ignore[arg-type]
        )


__all__ = [
    "ORACLE_SCHEMA_VERSION",
    "CollisionMode",
    "ControlledTwinOracleRecord",
    "DomainTruth",
    "OracleTier",
    "OracleValidationError",
    "SourceTruth",
    "SupportTruth",
]
