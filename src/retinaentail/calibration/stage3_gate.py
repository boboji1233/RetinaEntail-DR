"""Path-free Stage 3 input preflight that never grants dataset access."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

STAGE3_PREFLIGHT_SCHEMA_VERSION: Final = "3.0.0"
_DATASET_ID_PATTERN: Final = re.compile(r"^[a-z][a-z0-9_]*$")
_DATASET_VERSION_PATTERN: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}$")


class Stage3PreflightStatus(StrEnum):
    """Whether explicit inputs are present for a later Stage 2 license Gate attempt."""

    BLOCKED = "BLOCKED"
    READY_FOR_STAGE2_GATE = "READY_FOR_STAGE2_GATE"


class Stage3PreflightReason(StrEnum):
    """Controlled, path-free missing-input reasons."""

    DATASET_NOT_EXPLICIT = "DATASET_NOT_EXPLICIT"
    DATASET_VERSION_NOT_EXPLICIT = "DATASET_VERSION_NOT_EXPLICIT"
    APPROVED_REVIEW_NOT_PROVIDED = "APPROVED_REVIEW_NOT_PROVIDED"
    DATASET_ROOT_NOT_PROVIDED = "DATASET_ROOT_NOT_PROVIDED"
    ADAPTER_SPEC_NOT_PROVIDED = "ADAPTER_SPEC_NOT_PROVIDED"


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage3PreflightReport:
    """A zero-I/O presence report; READY is not an access capability."""

    status: Stage3PreflightStatus
    dataset_id: str | None
    dataset_version: str | None
    review_record_provided: bool
    dataset_root_provided: bool
    adapter_spec_provided: bool
    reasons: tuple[Stage3PreflightReason, ...]
    schema_version: str = STAGE3_PREFLIGHT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STAGE3_PREFLIGHT_SCHEMA_VERSION:
            raise ValueError("unsupported Stage 3 preflight schema_version")
        if not isinstance(self.status, Stage3PreflightStatus):
            raise TypeError("status must be Stage3PreflightStatus")
        if self.dataset_id is not None and (
            not isinstance(self.dataset_id, str)
            or _DATASET_ID_PATTERN.fullmatch(self.dataset_id) is None
        ):
            raise ValueError("dataset_id must be a normalized identifier or null")
        if self.dataset_version is not None and (
            not isinstance(self.dataset_version, str)
            or _DATASET_VERSION_PATTERN.fullmatch(self.dataset_version) is None
        ):
            raise ValueError("dataset_version must be a report-safe identifier or null")
        presence = (
            self.review_record_provided,
            self.dataset_root_provided,
            self.adapter_spec_provided,
        )
        if any(type(value) is not bool for value in presence):  # noqa: E721
            raise TypeError("preflight presence fields must be exact bool values")
        if not all(isinstance(reason, Stage3PreflightReason) for reason in self.reasons):
            raise TypeError("reasons must contain Stage3PreflightReason values")
        normalized = tuple(dict.fromkeys(self.reasons))
        object.__setattr__(self, "reasons", normalized)
        expected: list[Stage3PreflightReason] = []
        if self.dataset_id is None:
            expected.append(Stage3PreflightReason.DATASET_NOT_EXPLICIT)
        if self.dataset_version is None:
            expected.append(Stage3PreflightReason.DATASET_VERSION_NOT_EXPLICIT)
        if not self.review_record_provided:
            expected.append(Stage3PreflightReason.APPROVED_REVIEW_NOT_PROVIDED)
        if not self.dataset_root_provided:
            expected.append(Stage3PreflightReason.DATASET_ROOT_NOT_PROVIDED)
        if not self.adapter_spec_provided:
            expected.append(Stage3PreflightReason.ADAPTER_SPEC_NOT_PROVIDED)
        expected_status = (
            Stage3PreflightStatus.BLOCKED
            if expected
            else Stage3PreflightStatus.READY_FOR_STAGE2_GATE
        )
        if self.status is not expected_status or normalized != tuple(expected):
            raise ValueError("preflight status/reasons conflict with explicit input presence")

    @property
    def authorizes_data_access(self) -> bool:
        """A preflight can never substitute for a Stage 2 DatasetAccessPermit."""

        return False

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "review_record_provided": self.review_record_provided,
            "dataset_root_provided": self.dataset_root_provided,
            "adapter_spec_provided": self.adapter_spec_provided,
            "reasons": [reason.value for reason in self.reasons],
            "authorizes_data_access": False,
            "real_data_read": False,
            "sample_count": 0,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Stage3PreflightReport:
        required = {
            "schema_version",
            "status",
            "dataset_id",
            "dataset_version",
            "review_record_provided",
            "dataset_root_provided",
            "adapter_spec_provided",
            "reasons",
            "authorizes_data_access",
            "real_data_read",
            "sample_count",
        }
        if set(value) != required:
            raise ValueError("Stage 3 preflight has missing or unknown fields")
        if value["authorizes_data_access"] is not False:
            raise ValueError("preflight cannot authorize data access")
        if value["real_data_read"] is not False or value["sample_count"] != 0:
            raise ValueError("preflight must contain zero real-data activity")
        raw_reasons = value["reasons"]
        if not isinstance(raw_reasons, list) or not all(
            isinstance(reason, str) for reason in raw_reasons
        ):
            raise ValueError("preflight reasons must be a list of strings")
        raw_status = value["status"]
        if not isinstance(raw_status, str):
            raise ValueError("preflight status must be a string")
        try:
            status = Stage3PreflightStatus(raw_status)
            reasons = tuple(Stage3PreflightReason(reason) for reason in raw_reasons)
        except (TypeError, ValueError) as error:
            raise ValueError("preflight status or reason is invalid") from error
        schema_version = value["schema_version"]
        dataset_id = value["dataset_id"]
        dataset_version = value["dataset_version"]
        presence = (
            value["review_record_provided"],
            value["dataset_root_provided"],
            value["adapter_spec_provided"],
        )
        if not isinstance(schema_version, str):
            raise ValueError("preflight schema_version must be a string")
        if dataset_id is not None and not isinstance(dataset_id, str):
            raise ValueError("preflight dataset_id must be a string or null")
        if dataset_version is not None and not isinstance(dataset_version, str):
            raise ValueError("preflight dataset_version must be a string or null")
        if any(type(item) is not bool for item in presence):  # noqa: E721
            raise ValueError("preflight presence fields must be booleans")
        review_provided, root_provided, spec_provided = presence
        assert isinstance(review_provided, bool)
        assert isinstance(root_provided, bool)
        assert isinstance(spec_provided, bool)
        return cls(
            schema_version=schema_version,
            status=status,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            review_record_provided=review_provided,
            dataset_root_provided=root_provided,
            adapter_spec_provided=spec_provided,
            reasons=reasons,
        )


def assess_stage3_preflight(
    *,
    dataset_id: str | None,
    dataset_version: str | None,
    review_record_provided: bool,
    dataset_root_provided: bool,
    adapter_spec_provided: bool,
) -> Stage3PreflightReport:
    """Assess only values/presence; do not accept a path object or perform filesystem I/O."""

    presence = (review_record_provided, dataset_root_provided, adapter_spec_provided)
    if any(type(value) is not bool for value in presence):  # noqa: E721
        raise TypeError("preflight presence fields must be exact bool values")
    reasons: list[Stage3PreflightReason] = []
    if dataset_id is None:
        reasons.append(Stage3PreflightReason.DATASET_NOT_EXPLICIT)
    if dataset_version is None:
        reasons.append(Stage3PreflightReason.DATASET_VERSION_NOT_EXPLICIT)
    if not review_record_provided:
        reasons.append(Stage3PreflightReason.APPROVED_REVIEW_NOT_PROVIDED)
    if not dataset_root_provided:
        reasons.append(Stage3PreflightReason.DATASET_ROOT_NOT_PROVIDED)
    if not adapter_spec_provided:
        reasons.append(Stage3PreflightReason.ADAPTER_SPEC_NOT_PROVIDED)
    return Stage3PreflightReport(
        status=(
            Stage3PreflightStatus.BLOCKED
            if reasons
            else Stage3PreflightStatus.READY_FOR_STAGE2_GATE
        ),
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        review_record_provided=review_record_provided,
        dataset_root_provided=dataset_root_provided,
        adapter_spec_provided=adapter_spec_provided,
        reasons=tuple(reasons),
    )


__all__ = [
    "STAGE3_PREFLIGHT_SCHEMA_VERSION",
    "Stage3PreflightReason",
    "Stage3PreflightReport",
    "Stage3PreflightStatus",
    "assess_stage3_preflight",
]
