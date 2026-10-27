"""Independent loading and replay checks for Stage 0/1 witness packages."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from retinaentail.data.manifest import sha256_file
from retinaentail.degradations import replay_trace
from retinaentail.entailment.toy_baseline import normalized_mse
from retinaentail.interventions.toy import toy_lesion_predicate
from retinaentail.schemas import JSONValue, LesionRegion, WitnessRecord


def canonical_json_bytes(value: JSONValue | dict[str, JSONValue]) -> bytes:
    """Encode a JSON-safe object deterministically and reject NaN/Infinity."""

    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_bytes_fail_safe(path: Path, payload: bytes) -> None:
    """Create an artifact or accept an identical existing artifact; never clobber differences."""

    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite non-identical artifact: {path}") from None


def rgb_png_bytes(image: NDArray[np.uint8]) -> bytes:
    from io import BytesIO

    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("PNG encoder expects RGB uint8 HxWx3")
    stream = BytesIO()
    Image.fromarray(image, mode="RGB").save(stream, format="PNG", compress_level=9)
    return stream.getvalue()


def load_rgb(path: Path) -> NDArray[np.uint8]:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def _resolve_artifact(root: Path, relative_path: str) -> Path:
    resolved_root = root.resolve(strict=True)
    candidate = (resolved_root / Path(relative_path)).resolve(strict=True)
    try:
        candidate.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"artifact escapes witness root: {relative_path}") from error
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


@dataclass(frozen=True, slots=True)
class ReplayResult:
    witness_id: str
    replay_error: float
    predicate_valid: bool
    digest_valid: bool
    passed: bool


def verify_witness(
    root: Path,
    witness: WitnessRecord,
    region: LesionRegion,
    *,
    tolerance: float = 1e-12,
) -> ReplayResult:
    """Independently replay one witness and verify digest, error, and toy predicate."""

    if not isinstance(root, Path):
        raise TypeError("root must be a pathlib.Path")
    if not isinstance(witness, WitnessRecord):
        raise TypeError("witness must be a WitnessRecord")
    if not isinstance(region, LesionRegion):
        raise TypeError("region must be a LesionRegion")
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    latent_path = _resolve_artifact(root, witness.relative_path)
    observation_path = _resolve_artifact(root, witness.observation_path)
    digest_valid = sha256_file(latent_path) == witness.sha256
    latent = load_rgb(latent_path)
    observation = load_rgb(observation_path)
    replay = replay_trace(latent, witness.degradation_trace.to_dict())
    error = normalized_mse(replay, observation)
    predicate = toy_lesion_predicate(latent, region)
    expected_predicate = witness.hypothesis_side == "positive"
    predicate_valid = predicate is expected_predicate and witness.predicate_valid
    error_valid = abs(error - witness.replay_error) <= tolerance
    return ReplayResult(
        witness_id=witness.witness_id,
        replay_error=error,
        predicate_valid=predicate_valid,
        digest_valid=digest_valid,
        passed=digest_valid and predicate_valid and error_valid,
    )


def load_witness(path: Path) -> WitnessRecord:
    decoded: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("witness JSON must contain an object")
    return WitnessRecord.from_dict(decoded)


__all__ = [
    "ReplayResult",
    "canonical_json_bytes",
    "load_rgb",
    "load_witness",
    "rgb_png_bytes",
    "sha256_bytes",
    "verify_witness",
    "write_bytes_fail_safe",
]
