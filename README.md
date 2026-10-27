# RetinaEntail-DR

Lesion-level entailment auditing for diabetic retinopathy image enhancement.

RetinaEntail-DR asks whether a raw fundus observation supports a lesion-present
explanation, a lesion-absent explanation, both, or neither. It combines dual
feasibility checks with replayable witnesses and explicit model boundaries.
The method audits the evidence available in the raw observation; it is not an
image-enhancement network or a replacement for clinical assessment.

## Four-valued outputs

| Positive feasible | Negative feasible | Output |
| --- | --- | --- |
| Yes | No | `ENTAILED_PRESENT` |
| No | Yes | `ENTAILED_ABSENT` |
| Yes | Yes | `RAW_UNIDENTIFIABLE` |
| No | No | `OUT_OF_MODEL` |

Missing thresholds, non-finite scores, unsupported degradations, failed solvers,
and invalid witnesses produce `OUT_OF_MODEL`, rather than a determinate claim.

## Installation

Use Python 3.11 on Windows, or Python 3.11/3.12 on Linux. From the repository root:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Runtime dependencies are NumPy, Pillow, PyYAML, and cryptography. The core smoke
example uses the CPU and does not download datasets, model weights, or services.

## Quick start

```bash
python -m retinaentail.cli doctor
python -m retinaentail.cli registry validate
python -m retinaentail.cli smoke --output-dir outputs/smoke-example
python -m pytest tests -q
```

The smoke command generates a deterministic synthetic fundus patch and its
lesion-present/lesion-absent twins, applies a shared degradation trace, and writes
a certificate and replayable witnesses. Synthetic examples demonstrate software
behavior only; they are not clinical validation or paper-performance results.

Use a new output directory for each run; output writers preserve existing files.

## Repository layout

```text
src/retinaentail/       Core library and research implementations
configs/               Portable synthetic calibration and evaluation fixtures
data_registry/         Dataset metadata, adapter contracts, and JSON schemas
scripts/               Scientific experiment entry points
tests/                 Portable offline unit and property tests
docs/                  Method boundaries, oracle contract, and data-access guide
```

The library includes controlled-twin construction, degradation calibration,
Verify/Lite experiments, four-state reasoning, witness replay, and signed-package
verification. The public test suite exercises the reusable offline software.
Historical acquisition runs, preregistered run configurations, hash-bound research
archives, and manuscript-production workflows are maintained separately. Scientific
entry points require their matching run configuration and evidence bundle; their
presence here does not make historical dataset experiments self-contained.

## Data and reproducibility

Acquire datasets independently from the original sources and keep them outside
the repository. Registry entries describe particular releases; they do not grant
permission to obtain, process, or redistribute the data. Real-data execution
requires the applicable review records, explicit dataset roots, adapter contracts,
and matching evidence. Missing labels remain unknown, never negative.

Raw medical images, annotation archives, derived patient-level artifacts,
checkpoints, credentials, and local experiment outputs are not distributed here.
See [data access](docs/data_access.md), [method boundaries](docs/innovation_boundary.md),
and the [controlled-twin oracle](docs/controlled_twin_oracle_spec.md).

## Citation and licensing

Author: Haifeng Qiu (邱海峰). Citation metadata is available in
[`CITATION.cff`](CITATION.cff).

The package currently retains its original proprietary research-scaffold license
status; a separate open-source license has not been selected. Public repository
access does not grant additional reuse or redistribution rights. Dataset licenses
are independent of the software license.

This is research software, not a medical device or clinical decision system.
