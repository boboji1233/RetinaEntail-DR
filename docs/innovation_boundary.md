# Innovation boundary

RetinaEntail-DR asks whether a lesion proposition in an enhanced fundus image is
supported by the original low-quality observation, relative to a declared image
and degradation model, calibration distribution, region, and lesion class. It
tests both directions: whether a lesion-present explanation is feasible and
whether a lesion-absent explanation is feasible. Their joint result gives the
four model-relative states `ENTAILED_PRESENT`, `ENTAILED_ABSENT`,
`RAW_UNIDENTIFIABLE`, and `OUT_OF_MODEL`.

## What the project contributes

- A four-valued representation of lesion-level observational support.
- Symmetric lesion-present and lesion-absent feasibility tests.
- Explicit recognition that the raw observation can be non-identifying.
- Replayable counterexample witnesses tied to configuration, data, model, and
  solver provenance.
- Safety-audit rules for additive hallucination, subtractive hallucination,
  unsupported assertion, and out-of-scope assertion by an arbitrary enhancer.

All conclusions are model-relative and calibration-backed. They are not claims of
universal mathematical proof, clinical certification, or validity for every
camera, degradation, patient, and population.

## Not a new enhancement network

The enhancer is an object being audited, not the proposed core method. A GAN,
Transformer, diffusion model, lesion-preservation loss, or quality metric may be
used as a comparator or implementation component, but inventing another enhancer
is outside the central contribution. The four-state semantics and two-hypothesis
search must remain meaningful even if learned components are replaced with small
templates or constrained optimization.

## Not ordinary predictive uncertainty

Entropy, MC dropout, ensembles, confidence scores, or image-quality probabilities
describe variability or confidence of a predictor. They do not by themselves
construct two opposite latent explanations that both replay the same raw
observation. RetinaEntail-DR therefore cannot collapse its decision to one model
probability or rename low confidence as `RAW_UNIDENTIFIABLE`. Optimization
failure, unknown degradation, invalid witness, and domain mismatch fail closed to
`OUT_OF_MODEL`, not to a determined lesion state.

## Not conformal prediction alone

Conformal methods can calibrate positive and negative feasibility thresholds and
support finite-sample statistical error control under stated assumptions. They
are a guarantee mechanism, not the sole innovation. Applying a conformal wrapper
to a DR classifier would omit the lesion proposition, dual feasibility search,
model-relative four-valued semantics, counterexample witnesses, and enhancer
audit taxonomy that define this project.

## Claims that remain out of scope

- A new state-of-the-art enhancement, segmentation, or DR-grading model.
- Better PSNR, SSIM, Dice, AUC, or perceptual quality as the primary endpoint.
- Certainty outside the registered degradation/model domain.
- Treating a missing annotation as lesion absence.
- Treating a code repository license as a medical-data license.
- Fabricated performance, expert review, dataset access, or experimental results.

The engineering priority is to determine when the evidence is insufficient before
trying to increase the number of determined cases.
