# Data access and license gate

The registry records public information entry points and reviewed acquisition evidence. This
project does not bundle or redistribute medical data and does not provide a general downloader.
IDRiD, DDR, DeepDRiD, HRF quality, and the metadata-only FGADR S7-P0 selection have explicit review
records; each record remains limited to its listed purposes.

## Mandatory manual process

Before any real-data adapter is run:

1. Open the official entry point manually and identify the exact release.
2. Read the dataset terms, access agreement, privacy restrictions, permitted
   purposes, attribution requirements, and redistribution rules.
3. Save a dated local terms snapshot and a review record outside Git if the terms
   themselves may not be redistributed.
4. Obtain access through the official login, request, agreement, or download flow.
   Do not automate CAPTCHA, form submission, authentication, or acceptance.
5. Create a strict review record containing the exact dataset/version, approved or denied
   decision, permitted purpose, SHA-256 of the terms/evidence, reviewer identifier, and a
   timezone-aware review timestamp. Do not fabricate or copy a review record between versions.
6. Store data outside this repository. The registry `root_env` is a human configuration hint;
   Stage 2 requires the root to be passed explicitly and never treats an environment variable as
   authorization.
7. Create an explicit adapter spec from verified official metadata. Unknown patient, eye,
   follow-up, pair, or label fields remain null/unknown; never infer them from filenames.
8. Compute local checksums and build a relative-path manifest. Never place
   credentials, signed URLs, or protected data in a manifest.
9. Keep the dataset blocked whenever the data license, provenance, identity
   mapping, or label meaning remains unresolved.

A GitHub code license applies to code only unless the dataset terms explicitly
say otherwise. Repository visibility, a working download link, or possession of
files is not evidence of permission to use or redistribute the data.

## Recorded entry points

| Dataset | Official/public information entry point | Access handling | Current license state |
|---|---|---|---|
| IDRiD | <https://idrid.grand-challenge.org/Data/>; original record <https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid>; transport mirror <https://zenodo.org/records/17219542> | Official IEEE download is login-gated. A user-authorized, one-off acquisition used the CC BY 4.0 mirror; data remain outside Git | **Verified/reviewed** for exact version `idrid-ieee-h25w98-2026-01-12-zenodo17219542-v1` |
| DDR / OIA-DDR | <https://github.com/nkicsl/DDR-dataset> | Official public Google Drive contains ten split ZIP parts; keep raw data outside Git | **Verified/reviewed** for task-scoped non-commercial research/training on `ddr-oia-official-drive-2019-07-10-v1`; local transfer/hash preflight remains incomplete |
| DeepDRiD | <https://github.com/deepdrdoc/DeepDRiD> | Select and record an exact official release manually | Unknown; pending review |
| EyeQ | <https://github.com/hzfu/EyeQ> | Review EyeQ labels/code and underlying EyePACS image access separately | Unknown; pending review |
| Real-Fundus | <https://github.com/dengzhuo-AI/Real-Fundus> | Do not use or redistribute until explicit dataset terms are reviewed | **Ambiguous; pending review**. Code licensing is not a data license |
| FGADR | <https://csyizhou.github.io/FGADR/> | Manual signed Research Use Agreement and personal email-delivered link; never automate or share the link | **Restricted/reviewed for metadata validation only** on `fgadr-official-seg-set-1842-unnumbered-p0`; payload use remains blocked pending executed agreement and delivery-specific review |
| MAPLES-DR | <https://github.com/LIV4D/MAPLES-DR> (documentation: <https://liv4d.github.io/MAPLES-DR/en/>) | Acquire original MESSIDOR images separately and legally, then preserve the mapping | Unknown; pending review |
| FIVES | <https://figshare.com/articles/figure/FIVES_A_Fundus_Image_Dataset_for_AI-based_Vessel_Segmentation/19688169> | Review the terms for the exact Figshare version before manual acquisition | Unknown; pending review |
| HRF quality subset | <https://www5.informatik.uni-erlangen.de/indexbbb3.html?L=&id=1531&type=98> | Public `allQuality.zip`; 18 same-eye good/bad acquisition pairs; source images stay outside Git | **Verified/reviewed**, CC BY 4.0, exact archive SHA-256 `e0818e5e…d86ca78` |

## Dataset-specific constraints

- **IDRiD:** the official project page states CC BY 4.0 and identifies IEEE DataPort DOI
  `10.21227/H25W98` as the original record. The acquired archives used Zenodo record 17219542 v1
  only as a transport mirror after the official gated download failed. The mirror uploader is not
  the dataset creator and IEEE exposes no archive checksums, so the review records local SHA-256
  and mirror MD5 values but does not claim cryptographic byte identity with the IEEE-hosted copy.
  Patient, eye, follow-up, and intervention-pair identities are not inferred from filenames.
- **EyeQ:** EyeQ metadata/labels do not themselves authorize use of EyePACS
  images. Verify both provenance layers.
- **MAPLES-DR:** annotation access and MESSIDOR image access are separate. Never
  infer image rights from the annotation repository.
- **DDR:** the responsible operator's exact-version non-commercial review is recorded independently
  from the GitHub code license. The S4 recovery remains blocked until all ten official parts are
  complete, hashed, structurally verified and restricted to official TRAIN/VALID development data.
  The review does not authorize redistribution or locked-test use.
- **Real-Fundus:** remains fail-closed at the license gate while its registry is
  `license_status: ambiguous` and `license_review_status: pending_review`.
- **FGADR:** S7-P0 freezes the official unnumbered Seg-set described by the provider as 1,842
  images with MA/HE/EX/SE/IRMA/NV pixel labels. The current review authorizes metadata validation
  only. A signed agreement, personal provider link, delivery-specific review, and local hashes are
  mandatory before any payload or root read; the Grade-set is outside the protocol.
- **HRF quality subset:** only supports independent acquisition-error contract preflight. Its 18
  good/bad pairs are not lesion truth and are insufficient alone for confirmatory Tier B.
- **FIVES:** use only for vessel/structure support unless a future reviewed
  protocol gives it another justified role; it is not DR lesion ground truth.

No downloader is provided by design. If an official link is unavailable, record the issue and use
only a source whose redistribution permission and provenance have been explicitly reviewed; never
bypass access controls. A transport mirror must be identified as such, and missing publisher hashes
must remain an explicit provenance limitation.
