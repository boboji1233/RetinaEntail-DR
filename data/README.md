# Local data boundary

This directory is a documentation anchor, not a data distribution. No medical
image, annotation archive, derived patch, manifest containing credentials, or
access token belongs in Git.

## Local layout

Keep each legally acquired dataset outside the repository. A registry variable may help a human
locate its root, for example:

```powershell
$env:RETINAENTAIL_IDRID_ROOT = "D:\datasets\IDRiD"
```

Stage 2 does not read this variable as authorization: every real audit also requires an explicit
CLI root, adapter spec, matching approved review record, and verified registry state.
Environment variables are deliberately dataset-specific. Never commit their
values, machine-specific paths, credentials, cookies, or signed URLs. Registry
templates, schemas, checksum manifests, and split manifests may be committed only
when they contain metadata and relative paths, not protected content or secrets.

## Safety rules

- Follow [the manual access guide](../docs/data_access.md) before indexing data.
- Treat a missing label or mask as unknown, never as a negative finding.
- Preserve official source identifiers so all twins, crops, degradations, and
  enhancement outputs remain in the same split.
- Do not redistribute source images or annotations unless a separately reviewed
  license explicitly permits it.
- Do not create downloaders that bypass login, CAPTCHA, agreements, or access
  controls.

Stage 0/1 and the Stage 2 fixture do not require any real dataset. The smoke and fixture pipelines
are procedural and must run with all dataset environment variables unset. If no exact dataset
version passes the license Gate, real-data audit remains `BLOCKED` and performs no root I/O.
