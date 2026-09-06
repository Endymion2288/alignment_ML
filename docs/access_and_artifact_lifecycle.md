# T01 Access Boundary and Artifact Lifecycle

Fail-closed data access and immutable run directories.  Sealed test is not a
development flag.  No production, no event-content read.

## Scopes

| Scope | Allowed splits | May open sealed test |
| --- | --- | --- |
| `train` | train | no |
| `development_validation` | train, validation | no |
| `real_data_monitoring` | train, validation | no |
| `frozen_evaluation` | train, validation | no in T00–T12 |

`--allow-sealed-test` and `--evaluate-test` raise `AccessPolicyError`.  They
are not a `FrozenEvaluationCapability`.

Shared entry: `datasets.access_policy.load_curriculum_for_scope`.  Test sample
paths are authorized before `_require_path`, so missing sealed assets are never
resolved.

## Artifact store

`evaluation.artifact_store.ImmutableArtifactStore`:

- unique `run_id`
- exclusive directory
- temp file + same-filesystem `os.replace`
- second write → `FileExistsError`
- missing `COMPLETE.json` → incomplete

## Artifacts

`outputs/access_and_artifact_lifecycle_v1/<run_id>/`

- `access_policy_audit.json`
- `immutable_writer_smoke.json`
- `COMPLETE.json`
