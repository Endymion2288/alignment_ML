# T01 访问边界与产物生命周期

默认 fail-closed。Sealed test 不是开发开关。无 production，不读 event 内容。

## Scope

| Scope | 允许的 split | 可否打开 sealed test |
| --- | --- | --- |
| `train` | train | 否 |
| `development_validation` | train, validation | 否 |
| `real_data_monitoring` | train, validation | 否 |
| `frozen_evaluation` | train, validation | T00–T12 否 |

`--allow-sealed-test` 与 `--evaluate-test` 会抛 `AccessPolicyError`，不是
`FrozenEvaluationCapability`。

共享入口：`datasets.access_policy.load_curriculum_for_scope`。在
`_require_path` 之前授权，因此缺失的 sealed 资产不会被 resolve。

## Artifact store

`evaluation.artifact_store.ImmutableArtifactStore`：唯一 `run_id`、独占目录、
临时文件 + 同文件系统 `os.replace`；第二次写 `FileExistsError`；没有
`COMPLETE.json` 即未完成。

## 产物

`outputs/access_and_artifact_lifecycle_v1/<run_id>/` 下的
`access_policy_audit.json`、`immutable_writer_smoke.json`、`COMPLETE.json`。
