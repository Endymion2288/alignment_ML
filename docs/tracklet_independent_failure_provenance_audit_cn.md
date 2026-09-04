# Tracklet 独立验证失败源只读 provenance 审计 V1

Workbook 72 / 2026-09-03。条目 69 继续冻结
`cross_source_stable_core_independent_validation_fail`。本审计不
重开那次战役，不改 `S` 或 `rank_tolerance=0.01`，不删 rank 2/3/4
源，不强制 rank 5，也不把这些源当作 cluster-local 确认。

只读分类 coverage loss 与 pipeline / artifact corruption：

- `mc24_100043_00400_00499`（rank 2）
- `mc24_100043_00500_00599`（rank 3）
- `mc24_100044_00200_00299`（rank 4）

预注册 sibling 对照来自条目 69 已通过的独立验证名单，只作只读
比较：

- `mc24_100043_00300_00399`
- `mc24_100044_00400_00499`

不重新训练 V2/V3/Transformer，不改冻结的 pairwise/route policy，
不 Newton，不写 geometry，不产生 alignment payload。真实数据仍
是 `residual_dq_monitoring_only`。不打开 sealed test。不重跑
Athena。

## 冻结合同

- 真实数据仍是 `residual_dq_monitoring_only`。
- `geometry_write_allowed=false`。
- 冻结 V2 checkpoint SHA256
  `0c85a001677678163512c59bd5ceb6d08bdefaab79c0f4628659a4a8ad766a27`。
- 继承条目 69
  `cross_source_stable_core_independent_validation_fail`。
- 必须用冻结 `S = (5, 5, 5, 60, 60, 60, 0.12)` 和冻结
  `rank_tolerance = 0.01` 构造 `A = W^{1/2} J S`。原生冻结-rank
  SVD，禁止截成 5D。
- 严重 artifact：缺失 FD probe、plus/minus population 不对齐、
  非有限残差、访问 sealed test、held-out 点混入 SVD。这些才判
  pipeline corruption。
- pair covariance 病态只是警告。若 rank-5 sibling 也有，不能单
  独判 corruption。
- 源不得删除。rank 不得强制为 5。stable-core criterion 不得重
  定义。这些源不是 cluster-local confirmatory set。

## 判定

`tracklet_independent_failure_provenance_audit_complete_sources_not_dropped`

| 门 | 值 |
| --- | --- |
| 总体分类 | **normal_physics_or_coverage_loss** |
| 源被删除 | **false** |
| rank 被强制为 5 | **false** |
| stable-core criterion 被重定义 | **false** |
| 用作 cluster-local 确认 | **false** |
| 条目 69 结论 | **继续冻结** |
| geometry write | **false** |

`+/-` probe 完整，population 对齐，残差有限，没有近零
Jacobian 列，没有混入 held-out 点，没有打开 sealed test。原生
rank 2 / 3 / 4 被复现。共同的
`ill_conditioned_pair_covariance` flag 在两个 rank-5 sibling 上
同样出现，因此不能标成 pipeline corruption。

## 结果

配置
`configs/tracklet_independent_failure_provenance_audit_v1.yaml`，
SHA256
`b73e35148e7ee8b31a579db741d1a9d36a0571f2df80215e3b16f10f432ee410`。

| source | pairs | 官方 rank | 分类 |
| --- | ---: | ---: | --- |
| `mc24_100043_00400_00499` | 240 | **2** | coverage loss |
| `mc24_100043_00500_00599` | 234 | **3** | coverage loss |
| `mc24_100044_00200_00299` | 234 | **4** | coverage loss |
| sibling `mc24_100043_00300_00399` | 237 | 5 | 只读对照 |
| sibling `mc24_100044_00400_00499` | 235 | 5 | 只读对照 |

低 rank 来自冻结相对 cut，不是缺 probe。Station 覆盖仍是
IFT→1/2/3。

## 命令

```bash
source scripts/setup_environment.sh ml
pytest -q tests/test_tracklet_independent_failure_provenance.py
python scripts/report_tracklet_independent_failure_provenance.py
```

CUDA `True`（Tesla T4）。相关测试包含在 40 passed 中。没有
Condor。

报告：`outputs/tracklet_independent_failure_provenance_audit_v1/`。
记录 HEAD `a1fd01bb5c2c5daf432c13db3b021feb42bdf6d1`。

## 下一阶段

保持 residual DQ monitoring。不得通过删 source 或改阈值恢复
rank 5。不得把这些源并入 cluster-local 战役。
