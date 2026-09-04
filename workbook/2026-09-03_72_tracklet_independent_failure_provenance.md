# 2026-09-03 Tracklet 独立验证失败源只读 provenance 审计 V1

## 任务

条目 69 冻结 `cross_source_stable_core_independent_validation_fail`：
11 个 source-disjoint 确认源里有 3 个原生 identifiable rank 为
2 / 3 / 4，独立支持分数 8/11 = 0.727 < 0.80。本阶段 **不重开**
那次 stable-core 战役，不改 `rank_tolerance=0.01`，不反调 `S`，
不删失败 source，不强制 rank 5，不把这些源当作 cluster-local
confirmatory 样本。

只读审计三个失败源：

- `mc24_100043_00400_00499`（rank 2）
- `mc24_100043_00500_00599`（rank 3）
- `mc24_100044_00200_00299`（rank 4）

预注册 sibling 对照（条目 69 已通过、文件级互斥，不是新的
confirmatory core）：

- `mc24_100043_00300_00399`
- `mc24_100044_00400_00499`

目标是区分 **正常 physics / coverage loss** 与 **明确 artifact /
pipeline corruption**。绝不能为了恢复 rank 5 而删 source、重选
population 或改 threshold。

不训练、不改 pairwise/route、不 Newton、不写 geometry。真实
数据仍是 `residual_dq_monitoring_only`。封存 test 不打开。

## 仓库状态

```text
branch: master
HEAD:   a1fd01b Add survey/metrology IOV provenance and prior-interface feasibility chain (60-65).
ancestor check: a1fd01bb5c2c5daf432c13db3b021feb42bdf6d1 ⊂ HEAD
host:   lxplus909.cern.ch
CUDA:   True (Tesla T4)
```

Jacobian 语料仍是 hierarchical V1 iteration-00 physical FD 库。
未重跑 Athena。未提交 Condor。未访问 sealed test。

## 答案

冻结
`tracklet_independent_failure_provenance_audit_complete_sources_not_dropped`。

| 门 | 冻结值 |
| --- | --- |
| 总体分类 | **normal_physics_or_coverage_loss** |
| 失败源被删除 | **false** |
| rank 被强制为 5 | **false** |
| stable-core criterion 被重定义 | **false** |
| 用作 cluster-local 确认 | **false** |
| 条目 69 结论 | **继续冻结** |
| geometry write | **false** |

三个失败源的 `+/-` FD probe 完整、plus/minus population 对齐、
残差有限、没有近零 Jacobian 列、没有混入 held-out physical
points、没有打开 sealed test。原生 rank 2 / 3 / 4 被复现。
唯一共同 artifact flag 是 `ill_conditioned_pair_covariance`，
但两个 rank-5 sibling 对照也有同一 flag，因此它不能把失败源
判成 pipeline corruption。

失败机制是正常 physics / coverage 导致的原生 identifiable rank
不足，不是可修复的 probe / join / payload 损坏。按合同不删
source、不追阈值。

## 预注册合同

配置
`configs/tracklet_independent_failure_provenance_audit_v1.yaml`，
SHA256
`b73e35148e7ee8b31a579db741d1a9d36a0571f2df80215e3b16f10f432ee410`。

- 继承条目 69
  `cross_source_stable_core_independent_validation_fail`
- `A = W^{1/2} J S`，`S = (5, 5, 5, 60, 60, 60, 0.12)`，
  `rank_tolerance = 0.01`
- 原生冻结-rank SVD，禁止截成 5D
- sibling 只作只读对照，不是新 confirmatory set
- 严重 artifact：缺失 FD probe、plus/minus 不对齐、非有限
  残差、访问 sealed test、held-out 混入 SVD。这些才判
  `pipeline_or_artifact_corruption`
- `ill_conditioned_pair_covariance` 是警告，不是单独的
  corruption 判决；必须对照 sibling

## 失败源

| source | pairs | 官方 rank | probe +/- | population 对齐 | 近零列 | 分类 |
| --- | ---: | ---: | --- | --- | --- | --- |
| `mc24_100043_00400_00499` | 240 | **2** | 完整 | 是 | 无 | coverage loss |
| `mc24_100043_00500_00599` | 234 | **3** | 完整 | 是 | 无 | coverage loss |
| `mc24_100044_00200_00299` | 234 | **4** | 完整 | 是 | 无 | coverage loss |

奇异值（冻结 `A`）：

- rank 2：`8748, 694, 65.0, 56.3, 25.3, 3.87, 0.76`
  （σ3 / σ1 = 0.0074 < 0.01）
- rank 3：`11022, 640, 335, 58.8, 23.2, 1.47, 1.36`
  （σ4 / σ1 = 0.0053 < 0.01）
- rank 4：`4349, 633, 60.0, 47.2, 37.7, 2.21, 0.36`
  （σ5 / σ1 = 0.0087 < 0.01）

Normal-matrix 数值 rank 仍是 7，没有整列近零。低官方 rank 来自
冻结相对 cut，不是缺 probe。Station 覆盖仍是 IFT→1/2/3。
`held_out_physical_points_loaded=false`，
`test_data_accessed=false`。

## Sibling 对照

| source | pairs | 官方 rank | 同样的 cov-condition flag |
| --- | ---: | ---: | --- |
| `mc24_100043_00300_00399` | 237 | **5** | 是 |
| `mc24_100044_00400_00499` | 235 | **5** | 是 |

Pair covariance 病态在本 FD 语料里普遍，包括已通过条目 69 的
源。它不是把 rank 2/3/4 判成 pipeline corruption 的证据。
Sibling 不得被解释成“新的 portable core 确认”。

## 输入 / 命令

```bash
source scripts/setup_environment.sh ml
pytest -q tests/test_tracklet_independent_failure_provenance.py
python scripts/report_tracklet_independent_failure_provenance.py
```

CUDA `True`（Tesla T4）。相关测试包含在 40 passed 中。没有
Condor。

报告：`outputs/tracklet_independent_failure_provenance_audit_v1/`。
记录 HEAD `a1fd01bb5c2c5daf432c13db3b021feb42bdf6d1`。

## 未解除 / 明确不做

- 条目 69 独立验证失败继续冻结。本审计没有把它救活。
- 不得删除这三个源，不得把它们的 projector 截成 5D。
- 不得因为 sibling 是 rank 5 就重开 stable-core 或把它们并入
  cluster-local 门。
- 不得根据本谱反调 `S` 或 `rank_tolerance`。
- 条目 71 的 cluster-local 战役不得把这些失败源当成成功确认
  样本。

## 冻结结论

- Frozen V2 / association policy 不变。
- 真实数据继续 `residual_dq_monitoring_only`。
- `geometry_write_allowed=false`。
- 条目 69 的 rank 2/3/4 失败是正常 physics / coverage loss，
  不是 probe / payload / join 损坏。
- 停止通过删 source 或改阈值去恢复 rank 5。
