# Cross-source stable-core identifiable subspace 定义与独立验证 V1

Workbook 69 / 2026-09-03。条目 68 继续冻结
`tracker_only_identifiable_basis_unstable_solve_stopped`。本战役
不重开那次 V1 solve，不改 `rank_tolerance=0.01`，不反调物理
尺度 `S`，不删 rank-6 源，不改 selection / route / Frozen V2。

问题不是“把 rank 强行固定成 5”，而是：是否存在一个在不同
source 中持续出现、可搬运的 tracker-visible core subspace，同时
把 source-dependent 的第 6 模式明确隔离。

不重新训练 V2/V3/Transformer，不改冻结的 pairwise/route policy，
不继续堆 2024 r0022 collision-like tracks，不发明 cosine cut，不
根据 residual/cosine 调 selection，不运行 full-parameter Newton，
不写 geometry，不产生 alignment payload。Survey/metrology 仍只作
external cross-check。真实数据仍是
`residual_dq_monitoring_only`。Cluster-local transfer 修复是并行
支线，不是本战役的门。

## 冻结合同

- 真实数据仍是 `residual_dq_monitoring_only`。
- `geometry_write_allowed=false`。
- 冻结 V2 checkpoint SHA256
  `0c85a001677678163512c59bd5ceb6d08bdefaab79c0f4628659a4a8ad766a27`。
- 继承条目 68
  `tracker_only_identifiable_basis_unstable_solve_stopped`。
- 必须用冻结 `S = (5, 5, 5, 60, 60, 60, 0.12)` mm/mrad 和冻结
  `rank_tolerance=0.01` 构造 `A = W^{1/2} J S`。禁止对裸混合
  单位 Jacobian 做 SVD。
- 每个 source projector 使用 **原生冻结-rank SVD**，包括
  rank-6 的 `mc24_100047_00150_00199`。该源不得截成 5D，也不得
  删除。
- 条目 68 的 7 个源只做 **hypothesis construction**，不能当作
  confirmatory evidence。
- Core 维数由冻结的特征值与 persistence 门槛自动决定，不强制
  为 5。
- 独立确认使用已经产出、文件级互斥、条目 68 rank-flip 检查未
  用过的 hierarchical iteration-00 FD 库。禁止访问 sealed test。
  不重跑 Athena。看到新 source 后不得改 criterion。
- 若打开三臂，必须使用冻结 `V_core`，不得使用条目 68 pooled
  `V_id`。Arm-B 注入 core-orthogonal。
- `dz` 仍留在 7 参数模型中；移除它必须新开 config，理由来自
  既有 6-DoF 不可辨识结论，而不是本次 SVD。`dx`–`C_dx` 仍是
  可能的 observable / null combination。
- 若独立验证失败，冻结否定结论。不得通过调 scale、rank、
  selection、source 集合或模型来追结果。

## 判定

`cross_source_stable_core_independent_validation_fail`

| 门 | 值 |
| --- | --- |
| hypothesis `core_dimension` | **5**（自动，非强制） |
| hypothesis LOSO / source-bootstrap 稳定 | **true** |
| independent source-support fraction | **8/11 = 0.727 < 0.80** |
| `independent_validation_pass` | **false** |
| `null_injection_leakage_gate` | **null**（三臂未打开） |
| `mixed_injection_projected_closure` | **null**（三臂未打开） |
| Frozen-V2 unknown-association | **未授权** |
| 真实数据 alignment correction | **未授权** |
| geometry write | **false** |

Hypothesis 7 源上可以构造 5 维 consensus core，并把第 6 模式
隔离为 rank-6 源专有。该 core 在 LOSO 和 source-bootstrap 上
稳定。独立确认失败：3 个未见过的 FD 源原生 rank 为 2 / 3 /
4，预注册 persistence / principal-angle / projector 门全部失败。
不得追阈值。

## 预注册算法

配置
`configs/cross_source_stable_core_identifiable_subspace_v1.yaml`，
SHA256
`4b3b3fba9d46fc8925682566ace4c56ba8b05420359594db4eb34e914973ca50`。

- 每源 `P_s = V_id,s V_id,s^T`，来自 `A = W^{1/2} J S` 的原生
  冻结-rank SVD。
- 等权共识 `M = (1/N) Σ_s P_s`。
- 对称特征值分解，降序；符号约定为右向量绝对值最大分量取正。
- 前缀 core：特征值 ≥ 0.70 **且** `q^T P_s q` ≥ 0.85 的 source
  支持分数 ≥ 0.80。碰到第一个失败就停，不挑非连续
  eigenvector。
- LOSO：同维，最大 principal angle ≤ 15 deg，projector
  Frobenius ≤ 1.0。
- source-with-replacement bootstrap 40：最大角 ≤ 20 deg，
  projector Frobenius ≤ 1.2。
- 独立门：支持分数 ≥ 0.80；每源 persistence ≥ 0.85、角 ≤ 15
  deg、missing projector Frobenius ≤ 0.75。

Hypothesis 源（条目 68 rank-flip 集合，只用于构造）与独立确认
源见 workbook 69。独立源与 hypothesis 源文件级互斥，未打开
sealed test。

## Hypothesis consensus

`M` 的特征值
`[1.000000, 1.000000, 0.999995, 0.999964, 0.999558, 0.143304, 3.6e-5]`。

前 5 个特征值 ≈ 1，source 支持分数 1.0，标为 **core**。第 6
个特征值 0.143 < 0.70，支持分数 1/7，标为
**source_specific_or_null**。自动 core 维 **5**。第 6 模式
persistence 只在 `mc24_100047_00150_00199` 上 ≈ 1，其余源 ≈ 0。
这就是条目 68 的 source-dependent 第 6 奇异值，现已隔离出
core。

Core 主导 scaled 分量是 reconstruction-observable 线性组合，
**不是**单独机械参数：`ift_ry`、`ift_rx`、`ift_dy`、`ift_rz`、
以及 `C_dx` 与 `ift_dx` 的组合。Core-orthogonal 是已记录的
`dx`–`C_dx` 简并和不可辨识的 `dz`。`V_core^⊥ u_hat = 0` 仍是
gauge，不能写成“测得 orthogonal 为 0”。

## Hypothesis 稳定性

比较的是子空间，不是 signed vector 元素。

| 审计 | 稳定 | 最大角 | projector Frobenius |
| --- | --- | ---: | ---: |
| LOSO 7 | **是** | 0.433 deg | 0.0107 |
| source-with-replacement 40 | **是** | 1.049 deg | 0.0259 |
| event-bootstrap core persistence 56 | 56/56 通过 | — | — |

留下 rank-6 源时 core 维仍是 5（0.032 deg）。40 次 source
bootstrap 全部保持 5 维。`mc24_100047_00150_00199` 的 event
bootstrap 仍会 5↔6 rank-flip（8 次里 5 次 rank 6），但
`V_core` persistence 全部通过：flip 被隔离在第 6 模式。这只是
hypothesis 构造，不是独立确认。

## 独立验证

预注册支持分数门槛 0.80。结果 **8/11 = 0.727**。失败源原生
rank 为 2 / 3 / 4，core 有整段方向落在 source projector 外面。
这些源没有被截成 5D。不得改 `rank_tolerance`、删失败源，或不
得回头把 hypothesis 7 源当成确认。

三臂 **未打开**。若将来打开，必须用本战役冻结的 `V_core`。

## 命令

```bash
source scripts/setup_environment.sh ml
python -c "import torch; print(torch.cuda.is_available())"
pytest -q tests/test_cross_source_stable_core.py tests/test_tracker_only_identifiable_subspace.py
python scripts/report_cross_source_stable_core.py
```

CUDA `True`（Tesla T4）。相关 26 个测试通过。没有 Condor：独立
源已是现成 FD 库。

报告：`outputs/cross_source_stable_core_identifiable_subspace_v1/`。
记录 HEAD `a1fd01bb5c2c5daf432c13db3b021feb42bdf6d1`。

## 下一阶段

保持 residual DQ monitoring。不授权 Frozen-V2
unknown-association。Hypothesis 上的 5 维 core **不是**预注册
门下可独立搬运的 tracker-visible subspace。不得在本战役上移动
`rank_tolerance`、反调 `S`、删源或强制 rank 5。以后若更换
Jacobian、观测或 association control，必须预注册新 config 和新
workbook 条目。
