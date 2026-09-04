# 2026-09-03 Cross-source stable-core identifiable subspace 定义与独立验证 V1

## 任务

条目 68 已冻结
`tracker_only_identifiable_basis_unstable_solve_stopped`：pooled
identifiable rank 为 5，但冻结 `rank_tolerance=0.01` 下出现
source-dependent 第 6 奇异值，identifiable basis 跨 source /
bootstrap 不稳定，三臂和 Frozen-V2 均未打开。本阶段 **不重开**
那次 V1 solve，不改 `rank_tolerance`，不反调 `S`，不删
rank-6 源，不改 selection / route / Frozen V2。

新开独立预注册战役 **Cross-Source Stable-Core Identifiable
Subspace Definition & Independent Validation V1**。目标不是把
rank 强行固定成 5，而是检验是否存在一个在不同 source 中持续
出现、可搬运的 tracker-visible core subspace，同时把
source-dependent 的第 6 marginal mode 明确隔离。

合同：

1. 对每个 source 继续用冻结 `S`、`W` 和 `rank_tolerance=0.01`
   构造原生 `P_s = V_id,s V_id,s^T`。rank=6 的 validation 源
   绝不能人为截成 5D。
2. 共识算子 `M = (1/N) Σ_s P_s`（等权，不是 pair-count 加权
   pool）。
3. 算法、persistence、angle/projector gate、最低 source-support
   fraction、符号约定、dimension 是否自动决定，全部先写入
   config 再冻结。条目 68 的 7 个源只做 **hypothesis
   construction**，不能当作 confirmatory evidence。
4. 独立确认必须用 **source-disjoint、此前未用于 rank-flip
   检查** 的已有 hierarchical V1 FD 库。禁止访问 sealed test。
   看到新 source 结果后禁止改 criterion。
5. 只有独立验证通过后才允许用冻结 `V_core`（不是 V1 pooled
   `V_id`）开三臂。若独立验证失败，冻结否定结论，不得调 rank
   cut、删 source、重选 topology 或改模型追结果。

Cluster-local transfer repair 是并行的纯 provenance 支线，
**不是** 本战役的 gate。不训练、不改 pairwise/route、不增加
2024 r0022 collision-like 统计、不 Newton、不写 geometry /
alignment payload。真实数据仍是 `residual_dq_monitoring_only`。

成功标准不是得到 alignment correction，而是回答：是否存在跨
source 可搬运的 stable tracker-visible core subspace。

## 仓库状态

```text
branch: master
HEAD:   a1fd01b Add survey/metrology IOV provenance and prior-interface feasibility chain (60-65).
ancestor check: a1fd01bb5c2c5daf432c13db3b021feb42bdf6d1 ⊂ HEAD
host:   lxplus909.cern.ch
CUDA:   True (Tesla T4)
```

未访问封存 test。未重新训练 V2/V3/Transformer。未改 pairwise/
route policy。未写 geometry。未反调 `S` 或 `rank_tolerance`。
未截断 rank-6 源。未提交 Condor：独立 validation source 已是
现成 FD 库，交互节点短 SVD / LOSO / bootstrap 足够。

## 答案

冻结
`cross_source_stable_core_independent_validation_fail`。

| 门 | 冻结值 |
| --- | --- |
| hypothesis `core_dimension` | **5**（eigenvalue + persistence 自动决定，不是强制 5） |
| hypothesis LOSO / source-bootstrap 稳定 | **true** |
| independent source-support fraction | **8/11 = 0.727 < 0.80** |
| `independent_validation_pass` | **false** |
| `null_injection_leakage_gate` | **null**（三臂未打开） |
| `mixed_injection_projected_closure` | **null**（三臂未打开） |
| Frozen-V2 unknown-association | **未授权** |
| 真实数据 alignment correction | **未授权** |
| geometry write | **false** |

Hypothesis 7 源上可以构造一个 5 维 consensus core，并把第 6
mode 隔离为 rank-6 源专有。该 core 在 LOSO 和 source-bootstrap
上稳定。独立确认失败：11 个未见过的 FD 源中有 3 个原生 rank
掉到 2 / 3 / 4，预注册 persistence / principal-angle /
projector gate 全部失败。按合同停止，不调阈值、不删失败源、
不把 hypothesis 结果当成 confirmatory evidence。

## 预注册算法（先冻结，后看独立源）

配置
`configs/cross_source_stable_core_identifiable_subspace_v1.yaml`，
SHA256
`4b3b3fba9d46fc8925682566ace4c56ba8b05420359594db4eb34e914973ca50`。

继承条目 68：

- `A = W^{1/2} J S`
- `S = (5, 5, 5, 60, 60, 60, 0.12)` mm / mrad
- `rank_tolerance = 0.01`
- 符号：右向量绝对值最大分量取正
- Frozen V2 SHA256
  `0c85a001677678163512c59bd5ceb6d08bdefaab79c0f4628659a4a8ad766a27`

新冻结规则：

- 每个 source 用 **原生冻结-rank SVD** 的 `P_s`，包括
  `mc24_100047_00150_00199` 的 rank 6，禁止截成 5D
- `M = (1/N) Σ_s P_s`，等权
- core 选择：前缀规则。特征值 ≥ 0.70 **且**
  `q^T P_s q` ≥ 0.85 的 source 支持分数 ≥ 0.80。碰到第一个
  失败就停，不挑非连续 eigenvector，不强制维数 5
- LOSO：同维，最大 principal angle ≤ 15 deg，projector
  Frobenius ≤ 1.0
- source-with-replacement bootstrap 40：最大角 ≤ 20 deg，
  projector Frobenius ≤ 1.2
- 独立确认：支持分数 ≥ 0.80；每源 persistence ≥ 0.85、
  角 ≤ 15 deg、missing projector Frobenius ≤ 0.75
- 三臂只用冻结 `V_core`；Arm-B 注入 core-orthogonal，不是
  V1 pooled null
- `dz` 仍在 7 参数模型中；`dx`–`C_dx` 仍是可能的
  observable / null combination，不固定任一参数

Hypothesis 源（条目 68 看过 rank-flip，只用于构造）：

| split | source | pairs | native rank |
| --- | --- | ---: | ---: |
| train | `mc24_100043_00200_00299` | 232 | 5 |
| train | `mc24_100043_00600_00699` | 221 | 5 |
| train | `mc24_100044_00300_00399` | 240 | 5 |
| validation | `mc24_100047_00000_00049` | 231 | 5 |
| validation | `mc24_100047_00150_00199` | 240 | **6** |
| validation | `mc24_100048_00000_00049` | 216 | 5 |
| validation | `mc24_100048_00150_00199` | 223 | 5 |

独立确认源（同一 iteration-00 FD 语料、文件级互斥、条目 68
未用于 rank-flip 检查；无 sealed test；无 Athena 重跑）：

`mc24_100043_00300_00399`、`00400_00499`、`00500_00599`，
`mc24_100044_00200_00299`、`00400_00499`、`00500_00599`、
`00600_00699`，`mc24_100047_00050_00099`、`00100_00149`，
`mc24_100048_00050_00099`、`00100_00149`。

## Hypothesis consensus

`M` 的特征值
`[1.000000, 1.000000, 0.999995, 0.999964, 0.999558, 0.143304, 3.6e-5]`。

前 5 个特征值都 ≈ 1，source 支持分数都是 1.0，标为 **core**。
第 6 个特征值 0.143 < 0.70，支持分数 1/7 = 0.143，标为
**source_specific_or_null**。第 7 个 ≈ 0。自动 core 维 **5**，
不是强制 5。

第 6 模式 persistence 只在 rank-6 源
`mc24_100047_00150_00199` 上 ≈ 1.0，其余 6 个源 ≈ 0。这就是
条目 68 的 source-dependent 第 6 奇异值：它被隔离，没有进入
core。

Core 主导 scaled 分量（reconstruction-observable 线性组合，
**不是**单独机械参数）：

- 0：`ift_ry_mrad`
- 1：`ift_rx_mrad`
- 2：`ift_dy_mm`
- 3：`ift_rz_mrad`
- 4：`C_dx` 与 `ift_dx_mm` 的组合

Core-orthogonal：

- `ift_dx_mm` 0.815 / `C_dx` 0.578（已记录的 `dx`–`C_dx`
  简并）
- `ift_dz_mm` 1.000（6-DoF map 中不可辨识的 `dz`）

`V_core^⊥ u_hat = 0` 仍是 minimum-norm / gauge，不能写成
“测得 orthogonal mode 为 0”。不得为了改善本次 SVD 而移除
`dz`。

## Hypothesis 稳定性

比较的是子空间，不是 signed vector 元素。

| 审计 | 稳定 | 最大角 | projector Frobenius |
| --- | --- | ---: | ---: |
| LOSO 7 | **是** | 0.433 deg | 0.0107 |
| source-with-replacement 40 | **是** | 1.049 deg | 0.0259 |
| event-bootstrap core persistence 56 | 56/56 通过 | — | — |

LOSO 即使留下 rank-6 源，core 维仍是 5，角 0.032 deg。Source
bootstrap 40 次全部保持 5 维。Event bootstrap 中
`mc24_100047_00150_00199` 仍会出现 5/6 rank-flip
（8 次里 5 次 rank 6、3 次 rank 5），但 `V_core` 对原生
projector 的 persistence 全部通过：rank-flip 被隔离在第 6
mode，不再把 hypothesis core 判为不稳定。这只是 hypothesis
构造，不是独立确认。

## 独立验证（confirmatory）

预注册门槛：11 源支持分数 ≥ 0.80。结果 **8/11 = 0.727**。

| split | source | native rank | 通过 | 最大角 | min persistence | missing F |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| train | `mc24_100043_00300_00399` | 5 | 是 | 0.773 | 0.9998 | 0.019 |
| train | `mc24_100043_00400_00499` | **2** | **否** | 71.0 | 0.0002 | 1.732 |
| train | `mc24_100043_00500_00599` | **3** | **否** | 88.3 | 0.0015 | 1.414 |
| train | `mc24_100044_00200_00299` | **4** | **否** | 18.7 | 0.106 | 1.000 |
| train | `mc24_100044_00400_00499` | 5 | 是 | 1.135 | 0.9997 | 0.028 |
| train | `mc24_100044_00500_00599` | 5 | 是 | 2.932 | 0.9981 | 0.072 |
| train | `mc24_100044_00600_00699` | 5 | 是 | 0.195 | 1.0000 | 0.005 |
| validation | `mc24_100047_00050_00099` | 5 | 是 | 3.005 | 0.9973 | 0.074 |
| validation | `mc24_100047_00100_00149` | 5 | 是 | 2.491 | 0.9991 | 0.062 |
| validation | `mc24_100048_00050_00099` | 5 | 是 | 0.514 | 0.9999 | 0.015 |
| validation | `mc24_100048_00100_00149` | 5 | 是 | 0.365 | 1.0000 | 0.010 |

三个失败源没有被截成 5D。失败机制是原生 identifiable rank
不足，core 有整段方向落在 source projector 外面。不得因此把
`rank_tolerance` 从 0.01 改掉，也不得把失败源从独立集合里删
掉，也不得回头用 hypothesis 7 源充当确认。

三臂 **未打开**。若将来打开，必须用本战役冻结的 `V_core`，
禁止用条目 68 pooled `V_id`。

## 输入 / 命令

```bash
source scripts/setup_environment.sh ml
python -c "import torch; print(torch.cuda.is_available())"
pytest -q tests/test_cross_source_stable_core.py tests/test_tracker_only_identifiable_subspace.py
python scripts/report_cross_source_stable_core.py
```

CUDA `True`（Tesla T4）。相关 26 passed（含条目 68 回归）。
没有 Condor。

## 数值结果（artifact）

| 量 | 值 | 用途 |
| --- | ---: | --- |
| hypothesis core 维 | 5 | 自动，非强制 |
| consensus λ | 1, 1, 1.000, 1.000, 0.9996, **0.143**, 3.6e-5 | 第 6 模式隔离 |
| rank-6 源 | 保留、未截断 | 合同 |
| LOSO 最大角 | 0.433 deg | hypothesis 稳定 |
| source bootstrap 40 | 稳定 / 1.049 deg | hypothesis 稳定 |
| 独立支持分数 | 8/11 = 0.727 | **失败门** |
| 三臂 | skipped | 独立验证失败即停 |
| Frozen-V2 授权 | false | 不打开 unknown-association |
| geometry write | false | 合同 |

报告：`outputs/cross_source_stable_core_identifiable_subspace_v1/`。
配置 SHA256
`4b3b3fba9d46fc8925682566ace4c56ba8b05420359594db4eb34e914973ca50`。
记录 HEAD `a1fd01bb5c2c5daf432c13db3b021feb42bdf6d1`。

## 未解除 / 明确不做

- 条目 68 V1 失败结论继续冻结，本战役没有把它救活。
- 不得把 hypothesis core 叫成“已独立确认的 portable
  tracker-visible subspace”。
- 不得把 core mode 0 叫成“测得 station `ry`”，也不得把
  mode 4 叫成“测得 `C_dx`”。
- 不得把 `rank_tolerance` 从 0.01 改掉来吞掉独立失败源。
- 不得根据本谱反调 `S`，不得删失败源，不得把独立源 rank
  截成 5。
- 不得为改善本次 SVD 移除 `dz`。
- Cluster-local 修复（条目 70）不是本战役 gate。
- 不对真实数据求 correction，不写 geometry。

## 冻结结论

- Frozen V2 / association policy 不变。
- 真实数据继续 `residual_dq_monitoring_only`。
- `geometry_write_allowed=false`；无 alignment payload。
- Survey 继续只做外部交叉检验。
- Hypothesis 7 源上存在一个 LOSO/bootstrap 稳定的 5 维
  tracker-visible core，第 6 mode 被隔离为 rank-6 源专有。
- **该 core 未能通过预注册的独立 source-disjoint 验证**
  （8/11 < 0.80）。因此不存在合同意义上可搬运的 stable
  tracker-visible core subspace。
- 三臂未打开。Frozen-V2 unknown-association 未授权。
- 下一步：保持 DQ monitoring。不得在本战役上追阈值。若以后
  更换 Jacobian、观测或 association control，必须预注册新
  config 和新 workbook 条目。
