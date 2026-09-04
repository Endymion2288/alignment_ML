# Tracker-only identifiable subspace 定义与三臂闭环 V1

Workbook 68 / 2026-09-02。条目 60–67 已把 survey/metrology 冻成
external cross-check，而不是 alignment 输入。本阶段正式转向
tracker-only identifiable-mode alignment：从已验证的物理有限差分
Jacobian 定义 identifiable subspace，做 source-disjoint / bootstrap
子空间稳定性审计，**只有 basis 稳定才打开**三臂线性 MC closure。

不重新训练 V2/V3/Transformer，不改冻结的 pairwise/route policy，不
继续堆 2024 r0022 collision-like tracks，不发明 cosine cut，不根据
residual/cosine 调 selection，不运行 full-parameter Newton，不写
geometry，不产生 alignment payload。Survey prior 保持
`feasibility_only` / `unavailable`。真实数据仍是
`residual_dq_monitoring_only`。第一轮 solver control 是 truth-selected
association。Frozen-V2 unknown-association 闭环本阶段不打开。

## 冻结合同

- 真实数据仍是 `residual_dq_monitoring_only`。
- `geometry_write_allowed=false`。
- 冻结 V2 checkpoint SHA256
  `0c85a001677678163512c59bd5ceb6d08bdefaab79c0f4628659a4a8ad766a27`。
- Survey/metrology 只做外部交叉检验（条目 60–67）。
- 禁止对混合 mm/rad/mrad 单位的裸 Jacobian 做 SVD。必须用预声明
  尺度矩阵 `S` 和已有 WLS residual weight `W` 构造
  `A = W^{1/2} J S`。
- 不得根据 singular values 反调 `S` 或 rank cut。
- Rank tolerance 沿用冻结的 leakage/Fisher 相对阈值 `1e-2`。
- Identifiable mode 是 reconstruction-observable 线性组合，不得重新
  解释成单独的机械 `ry` 或 `C_dx`。
- `V_null^T u_hat = 0` 是 scaled 坐标下的 minimum-norm / gauge
  representative，绝不能写成“测得 null mode 为 0”。
- 逐 physical parameter 的 truth error 不是战役门。
- 若 identifiable basis 不稳定，停止后续 solve 并冻结否定结论。
  不得通过调 scale、rank、selection、route policy 或模型来追结果。

## 判定

`tracker_only_identifiable_basis_unstable_solve_stopped`

| 门 | 值 |
| --- | --- |
| `identifiable_rank` | **5** |
| `identifiable_basis_stable` | **false** |
| `null_injection_leakage_gate` | **null**（三臂未打开） |
| `mixed_injection_projected_closure` | **null**（三臂未打开） |
| Frozen-V2 unknown-association | **未授权** |
| 真实数据 alignment correction | **未授权** |
| geometry write | **false** |

失败原因是冻结相对阈值附近出现 source-dependent 的 rank 变化，
而不是同 rank 源之间出现大的 principal angle。不得因此改阈值。

## Jacobian、单位与尺度

定义 Jacobian 来自条目 44–45 已经产出的 hierarchical V1 物理中心
有限差分库：

- 语料
  `outputs/mc24_ift_hierarchical_v1_iteration00_trainval_physical_v1/`
- residual 四向量 `[rx, ry, rtx, rty]`，truth-selected mode-0
  physical edge（`q_over_p_mode=0`，`min_truth_match_fraction=0.99`）
- `W` 是已有 4×4 pair covariance 的 WLS 权重，不是独立的
  same-cluster 协方差
- 原生单位：平移 / `C_dx` 为 mm，转动为 mrad，与
  `physical_jacobian.COMPONENT_INDEX_AND_PAYLOAD_SCALE` 一致
- 第一轮 loader 只取 FD（reference + 轴向探针），不把 held-out
  physical point 混进 SVD
- 不重跑 Athena

使用 7 个 source-disjoint 文件（不是全部 18 源重载）：

| split | source | pairs | rank | `σ6/σ1` |
| --- | --- | ---: | ---: | ---: |
| train | `mc24_100043_00200_00299` | 232 | 5 | 0.001785 |
| train | `mc24_100043_00600_00699` | 221 | 5 | 0.003005 |
| train | `mc24_100044_00300_00399` | 240 | 5 | 0.002265 |
| validation | `mc24_100047_00000_00049` | 231 | 5 | 0.004125 |
| validation | `mc24_100047_00150_00199` | 240 | **6** | **0.012897** |
| validation | `mc24_100048_00000_00049` | 216 | 5 | 0.001695 |
| validation | `mc24_100048_00150_00199` | 223 | 5 | 0.003179 |

合计 1603 pairs、6412 observations、7 个参数。

预声明 `S`（冻结 `severity_scale`，从不根据 `Σ` 反调）：

| 参数 | 单位 | `S` |
| --- | --- | ---: |
| `ift_dx_mm` | mm | 5 |
| `ift_dy_mm` | mm | 5 |
| `ift_dz_mm` | mm | 5 |
| `ift_rx_mrad` | mrad | 60 |
| `ift_ry_mrad` | mrad | 60 |
| `ift_rz_mrad` | mrad | 60 |
| `C_dx` | mm | 0.12 |

右奇异向量生活在无量纲坐标 `u = S^{-1} θ`。Mode 符号约定：右向量
绝对值最大的分量取正。

## 对 `A = W^{1/2} J S` 的 pooled SVD

Pooled 奇异值

```text
[3729.82, 1747.30, 292.11, 170.81, 164.27, 28.16, 2.52]
```

相对 `σ1`：

```text
[1.000, 0.4685, 0.0783, 0.0458, 0.0440, 0.00755, 0.00067]
```

冻结阈值 `σ_k > 0.01 σ1` ⇒ pooled identifiable rank **5**，null
维数 **2**。WLS 法矩阵在 `rcond=1e-10` 下是 rank 7；那不是战役
rank。战役 rank 是冻结相对 SVD cut 作用在 `A` 上的结果。

Identifiable mode（scaled 组成，`|comp| > 0.15`）：

| mode | 主导 | 组成 | `σ` | `σ/σ1` |
| ---: | --- | --- | ---: | ---: |
| 0 | `ift_ry_mrad` | `ift_ry_mrad` 0.999 | 3729.82 | 1.000 |
| 1 | `ift_rx_mrad` | `ift_rx_mrad` 0.998 | 1747.30 | 0.468 |
| 2 | 混合 | `C_dx` 0.796，`ift_dx_mm` −0.571，`ift_rz_mrad` 0.164 | 292.11 | 0.078 |
| 3 | `ift_dy_mm` | `ift_dy_mm` 0.988 | 170.81 | 0.046 |
| 4 | `ift_rz_mrad` | `ift_rz_mrad` 0.984 | 164.27 | 0.044 |

Null mode：

| mode | 主导 | 组成 | `σ` | `σ/σ1` |
| ---: | --- | ---: | ---: | ---: |
| 0 | `ift_dx_mm` | `ift_dx_mm` 0.815，`C_dx` 0.575 | 28.16 | 0.00755 |
| 1 | `ift_dz_mm` | `ift_dz_mm` 1.000 | 2.52 | 0.00067 |

这些是 reconstruction-observable 线性组合。Mode 0 不是“station
`ry`”。Mode 2 不是“`C_dx`”。Null mode 0 是 hierarchical leakage
已经记录的 `dx`–`C_dx` 简并；null mode 1 是 6-DoF map 中不可辨识
的 `dz`。两个 null 方向都不是“测得为 0”。

## 子空间稳定性

稳定性比较的是 identifiable **子空间** 的 principal angles 和
projector Frobenius 距离，不要求单个奇异向量元素符号一致。Rank
变化直接失败预声明的 same-rank gate。

| 审计 | 稳定 | n | 同 rank 最大角 | 备注 |
| --- | --- | ---: | ---: | --- |
| source-disjoint vs pooled | **false** | 7 | 4.28 deg | `validation:mc24_100047_00150_00199` rank 6（`σ6/σ1=0.0129`） |
| train vs validation（pooled） | true | 1 | 5.89 deg | projector 0.145 |
| complete-truth-route topology | true | 1 | 0.07 deg | projector 0.0017 |
| event bootstrap | **false** | 40 | 3.88 deg | 7/40 翻到 rank 6 |
| event half-split | **false** | 24 halves | 3.90 deg | 6/24 翻到 rank 6 |

同 rank 源大约 3.5–4.3 deg / projector 0.09–0.11。不稳定来自冻结
`1e-2` 阈值紧挨着一个弱 source-dependent 的第 6 奇异值（一个
validation 源上约 `σ1` 的 1.3%，其余源 ≪1%）。这是 rank 定义
不稳定，不是五维子空间的大转动。

因为 basis 不稳定，三臂 solve **未打开**。Arm-A / Arm-B / Arm-C
没有在这个 Jacobian 上评估。合成 unit test 仍然证明求解器数学：
先解 mode amplitude `a`，再 `Δθ = S V_id a`；混合注入只恢复
`P_id q_truth`；`V_null^T u_hat = 0` 是 gauge。

## Cluster-local 诊断（不是战役门）

对真实 cluster-local `r_u` 的软件 FD `{station_dx, station_ry,
C_dx}` 只作为诊断。ry 列从 per-radian 换成 per-mrad，再用冻结的
cluster-local scale map 构造 `A = W^{1/2} J S`。禁止裸混合单位
SVD。

现场诊断被跳过：`cluster_local_reference_or_transfer_unavailable`。
Reference run 14973 可以加载；transfer run 14974 在 join 条目 57
的 cluster dump 后测量数为 0（该 dump 是 r14973 可行性文件）。
这不进入四个战役门，也不是真实数据 alignment solve。

## 明确没做的事

- 没有 full-parameter Newton。
- 没有 Frozen-V2 unknown-association 闭环。
- 没有真实数据 alignment correction。
- 没有 geometry / POOL / COOL / alignment payload。
- 没有反调 `S`、rank tolerance、selection、route policy 或模型。
- 没有访问封存 test。
- 没有继续堆 2024 r0022 collision-like tracks。
- 条目 66–67 的 survey 数字不是 alignment 输入。

## 报告

`outputs/tracker_only_identifiable_subspace_three_arm_closure_v1/`

配置 SHA256
`1657019a2c0746835d490fcf58ce91b70b1cb846f73bc29e2ad69e0b5a2f7237`。
记录的 HEAD `a1fd01bb5c2c5daf432c13db3b021feb42bdf6d1`。
12 个 unit/regression 测试通过。CUDA `True`（Tesla T4）。短 SVD /
bootstrap / 小型 closure 交互运行；没有 Condor。

## 下一阶段

继续 residual DQ monitoring。在没有稳定 identifiable subspace 之前，
**不授权** Frozen-V2 unknown-association。以后如果更换 Jacobian
（不同 residual、不同 association control，或真正的新观测量），必须
预注册新 config 和新 workbook 条目。不得把本战役的
`rank_tolerance` 从 `0.01` 改成 `0.013` 来吞掉
`mc24_100047_00150_00199`。
