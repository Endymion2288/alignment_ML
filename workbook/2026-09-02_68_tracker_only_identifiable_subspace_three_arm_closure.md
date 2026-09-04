# 2026-09-02 Tracker-only identifiable subspace 定义与三臂闭环 V1

## 任务

条目 60–67 已把 survey/metrology 冻成 external cross-check，官方槽位
保持 `feasibility_only` / `unavailable`，Stations `ry` 软件合同是绕
FASER 原点的全局左乘，但没有可 ingest 的测量协方差和 IOV。现有
association policy / V2 保持不变；真实数据仍是
`residual_dq_monitoring_only`；`geometry_write_allowed=false`。

本阶段正式转向 **tracker-only identifiable-mode alignment**：

1. 从已验证的 hierarchical V1 物理有限差分 Jacobian 出发，确认参数
   定义、单位、权重和冻结 physical scales。
2. 用预声明 `S` 和 residual weight `W` 构造无量纲加权矩阵
   `A = W^{1/2} J S`，沿用冻结 rank tolerance 做 SVD，分解
   `V_id` / `V_null`，冻结 `P_id = V_id V_id^T`。
3. 做 source-disjoint / bootstrap 子空间稳定性审计（principal
   angles / projector distance，不是单个奇异向量元素一致）。
4. **只有 basis 稳定**才打开预注册三臂 MC closure：Arm-A 只注入
   identifiable；Arm-B 只注入 null；Arm-C 混合，只要求
   `q_hat ≈ P_id q_truth`。
5. 冻结四个核心结论：`identifiable_rank`、
   `identifiable_basis_stable`、`null_injection_leakage_gate`、
   `mixed_injection_projected_closure`。

不训练、不改 pairwise/route policy、不增加 2024 r0022 collision-like
统计、不根据 residual/cosine 调 selection、不 Newton、不写 geometry /
alignment payload。禁止对裸混合单位 Jacobian 做 SVD，禁止根据
singular values 反调 `S` 或 rank cut。第一轮 solver control 是
truth-selected association；Frozen-V2 unknown-association 本阶段不
打开。不对真实数据求 alignment correction。

成功标准不是“得到一组漂亮的 station 参数”，而是严格证明 tracker
data 在什么低维 subspace 内可以稳定求 alignment，并证明不可辨识
方向不会被 solver 伪装成物理 correction。若三臂或跨 source basis
stability 不通过，直接冻结否定结论。

## 仓库状态

```text
branch: master
HEAD:   a1fd01b Add survey/metrology IOV provenance and prior-interface feasibility chain (60-65).
ancestor check: a1fd01bb5c2c5daf432c13db3b021feb42bdf6d1 ⊂ HEAD
host:   lxplus908.cern.ch
CUDA:   True (Tesla T4)
```

未访问封存 test。未重新训练 V2/V3/Transformer。未改 pairwise/route
policy。未写 geometry。

## 答案

冻结
`tracker_only_identifiable_basis_unstable_solve_stopped`。

四个核心结论：

| 门 | 冻结值 |
| --- | --- |
| `identifiable_rank` | **5** |
| `identifiable_basis_stable` | **false** |
| `null_injection_leakage_gate` | **null**（三臂未打开） |
| `mixed_injection_projected_closure` | **null**（三臂未打开） |

Frozen-V2 unknown-association **未授权**。真实数据 alignment
correction **未授权**。`geometry_write_allowed=false`。

失败不是同 rank 源之间出现大角度，而是冻结相对阈值 `1e-2` 紧挨着
一个 source-dependent 的第 6 奇异值：`validation:mc24_100047_00150_00199`
的 `σ6/σ1=0.0129 > 0.01`，因此 rank 变成 6；其余 6 个源 rank 5。
Event bootstrap 40 次中 7 次翻到 rank 6；event half-split 24 个
half 中 6 次翻到 rank 6。同 rank 比较的最大 principal angle 只有
约 4.3 deg（source）/ 5.9 deg（train vs validation）。按合同停止
solve，不调 `S`、rank、selection、route 或模型。

## 参数、单位、权重、尺度

定义 Jacobian 来自条目 44–45 已验证的 hierarchical V1 物理中心有限
差分库
`outputs/mc24_ift_hierarchical_v1_iteration00_trainval_physical_v1/`。
观测是 truth-selected mode-0 pair residual 四向量
`[rx, ry, rtx, rty]`；`W` 是既有 4×4 pair covariance WLS 权重。
原生单位：平移 / `C_dx` 为 mm，转动为 mrad。第一轮 loader 只取
FD（reference + 轴向探针）。

7 参数：`ift_dx/dy/dz_mm`、`ift_rx/ry/rz_mrad`、`C_dx`。冻结
`S = (5, 5, 5, 60, 60, 60, 0.12)`，对应冻结 `severity_scale`，
在 SVD 之前声明，从不根据奇异值反调。Rank tolerance `0.01` 是
条目 57–61 的 `LEAKAGE_RANK_RELATIVE_TOLERANCE`。Mode 符号约定：
右向量绝对值最大分量取正。

7 个 source-disjoint 文件，合计 1603 pairs / 6412 observations：

| split | source | pairs | rank | `σ6/σ1` |
| --- | --- | ---: | ---: | ---: |
| train | `mc24_100043_00200_00299` | 232 | 5 | 0.001785 |
| train | `mc24_100043_00600_00699` | 221 | 5 | 0.003005 |
| train | `mc24_100044_00300_00399` | 240 | 5 | 0.002265 |
| validation | `mc24_100047_00000_00049` | 231 | 5 | 0.004125 |
| validation | `mc24_100047_00150_00199` | 240 | **6** | **0.012897** |
| validation | `mc24_100048_00000_00049` | 216 | 5 | 0.001695 |
| validation | `mc24_100048_00150_00199` | 223 | 5 | 0.003179 |

## Pooled SVD

Pooled 奇异值
`[3729.82, 1747.30, 292.11, 170.81, 164.27, 28.16, 2.52]`，
相对 `σ1` 为
`[1.000, 0.4685, 0.0783, 0.0458, 0.0440, 0.00755, 0.00067]`。
冻结 cut ⇒ rank 5，null 维 2。WLS 法矩阵 `rcond=1e-10` 是 rank 7，
那不是战役 rank。

Identifiable mode（scaled，`|comp|>0.15`）：

- 0：`ift_ry_mrad` 0.999
- 1：`ift_rx_mrad` 0.998
- 2：`C_dx` 0.796 / `ift_dx_mm` −0.571 / `ift_rz_mrad` 0.164
- 3：`ift_dy_mm` 0.988
- 4：`ift_rz_mrad` 0.984

Null mode：

- 0：`ift_dx_mm` 0.815 / `C_dx` 0.575
- 1：`ift_dz_mm` 1.000

这些是 reconstruction-observable 线性组合，**不是**单独的机械
`ry` 或 `C_dx`。Null 0 是已记录的 `dx`–`C_dx` 简并；null 1 是
6-DoF map 中不可辨识的 `dz`。`V_null^T u_hat = 0` 只能解释为
gauge，不能写成“测得 null 为 0”。

## 稳定性审计

比较的是 identifiable **子空间**，不是 signed vector 元素。

| 审计 | 稳定 | 最大同 rank 角 | 失败机制 |
| --- | --- | ---: | --- |
| source-disjoint vs pooled | **否** | 4.28 deg | 一个 val 源 rank 6 |
| train vs validation | 是 | 5.89 deg | projector 0.145 |
| complete-truth-route topology | 是 | 0.07 deg | projector 0.0017 |
| event bootstrap 40 | **否** | 3.88 deg | 7/40 rank-flip |
| event half-split 24 | **否** | 3.90 deg | 6/24 rank-flip |

同 rank 源 projector Frobenius 约 0.09–0.11。不稳定是阈值贴着
第 6 奇异值，不是五维子空间大转动。按合同 **三臂未打开**。

合成 unit test 仍证明求解器：只解 `a` 再 `Δθ = S V_id a`；Arm-B
不产生显著 identifiable fake；Arm-C 只恢复 `P_id q_truth`；逐
physical parameter truth error 不是 gate。

## Cluster-local 诊断（不是门）

现场诊断跳过：`cluster_local_reference_or_transfer_unavailable`。
Reference r14973 可加载；transfer r14974 join 条目 57 dump
（r14973 可行性文件）后测量数为 0。ry 列已按 per-mrad 转换，
裸混合单位 SVD 被拒绝。不进入四个战役门，不是真实数据 solve。

## 输入 / 命令

- 配置：`configs/tracker_only_identifiable_subspace_three_arm_closure_v1.yaml`
- Jacobian：hierarchical V1 iteration-00 train/val 物理 FD 库
- 环境：`source scripts/setup_environment.sh ml`（`LCG_110_cuda`）

```bash
source scripts/setup_environment.sh ml
python -c "import torch; print(torch.cuda.is_available())"
pytest -q tests/test_tracker_only_identifiable_subspace.py
python scripts/report_tracker_only_identifiable_subspace.py
```

CUDA `True`（Tesla T4）。12 passed。没有提交 Condor：短 SVD /
bootstrap / unit test / 小型 closure，交互节点足够。

## 数值结果（artifact）

| 量 | 值 | 用途 |
| --- | ---: | --- |
| pooled identifiable rank | 5 | 冻结 `identifiable_rank` |
| null 维 | 2 | `dx`–`C_dx` 与 `dz` |
| pooled `σ` | 3729.82 … 2.52 | `A` 的谱，不反调 `S` |
| 失败源 `σ6/σ1` | 0.012897 | rank 变化，不改 cut |
| source 同 rank 最大角 | 4.28 deg | 子空间比较 |
| train–val 最大角 | 5.89 deg | 子空间比较 |
| bootstrap rank-flip | 7/40 | basis 不稳定 |
| half-split rank-flip | 6/24 | basis 不稳定 |
| 三臂 | skipped | basis 不稳定即停 |
| Frozen-V2 授权 | false | 不打开 unknown-association |
| geometry write | false | 合同 |

配置 SHA256 `1657019a2c0746835d490fcf58ce91b70b1cb846f73bc29e2ad69e0b5a2f7237`；
记录 HEAD `a1fd01bb5c2c5daf432c13db3b021feb42bdf6d1`。

报告：`outputs/tracker_only_identifiable_subspace_three_arm_closure_v1/`。

## 未解除 / 明确不做

- 条目 66–67 的 survey 数字仍不是 alignment 输入。
- 不得把 pooled mode 0 叫成“测得 station `ry`”，也不得把 mode 2
  叫成“测得 `C_dx`”。
- 不得把 `rank_tolerance` 从 `0.01` 改成 `0.013` 来吞掉失败源。
- 不得根据本谱反调 `S`。
- Cluster-local 诊断缺 transfer join，不升级为战役门。
- 不对真实数据求 correction，不写 geometry。

## 冻结结论

- Frozen V2 / association policy 不变。
- 真实数据继续 `residual_dq_monitoring_only`。
- `geometry_write_allowed=false`；无 alignment payload。
- Survey 继续只做外部交叉检验。
- Tracker-only identifiable rank 在本 Jacobian 上是 5，但
  **identifiable basis 跨 source / bootstrap 不稳定**，三臂未打开。
- 下一步：保持 DQ monitoring。在没有稳定 identifiable subspace
  之前，不授权 Frozen-V2 unknown-association。若以后更换 Jacobian
  或观测，必须预注册新 config 和新 workbook 条目，不得在本战役上
  追阈值。
