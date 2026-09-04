# 2026-09-03 Rigid-Station-Only Tracker Alignment Identifiability V1

## 任务

条目 68 冻结 `tracker_only_identifiable_basis_unstable_solve_stopped`
（7D tracklet-level identifiable subspace 跨 source / bootstrap
不稳定）。条目 69 冻结
`cross_source_stable_core_independent_validation_fail`（独立
source-disjoint 确认 8/11 = 0.727 < 0.80）。条目 70 只修复
cluster-local exact join provenance。条目 71 冻结
`cluster_local_observable_not_cross_run_portable`（exact join 下
slope tertile 仍把 rank 3→2）。条目 72 把条目 69 的 rank 2/3/4
源分类为正常 physics / coverage loss，不是 pipeline corruption。

**停止拯救 7D / cluster-local / stable-core observable。** 不得改
`rank_tolerance=0.01`，不得根据谱反调 `S`，不得删失败 source，不得
更换 residual、重定义 stable-core、根据 slope/residual/cosine 重选
population，也不得通过更复杂 ML、route 或 association policy 追
rank。

本阶段新开独立预注册战役 **Rigid-Station-Only Tracker Alignment
Identifiability & Closure V1**。物理模型边界在看结果前写死：

1. tracker fit 只允许 station rigid-body `dx, dy, rx, ry, rz`
   五个自由度。
2. `dz` 继续继承此前冻结的 track-unidentifiable /
   survey-or-gauge-constrained 结论，不交给 tracks。
3. 内部 plane/module geometry 和 `C_dx` 不再作为本次 tracker
   alignment 的自由参数，固定为当前 reconstruction geometry /
   metrology state。
4. 固定 `C_dx` 只是 rigid-station 模型边界，**不是**“测得
   `C_dx=0`”、不是“survey 已验证当前 `C_dx` 正确”，也不得把旧
   survey central value 强行写入 fit。Survey/metrology 继续只做
   external cross-check。

不得从旧 7D SVD 删掉 `dz/C_dx` 两列后宣布成功。必须用
`only_parameters` 重新构造原生 5 列 `A = W^{1/2} J S`。pooled
rank=5 不能代替 portability。所有预声明 source-disjoint 和
slope/topology coverage gate 都必须通过。若某个预声明 source 或
coverage slice 掉 rank，只做 provenance / physics diagnosis；不得
为了恢复 rank 5 而删 source、放宽 cut、调整 `S` 或选择“好看的”
population。

当前阶段 `three_arm_authorized=false`、
`frozen_v2_alignment_loop_authorized=false`、
`real_data_correction_authorized=false`、
`geometry_write_allowed=false`、
`official_conditions_write_allowed=false`。只有 5DoF
identifiability 在 source-disjoint 和 coverage gates 上稳定通过后，
才允许打开第一阶段 truth-selected joint closure。

不训练、不改 pairwise/route、不增加 2024 r0022 collision-like
统计、不 Newton、不写 geometry / alignment payload。真实数据仍是
`residual_dq_monitoring_only`。

成功标准不是得到一组漂亮的 station 参数，而是严格证明或否定：
**固定内部 geometry 后，station rigid-body 5DoF 是否在当前真实
FASER-like track topology 下具有跨 source、跨 slope coverage 的
稳定 tracker-only identifiability。** 若仍失败，直接冻结否定
结论，不得继续靠删 DoF 或调阈值救结果。

## 仓库状态

```text
branch: master
HEAD:   a1fd01b Add survey/metrology IOV provenance and prior-interface feasibility chain (60-65).
ancestor check: a1fd01bb5c2c5daf432c13db3b021feb42bdf6d1 ⊂ HEAD
host:   lxplus909.cern.ch
CUDA:   True (Tesla T4)
```

Jacobian 语料仍是 hierarchical V1 iteration-00 physical FD 库。
未重跑 Athena。未提交 Condor：18 源原生 5 列 Jacobian、SVD、
slope tertile、bootstrap 在交互节点约 96 s 完成。未访问封存
test。未重新训练 V2/V3/Transformer。

## 答案

冻结 `rigid_station_five_dof_not_source_or_coverage_portable`。

| 门 | 冻结值 |
| --- | --- |
| Jacobian 有效（probe / population / 有限响应 / 近零列） | **true** |
| pooled identifiable rank | **5**（不是 portability） |
| 每个预声明 source rank=5 | **false**（14/18） |
| source-disjoint 子空间稳定 | **false** |
| train vs validation 子空间 | **true**（同 rank 且角度 ~0） |
| slope tertile 无 rank drop | **false** |
| complete-route topology rank=5 | **false** |
| event bootstrap 稳定 | **false** |
| 失败分类 | **normal_physics_or_coverage_loss** |
| 源被删除 | **false** |
| 从 7D SVD 删列 | **false** |
| 三臂 / Frozen-V2 / 真实数据 correction | **未授权** |
| geometry write | **false** |

固定内部 geometry 后，当前 tracker data **不足以**跨 source /
coverage 稳定确定 station rigid-body 5DoF。失败不是 Jacobian
构造 artifact：全部 18 源 `+/-` probe 完整、plus/minus population
对齐、残差有限、没有近零列、`source_slope` 无缺失、没有混入
held-out physical points、没有打开 sealed test。原生 5 列
`A = W^{1/2} J S` 由 `only_parameters` 重建，不是旧 7D SVD 删列。

Pooled rank=5 不能代替 portability。4 个预声明 source 的原生
rank 低于 5；9 个 source 的 slope tertile 掉 rank；4 个 source 的
complete-route topology 也掉 rank。按合同停止，不调 `S`、rank、
selection、route 或模型，不删 source。下一步应转向物理上不同的
track coverage 或明确 gauge / external constraint，而不是继续
修改算法或再删 DoF。

## 预注册合同

配置
`configs/rigid_station_only_tracker_alignment_identifiability_v1.yaml`，
SHA256
`80b5ebebe71e6d2bb20604c8593b97540cd998cbafee86e94d201dd6b00719ba`。

- 参数顺序：`ift_dx_mm, ift_dy_mm, ift_rx_mrad, ift_ry_mrad, ift_rz_mrad`
- 单位：`mm, mm, mrad, mrad, mrad`
- FD step：`0.5 mm, 0.5 mm, 10 mrad, 10 mrad, 10 mrad`
- `S = (5, 5, 60, 60, 60)` mm/mrad
- `W`：名义 4×4 pair covariance 的 WLS
- `rank_tolerance = 0.01`
- 全部 18 个 hierarchical V1 train/validation 源，含条目 69 的
  rank 2/3/4 源
- slope：`hypot(source_tx, source_ty)`，锚点 source-station
  local SegmentFit 状态，不是 leftover residual `(rtx, rty)`
- topology：complete four-station truth routes
- 继承条目 68–72 的冻结否定结论
- 固定 `C_dx` 是模型边界，不是测量、不是 survey 验证
- 本战役不是条目 36 的 survey-`dz` curriculum，也不是 7D SVD 删列

## 原生 5 列谱

Pooled（4141 pairs）奇异值
`16327, 2824, 527, 424, 265`，相对阈值下 rank **5**，null
dimension 0。

| source | split | pairs | 官方 rank | σ/σ1 |
| --- | --- | ---: | ---: | --- |
| `mc24_100043_00200_00299` | train | 232 | 5 | 1, 0.590, 0.056, 0.053, 0.030 |
| `mc24_100043_00300_00399` | train | 237 | 5 | 1, 0.449, 0.042, 0.036, 0.018 |
| `mc24_100043_00400_00499` | train | 240 | **2** | 1, 0.079, **0.0074**, 0.0064, 0.0019 |
| `mc24_100043_00500_00599` | train | 234 | **3** | 1, 0.058, 0.030, **0.0053**, 0.0012 |
| `mc24_100043_00600_00699` | train | 221 | 5 | 1, 0.435, 0.046, 0.038, 0.020 |
| `mc24_100044_00200_00299` | train | 234 | **4** | 1, 0.146, 0.014, 0.011, **0.0051** |
| `mc24_100044_00300_00399` | train | 240 | 5 | 1, 0.662, 0.061, 0.051, 0.036 |
| `mc24_100044_00400_00499` | train | 235 | **4** | 1, 0.475, 0.045, 0.036, **0.0089** |
| `mc24_100044_00500_00599` | train | 228 | 5 | 1, 0.141, 0.057, 0.014, 0.011 |
| `mc24_100044_00600_00699` | train | 245 | 5 | 1, 0.385, 0.036, 0.034, 0.020 |
| `mc24_100047_00000_00049` | val | 231 | 5 | 1, 0.784, 0.073, 0.068, 0.054 |
| `mc24_100047_00050_00099` | val | 210 | 5 | 1, 0.768, 0.073, 0.063, 0.052 |
| `mc24_100047_00100_00149` | val | 230 | 5 | 1, 0.530, 0.052, 0.046, 0.030 |
| `mc24_100047_00150_00199` | val | 240 | 5 | 1, 0.459, 0.074, 0.049, 0.037 |
| `mc24_100048_00000_00049` | val | 216 | 5 | 1, 0.262, 0.049, 0.023, 0.018 |
| `mc24_100048_00050_00099` | val | 210 | 5 | 1, 0.840, 0.089, 0.068, 0.040 |
| `mc24_100048_00100_00149` | val | 235 | 5 | 1, 0.762, 0.075, 0.065, 0.035 |
| `mc24_100048_00150_00199` | val | 223 | 5 | 1, 0.828, 0.093, 0.078, 0.042 |

`mc24_100044_00400_00499` 在条目 72 的 **7D** sibling 对照中是
rank 5，但原生 5DoF 重建后是 rank 4。这不是删列 artifact：5 列
`A` 是独立 `only_parameters` 构造，σ5/σ1 = 0.0089 < 0.01。该源
未被删除。

## Coverage

Slope tertile（每 bin ≥8 events，54 个 usable bins）在 9 个源上
掉 rank，包括若干全样本 rank 5 的源。这与条目 71 的结论一致：
angular coverage 是真实 failure mechanism，不是可调阈值。

Complete four-station truth-route topology 在 4 个低 rank 源上
复现同样的 rank 2/3/4，不是 mask 过小。

## 稳定性

Train vs validation pooled 子空间同 rank、principal angle ~0。
失败来自预声明 source 和 coverage slice 的 rank 不一致，不是同
rank 源之间的大角度。Event bootstrap 40 次中 1 次翻 rank；
half-split 24 个 half 中 1 次翻 rank。同 rank 比较的 principal
angle 仍接近 0。按合同，rank 变化即不稳定。

## 输入 / 命令

```bash
source scripts/setup_environment.sh ml
pytest -q tests/test_rigid_station_only_identifiability.py
python scripts/report_rigid_station_only_identifiability.py
```

CUDA `True`（Tesla T4）。本战役单元测试 8 passed；与 tracker-only /
stable-core / failure-provenance 合计 35 passed。没有 Condor。

报告：`outputs/rigid_station_only_tracker_alignment_identifiability_v1/`。
记录 HEAD `a1fd01bb5c2c5daf432c13db3b021feb42bdf6d1`。

## 未解除 / 明确不做

- 不得把 pooled rank=5 写成“5DoF 可辨识且可搬运”。
- 不得删 4 个低 rank 源或 9 个 tertile-drop 源来恢复 rank 5。
- 不得根据本谱反调 `S` 或 `rank_tolerance`。
- 不得再删 `rz` 或其他 DoF 追结果。
- 不得把固定 `C_dx` 解释成测得零或 survey 已验证。
- 不得打开三臂、Frozen-V2 unknown-association、真实数据
  correction 或 geometry write。
- 不得重开 7D / cluster-local / stable-core 救援。
- 本战役不是条目 36。

## 冻结结论

- Frozen V2 / association policy 不变。
- 真实数据继续 `residual_dq_monitoring_only`。
- `geometry_write_allowed=false`。
- 固定内部 geometry 后，station rigid-body 5DoF **没有**跨
  source、跨 slope coverage 的稳定 tracker-only identifiability。
- 下一步：物理上不同的 track coverage，或明确 gauge / external
  constraint。不是继续改算法。条目 74 Stage 1 已打开该 inventory，
  决策 `residual_blind_export_authorized_fd_not_opened`；尚未跳到
  gauge / external constraint。
