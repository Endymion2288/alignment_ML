# Rigid-Station-Only Tracker Alignment Identifiability V1

Workbook 73 / 2026-09-03。条目 68–72 继续冻结。本战役不拯救
7D tracklet-level observable、cluster-local observable 或
cross-source stable core。不改 `rank_tolerance=0.01`，不根据谱
反调 `S`，不删失败 source，不更换 residual，不重定义
stable-core，不根据 slope / residual / cosine 重选 population。

物理模型在看任何 5DoF SVD 之前写死。tracker fit 只允许 station
rigid-body `dx, dy, rx, ry, rz`。`dz` 继续 track-unidentifiable /
survey-or-gauge-constrained，不交给 tracks。内部 plane/module
geometry 和 `C_dx` 不是自由参数，固定为当前 reconstruction
geometry / metrology state。固定 `C_dx` 是模型边界，不是“测得
`C_dx=0`”，不是 survey 已验证当前 `C_dx`，也不得把旧 survey
central value 强行写入 fit。Survey/metrology 仍只作 external
cross-check。

原生 5 列 `A = W^{1/2} J S` 必须用 hierarchical V1 物理 FD 库的
`only_parameters` 重建。禁止从旧 7D SVD 删掉 `dz/C_dx` 两列后
宣布成功。pooled rank=5 不是 portability。所有预声明
source-disjoint 验证和 slope/topology coverage slice 都必须通过。

不重新训练 V2/V3/Transformer，不改 Frozen V2，不 Newton，不写
geometry，不产生 alignment payload。真实数据仍是
`residual_dq_monitoring_only`。identifiability 通过前，三臂、
Frozen-V2 unknown-association 和真实数据 correction 保持关闭。

## 冻结合同

- 真实数据仍是 `residual_dq_monitoring_only`。
- `geometry_write_allowed=false`。
- 冻结 V2 checkpoint SHA256
  `0c85a001677678163512c59bd5ceb6d08bdefaab79c0f4628659a4a8ad766a27`。
- 继承条目 68
  `tracker_only_identifiable_basis_unstable_solve_stopped`；条目 69
  `cross_source_stable_core_independent_validation_fail`；条目 70
  `cluster_local_transfer_exact_join_restored_not_an_identifiability_campaign`；
  条目 71 `cluster_local_observable_not_cross_run_portable`；条目 72
  `tracklet_independent_failure_provenance_audit_complete_sources_not_dropped`。
- 参数顺序
  `ift_dx_mm, ift_dy_mm, ift_rx_mrad, ift_ry_mrad, ift_rz_mrad`。
- 单位 `mm, mm, mrad, mrad, mrad`。
- FD step `0.5 mm, 0.5 mm, 10 mrad, 10 mrad, 10 mrad`。
- `S = (5, 5, 60, 60, 60)` mm/mrad，SVD 前声明。
- `rank_tolerance = 0.01`。
- `W` 是 `solve_physical_finite_difference` 已用的名义 4×4 pair
  covariance WLS。
- 全部 18 个 hierarchical V1 train/validation 源，含条目 69 的
  rank 2/3/4 源。
- slope 是锚点 source-station tracklet 局部状态的
  `hypot(source_tx, source_ty)`，不是 leftover residual
  `(rtx, rty)`。
- topology 是 complete four-station truth-route mask。
- 本战役不是条目 36，也不是 7D 删列。

## 判定

`rigid_station_five_dof_not_source_or_coverage_portable`

| 门 | 值 |
| --- | --- |
| Jacobian 有效 | **true** |
| pooled identifiable rank | **5**（不是 portability） |
| 每个预声明 source rank 5 | **false**（14/18） |
| source-disjoint 子空间稳定 | **false** |
| train vs validation 子空间 | **true** |
| slope tertile 无 rank drop | **false** |
| complete-route topology rank 5 | **false** |
| event bootstrap 稳定 | **false** |
| 失败分类 | **normal_physics_or_coverage_loss** |
| 源被删除 | **false** |
| 7D 删列 | **false** |
| 三臂 / Frozen-V2 / 真实数据 correction | **未授权** |
| geometry write | **false** |

固定内部探测器 geometry 后，当前 tracker data **不足以**跨
source、跨 slope coverage 稳定确定 station rigid-body 5DoF。失败
不是 Jacobian 构造 artifact：`+/-` probe 完整、population 对齐、
残差有限、没有近零列、source slope 无缺失、没有混入 held-out
点、没有打开 sealed test。

## 官方谱

配置
`configs/rigid_station_only_tracker_alignment_identifiability_v1.yaml`，
SHA256
`80b5ebebe71e6d2bb20604c8593b97540cd998cbafee86e94d201dd6b00719ba`。

Pooled 4141 pairs：奇异值 `16327, 2824, 527, 424, 265`，官方 rank
**5**。

原生 rank 低于 5 的源：

| source | pairs | 官方 rank | 第一个低于阈值的相对奇异值 |
| --- | ---: | ---: | --- |
| `mc24_100043_00400_00499` | 240 | **2** | σ₃/σ₁ = 0.0074 |
| `mc24_100043_00500_00599` | 234 | **3** | σ₄/σ₁ = 0.0053 |
| `mc24_100044_00200_00299` | 234 | **4** | σ₅/σ₁ = 0.0051 |
| `mc24_100044_00400_00499` | 235 | **4** | σ₅/σ₁ = 0.0089 |

其中最后一个在条目 72 的 7D sibling 对照中是 rank 5，原生 5DoF
重建后是 rank 4。这符合合同：本战役没有从 7D SVD 删列。该源未被
删除。

Slope tertile（每 bin ≥8 events，54 个 usable bins）在 9 个源上
掉 rank，包括若干全样本 rank 5 的源。Angular coverage 仍是真实
failure mechanism，与条目 71 一致。Complete-route topology 在同样
四个低 rank 源上复现 rank 2/3/4。

Train 与 validation pooled 子空间重合。不稳定来自预声明 source、
tertile、topology、bootstrap（1/40）和 half-split（1/24）的 rank
变化，不是同 rank 源之间的大角度。

## 命令

```bash
source scripts/setup_environment.sh ml
pytest -q tests/test_rigid_station_only_identifiability.py
python scripts/report_rigid_station_only_identifiability.py
```

CUDA `True`（Tesla T4）。本战役测试 8 passed；与相关
identifiability 套件合计 35 passed。没有 Condor。

报告：
`outputs/rigid_station_only_tracker_alignment_identifiability_v1/`。
记录 HEAD `a1fd01bb5c2c5daf432c13db3b021feb42bdf6d1`。

## 下一阶段

不得靠删 source、再删 DoF 或调阈值恢复 rank 5。不得重开 7D /
cluster-local / stable-core 救援。真实数据继续 residual DQ
monitoring。条目 74 Stage 1 已打开物理不同 track-coverage
inventory，并冻结 `residual_blind_export_authorized_fd_not_opened`；
尚未跳到 gauge / external constraint。继续改算法仍然禁止。
