# 三站 \(q/p\) CKF / KalmanFitter-refit provenance（Yasu-S2H）

Workbook 124。执行 WB123 已授权的 \(\le 20\) 条 identity 诊断 dump。
不翻 WB119，不打开第 3 阶段，不改 fitter、seed、hit、geometry、
field、covariance scale、outlier 或 collection 定义。

唯一科学问题：WB123 的 `front_near_s2` 是否就是 KalmanFitter
refit failure / CKF-only fallback，以及异常 \(q/p\) 与 covariance
是在原始 CKF、KF refit、还是 persisted `front()` 选择中产生。

## 正式 run

Batch：`yasu_s2h_three_st_qp_refit_provenance_batch_20260909T150728Z_aebef4ed`

```
decision = three_st_qp_refit_provenance_recorded
ckf_only_fallback_drives_s2_front          = unresolved
kf_refit_introduces_qp_shift               = unresolved
persisted_front_state_selection_artifact   = rejected
covariance_changed_in_refit                = unresolved
official_mechanism = mixed/inconclusive
three_st_qp_trusted_observable = false
residual_conditional_authorized = false
s2_flipped_to_pass = false
focus_replaced_after_results = false
large_dump_submitted = false
```

20/20 条冻结 identity 均已写出。离线 WB119 join 20/20；truth
没有改写任何重建状态。

## 源码控制流（钉住的 Calypso `40892527e9c65409afd2378a2abfc25ddbddac03`）

文件哈希与 `configs/three_st_qp_refit_provenance_v1.yaml` 的
`pinned_calypso_sources` 一致（与 WB121 相同）。

- `CKF2.cxx:268-297`：`createTrack(..., fittedParams)` 之后总是调用
  `KalmanFitterTool::fit`。成功 persist `trk2`；失败 persist `trk`
  （CKF-only）。
- `CKF2Config.py:137`：`addFittedParamsToTrack=True`。
- `CreateTrkTrackTool.cxx:25-78`：非 backward 时 Acts 反向 states
  `insert(begin)`。`:82-93`：若有 `fittedParams`，队首再插 Hole
  TSOS，`FitQualityOnSurface(-99, 0)`。这就是 CKF-only 的 `front()`。
- `KalmanFitterTool.cxx:348-351`：MOT `< MinMeasurements`（默认 12）
  则失败。`:360`：origin = `front()->z()-10`。`:372-374`：输入协方差
  \(\times 10\)。`:423`：成功时 `createTrack` **不带** `fittedParams`，
  `front()` 是第一 MOT 平滑态。`:424-426`：失败返回 `nullptr`。
- `CircleFitTrackSeedTool.cxx:176-186`：forward 目标 \(z=\min Z-10\)。
  完整三站靠近 S1；缺 S1 时靠近 S2。
- `faser_reco.py:317-321`：`CKF_woIFT`，
  `maskedLayers=[0,1,2,3,4,5]`，
  `OutputCollection=CKFTrackCollectionWithoutIFT`，
  `BackwardPropagation=False`。

CKF2 只保留 `trk` / `trk2` 之一。原始 complementary state 不在
xAOD。本阶段只 inspect 已 persist 的 collection。没有编进二次
`KalmanFitter`：该私有头文件无法在独立插件里安全编译，而且二次
refit 的起点是 `front()->z()-10`，不是原始 CKF 目标面。

## 冻结 focus 清单

dump 检查前冻结，之后不替换。`skip_index` 是单文件 xAOD 上的
Athena 序（链式拷贝会重复 `event_id`）。前 180 事件内
`skip_index == event_id`。

6 条 WB119/123 identity，validation 5 + construction 3 条 skip
序最先的脏 S2-front（缺 S1 或 `n_mot<12`），以及 3+3 条完整
`n_mot=18` 的 S1-front 对照。

## 机制（dump 前冻结）

- `ckf_only_fallback_drives_s2_front`：S2-front 中 Hole-99 占比
  \(\ge 0.80\)（\(n_{\rm S2}\ge 4\)）。\(\le 0.20\)，或全部都是
  first-MOT 且无 Hole-99，则拒绝。
- `kf_refit_introduces_qp_shift`：需要真正的原始 pre/post \(q/p\) 对。
- `persisted_front_state_selection_artifact`：persisted `front()`
  与两种已知终态都不符。
- `covariance_changed_in_refit`：需要真正的原始 pre/post 协方差对。

官方机制是唯一成立的标签，否则 `mixed/inconclusive`。

## Batch 结论

`front_near_s2` **不是** 与 CKF-only fallback 的一一对应，也过不了
\(80\%\) 高概率线。

缺 S1 时种子 `minZ` 在 S2，因此 **两种结局都可以把 `front()` 放在
S2**：

| S2-front 子集 | \(n\) | persisted front | 原始 KF | `n_mot` |
| --- | ---: | --- | --- | --- |
| Hole \(\chi^2=-99\), ndof \(=0\) | 5 | CKF 目标面（pre-refit） | 失败 | 全部 \(<12\) |
| first MOT | 3 | 第一 MOT 平滑态（post-refit） | 成功 | 全部 \(=12\) |

fallback 占比 \(5/8=0.625\) → unresolved。3 条 `n_mot=12` 是合法
successful-refit first-MOT，因此不能接受“凡 S2-front 皆 fallback”。
也不能拒绝该假设：另外 5 条确实是 Hole-99 CKF-only。
`MinMeasurements=12` 与这个分裂完全对齐。

20 条 persisted front 都是 Hole-99 或 first-MOT，序列化 /
state-ordering 伪迹被拒绝。原始 pre→post \(q/p\) 与 covariance
仍未决：xAOD 从未同时保存两种状态。

6 条完整 S1 对照全是 successful first-MOT，
\(\tilde\sigma\approx 2.3\times 10^{-7}/\mathrm{MeV}\)。8 条
S2-front dirty 的 \(\tilde\sigma\approx 2.8\times 10^{-6}/\mathrm{MeV}\)
（大约大 12 倍）。这是终态之间的比较，不是同一径迹的 pre→post。

\(z\) 代理看不见的事实：

- WB119 灾难点 100048 / skip 15 是 \(z=-0.5\)（S1 窗口）的 Hole-99，
  `n_mot=5`，\(q/p=3.95\times 10^{-4}\)，pull \(=125\)。是 CKF-only
  fallback，**不是** S2-front。
- WB119 的 sign-flip（100048 / 6、9）和其他完整三站 large-pull 都是
  S1 上的 successful-refit first-MOT。

## 第 3 阶段状态

`three_st_qp_trusted_observable = false`。
`residual_conditional_authorized = false`。
WB119 仍为 `not_established`。

## 本阶段不得声称

解释清 S2-front provenance **不等于** 校准了三站 \(q/p\)，不等于翻
WB119，也不授权 \(E[r_{\mathrm{IFT}}\mid q/p]\)、alignment、
\(R_y/d_x\) 弱模、B14M/B15、MM V2、covariance 修复，或改
`MinMeasurements` / fitter。`front_near_s2` 不能当作 CKF-only
fallback 的同义词。二次诊断 KF 不能当作原始 pre/post。

后续为 Yasu-S2I / WB125：拟合无关的 measurement-level
`bending_raw`。第 3 阶段仍关闭。
