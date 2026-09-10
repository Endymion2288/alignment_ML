# Workbook 124: 3ST \(q/p\) CKF / KalmanFitter-refit provenance 最小诊断（Yasu-S2H）

日期：2026-09-09
状态：**完成** —— 已在 WB123 授权的 \(\le 20\) 条 focus identity 上机器验证 `front_near_s2` 与 refit 成败的关系。未进入 S3、alignment、weak-mode、B14M/B15 或 MM V2。未把 WB119 翻成 PASS。未改 fitter / seed / hit / geometry / field / covariance scale / outlier / collection。未提交 HTCondor。Focus 清单在看 dump 前冻结，看结果后未替换。

**最终判定：`mixed/inconclusive`**

- `decision = three_st_qp_refit_provenance_recorded`
- `contract_verdict = PASS`
- `diagnosis_verdict = RECORDED`
- `ckf_only_fallback_drives_s2_front`：**unresolved**（8 条 S2-front 中 5 条 Hole-99，占比 0.625；既不是 \(\ge 0.80\) 的一一/高概率关联，也不是 \(\le 0.20\) 的可拒绝）
- `kf_refit_introduces_qp_shift`：**unresolved**（xAOD 只保留 `trk` 或 `trk2`，没有同一径迹的原始 pre/post 对）
- `persisted_front_state_selection_artifact`：**rejected**（20/20 的 `front()` 都是 `ckf_target_hole` 或 `first_mot`，与源码控制流一致）
- `covariance_changed_in_refit`：**unresolved**（同样没有原始 pre/post covariance 对）
- `three_st_qp_trusted_observable = false`
- `residual_conditional_authorized = false`
- `s2_flipped_to_pass = false`
- `focus_replaced_after_results = false`
- `large_dump_submitted = false`

## 对唯一科学问题的回答

**`front_near_s2` 是不是 KalmanFitter refit failure / CKF-only fallback？异常 \(q/p\) 出在哪一步？**

不是一一对应，也不是可以单靠 \(z\) 继续推断的高概率关联。

缺 S1 时，种子 `minZ` 就是 S2，CKF 目标面在 \(z\approx\min Z-10\)，因此 **fallback 和 successful refit 都可以把 `front()` 落在 S2**：

| S2-front 子集 | \(n\) | persisted `front()` | 原始 KF | `n_mot` |
| --- | ---: | --- | --- | --- |
| Hole \(\chi^2=-99\), \(\mathrm{ndof}=0\) | 5 | CKF 目标面（pre-refit） | **失败**，persist `trk` | 全部 \(<12\) |
| Measurement / first MOT | 3 | 第一层 MOT 平滑态（post-refit） | **成功**，persist `trk2` | 全部 \(=12\) |

`KalmanFitterTool.cxx:348-351` 的默认 `MinMeasurements=12` 与这 8 条一一对齐：缺 S1 且 `n_mot<12` → 原始 refit 失败，S2 上的 Hole-99 就是 CKF-only fallback；缺 S1 且 `n_mot\ge 12` → S2-front 是 **合法 successful-refit 的 first-MOT**，必须拒绝“凡 S2-front 皆 fallback”的假设。

20 条里没有一条 persisted `front()` 与这两种源码终态不一致，因此 **不优先追序列化 / state ordering**。原始 complementary state（成功时的 CKF pre，失败时的 KF post）不在 xAOD 里：`CKF2.cxx:290-296` 只保留一边。本阶段没有、也不允许改 fitter 去重跑官方 collection。

## 源码锁定（Calypso `40892527e9c65409afd2378a2abfc25ddbddac03`）

路径均在 `Tracking/Acts/FaserActsKalmanFilter/`，哈希与 WB121/本 config 的 `pinned_calypso_sources` 一致。

1. `CKF2.cxx:268-297`：对每条 selected track，若 `addFittedParamsToTrack`（`CKF2Config.py:137 = True`）且有 reference surface，则在 CKF 目标面构造 `fittedParams`。`createTrack(..., fittedParams, backward=false)` 得到 `trk`。然后 `KalmanFitterTool::fit(trk, Zero(), isMC)` → `trk2`。成功 persist `trk2`；失败 persist `trk`，并警告 “Re-Fit either not performed or failed.”
2. `CreateTrkTrackTool.cxx:25-78`：Acts 反向 states 在 `!backward` 时 `insert(begin)`。`:82-93` 若有 `fittedParams`，在队首再插一个 Hole TSOS，`FitQualityOnSurface(-99, 0)`。这就是 CKF-only / pre-refit 的 `front()`。
3. `KalmanFitterTool.cxx:348-351`：MOT `< MinMeasurements`（默认 12）→ `nullptr`。`:360` origin = `front()->z()-10`。`:372-374` 输入协方差 \(\times 10\)。`:423` 成功时 `createTrack(gctx, track)` **不带** `fittedParams`，`front()` = 第一 MOT 平滑态。`:424-426` 失败返回 `nullptr`。
4. `CircleFitTrackSeedTool.cxx:176-186`：forward 目标 \(z=\min Z-10\)。完整三站通常靠近 S1；缺 S1 时 \(\min Z\) 在 S2，目标面就在 S2 上游。
5. `faser_reco.py:317-321`：`CKF_woIFT`，`maskedLayers=[0,1,2,3,4,5]`，`OutputCollection=CKFTrackCollectionWithoutIFT`，`BackwardPropagation=False`。

单位换算仍用 `1_MeV`（`CreateTrkTrackTool.cxx:104-126`）；本阶段不改。

## Focus 清单（dump 前冻结，未替换）

`skip_index` = 单文件 xAOD 上的 Athena 事件序。前 180 事件内 `skip_index == event_id`。选择规则只应用一次：6 条 WB119/123 identity；validation 5 + construction 3 条 skip 序最先的 S2-front dirty（缺 S1 或 `n_mot<12`）；各 3 条完整 `n_mot=18` 的 S1-front 对照。

| role | source | skip/event | 冻结理由 |
| --- | --- | ---: | --- |
| frozen_wb119 | 100043 | 8, 16 | large_pull |
| frozen_wb119 | 100048 | 6, 9 | sign_flip |
| frozen_wb119 | 100048 | 15, 16 | large_pull |
| validation_s2_front_dirty | 100048 | 50, 84, 150, 174, 177 | S2-front；缺 S1 / 两站 / 低 `n_mot` |
| construction_s2_front_dirty | 100043 | 50, 58, 92 | S2-front；缺 S1 |
| validation_s1_front_control | 100048 | 1, 5, 10 | 完整三站，`n_mot=18` |
| construction_s1_front_control | 100043 | 0, 1, 7 | 完整三站，`n_mot=18` |

只读两个 WB119 同款 xAOD；`SkipEvents=0`，`maxEvents=180`；只写 focus `(event_id, track_index)`。Truth 不进 seed / fit / prior / state 选择；dump 后按 identity 离线 join WB119 calibration，20/20 连上，`truth_rewrote_fit=false`。

## 机器结果

run：`yasu_s2h_three_st_qp_refit_provenance_batch_20260909T150728Z_aebef4ed`

20/20 条全部写出。`front()` 表面类型一律 `Trk::PlaneSurface`。

### 每条 identity

单位：\(z\) 为 mm，\(q/p\)、\(\sigma\)、\(\Delta\) 为 \(1/\mathrm{MeV}\)。

| role | run | skip | \(z\) | front | kind | `n_mot` | missing | KF | \(q_{\rm fit}\) | \(\sigma\) | \(\Delta\) | pull |
| --- | ---: | ---: | ---: | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| con S1 ctrl | 100043 | 0 | 21.4 | S1 | first_mot | 18 | complete | 成功 | \(-5.71\times10^{-7}\) | \(4.45\times10^{-7}\) | \(+2.56\times10^{-7}\) | 0.57 |
| con S1 ctrl | 100043 | 1 | 13.5 | S1 | first_mot | 18 | complete | 成功 | \(-2.62\times10^{-6}\) | \(2.80\times10^{-7}\) | \(-1.87\times10^{-7}\) | −0.67 |
| con S1 ctrl | 100043 | 7 | 17.4 | S1 | first_mot | 18 | complete | 成功 | \(-1.60\times10^{-5}\) | \(2.35\times10^{-6}\) | \(-2.73\times10^{-6}\) | −1.16 |
| WB119 pull | 100043 | 8 | 13.5 | S1 | first_mot | 18 | complete | 成功 | \(-8.85\times10^{-6}\) | \(1.18\times10^{-6}\) | \(-6.98\times10^{-6}\) | −5.90 |
| WB119 pull | 100043 | 16 | 9.5 | S1 | first_mot | 18 | complete | 成功 | \(-7.84\times10^{-5}\) | \(1.63\times10^{-6}\) | \(-1.55\times10^{-5}\) | −9.50 |
| con S2 dirty | 100043 | 50 | 1201.4 | S2 | **hole-99** | 10 | missing_s1 | **失败** | \(-6.09\times10^{-7}\) | \(2.48\times10^{-6}\) | \(+7.77\times10^{-7}\) | 0.31 |
| con S2 dirty | 100043 | 58 | 1203.5 | S2 | first_mot | 12 | missing_s1 | 成功 | \(-3.74\times10^{-6}\) | \(2.72\times10^{-6}\) | \(+8.01\times10^{-8}\) | 0.03 |
| con S2 dirty | 100043 | 92 | 1207.4 | S2 | first_mot | 12 | missing_s1 | 成功 | \(-1.09\times10^{-5}\) | \(2.95\times10^{-6}\) | \(-6.74\times10^{-7}\) | −0.23 |
| val S1 ctrl | 100048 | 1 | 9.5 | S1 | first_mot | 18 | complete | 成功 | \(+3.76\times10^{-7}\) | \(1.38\times10^{-7}\) | \(-5.18\times10^{-8}\) | −0.38 |
| val S1 ctrl | 100048 | 5 | 13.5 | S1 | first_mot | 18 | complete | 成功 | \(+4.58\times10^{-7}\) | \(1.74\times10^{-7}\) | \(+2.45\times10^{-8}\) | 0.14 |
| WB119 flip | 100048 | 6 | 13.5 | S1 | first_mot | 18 | complete | 成功 | \(-2.09\times10^{-6}\) | \(2.27\times10^{-6}\) | \(-2.68\times10^{-6}\) | −1.18 |
| WB119 flip | 100048 | 9 | 13.5 | S1 | first_mot | 18 | complete | 成功 | \(-6.34\times10^{-7}\) | \(2.54\times10^{-7}\) | \(-9.76\times10^{-7}\) | −3.84 |
| val S1 ctrl | 100048 | 10 | 21.4 | S1 | first_mot | 18 | complete | 成功 | \(+2.00\times10^{-6}\) | \(1.83\times10^{-7}\) | \(+2.73\times10^{-9}\) | 0.01 |
| WB119 pull | 100048 | 15 | −0.5 | S1 | **hole-99** | 5 | missing_s3 | **失败** | \(+3.95\times10^{-4}\) | \(3.15\times10^{-6}\) | \(+3.95\times10^{-4}\) | **125** |
| WB119 pull | 100048 | 16 | 13.5 | S1 | first_mot | 18 | complete | 成功 | \(+9.83\times10^{-6}\) | \(3.16\times10^{-7}\) | \(+1.90\times10^{-6}\) | 6.03 |
| val S2 dirty | 100048 | 50 | 1199.5 | S2 | first_mot | 12 | missing_s1 | 成功 | \(-3.59\times10^{-6}\) | \(3.21\times10^{-6}\) | \(-3.99\times10^{-6}\) | −1.24 |
| val S2 dirty | 100048 | 84 | 1189.5 | S2 | **hole-99** | 6 | missing_two+ | **失败** | \(-1.90\times10^{-9}\) | \(3.16\times10^{-6}\) | \(-6.62\times10^{-7}\) | −0.21 |
| val S2 dirty | 100048 | 150 | 1228.9 | S2 | **hole-99** | 8 | missing_s1 | **失败** | \(+5.51\times10^{-6}\) | \(2.51\times10^{-6}\) | \(-1.54\times10^{-6}\) | −0.61 |
| val S2 dirty | 100048 | 174 | 1193.5 | S2 | **hole-99** | 11 | missing_s1 | **失败** | \(+5.19\times10^{-6}\) | \(2.09\times10^{-6}\) | \(+1.11\times10^{-6}\) | 0.53 |
| val S2 dirty | 100048 | 177 | 1197.4 | S2 | **hole-99** | 4 | missing_two+ | **失败** | \(+1.00\times10^{-10}\) | \(3.16\times10^{-6}\) | \(-4.15\times10^{-7}\) | −0.13 |

### 组对比（诊断，不是新校准）

| 组 | \(n\) | hole | first_mot | \(\tilde\Delta\) | \(\tilde\sigma\) | \(\tilde\mathrm{pull}\) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S1-front 对照 | 6 | 0 | 6 | \(-2.5\times10^{-8}\) | \(2.32\times10^{-7}\) | −0.18 |
| 全部 S1-front | 12 | 1 | 11 | \(-1.2\times10^{-7}\) | \(3.80\times10^{-7}\) | −0.52 |
| S2-front / dirty | 8 | 5 | 3 | \(-5.4\times10^{-7}\) | \(2.84\times10^{-6}\) | −0.17 |

S2-front 的 \(\tilde\sigma\) 大约是完整 S1 对照的 **12 倍**。这是 persisted 终态上的比较，不是同一径迹的 pre→post。

### 不能用 \(z\) 单独下的额外事实

- WB119 灾难点 100048 / skip 15：`front()` 在 \(z=-0.5\)（S1 窗口内的 CKF 目标 Hole-99），`n_mot=5`，\(q/p=3.95\times10^{-4}\)，pull \(=125\)。这是 **CKF-only fallback**，但 **不是 S2-front**。
- WB119 两条 sign-flip（100048 / 6、9）和其余 large-pull 完整三站：全部是 **successful-refit first-MOT at S1**。符号翻转和大部分大 pull **不是** refit failure。
- 100048 / 84 与 177 的 persisted \(q/p\) 接近 0 或 \(10^{-10}\)，正是 CKF-only、低 `n_mot` 的脏态，不是 KF 成功后再被 `front()` 选错。

## 机制裁决（闸门在 dump 前冻结）

- 支持 fallback 驱动 S2-front：Hole-99 占 S2-front \(\ge 0.80\) 且 \(n_{\rm S2}\ge 4\)
- 拒绝：该比例 \(\le 0.20\)；或全部 S2-front 都是 first-MOT 且无 Hole-99
- 本样本 0.625 → **unresolved**。`one_to_one_s2_front_equals_fallback=false`，`legal_successful_refit_first_mot=false`
- persisted `front()` 与已知两种终态一致 → 选择伪迹 **rejected**
- 无原始 pre/post 对 → \(q/p\) shift 与 covariance 变化 **unresolved**（负结果保留）

## Provenance

- run：`yasu_s2h_three_st_qp_refit_provenance_batch_20260909T150728Z_aebef4ed`
- config SHA：`bd88ed9fb34ca5a3c1777407fc95c6f29867e3c583749a5606e85882ece748b4`
- code SHA：`55cf982302a3c62c57b74f368d5e3ba7723fd33a`
- contract SHA：`01bcf3a924ae4a0363ddbe6f53f295e9bdc6f728cf42ca83a6f2ac418657d4a4`
- dump 100043 tracks：`df3185b882dca0922904bff0d33399f4a736c00026b234dd32bf183a7c2b2569`
- dump 100048 tracks：`9c0d5d7d58deb9825738bbf2f63d715ce872e712bfea779c3d5abc75285a3487`
- WB119 仍为 `three_st_qp_calibration_not_established`（`b326532b381d7240ba0615bf3e4825f5a20c9b0b242a3c40cc52f25e97da3a43`）
- WB123 仍为 `three_st_qp_tail_source_recorded`（`446989727517e8c8815757c8a3077c880b1498fa0e657779cf7af876b7feda5a`）
- Calypso `40892527e9c65409afd2378a2abfc25ddbddac03`；Athena 24.0.41；ACTS 32.0.2
- geometry / field / conditions 哈希与 WB119 相同
- 测试：`tests/test_three_st_qp_refit_provenance.py` 8 passed
- HTCondor：无

Dump 是对已有 `CKFTrackCollectionWithoutIFT` 的 inspect-only 导出。没有编译进 live `KalmanFitterTool::fit`：该头文件会把 Acts 代数和 Athena `Amg` 缠在一起，而且二次 refit 的 origin 是 `front()->z()-10`，不是原始 CKF 目标面，不能冒充原始 pre/post。

## 禁止声称

- 三站 \(q/p\) 已校准或已是可信动量观测量
- WB119 / S2 被本阶段翻成 PASS
- 已授权 \(E[r_{\rm IFT}\mid q/p]\)、alignment、\(R_y/d_x\) weak-mode、B14M/B15、MM V2
- `front_near_s2` **就是** CKF-only fallback（5/8 是，3/8 是合法 first-MOT）
- 已在同一径迹上看到原始 KF 引入的 \(q/p\) 或 covariance 位移
- persisted `front()` 与 pre/post 不一致，因而该先查序列化
- 可用二次诊断 refit 改写或替换官方 collection
- 已授权 covariance 修复或改 fitter / `MinMeasurements` / seed
- 这 20 条足以重新定义正式均值

## 下一步

S3 保持关闭。S2-front / dirty 的 provenance 现在清楚了：它是 **缺 S1 拓扑 + `MinMeasurements=12` 闸门** 下的两种合法终态混合物，不是单一 fallback 标签。因此：

- 不再用 \(z\) 代理推断 refit 成败；
- 不因为“解释清楚了 dirty 径迹”就打开 residual conditional；
- **不授权**后续 covariance 修复研究：官方 xAOD 没有同一径迹的 pre/post 对，完整 S1 对照的 \(\Delta\) 也不是这个 20 事件合同要修的对象。若将来有人要同时留下 CKF `trk` 与 KF `trk2`，那是新的、必须单独预注册的 reconstruction 导出，且不得改拟合。

后续已在 WB125 / Yasu-S2I 用 measurement-level `bending_raw` 回答更基础的问题（不用 fitted \(q/p\)）。S3 仍关闭。
