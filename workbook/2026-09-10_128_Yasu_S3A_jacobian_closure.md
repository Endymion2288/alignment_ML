# Workbook 128: Yasu-S3A fixed-measurement curvature/geometry Jacobian

日期：2026-09-10
状态：**dump 已物化；阶段 FAIL，停止物理解读** —— 不使用 `qp_bending_proxy`，不乘 2，不拟合自由 scale，不提交 HTCondor，不进 B14M/B15/MM V2，不碰 held-out。Focus list 在看 Jacobian 数字前冻结，事后未替换。Mean-response only。未改 fitter / seed / hit selection / geometry payload / covariance scale。

**当前判定：`FAIL yasu_s3a_jacobian_recorded`**
artifact：`outputs/yasu_s3a_jacobian_closure_v1/yasu_s3a_jacobian_closure_20260910T193937Z_24be51ac`
`authorize_profiled_schur_next = false`，`weak_mode_claimed = false`

合法最终 claim 只允许是：

> 在固定且验证过的 measurement/surface contract 下，曲率误差能够/不能产生某种 IFT residual response，并与哪些 geometry/slope response 相似。

**禁止**声称自然数据中的 \(q/p\)-\(R_y\) weak mode 已建立。高 cosine similarity 不是 weak mode。

## 为什么开 S3A，而不是修 S2K

Reviewer P0：YZ `bending_raw` 是两段弦角差，response 是 \(W(z)\) 三角形曲率核，不是无权 \(\int(B_x dz-B_z dx)\)；均匀场圆轨道上现有 proxy \(\approx 0.5\times\) truth。S2K/WB127 已隔离。本阶段直接回答 Yasu 核心问题：一个合理大小的 track-curvature \(q/p\) 误差，是否能在 IFT 产生与 \(R_y\)、\(d_x\)、\(t_y\) 相似的 residual response。

不要求可信 reconstructed momentum。不把 native 5×5 当作 Jacobian 权重。

## 冻结输入合同

每条 track 身份：`file_sha256 + source_id + run_id + event_id + skip_index + collection + track_index + measurement_digest`。

- collection 必须是 `CKFTrackCollectionWithoutIFT`
- IFT association：MC 上 `SCT_SDO_Map` 的 truth-associated cluster，barcode 与 track plurality barcode 相等。**禁止 4ST survival/association**（隔离 P1-6）
- plurality ≠ majority；majority / tie 只记录
- sensor transform、local measurement axis、rotation pivot、geometry/field IOV 从 runtime/Calypso 读取。禁止手写 station response，禁止写 `/Tracker/Align`
- residual：真实 IFT sensor loc0，\(r = \mathrm{loc0}_{cluster} - \mathrm{loc0}_{predicted}\)，不是 “YZ bending” 几何猜测

## 冻结 20-track focus（看 Jacobian 前锁定）

同一组 WB124 身份，两个冻结 xAOD：

| 角色 | n | skip_index |
| --- | ---: | --- |
| construction S1 完整三站控制 | 3 | 100043: 0, 1, 7 |
| construction WB119 large_pull | 2 | 100043: 8, 16 |
| construction S2-front dirty | 3 | 100043: 50, 58, 92 |
| validation S1 完整三站控制 | 3 | 100048: 1, 5, 10 |
| validation WB119 sign_flip | 2 | 100048: 6, 9 |
| validation WB119 large_pull | 2 | 100048: 15, 16 |
| validation S2-front dirty | 5 | 100048: 50, 84, 150, 174, 177 |

xAOD SHA-256：

- 100043：`17fa93a76ad7fc5eb71c7fd6242c566d536a9d89579a5272ab6cac1ff2f13629`
- 100048：`800c7e22758be7b7507fb8308c4d52c950f0e3acfc02ee0d57f2c5cefd974533`

看结果后不得替换 focus。

## 四列 mean-response Jacobian

从同一冻结 3ST state/measurement set：

\[
J=\left[\frac{\partial r_{\rm IFT}}{\partial(q/p)},\frac{\partial r_{\rm IFT}}{\partial R_y},\frac{\partial r_{\rm IFT}}{\partial d_x},\frac{\partial r_{\rm IFT}}{\partial t_y}\right]
\]

中心有限差分 \(\pm\delta,\pm\delta/2,\pm\delta/4\)。运行前冻结的幅度：

| 参数 | \(\delta\) | 来源 |
| --- | --- | --- |
| \(q/p\) | \(10^{-6}\,\mathrm{MeV}^{-1}\) | IFT loc0 移动约十分之几 mm（\(B\sim0.5\,\mathrm{T}\), \(L\sim2\,\mathrm{m}\)） |
| \(R_y\) | \(10^{-3}\,\mathrm{rad}\) | 典型 station alignment 1 mrad |
| \(d_x\) | \(0.10\,\mathrm{mm}\) | 典型 station alignment 0.1 mm |
| \(t_y\) | \(10^{-4}\) | 斜率扰动，小于 detector 角分辨 |

几何：pivot = FASER origin；\(G=T\,R_z R_y R_x\) active left-multiply；\(R_y=[[c,0,s],[0,1,0],[-s,0,c]]\)。作用于 runtime sensor transform，不是手写 station Jacobian。

干预：

1. **fixed-state**：只改一个参数，其余 state 固定，同一 ACTS mean transport 到同一 IFT surface
2. **profiled-track**：本 dump **不改 fitter**。保留 S1/S2/S3 measurement likelihood 的再最小化是下一独立 dump；本阶段记录 `deferred_no_fitter_change_this_stage`

闭合：最后两级导数相对收敛 \(10^{-2}\)；近零响应用绝对 residual 容差 \(10^{-3}\,\mathrm{mm}\)；正负扰动线性闭合。surface identity、association、navigation path 或 FD 不闭合时立即停止物理解读。

独立控制（Python，dump 前已闭合）：

- 零场解析直线 / 旋转平面交点（\(q/p\) 列近零；\(R_y,d_x,t_y\) 用 runtime 轴，不是 YZ 猜测）
- 均匀场解析圆轨道（FD 对 truth 相对误差 \(<10^{-2}\)）
- energy-loss on/off 与固定 surface sequence：runtime dump 后填

## P1 blockers / side-audits

| id | 问题 | 是否污染本 20-track 实验 | 本阶段动作 |
| --- | --- | --- | --- |
| P1-1 | TrackTruthMatchingTool 是 plurality 不是严格 majority | 否 | 记录；S3A 用显式 SDO barcode 相等，并报告 majority/tie |
| P1-2 | S1 truth momentum 不是同表面 Geant local momentum | 否 | 记录；Jacobian 不用 S1 truth \(p\) |
| P1-3 | bound→curvilinear covariance / KF seed surface 未闭合 | 否 | 记录；mean-response only，不用 native 5×5 加权 |
| P1-4 | CircleFitTrackSeedTool 静态 `s_spacePointMap` 可能跨事件 stale | 否 | 记录；本阶段不 reseed、不改 fitter。pin SHA `1709335ca446d475a8fbd33d0530c8009b632c1ffbbe4049f92545b80c413bbd` 已独立核对 |
| P1-5 | measurement bending space-point fallback/去重不严 | 否 | 记录；S3A 不用 bending centroids |
| P1-6 | WB118 associated residual 受 4ST association conditioning | **是** | **隔离**：IFT association = truth-SDO，永不 4ST |

不因此重开十个诊断阶段。

## PASS 条件

数值导数闭合、正负扰动线性闭合，并能定量回答：达到 Yasu 所见 residual 量级所需的 \(\delta(q/p)\) 是多少，其 layer/local-coordinate pattern 是否与 \(R_y\)、\(d_x\)、\(t_y\) 近共线（cosine \(\ge 0.98\) 只作方向报告，不是 weak-mode 门）。若曲率 response 在方向或所需幅度上明显无法模拟该 residual，则否定这条具体机制，不再增加 momentum-proxy workbook。

仅当 **fixed-state** 显示显著近共线，才授权下一阶段：带 3ST measurement likelihood 的 Schur/profile reduced-normal-matrix weak-mode falsification。否则停止。

## 下一步

1. `scripts/build_ckf_yasu_s3a_jacobian_dump.sh`
2. `scripts/run_ckf_yasu_s3a_jacobian_dump.sh`（本地，两份冻结 xAOD，覆盖最高 skip_index；无 HTCondor）
3. `scripts/audit_yasu_s3a_jacobian_closure.py`

## Dump 物化

本地 Athena，无 HTCondor。Calypso `40892527` / Athena 24.0.41 / ACTS 32.0.2。

| 源 | n_focus | jsonl SHA-256 |
| --- | ---: | --- |
| `mc24_100043_00400_00499` | 8 | `c24fa593e949584ad39480830e07c0f01a0c407458f566e7e176cea67139e4db` |
| `mc24_100048_00000_00049` | 12 | `3c10c586b478b3b345cdd61b4b8117e6454dd7206256867d3265e2e093d93290` |

事件 jsonl：100043 `37acbb5ee6185b55cc636ed6648a9fbb60251fdcc0e9c72621816e95ca55ad65`；100048 `c22217788e55f06e26bc6a50fdd03051d309fd4b92723a5863b7d86995172d3a`。

合同：config SHA `d3d2e28f18943805e303ffb2676607bfc6e4d947c9ceebefad0c39c647cc671a`；git HEAD `b617614e7746ec400072347b7cf3f056a4b9a9a7`；C++ helper `CkfYasuS3AJacobianAlg.cxx` SHA `62dabaae11f32fd56d2880e75e8d8530ab097754d4caf5bc1003e7fcddf33cb4`。Association = `truth_sdo_barcode`。`four_station_association_used=false`，`used_qp_bending_proxy=false`，`native_5x5_weighted=false`。`profiled_rungs=null`（本阶段不改 fitter）。

## 阶段 FAIL 原因（停止物理解读）

20 条冻结身份全部 dump。Jacobian-eligible 12 条（S1 控制 + WB119）；coverage-only dirty 8 条不决定 PASS。

Eligible 合同破坏 3 条，因此整阶段 FAIL，`authorize_profiled_schur_next=false`：

| 身份 | 失败码 | 说明 |
| --- | --- | --- |
| 100048 skip 1，`validation_s1_front_control` | `navigation_path_changed` | 12 个 truth-SDO IFT cluster；名义残差可用，但 \(R_y\) 扰动 23/72 rungs 出现 `propagate_surface_failed`（ACTS `SurfaceError:1`），有限差分向量长度变化 |
| 100048 skip 9，`frozen_wb119` sign-flip | `navigation_path_changed` | 名义 5 个关联中 1 个已 `propagate_surface_failed`；四列 rung 都有失败 |
| 100048 skip 15，`frozen_wb119` large_pull | `propagate_surface_failed` | n_mot=5；6 个关联 IFT 全部 propagate 失败（`source_z=-0.5 mm`，`q/p≈4.0×10^{-4}/MeV`） |

Coverage-only 记录但不延伸：100048 skip 150 `truth_sdo_ift_unmatched`；skip 84/177 `propagate_surface_failed`。不得替换 focus。

## 已闭合 eligible 样本上的 mean-response（数字，不是 weak mode）

9/12 eligible 数值导数与奇偶线性闭合。残差在真实 IFT loc0。layer 符号：\(R_y\) 与 \(d_x\) 随 stereo side 交替；\(q/p\) 与 \(t_y\) 同号、随层缓慢变化。

闭合样本 RMS（每单位参数）：

| 列 | 典型 RMS | 单位 |
| --- | ---: | --- |
| \(\partial r/\partial(q/p)\) | \(2.56\times10^5\)–\(2.70\times10^5\) | mm / (MeV\(^{-1}\)) |
| \(\partial r/\partial t_y\) | \(1.82\times10^3\)–\(1.88\times10^3\) | mm / slope |
| \(\partial r/\partial R_y\) | \(36.8\)–\(37.3\) | mm / rad |
| \(\partial r/\partial d_x\) | \(0.020\) | mm / mm；\(\lvert\partial r\rvert\cdot\delta_{d_x}=0.002\,\mathrm{mm}\le 10^{-3}\,\mathrm{mm}\) 门，记为 near-zero |

成对 cosine（闭合 eligible）：

- \(\cos(q/p, t_y)\approx 0.99994\)（9/9 近共线）
- \(\cos(R_y, d_x)\approx -0.9999\)（stereo 交替，几何平移/绕原点转动）
- \(\cos(q/p, R_y)\) 与 \(\cos(q/p, d_x)\) 均 \(\lvert\cos\rvert\lesssim 0.007\)（**不正交门之外的近共线**）

达到 \(0.1\,\mathrm{mm}\) IFT loc0 RMS 所需 \(\lvert\delta(q/p)\rvert\) 中位数 \(3.74\times10^{-7}\,\mathrm{MeV}^{-1}\)（约 \(p\sim 200\,\mathrm{MeV}\) 量级的相对曲率变化，不是“乘 2 修补”）。energy-loss on/off 最大 \(\lvert\Delta r\rvert\) 在闭合样本上 \(4\times10^{-7}\)–\(0.014\,\mathrm{mm}\)；surface identity 在 eloss 控制上保持。

**不得**把 \(\cos(q/p,t_y)\approx 1\) 说成 \(q/p\)-\(R_y\) weak mode。fixed-state 曲率响应与 \(R_y/d_x\) 的 IFT loc0 方向不正交意义上的近共线。因 eligible 合同未全部闭合，本阶段 **不授权** Schur/profile 下一步，也 **不** 据此否定或建立自然数据 weak mode。

合法最终 claim：

> 在固定且验证过的 measurement/surface contract 下，曲率误差能够产生 IFT loc0 residual response，其 layer/local-coordinate pattern 与 \(t_y\) 近共线、与 \(R_y\) 和 \(d_x\) 不正交意义上的近共线。本 dump 有 3 条 eligible track 的 surface/navigation 未闭合，因此停止物理解读；自然数据中的 \(q/p\)-\(R_y\) weak mode 未建立。

## 下一步（最小，不再开十个诊断阶段）

1. **停止** S2K batch、×2 修补、自由 scale、HTCondor、B14M/B15/MM V2、held-out。
2. 仅针对 100048 skip 1/9/15 的 `SurfaceError:1` / `navigation_path_changed` 做最小隔离：记录失败 surface geo id 与 identifier，不改 fitter/seed/geometry payload。未闭合前不得做物理解读，不得授权 Schur。
3. 不因 dirty/missing-station 覆盖失败替换 focus。
4. P1-1…P1-6 保持登记；不延伸。
