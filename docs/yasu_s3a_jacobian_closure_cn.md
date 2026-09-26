# Yasu-S3A：固定测量的曲率/几何响应 Jacobian

Workbook 128。与 S2K/WB127 `qp_bending_proxy` 隔离。在不要求可信
reconstructed momentum 的前提下，回答一个合理大小的 \(q/p\) 误差能否在
IFT 产生与 \(R_y\)、\(d_x\)、\(t_y\) 相似的 residual response。

唯一合法的最终 claim：

> 在固定且验证过的 measurement/surface contract 下，曲率误差能够/不能
> 产生某种 IFT residual response，并与哪些 geometry/slope response 相似。

**不得**声称自然数据中的 \(q/p\)-\(R_y\) weak mode 已建立。高 cosine
不是 weak mode。

## S2K 隔离

Reviewer P0：`bending_raw` 是两段弦角差，response 是 \(W(z)\) 三角形核，
不是无权 \(\int(B_x dz-B_z dx)\)。均匀场圆轨道上现有 proxy 约为
\(0.5\times\) truth。不要乘 2、不要拟合自由 scale、不要继续 S2K batch。

## 冻结合同

- 身份：`file_sha256 + source_id + run_id + event_id + skip_index + collection + track_index + measurement_digest`
- collection：`CKFTrackCollectionWithoutIFT`
- IFT association：`SCT_SDO_Map` truth-associated cluster，禁止 4ST（隔离 P1-6）
- residual：真实 IFT sensor loc0
- 几何：runtime `SiDetectorElement` / ACTS wafer transform；pivot 为 FASER 原点；禁止手写 station response；禁止写 `/Tracker/Align`
- 只做 mean-response；不用失准 native 5×5 加权
- 不改 fitter / seed / hits / geometry payload / covariance scale
- 不进 B14M/B15/MM V2；不碰 held-out；不提交 HTCondor

20 条 development track 已在
`configs/yasu_s3a_jacobian_closure_v1.yaml` 中、在看 Jacobian 前冻结。
覆盖完整三站、WB119 sign-flip / large-pull、S2-front dirty。看结果后不得换名单。

## Jacobian

中心有限差分 \(\pm\delta,\pm\delta/2,\pm\delta/4\)。冻结幅度：
\(q/p=10^{-6}\,\mathrm{MeV}^{-1}\)，\(R_y=10^{-3}\,\mathrm{rad}\)，
\(d_x=0.10\,\mathrm{mm}\)，\(t_y=10^{-4}\)。报告最后两级相对收敛；近零响应用
\(10^{-3}\,\mathrm{mm}\) 绝对容差。

当前 dump 做 **fixed-state**。**profiled-track** 因本阶段不改 fitter 而推迟。

独立控制：零场直线/旋转平面、均匀场圆轨道、energy-loss on/off、固定 surface sequence。surface / association / navigation / FD 不闭合时立即停止物理解读。

## PASS

数值导数闭合、正负扰动线性闭合，并能定量给出达到 Yasu residual 量级所需的
\(\delta(q/p)\) 及其与 \(R_y/d_x/t_y\) 的方向是否近共线。若不能模拟，则否定该机制，不再增加 momentum-proxy workbook。仅当 fixed-state 显著近共线时，才授权后续 Schur/profile 约化法方程 falsification。

## 运行

```
scripts/build_ckf_yasu_s3a_jacobian_dump.sh
scripts/run_ckf_yasu_s3a_jacobian_dump.sh
scripts/audit_yasu_s3a_jacobian_closure.py
```

## 已记录 dump（2026-09-10）

本地 Athena，无 HTCondor。artifact
`outputs/yasu_s3a_jacobian_closure_v1/yasu_s3a_jacobian_closure_20260910T193937Z_24be51ac`。

**判定：FAIL `yasu_s3a_jacobian_recorded`。** `authorize_profiled_schur_next=false`。
`weak_mode_claimed=false`。config SHA
`d3d2e28f18943805e303ffb2676607bfc6e4d947c9ceebefad0c39c647cc671a`。
git HEAD `b617614e7746ec400072347b7cf3f056a4b9a9a7`。

20 条冻结身份均已 dump。12 条 Jacobian-eligible；8 条 dirty 只作覆盖、不决定 PASS。
3 条 eligible 破坏 surface/navigation 合同（100048 skip 1/9 `navigation_path_changed`，
skip 15 `propagate_surface_failed`），整阶段停止物理解读。

9 条闭合 eligible 上，IFT loc0 mean-response 数值闭合。
\(\partial r/\partial(q/p)\) RMS \(\sim 2.6\times10^5\,\mathrm{mm\,MeV}\)；
\(\delta(q/p)\approx 3.7\times10^{-7}\,\mathrm{MeV}^{-1}\) 对应 \(0.1\,\mathrm{mm}\)。
\(\cos(q/p,t_y)\approx 0.99994\)，与 \(R_y/d_x\) 的 \(\lvert\cos\rvert\lesssim 0.007\)。
与 \(t_y\) 高相关 **不是** \(q/p\)-\(R_y\) weak mode。未闭合前不授权 Schur。
