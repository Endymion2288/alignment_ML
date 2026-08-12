# 偏移几何下的 Segment Refit

## 结论

持久化 xAOD 中的 `SCT_ClusterContainer` 已足以支持 V1 的 station-level
`dx, dy` 偏移几何 segment refit。它保存了 `SegmentFitAlg` 所需的局部测量、局部协方差、
cluster/strip identifier 和 RDO identifier list。因此，对于刚体 station 平移，当前流程
**不需要** 回到 RDO。

这个结论同时来自源码审计和 MC24 100 GeV FASERnu muon 控制样本的运行验证。它仅适用于
从已有 cluster 重建 local segment；如果研究改变 clustering、digitization 或电荷标定，
结论不自动成立。

## 为什么 Cluster 层足够

输入 `CollectionTree` 将 `SCT_ClusterContainer` 持久化为
`Tracker::FaserSCT_ClusterContainer_p3`。p3 converter 保存并恢复：

- 局部位置 `m_localPos`；
- 2-by-2 局部协方差 `m_mat00`、`m_mat01`、`m_mat11`；
- cluster identifier 和压缩的 `m_rdoList` strip identifier；
- width 与 time-bin 信息。

读回时，`FaserSCT_ClusterContainerCnv_p3` 取得当前的
`SCT_DetectorElementCollection`，并用其中的 detector element 构造 transient cluster。
因此 `cluster.globalPosition()` 由当前 SCT alignment condition 计算，而不是读取一个
持久化的 global position。`SegmentFitAlg` 读取 `SCT_ClusterContainer`，使用 cluster 的
global position 和 local covariance，写出新的 `TrackCollection`；随后 `GhostBusters`
从该 refit 输出写出新的 segment collection。

关键源码位置：

- `TrackerEventTPCnv/src/TrackerPrepRawData/FaserSCT_ClusterCnv_p3.cxx`；
- `TrackerEventTPCnv/src/FaserSCT_ClusterContainerCnv_p3.cxx`；
- `TrackerSegmentFit/src/SegmentFitAlg.h` 和 `SegmentFitAlg.cxx`；
- `FaserActsKalmanFilter/src/GhostBusters.h`。

## 可复现 Refit Chain

`faser_ntuple_maker.py` 已加入 `--refit-segments`。它有意使用新的 StoreGate key
`SegmentFitRefit` 和 `SegmentsRefit`，以保留 xAOD 中的 nominal `SegmentFit`/`Segments`
供对照。

```bash
cd /eos/home-x/xcheng/FASER
source alignment_ML/scripts/setup_environment.sh calypso

faser_ntuple_maker.py INPUT-xAOD.root --isMC --useIFT --nevents 5 \
  --export-tracklets --export-tracklet-propagation --refit-segments \
  --outfile alignment_ML/outputs/example_refit_nominal/enhanced_tracklets.root

faser_ntuple_maker.py INPUT-xAOD.root --isMC --useIFT --nevents 5 \
  --export-tracklets --export-tracklet-propagation --refit-segments \
  --tracker-align-sqlite PAYLOAD/tracker_alignment.sqlite \
  --tracker-align-pool-catalog PAYLOAD/PoolFileCatalog.xml \
  --outfile alignment_ML/outputs/example_refit_displaced/enhanced_tracklets.root
```

在接受结果前，每个 job log 都必须确认：

- 偏移作业通过 SQLite override `/Tracker/Align`；
- 新建 `SCTAlignmentStore`；
- 新建 `FaserActsAlignment`；
- `SegmentFitAlg` 处理了指定 event；
- 作业成功结束。

将两个 enhanced 文件分别用 `convert_ntuple_tracklets.py` 与
`convert_ntuple_tracklet_propagations.py` 转换后，运行：

```bash
source alignment_ML/scripts/setup_environment.sh ml
cd alignment_ML

python scripts/audit_refit_geometry_response.py \
  --nominal-tracklets NOMINAL/tracklets.root \
  --nominal-propagations NOMINAL/propagations.root \
  --displaced-tracklets DISPLACED/tracklets.root \
  --displaced-propagations DISPLACED/propagations.root \
  --payload-manifest PAYLOAD/alignment_payload.json \
  --output-dir outputs/example_refit_response

python scripts/audit_propagation_mode_components.py \
  --tracklets NOMINAL/tracklets.root --propagations NOMINAL/propagations.root \
  --output-dir outputs/example_mode_audit

python scripts/run_refit_alignment_closure.py \
  --nominal-tracklets NOMINAL/tracklets.root \
  --nominal-propagations NOMINAL/propagations.root \
  --displaced-tracklets DISPLACED/tracklets.root \
  --displaced-propagations DISPLACED/propagations.root \
  --payload-manifest PAYLOAD/alignment_payload.json \
  --output-dir outputs/example_refit_closure --reference-station 0
```

三条命令均拒绝覆盖已有 artifact。

## Station 3 `+1 mm` 验证

保留的控制实验使用 global payload
`station:3 = [1, 0, 0, 0, 0, 0]` 与五个 MC24 event。

| 检查项 | 结果 |
| --- | --- |
| refit tracklet | 20 条，每个 event 均有四站 |
| truth-matched propagation pair | 每个 q/p mode 为 27 条，truth-match fraction >= 0.99 |
| S3 refit `delta x` 平均值 | `1.000000000000069 mm` |
| S3 position response 最大误差 | `5.5e-13 mm` |
| pair residual response 最大误差 | `5.5e-13 mm` |
| combined covariance 元素最大变化 | `5.7e-14` |
| slope residual 最大变化 | `2.9e-14` |

观测行为与刚体平移完全一致：只有 S3 的 global `x` 改变，终点为 S3 的 pair 满足
`delta r_x = +1 mm`；平移保持 slope 与 covariance 不变，剩余量均为浮点舍入。原始记录保存在
`outputs/mc24_muon_fasernu_5events_segment_refit_geometry_response/`。

固定 truth、使用所有 station pair 的加权解采用 mode 0，station 0 为 reference。其 rank 为 6，
恢复结果为：

```text
S0 = [0, 0] mm
S1 = [5.8e-15, 4.7e-13] mm
S2 = [6.8e-14, 4.8e-13] mm
S3 = [1.000000000000064, 4.8e-13] mm
```

可动 station 的最大误差为 `4.9e-13 mm`。这已是 physical displaced-geometry refit closure，
不是此前 coordinate-level surrogate。

## Propagation Mode 0 与 Mode 1

`audit_propagation_mode_components.py` 会为每个 mode 写一个 raw CSV，其中包含
`rx_mm`、`ry_mm`、`rtx`、`rty`、propagated/target/combined sigma、combined covariance 的
determinant 和 log-determinant、pull 与 4D chi2；同时输出同一 pair 的分解：

```text
chi2_1 - chi2_0 = residual_effect_at_S0 + covariance_effect_after_r1
```

27 条 nominal refit pair 的结果：

| 量 | 结果 |
| --- | ---: |
| mode 0 平均 chi2 | 202.36 |
| mode 1 平均 chi2 | 553.95 |
| 固定 mode-0 covariance 的平均 residual effect | 0.0880 |
| 切换 residual 后的平均 covariance effect | 351.49 |
| 平均 `log det(S_1) - log det(S_0)` | -14.205 |

因此 mode 1 chi2 的大幅增加几乎完全来自 propagated/combined covariance 明显收窄，而不是
raw residual 恶化。mode 0/mode 1 的原始比较在
`outputs/mc24_muon_fasernu_5events_segment_refit_mode_audit_nominal/`；偏移输出的结果结论相同。

当前 `SegmentFitAlg` 源码对每个 local segment 指定
`q/p = 1e-5 / MeV`、方差 `5e-6 / MeV^2`。refit q/p audit 确认 20 条 tracklet 全部得到
这两个相同值，且 19 条可做 truth 比较的记录中仅 11 条 sign 一致。因此 native local q/p
不是可用测量。mode 1 仅用于 MC truth-q/p 诊断；V1 的可运行 alignment closure 使用 mode 0。

## 范围与下一步 Scan

对于 V1 refit，不需要 RDO rerun。只有研究需要在变化的电子学/标定条件下重新 clustering、
改变 digitization，或 geometry effect 使已保存的 cluster measurement 本身失效时，才应回到 RDO。

`+1 mm` closure 已建立 physical response path，但尚不是 detector capture-range 测量。物理 scan
需要针对多个 `dx, dy` magnitude 创建独立 payload，对每一个 payload 重跑相同 refit chain，
并对每个结果执行 fixed-truth closure。这里两个相同 hit 的 refit 差值，其统计不确定度不能由
作为确定性 WLS weight 使用的 nominal covariance 直接代表。
