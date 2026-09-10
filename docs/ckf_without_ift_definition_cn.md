# CKFTrackCollectionWithoutIFT 定义契约（Yasu 第 0 阶段）

Workbook 117。本任务从 Calypso 源码、ROOT metadata 和一小份重建 dump
确定官方生产 collection `CKFTrackCollectionWithoutIFT` 的含义。定义未
PASS 之前不打开 3ST→IFT 预测链。

## 源码契约

`faser_reco.py` 总会实例化 `CKF_woIFT`：

```
maskedLayers = [0, 1, 2, 3, 4, 5]
OutputCollection = CKFTrackCollectionWithoutIFT
BackwardPropagation = False
actsOutputTag = {filestem}_3station_forward
```

`CircleFitTrackSeedTool` 的 wafer 编码是 `6 * station + 2 * layer + side`。
`SiDetectorElement::isInterface()` 就是 `station == 0`。因此 0–5 是 IFT
（station 0）全部 side。这些 cluster 在 CKF 之前就不进入 sourceLinks /
measurements。`CreateTrkTrackTool` 和 `KalmanFitterTool.fit` 只复用已经
在这条 track 上的测量。

这**不是** WB109 LTO。LTO 排除的是 1/2/3 站并保留 IFT。
这也**不是** `CKFTrackCollection`，那是四站 CKF。

## 原生状态

落盘参数是 `Trk::CurvilinearParameters`
`(loc1, loc2, phi, theta, q/p)`。`CreateTrkTrackTool` 把 ACTS `q/p` 从
`1/GeV` 转到 Athena `1/MeV`。电荷是 `q/p` 的符号。5×5 在存在时包含
q/p 交叉项。

## 机器检查

1. ROOT `CollectionTree` 含有 `CKFTrackCollectionWithoutIFT`。
2. 同一批事件含有 IFT cluster（`FaserSCT_ID.station == 0`）。
3. WithoutIFT 径迹在 `measurementsOnTrack` 和全部 TSOS（含 outlier）上
   都没有 station-0 hit。
4. 站号由 `FaserSCT_ID` 解码，不用猜 z。
5. 5×5 存在，且不是 SegmentFit dummy
   （`q/p = 1e-5 /MeV`，`var = 5e-6 /MeV²`）。
6. 不用 truth q/p。

## 禁止项

用 LTO 顶替；用四站 CKF 顶替；dummy q/p；truth q/p；协方差 rescale；
新建 source campaign；写 geometry；打开 B14M / B15 / Measurement Model
V2；读取 sealed test。

PASS 只授权下一步小样本 3ST→IFT 链。不声称 Yasu 的弱耦合机制成立。
