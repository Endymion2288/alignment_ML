# CKFTrackCollectionWithoutIFT Definition Contract (Yasu Stage 0)

Workbook 117.  This task establishes the official production meaning of
`CKFTrackCollectionWithoutIFT` from Calypso source, ROOT metadata, and a
small reconstruction dump.  It does not start the 3ST→IFT prediction
chain until the definition PASSes.

## Source contract

`faser_reco.py` always instantiates `CKF_woIFT` with

```
maskedLayers = [0, 1, 2, 3, 4, 5]
OutputCollection = CKFTrackCollectionWithoutIFT
BackwardPropagation = False
actsOutputTag = {filestem}_3station_forward
```

`CircleFitTrackSeedTool` encodes a wafer as `6 * station + 2 * layer + side`.
`SiDetectorElement::isInterface()` is `station == 0`.  Therefore codes
0–5 are all IFT (station 0) sides.  Those clusters are omitted from
sourceLinks and measurements before CKF runs.  `CreateTrkTrackTool` and
`KalmanFitterTool.fit` only reuse measurements already on that track.

This is **not** WB109 LTO.  LTO leaves out stations 1/2/3 and keeps IFT.
This is **not** `CKFTrackCollection`, which is the 4-station CKF.

## Native state

Persisted parameters are `Trk::CurvilinearParameters`
`(loc1, loc2, phi, theta, q/p)`.  `CreateTrkTrackTool` converts ACTS
`q/p` from `1/GeV` to Athena `1/MeV`.  Charge is the sign of `q/p`.
The 5×5 includes q/p cross terms when present.

## Machine checks

1. ROOT `CollectionTree` contains `CKFTrackCollectionWithoutIFT`.
2. The same events contain IFT clusters (`FaserSCT_ID.station == 0`).
3. WithoutIFT tracks have zero station-0 hits on `measurementsOnTrack`
   and on all TSOS, including outliers.
4. Station IDs are decoded by `FaserSCT_ID`, not by a guessed z map.
5. The 5×5 is present and is not the SegmentFit dummy
   (`q/p = 1e-5 /MeV`, `var = 5e-6 /MeV^2`).
6. Truth q/p is not used.

## Forbidden

LTO as a substitute; 4-station CKF as a substitute; dummy q/p; truth q/p;
covariance rescale; new source campaigns; geometry writes; opening
B14M / B15 / Measurement Model V2; sealed-test access.

PASS authorizes only the next small-sample 3ST→IFT chain.  It does not
claim Yasu’s weak-coupling mechanism.
