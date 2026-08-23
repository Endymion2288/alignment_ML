# Operating Protocol V1 论文结果摘要

这是对条目 54 冻结证据包的只读论文分析。不再打开新的 alignment mode，
不重训 V2，不改 occupancy / 报警 / `A`。Residual 或 χ² 下降只是
**DQ observable**。Implied `C_dx` 不是测量值。

**决策流。** `MC transfer PASS → real-data association PASS → Station calibration REJECT → reduced calibration REJECT → residual/DQ monitoring PASS`

**运行状态。** `real_data_operating_mode=residual_dq_monitoring_only`，
`geometry_write_allowed=false`，`station_calibration_mode_available=false`，
`cdx_mode_allowed=false`，`alignment_drift_candidate=false`。

## 三个层次

### 第 1 层. Frozen V2 association transfers from MC to real data

**状态：** PASS

**能够证明。** Frozen V2 association is transferable from MC to 2024 r0022 data.  The empty 100-event selected graph is statistics-limited, and full-segment route acceptance recovers.

**不能证明。** That a recoverable association graph is a solvable station alignment problem, or that residual decrease is closure.

### 第 2 层. Association usable is not alignment solvable

**状态：** REJECT_GEOMETRY_WRITE

**能够证明。** The real-data Station Jacobian has a gauge-like dz weak direction.  The 14974 5-DoF weak direction coincides with the frozen C_dx→station leakage direction A.  Reduced {dy,rx,rz} is Fisher-identifiable but does not transfer between independent calibration runs.  Geometry write is rejected by identifiability and cross-level ambiguity, not by a conservative threshold.

**不能证明。** A numerical value of C_dx, a transferable station correction, or that a chi-squared drop proves the geometry has been corrected.

### 第 3 层. Current-geometry frozen-V2 residual/DQ monitoring is stable

**状态：** PASS

**能够证明。** Under current official geometry, frozen V2 residual/DQ monitoring is stable on independent r0022 runs.  The seven new runs are nominal.  There is no persistent dy/rx drift.  Run 14977 is insufficient statistics, not an alignment anomaly.

**不能证明。** A geometry update, a survey trigger, or that monitoring residuals can be inverted into an alignment payload.

## 本文可以支持的 claim

- `mc_transfer_pass`: On source-disjoint MC, exclusive Station and IFT-Internal modes both independently close.  Frozen A is not updated.
- `real_data_association_statistics_limited_then_recovered`: At 100 events the selected graph is empty while the all-pairs candidate graph is not.  Full remaining segments recover selected-route counts 121/109/156/63/2 on 14973–14977, and 233/193/114/77/74/91/32 on the seven independent expansion runs.
- `station_jacobian_dz_gauge`: The six-DoF Station Jacobian has a near-null direction that is almost pure dz (gauge-like).  More events of the same topology do not lift that direction.
- `weak_direction_aligns_with_A`: After dropping survey dz, the 14974 five-DoF weak direction is almost pure dx and has cosine 0.994 with frozen A.  This is cross-level leakage geometry, not a C_dx measurement.
- `reduced_mode_not_transferable`: Reduced {dy,rx,rz} is full rank (cond. 73/86) and nearly orthogonal to A, but the self-nulling correction changes sign and magnitude between 14973 and 14974 (max ~258σ).  Linearized 14974→14973 chi-squared ratio ~16 is a DQ observable only.
- `geometry_write_rejected_by_physics`: Not writing geometry is a validated scientific conclusion: the observed real-data topology cannot isolate a transferable station update from C_dx leakage.  It is not an unfinished analysis and is not caused by an overly tight alarm threshold.
- `monitoring_nominal_on_independent_runs`: Seven independent r0022 runs are all nominal_monitoring under the frozen protocol.  Isolation |robust z| stays well below 3.  alignment_drift_candidate remains false.
- `low_stats_is_insufficient_not_anomaly`: Run 14977 has two selected routes and is classified insufficient_statistics_for_alignment_dq, not an alignment anomaly and not a detector-condition change.

## 本文不能支持的 claim

- `no_real_data_station_geometry_update`: The present real-data sample supports a station geometry write.
- `no_cdx_measurement`: Implied |C_dx| from a self-nulling station solve can be quoted as the true layer contrast.
- `no_residual_as_closure`: A residual or linearized chi-squared decrease can be read as proof that the station geometry has been corrected.
- `no_reduced_mode_payload`: Because {dy,rx,rz} is Fisher-identifiable it may be written as a reduced station correction.
- `no_drift_to_geometry`: A future alignment_drift_candidate may be inverted into a geometry payload without new external constraints.
- `no_v2_retune_needed`: The 100-event empty selected graph requires retuning V2 thresholds or unmatched penalty.
- `no_subset_rescue`: A quieter subset of the same r0022 topology can rescue geometry_write_allowed.

## 为什么“不写 geometry”本身就是结论

Not writing geometry is a real-data scientific result, not an unfinished state.

原因：

- Association recovery at full-segment scale shows the graph is usable, so the failure is not empty data or a broken V2.
- The six-DoF Jacobian is empirically degenerate along dz.
- The remaining five-DoF weak direction on 14974 coincides with frozen A, so a track-driven station update is cross-level contaminated.
- The only Fisher-safe reduced subspace {dy,rx,rz} does not transfer between independent calibration runs.
- Modes that still float dx or ry exceed the 1.5–1.7 µm isolation budget; more events cannot rescue that leakage.
- Independent monitoring runs stay nominal at current official geometry, so there is no empirical need to force a write.

这不是因为：

- an overly conservative alarm threshold
- an unfinished software path
- failure to search a quieter event subset

## Operating Protocol V2 的解锁条件

当前未满足。以下任一条件可以打开 V2；本包仍然不写 geometry：

1. 独立 survey / 外部约束固定 `dx/ry/dz`，并且独立测量证明真实
   `|C_dx|` 落在 1.5–1.7 µm isolation budget 内。
2. 新的独立真实 track topology 实证得到跨 run 可搬运的 calibration
   subspace。

在此之前唯一代码路径是：当前官方 geometry → 冻结 V2 → residual/DQ
monitoring。禁止从真实数据 self-nulling residual 生成 alignment payload。

## 限制

- Complete four-station routes remain rare; the Station solve is dominated by 2- and 3-station routes.
- No independent survey or external station constraint is available in this corpus to fix dx/ry/dz.
- True |C_dx| on these real runs is not measured; only the MC isolation budget 1.5–1.7 µm is known.
- Time order uses LHC fill, then run, then skip.  This corpus has no unix timestamp in occupancy.
- 14977 is too short for alignment DQ and is kept only as a statistics-control example.
- The sealed test split 100116/100117 is never opened.
- Workbook 03 still blocks unverified 2022 data0 IFT PHYS/xAOD re-export.

冻结 V2 SHA256：`0c85a001677678163512c59bd5ceb6d08bdefaab79c0f4628659a4a8ad766a27`。
