# 2026-08-09: 偏移几何 Segment Refit 与 Physical Closure

## 本阶段目标

确认现有 xAOD 的持久化 `SCT_ClusterContainer` 是否可在新的
`SCTAlignmentStore/FaserActsAlignment` condition 下重新构造 cluster 并运行
`SegmentFit/GhostBusters`，从而避免把 payload 仅应用于已持久化 segment state 的
coordinate-level surrogate。

## Cluster 层审计结论

结论：V1 station-level `dx, dy` 刚体平移的 local segment refit 可以直接从
`SCT_ClusterContainer` 开始，当前不需要回到 RDO。

源码证据：

- xAOD branch 类型为 `Tracker::FaserSCT_ClusterContainer_p3`；
- `FaserSCT_Cluster_p3` 保存 `m_localPos`、`m_mat00/m_mat01/m_mat11`、cluster/strip
  identifier、`m_rdoList`、width 与 time-bin；
- `FaserSCT_ClusterContainerCnv_p3` 读回时从当前
  `SCT_DetectorElementCollection` 取得 detector element，再构造 transient cluster；
- `SegmentFitAlg` 的输入 key 为 `SCT_ClusterContainer`，使用 cluster 的
  `globalPosition()`、`localCovariance()` 与 detector element；
- `GhostBusters` 可以用新的 `SegmentFitRefit` 写出 `SegmentsRefit`。

因此已持久化的是 local measurement 和 identity，不是不可变的 global measurement。

## 实现

修改 `calypso/PhysicsAnalysis/NtupleDumper/scripts/faser_ntuple_maker.py`：

- 新增 `--refit-segments`；
- 新增可配置的 cluster 输入、`SegmentFitRefit` 输出和 `SegmentsRefit` 输出；
- 禁止覆盖 xAOD 原有 `SegmentFit/Segments`；
- 在 payload SQLite 读取后调度 `SegmentFitAlgCfg` 与 `GhostBustersCfg`；
- NtupleDumper 改为导出新的 refit collection；
- 继续用 `FaserActsExtrapolationTool` 导出 field-aware propagation record。

1-event nominal 与 station-3 payload smoke test 均成功。随后运行了 5-event 对照：

- nominal：`outputs/mc24_muon_fasernu_5events_segment_refit_nominal/`；
- station 3 `dx=+1 mm`：
  `outputs/mc24_muon_fasernu_5events_segment_refit_station3_dx1mm/`；
- payload：
  `outputs/mc24_muon_fasernu_5events_physical_station3_dx1mm/payload/`。

两个 job 均处理 5 个 event、20 个 good station、20 个输出 local segment。偏移 job log
确认 SQLite override `/Tracker/Align`、`SCTAlignmentStore`、`FaserActsAlignment`、
`SegmentFitAlg` 与 `Execution succeeded`。

## Physical Response 验证

使用 `audit_refit_geometry_response.py` 比较两次独立 refit 输出。

- 20 条 tracklet identity 完全一致；
- S0/S1/S2 state 不变；
- S3 的平均 `delta x = 1.000000000000069 mm`；
- S3 位置 response 最大误差 `5.5e-13 mm`；
- 每个 q/p mode 有 27 条 truth-matched pair；
- 含 S3 target 的 `delta r_x` 为 `+1 mm`，pair response 最大误差 `5.5e-13 mm`；
- combined covariance 元素最大变化 `5.7e-14`，符合纯平移下 covariance 不变的预期；
- slope residual 最大变化 `2.9e-14`。

输出：

- `outputs/mc24_muon_fasernu_5events_segment_refit_geometry_response/tracklet_response.csv`；
- `outputs/mc24_muon_fasernu_5events_segment_refit_geometry_response/propagation_response_mode0.csv`；
- `outputs/mc24_muon_fasernu_5events_segment_refit_geometry_response/propagation_response_mode1.csv`；
- 完整数组保存为对应的 `.npz`，摘要为 `metrics.json`。

## q/p 与 Propagation Mode 审计

新的 `audit_propagation_mode_components.py` 每个 mode 输出 raw CSV，字段包括：

```text
rx_mm, ry_mm, rtx, rty,
propagated_sigma_*, target_sigma_*, combined_sigma_*,
combined_covariance_det, combined_covariance_logdet, chi2
```

nominal 5-event 的 27 个相同 truth pair：

| 量 | 值 |
| --- | ---: |
| mode 0 平均 chi2 | 202.36 |
| mode 1 平均 chi2 | 553.95 |
| 固定 mode-0 covariance 的平均 residual effect | 0.0880 |
| 切换 residual 后平均 covariance effect | 351.49 |
| 平均 `log det(S1)-log det(S0)` | -14.205 |

结论：mode 1 chi2 变大不是 residual 恶化，而是 covariance 大幅收窄。该结论也在 payload
refit 输出中复现。原始 artifact：

- `outputs/mc24_muon_fasernu_5events_segment_refit_mode_audit_nominal/`；
- `outputs/mc24_muon_fasernu_5events_segment_refit_mode_audit_station3_dx1mm/`。

`SegmentFitAlg.cxx` 当前固定设置 `qoverp = 1/100000`，方差为
`50000*qoverp^2 = 5e-6 / MeV^2`。refit q/p audit 确认 20 条 tracklet 的 native q/p 都是
`1e-5 / MeV`，truth sign 在 19 条可比较记录中仅 11 条一致。因此 native local q/p 不能
作为 V1 measurement；mode 1 只作为 MC truth-q/p 的 propagation audit，physical closure 使用
mode 0。

## Fixed-Truth Alignment Recovery

使用 `run_refit_alignment_closure.py`，station 0 固定为 reference，mode 0，全部 27 条
truth-matched pair 参与 WLS。结果：

```text
normal matrix rank = 6
S0 = [0, 0] mm
S1 = [5.8e-15, 4.7e-13] mm
S2 = [6.8e-14, 4.8e-13] mm
S3 = [1.000000000000064, 4.8e-13] mm
```

可动 station 的最大恢复误差为 `4.87e-13 mm`。详细 artifact：
`outputs/mc24_muon_fasernu_5events_segment_refit_station3_dx1mm_closure/`。

## 边界与下一步

- 这已不是 coordinate-level surrogate；segment 与 propagation 均在 payload condition 下重跑。
- `+1 mm` 是单一刚体平移 closure，尚不是 physical capture-range scan。
- physical scan 应生成多个 `dx,dy` payload，对每个 payload 重跑 cluster-to-segment refit，
  再调用 fixed-truth closure；不要使用旧 coordinate-level scan 代替。
- 两次 refit 使用相同 hits，nominal combined covariance 在当前 closure 中仅作为确定性 WLS
  weight，不代表两次 residual 差值的独立统计协方差。
- 只有当需要重新 clustering、digitization 或使已保存 cluster measurement 本身失效的 geometry/
  calibration 研究时，才需要进一步回到 RDO。
