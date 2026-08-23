# Operating Protocol V1 figure caption drafts

Captions for the paper-ready figures generated from the frozen entry-54 `chart_data`.  Residual or χ² decrease is a DQ observable only.  Implied `C_dx` is not a measurement.

## fig00_decision_flow

**EN.** Figure 0. Decision flow of Operating Protocol V1 on 2024 r0022 data.  Frozen V2 association transfers from MC and recovers at full-segment statistics (PASS).  Usable association is not a solvable station alignment: full-segment Station calibration is rejected by physical non-identifiability and cross-level contamination, and the reduced {dy,rx,rz} mode is rejected by cross-run non-transferability.  The surviving operation is current-geometry residual/DQ monitoring, which passes on independent runs.  Residual or χ² decrease is never treated as alignment closure.

**CN.** 图 0. Operating Protocol V1 在 2024 r0022 上的决策流。冻结 V2 关联从 MC 迁移到真实数据，并在 full-segment 统计下恢复（PASS）。关联可用并不等于 station alignment 可解：全段 Station calibration 因物理不可识别与跨层污染被否决，reduced {dy,rx,rz} 因跨 run 不可搬运被否决。幸存路径是当前官方 geometry 下的 residual/DQ monitoring，并在独立 run 上通过。Residual / χ² 下降从不作为 alignment closure。

## fig01_route_acceptance_scaling

**EN.** Figure 1. Frozen-V2 selected-route count versus reconstructed event statistics.  Every 100-event window has a nonempty all-pairs candidate graph but zero selected routes (statistics-limited).  Acceptance recovers on the full remaining segment for 14973–14976 and on seven independent expansion runs.  14977 saturates at two selected routes because the remaining segment is only 1687 events.  This is association transfer, not an alignment update.

**CN.** 图 1. 冻结 V2 selected-route 数随重建事例统计的标度。所有 100-event 窗口的 all-pairs 候选图非空，但 selected 图为空（statistics-limited）。14973–14976 的 full remaining segment 以及 7 个独立 expansion run 上 acceptance 恢复。14977 停在 2 条 selected route，因为剩余 segment 只有 1687 个事例。这是关联 迁移，不是 alignment 更新。

## fig02_route_composition

**EN.** Figure 2. Composition of frozen-V2 selected routes by endpoint length.  The sample is dominated by 2- and 3-station routes; complete 4-station routes remain rare.  14977 (two 3-station routes) is insufficient statistics for alignment DQ, not an alignment anomaly.  Route composition is a data-quality observable.

**CN.** 图 2. 冻结 V2 selected route 按端点站数的组成。样本以 2 站和 3 站 route 为主，完整 4 站 route 仍然很少。14977（两条 3 站 route）属于 alignment DQ 统计不足，不是 alignment 异常。Route 组成是数据质量观测量。

## fig03_dy_rx_robust_z

**EN.** Figure 3. Isolation residual robust-z for dy (residual_y) and rx (residual_ty) versus LHC fill, using the frozen 14973/14974 reference scale.  All statistically sufficient runs stay well inside |z|<3.  There is no persistent same-sign drift and no alignment_drift_candidate.  14977 is drawn hollow because it is insufficient statistics, not a detector-condition change.  These residuals are DQ observables only and are not inverted into a geometry correction.

**CN.** 图 3. Isolation residual 的 dy（residual_y）和 rx（residual_ty） robust-z 随 LHC fill 的变化，参考尺度冻结自 14973/14974。所有统计充分的 run 都远低于 |z|<3。没有持续同号漂移，也没有 alignment_drift_candidate。14977 用空心点表示，因为它是统计不足，不是探测器工况变化。这些 residual 只是 DQ observable，不会被反演成 geometry 改正。

## fig04_jacobian_spectrum

**EN.** Figure 4. Singular spectrum of the real-data Station Jacobian rebuilt from the already-captured finite-difference probes on 14973 and 14974.  The six-DoF spectrum has a near-null value whose direction is almost pure dz (gauge-like).  Dropping dz lifts the 14973 condition number to 114, but the 14974 five-DoF spectrum remains weak along a direction that is almost pure dx.  This is an identifiability diagnostic, not a closure test.  Implied C_dx is not plotted as a measurement.

**CN.** 图 4. 由 14973/14974 已捕获有限差分探针重建的真实数据 Station Jacobian 奇异谱。六自由度谱存在接近零的奇异值，方向几乎是纯 dz（规范型）。去掉 dz 后 14973 条件数降到 114，但 14974 的五自由度谱仍沿几乎纯 dx 的方向偏弱。这是可识别性诊断，不是 closure。Implied C_dx 不作为测量值绘制。

## fig05_reduced_inconsistency

**EN.** Figure 5. Self-nulling parameter updates of the reduced {dy,rx,rz} mode on the two independent calibration runs.  The mode is full rank and nearly orthogonal to frozen A, but the updates disagree in sign and magnitude (dy +1.08 mm versus −8.93 mm; rz −4.3 mrad versus +95 mrad; maximum separation ~258σ).  The linearized 14974→14973 χ² ratio is ~16.  Any χ² decrease in the opposite direction is a DQ observable only and is not alignment closure.  The mode is therefore not a writable geometry correction.

**CN.** 图 5. reduced {dy,rx,rz} 模式在两个独立 calibration run 上的 self-nulling 参数更新。该模式满秩且几乎与冻结 A 正交，但更新的符号和量级不一致（dy +1.08 mm 对 −8.93 mm；rz −4.3 mrad 对 +95 mrad；最大约 258σ）。14974→14973 线性化 χ² 比约 16。相反方向上的任何 χ² 下降都只是 DQ observable，不是 alignment closure。因此该模式不能写成 geometry 改正。

## fig06_a_versus_weak

**EN.** Figure 6. Geometry of the frozen C_dx→station leakage operator A and the real-data Station weak directions.  Left: after dropping dz, the 14974 five-DoF weakest right-singular vector is almost pure dx and has cosine 0.994 with unit-normalized A.  Right: the six-DoF weakest direction on both calibration runs is almost pure dz.  The alignment of the remaining weak direction with A is the empirical reason a track-driven station update is cross-level contaminated.  A is not retuned, and the figure is not a C_dx measurement.

**CN.** 图 6. 冻结 C_dx→station 泄漏算子 A 与真实数据 Station 弱方向的几何关系。左：去掉 dz 后，14974 五自由度最弱右奇异向量几乎是纯 dx，与单位化 A 的余弦为 0.994。右：两个 calibration run 的六自由度最弱方向几乎都是纯 dz。剩余弱方向与 A 对齐，是 track-driven station 更新被跨层污染的实证原因。A 不重调，本图也不是 C_dx 测量。
