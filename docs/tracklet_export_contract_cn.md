# Tracklet 导出契约

## 范围

V1 exporter 读取已有的、经过 ghost-busting 的 `Segments` local-track
collection，不重新进行 hit finding 或 global tracking。Calypso 先在正常 `nt`
tree 中写入 event-wise `Tracklet_*` vector，再由 Python converter 输出
`tracklets` flat tree，每行对应一条 local tracklet。

## 必需列

| 字段组 | 必需列 | 来源要求 |
| --- | --- | --- |
| 标识 | `run_id`, `event_id`, `station_id`, `tracklet_id` | `station_id` 必须来自关联 cluster 的 `FaserSCT_ID::station()`，不能由 z 区间猜测。 |
| 状态 | `x_mm`, `y_mm`, `z_mm`, `tx`, `ty` | position 和 direction 必须使用同一 fitted reference surface。 |
| 协方差 | `[x, y, tx, ty]` 顺序的十个 `cov_*` 列 | exporter 必须记录变换方法，并拒绝无效矩阵。 |
| 质量 | `chi2`, `ndof`, `n_hit`, `hit_pattern` | `hit_pattern` 是 station 内 layer bit mask。 |
| 溯源 | 可选 `module_ids`, `raw_hit_ids`, `source_file_id`；仅 synthetic 使用的 `origin_run_id`、`origin_event_id`、`origin_tracklet_id` | V1 探测器导出尚未写入 module/raw-hit 列。synthetic overlay 保留 origin identity，使同一 payload 的 Acts prediction 能够映射且不需要 coordinate-level propagation。增强输出保留逐 hit 的 module/cluster identifier；不把它宣称为 raw RDO ID。 |
| MC 标签 | `truth_particle_id`, `truth_pdg`, `truth_match_fraction` | 对每条 local segment 使用 `ITrackTruthMatchingTool`；未匹配的 data/fake 使用 `-1`。 |

## 已实现的导出通路

Calypso 修改通过 `faser_ntuple_maker.py --export-tracklets` 显式启用；默认
PHYS branch 不变，只有启用后才在 event-wise `nt` tree 中新增以下 vector：

- `Tracklet_station_id`、state、quality 和 `Tracklet_id` 来自同一条
  ghost-busted segment。station 从每个 `FaserSCT_ClusterOnTrack` 的
  identifier 获得；若一个 segment 跨越多个 station，会被拒绝。
- native track-parameter covariance 从 `(loc1, loc2, phi, theta, q/p)` 通过
  central numerical Jacobian 变换到 global `[x, y, tx, ty]`。每次扰动均由
  关联的 `Trk::Surface` 构造，因此会保留实际 module orientation。V1 state
  不含动量大小坐标，所以 `q/p` 的 Jacobian 列为零。无法安全变换时写入
  `Tracklet_has_covariance=false` 和 NaN covariance；converter 默认丢弃该行。
- 可选 q/p audit branch 保存 native `q/p`、由 reconstructed momentum 得到的等价值
  及 q/p covariance validity flag；原始 MC22 electron smoke test 不要求这些列。启用
  `--export-tracklet-propagation` 时，还会使用 `FaserActsExtrapolationTool` 对
  truth-matched source/target segment pair 做外推，并以独立 event-wise vector 写入
  field-aware validation 通路。
- `TrackletHit_*` vector 保存每个 cluster 的 station/layer/module metadata。
  `TrackletHit_cluster_identifier` 是 reconstruction-level cluster
  identifier，不宣称为 raw RDO identifier。在确认 RDO list 的 persistence
  语义前，不把它当作 raw hit ID。
- 仅对 MC，`Tracklet_truth_*` 对每条 local segment 独立调用一次
  `ITrackTruthMatchingTool::getTruthParticle(segment)` 获得；它与已有的
  global-track `t_barcode` branch 无关。

flat conversion 的可重复命令为：

```bash
source alignment_ML/scripts/setup_environment.sh ml
cd alignment_ML
python -m scripts.convert_ntuple_tracklets INPUT-PHYS.root \
  --output data/tracklets.root --include-truth
python -m scripts.inspect_root_schema data/tracklets.root
python -m scripts.audit_tracklets data/tracklets.root --require-mc-labels
python -m scripts.run_chi2_baseline \
  --input data/tracklets.root --config configs/baseline_chi2.yaml \
  --output-dir outputs/chi2_mc
```

`--include-truth` 只适用于 MC 导出；真实数据不应加入它。converter 会拒绝
event-wise branch 长度不一致的输入，默认只丢弃被显式标记为缺少 covariance
的行。

## Geometry 契约

每次运行还需要一个版本化的 geometry YAML/JSON，记录各 station nominal
transform、geometry tag、conditions tag 和 reference station。V1 alignment
只含 station-level `dx_mm`、`dy_mm`，其中一个 station 固定为 `(0, 0)`。

## 当前状态

已有 PHYS tree 有 local segment 的 position、momentum、chi-square、ndof，
但没有上述显式列。opt-in Calypso exporter 现已编译、安装，并已通过 canonical
loader 和 chi-square baseline 完成小样本 xAOD 导出验证。它不会伪造缺失的
covariance 或 station label。

两个验证样本被刻意分开处理：MC22 100 GeV 电子验证了 segment-level truth 和
electron 导出通路，但只实际给出 station ID `1,2,3`；MC24 FASERnu muon 验证了
station ID `0,1,2,3` 和 IFT 通路，但不是电子训练数据。最初检查的 MC24 FLUKA
slice 前 10 个 event 没有 local segment。命令和准确结果见
[baseline 验证](baseline_validation_cn.md)。

数值 station-transform sidecar、raw RDO ID，以及 IFT+三站 electron MC 输入
仍是未解决的关键项。当前允许在 MC24 muon 样本上进行受控的、truth-fixed、
baseline-subtracted residual-level x/y closure，但它不能替代 deformed-geometry ACTS
closure。在 covariance calibration、physical geometry injection 与受控 multi-track
association baseline 验证前，不应开始 Transformer association training。
