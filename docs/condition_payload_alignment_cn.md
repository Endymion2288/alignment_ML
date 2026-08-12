# Station Alignment Conditions Payload 与 Coordinate-Level Closure

## 状态更新：Physical Refit 已可用

原有 coordinate-level 研究仍是有价值的受控 surrogate，但它不再是唯一的 geometry response
路径。`--refit-segments` 现在读取持久化的 `SCT_ClusterContainer`，在当前 SCT/ACTS
conditions 下重建 `SegmentFitRefit` 和 `SegmentsRefit`，并导出新的 field-aware propagation
record。MC24 station 3 `+1 mm` physical closure 已用 20 条 refit tracklet 和 27 条
truth-matched pair 完成。数据契约、命令、原始审计和结果见
[偏移几何下的 Segment Refit](displaced_geometry_refit_cn.md)。

## 已验证内容

V1 需要一个由 Calypso conditions 锚定的 station-level `delta_x, delta_y` 约定，不能只
在 Python residual 公式中人为规定。工作区的 `TrackerAlignDBTool` 现接受
`station:<id>` alignment constant，其六个 global 分量为
`[dx_mm, dy_mm, dz_mm, rx_rad, ry_rad, rz_rad]`；V1 只接受 `dx_mm` 和 `dy_mm`。
payload writer 会生成 `/Tracker/Align` 的 SQLite、POOL 文件、XML catalog 与记录该约定的
JSON manifest。

已写出 `station:3 = [1, 0, 0, 0, 0, 0]` 的受控 payload，并交给 MC24 exporter。job log
确认条件链的必要步骤均已执行：`/Tracker/Align` 的 SQLite override、
`SCTAlignmentStore` 创建、`FaserActsAlignment` 创建以及 event processing 成功。

## 历史 Exporter-Only Control

在加入 refit chain 前，condition payload 已到达 SCT 和 ACTS condition algorithm，但 exporter
仍从 xAOD 读取持久化 local segment `TrackParameters`。state 没有改变，这是预期结果；该历史
negative control 保留在
`outputs/mc24_muon_fasernu_5events_physical_station3_dx1mm/condition_runtime_audit.json`。

它证明只重跑 exporter 不是 deformed-geometry reconstruction。新的 refit path 已在持久化
cluster 层跨过这个边界，且无需 raw-hit reconstruction。

## V1 Coordinate-Level Surrogate

`apply_station_alignment_payload.py` 读取已验证 manifest，生成新的 canonical ROOT 文件：

```text
x' = x + dx_station
y' = y + dy_station
```

它有意保持 slope 和 covariance 不变。这是仅平移的 coordinate-level surrogate，不是
reconstruction rerun。为避免使用不一致的 shifted source state，closure 与 scan 只使用
source 为零偏移 reference IFT（station 0）的 truth-matched propagation pair。

对于保留的 station 3 `+1 mm` payload，15 条 reference-source pair 通过 truth-match
要求（`>= 0.99`）。观测到的 coordinate increment 与 manifest 在 `8.9e-16 mm` 内一致；
reference-fixed weighted solver 以 rank 6 恢复 station 3 的 `[+1.0, 0.0] mm`。

## Coordinate-Level Capture Scan

payload-calibrated scan 使用 mode-1 MC truth q/p propagation record、固定 truth
association、每个 magnitude 100 个随机 offset direction、三次 refinement、chi-square
gate=25 和 `0.01 mm` recovery tolerance，共有 15 条 reference-source pair。

| 每个可动 station 的平移 [mm] | Capture fraction | 平均 active-pair fraction |
| ---: | ---: | ---: |
| 0、0.1、0.5、1、2、5 | 1.00 | 1.00 |
| 10 | 0.39 | 0.624 |
| 20 | 0.05 | 0.214 |
| 50 | 0.01 | 0.082 |

这只是当前 **coordinate-level、reference-source surrogate** 的 capture range，不能解释为
真实 FASER detector capture range。物理 scan 要求从 displaced hits 重新 fit local segment，
并从相应 displaced source surface 传播。

## 复现

创建新的 payload directory；所有命令都拒绝覆盖既有 artifact：

```bash
cd /eos/home-x/xcheng/FASER
source alignment_ML/scripts/setup_environment.sh calypso

python alignment_ML/scripts/write_station_alignment_payload.py \
  --output-dir alignment_ML/outputs/example_station3_dx1mm/payload \
  --offset 3:1.0:0.0
```

生成的 SQLite 与 catalog 可通过 workspace exporter 的
`--tracker-align-sqlite` 和 `--tracker-align-pool-catalog` 参数使用。对于 V1 surrogate，
切换到 ML 环境后对既有 canonical 文件应用 manifest：

```bash
source alignment_ML/scripts/setup_environment.sh ml
cd alignment_ML

python scripts/apply_station_alignment_payload.py \
  --input outputs/mc24_muon_fasernu_5events_fieldaware_v4_truthcontrol/tracklets.root \
  --output outputs/example_station3_dx1mm/coordinate_injected_tracklets.root \
  --payload-manifest outputs/example_station3_dx1mm/payload/alignment_payload.json
```

运行 `run_payload_alignment_closure.py` 或
`run_payload_coordinate_capture_scan.py` 前，必须先阅读 JSON metadata 中的
`deformed_geometry_repropagation=false` 与 `reference_source_only=true`。

## 下一步 Physical Scan

所需的 geometry-level refit 现已实现并验证。下一项剩余工作是 physical capture-range scan：
为多个 dx/dy 建立独立 payload，对每个 payload 重跑同一个 cluster-to-segment refit chain，
并对每份输出做 fixed-truth closure。不得把上文历史 coordinate-level capture table 解释为
detector-level scan。
