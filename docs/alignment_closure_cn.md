# 固定 Truth 的 Station-x/y Alignment Closure

## 当前 Physical Closure

所需的 displaced-geometry local-segment refit 已从持久化 `SCT_ClusterContainer`
实现。对于 MC24 station 3 `+1 mm` payload，refit 加 field-aware propagation 在 27 条
truth-matched pair 上以 `4.9e-13 mm` 的可动 station 最大误差恢复注入偏移。physical chain
使用全部 station pair，不再有历史 reference-source-only 限制。复现命令与边界见
[偏移几何 Segment Refit](displaced_geometry_refit_cn.md)。

## 状态与边界

这里没有使用 learned association model。所有研究都固定 truth association，只求解
station-level `delta_x, delta_y`，并固定 IFT（station 0）为 reference。

历史开发中有两个受控验证层次：

1. 原始 residual-level solver 用明确约定
   `r_injected = r_nominal + delta_target - delta_source` 验证线性的
   weighted least-squares objective。
2. 当前 payload-calibrated coordinate-level study 将同一约定锚定到真实 Calypso
   `/Tracker/Align` SQLite/POOL payload，然后把已验证平移应用到 canonical
   tracklet 文件。

这两个历史层次都不是 raw-hit deformed-geometry reconstruction。新的 cluster-to-segment
refit 已在 V1 刚体平移研究中跨过该限制，而无需回到 RDO。历史 control 及其含义见
[conditions-payload 验证（英文）](condition_payload_alignment.md)和
[中文版本](condition_payload_alignment_cn.md)。

## 当前 Payload-Calibrated Closure

保留的控制 payload 对 station 3 写入 global `delta_x = +1.0 mm`、
`delta_y = 0.0 mm`，其 condition chain 已在 Calypso 中验证。独立的 canonical
coordinate surrogate 再应用：

```text
x' = x + delta_x(station)
y' = y + delta_y(station)
```

使用 mode-1（MC truth q/p）propagation record，并要求 truth-match fraction 至少
`0.99`。为使 source state 与 nominal propagation record 保持一致，只使用 source 为
reference station 0 的 pair。

| 指标 | 结果 |
| --- | ---: |
| Reference filter 前的 truth-matched pair | 30 |
| Reference-source pair | 15 |
| Coordinate-increment 最大误差 | `8.9e-16 mm` |
| Normal-matrix rank | 6 |
| 恢复的 station-3 offset | `[+1.0, 0.0] mm` |
| 可动 station 最大恢复误差 | 数值零 |

这验证 V1 coordinate-level objective 与 transform sign，不量化 detector-level
alignment closure。

## 当前 Coordinate-Level Capture Scan

每个 magnitude 使用 100 个随机方向、三次 refinement、chi-square gate=25、零附加
noise 及 `0.01 mm` tolerance，reference-source surrogate 的结果为：

| 每个可动 station 的平移 [mm] | Capture fraction | 平均 active-pair fraction |
| ---: | ---: | ---: |
| 0、0.1、0.5、1、2、5 | 1.00 | 1.00 |
| 10 | 0.39 | 0.624 |
| 20 | 0.05 | 0.214 |
| 50 | 0.01 | 0.082 |

这些数值不是物理 detector capture range；它们依赖当前 pair covariance 与
reference-source 限制，只用于判断受控 V1 optimizer 是否准备好进入下一项数据产品验证。

## 复现

```bash
cd /eos/home-x/xcheng/FASER
source alignment_ML/scripts/setup_environment.sh ml
cd alignment_ML

python scripts/run_payload_alignment_closure.py \
  --nominal-tracklets outputs/mc24_muon_fasernu_5events_fieldaware_v4_truthcontrol/tracklets.root \
  --observed-tracklets outputs/mc24_muon_fasernu_5events_physical_station3_dx1mm/coordinate_injected_tracklets.root \
  --propagations outputs/mc24_muon_fasernu_5events_fieldaware_v4_truthcontrol/propagations.root \
  --payload-manifest outputs/mc24_muon_fasernu_5events_physical_station3_dx1mm/payload/alignment_payload.json \
  --output-dir outputs/mc24_muon_fasernu_5events_physical_station3_dx1mm/payload_coordinate_closure_rerun

python scripts/run_payload_coordinate_capture_scan.py \
  --tracklets outputs/mc24_muon_fasernu_5events_fieldaware_v4_truthcontrol/tracklets.root \
  --propagations outputs/mc24_muon_fasernu_5events_fieldaware_v4_truthcontrol/propagations.root \
  --payload-manifest outputs/mc24_muon_fasernu_5events_physical_station3_dx1mm/payload/alignment_payload.json \
  --output-dir outputs/mc24_muon_fasernu_5events_physical_station3_dx1mm/payload_coordinate_capture_scan_rerun
```

每个命令都会保存 resolved configuration、JSON metrics、图以及 pair 或 scan
diagnostic；已存在的 artifact path 不会被覆盖。

## 下一步 Physical Validation

单一 payload 的 physical closure 已完成。在比较 sequential pipeline 与 joint alignment
model 前，需要在多个 dx/dy payload 上重复 cluster-to-segment refit 与 fixed-truth closure，
完成 physical capture-range scan。之后才可引入 association ambiguity、synthetic overlay、MLP
或 Transformer。
