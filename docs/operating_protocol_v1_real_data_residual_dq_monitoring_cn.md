# Operating Protocol V1 真实数据 residual/DQ monitoring

条目 52 / 2026-08-23。条目 51 已表明 `{dy,rx,rz}` 在 Fisher 空间上
可辨识、且与冻结 `A` 基本正交，但 self-nulling correction 不能从
14973 搬到 14974。本步把该结论冻结成 production-like DQ protocol。

不再构造任何新的 Station calibration mode。不启动 C_dx Mode。不生成
FD probe，不求 Newton，不写 payload。继续使用冻结的 mode-0、V2
checkpoint、route policy、阈值 `0.001`、`unmatched_penalty=-1.0`、
`physical_edge_deduplicated` 观测，以及当前官方 geometry。

## 冻结不变量

- 传播：mode-0
- association：冻结 V2（`0c85a001…766a27`）
- 观测：`anchor_selected_field_edge` + `physical_edge_deduplicated`
- 几何：官方 `FASERNU-04` / `OFLCOND-FASER-06` / `r0022`
- calibration reference：只用 14973 + 14974
- `geometry_write_allowed=false`
- `station_calibration_mode_available=false`
- `cdx_mode_allowed=false`

禁止：写官方 conditions；写入任何 self-nulling 修正；启动 C_dx Mode；
联合 Newton；新增 layer/module DoF；打开 sealed test；把 residual
下降当成成功；把 `alignment_drift_candidate` 转成 geometry update。

## Residual observables

| 通道 | residual | 角色 |
| --- | --- | --- |
| `dy` | `residual_y_mm` | isolation DQ 观测量 |
| `rx` | `residual_ty` | isolation DQ 观测量 |
| `dx` | `residual_x_mm` | 仅报告，标记 cross-level-sensitive |
| `ry` | `residual_tx` | 仅报告，标记 cross-level-sensitive |
| `rz` | 无 | 没有独立 4-vector 通道，不伪造 |
| `dz` | 无 | 完全不是 track-driven observable |

robust standardized shift 定义为
`(run median − reference median) / (1.4826 × MAD)`，reference 是
14973/14974 合并后的 field-edge residual。这是 DQ 数字，不是修正。

## 报警规则

预先登记，禁止用本监测样本回改阈值。

1. `selected_routes == 0` 且 `n_all_pairs_candidates ≥ 50` →
   `association_or_reconstruction_degradation`
2. `0 < selected_routes < 10` →
   `insufficient_statistics_for_alignment_dq`（不是 alignment 异常）
3. 有 selected route，但单事件占比超过 50% →
   `association_or_reconstruction_degradation`
4. isolation 通道相对 14973/14974 的 `|robust z| > 5` →
   `detector_condition_change`
5. 否则 `nominal_monitoring`

至少两个非 reference、非 insufficient run 上出现同号 isolation
`|robust z| ≥ 3` 时，只标记 `alignment_drift_candidate`，禁止反演成
station payload。

## 第一批监测语料

五个已有 full-segment current-geometry association 构成第一次长期扫描。
邻近 2024 r0022 run（14971、14972、14980、14981、14985、14989、15007）
只是 expansion candidates。它们必须先完成 residual-blind occupancy、
current-geometry Athena 和冻结 V2 association 才能进入同一套报告。
14973–14977 的 occupancy 窗口保持冻结，本步不再重选。

| run | 角色 | selected | 2/3/4-st | 最大事件占比 | `dy` z | `rx` z | 状态 |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| 14973 | calibration_reference | 121 | 38/83/0 | 0.008 | −0.00 | −0.02 | nominal |
| 14974 | calibration_reference | 109 | 34/73/2 | 0.009 | 0.00 | 0.03 | nominal |
| 14975 | monitoring | 156 | 60/95/1 | 0.006 | −0.13 | 0.01 | nominal |
| 14976 | monitoring | 63 | 18/45/0 | 0.016 | −0.07 | 0.03 | nominal |
| 14977 | monitoring | 2 | 0/2/0 | 0.50 | −0.06 | −0.14 | insufficient |

14975/14976 仍在 calibration-reference 带内。以 run 号为时间代理的
Theil–Sen 斜率是 `dy ≈ −0.12 mm/run`、`rx ≈ −1.9×10⁻⁵ /run`，相对
reference robust scale（`dy` 6.71 mm，`rx` 0.00524）不是可重复长期
漂移。本语料没有 `alignment_drift_candidate`。14977 自动归为
`insufficient_statistics_for_alignment_dq`。

## 唯一结论

**`real_data_residual_dq_monitoring_only`**

- `geometry_write_allowed=false`
- `station_calibration_mode_available=false`
- `cdx_mode_allowed=false`
- 不从当前 self-nulling residual 提取任何 alignment payload

未来重新开启 geometry calibration，必须先满足以下外部前提之一：

1. 独立 survey / external station constraints 能固定 `dx`、`ry`、`dz`，
   **并且** 独立证明 `C_dx` 落在 1.5–1.7 µm isolation budget 内；或
2. 提供新的独立真实数据 topology / track sample，使跨 run transferable
   subspace 得到实证。

在这些条件满足前，当前真实数据只支持 residual/DQ monitoring。

## 输出

`outputs/operating_protocol_v1_real_data_residual_dq_monitoring_v1/`

- `run_level_dq_report.json`
- `time_stability_report.json`
- `alignment_drift_candidate_report.json`
- `operating_protocol_monitoring_decision.json`
