# 2026-08-21 (54) 四站 matched V2 retraining：source-disjoint association 闸

## 任务边界

条目 53 已提交 15 维相对 curriculum 的真实物理生产。本条目在 Condor
完成后做 assemble / overlay / 冻结 V2 对照 / GPU matched retraining /
validation association 闸。**没有**打开 15 维未知关联 WLS，**没有**用条目
52 的两个 held-out overlay 选 threshold / unmatched penalty / calibration。

## 物理 bank 与 overlay

Cluster 1000434：4 source × 7 点 = 28/28 完成，0 `failure.json`。
`nevents=50`。train 99 个物理事件、validation 100 个。

Overlay 配方与冻结 V2 相同：
`configs/physical_alignment_iteration_four_station_relative_trainval.yaml`，
`events_per_payload: 120` × 2 source、3 轨/事件。14 个 sample（7 train + 7
validation）。条件轴 `four_station_relative_l2`，幅度 0（nominal）或 1
（相对族及其 left-SE(3) twin）。密封 test 未打开。

## 冻结历史 V2 对照（同一新 validation overlay，冻结 OP）

Checkpoint
`/eos/home-x/xcheng/FASER/alignment_ML/outputs/mc24_v3_expanded_trainval_v2_bce_control_v1`
（sha256 `0c85a001…`），阈值 0.001，unmatched penalty −1.0。Raw chain
recall 全部 1.0。

| payload | efficiency | 相对 nominal 下降 | 2→3 | purity | fake |
| --- | ---: | ---: | ---: | ---: | ---: |
| reference | 0.714 | — | 0.604 | 0.971 | 0.031 |
| hard_s3_ry | 0.584 | 0.130 | 0.508 | 0.947 | 0.041 |
| hard_s3_ry_plus_common | 0.706 | 0.009 | 0.593 | 0.929 | 0.066 |
| draw_00 | 0.677 | 0.037 | 0.589 | 0.940 | 0.056 |
| draw_00_plus_common | 0.742 | −0.028 | 0.625 | 0.930 | 0.064 |
| draw_01 | 0.539 | 0.175 | 0.448 | 0.912 | 0.065 |
| draw_01_plus_common | 0.504 | 0.210 | 0.431 | 0.896 | 0.086 |

同一 `ΔT_ij` 的 twin **不一致**：`hard_s3_ry` 0.584 vs plus-common 0.706。
冻结 V2 仍在用「哪一站接近 nominal」这类绝对图伪特征。对照闸未过，失败类
`association_domain_shift`。

## Retrained V2

架构 / `residual_v1` / unit-capacity / 30 epoch 预算不变。best epoch 27。
Validation 选 OP（允许，且不是条目 52 held-out）：

- unmatched penalty **0.5**（历史 −1.0）
- 阈值 `0→1=0.001`，`1→2=0.5`，`2→3=0.5`

历史 capture_success（efficiency≥0.70、purity≥0.95、fake≤0.05）在 0/1 两个
幅度上都是 false：新 OP 把 fake 抬到 ~0.10–0.13。这是 validation 网格搜索的
结果，不是再调 unmatched penalty 去硬拉 2→3。

Checkpoint sha256
`ec3d40ea43a533e4883becf256d32a7e1966f1bd226a01988f42cced24c1f9a4`。

| payload | efficiency | 相对 nominal 下降 | 2→3 | 2→3 下降 | purity | fake |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| reference | 0.747 | — | 0.891 | — | 0.948 | 0.103 |
| hard_s3_ry | 0.701 | 0.045 | 0.859 | 0.032 | 0.936 | 0.111 |
| hard_s3_ry_plus_common | 0.727 | 0.019 | 0.838 | 0.052 | 0.939 | 0.114 |
| draw_00 | 0.682 | 0.065 | 0.864 | 0.027 | 0.921 | 0.127 |
| draw_00_plus_common | 0.643 | **0.104** | 0.813 | 0.077 | 0.934 | 0.130 |
| draw_01 | 0.660 | 0.087 | 0.847 | 0.044 | 0.924 | 0.124 |
| draw_01_plus_common | 0.665 | 0.082 | 0.854 | 0.037 | 0.930 | 0.121 |

Raw chain recall 仍全部 1.0。2→3 / S3 **恢复**：hard 例从冻结 0.508 到
0.859，且每个 payload 都有 2→3 选中边。Gauge twin 全部通过预注册差限
（efficiency 差 ≤0.039，2→3 差 ≤0.051）。

未过的闸只有 complete-track efficiency vs-nominal：`draw_00_plus_common`
下降 **0.1039 > 0.10**。purity / fake 的 vs-nominal 界限全部通过。因此
`continue_to_15d_relative_wls=false`，失败类仍记为
`association_domain_shift`。`representation_gap_suspect=false`：candidate
recall=1 时 2→3 **没有**系统性掉光。

新 OP 的 2→3 阈值 0.5 会丢掉一部分完整 truth chain 的 score-retention
（例如 plus-common 352/462），这与条目 52 冻结 OP 下 473/473 不同。禁止为此
再调 unmatched penalty。

## 对条目 52 那个问题的回答

四站同时错位造成的 **2→3 / S3 崩塌，主要是历史 V2 的训练分布失配**：
matched retraining 把它拉了回来，并且同一 `ΔT_ij` 的 left-SE(3) twin 变得
统计一致。现有 V2 表示 **不是**完全缺少 relative four-station 归纳偏置。

它也 **还不够** 被冻结进 15 维未知关联 WLS：预注册 efficiency 下降 ≤0.10
在一个 gauge-control 点上差 0.004，且 validation 选出的 OP 明显更吵
（fake ~0.11）。下一步若继续，应在不碰条目 52 held-out、不靠 unmatched
penalty 硬拉的前提下，诊断该点的 route packing / score 校准，而不是宣称
15 维几何已经恢复。

## 产物

- 物理 bank：`outputs/mc24_four_station_relative_association_retrain_v1/`
- overlay：`.../overlay_synthetic_v1/`
- 冻结对照：`.../frozen_v2_control/`
- retrained V2：`.../retrained_v2/`
- per-payload 推理：`.../retrained_v2_validation/`
- 闸门：`.../association_gate_decision.json`
