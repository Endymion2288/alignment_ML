# Geometry-Aware Transformer V1：实现、冻结与多方向测试

## 本轮目标

在已经建立的真实物理链上正式启动 Geometry-Aware Transformer V1，并严格保持：

1. 原始 xAOD source-file 的 train/validation/test 隔离；
2. cluster -> segment refit -> mode-0 Acts propagation 的 physical 数据定义；
3. 既有未加 gate 的 physical candidate graph；
4. 既有 IFT -> S1 -> S2 -> S3 相邻站 unit-capacity route assignment；
5. validation-only 的模型、calibration、threshold 与 dustbin penalty 选择；
6. 只在所有选择冻结后执行一次多方向 test scan。

本轮没有重跑 Calypso refit、没有 coordinate/residual surrogate、没有用 local q/p
作为可靠 measurement、没有在 test 上调整模型或运行点。

## 实现

新增的核心模块：

- models/encoder.py：带 learnable station embedding 的 128 维 tracklet/state encoder。
- models/geometric_attention.py：只在已有 candidate edge 上计算的 8-head sparse attention；
  支持 MLP(g_ij) 和可开关的 -lambda*chi2/(2*tau) 项。
- models/transformer.py：固定 4 个 block、d_model=128、FFN 128->256->128 的 V1。
- training/geometry_aware_transformer.py：物理图构造、train-only standardizer、
  weighted BCE/focal、hard-negative weight、curriculum、Platt calibration。
- training/transformer_route_selection.py：validation-only route operating-point 搜索；
  先用确定 subset 提议，再在完整 validation 上重排前 24 个候选。
- scripts/train_geometry_aware_transformer_v1.py：明确只加载 train/validation event。
- scripts/run_frozen_geometry_aware_transformer_v1_scan.py：仅 test inference；
  拒绝 test-time training、calibration 或 threshold selection。
- evaluation/pairwise_metrics.py：增加 station-pair Platt scaling。
- configs/geometry_aware_transformer_v1*.yaml：V1、local pretrain、Platt、完整
  validation rerank 与最终 ablation 配置。

图的 node feature 使用持久化 state、z、covariance diagonal、local fit quality、hit count
和六个 layer-side bit。attention 的 g_ij 使用已验证的
rx、ry、rtx、rty、四个 pull、log1p(chi2)、combined covariance logdet 和 dz；
station-pair 与 message direction 由 embedding 表示。所有 six station-pair candidate
都是 full-event message edge，输出只对相邻 0->1、1->2、2->3 edge 打分。

geometry-aware 配置额外有 local edge residual decoder，并先在同一物理 train curriculum
上预训练；它不读取 truth、MLP score 或 fake role。no-multi-station-context 保留此 decoder，
只将每个相邻 station pair 做成独立二站图。因此 full 与 no-context 的比较是当前
cross-station message 的主要消融。ordinary sparse control 关闭了 geometry bias 和
local decoder，必须把它解释为非几何参考，不能把二者差异完全归因于一项 attention bias。

## 数据与封存契约

主训练 manifest：

    outputs/mc24_muon_2dfluka_curriculum_external_full_train_aug_synthetic_v1/
    synthetic_corpus_manifest.json

source-file split 为 train/validation/test = 10/2/2，source event = 99/19/19。
validation run contract 记录：

    loaded_event_splits = [train, validation]
    test_events_loaded = false
    test_opened = false
    candidate_graph = existing_mode0_acts_physical_candidates_all_six_station_pairs

test 使用历史冻结 MLP route scan 完全相同的物理多方向 manifest：

    outputs/mc24_muon_2dfluka_multidirection_test_synthetic_controlled_v2/
    synthetic_corpus_manifest.json

test source 仍仅为 mc24_100116_00030_00039 和 mc24_100117_00030_00039。
0 mm 只有一个真实 refit trial；0.1、1、5、10、50 mm 各有三个独立 dx/dy direction。
所有 trial 的物理几何响应在前一轮已经由 payload -> segment refit -> Acts audit 验证。

## Validation-only 选择

四个模型均通过 validation raw AP、station-pair Platt calibration 和完整 validation route
重排后冻结。所有模型的 selection_split/fit_split 均为 validation_only，test_opened=false。

| 变体 | raw AP | raw AUC | validation nominal eff | purity | fake |
| --- | ---: | ---: | ---: | ---: | ---: |
| ordinary sparse | 0.8153 | 0.9304 | 0.570 | 0.910 | 0.161 |
| geometry-aware | 0.8726 | 0.9429 | 0.753 | 0.940 | 0.088 |
| geometry-aware，无 raw chi2 项 | 0.8726 | 0.9430 | 0.744 | 0.940 | 0.087 |
| geometry-aware，无 multi-station context | 0.8716 | 0.9432 | 0.655 | 0.891 | 0.118 |

预声明 primary 为 eff >= 0.70、purity >= 0.95、inclusive fake <= 0.05。没有任何
Transformer 在 validation nominal 通过。即使某个 test direction 表现较好，也没有用它
重新选阈值或重新标记模型为通过。

geometry-aware 的 raw chi2 hyperparameter 在 validation AP 上选择为
lambda=1、tau=2.0e6；no-chi2 结果几乎相同，说明当前尺度的显式项贡献很小。

## 一次封存 test scan

最终输出：

    outputs/geometry_aware_transformer_v1f_final_multidirection_test_v1/

根 frozen_evaluation_contract.json 记录：

    physical_geometry_repropagation = true
    q_over_p_mode = 0
    candidate_chi2_gate = null
    no_test_time_model_selection = true
    no_test_time_calibration = true
    no_test_time_threshold_selection = true

每个 Transformer 目录还保存 checkpoint、validation model selection、validation calibration
和 validation operating point 的 SHA-256。MLP reference 的 manifest hash 与此次 test manifest
相同。

因为所有方法共享物理 candidate graph，raw complete truth-chain candidate recall 在所有
幅度均为 1.0。capture 成功数如下：

| 方法 | 0 | 0.1 | 1 | 5 | 10 | 50 mm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 冻结 pairwise MLP | 1/1 | 3/3 | 2/3 | 0/3 | 0/3 | 0/3 |
| ordinary sparse | 0/1 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| geometry-aware | 1/1 | 1/3 | 2/3 | 0/3 | 0/3 | 0/3 |
| geometry-aware，无 raw chi2 | 1/1 | 1/3 | 2/3 | 0/3 | 0/3 | 0/3 |
| geometry-aware，无 multi-station context | 1/1 | 3/3 | 2/3 | 0/3 | 0/3 | 0/3 |

full geometry-aware 在 5 mm 的方向均值为 threshold truth-chain recall=0.901、
complete efficiency=0.705 +/- 0.021、purity=0.985 +/- 0.003、fake=0.065 +/- 0.005、
missing recovery=0.973 +/- 0.004；10 mm 对应 0.884、0.632 +/- 0.085、
0.974 +/- 0.005、0.069 +/- 0.017、0.961 +/- 0.004。它相对 MLP 在 5/10 mm 的
efficiency 与 score retention 有改善，但 fake rate 超过固定 0.05 条件，capture fraction
仍为 0。50 mm 时 full model 没有选出 complete route。

no-multi-station-context 在 test 的 nominal、0.1 与若干中等偏移 route metric 上反而比
full context 高。这不是选择新模型的证据，因为 validation primary 已经失败；它是需要在
下一轮独立训练/验证中诊断的反例。

详细 station-pair 结果均已写到每个模型的 station_pair_trial_metrics.csv，包含
0->1、1->2、2->3 的 candidate recall、AUC/AP、ECE、association efficiency/purity/fake
rate 与 unmatched 指标。根 comparison_magnitude_summary.csv 和两个 PNG 统一保存
capture fraction 与 complete-track efficiency 曲线。

## 结论

本轮正式实现、训练、冻结并测试了用户要求的 Geometry-Aware Transformer V1 与四个
ablation。结果没有证明 multi-station Transformer 扩大 physical association/alignment
capture range：5、10、50 mm 的所有 Transformer capture fraction 都为 0；full context
也没有胜过其 no-context 消融。

这是需要保留的负结果，而不是调 test threshold 的理由。V1 test 已封存，后续不能再使用
这批 test source 做 architecture、calibration、temperature 或 threshold 搜索。下一步应在
train/validation 上先研究 full context 退化的原因，再准备独立 test bank 验证一个预先冻结的
V2 假设。

