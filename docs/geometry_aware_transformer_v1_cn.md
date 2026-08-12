# 几何感知稀疏 Transformer V1

## 范围与冻结输入契约

这是物理四站 route baseline 之后的第一版 Transformer 研究。它不改变已有数据
定义、传播、candidate graph、route solver 或 source split。

每条 tracklet 与 candidate 都来自已经物理化的链路：

    SCT_ClusterContainer -> SegmentFitRefit -> SegmentsRefit -> NtupleDumper
    -> FaserActsExtrapolationTool (mode 0)

candidate graph 是既有 mode-0 Acts 六个前向 station pair 的全部未加 gate 候选。
local segment 的 q/p 仍只是 propagation seed。没有坐标平移、缓存 residual、
residual-level surrogate、重新 tracking，也没有 test-time candidate construction。

原始 xAOD 文件是唯一 split 单位：train/validation/test 分别为 10/2/2 个 source，
对应 99/19/19 个 source event，三组 source-event UID 严格不重叠。封存 test 使用与
冻结 MLP route baseline 完全相同的多方向物理 manifest：0 mm 一个 trial，每个非零
幅度各三个独立的 dx/dy direction trial。

## V1 模型

TrackletStateEncoder 使用持久化 state、covariance diagonal、拟合质量、hit 数、
六个 layer-side occupancy bit 和可学习 station embedding，输出为 128 维。稀疏网络
固定为四个 pre-normalized block、八个 attention head 与 128 -> 256 -> 128 FFN。

attention 只在已经存在的物理 candidate edge 上计算；同一前向候选可以产生两个 message
方向：

    QK / sqrt(d) + MLP(g_ij) - lambda * chi2_ij / (2 * tau)

几何向量为 [rx, ry, rtx, rty, pull_x, pull_y, pull_tx, pull_ty, log1p(chi2),
logdet(S), dz]。attention MLP 还接收 station-pair 与 message-direction embedding。
raw chi-square 物理项可以独立开关。edge scorer 只输出 0->1、1->2、2->3 的概率；
后端完全复用既有带 dustbin 的相邻站 unit-capacity route assignment。

三个 geometry-aware 配置先在相同物理 feature 与 train split 上训练一个小的 local edge
residual head，再保留它训练 sparse context stack。它从不读取 truth、已有 MLP score
或 synthetic role。因而 no multi-station context 是 cross-station message 的受控消融。
ordinary sparse control 同时关闭 geometric attention bias 与 local residual decoder，
所以它是非几何参考，不是与 full model 只差一个变量的比较。

损失为 weighted BCE（focal_gamma=0），以 event graph 等权采样，正样本不平衡权重
上限为 30，hard negative 权重为 2。local 阶段在物理 0/0.1 -> 5 -> 50 mm curriculum
上训练 25/30/35 epochs；sparse-context 阶段使用相同 curriculum 训练 8/10/12 epochs。
模型训练使用 CUDA。

## 仅 validation 的模型选择

所有 feature standardizer 只由 train graph 拟合。geometry chi-square 的两个候选
（lambda=1，tau=0.5e6 与 2.0e6）按 validation candidate AP 选择。每个 station pair
的单调 Platt calibration，以及 route threshold/dustbin penalty，也只在 validation 上
拟合。先由确定性 event subset 提议有限控制量，再将前 24 个候选在全部 validation event
上重新排序，随后冻结运行点。

| 变体 | validation raw AP | validation raw AUC | 选定 nominal efficiency / purity / fake rate |
| --- | ---: | ---: | ---: |
| Ordinary sparse Transformer | 0.8153 | 0.9304 | 0.570 / 0.910 / 0.161 |
| Geometry-aware | 0.8726 | 0.9429 | 0.753 / 0.940 / 0.088 |
| Geometry-aware，去 raw chi2 项 | 0.8726 | 0.9430 | 0.744 / 0.940 / 0.087 |
| Geometry-aware，去 multi-station context | 0.8716 | 0.9432 | 0.655 / 0.891 / 0.118 |

预先声明的 primary point 为 efficiency >= 0.70、purity >= 0.95、inclusive fake rate
<= 0.05。四个 Transformer 变体在 validation 都没有通过。这个事实保留，不使用
test 改变模型或运行点。

## 一次封存的多方向 test scan

test 输出在 outputs/geometry_aware_transformer_v1f_final_multidirection_test_v1/。
它的 frozen contract 保存 test-manifest hash、所有 checkpoint 与 validation artifact hash、
严格 test-source membership，并明确记录 no_test_time_model_selection=true、
no_test_time_calibration=true 和 no_test_time_threshold_selection=true。

所有方法使用同一个物理 candidate graph，所以各幅度的 raw complete truth-chain
candidate recall 都是 1.0。下表为各方向的 capture 成功数：

| 方法 | 0 mm | 0.1 mm | 1 mm | 5 mm | 10 mm | 50 mm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 冻结 pairwise MLP route baseline | 1/1 | 3/3 | 2/3 | 0/3 | 0/3 | 0/3 |
| Ordinary sparse Transformer | 0/1 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| Geometry-aware Transformer | 1/1 | 1/3 | 2/3 | 0/3 | 0/3 | 0/3 |
| Geometry-aware，去 raw chi2 项 | 1/1 | 1/3 | 2/3 | 0/3 | 0/3 | 0/3 |
| Geometry-aware，去 multi-station context | 1/1 | 3/3 | 2/3 | 0/3 | 0/3 | 0/3 |

对于完整 Geometry-Aware 模型，5 mm 的方向均值为：threshold-chain recall 0.901、
complete-track efficiency 0.705 +/- 0.021、purity 0.985 +/- 0.003、fake rate
0.065 +/- 0.005、missing recovery 0.973 +/- 0.004。10 mm 对应为 0.884、
0.632 +/- 0.085、0.974 +/- 0.005、0.069 +/- 0.017、0.961 +/- 0.004。
它在这两个幅度的 score retention 与 efficiency 高于 MLP，但固定 fake-rate 条件仍然
失败，因而没有证明 capture-range 扩大。50 mm 时完整模型没有选出 complete route。

去除 multi-station context 的消融在封存 test 的小到中等偏移下反而强于 full-context
模型。因此 V1 不支持“当前 sparse multi-station context 是扩大 association capture range
的机制”这一结论。去掉显式 raw chi-square 项的影响很小，这与 validation 选择的
tau=2.0e6 尺度一致。

每个模型目录保留了不丢失信息的详细 route 与 station-pair breakdown：

- trial_metrics.csv：每个方向的 raw/threshold truth-chain recall、complete efficiency、
  purity、fake/duplicate rate 与 missing recovery。
- station_pair_trial_metrics.csv：0->1、1->2、2->3 的 candidate recall、AUC/AP、
  calibration、efficiency/purity/fake rate 和 unmatched 指标。
- magnitude_summary.csv：方向均值、population spread、pooled count 与 capture fraction。
- comparison_magnitude_summary.csv 和两个 PNG：与冻结 MLP baseline 的共同比较。

## 结论与下一轮边界

V1 是有效的负结果。它相对 ordinary sparse control 提高了 candidate-level AP，并保留了
全部 raw physical truth edge，但没有在 5、10 或 50 mm 保住预先声明的 route 运行点。
因此它没有证明 association 或 alignment capture range 超过冻结的 pairwise MLP baseline。

封存 test 不能再用于 threshold、calibration、architecture 或新增 ablation 的选择。任何
V2 必须先在 train/validation 上设计和选择，再使用新保留的 source 或明确独立的 test bank。
下一步应先诊断为什么 full message-passing context 不如 local no-context 消融，而不是直接
增加更大的 Transformer。

## 复现

    cd /eos/home-x/xcheng/FASER/alignment_ML
    source scripts/setup_environment.sh ml

    # 只训练并冻结 train/validation artifact；使用新的输出路径。
    python scripts/train_geometry_aware_transformer_v1.py \
      --synthetic-manifest outputs/mc24_muon_2dfluka_curriculum_external_full_train_aug_synthetic_v1/synthetic_corpus_manifest.json \
      --config configs/geometry_aware_transformer_v1f_final_ablations.yaml \
      --output-dir outputs/NEW_TRANSFORMER_VALIDATION_RUN

    # 只在 validation 冻结后执行一次 test-only evaluator。
    python scripts/run_frozen_geometry_aware_transformer_v1_scan.py \
      --synthetic-manifest outputs/mc24_muon_2dfluka_multidirection_test_synthetic_controlled_v2/synthetic_corpus_manifest.json \
      --validation-output-dir outputs/NEW_TRANSFORMER_VALIDATION_RUN \
      --config configs/frozen_geometry_aware_transformer_v1_multidirection_test.yaml \
      --output-dir outputs/NEW_TRANSFORMER_TEST_RUN

