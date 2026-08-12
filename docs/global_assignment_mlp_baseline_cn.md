# Pairwise MLP 与全局指派基线

## 范围

本研究严格停留在 Transformer 之前。先建立低容量 pairwise MLP 加显式
station-pair 一对一指派的基线，再考虑多站上下文模型。

所有关联输入都来自真实物理链：

```text
SCT_ClusterContainer -> SegmentFitRefit -> SegmentsRefit -> NtupleDumper
-> FaserActsExtrapolationTool (mode 0)
```

每个几何点都生成独立 SQLite conditions payload 并重新拟合 segment。
不接受坐标平移或 residual 级替代物。local segment 的 `q/p` 仅是固定传播
seed，不是测量量，因此 V1 固定 `q_over_p_mode=0`。

## 严格 source 隔离的 nominal pilot

nominal pilot 使用 8 个彼此独立的 MC24 2D-FLUKA xAOD 文件：4 个训练、
2 个验证、2 个测试，来自正 muon source `100116` 和负 muon source
`100117`。划分单位是原始 xAOD 文件；synthetic materialize 前和 MLP
加载时均会检查 source provenance。

| Split | 物理 source event | Synthetic event | True tracklet | Random easy fake | 被省略 true tracklet |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 39 | 480 | 5,178 | 443 | 582 |
| Validation | 19 | 240 | 2,577 | 223 | 303 |
| Test | 19 | 240 | 2,600 | 242 | 280 |

每个 synthetic event 叠加 3 条彼此不同的 single-muon 物理径迹。此 nominal
pilot 有 random easy fake 和 missing tracklet，但刻意没有注入 field-aware
hard negative（数量为 `0`）。这不是把随机 fake 分布误当成物理背景；
hard-negative 构造已作为错位 curriculum corpus 的必须后续工作记录。

验证集 candidate coverage 为 `3,461 / 3,461 = 1.000`，所有 station pair
都覆盖。测试集覆盖率较低源于真实 exporter/refit 的可用性，而不是坐标替代：
最终选择的 ungated test graph candidate recall 为 `0.98493`。

## Field-Aware Hard-Negative Smoke Test

第一版采用 `hard_negative_mean_per_target_station = 0.50`，从真实同 payload
mode-0 Acts 的 `[1,1000]` chi2 区间选择 hard negative。它产生了训练/验证
`731/360` 个 hard negative，但审计发现约 `40--60%` 比对应 retained truth
endpoint 更接近 source。这是会系统性压过 truth 的对抗式构造，不符合普通几何近邻
背景的定义，因此不再作为当前 curriculum 定义。

修正后的构造仍使用真实 field propagation 与相同绝对 chi2 区间，但当 truth endpoint
存在时额外要求 `1.10 <= hard_chi2 / truth_chi2 <= 10.0`。只用训练/验证的 smoke
实际得到 `715/304` 个 hard negative；选择失败 `24/48` 会保留在 provenance 中，
不会用 surrogate 填充。所有可与 truth endpoint 比较的 hard candidate 均满足
`hard_chi2 > truth_chi2`。random easy fake 仍要求 anchor chi2 至少为 1000。

在这个刻意严格的 nominal hard-negative stress 上，当前 shared
`state_augmented_v2` MLP 尚未满足主运行点：最佳质量配置在 250 gate 的 efficiency
约为 0.45，但 candidate recall 只有约 0.52；ungated graph 可达约 0.83 efficiency，
但 inclusive fake rate 约为 0.091。station-pair ensemble 也在同一 validation-only
规则下测试，结果更差。所有 hard-negative smoke 的 test 文件均未打开。下一步是增加
严格 source 隔离的物理训练数据，而不是启动 Transformer。

同一组 pairwise MLP score 还进行了 exact multi-station flow 的小型
validation-only threshold/penalty 扫描。最佳质量点使用 ungated graph 和
unmatched penalty=0.5：efficiency `0.57835`、inclusive purity `0.95520`、
fake rate `0.04480`、candidate recall `1.0`。这比相同 stress 下 pairwise
dustbin assignment 的质量约束效率明显更高，但仍低于预先声明的 0.70 efficiency
门槛。它说明跨站全局一致性确实有帮助，但目前仍不能替代更充分的 source-disjoint
curriculum baseline。

## 模型与指派

第一版只用 residual/pull 的 pair MLP 不足：validation AUC 为 `0.9171`、
AP 为 `0.8302`，但最好的 Sinkhorn 指派在 `95.36%` purity 时 efficiency
仅 `18.98%`。诊断显示 high-score 的 different-truth pair，尤其在
`S0 -> S2` 和 `S0 -> S3`。

修正后的低容量 `state_augmented_v2` MLP 在物理 residual/pull feature
之外加入持久化的 refitted source/target state `(x, y, tx, ty)`。它不使用
truth ID、synthetic-role 标签、Transformer，也不把 `q/p` 当作关联测量。
其 ungated validation candidate 指标为 AUC `0.96307`、AP `0.89888`、
calibrated ECE `0.05165`。

每个有序 station pair 都由 MLP score 填充完整 candidate score matrix。
以下四种方法均不读取 MC truth：

1. `greedy`：按分数递减的一对一参考。
2. `hungarian`：阈值后的全局二部图指派。
3. `dustbin_hungarian`：source/target 专属 dummy node；等价于带
   unmatched penalty 的 unit-capacity min-cost-flow。
4. `sinkhorn_hungarian`：对 dustbin matrix 做 log-domain Sinkhorn
   normalisation，随后 Hungarian rounding。

只用 validation 拟合 score calibration，并扫描 candidate chi2 gate、
decision threshold、unmatched penalty 和 Sinkhorn temperature。五个 candidate
gate（`25`、`250`、`1000`、`5000`、ungated）全部进入扫描。只有运行点
固定后才打开 test。预先声明的主约束是 inclusive fake rate <= `0.05` 且
inclusive purity >= `0.95`，并在约束内最大化 validation efficiency。

## 封存的 nominal test 结果

| 方法 | 选择的 gate | Test efficiency | Inclusive purity | Fake rate | Missing unmatched recall |
| --- | --- | ---: | ---: | ---: | ---: |
| Greedy | `5000` | 0.82514 | 0.98540 | 0.01460 | 0.99347 |
| Hungarian | ungated | 0.82940 | 0.98781 | 0.01219 | 0.99217 |
| Dustbin Hungarian | ungated | 0.93261 | 0.97590 | 0.02410 | 0.96997 |
| Sinkhorn -> Hungarian | ungated | 0.93261 | 0.97590 | 0.02410 | 0.96997 |

因此 core nominal primary operating point 已满足；但 hard-negative stress 仍未通过，
其 test split 继续封存。完整 physical curriculum 的结果与下一步约束记录在下一节；
这些结果仍不构成启动 Transformer 的理由。

## 完整物理 hard-background 更新

完整 2D-FLUKA curriculum 的原始 source split 为训练/验证/测试
`6 / 2 / 2` 个文件，对应 `59 / 19 / 19` 个 source event，跨 split event UID
重叠为零。另有两个独立生成的 MC24 `100043` muon-minus 5 mrad FLUKA-E
xAOD 文件完成了全部六个真实 conditions payload，且只加入训练集。扩展后的
source split 为 `8 / 2 / 2` 个文件、`79 / 19 / 19` 个 source event；每个新增
source 的每个错位点均通过
`SCT_ClusterContainer -> SegmentFitRefit -> SegmentsRefit -> NtupleDumper ->
FaserActsExtrapolationTool(mode 0)` 物理链。

synthetic background 的定义保持可审计和物理一致：每个 event overlay 三条独立
source track；missing tracklet 保留；random easy fake 要求较大 Acts chi2；hard fake
从同一 payload 的 field-aware mode-0 Acts 候选中选择，并在 truth endpoint 存在时
要求 `1.10 <= hard_chi2 / truth_chi2 <= 10.0`。审计确认没有 hard fake 比保留的
truth counterpart 更兼容。完整 stress 的默认密度为每个非参考 target station 平均
`0.50` 个 hard fake；受控的 `0.10` 和 `0.25` 密度也未达到主运行点，最佳质量约束
efficiency 分别为 `0.3790` 和 `0.3851`。

以下比较均使用扩展训练 bank、相同封存 validation source，以及只在 validation
进行的控制量选择。预先声明的 nominal 主运行点是 efficiency >= `0.70`、inclusive
purity >= `0.95`、inclusive fake rate <= `0.05`。

| 基线 | 最佳 nominal 质量运行点 | Efficiency | Inclusive purity | Fake rate | Candidate recall | Test |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Shared `state_augmented_v2` | Sinkhorn + Hungarian, chi2 <= 250 | 0.4476 | 0.9580 | 0.0420 | 0.5198 | 封存 |
| Shared v2，station-pair temperature | Sinkhorn + Hungarian, chi2 <= 250 | 0.4722 | 0.9523 | 0.0477 | 0.5198 | 封存 |
| Exact four-station flow | ungated，10,000 hypothesis 上限 | 0.4205 | 0.9517 | 0.0483 | 1.0000 | 封存 |
| 六个独立 pair MLP | station-pair temperature，chi2 <= 250 | 0.4361 | 0.9521 | 0.0479 | 0.5198 | 封存 |
| Shared v2，threshold vector | station-pair Sinkhorn threshold | 0.5986 | 0.9509 | 0.0491 | 0.6032 score-edge retention | 封存 |
| 六个独立 pair MLP，精确密集 threshold vector | station-pair Sinkhorn threshold | 0.5209 | 0.9500 | 0.0500 | 1.0000；0.5590 score-edge retention | 封存 |
| `state_hitpattern_v4` | 持久化 layer-side occupancy bit | 0.3576 | 0.9597 | 0.0403 | 1.0000 | 封存 |

threshold-vector 是一个显式有界、仅 validation 的搜索：五个 threshold、两个初值、
两轮 coordinate sweep、三个 unmatched penalty。选择的阈值向量按
`0->1, 0->2, 0->3, 1->2, 1->3, 2->3` 依次为
`0.35, 0.60, 0.60, 0.20, 0.20, 0.20`。它明显优于全局单阈值，但仍未达到预定义的
0.70 efficiency 门槛。其 nominal score-edge truth retention 在 `0->2` 和 `0->3`
仅为 `0.068` 和 `0.142`，短基线 pair 为 `0.70--0.93`；当前瓶颈由长基线的 pairwise
score separation 主导。

六个独立 pair MLP 还使用精确的有限 threshold-vector 选择器重跑，而不依赖
coordinate descent。对每个固定 dustbin penalty 和 Sinkhorn temperature，先针对每个
station pair/threshold 独立做 truth-blind bipartite assignment；随后只在 validation
上，以可加的 correct/predicted 计数构造 Pareto frontier。密集审计覆盖 16 个
threshold、3 个 penalty 和 `0.2, 0.5, 1.0` 三个 temperature；最佳满足质量约束的
nominal 点为 efficiency `0.5209`、inclusive purity `0.9500`、fake rate `0.0500`。
它使用 penalty=`1.0`、temperature=`0.2`，按既定 pair 顺序的 threshold 为
`0.10, 0.90, 0.90, 0.30, 0.60, 0.20`。在该点长基线 `0->2` 和 `0->3` 没有保留任何
truth edge。因此达不到 0.70 不是 coordinate-descent 局部最优，也不是五点 threshold
网格过粗造成的假象。

使用 `Tracklet_hit_pattern` 前已经核对 NtupleDumper 源码：这是六位 SCT
layer-side pattern，第 `2 * layer + side` 位对应一个 layer-side。训练/验证审计值在
`[15, 63]`。将 source/target 的 12 个 bit 加入 `state_hitpattern_v4` 后 validation
反而变差，因此它只作为记录完备的负 ablation，不进入基线。

这一轮 pre-source-expansion 结果没有启动 Transformer、attention layer 或 raw-hit
retracking。随后在同一 source 隔离和 test 封存规则下进行的修复是增加物理 source
覆盖；结果记录在下一节。

## 扩展 source 与精确阈值向量结果

下一步 source 覆盖修复现已完成。两个独立的 MC24 `100044` 正 muon、5 mrad
FLUKA-E xAOD 文件均已走完六个真实 conditions payload，并且只加入训练集。新的
physical corpus 在训练/验证/测试中包含 `10 / 2 / 2` 个 source 文件和
`99 / 19 / 19` 个 source event；所有 source-event UID 交集仍为空。train/validation
synthetic 审计分别有 `77,797 / 15,574` 条 true tracklet、
`7,254 / 1,394` 条 random-easy fake 和 `9,590 / 1,701` 条 field-aware hard fake。
每一个可与 retained truth counterpart 比较的 hard fake 都更不兼容。

共享的 `state_augmented_v2` MLP 在扩展后的 physical bank 上使用 CUDA 重训。
普通的单一阈值 global scan 仍未达到主运行点，最佳质量诊断 efficiency 为
`0.44675`。随后固定 checkpoint，只在 validation 做精确、有限的
station-pair threshold-vector 搜索。对于 16 个 score threshold、3 个 unmatched
penalty 和 3 个 Sinkhorn temperature 的每种组合，求解器先在每个 station pair 内做
不读取 truth 的一对一指派，再以可加的 correct/predicted count 构造 Pareto
frontier；truth label 只在指派之后用于选择 validation 运行点。

选择的 validation 配置为 Sinkhorn 后 Hungarian rounding、penalty=`1.0`、
Sinkhorn temperature=`1.0`，并按
`0->1, 0->2, 0->3, 1->2, 1->3, 2->3` 使用阈值
`0.05, 0.20, 0.90, 0.30, 0.35, 0.35`。其 efficiency=`0.71313`、inclusive
purity=`0.95075`、inclusive fake rate=`0.04925`，这是首个满足预先声明的
inclusive primary 的 physical hard-background validation 点。配置随后被冻结，并只对
封存 test source 执行一次评估。

| 注入错位幅度 [mm] | Test efficiency | Inclusive purity | Inclusive fake rate | 主质量约束 |
| ---: | ---: | ---: | ---: | --- |
| 0.0 | 0.77114 | 0.95758 | 0.04242 | 通过 |
| 0.1 | 0.77702 | 0.95715 | 0.04285 | 通过 |
| 1.0 | 0.76236 | 0.94686 | 0.05314 | 未通过 |
| 5.0 | 0.75257 | 0.95257 | 0.04743 | 通过 |
| 10.0 | 0.69620 | 0.93333 | 0.06667 | 未通过 |
| 50.0 | 0.07546 | 0.45392 | 0.54608 | 未通过 |

这还不是 capture fraction 曲线：封存 test 的每个 magnitude 目前只对应一个真实
payload direction，因此 `1` 与 `5 mm` 的非单调性不能被解释为确定的收敛范围。它已经
显示了大错位下的清晰失效，而大部分 station pair 的物理 candidate recall 依然很高。
例如在 `50 mm`，五个 pair 的 raw candidate recall 为 `1.0`，`0->3` 为 `0.929`，
但全局 score-threshold truth recall 已降至 `0.253`。

还存在一个重要的拓扑限制。validation 选择的 `0->3` 阈值 `0.90` 没有保留任何直接
`S0->S3` truth edge。因此该结果满足预先声明的 *inclusive pairwise* primary，却不能
证明完整的直接四站关联链。下一项非 Transformer 工作应在多个真实 payload direction
上重复物理扫描，并用相邻站链式关联或 route-level unit-capacity flow，加上明确的四站
完整度指标。没有启动 Transformer。

## 复现

```bash
cd /eos/home-x/xcheng/FASER/alignment_ML
source scripts/setup_environment.sh ml

# 完整 physical conditions/refit bank，每次处理一个 source。
python scripts/build_physical_curriculum_corpus.py \
  --config configs/physical_global_assignment_muon_2dfluka_curriculum.yaml \
  --output-dir outputs/mc24_muon_2dfluka_curriculum_physical_v1

# source bank 完成后，先封存 provenance 并审计每一个 refitted payload。
python scripts/assemble_physical_curriculum_manifest.py \
  --config configs/physical_global_assignment_muon_2dfluka_curriculum.yaml \
  --input-manifest outputs/mc24_muon_2dfluka_curriculum_physical_v1/physical_corpus_manifest.json \
  --output outputs/mc24_muon_2dfluka_curriculum_physical_v1/combined_physical_corpus_manifest.json

python scripts/audit_physical_curriculum_bank.py \
  --physical-manifest outputs/mc24_muon_2dfluka_curriculum_physical_v1/combined_physical_corpus_manifest.json \
  --output-dir outputs/mc24_muon_2dfluka_curriculum_physical_bank_audit_v1

# 审计通过后，生成严格 source 隔离的 synthetic 数据。
python scripts/materialize_pooled_curriculum_synthetics.py \
  --physical-manifest outputs/mc24_muon_2dfluka_curriculum_physical_v1/combined_physical_corpus_manifest.json \
  --config configs/physical_global_assignment_muon_2dfluka_curriculum.yaml \
  --output-dir outputs/mc24_muon_2dfluka_curriculum_synthetic_v1

python scripts/run_global_assignment_mlp_baseline.py \
  --config configs/physical_global_assignment_muon_2dfluka_curriculum.yaml \
  --synthetic-manifest outputs/mc24_muon_2dfluka_curriculum_synthetic_v1/synthetic_corpus_manifest.json \
  --output-dir outputs/mc24_muon_2dfluka_curriculum_mlp_v1

# 使用固定 shared MLP 做有界的 station-pair threshold-vector 搜索。
# 在 primary validation 通过前，默认不打开 test。
python scripts/run_station_pair_threshold_baseline.py \
  --config configs/physical_global_assignment_muon_2dfluka_curriculum_external_mumi_station_pair_thresholds.yaml \
  --synthetic-manifest outputs/mc24_muon_2dfluka_curriculum_external_mumi_train_aug_synthetic_v1/synthetic_corpus_manifest.json \
  --checkpoint outputs/mc24_muon_2dfluka_curriculum_external_mumi_train_aug_mlp_validation_v1/mlp_pair_classifier.pt \
  --output-dir outputs/mc24_muon_2dfluka_curriculum_external_mumi_station_pair_thresholds_validation_v1

# 对已训练的独立 pair-MLP ensemble 做精确、密集的 validation-only 网格。
# 此命令不打开封存 test。
python scripts/run_station_pair_threshold_baseline.py \
  --config configs/physical_global_assignment_muon_2dfluka_curriculum_external_mumi_pair_ensemble_station_pair_thresholds_exact_dense.yaml \
  --synthetic-manifest outputs/mc24_muon_2dfluka_curriculum_external_mumi_train_aug_synthetic_v1/synthetic_corpus_manifest.json \
  --checkpoint-dir outputs/mc24_muon_2dfluka_curriculum_external_mumi_pair_ensemble_validation_v1 \
  --output-dir outputs/mc24_muon_2dfluka_curriculum_external_mumi_pair_ensemble_station_pair_thresholds_exact_dense_validation_v1

# 扩展后的 100043 + 100044 训练 bank：只训练并在 validation 选择。
python scripts/run_global_assignment_mlp_baseline.py \
  --config configs/physical_global_assignment_muon_2dfluka_curriculum_external_train_aug.yaml \
  --synthetic-manifest outputs/mc24_muon_2dfluka_curriculum_external_full_train_aug_synthetic_v1/synthetic_corpus_manifest.json \
  --output-dir outputs/mc24_muon_2dfluka_curriculum_external_full_train_aug_mlp_validation_v1 \
  --validation-only

# 固定 checkpoint 后，进行精确的 validation 搜索。
python scripts/run_station_pair_threshold_baseline.py \
  --config configs/physical_global_assignment_muon_2dfluka_curriculum_external_full_train_aug_shared_station_pair_thresholds_exact_dense.yaml \
  --synthetic-manifest outputs/mc24_muon_2dfluka_curriculum_external_full_train_aug_synthetic_v1/synthetic_corpus_manifest.json \
  --checkpoint outputs/mc24_muon_2dfluka_curriculum_external_full_train_aug_mlp_validation_v1/mlp_pair_classifier.pt \
  --output-dir outputs/mc24_muon_2dfluka_curriculum_external_full_train_aug_shared_station_pair_thresholds_exact_dense_validation_v1

# 仅在冻结的 validation primary 通过后，以完全相同的命令打开 test。
python scripts/run_station_pair_threshold_baseline.py \
  --config configs/physical_global_assignment_muon_2dfluka_curriculum_external_full_train_aug_shared_station_pair_thresholds_exact_dense.yaml \
  --synthetic-manifest outputs/mc24_muon_2dfluka_curriculum_external_full_train_aug_synthetic_v1/synthetic_corpus_manifest.json \
  --checkpoint outputs/mc24_muon_2dfluka_curriculum_external_full_train_aug_mlp_validation_v1/mlp_pair_classifier.pt \
  --output-dir outputs/mc24_muon_2dfluka_curriculum_external_full_train_aug_shared_station_pair_thresholds_exact_dense_test_v1 \
  --evaluate-test
```

运行会写出 candidate recall、global matching efficiency、purity/fake rate、
unmatched recall、score calibration，以及 station-pair/misalignment breakdown。
