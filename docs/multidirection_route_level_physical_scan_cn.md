# 多方向物理扫描与相邻站 route 基线

## 范围

本轮仍停留在 Transformer 实现之前。研究对象是已冻结的低容量 pairwise MLP，配合
不读取 truth 的四站 route 求解器，并在新生成的真实几何 trial 上评估。

每个物理点严格运行：

```text
persisted SCT_ClusterContainer -> SegmentFitRefit -> SegmentsRefit
-> NtupleDumper -> FaserActsExtrapolationTool (mode 0)
```

没有平移 tracklet 坐标、复用 residual、使用 residual-level surrogate、重做 raw-hit
tracking，也没有 test-time calibration 或 test-time 阈值选择。local segment 的 `q/p`
仍仅作传播 seed；候选构造固定使用 `q_over_p_mode=0`。

## 物理 trial bank

封存的 source-file test split 没有变化：

| Source xAOD ID | 产生过程 | Source event 数 |
| --- | --- | ---: |
| `mc24_100116_00030_00039` | MC24 FASERnu 2D-FLUKA mu+ | 10 |
| `mc24_100117_00030_00039` | MC24 FASERnu 2D-FLUKA mu- | 9 |

station 0（IFT）固定为 `(0, 0)`。station 1--3 在每个非零幅度均注入彼此独立、已
归一化的 `(dx, dy)` direction。固定 configuration seed 是 `20260812`。

| 注入幅度 [mm] | 独立 direction trial | 每个 source 新物理点 |
| ---: | ---: | ---: |
| 0 | 1 | 1 |
| 0.1、1、5、10、50 | 每个 3 个 | 15 |

`0 mm` 只做一次新的 refit，因为零向量没有物理 direction。最终两个 xAOD 各有 16 点，
即总共 32 个新写入的 `/Tracker/Align` SQLite/POOL payload 及独立 refit/Acts 输出。

`outputs/mc24_muon_2dfluka_multidirection_test_physical_audit_v2/` 确认 32 个点均有
完整资产、MC truth 和正定 tracklet covariance。每个非零幅度的三个 direction 合计在两个
source 中有 57 个 refitted event、213 条 tracklet。

对 source `100116` 的 `5 mm/test_02` 做了显式响应审计，确认是重新拟合的几何响应：
station 1--3 refitted state 的 median shift 分别是 `(4.8633, 1.1610)`、
`(-3.3355, -3.7249)` 和 `(1.8542, 4.6435) mm`，与写入 payload 的 translation 一致。
mode-0 truth-pair residual response error 的 p95 为 x `0.00064 mm`、y `0.00832 mm`。
这验证了 displaced-geometry refit，而不是坐标 surrogate。

## 受控 synthetic overlay

每个物理 payload 只在两个封存 test source 内 pool，并 materialize 成 240 个 synthetic
event；每个 event 叠加三条彼此不同的 source-muon track，保留 missing tracklet、
random-easy fake 和 field-aware hard negative。评估前会检查 source-file membership。

同一幅度的不同 direction 使用相同 overlay RNG seed
（`overlay_seed_scope: magnitude_shared_across_direction_trials`）。因此 source-track 与
random-fake draw 在 direction 之间受控。hard-negative 是否被接受仍可能因该 direction
下真实 mode-0 Acts 候选而不同；这是物理效应，不是人为坐标扰动。

背景审计中有 4,595 个可与 retained truth counterpart 比较的 hard-negative candidate；
没有一个满足 `hard_chi2 <= truth_chi2`。synthetic role provenance 只用于 audit，不进入
MLP feature 或 route assignment。

## 冻结的 pairwise 契约

| 项目 | 固定值 |
| --- | --- |
| MLP checkpoint | `...full_train_aug_mlp_validation_v1/mlp_pair_classifier.pt` |
| Calibration | 按 station pair 的 temperature，仅用 validation 拟合 |
| Pair threshold 来源 | 仅 validation 的 exact threshold-vector 选择 |
| 相邻站 threshold | `0->1: 0.05`、`1->2: 0.30`、`2->3: 0.35` |
| Dustbin penalty | `1.0` |
| Test-time calibration/selection | 禁止 |

route 求解器只使用校准后的 `IFT -> S1 -> S2 -> S3` 相邻站 score。它枚举长度至少为二的
连续 path，并用精确的小事件 binary set-packing 选择端点互斥的 path；这等价于每个端点有
单位容量、未使用端点进入显式 dustbin 的 route flow。求解器从不读取 truth，也不要求直接
`0->3` edge。

固定 capture 判据为：

```text
complete-track efficiency >= 0.70
complete-track purity     >= 0.95
route fake rate           <= 0.05
```

这里的 route fake rate 使用 inclusive 定义：任何不是 truth-consistent 的 selected route
都计为假，包括错误连接的真端点和 synthetic fake endpoint。后者另以
`fake_endpoint_route_rate` 单独报告。duplicate rate 作为 route fragmentation 诊断，但不
作为新调出的 capture 条件。

## 结果

下表是 direction trial 的均值；`+/-` 是 direction 间总体标准差。raw candidate-chain
recall 在 MLP threshold 前计算；score-chain recall 要求三条 truth 相邻边均通过冻结阈值。

| 幅度 [mm] | 通过 trial | Raw candidate-chain recall | Score-chain recall | Complete-track efficiency | Complete purity | Route fake rate | Duplicate rate | Missing-station recovery |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 / 1 | 1.000 | 0.932 | 0.778 | 0.977 | 0.037 | 0.081 | 0.970 |
| 0.1 | 3 / 3 | 1.000 | 0.958 +/- 0.030 | 0.790 +/- 0.042 | 0.984 +/- 0.000 | 0.037 | 0.089 | 0.970 |
| 1 | 2 / 3 | 1.000 | 0.954 +/- 0.033 | 0.802 +/- 0.040 | 0.987 +/- 0.004 | 0.048 | 0.069 | 0.965 |
| 5 | 0 / 3 | 1.000 | 0.793 +/- 0.210 | 0.529 +/- 0.251 | 0.962 +/- 0.028 | 0.090 | 0.096 | 0.965 |
| 10 | 0 / 3 | 1.000 | 0.620 +/- 0.246 | 0.260 +/- 0.289 | 0.946 +/- 0.042 | 0.090 | 0.141 | 0.942 |
| 50 | 0 / 3 | 1.000 | 0.000 | 0.000 | -- | 0.425 | 0.000 | 0.947 |

`50 mm` 没有被选中的 complete route，因此 complete purity 无定义。但三条 direction 的
candidate graph 仍保留所有 complete truth chain。失效发生在物理候选生成之后：冻结 MLP
的 score separation、阈值和全局 route 选择失效，而不是 chi2 candidate gate 丢失真边。

在 `5 mm` 已有相同结论：所有 direction 都保留 raw truth chain，但没有任何一个通过固定
capture 判据。每个 station pair 的 candidate recall、AUC/AP、association
efficiency/purity/fake rate 和 unmatched recall 在 `station_pair_trial_metrics.csv`；至
`50 mm` 为止所有 raw adjacent-pair candidate recall 都为 1.0。

图：

- `capture_fraction_vs_misalignment.png`
- `route_quality_vs_misalignment.png`

## 判定

用户定义的 Transformer 触发条件已经满足：在大真实错位 `5`、`10`、`50 mm`，物理
candidate graph 保留 truth chain，而冻结 pairwise MLP + unit-capacity route assignment 的
capture fraction 均为零。这是 pairwise local score 的受控失效，不是 chi2 gate、单一
payload direction 或坐标 surrogate 的失效。

这还不能证明 Transformer 会扩大 capture range；它只建立了下一步比较所需要的基线条件：
Geometry-Aware Transformer 必须与这个冻结、source-disjoint、真实物理 route baseline 对比。
本研究没有实现 Transformer。

## 复现

```bash
cd /eos/home-x/xcheng/FASER/alignment_ML
source scripts/setup_environment.sh ml

# 准备并运行所有 test-source 物理点。每一点调用 Calypso 的
# cluster -> segment refit -> Acts 链，而不是 residual surrogate。
python scripts/build_physical_curriculum_corpus.py \
  --config configs/physical_global_assignment_muon_2dfluka_curriculum_external_multidirection_test.yaml \
  --output-dir outputs/mc24_muon_2dfluka_multidirection_test_physical_v1 \
  --prepare-only

python scripts/build_physical_curriculum_corpus.py \
  --config configs/physical_global_assignment_muon_2dfluka_curriculum_external_multidirection_test.yaml \
  --output-dir outputs/mc24_muon_2dfluka_multidirection_test_physical_v1 \
  --resume \
  --source-id mc24_100116_00030_00039 \
  --source-id mc24_100117_00030_00039

# 只 materialize 封存的 test source pool；同一幅度的 direction 共用
# overlay RNG stream。
python scripts/materialize_pooled_curriculum_synthetics.py \
  --physical-manifest outputs/mc24_muon_2dfluka_multidirection_test_physical_v1/physical_corpus_manifest.json \
  --config configs/physical_global_assignment_muon_2dfluka_curriculum_external_multidirection_test.yaml \
  --output-dir outputs/mc24_muon_2dfluka_multidirection_test_synthetic_controlled_v2 \
  --split test

python scripts/run_frozen_route_level_scan.py \
  --config configs/frozen_route_level_multidirection_test.yaml \
  --synthetic-manifest outputs/mc24_muon_2dfluka_multidirection_test_synthetic_controlled_v2/synthetic_corpus_manifest.json \
  --checkpoint outputs/mc24_muon_2dfluka_curriculum_external_full_train_aug_mlp_validation_v1/mlp_pair_classifier.pt \
  --frozen-calibration outputs/mc24_muon_2dfluka_curriculum_external_full_train_aug_shared_station_pair_thresholds_exact_dense_validation_v1/calibration.json \
  --frozen-operating-point outputs/mc24_muon_2dfluka_curriculum_external_full_train_aug_shared_station_pair_thresholds_exact_dense_validation_v1/validation_selected_operating_point.json \
  --output-dir outputs/mc24_muon_2dfluka_multidirection_test_route_level_controlled_v4
```

`frozen_evaluation_contract.json` 记录 config、manifest、checkpoint、calibration 和
validation operating point 的 SHA-256 hash。
