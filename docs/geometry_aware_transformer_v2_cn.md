# Geometry-Aware Transformer V2：route-aware 验证研究

## 判定

V1 现有 test source 已永久封存。V2 的 architecture、loss、calibration、threshold 和
route utility 选择只使用按原始 source file 严格隔离的 train/validation split。本轮没有生成
新 test bank，也没有打开 test event 或 test artifact。

V2 validation hypothesis 未通过。validation 选择出的 route-context weight 是 `0.0`，它与
冻结的 V2 edge-level route-assignment control 完全等价。因此本轮不能宣称 route-aware 增益、
association capture range 扩大，也不能据此运行新的最终 test。

## 不变的物理输入链

所有 V2 输入继续来自真实 displaced-geometry 链路：

```text
SCT_ClusterContainer -> SegmentFitRefit -> SegmentsRefit -> NtupleDumper
-> FaserActsExtrapolationTool (mode 0)
```

模型使用 ungated、既有的 mode-0 Acts physical candidate graph。不使用 coordinate-level
surrogate、不重新生成 residual、不制造新 edge，也不把 local segment `q/p` 当成 measurement。
train 与 validation source file 严格隔离；manifest loader 以明确的 `allowed_splits` 调用，
因此 excluded test asset path 不会被 resolve。

以下输出 contract 都记录 `test_events_loaded=false` 和
`test_artifacts_opened=false`：

- `outputs/geometry_aware_transformer_v2_validation_v1/`
- `outputs/geometry_aware_transformer_v2_route_residual_utility_validation_v1/`
- `outputs/geometry_aware_transformer_v2_validation_assessment_v1/`

## V1 full context 为什么没有更好

此前的 validation-only mechanism diagnosis 见
[V2 机制诊断记录](geometry_aware_transformer_v2_diagnostics_cn.md)。它使用 1,440 个
validation event、18,669 个 node、175,084 条有向 physical message edge 和 44,921 条相邻
输出 edge。

message-depth、hop direction、adjacent-only 与 leave-one-station-out probe 表明：冻结 V1 的
edge ranking 几乎不随 global message 改变。depth-zero 到 four-layer full-message 的 score MAE
为 `0.01350`、rank correlation 为 `0.99937`；depth-three 到 full 为 `0.00435` 和
`0.99989`。edge AP 只发生很小变化：

| station pair | local depth 0 AP | full depth 4 AP |
| --- | ---: | ---: |
| `0->1` | 0.8433 | 0.8405 |
| `1->2` | 0.8657 | 0.8666 |
| `2->3` | 0.8962 | 0.8983 |

与此同时，node-state 的 station 内平均 cosine similarity 从 `0.319` 上升到 `0.549`，平均
norm 从 `8.01` 上升到 `21.25`。这支持 representation mixing/over-smoothing 的判断，而 local
residual edge decoder 仍主导大部分排序。10 mm 时选出的 false route 同时包含 fake endpoint 和
mixed-truth chain；限制 message direction 只是在 efficiency 与 fake rate 之间交换，不能达到
固定 operating point。

## V2 route-aware 模型

`models/route_transformer.py` 保留固定的 V1 backbone：width 128、4 个 sparse block、8 个
head、FFN `128 -> 256 -> 128`。只有当已有 physical edge 能连成完整
`IFT -> S1 -> S2 -> S3` 时才构造 route query。query 聚合：

- 有序的四个 encoded node state；
- 三条既有相邻 physical edge 的 feature；
- station-pair embedding；
- 三个 backbone edge logit。

candidate bank 完全物理、truth-blind。train 的 7,200 个 graph 有 800,228 条完整 route；
validation 的 1,440 个 graph 有 148,551 条。validation bank 中有 2,832 条 truth-consistent
route、71,756 条 fake-endpoint route 和 45,142 条 hard-negative route。truth particle ID 和
synthetic role 只用作 supervision/audit label，不进入网络输入。

训练 loss 按预先定义为 weighted focal edge BCE、route-truth consistency focal BCE、
endpoint-incidence log-sum-exp one-to-one competition 和 fake-endpoint softplus penalty 的和。
route query 不替换 unit-capacity route assignment backend。

## Direct query 诊断与 residual utility

初始 direct-query integration 用校准后的完整 route log-odds 直接替换三条 edge log-odds 的和。
这与 solver 不一致：完整 route 会和保留三条强 edge utility 的 partial route 竞争。因此该验证网格中
没有任何完整 route 被选中。这是 route utility formulation 失败，并非 physical truth candidate
丢失。

route query 在进入 solver 前仍具有非零区分能力：148,551 条 physical route 上 raw AP 为
`0.4095`、ROC AUC 为 `0.9449`。只在 validation 拟合的 Platt calibration 将 ECE 从
`0.0177` 降至 `0.00312`，但不能自动使该分数与 partial-route objective 相容。

`baselines/route_assignment.py` 因而增加 complete route 的 residual composition：

```text
L = L_edge + w * (L_route - L_edge)
L_edge = sum(logit(p_edge))
L_route = logit(p_route)
```

`w=0` 精确回到 edge-only route solver，`w=1` 回到 direct replacement。partial route 始终保留
已有 physical edge utility，因此 missing-station/dustbin recovery 仍可用。residual scan 不裁剪
任何已有完整 physical route（`complete_route_score_threshold=0`），只在 validation 上选择 `w` 与
共享 dustbin utility。冻结 V2 的相邻 edge calibration/threshold（`0->1=0.001`、
`1->2=0.05`、`2->3=0.05`）不重新拟合。

## Validation 结果

residual grid 为 `w in {0, 0.025, 0.05, 0.10, 0.20, 0.40, 0.70, 1.0}`，dustbin penalty 为
`{-0.5, 0, 0.5, 1.0}`。validation-selected point 为 `w=0`、dustbin penalty `1.0`；所有
非零 route-context weight 都降低 complete-route efficiency。`w=0` 时 same-checkpoint edge-only
control 与 route-aware stream 数值完全一致。

| injected offset | raw truth-chain recall | edge-threshold chain recall | complete efficiency | complete purity | fake rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 mm | 1.0000 | 0.9978 | 0.5293 | 0.8561 | 0.2183 |
| 5 mm | 1.0000 | 0.9872 | 0.5556 | 0.8754 | 0.2245 |
| 10 mm | 1.0000 | 0.9782 | 0.4858 | 0.8610 | 0.2356 |
| 50 mm | 1.0000 | 0.6253 | 0.0683 | 0.6600 | 0.3435 |

所以主要问题发生在 raw candidate graph 之后。5 和 10 mm 时，大多数 physical truth chain 仍是
candidate 且可由 edge score 到达，但 route-level 的 false/mixed/fake competition 阻止模型达到
所需 operating point。

预先定义的 validation gate 为 nominal efficiency `>=0.70`、purity `>=0.95`、fake rate
`<=0.05`，并要求在 5/10 mm 相对 pairwise MLP、V1 full-context、V1 no-context 及
same-checkpoint edge control 有明确增益。post-selection assessment 的所有 gate flag 都为 false；
特别地，route-aware stream 相对完全相同 checkpoint control 没有增益。

## 结论

本结果不允许进行 test。现有 V1 test 不能被用于选择 route weight、threshold、calibration、loss 或
architecture，也不创建新的 source-disjoint test bank。

之后若继续 V2，应作为新的 train/validation hypothesis：例如在训练中将 route logit 显式建模为
edge-route utility 的 residual，并引入 event-level assignment surrogate，而不是先学习绝对 rare-route
probability 后再在训练外混合。这不是重新打开已封存 test 的理由。

## 复现

```bash
cd /eos/home-x/xcheng/FASER/alignment_ML
source scripts/setup_environment.sh ml

# 仅在 train/validation physical payload 上训练 V2。
python scripts/train_route_aware_transformer_v2.py \
  --synthetic-manifest outputs/mc24_muon_2dfluka_curriculum_external_full_train_aug_synthetic_v1/synthetic_corpus_manifest.json \
  --config configs/geometry_aware_transformer_v2.yaml \
  --output-dir outputs/NEW_V2_VALIDATION

# 仅 validation 评估 route-query residual，复用冻结的 V2 edge controls。
python scripts/evaluate_route_aware_transformer_v2_direct_routes.py \
  --synthetic-manifest outputs/mc24_muon_2dfluka_curriculum_external_full_train_aug_synthetic_v1/synthetic_corpus_manifest.json \
  --v2-validation-dir outputs/NEW_V2_VALIDATION \
  --config configs/geometry_aware_transformer_v2_route_residual_utility.yaml \
  --output-dir outputs/NEW_V2_ROUTE_RESIDUAL_VALIDATION \
  --device cuda

# 只比较 validation artifact；此命令不打开 event data。
python scripts/assess_route_aware_transformer_v2_validation.py \
  --v2-output outputs/NEW_V2_VALIDATION \
  --direct-route-output outputs/NEW_V2_ROUTE_RESIDUAL_VALIDATION \
  --mlp-route-output outputs/pairwise_mlp_route_validation_v2_control_v1 \
  --v1-validation-dir outputs/geometry_aware_transformer_v1f_final_validation_v1 \
  --output-dir outputs/NEW_V2_VALIDATION_ASSESSMENT
```
