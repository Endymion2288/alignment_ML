# Geometry-Aware Transformer V2：route-aware 验证与 test 封存结论

## 本轮硬边界

- V1 test source `mc24_100116_00030_00039`、`mc24_100117_00030_00039` 继续永久封存。
- architecture、loss、calibration、threshold、route utility 的选择只允许 train/validation。
- 本轮没有生成新的 test bank；没有打开 test event 或 test artifact。
- 物理数据链保持不变：`SCT_ClusterContainer -> SegmentFitRefit -> SegmentsRefit -> NtupleDumper -> FaserActsExtrapolationTool(mode 0)`。

每个新 V2 输出均记录 `test_events_loaded=false`、`test_artifacts_opened=false`。loader 使用
`allowed_splits=(train, validation)` 或 `allowed_splits=(validation,)`，excluded test path 不会被
resolve。

## V1 机制诊断回顾

validation-only probe 已完成：message depth 0--4、adjacent-only、forward-only、backward-only、
leave-one-station-out。使用 1,440 validation event、18,669 node、175,084 条有向 physical message
edge、44,921 条相邻输出 edge。

- depth 0 到 full 的 score MAE=0.01350、rank corr=0.99937；depth 3 到 full 为 0.00435、0.99989。
- 0->1/1->2/2->3 AP 从 0.8433/0.8657/0.8962 变为 0.8405/0.8666/0.8983。
- station 内 cosine similarity 0.319 -> 0.549，state norm 8.01 -> 21.25。
- 结论：full message 有 mixing/over-smoothing，local edge decoder 主导排序；false route 的核心是
  fake endpoint 与 mixed-truth competition，而不是 raw truth edge 被 candidate graph 删除。

详细 CSV：

    outputs/geometry_aware_transformer_v2_mechanism_diagnostics_validation_v1/

## V2 实现

新增：

- `models/route_transformer.py`：保留 d=128、4 block、8 head、FFN 128->256->128 的 V1 backbone；
  只对已有 0->1->2->3 physical chain 建 route query。
- `training/route_aware_transformer.py`：route truth consistency、endpoint one-to-one
  competition、fake-route penalty；truth/synthetic role 仅作 label/audit。
- `baselines/route_assignment.py`：完整 route score 支持 `replace` 与 `residual` composition。
- `scripts/evaluate_route_aware_transformer_v2_direct_routes.py`：只加载 validation，复用冻结 V2
  edge calibration/threshold，对 route utility 做 validation-only 选择。
- `configs/geometry_aware_transformer_v2_route_residual_utility.yaml`：route residual 实验配置。

route candidate bank audit：train 7,200 graph / 800,228 route / 14,203 truth-consistent / 416,825
fake-endpoint；validation 1,440 graph / 148,551 route / 2,832 truth-consistent / 71,756 fake-endpoint。
完整 route 只由已有 physical adjacent candidate 组成，不用 truth、score 或 chi2 截断构造。

## Direct route utility 的机制问题

初始 direct integration 直接用 complete-route log-odds 替换三个 edge log-odds。它让完整 route 和
两个 partial route 比较时失去三条高分 edge 的累计 utility，78 个 validation control point 都没有
选择完整 route。这是 solver objective 不一致，而不是 route query 或 candidate graph 为零。

route query 本身在 validation 的 raw AP=0.4095、ROC AUC=0.9449。Platt calibration 的 ECE 从
0.0177 降到 0.00312，但校准后的 absolute rare-route probability 不能直接替代 partial route 的
edge utility。

因此改为：

    L = L_edge + w * (L_route - L_edge)
    L_edge = sum(logit(p_edge))
    L_route = logit(p_route)

`w=0` 精确是 edge-only control；`w=1` 是 direct replacement。partial route 保留 edge utility。
residual scan 不做 route score 硬切（threshold=0），所有既有 physical complete route 都可参与。

## Residual validation 结果

扫描：

    w = 0, 0.025, 0.05, 0.10, 0.20, 0.40, 0.70, 1.0
    dustbin penalty = -0.5, 0, 0.5, 1.0

冻结 edge threshold：0->1=0.001、1->2=0.05、2->3=0.05。所有选择仅 validation。

validation-selected point 是 `w=0`、dustbin=1.0。所有非零 w 都降低 complete efficiency，说明当前
route query 未提供可用的 route-level 增益。`w=0` 与 same-checkpoint edge-only control 数值完全相同。

| magnitude | raw chain recall | threshold chain recall | eff | purity | fake |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 mm | 1.0000 | 0.9978 | 0.5293 | 0.8561 | 0.2183 |
| 5 mm | 1.0000 | 0.9872 | 0.5556 | 0.8754 | 0.2245 |
| 10 mm | 1.0000 | 0.9782 | 0.4858 | 0.8610 | 0.2356 |
| 50 mm | 1.0000 | 0.6253 | 0.0683 | 0.6600 | 0.3435 |

因此 5/10 mm 的主要问题在 candidate graph 之后：truth chain 仍保留，但 false/mixed/fake route
competition 使 global assignment 不能达到 primary operating point。

## Gate 与后续边界

预先定义 gate：nominal eff>=0.70、purity>=0.95、fake<=0.05；并在 5/10 mm 明显优于 MLP、V1
full-context、V1 no-context 和 same-checkpoint control。结果：

    nominal_primary_point_passed = false
    all_5_10mm_baseline_gains_passed = false
    same_checkpoint_route_context_gain_passed = false
    v2_hypothesis_passed = false

因此严格禁止生成新的 final test bank 或运行 test。下一步若继续，必须提出新的 train/validation
假设，例如训练时显式学习 edge-route utility residual 并加入 event-level assignment surrogate；不能
把本轮失败的 validation/test boundary 当成可调 threshold 的机会。

关键输出：

    outputs/geometry_aware_transformer_v2_validation_v1/
    outputs/geometry_aware_transformer_v2_route_score_audit_validation_v1.json
    outputs/geometry_aware_transformer_v2_route_residual_utility_validation_v1/
    outputs/geometry_aware_transformer_v2_validation_assessment_v1/
