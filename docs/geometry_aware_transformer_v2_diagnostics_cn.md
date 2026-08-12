# Geometry-Aware Transformer V2：V1 机制诊断

## 封存边界

V1 test source 已永久封存。本次诊断只打开已有 physical curriculum 的两个 validation
source：`mc24_100116_00010_00019` 与 `mc24_100117_00010_00019`。

`diagnostic_contract.json` 明确记录 `loaded_splits=["validation"]`、
`forbidden_splits=["test"]`、`test_events_loaded=false` 和
`test_artifacts_opened=false`。输入链路与 candidate graph 未改变：

    SCT_ClusterContainer -> SegmentFitRefit -> SegmentsRefit -> NtupleDumper
    -> FaserActsExtrapolationTool (mode 0)

validation 图包含 1,440 个 full event、18,669 个 node、175,084 条有向 physical message
edge，以及 44,921 条相邻输出 edge（其中 10,511 条为 truth positive）。复用了冻结的 V1
full-context checkpoint、Platt calibration 与 route controls
（0->1=0.001、1->2=0.5、2->3=0.05、dustbin penalty=1.0），没有重新拟合。

## 干预项

以下干预只修改冻结 V1 实际执行的 message，不改输出 candidate edge、物理
residual/covariance feature、calibration、threshold 或 route assignment backend：

- message depth 0、1、2、3、4；
- 四层 adjacent-only、forward-only、backward-only；
- 四个 leave-one-station-out message 干预。

完整记录在
`outputs/geometry_aware_transformer_v2_mechanism_diagnostics_validation_v1/`：
`layer_station_pair_edge_metrics.csv`、`node_state_depth_summary.csv`、
`attention_by_layer_station_pair.csv`、`score_drift_from_full_context.csv`、
`fake_route_contributions.csv` 与冻结 route metric 表。

## 结论

message depth 对 score ranking 的影响很小。相对四层 full graph，depth-zero local score 的
平均绝对概率漂移为 0.0135，rank correlation 为 0.99937；depth-three 分别为 0.00435 和
0.99989。相邻 edge AP 的变化也很小：depth zero 的 0->1/1->2/2->3 AP 为
0.8433/0.8657/0.8962，而 full depth 为 0.8405/0.8666/0.8983。

但 node representation 的混合明显增强。所有 node 的 station 内平均 cosine similarity
从 depth zero 的 0.319 升至 depth four 的 0.549，平均 state norm 从 8.01 升至 21.25。
这与过平滑或尺度增长相符，同时 local residual edge decoder 仍主导最终排序。

route 失败也不是 raw candidate 丢失造成：5 mm 的 threshold truth-chain recall 对 local 和
full message 都约为 0.825。问题在于 full context 没有充分压制 false route。使用冻结 V1
controls 时，10 mm full depth 的 complete efficiency=0.573、purity=0.913、fake=0.129；
false route 同时包含 fake endpoint 与 mixed-truth chain。单向 message 可以改变 efficiency，
但会以 fake rate 为代价，仍无法达到预先定义的 primary point。

因此 V2 的假设被限定为：保持 physical candidate graph 与 V1 backbone，但直接训练完整
route consistency 和 endpoint competition，而不是期待独立 edge score 加普通 message passing
自行解决四站组合问题。

## V2 预先冻结设计

只有当三条已存在的相邻 physical edge 连成 `IFT -> S1 -> S2 -> S3` 时，V2 才构造一个
route candidate。route query 聚合四个 node state、三条 physical edge descriptor、station-pair
embedding 与 base edge evidence，输出 route logit，并将 correction 回投到原来的相邻 edge row。
后端仍使用不变的 unit-capacity route assignment。

损失是预先声明的 weighted edge BCE/focal、route truth-consistency BCE/focal、
endpoint-incidence log-sum-exp one-to-one competition 与 fake-endpoint route softplus penalty
之和。truth ID 与 synthetic role 仅作为 label/loss audit，不作为网络输入。

V2 只会使用 train/validation。same-checkpoint control 仅关闭 route-derived edge correction，
但复用 V2 的 calibration 与 route operating point；它不会得到独立的 threshold 或 calibration
search。只有 V2 先满足 validation 门槛，才会建立新的独立 test bank。

## 复现

    cd /eos/home-x/xcheng/FASER/alignment_ML
    source scripts/setup_environment.sh ml

    python scripts/diagnose_geometry_aware_transformer_v1_context.py \
      --synthetic-manifest outputs/mc24_muon_2dfluka_curriculum_external_full_train_aug_synthetic_v1/synthetic_corpus_manifest.json \
      --v1-validation-dir outputs/geometry_aware_transformer_v1f_final_validation_v1 \
      --output-dir outputs/geometry_aware_transformer_v2_mechanism_diagnostics_validation_v1
