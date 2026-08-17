# Geometry-Aware Transformer V3：结构化全局指派学习

## 当前状态与边界

V3 已实现为只允许 train/validation 的研究假设。扩展后的真实物理语料已经完成，但 V3 尚未在其上完成训练和评估；因此本记录只给出已完成 control 的结果，不给出 V3 性能结论。

V1 test source 继续永久封存。V3 的配置、训练、calibration、threshold 选择和物理 source production 都显式排除 `test`；不会打开 V1 test event 或 artifact，也不创建新的 test bank。

物理输入链保持不变：

```text
SCT_ClusterContainer -> SegmentFitRefit -> SegmentsRefit -> NtupleDumper
-> FaserActsExtrapolationTool (mode 0)
```

每个 curriculum point 都由真实 `/Tracker/Align` conditions payload 开始，随后重新执行 segment refit 和 Acts propagation。V3 不使用 coordinate-level surrogate，不修改持久化 tracklet state，也不把 local segment `q/p` 当作 measurement。

## 固定模型与 Solver

V3 保留 V1/V2 route encoder：`d_model=128`、4 个 sparse block、8 个 attention head、FFN `128 -> 256 -> 128`。输入仍为既有的 ungated mode-0 physical candidate graph 及 residual/covariance feature。完整 candidate 只由已有相邻 physical edge 拼接：

```text
IFT -> S1 -> S2 -> S3
```

既有的 unit-capacity route solver 继续作为 inference backend。V3 只新增通用的精确 unit-capacity packing 接口供 loss-augmented training 使用；不改变 candidate graph、matching 数据定义或最终 solver。

完整 route 的学习 utility 为：

```text
U_theta(r) = sum_{e in r} edge_logit_theta(e) + route_residual_theta(r).
```

partial/missing-station route 继续使用既有 edge-based utility。truth ID、fake 标签、hard-negative 标签与 source provenance 都只作监督或审计字段，绝不作为网络输入。

## 结构化目标

对每个 event，V3 先构造 `Y`：它是 complete route 中 truth-consistent、maximum-cardinality、unit-capacity 的全局指派。随后用相同的精确 solver 找到最强可行竞争 assignment：

```text
Y_hat = argmax_A [ U_theta(A) + m * Delta(A, Y) ]
L = max(0, U_theta(Y_hat) + m * Delta(Y_hat, Y) - U_theta(Y)).
```

`Delta` 对 mixed-truth route、fake-endpoint route、duplicate truth alternative 和 endpoint-conflicting alternative 施加 loss augmentation；hard negative 叠加明确的严重度项。训练因此直接面对 V2 已观测到的 event-level 竞争，而不是独立逐行 BCE。

主 V3 run 关闭 independent edge BCE 与 independent route BCE。early stopping 只按 validation structured hinge，再按 validation violation fraction 选择；calibration 和 route threshold 不参与 checkpoint 选择。

完整 route 是四 endpoint hyperedge，普通矩阵 Sinkhorn 不是精确 relax。项目另提供一个明确标记的 capacity-normalized differentiable soft-assignment hyperedge surrogate 作为对照。默认权重为零，不改变主 structured-margin 结果。

## Validation 协议

checkpoint 固定后，`evaluate_structured_assignment_v3.py` 只打开 validation sample。它仅在 validation 拟合 station-pair edge calibration，并选择 edge threshold、unmatched utility 与 V3 route-utility temperature；同时报告 V3 route-utility stream 和同一 checkpoint 的 edge-only route-solver control。

冻结的 validation gate：

```text
nominal complete efficiency >= 0.70
nominal complete purity     >= 0.95
nominal fake rate           <= 0.05
```

还要求 5/10 mm 相对在相同 expanded corpus 上重新训练的 MLP 和 V1 control 有清晰增益。同时预先定义同 architecture 的 expanded V2 BCE/focal-loss control，用来把 structured objective 与 route-encoder capacity、data volume 区分开。冻结 V2 结果只作为补充历史背景。只有该 validation 假设通过后，才允许建立新的 source-disjoint multidirection test bank。

## 匹配的 Control 重训

扩展语料比较已具备明确的 train/validation-only control。`run_global_assignment_mlp_baseline.py --validation-only` 现在会在解析 test asset 前过滤 manifest，并为 ungated station-pair route control 输出 `route_calibration.json`。`train_geometry_aware_transformer_v1.py` 也会在加载前过滤，且不再解析 legacy `frozen_route_test` reference。

共享 pairwise MLP 使用 `physical_global_assignment_mlp_v3_expanded_control.yaml`，四个 V1 ablation 使用 `geometry_aware_transformer_v1_expanded_control.yaml`。两者都声明 `allowed_splits=[train, validation]`、`forbidden_splits=[test]`、不变的 mode-0 physical graph 和 GPU training。`geometry_aware_transformer_v2_expanded_bce_control.yaml` 是匹配的 route-encoder BCE/focal-loss control：它与 V3 有相同 128 维四层 backbone，且不含历史或 test reference。

## 已完成的扩展 MLP Control

重新训练的 pairwise MLP 已在 `outputs/mc24_v3_expanded_trainval_mlp_control_v1` 完成 train/validation-only run。其 contract 记录 994 个 train 与 796 个 validation original source event、UID 交集为零、`test_events_loaded=false`，且没有访问 test artifact。对 ungated station-pair 的温度 calibration 在 373,325 条 validation candidate 上将 aggregate ECE 从 `0.0992` 降至 `0.0344`。

在 validation 选择的 station-pair multistation-flow 点（score threshold `0.10`、unmatched penalty `0.5`），nominal association efficiency 为 `0.9578`、purity 为 `0.9938`、fake rate 为 `0.00621`、candidate truth-edge recall 为 `0.99815`。这不是四站 primary operating point：到 `5 mm` 时，station-pair efficiency 降至 `0.4447`、purity `0.9273`、fake rate `0.0727`，而 candidate truth-edge recall 仍有 `0.99775`。

独立的四站 route control 位于 `outputs/mc24_v3_expanded_trainval_mlp_route_validation_v1`。它固定 MLP 权重和 station-pair calibration，只在 validation 选择 route threshold 与 dustbin utility；最终选择每条 adjacent pair 的 threshold 都为 `0.001`、unmatched penalty 为 `0.0`。nominal complete-track efficiency 为 `0.9094`、complete-track purity 为 `0.9724`，但 route fake rate 为 `0.06286`，高于冻结的 `0.05` 上限，因此 nominal capture 也为 false。到 `5/10/50 mm`，complete-track efficiency 都为 `0.0`；同时每个扫描 magnitude 的 physical candidate complete-truth-chain recall 都保持 `1.0`。这说明当前瓶颈在 score/structured competition，而不是 candidate graph 丢失 truth chain。它只是 validation-only control，不允许建立新 test bank，并将作为 V1/V2/V3 的明确基线。

validation-refreeze utility 现在也只从 partial corpus manifest 加载 `train` 与 `validation` entry，不要求未来的 test split；旧版可选 `selection_policy` metadata 具有明确默认值，扩展 control 不会在计算完成后因缺少这一个说明字段而失败。

## 已完成的扩展 V1 Control

四个 V1 control 现在都已得到 source-disjoint validation operating point。ordinary sparse 与 full-context geometry-aware checkpoint 只训练一次，随后以不改权重的方式 validation-refreeze；no-chi2 与 no-context 则独立训练。下表全部使用同一 physical mode-0 candidate graph、station-pair Platt calibration、validation-only route grid，且没有读取任何 test asset。

| V1 control | candidate AP | nominal eff / purity / fake | 5 mm eff / purity / fake | 10 mm eff / purity / fake | 通过 capture 的 magnitude |
| --- | ---: | --- | --- | --- | --- |
| ordinary sparse | 0.8858 | 0.9637 / 0.9708 / 0.0350 | 0.9125 / 0.9464 / 0.0586 | 0.2980 / 0.7934 / 0.1217 | 0, 0.1, 1 mm |
| geometry-aware | 0.8908 | 0.9547 / 0.9664 / 0.0432 | 0.8833 / 0.9484 / 0.0593 | 0.2802 / 0.7761 / 0.1200 | 0, 0.1, 1 mm |
| geometry-aware，无显式 chi2 项 | 0.8938 | 0.9552 / 0.9659 / 0.0439 | 0.9079 / 0.9502 / 0.0569 | 0.3297 / 0.7907 / 0.1170 | 0, 0.1, 1 mm |
| geometry-aware，无多站 context | 0.9157 | 0.9358 / 0.9554 / 0.0481 | 0.8690 / 0.9460 / 0.0586 | 0.5760 / 0.8879 / 0.0959 | 0, 1 mm |

没有任何 V1 control 在 5 mm 通过冻结的 primary criterion，主要是 route purity/fake rate 不满足，即使 candidate truth-chain recall 仍被保留。full-event geometric context 在这一组匹配 V1 比较中没有带来改善。该结果支持继续测试 V2 route objective 与 V3 structured assignment objective，但不允许创建 test。

## Exact Oracle 吞吐量

`scripts/audit_structured_route_oracle.py` 只审计声明的 train 或 validation sample。在 7,200 个 expanded train graph 上，平均每个 graph 有 116.8 个 complete physical route candidate（median 108、p99 336、maximum 720）；endpoint-conflict component 通常覆盖整个 event，而不是若干互相独立的 track。对固定的 64-event deterministic sample，原 generic MILP benchmark 平均为 24.2 ms/event。

exact unit-capacity solver 现对完整的固定顺序 IFT/S1/S2/S3 tuple component 使用经过验证的 multipartite dynamic-programming fast path；mixed route length、endpoint position 不可分或 state 过大的情形仍回退到既有 MILP。单元测试将其与 brute-force optimum 比较，并保留 MILP fallback 覆盖。当前已修正的 64-event audit 平均为 4.61 ms/event（median 2.99 ms），候选图与数学目标均不变。这消除了 V3 structured oracle 的实际吞吐量瓶颈，而没有改变 V3 监督定义。

## 扩展物理语料

`configs/physical_curriculum_v3_expanded_trainval.yaml` 定义 10 个 MC24 train source file 与 8 个 source-file-disjoint validation file。两个 split 都含 `mumi` 与 `mupl` production chunk。重建接受前的请求目标是最多 1,000 个 train 与 800 个 validation source-event slot；实际被接受数量只在每个物理链完成后记录。

本轮 production 已完成。单进程 physical-manifest refresh 接纳了全部 `108/108` 个计划 point：10 个 train source file 的 60 个点和 8 个 validation source file 的 48 个点。Calypso export 后的 provenance 含 994 个不同 train 与 796 个不同 validation `source_event_uid`，两者交集为零；本语料不含任何 test source 或 artifact。每个 accepted point 的 payload 都与 scan plan 的逐站 dx/dy 注入一致，具有四站 refitted tracklet、全部 positive-definite covariance，以及覆盖六个 forward station pair 的 finite、successful mode-0 Acts record。按 train 后 validation 的 pooled materialization 为每个 split 生成 6 个 sample；每个 train/validation payload 分别汇聚准确的 994/796 个 physical source event，所有 candidate export 均报告 `q_over_p_mode=0` 且 target-z mismatch 为零。

完整 nominal refitted-state audit 保存在 `outputs/mc24_v3_source_audit_v1/refitted_source_diversity_expanded_trainval.json`。18 个 source 合计观测到 1,790 个 event、6,690 条 tracklet、6,488 条 truth-matched muon tracklet，且两种 truth PDG charge 都出现。实际 state 包络为 `x=[-123.13, 123.94] mm`、`y=[-121.99, 125.97] mm`、`tx=[-0.1905, 0.2244]`、`ty=[-0.0991, 0.1617]`。这说明被接纳语料具有实际的 charge 与 refitted state 多样性，但不把它夸大为对全部 physics production coverage 的穷尽证明。

一个独立 train source 和一个 validation source 已完成真实 nominal smoke：分别导出 100/99 个 event 与 373/371 条 tracklet，所有导出 tracklet covariance 都 positive definite，且有每个 station pair 的 mode-0 propagation record。`audit_refitted_source_diversity.py` 从 refitted tracklet 而非文件名测量覆盖：前两个 source 分别有 360/364 条 truth-matched muon tracklet，观测到的 `x/y` 覆盖约为 `[-115, 120] mm` 与 `[-104, 123] mm`，且 `tx/ty` 有非零 spread。其余 source 和 payload point 在各自 output audit 通过前不得进入训练。

新增 nominal train source `mc24_100043_00200_00299` 的配对 propagation-component audit 进一步确认 mode 选择。505 条同时存在于两个 mode 的 truth-matched record 中，median chi2 从 mode-0 的 `5.9670` 变为 mode-1 的 `19.9836`。固定 mode-0 combined covariance、仅替换 residual 的 median effect 是 `-2.56e-05`；在 mode-1 residual 下替换 covariance 的贡献为 `6.4121`，combined-covariance log-determinant 的 median change 为 `-14.0282`。因此 chi2 增大主要是 uncertainty 收窄，而不是 raw residual 恶化。V3 继续只用 mode-0；mode-1 的固定 q/p seed 不被当成可靠的 local momentum measurement。

仅凭 production 文件名仍不能证明全部角度、位置或 conditions diversity；refitted-state audit 是观测覆盖的接受证据，若要声称更广 production 覆盖仍需单独的 Calypso/truth audit。

首次 EOSSubmit cluster `990949` 在任何 payload 工作前退出，原因是 staged worker 从 Condor scratch 路径推导 project root。后续 cluster `990951` 暴露第二个仅启动阶段的问题：严格 nounset shell option 被外部 LCG setup 脚本继承。worker 现已显式接收 EOS project root，并且只在 source setup 脚本的短作用域内关闭 nounset。当前有效提交是 Condor cluster `990952`，共 18 个 worker，每个 source 处理 6 个真实 payload magnitude。worker 不并发写 shared corpus manifest；成功 output audit 后由单一聚合步骤刷新 provenance。现在 physical point 只有当其持久化 `/Tracker/Align` payload 与 scan plan 的逐站注入一致、具备 refitted tracklets、无 `failure.json`，且 `content_audit.json` 确认 MC label、四站覆盖和每条导出 tracklet covariance positive definite 时，才会被接纳；其 propagation ROOT 还必须为每个 configured forward station pair 含有 finite、successful、带 covariance 的 mode-0 record。

## 复现

```bash
cd /eos/home-x/xcheng/FASER/alignment_ML
source scripts/setup_environment.sh ml

# 仅准备或刷新 source-local 配置和 provenance，不使用 test source。
python scripts/build_physical_curriculum_corpus.py \
  --config configs/physical_curriculum_v3_expanded_trainval.yaml \
  --output-dir outputs/mc24_v3_expanded_trainval_physical_v1 \
  --prepare-only --resume

# 创建 EOSSubmit 兼容 worker。加入 --submit 才提交 bank。
python scripts/submit_physical_curriculum_condor.py \
  --config configs/physical_curriculum_v3_expanded_trainval.yaml \
  --physical-output-dir outputs/mc24_v3_expanded_trainval_physical_v1 \
  --submit-dir outputs/NEW_CONDOR_SUBMISSION \
  --skip-complete

# 所有需要的 physical point 通过后，刷新 manifest，再分别 materialize split。
python scripts/materialize_pooled_curriculum_synthetics.py \
  --physical-manifest outputs/mc24_v3_expanded_trainval_physical_v1/physical_corpus_manifest.json \
  --config configs/physical_curriculum_v3_expanded_trainval.yaml \
  --output-dir outputs/NEW_V3_SYNTHETIC --split train
python scripts/materialize_pooled_curriculum_synthetics.py \
  --physical-manifest outputs/mc24_v3_expanded_trainval_physical_v1/physical_corpus_manifest.json \
  --config configs/physical_curriculum_v3_expanded_trainval.yaml \
  --output-dir outputs/NEW_V3_SYNTHETIC --split validation --resume

`--resume` 现在只替换本次显式请求的 split，并保留已经 materialize 的另一 split entry；它会拒绝来自不同 physical corpus 的 manifest 和重复的 `(split, payload_id)`。因此第二次仅 validation 的调用不会覆盖 train provenance。

python scripts/train_structured_assignment_v3.py \
  --synthetic-manifest outputs/NEW_V3_SYNTHETIC/synthetic_corpus_manifest.json \
  --config configs/geometry_aware_transformer_v3.yaml \
  --output-dir outputs/NEW_V3_TRAIN_VALIDATION
python scripts/evaluate_structured_assignment_v3.py \
  --synthetic-manifest outputs/NEW_V3_SYNTHETIC/synthetic_corpus_manifest.json \
  --training-dir outputs/NEW_V3_TRAIN_VALIDATION \
  --config configs/geometry_aware_transformer_v3.yaml \
  --output-dir outputs/NEW_V3_VALIDATION
```

同一 synthetic manifest 完成后，匹配 MLP/V1 control 的命令为：

```bash
python scripts/run_global_assignment_mlp_baseline.py \
  --synthetic-manifest outputs/NEW_V3_SYNTHETIC/synthetic_corpus_manifest.json \
  --config configs/physical_global_assignment_mlp_v3_expanded_control.yaml \
  --output-dir outputs/NEW_EXPANDED_MLP_VALIDATION --validation-only

python scripts/evaluate_pairwise_mlp_route_validation.py \
  --synthetic-manifest outputs/NEW_V3_SYNTHETIC/synthetic_corpus_manifest.json \
  --checkpoint outputs/NEW_EXPANDED_MLP_VALIDATION/mlp_pair_classifier.pt \
  --frozen-calibration outputs/NEW_EXPANDED_MLP_VALIDATION/route_calibration.json \
  --config configs/pairwise_mlp_route_expanded_control.yaml \
  --output-dir outputs/NEW_EXPANDED_MLP_ROUTE_VALIDATION --device cuda

python scripts/train_geometry_aware_transformer_v1.py \
  --synthetic-manifest outputs/NEW_V3_SYNTHETIC/synthetic_corpus_manifest.json \
  --config configs/geometry_aware_transformer_v1_expanded_control.yaml \
  --output-dir outputs/NEW_EXPANDED_V1_VALIDATION

python scripts/train_route_aware_transformer_v2.py \
  --synthetic-manifest outputs/NEW_V3_SYNTHETIC/synthetic_corpus_manifest.json \
  --config configs/geometry_aware_transformer_v2_expanded_bce_control.yaml \
  --output-dir outputs/NEW_EXPANDED_V2_BCE_VALIDATION
python scripts/evaluate_route_aware_transformer_v2_direct_routes.py \
  --synthetic-manifest outputs/NEW_V3_SYNTHETIC/synthetic_corpus_manifest.json \
  --v2-validation-dir outputs/NEW_EXPANDED_V2_BCE_VALIDATION \
  --config configs/geometry_aware_transformer_v2_expanded_direct_routes.yaml \
  --output-dir outputs/NEW_EXPANDED_V2_BCE_ROUTE_VALIDATION --device cuda
```

训练与评估命令要求 completed manifest 只包含声明的 train 与 validation 输入；违反 test 封存边界会被契约检查拒绝。

## 已完成的扩展语料 Validation

下表全部使用同一真实 mode-0 physical candidate graph、按 original xAOD file 严格隔离的 source split，以及仅 validation 的 calibration/control 选择。没有结果打开 test event 或 artifact。primary capture 门槛固定为 `efficiency >= 0.70`、`purity >= 0.95`、`fake rate <= 0.05`。

| control | nominal eff / purity / fake | 5 mm eff / purity / fake | 10 mm eff / purity / fake | 通过 capture 的 magnitude [mm] |
| --- | --- | --- | --- | --- |
| pairwise MLP + route solver | 0.9094 / 0.9724 / 0.0629 | 0.0000 / 0.0000 / 0.7481 | 0.0000 / 0.0000 / 0.9772 | 无 |
| V1 ordinary sparse | 0.9637 / 0.9708 / 0.0350 | 0.9125 / 0.9464 / 0.0586 | 0.2980 / 0.7934 / 0.1217 | 0, 0.1, 1 |
| V1 geometry-aware | 0.9547 / 0.9664 / 0.0432 | 0.8833 / 0.9484 / 0.0593 | 0.2802 / 0.7761 / 0.1200 | 0, 0.1, 1 |
| V2 BCE route query | 0.9163 / 0.9667 / 0.0443 | 0.7559 / 0.9462 / 0.0725 | 0.2360 / 0.7935 / 0.1446 | 0, 0.1, 1 |
| V3 exact structured margin | 0.7988 / 0.8639 / 0.2523 | 0.5450 / 0.8294 / 0.2638 | 0.2188 / 0.5953 / 0.3218 | 无 |
| V3 exact margin + soft-assignment control | 0.7335 / 0.8509 / 0.2779 | 0.3884 / 0.7256 / 0.3141 | 0.1546 / 0.5405 / 0.3810 | 无 |

扩展 V2 control 通过 nominal validation，但在 5 mm 同时不满足 purity 和 fake-rate 门槛。两个 V3 variant 都未通过 nominal validation。对 exact-margin V3，raw physical candidate 的 complete truth-chain recall 在所有 magnitude 都为 `1.0`；经过 score threshold 后，nominal、5 mm、10 mm 分别为 `0.8952`、`0.9110`、`0.9014`。route utility 将同 checkpoint edge-only control 的 nominal complete-track efficiency 从 `0.4460` 提升到 `0.7988`，但引入过多 fake route。soft control 在 retention 和 route quality 两方面都更差。因此当前 V3 的限制是 score calibration/selectivity 与 structured-loss optimization，而不是 physical candidate graph 丢失 truth chain。

primary exact-margin checkpoint 在 validation epoch 2 被选中，soft-control checkpoint 在 epoch 1 被选中。primary run 在 soft weight 为零时严格跳过可选 surrogate；labelled control 才启用它。GPU 上 batch size 设为 16 以提高吞吐量，encoder 和 event-level objective 均未改变。exact oracle audit 现正确保留浮点最大值（`29.93 ms`），并在固定的 train-only 64-event 样本上得到平均 `4.61 ms/event`。

`scripts/plot_expanded_validation_route_controls.py` 与 `configs/expanded_validation_route_controls.yaml` 可复现 MLP/V2/V3 的 validation-only 对照图和 CSV。生成图为 `outputs/mc24_v3_expanded_trainval_validation_control_plots_v3/route_validation_control_comparison.png`。

因此 V3 validation hypothesis 不会被冻结为成功，也不会生成或打开新的 multidirection test bank。后续机制工作必须继续只用 train/validation，针对 route-score selectivity 或 structured objective，而后才可重新讨论 test 决策。
