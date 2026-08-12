# 物理错位增强的 MLP 基线

## 范围

本研究仍严格停留在 Transformer 之前。目标是在真实偏移几何重新拟合的样本上训练一个小型 pairwise MLP，并测量它的鲁棒性上限。V1 强制使用
`q_over_p_mode=0`，因为 local segment 的 `q/p` 只是 seed，不能当作可靠测量。

每一个 payload 点都经过：

```text
持久化 SCT_ClusterContainer
  -> SegmentFitRefit -> SegmentsRefit -> NtupleDumper
  -> canonical tracklets + FaserActs mode-0 propagation
  -> synthetic overlay + 物理候选展开
```

候选导出器只能复制同一个 payload 下的 Acts prediction，并且必须先通过 target reference-z 的精确检查。禁止坐标平移或复用缓存 residual。

## Source 切分约定

`configs/physical_curriculum_mlp_muon.yaml` 将 MC24 `s0012-r0019` xAOD 分块分配为 train、validation、test。切分发生在生成 synthetic overlay 之前，单位是原始输入文件。manifest 对每个原始 event 记录
`source_id:run_id:event_id`；加入 `source_id` 是为了防止不同 xAOD 分块发生 run/event 编号重复时泄漏。每个 payload 还会记录实际 refit 后的 event membership。物理重建可以造成 event 丢失，但 payload 绝不能引入来自其他 source split 的 event。

初始 bank 使用五个输入分块：三个训练、一个验证、一个测试。除严格的 source-event 切分外，随机但固定 seed 的 geometry direction trial 也按 split 留出。

## Payload Bank 与课程训练

真实 station-level `dx/dy` payload 的大小为
`0, 0.1, 1, 5, 10, 50 mm`。所有非零方向均由固定 seed 生成，并写入各 source 的 scan config。训练阶段只读取真实 refit 输出，按下列顺序扩展：

1. `0` 到 `0.1 mm`
2. 扩展至 `5 mm`
3. 扩展至 `50 mm`

各阶段共享一个只由 train-split candidate row 拟合的 feature standardizer，并从上阶段权重继续训练。synthetic overlay 只加入 missing tracklet 和 fake/background tracklet，不改变其底层物理 refit state。

在每个阶段中，采样单位是物理 `(source_id, payload_id)` group，所有 group 的抽样概率相同，而不是按 candidate row 数目抽样。因此宽 gate 或 fake 较多的 payload 不会主导训练 exposure。

## 命令

```bash
source alignment_ML/scripts/setup_environment.sh ml
cd alignment_ML

python scripts/build_physical_curriculum_corpus.py \
  --config configs/physical_curriculum_mlp_muon.yaml \
  --output-dir outputs/mc24_muon_curriculum_physical_v1 --resume

python scripts/materialize_curriculum_synthetics.py \
  --physical-manifest outputs/mc24_muon_curriculum_physical_v1/physical_corpus_manifest.json \
  --output-dir outputs/mc24_muon_curriculum_synthetic_v1 --resume

python scripts/run_curriculum_mlp_baseline.py \
  --synthetic-manifest outputs/mc24_muon_curriculum_synthetic_v1/synthetic_corpus_manifest.json \
  --config configs/physical_curriculum_mlp_muon.yaml \
  --output-dir outputs/mc24_muon_curriculum_mlp_v1
```

## 评估约定

每个 candidate `chi2` gate 分别训练一个 MLP。温度缩放只能使用 validation candidate；随后在 validation 的 greedy one-to-one association 上扫描 decision threshold。在给定 inclusive fake-rate 和 inclusive purity 两种约束下，各自选择 efficiency 最高的阈值。测试输出包括：

- 在全部 validation payload 量级上选择的单一全局阈值，用于检验单一部署阈值；
- 对每个 injected magnitude，只从同一量级的 validation payload 选择阈值，用于构造真正可比的固定 fake-rate 与固定 purity 扫描曲线。

这两类阈值都不会读取 test score。测试输出包括：

- MLP 选择前的 candidate truth recall；
- ROC AUC 和 average precision；
- calibration 前后 Brier score、NLL 与 ECE；
- 随 injected misalignment 大小变化的 association efficiency、inclusive purity、inclusive fake rate；
- 各 station pair 的同类指标。

即使 learned operating point 无法满足指定约束，仍会写出
`candidate_gate_test_by_magnitude.csv`，从而把 candidate-recall 的损失与
classifier/assignment 的损失分开审计。

`test_candidate_metrics_by_magnitude.csv` 与
`test_candidate_metrics_by_station_pair.csv` 也一定会写出；即使不存在非空 assignment operating point，它们仍保存 AUC、PR 以及校准前后的 reliability 指标。

每条结果都会保存 `threshold_scope` 与 `validation_magnitude_mm`，避免把单一部署阈值与按量级受控比较混为一谈。

阈值扫描会保留配置文件中声明的网格，并在温度缩放后加入 validation score 的分位数。因此当校准后的分数落在很窄区间时，不会因粗网格漏掉有效的高分 operating point。

若只需重新计算 calibration 与阈值，而不重新训练已完成的模型，可使用新的输出目录并添加
`--checkpoint-root outputs/mc24_muon_curriculum_mlp_v1`。

## 受控试验结果

完成的 MC24 muon 试验使用 45 个真实 physical payload，来自 25 个原始 MC event：train 15 个、validation 5 个、test 5 个，分别来自五个互斥 xAOD 分块。对应的 synthetic multi-track event 数为
`2640/480/480`。因此它适合验证可复现性和失效模式，但还不是统计精度的最终结论。

在 50 mm，candidate truth recall 分别为：`chi2=250` 时 `0.298`，`chi2=1000` 时 `0.567`，`chi2=5000` 时 `0.891`，不加 gate 时为 `1.000`。无 gate MLP 的 test ROC AUC 从 0 mm 的 `0.846` 变为 50 mm 的 `0.804`，average precision 从 `0.466` 变为 `0.468`。仅使用 validation 的 temperature scaling 将无 gate test ECE 在 0 mm 从 `0.146` 降至 `0.063`，在 50 mm 从 `0.154` 降至 `0.096`。

预先声明的主 operating target 是 inclusive fake rate <= `0.05` 和 inclusive purity >= `0.95`。所有 gate（包括 nominal geometry）都不存在非空的 validation 解。因此没有报告主 fixed-operating-point 下的 association-efficiency 曲线；在约束没有满足时伪造一条曲线会造成误导。无 gate 的 candidate recall 已经足够，但 pairwise MLP 在分离高错位退化之前，甚至 nominal 就不能满足主 association 要求。它尚不满足预先声明的启动 Transformer 证据条件。

作为量化背景，四个 gate 的最小非空 validation fake rate 分别为：`250` 时 `7.1%`，`1000` 时 `11.2%`，`5000` 时 `42.9%`，无 gate 时 `38.5%`。紧 gate 最接近主要求，但会损失 50 mm candidate recall；放宽 gate 虽恢复候选，却暴露当前 greedy pairwise assignment 的限制。

结果位于
`outputs/mc24_muon_curriculum_mlp_batch8192_reassessed_v1/`，其中包括
`candidate_metrics_vs_misalignment.png`、按错位大小和 station pair 的 candidate 指标、calibration 以及严格 source split 审计。

只有当大偏移点的 candidate recall 仍高，而已校准 MLP 的 efficiency 相比小偏移明显降低时，测试曲线才构成启动 Transformer 的证据。
`metrics.json` 会按预先声明的 screening criterion 汇总证据，但绝不会自动启动 Transformer。
