# 物理错位增强与 Curriculum MLP 基线

状态：完成（受控 pilot；尚非统计精度结论）。

## 本阶段目的

在不启动 Transformer 的前提下，先测量经由真实 displaced-geometry refit 数据增强后的低容量 pairwise MLP 的鲁棒性上限。所有训练和评估使用 mode 0；local segment 的 `q/p` 不作为可测量输入。

## 生产版 MC source 切分

使用 `/eos/experiment/faser/data0/sim/mc24/particle_gun/100012/rec/s0012-r0019/` 中的五个独立 xAOD 分块：

| split | source 文件区间 | 数量 |
| --- | --- | --- |
| train | 00000--00004, 00005--00009, 00010--00014 | 3 |
| validation | 00015--00019 | 1 |
| test | 00020--00024 | 1 |

切分单位是原始 xAOD 文件，而不是 synthetic event ID。manifest 的原始 event 唯一键为 `source_id:run_id:event_id`，因此不同输入文件不会因 run/event 编号碰撞而泄漏。

## 生产版 refit smoke

对 `00000--00004` 的第一 event 使用真实零偏移 `/Tracker/Align` payload 完成：

```text
SCT_ClusterContainer -> SegmentFitRefit -> SegmentsRefit
-> NtupleDumper -> canonical tracklets + Acts propagation
```

结果：四个 station 各有一条 tracklet，四条协方差均有效；导出 18 条 propagation record，全部成功，其中 12 条有 propagation covariance。说明生产版 `s0012-r0019` 与此前验证的 dev 样本链兼容。

## 物理 payload bank

配置：`configs/physical_curriculum_mlp_muon.yaml`。

- 量级：`0, 0.1, 1, 5, 10, 50 mm`；
- station 0 固定为 reference；
- station 1--3 的非零 `dx/dy` 方向由 seed `20260810` 生成，并写入每个 source 的实际 scan config；
- train 有两个方向，validation/test 各有一个未见方向；
- 每个 payload 必须实际生成 SQLite/POOL conditions，并重新跑 cluster 到 Acts；不允许坐标平移或缓存 residual。

对 train source `00000--00004`，零偏移与 `0.1 mm` 的 `train_00` 输出已直接比较：

| station | refit state 的平均 `(dx,dy)` [mm] |
| --- | --- |
| 0 | `(0, 0)` |
| 1 | `(0.03162205145, -0.09486857152)` |
| 2 | `(-0.01325851571, -0.09911716179)` |
| 3 | `(-0.07167666335, -0.06973131241)` |

这些数值与注入 payload 一致；`tx,ty` 只有浮点舍入级差异。两点均有 24 条 truth-matched mode-0 field-aware pair，场传播结果从新输出计算，未复用历史 residual。

## 当前实现

新增模块：

- `scripts/build_physical_curriculum_corpus.py`：生成 source 分离的真实 payload/refit bank；
- `scripts/materialize_curriculum_synthetics.py`：仅从同一 payload 的 tracklet 和 Acts record 生成 overlay/candidate；
- `training/curriculum_mlp.py`：按 `0--0.1`、到 `5`、到 `50 mm` 继续训练，并对 `(source_id,payload_id)` 等概率采样；
- `evaluation/pairwise_metrics.py`：ROC AUC、average precision、Brier、NLL、ECE 和 validation-only temperature scaling；
- `scripts/run_curriculum_mlp_baseline.py`：扫描 candidate chi2 gate、validation threshold、固定 fake-rate/purity operating point，并输出按错位大小和 station pair 的测试指标。

`pytest -q`：21 passed（本阶段最终代码验证）。

## 语料完成审计

物理 corpus：`outputs/mc24_muon_curriculum_physical_v1/physical_corpus_manifest.json`。

- 5 个原始 xAOD source，25 个原始 MC event：train/validation/test=`15/5/5`；所有 event UID overlap 均为零；
- 45/45 个物理 payload 完成：train 33、validation 6、test 6；累计 3,753 条真实 Acts propagation record；
- 每个 payload 都由 SCT cluster 至 segment refit、tracklet exporter 和 mode-0 Acts propagation 重新产生；
- synthetic corpus 有 45 个同 payload 样本，train/validation/test synthetic event=`2640/480/480`，tracklet row=`43186/7867/7870`，field candidate record=`264541/48367/48407`；
- loader 现在强制每个 synthetic payload 给出 `physical_event_uids`，并验证它是原始 source UID 的子集，避免 source split 泄漏。

## MLP 训练与评估

最终训练输出：`outputs/mc24_muon_curriculum_mlp_batch8192_v1/`；只读 checkpoint 的最终 reassessment：`outputs/mc24_muon_curriculum_mlp_batch8192_reassessed_v1/`。

- curriculum 保持 `0--0.1 mm -> 5 mm -> 50 mm` 与 `50/60/80` epoch；
- 共享 GPU 被占用，故使用 CPU；batch 从 1024 调为 8192 以避免宽 candidate graph 上的小 batch 线程开销，四个 gate 统一使用此配置；
- 训练后对 validation score 做 temperature scaling；threshold grid 保留显式网格并加入 101 个 validation-score quantile；
- fixed fake-rate/purity 阈值均只在 validation 选择；test 从不参与 standardizer、训练、calibration 或阈值选择；
- 即使 operating point 不存在，仍写出按错位与 station pair 的 candidate AUC、AP、raw/calibrated Brier、NLL、ECE。

## 结果

50 mm 的 test candidate truth recall：

| candidate gate | recall |
| --- | --- |
| 250 | 0.298 |
| 1000 | 0.567 |
| 5000 | 0.891 |
| 无 gate | 1.000 |

无 gate 是 candidate coverage 足够的关键对照：test ROC AUC 从 0 mm 的 0.846 变为 50 mm 的 0.804，AP 从 0.466 变为 0.468；ECE 经 validation-only temperature scaling 后，在 0 mm 为 `0.146 -> 0.063`，在 50 mm 为 `0.154 -> 0.096`。

按 station pair 的无 gate AP 中，`1->3=0.291` 最弱；`0->1=0.670`、`0->2=0.603`、`2->3=0.669` 较高。这说明单对分类分辨能力在 station pair 间差异明显，不能只看全局 AUC。

预先固定的主 operating target 为 inclusive fake rate <= 0.05 或 inclusive purity >= 0.95。四个 gate 的 validation 都没有任何非空 matching threshold 达到该要求，包括 nominal geometry。因此：

1. 主 fixed-operating-point association-efficiency 曲线为空，这是约束不可达的真实结果，不是缺失数据；
2. 不可将没有满足约束的阈值替换成任意阈值后称为固定 fake rate；
3. 无 gate 在 50 mm 的 recall 已为 1.0，但 pairwise MLP 在 nominal 就无法满足主 association 要求；
4. 因而当前无法验证“仅在大错位时 efficiency 明显下降而 candidate recall 仍足够”的预注册 Transformer 触发条件；本阶段不启动 Transformer。

最小非空 validation fake rate 的审计为：`chi2=250: 7.1%`、`1000: 11.2%`、`5000: 42.9%`、无 gate: `38.5%`。紧 gate 最接近主要求，却在 50 mm 丢掉大量 candidate；无 gate 则保留全部 truth candidate，但 greedy pairwise assignment 的 purity 明显不足。

图：`outputs/mc24_muon_curriculum_mlp_batch8192_reassessed_v1/candidate_metrics_vs_misalignment.png`。

## 后续判断

下一轮应先扩展独立 source xAOD 分块的数量，并在保持相同真实 physical refit/Acts 条件链的前提下，审计 pairwise one-to-one assignment 的操作点定义与 synthetic fake 难度。只有在更大 source-independent 样本上，无 gate candidate recall 仍高且可达到明确 primary operating point 后，才可以用该曲线判断是否需要多站全局上下文；目前不启动 Transformer。
