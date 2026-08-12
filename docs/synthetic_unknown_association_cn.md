# 真实 Payload 的 Synthetic Unknown-Association Baseline

该受控研究只在完整的固定 truth physical capture scan 结束后启动。它提供两个不使用 Transformer 的关联参考：传统 field-aware chi-square matching，以及仅在 nominal geometry 训练的小型 MLP pair classifier。

## 物理输入契约

每个 physical scan point 的输入均来自该点独立 refit 后的 canonical tracklet 文件和 mode-0 `FaserActsExtrapolationTool` propagation export：

```text
本点专属 /Tracker/Align payload
  -> SCT_ClusterContainer -> SegmentFitRefit -> SegmentsRefit
  -> tracklets.root + mode-0 propagations.root
  -> 确定性的 synthetic multi-track overlay
  -> field-aware candidate records
  -> unknown-association baseline 评估
```

overlay 会为每条 synthetic tracklet（包括复制得到的 fake row）保存 `origin_run_id`、`origin_event_id` 和 `origin_tracklet_id`。candidate prediction 只会从 source provenance 与 target station 都一致的同一条 mode-0 Acts record 复制，并且必须通过 exported target z 与 synthetic target z 的一致性检查。因此 event overlay 不会把 propagation 替换为坐标平移或直线代理。candidate ROOT tree 中的 propagation truth 字段刻意写为 `-1`；truth 不会进入 candidate 构造或 matching。

当前 MC24 control 的 mode-0 station reference plane 在各 event 间相同，因此这种 fan-out 成立。若 target-z 不匹配，会被计数并丢弃，绝不静默近似。

## Synthetic 样本与数据划分

`configs/synthetic_unknown_association_muon.yaml` 定义 100 个确定性的 synthetic events；每个 event 叠加四条 source tracks，真实 tracklet missing probability 为 0.10，每个 station fake row 的 Poisson 均值为 0.50。所有 physical payload point 使用相同 RNG seed 和相同的 overlay layout。

MLP 只在 nominal-payload synthetic sample 训练一次。确定性的 70/30 event split 会写入 `heldout_event_ids.json`。传统 field-chi2 和冻结的 MLP 对每一个 injected geometry 都只在这同一份 held-out event 子集上评估，避免将 MLP 的 nominal 训练 events 当作其报告的 baseline performance。

当前 source control 只从五个 xAOD events 中选出四条满足 complete 条件的 muon tracks。overlay 按设计复用这些状态，因此曲线测量的是受控的 geometry robustness，而不是独立的 particle-level 泛化能力。

## Baseline

field-chi2 reference 考虑全部有序的 forward station pairs，并在每个 station pair 内采用传统的 ascending-chi2 greedy one-to-one assignment。其固定 `1500` gate 在评估 displaced points 之前已经确定；输出中保存 nominal truth-chi2 audit。

MLP 使用两层宽度为 64 的 ReLU hidden layers。输入包括 field-aware residual/pull 分量、`log1p(chi2)`、combined-covariance log determinant、delta-z、local fit quality、hit count 和 station-pair one-hot 特征。它不读取 truth、q/p 或 injected alignment。validation candidate-F1 scan 固定输出 threshold；在该 baseline 中它对应 unmatched 决策。

全流程强制 `q_over_p_mode=0`。segment q/p 只是 seed，而不是可信 local measurement，因此没有 q/p feature。

## 指标

每个 geometry point 与 station pair 都保存 candidate 数、truth-possible match 数、prediction 数和 correct 数。scan 绘制并比较：

- association efficiency：correct / possible truth matches；
- inclusive association purity：correct / 全部 predictions；
- inclusive fake rate：non-correct / 全部 predictions。

inclusive 指标会将 synthetic fake 或 unknown endpoint 计作 false prediction。为保持诊断连续性，每个 point 的 metrics JSON 仍保留 legacy 的仅对 scorable endpoint 计算的 purity 和 fake rate。

## 当前 Held-Out 结果

完成的结果位于
`outputs/mc24_muon_fasernu_physical_unknown_association_v2_heldout/`。30 个 held-out
synthetic events 含 572 个 truth-possible pair matches。零偏移时，field-chi2 的
efficiency/purity/inclusive-fake-rate 为 `0.834/0.796/0.204`；nominal MLP 为
`0.892/0.875/0.125`。

在 `10 mm`，field-chi2 的平均 efficiency 为 `0.812`，冻结 MLP 为 `0.449`。此时 MLP
表面上稳定的 purity 来自 nominal threshold 拒绝大量 candidate，而不是稳健的 recall。在
`50 mm`，平均 efficiency 分别为 `0.280` 和 `0.134`；在 `250 mm` 为 `0.075` 和
`0.017`。在 `500 mm`，MLP 在全部方向的 efficiency 都为零；在 `1000 mm`，它不产生 match。
chi-square 在极端偏移的 purity 有 survivor bias，因为固定 gate 只留下很少的 candidate。

这些证据不支持现在启动 Transformer。下一步 association 工作应是使用更大且 source-event
independent 的 MC 做 misalignment augmentation 与 threshold/calibration study，再比较
alignment-aware candidate gating。

## 复现

只有当所有 physical points 都已经具有完成的 `closure.json` 后，才运行：

```bash
cd /eos/home-x/xcheng/FASER
source alignment_ML/scripts/setup_environment.sh ml
python3 alignment_ML/scripts/run_synthetic_unknown_association_scan.py \
  --physical-scan-root alignment_ML/outputs/mc24_muon_fasernu_physical_capture_scan_v1 \
  --output-dir alignment_ML/outputs/mc24_muon_fasernu_physical_unknown_association_v1 \
  --config alignment_ML/configs/synthetic_unknown_association_muon.yaml
```

只可对相同 physical scan 和相同配置使用 `--resume`。输出包含每点的 synthetic ROOT、provenance/target-z audit、baseline log 与 metrics、nominal MLP checkpoint、held-out IDs、`unknown_association_scan_points.csv` 和 `unknown_association_scan.png`。
