# 受控 Synthetic Multi-Track Overlay

## 目的

该数据集在 propagation 与 alignment closure 之后提供第一套受控 association 输入。它将完整的
single-muon source track 叠加到一个 synthetic event，再加入 missing tracklet 与
fake/background tracklet。它现在支持文档中定义的 MLP pair-classifier baseline，但仍不训练
Transformer。

## 构造方式

- source track 必须在每个指定 station 都有一条 truth-known segment，且 truth-match
  fraction 不低于 0.99。
- 若同一 truth particle 在同一 station 有多条 segment，依次选择 truth-match
  fraction 最高、local chi-square 最低、原始 tracklet ID 最小者。这是确定性的 source
  selection rule。
- source truth ID 会按 synthetic event 与 track slot 重新命名，因此 overlay 后的 truth
  association label 仍唯一。
- 每条 true segment 以 `missing_tracklet_probability` 独立被移除。
- 每个 station 抽取 Poisson 数量的 fake row；其 local state 和 covariance 从原始同站
  pool 采样，但 truth ID 设为 `-1`。
- 所有 row 随机打乱，并重新赋予 event-local `tracklet_id`。
- 每条 row 保留 `origin_run_id`、`origin_event_id` 和 `origin_tracklet_id`，使同一物理
  payload 的 mode-0 Acts source prediction 能够映射到 synthetic event，而无需
  coordinate-level propagation 近似。

## 当前 Artifact

`outputs/mc24_muon_fasernu_5events_fieldaware_v4_truthcontrol/synthetic_multitrack_100events.root`
来自 5 条可用 MC24 source track：每个 synthetic event 叠加 4 条 track、true tracklet
移除率 10%、每站 fake mean 为 0.5。seed `20260808` 下输出 100 个 event、1,642 条
row：已知 truth 1,449 条、被移除的 true row 151 条、fake/unknown 193 条。四个 station
均存在；可选 q/p 字段仅为 schema compatibility 保留，不是已验证的 local q/p measurement。

这里只有 5 条物理 source trajectory 被重复使用，适合 pipeline test 和受控 association
ablation，不适合无偏的最终训练/评估划分。MLP scan 的 held-out synthetic-event split 只能控制
overlay event leakage，并不是独立的 source-event split。

## 复现

```bash
cd /eos/home-x/xcheng/FASER
source alignment_ML/scripts/setup_environment.sh ml
cd alignment_ML

python scripts/make_synthetic_multitrack.py \
  --config configs/synthetic_multitrack_muon.yaml \
  --input outputs/mc24_muon_fasernu_5events_fieldaware_v4_truthcontrol/tracklets.root \
  --output outputs/mc24_muon_fasernu_5events_fieldaware_v4_truthcontrol/synthetic_multitrack_100events.root
```

生成器会写入 ROOT metadata 和同目录的
`synthetic_multitrack_100events.resolved_config.yaml`。只需改变 YAML 参数，即可构造
missing-tracklet、fake-rate 与 multiplicity scan。
