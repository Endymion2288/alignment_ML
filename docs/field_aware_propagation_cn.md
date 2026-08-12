# 含磁场 Muon Propagation 与 q/p 审计

## 范围

当前四站控制样本为 MC24 100 GeV FASERnu muon：

`/eos/experiment/faser/sim/mc24/particle_gun/100012/rec/dev/FaserMC-MC24_PG_muon_fasernu_100GeV-100012-00000-00004-xAOD.root`

它包含 IFT 与 1、2、3 号 station。MC22 100 GeV electron 继续作为仅含 1--3 站的
exporter、covariance 与 loader smoke test，不用于建立本四站控制结论。

V4 导出通过 `FaserActsExtrapolationTool` 使用磁场图 `FaserFieldTable_v2.root`；该图
由配置好的 FASER ReleaseData 目录提供。审计状态约定为存储 reference z 平面上的 global
`[x, y, tx, ty]`；residual 定义为 target state 减 propagated source state。

## Local q/p 审计

五个 event 的 canonical 文件含 24 条 local tracklet，IFT/1/2/3 station 计数为
5/6/8/5。24/24 的 reconstructed q/p 均有限，并与由 reconstructed momentum
magnitude 计算的 charge/p 自洽。这只验证序列化，因为两者都来自同一个
`TrackParameters` 对象。

独立 MC truth 比较才是决定性检查：22 条 tracklet 有有限 truth q/p，其中只有 14/22
符号一致（`0.6364`），local q/p uncertainty 相对 truth q/p magnitude 的中位数为
`223.61`。因此当前 reconstructed local q/p **不能作为 V1 local measurement**。它可
保留为 global-track latent parameter 或 MC 审计量，但不能作为部署时的 q/p 输入，也不能
用于 uncertainty calibration。

## Propagation 控制量

exporter 对每个 truth-matched 的有序 station pair 写出三个明确标记的 variant：

| Mode | Source state | 用途 |
| --- | --- | --- |
| 0 | Reconstructed local state 与 reconstructed q/p | 当前最接近可部署的控制，但其 q/p covariance 尚未标定。 |
| 1 | Reconstructed position/direction 加 MC truth q/p，并去除 q/p seed covariance | MC-only 的磁场传播控制，供历史 coordinate-level alignment 使用，不用于 physical refit closure。 |
| 2 | MC truth position、direction 与 q/p | 独立 truth-state transport diagnostic，不是 local-tracklet fit。 |

在 truth-match fraction 至少为 `0.99` 时，mode 0 和 mode 1 各保留 30 条带 covariance
的 pair。mode-0 的 4D chi-square 中位数为 `60.05`，而 straight-line 为 `432.03`。
这是有用的 field-aware 相对比较，但宽大的 q/p covariance 使其 chi-square 尺度不能作为
已标定的物理 gate。MC truth-q/p 控制（mode 1）的中位数为 `427.55`；x residual RMS 为
`43.97 mm`，y residual RMS 为 `1.513 mm`，`[x, y, tx, ty]` 的 pull RMS 为
`[1.035, 0.553, 1.002, 0.639]`。

mode 2 生成 36 条成功的 truth-state record，只适合 transport sanity check：
`[x, y, tx, ty]` 的 RMS residual 分别为 `1.859 mm`、`1.349 mm`、`9.03e-4` 和
`8.56e-4`。它不会把持久化的 local tracklet 变成 truth-calibrated measurement。

这些结果验证 exporter 与 field-aware propagation 数据通路，但不证明 covariance 已标定、
nominal detector alignment 已收敛，或存在可用于 real data 的 q/p 输入。

## 复现

保留的 V4 artifact 位于
`outputs/mc24_muon_fasernu_5events_fieldaware_v4_truthcontrol/`：

```bash
cd /eos/home-x/xcheng/FASER
source alignment_ML/scripts/setup_environment.sh ml
cd alignment_ML

python scripts/audit_tracklet_qoverp.py \
  outputs/mc24_muon_fasernu_5events_fieldaware_v4_truthcontrol/tracklets.root \
  --output outputs/mc24_muon_fasernu_5events_fieldaware_v4_truthcontrol/qoverp_audit.json

python scripts/evaluate_field_propagation.py \
  --tracklets outputs/mc24_muon_fasernu_5events_fieldaware_v4_truthcontrol/tracklets.root \
  --propagations outputs/mc24_muon_fasernu_5events_fieldaware_v4_truthcontrol/propagations.root \
  --q-over-p-mode 0 \
  --min-truth-match-fraction 0.99 \
  --output-dir outputs/mc24_muon_fasernu_5events_fieldaware_v4_truthcontrol/field_propagation_mode0_rerun
```

`--q-over-p-mode 1` 只能用于 MC truth-q/p control。评估器会保存 JSON metrics、
per-station-pair CSV、residual/pull/chi-square 图和 accepted pair tensor。

## 必需的后续验证

在 association 研究中使用 chi-square gate 前，必须在更大的独立 muon 样本上审计
propagated covariance 与 Jacobian convention。四站 electron MC 仍适合检验 particle
type 与 magnetic-field 泛化，但不再阻塞受控 station-x/y alignment 工作。
