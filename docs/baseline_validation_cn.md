# 基线验证：2026-08-08

## 已完成的验证

以下链路已经在 lxplus 上实际执行，而不只是通过单元测试：

1. Calypso 读取 xAOD，并从 ghost-busted `Segments` collection 写出按 event
   组织的 `Tracklet_*` branch。
2. `scripts.convert_ntuple_tracklets` 检查 vector 长度后写入 canonical flat
   `tracklets` tree。
3. `scripts.audit_tracklets` 检查实际 station 覆盖、协方差正定性、hit/fit
   摘要和 MC 标签完整性。
4. `scripts.run_chi2_baseline` 以 straight-line propagation 和 covariance
   gate 构造候选，再做 greedy one-to-one assignment。

baseline 的 CSV 现已保存 source/target truth barcode 和 `truth_relation`。
端点 truth 未知或重复的 prediction 被标成 `unscorable`，不会再被当成 fake
纳入 purity/fake-rate 分母。

## MC 样本与实际导出内容

| 样本 | 输入 | 使用的 geometry | events | canonical tracklet | 实际 station ID |
| --- | --- | --- | ---: | ---: | --- |
| MC22 100 GeV 电子 particle gun | `/eos/experiment/faser/sim/mc22/particle_gun/100022/rec/s0012-r0019/FaserMC-MC22_PG_elec_100GeV-100022-00000-00004-s0012-r0019-xAOD.root` | `TI12MC03` | 10 | 28 | `1, 2, 3` |
| MC24 100 GeV FASERnu muon particle gun 开发样本 | `/eos/experiment/faser/sim/mc24/particle_gun/100012/rec/dev/FaserMC-MC24_PG_muon_fasernu_100GeV-100012-00000-00004-xAOD.root` | 自动配置的 `FASERNU-04` | 5 | 24 | `0, 1, 2, 3` |

MC22 电子样本的 28 行协方差均为正定，且每条 tracklet 都有已知 truth 标签。
它是三站样本，不含 station `0`。MC24 FASERnu 样本的 24 行协方差也全为正定；
每个 event 都观察到 station `0`，运行时 geometry 加载了 `/Tracker/Align` 的
`TRACKER-ALIGN-02`。

station 标签由 cluster identifier 直接给出。文中 z 值仅作审计交叉检查，
不是 station 标签的来源。

## Straight-Line Baseline 结果

`possible` 是在两个 station 中各恰好出现一次的 truth ID 数。`raw/scored/
unscorable` 依次是全部 greedy assignment、两端都有非重复已知 truth 标签的
可评分子集、以及其余不可评分子集。`1e6` 的宽 gate 只用于这些很小样本的诊断，
不是生产分析的 gate 选择。

| 样本 | station 对 / gate | possible / correct | raw / scored / unscorable | efficiency | purity | fake rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MC22 电子 | `1 -> 2`, 50 | 9 / 3 | 3 / 3 / 0 | 0.333 | 1.000 | 0.000 |
| MC22 电子 | `1 -> 2`, `1e6` | 9 / 9 | 10 / 9 / 1 | 1.000 | 1.000 | 0.000 |
| MC22 电子 | `2 -> 3`, `1e6` | 4 / 4 | 5 / 4 / 1 | 1.000 | 1.000 | 0.000 |
| MC24 FASERnu muon | `0 -> 1`, `1e6` | 4 / 4 | 5 / 4 / 1 | 1.000 | 1.000 | 0.000 |
| MC24 FASERnu muon | `1 -> 2`, `1e6` | 3 / 3 | 5 / 3 / 2 | 1.000 | 1.000 | 0.000 |
| MC24 FASERnu muon | `2 -> 3`, `1e6` | 4 / 4 | 5 / 4 / 1 | 1.000 | 1.000 | 0.000 |

在 MC24 FASERnu 样本上，默认 50 的 gate 对 `0 -> 1` 不产生任何 candidate。
临时 straight-line propagator 下，相同 truth 的 `0 -> 1` candidate chi2 约为
118 到 1,617。这说明该跨度跨磁场时 straight line 不足以作为物理 gate，不是
station 缺失或 covariance 无效的证据。

这些小规模 particle-gun 样本没有可用于最终性能结论的 false-pair population：
MC22 的 `1 -> 2` 宽 gate 共 13 个 candidate，全部拥有同一个已知 truth barcode。
因此表格验证的是数据契约和 baseline 的执行，不是最终 association 性能结论。

## 负向检查

`/eos/experiment/faser/sim/mc24/fluka/210010/rec/s0013-r0019/FaserMC-MC24_Fluka_2023_exp001_z448p6_d31p4_zsim3p99-210010-00000-00007-s0013-xAOD.root`
的前 10 个 event 中，`SegmentFit` 与 ghost-busted `Segments` 都为空。exporter
写出了合法的空 canonical tree，而没有伪造记录。该 FLUKA slice 在未经过含
local segment 的选择前，不可用于 baseline 验证。

## 可重复命令

```bash
cd /eos/home-x/xcheng/FASER
alignment_ML/scripts/export_mc_tracklets.sh \
  --input /eos/experiment/faser/sim/mc24/particle_gun/100012/rec/dev/FaserMC-MC24_PG_muon_fasernu_100GeV-100012-00000-00004-xAOD.root \
  --output-dir alignment_ML/outputs/mc24_muon_fasernu_check \
  --nevents 5 \
  --source-station 0 --target-station 1 --chi2-gate 1e6

source alignment_ML/scripts/setup_environment.sh ml
cd alignment_ML
python -m scripts.audit_tracklets \
  outputs/mc22_elec_100gev_10events/tracklets.root \
  --require-mc-labels --source-station 1 --target-station 2 \
  --chi2-gate 1e6
```

第一条命令会创建新目录，且拒绝覆盖已有 artifact。它保存
`content_audit.json`、`metrics.json`、`matches.csv`、解析后的配置和
candidate-chi2 图。

## Transformer 前仍缺少的关键输入

1. 该电子研究需要一个已确认含 station `0,1,2,3` local segment 的 MC24
   electron xAOD，或新产生同等样本。对当前 MC24 particle-gun 目录的文件名检索
   没有找到包含 `elec` 或 `electron` 的 xAOD 名称，但这不能证明不存在适用
   campaign。
2. 仍需版本化的数值 geometry sidecar：各 station nominal transform、
   geometry/conditions tag、reference station，以及每个 episode 的注入
   `dx_mm`、`dy_mm`。运行时 conditions tag 已知，但不从 tracklet z 推断数值
   transform。
3. 增强输出有 `TrackletHit_cluster_identifier`，但它不是 raw RDO ID。raw hit
   ID 在其 persistence 语义确认前仍未实现。
4. 在 candidate gate 的阈值可作物理解释前，需要 field-aware 的一阶传播或经过
   验证的 Calypso/Acts propagation payload。当前 straight-line 实现仍只是刻意
   保持简单的 control baseline。
