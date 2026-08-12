# 数据审计：2026-08-08

## 已确认输入

- 真实 IFT PHYS 样本：
  `/eos/experiment/faser/phys/2024_ift/dev/014975/Faser-Physics-014975-00550-00553-PHYS.root`，
  包含 73,083 events 的 `nt` tree。
- MC24 样本：
  `/eos/experiment/faser/sim/mc24/fluka/210010/phy/s0013-r0019/FaserMC-MC24_Fluka_2023_exp001_z448p6_d31p4_zsim3p99-210010-00000-00007-s0013-s0012-PHYS.root`，
  包含 200,000 events。
- 对应 MC xAOD 保留 `Segments`、`SegmentFit`、`SCT_ClusterContainer`、
  `SCT_SDO_Map`、`TruthParticles` 和 `SCT_Hits`。
- 电子 PHYS 审计样本：
  `/eos/experiment/faser/sim/mc22/particle_gun/100020/phy/r0013/FaserMC-MC22_PG_elec_logE-100020-00000-00004-s0008-r0013-PHYS.root`，
  含 2,500 个 event。前 200 个 event 中有 193 个含 local segment，global-track
  truth 中确实有 `pdg=+/-11`。它是旧的三站 PHYS 输出，不是 IFT+三站训练数据集。
- 电子 xAOD 验证样本：
  `/eos/experiment/faser/sim/mc22/particle_gun/100022/rec/s0012-r0019/FaserMC-MC22_PG_elec_100GeV-100022-00000-00004-s0012-r0019-xAOD.root`。
  它是 MC22 三站样本，运行时必须指定 `TI12MC03`。
- IFT+三站结构验证样本：
  `/eos/experiment/faser/sim/mc24/particle_gun/100012/rec/dev/FaserMC-MC24_PG_muon_fasernu_100GeV-100012-00000-00004-xAOD.root`。
  其 runtime metadata 选择 `FASERNU-04`；这是 muon control 样本，不能代替最终
  electron 训练输入。

## 已执行的导出检查

- 选定 MC24 FLUKA xAOD 的前 10 个 event 中，`SegmentFit` 和 `Segments`
  collection 都为空。因此 exporter 输出的是合法空 canonical tree；这是样本内容
  的结果，不是 exporter 失败。
- MC22 电子样本在 10 个 event 中导出 28 条 canonical tracklet，实际 station ID
  为 `1,2,3`，每行 covariance 有效，并有 segment-level MC 标签。它验证电子导出
  通路，但不覆盖 IFT。
- MC24 FASERnu muon 样本在 5 个 event 中导出 24 条 canonical tracklet，实际
  station ID 为 `0,1,2,3`，且每个 event 都包含 station `0`。它验证 IFT+三站的
  schema 通路，但不能替代电子训练样本。

详细命令、内容摘要和 chi-square baseline 结果见
[baseline 验证](baseline_validation_cn.md)。

## data0 复审

- MC22 100 GeV electron xAOD 也位于
  `/eos/experiment/faser/data0/sim/mc22/particle_gun/100022/rec/s0012-r0019/`。
  它与上文已经验证的路径是同一个 EOS inode。其 10-event canonical 导出有 28 条
  tracklet，station 为 `1,2,3`，每行 covariance 正定，truth PDG 为 `+/-11`。
  因而只能用于三站电子 smoke test。
- 在 `data0/sim/mc22` 下没有找到带 `r0022` tag 或文件名含 `IFT` 的重建 xAOD。
  这只是对当前 filename/path 的审计，不能证明其它 production 不存在；但已找到的
  MC22 particle-gun 不能被当作 IFT+后三站数据集。
- `data0/phys/2022_ift/r0022/008090` 已有一个 PHYS ntuple，共 18,363 行，其中
  18,116 行满足 `TrackSegments>0`。可见的对应 xAOD 路径
  `data0/rec/2022/r0022/008090` 有 44,639-entry `CollectionTree`，并持久化
  `SegmentFit` 与 `Segments` key。
- 对其前 100 个 xAOD event，以 `--useIFT --export-tracklets`、关闭 global-track
  与 stable-beam filter、保留默认 blinding、并使用 `--skip-ghostbusters` 重新导出后，
  得到 100 行 ntuple，但 `TrackSegments`、`GhostBustedTrackSegments` 和详细 tracklet
  都为零。这些导出行的 event ID 是 `1...100`，而已有 PHYS 文件从 `767688` 开始。
  因此该 xAOD/PHYS 配对的 provenance/content 仍未证实；在确定精确 production 输入前，
  不可用于训练或 alignment。
- 部分 MC22 FASER-FORESEE `Aee` PHYS 文件含 multi-lepton truth 内容，但已检查的
  `110052/rec` 目录没有 xAOD 输入。其 legacy PHYS 仍缺 station ID、covariance、hit
  信息和 per-tracklet truth label，若不重新 reconstruction，不能作为 V1 输入。

NtupleDumper 现在有 opt-in 的 `--skip-ghostbusters`，用于已经持久化 `Segments` 的
输入，可避免向同一个 key 再安排一个 writer。Calypso setup helper 会把工作区 CLI 放在
已安装副本之前，因此无需重编 runtime library 即可使用该 Python-only 选项。

## 现有 PHYS 字段映射

| 所需量 | 现有 PHYS 字段 | 状态 |
| --- | --- | --- |
| run/event ID | `run`、`eventID` | 有 |
| tracklet position | `TrackSegment_x/y/z` | 有 |
| slopes | 可由 `TrackSegment_px/pz`、`py/pz` 计算 | 未显式保存 |
| local chi2/ndf | `TrackSegment_Chi2`、`TrackSegment_nDoF` | 有 |
| station ID | 无 | 缺失 |
| covariance 或 errors | 无 | 缺失 |
| tracklet hit count/pattern | 无 | 缺失 |
| module/raw-hit ID | 无 | 缺失 |
| 每条 segment 的 MC truth ID | 无 | 缺失 |
| nominal station transform | 无 | PHYS 中缺失，只能从 Calypso geometry/conditions 取得 |
| injected misalignment | 无 | 必须由本研究生成并记录 |

旧版真实 PHYS 输出的 `TrueTrackSegments` 是 ghost-busted segment collection
的历史名称，不是 truth label。MC 的 `t_barcode` 属于全局 CKF track，不能
作为 local segment association label。

## 已确认的 Geometry 语义

当前 FASER geometry factory 的 `SCT_DetectorFactory.cxx` 按顺序建立
`Interface`、`StationA`、`StationB`、`StationC`，并分别赋予 identifier
`0`、`1`、`2`、`3`。导出的 station ID 将从每个 cluster identifier 获取，
不从观察到的 segment z 位置推断。已观察到 runtime conditions tag，但当前审计
尚未得到权威的数值 station-level nominal transform payload；该 transform 和
injected alignment payload 仍需作为 alignment 实验的 geometry sidecar metadata。
