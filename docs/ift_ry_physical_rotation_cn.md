# IFT R_y 真实转动研究

## 范围与隔离

本研究暂停 V3 structured-margin objective。不会读取、修改、重新校准或以其他方式
使用已封存的 V1/V2/V3 test source。rotation curriculum 仅继承现有严格按 source file
隔离的 train 和 validation source。下游三站坐标系固定：每个 payload 中 station 1、2、3
均为单位变换，IFT 为 station 0。

物理语料配置为
`configs/physical_curriculum_ift_ry_expanded_trainval.yaml`。其中定义 IFT 的
`0, +/-10, +/-25, +/-40, +/-60 mrad` 纯转动，以及四个
`|R_y|=40 mrad` 的受控 `dx/dy + R_y` 联合 trial。本研究不接受任何
coordinate-level injection。

## 已确认的 conditions 语义

Calypso 的 `TrackerAlignDBTool::stationAlignment` 对每个 station 校验六个常数，顺序为：

```text
[dx_mm, dy_mm, dz_mm, rx_rad, ry_rad, rz_rad]
```

其全局变换为：

```text
T(dx, dy, dz) * Rz(rz) * Ry(ry) * Rx(rx)
```

平移单位为 mm，转动单位为 rad。tracker conditions algorithm 先生成
`SCTAlignmentStore`；随后 `FaserActsAlignmentCondAlg` 复制该 store，并将每个已对准的
detector-element transform 缓存到 `FaserActsAlignmentStore`。ACTS detector element 从
geometry context 读取该 store，因此真实 payload 同时改变 segment refit geometry 和
mode-0 propagation 所使用的 ACTS surface transform。

## 可复现 smoke

10-event、source-isolated 的 smoke plan 位于
`configs/physical_refit_ift_ry_smoke_mc24_100043.yaml`。每个选定点均执行：

```text
/Tracker/Align SQLite/POOL
  -> SCT_ClusterContainer
  -> SegmentFitRefit
  -> SegmentsRefit
  -> NtupleDumper
  -> FaserActsExtrapolationTool, q/p mode 0
```

全部 13 个 frozen smoke point 已完成并位于
`outputs/mc24_ift_ry_physical_smoke_v1`：纯转动为
`0, +/-10, +/-25, +/-40, +/-60 mrad`，另有四个 `+/-40 mrad + dx/dy` joint trial。

在 `+10 mrad`，truth-matched IFT state 的平均变化为
`delta x = -18.5981 mm`、`delta tx = +0.0100209`；在 `-10 mrad`，对应结果为
`+18.5970 mm`、`-0.0100143`。这种近似反对称性来自真实 refit response，而不是人为施加
的坐标偏移。全部 13 点中每个相邻 truth edge 和完整
IFT -> S1 -> S2 -> S3 truth chain 都在 10 个 event 中保留；raw candidate truth-chain recall 为
`1.0`，包括 `+/-60 mrad`。`+60 mrad` 的 mean IFT state response 为
`delta x=-111.5404 mm, delta tx=+0.0602876`，`-60 mrad` 为
`delta x=+111.4999 mm, delta tx=-0.0600532`。

日志中即使在 nominal geometry 也出现 ACTS layer-overlap 和 surface-error 信息。因此审计
会把这些 log count 与精确 mode-0 candidate edge 分开记录：job-level log error 不等同于
truth-matched propagated state 或 covariance 失败。所有 completed point 均已输出 state、
residual、pull、chi2、combined covariance/logdet 和相对 nominal 的 `x/y/tx/ty` observed
slope/correlation；10-event smoke 只作为机制检查，不冻结位置依赖的统计结论。

## 工具

- `scripts/write_station_alignment_payload.py` 支持完整六分量的 `--transform` 参数，同时
  保留旧的 `--offset`。
- `scripts/run_physical_refit_capture_scan.py` 新增显式 `ift_ry_rotation` scan mode，rotation
  不会进入 coordinate surrogate。
- `scripts/audit_ift_ry_refit_response.py` 相对 nominal 审计 truth-matched tracklet state、
  covariance、residual、pull 和 chi2 response。
- `scripts/audit_physical_route_candidate_graph.py` 审计 raw mode-0 physical candidate 与完整
  truth-chain retention，并单独记录 ACTS surface response。
- `scripts/run_refit_ift_ry_closure.py` 使用独立真实 refit 的 `+R_y`、`-R_y` probe 做单参数
  central finite-difference WLS closure。该程序固定 station 1--3，对单参数 closure 拒绝
  joint translation trial。
- `scripts/run_refit_ift_ry_scan_closure.py` 对多个独立 physical response point 做 held-out
  scalar closure；它不是 coordinate/residual surrogate。
- `scripts/run_physical_refit_scan_condor.sh` 和
  `scripts/submit_physical_refit_scan_condor.py` 将任意冻结 scan 作为单个 Condor job 执行，
  保留同一 real payload/refit/Acts driver。
- `scripts/audit_ift_ry_physical_corpus.py` 对完成的 source-disjoint train/validation corpus
  做只读汇总，输出 actual mode-0 candidate/route/surface 与针对 nominal `x/y/tx/ty` 的
  position-dependence regression；它拒绝 incomplete point 和 test source。

目前的 station-0 closure 可辨识但尚未达到 1 mrad：`+/-10 mrad` central probe 对 `+/-60 mrad`
给出 `5.775/5.191 mrad` absolute error；held-out physical curve 的二次结果为
`4.233/7.425 mrad`。因此已提交 `0,+/-45,+/-50,+/-55 mrad` 的真实局部 refinement probe，
`+/-60 mrad` 仍只作为 held-out observation。同时，expanded train/validation-only 13-point
rotation bank 正在 Condor 上生成；它完成并通过 physical manifest 审计后，才可 materialize
synthetic multi-track 并在 validation 上重训 MLP/V1/V2。sealed test 不会被读取或重新校准。

局部 refinement 已完成并通过：`+50 mrad` anchor、`+45/+55 mrad` physical probe 对 held-out
`+60 mrad` 恢复 `+60.06898 mrad`（absolute error `0.06898 mrad`）；`-50 mrad` anchor、
`-55/-45 mrad` probe 对 held-out `-60 mrad` 恢复 `-60.41250 mrad`（absolute error
`0.41250 mrad`）。normal-matrix rank 均为 1、condition number 均为 1。这表明参数在真实
physical refit/Acts 链中可通过 coarse-to-local iteration 恢复；此前失败的是大范围的一次
linearization，并非 rotation 不可辨识。

expanded physical bank 已完成并通过 manifest audit：10 个 train source/994 event、8 个
validation source/796 event、234/234 point complete、无 test source。大样本 raw candidate
complete truth-chain recall 在 validation 中为 nominal `0.9493`、`-60 mrad` `0.9542`、`+60 mrad`
`0.9678`，所以 rotation 没有导致 candidate truth route 消失；`0->1` 是主要损失对。对应的
source `x/y/tx/ty` position-dependence 只读审计已经输出。

## 冻结 validation control 结果

MLP、V1、V2 已全部完成。它们共享严格 source-disjoint 的 physical train/validation 语料、
mode-0 Acts candidate 和相同的相邻站 unit-capacity route solver；历史 test source 没有被打开。
V1/V2 的参数优化只使用 train source，early stopping、calibration 和 route-control 选择只使用
validation。MLP route 后处理位于 `outputs/ift_ry_mlp_route_validation_v1`，只加载 validation event，复用
冻结的 MLP checkpoint 与 station-pair calibration，并记录
`test_events_loaded=false`、`test_artifacts_opened=false`、`calibration_refit=false`。其中
checkpoint/calibration 的 SHA-256 与声明的冻结文件一致。

预注册 primary 条件为 complete-track efficiency >= 0.70、purity >= 0.95、fake rate <= 0.05。
pooled validation 的 nominal / `|R_y|=60 mrad` 结果为：

| Control | Efficiency | Purity | Fake rate | Capture |
| --- | --- | --- | --- | --- |
| Pairwise MLP + route assignment | 0.9489 / 0.9370 | 0.9888 / 0.9878 | 0.0312 / 0.0270 | 5/5 point |
| V1 geometry-aware full context | 0.9651 / 0.9626 | 0.9819 / 0.9859 | 0.0295 / 0.0253 | 5/5 point |
| V1 无 multi-station context | 0.9382 / 0.9429 | 0.9848 / 0.9870 | 0.0210 / 0.0192 | 5/5 point |
| V2 BCE route-query | 0.9355 / 0.9381 | 0.9797 / 0.9811 | 0.0404 / 0.0368 | 5/5 point |

同一 checkpoint 的 V2 edge-only 对照不通过 primary：其 nominal fake rate 为 `0.0628`。因此
V2 route correction 可以把 false route 降到自身能通过条件的水平，但在这个 rotation bank 中并未
超过 V1 full-context 的 efficiency。Pairwise MLP 的 validation-only 选择为
`0->1=0.20`、`1->2=0.05`、`2->3=0.05`、unmatched penalty `1.0`；其冻结 calibration 后的
candidate AP 为 `0.9738`、ECE 为 `0.00520`。

所有 synthetic-overlay control 的 candidate complete-truth-chain recall 都是 `1.0`。这不能与上文
raw physical recall 用同一个分母比较：multi-track overlay 只选取 complete physical truth track，
而 raw audit 包含每一个可重拟合的 physical truth segment。纯 `+/-60 mrad` 下，MLP 的 complete-track
efficiency 范围为 `0.9366--0.9374`，V1 full-context 为 `0.9620--0.9632`，V2 为
`0.9342--0.9420`；每个符号方向均 capture。`|R_y|=40 mrad` 的四个联合 `dx/dy` trial 对每个列出的
control 也均 capture。

rotation 的响应不只是平均值平移。validation raw physical audit 在 `-60 mrad` 下给出 IFT
`delta t_x` 与 source `t_x/t_y` 的相关分别为 `-0.415/-0.414`，IFT `delta t_y` 与 source `t_y`
的相关为 `-0.627`。对 mode-0 `0->1`，`delta r_{t_x}` 与 source `y` 的相关在 `-60/+60 mrad`
分别为 `+0.576/-0.337`。这些均来自 observed refit/Acts response，不是 coordinate surrogate。

因此当前 bank 不满足启动新的 hard-route contrastive objective 的条件：physical truth candidate
仍保留，且现有 pairwise/route model 到 60 mrad 均未出现 primary-point 失败。不会生成新的 test bank，
V3 structured-margin 继续暂停。下一步应该优先构造更困难的真实 background/association 语料，或扩大
physical closure campaign，而不是加深 Transformer 或进行 test-time 调参。统一只读曲线和表位于
`outputs/ift_ry_validation_control_assessment_v2`。
