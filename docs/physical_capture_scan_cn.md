# 物理 Capture-Range 扫描

本扫描用真实重建得到的探测器数据测量固定 truth 的 station-level alignment 求解器的
capture range，不使用 coordinate-level 代理，也不在此阶段引入 learned association
模型。

## 重建链

每一个扫描点都会新建独立的 `/Tracker/Align` SQLite/POOL payload，并重新执行：

```text
持久化 SCT_ClusterContainer
  -> SegmentFitRefit
  -> SegmentsRefit
  -> NtupleDumper tracklet 与 propagation 导出
  -> FaserActsExtrapolationTool（q/p mode 0）
  -> 固定 truth 的 WLS alignment closure
```

各点共用原始 xAOD 输入，但绝不复用 segment 坐标或 propagation residual。mode 0 是
强制选择，因为 `SegmentFitAlg` 中 local segment 的 `q/p` 是固定 seed，而不是可靠的
测量量。

单独重新 refit 的零 payload control 只作为构造 residual increment 时共同的物理 WLS
baseline。它不会被平移或以其他方式变换来代替任何非零 injection；每一条非零 observed
residual 都由该点自身的 Calypso refit 导出。

## 扫描定义

`configs/physical_refit_capture_scan_muon.yaml` 定义 MC24 100 GeV FASERnu muon
输入、五个 events、station 0 reference、station 1--3 `dx/dy` 空间中的三组确定方向，
以及逐步增大的偏移幅度。零偏移只跑一次；每一个非零幅度跑全部三组方向，因此 capture
fraction 来自方向样本，而不是把单个 injection 错当作统计分数。

当前成功条件为：

1. 物理 refit 与 propagation 正常完成；
2. WLS gate 后至少保留一条固定 truth pair；
3. 三个 movable stations 对应的 normal-matrix rank 为六；
4. movable stations 的最大恢复误差范数不超过 `0.01 mm`。

## 当前 Control 结果

完成的 MC24 五 event control 位于
`outputs/mc24_muon_fasernu_physical_capture_scan_v1/`。它包含 25 个物理点：一个零偏移点，
以及每个非零幅度的三组方向。每个点均完成自己的 refit/export/propagation chain。

在配置的 `0.01 mm` 判据下，零偏移点和全部三个 `0.1 mm` trials 均 capture；它们的最大
station-norm recovery error 分别为 `0`、`0.00493`、`0.00330` 和 `0.00132 mm`。全部三个
`1 mm` trials 虽保留 rank six 和 truth pairs，却超过严格 tolerance（`0.0487`、`0.0334` 和
`0.0133 mm`）。更大幅度均不 capture；部分 `500` 和 `1000 mm` 方向在 WLS gate 后还会失去
normal-matrix rank。因此，在这一确切 control 配置和 tolerance 下，capture range 至多为
`0.1 mm`。

这不是探测器整体性能结论。它依赖五个 source events、选定的三组 displacement direction、
固定 truth association、mode-0 seed q/p、WLS gate 和 refit 配置。

## 复现命令

从干净的 Python 环境启动 orchestration。脚本会为 Calypso 与 ML 子步骤分别设置环境，
避免 LCG Python/动态库串扰。

```bash
cd /eos/home-x/xcheng/FASER
env -i HOME="$HOME" USER="$USER" LOGNAME="$USER" \
  PATH=/usr/local/bin:/usr/bin:/bin SHELL=/bin/bash \
  /usr/bin/python3 alignment_ML/scripts/run_physical_refit_capture_scan.py \
  --config alignment_ML/configs/physical_refit_capture_scan_muon.yaml \
  --output-dir alignment_ML/outputs/mc24_muon_fasernu_physical_capture_scan_v1
```

相同 scan 被中断后可使用 `--resume` 续跑。脚本会拒绝非空目录的非续跑执行，并在续跑前
比较已保存的 scan plan。

## 输出

每个 `points/<point>/` 目录保存本点的 conditions payload、Athena log、refit ROOT、
canonical tracklets、mode-0 propagation records、内容审计及 closure 产物。scan 根目录
保存：

- `scan_plan.json` 与 `resolved_config.yaml`
- `capture_scan_summary.json` 与 `capture_scan_points.csv`
- `capture_scan_station_pair_diagnostics.csv`：每个 station pair 的 field-aware
  `rx`、`ry`、`rtx`、`rty`、pull 和 chi-square 统计
- `capture_scan.png`：injected magnitude 对应的 capture fraction 与 median recovery
  error

每个点的 `closure.json` 还保留可用/active truth-pair 数、WLS normal-matrix rank 与
condition number、注入和恢复的 offset、各 station 的绝对恢复误差、covariance response
诊断，以及原始 field-aware residual/pull 汇总。
