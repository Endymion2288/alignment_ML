# 规范流水线

当前科学主线的导航。历史 V1/MLP/V2/V3、rank rescue 与 Frozen-V2 association
只作控制，不是未来 estimator 路径。

权威审查与任务表：

- [PROJECT_MASTER_AUDIT.md](PROJECT_MASTER_AUDIT.md) — 冻结于 2026-09-05/06，审查远端 master `0c8a6ca34c6c8b4f505dc153dd0ca8cc7949ce56`
- [CODE_ROADMAP.md](CODE_ROADMAP.md) — T00–T41 执行计划

不得把后续 workbook 倒填进该审查。Workbook 75–87 由 T00 记为 post-audit lineage。

## 三条不得混用的入口

| 入口 | 消费什么 | 可以主张 | 不得主张 |
| --- | --- | --- | --- |
| **inference** | 冻结 V2 / MLP / route packing | association 分数、route 占用、冻结策略后的 residual/DQ | 可部署 alignment correction |
| **paired-response analysis** | 同事件 FD residual、WLS、rank | 在同事件上恢复已知 geometry 差 | 绝对 data alignment、探测器结构性零空间 |
| **real-data monitoring** | 官方 geometry + 冻结 association | residual/DQ 告警 | 写 geometry、机械 correction |

T00 只按这三类登记 E01–E09。

## 当前科学问题

```text
真实测量 → 含场全局 track likelihood
         → 轨迹 nuisance 剖面化
         → alignment 信息 / 协方差 / coverage
```

问题是真实测量里有多少 alignment 信息，而不是更大网络能否拟合 paired residual。

已停止：V4/GNN/Transformer 调参、新 parameterization rescue、继续找
tracker-only 5DoF rank 子空间、改 rank 门、打开 sealed test、改写冻结负结果。

## 每条新主张必须带的 provenance

code SHA、dirty-tree SHA、resolved-config SHA、input GUID/SHA、output SHA、
schema、seed、source/condition split、UTC、argv、job ID、closure 类型、
负结果/替代状态。

仓外 Calypso/Athena/ACTS 针脚只作 provenance。若不能证明当前二进制等于历史
production，标 `historical_runtime_unverified`，不补造 commit。

## 冻结权限

`geometry_write_allowed=false`，`real_data_alignment_authorized=false`，
`measurement_model_validated=false`，`held_out_accessed=false`。
