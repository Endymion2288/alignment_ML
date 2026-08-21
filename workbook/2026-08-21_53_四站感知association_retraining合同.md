# 2026-08-21 (53) 四站感知 association retraining：合同与 15 维相对 curriculum

## 任务边界

条目 52 把冻结历史 V2 在四站同时错位上的失败定为
`association_domain_shift`：raw truth-chain recall=1，score 阈值仍留住完整
chain，但 complete-track efficiency 相对本 bank nominal 下降约 0.19–0.20，
集中在 2→3 / S3。本条目正式进入 matched retraining pilot。

本阶段**不**重设计 Transformer，也**不**打开 15 维未知关联 WLS。问题只有
一个：四站同时错位造成的失败，究竟只是历史 V2 的训练分布失配，还是现有
表示缺少对 relative four-station geometry 的正确归纳偏置。

继续冻结：物理链、mode-0、unit-capacity route solver、
`physical_edge_deduplicated`、已准入 15 维相对子空间、预注册 `ΔT_ij`
capture、密封 test。禁止用条目 52 两个 held-out overlay 调 threshold、
unmatched penalty 或 calibration。若重训后 2→3 仍系统性掉而 candidate
recall=1，先诊断对 absolute station state / station embedding / route
utility 的依赖，不允许靠调 unmatched penalty 把效率硬拉回来。

## 预注册（看生产结果之前）

闸门写在
`configs/physical_four_station_association_retraining_gates.yaml`。

### 源

| split | source | 理由 |
| --- | --- | --- |
| train | `mc24_100043_00200_00299`（μ−） | 当前四站 identifiability pilot 已完成 |
| train | `mc24_100044_00300_00399`（μ+） | 同上 |
| validation | `mc24_100047_00050_00099`（μ−） | expanded contract 内、从未进入该 pilot |
| validation | `mc24_100048_00050_00099`（μ+） | 同上 |

不使用 `100047_00000_00049` / `100048_00000_00049`（identifiability YAML
里留给后续 closure bank）。不用 `100047_00150_00199`（历史 outlier）。
密封 100116/100117 仍关闭。

### 采样，不是 20 维事后去 gauge

每个 source 7 个真实物理点（1 nominal + 3 个相对族 × 2）：

1. 在 S0-identity **采样图**中写 15 维相对构型（S0 只是坐标选择，不是物理真值）。
2. 随机左乘一个公共 SE(3) 作为明确的 gauge-control twin，使模型不能依赖
   「哪一站接近 nominal」。
3. 相对幅度锁在条目 50 已验证局部区：平移 ≤0.50 mm、转动 ≤5 mrad。
4. 固定纳入 `hard_s3_ry`：`s1_dx=0.30 mm`、`s2_dy=−0.20 mm`、
   `s3_ry=5 mrad`，针对 2→3 难例。这是训练分布里的 matched hard case，
   **不是**把条目 52 overlay 拿来选 OP。
5. 另两族在同一盒子内随机抽取。公共左乘幅度 ≤0.30 mm / 3 mrad，且不扩大
   `ΔT_ij`。

禁止：先抽 20 个绝对参数再投影到 15 维。物理 payload 仍是
`unconstrained_full`（四站都写）。不生成 48 个 FD probe。

### Overlay 与模型

生产链仍是 `/Tracker/Align → SegmentFitRefit → SegmentsRefit →
NtupleDumper → Acts(mode 0)`。图用与冻结 V2 **完全相同**的 overlay
recipe：`configs/physical_alignment_iteration_trainval.yaml`
（`events_per_payload: 120`，3 轨/事件，hard/easy fake，
`overlay_seed_scope: alignment_iteration_shared_across_payloads`）。

Matched comparison：

- 对照：同一套新四站 overlay + 冻结历史 V2（阈值 0.001，unmatched
  penalty −1.0）。
- 唯一新模型：同一 overlay + retrained V2。架构、feature、route solver、
  30 epoch 预算保持历史 V2；只允许 train 训练、validation early stopping /
  calibration / OP 选择。GPU-only。test 永不打开。

### Association 闸（source-disjoint validation）

相对该 bank **自己的** nominal：

- raw complete truth-chain recall ≥ 0.90
- complete-track efficiency 下降 ≤ 0.10
- purity 下降 ≤ 0.05、fake 增加 ≤ 0.05（条目 51 冻结界限）
- 尤其检查 2→3 与 S3 participation 是否恢复

Gauge-invariance audit：同一 `ΔT_ij`、不同公共 SE(3) 左乘的 twin，
efficiency / purity / fake 的绝对差分别不超过 0.05，2→3 efficiency 差
不超过 0.08。

只有上述闸在 validation 上通过，才冻结 retrained V2，并接到已经
truth-selected 通过的 15 维相对 WLS，做未知关联 `ΔT_ij` closure。

## 本条目完成的工作

- 采样与 compiler：`alignment/four_station.py` 的 15 维 S0 图 +
  `left_multiply_all`；`scripts/prepare_four_station_relative_curriculum.py`
- 源与闸门 YAML；V2 matched 训练配置（条件轴改为
  `four_station_relative_l2`，幅度只分 0/1，总 epoch 仍为 8+22=30）
- `_build_plan` 允许该 curriculum 跳过 FD probe，其它扫描仍强制 FD
- 单测：禁止 20 维事后去 gauge、7 点无 FD、越出局部区拒绝、非该
  curriculum 不得跳过 FD

Athena / Condor 生产已提交：cluster **1000434**，4 个 source job（flavour
`tomorrow`，6000 MB）。完成后才做 overlay、冻结 V2 对照、GPU 重训与闸门
评估。15 维未知关联 WLS 仍关闭。
