# 2026-08-11：2D-FLUKA nominal 的 Pairwise MLP 全局指派基线

## 目的

在启动任何 Transformer 前，验证 pairwise MLP 加显式一对一全局指派能否在
严格 source-file 隔离、真实物理 refit/Acts 输入下达到 nominal 主运行点。
预先定义的主约束为 inclusive purity >= 0.95 且 inclusive fake rate <= 0.05，
在此范围内最大化 association efficiency。

## 数据与物理链

- MC24 particle-gun 2D-FLUKA：正 muon `100116` 与负 muon `100117`。
- 8 个独立 xAOD source 文件：训练 4、验证 2、测试 2；原始 source 文件是
  唯一 split 单位，materialise 和加载均验证 provenance。
- 每个输入都来自
  `SCT_ClusterContainer -> SegmentFitRefit -> SegmentsRefit -> NtupleDumper
  -> FaserActsExtrapolationTool(mode 0)`。
- local segment `q/p` 已审计为固定 propagation seed，不作为 association
  measurement，V1 使用 `q_over_p_mode=0`。
- nominal pilot 的 synthetic overlay 每个 event 取 3 条不同 single-muon
  物理轨迹，训练/验证/测试分别为 480/240/240 个 synthetic event。

真实 tracklet 数分别为 5178/2577/2600；random easy fake 为 443/223/242；
missing true tracklet 为 582/303/280。此 pilot 的 field-aware hard negative
为 0，故不能把该 fake 分布称为完整物理背景模型。后续 curriculum bank 必须
继续审计并加入 hard-negative 机制。

## 先前失败与修正

只使用 propagation residual/pull 的 MLP 在 validation 上 AUC=0.9171、
AP=0.8302；即使使用 Sinkhorn 全局指派，最佳结果只有 18.98% efficiency
且 purity=95.36%。检查发现 S0->S2 与 S0->S3 存在大量 high-score
different-truth pair。

因此增加持久化、重新拟合得到的 source/target `(x,y,tx,ty)` 状态，形成仍然
低容量的 `state_augmented_v2` MLP。没有加入 truth ID、role 标签、Transformer
或把 q/p 当测量。修正后 ungated validation candidate：

- AUC=0.96307
- AP=0.89888
- calibrated ECE=0.05165
- candidate coverage=3461/3461=1.0（每个 station pair 都为 1.0）

## 指派比较与验证选择

比较 greedy、Hungarian、带 source/target 私有 dustbin 的 Hungarian，以及
Sinkhorn 后 Hungarian。dustbin Hungarian 等价于具有 unmatched penalty 的
unit-capacity min-cost-flow；所有方法均不读取 MC truth。

验证集独立完成 temperature calibration，并对所有 chi2 gate
`25/250/1000/5000/ungated` 扫描 score threshold、unmatched penalty 和
Sinkhorn temperature。test 在运行点锁定后才打开。

## 封存测试结果

| 方法 | gate | Efficiency | Inclusive purity | Fake rate | Missing unmatched recall |
| --- | --- | ---: | ---: | ---: | ---: |
| Greedy | 5000 | 0.82514 | 0.98540 | 0.01460 | 0.99347 |
| Hungarian | ungated | 0.82940 | 0.98781 | 0.01219 | 0.99217 |
| Dustbin Hungarian | ungated | 0.93261 | 0.97590 | 0.02410 | 0.96997 |
| Sinkhorn -> Hungarian | ungated | 0.93261 | 0.97590 | 0.02410 | 0.96997 |

selected ungated test graph 的 candidate recall=0.98493。测试集少量物理
exporter/refit 不可用导致其未达到验证集的 1.0，已显式保留在结果中。

结论：nominal primary operating point 已达成。可以进行真实 conditions payload
的 `0, 0.1, 1, 5, 10, 50 mm` curriculum scan；但目前没有任何理由启动
Transformer。下一道门槛是：在严格物理错位数据上，查看 candidate recall
是否仍足够，以及 pairwise MLP + global assignment 的 efficiency 是否明显退化。

## Hard-negative 机制 smoke test

为避免当前 fake 只包含随机远离组合，curriculum 配置恢复
`hard_negative_mean_per_target_station=0.50`。在完整错位 bank 尚未完成前，先对
封存 nominal physical bank 做了仅 training/validation 的 materialise 与审计：

- field-aware hard negative 实际写入：训练 731、验证 360；两者 selection failure 均为 0。
- 每个 hard target 都由同一真实 payload 的 mode-0 Acts source prediction 选择，
  source-anchor chi2 被显式限制在 `[1,1000]`。
- 按 station 的 hard-anchor chi2 中位数约为 232--384；因此它们是明确的几何近邻
  困难负样本，而不是对 state 或 residual 施加人工扰动。
- audit 没有打开 test split（`test_opened=false`）。

这次 smoke 只验证 synthetic background 生成机制，不用于替代后续 `0.1--50 mm`
真实 payload 的训练/评估数据。

## Hard-negative 定义修正与 validation 结果

绝对 `[1,1000]` chi2 band 的第一版虽然生成成功（训练/验证 731/360 个，
无选择失败），但 audit 显示约 40--60% hard fake 比同 source 的 retained truth
endpoint 更接近。这会制造系统性 truth-dominating 负样本，不适合作为普通几何近邻
背景。

因此 materializer 已改为仍用同 payload mode-0 Acts，但额外要求
`1.10 <= hard_chi2 / truth_chi2 <= 10.0`。修正后的训练/验证 smoke：

- hard negative 写入 715/304，选择失败 24/48；失败保留为可审计的空位。
- 每个可比较 hard/truth pair 的 `hard_chi2 > truth_chi2`，fraction=0 的
  truth-dominating hard fake。
- shared `state_augmented_v2` MLP 的 hard-stress validation 仍未满足
  primary：质量约束下最佳 efficiency 约 0.45（gate 250，candidate recall 约 0.52）；
  ungated 时 efficiency 约 0.83，但 inclusive fake rate 约 0.091。
- station-pair MLP ensemble 也在 validation-only 下测试，结果更差；两个 smoke
  均未打开 test。

因此现在不能把 hard-stress 的失败归咎于还未尝试的 Transformer，也不能悄悄把
hard fake 删除。下一步先增加真实训练 source，并在完整 physical misalignment
corpus 上重新评估 shared MLP 和全局指派。

## Source 扩展

curriculum 配置已从 8 个 source 扩展到 10 个：新增
`100116-00040--00049` 和 `100117-00040--00049` 到 train。validation/test
仍各保留 2 个独立原始 xAOD 文件。新增两个 source 同样需要各自完整运行
`0/0.1/1/5/10/50 mm` 的 SQLite payload、segment refit 与 Acts export。

## 产物

- `outputs/mc24_muon_2dfluka_nominal_pilot_state_v2_full_gate_validation_v2`
- `outputs/mc24_muon_2dfluka_nominal_pilot_state_v2_full_gate_test_v1`
- `docs/global_assignment_mlp_baseline.md`
- `docs/global_assignment_mlp_baseline_cn.md`
