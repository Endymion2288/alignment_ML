# Pairwise MLP + 全局指派基线

日期：2026-08-10

## 本阶段目标

在不启动 Transformer 的前提下，验证 pairwise MLP 经过显式全局一对一指派后，是否能够达到 nominal primary operating point。只有 nominal 达标且较大错位明显退化，才有理由比较多站上下文模型。

## 已完成实现

1. 新增 `baselines/global_assignment.py`。
   - `greedy`：现有分数贪婪指派参考；
   - `hungarian`：完整 station-pair score matrix 上的常规全局指派；
   - `dustbin_hungarian`：source/target 专属 dummy node，等价于允许 unmatched 的 unit-capacity min-cost-flow；
   - `sinkhorn_hungarian`：dustbin matrix 的 log-domain Sinkhorn 后 Hungarian rounding。
2. 新增 unmatched 评估。
   - `missing_truth_unmatched_recall`：一侧真实 tracklet 因 missing counterpart 应该进入 dustbin 时，正确未匹配的比例；
   - `fake_unmatched_recall`：synthetic fake endpoint 被正确未匹配的比例；
   - fake 与真实 missing 分母分开保存，不与 association efficiency 混合。
3. 新增 `scripts/run_global_assignment_mlp_baseline.py`。
   - MLP 使用 CUDA；
   - scalar calibration temperature、score threshold、unmatched penalty、Sinkhorn temperature 只由 validation 决定；
   - primary selection scope 为 nominal-only validation；
   - test 只对固定运行点输出 nominal 至 50 mm 的 candidate/global/unmatched/station-pair 指标。
4. synthetic overlay 改进。
   - 每个 event 使用 3 条不同的完整 single-muon source track，超过可用数量时直接失败，禁止复制同一物理径迹；
   - ROOT 中增加 evaluation-only 的 `synthetic_role`、hard-anchor station 与 hard-anchor chi2；这些字段不进入 MLP feature；
   - `random_easy_fake` 必须对每个可形成有序跨站 candidate 的 true anchor 的最小真实 Acts chi2 不小于 1000；
   - `field_aware_hard_negative` 必须由同一 displaced payload 的真实 mode-0 Acts prediction 选出，满足 `1 <= chi2 <= 1000`；
   - `scripts/audit_synthetic_background.py` 输出角色数量、candidate chi2 分类、hard-anchor chi2 与 missing endpoint 审计。
5. 新增 `scripts/assemble_physical_curriculum_manifest.py`。
   - 仅组合完整物理 refit asset；
   - 检查原始 xAOD 文件、split、source event UID、物理链和 mode-0 一致性；
   - 不改写坐标、conditions 或 residual。

## 新数据划分

最终计划使用 8 个独立 MC24 xAOD source file：

| split | source 数 | source range |
| --- | ---: | --- |
| train | 4 | 00000-00004, 00005-00009, 00010-00014, 00025-00029 |
| validation | 2 | 00015-00019, 00035-00039 |
| test | 2 | 00020-00024, 00050-00054 |

前 5 个 source 复用此前已经完整运行并验证过的同一 physical payload bank；新增的 3 个 source 逐点重新运行：

```text
SCT_ClusterContainer -> SegmentFitRefit -> SegmentsRefit -> NtupleDumper
-> FaserActsExtrapolationTool(mode 0)
```

因此新增物理计算为 train 11 点、validation 6 点、test 6 点，共 23 个 conditions payload/refit/Acts 点；不是 residual 或坐标替代物。

## 物理 smoke

新增 train source `mc24_00025_00029` 的 nominal point 已成功完成。Calypso 日志记录 `Execution succeeded`，单点 Athena execution 时间约 367 秒。

在该真实 nominal refit/Acts 输出上做了 80-event synthetic smoke：

| 项目 | 数量 |
| --- | ---: |
| 可用独立完整 source track | 4 |
| 写入 true tracklet | 861 |
| missing true tracklet | 99 |
| field-aware hard fake | 82 |
| random-easy fake | 26 |
| hard fake 选择失败 | 36 |
| easy fake 选择失败 | 56 |

hard 上限最初设为 100 时产额偏低；在保持真实 mode-0 Acts 判别而不改变坐标的条件下，改为 `chi2 <= 1000` 后取得上述产额。random-easy 的最小 chi2 仍为 1000，因此两类构造在定义上分离。选择失败也会被保存，避免把配置期望的 fake 均值误报为实际样本分布。

## 软件验证

`source scripts/setup_environment.sh ml && pytest -q`：29 项通过。

CLI smoke 使用已有 physical synthetic corpus、1 epoch 和放宽的测试型 operating target，确认 CUDA MLP、四种 assignment、CSV、checkpoint、图和 station-pair 输出可完整生成。该 smoke 配置不用于任何物理结论。

## 当前状态与下一步

当前 `mc24_00025_00029` 的 11 个真实 payload 正在继续运行。完成后依次生成新增 validation `mc24_00035_00039` 与 test `mc24_00050_00054`，装配 8-source manifest，materialize/audit synthetic corpus，并运行 GPU curriculum MLP 和 validation-only global-assignment scan。

Transformer 未启动。
