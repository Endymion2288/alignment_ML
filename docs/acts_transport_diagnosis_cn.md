# ACTS 输运协方差诊断（Stage B / Task B3）

Workbook 99。WB98 已经物化 `C0`、`F`、`C1`、`Q_ACTS`，但冻结 closure gate
仍失败。本任务只诊断为什么

    C1 = F Cin F^T + Q_ACTS

盖不住

    e = x_prop - x_target。

不修复 closure，不进入 Measurement Model V2，不进入 alignment，不调 Q，
不 rescale 协方差，不删除高 χ² 事件。

## 审计内容

- **B3.1** 状态与 surface 契约：Athena 原生
  `(loc1, loc2, phi, theta, q/p)` → Acts bound → 导出
  `(x, y, tx, ty, q/p)`，目标面 `PlaneSurface((0,0,z), n=ẑ)`，
  正向/反向，q/p 符号，以及数值 Jacobian。
- **B3.2** Jacobian 一致性：直接比较 dump 中的 `C0` 与 `F Cin F^T`。
  禁止从协方差反推 `F`。
- **B3.3** 残差分解：`x, y, tx, ty, q/p`，冻结 construction/validation
  划分与冻结 `|q/p|` bin。
- **B3.4** 最高 1% / 0.1% χ² 溯源。全部保留，不 clip、不拒绝。

## 失败分类

| 情况 | 含义 | 下一步 |
| --- | --- | --- |
| A | 状态、surface 或 Jacobian 契约有问题 | 修 transport contract |
| B | 材料 / 散射 / process-noise 模型不匹配 | ACTS material diagnosis |
| C | 少数高 χ² 事件主导均值 | 分析 uncertainty model，不调协方差 |
| D | transport 契约正确但 closure 仍失败 | 重新评估 Measurement Model V2 **输入**模型 |

即使判为 D，B3 也不进入 V2。

正式 run `sbb3_acts_transport_diagnosis_20260906T180100Z_574d6429`
为 **Case C**（高 χ² 尾巴主导），次因为 Case B（`Q=C1−C0` 常非 PSD）
与 Case D（主体 x 方向 overcover）。状态/surface 契约成立；
典型事件 `C0 ≈ F Cin F^T`。

## 冻结继承

WB87–WB98 结论只读。WB98 保持 `acts_q_materialized_closure_failed`。
Dump 仍在 `outputs/acts_transport_dump_v1/dumps/`，本任务不覆盖。
