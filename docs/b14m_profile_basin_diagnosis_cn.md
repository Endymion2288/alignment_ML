# Profile Basin 拓扑与 Flat-Direction 终止诊断（任务 B14M-S）

Workbook 130。上一本冻结是 **WB129**。本任务**不修 optimizer**，**不提交 1989**。

正式判定：

```
decision = flat_direction_termination_not_stationary
run      = sbb14ms_profile_basin_20260909T214204Z_45f7bda5
```

判断顺序：先确认高 χ² endpoint 是不是 stationary point。只有两个
endpoint 都真正 stationary，才有资格称为两个 likelihood basin。

三个失败 identity 的高 χ² endpoint 在 supported / profile 子空间仍有
明确下降。固定 α 后 ν seed 收敛到同一 χ²；A→B / B→A continuation 重合。
不是已证明的 profile multibasin。

已通过 identity 不得重新解释为失败。nuisance 可以不唯一，只要
profiled objective 和 physical prediction 保持不变。

```
restart_invariance_established = false
full_sample_authorized = false
```
