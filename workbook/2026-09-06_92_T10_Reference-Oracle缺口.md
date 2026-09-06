# Workbook 92: T10 Reference-Oracle 缺口

日期：2026-09-06
状态：**完成** —— 只重算冻结 iteration-01 数组，无新 refit，无 sealed test。

**最终判定：`FAIL`**

旧 WLS 只作 diagnostic，不得进入 data correction。Zero-target 不是正确
likelihood。下一步仍是 T11/T12，不是 V2 重训。

- `old_wls_authorized_for_data_correction = false`
- `zero_target_is_not_correct_likelihood = true`
- 门未改：`0.1 mm / 1 mrad`

## 结果

Validation 467 边复现审查 E02：paired remaining 过门，zero remaining
`(-0.2476 mm, -0.0495 mm, -3.438 mrad)` 失败。Train 623 边同样：paired
过门，zero remaining `(0.937 mm, -1.268 mm, 1.437 mrad)` 失败。

Paired 增量与保存值逐分量 `1e-8` 一致。毒化 target residual 不改变
`delta_zero`。

Run ID：`t10_reference_oracle_20260906T131146Z_ddf00f66`

Config SHA：`0e4d4d8d346687ae65f1a1a0cf1bed9fd729804ce103f5d435ae9a589e2087bb`
