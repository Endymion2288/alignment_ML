# Profile Globalization 修复与 Stationarity 再合同（任务 B14M-T）

Workbook 131。上一本冻结是 **WB130**。本任务**只修数值 globalization /
termination**，**不提交 1989**。

```
WB128 Jacobian PASS
        ↓
WB129 restart invariance FAIL          （历史结果；不再单独当 validity certificate）
        ↓
WB130 无真正 multibasin / 无 hysteresis
      高 χ² endpoint 非 stationary
      flat_direction 终止不合法
        ↓
WB131 显式 nuisance profiling
      + range-space trust-region
      + 有 stationarity 意识的终止
      重新认证全部 12 identities × 4 restarts
```

正式统计 objective 始终是原始 `chi2`。trust-region / LM `λ` 只是算法
globalization multiplier，不是 prior，不是 ridge，不是 information。

Trust-region 常数已在任何 37/86 新结果出现前写入
`configs/b14m_profile_globalization_repair_v1.yaml`，不得按 WB131 结果回调。

即使 Case A：

```
restart_invariance_established = true
restart_invariance_authorized = true
full_sample_authorized = false
```

下一本才是 WB132 full-sample preflight。
