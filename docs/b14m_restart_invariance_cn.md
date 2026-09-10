# Profile 似然 Smoke 重开与 Restart 不变性（Stage B / Task B14M-R）

Workbook 129。冻结入口是 **WB128**。Jacobian contract 已经闭合。
本任务只重开 login-scale B14M smoke。

```
WB127 mean function PASS
        ↓
WB128 独立导数 reference PASS
      Jacobian contract PASS
      B14M 重开
        ↓
WB129 profile likelihood smoke
      + 四个预注册 restart
      + objective / prediction invariance
        ↓
只有 PASS 才有 restart_invariance_established = true
        ↓
WB130 1989 preflight
```

统计模型未改：

```
chi2(theta) = Σ r_i(theta)^T R_i^{-1} r_i(theta)
chi2_prof(alpha) = min_nu chi2(alpha, nu)
R_i = (0.08 mm)^2 / 12
```

无 prior，无 ridge，无 truth q/p，fit 不读 target measurement。
q/p 仍是 explicit nuisance。数值 Jacobian 是继承的 WB123 repaired
tangent + WB127 certified mean，不是新模型。

Restart tolerance 从冻结的 WB114/WB115 恢复，不是看完本跑再设的。

## 正式结果

Official run：`sbb14mr_restart_invariance_20260909T183700Z_f1c280a7`

```
decision = profile_optimizer_restart_sensitive
verdict  = FAIL
b14m_smoke_passed = false
restart_invariance_established = false
restart_invariance_authorized = false
jacobian_contract_established = true
full_sample_authorized = false
```

```
config SHA      4ce8379b4cab239b5124cafea4f17ae8805a2b676eb7b506c597ae6e38d68890
decision SHA    67aa191def9863d39dec6643bf117516aa95039d6a90e0e38a09ac5b161059ea
helper SHA      686d980fbf72a872af74b4cb9b4a0bdbbd608e615a787dcc55b4986027c6ec4d
```

Dump：`outputs/leave_target_out_dump_v1/b14m_reopen_smoke/`

control `100043/0` 和 `100043/1` 的 profiled objective 与预测对
restart 不变；nuisance 不唯一（Case C）。`100043/37` 收敛且 χ² 很大
（~1e4–1e5），样本中保留；T1/T2 restart-sensitive，T3 invariant。
`100048/86` T1 是 profile 层的 multimodality / globalization 失败，
不是 Jacobian 失败；T2/T3 invariant。target leakage = 0。未提交 1989。

下一本必须留在 B14M-R。不能开始 WB130，也不能提交 1989。
