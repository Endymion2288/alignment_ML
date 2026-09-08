# Profile Likelihood 数值稳定化（Stage B / Task B14N）

Workbook 115。WB114 已经在数学上、以及在合成线性问题上建立了
measurement-level profile likelihood。同一 likelihood 在真实 ACTS
几何和磁场下**数值不稳定**。本任务不改这个 likelihood。它只问：
同一个目标函数能否被稳定、可复现、与初值无关地求值和 profile。

```
theta = (loc0, loc1, phi, theta, q/p)
alpha = (loc0, theta)
nu    = (loc1, phi, q/p)
chi2_prof(alpha) = min_nu chi2(alpha, nu)
```

`h_i` 仍是场感知 ACTS transport 加上生产 loc0 strip 投影。
`R_i = (0.08 mm)² / 12`。Target 站 hits 不进入。不加 prior。不加
作为统计信息的 ridge。不用人为 χ² penalty 代替传播失败。不恢复
5D Cin。不进 B15。不进 Measurement Model V2。

## 允许的数值修复

- 按物理 z 顺序逐面 hop，而不是每次都从 source 独立跳到每个面
- 与 Kalman 对齐的自适应 `maxStepSize`（去掉 10 m 上限）
- 逐 measurement 传播审计
- 固定数值单位缩放 `z_i = (θ_i − θ_ref_i) / s_i`，
  `s = (1 mm, 1 mm, 1 mrad, 1 mrad, 1e-3 /GeV)`
- Gauss–Newton + backtracking line search
- 仍只用 WB114 预注册的四个 restart

物理 `χ²(θ(z))` 不被 scaling 改变。这些尺度不是 prior 协方差，
也不是从 truth、Hessian 或 seed campaign 调出来的。

## 什么不是物理诊断

传播失败不是不可识别。Hessian rank 3/5 是允许的：profile
likelihood 的 nuisance 可以有 null direction。真正要认证的是
profile 目标、target observable，以及 observable 不确定度。

## 分类

- A：`profile_numerical_contract_established` — 然后在稳定求解器上
  重新执行 B14M 物理 / 不确定度
- B：`profile_transport_contract_broken` — 先修独立 evaluator；
  “都用了 ACTS”不等于合同相同
- C：`profile_derivative_contract_broken` — 先修 `∂r/∂θ`，再解释
  optimizer
- D：`profile_optimizer_contract_broken` — evaluator 和 Jacobian
  已可信，globalization 还没有
- E：`physical_nonidentifiability_exposed` — 只有数值合同 A 类
  通过后才允许
- F：混合，或还没有 smoke

## Smoke gate

Login 节点只跑 `100043` 的 `0,1,37` 和 `100048` 的 `86`。除非
全部通过——包括 `loc1+1 mm` restart、Jacobian、nominal 传播、
37/86 不再只因 step-limit 无法求值、没有隐藏 ridge/prior、
0 leaked target measurements——否则**不**提交 1989 行全样本。

## 正式 token

Official run `sbb14n_profile_numerics_20260907T150627Z_d77ddddc`：

```
decision = profile_transport_contract_broken
smoke_gate_passed = false
full_sample_authorized = false
synthetic_profile_passed = true
target_exclusion_holds = true
b15_authorized = false
do_not_force_5d_lto_covariance = true
```

Event 0/1 的 profile 求解器和 `loc1+1 mm` restart 已经稳定。
`100043/37` 仍不能完成跨越磁铁的 hop。`100048/86` 能过磁铁，
但有一个 stereo partner 不在面上。冻结权限保持 false。不覆盖
WB109 / WB114 dumps。
