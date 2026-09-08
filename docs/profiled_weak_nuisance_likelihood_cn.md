# Profiled 弱 nuisance 测量似然（Stage B / Task B14M）

Workbook 114。WB113 已经证明没有合法的 target-independent prior，
因此本任务不再要求一个 seed-independent 的 5×5 LTO Cin。它从幸存
LTO hits 构造 measurement likelihood，并对弱参数做 **profile**。

Profiling 是主合同：

```
chi2_prof(alpha) = min_nu chi2(alpha, nu)
```

**不**默认做 marginalization。没有合法 prior 或独立定义的 measure
时，`∫ L dν` 不能被叫做合法 marginal likelihood。

这**不是** Measurement Model V2。**不**进入 B15 或 V4 C/D。

## Likelihood

不要使用 WB109 Cin 或 WB107 Cin。

```
r_i(theta) = m_i - h_i(theta)
chi2(theta) = sum_i r_i^T R_i^{-1} r_i
```

`theta` 是 source-plane native chart
`(loc0, loc1, phi, theta, q/p)`。`h_i` 是场感知 ACTS transport 加上
生产 loc0 strip 投影。`R_i = (0.08 mm)² / 12`，与 CKF / LTO helper
的 pitch/√12 相同。

Target 站测量不进入 fit。

## 划分

WB112 分类保持冻结。这**不是**截断状态或把 nuisance 钉到 seed 的
许可证。

```
alpha = (loc0, theta)   ~ (x, ty)
nu    = (loc1, phi, q/p) ~ (y, tx, q/p)
```

耦合保留。`H_nn` 亏秩是物理诊断，不是加 ridge 的理由。

## 数值合同

B14M 单独预注册 `pinv_relative = 1e-8`。这不是历史 alignment 的
`rank_tolerance=0.01`。在受控线性问题上，joint NLS、显式 nuisance
最小化和 Schur complement 必须一致。

## 分类

- A：`profiled_measurement_likelihood_validated` — 排除、优化器、
  可观测预测对 seed 基本不变、null space 显式处理、uncertainty
  语义成立，且冻结 construction/validation calibration 通过。然后
  再决定 Gaussian-state V4 是否还是正确抽象。
- B：nuisance 本身测不准，但 target prediction observable 可识别
  → 下一步是 observable-space 模型，不是 5D Cin。
- C：prediction 本身依赖 seed 或没有有限 uncertainty → 停止当前
  LTO estimator。
- D：数值不稳定 → 先修实现，不用 ridge/prior 掩盖。
- E：混合

## 官方标记

正式 B14M run 落在 Case D：

```
decision = profile_likelihood_numerically_unstable
profiling_executed = true
marginalization_executed = false
prior_introduced = false
lto_cin_contract_established = false
b15_authorized = false
do_not_force_5d_lto_covariance = true
```

synthetic joint / profile / Schur 合同通过。ACTS measurement-only
helper 还不稳定：`100043/37` 和 `100048/86` 在场感知 `h_i(theta)`
上失败，`loc1+1mm` restart 回不到同一 χ²。这是欠收敛，不是挑选
最好 seed 或继续修 5D Cin 的许可证。
