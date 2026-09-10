# 三站 \(q/p\) 总体均值 / 重建偏置分解（Yasu-S2F）

Workbook 122。WB119 已冻结：官方 WithoutIFT \(q/p\) 不是可信
reconstructed momentum。WB121 已把高 \(p\) 信息极限与协方差失准拆开。
本阶段不修复这些失败，也不打开第 3 阶段。

只问 MC 中 fitted \(q/p\) 的**总体**非零均值应如何读：物理电荷 /
动量组成、reconstruction bias，还是极少数极端尾巴。

## 正式 run

Smoke：`yasu_s2f_three_st_qp_mean_decomposition_smoke_20260908T222355Z_63018db8`

```
decision = three_st_qp_mean_decomposition_contract_established
diagnosis_verdict = INCONCLUSIVE
three_st_qp_trusted_observable = false
residual_conditional_authorized = false
official_mean_trimmed = false
```

Batch：`yasu_s2f_three_st_qp_mean_decomposition_batch_20260908T224209Z_06d45778`
（同一 3 398 774 条主样本、243 758 条 sign-flip）

```
decision = three_st_qp_mean_decomposition_recorded
construction = population_mean_dominates
validation   = reconstruction_bias_dominates
pooled       = tail_dominated_mean   （不是稳定的跨 split 结论）
official_mechanism = mixed/inconclusive
train_validation_direction_consistent = false
three_st_qp_trusted_observable = false
residual_conditional_authorized = false
official_mean_trimmed = false
```

## 冻结定义

未修剪的主样本上：

\[
\mu_{\rm fit}=E[(q/p)_{\rm fit}],\quad
\mu_{\rm truth}=E[(q/p)_{\rm truth}],\quad
\mu_{\rm bias}=E[\Delta(q/p)].
\]

恒等式 \(\mu_{\rm fit}=\mu_{\rm truth}+\mu_{\rm bias}\) 检查到
\(10^{-18}\)。正式均值不得换成中位数、修剪均值或重加权均值。

电荷平衡（\(0.5\mu_++0.5\mu_-\)）和运动学匹配（两电荷都重加权到合并
\((p,t_x,t_y)\)）只是**诊断 counterfactual**，不是新的校准样本。

按 \(\lvert\Delta\rvert\) 与 \(\lvert q/p_{\rm fit}\rvert\) 取 top
0.1%、0.01%，只报告它们对正式均值的贡献。这些径迹留在样本里。

## 机制（看 batch 前冻结）

优先：tail → population → reconstruction，否则 `mixed/inconclusive`。

- `tail_dominated_mean`：top 0.1% \(\lvert\Delta\rvert\) 占
  \(\sum(q/p)_{\rm fit}\) \(\ge 50\%\)
- `population_mean_dominates`：\(\mu_{\rm truth}\) 与 \(\mu_{\rm fit}\)
  同号且至少占其幅度的 60%；\(\lvert\mu_{\rm bias}\rvert\) 低于 50%
- `reconstruction_bias_dominates`：\(\mu_{\rm truth}\) 低于
  \(\lvert\mu_{\rm fit}\rvert\) 的 20%，且 \(\mu_{\rm bias}\) 同号并至少
  占 60%

construction 与 validation 必须同一机制且方向一致，否则官方机制为
`mixed/inconclusive`。

## 批量结论

Construction 有 60% \(\mu^-\)。truth 已有同号均值
（\(\mu_{\rm truth}=-1.95\times 10^{-6}/\mathrm{MeV}\)，
\(\mu_{\rm fit}=-1.83\times 10^{-6}\)）。电荷平衡后 \(\mu_{\rm truth}\)
到 \(\sim 0\)。

Validation 是 50/50。\(\mu_{\rm truth}\approx 0\)，
\(\mu_{\rm fit}=-6.87\times 10^{-6}\) 的 94% 来自 \(\Delta\)。
\(\mu^+\) 的 fitted 均值甚至与 truth 反号。

合并正式均值 \(-3.28\times 10^{-6}\) 是中位数的六倍。top 0.1%
\(\lvert\Delta\rvert\)（3399 条）占 \(\sum(q/p)_{\rm fit}\) 的 52.6%。
这些径迹没有被删。

两个 source-disjoint split 机制不同，官方结论因此是
`mixed/inconclusive`。\(\mu_{\rm fit}\neq 0\) 仍 **不授权** 第 3 阶段。

## 本阶段不得声称

\(\mu_{\rm fit}\neq 0\) **不授权** \(E[r_{\rm IFT}\mid q/p]\)、IFT
alignment 或 \(R_y/d_x\) 弱模。它只说明 Yasu 的“分布中心不在 0”在这份
MC 里是 population、reconstruction 还是 tail。
