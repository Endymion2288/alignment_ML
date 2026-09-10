# 三站 \(q/p\) 极端尾巴与 split/source 不稳定性审计（Yasu-S2G）

Workbook 123。WB119 已冻结：官方 WithoutIFT \(q/p\) 不是可信
reconstructed momentum。WB122 表明 construction 是 population 均值效应，
validation 是 reconstruction bias，合并正式均值被尾巴主导。本阶段不
修复这些结论，也不打开第 3 阶段。

只问两个尚未解决的问题：

1. 为什么 validation 出现强 reconstruction bias
   （\(\mu_{\rm bias}=-6.48\times 10^{-6}/\mathrm{MeV}\)）？
2. 为什么极少数 track 能主导总体 \(q/p\) 均值和 pull RMS？

## 正式 run

Smoke：`yasu_s2g_three_st_qp_tail_source_smoke_20260909T075818Z_45694506`

```
decision = three_st_qp_tail_source_contract_established
diagnosis_verdict = INCONCLUSIVE
three_st_qp_trusted_observable = false
residual_conditional_authorized = false
official_mean_trimmed = false
new_reconstruction_dump_authorized = false
```

Batch：`yasu_s2g_three_st_qp_tail_source_batch_20260909T081928Z_89657e0f`
（同一 3 398 774 条主样本、243 758 条 sign-flip）

```
decision = three_st_qp_tail_source_recorded
tail_from_sparse_catastrophic_fits     = rejected
broad_reconstruction_shift             = unresolved
source_specific_reconstruction_response = supported
population/composition_explains_split  = rejected
refit_provenance_suspected             = supported
official_mechanism = mixed/inconclusive
repeatable_reconstruction_bias = false
three_st_qp_trusted_observable = false
residual_conditional_authorized = false
official_mean_trimmed = false
new_reconstruction_dump_authorized = true   # 仅合同；本阶段未 dump
```

## 冻结定义

未修剪的主样本上：

\[
\mu_{\rm fit}=E[(q/p)_{\rm fit}],\quad
\mu_{\rm truth}=E[(q/p)_{\rm truth}],\quad
\mu_{\rm bias}=E[\Delta(q/p)].
\]

每一类报告对 \(\sum\Delta\)、\(\sum(q/p)_{\rm fit}\)、\(\sum z^2\)、
sign-flip 和 coverage failure（\(\lvert z\rvert\ge 1\)）的贡献。正式
均值不删径迹。

看结果前冻结的 tail 定义：

- \(\lvert z\rvert\ge 10\) 与 \(\lvert z\rvert\ge 100\)
- top \(0.1\%\) 与 \(0.01\%\) \(\lvert\Delta(q/p)\rvert\)

“移除某类后剩余均值”只是 **influence diagnostic**。相同电荷、相同
\((p,t_x,t_y)\)、相同 topology 的 matched 比较 **不是** 新的校准样本。

## 机制（看 batch 前冻结）

A/B 主要看 validation，C 看合并 source，D 看 construction/validation
匹配，E 看 validation 的 front-\(z\)。官方机制取唯一成立的标签，否则
`mixed/inconclusive`。

- `tail_from_sparse_catastrophic_fits`：去掉 \(\lvert z\rvert\ge 10\)
  **且** 去掉 top 0.1% \(\lvert\Delta\rvert\) 后，剩余 \(\lvert\mu_{\rm bias}\rvert\)
  落到原值的 20% 以下（或低于 \(10^{-7}/\mathrm{MeV}\)）
- `broad_reconstruction_shift`：两次移除后剩余同号且至少保留 60%
- `source_specific_reconstruction_response`：同电荷 source 的
  \(\mu_{\rm bias}\) 相对极差 \(\ge 0.50\) 或反号
- `population/composition_explains_split`：matched-cell gap
  \(\le 30\%\) raw split gap
- `refit_provenance_suspected`：非 `front_near_s1` 的 front-\(z\) 代理
  占 validation \(\sum\Delta\)、\(\sum z^2\) 或 top 0.1% tail 的
  \(\ge 50\%\)

仅当该代理可能解释 validation bias 或 tail 的主份额，**并且** 现有
字段无法区分 `CKF-only fallback` 与 `KalmanFitter refit success` 时，
才授权后续 \(\le 20\) 个 focus event 的最小 diagnostic dump。字段限定
为 `kf_refit_succeeded`、pre/post-refit \(q/p\) 与 covariance、front
surface type/\(z\)、track-state provenance。禁止改 fitter、seed、hit
selection、geometry、covariance scale，也禁止用 truth 改 fit。

## 批量结论

Validation \(\mu_{\rm bias}=-6.48\times 10^{-6}/\mathrm{MeV}\) **不是**
clean 三站主体上的持续偏移。58% clean 的 \(\mu_{\rm bias}=-3.6\times 10^{-8}\approx 0\)，
只占 \(\sum\Delta\) 的 0.3%。dirty 子集占 \(\sum\Delta\) 的 99.7%，以及
几乎全部 \(\sum z^2\)。

集中点是 topology / provenance：

- 缺 \(\ge 2\) 站（2.5% 径迹）占 \(\sum\Delta\) 的 **103.9%**、占
  \(\sum z^2\) 的 99.7%
- `front()` 在 S2（3.9%）占 \(\sum\Delta\) 的 **75.4%**，
  \(\mu_{\rm bias}=-1.25\times 10^{-4}\)
- `n_mot<12` 几乎占满 signed 总和

top 0.1% \(\lvert\Delta\rvert\)（981 条）**并不**拥有 validation 的
signed 均值：其对 \(\sum\Delta\) 的份额是 \(-38.6\%\)。去掉它会让
\(\mu_{\rm bias}\) 更负。去掉 \(\lvert z\rvert\ge 10\)（3.8%）后剩余
原值的 24%——低于 60% 的 broad 线、高于 20% 的 catastrophic 线——故 A
被拒绝、B 未决。

Construction 里同一 S2-front 类的符号相反（\(+8.4\times 10^{-5}\)）。
同电荷 source 互相不同意，\(\mu^-\) 文件甚至反号。matched 格子（同电荷、
\(p\)、\(t_x\)、\(t_y\)、topology）仍保留 raw split gap 的 74.5%，
所以 composition 解释不了 split。train / validation 的 \(\mu_{\rm bias}\)
反号：reconstruction bias **不可重复**。

Pull RMS 与正式均值不是同一句话。Validation 上 \(\lvert z\rvert\ge 10\)
占 \(\sum z^2\) 的 99.9997%；\(\lvert z\rvert\ge 100\)（3226 条）仍占
99.998%。合并样本 top 0.1% \(\lvert\Delta\rvert\) 仍占
\(\sum(q/p)_{\rm fit}\) 的 52.6%，与 WB122 一致。这些径迹没有被删。

C 与 E 同时成立，官方机制因此是 `mixed/inconclusive`。unusual-front
代理占 validation \(\sum\Delta\) 的 75.3%，现有 dump 没有
`kf_refit_succeeded` 标签，故最小 diagnostic-export contract **已授权**。
本阶段没有提交该 dump。WB124 已执行该 dump：S2-front 是 Hole-99
fallback 与合法 first-MOT 的混合物，不是一一对应的 fallback 标签。

## 第 3 阶段状态

`three_st_qp_trusted_observable = false`。
`residual_conditional_authorized = false`。
WB119 仍为 `not_established`。

## 本阶段不得声称

validation 上 \(\mu_{\rm bias}\neq 0\) **不授权**
\(E[r_{\mathrm{IFT}}\mid q/p]\)、IFT alignment 或 \(R_y/d_x\) 弱模。
influence remaining mean 与 matched / 重加权数字不是新的正式均值，也
不是新的 calibration population。S2-front 代理还不是
“这些径迹就是 CKF-only fallback”的证明。
