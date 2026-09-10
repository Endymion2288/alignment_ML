# Workbook 123: 3ST \(q/p\) 极端尾巴与 split/source 不稳定性根因审计（Yasu-S2G）

日期：2026-09-09
状态：**完成** —— 尾巴 / split / source 根因已记录。未进入 S3、alignment、weak-mode、B14M/B15 或 MM V2。未把 WB119 翻成 PASS。未 trim / winsorize 正式均值。matched / reweight / influence 只作诊断。本阶段**没有提交**新 reconstruction dump，但最小 diagnostic-export contract **已授权**后续 \(\le 20\) 个 focus event。

**最终判定：`mixed/inconclusive`（C 与 E 同时成立；train / validation 重建偏置不可重复）**

- `decision = three_st_qp_tail_source_recorded`
- `contract_verdict = PASS`
- `diagnosis_verdict = RECORDED`
- `tail_from_sparse_catastrophic_fits`：**rejected**
- `broad_reconstruction_shift`：**unresolved**
- `source_specific_reconstruction_response`：**supported**
- `population/composition_explains_split`：**rejected**
- `refit_provenance_suspected`：**supported**
- `repeatable_reconstruction_bias = false`
- `three_st_qp_trusted_observable = false`
- `residual_conditional_authorized = false`
- `new_reconstruction_dump_authorized = true`（仅最小 diagnostic-export；本阶段未 dump）
- `official_mean_trimmed = false`
- 恒等式残差 \(\sim 10^{-19}\)–\(10^{-20}\)

## 对两个未决问题的回答

**1. 为什么 validation 出现强 reconstruction bias（\(\mu_{\rm bias}=-6.48\times 10^{-6}/\mathrm{MeV}\)）？**

不是主体 clean 三站样本上的持续偏移。Validation 的 58% clean 径迹 \(\mu_{\rm bias}=-3.6\times 10^{-8}\approx 0\)，只占 \(\sum\Delta\) 的 0.3%。偏置几乎全部来自 dirty / 不完整 topology：缺 \(\ge 2\) 站（2.5% 径迹，占 \(\sum\Delta\) 的 **103.9%**）、`n_mot<12`、以及 `front()` 落在 S2 而非 S1（3.9% 径迹，占 \(\sum\Delta\) 的 **75.4%**，\(\mu_{\rm bias}=-1.25\times 10^{-4}\)）。

也不是 top 0.1% \(\lvert\Delta\rvert\) 那 981 条：它们对 signed \(\sum\Delta\) 的份额是 **-38.6%**（把均值往回拉）。去掉它们之后剩余 \(\mu_{\rm bias}\) 反而更负（\(-8.99\times 10^{-6}\)，原值的 139%）。\(\lvert z\rvert\ge 10\)（3.8%）占 \(\sum\Delta\) 的 76.7%、占 \(\sum z^2\) 的 \(\approx 100\%\)；去掉后剩余 \(\mu_{\rm bias}=-1.57\times 10^{-6}\)（24%），过不了“灾难点 \(<20\%\)”线，也过不了“主体仍保留 \(60\%\)”线。

同电荷、同 \((p,t_x,t_y)\)、同 topology 的 matched gap 仍是 raw split gap 的 **74.5%**。split 差异不是 population/composition，而是 **不可转移的 reconstruction response**。同电荷 source 之间 \(\mu_{\rm bias}\) 可反号。

**2. 为什么极少数 track 能主导总体均值和 pull RMS？**

Pull RMS：是的，极端 \(z\) 尾巴几乎独占 \(\sum z^2\)。Validation 上 \(\lvert z\rvert\ge 10\) 占 \(\sum z^2\) 的 99.9997%；\(\lvert z\rvert\ge 100\)（3226 条）仍占 99.998%。这与 WB121 的协方差 / 信息极限失败是同一条尾巴。

正式均值：合并样本 top 0.1% \(\lvert\Delta\rvert\)（3399 条）仍占 \(\sum(q/p)_{\rm fit}\) 的 **52.6%**（与 WB122 一致）。但 validation 的 **signed** \(\mu_{\rm bias}\) 并不由这 0.1% 拥有，而是由约 4% 的 S2-front / 缺站 / 低 `n_mot` dirty 径迹拥有。中位数 \(\tilde\Delta\approx -2.7\times 10^{-9}\)，正式均值是中位数的数千倍——**正式均值未改成中位数。**

## 裁决

官方机制是 **`mixed/inconclusive`**，因为预注册的五条里同时成立两条：source 响应不可转移，以及 front-\(z\) 代理足以怀疑 refit / `front()` provenance。train / validation 的 \(\mu_{\rm bias}\) 反号（\(+0.12\) vs \(-6.48\times 10^{-6}\)），**不存在可重复的 reconstruction bias**。

现有字段不能区分 `CKF-only fallback` 与 `KalmanFitter refit success`，但非 `front_near_s1` 已占 validation \(\sum\Delta\) 的 75.3%。因此授权后续 \(\le 20\) 个 focus event 的最小 diagnostic dump，字段仅限 `kf_refit_succeeded`、pre/post-refit \(q/p\) 与 covariance、front surface type/\(z\)、track-state provenance。禁止改 fitter / seed / hit / geometry / covariance scale，也禁止用 truth 改 fit。**本阶段没有提交该 dump。**

`three_st_qp_trusted_observable` 与 residual conditional 继续冻结。即使 validation bias 已局部到 dirty / S2-front，也不能说它造成了 IFT residual、\(R_y\) 或 \(d_x\)。

## 问题

在不把单条 \(q/p\) 当可信动量、不删尾巴、不重定义正式均值的前提下，解释 validation reconstruction bias 的来源，以及极端尾巴对均值和 pull RMS 的贡献。

## 输入

- 只读复用 WB119 dump；分母与 WB119 **逐项相同**：3 398 774 primary，243 758 sign-flip
- construction → `train`，validation → `validation`

## 方法（看 batch 前冻结）

正式量：\(\mu_{\rm fit},\mu_{\rm truth},\mu_{\rm bias}\)，禁止 trim / winsorize / 改用中位数。  
贡献式分解：每一类对 \(\sum\Delta\)、\(\sum q_{\rm fit}\)、\(\sum z^2\)、sign-flip、coverage failure 的占比。  
Influence：“移除某类后剩余均值”只诊断，不是新裁决。  
Matched：相同电荷、相同 \((p,t_x,t_y)\)、相同 topology；不是新校准样本。  
机制见上文五条；多于一条成立则为 mixed。

## 机器结果（smoke）

`yasu_s2g_three_st_qp_tail_source_smoke_20260909T075818Z_45694506`  
39 primary，2 sign-flip。\(n<200\) → INCONCLUSIVE。`new_dump=false`。

## 机器结果（batch）

`yasu_s2g_three_st_qp_tail_source_batch_20260909T081928Z_89657e0f`

单位：\(1/\mathrm{MeV}\)。正式均值未修剪。

| 样本 | \(n\) | \(\pi_+\) | \(\mu_{\rm fit}\) | \(\mu_{\rm truth}\) | \(\mu_{\rm bias}\) | \(\tilde\Delta\) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| construction | 2 418 513 | 0.400 | \(-1.83\times 10^{-6}\) | \(-1.95\times 10^{-6}\) | \(+0.12\times 10^{-6}\) | — |
| validation | 980 261 | 0.500 | \(-6.87\times 10^{-6}\) | \(-0.40\times 10^{-6}\) | \(-6.48\times 10^{-6}\) | \(-2.7\times 10^{-9}\) |
| pooled | 3 398 774 | 0.429 | \(-3.28\times 10^{-6}\) | \(-1.50\times 10^{-6}\) | \(-1.78\times 10^{-6}\) | \(-0.52\times 10^{-6}\) |

Validation 电荷：\(\mu^+_{\rm bias}=-14.9\times 10^{-6}\)，\(\mu^-_{\rm bias}=+1.97\times 10^{-6}\)。\(\Delta\) 符号几乎 50/50（`frac_delta_neg=0.503`）。

### 贡献（validation，未删除）

| 类 | \(n\) | 占 \(\sum\Delta\) | 占 \(\sum z^2\) | 移除后剩余 \(\mu_{\rm bias}\)（诊断） |
| --- | ---: | ---: | ---: | ---: |
| clean | 567 980 | 0.3% | \(\sim 0\) | \(-15.4\times 10^{-6}\) |
| dirty | 412 281 | **99.7%** | **100%** | \(-0.036\times 10^{-6}\) |
| \(\lvert z\rvert\ge 10\) | 36 967 | 76.7% | 100% | \(-1.57\times 10^{-6}\)（24%） |
| \(\lvert z\rvert\ge 100\) | 3 226 | 74.2% | 100% | \(-1.68\times 10^{-6}\)（26%） |
| top 0.1% \(\lvert\Delta\rvert\) | 981 | **-38.6%** | 65.3% | \(-8.99\times 10^{-6}\)（139%） |
| top 0.01% \(\lvert\Delta\rvert\) | 99 | 24.4% | 7.1% | \(-4.90\times 10^{-6}\)（76%） |
| `front_near_s2` | 38 376 | **75.4%** | 39.3% | — |
| unusual front（非 S1） | 45 465 | **75.3%** | 39.3% | \(-1.67\times 10^{-6}\)（26%） |
| missing \(\ge 2\) stations | 24 384 | **103.9%** | **99.7%** | \(+0.26\times 10^{-6}\) |
| `n_mot` 6–11 | 44 858 | 58.7% | 80.5% | \(-2.80\times 10^{-6}\) |
| `n_mot`<6 | 7 025 | 41.4% | 19.4% | \(-3.82\times 10^{-6}\) |
| complete S1+S2+S3 | 890 597 | 0.2% | 0.1% | \(-70.7\times 10^{-6}\) |
| \(\sigma\in[10^{-6},10^{-5}]\) | 174 847 | 73.0% | 99.9% | \(-2.13\times 10^{-6}\) |
| \(\mathrm{cond}\in[20,30]\) | 277 888 | 94.2% | 99.0% | \(-0.53\times 10^{-6}\) |
| non-SPD | 807 | -2.8% | 0.17% | \(-6.66\times 10^{-6}\) |

Construction 的 `front_near_s2` 同样极端，但 \(\mu_{\rm bias}=+8.42\times 10^{-5}\)，与 validation 的 \(-1.25\times 10^{-4}\) **反号**。同一 front-\(z\) 代理在两个 split 上给出相反 reconstruction response。

### Matched diagnostic（不是校准样本）

| | |
| --- | ---: |
| raw \(\mu_{\rm bias}\) gap (val−con) | \(-6.60\times 10^{-6}\) |
| matched-cell gap（232 格，\(\min n\ge 50\)） | \(-4.92\times 10^{-6}\)（raw 的 74.5%） |
| 电荷-only 平均 gap | \(-5.56\times 10^{-6}\) |
| matched 格反号 | 54 / 232 |

\(0.745>0.70\) → composition **rejected**。

### Source（pooled，同电荷）

- \(\mu^+\) 四文件：\(-0.94\) 到 \(-16.3\times 10^{-6}\)，相对极差 0.94
- \(\mu^-\) 五文件：\(+5.05\) 到 \(-4.16\times 10^{-6}\)，反号，相对极差 1.83

### 最小 diagnostic-export contract

- `authorized = true`
- 理由：unusual front-\(z\) 占 validation \(\sum\Delta\) 的 75.3%；现有字段无法打 `kf_refit_succeeded`
- \(\le 20\) 个 focus event；复用 WB119 六条 identity，可加极端 S2-front / 缺站行
- 字段：`kf_refit_succeeded`、pre/post-refit \(q/p\)、pre/post-refit covariance、front surface type/\(z\)、track-state provenance
- 禁止改 fitter、seed、hit selection、geometry、covariance scale；禁止用 truth 改 fit；禁止翻 WB119 或开 S3

## Provenance

- 代码 SHA（`git_head_sha`）：`55cf982302a3c62c57b74f368d5e3ba7723fd33a`
- config SHA：`a757c6ac6576d4989522259c4b45b9e79ca713666f397ecae73986c97051f3b0`
- smoke contract SHA：`43704b73addf7bd8e86517adea92cca41cd3ac44d49a295ba55d6f53c77125be`
- batch contract SHA：`446989727517e8c8815757c8a3077c880b1498fa0e657779cf7af876b7feda5a`
- WB119 batch：`b326532b381d7240ba0615bf3e4825f5a20c9b0b242a3c40cc52f25e97da3a43`
- WB122 batch：`27f61c7f36aa9ad135fedcd2c7b22403abccb27f2a5d4b303d84a31c26f5cb4f`
- geometry `4e965f3631dd4ff6e04b141ac36efc49603a1dfc0625de5ea7558dd929486b15`
- field `60432de5864cfba361f91f47a694dc111bbb06dcec44434e9682548680460e9c`
- conditions `d30733216d6eb392dbdbcf5a252fed01ff56a045ade59731ca8419febce7a63d`
- Calypso `40892527e9c65409afd2378a2abfc25ddbddac03`；Athena 24.0.41；ACTS 32.0.2
- 测试：`tests/test_three_st_qp_tail_source_audit.py` 9 passed
- HTCondor：无新作业；复用 cluster 1112283 的 WB119 dump

## 禁止声称

- 三站 \(q/p\) 已校准或已是可信动量观测量
- S2 / WB119 被本阶段翻成 PASS
- 已授权 \(E[r_{\rm IFT}\mid q/p]\)、alignment、weak-mode、B14M/B15、MM V2
- \(\mu_{\rm bias}\neq 0\) 导致了 IFT residual、\(R_y\) 或 \(d_x\)
- 可用 trim / 中位数 / clean / 重加权重定义正式均值
- matched / reweighted 结果是新的 calibration population
- influence remaining mean 是新的物理裁决
- 已证明这些 S2-front 径迹就是 CKF-only fallback（只是代理成立，标签尚未导出）
- 高 \(p\) 信息极限或 5×5 胖尾已被本阶段修复

## 下一步

S3 保持关闭。本阶段已经回答：validation 的强 bias **不是** clean 三站主体上的可重复偏移，而是 **source-specific、不可转移、集中在 S2-front / 缺站 / 低 `n_mot` 的 dirty 子集**；pull RMS 则由 \(\lvert z\rvert\ge 10\) 独占。已授权、但本阶段尚未提交的下一步是 \(\le 20\) 个 focus event 的最小 diagnostic dump，用来核对 `kf_refit_succeeded` 是否就是这条 front-\(z\) 代理。该 dump 已由 WB124（Yasu-S2H）完成：S2-front 不是 fallback 的一一标签。不授权改 fitter，也不授权 residual conditional。
