# Workbook 126: 3ST measurement-level bending 批量校准与 CKF mapping 审计（Yasu-S2J）

日期：2026-09-09 / 2026-09-10
状态：**完成** —— 闸门在看批量结果前冻结；9 个 inspect-only dump 已跑完；批量审计已记录。未进入 S3、alignment、weak-mode、B14M/B15 或 MM V2。未把 WB119 翻成 PASS。未改 fitter / seed / hit / geometry / field / covariance / collection。未改 WB125 的 YZ `bending_raw`。未构造正式 \(q/p\)-like，未虚构 \(\sigma_b\)。

**最终判定：`mixed/inconclusive`**

- `decision = three_st_qp_measurement_bending_batch_recorded`
- `official_mechanism = mixed/inconclusive`
- `raw_bending_information_present`：**supported**
- `raw_bending_high_p_limit`：**rejected**
- `fit_additional_sign_failure`：**rejected**（冻结闸门；见下）
- `raw_measurement_source_dependence`：**rejected**（匹配后降级为 population-spectrum effect）
- `mapping_nontransferability`：**supported**
- `three_st_qp_trusted_observable = false`
- `residual_conditional_authorized = false`
- `official_qp_like_jacobian_authorized = false`
- `s2_flipped_to_pass = false`

官方 run：`yasu_s2j_three_st_qp_measurement_bending_batch_batch_20260910T082512Z_f61fa2a8`

## 对两个批量问题的回答

### 1. 243 758 条 fitted sign-flip 有多少在 measurement 层、有多少是 fit 新增

truth 参考条数 \(n_{\mathrm{fit,all}}=3\,398\,774\)，fitted 翻号 **正好 243 758**，与 WB119 对齐。

| 类别 | \(n\) | 占 243 758 | 含义 |
| --- | ---: | ---: | --- |
| A：raw bending 已与 truth 同号 | 113 144 | **46.4%** | reconstruction mapping 额外引入 |
| C：raw bending 自身已翻号 | 27 206 | **11.2%** | measurement 层已经失去符号 |
| 不可分辨 \(\lvert b\rvert<10^{-5}\) | 2 135 | 0.88% | 近零弯曲 |
| 无可构造三站 bending | 101 273 | **41.5%** | 冻结质心定义下无法在 measurement 层分类 |

可分类的 142 485 条中：A **79.4%**，C **19.1%**，近零 1.5%。mapping 闸门因此 **supported**。

clean 18-hit（90 252 条 fit-flip）：A 86.3%，C 12.5%。

正式条件概率（clean 18-hit）：

- \(\Pr(\mathrm{fit\ flip}\mid\mathrm{raw\ bending\ correct})=0.0404\)（78 434 / 1 941 637）
- \(\Pr(\mathrm{raw\ bending\ flip})=0.0329\)（65 994 / 2 007 651）

整体 fit 翻号率仍是 243 758 / 3 398 774 = **0.0717**（WB119 的 0.928 符号一致）。其中将近一半是“bending 对、fit 错”，约九分之一是“bending 已经错”，剩下四成没有完整三站质心、不能用本 observable 归因。

### 2. source dependence 与高 \(p\) 是否在匹配后仍成立

**Source：不成立。** 未匹配 charge-odd 相对极差 0.397（未过 0.50）。按电荷、\(p\)、\(t_x\)、\(t_y\)、topology 匹配后（165 cell，匹配分数 0.999）：construction \(6.59\times10^{-4}\)，validation \(6.34\times10^{-4}\)，相对极差 **0.038**，同号。WB125 小样本上的 source 闸门是能谱/电荷构成效应，正式标记 `raw_measurement_source_dependence=rejected`。

**高 \(p\)：raw 不按冻结闸门退化。** clean \(\ge2\,\mathrm{TeV}\) \(n=176\,492\)，符号一致 **0.923**；\(<200\,\mathrm{GeV}\) \(n=510\,347\)，0.983。下降 0.060 < 0.15。1–2 TeV 为 0.951。**不得**沿用 WB125 \(n=26\) / 100 GeV 分割，也 **不得** 说 raw bending 在高 \(p\) 同样死亡。`raw_bending_high_p_limit=rejected`。

同层 fitted \(q/p\)：\(\ge2\,\mathrm{TeV}\) 符号一致 0.853（比 raw 低 0.070，未过冻结的 0.15 差距闸门），故 `fit_additional_sign_failure` 按冻结规则为 **rejected**。数字上 raw 在 TeV 区仍明显好于 fit，但未达到预注册的 high-\(p\) extra 门槛；mapping 支持来自 fit-flip 样本的 79% class A，不是来自总体 5% 翻号率。

## 冻结定义（看 dump 之前，未改）

YZ `bending_raw` 与 WB125 相同。质心：MOT space point 站平均，否则 `localToGlobal`。匹配斜率用 S1–S3 测量弦。truth 只作 source-disjoint 校准参考。

Denominator：WB119 全部 truth-matched primary。clean：`n_mot=18` 且站 \(\{1,2,3\}\) 且无 IFT leak。dirty 保留。

\(p\) 边与 WB119 相同；高 \(p\) 限制定义为 **`p_ge_2000gev`**。source 匹配键：`(charge, p_bin, tx_bin, ty_bin, topology)`。

## 批量主量（clean 18-hit，\(n=2\,007\,715\)）

| 量 | 值 |
| --- | ---: |
| 与 truth charge 符号一致 | **0.967** |
| fitted \(q/p\) 符号一致 | 0.955 |
| Spearman(`bending_raw`, \(q/p_{\rm truth}\)) | **0.954** |
| Spearman(`bending_x`) | **0.0005** |
| charge-odd 均值 | \(+8.74\times10^{-4}\) |
| \(\Pr(\mathrm{fit\ flip}\mid\mathrm{bending\ correct})\) | 0.0404 |
| \(\Pr(\mathrm{raw\ bending\ flip})\) | 0.0329 |

完整三站（\(n=3\,075\,701\)）：符号一致 0.942，Spearman 因同一构造；dirty（可构造 \(n=1\,067\,986\)）符号一致 0.896，保留。

按 \(p\)（clean 符号一致 / fit 符号一致）：

| \(p\) | \(n_{\mathrm{sign}}\) | raw | fit |
| --- | ---: | ---: | ---: |
| \(<200\,\mathrm{GeV}\) | 510 347 | 0.983 | 0.992 |
| 200–500 GeV | 487 948 | 0.981 | 0.984 |
| 500–1000 GeV | 343 361 | 0.970 | 0.961 |
| **1–2 TeV** | 489 443 | **0.951** | 0.920 |
| **\(\ge2\,\mathrm{TeV}\)** | 176 492 | **0.923** | 0.853 |

大 \(|t_x|\) 上 raw 仍 \(\approx 0.91\)；大 \(|t_y|\) 反而更高（\(\approx 0.996\)）。\(\lvert b\rvert\) 很小的箱子 raw 符号一致降到 0.53–0.81，fitted 翻号率升到 0.20。

dirty \(\ge2\,\mathrm{TeV}\) 单独报告：raw 符号一致仅 0.724（\(n=81\,208\)）。官方高 \(p\) 闸门只用 clean。完整三站 \(\ge2\,\mathrm{TeV}\) raw 0.861 / fit 0.847，被 dirty 拉低，不得与 clean 闸门混用。

审计 harness 的 `verdict=PASS` 只表示批量 contract 已按冻结闸门写完，**不是** 三站 \(q/p\) 科学 PASS，也不是把 WB119 翻面。

## Smoke / 本地 dump / HTCondor

复用 WB125 contract dump：clean 0.962、Spearman 0.908、class A 身份 `100048` skip 6、9。本地 5 事件与 WB125 skip 0–4 逐轨一致。

Cluster **1113523**（`bigbird24`，`workday`，12 GB）：9/9 正常结束。truth 参考 3 398 774 = WB119 \(n_{\mathrm{primary}}\)。

## 场表（只读）

`FaserFieldTable_v2.root` 已找到。单 zone、毫米边界、`bscale=1e-7`。\(x=y=0\) 中心 \(\sim-0.56\,\mathrm{T}\)（若 kT 注释成立）。S1→S3 直线积分非正式 \(\sim 1.13\,\mathrm{T\cdot m}\)。**路径积分 / 单位 / Jacobian 未锁定，禁止把 bending 转成动量。**

## SHA / provenance

| 项 | 值 |
| --- | --- |
| config | `7871a77ec5b7698ce2bb1518868d4f97f2dc3ff03db485461d36236575e8de55` |
| 正式 batch contract | `9358ee1442426a0db97d069ab525bee218f39e55ec4315aaa935d60fefb12c5b` |
| smoke contract | `c882510e2cf8f879457281cc614103ced99bcf44658db5df10efed089ff2caee` |
| git HEAD | `55cf982302a3c62c57b74f368d5e3ba7723fd33a`（S2J 文件尚未入库） |
| field contract | `d6879d5c020558a3d4ad72fcc1c5beaa0790d3c67fa570261a85408e79a38fac` |
| Calypso | `40892527e9c65409afd2378a2abfc25ddbddac03` |
| Athena / ACTS | 24.0.41 / 32.0.2 |
| 几何 / 条件 / 场 | FASERNU-04，OFLCOND-FASER-06，OFLP200，`GLOBAL-BField-Maps-03` |
| 继承 | WB117–125 哈希核验；WB119 保持 `not_established` |

Dump SHA-256（`three_st_qp_measurement_bending.jsonl`）：

| source | SHA |
| --- | --- |
| 100043_00200_00299 | `76b0e6012ab9c3fe678c7f323889f9c6d20659bd2d3c0ecf318fce98b5bfef32` |
| 100043_00300_00399 | `af4c8d499a2efe7d8d1ffecae1dedf72bcfbd6f1abe8c09e57ddd5a02f5b1b09` |
| 100043_00400_00499 | `51607540ba0e053f328a91ad197094e48b07a45c2ef6fe6615c0e6db7febc8a3` |
| 100044_00200_00299 | `7eb2622e073727c96324361c83dca4aa27a40fb54558bc844f3a05641ac2172d` |
| 100044_00300_00399 | `38fe56114fe70faf65a769b1a442889784f141a9f195777bf3838f3b93d6fb44` |
| 100047_00000_00049 | `88e84c0f1da2bb6793417a33d9c02bf5d1cd03fde91a6c749bbfaae2cc251e0e` |
| 100047_00050_00099 | `c024f8594b046888d879a6a7ccefbae0dff6a1038139e648e604f503984a344e` |
| 100048_00000_00049 | `97221c4f007a1c05af2160c34eb498d1a40470d5c4941ab8bfda9d63b46d43b4` |
| 100048_00050_00099 | `91ea6e00ef688ce54e0ef988264c1251181a175f6d24ed1f34817f98498eba2f` |

## 禁止声称

- 三站 \(q/p\) 已校准，或 `three_st_qp_trusted_observable=true`
- WB119 / S2 被翻成 PASS
- 已授权 \(E[r_{\mathrm{IFT}}\mid q/p]\) 或 \(E[r_{\mathrm{IFT}}\mid\mathrm{bending}]\)
- `bending_raw` 已是 residual-conditioning 观测量
- 0.55 T 或场表直线积分已是正式 Jacobian
- 已有 \(\sigma_b\)
- raw bending 在 \(\ge2\,\mathrm{TeV}\) 按冻结闸门退化
- WB125 的 unmatched source 差已证明 measurement bias
- 101 273 条无 bending 的 fit-flip 已被本 observable 归因
- 已改 reconstruction

## 下一步

S3 保持关闭。`bending_raw` **不得**偷换成已授权的 residual-conditioning observable。若要研究那 41.5% 无完整三站质心的 fit-flip，必须另开、单独预注册，且不得改官方 collection。场表 Jacobian 仍须单独锁定之后才能谈 \(q/p\)-like。
