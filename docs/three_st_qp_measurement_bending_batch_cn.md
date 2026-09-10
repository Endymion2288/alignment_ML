# 3ST measurement-level bending 批量校准 / CKF mapping（Yasu-S2J）

Workbook 126。对已冻结的 WB125 YZ `bending_raw` 做批量校准，并把
WB119 的 fitted-\(q/p\) sign-flip 拆成 measurement 层丢失与
reconstruction mapping。不把 WB119 翻成 PASS，不进入第 3 阶段，
不改 bending 定义、fitter、seed、hit、geometry、field 或 collection。

## 官方 run

`yasu_s2j_three_st_qp_measurement_bending_batch_batch_20260910T082512Z_f61fa2a8`

```
decision = three_st_qp_measurement_bending_batch_recorded
raw_bending_information_present     = supported
raw_bending_high_p_limit            = rejected
fit_additional_sign_failure         = rejected
raw_measurement_source_dependence   = rejected
mapping_nontransferability          = supported
official_mechanism = mixed/inconclusive
three_st_qp_trusted_observable = false
residual_conditional_authorized = false
official_qp_like_jacobian_authorized = false
s2_flipped_to_pass = false
```

HTCondor cluster `1113523`（9/9 正常结束）。truth 参考
\(3\,398\,774\) 与 fitted 翻号 **243 758** 与 WB119 完全一致。

## 回答

**243 758 条 fitted sign-flip**

| 类 | \(n\) | 占 243 758 |
| --- | ---: | ---: |
| A raw bending 已正确 | 113 144 | 46.4% |
| C raw bending 已翻号 | 27 206 | 11.2% |
| \(\lvert b\rvert<10^{-5}\) | 2 135 | 0.88% |
| 无可构造三站 bending | 101 273 | 41.5% |

可分类的 142 485 条中 A 为 79.4%、C 为 19.1%。
`mapping_nontransferability` 因此为 **supported**。

clean 18-hit（\(n=2\,007\,715\)）正式条件概率：

- \(\Pr(\mathrm{fit\ flip}\mid\mathrm{raw\ bending\ correct})=0.0404\)
- \(\Pr(\mathrm{raw\ bending\ flip})=0.0329\)

**Source dependence** 在匹配后不成立。未匹配 charge-odd 相对极差
0.397。按电荷、\(p\)、\(t_x\)、\(t_y\)、topology 匹配后（165 cell，
匹配分数 0.999）相对极差 0.038。WB125 的未匹配闸门是能谱效应。
正式：`raw_measurement_source_dependence=rejected`。

**高 \(p\)** 未过冻结退化闸门。clean \(\ge2\,\mathrm{TeV}\)
（\(n=176\,492\)）raw 符号一致 0.923，相对 \(<200\,\mathrm{GeV}\)
的 0.983 只降 0.060。1–2 TeV 为 0.951。**不得**沿用 WB125 的
\(n=26\) / 100 GeV 拒绝。`raw_bending_high_p_limit=rejected`。
fitted \(q/p\) 在 \(\ge2\,\mathrm{TeV}\) 为 0.853；raw−fit 差距
0.070 低于冻结的 0.15，故 `fit_additional_sign_failure=rejected`，
尽管该层 raw 仍明显好于 fit。dirty \(\ge2\,\mathrm{TeV}\) raw
符号一致仅 0.724（\(n=81\,208\)），不进官方闸门。

## 冻结 observable 与样本

YZ `bending_raw` 与 WB125 相同。质心为 MOT 匹配 space point，否则
`localToGlobal`。匹配斜率用 S1–S3 测量弦。truth 只作
source-disjoint 校准参考。源是 WB119 的同一 9 个 xAOD。clean：
`n_mot=18` 且站 \(\{1,2,3\}\) 且无 IFT leak；dirty 保留。

## Clean 18-hit 校准

| 量 | 值 |
| --- | ---: |
| sign(`bending_raw`) 对 truth charge | 0.967 |
| sign(fitted \(q/p\)) 对 truth | 0.955 |
| Spearman(`bending_raw`, \(q/p_{\rm truth}\)) | 0.954 |
| Spearman(`bending_x`) | 0.0005 |
| charge-odd 均值 | \(+8.74\times10^{-4}\) |

完整三站（\(n=3\,075\,701\)）符号一致 0.942。dirty 可构造
（\(n=1\,067\,986\)）为 0.896，不删除。

## 场表（只读）

`FaserFieldTable_v2.root` 位于
`/cvmfs/faser.cern.ch/repo/sw/software/22.0/faser/offline/ReleaseData/v20/MagneticFieldMaps/FaserFieldTable_v2.root`。
单 zone，毫米边界，`bscale=1e-7`。\(x=y=0\) 上存储 \(B_x\) 若按
kT 注释约 \(-0.56\,\mathrm{T}\)。S1→S3 直线积分非正式约为
\(1.13\,\mathrm{T\cdot m}\)。

路径积分、单位和 Jacobian **未锁定**。不得把 `bending_raw` 转成
动量。

## Provenance

| 项 | SHA / id |
| --- | --- |
| config | `7871a77ec5b7698ce2bb1518868d4f97f2dc3ff03db485461d36236575e8de55` |
| batch contract | `9358ee1442426a0db97d069ab525bee218f39e55ec4315aaa935d60fefb12c5b` |
| smoke contract | `c882510e2cf8f879457281cc614103ced99bcf44658db5df10efed089ff2caee` |
| git HEAD | `55cf982302a3c62c57b74f368d5e3ba7723fd33a` |
| field contract | `d6879d5c020558a3d4ad72fcc1c5beaa0790d3c67fa570261a85408e79a38fac` |
| Calypso | `40892527e9c65409afd2378a2abfc25ddbddac03` |
| Athena / ACTS | 24.0.41 / 32.0.2 |
| HTCondor | cluster `1113523` |

WB117–125 哈希已核验。WB119 保持
`three_st_qp_calibration_not_established`。

各 dump 的 SHA-256 见
`workbook/2026-09-09_126_3ST_qp_measurement_bending_batch.md`。

## 禁止声称

三站 \(q/p\) 仍不可信。第 3 阶段关闭。`bending_raw` 不是已授权的
residual-conditioning 观测量。0.55 T 与非正式直线积分不是正式
Jacobian。未虚构 \(\sigma_b\)。冻结闸门下 raw bending 并未在
\(\ge2\,\mathrm{TeV}\) 死亡。101 273 条无完整质心的 flip 不能用
本 observable 归因。

## 下一步

第 3 阶段保持关闭。任何未来的
\(E[r_{\mathrm{IFT}}\mid\mathrm{bending}]\) 必须另行预注册。
占 WB119 flip 41.5% 的无三站质心样本若要研究，必须新开并单独冻结。
在场表 Jacobian 锁定前，\(q/p\)-like 换算仍未授权。
