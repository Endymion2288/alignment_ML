# 3ST measurement-level bending / 拟合无关曲率代理（Yasu-S2I）

Workbook 125。只从 S1/S2/S3 measurement geometry 构造有符号、可校准的
三站弯曲观测量。不修复 `CKFTrackCollectionWithoutIFT` 的 \(q/p\)，
不重开 covariance / refit，不追 `front()` provenance，不把 WB119
翻成 PASS，不进入第 3 阶段。

## 官方 run

Contract：`yasu_s2i_three_st_qp_measurement_bending_contract_20260909T170948Z_9164bf14`

```
decision = three_st_qp_measurement_bending_contract_established
raw_bending_information_present              = supported
raw_bending_information_limited_at_high_p    = rejected
fit_additional_sign_failure                  = supported
raw_measurement_bias/source_dependence       = supported
official_mechanism = mixed/inconclusive
three_st_qp_trusted_observable = false
residual_conditional_authorized = false
official_qp_like_jacobian_authorized = false
measurement_uncertainty_propagated = false
s2_flipped_to_pass = false
htcondor_submitted = false
```

两个冻结 xAOD 各 20 事件。39 条 WithoutIFT；26 条 clean 18-hit；
37 条完整三站；13 条 dirty（保留）。37/39 可构造质心。

## 物理定义（看 dump 前冻结）

弯曲平面是 **YZ**，不是 XZ。锁定自 pinned Calypso
`40892527e9c65409afd2378a2abfc25ddbddac03`：

- `CircleFit.cxx:6-8` 把 space point 的 \((z,y)\) 送入 Taubin 圆拟合
- `CircleFitTrackSeedTool.cxx:310-314` 用 \((z,x)\) 做非弯曲直线
- `CircleFitTrackSeedTool.cxx:335,345`：`charge = (cy < 0) ? +1 : -1`
- Lorentz：\(v\sim\hat z\)，偶极 \(B\sim\hat x\) \(\Rightarrow\)
  \(v\times B\) 在 \(y\)。\(\mu^+\) \(\Rightarrow\) \(+y\) sagitta

正式 primary（弧度）：

\[
\texttt{bending\_raw}
=\arctan\frac{y_2-y_1}{z_2-z_1}
-\arctan\frac{y_3-y_2}{z_3-z_2}
\]

伴随 sagitta
\(s_y=y_2-[y_1+(y_3-y_1)(z_2-z_1)/(z_3-z_1)]\)。
\(+\texttt{bending\_raw}\Leftrightarrow +s_y\Leftrightarrow\)
CircleFit 电荷 \(+1\)。正交控制 `bending_x` 对 \(x\) 用同一公式。

输入是 MOT 匹配的 `SCT_SpacePoint` 全局坐标按站平均；否则用
`SCT_DetectorManager` 对 MOT cluster 做 `localToGlobal`。禁止
persisted `front()` \(xyz\)、fitted \(q/p\)、5×5、truth \(q/p\)、
seed 圆心。

seed 常数 \(0.55\,\mathrm{T}\) 与 \(0.3\)
（`CircleFitTrackSeedTool.cxx:334-345`）**不是**正式
\(\int B_\perp dl\)。未授权 \(q/p\)-like 换算。不虚构 \(\sigma_b\)：
cluster `local_cov00` 是一维 strip 方差，没有已验证的立体声
Jacobian。本阶段只做 point-estimator 校准。

站 \(z\)（mm，继承，不重导）：IFT \(-1860.15\)，S1 \(47.4\)，
S2 \(1237.4\)，S3 \(2427.4\)。

## 几何 / 磁场 provenance

标签：FASERNU-04，OFLCOND-FASER-06，OFLP200，
`GLOBAL-BField-Maps-03/FaserFieldTable_v2.root`，
`GLOBAL-BField-Scale-03`，reco `s0013-r0022`。
场图表线积分 **未** 在本阶段机器确认。

第一次 `dumps/smoke/` 写出后，ROT `globalPosition()` 全空
（POOL 读回的 `FaserSCT_ClusterOnTrack` 没有 detector element）。
这些文件保留作法医记录，**不是**正式 contract。正式 dump 在
`dumps/contract/`，按 Identifier 把 MOT cluster 配到 space point
（与 CircleFit 同源）。

## 冻结 denominator 与 SHA

Construction `mc24_100043_00400_00499`（\(\mu^-\)），
validation `mc24_100048_00000_00049`（\(\mu^+\)）。
`SkipEvents=0`，20 事件，只跑本地。

| 项 | SHA-256 / SHA |
| --- | --- |
| config | `b03976103cf8df63ebc10a6db6bc709a077e6b61bc0b8d7df2078f199fe2b4a4` |
| contract | `d11537b731846b4fc4b7e5cd71d95b62258a32bea24c79a9286166c6edac3f97` |
| git HEAD | `55cf982302a3c62c57b74f368d5e3ba7723fd33a` |
| dump 100043 | `94fa6f95cefd07a650b06ed19db5b6fe7f3e674e467c7dbd10fdb108ca72b1de` |
| dump 100048 | `aa2f1429f5ea73b48114c3eaca7fbedb71c7f3a11b7802d85a99a54e827819b1` |

WB117–124 继承哈希与 S2H config 一致。CircleFit 源码已 pin 并核验。
Truth 只作 source-disjoint MC 校准参考。

## 校准与源稳定性（clean 18-hit，\(n=26\)）

| 量 | 值 |
| --- | ---: |
| `bending_raw` 与 truth charge 符号一致 | 0.962（25/26） |
| fitted \(q/p\) 与 truth 符号一致 | 0.923（24/26） |
| Spearman(`bending_raw`, \(q/p_{\rm truth}\)) | 0.908 |
| Spearman(`bending_x`, \(q/p_{\rm truth}\)) | 0.010 |
| charge-odd 均值 | \(+1.03\times10^{-3}\) |
| 高 \(p\)（\(p\ge 100\,\mathrm{GeV}\)，\(n=24\)）符号一致 | 0.958 |
| 低 \(p\)（\(n=2\)）符号一致 | 1.0 |
| construction charge-odd 均值 | \(+1.68\times10^{-3}\)（\(n=14\)） |
| validation charge-odd 均值 | \(+2.64\times10^{-4}\)（\(n=12\)） |
| 源相对极差 | 0.842（同号） |

可构造的 dirty（\(n=11\)）符号一致率为 1.0，样本未删。完整三站
（\(n=37\)）符号一致率 0.973。

源极差闸门触发。两个 charge-odd 均值同号；约 6 倍差距仍可能是
两套 particle-gun 能谱，**不是**已证明的 charge-even 测量偏置。

高 \(p\) 退化在本 contract **被拒绝**：中位 \(p\) 为
\(879\,\mathrm{GeV}\)，100 GeV 分割只降 0.042。这 **不能** 巩固
WB121 的本征曲率信息极限。

## 与 fitted \(q/p\) 的逐轨比较

| 类 | \(n\)（clean） | 身份 |
| --- | ---: | --- |
| A truth 与 bending 同号，fit 翻号 | 2 | 100048 / skip 6、9（WB119 sign-flip） |
| B \(\lvert\mathrm{bending}\rvert<10^{-5}\) | 0 | — |
| C bending 相对 truth 翻号 | 1 | 100043 / skip 10（fit 与 truth 同号） |
| D \(\lvert\mathrm{pull}\rvert\ge 10\) | 0 | — |

两条 A 类都在 \(\lvert\mathrm{bending\_raw}\rvert\) 最小四分位
（2/7）。其余三个四分位 fitted 翻号为 0。WB119 的 243 758 条
sign-flip **未**在本阶段量化。

本窗口里两条冻结 WB119 sign-flip 在 measurement-level 仍有电荷
信息。reconstruction mapping 可以作为**以后**单独预注册的理由；
本阶段不打开。

## 本阶段不得声称

三站 \(q/p\) 未校准。WB119 未翻面。未授权
\(E[r_{\mathrm{IFT}}\mid q/p]\) 或
\(E[r_{\mathrm{IFT}}\mid\mathrm{bending}]\)。
`bending_raw` 不是已授权的 residual-conditioning 观测量。
0.55 T 不是 \(\int B_\perp dl\)。没有 \(\sigma_b\)。243 758
的分数在本阶段未知（WB126 已量化）。不能说 raw bending 在高 \(p\) 同样退化。源均值差
不是已证明的测量偏置。第一次空质心 smoke 不能当 bending 输入。

## 下一步

第 3 阶段保持关闭。任何未来的
\(E[r_{\mathrm{IFT}}\mid\mathrm{bending}]\) 必须另行预注册。
正式 \(q/p\)-like 换算仍要求场图表 \(\int B_x\,dl\) 的机器确认。

Yasu-S2J / WB126 **已记录**（`mixed/inconclusive`）：cluster
`1113523` 已结束；243 758 条 WB119 flip 已拆分（46.4% 为 fit
新增、11.2% 在 measurement 已翻、41.5% 无可构造三站 bending）。
高 \(p\) raw 退化与匹配后 source 依赖被拒绝。trusted / S3 仍为
false。
