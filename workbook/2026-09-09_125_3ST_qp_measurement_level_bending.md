# Workbook 125: 3ST measurement-level bending / 拟合无关曲率代理（Yasu-S2I）

日期：2026-09-09
状态：**完成** —— 已在两个冻结 xAOD 上各 20 事件做小样本 contract。未进入 S3、alignment、weak-mode、B14M/B15 或 MM V2。未把 WB119 翻成 PASS。未改 fitter / seed / hit / geometry / field / covariance scale / outlier / collection。未提交 HTCondor。未追 `front()` provenance。未计算 \(E[r_{\mathrm{IFT}}\mid q/p]\) 或 \(E[r_{\mathrm{IFT}}\mid \mathrm{bending}]\)。

**最终判定：`mixed/inconclusive`**

- `decision = three_st_qp_measurement_bending_contract_established`
- `contract_verdict = PASS`
- `diagnosis_verdict = RECORDED`
- `raw_bending_information_present`：**supported**
- `raw_bending_information_limited_at_high_p`：**rejected**（本 contract 的 \(p\ge 100\,\mathrm{GeV}\) 分割）
- `fit_additional_sign_failure`：**supported**
- `raw_measurement_bias/source_dependence`：**supported**（闸门；物理上仍可能是两源 \(p\) 谱不同）
- `official_mechanism = mixed/inconclusive`（三条同时 supported）
- `three_st_qp_trusted_observable = false`
- `residual_conditional_authorized = false`
- `official_qp_like_jacobian_authorized = false`
- `measurement_uncertainty_propagated = false`
- `s2_flipped_to_pass = false`
- `large_dump_submitted = false`
- `htcondor_submitted = false`

## 对唯一科学问题的回答

**不用 CKF fitted \(q/p\) 和 5×5，仅从 S1/S2/S3 measurement geometry 与已确认磁场信息，能否构造有符号、可校准的三站弯曲观测量？fitted \(q/p\) 的非零中心和 sign flip 是已经写在原始弯曲里，还是主要由 CKF/KF reconstruction 引入？**

能构造。正式 primary 是无单位角度 `bending_raw`（弧度），平面是 **YZ**，不是 XZ。本 contract 的 clean 18-hit（\(n=26\)）上：

| 量 | clean 18-hit | 完整三站 | dirty（保留） |
| --- | ---: | ---: | ---: |
| \(n\) | 26 | 37 | 13（11 可构造） |
| 与 truth charge 符号一致 | **0.962**（25/26） | 0.973（36/37） | 1.000（11/11） |
| fitted \(q/p\) 符号一致 | 0.923（24/26） | 0.946 | 1.000 |
| Spearman(`bending_raw`, \(q/p_{\rm truth}\)) | **0.908** | 0.910 | 0.782 |
| Spearman(`bending_x`, \(q/p_{\rm truth}\)) | **0.010** | −0.017 | −0.282 |
| charge-odd 均值 | \(+1.03\times10^{-3}\) | \(+9.06\times10^{-4}\) | \(+6.23\times10^{-4}\) |

XZ 控制面几乎不携带电荷信息，YZ 才是磁弯曲平面。这与 pinned CircleFit 源码一致，不是先假设 \(x\) 或 \(y\)。

同轨对照的四类事件（clean 18-hit）：

| 类 | 定义 | \(n\) | 本样本身份 |
| --- | --- | ---: | --- |
| A | truth 与 bending 同号，fitted \(q/p\) 翻号 | 2 | 100048 / skip 6、9（WB119 冻结 sign-flip） |
| B | \(\lvert\mathrm{bending}\rvert < 10^{-5}\) | 0 | — |
| C | bending 相对 truth 翻号 | 1 | 100043 / skip 10（fit 反而与 truth 同号） |
| D | fitted \(\lvert\mathrm{pull}\rvert\ge 10\) | 0 | — |

两条 WB119 冻结 sign-flip 在 measurement-level bending 上**没有**丢符号：`bending_raw` 为正，与 \(\mu^+\) truth 一致，fitted \(q/p\) 为负。因此至少这两条的 sign-flip 是 **reconstruction 额外引入**，不是原始弯曲已经翻号。WB119 全集 243 758 条 sign-flip 有多少已经在 measurement 上失去符号，**本阶段未量化**（小样本不能覆盖 batch）。

高 \(p\)：clean 中 24/26 条 \(p\ge 100\,\mathrm{GeV}\)（中位 \(p\approx 879\,\mathrm{GeV}\)，最高 \(3.1\,\mathrm{TeV}\)），符号一致率 0.958，相对低 \(p\)（\(n=2\)）只降 0.042，未过冻结闸门 0.15。**不能**用本 contract 巩固 WB121 的 “高 \(p\) 本征曲率信息极限”，也 **不能** 说 raw bending 在高 \(p\) 同样退化。闸门 `raw_bending_information_limited_at_high_p` 被拒绝。

## 冻结定义（看 dump 之前）

### 弯曲平面与符号

Pinned Calypso `40892527e9c65409afd2378a2abfc25ddbddac03`：

1. `CircleFit.cxx:6-8`：space point 送入 Taubin 圆拟合的是 \((z,y)\)，不是 \((z,x)\)。
2. `CircleFitTrackSeedTool.cxx:310-314`：\((z,x)\) 只做非弯曲平面直线拟合。
3. `CircleFitTrackSeedTool.cxx:335,345`：`charge = (cy < 0) ? +1 : -1`。
4. Lorentz：\(v\sim\hat z\)，FASER 偶极 \(B\sim\hat x\) \(\Rightarrow\) \(v\times B\) 在 \(y\)。\(\mu^+\)（\(q>0\)）受力 \(+y\)，正 sagitta，正 `bending_raw`。

禁止把 seed 手写常数当正式 Jacobian：`Seed::fit` 用 \(p = r\cdot 0.001\cdot 0.3\cdot 0.55\)（`CircleFitTrackSeedTool.cxx:334`），`fakeFit(B=0.55)`。**0.55 T 不是机器确认的 \(\int B_\perp dl\)**。正式结果保持 `bending_raw`（弧度）。未授权 \(q/p\)-like 换算。

站 \(z\)（不重导，mm）：IFT \(0=-1860.15\)，S1 \(1=47.4\)，S2 \(2=1237.4\)，S3 \(3=2427.4\)。S2−S1 = S3−S2 = 1190 mm。质心用测量到的 \((y,z)\)，不用这张表代替斜率分母。

### Primary observable

\[
t_{y,12}=\frac{y_2-y_1}{z_2-z_1},\quad
t_{y,23}=\frac{y_3-y_2}{z_3-z_2},\quad
\texttt{bending\_raw}=\arctan t_{y,12}-\arctan t_{y,23}
\]

伴随量 \(s_y=y_2-[y_1+(y_3-y_1)(z_2-z_1)/(z_3-z_1)]\)。符号：\(\texttt{bending\_raw}>0 \Leftrightarrow s_y>0 \Leftrightarrow\) CircleFit \(+1\)。

正交控制：`bending_x` 对 \(x\) 做同一公式，不得通过 “present” 闸门。

输入：MOT 匹配的 `SCT_SpacePoint` 三维点按站平均；若无 space point，用 `SCT_DetectorManager` 把 MOT cluster 的 \((\mathrm{loc}X, \mathrm{positionAlongStrip})\) `localToGlobal`。禁止：persisted `front()` xyz、fitted \(q/p\)、5×5、truth \(q/p\)、seed 圆心。

\(\sigma_b\)：cluster `local_cov00` 是一维 strip 方差（本 dump \(\approx 5.88\times10^{-4}\,\mathrm{mm}^2\)），没有已验证的立体声 Jacobian，**不虚构 \(\sigma_b\)**。本阶段只做 point-estimator 校准。\(P(\mathrm{sign\ flip}\mid \lvert b\rvert/\sigma_b)\) 改为按 \(\lvert\mathrm{bending\_raw}\rvert\) 四分位报告 fitted 翻号率。

### 闸门（看结果前冻结）

- present：clean 符号一致 \(\ge 0.85\) 且 \(\lvert\mathrm{Spearman}\rvert\ge 0.30\)，且 \(\lvert\mathrm{Spearman}_x\rvert\le 0.25\)
- high-\(p\) 退化：\(p\ge 100\,\mathrm{GeV}\) 符号一致率比低 \(p\) 低 \(\ge 0.15\)（至少 5 条高 \(p\)）
- fit extra：在 bending 与 truth 同号的子集上，fitted 翻号率 \(\ge 0.05\)
- source：construction / validation 的 charge-odd 均值相对极差 \(\ge 0.50\) 或异号
- clean：`n_mot=18` 且站 \(\{1,2,3\}\) 且无 IFT leak。dirty **不删**。

## 坐标实现（第一次 smoke 的教训）

第一次 `dumps/smoke/`（20+19 条）站号和 `n_mot` 正确，但 ROT `globalPosition()` 全是 NaN：POOL 读回的 `FaserSCT_ClusterOnTrack` 不带 detector element，`setValues` 未把全局坐标补上。这些 smoke 文件保留作法医记录，**不是**正式 contract。

正式 dump 是 `dumps/contract/`：事件内读 `SCT_SpacePointContainer` + `SCT_ClusterContainer`，按 Identifier / IdentifierHash 把 MOT cluster 配到 space point（与 CircleFit 同源）。100043：343/352 hit 来自 space point；100048：320/322。质心 \(z\) 落在各站窗口内（S1 \(\sim 50\,\mathrm{mm}\)，S2 \(\sim 1240\,\mathrm{mm}\)，S3 \(\sim 2430\,\mathrm{mm}\)）。

## 冻结 denominator 与 provenance

- 源：与 WB119/124 相同的两个 xAOD，`s0013-r0022`，construction `100043`（\(\mu^-\)）/ validation `100048`（\(\mu^+\)）
- `SkipEvents=0`，`maxEvents=20`，本地，不覆盖
- 几何 / 场 / 条件：FASERNU-04，OFLCOND-FASER-06，OFLP200，`GLOBAL-BField-Maps-03/FaserFieldTable_v2.root`。场表积分 **未** 在本阶段机器确认，故无正式 \(q/p\)-like
- Athena 24.0.41，ACTS 32.0.2
- config SHA `b03976103cf8df63ebc10a6db6bc709a077e6b61bc0b8d7df2078f199fe2b4a4`
- contract SHA `d11537b731846b4fc4b7e5cd71d95b62258a32bea24c79a9286166c6edac3f97`
- code SHA（git HEAD）`55cf982302a3c62c57b74f368d5e3ba7723fd33a`（S2I 文件尚未入库）
- dump 100043 `94fa6f95cefd07a650b06ed19db5b6fe7f3e674e467c7dbd10fdb108ca72b1de`
- dump 100048 `aa2f1429f5ea73b48114c3eaca7fbedb71c7f3a11b7802d85a99a54e827819b1`
- 官方 run：`yasu_s2i_three_st_qp_measurement_bending_contract_20260909T170948Z_9164bf14`
- 继承 WB117–124 哈希与 S2H config 一致；CircleFit 四文件已 pin 并核验

Truth 只作 source-disjoint MC 校准参考，不进 construction / fit / selection / prior。dump 里的 fitted \(q/p\) 只用于事后同轨对照。

## Source stability

charge-odd 均值（`bending_raw * sign(truth)`）：construction \(+1.68\times10^{-3}\)（\(n=14\)），validation \(+2.64\times10^{-4}\)（\(n=12\)），同号，相对极差 0.842，过闸门 0.50。**不能**据此断言存在与电荷无关的测量偏置：两源是不同 particle-gun 电荷/能谱。需要以后在 \(p\) 匹配后再看。闸门按冻结规则记 **supported**。

## 与 fitted \(q/p\) 的逐轨比较

Clean 上 fitted 翻号 2/26，全部落在 \(\lvert\mathrm{bending\_raw}\rvert\) 最小四分位（2/7）。较大弯曲的三个四分位 fitted 翻号为 0。这与 “弱弯曲时 KF 更容易丢符号、强弯曲时 raw 与 fit 都稳” 相容，但 \(n\) 太小，不能外推到 243 758。

100043 / skip 10 是唯一 raw 翻号：`bending_raw=+1.18\times10^{-4}\)，truth 与 fit 都是负。说明 raw bending 也会错，只是本样本里远少于 fitted 翻号。

## 禁止声称

- 三站 \(q/p\) 已校准或 `three_st_qp_trusted_observable=true`
- WB119 / S2 被本阶段翻成 PASS
- 已授权 \(E[r_{\mathrm{IFT}}\mid q/p]\) 或 \(E[r_{\mathrm{IFT}}\mid \mathrm{bending}]\)
- `bending_raw` 已是 residual-conditioning 观测量
- 0.55 T / 0.3 是正式 \(\int B_\perp dl\) 或 \(q/p\) Jacobian
- 已有 \(\sigma_b\) 或 pull/coverage 校准
- 已量化 WB119 的 243 758 条 sign-flip 有多少在 measurement 上丢符号
- raw bending 在高 \(p\) 同样退化，因而巩固了 WB121（本闸门被拒绝）
- construction/validation 均值差已经证明 raw measurement bias（能谱未匹配）
- 已改 fitter / seed / hit / geometry / field / covariance / collection
- 第一次 smoke 的空质心可以当 bending 输入

## 下一步

S3 保持关闭。`bending_raw` **不得**偷换成已授权的 residual-conditioning observable。任何未来 \(E[r_{\mathrm{IFT}}\mid \mathrm{bending}]\) 必须在本阶段独立通过校准后另行预注册。

本 contract 已经证明：YZ measurement-level bending 在 clean 18-hit 上携带电荷符号，并且 WB119 的两条冻结 sign-flip 属于 fit-additional。因此 **有理由以后单独预注册 reconstruction-mapping 研究**（不在本阶段做，也不改官方 collection）。

若要回答 243 758 的分数，需要授权一次更大的、仍然 inspect-only 的 measurement dump（同一 observable，禁止 HTCondor 自动提交）。高 \(p\) 分辨与 source 稳定性应在 \(p\) 匹配的更大样本上重评。场图表 \(\int B_x\,dl\) 仍未机器确认；在确认之前不得发布正式 \(q/p\)-like。

**Yasu-S2J / WB126 已记录**（`mixed/inconclusive`）：cluster `1113523` 9/9 结束；243 758 已拆分（46.4% fit 新增、11.2% measurement 已翻、41.5% 无可构造三站 bending）；高 \(p\) raw 极限与匹配后 source 依赖被拒绝。S3 仍然关闭。
