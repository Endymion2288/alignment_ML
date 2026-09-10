# Yasu 3ST q/p ↔ IFT 几何弱耦合研究

`yasuCheck` 分支上的预注册研究线。复用 WB95–116 的物理 CKF q/p、5×5、
LTO、ACTS transport、profile likelihood 和 identifiability 基础设施。
**不从** SegmentFit dummy `q/p = 10^{-5}/MeV` 出发。

## 科学问题

官方 S1+S2+S3 CKF（`CKFTrackCollectionWithoutIFT`）给出的带符号 `q/p`
是否有偏置或非零均值，以及它是否与 IFT alignment 自由度（`R_y`、
`d_x`、类斜率 `dy/dz`）形成真正的弱模，从而造成 IFT unbiased residual
非零均值或四站 alignment 不稳定。

仅有 residual 相关不够。必须区分

- q/p 重建偏置
- 几何弱模
- ACTS 传播 / 导航失败
- 协方差未校准
- candidate / 选择损失

## 阶段

| 阶段 | 闸门 | 授权 |
| --- | --- | --- |
| 0 | 源码 + ROOT + dump 锁定官方 WithoutIFT 定义 | 3ST→IFT 链 — **PASS，WB117** |
| 1 | 独立 3ST→IFT 预测；IFT hit 不进 fit | MC q/p 校准 — **PASS，WB118** |
| 2 | truth-known、source-disjoint MC：符号、尺度、不确定度、5×5 | residual 条件期望 — **FAIL，WB119；第 3 阶段关闭** |
| 2D | 诊断 WB119 FAIL（曲率 / 协方差 / 参考 / 电荷） | residual 条件期望 — **已记录 mixed/inconclusive，WB120；第 3 阶段仍关闭** |
| 2E | 拆开曲率信息极限与协方差失败（A/B/C/D） | residual 条件期望 — **已记录 A+D 支持、C 拒绝、B 未决，WB121；第 3 阶段仍关闭** |
| 2F | 分解 \(E[q/p]\) 的 population / reconstruction / tail | residual 条件期望 — **已记录 mixed/inconclusive，WB122；第 3 阶段仍关闭** |
| 2G | 3ST \(q/p\) 极端尾巴与 split/source 不稳定性 | residual 条件期望 — **已记录 mixed/inconclusive，WB123；第 3 阶段仍关闭；已授权 ≤20 事件 refit 诊断 dump** |
| 2H | ≤20 条冻结 identity 上的 CKF / KF-refit provenance | residual 条件期望 — **已记录 mixed/inconclusive，WB124；第 3 阶段仍关闭；S2-front 不是 fallback 的一一标签** |
| 2I | 拟合无关的 3ST measurement-level bending / 曲率代理 | residual 条件期望 — **contract 已建立，mixed/inconclusive，WB125；第 3 阶段仍关闭；clean 18-hit 上 YZ `bending_raw` 有信息；两条 WB119 sign-flip 属 fit-additional；243 758 在 2J 量化** |
| 2J | WB125 `bending_raw` 批量校准与 CKF mapping 审计 | residual 条件期望 — **已记录 mixed/inconclusive，WB126；第 3 阶段仍关闭；raw bending 有信息（符号 0.967，Spearman 0.954）；243 758 条 flip 中 46.4% 为 fit 新增、11.2% 在 measurement 已翻、41.5% 无可构造三站 bending；高 \(p\) raw 极限与匹配后 source 依赖被拒绝；trusted/S3 仍为 false** |
| 3 | `E[r_IFT|q/p]`、`E[r_IFT|t_y]`、电荷 / p / 斜率依赖 | Jacobian / profile |
| 4 | `∂(d_x,R_y)/∂(q/p)`、profile likelihood、Fisher 奇异向量 | 仅在需要时做 payload |
| 5 | 先确认 pivot / convention，再做受控物理 payload | 可选 |
| 6 | 如需检验 boundary 假设，固定同一批事件和几何扫描 ACTS | 可选 |
| 7 | 仅在 MC closure 之后做真实数据 residual 诊断 | 只监测 |

小样本 → 独立 MC closure → 批量统计 → 真实数据 residual 诊断。
第 0–1 阶段完成前不跑大规模作业。

## 继承规则

fail-closed 访问策略。sealed / final-blind 路径保持关闭。既有产物不可
覆盖。construction / validation 源保持文件级不重叠。负结果保留。不得
靠改 threshold、删异常、truth q/p、人为 rescale 协方差、ridge 或
临时 prior 强行 PASS。

LTO 是可复用基础设施，不能顶替官方三站 collection。WB109 LTO 保留 IFT。

## 在对应阶段 PASS 之前禁止声称

- Yasu 的观察已被 q/p–几何弱模解释
- 3ST q/p 已经校准
- IFT residual 偏置由 `R_y` 或 `d_x` 造成
- 四站 alignment 不稳定因此得到解释
- ACTS warning 计数等于重建损失
