> **If I took over this project today, I would build a physics-informed, explicit multi-length route-energy model with a trainable shared representation and exact set packing.**
>
> **The single most important next experiment is a source-and-geometry-held-out comparison of frozen latent features, raw physical route features, and a trainable representation, using identical candidates, canonical utilities, and the same exact solver.**
>
> **The current direction I would stop or deprioritize is frozen-backbone Head-Only correction as the main research programme, including further V4 loss sweeps and automatic promotion of bounded V5A to full training.**

# 1. Executive Summary

**独立审查结论：研究问题值得继续；当前 association 的训练目标、route utility 接口和证据解释需要重建，alignment 的科学验证尚未完成。** 保留局部重建、物理传播和 endpoint-capacity packing 的基础设施，不保留“已有 SVD/closure 已经证明 alignment 可用”“B=0 排除了 ranking 问题”“train 拟合成功证明 frozen representation 足够”等结论。

最大的瓶颈是**对完整径迹和竞争径迹集合的效用建模**，叠加训练输入分布与 deployment 的偏离。现有完整径迹效用由边分数相加而来；完整 route 的修正又通过 sigmoid→clip→logit 接口，和 fragment 使用的尺度不一致。一个无须数据、可确定性复现的反例显示：**即使 correction=0，完整径迹也会变成 fragment**。局部 packing margin 比较一条竞争 route，不能代表多个兼容竞争 routes 的总效用。这两项必须先修复，才能可靠解释架构实验。

**STOP：继续把 frozen-backbone Head-Only 当主路线。** V4 的失败不是“所有 frozen head 在数学上不可能成功”的证明，但已足以否决在没有新的信息瓶颈实验前继续投入 head/loss/bound 扫描。V5A 四个训练 checkpoint 均完成30个epoch，最新两个source-transfer folds已核对为 **FAIL**。新增W73/W74提供了有价值的raw-pair表示线索和代码，但其归因/metric仍有缺口，不支持恢复Head-Only主线。

**主路线：显式、物理输入驱动、同时覆盖 2/3/4 站的 route energy，使用可训练共享表示和同一 exact set-packing objective。** 先做低容量、严格匹配的表示对照，再确定是否需要现有 Transformer；不先堆更大网络。备选路线是 field-aware 的共同径迹拟合与统计 route likelihood，在必要时仅学习其残差。

接下来最重要的三个动作：

1. 修复并锁定 utility/solver contract，补充零初始化、饱和分数、多竞争集合和 fail-closed gate 测试；保留历史结果的原始语义。
2. 冻结 source/event/payload 与 checkpoint ancestry 清单，开展上述表示对照；所有正式训练、批量评估均走 HTCondor，完全不接触 final-blind/sealed。
3. 并行资格审查 truth-association alignment：区分 SE(3) 代数恒等式与固定磁场中的观测对称性，验证独立噪声、真实更新后 refit、弱模及 uncertainty coverage。

## 审查范围、时间与证据等级

- 审查日期：2026-09-05；本地分支 `4station`。起始快照为 `1ad9939e9a2d066d1da6f65e284e5edbdeb62d2c`（2026-09-04 20:39:38 +0200）；审查期间分支前进，已增量审查到交付快照 **`6f63dbb0cc116e58cdc5bbbff1ffa9cba8cda00d`**。远端实时HEAD未核实：SSH认证失败，网页缓存不能替代本地快照。结论绑定上述交付SHA。
- 最近历史：`5a61d14` 加入 V5A evaluation/Condor；`f90b2bd` 加入 bounded head/CV；`2b2fb89` 为 Workbook 71 结果；`1ad9939` 为 Workbook 72 执行状态。其后 `e571a7e` 融合single-pass evaluation，`a77812d`记录V5A gate FAIL，`2cc7b5b`加入W73 domain audit，`6f63dbb`加入W74 physical-pair encoder。最新21个变更文件已全文静态读取/语法检查，并深查模型、probe、metric/gate与测试；W74模型效果仍待实验。
- 系统索引并静态扫描全部 tracked 文本/Python，重点逐函数核查当前科学链。覆盖两份 README、`docs/`、`workbook/`、`configs/`、`models/`、`alignment/`、`geometry/`、`datasets/`、`evaluation/`、`baselines/`、`training/`、`scripts/`、`tests/` 和 `pyproject.toml`。起始快照规模包括137个config、210个script、84个test文件、76个workbook；另外核查上述21文件增量；不声称逐行人工审计了全部历史脚本，也不声称重跑全部历史实验。
- **CODE**：实际实现/配置；**ARTIFACT**：此次核对的已有允许访问产物；**REPRODUCED**：本次测试或确定性 sanity check；**HISTORICAL**：文档报告而未独立重跑；**PROPOSED**：本文件未来任务；**UNKNOWN / NEEDS EXPERIMENT**：当前不能下结论。下文按此区分，不能把计划写成结果。
- 本次测试：`source scripts/setup_environment.sh ml` 后，以单线程 BLAS/OpenMP 运行 `pytest -q`，起始快照 **445 passed, 1 failed，30.94 s**；交付快照重跑为 **463 passed, 1 skipped, 1 failed，54.14 s**，skip为新增GPU测试在CPU环境跳过。失败为 `test_registered_disk_contract_matches_workbook_45` 缺外部 artifact，见第 6 节。另运行纯合成 utility/packing 和 dropout sanity checks。
- 只读取已有允许访问的 train/development 审计 JSON/JSONL 和训练 checkpoint；没有运行正式训练，没有进行新 development ROOT 推理，没有打开 final-blind 或 sealed test，也没有修改研究实现。

# 2. Current System Reconstruction

## 2.1 真正的数据流

```text
MC / raw / xAOD + geometry payload + nominal field/material configuration
  → external Calypso station-local cluster refit
  → local tracklets: station, ID, (x,y,tx,ty), covariance, quality/hit pattern
  → ACTS source-tracklet propagation to downstream station planes
  → physical candidate export
  → synthetic overlay / optional candidate fan-out to target tracklets
  → node features + directed candidate residual/covariance features
  → contextual graph/route Transformer → adjacent edge logits
  → enumerate contiguous 2/3/4-station hypotheses
       ↘ V4/V5: complete-route additive correction
  → edge/route score conversion + dustbin-derived route utility
  → exact unit-capacity set packing → selected fragments/complete tracks
  → select/deduplicate associated physical observations
  → nominal/finite-difference/observed residual-bank intersection
  → nominal-subtracted WLS in a fixed-reference 15-column chart
  → relative-transform response/capture report

Truth metadata ──→ pool selection, synthetic labels, evaluation denominators
              └─→ some export/enumeration requirements (must decouple for deployment)
```

当前可复现链条的末端是 **conditional response closure**；还不是经过独立资格审查的、对未知真实几何迭代收敛的 alignment 系统。

## 2.2 输入和模型输出

`training/geometry_aware_transformer.py::_node_features` 使用约 17 维节点信息：四维局部状态、z、协方差对角尺度、fit quality、hit count/pattern。边特征包含四维传播 residual、逐分量 pull、chi²、logdet、dz；完整协方差相关结构没有直接作为矩阵输入。位置和隐藏表示不是 SE(3)-equivariant。

`models/route_transformer.py` 的 backbone 对候选图做上下文建模；历史 W64 还把 route context 聚合回 edge logits。因此“edge-only”描述的是最终 route utility 的可加形式，**并不表示每条边完全独立、没有 route 信息**。

模型真正输出的是监督分类/排序用的分数。weighted/focal loss、负样本构造和 route correction 均改变其统计意义；未经验证不能称为真实条件概率或 likelihood ratio。V4/V5 输出完整 route 的 raw correction，同时保留 frozen W64 edge/fragment 通道。

## 2.3 Solver 优化什么

在固定候选集合 \(\mathcal R\) 上，solver 求

\[
\max_{y_r\in\{0,1\}}\sum_{r\in\mathcal R}U_r y_r,
\qquad\sum_{r:v\in r}y_r\leq1\quad\forall\text{ tracklet }v.
\]

W64 当前 operating point 的 complete utility 为 \(\sum_{e\in r}\operatorname{logit}(p_e)-4\)，三站为两条 logit 之和减 3。这个常数来自当前 unmatched/dustbin convention；不能脱离该 convention 宣称是正确的贝叶斯轨迹先验。未选 tracklet 隐含进入 dustbin；正效用 route 可参与选择，低边阈值仍会提前排除候选。

solver 最大化**提供给它的效用和**，不是 complete-track efficiency，不是正确径迹数，也不是 alignment information。精确优化错误效用，会稳定选出错误答案。Endpoint capacity 也不自动等于原始 hit capacity：不同 tracklet 若共享 cluster，还需要额外冲突定义。

## 2.4 Association 与 alignment 的耦合

misalignment 先影响局部 refit 的 state/covariance、传播 residual、候选可用性和绝对输入，再影响 edge/route score 与 packing。association 决定哪些 residual 被送入 alignment；错误匹配产生 outlier，而依几何/分数做选择会产生 selection bias。

当前 route-selected closure 在 identity/reference 上选定关联，再跨 nominal/FD/observed banks 对齐同一组 observation。这测试的是“给定这组选择，几何响应能否恢复”；没有充分测试“在未知 misalignment 上重新关联，再更新几何，再重新关联”的反馈环。

信息流应明确为：station pose → clusters/refitted states → propagated residual distribution → association-dependent observation selection → Jacobian/weight → relative pose estimate。标签不是 measurement；nominal 同事件 residual 也不是 deployment 可以观测的真值基线。

# 3. Scientific Audit

## 3.1 Alignment

### 15 DoF：有条件保留，不能由列数直接证明

四站每站五个活动坐标是 20 维；若确有五个共同 gauge directions，局部 quotient 是 15 维。这里被去掉的 chart 分量是 **dx、dy、rx、ry、rz，即两平移三旋转**，不是 README 所述“三平移两旋转”。dz 被固定或外部约束，不能同时把它当成已包含的自由平移。

更完整的论证应从四个 SE(3) pose 的 24 维与 6 维全局坐标自由度出发，再说明 longitudinal survey 怎样约束剩下的相对自由度。硬设每站 dz=0 的五维 chart 对有限 SE(3) composition 不闭合：旋转与平移组合能产生 dz。因此 15D 是需要定义参考框架与近似阶数的局部模型，不是任意 finite pose 的全局子群。

`alignment/four_station.py` 中对四个 payload 做共同左乘后，\(g_i^{-1}g_j\) 不变，这是正确的代数恒等式。然而 alignment 的观测是否也不变，还取决于磁场、材料、beam/survey reference 和坐标是否共同变换。固定实验室 B 场中只旋转 tracker，并不一般等价于坐标变换。坐标变换必须同时作用于场，例如 \(B'(x')=R B(R^{-1}(x'-t))\)，以及 surface、track state 和 covariance。

**证据边界：**历史 full-20 数值上 rank=20；共同 dx/dy 的微小响应比五个严格 null modes 更有直接证据。SVD 已做过，不等于“五个 gauge 已经被观测模型证明”。需要在相同物理约定下检验 \(JG\)，并分别报告固定外场和整体坐标变换两种实验。此处不是宣布 15D 错，而是拒绝无条件认证。

### Weak modes、尺度和 conditioning

窄角分布、磁弯曲与动量分布、站间 lever arm，以及有限 acceptance 都可能造成平移/倾角、曲率/偏移等近似退化；是否发生、方向是什么必须由 weighted design 的奇异向量和分层实验给出，不能只凭 station 编号解释。

`alignment/physical_jacobian.py::solve_physical_finite_difference` 对 **normal matrix** 做 SVD。已有 S0-fixed condition ≈24,781.94 因而在同一 full-rank scaling 下对应 whitened design 的 condition ≈157.4，而不是 24,781.94。它不是单独的“健康/不健康”判据；还需弱模 uncertainty、FD 稳定性和所需物理精度。当前 5 mm/60 mrad 的数值尺度与局部注入 envelope 不同，必须区分数值预条件、物理 prior 和误差单位。

`outputs/mc24_four_station_identifiability_pilot_v1/relative_closure_relative.json`：499 pairs、rank 15；S0-fixed response chi²≈0.0002169，S3-fixed≈58.1926；后者的历史最大 rz error 约 2.17 mrad。共同变换样本对应 chi²约 0.143 和 61.70。虽然 capture/agreement 标志为真，宽松的 rz tolerance=4 mrad 可以容纳这种误差。**通过 capture 不等于 reference-choice 下高精度等价。** 这些数值未归一成统计检验，不能据其大小宣称 p-value 或不确定度覆盖正确。

### Parameterization / sign / frame

`alignment/four_station.py` 使用平移与 `Rz Ry Rx` Euler composition；物理参数使用 mm/mrad，而 payload 旋转用 rad。GeoModel delta 是左作用：\(P_i=g_iN_i\)。必须分别报告 payload relative \(g_i^{-1}g_j\) 和 physical pose relative \(N_i^{-1}g_i^{-1}g_jN_j\)；后者含 nominal station separation 和旋转 lever arm，不能混称同一表。

reference choice 的正确比较是将同一物理解经一致群作用映射到另一坐标表示，再比较 physical relative transforms。简单删掉 S0 或 S3 五列并各自线性求解，只在相应局部近似中等价。Euler 分量相减也不是有限旋转的通用误差度量；新增基于 \(\log(T_{ij,\rm truth}^{-1}T_{ij,\rm fit})\) 的误差，明确其表达 frame 和单位。

Jacobian 的 `(positive-negative)/(plus-minus)` 与 `observed-nominal` 在当前线性回归中同号一致；但“估计注入量”与“对几何应用补偿量”符号不同，现有 response closure 不能代替修正后 refit 的 sign test。

### WLS 的能力与缺口

`solve_physical_finite_difference` 实际求解：

\[
d=r_{\rm observed}-r_{\rm nominal},\quad
N=\sum J^TC_{\rm nominal}^{-1}J,\quad
b=\sum J^TC_{\rm nominal}^{-1}d,\quad
\hat a=N^+b.
\]

它提供 FD 线性响应拟合、scaling、rank/pseudoinverse 和形式协方差。实现自己说明 nominal covariance 是同 clusters 两次 refit 差分的确定性权重，并不是两次独立 measurement difference 的 covariance。因此 \(N^{-1}\) 暂不能解释为校准过的 frequentist uncertainty。

核心缺口：

1. 同事件 nominal residual 被减去，在数据中没有同样可观测的无 misalignment counterfactual；结果主要验证 response bank 的一致性。
2. 共享 tracklet 的不同边相关。physical edge 去重正确，但不消除不同边之间的相关性。`models/field_route_fitter.py` 已注明 pairwise chi² 之和只是诊断，不是全局 likelihood。
3. 当前 closure 调用没有完成带 survey uncertainty 的联合求解：15D 删除 dz 列后写零，不等于应用了 5 mm dz prior。通用 solver 支持某些对角 prior，不代表这条执行路径实际传入了它。
4. 缺少已验证的 robust outlier model、迭代 relinearization/trust region、实际补偿后 refit、独立 stopping rule 和 ensemble coverage。
5. `primary_delta_t_capture_success` 不把 rank、conditioning、全部 Left-SE(3) 条件作为硬合取，存在摘要 PASS 大于配置要求的风险。

**如果给我 perfect truth association，当前 solver 是否足够可信作为 downstream oracle？答案：NO。** 可保留为局部物理响应 regression/软件 contract oracle；在独立几何更新与统计验证通过前，不能认证 association 的科学下游效果。

替代目标应是以每条真实径迹的共同状态（包括适当的 q/p nuisance）连接各站 measurement，正确处理 field/material transport、局部参数和全局 alignment 参数。消去每径迹 nuisance 的 Schur complement 是标准 global/local alignment 技术，不等于恢复历史已否决的 station-layer 模型。参见 [Millepede 官方说明](https://millepede.pages.desy.de/millepede-ii/) 和 [Blobel–Kleinwort 原始方法](https://arxiv.org/abs/hep-ex/0208021)。这是一项待实现、待验证的方案，而不是把当前 pair chi² 求和改名为 global fit。

## 3.2 Association：数学对象是否正确

物理目标是对允许缺站、fake/clone 和可能共享 hit 的 tracklets 建立径迹集合。给每条边一个二分类标签是可用辅助任务，但“边正确”不是最终随机变量，也不足以规定相互排斥的完整 route posterior。

若能严格把 route log posterior 分解为边项、长度/出生/消失先验与 dustbin 项，则可加 score 有依据。目前边网络已有重叠上下文、三个预测共享中间 tracklet，且使用 weighted/focal 分类目标。相加 edge logit 会重复计入证据和训练先验，不能自动解释为完整径迹 log odds。

有些完整链一致性无法被固定 pairwise potentials 表达，例如跨三段的共同 curvature、残差的相关方向，或依整体几何确定的 ambiguous crossing。现有 contextual edge network 部分缓解这一点，但把多个 route 的上下文压回共享边后，同一条边不能对每个 competing route 保留独立效用。因此 **可加输出是值得检验的限制，而不是已证明的 universal impossibility**。

### A–J 设计空间

| 思想 | 对 FASER 的适配与限制 | 本次取舍 |
|---|---|---|
| A. Pairwise edge classification | 简单、可校准、好定位；不能表达全链一致性及集合竞争；保留辅助监督 | 不作最终主目标 |
| B. Edge model + structured solver | 目前主系统，capacity 有明确物理含义；solver 不能补偿错误的因子分解/尺度 | 保留 baseline |
| C. Explicit route scoring | 四站、低占据度可显式枚举；可使用共同曲率、协方差与缺站 mask；需控制高占据组合数 | **主路线** |
| D. Set/track-level prediction | 直接输出径迹集合，目标贴近任务；no-object、重复径迹与可变径迹数训练需要更多真实支持 | 暂缓 |
| E. Hypergraph | route 本身就是连接 2–4 个节点的 hyperedge；当前 packing 已是 hypergraph optimization，无需先换品牌 | 显式 route energy 自然采用 |
| F. Bipartite/multi-partite matching | 相邻二分匹配易解，但局部最优不保全链；纯可加成本可用 min-cost flow，多边整体成本一般不再如此 | 统计/算法 baseline |
| G. Autoregressive track construction | 扩展到高占据度时可节省枚举；顺序偏置、early error、beam-search 截断和 STOP 校准增加复杂度 | 当前不优先 |
| H. Learned structured inference | 可训练 proposal/pruning 或近似求解；小事件已有 exact oracle，替换它损失可解释性 | 仅在测得规模瓶颈后研究 |
| I. Permutation-invariant set prediction | 输入 tracklet 顺序应不影响物理结果；DETR 式 matching loss 可处理输出集合，但不能直接保证 hit/endpoint capacity | 保留为远期竞争路线 |
| J. Differentiable assignment / OT | 二分 soft correspondence 有用；逐站 Sinkhorn 不等于四站整数可行集合，rounding 有 mismatch | 可作辅助松弛，不作主 solver |

FASER 典型占据度较低支持先做显式 route 和 exact solver，但多 muon/异常占据尾部必须实测，不能从“典型低占据”推出无组合风险。[FASER tracker 官方介绍](https://faser.web.cern.ch/tracker)。集合预测与 OT 的原始参照分别为 [DETR](https://arxiv.org/abs/2005.12872)、[SuperGlue](https://arxiv.org/abs/1911.11763)；不能把二分匹配结果直接推广成四站 assignment 的保证。

## 3.3 Structured inference

### 保留可行域，重建统一效用

packing 的 unit capacity 是正确且可解释的基本约束。候选允许 2/3/4 站 contiguous fragments 很有必要，但要统一分数定义、长度先验及未匹配节点成本。将所有分数称为 probability 会掩盖不同层面的 prior 和 clipping。

**REPRODUCED：零修正不保 solver identity。** W64 对每条边 probability 以 `1e-6` clip 后取 logit，再相加；V4/V5 先把完整 route logit 相加、sigmoid，再对整个 route probability clip 后取 logit。由此

\[
\operatorname{clip}(z_1+z_2+z_3)\ne
\operatorname{clip}(z_1)+\operatorname{clip}(z_2)+\operatorname{clip}(z_3).
\]

使用三个真实边的 logit 均为 8、unmatched=-1 的合成 chain：edge-only complete utility=20，三站 fragment utility=13；zero-delta 完整 route utility 变为约 9.81551，exact solver 改选 fragment。涉及 `training/route_aware_transformer.py::predict_route_aware_scores` 和 evaluation 使用的 `_route_hypotheses`。float32 sigmoid 的舍入饱和还会产生额外差异。

推荐 `raw_energy_v1` 类型化接口：baseline identity 使用实际 canonical edge logit 的和，再直接加 correction；完整和 fragment 都以 raw utility 送 solver，诊断概率单独输出。新 explicit model 可以采用不同 energy，但必须显式升版并重新验证 operating point。**反例证明 universal identity 声明错误；它没有证明全部 development 损失由此 bug 引起。** 需用已保存允许访问分数统计饱和发生率和 assignment 变化。

### Packing-aware 当前仍是单竞争 route margin

`training/gauge_consistent_route.py::packing_route_competition_loss` 取与 truth route 共享 endpoint 的单条最强竞争 hypothesis（及 dustbin）。假设 truth utility=10，两个互相兼容的 fake routes 各为 6，分别占 truth 的不同 endpoint。每个单竞争 margin=4，可能完全通过训练 margin；solver 却选 6+6=12。合成 exact-packing check 已复现。

需要真正的 global inclusion gap：

\[
M_r=\max_{Y:r\in Y}U(Y)-\max_{Y:r\notin Y}U(Y).
\]

在 ties/tolerance 明确的前提下，这才衡量完整 route 与最优可行竞争集合的关系。训练可以使用 exact loss-augmented inference 的 structured hinge，辅以稳定 pointwise supervision；不得把仅 complete4 的训练候选表和 deployment 的 2/3/4 表混用。历史 `training/structured_assignment.py` V3 的失败并未否决所有一致的 structured learning：它的 candidate/loss 设置与当前生产 fragments 不同。

训练真值集合也必须明确：一条物理 truth component 不应因为拆成两段就获得两份独立奖励。对缺站规定 maximal contiguous truth components，对 clone/等价表示规定允许的 target set；若未来加入 skip edges，单独版本化 topology，不能暗改 denominator。

## 3.4 ML formulation：V4/Head-Only 的独立裁决

### “Relative” 并没有引入所声称的物理不变性

`models/route_transformer.py::RouteAwareSparseTransformer.forward` 的 relative 表示为 `[h0,h1-h0,h2-h0,h3-h0]`。保留 h0 后，这是四个隐藏向量 concatenation 的可逆线性重参数化；后面的自由 Linear 层可以恢复 absolute 表示。并且 attention pooling 仍使用原始 h，edge features/base logits 也保留绝对信息。

因此该对照主要改变优化坐标、初始化和正则化，不构成“relative 几何信息”对“absolute 信息”的干净对照。一般非线性 absolute backbone 的 hidden difference 没有 SE(3) 意义。**V4 relative arm 失败不能否决物理相对表示；成功也不能证明 gauge invariance。**

### Head-Only 失败意味着什么

| 假设 | 当前证据与裁决 |
|---|---|
| Frozen backbone 丢失关键 route 信息 | train 能拟合不足以排除 OOD 信息瓶颈；需 raw-feature/trainable 对照。**UNKNOWN** |
| Frozen logits 数学上禁止任何 correction | 无界加法可表达任意有限补偿（若输入/网络足够）；但 fragment 不变、candidate 固定、bounded residual 有硬限制。不能泛化断言 |
| Additive complete-only correction 足够 | 不能共同学习 fragment/dustbin 尺度，且当前接口破坏 identity；作为长期接口 **REPLACE** |
| Relative 表示更物理 | 当前是可逆 latent 重参数化并保留绝对旁路；该命题没有被正确测试 |
| Loss 对齐 packing | 当前单竞争 margin 不能约束集合竞争；clipping 后极端分数梯度还可能消失 |
| 仅是 optimization instability | 训练期 output gradient 符号正确不能排除 shared parameter gradient conflict；缺跨 seed/早期动态证据 |
| Capacity 不足 | 没有公平容量/输入对照，不能下结论，更不能直接加深 Transformer |
| 正负样本/权重无关 | 大量复用 fake routes、weighted/focal loss 和 train/development support 差异仍是混杂因素 |
| Head-Only 值得继续作主线 | **STOP**；当前证据不足以支持再做一轮同类救火训练 |

`RelativeRouteV4Inference` 的冻结副本处于 eval/no_grad 是正确的；但 trainable copy 的 frozen backbone 在 `wrapper.train()` 后仍有 dropout 处于训练模式。本次检查发现其 8 个 dropout active。冻结梯度不等于固定 representation。可以把 dropout 当明确实验的正则化，但不能同时声称训练时两路 representation 完全确定一致。

weighted BCE 即便没有 focal，也使最优 logit 相对原始 posterior 增加与 class weight 有关的偏移；有 route sampling 后还涉及采样 prior。Focal loss 的原始概率解释也不严格 proper，因此不能把训练输出无条件当 calibrated posterior。[关于 focal probability estimation 的原始研究](https://arxiv.org/abs/2011.09172)。可执行选择是把输出称为 energy，并在独立 validation 上校准 operating costs；概率报告另做 reliability/log-loss 检验。

## 3.5 Dataset / domain shift

### 已有隔离与不能过度解释的部分

六源实际是四个 DSID、两个成组的 source families：family1 包含 `100043/100044` 的 `00200_00299` 与 `00300_00399`；family2 是 `100047/100048` 的 `00100_00149`。六个 source 文件不等于六种独立物理分布；是否代表不同生成过程仍需 provenance，不从 DSID 猜测物理原因。

`training/source_transfer_cv.py::validate_fold_sources` 检查 disjoint/union；当前生成的 folds 按 family 分开是正确的，但 validator 本身没有强制整个 family 不被拆开。V5A/W74只在head training上source-disjoint；其 frozen W64 backbone 和 normalizers 见过全部六源。因此它是 **head-transfer audit**，不是严格全模型 unseen-family generalization。

### Truth 使用与 deployment 输入条件

`datasets/synthetic_overlay.py::select_complete_truth_tracks` 先要求四站共同 truth、每站高 truth fraction，再按 truth fraction/chi²/ID 选唯一 segment。这是 truth-assisted 的干净 pool 构造。它不等同于模型直接使用 truth 特征，但把 clone、局部不纯 tracklet、重建失败等困难从输入分布中移走。

`datasets/synthetic_field_propagation.py::write_synthetic_field_candidate_root` 将已有 source propagation 扩展到同平面各 target。candidate recall=1 因而是对这一清洗/可传播池的条件结果。`enumerate_complete_route_candidates` 仍要求 truth ID 存在，候选构造与标签附加应分离；模型在数据上运行不应要求 MC truth branch。

外部依赖静态核查（不属于本仓库固定 SHA）：Calypso `NtupleDumperAlg.cxx::appendTrackletPropagationAudit` 的非 `allPairs` 路径使用 truth availability/matching，`run_physical_refit_capture_scan.py` 当前默认导出命令未开启 all-pairs。已有 opt-in truth-free 候选方向，但**真实未知关联导出是否覆盖所有 tracklets：UNKNOWN / NEEDS EXPERIMENT**。必须固定并记录外部源码/二进制版本，再在无 truth 输入模式测试。

### 重用、分布和混杂

完整 train corpus 有 7 payload×720=5040 overlay events；manifest 中只有约 299 个独立 physical event UIDs。跨 overlay/payload 复用使几千 truth routes 或几十万候选 routes 不能视为同数量独立观测。置信区间要按底层 event/source cluster，不能按 route IID bootstrap。

同一 overlay seed 不保证跨 payload 完全配对：truth pool/best segment/acceptance 可变，rejection sampling 消耗 RNG 顺序可变。station 3 shift 可能包括几何响应、运动学、协方差、source provenance、假 tracklet recipe 与 selection 差异。当前七个 payload 不能代表完整 15D deployment envelope；需要 source×event×payload 的分离设计。

真实 tracker local q/p seed、ACTS covariance/material policy 也属于输入分布。外部 SegmentFit 有固定 q/p seed，ACTS extrapolator 默认 scattering/eloss flags 不应被“完整无近似传播”措辞掩盖。历史 covariance mode scan 提供工程 tradeoff，不证明当前 uncertainty 是真实 likelihood；ACTS 本身提供的是可配置 tracking framework。[ACTS 原始论文](https://arxiv.org/abs/2106.13593)。不要直接改成 truth-q/p 或武断采用另一个 mode。


# 4. Failure Analysis

## 4.1 Candidate recall 100% 为什么仍损失完整径迹

这四个解释必须同时考虑：

1. **概率建模：**真实边存在不保证三个 score 的联合排序正确；三个共享 tracklet 的事件不是独立 Bernoulli trials。候选召回只给出该条件图上的上限。
2. **Calibration：**阈值 0.001 对应 logit≈−6.907，完整径迹却要求三个 logit 的和超过 4 才有正效用。所有边过阈值与完整 route 可获选之间存在很大空隙。class weighting/sampling/domain shift 均可改变 offset/scale。
3. **Structured inference：**即使 complete utility>0，它仍可能输给相容的 fragments/fakes 集合。当前 baseline 从三站 prefix 扩展到四站需要末边 logit>1 才增加效用，尚未计竞争、tie 和 correction。
4. **物理几何：**misalignment、field transport、q/p/covariance 与 track acceptance 改变 residual 的联合结构；true edge 可在宽 candidate gate 内但落到模型训练 support 之外。

因此，“candidate recall=100%”仅排除了**已选定 truth pool、已构建图、当前 payload 上的候选缺失**。它不能排除 raw→tracklet reconstruction、truth-filtered export、候选截断、missing stations、clone ambiguity 或新 deployment 分布的瓶颈。

## 4.2 C/D 的语义和反证

`training/route_operating_audit.py::classify_truth_chain_loss` 顺序判定 selected、A(candidate missing)、B(edge threshold rejected)、C(nonpositive/missing utility)、D(剩余未选)。在其合格 unique complete truth chain 定义及当前 operating mode 下，这是一套 mutually exclusive、基本 exhaustive 的**阶段记账**；不是 mutually exclusive 的因果机制模型。

D 是 catch-all，可包含 ranking、集合竞争、tie、route lookup、rounding、错误 utility 和 solver interface 问题。C 也可由 ranking 或错误几何表示造成。早期 B 曾描述为 ranking failure，实际执行 B 是 threshold failure；B=0 不能推出 ranking 无关。额外 E 类计数为零也不能证明 representation failure 被排除，因为它会被吸收到 C/D。

**ARTIFACT + 本次重新计数：**对 `outputs/mc24_four_station_blind_failure_localization_v1/blind_failure_rows.jsonl` 的 3378 条历史记录检查 `truth_rank_among_source_calibrated`：

| 最终阶段 | truth chains | 至少一条 true edge 非 source rank-1 |
|---|---:|---:|
| selected | 3008 | 41 |
| C | 200 | 49 |
| D | 170 | 80 |

所以至少 129 条 C/D truth routes 同时具有该定义下的 ranking deficiency。它不自动说明 rank-1 是充分/必要条件，却直接否决“阈值零失败意味着排序已排除”。应将 stage、rank、score support、全局 inclusion gap、domain/kinematics 做交叉表。

## 4.3 量化证据、文档 drift 和因果边界

`outputs/mc24_four_station_source_diversity_v1/blind_gate_decision.json` 给出的 reference 是：480 条可恢复 truth、466 条 selected complete、448 条 correct，efficiency=448/480≈0.93333、purity≈0.96137、all-route fake≈0.03828。draw00：480 truth、427 selected complete、392 correct，efficiency≈0.81667、purity≈0.91803、all-route fake≈0.07278。Workbook 65 表中的 selected-complete=480 与产物冲突；480 是 truth denominator，不能直接当预测数。

reference C14/D18，draw00 C49/D39；净增 C35+D21=56，对应效率差 56/480。62.5%/37.5% 只是**净计数分解**，不是对每条新增损失做 counterfactual 匹配后的因果贡献。严格共同 origin 的子集只有约 130 条，某些其他 payload truth 数为 489/484/485，直接相减还混入样本组成变化。

历史末站对 2→3 mean logit 从约 2.180 降到 1.828，解释了 marginal extension 偏好为何变脆弱；但具体物理起因 **UNKNOWN**。单独 hard-S3-ry 的效率约 0.929，接近 nominal 0.933，不能以 draw00 0.817 的下降证明“就是 S3 ry”。应比较同 origin、同 kinematics、同 quality 和共同可用 graph 的 paired residual/score，另报告非共同 selection population。

## 4.4 W71：head 在 development 上制造新 C

本次核对 `outputs/mc24_four_station_relative_route_v4_generalization_audit_v1/` 内 contract、decision、summaries 和 transition matrices，而非仅复制 workbook：

| population | Arm0 W64 C/D/selected | Arm1 absolute C/D/selected | Arm2 relative C/D/selected |
|---|---|---|---|
| train，9893 truth routes | 0 / 38 / 9855 | 0 / 33 / 9860 | 0 / 25 / 9868 |
| development，3378 truth routes | 200 / 170 / 3008 | 327 / 123 / 2928 | 441 / 135 / 2802 |

这里的 pooled efficiency 约 0.8905/0.8668/0.8295，不能与 README 单个 reference 的 0.933/0.917/0.888 混用。原始 200 条 C 在两 head 中全部仍是 C；absolute 新增 selected→C72、D→C55，relative 为153和88。train baseline 原本 C=0，因此 train 的成功不能证明模型学会了在真实低 utility support 上救 C。

train fake-route correction 的历史均值约 −131/−141，而 development truth 的最差 correction 可低到约 −53，q01 约 −33.9/−36.2；train 没有末边 logit<0 的 truth，development 有 50 条。这支持**support shift 下 head 把正确 route 当 fake 强烈压低**，但不能单独定位 feature insufficiency、sampling/weighting、shared-parameter gradient conflict 或过度自信。

最后一 epoch 部分 batch 的 `d loss / d correction` 符号正确，不能排除 `grad_theta` 上 fake/truth 的相互干扰、早期轨迹和 OOD 情况。把它写成“架构/imbalance/优化已经排除”超出了证据。

## 4.5 V5A：最新两fold均FAIL，不再是待评估

`configs/v5a_absolute_bounded_primary_source_transfer.yaml` 的 B=4、`delta=B*tanh(raw/B)` 是稳定性控制，不是domain-generalization原理。小于4的正utility仍可被推成C。

初始审查时只见四个30-epoch checkpoint；交付前新增evaluation已从 **`outputs/mc24_four_station_relative_route_v5a_source_transfer_v1/evaluation/source_transfer_summary.json`** 独立核对，产物绑定`e571a7e`、dirty=false：

| held-out family | arm | C | D | selected correct complete | efficiency | complete purity |
|---|---|---:|---:|---:|---:|---:|
| family1，6789 truth | W64 | 6 | 74 | 6709 | 0.98822 | 0.98228 |
| family1 | unbounded | 203 | 144 | 6442 | 0.94889 | 0.98517 |
| family1 | bounded | 237 | 151 | 6401 | 0.94285 | 0.98401 |
| family2，3248 truth | W64 | 0 | 43 | 3205 | 0.98676 | 0.97416 |
| family2 | unbounded | 12 | 57 | 3179 | 0.97876 | 0.97336 |
| family2 | bounded | 2 | 54 | 3192 | 0.98276 | 0.97139 |

family2→family1方向，bounded catastrophic=215而unbounded=184，new-C=231而197；反方向分别2而12。两fold gate均FAIL。安全bound通过但transfer失败，进一步支持停止本head-only主线；仍不能把“唯一原因必然是absolute node latent”当已证明。

`_gate_decision` 的purity/fake漏约束、missing stats默认值、empty-fold风险仍存在。它们不撤销此次已有的明确FAIL，也不提供重开该训练分支的理由。

最新single-pass优化减少重复solver调用值得保留，但引入/沿用了一个错误metric：`_metrics_from_routemetrics` 设置 `fake_selected_routes=selected_routes-correct_complete_routes`，会把正确2/3站fragments也算fake。真正的all-route fake分子在`evaluation/route_metrics.py::RouteMetrics.as_dict`是 `selected_routes-truth_consistent_routes`。因此W72表中的2943/3297等“fake”不能解释为假径迹数；必须重建正确counter再谈fake guardrail。complete efficiency/purity和C/D计数不受这个特定命名/分子错误影响。

## 4.6 W73：新representation证据有价值，但“唯一根因定位”不成立

本次核对 `outputs/mc24_four_station_route_representation_domain_audit_v1/decision.json`、W73以及`extract_route_representations.py`、`analyze_route_representation_domain.py`、`make_route_representation_decision.py`。R_phys包含33维raw pair features加3个W64 production logits；线性probe在family2→family1的truth FNR约0.00162，R6约0.03594。这是值得检验的输入简化线索，**不是已经完成的solver-level模型改进**。

必须纠正七项推断：

1. **Linear probe能力不是全部信息量。** R3/R4单独线性分类差，不证明节点状态没association信息；节点间几何一致性常需要差值的平方、乘积或共同trajectory约束。不能用低linear AUC证明这些输入有害。
2. **不是去掉一个变量的因果ablation。** R_phys与R5/R6在维度、非线性映射、正则化、logit通道和训练历史上不同；FNR接近catastrophic rate不证明同一cohort，更不证明唯一原因。
3. **未固定跨level同一抽样。** `cross_family_transfer`和`source_predictability`在level循环内连续消耗同一个RNG，train/test和balanced negatives每层不同。in-family split按route行随机，相关event/track/payload可跨边界。R0/R1是可逆标准化后又做StandardScaler，出现不同probe结果本身提醒抽样/数值不是完全matched。
4. **Parent已经见过两family。** R_phys/R2使用all-six W64 logits，不构成全模型未见domain的证据。固定fold1 head对两family抽R5/R6消除了“不同extractor”混杂，但其训练只见family2，两个transfer方向的extractor ancestry仍不对称。
5. **阈值不对齐。** 1:1:1:1抽样、balanced logistic、threshold0.5的FNR不是route utility≤0或被exact packing选中的概率。缺少相同false-positive operating point、global margins及solver-level对照；仅降低FNR可能增加大量fake。
6. **小SMD/MMD/NN差不排除tail/support shift。** 没有独立event bootstrap、null/permutation分布、tail conditional calibration；高维距离和kernel bandwidth不能跨表示直接比较。全分布median相近不能排除造成约3%失败的尾部。
7. **decision不是自动验证过的decision tree。** `make_route_representation_decision.py::main`把`decision_case='case_1'`、唯一根因叙述及215写死；输入数据若反转也会输出同一architecture选择。`catastrophic_cohort`用raw `L_edge>4,L_corrected<=4`得到249，既非exact selected→C cohort，也未通过逐route/key证明与215一致。量级相近不能作为semantic-bug已排除的验证。

此外，extractor未保存完整source/origin key，非整数payload ID会被写成−1；这会妨碍后续严格event/payload分层。R0–R7的数字层级不全是同一条纯前馈信息链，部分来自不同branch/hook；“最早泄露层”只能描述这些观测，不能当网络因果路径。

**W73裁决：MODIFY。** 保留R_phys候选，新增logits-only低维control、同一event/sampling/group split、相同FPR比较和exact solver replay；停止“去掉absolute node latent已消除根因”的叙事。已用这些families选择架构后，它们是已见研究benchmark，不能再称全新blind domain。

## 4.7 W74：输入删旁路属实；仍是complete-only frozen residual

`models/route_transformer.py`新增 `route_representation_mode='physical_pair_relative'`。代码确实不构造route query、node key、edge projection和pair embedding，直接拼接三个**标准化**pair feature和三个frozen production logits；head参数从172,513降至21,377。它减少了absolute latent的**直接**旁路，但production logits仍来自含absolute节点的W64，不能称整体gauge/source independent。标准化R1与W73 raw R0在有限方差下信息等价，优化/正则化并不相同。

模型仍仅修正complete4 route、冻结fragment与all-six parent、B=4，并沿用sigmoid/clip和单竞争loss。因此它是有意义的新输入ablation，**不是本review推荐的多长度统一route-energy系统**。参数量、随机数消耗和隐式正则化也同时改变，不能把全部收益唯一归因于“移除absolute信息”。standalone forward允许production logits缺失时退回base logits，必须显式区分该fallback与wrapper生产语义。

新增 `scripts/zero_init_replay_physical_pair_relative.py` 默认每payload三events，检查truth selected/class和utility容差；没有比较全部fake/fragment assignment，注释“两个转换只差约1e-5”不具有普适性。第3节logit=8反例依然适用，新toy测试也没有覆盖它。

新evaluation的 `_gate_c_fold` 已加入purity/fake guardrails，这是改进；但继承上述错误fake分子。Gate A要求两fold都严格小于control，若control某项为0，则primary即使同为0也永不通过；这是应预注册处理的floor case。Gate B从W72的零新增C改为≤1%，属于**新协议阈值**，不能用于翻案W72；且新fold_pass没有合取所有bounded/finiteness诊断。

截至交付检查，指定W74四个`checkpoint_freeze.json`及`evaluation/source_transfer_summary.json`均不存在；**CODE IMPLEMENTED / SCIENTIFIC RESULT UNKNOWN**，不据此推断是否有排队/运行中的作业。新增18个CPU测试通过，GPU测试跳过；不是生产transfer PASS。可复用该36维输入作为WP4的R/L controls，修完P0后再做有限对照，不自动继续Head-Only主线。

# 5. Code Audit

以下按对科学结论的影响排序；FIX 均为后续任务，本次只添加审查文档。

| 级别 | File → function/class/config | Problem → consequence | Recommended fix |
|---|---|---|---|
| P0 | `scripts/evaluate_relative_route_v5a_source_transfer.py::_metrics_from_routemetrics` → W74 `_gate_c_fold` | selected−correct-complete把正确fragments算fake，污染guardrail | 用RouteMetrics真伪counter分别输出complete/all-route/fragmentation，补纯真fragment fixture |
| P0 | `scripts/analyze_route_representation_domain.py::cross_family_transfer` / `source_predictability` | route-row split、逐level不同抽样；linear probe被解释为信息/因果结论 | physical-origin group split、固定样本索引、logits-only与同FPR对照、cluster CI |
| P0 | `scripts/make_route_representation_decision.py::main` | case_1与根因文字hardcoded，不随输入证据改变 | 用冻结数值规则产出case/UNKNOWN，并测试反向/缺失证据 |
| P1 | `scripts/extract_route_representations.py::extract_family` | 非整数payload变−1、缺完整origin key、同路径覆盖extract | 保存字符串payload和source/origin全key、immutable版本，禁止覆盖历史extract |
| P1 | `scripts/zero_init_replay_physical_pair_relative.py::main` | tiny truth-only比较不足以证明全部assignment identity | 加饱和scores、全部fake/fragment set和randomized exact solver tests |
| P1 | W74 `RouteAwareSparseTransformer.forward` / eval gates | fallback base/production语义不同，strict-improvement零floor、fake错误与bounds缺合取 | typed显式logit来源；新gate版本处理floor与完整safety；仍不复活W72 |
| P0 | `training/route_aware_transformer.py::predict_route_aware_scores` → `baselines/route_assignment.py::_route_hypotheses` | complete sigmoid/clip 与 edge-wise clip 不交换；zero-init 改 assignment | raw utility contract；保留 legacy decoder；饱和 synthetic chain regression |
| P0 | `training/gauge_consistent_route.py::packing_route_competition_loss` | 单 rival 不等于 feasible rival set；正 margin 仍输 global packing | exact forced-in/out gap 和 all-topology loss-augmented inference |
| P0 | `scripts/evaluate_relative_route_v5a_source_transfer.py::_gate_decision` | purity/fake 未进入 PASS，空/missing folds 不 fail closed | 冻结 config 中定义 schema、guardrails、required-fold 集合；失败路径测试 |
| P0 | `alignment/physical_jacobian.py::solve_physical_finite_difference` | 同-event baseline difference、nominal covariance、边独立假设；形式 covariance 非认证 uncertainty | 明确 response-only API；独立 common-track likelihood/coverage 实现，不静默改旧结果 |
| P0 | `scripts/run_four_station_relative_closure.py::close_observed_point` → `_fit` | capture PASS 不包含全部 admission 条件；dz prior 未在该 15D 路径落实 | gate 合取、分开 fixed-dz/prior 模式、记录所有拒绝原因 |
| P0 | `scripts/run_four_station_route_selected_relative_closure.py` → identity selection / bank intersection | 固定选择和 survivor intersection 被解释成闭环重关联 | 命名 conditional response；增加当前几何 selection→update→refit 和 survivor 统计 |
| P0 | `models/route_transformer.py::RouteAwareSparseTransformer.forward` | relative 隐藏向量可逆重参数化、absolute pooling 旁路保留 | 停用 invariant 声明；显式 transport-relative raw route schema/对照 |
| P0 | `models/route_transformer.py::RelativeRouteV4Inference` | trainable copy 的 frozen backbone dropout 仍 active | train() override 固定该 backbone.eval；或显式声明 dropout 实验 |
| P0 | `training/route_operating_audit.py::classify_truth_chain_loss` / `audit_event_truth_chains` | D catch-all 被因果化；query lookup 在 missing route 情况可能失败，composition 假定 replace | 阶段与诊断分离；lookup-safe；直接调用 canonical decoder，测试 residual/replace/threshold |
| P0 | `training/route_aware_transformer.py::enumerate_complete_route_candidates` / graph loading | inference enumeration 要求 truth，候选与 labels 耦合 | truth-free enumeration；labels 可选后附加；truth 删除/置换不变性测试 |
| P0 | `scripts/run_physical_refit_capture_scan.py` → export command | 默认未启用 all-pairs；外部 exporter 有 truth selection | 固定 Calypso revision，导出全体 source truth-free propagation，真值只附标签 |
| P0 | `training/source_transfer_cv.py::validate_fold_sources` | 只检验 source set，未强制 family 或 parent ancestry | family closure、normalizer/backbone provenance validator，明确 head-only 限定 |
| P1 | `training/structured_assignment.py` → candidate/objective | complete-only 训练与 production fragments 不同 | 共用 2/3/4 builder、canonical target components 和 exact solver |
| P1 | `datasets/synthetic_overlay.py::select_complete_truth_tracks` | truth quality 清洗和唯一 segment 选择降低真实困难 | 保留 clean benchmark，另建 raw/clone-aware 层，报告选择效率 |
| P1 | `datasets/synthetic_field_propagation.py::write_synthetic_field_candidate_root` | fan-out 不能补回缺 source propagation | 分解 source coverage、target fan-out coverage、candidate recall |
| P1 | `alignment/route_selected_update.py::apply_observation_statistics` / `align_route_selected_observations` | 去重不消除共享 tracklet covariance，intersection 有偏 | correlation blocks、lost-observation ledger、cross-fit validation |
| P1 | `alignment/four_station.py` → relative tables / errors | payload/composed relative 容易混用，Euler subtraction 非有限旋转标准误差 | typed frame/unit records、Lie-log errors、非零 baseline pose tests |
| P1 | `baselines/field_chi2_matching.py::_valid_covariance` → `geometry/propagation.py::mahalanobis_chi2` | 正对角/对称不保证 SPD，solve 可产生负 chi² | Cholesky gate、明确 jitter policy、indefinite fixture，统一上下游行为 |
| P1 | `scripts/train_relative_route_v4_head_only.py` / V5A worker config assembly | 部分参数依赖默认值/hardcode，epoch/OP 元数据可能与运行值分离 | resolved config single source of truth，实际值/hash 入 checkpoint，拒绝未知键 |
| P1 | `scripts/run_relative_route_v5a_source_transfer_condor.sh` / submitter | unsets `CUDA_VISIBLE_DEVICES`，共享可变 repo/环境 | 保留 GPU allocation，固定源码快照/环境，记录实际 GPU/driver |
| P2 | training checkpoint save/load 路径 | freeze/hash 有价值，缺 optimizer/RNG/scheduler 完整 restart contract | 区分 restart 与 immutable selection checkpoint，测试中断续训 |
| P2 | `pyproject.toml` / `scripts/setup_environment.sh` | Torch/SciPy/test 依赖与环境未被完整声明 | ML/test extras + pinned lock/container/LCG manifest，声明 Python 支持 |
| P2 | `baselines/route_assignment.py` → hypothesis generation | positive-hypothesis 上限不必限制此前 Cartesian enumeration | component enumeration、runtime/内存记录和 safe pruning，禁止 truth-based truncation |
| P2 | `tests/test_calibration_modes.py::test_registered_disk_contract_matches_workbook_45` | suite 依赖未提供的历史 output | tracked 小型 fixture；完整 artifact 检查标 integration 并列 prerequisites |

## Architecture 与文档 drift

script 层同时承担 config、训练、decoder、gate 和历史分析。多个评估入口重复 score composition，使 mask/clip/threshold 修复无法自动传播。先抽出 `CandidateTable`、`RouteEnergyTable`、`AssignmentResult`、`EvaluationContract` 四个共享 contract，历史脚本通过 versioned adapters 调用，不整体重写所有历史代码。

需纠正并保留历史时间语境：

- README 的 15D gauge 文字错误；四站 recovery 的 `4.9e-13 mm` 来自早期单 S3 +1 mm、27 pair 结果，不代表当前四站多旋转精度。
- “5 mm dz survey prior”与实际 fixed-zero 路径不同；`physical_geometry_repropagation=True` 不能证明应用解后重新 refit。
- W65 selected-complete denominator 和概率相加式与 artifact/code 不符，ranking 与 threshold 混淆。
- W71 对架构、梯度和 imbalance 的排除性判断应降级为有限观察。
- W72最新已是两个fold FAIL；六源CV仍仅head层隔离。W73的唯一根因/自动decision-tree声明超出代码和probe证据，W74尚无冻结训练/eval结果。
- 本次环境 Python 3.13.11、NumPy 2.4.4、Torch 2.11.0 / LCG 110 CUDA EL9 gcc13；不能用 workbook 的另一平台标签代替环境记录。

# 6. Experiment & Evaluation Audit

## 6.1 Leakage / independence

| 风险 | 已观察事实 | 判定与要求 |
|---|---|---|
| Direct truth-derived feature leakage | 特征清单未直接含 truth ID；pool/export/enumerator 依赖 truth | 不指控标签直接入网络；证明删除 truth 后 inference 可执行且不变 |
| Source leakage | loader 检查 source ownership；V5A parent/normalizer 见过全部六源 | V5A 是 head transfer；全模型 transfer 使用 fold-specific parent/preprocessing |
| Event / track leakage | overlay/payload 重用底层 event，UID namespaced by source | 不能排除换名重复文件；需 GUID/content fingerprint，按 origin split |
| Geometry leakage | 少量 payload/twins，source/payload holdout 未正交 | 新 geometry seed/方向留出，known-payload interpolation 分开报告 |
| Train–validation mismatch | clean pool/fake recipe 与未知关联输入不同 | controlled/realistic 两层，报告 acceptance 与 quality selection |
| Checkpoint-selection leakage | freeze/hash 有记录，反复 development 影响架构选择 | 已看 development 永远是 development；architecture/OP 选择也算使用 |
| Repeated evaluation contamination | reserved-blind 已被多轮诊断 | 禁止恢复 blind 名称；新模型选择仅用允许训练池重新设计 splits |
| Final-blind / sealed leakage | 本次未访问，不能仅靠文档证明历史无人访问 | registry/custodian 审计，不为本次检查打开内容或扫描 forbidden assets |

manifest loader 在允许 split 过滤后才 resolve 资产路径是有用保护。需增强底层身份、模型祖先和历次使用记录。历史是否曾暴露与现在“永久关闭”是不同命题；本次不访问 sealed 产物求证，也不建议重开。

## 6.2 四级指标体系

所有指标附分子/分母、mask、candidate/threshold/solver版本、micro/macro 聚合方式。先逐 source/payload/multiplicity 报告，再给 pooled；uncertainty 按底层 physical-event cluster，event bootstrap 不能证明跨 source 外推。

### Primary metrics

| 指标 | 定义/用途 |
|---|---|
| Complete-track efficiency | correct complete unique tracks / 有四站可用 tracklets 的合格 truth；另报相对原始可接受 truth 的 end-to-end efficiency |
| Fake complete-track rate / purity | fake complete / selected complete；purity=correct complete / selected complete；明确 ambiguous truth 处理，不用 all-route fake 替代 |
| Worst-payload performance | 冻结 envelope 中最差 efficiency、相对 nominal 降幅；限制 nominal 绝对值，防止压低 nominal 伪造稳定性 |
| Assignment success | 与允许 truth-equivalent target set 一致的 event 比例；分 multiplicity/缺站/clone |

### Secondary metrics

- Fragmentation：每条 truth 的 selected components 数、complete→fragment 概率、unmatched truth nodes。
- All-route purity/fake、route length spectrum、station-pair efficiency、edge fake rate。
- 每事件 correct/missed/fake tracks；runtime、candidate count、solver optimality status 和内存尾部。
- Candidate recall 在 raw eligible truth、source-propagation available、clean overlay 三层分别报告。

### Diagnostic metrics

- Edge ROC/PR/BCE/AUC 辅助；source-conditional rank、threshold efficiency 和 ranking deficiency 交叉表。
- True/fake route energy、length offsets、global forced-in/out margin、best-set gap、ties、complete-vs-fragment extension gap。
- 若声明 probability：Brier、log-loss、可靠性曲线、预定义分箱 ECE，注明抽样 prior。Energy 只报告排序/尺度稳定性。
- C/D 与 saturation、domain support、kinematics、geometry、selection survivor cross-tabs；配对与非配对分开。
- Truth-only/physics-only/ML 的 candidate/assignment oracle 上限，以及完整 route 的非局部信息增益。

### Alignment-level metrics

- 15D chart recovery 与 physical relative-transform Lie-log error，明确 frame/mm/mrad。
- Bias、RMS、weak-mode projection、scaled whitened-J singular spectrum、FD step stability。
- 应用解并物理 refit 后的独立 residual closure、robust rejection、selection bias。
- Interval coverage、pull mean/width、协方差相关项；truth 与 ML 配对下结果之差。
- versus misalignment 大小/方向、momentum/charge/angles、occupancy/fake rate、survey uncertainty 的 response curves。

## 6.3 Tests：实现证明与科学证明

交付快照本次 `463 passed`（另1 skipped、1 failed）支持若干局部契约：toy packing/brute-force、站顺序、zero terminal initialization、冻结梯度、checkpoint roundtrip、bounded 范围和代数 relative transforms。不能证明固定 field 中的观测 gauge invariance、source independence、calibrated posterior、完整 closure 或 OOD 信息充分性。

唯一 failure：`tests/test_calibration_modes.py::test_registered_disk_contract_matches_workbook_45` 缺少 `outputs/mc24_ift_calibration_mode_validity_contract_v1/mode_validity_contract.json`。这是 non-hermetic artifact dependency，不能冒称全部测试通过，也不能据此判定四站科学算法错误。

| 类型 | 必须补的验证 |
|---|---|
| Implementation | raw/legacy utility、饱和±logit、2/3/4 lengths、zero-init **assignment** identity |
| Implementation | 两兼容 rivals 反例、forced-in/out 与 brute-force 全子集、ties 的等价最优集合 |
| Implementation | node/route permutation 后 assignment equivariance，station ID 与输入顺序解耦；物理站互换不是所需对称性 |
| Implementation | truth missing/permuted 不改变 features/candidates/scores，labels 只影响监督 |
| Implementation | masks、空 graph/truth/fold、NaN/Inf、indefinite covariance、missing key、bound/threshold fail closed |
| Implementation | frozen dropout policy、gradient routing、checkpoint parity、resume RNG/optimizer、拒绝 unknown config keys |
| Scientific | left/right 更新、非零 nominal pose、单位/sign，群变换同时作用于 field/priors |
| Scientific | fixed-field JG、FD h/2/h/2h、独立 source/noise、更新后真实 refit |
| Scientific | all-topology target set、shared-hit exclusion、source/event/payload independence、covariance coverage |

GPU seed/cuDNN deterministic 不等于跨硬件位级复现，尤其 sparse scatter；记录版本、硬件和容差，在固定平台验证指标与 selection contract。


# 7. Competing Future Directions

以下是有不同科学假设的路线，不是同一模型的超参数列表。解决 C/D 的机会为基于当前证据的**定性判断**，不是统计预测概率。

| 项目 | A：有界 frozen-head | B：现有 edge+route 联合微调 | C：显式物理 route energy | D：直接 set/hypergraph network | E：统计共同径迹模型 |
|---|---|---|---|---|---|
| Scientific rationale | 限制 OOD correction 破坏 | 让表征适应 route/packing 目标 | 对完整及缺站径迹直接建模联合证据 | 学习集合和候选间高阶竞争 | 用共同 trajectory nuisance 解释测量 |
| Expected benefit | 降低极端负修正 | 修复 frozen 信息瓶颈 | 同时修复因子分解与 length objective mismatch | 全事件相互作用，可自适应径迹数 | 可解释、数据效率高、与 alignment 一致 |
| Main risk | 保留片段尺度、输入与 support 缺口 | 过拟合绝对坐标/source，继承 edge瓶颈 | 枚举规模、协方差/动量输入不足 | 数据不足、训练/离散约束复杂 | field/material/common-state contract 尚缺 |
| Engineering cost | 低，但需先修通用 contract | 中 | 中；复用图/solver，新增共享多长度 scorer | 高 | 中到高，主要在物理 export/fit |
| Required new data | 不新增；仅已有允许 folds | 训练池内新 payload 与严格 source folds | 同 B，另有未清洗 tracklet 层 | 更广 multiplicity/clone/missing 覆盖 | common-state/transport/covariance export |
| Required training | V5A诊断已FAIL；W74仅输入对照 | fold-specific parent+联合训练 | 低容量 raw scorer + trainable 表示对照 | 新模型、matching/structured loss | 无大网络；统计参数/少量 residual 可选 |
| Interpretability | 中低，latent additive correction | 中低 | 中高，route residual/energy 可分解检查 | 低到中 | 高，但依赖物理模型正确 |
| Alignment compatibility | 间接，固定 association selection | 间接 | 明确 route→common observation 接口 | 可做但选择偏差更复杂 | 最直接 |
| Chance of solving C/D | 低到中；C减少可能换成fake/D | 中，取决于表征 | **中高，最值得首先检验** | 不确定，暂无资源优势证据 | 中高，若物理信息足够 |
| Kill criterion | 修复 gate 后仍无 transfer 或牺牲 purity；即便通过也不自动主线化 | 公平对照不优于 raw scorer/统计baseline | 预注册 held-out 两fold无增益且CI足够窄；或真实输入不可得 | 小预算不能优于 exact-route baseline，或约束不满足 | truth-only likelihood/coverage 不通过且缺口不可修复 |
| 排序 | 5，诊断已完成且FAIL | 3，作为表示对照 | **1，主推荐** | 4，延期 | **2，备选推荐** |

B 与 C 的区别：B 保留主要由 edge/correction 定义的模型；C 以统一 2/3/4 route energy 为输出对象，edge classification 只作辅助任务。C 仍可复用并训练当前 backbone，但不要求必须采用它。D 的新 network 与“routes 作为 hyperedges 的现有 packing”也不是同一件事。

# 8. Recommended Research Direction

## Primary recommendation

**显式 physics-informed multi-length route energy + trainable shared representation + exact packing。** 先用三种输入/表征对照决定信息来自哪里，再在获胜表示上完成结构化学习。保留 exact solver 作为清晰的离散优化层。

可执行的模型契约：

1. 所有 2/3/4 站 contiguous candidates 由一个 truth-free builder 构造。输入顺序可变，但 station ID/物理 z 有显式语义；missing mask 独立编码。shared-hit conflicts 若存在必须进入可行域。
2. raw route 输入包括 states、实际可用的 propagation residual、完整 pair covariance 或合法 whitened residual、质量与站间几何；使用同一参考 transport 构造相对量。**不得在没有 common-state covariance 时把 pairwise whitened residual 拼接说成联合白化。** 场位置/方向是物理条件，应保留必要信息而非强行删除全部绝对坐标。
3. 一个共享 scorer 输出各长度的 raw energy；显式长度/未匹配成本只出现一次。候选 route energy 不经过 sigmoid 回转。edge BCE 为辅助，route pointwise 项稳定分数，exact set loss 对齐真正的竞争集合。
4. 先做两层、宽度128的低容量 route scorer；隐藏 pooling 使用 mask。复杂 backbone 仅在 matched held-out 实验显示收益后保留，不以现有代码体量作为保留理由。
5. 不把名为 Relative 的 latent subtraction 当物理 invariant。坐标变换测试必须包含 field/pose 条件；真实 misalignment 是待估计信号，不能被错误的“invariance”消掉。

## Backup recommendation

**共同 field-aware trajectory fit + calibrated statistical route energy，必要时只学习物理残差。** 当 raw route 和 trainable 模型不能在严格 transfer 上胜过统计 baseline，优先补 q/p/transport/covariance 信息，而不是改成更大的 set network。此方案也为最终 alignment 提供统一的 likelihood；当前 `field_route_fitter` 只能作为开发起点，不能冒充已完成的联合拟合。

# 9. Go / Modify / Replace / Stop Table

| 主要假设 | 决策 | 理由与继续条件 |
|---|---|---|
| 15D relative alignment | **MODIFY** | 保留局部 relative-subspace 思想，但必须证明观测模型的 gauge 与 longitudinal prior。固定站列删除、代数恒等式和 rank15 不能独立完成认证。先补 fixed-field/共同坐标变换、尺度/弱模及真实更新测试。 |
| Current propagation/candidate builder | **MODIFY** | 物理传播与宽候选图有价值。100% recall 仅适用于目前 truth-clean 条件池，truth-free export、covariance 和占据尾部尚未认证。保留实现骨架，替换标签耦合输入契约。 |
| Current backbone | **MODIFY** | 训练拟合不证明 OOD 信息充分性，也未证明它已经必然成为性能上限。允许在 matched probe 中冻结/训练，并与 raw physics features 对照。未取得独立收益前不扩大容量。 |
| Edge-only scoring | **REPLACE** | 边分类适合辅助，但最终可加 utility 缺乏验证过的联合概率解释。重叠上下文与长度先验会重复计证据。主输出改为多长度 route energy，保留 baseline 作对照。 |
| Route-level correction | **REPLACE** | 完整 route correction 可作诊断，但与 frozen fragments 的分离尺度和 clipping 不适合作长期接口。替换为同一 contract 下的完整/片段 energy。任何新 correction baseline 仍须证明 zero-init identity。 |
| Frozen-backbone Head-Only | **STOP** | development 明确恶化，现有排除性诊断不足。继续 loss/bound 调参不能回答信息瓶颈。已有V5A两fold已FAIL；结束该主线。W74仅保留输入对照价值，停止自动full-six-source升级。 |
| Solver-aware margin | **REPLACE** | Dustbin 与单 rival margin 有诊断意义，但不是 set-packing margin。以 exact feasible-set loss/inclusion gap 替代主训练约束，保证所有长度与 inference 一致。 |
| Set-packing formulation | **KEEP** | Unit-capacity 可行域符合基本 tracklet exclusivity，exact solver 是研究优势。效用、shared-hit conflict 和 target-set 定义需扩展，不需要先换 learned solver。 |
| Current data curriculum | **REPLACE** | 保留为历史 controlled benchmark，但不足以代表15D/source/real-tracklet分布。新主实验采用 origin-disjoint、family-disjoint、payload-held-out crossed design，truth清洗层与真实输入层分开。 |
| Current blind-evaluation protocol | **MODIFY** | 继续严格保护 final-blind/sealed，保留冻结/allowlist原则。所有已读 reserved-blind 永久标 development；补 ancestry/usage ledger 和 fail-closed gates。不得因模型失败变更 blind 标准。 |

# 10. Prioritized Engineering Roadmap

所有新增文件名和命令在第11节明确标记 **NEW / 尚未实现**；不是声称仓库现在已有这些入口。预计为数周工作，排期是工作量参考，不是实验成功承诺。

| 优先级 | 工作包 | 顺序与产出 | 不允许跳过的条件 |
|---|---|---|---|
| **P0：研究前提** | WP0 snapshot；WP1 utility/solver/gates；WP2 alignment资格设计；WP3 data契约 | 第1周，建立可解释可复現输入/输出；WP2先完成资格裁决 | 不得在 utility、split、truth-free 输入未明确前正式比较模型；alignment未认证不得称downstream oracle |
| **P1：决定主路线** | WP4 matched representation experiment | 第2周，有限 pilot 后只做预注册replication，产生唯一模型方向裁决 | 不用旧development选checkpoint/OP；不消费final-blind |
| **P2：获胜路线工程化** | WP5 winning route model；WP6 reproducibility完善 | 第3周起，固定接口、训练/评估共用逻辑、规模测试 | WP6最小Condor快照/恢复功能须在WP4前具备；余项在P2完成 |
| **P3：科学闭环** | WP7 independent alignment closure | WP2通过方法资格且WP5模型冻结后，truth与ML关联并行验证 | 必须实际更新geometry/refit、独立validation与coverage，不以response-bank代替 |
| **P4：一次最终检验** | WP8 final-blind preregistration/evaluation | 所有门槛通过后才进入 | sealed永久关闭；本次roadmap执行前期不打开final-blind |

依赖图：

```text
WP0 → WP1 ───────────────┐
  ├→ WP2 qualification ─┼──────────────→ WP7 → WP8
  └→ WP3 → WP6(minimal) → WP4 → WP5 ──┘
                       WP6(full) ─────┘
```

WP2 的结果可以是“alignment oracle 未认证”；这不阻止明确标为 association-only 的 WP4，但会阻断 alignment 效用和 final-blind 的科学成功声明。WP2若发现输入无法满足真实 association，WP3/WP4也必须等待相应输入修复。

# 11. Work Packages

## 公共执行与产物约定

- 本节全部未来工作为 **PROPOSED**。仅测试/static/tiny smoke 可交互；正式训练、多 epoch、corpus生产、批量评估、ablation、参数扫描和 physical refit 一律 HTCondor。
- 公共输出：`outputs/research_review_v1/<wp>/<run_id>/`，run_id由源码SHA、resolved-config SHA、split-manifest SHA、seed构成；同 run_id 禁止覆盖。日志在 `logs/`，训练恢复在 `restart/`，冻结模型在 `frozen/`，evaluation与gate分目录。
- 新建 `scripts/submit_research_review_jobs.py` 和 `scripts/run_research_review_job.sh`，以及 `configs/research_review/`。submitter先生成不可变源码/配置快照和 DAG，再提交；代码入口不存在时仅可 dry-run，拒绝伪提交成功。
- `run_metadata.json` 必含源码与dirty diff hash、外部Calypso/ACTS/field/material/geometry版本、完整命令、resolved config、允许数据manifest与底层GUID/hash、模型祖先/normalizer hash、Python/package/LCG或container、CPU/GPU/CUDA/driver、seed、Condor Cluster/Proc、启动/退出、输出hash。checkpoint 保存 optimizer/scheduler/RNG/epoch/step用于恢复。
- 保留 Condor 的 `CUDA_VISIBLE_DEVICES`；CPU job默认4核/8GB，GPU初始1卡/4核/16GB host memory，资源经tiny memory probe确认，不通过取消资源隔离解决。stdout/stderr/eventlog分别保存；resource failure重试不改变科学config。
- 成功标志需要有效schema、非空required splits、产物hash和退出码；`done.json` 不等于 `gate_pass.json`。DAG依赖看两者的相应职责。
- 用 `20260905` 作为首个训练seed，重复seed为 `20260906,20260907`；split seed=`1729`，训练payload seed=`271828`，held-out payload seed=`314159`。冻结前生成具体payload表并检查不重合，不能仅写seed而不保存表。

## WP0 — Reproduce and freeze the reviewable state（P0）

- **Objective：**建立可追溯基线，区分代码、已执行产物和未完成计划。
- **Scientific hypothesis：**当前的关键对比可从确定的版本/分数重建；若不能，后续收益无法归因。
- **Inputs：**上述本地SHA、两份README、W64–72文档、允许train/development审计产物、四个V5A checkpoint和最新W72/73 train-side summaries；不对整个outputs做递归搜索。
- **Files/modules to inspect or modify：**`pyproject.toml`、`scripts/setup_environment.sh`、`tests/test_calibration_modes.py`、相关freeze/summary scripts；README更正需引用历史范围。
- **New code required：**NEW `scripts/audit_research_review_state.py`；artifact allowlist/usage ledger schema；tracked calibration-mode小fixture和独立integration marker。
- **Tests required：**缺产物、hash mismatch、未知schema、forbidden path均fail closed；hermetic suite不读外部历史output。
- **Experiment required：**无训练；复核现有score/summary的分母与版本，记录V5A两fold FAIL及W74仅代码状态；不把W73 probe当生产结果。
- **Execution command：**现有：`source scripts/setup_environment.sh ml`，随后 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 pytest -q`。NEW入口实现后：`python scripts/audit_research_review_state.py --config configs/research_review/wp0_state.yaml`。
- **Metrics：**artifact可追溯率、test结果、schema/hash不一致清单。
- **Acceptance criteria：**hermetic tests全通过；外部integration明确独立且缺失时不能记PASS；每项主结论有source/code/artifact路径和SHA。
- **Kill criteria / failure handling：**无法定位baseline checkpoint或decoder版本则暂停模型效果比较；补provenance而不是重新训练覆盖历史。
- **Dependency：**无。
- **Expected artifact：**`wp0/<run_id>/evidence_ledger.json`、`environment.json`、`tests.txt`、`baseline_status.md`。

## WP1 — Canonical energy and exact competition contract（P0）

- **Objective：**消除score接口伪效应，建立真实集合竞争诊断，修复V5A gate。
- **Scientific hypothesis：**训练/评估必须对相同候选最大化同一效用；单竞争margin不是充分条件。
- **Inputs：**W64 edge scores、V4/V5模型契约和纯合成route fixtures；历史development只允许既有score重新解释，不做新OP选择。
- **Files/modules to inspect or modify：**`baselines/route_assignment.py`、`training/route_aware_transformer.py`、`training/gauge_consistent_route.py`、`training/route_operating_audit.py`、`scripts/evaluate_relative_route_v5a_source_transfer.py`。
- **New code required：**NEW `models/route_energy.py`定义typed raw-energy table；NEW `evaluation/route_counterfactuals.py`提供forced-in/out oracle；legacy score adapter；gate config schema。
- **Tests required：**本报告三个logit=8反例、±100、NaN/Inf、所有route长度、zero-init assignment identity、两兼容rivals、与全子集枚举相等、tie/空fold/missing-purity拒绝。
- **Experiment required：**先tiny deterministic；若需要重建正确fake counters，大规模saved-score replay走CPU Condor；V5A现有FAIL不需要为继续训练再做一次CV。legacy与new semantics并列，不能回写历史gate。
- **Execution command：**NEW：`python scripts/submit_research_review_jobs.py --config configs/research_review/wp1_energy_contract.yaml --submit`。测试：`python -m pytest -q tests/test_route_energy_contract.py tests/test_route_counterfactuals.py tests/test_review_gates.py`（均NEW）。
- **Metrics：**identity违反数、saturation数、assignment变化、global margin；V5A逐fold/payload C/D、new-C transition、complete purity与fake。
- **Acceptance criteria：**zero-init任意有限合法分数下canonical baseline assignment/utility相同（ties按等价集合），brute-force oracle一致；缺fold/缺metric永不PASS。V5A保留原始FAIL；重建fake counters仅新增metric版本。W74 gate补floor/missing/bounds测试，不能套新阈值翻案W72。
- **Kill criteria / failure handling：**若接口改变历史assignment，保留旧结果并发布semantic break；不得把修复后分数称为原实验。V5A分支已结束；W74作为对照不自动触发full-six training。
- **Dependency：**WP0。
- **Expected artifact：**`energy_contract_v1.json`、`legacy_vs_canonical.json`、`counterfactual_tests.json`、`v5a_gate_amendment.md`及已有FAIL的metric更正记录。

## WP2 — Qualify the alignment model before calling it an oracle（P0）

- **Objective：**证明或否定15D/gauge/FD的物理和统计适用范围。
- **Scientific hypothesis：**15D是明确外场/priors约定下稳定的局部子空间；当前response closure不能替代真实alignment。
- **Inputs：**允许train nominal/FD banks、geometry payload、外部field/material/config；不使用final-blind。
- **Files/modules to inspect or modify：**`alignment/four_station.py`、`alignment/physical_jacobian.py`、`scripts/run_four_station_relative_closure.py`、`scripts/audit_6dof_identifiability.py`、`models/field_route_fitter.py`、`scripts/audit_field_global_fit_contract.py`。
- **New code required：**NEW `alignment/relative_pose_contract.py`，明确Left-SE(3)、frame/unit、dz fixed/prior；NEW `scripts/qualify_relative_alignment.py`，输出whitened-J spectrum/JG与完整gate；文档推导。
- **Tests required：**非零nominal pose下composition；共同坐标变换连同B/state/covariance；固定B的物理位移单独测试；补偿sign；FD h/2/h/2h；S0/S3经一致群映射比较。
- **Experiment required：**truth association，仅允许train physical events；Condor重建小型独立FD银行。分别fixed-dz与带survey uncertainty模式，比较同-event response和独立measurement，不混为一项。
- **Execution command：**NEW：`python scripts/submit_research_review_jobs.py --config configs/research_review/wp2_alignment_qualification.yaml --submit`。
- **Metrics：**JG相对响应、scaled design singular vectors/rank、FD稳定性、frame变换误差、独立relative recovery/coverage。
- **Acceptance criteria：**FD列在h/2/h/2h下相对变化≤1%（近零列用数值噪声绝对阈值）；稳定可观测子空间；代数测试float64误差≤1e-9按单位归一；物理对称性误差必须低于实测propagation噪声阈值。任何非零JG若来自固定外场，写入物理模型而不是强删成gauge。输出`oracle_qualified=false`直到WP7独立closure完成。
- **Kill criteria / failure handling：**若五共同方向不成立，停止“20−5已证明”叙事，改为明确外部reference下参数/priors；若FD非线性，缩小有效envelope或做iterative relinearization，不能以放宽capture容差过关。
- **Dependency：**WP0；WP1不阻止truth-only资格检查。
- **Expected artifact：**`alignment_parameterization.md`、`gauge_field_audit.json`、`fd_stability.json`、`oracle_qualification.json`。

## WP3 — Truth-free inputs and source×event×geometry registry（P0）

- **Objective：**建立模型选择用的合法数据契约，分离source transfer、geometry shift与synthetic selection。
- **Scientific hypothesis：**现有失败至少部分来自训练support缺口；严格crossed split能区分它与表示/求解器问题。
- **Inputs：**只使用现有授权train source families及其原始数据。独立新simulation如必要，仅作为预先命名的train/dev资产；禁止借用final-blind来补量。
- **Files/modules to inspect or modify：**`training/source_transfer_cv.py`、manifest loader、`datasets/synthetic_overlay.py`、`datasets/synthetic_field_propagation.py`、`training/route_aware_transformer.py`；外部Calypso exporter单独固定revision。
- **New code required：**NEW `datasets/review_split_registry.py`；修复W73 probe的group split、null/permutation/cluster区间和数据驱动decision；truth-free候选/可选label接口、origin/GUID/hit fingerprint、payload table、candidate coverage分层report。
- **Tests required：**W73所有level共用相同origin/group split与抽样索引；same event换名仍不能跨split；family整体划分；parent/normalizer祖先检查；truth branch移除后graph/score不变；payload/hash重复拒绝；allowlist过滤先于asset resolve。
- **Experiment required：**从允许source导出少量未清洗tracklets和all-pairs，先tiny比较coverage；正式corpus/physical payload生成走Condor。controlled clean层与raw clone-aware层同时保留。
- **Split design：**外层两次family holdout；内层仅训练family按physical-event hash做70%train/15%checkpoint-validation/15%calibration，charge/kinematics尽可能分层；所有该event的overlay/payload同属一split。训练payload之外生成16个冻结held-out payload：8个独立方向和它们在同一明确chart的反向点，幅度取既定envelope的0.5及1.0交替，另保留nominal；twin整体作为同一geometry group，不能跨train/holdout。source seen/unseen × geometry seen/unseen四格分别评估。
- **Execution command：**NEW：`python scripts/submit_research_review_jobs.py --config configs/research_review/wp3_data_registry.yaml --submit`。
- **Metrics：**unique physical events、重复/重合、各cell支持数、raw/source/target/candidate coverage、clone/shared-hit/missing分布。
- **Acceptance criteria：**底层origin和payload group零交叉；全模型transfer的所有学习祖先只见train family；truth-free tiny inference成功；每fold至少100个独立held-out physical events用于方向性pilot，统计精度另由gate决定。若输入只有tracklet而没有raw truth acceptance，明确不能报告end-to-end效率。
- **Kill criteria / failure handling：**无足够独立events则标precision不足，补允许train/dev专用simulation而不打开blind；all-pairs仍不能覆盖真实输入则先修export，不启动WP4。
- **Dependency：**WP0，采用WP1数据/energy版本；WP2给出允许geometry envelope/约定。
- **Expected artifact：**`split_registry.json`、`usage_ledger.json`、`payloads.json`、`corpus_manifest.json`、`input_independence_report.md`。


## WP4 — The decisive representation experiment（P1）

- **Objective：**决定保留/解冻backbone、使用raw route physics，还是转向统计模型；不再用head-only反复失败推测原因。
- **Scientific hypothesis：**在相同候选/solver/监督下，raw physical或trainable表示能比frozen latent更稳定地分辨低utility truth与fake竞争集合。
- **Inputs：**WP3四格数据（已见families的预注册transfer benchmark，不再叫未见domain blind）；每fold独立训练的parent/normalizer；WP1 canonical route table。旧all-six W64仅作“exposed-parent历史control”，不得混入unseen-family主要结论。
- **Files/modules to inspect or modify：**`models/route_transformer.py`、`training/route_aware_transformer.py`、`training/structured_assignment.py`、`training/source_transfer_cv.py`。
- **New code required：**NEW `models/physical_route_energy.py`、`training/route_energy_training.py`、`evaluation/representation_probe.py`；统一2/3/4-length decoder和exact competitor接口。
- **Arms：**P：固定物理score + calibration-split上的正则化低维length/cost校准（pair chi²模型明确标approximate）；L：仅三个fold-parent production logits的正则化logistic/energy校准control，用于判断R_phys是否只是重复利用W64分数；F：eval模式frozen fold-parent latent + 两层128宽共享route scorer；R：raw route physics + 相同容量scorer；T：与F相同的route模型但允许共享backbone训练。所有arms使用同一candidate graph和无额外ML阈值的同一可行route集合；历史0.001-threshold W64另作为legacy baseline。不得通过每arm不同pruning制造收益。
- **Training design：**F/R/T统一30 epochs、AdamW、head学习率1e-3、weight decay1e-4；T backbone学习率1e-4，无warm-up扫描。pointwise route BCE按event归一化、使用一致采样权重，加0.1倍event-normalized exact structured hinge；阈值/length calibration仅内层calibration。内层validation按预注册primary utility选checkpoint，tie选更早epoch；外层fold从不选checkpoint。这里的数值是一次有限probe的预注册起点，不宣称最优。
- **Target / negatives：**target为canonical maximal truth components；loss-augmented task cost按missed truth component与fake selected component各1定义，ties允许truth-equivalent target。每event保留所有positive、exact当前错误集合中的全部negative，再按固定seed均匀抽最多128个其他negative；记录inclusion probability并对pointwise项校正。exact hinge仍用完整可行候选集；OOM必须修规模实现，不静默丢hard negatives。
- **Tests required：**F/R/T输入、mask和监督相同范围；parameter/gradient/dropout报告；不同route长度成本不重复；loss-augmented solver和brute-force一致；无truth输入推理；normalizer无holdout ancestry。
- **Experiment required：**两fold×F/R/T首seed共6个scorer训练，另两fold parent训练共2个，共8个正式训练job；P/L用CPU job。只有pilot无输入/数值失败才执行预注册另外两seed，共计24个训练job（含parents）；不根据外层fold结果改模型或loss后继续称同一experiment。四格评估是同一模型的固定评估，不新增调参。
- **Execution command：**NEW：`python scripts/submit_research_review_jobs.py --config configs/research_review/wp4_representation_pilot.yaml --submit`；预注册重复：`python scripts/submit_research_review_jobs.py --config configs/research_review/wp4_representation_replication.yaml --submit`。
- **Metrics：**第6节primary全套；逐fold/geometry的efficiency差、fake/purity、global margin、new-C、rank/support；event-cluster paired bootstrap 95%CI，按预注册比较使用同时区间；两family分别报，不把两个点当普遍domain证据。
- **Acceptance criteria：**候选主arm相对F、P、L的held-out pooled efficiency提升均≥2个百分点，paired simultaneous 95%CI下界>0；每fold方向一致；fake-complete和all-route-fake相对P的增幅95%CI上界≤0.5个百分点；不允许靠降低nominal换取relative robustness。完整科学gate另见第12节。
- **Decision rule：**R胜F/L且T无额外可靠收益→选低容量raw route作为主实现，取消复杂backbone依赖；T胜F/R→保留可训练共享表示；P与最佳ML在预注册精度内等价→转备选统计路线；所有模型都差且oracle candidates可恢复→检查输入缺信息/目标，暂停网络扩容。F胜出仅说明本probe表示可用，不自动复活旧complete-only V4。
- **Kill criteria / failure handling：**若CI已足以排除2个百分点收益，停止该ML路线；CI过宽则UNKNOWN，最多按预注册一次补充独立train/dev事件的replication，禁止扫超参数。若P依赖的covariance不合法，不能用它的失败宣传ML优越性。
- **Dependency：**WP0/1/3、WP2参数约定和WP6最小可复现提交；alignment oracle未认证时只报告association结论。
- **Expected artifact：**`preregistered_comparison.json`、24个或pilot停止时实际checkpoint及hash、`four_cell_metrics.json`、`representation_decision.md`，明确唯一获胜方向或NO-GO。

## WP5 — Implement the winning association system（P2）

- **Objective：**将WP4获胜表示落实为训练、推理、诊断共享的真实多长度route模型。
- **Scientific hypothesis：**在不改变数据/OP的前提下，统一输出契约和matched structured supervision可保留transfer增益。
- **Inputs：**WP4冻结裁决、WP3真实/controlled双层corpus、WP1 energy/solver contracts。
- **Files/modules to inspect or modify：**`models/physical_route_energy.py`（WP4新增）、`training/route_energy_training.py`、`baselines/route_assignment.py`、`evaluation/`公共入口；legacy通过adapter保留。
- **New code required：**typed候选/score/assignment records，2/3/4共享scorer，必要shared-hit conflict，component级批处理与可证明safe pruning，独立calibration adapter。
- **Tests required：**输入/route排列等价、station ordering、mask、缺站/clone/fake、zero candidate、one event vs batch、save/load inference一致、shared-hit exclusion、solver optimum和pruning不改变解。
- **Experiment required：**只用冻结结构在允许train/validation/calibration上最终训练；HTCondor一次受控development评估，development结果不回流选checkpoint。occupancy scale测2/4/8/16 tracklets每站合成stress，明确其工程而非物理频率意义。
- **Execution command：**NEW：`python scripts/submit_research_review_jobs.py --config configs/research_review/wp5_selected_route_model.yaml --submit`。
- **Metrics：**primary/secondary，controlled→raw degradation，99th percentile runtime/memory、枚举规模、optimality status、calibration稳定性。
- **Acceptance criteria：**复现WP4方向和第12节性能guardrails；真实输入不依赖truth；所有产物可由固定SHA/config复建。规模超限必须显式拒绝/转fallback，不得输出被截断却标exact的assignment。
- **Kill criteria / failure handling：**raw未清洗层增益消失则停止deployment声称，回WP3分析selection；若scale瓶颈才准进入proposal/pruning研究，不直接替换exact solver。
- **Dependency：**WP4成功裁决；WP6完整执行契约。
- **Expected artifact：**`association_model_card.md`、`frozen/`、`decoder_contract.json`、`development_once.json`、`occupancy_scaling.json`。

## WP6 — HTCondor reproducibility and experiment integrity（P0最低配，P2完成）

- **Objective：**每个科学比较绑定不变源码、数据、模型祖先与环境，支持preemption恢复。
- **Scientific hypothesis：**作业/配置漂移能够造成无法归因的差异；重复seed结果只有在相同语义下才有意义。
- **Inputs：**现有V4/V5 submitter/worker、checkpoint freeze、environment脚本；WP0/3 registry。
- **Files/modules to inspect or modify：**`scripts/submit_relative_route_v5a_source_transfer_condor.py`、对应worker、`pyproject.toml`、训练save/load逻辑。
- **New code required：**公共submitter/worker、schema strict validation、resolved config snapshot、DAG generator、原子done/frozen records、optimizer/scheduler/RNG恢复。
- **Tests required：**source提交后变动不影响snapshot；GPU allocation保留；unknown config拒绝；中断/恢复与不中断短toy训练指标一致到预定容差；missing output非成功；禁止覆盖run_id；forbidden path在解析前拒绝。
- **Experiment required：**tiny CPU和单GPU smoke各一job、一次受控preemption/restart；不把长训练当基础设施测试。正式评估/scan统一Condor，不交互等待长计算。
- **Execution command：**NEW：`python scripts/submit_research_review_jobs.py --config configs/research_review/wp6_condor_smoke.yaml --submit`。
- **Metrics：**hash一致、恢复step一致、作业退出/产物完整性、实际资源、固定平台seed差异。
- **Acceptance criteria：**DAG完整包含`SNAPSHOT → CORPUS → FOLD_PARENT → SCORER → FREEZE → EVALUATE → GATE`；evaluation只加载指定冻结hash；metadata由实际运行生成而非hardcoded epochs/parameter counts。
- **Kill criteria / failure handling：**任一模型/数据祖先不可追溯则作业不进入比较；只重试相同配置的基础设施故障，科学失败不自动修改配置重试。
- **Dependency：**WP0/3；最小snapshot/allowlist/metadata/GPU隔离在WP4前完成。
- **Expected artifact：**`submit.dag`、`jobs/*.sub`、resolved configs、环境lock、`reproducibility_report.md`和可恢复checkpoint。

## WP7 — Independent truth and ML alignment closure（P3）

- **Objective：**建立真正可用的alignment oracle，并测量association引入的bias/uncertainty损失。
- **Scientific hypothesis：**共同trajectory likelihood和一致SE(3)更新能在独立数据上恢复relative geometry；selected-route conditioning不会产生不可接受bias。
- **Inputs：**WP2经验证的参数约定；允许train/dev专用独立physical事件；WP5冻结association；固定field/material/beam/survey模型。
- **Files/modules to inspect or modify：**`alignment/physical_jacobian.py`保留response工具；`models/field_route_fitter.py`、`alignment/route_selected_update.py`；外部common-state/transport export。
- **New code required：**NEW `alignment/common_track_solver.py`，per-track nuisance/common q/p与global pose，合法joint covariance、survey prior、Schur消元、robust IRLS、relinearization与damped update；NEW `scripts/run_independent_alignment_closure.py`。不把现有pair chi² sum重命名交付。
- **Tests required：**linear Gaussian toy解析解和coverage、共享measurement相关性、outlier注入、left-update/sign、reference equivalence、prior-zero/finite两模式、数值奇异拒绝、真实修正后refit；truth和ML路径共用measurement code。
- **Experiment required：**先truth-only，再固定ML association，再迭代reassociate；每个geometry条件独立训练/拟合与validation事件。预注册100个独立measurement replicas用于初步coverage，不能用重叠overlay伪造100份；5个条件为nominal、1个mixed方向在0.5/1.0 envelope、1个weak方向在0.5/1.0 envelope。方向来自允许训练Jacobian并在生成replicas前冻结。所有生产、fit、replicas走HTCondor。
- **Update / convergence：**每步在当前geometry重建residual/J；用预注册step damping与最多10次迭代，连续两步scaled update norm<1e-3且independent-validation objective相对变化<1e-3才称收敛；超过上限记录nonconvergence，不以最后一步自动PASS。robust阈值只由独立train residual校准，禁止按注入真值调节。
- **Execution command：**NEW：`python scripts/submit_research_review_jobs.py --config configs/research_review/wp7_independent_closure.yaml --submit`。
- **Metrics：**relative-transform bias/RMS、truth→ML差、weak-mode uncertainty、coverage/pulls、独立refit residual、convergence、track/momentum downstream sensitivity。
- **Acceptance criteria：**每个condition的各预注册mode的95% interval coverage与95%在binomial 95%区间内相容，pull mean绝对值<0.2、width在0.8–1.2；多条件检验校正，完整结果不择优。relative bias先用工程screen：dx/dy≤0.1mm、rx/ry≤0.1mrad、rz≤1mrad，且ML相对truth的增量不超过各预算的20%。这些是本review提出的筛选阈值，**不是合作组已确认的物理精度要求**；必须在allowed simulation上冻结“参数误差→momentum/acceptance”预算后，采用更严格的科学阈值作为最终验收。若数据精度不足验证则UNKNOWN，不以宽CI称通过。
- **Kill criteria / failure handling：**truth-only失败就停止用alignment评价ML；不能用robust loss隐藏错误covariance/field。truth通过而ML bias超限，则返回association selection/cross-fitting，不能修改blind或只报selected residual变小。
- **Dependency：**WP2，WP5/6；truth-only阶段不依赖ML成功，可提前做。
- **Expected artifact：**`measurement_likelihood_contract.md`、`physics_tolerance_budget.json`、`truth_oracle_qualification.json`、`independent_closure.json`、`coverage_report.md`。

## WP8 — Freeze and execute one final-blind test（P4）

- **Objective：**在全部方法/门槛冻结后进行一次不可用于再选择的科学检验；sealed保持关闭。
- **Scientific hypothesis：**冻结系统在真正未用于模型/协议选择的事件与geometry上满足预注册性能与closure目标。
- **Inputs：**WP0–7完整PASS证据、模型/normalizer/decoder/calibration/solver hashes、数据usage registry、final-blind custodian的未使用状态证明；本阶段之前不读其资产。
- **Files/modules to inspect or modify：**NEW `configs/research_review/final_blind_contract.yaml`、公共eval/gate入口；现有blind allowlist规则不得被通用glob绕过。
- **New code required：**schema-checked一次性evaluation包、内容访问前验证hash/phase/registry、immutable access/result ledger；不增加sealed访问功能。
- **Tests required：**fake final identifiers的合成测试；wrong hash/缺gate/重复run拒绝；禁止eval中更新权重、normalizer、temperature、阈值和tie policy。
- **Experiment required：**全部门槛通过后单次HTCondor frozen evaluation；没有根据输出再跑不同checkpoint/OP。若确认纯基础设施失败，只允许相同hash配置恢复；若科学失败，记录失败。
- **Execution command：**NEW，仅满足第13节后：`python scripts/submit_research_review_jobs.py --config configs/research_review/final_blind_contract.yaml --submit`。在P0–P3该配置必须被phase guard拒绝。
- **Metrics：**预冻结primary、alignment-level与预定义分层，无事后挑选有利payload/metric。
- **Acceptance criteria：**所有预注册条件与coverage/物理预算同时满足，且access ledger无提前使用；报告成功与失败的全部分母。
- **Kill criteria / failure handling：**任一科学门槛失败则本轮NO-GO；该数据永久变为已见，不再用于“blind确认”。未来方法用新train/dev研究，新确认集须独立预注册；不得移动sealed来救结果。
- **Dependency：**WP0–7，全模型/数据/协议冻结，明确下一阶段执行授权；本review本身不执行此WP。
- **Expected artifact：**`final_blind_preregistration.md`、access ledger、冻结结果及完整`final_scientific_decision.md`；sealed无任何新产物。

# 12. Validation Gates

下表是**PROPOSED、须在新实验前锁定**的决策标准，不追溯修改历史实验。数值是工程/研究继续投入的阈值，不声称从探测器物理需求推导；最终alignment科学容差由WP7预算另行收紧。

| Gate | 必须满足 | 不通过如何处理 |
|---|---|---|
| G0 Provenance | snapshot/config/allowed data/model ancestry可追溯，tests无未解释失败；missing artifact与scientific FAIL分开 | 暂停比较，补证据，禁止重写历史 |
| G1 Utility/solver | zero-init assignment identity、raw-score一致、all-length exact competition、empty/missing gates fail closed | 不启动新训练 |
| G2 Input/split | truth-free inference，physical origin/family/payload分离，external exporter版本与输入coverage明确 | 修输入或补允许train/dev，不消耗blind |
| G3 Representation | WP4 ≥2pp改善及CI、两fold一致、fake非劣；matched model/solver条件成立 | 按预注册decision rule转统计backup或UNKNOWN；不扫head超参 |
| G4 Association | 新协议下nominal complete efficiency≥0.90，每个冻结payload≥0.80，最大相对nominal绝对降幅≤0.10；complete purity≥0.95；all-route fake≤0.05；同时满足相对baseline fake非劣 | 停止production推进；这些是新增目标，不宣称当前模型已通过 |
| G5 Alignment | truth-only及ML独立closure、有效15D/priors、真实refit、coverage和冻结physics tolerance budget全通过 | 不称downstream oracle，不进入final科学结论 |
| G6 Freeze | 一个模型、一个calibration/OP、一个solver、一个统计analysis plan、一次final run；usage ledger证明未见 | final-blind入口保持关闭 |

G4先按每payload aggregate点估计筛查，最终验收要求efficiency/purity的单侧95%下界及fake的上界满足阈值；worst-payload/multiple-comparison处理在manifest冻结时确定。若独立event数量不足，结论是precision不足，不是PASS，也不是允许扩大看过的数据后假称新的blind。

新模型选择允许使用新train/validation/calibration和已经明确叫development的数据，但不同职责不能混用：训练/结构选择使用WP3内层与预注册WP4；旧development只作最终development复核。若根据它再修改模型，该修改必须登记为新的development-informed方案，并回到新的允许训练池对照；不能声称完全未用development。

# 13. Final-Blind Strategy

**P0–P3 不打开 final-blind；sealed test 永久关闭。** 本次未访问这两类内容，也不建议为了查重复、估计分布或验证想法去读取它们。可以让custodian基于既有registry提供使用状态/重合证明，不让建模agent提前看到内容。

进入条件必须同时满足：G0–G6、唯一模型及所有祖先hash冻结、完整evaluation和统计分析代码冻结、physics tolerance budget冻结、final-blind从未参与checkpoint/architecture/threshold/normalizer选择的登记证明。既有final source命名保留，不能因为它与train家族相近/不同就重新挑一组更容易或更困难的数据。

final-blind要检验什么也必须提前写清：同一source的未见event只证明对应的event generalization，不自动证明新physics-family transfer；新geometry的范围仅限冻结envelope，不自动覆盖任意15D幅度。最终paper/report必须按实际sampling frame描述。

失败处理：科学失败公布失败及不确定性，冻结模型不再修改后重看同一数据。纯损坏文件/资源中断可由相同hash任务恢复，并在access ledger记录；不能把模型预测不好认定为“技术故障”。任何未来新确认集都应在新方法开发之前由独立流程命名和冻结，sealed不参与补救。

# 14. Open Scientific Questions

以下均为 **UNKNOWN / NEEDS EXPERIMENT**，不得用措辞强度代替数据：

1. 固定FASER field/material/beam/survey下，五个共同tracker方向到底是精确gauge、近似weak modes还是可观测物理自由度？WP2的JG和整体坐标变换实验回答。
2. 当前q/p seed与covariance/material policy是否提供真实不确定度？需要truth-independent common-state export与独立coverage，不能用candidate recall或当前mode Pareto代替。
3. 未经truth清洗的真实tracklet池里，local reconstruction/propagation coverage、clone/shared-hit和missing stations各损失多少？WP3分层统计回答。
4. frozen latent是否丢失route判别所需信息，还是相同信息被错误loss/scale使用？只有WP4 matched F/R/T/P对照能区分。
5. 非局部route特征相对contextual edge潜变量增加多少可泛化信息？不能把“有Transformer”当作信息充分证明。
6. C与D在修复utility接口、精确global margin、同origin匹配后还剩哪些共同上游原因？旧C/D标签不是答案。
7. station3下降有多少来自物理lever arm/field/动量，有多少来自source/acceptance/fake recipe？需固定origin与kinematics的因子实验；当前没有单一物理解释。
8. 两个family能否代表未来数据的变化？即使crossed CV通过，也只支持这两个family与冻结geometry范围，不证明全deployment。
9. 是否存在 association–alignment 自洽但错误的局部解？需truth-first、独立validation、reassociation对照和多初值物理实验；低selected residual不够。
10. 哪些relative modes决定最终momentum/vertex/acceptance偏差，真正可接受的alignment预算是多少？目前repository的capture阈值不能替代物理需求。
11. 显式枚举在真实occupancy尾部是否仍可控？若确实不可控，再比较safe proposal、flow松弛或learned inference，而非现在提前升级复杂度。
12. W74若取得收益，来自R_phys输入、logits-only信息、参数量减少还是优化/正则化？V5A的FAIL已确定，W74效果仍UNKNOWN；不能仅凭W73 probe恢复“Head-Only已被验证”结论。

本项目值得下一阶段投入的前提，是先把**物理可观测性、统一效用、严格数据分离和可证伪的表示对照**做成共同基线。后续Coding Agent应先执行WP0/1/2/3和WP6最低配，提交可审查的contract与测试产物，再开展WP4；不得直接继续V4/V5大训练，也不得提前开启final-blind或sealed。
