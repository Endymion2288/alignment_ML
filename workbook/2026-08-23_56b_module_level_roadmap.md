# FASER Alignment ML 下一阶段路线图：从 Station-Level 到 Module-Level / Cluster-Level

## 0. 当前阶段的核心结论

目前项目已经完成了一条完整的第一阶段路线：

> **Station-level tracklet association → station-level alignment → real-data validation**

现有结果同时说明：

1. **冻结 V2 association 可以从 MC 迁移到真实 FASER data。**  
   在足够大的真实数据统计量下，V2 可以稳定产生 selected routes，因此当前主要瓶颈不是“模型完全无法在真实数据上做 association”。

2. **Association 可用，并不意味着 station-level alignment 可解。**  
   当前真实数据上已经观察到：
   - `dz` 是近似 gauge-like weak direction；
   - station `dx/ry` 与内部 layer-level `C_dx` 存在明显 cross-level degeneracy；
   - reduced `{dy,rx,rz}` 虽然 Fisher-identifiable，但在不同 calibration run 之间无法稳定 transfer；
   - 因此当前真实数据不能可靠地产生可写入 conditions 的 station geometry update。

这提示我们：**问题可能不在 association 本身，而在观测与 alignment 参数化的层级太高。**

当前 pipeline 会把 cluster / module 几何信息先压缩成 station tracklet，再进一步压缩成 station-to-station residual。这样可能丢失区分以下几类几何变化所需的局部信息：

- station rigid-body motion；
- layer relative deformation；
- individual module displacement。

下一阶段的核心问题因此改为：

> **如果把 alignment observable 下沉到 module / cluster level，并进一步把 ML association 的对象也逐步下沉，能否恢复 alignment identifiability？**

---

# 1. 长期目标

最终希望建立如下层级化 reconstruction + alignment 链：

```text
SCT clusters / module-local measurements
            ↓
local object construction
            ↓
learned cross-layer / cross-module association
            ↓
global track hypothesis
            ↓
physics track fit
            ↓
module-level residuals
            ↓
hierarchical detector alignment
            ↓
updated geometry
            ↓
re-association / re-fit
```

长期目标是形成：

```text
Association
    ↕
Track Reconstruction
    ↕
Hierarchical Alignment
```

但不能一步做到。为了避免同时修改模型、观测、参数化和 geometry 后无法定位收益来源，后续分成三个递进阶段：

1. **先下沉 alignment observable，不改现有 V2。**
2. **再下沉 ML association object 到 layer/module-local object。**
3. **最后才考虑 cluster-level learned tracking / association。**

---

# 2. 后续工作的基本原则

## 2.1 不直接假定“每个 module × 6 DoF 都可测”

不能一开始就定义：

$$
\theta=
\{dx_m,dy_m,dz_m,rx_m,ry_m,rz_m\}_{m=1}^{N_{\rm module}}
$$

然后直接求解。

参数更多并不会自动解决 identifiability，反而可能引入更多：

- weak modes；
- common modes；
- layer coherent modes；
- station/module gauge redundancy；
- track-geometry degeneracy。

正确顺序应该是：

```text
measurement space
      ↓
Jacobian
      ↓
SVD / Fisher / null-space
      ↓
identify measurable modes
      ↓
define calibration parameterization
```

即：

> **先让数据告诉我们哪些方向可测，再决定 alignment basis。**

---

## 2.2 区分 Fit-able、Identifiable 和 Transferable

后续所有报告始终区分：

### Fit-able

矩阵数值上满秩，可以求解。

### Identifiable

参数方向真正被数据约束，而不是 weak/null mode。

### Transferable

在不同 run / time block / independent sample 上得到的 correction 可以稳定搬运。

必须牢记：

$$
\text{fit-able}
\not\Rightarrow
\text{identifiable}
\not\Rightarrow
\text{transferable}
$$

最终 geometry write 只能依赖 **identifiable + transferable** 的模式。

---

# 3. Stage 0：冻结当前 Station-Level 结果

现有 Operating Protocol V1 继续作为 baseline：

```text
real_data_operating_mode = residual_dq_monitoring_only
geometry_write_allowed = false
station_calibration_mode_available = false
cdx_mode_allowed = false
```

其意义是：

- 当前 V2 association 在真实数据上可用；
- 当前 rigid-station calibration 不可安全写 geometry；
- 当前 official geometry + frozen V2 可以用于长期 residual/DQ monitoring。

任何新的 module-level / cluster-level 方法都必须回答：

> 相对于这个 baseline，它是否真正增加了 alignment identifiability，而不只是降低 residual？

---

# 4. Stage 1：保持现有 V2，只把 alignment observable 下沉

## 4.1 核心问题

Stage 1 只回答一个问题：

> **当前 station-level non-identifiability 是真实数据本身缺信息，还是 measurement 被压缩到 station level 后丢失了内部几何信息？**

这一阶段 **不训练新模型**。

仍然使用当前冻结 V2：

```text
SegmentFit tracklets
        ↓
Frozen V2 association
        ↓
selected global track hypotheses
```

但不再直接使用 station-to-station edge residual 做 alignment，而改为：

```text
V2 selected track
        ↓
回取该 track 对应的 SCT clusters / module measurements
        ↓
重新做 global track fit
        ↓
得到 module-level unbiased residual
        ↓
建立 module/layer Jacobian
```

这是下一阶段优先级最高的工作。

---

# 5. Stage 1A：建立 module-level residual 数据链

## 5.1 工程目标

需要建立从：

```text
V2 selected tracklet / route
```

回溯到：

```text
station
→ layer
→ module
→ cluster
```

的 mapping。

最终每个 measurement 至少记录：

```text
event_id
track_id / route_id
station_id
layer_id
module_id
cluster_id

global position
module-local coordinate
measurement uncertainty
track predicted intercept
unbiased residual
track direction
```

核心要求是：

> alignment residual 必须知道“这一条 residual 来自哪个具体 module”。

---

## 5.2 优先使用 unbiased residual

对于参与 track fit 的 measurement，不应直接使用 biased residual。

目标是：

$$
r_i^{\rm unbiased}
=
m_i-h_i(\hat{x}_{-i},\theta)
$$

也就是在计算第 $i$ 个 measurement residual 时，不让该 measurement 自己决定对应的 track prediction。

如果现有 Calypso / Acts 可以直接导出 unbiased residual，优先复用；否则实现 leave-one-measurement-out 或等价近似。

---

# 6. Stage 1B：Module-Level Finite-Difference Jacobian

这一阶段仍然 **不求 alignment correction**。

先建立：

$$
J_{\rm module}
=
\frac{\partial r_{\rm module}}
{\partial \theta_{\rm module}}
$$

初始可以扫描：

$$
dx,dy,dz,rx,ry,rz
$$

但这里只是测 sensitivity，并不表示六个自由度最终都会进入 production solve。

建议先做小规模 smoke test：

```text
一个 station
→ 一个 layer
→ 2–4 个 representative modules
→ 六轴 FD
```

检查：

- geometry payload 是否真正作用到目标 module；
- cluster reconstruction 是否按预期变化；
- residual derivative 符号是否合理；
- FD step 是否位于线性区；
- 不同 FD step 下 derivative 是否稳定。

通过后再扩展完整 detector hierarchy。

---

# 7. Stage 1C：先做 identifiability map，不要马上求 correction

有了 $J_{\rm module}$ 后，先分析：

$$
F=J^T WJ
$$

重点输出：

- rank；
- singular spectrum；
- condition number；
- eigenvectors；
- parameter correlation；
- null-space；
- station-common modes；
- layer coherent modes；
- module-local modes。

最终形成一个 **Module-Level Identifiability Map**：

| DoF  | Module A | Module B | Module C |
| ---- | -------: | -------: | -------: |
| dx   |   strong |   strong |   medium |
| dy   |   medium |   strong |   strong |
| dz   |     null |     weak |     null |
| rx   |     weak |   medium |   medium |
| ry   |   strong |     weak |   strong |
| rz   |   strong |   strong |   medium |

这一步真正要回答：

> 哪些 module-level DoF 被真实 tracks 真正约束？

---

# 8. Stage 1D：重新检查当前 cross-level degeneracy

当前 station-level 最严重的问题之一是：

$$
C_{dx}
\longrightarrow
(dx_{\rm station},ry_{\rm station})
$$

的近退化。

Module-level residual space 应专门测试这个问题。

例如 rigid station `dx` 的 residual signature 可能近似：

$$
(+1,+1,+1)
$$

而 layer antisymmetric deformation $C_{dx}$ 可能更接近：

$$
(+1,0,-1)
$$

在 station-tracklet observation space 中二者可能高度相关，但在 module residual space 中应该更容易区分。

因此 Stage 1 最关键的 quantitative metrics 包括：

$$
\cos(J_{\rm station\,dx},J_{C_{dx}})
$$

以及：

$$
\cos(J_{\rm station\,ry},J_{C_{dx}})
$$

比较它们从 station-level observation space 到 module-level observation space 是否显著下降。

如果出现：

```text
station-level cosine ≈ 1
module-level cosine << 1
```

则说明：

> 之前的 degeneracy 很大程度来自 measurement compression。

这是 Stage 1 最重要的成功信号之一。

---

# 9. Stage 1E：寻找 deformation eigenmodes，而不是直接使用所有 raw module DoF

如果完整 module Jacobian 很大，建议做：

$$
J=U\Sigma V^T
$$

其中 $V$ 中的方向可以解释为 detector deformation eigenmodes。

可能自然出现：

- rigid station translation；
- rigid station rotation；
- layer antisymmetric shift；
- layer shear；
- module stagger mode；
- bow-like deformation；
- twist mode；
- weak longitudinal mode。

最终 production alignment 参数不一定是：

```text
module_0_dx
module_1_dx
module_2_dx
...
```

而可能是：

$$
\theta=
\alpha_1v_1+
\alpha_2v_2+
\cdots+
\alpha_kv_k
$$

即只保留真正被数据约束的 deformation modes。

---

# 10. Stage 1 的 Go / No-Go

## GO 条件

如果 module-level residual 能做到：

1. station `dx/ry` 与 internal deformation 的相关性明显下降；
2. 原 station-level weak modes 被拆分成可解释的 detector modes；
3. 至少存在一组低维 mode 在多个 run 上：
   - full rank；
   - condition 稳定；
   - correction 一致；
   - cross-run transfer 不恶化；
4. blind run 上 residual/DQ 稳定；

则说明：

> **alignment observable 下沉有效。**

进入 Stage 2。

## NO-GO 条件

如果 module-level residual 下：

- 原有 weak direction 仍存在；
- module DoF 之间出现更严重 null-space；
- 不同 run correction 仍完全不一致；
- 增加 module information 仍无法解除 cross-level ambiguity；

则说明：

> 当前 track topology 本身无法约束这些 alignment modes。

此时不应继续堆复杂 ML，而应优先寻找：

- external survey；
- cosmic / beam-halo / alternative topology；
- 更长 track；
- 不同 beam condition；
- 其它 reconstruction sample。

---

# 11. Stage 2：把 ML 输入下沉到 layer/module-local object

只有 Stage 1 证明 finer-grained residual 确实增加 identifiability 后，才值得修改 ML。

Stage 2 要回答：

> **如果 ML 不再只看到已经压缩好的 station tracklet，而能看到 layer/module-level local objects，association 是否能保留更多 alignment-sensitive geometry information？**

---

# 12. Stage 2A：不要直接从 raw strip hit 开始

推荐先定义中间层对象：

```text
cluster
↓
module-local hit / compatible cluster pair
↓
layer-local segment
↓
learned association
```

ML node 可以先是：

- module-local measurement；
- layer-local short segment；
- local compatible cluster pair。

不建议第一版直接把每个 raw strip hit 当 node。

---

# 13. Stage 2B：新的 node / edge

一个 local object 的 feature 可以包括：

$$
x_i=
(
x,y,z,
u_{\rm local},
v_{\rm local},
t_x,t_y,
\text{station ID},
\text{layer ID},
\text{module ID},
\text{cluster width},
\text{uncertainty}
)
$$

其中：

- global coordinate 用于 track geometry；
- local coordinate 用于 module alignment；
- detector hierarchy ID 用于 topology；
- uncertainty 用于 physics-aware scoring。

Edge 表示：

$$
P(i,j\text{ originate from the same particle})
$$

candidate graph 应继续使用 physics constraints 预筛，例如：

```text
adjacent layers
adjacent stations
physically reachable module pairs
slope / bending compatibility
```

避免完全 dense all-pairs。

---

# 14. Stage 2C：Hierarchical Association

不建议直接做一个巨大的 global Transformer。

更自然的结构是：

```text
Module / Layer objects
        ↓
local association
        ↓
station-level hypothesis
        ↓
cross-station association
        ↓
global track
```

即同时保留：

- local geometric information；
- global trajectory consistency。

---

# 15. Stage 2D：与现有 V2 做严格对照

新的 lower-level association 必须与当前 V2 公平比较。

### MC association metrics

可以比较：

- edge AP / AUC；
- route efficiency；
- route purity；
- fake；
- complete-track efficiency。

### Real-data DQ

真实数据无 truth，因此只比较：

- selected-route scaling；
- score distribution；
- event concentration；
- route composition；
- residual/DQ stability。

### 最关键的 alignment 指标

比较：

```text
station-tracklet V2
vs.
lower-level association
```

产生的 alignment Jacobian：

- smallest singular value；
- condition number；
- leakage cosine；
- transferable dimension；
- run-to-run correction consistency。

最终评价标准不是：

> 新模型 association accuracy 提高多少。

而是：

> **新模型是否让 alignment inverse problem 更可辨识。**

---

# 16. Stage 3：Cluster-Level Learned Reconstruction

只有 Stage 2 证明 lower-level ML object 确实有收益后，才进入 cluster-level。

Stage 3 的目标是：

$$
\text{SCT clusters}
\rightarrow
\text{learned global trajectory reconstruction}
$$

此时 ML 不再只是现有 reconstruction 的后处理，而开始承担真正的 track construction。

---

# 17. Stage 3A：Cluster graph

每个 cluster 作为 node：

$$
x_i=
(
u_{\rm local},
z,
\text{module},
\text{layer},
\text{station},
\text{cluster shape},
\sigma_i,
\ldots
)
$$

candidate edge 由物理几何预筛：

```text
reachable layer pairs
compatible slope window
compatible magnetic bending
module adjacency / acceptance
```

避免 $O(N^2)$ 完全连接。

---

# 18. Stage 3B：模型结构

可以采用：

```text
cluster graph
→ edge classifier / Transformer / GNN
→ track candidate graph
→ structured route solver
→ global track fit
```

可探索：

- object-centric Transformer；
- graph Transformer；
- learned message passing；
- differentiable soft assignment。

但 route consistency 建议继续保留显式物理结构，而不是完全交给黑盒网络。

---

# 19. Stage 3C：Alignment-aware training

到这一阶段才值得考虑真正的 geometry-robust training。

训练中可以随机注入：

$$
\theta_{\rm station}
+
\theta_{\rm layer}
+
\theta_{\rm module}
$$

模型目标仍以 association 为主，例如：

$$
\mathcal L
=
\mathcal L_{\rm association}
+
\lambda_{\rm geom}
\mathcal L_{\rm geometry\ robustness}
$$

但不建议直接让网络输出 alignment constants。

推荐继续保持：

```text
ML:
recover associations

physics estimator:
solve alignment
```

以保留可解释性与物理验证能力。

---

# 20. Hierarchical Alignment 参数化

长期推荐采用：

$$
\Delta T_{s,l,m}
=
\Delta T_s^{\rm rigid}
+
\Delta T_{s,l}^{\rm internal}
+
\delta T_{s,l,m}^{\rm module}
$$

其中：

- $s$：station；
- $l$：layer；
- $m$：module。

为了避免 gauge redundancy，需要显式约束，例如：

$$
\sum_l
\Delta x_{s,l}^{\rm internal}
=0
$$

以及：

$$
\sum_m
\delta x_{s,l,m}^{\rm module}
=0
$$

这样：

```text
station term
= 整体刚体运动

layer term
= station 内部 coherent deformation

module term
= layer 内局部偏差
```

三者不会简单重复。

---

# 21. Year / IOV-dependent Alignment

## 21.1 不默认不同年份共享完全相同的 alignment

不同年份可能经历：

- shutdown；
- detector access；
- thermal cycle；
- mechanical relaxation；
- power cycling；
- beam-condition change。

因此应允许：

$$
\theta_{2022},
\theta_{2023},
\theta_{2024}
$$

不同。

但第一版不建议完全独立，而采用：

$$
\theta^{(y)}
=
\theta^{\rm static}
+
\Delta\theta^{(y)}
$$

其中：

- $\theta^{\rm static}$：长期固定制造/安装误差；
- $\Delta\theta^{(y)}$：year / IOV-specific variation。

---

# 22. 时间层级可以逐步增加

未来可以进一步扩展：

$$
\theta(t)
=
\theta_{\rm static}
+
\theta_{\rm year}
+
\theta_{\rm fill}
+
\epsilon_{\rm run}
$$

但不要一开始全部浮动。

推荐顺序：

```text
Year-level
↓
发现显著变化后
↓
Fill-level
↓
确有必要时
↓
Run-level
```

因为时间粒度越细，统计量和 gauge 问题越严重。

---

# 23. 如何证明不同年份真的需要不同 alignment constants

一个 year-dependent mode 至少要满足：

1. statistically significant；
2. 在多个 independent runs 中重复出现；
3. 不是 association score shift；
4. 不是 occupancy / beam-condition effect；
5. geometry update 后能够跨 run transfer。

只有这些都满足，才应该创建新的 IOV。

---

# 24. 后续可以形成的几个核心科学问题

## Q1. Measurement compression 是否导致 alignment degeneracy？

比较：

```text
station-level residual
vs.
module-level residual
```

下的：

- singular spectrum；
- weak modes；
- leakage；
- transferability。

## Q2. Finer-grained association 是否改善 alignment identifiability？

比较：

```text
station-tracklet V2
vs.
layer/module-level model
```

最终 alignment 信息量。

## Q3. Detector deformation 的自然 basis 是什么？

不预设所有 module 六自由度，而通过 $J_{\rm module}$ 的 SVD/PCA 找出真实可测 deformation modes。

## Q4. Geometry 是否需要 time-dependent IOV？

比较不同年份 / fill 的 identifiable alignment modes。

## Q5. Association 与 Alignment 是否能形成真正迭代闭环？

最终目标：

```text
misaligned geometry
      ↓
learned association
      ↓
global tracks
      ↓
module-level alignment
      ↓
updated geometry
      ↓
improved association
      ↓
convergence
```

---

# 25. 推荐的实际执行顺序

## Phase 1：Module residual proof-of-concept

**不训练新模型。**

任务：

1. 选一个 station；
2. 从当前 V2 selected tracks 回取 module cluster；
3. 生成 unbiased module residual；
4. 做少数 modules 的 FD；
5. 验证 Jacobian；
6. 比较 station `dx/ry` 与 `C_dx` 在 module residual space 的夹角。

### Go / No-Go

如果 leakage 明显下降：

```text
GO → Phase 2
```

否则：

```text
STOP → 当前 topology 本身信息不足
```

---

## Phase 2：Full module-level identifiability map

任务：

1. 扩展所有 relevant modules；
2. 建立完整 $J_{\rm module}$；
3. SVD / Fisher / null-space；
4. 找 deformation eigenmodes；
5. 建立 measurable / non-measurable DoF map；
6. 设计 hierarchical gauge constraints。

输出：

```text
Module-Level Alignment Basis V1
```

---

## Phase 3：Run-to-run transfer test

在至少两个 independent calibration runs：

1. 使用完全相同 basis；
2. 独立求解；
3. 比较 correction；
4. cross-apply；
5. blind validation。

要求：

$$
\theta_A \approx \theta_B
$$

且：

```text
A-derived geometry works on B
B-derived geometry works on A
```

否则不能写 geometry。

---

## Phase 4：Year / IOV study

选择不同年份的 source-disjoint runs，例如：

```text
2022
2023
2024
```

只在 Phase 3 已经证明 transferable 的 modes 上研究：

$$
\theta^{2022},
\theta^{2023},
\theta^{2024}
$$

不要重新打开已经不可辨识的 modes。

---

## Phase 5：Lower-level ML prototype

把 V2 input 从 station tracklet 下沉到：

```text
layer-local segment
或
module-local object
```

先做小规模 prototype。

同时比较：

```text
association quality
+
alignment information gain
```

---

## Phase 6：Cluster-level learned tracking

只有前面已经证明 finer granularity 有明确收益后才进入。

任务：

```text
cluster graph
→ learned association
→ global tracks
→ module residuals
→ hierarchical alignment
```

这是长期最终版本。

---

# 26. 每个阶段都必须保留的 validation philosophy

任何新的 calibration mode 都至少需要：

```text
MC closure
↓
source-disjoint MC transfer
↓
real-data numerical identifiability
↓
cross-run transferability
↓
blind validation
↓
geometry-write decision
```

并始终遵守：

> **Residual decrease is a DQ observable, not alignment correctness evidence.**

真正的 alignment correctness 需要依赖：

- known-injection closure；
- independent transfer；
- identifiable parameter space；
- blind validation；
- physical mode-validity contract。

---

# 27. 最终希望得到的系统

如果路线成功，最终系统不应是“每个 module 随便求 6 DoF”，而更可能是：

```text
Learned association:
cluster/module-aware

Track reconstruction:
physics-constrained global fit

Alignment basis:
data-driven identifiable deformation modes

Geometry hierarchy:
station + layer + module

Time structure:
static + year/IOV variation

Safety:
survey priors + gauge constraints + blind transfer gates
```

最终 alignment 可以表示成：

$$
\theta
=
\theta_{\rm survey-fixed}
+
\sum_{k=1}^{K}\alpha_k v_k
$$

其中 $v_k$ 是通过真实 track sensitivity 得到的 identifiable detector deformation modes。

---

# 28. 当前最值得马上做的下一步

当前优先级最高的不是重新训练网络，而是：

> **Module-Level Residual & Identifiability Proof-of-Concept V1**

只做：

```text
现有 frozen V2 selected tracks
        ↓
回取 SCT clusters / module information
        ↓
构造 unbiased module residual
        ↓
选择一个 station / 少量 modules
        ↓
module + layer FD perturbation
        ↓
J_module
        ↓
SVD / leakage analysis
```

第一阶段必须直接回答：

$$
\boxed{
\text{station }dx/ry
\leftrightarrow
C_{dx}
\text{ 的 degeneracy 是否在 module residual space 中被解除？}
}
$$

如果答案是 **Yes**，再全面进入 module-level alignment，并进一步下沉 ML。

如果答案是 **No**，则说明瓶颈主要来自 track topology 本身，此时应该优先寻找新的 track sample / external survey，而不是增加模型复杂度。

---

# 29. 一句话路线总结

> **先把 alignment 从 station residual 下沉到 module residual，确认细粒度 measurement 是否恢复几何可辨识性；如果有效，再把 ML association 从 station tracklet 下沉到 layer/module-local object；只有这两步都证明有收益后，才进入 cluster-level learned tracking，并最终构建具有 station-layer-module hierarchy 与 year/IOV dependence 的 association–alignment 闭环。**