# Current master 科学与软件深度审阅

## 1. Executive Verdict

1. **审阅对象是实时 GitHub master `c749e45aac9157ab59ab672f9527cafa6234dc46`，不是旧审阅的 `0c8a6ca`，也不是 WB117–130 引用的 `55cf982`。** 最新编号 workbook 是 131，代码实现到 B14M-T；没有 B14M-T 成功认证，更没有 full-sample 授权。[S1][S2]
2. **当前结果比 workbook 131 的“等待 dump”更明确：两个本地 smoke 文件已有 60 rows，其中 48 optimize rows 全部不满足 stationarity，`valid_solution=0/48`。** 当前 HEAD 的审计函数只读重算得到 `inner_nuisance_profile_not_converged / FAIL`，12/12 identities 均为 Case C。这里是审阅复算，不是伪造一次官方 immutable audit；原官方 B14M-T decision 目录仍不存在。[E1][E2]
3. **继续 measurement likelihood + nuisance elimination 是正确方向；但“已经只剩 optimizer，所有 derivative contract 都彻底关闭”不成立。** nominal mean 的物理语义比旧路线可信得多；完整 production objective 在优化端点的导数精度、branch 有效性和小梯度可认证性仍未建立。[S3–S7]
4. **B14M-T 有实质正确实现：固定 alpha 的 inner 循环、每个 outer trial 重新 profile、原始 chi2 acceptance、接受后才更新 nuisance、独立 inner/outer radius、没有 statistical ridge。** 没发现 stale-nuisance outer acceptance 或主要公式 factor-of-two 错误。[S3]
5. **存在确定的假 PASS 审计漏洞。** 删除一个 restart 的 target prediction、让 alpha 相差 10 mm、清空 surviving predictions，或在通过行之前插入同 restart 的失败行，当前 `audit_identity` 仍返回 Case A。57 个相关 Python 测试全部通过，未捕获这些反例。[S8][S9]
6. **“supported stationary”不等于原始 nuisance minimum。** 数值截断后的弱方向可以有非零下降；代码只验证投影梯度，还把未计算的 outer/joint gradient 默认写为 0。不能把这些 0 解释成 stationary，更不能把结果无条件叫作 `min_nu chi2`。[S3][S4]
7. **WB128 独立参考的证明范围被夸大。** frozen-grid shadow FD 收敛、与 repaired tangent 在所测三段上相差小于 5%有实际数值支持；但 perturbed-arm “same mean/no branch switch”没有逐臂调用 production propagator，主要比较同一冻结 ledger 的复制结果。它不是 production perturbed map 等价证明。[S5][S6][E3]
8. **存在 fail-closed 基础漏洞和数值缺陷，必须先处理。** ACTS propagation error 在可作 supporting-plane projection 时仍可能被标记 `ok=true`；TR 小半径 secular solve 的绝对容差和事后 radial clipping不能保证所声明的子问题解。后者已用原 C++ 函数编译反例验证。[S3][S5]
9. **阶段判定：A/B 边界；若必须从 A–E 单选，以“可信 estimator 基础设施是否已可靠”为标准选 A。** 主要日常调试对象已经是 B 层的 profiling/optimization，但 B 的完整前提尚未满足；绝不是 C/D，也没有证据支持 E 式放弃整体物理推断框架。
10. **未来两周只做三件事：关严有效性与审计门；认证 endpoint objective–derivative 精度；通过后重跑完整固定 48-run smoke。** 不提交 1989，不写 geometry，不训练更大 ML/GNN。永久冻结旧 rank-rescue 作为主线的企图，保留全部控制和负结果。

本文证据等级：`PROVEN` 仅指明确范围内的源码事实、可重算算术或反例；`SUPPORTED` 指有实质证据但未普遍认证；`PLAUSIBLE` 指有机制理由；`UNVERIFIED` 指缺决定性证据；`CONTRADICTED` 指具体强主张与实现/反例矛盾。`P0` 表示可能使一类科学证书失效，不表示所有历史物理结果作废。

## 2. Current Master State

### 2.1 代码基准与版本关系

审阅日期为 2026-09-11。GitHub branch API、recent-commit API 与本地 HEAD 一致；本地核心 C++、`.inc`、Python audit 和 B14M-T YAML 的字节内容也逐项与 HEAD blob 比较一致。GitHub connector 实际读取了固定 SHA 的 `ProfileGlobalizationRepair.inc`；其余核心链从同 SHA 的本地仓库读取。[S1–S3]

| 项目 | 核实状态 |
|---|---|
| 当前 master | `c749e45aac9157ab59ab672f9527cafa6234dc46` |
| commit 时间 | 2026-09-10 15:18:10 UTC |
| tree SHA | `80d0977f64a05876a90ec018e8cc61658cc6a697` |
| parent | `55cf982302a3c62c57b74f368d5e3ba7723fd33a`，2026-09-08 11:52:30 UTC，B0–B14T/WB98–116 |
| 再前一提交 | `32c9044a8543845aaa0767d8089ba6fd731f31a9`，2026-09-06 16:16:07 UTC，process-noise export 仍未建立 |
| 近期更早工作 | `df863df` A2 q/p export；`77c99b1` Task A；`f31cfa1` T12 blocked；`4676e39` T11 fail；`4d40852` T10 oracle falsification；`776a52c` T03；`95f44dc` T02 |
| 最新编号 workbook | `2026-09-10_131_TaskB14M-T_Profile_Globalization修复与Stationarity再合同.md` |
| 实际实现到的 task | B14M-T explicit profile/range-space TR 的 smoke 实现，不是通过后的 estimator |
| Git tree | 1207 tracked files；55 个 `workbook/` 下文件；`outputs/` 仅跟踪 `.gitkeep` |
| 顶层目录 | `alignment, baselines, configs, data, datasets, docs, evaluation, geometry, models, scripts, tests, training, workbook, outputs` |

**仓库和实验文件必须分开引用。** 本轮读取的 B14M-T JSONL、WB123/127/128/129/130 immutable artifacts、plugin binary 均来自共享工作区，未包含在 GitHub tree 中。读到了这些证据不等于 GitHub clone 可以自动复现实验。

### 2.2 文档 reconciliation

| 文档/历史结论 | 当前判断 |
|---|---|
| `docs/PROJECT_MASTER_AUDIT.md` | 明确审阅 `0c8a6ca34c6c8b4f505dc153dd0ca8cc7949ce56`，是有用的冻结历史报告；作为当前状态入口已 stale，不能把其 P0 清单直接套用新 HEAD |
| `docs/CODE_ROADMAP.md` | 同样以 `0c8a6ca` 为执行基准，仍列 T00–T41，未提供 B14M-T 的当前状态和 supersession 导航 |
| `docs/CANONICAL_PIPELINE.md` / `_cn` | 科学方向和三类产物区分仍有用；版本说明停在 WB75–87，缺 B14J–B14M-T reconciliation |
| README 中英文 | 开头已声明新主线；正文仍有历史“current mainline”措辞和旧 rank-study 状态，虽有 historical-control 告示，仍容易误导从中间进入的读者 |
| WB117–130 的 `HEAD=55cf982` | 是当时 parent/checkpoint，不是包含这些新实现的 clean commit；本次这些实现已纳入 `c749e45`，不能只 checkout `55cf982` 重现实验 |
| WB129 | 把 `flat_direction` 写成有效终止、把部分结果描述为 converged/multimodal；被 WB130 的 nonstationarity 诊断实质 supersede |
| WB130 | 正确暴露非 stationary endpoints；仍保留 37/T3 的历史 restart-PASS。应保留记录，但必须明确它没有 estimator-validity 含义 |
| WB131 | 正确撤销“restart invariance 单独认证 optimizer”；“待 dump”已落后于本地文件现状，尚无官方最终判定 |

旧审阅指出的 bootstrap multiplicity 与非法 covariance 接受问题，当前已分别有 `bootstrap_subspaces_report(...pair_indices=info.row_indices)` 和 Cholesky SPD 检查，不能重复作为“当前未修复 P0”。这不恢复旧 paired-WLS 的绝对 alignment 解释。[S12]

建议新增当前状态索引并标记 supersession，**不要改写旧 workbook 的原始结果或历史 PASS/FAIL**。本报告不占用原计划“WB132 full-sample preflight”的 task 编号，也不是 full-sample preflight 通过证明。

### 2.3 本地 48-run smoke 的实际结果

严格按 `run,event,target,restart` 检查，当前文件恰有 12 identities × 4 restarts；本次各 event 只有 `track_index=0`。两份文件共 60 rows：12 evaluate-only、48 optimize。现有代码完整调用 `load_config → inherit_frozen_stage → recover_optimizer_contract → inventory_and_audit → decide`，得到：

```text
complete_48 = true
decision = inner_nuisance_profile_not_converged
verdict = FAIL
12/12 identities = Case C
valid_solution = 0/48
outer iterations = 0/48
35 terminations = globalization_failure
13 terminations = max_iterations
```

下表 chi2 是**失败 inner endpoint 的原始 objective**；虽然 JSON 键叫 `chi2_prof`，不能当作已经求得 profile minimum。[E1][E2]

| Identity | 四 restart chi2 范围（约） | `norm_g_n_R` 范围 | 有效 restart |
|---|---:|---:|---:|
| 100043/0 T1 | 25.072865 | 7.17e-5–1.32e-3 | 0/4 |
| 100043/0 T2 | 25.469964 | 6.61e-4–7.07e-4 | 0/4 |
| 100043/0 T3 | 27.471999 | 2.39e-4–4.48e-4 | 0/4 |
| 100043/1 T1 | 14.684343 | 1.26e-3–3.71e-3 | 0/4 |
| 100043/1 T2 | 12.866828 | 1.16e-3–1.31e-3 | 0/4 |
| 100043/1 T3 | 11.083445 | 6.22e-4–7.52e-4 | 0/4 |
| 100043/37 T1 | 3931.09–3964.56 | 12.97–13.04 | 0/4 |
| 100043/37 T2 | 6494.67–6540.52 | 22.82–22.89 | 0/4 |
| 100043/37 T3 | 3048.73–3072.79 | 10.81–10.85 | 0/4 |
| 100048/86 T1 | 12382.513 | 0.00477–0.148 | 0/4 |
| 100048/86 T2 | 9284.436 | 0.000616–0.0152 | 0/4 |
| 100048/86 T3 | 6910.7433–6910.7434 | 0.00186–0.0107 | 0/4 |

这也说明不应把本次近乎相同的 chi2 称为 restart success。所有 outer/profile-gradient 数值为 0 是 early return 后的默认值，**不是 outer stationary 的测量证据**。

## 3. Scientific Contract Audit

### 3.1 Measurement likelihood：形式明确，真实统计模型尚未认证

完整生产调用链为：

```text
dump_ckf_leave_target_out.py::main / build_flags
 → CkfLeaveTargetOutDumpAlg::execute
 → CKF TSOS hits（包括 recovered IFT outliers）
 → used/excluded 按 target station 分割
 → ProfileHit(local locX, variance=0.08²/12, surface)
 → seedFromOfficialMean
 → runProfileGlobalizationRepair
 → runExplicitProfileTR → profileNuisanceTR
 → evaluateScaledSystem → evaluateSequentialResiduals
 → propagateOneHit → ACTS mean / supporting-plane chart
```

`evaluateSequentialResiduals` 累加 `r_i=hit.loc0−predicted_loc0` 的 `r_i²/variance`，`evaluateScaledSystem` 建立 `H=JᵀWJ,g=JᵀWr`。无测量 update，无 prior/ridge 项，没有把 full-track CKF covariance 放进该 objective。`seedFromOfficialMean` 虽创建 seed covariance，进入 evaluator 时重新构造 `std::nullopt` covariance；导数路径的 dummy covariance 只是开启 tangent transport，不被加到 likelihood。上述是 **PROVEN 的代码语义**。[S3–S5]

但要区分以下统计缺口：

- **R 的数值被实现，不等于 R 已被数据验证。** `0.08²/12=0.000533333… mm²`，sigma≈0.023094 mm，是固定 strip-resolution 模型。它没有描述 cluster-size/incidence dependence、correlations、outlier contamination 或不同 sensor/IOV calibration。
- `multipleScattering=true` 不会使这条 objective 自动含 scattering uncertainty。正式 mean parameters 无 covariance，objective 仍只用固定 R；没有 scattering-angle local nuisance、相应 Q penalty 或 marginal correlated covariance。真实材料中的随机散射和能损涨落不由确定性 Bethe mean 代替。
- 这是**条件于已有 CKF hit association/selection** 的 measurement likelihood。还不是在全部 raw clusters 上完成 association uncertainty 的生成模型。
- `used_measurement_ids/count` 在 geometry lookup 之前构造；缺 identifier/surface 可被后续 `continue` 静默跳过。审计应对实际 `profileHits` 的精确 ID 集合、权重和计数，而不能只对前一个集合的自报计数。

**target exclusion 的准确表述：** 直接目标 station 的 residual 不进入 inner/outer solver，代码支持；但 `front` 来自 full-track CKF，seed mean 和已有 hit selection 可能依赖 target。`baseRow` 本身还保存了 held-out loc0，尽管求解器不读取它。因而“无直接 target residual term”是 SUPPORTED/源码可证，“整个 estimator 对 target counterfactually independent”是 UNVERIFIED，字面“target data inaccessible”过强。

最小决定性实验：保持 surviving measurements/geometry 固定，删除或置毒 target 数值，验证 solver state/trace 不变；另作**重新生成 target-blind seed/association**的 end-to-end 对照。四个小扰动 restart 不能替代后一个检验，尤其它们没有扰动初始 alpha。[S5]

### 3.2 Mean transport：生产语义确实查实，但不是物理真值认证

本次不仅阅读 WB127：还读取当前安装的 ACTS 32.0.2 headers，并检查 AthenaExternals 24.0.41 的 `libActsCore.so` 在 `0x2afce0` 的 `evaluatePointwiseMaterialInteraction(bool,bool)` 编译体，明确看到 `computeEnergyLossBethe` 调用和写入 Eloss。`PointwiseMaterialInteraction::updateState` 的 p/E→q/p、10 MeV floor，与 shadow 的 `qopAfterEnergyLoss` 对应。这支持 WB127 对**这条 pinned surface-interaction path**的判断。[E4]

`MeanMaterialProbeActor` 调编译的 slab/effect evaluation，`MeanPreMaterialActor/MeanPostMaterialActor` 记录前后状态；shadow 使用 Bethe、slab validity 和 update-state gating。WB127 decision/config/dump 继承哈希可核对，三个 required hop 的 nominal mean closure有保存证据。这个工作有真实信息增益。

限制有三点：

1. 不要推广为“所有 ACTS transport 都用 Bethe-only”。dense-volume extension、粒子种类、release 可能不同。
2. 名义 state 的 matching 证明实现语义接近，不证明 field/material map 对真实探测器准确，更不证明所有优化 trial 都在已认证区域。
3. `propagateOneHit` 的 supporting-plane 路径允许在 `propRes` 失败后，只要直线投影距离小于 50 mm，就置 `hop.ok=true`。`propagateOfficialPathJacobianHop` 有同样结构。随后 likelihood 只看 `hop.ok`。因此 **“任何 transport error 必定 fail closed”被代码反驳**；必须将 diagnostics projection 与 estimator-valid transport 分开。[S5：约 1221–1286、2154–2175]

### 3.3 Field-gradient tangent：正确修复了一个缺项，不是完整精确导数

当前安装的 `GenericDefaultExtension::transportMatrix` 把 `dGdx` 留为零，场 mean 却在 RKN stages 的不同位置采样 B。对 `dT/ds=q T×B(x)`，缺少的项确为 `q[T]×(∂B/∂x)δx`。`FieldGradientDefaultExtension` 的 stage positions、`dk1…dk4`、额外位置耦合与该 chain rule 一致；零梯度时应退回旧场 tangent。官方 wrapper 对 B、gradient 使用同一单位转换。[S7][E4][X1]

这不是 finite-difference Jacobian 本体；它是**有独立 FD 对照的 repaired variational tangent**。JSON 的 `validated_jacobian_is_numerical_derivative=true` 不应被理解为 production 直接调用 numerical FD。

目前仍有明确的 precision 边界：

- `transportMatrix` 忽略 `gradientOk`。当前 FASER wrapper 总返回 success，使该 error 分支通常不触发，但模块合同仍不 fail closed。
- ACTS 在乘 D 后显式 normalize direction；当前 extension 未加入这个离散 normalization 的 Jacobian。需要验证被遗漏分量对实际 bound derivatives 的影响，而不是自动假设无关。
- 材料 q/p 更新的映射非恒等。安装的 `EigenStepper::update(position,direction,qop,time)` 只改 parameters，不乘 material tangent；当前 `FreeTransportJacobianCollector` 主要累计 RK transport，没有显式补上 `d(qop_after)/d(state_before)`。因此完整 mean 的精确 chain rule 尚未闭合。WB128 允许的百分比残差可能包含这一项；它是否解释当前端点 gradient floor仍 **UNVERIFIED**。
- `repairedSourceLoc0Jacobian` 在 official mean endpoint 上拼 projection 与另一条启用 covariance 的 propagation tangent，但运行时不检查该 tangent path 与 official mean 的 endpoint/branch 同一性；早期 nominal audit 不能自动覆盖全部 trial。

不能据此否认场梯度修复的价值。正确结论是：**遗漏 nonuniform-field coupling 的机制被查明并修复；完整 production derivative 在 `1e-6` stationarity 精度上的适用性未证实。**

### 3.4 WB128 的“独立 certified reference”实际认证了什么

本次读取并复算 `repaired_tangent_vs_certified_shadow.json`，三个 required hop 的 q/p 相对误差分别为 `0.0120387144, 0.0083941700, 0.0003188583`，与保存值一致；各列 ladder last-pair 的算术也符合 5% gate。对 frozen shadow map 的独立差分对照是实质证据。[E3]

但是 `certifiedMeanCommonGridFdJson` 的每个 ±arm：

```text
shadow.replay(perturbed_start, prod.events, ...)
 → 比较 nField、nMaterial、materialIdentity
 → arm_mean_contract_same = projected && counts_equal && identities_equal
```

**没有** `production_propagate(perturbed_start)` 再与该 arm shadow 比较的步骤。`replay` 直接复用 nominal `src.slab`、material gating 和 step partition；相同 material IDs/counts 很大程度由构造保证。特别是 path correction、材料交点/分区随扰动变化，并不由“复制同一个 slab”验证。

因此必须分开三条 claim：

| Claim | Verdict |
|---|---|
| nominal production mean 与 shadow mean 在已测三段闭合 | SUPPORTED |
| frozen-ledger shadow 的 FD ladder 收敛并在 5% 内匹配 repaired tangent | PROVEN（已存算术）；适用域限这些 points/columns |
| 每个 perturbed arm 已证明与 production mean 同函数、无 production branch switch | CONTRADICTED（声称做过的检查缺失）；实际等价性 UNVERIFIED |

最小补证不是重开无穷 derivative campaign：只取本次固定失败端点及 WB128 三段，逐 ±arm 记录 production mean、shadow mean、完整 material/navigation signature、direction normalization 和 q/p update；比较 directional derivative 的绝对误差是否足以认证当前 gradient threshold。**保持旧 5% gate/旧结果不变，新增适用于 endpoint stationarity 的独立 precision contract。**

### 3.5 Profile framework 的物理定位

用 `gamma` 表示待求的 geometry、用 `q_t` 表示每条 track 的参数，则 alignment 的核心应是：

$$
L(\gamma,\{q_t\})=\sum_t L_t(m_t\mid q_t,\gamma)+L_{\rm survey}(\gamma),
\qquad L_{\rm prof}(\gamma)=\min_{\{q_t\}}L.
$$

当前 `alpha=(loc0,theta),nu=(loc1,phi,q/p)` **全部是同一条 track 的参数**，不是两个 geometry DoF 加三个 track nuisance。当前 2+3 分区是 LTO supported/weak-coordinate diagnostic；最终 geometry solve 必须把每条轨迹的全部必要 local 参数消去，并明确 gauge。成熟 global/local block elimination 的统计结构可参考 Millepede II，但不要求现在迁移整个软件框架。[X2]

## 4. Numerical Optimization Audit

### 4.1 符号、缩放和 Schur

本代码用的是半梯度 convention：

$$
r=m-h,\quad J=\partial h/\partial\theta,\quad
g=J^TWr=-\tfrac12\nabla\chi^2,\quad H=J^TWJ.
$$

因此应求 `H delta=g`，参数做 `theta+=delta`。局部模型为

$$
m(\delta)=\chi^2-2g^T\delta+\delta^TH\delta,
\qquad \mathrm{pred}=2g^T\delta-\delta^TH\delta.
$$

`rangeSpaceTrustRegionStep` 与 actual=`chi2_old−chi2_new` 的 factor-of-two 一致；没有把 `1/2 chi2` 和 chi2 混用于 rho。`kNumericScale=(1,1,1e-3,1e-3,1e-3)`，在 ACTS native units 下 `H_z=SᵀHS,g_z=Sᵀg`，这一变换不改变 objective。[S3][S4]

`extractSchurAlpha` 使用

$$
H_{\rm prof}^{GN}=H_{aa}-H_{an}H_{nn}^{+}H_{na},\qquad
g_{\rm prof}^{GN}=g_a-H_{an}H_{nn}^{+}g_n.
$$

这在**线性化 least squares 的局部消元模型**中正确。若 inner 真正 stationary，envelope theorem 给 `∇_alpha chi2_prof=−2g_a`；修正项应消失。非零 `g_n` 时该式是 eliminated quadratic RHS，不是对尚未求准的 nonlinear profile gradient 的自动认证。GN Schur 也不是一般 nonlinear profile 的 exact Hessian：缺少 residual-weighted second derivatives。局部 convex GN matrix不能排除真实 objective 的 saddle、其他盆地或全局不可辨识。

`H_nn` 的 pseudoinverse 只在保留的正谱上工作，没有加 ridge。对于精确 PSD normal matrix，其 exact null 与 cross block 满足兼容条件，Schur 消元合法。但是数值截断“很弱”不等于 exact null；应同时记录 dropped gradient/cross-block residual，不得把 finite step 无法分辨和物理无信息混为一谈。对 H 作 `1e-8` 相对阈值，对应加权 Jacobian 奇异值比约 `1e-4`，不是 Jacobian 的 `1e-8` 阈值。

还有一个跨语言差异：Python `symmetric_pinv` 使用 `max(max_abs_eigenvalue,1)`；C++ `symmetricPinv3/5` 只在最大值非正或非法时改成 1。故 `diag(1e-10,2e-10)` 在 Python rank=0、C++ rank=2。常数名字相同不代表同一 numerical contract。[S4][S11]

### 4.2 Inner nuisance minimization

`profileNuisanceTR` 是真正的 iterative 3D inner solver；`applyNuDelta` 保持 alpha 两坐标不动，反复重算 residual/J/H、trial evaluation、rho 和 radius。不是一次 joint update 冒充 profile。[S3：250–437]

但它至多寻找到局部 retained-subspace stationary point，不能保证全局 `argmin_nu`。当前 48-run 全部未达它自己的 `1e-6` gate。它的 inner trace 还不能充分复算：没有每一步完整 theta、实际 post-bound delta、J/W/H、gradient dropped component，子调用的日志父子关系也不完整。

四个初始 seed 是代码硬编码的 nominal、loc1+1 mm、phi+1e-3、q/p×1.1。后续 inner 从 accepted nuisance warm start；没有选四者 chi2 最低值作为正式 estimator，这是正确的。现有证据只说明这四个 seed 的行为，不证明任意 seed/basin invariance。

### 4.3 Outer candidate 是否重新 profile

**是。** 代码 573–598 行先产生 alpha candidate，恢复上一 accepted theta 的 nuisance，调用 `profileNuisanceTR`；只有 trial inner `ok && innerStationary` 后才计算 `actual=sys.chi2−trialInner.chi2`，接受后更新 `theta=trialInner.theta`。拒绝时不会把 trial nuisance 赋回 accepted theta。[S3]

这条逻辑值得保留。不过当前真实 smoke 没有进入 outer optimization，故“源码设计正确”与“真实 outer 已经工作”必须分开：后者 **UNVERIFIED**。需要独立 analytic profile benchmark、nonlinear profile 对照和 rejected-trial state-immutability 测试。

### 4.4 Trust-region 十项核查

| 要求 | 当前实现与判断 |
|---|---|
| 支持空间中的 TR 子问题 | 对有限 PSD H、普通半径，SVD range + secular multiplier 结构正确；缺有效性和残差检查，小半径存在确定缺陷 |
| 严格 range-space step | 原始 delta 在 V_R；`applyPhysicalBounds` clipping 后实际 displacement 未必仍在该空间 |
| predicted reduction 符号/2 因子 | 对未 clipping 的 GN 模型正确 |
| rho convention | actual 与 predicted 均为 chi2 decrease，正确 |
| rejection radius | 乘 0.25，并 floor 到 Delta_min；无无限循环，但 `< Delta_min` 分支通常不可达，会在 floor 重复同一 trial |
| accepted state | outer 只在 rho 接受后赋值，正确；最末重新 inner 的失败/trace语义需加有效性断言 |
| inner/outer radius | 局部变量独立；每次 inner 新建半径，没有错误共享 |
| LM multiplier | 仅用于 `(H_R+lambda I)` 的步，不改 chi2，也不加到统计信息矩阵，正确 |
| fail closed | B14M-T 的大梯度 stagnation 本次确实返回失败；底层 transport/branch/NaN 合同不完整，不能整体认证 |
| flat_direction 路径 | 新 solver 不以此成功停止；旧 `runOneProfileNumerics` 和 `runNuisanceOnly` 仍保留，必须标为 legacy/diagnosis 并禁止混用 |

`Delta0` 实际实现还加了 `max(normDelta,Delta_min)`，并先用 `Delta_max` 限制的解估算，再在第零步重复初始化；不完全等于 YAML 文本 `min(1,||delta_GN,R||)`。通常不影响非零大步的初始值，但执行合同应记录真实数值规则，不能以 prose 代替。

### 4.5 直接编译原 C++ 内核的反例

本轮只在 `/tmp` 抽取原 `RangeTRStep` 和 `rangeSpaceTrustRegionStep` 编译，不修改项目源文件。

普通 convex 对照：`H=diag(4,1),g=(2,1),Delta=10`，返回 `(0.5,1)`，pred=2；`Delta=0.2` 的 KKT residual≈4.6e-16。exact-null 对照 `H=diag(1,0),g=(2,0)` 返回 rank=1、delta=(2,0)。这些支持其基本符号和 range 求解。

**小半径缺陷：** secular Newton 用 `abs(f)<=1e-14*max(Delta²,1)`，在 Delta=1e-8 时容差是 1e-14，而目标平方半径是 1e-16。过早停止后再径向缩放并非一般各向异性 TR 的解。

使用冻结 `pinv_relative=1e-8`、`H=diag(1e6,1)`、`g=(0.01,0.0001)`、`Delta=1e-8`：

```text
原 C++ delta = (1.45988235e-9, 9.89286326e-9)
独立 bisection secular 解 = (9.63834678e-9, 2.66500871e-9)
原 predicted decrease = 2.90448653e-11
最优 predicted decrease = 1.00402202e-10
```

原步只得到最优模型下降的约 28.9%。这不是 `chi2` factor-of-two 错误；它是 subproblem accuracy/contract 缺陷。近似 TR 也可以有收敛理论，但当前既没说明采用何种近似保证，也没检验 Cauchy/KKT 条件，不能称作已认证精确 subproblem。**该缺陷对当前 Athena stall 的因果贡献仍 UNVERIFIED**，不得仅凭 toy 就宣称已找到全部根因。

最小修复：对 PSD supported spectrum 用 safeguarded/bracketed secular solve，验证 radius feasibility、scaled KKT residual 和 predicted decrease；非有限/明显非 PSD 输入明确失败。不可通过改 rho/gradient gate 隐藏失败。

### 4.6 Stationarity 的漏洞与正确解释

`runExplicitProfileTR` 最后记录 joint `normGR`，却只把 inner/outer 两个 projected gradients 组成 `validStationary`。Python `_row_stationary` 也只看这两个数，不看 joint norm，不检验 dropped-space gradient；变量名 `norm_g_alpha_prof_z` 实际是 profile range 投影 norm。[S3：651–691；S8：386–420]

实际编译反例：`H=diag(1,1e-10),g=(0,1e-3)`，阈值 1e-8 截掉第二方向，返回 `g_R=0`。然而在 Delta=1 内沿第二方向的模型下降是 `0.0019999999`。它可以来自合法 quadratic least squares，例如 J 的第二列尺度 1e-5、相应 residual=100；不是任意拼了不相容的 g/H。

故允许报告“此截断子空间 stationary”；**不能**由此直接认证完整 `min_nu chi2`。处理方式是输出完整/dropped gradients、谱和可解析下降界，区分 exact structural null、weak identifiable、precision unresolved、numerical failure。不是盲目降低 rank threshold，也不是用 pinv 把 dropped uncertainty 填零。

37/T3 现在确实被 inner gate 拦住，堵住了 WB129 的那一个已知反例；但因为以上投影和审计漏洞，不能宣称“一切 restart-invariant false PASS 已从逻辑上排除”。

### 4.7 当前失败的真实层级

**37：initial-inner budget/尺度问题已有直接证据。** nominal T1/T2/T3 都连续接受 50 步、最后 radius=10，末步 rho 分别约 `1.00163/1.00155/1.00145`，actual decrease 仍约 `261.65/464.78/219.75`。此时不是“全拒 globalization”，也不是 stationary basin；新流程在固定初始 alpha 下消耗完 inner budget。不能和 WB130 在另一组 alpha（旧 joint endpoints）上的固定-alpha minima 直接比较优劣。

**0/1/86：微小下降与 objective/derivative precision 的一致性待判。** 例如 0/T1 nominal，在 Delta_min 预测下降 1.69e-11，actual≈−7.60e-8、rho≈−4502；86/T1 nominal 预测下降≈9.36e-11，actual≈−5.46e-12。存在舍入/float material/adaptive partition/近似 derivative 的候选机制，但本轮没有实际 endpoint 重传播来区分。

不要把所有这些情况统称为“优化器不够强”，更不要把它们归因于 detector 不可辨识。下一项决定性实验应测量同一 endpoint 的 objective repeatability 和 directional-derivative error envelope；先问 `1e-6` gradient 判据在当前 evaluator 上是否可被证明满足，再决定数值实现。

## 5. Code Quality / Architecture

### 5.1 P0：会破坏科学证书的确定问题

**P0-1：restart/预测完整性审计可假 PASS。**

- 文件/函数：`datasets/b14m_profile_globalization_repair.py::audit_identity, inventory_and_audit`，约 433、487–560、748 行；辅助 `_pred_meas` 在 `datasets/b14m_restart_invariance.py`。
- bug：`by_restart={...}` 最后一条覆盖前一条；complete 使用 `n_optimize>=48`；部分 missing prediction pair 被筛掉；空 surviving set 可产生 0 difference；`alpha_ok` 被计算却不参与 Case A/E 决策。
- 影响：restart completeness、supported-parameter invariance、prediction invariance 的 PASS 不可靠。重复失败行可能被消失；未来多 track 同 event 还会因 identity 未含 `source/track_index` 被合并。
- 最小修复：完整唯一 UID（source/input GUID/run/event/track/target/restart）、恰好 expected rows、缺失/额外/冲突显式 FAIL；所有 prediction 必须 finite 且实际 measurement IDs 完整；将 alpha gate 纳入或明确将其剥离为另外的、不能混称的诊断。
- 必须重跑：mutation tests、所有用这些 gate 认证的 restart summaries、固定 48-run audit；若只改 Python 分类且 raw dump 足够，不需重跑 Athena。当前 48 rows 本来全 FAIL，不因修复变成 PASS。

**P0-2：失败传播可能作为有效 likelihood。**

- 文件/函数：`CkfLeaveTargetOutDumpAlg.cxx::propagateOneHit, propagateOfficialPathJacobianHop`。
- bug：`!propRes.ok()` 只记录错误；后续 `if(projected) ok=true` 覆盖了 estimator validity 含义。
- 影响：fail-closed transport claim失效；可以拿未完成 trajectory 的直线投影构造 residual/Jacobian。
- 最小修复：区分 `diagnostic_projection_available` 与 `transport_valid`；estimator 要求真实成功、finite end state、允许的 branch/continuation。诊断数据仍保留。
- 必须重跑：人工注入 transport failure 的测试；现有 raw hops 筛查；修改生效后原四事件 smoke。缺 final/trial hops 的 dump 不能靠自报 `ok` 排除影响。

**P0-3：WB128 perturbed-production certification 未实际执行。**

- 文件/函数：`CkfLeaveTargetOutDumpAlg.cxx::certifiedMeanCommonGridFdJson`；`MeanTransportContract.hpp::replay`；其 Python decision/audit。
- bug：用 frozen shadow 臂之间的相同 ledger 条件，认证了未被比较的 perturbed production mean/branch。
- 影响：使“production Jacobian 已完整 certified，之后不可再检查”的强结论失效。不是推翻 nominal Bethe closure，也不是否定保存的 5% 算术。
- 最小修复：收窄旧 claim，增加 required arms 的独立 production 对照及 full-chain/endpoint precision check，保留原 ledger 与 frozen gate。
- 必须重跑：WB128 required arms 的补充实验及当前 endpoint derivative certification；仅 hash 重算和现有五个 Python tests 不够。

### 5.2 P1：必须在下一次数值认证前处理

| 问题 | 最小行动 |
|---|---|
| truncated-gradient 被当完整 profile stationarity | 报完整/保留/丢弃梯度、谱、rank semantics，增加 discarded-descent certificate |
| tiny-radius TR secular tolerance + radial clipping | 原数学内核加解析/KKT oracle，修求根精度与失败返回 |
| physical bound clipping 后 rho 用原步 | 记录实际 delta，边界 trial 明确拒绝或采用一致 constrained subproblem；保持 p/angle domain 声明 |
| finite/PSD checks 不全，`RangeTRStep.ok` 几乎恒 true | 对输入、谱分解结果、delta/pred/rho/end state 强制检查，并贯穿 result validity |
| `gradientOk` 被忽略、material/normalization tangent 不完整 | 先测量误差并作有限修补，不以百分比 agreement 自动认证高精度 gradient |
| B14M-T 失败 row 的 `fit_success/profile_success` 可 true | 区分 evaluator success、inner success、estimator valid；未计算的梯度用 null，禁止 0 占位 |
| audit YAML 与 C++ constexpr 并行硬编码 | 一个执行合同或启动时严格比对；dump resolved constants/hash，不能只 hash 1920 行 YAML |
| 直接 dump truncate + mutable worker/plugin 路径 | exclusive attempt 输出、atomic finalize、运行时 binary/config/input pins |
| final inner trace/radius/count 不完整 | 明确 parent outer/trial IDs、accepted theta 与状态谱；避免反复重复同一 Delta_min trial |

`applyPhysicalBounds` 还把 theta 限在 `[1e-3,pi−1e-3]`、`|q/p|` 限在 `[1e-9,1] /GeV`。这不是 statistical ridge，但确实限定了允许的参数域（例如 p≥1 GeV）；必须说明它的物理依据、边界触及状态和 KKT 语义。不能宣称无任何边界的 argmin。

### 5.3 P2：最小科学风险的架构改进

主 `.cxx` 6890 行，Basin `.inc` 772 行、Globalization `.inc` 835 行、Mean header 714 行；B14M-T YAML 1920 行；Python audit module 850 行。新增 optimizer 依赖诊断 `.inc` 中的 `ScaledSystem/symmetricPinv3/evaluateScaledSystem`，include 顺序具有语义。大量 workbook flags、重复 profile implementations、hard-coded event roles，使增加 B15/T20 的科学风险持续上升。

**不适合继续原样向主 `.cxx` 堆 B15/T20。** 但不需要大重写。

现在必须做的最小边界：

1. 将 `RangeSpaceTrustRegion` 与 Schur/normal assembly 抽为不依赖 Athena/JSON 的小内核，或提供可直接编译的测试入口。保留冻结实现作 golden reference；新旧只通过显式 solver ID 切换。
2. 单一 `EvaluationResult` 返回 residual、实际 hit IDs、weights、Jacobian、domain/branch/finite status。`NuisanceProfiler/ProfileOptimizer` 不读取 workbook JSON，也不触达 held-out payload。
3. `AuditRecorder` 只序列化求解器返回的事实，不自行写 `statistical_model_unchanged=true` 之类无法核验的证书。
4. 去掉轻量 audit 的非必要训练栈 import。当前普通 Python 跑 B14M-T tests 会沿 survey/legacy 模块间接导入 `models.__init__ → torch`；LCG 环境能跑不等于模块依赖合理。

full-sample 之后可以做：拆 `MeasurementLikelihood/TransportEvaluator/DerivativeProvider` 的实现文件、历史 campaign registry、共享 artifact inventory；优化日志开销和 source-local batch execution。

现在不要做：全面改 ACTS/Athena runtime、全部重写 Python/C++、删除旧 integrators/`.inc`/negative artifacts、把 B14M 二参数 alpha 重命名后冒充 geometry solver。重构验收应保持 frozen source/dump hashes 和逐步数值对照，科学修复与纯文件搬迁分开。

## 6. Test Adequacy

### 6.1 实际运行与限制

LCG_110_cuda 环境下运行以下八个文件，结果 **57 passed in 101.58s**：

```text
test_b14m_profile_globalization_repair.py
test_b14m_profile_basin_diagnosis.py
test_b14m_restart_invariance.py
test_certified_mean_common_grid_fd.py
test_surface_energy_loss_mean_semantics.py
test_field_gradient_variational_repair.py
test_profiled_information.py
test_profiled_weak_nuisance_likelihood.py
```

普通系统 Python 首次 collection 因间接缺 torch 失败，随后用项目指定 LCG 环境完成。未重跑 Athena、1989、完整仓库 tests、训练或 sealed-test events。原 C++ TR 内核另外编译执行了前述 quadratic/KKT probes；没有把它伪称为完整 C++ profile integration suite。

### 6.2 有证明力与形式性测试的区别

有意义的现有测试包括：Python `test_profiled_information` 的 Schur/joint solve/covariance 等价、information 不增长；`test_profiled_weak_nuisance_likelihood` 的线性 profile 对照、rank-deficient nuisance、prediction 遇 null direction 返回非有限 uncertainty。这些是真正的数学单元测试，不能一概贬为 schema tests。[S10][S11]

但 **B14M-T 八个 tests 没有调用 C++ solver**。主要验证 YAML 字段、继承的哈希、人工 JSON rows 的分类与 `refuse_*()` 抛错。直接调用一个永远抛错的 `refuse_prior()`，不能证明 production 无 prior；`audit_equation_contract` 返回 `g_equals_zero_reduces_to_acts_d=true`，再 assert 该布尔，也不验证方程。

| 必需 adversarial test | 当前证明状态 |
|---|---|
| 已知解 convex quadratic，实际 C++ solver | 仓库缺；本轮只验证了 TR 单步内核 |
| exact null + identifiable directions | Python helper 有；C++ inner/outer 缺 |
| weak-but-nonzero direction 不得伪装 exact null | 缺；本轮原 C++ 反例触发 |
| nonlinear GN overshoot、TR 收敛 | B14M-T 缺 |
| analytic fixed-alpha inner vs numerical inner | Python 线性 helper 有；实际 C++ profiler 缺 |
| outer Schur vs brute-force nonlinear profile | 缺 |
| rejected trial 不得改变 accepted nuisance | 源码支持；缺直接 regression test |
| restart invariance 前必须 stationarity | JSON 层有，数学认证不完整 |
| 37/T3 invariant-but-nonstationary | 当前真实 dump 被拦；应建立永久 fixture 和 endpoint 数值回归 |
| target poisoning/deletion 不影响 fit | 未见实际求解路径的 impossibility test |
| branch-changing trial 必须失败 | 缺；运行时未比较完整 trial branch |
| NaN/field/transport failure propagation | 缺关键调用链测试 |
| Delta_min + 大梯度返回 failure | 分类层有模拟，缺实际 solver 回归；当前真实 smoke 恰有失败实例 |
| 缺失/重复/额外 row、prediction、alpha 越门 | 缺；本轮可得 false PASS |
| Python/C++ rank threshold 一致 | 缺，且已有可构造不一致 |

### 6.3 已执行的审计 mutation probes

从现有 test `_four()` 生成完全相同的四个合法 fixture，独立修改一项，直接调用当前 `audit_identity`：

| 修改 | 当前返回 | 正确处理 |
|---|---|---|
| 一行 `predicted_target_loc0=None` | Case A，prediction=true | 缺少完整 prediction evidence，FAIL |
| 一行 alpha.loc0=10 mm | Case A，alpha_invariance=false | 不得通过冻结的 supported-parameter gate |
| 四行 `norm_g_R=100` | Case A | 与声称的 joint stationary 合同冲突，FAIL/显式 scope 限定 |
| 四行 surviving predictions=[] | Case A，prediction=true | 实际测量预测缺失，FAIL |
| 同一 nominal 的失败行放在合法行前 | Case A | UID 冲突必须保留并 FAIL，不能 last-write-wins |

这些是证书逻辑的决定性反例；不是随机输入 fuzz 猜测，也不表示当前 48 条失败 row 被错误地判为 PASS。修复测试的价值在于防止未来看起来漂亮的输出越门。

## 7. Claim–Evidence Matrix

| Claim | Code Evidence | Test/Artifact | Verdict | Risk |
|---|---|---|---|---|
| master 已到 B14M-T | `runProfileGlobalizationRepair`、commit | GitHub HEAD、WB131 | PROVEN | 不可误读为任务成功 |
| 48-run repair 已通过 | actual inner/outer validity | E1/E2、只读 `decide` | CONTRADICTED，0/48 valid | 极高 |
| objective 是固定 R 的 measurement chi2 | `evaluateSequentialResiduals` | R 常数与 residual assembly | PROVEN（形式） | 不认证真实 calibration |
| 无 CKF covariance prior/ridge | nullopt eval、原始 chi2 | C++ 全调用链 | PROVEN（该路径） | seed/selection 仍可能 target-dependent |
| held-out target residual 不进 fit | `used` filter、profileHits 参数 | used IDs、目标计数 | SUPPORTED | 缺 counterfactual independence test |
| full estimator target-independent | CKF front seed/TSOS selection | 无 end-to-end blind reconstruction 对照 | UNVERIFIED | 高 |
| surface deterministic energy loss=Bethe | installed compiled body、MeanTransportShadow | E4、WB127 hashes/ledger | PROVEN（pinned path） | 外推 runtime/physics 不成立 |
| nominal supporting-plane mean closure | production/shadow ledgers | WB127 三段 | SUPPORTED | 限 points/domain |
| ACTS default 缺 field-gradient position coupling | GenericDefaultExtension/mean stepper | installed source + upstream v32.0.2 | PROVEN（pinned default） | 不推广全部 ACTS |
| repaired field tangent改善已测列 | FieldGradientDefaultExtension | WB123 controls、E3 | SUPPORTED | 非完整 exact derivative |
| WB128 frozen-shadow FD 收敛且 A≈C | `certifiedMeanCommonGridFdJson` | E3 重算，最差约1.204% | PROVEN（保存算术） | 限冻结 shadow map |
| perturbed production branch/mean 已逐臂认证 | `replay(prod.events)`、armMean predicates | 没有 production ±arm 对照 | CONTRADICTED（检查已完成的主张） | P0 certification scope |
| inner 固定 alpha 迭代优化 | `applyNuDelta/profileNuisanceTR` | source；实际均失败 | PROVEN（实现结构） | argmin UNVERIFIED |
| outer 每次 reprofile | `runExplicitProfileTR` 573–598 | source；没有真实 outer steps | SUPPORTED | 缺 integration test |
| profile GN RHS/Schur 正确 | `extractSchurAlpha` | Python linear tests | SUPPORTED（GN/compatible rank） | 不是 exact nonlinear curvature |
| TR pred/rho factor正确 | `2*gTd−dHd` | convex C++ probe | PROVEN（未clip模型） | bound clipping 例外 |
| LM 不当 statistical ridge | solve-only lambda | actual chi2 acceptance | PROVEN | rank truncation 另论 |
| 原始完整 objective 已 stationary | projected norms、未gate joint norm | weak-direction probe | CONTRADICTED（若无 scope 限定） | 高 |
| 37/T3 不再靠 invariance 假通过 | inner gate | actual 4/4 FAIL | PROVEN（本 smoke） | 普遍防假PASS仍不足 |
| nuisance 非唯一但 prediction 有限 | Python null-aware uncertainty helper | linear synthetic tests | SUPPORTED（helper） | B14M-T Case E 只是seed差异，不是非辨识证明 |
| B14M-T audit fail closed | duplicate/missing/alpha gates | mutation probes | CONTRADICTED | P0 |
| 因此可 full-sample/alignment | `full_sample_authorized=false` | failure evidence | CONTRADICTED | 禁止越门 |
| real-data validity/coverage | 无合格闭环 | 无独立 calibration/coverage evidence | UNVERIFIED | 最高下游风险 |

## 8. Top Risks

| 排序 | 风险 | 严重度 × 当前可能性 | 何种证据能降低风险 |
|---|---|---|---|
| 1 | 证书实现与证书文字不一致：false PASS、frozen-shadow 自认证 | 极高 × 已证实 | adversarial gate suite、真实 production-arm 对照 |
| 2 | optimizer/oracle precision 不相容，所有 inner fail | 极高 × 已发生 | endpoint error budget、正确 TR oracle、全48 stationarity |
| 3 | measurement covariance / scattering model misspecification | 极高 × 高但未量化 | whitened residual结构、独立 sigma/Q 校准、coverage，不靠rescale追chi2 |
| 4 | nuisance 弱可辨识与数值截断混淆 | 高 × 高 | full spectrum/drop-gradient、profile intervals、prediction estimability |
| 5 | target-dependent seed/CKF selection 被当作完全 LTO | 高 × 结构性存在依赖途径 | blind seed/selection 对照、target poison/deletion |
| 6 | material/field derivative与真实 model mismatch | 高 × 存在确定未闭合项 | material/normalization tangent、field/material nuisance、控制样本 |
| 7 | provenance 无法绑定运行 binary 与 dirty source/config | 高 × 当前链有缺口 | execution manifest、binary/source/config/input 内容哈希 |
| 8 | association uncertainty 未建模 | 高 × 下游必遇 | hard-assignment baseline偏差、共享hit/未匹配/竞争假设校准 |
| 9 | geometry gauge/外约束不足 | 极高 × 下游必遇 | global geometry model、真实 gauge basis、认证 survey covariance/IOV |
| 10 | simulation-to-real transfer 和重复使用 validation | 极高 × 尚未检验 | 新冻结独立 protocol、真实监测、coverage/selection/IOV分析 |

**当前最急的是 1–2，最终方法学最危险的是 3、5、9–10。** 一个完全收敛的错误 likelihood 会给出比显式数值 FAIL 更危险的伪精度。旧控制样本反复用于 debug 可以，但不能再把它们称为 untouched confirmatory sample。

### Reproducibility / provenance 专项

已有优点：WB inheritance 对 config/decision/dump SHA 的逐项核对；unique-run `ImmutableArtifactStore`；原始 negatives 保留；固定 raw/eligible denominator；提交计划保存 plugin SHA。WB123、127、128、129、130 的 decision SHA 本轮均匹配声明。

还不能回答完整 publication-grade lineage：

| 问题 | 当前能回答什么 | 缺口 |
|---|---|---|
| 哪个 commit/dirty patch | 新 HEAD 可锁定；旧 artifacts 常记 `55cf982`+helper SHA | hash 不等于可恢复的 dirty source bundle，特别是 include headers/`.inc` |
| 哪个 config | audit YAML SHA 精确 | worker 不把此 YAML 传给 C++ TR；constexpr 与 Python 合同未强绑定 |
| 哪个输入 | source ID、完整 xAOD path | dump缺执行时 GUID/content hash、schema/selection manifest |
| 哪个 runtime | 指定 Athena/ACTS/Calypso path；当前 plugin SHA匹配提交记录 | 作业启动时未锁 snapshot，mutable build路径、DBRelease/current、实际 loaded libraries未绑定 |
| 哪个 geometry/field/material | `_hashes` 计算标签/材料内容相关hash | `build_flags` 实际 hard-code tags/material path，不完全消费同一 config；label hash不能代替payload/IOV身份 |
| 哪个输出 | raw JSONL可哈希，部分旧 immutable audit有SHA | 本次正式audit缺失；B14M-T audit不对raw dump SHA做完整发布绑定 |
| 哪个 Condor | WB131：9291953、bigbird28、2 jobs | cluster主要是workbook文字；execution manifest没有逐attempt记录 |
| 哪个 denominator | raw 2680、ineligible691、contracted1989、official pairs1974 | 1989不是1989独立事件；须逐UID说明1974与1989差别，禁止成功行幸存者分母 |
| 哪些失败 | 本次48固定rows全部可见 | missing event/input/geometry可早返或skip；异常row可能不满足loader的optimize筛选条件 |

`ImmutableArtifactStore` 对通常单writer新run有价值，但 `COMPLETE.json` 没有自动列全 artifact hashes；`is_complete` 只看文件存在；`exists()` 后 `os.replace()` 也不是强原子 no-clobber 锁。更迫切的是 raw C++ `finalize` 使用 truncate，worker仅提前判断文件存在，submit plan可被覆盖，retry无独立attempt namespace。对 full sample 应先定义不可变attempt、失败记录、精确UID merge和显式supersession，不能靠扫描目录glob来恢复。

当前 evidence chain足以支持**范围限定的开发诊断和内部审阅**；不足以单靠 GitHub clone重现全部 scientific claims，更不足以支持“可用真实 alignment estimator”的论文主张。

## 9. Go / No-Go Decisions

| 行动 | 决定 | 理由/开放条件 |
|---|---|---|
| 继续 measurement-likelihood + profiling 研究 | GO | 修复了真实缺项、能明确分开统计/数值/信息问题，比paired residual目标更合理 |
| 现在跑1989 full sample | **NO-GO** | 0/48 stationary、outer未验证、certificate漏洞、provenance不完整 |
| 只读登记本次smoke FAIL | GO | 结果已落盘，可精确复算；新不可变判定不得改原常数 |
| geometry alignment solve | **NO-GO（当前真实样本）** | 2+3是track分区；全局geometry derivatives/gauge/coverage尚未建立；纯数学prototype可在独立toy上做 |
| external survey/metrology | GO（资料认证并行） | frame/IOV/covariance/相关性清单有独立价值；未认证数据不能进fit救数值失败 |
| Measurement Model V2 | 设计/证伪计划GO，正式替换NO-GO | optimizer误差与模型失配尚混杂；先固定V1并测残差结构，再依据新信息定义V2 |
| association uncertainty | 准备接口/小toy GO，真实性能claim NO-GO | 需要可信track likelihood及可比较候选 |
| joint association–alignment | NO-GO | 当前两层独立baseline均未齐备，joint优化会掩盖根因 |
| 重新训练ML/GNN | **NO-GO** | 没有证据显示瓶颈来自classical association表示能力；新网络不补measurement/gauge信息 |
| real-data geometry/conditions write | **NO-GO** | 无有效estimator、覆盖率与真实闭环认证 |

**冻结的历史路线：** fixed-residual WLS只保留paired-response control；rank-rescue、S/rank-threshold retuning、删source/删DoF追rank、V1/V2/V3/MLP扩容作为alignment主线永久停止。Frozen V2和route packing作为association controls保留，并非删除它们。WB124 vacuum DOPRI5、WB125 RK4+Mean失败配方不得再冒充production derivative参考；其代码/反例仍应保存。

“永久冻结”针对已经证伪的做法与解释，不意味着永远禁止在**新information source、不同objective、独立预注册验证**下研究相关技术。

## 10. Next Task

**唯一下一任务：B14M-U — 固定48-run失败收口与 profile numerical-oracle 一致性认证。** 任务的唯一问题是：当前 evaluator/derivative/TR/validity组合能否提供可信的局部 profiling certificate；不是尝试把现有FAIL调成PASS。

| 字段 | 具体要求 |
|---|---|
| 目标 | 保留本次Case C，堵住假PASS，并解释inner为何达不到预注册stationarity |
| 输入 | E1/E2固定dump、WB128三段、同四事件12identity，不增加样本/随机seed、不读target residual选算法 |
| 先修文件 | `datasets/b14m_profile_globalization_repair.py`及tests；TR小内核；C++ transport validity/recorder；不大改统计模型 |
| 数值实验 | 原C++解析quadratic/rank/小半径KKT；固定alpha analytic profile；nonlinear overshoot；rejected-state不变；branch/NaN失败注入 |
| 端点实验 | 固定0/T1、1/T1、37/T1/T2/T3、86/T1/T2的当前nominal endpoint，重复objective、全/保留/丢弃gradient、同方向production与shadow FD；不得仅测旧nominal seed |
| 必须记录 | 实际alpha/nu、J/W/H/Schur、eigenspectra、post-bound step、pred/actual/rho、parent trial、完整branch/material IDs、runtime hashes |
| PASS | 所有真假证书反例正确分类；解析数学oracle通过；对每个声明stationary点，独立error bound足以证明冻结gradient gate，且没有未解释的dropped descent/branch/transport失效 |
| 若FAIL | 返回明确分类：TR数学缺陷、derivative mismatch、oracle precision unresolved、boundary constrained、或inner budget insufficiency；不得混称multibasin |
| 禁止 | 改1e-6/1e-8/5%等旧gate追结果；拿四restart最小chi2当修复；增加更小line-search λ；改R、truth q/p、target选盆地、提交1989 |

若当前 oracle 的确定误差界确实高于目标 precision，应先改可证明的numerical implementation。只有独立误差预算和物理精度需求共同支持，才可另立明确标记的新数值合同；旧 B14M-T 仍是 FAIL，不能回填通过。

## 11. Next 3 Tasks

这也回答“导师会让学生接下来两周只做哪三件事”。三个任务依赖有序；后项不得绕过前项失败。

| 顺序 | 任务 | 交付与决策 |
|---|---|---|
| 1，约第1–4天 | **Certificate + numerical kernel hardening** | 登记本次固定smoke失败；P0-1/2最小修复；C++数学golden tests与Python/C++合同一致；所有missing/duplicate/branch/NaN反例fail closed |
| 2，约第5–9天 | **Endpoint oracle/derivative certification** | 对上述冻结端点测objective precision、material/normalization/branch导数；补WB128 production ±arm；判断37是否只是预算、0/1/86是否是derivative/oracle floor。必要修复限已证明缺项 |
| 3，约第10–14天，前两项通过后 | **完整48-run recontract** | 12/12 identities、4/4 restarts完整；inner与outer分别stationary；全局/丢弃梯度、alpha/objective/prediction/branch各gate独立。全部通过才交付full-sample preflight规格，不自动提交1989 |

若第2项未过，第3项变成记录明确阻塞与最小补证，不以赶两周节点为由执行新campaign。外部survey资料认证可由已有资料只读整理，但不挤占这三项核心工作，也不联系他人或引入未认证prior。

## 12. 1–2 Month Roadmap

### Phase 1：数值与证书闭合（第1–2周）

- **目标/blocker：** 见上面三任务，解决inner全FAIL及certificate scope。
- **文件：** 当前 `.inc`、evaluator、Python audit、最小纯数值test入口。
- **运行：** toy/math先行，再固定4事件48-run。
- **PASS：** 每条有效解都带可验证stationarity/transport/domain证书，完整restart gates通过；无默认0误读。
- **FAIL去向：** 只修证明了的numerics/derivative缺项，或给出precision不可达的明确证据；不拓展统计模型来遮蔽失败。
- **禁止：** 全样本、alignment、模型训练。

### Phase 2：full-sample preflight与冻结V1 population诊断（第3–4周，条件开启）

- **目标：** 证明smoke以外仍可运行，并量化V1模型的有效域，不能仅统计成功样本。
- **blocker：** 1989 contracted rows 的exact UID、原始2680/排除691/official1974关系、runtime pins、failure accounting。
- **文件：** 新preflight config/manifest、source-local runner、artifact store/aggregator；不重用“只为2 jobs写死”的B14M-T submit脚本冒充通用生产。
- **运行：** 冻结输入manifest、资源上限、分层诊断和失败政策；预先规定confirmatory subset。preflight过后才跑1989，失败/无输出仍计入分母。
- **PASS：** 100% attempted identities有唯一有效输出或明确失败；零未解释schema/provenance/selection损坏；solver适用率按source/charge/momentum/coverage报告。科学coverage/偏差criterion必须在查看新结果前注册，不能事后挑幸存分母。
- **FAIL去向：** numerical/domain failure回Phase1；已收敛但结构性residual/coverage失败进入受控measurement-model诊断；统计不足用功效预算判断，不无限加样本。
- **禁止：** 将1989完成率、chi2降低或local Hessian可逆称作真实alignment成功。

### Phase 3：measurement model adequacy与外信息可用性（第4–6周，可资料并行）

- **目标：** 把“优化正确”和“统计模型正确”分开。
- **blocker：** 固定R与真实cluster误差/散射、能损涨落、field/material系统差异；survey covariance/frame/IOV缺口。
- **文件：** measurement adapter、calibration/whitening diagnostics、显式model-version config；survey factor接口只接受认证输入。
- **运行：** source/angle/momentum分层的whitened residual correlations、predictive intervals/coverage、target-blind seed和association对照。若有证据需要V2，增加来自物理/独立calibration的scattering或measurement因子，保留V1control。
- **PASS：** error bars校准、相关结构有模型解释、bias满足预注册物理预算；外约束有可追溯mean/covariance/frame/pivot/IOV且与validation数据独立。
- **FAIL去向：** 清晰区分measurement misspecification、缺外信息、源域不可转移；退回monitoring/受限MC诊断。禁止经验rescale使chi2≈1就算通过。
- **禁止：** 把survey组装精度当逐DoF独立Gaussian prior，或用prior吸收optimizer错误。

### Phase 4：小规模global geometry likelihood（第6–8周，严格条件开启）

- **目标：** 第一次建立真正geometry参数gamma和track-local nuisance的joint/profile等价、gauge与no-oracle closure。
- **blocker：** geometry derivatives、真实gauge basis、合格外因子、可信track likelihood。
- **文件：** `MeasurementLikelihood`接口、geometry derivative provider、global/local Schur组装、survey factor、shadow geometry/refit driver；以现有physical chain为adapter，避免重写reconstruction。
- **运行：** 先真值association控制（truth只确定对应关系和评估，不提供q/p/geometry解）；known perturbation closure必须用measurement likelihood自身求解，不减同事件reference residual。比较direct joint与Schur、profile intervals与empirical coverage；之后才接frozen association control。
- **PASS：** anchor/gauge一致、对扰动的recovery与coverage均达预注册预算、无truth-q/p、无reference-oracle、完整failure accounting、held-out只评估。
- **FAIL去向：** 如果track-only方向弱且有合格survey，进入明确受约束分支；若没有可用信息，停止该部署主张，保留monitoring。不得用DoF删减/更大网络救rank。
- **禁止：** 写official geometry；把MC闭环直接外推到real data。

月度成功不要求完成alignment，而要求每个blocker都有可证伪的判定、失败证据保留、没有未经证据授权的路线扩张。

## 13. Longer-Term Research Direction

### Q1：profile likelihood + nuisance elimination 是正确核心吗？

**是，作为可验证的classical inference核心值得继续。** 它让measurement information、track自由度、geometry gauge、survey信息能在同一目标中清楚分离；Schur消元是计算方法，不凭空创造信息。当前错误主要是数值证书和模型完整性，不能由这些错误推导“profile框架不合适”。[X2]

但 profiling 不是唯一最终方案。若将scattering、energy-loss或association作为随机潜变量，marginal likelihood可能比单纯MAP/profile更适合某些coverage问题；必须清楚处理归一化、先验来源和integration approximation，不能把两者混称。参数依赖R/Q或积分后的有效covariance通常需要相应log-determinant，不能只留下平方残差项。

### Q2：当前2+3 parameterization自然吗？

它适合追踪当前LTO中较支持和较弱的track坐标，不是最终alignment模型。phi/theta在接近beam轴时可能有chart-conditioning问题；loc/角的支持性也随source/field/coverage而变。建议先在**同一objective、同一domain**下比较ACTS bound chart与局部direction tangent或slope chart的等价性和误差预算；这属于数值坐标检验，不是删除q/p或重新筛选可过门的DoF。

最终几何参数使用明确的station/module SE(3)、pivot和IOV；每条track的所有必要local参数进入nuisance，不能把现有alpha当成mechanical displacement。

### Q3：optimizer稳定后最值得做哪一个？

**先做1989 full-sample preflight，再做冻结V1 full-sample诊断。** 它最直接检验small-smoke是否可推广，并为measurement adequacy提供完整分母。外部metrology的资料认证应并行，因为它是独立的信息依赖。接下来优先顺序由证据决定：若V1 calibrated，进入带gauge的小global alignment；若V1 residual/coverage明显失配，先Measurement Model V2。association uncertainty在可信likelihood后接入；joint association–alignment最后。

这不是建议等待所有外约束完美才运行track diagnostics，也不是建议用external prior修当前optimization。

### Q4：何时才有理由重新引入ML/GNN？

必须先存在一个**量化的classical baseline残余问题**：例如高occupancy下候选组合爆炸、冻结candidate generation漏掉真track、模型未使用的cluster-shape/timing信息、或需要可校准的背景/缺失概率。ML可作为proposal/ranking或amortized inference，最终仍由明确likelihood、capacity/shared-hit规则和uncertainty calibration审查。

开启条件：物理likelihood和uncertainty已可信；错误分解表明主损失来自association而非transport/measurement/gauge；新增information或objective明确；冻结资源相同的classical baseline与独立source-disjoint evaluation。没有新information/objective/uncertainty处理，只换更大网络，不值得做。

### Q5：最大的methodological risk是什么？

目前是**以不足的数值证书宣称求得profile optimum，然后把该optimum的local curvature当成物理可辨识和真实precision**。即便堵住该风险，measurement covariance/随机材料效应、target-conditioned selection与simulation-to-real差异仍可能主导bias和coverage。geometry gauge与external constraints是全局alignment不可回避的下一层；association是重要误差源，但当前证据未把它提升为首要blocker。

### Q6：两周只做哪三件事？

就是第11节三件事：**证书与数值内核；端点导数/精度；固定48-run再认证。** 不增加第四条训练路线，不把失败隐藏到新编号campaign。

长远最可信的研究贡献是：在真实FASER geometry/field/material/selection约束下，建立具有明确适用域、uncertainty calibration、association-induced bias测量和shadow闭环的alignment methodology。FASER tracker文献可提供探测器结构和metrology背景，但设计/组装指标不能替代本数据IOV的实际校准因子。[X3]

## 14. Files / Functions That Need Attention

以下行号针对 `c749e45`。引用具体function优先于单一行号，防止后续重构移动位置。

| 文件 | 函数/位置 | 要点 |
|---|---|---|
| `alignment/leave_target_out_dump/ProfileGlobalizationRepair.inc` | `rangeSpaceTrustRegionStep`，约41–138 | PSD/finite/KKT、小Delta secular solve、返回真实有效状态 |
| 同上 | `extractNuBlocks/extractSchurAlpha`，约178–232 | 缩放和rank semantics、full/drop-gradient、GN vs exact profile命名 |
| 同上 | `applyNuDelta/applyAlphaDelta`，约234–264 | bound clipping、实际step与rho一致 |
| 同上 | `profileNuisanceTR`，约266–437 | inner精度、失败分类、accepted-state日志、Delta_min |
| 同上 | `runExplicitProfileTR`，约439–694 | final sys.ok、joint/profile stationarity、完整inner trace与trial计数 |
| 同上 | `profileGlobalizationRow`，约696–783 | 未计算值null、profile success≠eval success、J/H/branch证据 |
| 同上 | `runProfileGlobalizationRepair`，785–835 | 执行常数与config一致、固定seed身份 |
| `alignment/leave_target_out_dump/ProfileBasinDiagnosis.inc` | `symmetricPinv3/evaluateScaledSystem`，12、153 | 不能把diagnostic inc当无检验production依赖；finite/rank/derivative status |
| 同上 | `frozenLineSearchReplayJson/runNuisanceOnly/runBasinDiagnosis` | legacy stop规则保留但隔离；单条continuation不证明全局单盆地；历史predicted_decrease字段不是B14M-T的完整模型下降 |
| `alignment/leave_target_out_dump/CkfLeaveTargetOutDumpAlg.cxx` | `seedFromOfficialMean`，366 | full-track seed来源、source-plane chart/target independence |
| 同上 | `applyPhysicalBounds`，562 | 参数域、裁剪与constrained stationarity |
| 同上 | `projectSupportingPlane/propagateOneHit`，1044/1087 | projection与transport validity分离、branch/fallback/finite |
| 同上 | `evaluateSequentialResiduals`，1290 | 实际hit分母、R有效性、finite objective、失败传播 |
| 同上 | `FreeTransportJacobianCollector`，1932 | RK reset累计、material/normalization/event-surface chain |
| 同上 | `propagateOfficialPathJacobianHop`，2073 | 错误仍ok、mean-vs-derivative path一致性 |
| 同上 | `certifiedMeanCommonGridFdJson`，3301 | production ±arm补证，不能只比较复制ledger |
| 同上 | `repairedSourceLoc0Jacobian`，5476 | 完整source→measurement chain、endpoint精度 |
| 同上 | `runOneProfileNumerics`，5570 | 旧flat_direction成功路径保留为明确control |
| 同上 | `execute/finalize`，6172/6879 | exact UID、geometry lookup失败计数、exception rows、atomic output |
| `alignment/leave_target_out_dump/CkfLeaveTargetOutDumpAlg.h` | Gaudi properties | 合同/flags爆炸、状态模式互斥与执行provenance |
| `alignment/leave_target_out_dump/FieldGradientDefaultExtension.hpp` | `officialGradient/transportMatrix` | gradient error、normalization与完整tangent范围 |
| `alignment/leave_target_out_dump/MeanTransportContract.hpp` | `qopAfterEnergyLoss/applyProductionEnergyLoss/replay` | nominal材料ledger vs perturbed event/map、Bethe精度与gating |
| `alignment/leave_target_out_dump/CommonGridShadowIntegrator.hpp` | `applyEnergyLossMean/integrate` | 已失败历史RK4+Mean参考，保持冻结，禁止作为production C复用 |
| `alignment/leave_target_out_dump/IndependentMeanOdeIntegrator.hpp` | `integrateToPlane/variationalRhs` | vacuum ODE control，不是含材料production参考 |
| `datasets/b14m_profile_globalization_repair.py` | `_row_stationary/_row_valid/audit_identity/inventory_and_audit` | 本报告P0-1的核心 |
| `datasets/b14m_restart_invariance.py` | `_chi2/_pred_meas/_branch/recover_optimizer_contract` | fallback/缺失值、完整branch定义、跨版本threshold |
| `alignment/profiled_measurement_likelihood.py` | `symmetric_pinv/null_basis/prediction_uncertainty_from_hessian` | 跨语言rank差异；保留null-aware uncertainty优点 |
| `scripts/dump_ckf_leave_target_out.py` | `_hashes/build_flags/main` | config标签与执行标签统一、实际runtime/input/output pins |
| `scripts/run_b14m_profile_globalization_repair.py` | `main` | 只是只读preflight，不是实际solver runner；命名/状态导航明确 |
| `scripts/audit_b14m_profile_globalization_repair.py` | `main` | 原始dump哈希/唯一attempt、执行合同验证、不可自行宣告true |
| `scripts/submit_b14m_profile_globalization_repair_condor.py` | `main` | mutable worker/plugin、重复submit plan、cluster/attempt provenance |
| `scripts/run_b14m_profile_globalization_repair_condor.sh` | worker | snapshot binary/config、禁止覆盖、独立attempt、source精确验证 |
| `configs/b14m_profile_globalization_repair_v1.yaml` | inherited contracts + active section | 1920行拷贝式合同；保留旧版本，新版本resolved execution capsule |
| `tests/test_b14m_profile_globalization_repair.py` | 全部8项 | 增加实际numerical kernels和证书反例，不只token/schema |
| `evaluation/artifact_store.py` | `write_json_atomic/finalize/is_complete` | per-artifact hash、no-clobber、completed validation |
| `docs/CODE_ROADMAP.md`, README, canonical docs | current-state入口 | 新建reconciliation指向本次HEAD；历史审阅不回填 |

### 14.1 可复算记录与最小命令

本次只新增审阅Markdown；没有改动实现、config或tests，没有提交Condor、重建plugin、写geometry、训练、提交Git commit或发布外站。临时数学探针和只读audit中间结果在 `/tmp/faser-master-review-c749e45/`，不作为永久证据依赖。

测试复现：

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_110_cuda/x86_64-el9-gcc13-opt/setup.sh
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider \
  tests/test_b14m_profile_globalization_repair.py \
  tests/test_b14m_profile_basin_diagnosis.py \
  tests/test_b14m_restart_invariance.py \
  tests/test_certified_mean_common_grid_fd.py \
  tests/test_surface_energy_loss_mean_semantics.py \
  tests/test_field_gradient_variational_repair.py \
  tests/test_profiled_information.py \
  tests/test_profiled_weak_nuisance_likelihood.py
```

只读正式函数复算，不调用会创建official audit artifact的script：

```python
from datasets.b14m_profile_globalization_repair import (
    load_config, inherit_frozen_stage, inventory_and_audit, decide,
)
from datasets.b14m_restart_invariance import recover_optimizer_contract
c = load_config()
inherited = inherit_frozen_stage(c)
contract = recover_optimizer_contract(c)
inventory = inventory_and_audit(c, contract)
decision = decide(inventory, inherited, contract)
print(inventory['loaded'], inventory['complete_48'])
print(decision['decision'], decision['verdict'])
# 60 rows, 48 optimize, True
# inner_nuisance_profile_not_converged FAIL
```

可复现的最小假PASS：

```python
import runpy
from datasets.b14m_profile_globalization_repair import audit_identity
fixture = runpy.run_path('tests/test_b14m_profile_globalization_repair.py')
rows = fixture['_four']()
rows[0]['predicted_target_loc0'] = None
contract = dict(chi2_rel_tolerance=0.01, prediction_abs_tolerance_mm=0.1,
                supported_abs_tolerance={'loc0_mm': 0.05, 'theta': 1e-4})
a = audit_identity((100043, 37, 3), rows, contract, {'gradient_norm_z': 1e-6})
print(a['case'], a['prediction_invariance'])
# profile_globalization_and_restart_contract_established True
```

C++内核反例复现方式：从固定SHA的 `ProfileGlobalizationRepair.inc` 取第一个 `template <int N>` 至 `struct ProfileSolveResult` 之前的原文（即struct+function）；加 `Eigen/Dense,array,cmath,algorithm` headers，以 `g++ -O2 -I/usr/include/eigen3` 编译，传入第4.5/4.6节的矩阵和向量。独立TR oracle在lambda≥0上二分求 `||(H+lambda I)^−1g||=Delta`，检查KKT与原始模型下降。无需Athena或事件内容，不修改冻结阈值。

### 14.2 关键内容哈希

| 内容 | SHA256 |
|---|---|
| HEAD `CkfLeaveTargetOutDumpAlg.cxx` | `34f0edc6f197e1677b3261de38e02a25f2623a833bb9e8898ee79d6d8eea5671` |
| HEAD `ProfileGlobalizationRepair.inc` | `805ce3a8338aadd0a696859ebaf078c40dbaa443f42c145626e16d7aadc147bf` |
| HEAD `ProfileBasinDiagnosis.inc` | `c9489457aaee5bf7509ee1273ef5e24255eda822e9bf3f8099eff69b11525581` |
| HEAD `FieldGradientDefaultExtension.hpp` | `ca5e4f0ef1a7a09edd1c24099e7ce689c4f0511e2d4afab3a160ebd2d6ff03d8` |
| HEAD `MeanTransportContract.hpp` | `8f9d162e59f590770baba1f0c393a7f10861a85495dd7bfbac676d8b59bd15da` |
| HEAD B14M-T YAML | `395289a77b6a78d4533ab34ef12eb6617f5aba812922486c3493268cce0d6fc4` |
| HEAD B14M-T Python audit module | `0d565319cfb22f98688ba8a1e2a564fd1c114dc3f52fb20bbcd51920b85d280b` |
| 当前plugin，与WB131提交记录相符 | `14c93af3db23beac3035dc343f8ce3bf51c85252918aa7f54cb70c563be188f6` |
| E1：100043 smoke JSONL | `acdffc04bbf57aae1a7cfabb91220d762bdc563ab337cc34e10923aefdaeffce` |
| E2：100048 smoke JSONL | `0e8633fd8fdc46665e5275d1a0c349a640a6397603657ace0d4e5a75432ec400` |
| E3：WB128 tangent-vs-shadow artifact | `83a9a7fb7e8c54cea3d4b7d5006088c127df84ad0dcaae507ff6884c3ade12dc` |
| WB127 decision | `aa80751fb532cf36350a8cf78b25f9d74da5bc707b9739ba7b683ef294fe04c4` |
| WB128 decision | `7f9d411c31d3eb94d63fb13e24f1ebd57b3095f0007039a5c3048704007141b1` |
| WB129 decision | `67aa191def9863d39dec6643bf117516aa95039d6a90e0e38a09ac5b161059ea` |
| WB130 decision | `cb8d178ea67870ddf04bf6babd2957f5be6c29cb0709d31b35696ac535904d91` |

### 14.3 Sources / evidence index

代码来源均为 Endymion2288/alignment_ML 固定 `c749e45`，访问日期2026-09-11；历史workbook作为provenance，不作为独立数学证明。代码引用用固定commit链接，避免master继续变化后误指。

- [S1] [GitHub current master commit](https://github.com/Endymion2288/alignment_ML/commit/c749e45aac9157ab59ab672f9527cafa6234dc46)。实时branch和recent commits API核实；[parent checkpoint](https://github.com/Endymion2288/alignment_ML/commit/55cf982302a3c62c57b74f368d5e3ba7723fd33a)。
- [S2] [固定HEAD workbook tree](https://github.com/Endymion2288/alignment_ML/tree/c749e45aac9157ab59ab672f9527cafa6234dc46/workbook)，重点WB117–131；[WB131本地正文](2026-09-10_131_TaskB14M-T_Profile_Globalization修复与Stationarity再合同.md)。
- [S3] [ProfileGlobalizationRepair.inc](https://github.com/Endymion2288/alignment_ML/blob/c749e45aac9157ab59ab672f9527cafa6234dc46/alignment/leave_target_out_dump/ProfileGlobalizationRepair.inc)。
- [S4] [ProfileBasinDiagnosis.inc](https://github.com/Endymion2288/alignment_ML/blob/c749e45aac9157ab59ab672f9527cafa6234dc46/alignment/leave_target_out_dump/ProfileBasinDiagnosis.inc)。
- [S5] [CkfLeaveTargetOutDumpAlg.cxx](https://github.com/Endymion2288/alignment_ML/blob/c749e45aac9157ab59ab672f9527cafa6234dc46/alignment/leave_target_out_dump/CkfLeaveTargetOutDumpAlg.cxx)，含measurement/transport/Jacobian/entry/finalize关键调用链。
- [S6] [MeanTransportContract.hpp](https://github.com/Endymion2288/alignment_ML/blob/c749e45aac9157ab59ab672f9527cafa6234dc46/alignment/leave_target_out_dump/MeanTransportContract.hpp)。
- [S7] [FieldGradientDefaultExtension.hpp](https://github.com/Endymion2288/alignment_ML/blob/c749e45aac9157ab59ab672f9527cafa6234dc46/alignment/leave_target_out_dump/FieldGradientDefaultExtension.hpp)。
- [S8] [B14M-T Python classification/audit](https://github.com/Endymion2288/alignment_ML/blob/c749e45aac9157ab59ab672f9527cafa6234dc46/datasets/b14m_profile_globalization_repair.py)。
- [S9] [B14M-T tests](https://github.com/Endymion2288/alignment_ML/blob/c749e45aac9157ab59ab672f9527cafa6234dc46/tests/test_b14m_profile_globalization_repair.py)。
- [S10] [Profiled-information tests](https://github.com/Endymion2288/alignment_ML/blob/c749e45aac9157ab59ab672f9527cafa6234dc46/tests/test_profiled_information.py)。
- [S11] [Profiled measurement likelihood numerical helpers](https://github.com/Endymion2288/alignment_ML/blob/c749e45aac9157ab59ab672f9527cafa6234dc46/alignment/profiled_measurement_likelihood.py)；[其线性与null-aware tests](https://github.com/Endymion2288/alignment_ML/blob/c749e45aac9157ab59ab672f9527cafa6234dc46/tests/test_profiled_weak_nuisance_likelihood.py)。
- [S12] [当前bootstrap实现](https://github.com/Endymion2288/alignment_ML/blob/c749e45aac9157ab59ab672f9527cafa6234dc46/alignment/tracker_only_identifiable_subspace.py)；[当前physical covariance检查](https://github.com/Endymion2288/alignment_ML/blob/c749e45aac9157ab59ab672f9527cafa6234dc46/alignment/physical_jacobian.py)。
- [S13] [B14M-T YAML](https://github.com/Endymion2288/alignment_ML/blob/c749e45aac9157ab59ab672f9527cafa6234dc46/configs/b14m_profile_globalization_repair_v1.yaml)；[dump入口](https://github.com/Endymion2288/alignment_ML/blob/c749e45aac9157ab59ab672f9527cafa6234dc46/scripts/dump_ckf_leave_target_out.py)；[artifact store](https://github.com/Endymion2288/alignment_ML/blob/c749e45aac9157ab59ab672f9527cafa6234dc46/evaluation/artifact_store.py)。
- [E1] 本地raw evidence：[100043 part](../outputs/leave_target_out_dump_v1/b14mt_repair_smoke/mc24_100043_00400_00499/part_evt0_1_37.jsonl)。45 rows，约1.76MB；GitHub未跟踪。
- [E2] 本地raw evidence：[100048 part](../outputs/leave_target_out_dump_v1/b14mt_repair_smoke/mc24_100048_00000_00049/part_evt86.jsonl)。15 rows，约0.71MB；GitHub未跟踪。
- [E3] 本地WB128：[repaired tangent vs certified shadow](../outputs/certified_mean_common_grid_fd_v1/sbb14zc_certified_mean_fd_20260909T165606Z_5bd09dfe/repaired_tangent_vs_certified_shadow.json)。保存numerical values本轮独立复算。
- [E4] 已实际读取的runtime source/binary：`/cvmfs/atlas.cern.ch/repo/sw/software/24.0/AthenaExternals/24.0.41/InstallArea/x86_64-el9-gcc13-opt/` 下 `include/Acts/Propagator/{EigenStepper.ipp,MaterialInteractor.hpp,detail/GenericDefaultExtension.hpp,detail/PointwiseMaterialInteraction.hpp}` 与 `lib/libActsCore.so`；以及工作区Calypso的 `FASERMagneticFieldWrapper.h`。本轮确认当前文件行为，不声称已恢复每次历史作业实际loaded-libraries完整快照。
- [X1] ACTS project，[GenericDefaultExtension.hpp, v32.0.2](https://github.com/acts-project/acts/blob/v32.0.2/Core/include/Acts/Propagator/detail/GenericDefaultExtension.hpp)，与pinned default field tangent问题有关；不作为其他release的一般性结论。
- [X2] V. Blobel/C. Kleinwort等，[Millepede II Draft Manual](https://www.desy.de/~kleinwrt/MP2/doc/html/draftman_page.html)，global/local参数消元和alignment框架的一手方法参考。
- [X3] FASER Collaboration，[The tracking detector of the FASER experiment](https://arxiv.org/abs/2112.01116)，探测器结构与metrology背景的一手论文；不替代本项目实际IOV校准。

证据仍未完成的决定性工作：完整production perturbed-arm/endpoint derivative实验、实际C++ inner/outer integration数学测试、真实uncertainty calibration、无target-dependent seed/selection闭环、global geometry/gauge与合格外约束、独立real-data shadow validation。这些是明确的下一阶段gate，本报告没有把它们暗中视为完成。
