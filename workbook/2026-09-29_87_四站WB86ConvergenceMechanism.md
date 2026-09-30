# Workbook 87 — WB86 convergence mechanism, retrospective autopsy

日期： 2026-09-29
分支： `4station`
HEAD at inspection： `eea34212265e95060f1facf9bb744e104e5e2958`
任务： 只读机制研究。解释 WB86 官方非线性 common-track 迭代为什么在几乎全部 official-chain 事件上不收敛。不重跑 WB86，不修改 WB86 判决，不把任何新结果叫作 WB86。

```text
WB86 outcome is frozen and remains FAIL.
This study does not requalify WB86.
This is a prospective post-failure mechanism study.
ml_alignment_eval_authorized = false
```

---

## 0. 冻结的 WB86 事实

```text
execution_qualification   = FAIL
statistical_qualification = null
physics_screening         = FAIL
overall_qualification     = FAIL
alignment_oracle_qualified_for_physical_FASER = false
```

`T_LS` 与 `T_cov` 没有计算。不得把 WB85b 写成统计 FAIL。

2392 个 official-chain SUCCESS 事件：

```text
max_iterations = 2352
converged      = 40
WB86Error      = 2
```

主因是 nonconvergence。`solver_status = ok` 很常见，但冻结规则要求

```text
scaled_update_norm < 1e-3
AND
|chi2_k - chi2_{k-1}| / |chi2_{k-1}| < 1e-3
```

连续 `CONSECUTIVE_REQUIRED = 2` 次，且 `MAX_ITERATIONS = 10`。到达 10 次不是 PASS。

---

## 1. 已有产物里没有迭代轨迹

WB86 worker 只落盘最终 sealed chart 和 technical 计数。`iterate_official_event` 在内存里建了 history，返回后没有写入 `technical.json` / `sealed/*.json`。Condor scratch

```text
/pool/condor/dir_*/tmp/wb86_event_*
```

在 2026-09-29 已不存在。shard stdout 只有 stage 头尾，没有逐步 chi2。

因此本轮不能画出 iteration-versus-scaled-update 的实测曲线，也不能回答 “k>10 之后会不会收敛”。没有 k>10 的冻结轨迹，就不推断。

能用的冻结量是：

```text
最终 chart theta_hat, theta_true
最后一次 solve 的 n_dropped / solver_status / sign_frame
technical.n_official_refits
technical.iterations
```

`n_official_refits` 只在 payload cache miss 时加一。它是几何是否重复的计数器，不是科学残差。

---

## 2. 几何预算分类

一次迭代若每次都产生新几何，官方 backend 的 cache 给出确定的 refit 数。`P` 是 chart 维数。

```text
plateau after iteration 1     = P + 2
full unique through iteration 10 = 10 P + 11
```

`P+2` 是：名义点 1 次，P 个有限差分，1 次更新后的 rerefit；第 2 步起预测又落回同一 payload。`10P+11` 是 10 步里每一步的差分几何和更新几何都没有撞上 cache。介于两者之间表示 10 步之前已经出现重复几何。

| stratum | P | n | plateau `P+2` | repeated before 10 | full unique `10P+11` | no solve |
|---|---:|---:|---:|---:|---:|---:|
| identity_fixed_dz | 6 | 299 | 299 | 0 | 0 | 0 |
| identity_finite_survey_prior | 9 | 303 | 302 | 0 | 0 | 1 |
| identifiable_translation_fixed_dz | 6 | 311 | 0 | 46 | 264 | 1 |
| identifiable_translation_finite_survey_prior | 9 | 298 | 0 | 4 | 294 | 0 |
| identifiable_rotation_fixed_dz | 2 | 300 | 0 | 1 | 299 | 0 |
| identifiable_rotation_finite_survey_prior | 4 | 283 | 0 | 1 | 282 | 0 |
| weak_jg_diagnostic_fixed_dz | 2 | 302 | 0 | 19 | 283 | 0 |
| weak_jg_diagnostic_finite_survey_prior | 3 | 298 | 0 | 4 | 294 | 0 |

合计：

```text
plateau_after_iteration_1              = 601
repeated_geometry_before_iteration_10  = 75
full_unique_through_iteration_10       = 1716
no_solve                               = 2
```

图：`outputs/mc24_four_station_wb87_convergence_autopsy_v1/figures/wb86_geometry_budget_by_stratum.png`

这不是收敛阈值扫描。阈值仍是 `1e-3` / `1e-3` / 连续 2 次。

---

## 3. 形态，用最终 chart 对照 refit 类

最终 chart 的 L2 用 WB86 同一套 chart 坐标：平移 mm，转动 mrad。

### A. 恒等层：更新停了，判据没有停

601 个 plateau 事件全部是两个 identity stratum，最终 chart 与 truth 逐分量相等，L2 = 0。

代表：index 0，`identity_fixed_dz`，10 次迭代，8 次 refit，`solver_status = ok`，`theta_hat = 0`。

这不是慢收敛，也不是振荡证据能证明的那种发散。几何在第 1 次更新之后不再变。冻结判据仍然失败，因为 `validation_relative_change` 要有上一次 chi2。chi2 一旦到 0，下一次相对变化的分母按代码写成

```text
abs(previous_chi2) > 0
```

不成立，`rel` 保持 `None`，连续计数清零。再下一步分母仍是 0，于是 10 步全部失败。恒等事件正好是 chi2 可以到 0 的层。

index 510 的 `n_dropped = 3` 与最终 `theta_hat = 0` 同时出现。sealed 的 `n_dropped` 只保留最后一次 solve。最后一次几何已经回到名义点，dropped flag 不能证明某个物理参数在十步里一直被丢掉。这是迭代依赖的最后一次秩标记，不是一条可恢复的奇异向量。WB86 没有保存右奇异向量、奇异值或被丢的参数名。

### B. 非零注入层：几何一直在变，终点几乎就是真值

1716 个 full-unique 事件里，绝大多数最终 |chart error| 远小于 `1e-6`：

```text
translation fixed / finite     median L2 ~ 1e-13
rotation fixed / finite        median L2 ~ 1e-11
weak fixed                     median L2 ~ 1e-11
```

代表：

```text
index 105  translation fixed     refits 71   L2 2.6e-12
index 151  translation finite    refits 101  L2 8.3e-13
```

`10P+11` 的意思是第 10 步的更新几何仍然是新的。所以这些事件不是 “停在一个远离真值的平台上”。终点已经在真值的数值噪声里，但 10 步之内没有出现连续两次 “步长 < 1e-3 且 chi2 相对变化 < 1e-3”。

没有逐步 chi2，就不能把这一类再分成单调、两周期或过冲。能排除的是：它们不是被 `k=10` 截断的、肉眼可见的大残差慢爬。终点残差已经在 `1e-11`–`1e-13` chart 单位。失败发生在判据，或发生在判据所看的 chi2 序列，而不是发生在最终参数离真值很远。

75 个 “10 步前就重复几何” 的事件同样几乎都在这个数值地板上，包括 32 个被标成 converged 的成功事件。重复几何与 converged 标签相容，因为连续两次小步会撞上已经到过的 payload。

### C. 少数大终点：不是主样本

最终 chart L2 > 0.05 的事件只有 19 个。它们解释不了 98% 的 nonconvergence。

旋转 fixed：

```text
index 239  max_iterations  sign_frame=true
  s1_rz = -103.13 mrad    truth s1 = -0.5, truth s3 = +0.5
  s3_rz error ~ 1e-11
  refits = 31 = full unique

index 247  max_iterations  sign_frame=false
  s1_rz error = +33.69 mrad
  s3_rz error = +12.43 mrad
  refits = 31

index 657  converged, analyzable, the WB86 z = -496 event
  s1_rz error ~ 1e-11
  s3_rz error = -63.11 mrad
  a_hat = -43.918 mrad, a_true = +0.707 mrad, sigma_a = 0.090 mrad
  refits = 28, so the geometry repeated before iteration 10
```

投影 `u` 是 `s1_rz = -0.5` 与 `s3_rz = +0.5` 的单位方向。index 657 把约 63 mrad 放在 S3，S1 仍在真值上。这不是整支旋转的全局符号反号，也不是 `a_hat` 相对 `a_true` 的一次整体换号。它是单站、大幅、错误幅度，另一站已经对齐。index 239 把约 103 mrad 放在 S1，S3 已对齐，并且触发了 `sign_frame_failure`（`|a_hat| > 50 * max(1, |a_true|)`）。index 247 两站都偏，且与真值同号。三种形态同时存在，所以不能把旋转失败收成一条符号约定错误。

旋转 finite 的大事件是 index 1104（converged 标签，`n_dropped = 2`，S1 `rz` 误差 0.530 mrad）和 index 2285（`n_dropped = 2`，S3 `rz` 误差 -27.3 mrad）。dropped 与大旋转绑在最后一次 solve 上，仍然没有逐次奇异向量。

平移大终点是个位数事件，例如 index 1335 的 S1 `dx` 误差 -37.6 mm，index 1805 的 S2 `dx` 误差 +21.8 mm。它们 full-unique，所以更新一直在走，不是平台。

弱 JG fixed 的 index 1586：`s3_dx = -153.4 mm`，`s3_ry = +65.2 mrad`，`sign_frame_failure = true`。这是弱模式的个例发散，不是弱层的中位行为。弱层中位 L2 仍是 `1e-11`。

### D. 弱 finite 的 dz 地板

`weak_jg_diagnostic_finite_survey_prior` 的 298 个事件里，293 个 `s3_dz` 误差正好是 `-2.0e-5 mm`。`s3_dx` 与 `s3_ry` 仍在 `1e-11` 量级。代表 index 358，最终 L2 = `2.0e-5`，refits = 41，full unique。

这个数等于有限差分步长。`physical_finite_difference_blocks` 对每个 chart 分量做一步 left-SE(3)，步长

```text
FD_CHART_STEP = 1.0e-2
```

chart 里 `dz_mm` 的 0.01 mm，经 `se3_exp` 小角公式

```text
V = I + 0.5 [omega]^
```

会在平移槽留下与转角步长同量级的耦合。转动步长是 `1e-5 rad`。`0.5 * 1e-5 = 5e-6` 不是 `2e-5`，所以不能把 `-2e-5` 写成已经证明的 SE(3) 耦合公式。能写的是：这个地板事件间重复、大小等于差分步长的量级，并且只出现在带 `dz` 的弱 chart 上。平移 finite chart 的 `dz` 中位误差是 `1e-22`，没有这层地板。所以它是弱 chart 的差分/更新残留，不是全体 finite-survey 的结构奇异。

弱层因此几乎不可能满足 `scaled_update_norm < 1e-3`。`dz` 的 scale 是 5 mm，但若每步仍在写 `2e-5 mm` 量级的新几何，cache 不命中，10 步全部算作新几何。这与 294/298 full-unique 一致。

---

## 4. 假设对照

| 假设 | 与冻结计数的关系 |
|---|---|
| A. 收敛判据语义 | 恒等层 601/601 被它单独解释：几何已停、终点是真值、相对 chi2 在 0 上没有定义 |
| B. 步长过冲 | 19 个大终点支持存在过冲个例；解释不了 1700 个终点已在 `1e-12` 的 full-unique 事件 |
| C. 重线性化不一致 | 未测。需要逐步 `J delta` 与物理预测差 |
| D. Jacobian 信任域 | 弱 finite 的 `2e-5 mm` dz 地板指向差分步长；尚未用 rho 定量 |
| E. 规范/冗余耦合 | 恒等层不是这个。弱 JG 是已知弱 chart，但中位终点仍然贴真值 |
| F. left-SE(3) 复合 | 大旋转个例要查，但三件事站、同号事件同时存在，不能先改符号 |
| G. validation 残差定义 | 与 A 是同一条代码路径。validation 用的是 fit chi2 的相对变化，不是独立验证样本 |
| H. Calypso/ACTS 非线性 | 可能贡献大终点个例。终点已贴真值的多数事件不需要它作为主解释 |
| I. 旋转符号/框架 | 真实、单独、少数。不是 98% 的机制 |
| J. 组合 | 主样本是 A，加上弱 finite 的差分地板；I 与少数过冲是另一类 |

主结论：

```text
约 98% 的 official-chain nonconvergence
不是 “线性解 ok、物理步却停在错误几何”。

601 个恒等事件停在真值，判据在 chi2=0 上无法连续成立。
约 1700 个非零事件十步都在产生新几何，但最终 chart 已经在真值的 1e-11 以下。
失败的是冻结的 (步长, 相对 chi2, 连续两次) 判据，
或是这个判据所看的 chi2 序列在数值地板上仍不连续安静。
```

不能把 “把 MAX_ITERATIONS 加到 30” 写成修复。没有 k>10 的轨迹。终点已经贴真值的那一类，多跑几步也不会变成另一个物理答案，除非 chi2 序列本身还在跳动。这一点必须用逐步记录证明，而不是用终点代替轨迹。

---

## 5. 线性预测对物理响应

冻结产物里没有 `J`，也没有逐步预测。rho 未在 WB86 样本上计算。

已定义、并在线性 hermetic double 上核对的量：

```text
rho = || actual_prediction_change - J delta ||
      / max(|| J delta ||, 1e-12)
```

`J` 是 `physical_finite_difference_blocks` 的单边 chart Jacobian，步长 0.01。`actual_prediction_change` 是同一次 left-SE(3) 更新前后，官方 stacked `(x_mm, y_mm, tx, ty)` 预测之差。rho 无量纲。0 表示物理步落在该线性化上。1 表示偏差与线性预测本身同量级。

`tests/test_wb87_trajectory_recorder.py` 用 WB86 已有的线性 double，不调用 Calypso：

```text
iteration 1 rho ~ 8e-16
iteration 2 rho ~ 6e-16
iteration 3 trajectory_status = cached_repeat
theta_hat 回到注入的 s1_dx = 0.30 mm, s2_dy = -0.20 mm
```

这个测试只证明记录器在线性响应上自洽。它不是 WB86 的 Calypso 结果。

---

## 6. 旋转与 dropped mode

旋转不并进总 nonconvergence 率。

三个 fixed-dz 大事件的站别不同：S1 独大、S3 独大、两站同号。投影 z = -496 来自 S3 的 -63 mrad 与很小的 `sigma_a = 0.09 mrad`，不是来自单位方向 `u` 的符号写反。若只是 `u` 反了，S1 与 S3 会一起反号。这里另一站贴着真值。

dropped mode：WB86 sealed 只有最后一次 `n_dropped` 的整数。没有参数名、奇异值、右奇异向量或条件数序列。在这个诊断阶段不改秩阈值 `1e-12 * sigma_max`。现有证据只够说：

```text
identity index 510: 最后一次 n_dropped = 3，同时 theta_hat = 0
rotation index 1104, 2285: 最后一次 n_dropped = 2，同时有 O(0.5)–O(30) mrad 误差
translation / weak 的少数事件: 最后一次 n_dropped > 0，多数同时有可见 chart 误差
```

最后一次秩与 “十步一直是结构弱模式” 不是同一句话。要区分，必须有每一步的 spectrum。

---

## 7. 两个 technical FAIL

与主机制分开。

```text
index 0489  identity_finite_survey_prior
  iterations = 0, refits 未完成 solve
  missing truth-associated hit after official rerefit
  particle 10001 station 3

index 1745  identifiable_translation_fixed_dz
  iterations = 0
  missing truth-associated hit after official rerefit
  particle 10001 station 1
```

两者都发生在第一次有限差分 `predict`，也就是名义几何附近的官方 rerefit，还没有 alignment 更新。WB86 代码把 “真值粒子在该站没有 hit” 写成硬拒绝。这更像关联身份或重建没给出该站 hit，而不是迭代发散。它们是 2/2394，不主导 98% 的 nonconvergence。本轮没有重跑它们。

---

## 8. 下一步只授权一件事

推荐的下一步不是阻尼，不是放大 `MAX_ITERATIONS`，也不是改符号。

证据已经把主样本分成 “几何停在真值” 和 “几何还在变但终点贴真值”。这两类都要求看到 chi2、scaled update 和相对变化的逐步序列，才能判断判据是在 0 上无定义，还是 chi2 在数值地板上跳动。阻尼假设要等 rho 在大终点个例上偏大之后才成立。现在没有 rho。

因此下一实验是：**同一冻结迭代，只多写轨迹。**

已放入 `alignment/wb87_convergence_autopsy.py` 的 `record_frozen_trajectory`。它调用冻结的

```text
physical_finite_difference_blocks
solve_common_track
apply_parameter_update(..., UPDATE_LEFT_SE3)
WB86OfficialPhysicalBackend
MAX_ITERATIONS = 10
SCALED_UPDATE_TOL = 1e-3
VALIDATION_REL_TOL = 1e-3
CONSECUTIVE_REQUIRED = 2
```

不改这些对象。输出写到新目录，不写回 WB86。

### 预注册

```text
research question:
  在冻结 WB86 迭代上，chi2 与 scaled update 的逐步序列
  是 “几何已停但相对 chi2 无定义”，
  还是 “终点贴真值而 chi2 仍在跳动”，
  以及大终点个例的 rho 是否接近 1。

allowed code change:
  只允许调用 record_frozen_trajectory。
  不允许改 solver、阻尼、秩阈值、步长、关联、corpus、MAX_ITERATIONS。

dataset:
  回顾性、固定索引，取自已经跑完的 WB86 事件。
  不是新的 qualification 样本，也不是 Final Blind。

event subset, 13:
  0, 510, 105, 151, 239, 247, 657, 1104, 2285, 1586, 358, 489, 1745

metrics, each iteration:
  theta, delta_theta, scaled_update_norm, validation_relative_change,
  chi2, n_dropped, condition_spectrum, residual norm before/after,
  rho, payload sha256, trajectory_status

candidate algorithm:
  none.  H0 is the frozen full left-SE(3) update.
  H1 damping, H2 trust region, H3 backtracking are not opened.

decision rule for this diagnostic, not a qualification:
  identity plateau:
    PASS the mechanism claim if iterations >= 2 have
    scaled_update_norm == 0 and chi2 == 0 and validation_relative_change is null.
  numerical full-unique:
    PASS the mechanism claim if the last two theta rows differ by < 1e-6
    in chart L2 while validation_relative_change stays >= 1e-3
    or is null.
  large-endpoint rotation:
    report rho and which station moved.  Do not change sign.
  technical FAIL 489 and 1745:
    expect the same missing-hit refusal at iteration 0.
    A different exception is an execution mismatch, not a new solver result.

failure of the diagnostic:
  trajectory_status is accepted_record_missing on a step that
  n_official_refits says was a new geometry.
  That means the recorder indexing is wrong and the curves are void.

maximum physical rerefits:
  the frozen budget, at most 10*P+11 per event.
  13 events, P <= 9, so at most about 1300 official rerefits.
  Do not raise it.

success does not authorize:
  WB86 PASS, a new qualification, or ML alignment.
```

输出目录：

```text
outputs/mc24_four_station_wb87_convergence_autopsy_v1/
```

本 workbook 写成时，该目录里只有回顾性 census 和图。13 个事件的 Calypso 轨迹没有提交、没有运行。

census sha256：

```text
f19555b3640ab79f6e49dc6e76ed0a115a44bfe03482e5befb69e2e4d75612f3
  wb87_retrospective_census.json
fd763e91e241726a543c235a4bdc9c53f0d8bbce12c3098cdc78c13640337d4a
  wb87_event_geometry_class.json
```

---

## 9. 明确没做的事

- 没有重跑 WB86，没有改 WB86 JSON
- 没有提高 `MAX_ITERATIONS` 或放宽 `1e-3`
- 没有删事件，没有改 α、critical value、association、秩阈值
- 没有引入 IRLS 或阻尼
- 没有启动 ML-vs-truth
- 没有打开 Final Blind

---

## 10. Prospective results: the frozen 13-event trajectory

This section is the execution of the experiment preregistered in section 8.
It does not requalify WB86. WB86 remains

```text
execution_qualification   = FAIL
statistical_qualification = null
physics_screening         = FAIL
overall_qualification     = FAIL
alignment_oracle_qualified_for_physical_FASER = false
ml_alignment_eval_authorized = false
```

The preregistration was hashed before any physical trajectory was read:

```text
7c7eab81e743c413bbe2247a9bcb4fe1bea655f6c9cc12aea5f992dd58817124
  outputs/mc24_four_station_wb87_convergence_autopsy_v1/wb87_preregistration.json
```

Execution is Condor cluster `1160511` on `bigbird24`, 13 jobs, one per frozen index.
Every event stayed inside `10*P+11` official physical rerefits. No step has
`trajectory_status = accepted_record_missing` on a new geometry, so the curves
are not void. Right singular vectors, dropped parameter names, and the numeric
rank threshold are `spectrum_unavailable`: the frozen `CommonTrackSolution`
exposes `condition_spectrum`, `n_retained`, and `n_dropped`, and no second SVD
was computed.

Summary sha256:

```text
b8c6c965ef4b9ea4329e72038aa690215c68aa9b2f65e83b25cd6f4a4f1579b1
  wb87_trajectory_summary.json
```

### Q1

Yes. For identity events 0 and 510 the nonconvergence is the
`previous_chi2 == 0` denominator guard.

```text
identity_zero_chi2_relative_change_pathology = confirmed
```

On both events, every iteration has

```text
scaled_update_norm = 0
chi2 = 0
theta_hat = theta_true = 0
consecutive_convergence_count = 0
validation_relative_change = null
```

`previous_chi2` is null on iteration 1 and 0 from iteration 2. The frozen rule
defines the relative change only when `abs(previous_chi2) > 0`, so the
consecutive-2 gate cannot fire. Iteration 1 does rewrite the payload SHA once:
the left-SE(3) map turns `+0.0` station `ry` into `-0.0` while the chart stays
exactly 0. Every later iteration is a cache hit on that payload
(`n_official_refits` is 8 for event 0, `P+2`, and 11 for event 510).

### Q2

For the ordinary nonzero-injection events whose final chart lies near truth,
the blocking quantity is the relative chi2 change.

```text
105 translation_fixed   terminal scaled update 6.8e-12   relative change 0.81
151 translation_finite  terminal scaled update 1.6e-13   relative change 3.44
358 weak_finite         terminal scaled update 5.2e-11   relative change 2.05
```

After iteration 1 the solver-scaled movement is far below `1e-3` and the
physical residual norm is below `1e-10`. The chi2 itself is already around
`1e-20` to `1e-19`. Its relative change keeps jumping above `1e-3` because the
denominator is that numerical-floor chi2. The consecutive count stays 0.
The missing/null case is the identity path in Q1. These three events have a
defined relative change, and that defined change is what blocks the gate.

### Q3

The ordinary physical updates are locally consistent with `J delta` on the step
that actually moves the geometry. There is no evidence of a nonlinear
trust-region problem in this ordinary population.

```text
event 105 iteration 1   ||J delta|| = 0.361   rho = 0
event 151 iteration 1   ||J delta|| = 0.361   rho = 0
event 358 iteration 1   ||J delta|| = 0.685   rho = 0
```

Later ordinary steps have `||J delta||` around `1e-11` to `1e-12`. A rho of
order 1 on those steps is a comparison of two numerical-floor vectors. It is
classified as numerical-zero / not informative, and it is not used as a damping
signal.

### Q4

Yes. The large rotation and weak-JG endpoints are separated from the ordinary
nonconverged population by large rho, rank changes, or a station-localized jump.

```text
239  S3 rz reaches truth on iteration 1. S1 rz walks by about 0.01 mrad per
     step, then jumps to -840 mrad on iteration 4 (rho = 1.22, scaled step = 91).
     Terminal S1 rz = -103.63 mrad, S3 rz = +0.50 mrad. drop = 0 throughout.
247  Both stations stay near truth through iteration 8. Iteration 9 jumps S1
     rz to +47.2 mrad (rho = 11.8). Iteration 10 moves both:
     S1 rz = +33.19, S3 rz = +12.93. drop = 0 throughout.
657  The large S3 error is introduced in one physical update, iteration 1:
     S3 rz goes from 0 to -38.97 mrad (rho = 0.25, drop = 0, no payload repeat).
     S1 rz is already on truth. Later steps oscillate S3 and the frozen gate
     then converges at iteration 9 with S3 rz = -62.61 mrad and rho = 1.00.
1104 Rank changes every iteration (n_dropped = 3,3,1,2,3,3,2). S3 rz reaches
     truth on iteration 4. The gate converges at iteration 7 with S1 rz still
     at +0.030 mrad (truth -0.50) and n_dropped = 2. Several steps have
     rho > 1000 because ||J delta|| is ~1e-6 while the physical response is
     ~0.01.
2285 S3 rz jumps to +51.7 mrad on iteration 1 (rho = 13.3, drop = 0), returns
     near truth on iteration 7, and jumps again on iteration 8 (rho = 13.4).
     Rank drops to n_dropped = 2 on iterations 6 and 10. Terminal S3 rz =
     -26.82 mrad.
1586 The weak chart jumps on iteration 1 to S3 dx = -153.6 mm and S3 ry =
     +66.0 mrad (rho = 0.054, drop = 0) and stays there. Later rho values of
     order 1 have ||J delta|| of order 16 while the physical response is
     ~0.1 or ~20.
```

Ordinary events 105, 151, and 358 have no rank change and no station jump of
this size. Their first-step rho is 0.

### Q5

Yes. On event 358 the `dz ≈ -2e-5 mm` floor is the left-SE(3) half-angle
coupling of the first update, and the same coupling is repeated by the
finite-difference column.

Iteration 1 has solver `delta s3_dz = 0`. The chart after the update is

```text
s3_dx = +0.20000119 mm
s3_ry = +0.20000000 mrad
s3_dz = -2.00001184e-5 mm
```

The frozen map is `T ← Exp(ξ) T`. For a simultaneous translation `dx` and
rotation `ry` (radians) at the identity, the reconstructed `dz` is
`-0.5 * dx_mm * ry_rad`:

```text
-0.5 * 0.2000011908765422 * 0.00019999999245121574
  = -2.000011833e-5 mm
```

The residual against the recorded payload is `7.8e-14 mm`. From iteration 2
onward, the finite-difference column for `s3_ry` at the frozen step `0.01 mrad`
changes `s3_dz` by `-2.0e-6 mm`, which is the same half-angle factor
`-dx * (0.01/1000)`. The solver's own `delta s3_dz` on those later steps stays
around `1e-14 mm`, so the floor is not a sequence of solver dz updates.

### Technical failures

Events 489 and 1745 reproduced the historical refusal. Both stopped before a
solver trajectory.

```text
489   missing truth-associated hit, particle 10001, station 3
1745  missing truth-associated hit, particle 10001, station 1
execution class = reproduced_missing_hit
```

### Next hypothesis, not executed

Exactly one next hypothesis:

```text
convergence-metric repair for a zero or numerical-floor chi2
```

This is the mechanism confirmed on events 0, 510, 105, 151, and 358.
Trust-region / damping, Jacobian-step revision, rotation-chart correction, and
rank handling remain separate hypotheses for the large-endpoint population.
They are not the next experiment.

Workbook 88 protocol, frozen before any code change:

```text
research question:
  On the same frozen H0 iteration, does a preregistered numerical-floor
  rule mark an iteration successful when
    scaled_update_norm < 1e-3
    and (validation_relative_change < 1e-3
         or chi2 == 0
         or abs(chi2) is below a preregistered absolute floor)
  for two consecutive iterations?

allowed change:
  Only the convergence predicate used to increment the consecutive count.
  No change to solve_common_track, the rank cut, FD step, left-SE(3),
  association, corpus, MAX_ITERATIONS, SCALED_UPDATE_TOL, or VALIDATION_REL_TOL.

dataset:
  The same 13 frozen indices first, as a mechanism check.
  A full WB86-sized rerun is a later workbook and is not authorized here.

decision:
  The rule is supported on this subset if events 0, 510, 105, 151, and 358
  reach consecutive count 2 while their terminal chart stays at the same
  geometry this workbook recorded.
  Events 239, 247, 657, 1104, 2285, and 1586 must remain nonconverged or
  must be reported separately if the new predicate also fires on a large
  wrong chart. A pass on a wrong chart is a failure of the rule.
  Events 489 and 1745 must still refuse.

non-authorization:
  WB86 stays FAIL. ML alignment stays unauthorized. Final Blind stays closed.
```
