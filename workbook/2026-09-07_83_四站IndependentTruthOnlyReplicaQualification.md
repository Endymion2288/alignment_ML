# Workbook 83 — Independent Truth-only Alignment Replica Qualification

日期： 2026-09-07
分支： `4station`
任务： 在独立 measurement replicas、预注册几何条件、以及真正的 iterative update / relinearize 下，资格审查 common-track alignment solver。

---

## 0. 结论（先写）

```text
association_default_system =
    frozen_W64_raw_energy_plus_exact_solver

v1_1109787 = official FAIL
v2_1109806 = final solver-fix rerun

solver_implementation_fixed                      = true
common_track_solver_qualified_under_toy_model    = false
alignment_oracle_qualified_for_physical_FASER    = false
alignment_oracle_qualified                       = false
ml_alignment_eval_authorized                     = false
further_identical_rerun_authorized               = false
physical_alignment_qualification_blocked_by_data = true

continue_v5a_frozen_head         = false
continue_route_energy_rewrite    = false
continue_hybrid_expansion        = false
continue_residual_calibration    = false
continue_to_15d_relative_wls     = false

final_blind_eval_authorized      = false
sealed_test_accessed             = false
development_00350_used           = false
```

**WB83 已封账。** 不得再用同一 matrix / seeds / thresholds 重跑，不得再 patch solver，不得把 FAIL 重新解释成 PASS，也不得把剩余两 cell 说成已证明的 systematic bias。

```text
WB83 is closed as FAIL under its preregistered contract.

Further alignment qualification requires a new prospective statistical protocol
and, separately, sufficient independent Calypso physical replicas.
```

v1 `1109787` 是 **official FAIL**，永久保留。  
v2 `1109806` 是 **final solver-fix rerun**。实现已修，toy qualification 仍 FAIL。

第三层物理 FASER oracle **必须保持 false**：provenance 是 `toy_uniform_By_independent_v1`，不是 Calypso physical replicas。即便 14 cell 全过，也不把第三层设为 true。

WB82 `smoke_pass = true` **不是** 本阶段资格结果，也不得单独用 `validation_chi2_after < before` 证明 alignment 正确。

当前数据合同 **不能** 在不使用 overlay、不打开 00350 / Final Blind / sealed 的前提下，提供 100 个独立 Calypso 物理事件 replica。因此 WB83 的测量模型明确为：

```text
reconstruction_provenance = toy_uniform_By_independent_v1
calypso_physical_replicas_available = false
overlay_replicas_forbidden = true
```

每个 replica 是独立抽取的 track ensemble + 独立 hit noise，不是同一 physical event 的 overlay，不是同一 track 的重复 sampling。

不改预注册门槛，不用宽 CI 翻案，不回到 W64 / V5A / route-energy / hybrid / calibration。不建立 WB84。

---

## 1. 只回答的问题

> 给定 perfect truth association，common-track alignment solver 在独立 measurement replicas、多个 geometry directions 和真实 iterative update/relinearization 下，是否具有可接受的 bias、uncertainty coverage、weak-mode behavior 和 convergence？

只有全部 14 cell 在冻结规则下通过，才允许 `common_track_solver_qualified_under_toy_model = true`。  
`alignment_oracle_qualified_for_physical_FASER` 在 toy provenance 下必须保持 `false`。现在两层资格都是 false。不得做 ML-vs-truth。

---

## 2. 预注册几何矩阵（生成 replica 前冻结）

| condition | 方向 | 幅度 | chart |
|---|---|---|---|
| C0 nominal | 无注入 | 0 | S1/S2/S3 `dx,dy` |
| C1 translation | `(4,1,-2)/√21` 的 S1/S2/S3 相对 dx，正交于 `z` | 0.5× / 1.0× envelope `0.5 mm` | 同上 |
| C2 rotation | S1/S3 混合 `rz`，**不含 ry** | 0.5× / 1.0× envelope `1.0 mrad` | S1/S3 `rx,rz` |
| C3 weak | design-draw 白化 SVD 最弱保留模 | 0.5× / 1.0× envelope `0.5 mm` | S1/S2/S3 `dx,dy` |

每个条件都分别跑：

```text
fixed_dz
finite_survey_prior
```

两套结果不得合并。`finite_survey_prior` 额外纳入 S3 `dz`（σ = 5 mm）。  
full chart 若有 dropped SVD mode，禁止称 full-rank PASS。

Weak mode 在 `scripts/freeze_wb83_geometry_matrix.py` 里用 design seed `202609070` 冻结，禁止看完 replica 再改方向。

---

## 3. 预注册裁决（事后不得改）

```text
n_independent_replicas     = 100   # 不足 → UNKNOWN
|pull_mean|                < 0.2
0.8 <= pull_width          <= 1.2
95% coverage               与 p=0.95 的 Clopper-Pearson 相容
coverage CI width          <= 0.15
consecutive convergence    = 2
max iterations             = 10   # 到顶 = nonconverged，不是 PASS
```

工程 screening（**不是**合作组最终物理要求）：

```text
dx/dy bias <= 0.1 mm     # C0 / C1
rz bias    <= 1 mrad     # C2
```

C3 不套 0.1 mm screening：弱模的 mean 涨落可以大于该阈值。C3 仍必须过 pull / coverage / convergence，且不得把 dropped mode 叫 full-rank。

Robust IRLS 本阶段 `robust_authorized = false`。

---

## 4. Replica 闭环

```text
inject → generate independent hits → truth association
    → common-track solve → apply update → refit ξ → relinearize
```

禁止固定第一次 Jacobian 迭代。Fit / validation 分轨。truth 只用于最终 bias / coverage，不调 damping / FD / stopping。

---

## 5. 执行

正式 replica / coverage / gate：HTCondor DAG。

```text
SNAPSHOT / GEOMETRY freeze   (写入 immutable matrix)
    ↓
14 cells × 100 replicas
    ↓
COVERAGE_SUMMARY
    ↓
QUALIFICATION_GATE          (fail-closed)
```

输出根：`outputs/mc24_four_station_wb83_truth_only_replicas_v1/`（official FAIL，不可变）  
v2 输出根：`outputs/mc24_four_station_wb83_truth_only_replicas_v2/`（同一 matrix 的 solver-fix rerun）

---

## 6. 明确禁止

W64 / V5A / route-energy 重训、Arm C、hybrid、calibration、ML association closure、00350、00800–00849、100116 / 100117、Final Blind、sealed、overlay 充数、旧 pairwise WLS 替代 common-track、用 WB82 smoke 数字替代 replica coverage、truth-only FAIL 后转去调 association。

---

## 7. 数字（只从 JSON 抄）

Freeze：`outputs/mc24_four_station_wb83_truth_only_replicas_v1/`

| 量 | 值 |
|---|---|
| `matrix_sha256` | `ca895e44082d4021fe6a3f9d008cf594e03c6330044466a883ca6355e8ac2720` |
| n cells | 14 |
| n declared replica rows | 1400 |
| design seed | `202609070` |
| weak condition number | `3500003.6520249434` |
| weak n_dropped | 0 |
| weak direction (S1/S2/S3 dx) | `0.2673 / 0.5345 / 0.8018`（≈ `z` 斜率，物理 JG） |
| `alignment_oracle_qualified` | **`false`** |
| v1 qualification | **`FAIL`** (`one_or_more_gates_failed`) |
| n independent / n rows | 100 / 1400 |
| DAG | `1109787` on `bigbird24`, 15/15 nodes `DAG_STATUS_OK` |

v1 cell 表（`coverage_report.json`）：

| cell | conv | pull mean / width | cov | 裁决 |
|---|---|---|---|---|
| C0 nominal fixed_dz | 100/100 | 0.110 / 0.889 | 0.99 | PASS |
| C1 trans 0.5× / 1× fixed_dz | 100/100 | −0.079 / 0.889 ； 0.061 / 0.958 | 0.98 / 0.96 | PASS |
| C2 rot 0.5× / 1× fixed_dz | **0/100** | 0.128 / 0.916 ； −0.089 / 0.929 | 0.97 / 0.97 | FAIL `nonconverged`（原因：第 2 步 `non_spd`） |
| C3 weak 0.5× fixed_dz | 100/100 | **−0.263** / 0.889 | 0.97 | FAIL `pulls_failed` |
| C3 weak 1× fixed_dz | 100/100 | −0.010 / 0.937 | 0.97 | PASS |
| C0 / C1 / C2 / C3 finite_survey_prior | 几乎全到第 10 步 | pulls 大多合格 | C3 1× cov 0.89 FAIL | FAIL `nonconverged`（`dz` increment prior 慢爬） |

FD `h/2,h,2h` PASS。禁止资产未访问。`bias_qualified` 在 screening 条件下 PASS。

v1 失败机制（不改门槛）：

1. 旋转更新后 FD 的 `J^T W J` 有 1e-8 级不对称，`require_spd` 用 `atol=1e-12` 把可逆的局部法向打成 `non_spd`。
2. `finite_survey_prior` 的 `dz` prior 作用在 Newton increment=0 上，而不是绝对 survey 残差，每步留下 ~0.1 scaled `dz` 更新，10 步到顶。

这两处是 alignment 实现缺陷。修 solver 后用 **同一冻结 matrix** 重跑 v2。不放宽 `|pull_mean|<0.2`，不回到 association。

```text
v1 condor_submit_dag cluster = 1109787   # official FAIL, artifacts kept
v2 condor_submit_dag cluster = 1109806   # WB83-v2 solver-fix rerun
schedd                 = bigbird24.cern.ch
```

---

## 8. WB83-v2 solver-fix rerun（官方数字只从 JSON 抄）

输出根：`outputs/mc24_four_station_wb83_truth_only_replicas_v2/`  
v1 根未改写（`truth_oracle_qualification.json` mtime `2026-09-07 00:57`）。

```text
matrix_sha256 =
ca895e44082d4021fe6a3f9d008cf594e03c6330044466a883ca6355e8ac2720

14 cells
100 replicas/cell
n_rows = 1400
same seeds / geometry directions / amplitudes / thresholds
same FD contract / convergence rules
DAG 1109806 on bigbird24, 15/15 nodes DAG_STATUS_OK
summarize job = 1109821
```

Hermetic regressions（正式 v2 结果前）：`tests/test_wb83_v2_solver_fix.py` + solver/WB83 合同测试 **31 passed**。

1. SPD + `1e-8` antisymmetric perturbation → symmetrize 后 Cholesky 通过，并记录 antisymmetry / min eigenvalue / condition number  
2. true-indefinite 拒绝；`1e-12` ridge 不能把它变成 SPD，代码禁止加 ridge  
3. scalar analytic survey-prior MAP  
4. `a_current == a_survey` → prior gradient = 0 → update = 0  
5. iterative prior 收敛，禁止恒定 `dz` increment  
6. `fixed_dz` regression parity  

v2 只允许的两项实现修复：

1. `H = 0.5 (H_raw + H_raw.T)` 后再做严格 SPD/Cholesky。不放宽阈值，不加 ridge/jitter 掩盖 indefinite system。  
2. finite survey prior 改为 absolute-state MAP：作用于 `a_current + delta_a - a_survey`。`a_current == a_survey` 时 prior gradient 为 0。

`1109806` worker 在进程启动时 import 的 local inverse 仍是 symmetrize 之后加 `1e-12 I`。该 jitter 相对 C2 的 min eigenvalue（~1e3）是 `1e-15` 量级，**不能**把真正 indefinite 的系统改写成 SPD。当前树已去掉这条 ridge；`solver_implementation_fixed = true`。

官方三层（`truth_oracle_qualification.json`）：

```text
solver_implementation_fixed                      = true
common_track_solver_qualified_under_toy_model    = false
alignment_oracle_qualified_for_physical_FASER    = false
qualification                                    = FAIL
reason                                           = one_or_more_gates_failed
```

gates：

| gate | v2 |
|---|---|
| `convergence_qualified` | **true**（14/14 cell 100/100） |
| `fd_stability_qualified` | true |
| `bias_qualified` | true |
| `no_forbidden_assets_accessed` | true |
| `pulls_qualified` | **false** |
| `coverage_qualified` | **false** |
| `all_cells_pass` | false |

v2 cell 表（`coverage_report.json`）：

| cell | conv | pull mean / width | cov | 裁决 |
|---|---|---|---|---|
| C0 nominal fixed_dz | 100/100 | 0.110 / 0.889 | 0.99 | PASS |
| C1 trans 0.5× / 1× fixed_dz | 100/100 | −0.079 / 0.889 ； 0.061 / 0.958 | 0.98 / 0.96 | PASS |
| C2 rot 0.5× / 1× fixed_dz | 100/100 | 0.128 / 0.915 ； −0.089 / 0.930 | 0.97 / 0.97 | PASS（v1 为 0/100 `non_spd`） |
| C3 weak 0.5× fixed_dz | 100/100 | **−0.263** / 0.889 | 0.97 | FAIL `pulls_failed` |
| C3 weak 1× fixed_dz | 100/100 | −0.010 / 0.937 | 0.97 | PASS |
| C0 nominal finite | 100/100 | 0.047 / 1.148 | 0.92 | PASS（v1 几乎全 `dz` walk） |
| C1 trans 0.5× / 1× finite | 100/100 | 0.029 / 0.944 ； −0.002 / 1.116 | 0.95 / 0.91 | PASS |
| C2 rot 0.5× / 1× finite | 100/100 | −0.166 / 1.126 ； −0.106 / 1.045 | 0.93 / 0.95 | PASS |
| C3 weak 0.5× finite | 100/100 | −0.030 / 0.931 | 0.98 | PASS |
| C3 weak 1× finite | 100/100 | −0.065 / 1.159 | **0.89** | FAIL `coverage_failed` |

实现修复的直接效果：

* C2：第 2 步不再被 FD `J^T W J` 的 1e-8 反对称打死；100/100 收敛。  
* 全部 `finite_survey_prior`：不再每步恒定 `dz` increment；100/100 收敛（3–4 步）。  
* C0/C1/`C3 0.5× fixed_dz` 与 v1 数字在 `1e-8`–`1e-10` 内重合：这些 cell 本来就收敛，solver-fix 不应改它们。

---

## 9. 剩余 alignment 失败机制（不是门槛问题）

两项实现缺陷已经关闭。剩下的 FAIL **不是** SPD、也不是 increment prior。禁止放宽 `|pull_mean|<0.2`、coverage 规则、max iteration、damping、geometry matrix、weak direction、prior sigma。

诊断输出（不是新 gate）：`outputs/mc24_four_station_wb83_truth_only_replicas_v2/remaining_alignment_mechanism.json`

### 9.1 C3 weak 0.5× `fixed_dz`：独立预注册 `pulls_failed`

官方（`coverage_report.json`）：

```text
pull_mean = -0.26289344212158866
v1        = -0.26289344205610293
max |v2-v1 pull| = 1.9e-9
true_coefficient = 0.25
estimated_coefficient mean = 2.258
residual_coefficient mean = -2.008
mean_sigma = 7.638   # 100 个 replica 共用同一个 identity-linearized σ
iterations = 3 / 3 / 3
```

机制（同一 0.5× seeds，诊断重解；`remaining_alignment_mechanism.json`）：

```text
first_step_pull_mean                 = -0.262893435
iterate_pull_mean                    = -0.262893442
iterate_minus_first_pull_mean        = -7.0e-9
last_step_sigma_mean                 = 7.637630051   # ≈ identity 7.637630050
amp 0 × same seeds first_step_pull   = -0.262893436
amp 1 × same seeds first_step_pull   = -0.262893434
pull_mean z vs N(0, width/√100)      = 2.959
two-sided p                          = 0.00309
```

结论：**不是迭代偏差，不是幅度相关非线性，不是末步 covariance 写错。**  
−0.263 是冻结 100-replica **hit-noise ensemble** 沿弱模（`z` 斜率 JG）的线性 GLS 样本均值。同一 seeds 改注入 0× / 1×，pull 均值不变。同方向 1× `fixed_dz` 用的是另一组 seeds，`pull_mean = −0.010` PASS。

这仍是预注册 FAIL。不得加 replica 翻案，不得把 `|0.2|` 改成 0.3。

### 9.2 C3 weak 1× `finite_survey_prior`：收敛之后仍 `coverage_failed`

官方：

```text
n_covered_95 = 89/100
coverage     = 0.89
Clopper-Pearson = [0.8117, 0.9438]   # 不含 0.95；宽度 0.132 <= 0.15
pull_mean / width = -0.065 / 1.159   # pulls 规则本身 PASS
residual std = 8.853
mean_sigma   = 7.638
iterations   = 4 / 4 / 4
```

机制：

```text
first_step ≈ iterate                 (Δ pull mean = −4.0e-6)
last_step_sigma − identity_sigma     = +0.00024
coverage with last-step σ            = 0.89   # 不能救回
same 1×-finite seeds, amp 0 / 0.5 / 1.0:
    pull_width = 1.15912246
    residual_std = 8.85295
    coverage = 0.89
P(K≤89 | n=100, p=0.95)              = 0.01147
corr(weak residual, s3_dz)           = 0.13
```

v1 同一 cell 已是 89/100；当时未收敛。v2 收敛后 coverage 不变。  
**不是 prior walk，不是 Jacobian 随 1× 幅度变胖，不是 last-step σ 漏算 dz。**  
8.85 vs 7.64 的散度是这组冻结 1×-finite seeds 的噪声实现。0.5× finite 用另一组 seeds，std 7.11、coverage 0.98 PASS。

---

## 10. 诊断结论（仍 FAIL，不改门槛）

```text
c3_0p5x_fixed_is_frozen_ensemble_mean = true
c3_0p5x_fixed_not_iteration_bias = true
c3_0p5x_fixed_not_amplitude_dependent = true

c3_1x_finite_is_frozen_ensemble_width = true
c3_1x_finite_not_iteration_bias = true
c3_1x_finite_not_amplitude_dependent = true
c3_1x_finite_last_step_sigma_does_not_restore_coverage = true
```

弱模本身几乎不可观：whitened 条件数 `3.50e6`，identity σ ≈ 7.64 mm，注入只有 0.25 / 0.50 mm。Pull / coverage 门检验的是估计量是否无偏、是否校准，不是这个 JG 能不能用于物理。n=100 时 `|pull_mean|<0.2` 约等于要求 100 个 N(0,1) 的均值落在 ±0.2（≈2.25σ）；无偏估计量也会以百分之几的概率 FAIL。本冻结 seed 集就落在这一侧。

这 **不能** 用来重开门槛、加 replica、或宣称 toy qualification PASS。

---

## 11. WB83 封账

不可变 closure：`outputs/mc24_four_station_wb83_final_closure_v1/wb83_final_closure.json`  
未改写 v1/v2 replica 输出。

```text
v1_status = FAIL
v2_status = FAIL
solver_implementation_fixed = true
convergence_qualified = true
fd_stability_qualified = true
bias_screening_qualified = true
pulls_qualified = false
coverage_qualified = false
toy_qualification = FAIL
physical_FASER_oracle_qualified = false
further_identical_rerun_authorized = false
```

```text
matrix_sha256 =
ca895e44082d4021fe6a3f9d008cf594e03c6330044466a883ca6355e8ac2720
master_seed = 20260907
design_seed = 202609070
```

剩余正式 FAIL（不得翻案）：

```text
C3_weak_0.5x_fixed_dz            pull_mean = -0.26289344212158866
C3_weak_1x_finite_survey_prior   coverage = 89/100
```

解释合同：

```text
remaining WB83 failures are frozen-ensemble statistical failures
under the preregistered finite-replica qualification rule
```

**不是** `common-track estimator has proven systematic bias`。  
**不是** 新的 implementation defect。

禁止：放宽 `|0.2|`、改 coverage、加 WB83 replica、换 weak direction / seeds / prior σ / damping、再 patch solver、把 WB83 说成 PASS。

---

## 12. 下一代统计资格协议（只设计，不生成 replica）

设计稿：`outputs/mc24_four_station_wb83_final_closure_v1/prospective_statistical_protocol_design.json`  
`executable = false`，`new_replicas_authorized = false`，`is_wb83_rerun = false`。  
WB83 只说明旧 protocol 的统计行为值得重设计，**不能**用来训练新 gate，不能按 −0.263 或 89/100 优化阈值。

必须区分：

```text
single-cell calibration
    每个 cell 的 mean / width / coverage 及其抽样误差。这是诊断和功效计算。

global multi-cell qualification
    在预先声明的 simultaneous error rate 下，对 solver 做一次裁决。
    不是“每个 cell 都必须独立通过未校正门槛”。
```

旧规则 `every cell must independently satisfy an unadjusted threshold` 的问题：14 个未校正门的交会检验，族错误率远高于单 cell 名义水平。无偏、校准的估计量也会以不可忽略的概率被整表拒绝。这是 protocol 设计问题，不是把 WB83 改判 PASS 的理由。

三类 prospective 方法（必须在看到新数据之前选定，禁止挑“最容易让 WB83 过”的那个）：

1. **Multiplicity-controlled cellwise（M1）**  
   保留 cellwise 检验，但控制预注册的 FWER / FDR（Holm–Bonferroni、有依据时的 Hochberg、或 global→family→cell 的封闭检验）。必须先冻结：哪些量是检验、哪些只是报告；family 划分；同时错误率；进入校正的检验个数。

2. **Global pull GOF（M2）**  
   对所有预注册 mode/cell 的 standardized residual 做联合 mean / width 校准检验（Hotelling / χ²、声明协方差后的全局尺度检验、单一接受域的 location-scale 统计量）。科学声明是“solver 校准”，不是 14 条独立声明。必须预注册堆叠方式和单一接受域，并预注册 complementary lack-of-fit 分解，禁止事后拆 cell。

3. **Hierarchical / random-effects（M3）**  
   把 persistent bias 与 finite-replica ensemble fluctuation 分开。只对 persistent 分量做资格裁决（例如 cell mean ~ N(0, τ²)，抽样方差 σ_c²/n；τ² 与 0 不相容才拒）。弱模可以有自己的方差分量，而不是 0.1 mm screening。必须先冻结随机效应律和 τ² 的决策阈值。

未来若重新资格测试，那是 **new prospective qualification experiment**，不是 WB83 rerun。必须使用 **新的独立 replica seeds**，并在执行前冻结。WB83 的 FAIL 永远保留。

任何新 replica 生成前必须冻结：

```text
global null
cell/mode family
simultaneous error rate
sample size
acceptance statistic
multiplicity handling
UNKNOWN criterion          # n 不足 → UNKNOWN，不是 PASS
bias/coverage/pull 报告规则
weak-mode treatment
new independent replica seeds
```

sample size 由对科学相关 alternative 的功效决定，不由“怎样才能翻掉旧 FAIL”决定。

---

## 13. Physical-data blocker（只读）

`outputs/mc24_four_station_wb83_final_closure_v1/calypso_physical_replica_feasibility.json`

未打开 overlay / 00350 / Final Blind / sealed。

```text
physical_alignment_qualification_blocked_by_data = true
calypso_physical_replicas_available = false
```

现有可用与缺失：

| 资产 | 状态 |
|---|---|
| 六源授权 train xAOD（W64 已见） | 生成量级 2.5M，**不是** alignment-ready 独立 replica |
| WB78 源纯 refit 子集 | 每源约 50 events；远不够多 cell 资格 |
| overlay 语料（seed `20260813`） | 存在，但 **禁止** 当独立 replica |
| held-out geometry seed `314159` | 表在，`physical_refit_authorized = false`，事件 = 0 |
| 00350 / 00800–00849 / 100116 / 100117 | 禁止打开 |

缺失：独立 Calypso 事件、一次重建、物理 covariance / field / material、已知注入或 survey payload、truth-only association、以及未来全局协议冻结 n 之后够用的事件数。

需要的新生产（尚未开始）：

```text
Calypso /Tracker/Align → physical refit → ACTS
independent events, one reconstruction per event
provenance = calypso_physical_independent_v1
```

若将来复用 train-range xAOD，必须披露 W64 见过这六源。truth-only alignment 仍可用新重建事件，但它们对默认 association 系统不是 source-unseen。

当前默认 association 仍是 `frozen_W64_raw_energy_plus_exact_solver`。alignment **暂时不能**作为它的 downstream certified oracle。

---

## 14. 最终两条

1. `WB83 is closed as FAIL under its preregistered contract.`

2. `Further alignment qualification requires a new prospective statistical protocol and, separately, sufficient independent Calypso physical replicas.`

没有开始新的 qualification run。不返回 association。

---

## 15. 独立 consistency audit（2026-09-07 下午）

不重跑 WB83 qualification。对照仓库和 artifacts，不对照 workbook 文本。10 项全部通过。

| # | 要求 | 证据 |
|---|---|---|
| 1 | `wb83_final_closure.json` 存在且声明不可变 | `outputs/mc24_four_station_wb83_final_closure_v1/wb83_final_closure.json`；`immutable=true`；SHA256 `5b580f764829fff80f9de70bd74c6b691e369db724501de734904fbc6fe05599`；mtime `2026-09-07 08:59` |
| 2 | v1 / v2 root 存在，v1 未被覆盖 | v1 gate mtime `2026-09-07 00:57`，SHA 仍为 closure 记录的 `a5ecf9ea…`；v2 gate mtime `2026-09-07 01:33` |
| 3 | matrix SHA256 | v1、v2、closure 均为 `ca895e44082d4021fe6a3f9d008cf594e03c6330044466a883ca6355e8ac2720` |
| 4 | v2 14 × 100 完整 | 14 个 jsonl，各 100 行；`coverage_report.n_rows = 1400` |
| 5 | 两项 solver 修复在代码中 | `symmetrize_normal` + `inverse_spd_after_symmetrize`（无 `1e-12 I` ridge）；`absolute_survey_prior_terms` 作用于 `a_current + theta - a_survey` |
| 6 | regression 覆盖要求的四点 | `tests/test_wb83_v2_solver_fix.py`：true-indefinite rejection、analytic prior MAP、zero prior residual、fixed-dz parity；本 audit 重跑 5 个测试，全部通过 |
| 7 | 剩余正式 FAIL 只有两格 | `C3_weak_0.5x_fixed_dz` `pull_mean = -0.26289344212158866`；`C3_weak_1x_finite_survey_prior` `coverage = 89/100` |
| 8 | 禁止资产未访问 | v1/v2 `development_00350_used=false`，`final_blind`/`sealed` 未授权；1400 条 v2 replica 的 `overlay_used`/`w64_used`/`ml_association_used` 全为 0 |
| 9 | 没有第三次 identical-seed qualification | 无 v3 root；DAG 只有 `1109787` 与 `1109806`；无 WB84 |
| 10 | toy / physical oracle 未改成 PASS | v1/v2/closure 均为 FAIL；`common_track_solver_qualified_under_toy_model=false`；`alignment_oracle_qualified_for_physical_FASER=false` |

solver 源文件 SHA 与 closure 一致。无需为了“改结论”去修 artifact。

不可变证据是 **内容 SHA256**，不是 EOS 上的 `chmod 0444`：属主仍可写入。audit 曾用 identical rewrite 探测只写位，v1 gate 内容哈希未变，mtime 被碰过一次后已恢复为 `2026-09-07T00:57:22.824541`。

---

## 16. Physical replica production contract（只设计）

输出目录：`outputs/mc24_four_station_calypso_physical_replica_design_v1/`

```text
calypso_physical_replica_production_protocol.md
calypso_physical_replica_production_plan.json
physical_replica_capacity_estimate.json
provenance_contract.json
```

```text
physical_alignment_qualification_blocked_by_data = true
physical_replica_production_feasible             = true
physical_replica_production_authorized           = false
alignment_oracle_qualified                       = false
```

六源 train-range xAOD 在 EOS 上存在（约 2.5M 生成事件）。CERN Calypso / HTCondor 链可以把它们做成 `calypso_physical_independent_v1`。当前没有现成独立 physical replica 语料，所以 **qualification 仍被数据挡住**。规划容量是 8 conditions × 200 admitted replicas，admission yield 0.50 → **3200 个互不重复的独立事件**。n 来自 0.25σ persistent-bias 功效（m=3 Holm），不是 WB83 的 −0.263 或 89/100。

未开始生产，未提交 Calypso job，未生成新 replica，未做 ML-vs-truth，未返回 association。
