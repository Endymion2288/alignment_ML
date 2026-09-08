# Workbook 81 — Physics-constrained Route Energy Calibration

日期： 2026-09-06
分支： `4station`
任务： WB81a 物化 Arm A 边界并证明 `ΔU=0` 恒等；WB81b 在冻结 §4 目标下训练线性残差 Arm D。

---

## 0. 结论（先写）

```text
training_authorized              = true    # 仅 WB81b family-CV calibration
wb81b_authorized                 = true
dataset_export_authorized        = true    # 仅使用 WB81a 已定义/物化的边界合同
final_blind_eval_authorized      = false
sealed_test_accessed             = false
continue_v5a_frozen_head         = false
continue_to_15d_relative_wls     = false
arm_c_authorized                 = false
wp4_arm_c_authorized             = false
hybrid_expansion_authorized      = false
replacement_model_authorized     = false
development_00350_used           = false
domain_adaptation_authorized     = false
```

**WB81a PASS。WB81b 已跑完。校准臂叫 Arm D，不是 Arm C。**

```text
physics_constrained_calibration_supported          = false
calibration_has_no_accepted_increment              = false   # 两 fold 裁决不一致，不能用这一条概括
calibration_unacceptable_under_zero_fake_slack     = false   # 仅 family2 触发；family1 有过门 checkpoint
default_system                                     = frozen_W64_raw_energy_plus_exact_solver
```

| fold | identity | same-family fake gate | holdout Δeff | holdout Δfake | fold decision |
|---|---|---|---|---|---|
| holdout_family1 | PASS | 19/20 epoch 过门；选 epoch 17 | +0.093 | +0.00114 | `no_accepted_increment` |
| holdout_family2 | PASS | 0/20 过门；恢复 `ΔU=0` | 0 | 0 | `reject_calibration_under_zero_fake_slack` |

**本阶段问题的答案：不能。** 在冻结 W64、`|ΔU|≤0.25`、complete fake rate 不允许恶化的条件下，raw physical residual **不能安全**恢复 W64 边界 truth。family1 能捞回 691 条 `T_neg`，但 holdout fake 从 77 升到 93；family2 被零 slack 硬门整折打死。两 fold 都不满足 `fake_new ≤ fake_A` 且 `eff_new ≥ eff_A + 0.01`。不改 `+0.01`、`δ_max`、fake slack。

W64 见过全部六个授权 train source。Arm A 是 production baseline，不是 fully source-unseen representation。`gate_pass = null`。

81a 已在本地 CPU 跑完（未提交 Condor）。`θ=0` / `ΔU=0` 与 Arm A 的 selected / eff / fake / purity 恒等。边界计数与 WB80 checksum 对齐。合同守卫通过。81a 没有训练；81b 才在 Condor CPU 上训练线性 `ΔU`。没有扫 `δ_max`，没有放 fake slack。

| family | identity | checksum | T_neg | T_near | T_comf | F_A | competing | `|M_A|<0.25` 的 T_neg |
|---|---|---|---|---|---|---|---|---|
| family1 | PASS | PASS | 1071 | 5718 | 0 | 77 | 7941 | 919 |
| family2 | PASS | PASS | 463 | 2785 | 0 | 56 | 3779 | 458 |

可翻盘计数是 **上界**：假设 rival 的 `ΔU=0`。真实校准会对 2/3/4 一视同仁地加残差，所以 81b 仍可能被 fake 硬门打死。

81a 通过只说明：**残差槽位在零点安全，边界集真实非空，值得单独授权 81b。** 81b 现已授权，但仍不得改 §4、放宽 fake slack、加大 `δ_max`、或改读 WB80。默认在 81b 裁决写出之前仍停在冻结 W64。

---

## 1. 冻结边界

不重训 V5A / WB75 / W64 / WB79 Arm B / WB80 C。不改 W64 architecture 或 loss。不改 `baselines/route_assignment.py` 默认 legacy decoder。不打开：

```text
mc24_100047_00350_00399
mc24_100048_00350_00399
mc24_100047_00800_00849
mc24_100048_00800_00849
mc24_100116_*
mc24_100117_*
```

不做 domain adaptation。不做 wp4 Arm C。不扩大 hybrid（`ΔU` 不得读 W64 logits）。不把 B/C 当 parent。不扫 `δ_max` / fake slack。

---

## 2. Failure case analysis

全部数字抄自 WB79 / WB80 JSON。Arm A = 冻结 W64 `sum(raw logits) + n*penalty`。

### 2.1 不是“个别难例”，是尺度问题

| fold | complete truth | A selected（near_zero） | A missed（negative） | A comfortable `M≥1` | A complete fake | A 全部 selected 的 `|U|<1` |
|---|---|---|---|---|---|---|
| family1 holdout | 6789 | 5718 | 1071 | **0** | 77 | 5795 / 5795 |
| family2 holdout | 3248 | 2785 | 463 | **0** | 56 | 2841 / 2841 |

mean `M_A`：near_zero 0.167 / 0.143；negative −0.139 / −0.065。  
W64 的 packing 全部贴着 inclusion 边界。**不存在可以原封不动留下的舒适区。**

`|U_W64|<1` 盖住 A 的全部 selected，所以 near-dustbin 不能再当 A 内部的分层。边界集必须用 **inclusion gap**，不能只用 `|U|`。

### 2.2 七类失败 / 风险

**Case N — negative-gap complete truth（A miss）**  
1071 / 463。这是 A 仅有的 recall 头寸。C 在这里 recall 0.821 / 0.868，同时假率上升。若 `|ΔU|≤0.25` 且只希望抬 truth、不抬 rival，则只有 `|M_A| ≲ 0.25` 的子集物理上可翻盘。mean `|M_A|` 已是 0.14 / 0.07，所以头寸存在，但不是“全捞”。

**Case Z — near-zero complete truth（A hit，脆）**  
5718 / 2785。B 丢掉其中 0.217 / 0.144。rival 若拿到 `+0.2` 的 `ΔU`，A 已选的 truth 会被踢出。校准必须 **保护** 这类，而不是只优化 miss。

**Case C — comfortable**  
空集。任何“只动边界、不动内部”的故事在当前 A 下没有内部。

**Case F+ — A 已选 complete fake**  
77 / 56。它们已经在 A 的最优集里。对“看起来像 complete”的物理模式加正 `ΔU`，这部分会先涨。硬 fake 门首先针对它们和下面的 F−。

**Case F− — A 未选、C 选中的 fake**  
C 的 complete fake 497 / 174。这是 WB80 的机制警告：边界能量上也住着 fake。无约束 inclusion hinge 会把它们一起抬进来。

**Case P — fake pair / fragment**  
A 选 0 条 fragment。B 在 family1 选了 26 条 fake 2-station。若 `ΔU` 只加在 complete 上，会重开 V4/V5 的 complete-vs-fragment 不对称；若 `ΔU` 也加在 pair 上且无界，会重开 B 的 fragment fake。协议：同一有界函数作用于 2/3/4，禁止 complete-only 非对称 head。

**Case H — WB76 残差破坏（历史，禁止重演）**  
V5A：`U = L_edge + delta − 4`。OOD 上中等负 `delta` × 低 W64 margin 把已选 truth 打过零。WB81 用的是 **同一个代数槽**。差别必须是：`delta` 有界、默认 0、fake 硬拒绝、00350 仍不打开。本设计 **不声称** OOD 安全。

### 2.3 对 replacement / hybrid 的含义

| 实验 | 对 A 做了什么 | 结果 |
|---|---|---|
| B | 丢掉 `U_W64`，用物理 MLP 重写能量 | holdout 更差；Spearman(`U_A`,`U_B`) 0.27 |
| C | 物理 + W64 logits 重写能量 | eff +0.061 / +0.076，fake +0.062 / +0.035；不 beat A |
| 校准（设计） | 冻结 `U_W64`，只学有界 `ΔU` | 尚未实现 |

C 证明：边界上 **有** 可挖的 recall，但无 fake 硬约束时会把 F− 一起挖出来。这不是“再训一个更大 hybrid”的理由，而是“残差必须小于重写、并以 fake 作拒绝”的理由。

---

## 3. W64 margin boundary dataset（已物化）

`dataset_export_authorized = true`（仅 81a）。成员规则如下；行表已写盘，计数与 WB80 对齐。

### 3.1 宇宙

与 WB79 / WB80 相同的 family-CV historical overlay（seed `20260813`）：

```text
family1 = outputs/mc24_four_station_relative_route_v5a_source_transfer_v1/corpora/family1/...
family2 = outputs/mc24_four_station_relative_route_v5a_source_transfer_v1/corpora/family2/...
```

同一 candidate、同一 truth-free 2/3/4 enumeration、同一 exact solver。能量只许 Arm A。禁止用 B/C 的 utility 定义成员。

### 3.2 成员

对每个 event 建 `raw_energy_v1` 表，`correction=0`。对每条 **complete truth** `r` 算 `M_A(r)`。

| 子集 | 规则 | family1 n | family2 n | 用途 |
|---|---|---|---|---|
| `T_neg` | complete truth，`M_A < 0` | 1071 | 463 | 主召回头寸 |
| `T_near` | complete truth，`0 ≤ M_A < 1` | 5718 | 2785 | 必须保护 |
| `T_comf` | complete truth，`M_A ≥ 1` | 0 | 0 | 空；保留定义 |
| `R(r)` | `r` 的 forced-out packing 里、与 `r` 竞争的 routes | 每条 truth 一条集合 | 校准对照，不是第二套候选 |
| `F_A` | A 选中的 complete fake | 77 | 56 | fake 硬门的已有底数 |
| `P_A` | A 选中的 fragment | 0 | 0 | 回归：校准后应仍接近 0 |

已写盘行带：`U_W64`、`inclusion_gap_A`、`single_rival_margin_A`、`selected_A`、`bin`、endpoints、forced-out competitors 的 `U_W64`。81a 不存 54-D 物理特征、不存 W64 latent。不把 00350 / 00800 / sealed 写进宇宙。

### 3.3 Split 纪律

| 行来自 | 允许 |
|---|---|
| train-family、非 val payload | 若以后授权训练，才可进 `ΔU` 的 loss |
| train-family val payload | 只做 fake 硬拒绝与 early stop |
| holdout family | 只做最终报告；不选模、不调 `δ_max` |

物化之后每个 fold 必须有：dataset manifest、Arm A 复现表（与 WB79/80 逐格重合）、边界计数 checksum（上表）。对不上则 fail-closed。

### 3.4 可翻盘上界（已从物化集算出）

在 `|ΔU|≤0.25` 且 rival `ΔU` 不帮倒忙时，只有 `|M_A| < 0.25` 的 `T_neg` 有机会被校准翻成 selected。导出后的上界：family1 **919 / 1071**，family2 **458 / 463**。头寸非空，但不是“全捞”，也不能在 rival 同样拿到残差时保证翻盘。

---

## 4. Calibration objective

### 4.1 能量

```text
U_W64(r) = sum_e raw_base_edge_logit_e + n_stations(r) * unmatched_penalty
U_new(r)  = U_W64(r) + ΔU_θ(x_phys(r))
ΔU_θ      = δ_max * tanh(g_θ(x_phys))
δ_max     = 0.25
x_phys    = raw_physical_route_v1   # 54-D；禁止 W64 logits / latent
```

`U_W64` 冻结。`ΔU` 写入已有 `RouteEnergyRecord.correction`。`g_θ` 是线性层，或 width≤16 的单隐层；默认线性。2/3/4 共用同一 `g_θ`。`θ=0` ⇒ `ΔU=0` ⇒ 与 Arm A 的 selected 集相同。这一条必须先做成 hermetic test，再谈任何训练授权。

不走 `sigmoid → clip → logit`。不走 `_route_hypotheses` / `utility>0` 预过滤。

### 4.2 训练目标（81b 已授权；公式冻结）

禁止再用 WB79/80 的 LAI Hamming。C 的过包含来自“整事件结构化抢份额”。校准只动 inclusion gap。

对 complete truth：

```text
L_gap = mean_r  w(bin_A(r)) * relu(m - M_new(r))
L_keep = mean_{r in T_near ∪ {A-selected truths}} relu(-M_new(r))
L = L_gap + L_keep
```

```text
w(negative) = 2
w(near_zero) = 1
w(comfortable) = 0
m = 0
```

`M_new` 用 **同一** exact solver 的 forced-in / forced-out。梯度只走能量，不走 MILP。可选 `||ΔU||^2`，不得拿它代替 fake 门。

### 4.3 Fake 硬约束

不是 loss 权重，不是 Lagrange 可调项。

```text
split     = same-family val payloads
rule      = complete_fake_rate(U_new) <= complete_fake_rate(U_W64)
slack     = 0
action    = 拒绝该 checkpoint
holdout   = 不参与选择
```

若所有 epoch 都被拒绝：实验结论写成 **“在硬 fake 门下校准不能被接受”**，恢复 `ΔU=0`，停止。不放宽 slack，不加大 `δ_max`，不改成 hybrid。

### 4.4 成功怎么读（81b 预注册，不得事后改）

不是 production 门。`gate_pass` 仍可以是 `null`。预注册：

| 命题 | 规则 |
|---|---|
| identity 成立 | `ΔU=0` 与 Arm A 的 selected 完全一致 |
| 校准可接受 | holdout `fake_new ≤ fake_A` **且** `eff_new ≥ eff_A + 0.01` |
| 校准无增量 | identity 成立，且不满足可接受 |
| 校准不可接受 | 没有任何 checkpoint 通过 same-family fake 硬门 |

`+0.01` 低于 WB80 的 “C beats A +0.02”，因为目标不是替代 W64。假率不允许变差。

---

## 5. 81b 授权与执行协议

**81a 已做完。81b 现已单独授权。§4 目标冻结，不得改写。**

| 已授权 | 仍禁止 |
|---|---|
| 线性 `g_θ`，`ΔU = 0.25 * tanh(g_θ(x_phys))` | 改 loss / 放 fake slack / 加大 `δ_max` |
| family-CV 两 fold，same-family val 硬门 | 扩大网络或 width/depth sweep |
| 使用 WB81a checksum / JSONL | 读 W64 logits / latent / B/C 作为 `ΔU` 输入 |
| Condor CPU 正式训练 + 批量评估 | 进入 wp4 Arm C / 扩大 hybrid |
| 校准臂名为 Arm D | 00350 / Final Blind / sealed |
| | 事后改 `+0.01` 或 success rule |

训练前必须在正式代码路径再验：

```text
θ = 0
ΔU = 0
selected_new == selected_A
complete_eff_new == complete_eff_A
complete_fake_new == complete_fake_A
all_route_purity_new == all_route_purity_A
```

任一 fold identity mismatch → fail-closed，不训练。WB81a checksum 必须仍是：

```text
family1: T_neg=1071  T_near=5718  T_comf=0  F_A=77  |M_A|<0.25 → 919
family2: T_neg=463   T_near=2785  T_comf=0  F_A=56  |M_A|<0.25 → 458
```

对不上则停止，不自行修数字。

Checkpoint：先过 same-family val `complete_fake_rate(U_new) <= complete_fake_rate(U_W64)`，`fake_slack = 0`。未过门的 epoch 直接拒绝。通过后再按 val complete efficiency 选，并列用 val `L`，再并列用更早 epoch。Holdout 不参与选模、early stop、阈值或任何超参。20 epoch 跑完，不用 patience 提前停掉后续可能过门的点。若没有任何 epoch 过门：

```text
calibration_acceptable = false
decision = reject_calibration_under_zero_fake_slack
restore = DeltaU = 0
```

并立即停止。禁止放宽 slack、加大 bound、改 loss、扩大网络、改成 WB80 hybrid。

预注册最终裁决（holdout，选模之后）：

```text
calibration_acceptable =
    holdout_fake_new <= holdout_fake_A
    AND
    holdout_eff_new >= holdout_eff_A + 0.01
```

两 fold 都满足才可写 `physics_constrained_calibration_supported = true`。  
identity 成立但不满足上式：`calibration_has_no_accepted_increment = true`，默认回到冻结 W64。  
没有任何 checkpoint 过 same-family fake 硬门：`calibration_unacceptable_under_zero_fake_slack = true`，默认回到冻结 W64。

本阶段只回答：

> 在完全冻结 W64 baseline、严格 `|ΔU|<=0.25`、且 complete fake rate 不允许恶化的条件下，raw physical information 能否安全恢复一部分 W64 boundary truth？

不是 wp4 representation study，不是 OOD safety，不是 production / alignment readiness。

---

## 6. 81b 结果（只抄 JSON）

输出树：`outputs/mc24_four_station_wb81b_calibration_v1/`。不覆盖 WB74–WB81a。  
Condor `1109770` / `1109771`，`bigbird24`，CPU，均 `return 0`。git `afc8eeb`（工作区 dirty，porcelain 已入库）。WB81a checksum 两 fold 复验 PASS（1071/5718/0/77/919 与 463/2785/0/56/458）。`θ=0` identity 两 fold PASS。W64 SHA256 仍是冻结 parent。

### 6.1 Holdout Arm A vs Arm D

| fold | arm | complete eff | complete fake | all-route purity | fragmentation | selected 2/3/4 |
|---|---|---|---|---|---|---|
| family1 | A | 0.842 (5718/6789) | 0.0133 (77/5795) | 0.9867 | 0 | 0 / 0 / 5795 |
| family1 | D | 0.936 (6352/6789) | 0.0144 (93/6445) | 0.9856 | 0 | 0 / 1 / 6445 |
| family2 | A | 0.857 (2785/3248) | 0.0197 (56/2841) | 0.9803 | 0 | 0 / 0 / 2841 |
| family2 | D | 0.857 (2785/3248) | 0.0197 (56/2841) | 0.9803 | 0 | 0 / 0 / 2841 |

family1：`Δeff = +0.093`，`Δfake = +0.00114`。eff 过了 `+0.01`，fake 变差 → 预注册 `calibration_acceptable = false`。  
family2：没有任何 epoch 过 same-family fake 硬门，恢复 `ΔU=0`，D 与 A 逐格相同。

### 6.2 安全召回账本

```text
safe recall gain = recovered truth without increasing complete fake rate
```

| fold | recovered_from_T_neg | lost_from_T_near | new_complete_fake | A-selected truth lost | A fake retained / removed | T_neg recall A→D | T_near preservation A→D |
|---|---|---|---|---|---|---|---|
| family1 | **691** / 1071 | **57** / 5718 | **29** | 57 | 64 / 13 | 0 → 0.645 | 1 → 0.990 |
| family2 | **0** / 463 | **0** / 2785 | **0** | 0 | 56 / 0 | 0 → 0 | 1 → 1 |

family1 净对 complete truth +634（691−57），与 6352−5718 对齐；complete fake 77−13+29=93。这是 **recall 头寸，不是 safe recall gain**：holdout fake 上升。

### 6.3 `ΔU` 与 inclusion gap

| fold | ΔU truth mean | ΔU fake mean | \|ΔU\| near 0.25 | inclusion gap A→D mean |
|---|---|---|---|---|
| family1 | −0.130 | +0.179 | 0.0097 | 0.119 → 0.283 |
| family2 | 0 | 0 | 0 | 0.113 → 0.113 |

饱和很少。family1 的 fake 平均拿到正残差，truth 平均略负；边界翻盘来自相对位移，不是全体 truth 被抬到饱和。

### 6.4 Same-family val 硬门

family1 的 val fake_A = 0.02384。epoch 1–19 的 val fake ≤ 该值，epoch 20 升到 0.02465 被拒。选模：过门后最高 val complete eff，并列最低 val L → **epoch 17**（val eff 0.983，val fake 0.0225）。Holdout 未参与选模。  
family2 的 val fake_A = 0.01024。20 个 epoch 的 val fake 都在 0.01044–0.0110，全部拒绝。

```text
physics_constrained_calibration_supported = false
default_system = frozen_W64_raw_energy_plus_exact_solver
```

---

## 7. 文件

本 workbook 新增 / 更新：

- `workbook/2026-09-06_81_四站PhysicsConstrainedCalibration.md`
- `configs/research_review/wp81_physics_constrained_calibration.yaml`
- `training/wb81_calibration_contract.py`
- `training/wb81b_calibration.py`
- `models/physics_constrained_calibration.py`
- `scripts/materialize_wb81a_boundary.py`
- `scripts/train_eval_wb81b_calibration.py`
- `scripts/run_wb81b_calibration_condor.sh`
- `scripts/submit_wb81b_calibration_condor.py`
- `scripts/summarize_wb81b_calibration.py`
- `tests/test_wb81a_calibration_foundation.py`
- `tests/test_wb81b_calibration.py`

修改：

- `workbook/2026-09-06_80_四站ArmB_FailureMechanism.md` §9 — 只指向本设计
- `configs/research_review/wp4_representation_study.yaml` — 注明 WB81 / 81b 不是 Arm C

未改：W64 / V5A / WB79 / WB80 权重与输出树、`baselines/route_assignment.py`。

81a 输出：`outputs/mc24_four_station_wb81a_calibration_foundation_v1/`  
81b 输出：`outputs/mc24_four_station_wb81b_calibration_v1/`

---

## 8. 明确停止的方向

- 进入 wp4 Arm C
- 放大 WB80 hybrid
- 再训一个 replacement route-energy
- 用 00350 选模
- 打开 Final Blind / sealed
- 把 `δ_max` / fake slack / `+0.01` 当超参扫
- complete-only legacy correction head
- 事后把 WB80 `hybrid_worth_entering` 改写成 true
- 把本实验解释成 representation study / OOD safety / production readiness

---

## 9. 下一允许步骤

**停止。不值得在本冻结合同上继续。**

81b 已经回答了唯一的科学问题：raw physical residual 在 `|ΔU|≤0.25` 且 fake 不允许变差时，**不能**安全恢复 W64 边界 truth。family1 证明头寸存在但不安全；family2 证明零 slack 可以把残差整折拒绝。这不是授权放宽 slack、加大 `δ_max`、改 `+0.01`、扩大网络、或改回 WB80 hybrid 的理由。

默认系统：

```text
frozen_W64_raw_energy_plus_exact_solver
```

仍不授权：wp4 Arm C、hybrid 扩大、replacement model、00350、Final Blind、sealed、domain adaptation、V5A/W64 重训。
