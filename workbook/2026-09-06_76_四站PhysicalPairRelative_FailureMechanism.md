# Workbook 76 — Physical Pair-Relative failure mechanism attribution

日期： 2026-09-06
分支： `4station`
任务： **只读归因**。不重训、不改 architecture / loss / B=4、不打开 Final Blind / sealed test。

---

## 0. 继承状态（不推翻）

```text
WB74: source_transfer_gate = true
      representation_transfer_hypothesis_supported = true

WB75: development_gate = FAIL
      six_source_production_candidate_supported = false
      W64 dev: C=200 D=170 eff=0.8905
      Primary: C=585 D=148 eff=0.7830
      catastrophic = 274 / 3008 = 9.11%
```

唯一问题：

> 为什么同一个 Physical Pair-Relative representation 在 two-fold source-transfer head-only 中安全，但 six-source training 后在 frozen development 上产生大量 destructive negative corrections？

---

## 1. Git / 冻结边界

```text
audit commit = 617b6598371c8f9f6418f1e10eb6852e7b2a1fdf
local HEAD   = (this workbook commit)
origin/4station = cc8d22036be5f0381fa7ad9441477dae155d5a66
WB74 HEAD    = 7a8c894  (ancestor, retained)
WB75 HEAD    = bbe9cdf  (ancestor, retained)
```

```text
retrained = false
architecture_changed = false
loss_changed = false
bound_changed = false
new_final_blind_content_accessed = false
sealed_test_accessed = false
continue_to_15d_relative_wls = false
```

Condor CPU：`cluster 1109741`，hostname `b9p13p1867.cern.ch`，return 0。

---

## 2. Step 1 — Frozen checkpoint audit

读取 `outputs/mc24_four_station_physical_pair_relative_route_v1_six_source/`，不重训。

```text
checkpoint SHA256 = b961cefdf499f648487f05bac381e2bc02d413769419c9f041961fa072d94f02   OK
frozen W64 parent = 0c3a28704cc01151fab7ac943e41338e859b5dde1502c0b10e2f12ada04e6236   OK
route_representation_mode = physical_pair_relative   OK
route_input_width = 36   OK
B = 4.0   OK
absolute node latent modules absent   OK
WB75 freeze: post-training max(|edge-W64|) = 2.86e-06   OK
```

`checkpoint_contract.json` 通过。

---

## 3. Step 2 — Delta mechanism：同一 head，train vs development

对照组（全部 inference-only）：

| Head | Corpus | C | D | selected | eff | catastrophic vs W64 |
|------|--------|---|---|----------|-----|---------------------|
| W64 | six-source **train** | 0 | 38 | 9855 | 0.9962 | — |
| WB75 six-source | six-source **train** | **1** | 43 | 9849 | 0.9956 | **1** |
| W64 | development 00350 | 200 | 170 | 3008 | 0.8905 | — |
| WB75 six-source | development 00350 | **585** | 148 | 2645 | 0.7830 | **274** |
| WB74 Fold1 (Family2-only) | development 00350 | **432** | 196 | 2750 | 0.8141 | **136** |
| WB74 Fold2 (Family1-only) | development 00350 | **478** | 184 | 2716 | 0.8040 | **180** |

Truth-route bounded delta：

```text
WB75 on TRAIN truth (n=9893):
  mean=+0.400  median=+0.422  frac_neg=0.008  frac_sat_neg=0.000  min=-1.96

WB75 on DEV truth (n=3378):
  mean=-0.359  median=+0.303  frac_neg=0.317  frac_sat_neg=0.060  min=-4.0

WB74 Fold1 (family2-only) on DEV truth:
  mean=-0.303  median=+0.030  frac_neg=0.457  frac_sat_neg=0.016  catastrophic=136

WB74 Fold2 (family1-only) on DEV truth:
  mean=+0.064  median=+0.501  frac_neg=0.301  frac_sat_neg=0.024  catastrophic=180
```

**为什么会看到 delta≈−4？**

1. **In-sample 不会。** 六源训练集上 truth `frac_sat_neg=0`，C=1。head 在训练支持内把 truth 推向 +0.4，把 fake 推向 −4（fake sat_neg=0.993）。
2. **Development 上 202 条 truth 饱和到 −4，但其中只有 15 条是新 catastrophic。** 其余主要是 W64 已经 `U<=0` 的 200 条（already_nonpositive：`frac_sat_neg=0.87`）。head 在这些路上“同意”W64。
3. **真正的 274 条 catastrophic（W64 选中 → U<=0）中位 delta=−2.23，不是 −4。** 只有 5.5% 饱和。破坏来自 **中等负 correction × 低 W64 margin**：`U = L_edge + delta − 4`。若 W64 的 `U≈1`，`delta≈−2` 就过零。

因此 “delta≈−4” 不是 274 条新破坏的主形态；主形态是对 **W64 弱 / near-dustbin truth** 的 OOD 负外推。

---

## 4. Step 3 — Stratified failure

### 4.1 Source family

Manifest 合同（不是猜测）：

```text
Train sources = {100043_00200, 100043_00300, 100044_00200, 100044_00300, 100047_00100, 100048_00100}
Dev  sources  = {100047_00350, 100048_00350}
```

单事件 source 标签：six-source / development 样本的 `sample.source_id` 是 `pooled_train` / `pooled_validation`。uid 映射在本 run 未落到单事件（见 §4.4），因此 **不能** 再按 100043 vs 100044 切开 train truth。Development 在 corpus 层只有 Family2 的 00350 段。

关键对照已经足够：

- Family2-only head（WB74 Fold1，训练 100047/100048 的 **00100_00149**）在 development **00350_00399** 上 C=432、catastrophic=136。
- 所以失败 **不是** “六源混合物” 独有。同 DSID、不同 run range 已经失败。

### 4.2 Station pair（S3 敏感，与 WB65 一致）

Development truth 的 R_phys KS（train truth vs dev truth）最大维：

```text
e12_delta_z_mm   KS=0.429
L23              KS=0.341   mean shift = −2.20 train-std
e01_delta_z_mm   KS=0.287
e23_delta_z_mm   KS=0.286
e23 residual_y / pull_y / residual_ty ...
```

Catastrophic vs survived 的冻结 W64 边 logits：

```text
          L01 mean   L12 mean   L23 mean
train     2.26       1.42       2.28
survived  2.21       1.38       2.20
destroyed 1.77       1.29       1.81
```

破坏集在 **L23 和 L01 上明显更弱**。`corr(delta, L_edge)=+0.79`：W64 越弱，head 给的负 correction 越大。

### 4.3 Route quality（W64 U 分层）

```text
W64 U bin            n     C_like   catastrophic   mean delta   sat_neg
already_nonpositive  200   200      0              -3.99        0.87
near_dustbin (0,1]   282   279      179            -2.85        0.064
low_margin (1,2]    2230   103       93            +0.06        0.004
high_margin (>2)     666     3        2            +0.39        0.002
```

W64 已经失败的 200 条被进一步压到 −4。**新破坏集中在 near-dustbin（179/274）和 low-margin（93/274）**。高 margin truth 几乎不被毁。

W64 `packing_competition`（fragment / D）170 条：mean delta=−2.18，C_like=111（D→C）。

### 4.4 Particle sign

代码合同（`training/source_diversity_audit.py`，非猜测）：

```text
SOURCE_CHARGE = {100043: mu_minus, 100044: mu_plus, 100047: mu_minus, 100048: mu_plus}
EventTracklets.truth_pdg 存在（13 / −13）。
```

本 run 单事件 charge 切片 **未完成**：pooled `source_id` 使 `source_dsid` 得不到 100047/100048，`agreement_rate=0`。因此 **不报告 mu+/mu− 对比，避免编造**。Development corpus 在 manifest 层同时含 mu−（100047）与 mu+（100048）。

---

## 5. Step 4 — R_phys boundary

36 维 R_phys（标准化 pair-relative 11×3 + W64 production L01/L12/L23）。

```text
max KS (train truth vs dev truth) = 0.429
dominant: delta_z_mm on 01/12/23, and L23
```

NN 距离（dev truth → 最近 train truth R_phys）：

```text
all dev truth     median=1.07  mean=1.42  q99=6.30
survived          median=0.99  mean=1.06  max=4.84
destroyed         median=1.67  mean=3.02  max=23.0
```

destroyed−survived median gap = **+0.68**。NN top-10% 的 catastrophic 比例 0.228，相对 overall 0.091 高 2.5 倍。

**存在 development R_phys 落在训练未覆盖区，且该区与破坏同向。**

---

## 6. Step 5 — Function extrapolation

`R_phys → delta`（不是 representation 定义问题）：

```text
corr(delta, NN distance to train truth) = −0.50
corr(delta, L_edge)                     = +0.79
```

机制链：

```text
OOD / weak-L23 R_phys
        |
        v
route head 负外推（常为 −2，少数到 −4）
        |
        v
U = L_edge + delta − 4 过零
        |
        v
W64-selected truth destroyed
```

In-sample 该映射对 truth 是小正 correction；只在 development 支持外变成破坏。

---

## 7. Step 6 — Loss vs solver

训练目标（未改）：route body + packing competition + dustbin margin + gauge twin。
Solver：`U_complete = L_corrected − 4`，fragment 竞争，dustbin=0。

```text
WB75 TRAIN: fake sat_neg=0.993 , truth sat_neg=0.000 , C=1
WB75 DEV:   fake sat_neg=0.992 , truth sat_neg=0.060 , C=585
```

Loss **在训练集上做到了它被要求的事**：猛压 fake、几乎不压 train truth。  
Solver 在 development 上看到的是：对 **弱 W64 truth** 的负外推把 `U` 推过 0。

所以 loss–solver 不一致是 **次级放大器**（目标不保护 “W64-weak but selected truth 的 U>0”），**不是** WB75 崩溃的主因——否则 train 上也会大规模毁 truth。

---

## 8. Hypothesis table

| Hypothesis | Evidence | Status |
|---|---|---|
| data boundary shift | max KS=0.43（`delta_z`, **L23**）；destroyed NN median 1.67 vs survived 0.99；corr(delta,NN)=−0.50 | **supported（主因）** |
| source mixture conflict | Family2-only WB74 Fold1 在 dev 上已 C=432 / cat=136；六源只是更差（585/274），不是开关 | **not primary** |
| nonlinear extrapolation | train truth sat_neg=0 且 mean delta=+0.40；dev 上弱/OOD 点被推到负；corr(delta,L_edge)=+0.79 | **supported（机制）** |
| loss-solver mismatch | train 上 fake≈−4、truth 安全；solver `U=L+delta−4` 放大 near-dustbin | **supported（放大器，非主因）** |

---

## 9. Failure cohort

WB75 on development，**catastrophic**（W64 selected → `U_truth<=0`）：

```text
n_routes = 274
n_events = 255
C contribution = 274 / 585 的新破坏来自 W64-selected；另有 new_C_from_D = 111
source (corpus) = {mc24_100047_00350_00399, mc24_100048_00350_00399}
delta: mean=-2.40  median=-2.23  min=-4  max=-1.00  frac_sat_neg=0.055
L23 mean=1.81  (survived 2.20, train 2.28)
NN-to-train-truth median=1.67 (survived 0.99)
```

数组：`failure_cohort_catastrophic.npz`。

---

## 10. Decision（三选一）

```text
C. Data contract mismatch
```

理由（机械规则 + 上表）：

1. 同一 architecture 在 **六源 train 上几乎完美**（C=1，truth sat_neg=0）。representation + 训练目标在训练合同内成立。
2. **WB74 两个 family-only head 在 development 上同样 FAIL**（C=432 / 478）。根因不是 six-source mixture，也不是 “WB75 训练坏了 representation”。
3. Development `00350_00399` 的 R_phys / **L23** 相对 train-range 有大分布移动；OOD 距离预测负 delta 与破坏。
4. WB74 PASS 测的是 **训练 run-range 的跨 family holdout**（00100/00200/00300），从来没有测过 **00350 development**。WB75 第一次打开它，合同就破了。

不是 A（训练目标单独失败）：in-sample 目标成功。  
不是 B（representation 本身不够）：R_phys 在 WB74 的 train-range 合同上仍然可迁移。

---

## 11. 必须回答的四个问题

### 1. 为什么 WB74 PASS → WB75 FAIL？

WB74 证明的是：在 **已授权六源的 train-range 子集** 上，去掉 absolute node latent 后，跨 family 的 truth/fake 边界可迁移。  
WB75 把同一个 head 训满六源后，第一次在 **不同 run range 的 development（00350）** 上评估。00350 的 pair-relative 几何（尤其 `delta_z` 与 **L23**）不在 train support 内。任何已训的 pair-relative head（六源或单 family）都会在那里对弱 W64 truth 给出负 correction，solver 把 near-dustbin 真理打成 C。六源训练加重了破坏（585 vs 432），但 **不是开关**。

### 2. 失败发生在哪一层？

```text
representation          OK   （WB74 train-range 仍成立；36-dim R_phys 未改）
        |
feature distribution    FAIL 主因  （dev vs train KS / L23 / NN OOD）
        |
route head              FAIL 机制  （OOD/弱 L → 负外推）
        |
loss                    OK in-sample （train C=1；不解释 dev 崩溃）
        |
solver                  放大器  （U=L+delta−4；near-dustbin 过零）
```

### 3. 是否值得继续 Physical Pair-Relative？

**值得作为 representation 假设留下**（WB74 未推翻）。  
**不值得作为 “再训一个更大 head / 调 B / 调 loss 然后进 Final Blind” 的 production 路线。**  
当前阻塞是 **数据合同：development 00350 与 train-range 不可交换**，不是 “换一个网络就能过 development gate”。

### 4. 若继续，下一 Workbook 研究什么？

只允许 **data-contract / run-range audit**（仍只读，仍禁止 Final Blind）：

- 100047/100048 的 `00100_00149` vs `00350_00399`：物理 `delta_z`、S3 边、W64 L23 为何移动；
- 这些移动是 reconstruction / geometry / overlay curriculum 哪一层引入的（查 ROOT metadata 与已有 repropagation 合同，不猜）；
- 在 **不改模型** 的前提下，development 是否根本不该作为 “与 train 可交换的 gate”。

不授权：新模型、B sweep、loss reweight、domain adversarial、Final Blind、15D WLS。

---

## Outputs

```text
outputs/mc24_four_station_physical_pair_relative_route_v1_failure_audit/
  checkpoint_contract.json
  failure_mechanism_summary.json
  decision.json
  failure_cohort_catastrophic.npz
  condor/   # 1109741
```
