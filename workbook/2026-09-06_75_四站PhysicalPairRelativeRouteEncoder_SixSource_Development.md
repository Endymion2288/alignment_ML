# Workbook 75 — 四站 Physical Pair-Relative Route Encoder V1: full six-source training + frozen development

日期： 2026-09-05/06
分支： `4station`
任务： 在 **冻结** Workbook 74 architecture 后，将 Physical Pair-Relative Route Encoder V1 训练到 Workbook-64 六源 production candidate，并在严格隔离的 frozen development 上评估。

**不重新设计模型。不推翻 Workbook 74 的 representation-transfer 结论。**

---

## 0. 继承的科学状态（Workbook 74）

```text
source_transfer_gate = true
representation_transfer_hypothesis_supported = true
full_six_source_training_authorized = true_for_workbook75_physical_pair_relative
```

Workbook 74 已证明 R_phys 优于 absolute node latent，且 2-fold source-transfer Gate A/B/C 全 PASS。
本 Workbook 的问题不再是 representation，而是：

> 冻结 architecture 后，六源 full training 的 production candidate 能否在 frozen development 上保持 transfer safety？

---

## 1. Git state

```text
local HEAD at training/eval = 841576baeb7dcfef7ff31d28e6831ac3654aa920
origin/4station             = cc8d22036be5f0381fa7ad9441477dae155d5a66
Workbook 74 HEAD            = 7a8c894970a7cd9c08df08f6a35800b2db3fc55d   (ancestor, retained)
```

未 reset，未覆盖 local history。`git push` 仍被 SSH passphrase 非交互阻塞。
本地 repository + `/eos` outputs 是科学真值。

---

## 2. Architecture freeze

唯一允许模型：Physical Pair-Relative Route Encoder V1。

```text
route_representation_mode = physical_pair_relative
R_phys = [edge01 phys(11), edge12 phys(11), edge23 phys(11), L01, L12, L23] = 36
B = 4.0
delta = 4 * tanh(raw_delta / 4)
L01/L12/L23 from frozen Workbook-64 production edge scorer
```

禁止并实际未加入：`s0..s3`、absolute node latent、Station-0 latent subtraction、
SE(3) claim、domain adversarial、new geometry features。

Pre-training audit（`pretraining_audit.json`）：

```text
training_authorized = true
trainable = 21377
frozen    = 614947
W64 parent SHA = 0c3a28704cc01151fab7ac943e41338e859b5dde1502c0b10e2f12ada04e6236
six authorized sources only; 00350_00399 / 00800_00849 / sealed absent
WB75 config vs WB74 Primary: only arm_name / description / workbook / forbidden_sources annotation differ
```

---

## 3. Training contract（继承 Workbook 74，未调参）

```text
seed = 20260822
AdamW  lr = 2e-4  weight_decay = 1e-4
batch_size = 32
30 epochs = 8 nominal + 22 local_relative_and_gauge_control
early_stopping = false
checkpoint = last_completed_epoch
```

六源（Workbook 64，未加入 development / Final Blind）：

```text
mc24_100043_00200_00299
mc24_100044_00300_00399
mc24_100043_00300_00399
mc24_100044_00200_00299
mc24_100047_00100_00149
mc24_100048_00100_00149
```

manifest SHA256：`d30b66a585ee288985d0bf9c7af3fac64bf2f9ba4d53891306349e24af8c38ad`
config SHA256：`335ff594f594bc4bf6bc6eea8837f23ce4cfe0dcc218b83433fa9fcade826299`

---

## 4. Condor jobs

Training（GPU，禁止 interactive lxplus 长训）：

```text
cluster 1109482
schedd   bigbird24.cern.ch
hostname b9pgpun209.cern.ch
GPU      NVIDIA H100L-2-24C MIG 2g.24gb
git      841576b  dirty=false
return   0
```

Development evaluation（CPU）：

```text
cluster 1109484
hostname b9g23p4601.cern.ch
git      841576b  dirty=false
return   0
```

Condor return 0 不是科学 PASS。

---

## 5. Stage 1 — training sanity

```text
full_training_status = completed_return_0
epochs_completed     = 30
graphs               = 5040 (nominal stage 720)
checkpoint SHA256    = b961cefdf499f648487f05bac381e2bc02d413769419c9f041961fa072d94f02
parent SHA256        = 0c3a28704cc01151fab7ac943e41338e859b5dde1502c0b10e2f12ada04e6236
trainable / frozen   = 21377 / 614947
zero-init max|delta| = 0
post-training max(|edge_new-edge_W64|) = 2.86e-06
frozen hash invariant = true
W64 replica hash invariant = true
```

Loss（finite，各 stage 内下降）：

```text
epoch 01 nominal     total = 0.2066
epoch 08 nominal     total = 0.0902
epoch 09 curriculum  total = 0.1209
epoch 30 curriculum  total = 0.0764
```

Train-split delta（含大量 fake；median=-4 是 fake 抑制，不是 truth 诊断）：

```text
count=597781  min=-4  max=+0.600  mean=-3.913  median=-4
```

---

## 6. Stage 2 — frozen development evaluation

仅使用：

```text
mc24_100047_00350_00399
mc24_100048_00350_00399
```

未根据 development 修改 architecture。未打开 Final Blind / sealed test。

W64 baseline reproduction（sanity）：

```text
C=200  D=170
```

与 Workbook 65/70 冻结的 W64 development C/D 完全一致。

| Arm | C | D | selected | efficiency | purity | fake |
|-----|----|----|----------|------------|--------|------|
| W64 | 200 | 170 | 3008 | 0.8905 | 0.9486 | 1511 |
| Primary | **585** | 148 | 2645 | **0.7830** | 0.9545 | 1886 |

```text
catastrophic_truth_destruction (W64-selected -> U_truth<=0) = 274
catastrophic / W64_selected_truth = 274/3008 = 0.0911
C_primary / truth_count           = 585/3378 = 0.1732
new_C_from_selected = 274
new_C_from_D        = 111
efficiency drop vs W64 = 0.1075
fake_rate_W64     = 0.3344
fake_rate_primary = 0.4162   (increase = 0.0819)
purity_primary    = 0.9545 >= purity_W64 0.9486   (quality 中唯一通过的一条)
```

W64 → Primary 转移（truth routes）：

```text
selected -> utility_nonpositive     274   # catastrophic
selected -> selected               2640
selected -> packing_competition      94
packing_competition -> utility_nonpositive  111
packing_competition -> selected       5
```

Primary development truth delta：

```text
mean=-0.359  median=+0.303  frac_neg=0.317
frac_saturated_neg (near -4) = 0.060     # WB74 held-out truth near-4 = 0.000
max_abs=4.0
```

与 Workbook 74 held-out truth（near-4 = 0，mean 正 correction）对比：
六源训练后的 encoder 对 development truth 给出了可饱和的负 correction，
这是 catastrophic amplifier 再次出现的直接证据。

---

## 7. Development gate（训练前冻结，沿用 Workbook 74 Gate B+C）

**Safety** FAIL：

```text
C_primary / truth_count = 0.173  <= 0.01   FAIL
catastrophic / W64_sel  = 0.091  <= 0.01   FAIL
efficiency >= W64-0.01                    FAIL  (drop 0.107)
```

**Quality** FAIL：

```text
purity >= W64-0.01   PASS  (实际升高 0.006)
fake_rate increase <= 0.01   FAIL  (increase 0.082)
```

**C/D vs W64**：C 恶化（585 > 200）；D 略改善（148 < 170）。C 恶化主导。

```text
development_gate = FAIL
```

---

## 8. Final decision

Workbook 74 的 representation-transfer 结论 **保持**：R_phys 仍优于 absolute node latent，
source-transfer CV 仍 PASS。本 Workbook **没有**重开那个问题。

Workbook 75 新结论：

```text
full_training_status = completed_return_0
checkpoint SHA256    = b961cefdf499f648487f05bac381e2bc02d413769419c9f041961fa072d94f02
development_gate     = FAIL
six_source_production_candidate_supported = false
stop_physical_pair_relative_v1_six_source_production = true
```

六源 full training 没有把 Workbook 74 的 transfer safety 兑现到 frozen development。
不得因为 development FAIL 去改 B / loss / architecture（避免 `CV → look at dev → rescue`）。

---

## 9. 边界状态

```text
development used for evaluation only; not used to change architecture
new_final_blind_content_accessed = false
sealed_test_accessed             = false
continue_to_15d_relative_wls     = false
final_blind_eval_authorized      = false
next_workbook_authorized         = false
```

Final Blind `00800_00849` 未以任何形式访问。

---

## 10. 是否授权下一 Workbook

**否。** 不授权 Final Blind。不授权继续放大 MLP / 调 B / 调 loss / domain adversarial。

若未来另开 Workbook，唯一科学方向是 **failure-mechanism audit**：
比较 diagnostic logistic R_phys 与本六源非线性 encoder 为何在 development 上分叉。
该方向 **本任务不执行**。

---

## Outputs

```text
outputs/mc24_four_station_physical_pair_relative_route_v1_six_source/
  pretraining_audit.json
  training_sanity.json
  decision.json
  training/primary/checkpoint_last.pt
  training/primary/checkpoint_freeze.json
  training/condor/          # cluster 1109482
  development_eval/development_evaluation.json
  development_eval/condor/  # cluster 1109484
```
