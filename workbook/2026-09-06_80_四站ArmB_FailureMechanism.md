# Workbook 80 — Arm B Failure Mechanism

日期： 2026-09-06
分支： `4station`
任务： 分析 WB79 Arm B 相对冻结 W64 的失败机制，而不是调参。

---

## 0. 结论（先写）

```text
training_authorized              = true   # 仅 WB80 hybrid ablation C family-CV
final_blind_eval_authorized      = false
sealed_test_accessed             = false
continue_v5a_frozen_head         = false
continue_to_15d_relative_wls     = false
arm_c_authorized                 = false  # wp4 representation Arm C
wp4_arm_c_authorized             = false
v5a_retrained                    = false
domain_adaptation_authorized     = false
development_00350_used           = false
legacy_decoder_used              = false
```

本 workbook **不是** 调参，也 **不是** wp4 representation study 解冻。

本 workbook **是** 在固定 candidate / enumeration / `raw_energy_v1` / exact solver / `route_accounting_v2` 下，对 WB79 Arm B vs W64 做 route-level stratified audit，并用三个 offline scorer 归因：

| Arm | 内容 | 训练 |
|---|---|---|
| A | 冻结 W64 `sum(raw adjacent base-edge logits) + n_stations * unmatched_penalty` | 否 |
| B | 冻结 WB79 `raw_physical_route_v1` width-64 MLP | 否，读 WB79 checkpoint |
| C | raw physical 54-D + 3 hop W64 logits + logit sum + hop count → width-64 MLP | 是，family holdout |

披露：W64 parent 见过全部六个授权 train sources。Arm A 是 production baseline，不是 fully source-unseen representation。

Physical Pair-Relative 在正确 train-range 内的 WB74 GATE PASS **不被本实验推翻或重做**。

CPU Condor `1109766` / `1109767` 已完成（return 0）。A/B 数字与 WB79 逐格重合。`gate_pass = null`。`interpretation.json` 两 fold 预注册命题全部为 false。

| 问题 | 预注册判定 |
|---|---|
| W64 superiority 来自 edge information 还是 solver aggregation？ | **inconclusive**。`w64_superiority_from_edge_information = false`，`w64_superiority_from_solver_aggregation = false`。 |
| physical feature 是否缺少信息？ | **不能写成“缺 in-sample 信息”**。`physical_features_lack_information = false`。 |
| 是否值得进入 hybrid route-energy？ | **否**。`hybrid_worth_entering = false`。 |

观察（不改规则）：C 提高 holdout complete eff.（+0.061 / +0.076）但 fake 也更高（+0.062 / +0.035），因此不 beat A。B 的 complete-truth Spearman(`U_A`,`U_B`) 只有 0.273 / 0.275，不是“同一套分数、solver 用法不同”。C 也没有复制 W64 sum（Spearman(`U_A`,`U_C`) 0.202 / 0.032）。same-family val 不对称：B 在小 family 上 Δeff=−0.224，在大 family 上 Δeff=+0.037。默认维持暂停，不调参。

---

## 1. 冻结边界

不重训 V5A / WB75 / W64 / WB79 Arm B。不改 W64 / V5A architecture 或 loss。不改 `baselines/route_assignment.py` 默认 legacy decoder。不打开：

```text
mc24_100047_00350_00399
mc24_100048_00350_00399
mc24_100047_00800_00849
mc24_100048_00800_00849
mc24_100116_*
mc24_100117_*
```

不用 00350 做 holdout / 选模 / 选 OP。不做 domain adaptation。不做 wp4 Arm C。不 refit seed `271828` / `314159` 几何表。不超参扫描。

---

## 2. 合同

```text
utility_contract        = raw_energy_v1
solver                  = exact_unit_capacity
metric_version          = route_accounting_v2
enumeration             = truth-free contiguous 2/3/4
U(r)                    = core(r) + n_stations(r) * unmatched_penalty
unmatched_penalty       = -1.0
parent_w64_sha256       = 0c3a28704cc01151fab7ac943e41338e859b5dde1502c0b10e2f12ada04e6236
arm_b_model_contract    = explicit_route_energy_b_v1
hybrid_model_contract   = hybrid_route_energy_c_v1
hybrid_feature_version  = raw_physical_plus_w64_logits_v1
hybrid_feature_dim      = 59
hidden_width            = 64
seed                    = 20260905
epochs / patience / lr  = 20 / 6 / 3e-4
device                  = cpu
```

Arm A 用 `_forward_route_batch(...).base_edge_logits`，**不**走 `sigmoid → clip → logit`，也**不**走 `_route_hypotheses` / `utility > 0` 预过滤。

Ablation C **不是** wp4 的 trainable physical representation。它只是把冻结 W64 raw logits 接到同一低容量 energy scorer 上，用来判断 B 的缺口是缺边信息还是 scorer / solver 用法。

---

## 3. Audit strata

对 holdout 每个 event：同一 candidate、同一 2/3/4 集合、三个 scorer、同一 exact solver。

| 层 | 定义 |
|---|---|
| 2/3/4 station | `len(route.stations)` |
| route length | 同上；selected 集合按 length 计 truth / fake |
| W64 margin | 仅 complete truth；Arm A inclusion gap：`negative` `<0`，`near_zero` `[0,1)`，`comfortable` `≥1` |
| truth / fake | `truth_consistent` |
| near-dustbin | `|U| < 1.0` |

另评 **same-family val payloads**（`iteration_00_draw_01`, `iteration_00_draw_01_plus_common`）：同一 family、未进入 B/C 训练。用来把 “缺特征” 和 “source transfer” 分开。

Spearman / Pearson 只在 complete-truth `U(r)` 上算。

---

## 4. 预注册读法

缺测保持 `None`，fail-closed。不改阈值迁就数字。

| 命题 | 规则 |
|---|---|
| C recovers A | `\|eff_C−eff_A\|≤0.02` 且 `\|fake_C−fake_A\|≤0.02` |
| C beats A | `eff_C ≥ eff_A+0.02` 且 `fake_C ≤ fake_A` |
| B tracks A ranking | Spearman(`U_A`,`U_B`) on complete truths `≥ 0.80` |
| physical 缺 in-sample 信息 | same-family val `eff_B−eff_A ≤ −0.03` **且** C recovers A |
| W64 superiority 来自 edge information | C recovers A **且** B 不 track A ranking |
| W64 superiority 来自 solver aggregation | B tracks A ranking **且** C 不 recover A |
| 值得进入 hybrid | C beats A |
| hybrid 相对 W64 sum 无增量 | C recovers A **且** 不 beat A |
| scorer 用不上 W64 logits | C ≈ B（eff 差 `≤0.02`）且 C 不 recover A 且 `eff_C ≤ eff_A` |

三问对应：

1. edge vs solver：上表第 5 / 6 行。两者都假则写 inconclusive，不编故事。
2. physical 是否缺信息：第 4 行；若只在 holdout 差、same-family 不差，则读成 transfer，不读成特征空洞。
3. 是否进 hybrid：第 7 行。C 只是追上 A 则停在 W64 sum，不升 hybrid 为默认。

---

## 5. 文件修改列表

新增：

- `models/hybrid_route_energy.py`
- `training/wb80_failure_mechanism.py`
- `training/explicit_route_energy.py` 中 `w64_edge_logit_matrix`
- `tests/test_wb80_failure_mechanism.py`
- `scripts/train_eval_wb80_failure_mechanism.py`
- `scripts/run_wb80_failure_mechanism_condor.sh`
- `scripts/submit_wb80_failure_mechanism_condor.py`
- `scripts/summarize_wb80_failure_mechanism.py`
- `configs/research_review/wp80_arm_b_failure_mechanism.yaml`
- `workbook/2026-09-06_80_四站ArmB_FailureMechanism.md`

修改：

- `configs/research_review/wp4_representation_study.yaml` — 仅注明 WB80 hybrid C 不解冻 geometry-holdout A/B/C

未改：W64 checkpoint、V5A/WB75、WB79 Arm B 权重、`baselines/route_assignment.py`、`dataset_manifest_v2.protocol.json`、WB74–WB79 输出树。

---

## 6. Condor job

```text
worker      = scripts/run_wb80_failure_mechanism_condor.sh
submitter   = scripts/submit_wb80_failure_mechanism_condor.py
flavour     = tomorrow
device      = cpu
request     = 8 CPU, 32 GB
output root = outputs/mc24_four_station_wb80_failure_mechanism_v1/
```

GPU V100 / H100 VO 失败只记为运维，不进科学表。正式作业必须 CPU。

| fold | cluster | host | return | Usr | git SHA |
|---|---|---|---|---|---|
| holdout_family1 | 1109766 | `b9g03p3851.cern.ch` | 0 | 66 min | `afc8eeb`（工作区 dirty，含未提交 WB79/WB80） |
| holdout_family2 | 1109767 | `b9p19p0336.cern.ch` | 0 | 117 min | `afc8eeb`（同上） |

`bigbird24` 把 `request_cpus=8` / `request_memory=32000` 改写成 `11` / `33000`（约 3 GB/CPU）。实际 MemoryUsage 887 / 977 MB。这是运维，不是科学结果。

C 早停：family1 恢复 epoch 1（val 从 1.566 起变差，epoch 7 停）；family2 恢复 epoch 3（val 0.450，epoch 9 停）。汇总：`outputs/mc24_four_station_wb80_failure_mechanism_v1/summary.json`。

---

## 7. 结果表

指标一律 `route_accounting_v2`。空分母保持 `None`。本表不是 production 晋升门。`gate_pass = null`。数字只从 fold JSON 抄。

### 7.1 Holdout A / B / C

| fold | holdout | Arm | complete eff. | complete fake | all-route purity | n events | n truth |
|---|---|---|---|---|---|---|---|
| holdout_family1 | family1 | A W64 | 0.842 (5718/6789) | 0.013 (77/5795) | 0.987 | 3360 | 6789 |
| holdout_family1 | family1 | B physical | 0.757 (5141/6789) | 0.155 (944/6085) | 0.840 | 3360 | 6789 |
| holdout_family1 | family1 | C hybrid | 0.903 (6133/6789) | 0.075 (497/6630) | 0.925 | 3360 | 6789 |
| holdout_family2 | family2 | A W64 | 0.857 (2785/3248) | 0.020 (56/2841) | 0.980 | 1680 | 3248 |
| holdout_family2 | family2 | B physical | 0.836 (2714/3248) | 0.103 (312/3026) | 0.897 | 1680 | 3248 |
| holdout_family2 | family2 | C hybrid | 0.934 (3033/3248) | 0.054 (174/3207) | 0.946 | 1680 | 3248 |

A/B 与 WB79 逐格重合。C−A：family1 Δeff +0.061、Δfake +0.062；family2 Δeff +0.076、Δfake +0.035。A 只选 complete；B 有少量 fragment（family1 37，family2 1）；C 几乎不选 fragment（family1 1 条 truth 3-station）。

### 7.2 Same-family val

| fold | 训练 family | val family | Arm | complete eff. | complete fake | n events | n truth |
|---|---|---|---|---|---|---|---|
| holdout_family1 | family2 | family2 | A | 0.838 (778/928) | 0.024 (19/797) | 480 | 928 |
| holdout_family1 | family2 | family2 | B | 0.614 (570/928) | 0.232 (172/742) | 480 | 928 |
| holdout_family1 | family2 | family2 | C | 0.891 (827/928) | 0.098 (90/917) | 480 | 928 |
| holdout_family2 | family1 | family1 | A | 0.847 (1643/1940) | 0.010 (17/1660) | 960 | 1940 |
| holdout_family2 | family1 | family1 | B | 0.884 (1714/1940) | 0.096 (182/1896) | 960 | 1940 |
| holdout_family2 | family1 | family1 | C | 0.965 (1872/1940) | 0.039 (75/1947) | 960 | 1940 |

B−A same-family Δeff：−0.224（小 family 上训）/ +0.037（大 family 上训）。不是两侧同号的“特征空洞”。

### 7.3 Stratified audit

**by_length（holdout selected）**

| fold | Arm | 2-station | 3-station | 4-station truth | 4-station fake | 4-station near-dustbin |
|---|---|---|---|---|---|---|
| family1 | A | 0 | 0 | 5718 | 77 | 5795 / 5795 |
| family1 | B | 26 fake | 11 (3 truth / 8 fake) | 5141 | 944 | 1599 / 6085 |
| family1 | C | 0 | 1 truth | 6133 | 497 | 1260 / 6630 |
| family2 | A | 0 | 0 | 2785 | 56 | 2841 / 2841 |
| family2 | B | 1 fake | 0 | 2714 | 312 | 604 / 3026 |
| family2 | C | 0 | 0 | 3033 | 174 | 166 / 3207 |

A 的全部 selected 都落在 `|U|<1`。W64 complete energy 贴着 dustbin，near-dustbin 层不能分开 A 的决策。

**W64 inclusion-gap bins（仅 complete truth）**

| fold | bin | n | recall A | recall B | recall C | mean gap A |
|---|---|---|---|---|---|---|
| family1 | comfortable `≥1` | 0 | — | — | — | — |
| family1 | near_zero `[0,1)` | 5718 | 1.000 | 0.783 | 0.919 | 0.167 |
| family1 | negative `<0` | 1071 | 0.000 | 0.617 | 0.821 | −0.139 |
| family2 | comfortable `≥1` | 0 | — | — | — | — |
| family2 | near_zero `[0,1)` | 2785 | 1.000 | 0.856 | 0.945 | 0.143 |
| family2 | negative `<0` | 463 | 0.000 | 0.711 | 0.868 | −0.065 |

没有一条 complete truth 的 A inclusion gap ≥ 1。C 主要在 A 的 negative bin 里多捞 truth，同时 fake 上升。

**complete-truth energy 相关与 B 错分**

| fold | Spearman A–B | Spearman A–C | Pearson A–B | Pearson A–C | B extra fake | 其中 near-dustbin B | B missed A-selected truth |
|---|---|---|---|---|---|---|---|
| family1 | 0.273 | 0.202 | 0.254 | 0.193 | 944 | 428 | 1238（全部 `|U_A|<1`） |
| family2 | 0.275 | 0.032 | 0.261 | 0.014 | 282 | 88 | 400（全部 `|U_A|<1`） |

### 7.4 预注册命题

从 `interpretation.json` 抄，不改写。两 fold 相同：

```text
c_recovers_a                              = false
c_beats_a                                 = false
b_tracks_a_ranking                        = false
w64_superiority_from_edge_information     = false
w64_superiority_from_solver_aggregation   = false
physical_features_lack_information        = false
hybrid_worth_entering                     = false
hybrid_not_justified_beyond_w64_sum       = false
scorer_cannot_use_w64_logits              = false
```

family1 触发 C 不 recover 的是 `|Δeff|=0.061` 与 `|Δfake|=0.062`，都超过 0.02；`eff_C ≥ eff_A+0.02` 成立但 `fake_C ≤ fake_A` 不成立。family2 同形：`|Δeff|=0.076`，`|Δfake|=0.035`。Spearman A–B 0.27 < 0.80。

---

## 8. 明确停止的方向

- 重训 V5A / WB75 / W64 / WB79 Arm B
- architecture / loss / bound sweep
- 把 C 的高 recall / 高 fake 当成调参目标
- wp4 Arm C
- domain adaptation
- 用 00350 做 holdout 或选模
- 打开 Final Blind / sealed test
- 15D relative WLS
- 把本实验读成 representation study PASS/FAIL
- 事后改预注册阈值让 `hybrid_worth_entering` 变 true

---

## 9. 下一允许步骤

预注册三选一落在第 3 条：**命题 inconclusive，维持暂停，不调参。** 不进入 hybrid 为默认，也不把“C 用不上 logits”写成结论（C 相对 B 的 holdout Δeff 为 +0.146 / +0.098，大于 0.02）。

下一设计（不在本 workbook 授权）：Workbook 81 Physics-constrained Route Energy Calibration。仍不授权：wp4 Arm C、hybrid 扩大、replacement model、00350 holdout、Final Blind、V5A 重训。
