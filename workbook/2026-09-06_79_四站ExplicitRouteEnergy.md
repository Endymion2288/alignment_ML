# Workbook 79 — Explicit Route Energy Model v1

日期： 2026-09-06
分支： `4station`
任务： 在固定 candidate generation、truth-free 2/3/4 enumeration、`raw_energy_v1`、`route_accounting_v2`、exact solver 下，训练第一个低容量 route-energy scorer，并与冻结 W64 canonical energy 对照。

---

## 0. 结论（先写）

```text
training_authorized              = true   # 仅 WB79 Arm B family-CV
final_blind_eval_authorized      = false
sealed_test_accessed             = false
continue_v5a_frozen_head         = false
continue_to_15d_relative_wls     = false
arm_c_authorized                 = false
development_00350_used           = false
geometry_holdout_refit_used      = false
legacy_decoder_used              = false
```

本 workbook **不是** WB77/wp4 representation study 的解冻。wp4 仍暂停：geometry-holdout A/B/C 与 Arm C 仍未授权。

本 workbook **是** Dataset contract v2 下的第一阶段 scorer 实验：

| Arm | 内容 | 训练 |
|---|---|---|
| A | 冻结 W64 `sum(raw adjacent base-edge logits) + n_stations * unmatched_penalty` | 否 |
| B | `raw_physical_route_v1` → width-64 MLP → raw route core energy | 是，family holdout |

披露：W64 parent 见过全部六个授权 train sources。Arm A 是 production baseline，不是 fully source-unseen representation。

Physical Pair-Relative 在正确 train-range 内的 WB74 GATE PASS **不被本实验推翻或重做**。

CPU Condor `1109760` / `1109761` 已完成（return 0）。在固定 `raw_energy_v1` + `route_accounting_v2` + exact solver 下，**低容量 Arm B 没有超过冻结 W64 Arm A**。

| fold | holdout | Δ complete eff. (B−A) | Δ complete fake (B−A) | Δ all-route purity (B−A) |
|---|---|---|---|---|
| `holdout_family1` | family1（B 只训 family2） | −0.085 | +0.142 | −0.146 |
| `holdout_family2` | family2（B 只训 family1） | −0.022 | +0.083 | −0.084 |

`contract_gate.gate_pass = null`。这不是 production 晋升，也不是 representation study PASS/FAIL。Arm B 在更大的 train family（family1）上差距更小，但仍低于六源 W64。

---

## 1. 冻结边界

不重训 V5A / WB75 / W64。不改 W64 / V5A architecture 或 loss。不改 `baselines/route_assignment.py` 默认 legacy decoder。不打开：

```text
mc24_100047_00350_00399
mc24_100048_00350_00399
mc24_100047_00800_00849
mc24_100048_00800_00849
mc24_100116_*
mc24_100117_*
```

不用 00350 做 holdout / 选模 / 选 OP。不做 domain adaptation。不做 Arm C。不 refit seed `271828` / `314159` 几何表。

---

## 2. Model contract

```text
model_contract     = explicit_route_energy_b_v1
feature_version    = raw_physical_route_v1
feature_dim        = 54
hidden_width       = 64
utility_contract   = raw_energy_v1
solver             = exact_unit_capacity
metric_version     = route_accounting_v2
output             = raw route core energy; no sigmoid / clip / logit
U(r)               = core(r) + n_stations(r) * unmatched_penalty
unmatched_penalty  = -1.0
parent_w64_sha256  = 0c3a28704cc01151fab7ac943e41338e859b5dde1502c0b10e2f12ada04e6236
```

54-D 布局：

```text
3 adjacent hops × 11 EDGE_FEATURE_NAMES
4 stations × (x_mm, y_mm, tx, ty)
route_length
missing_station_0..3
```

缺 hop / 缺站为零；不读 W64 latent；没有 complete-only correction head。

Arm A 用 `_forward_route_batch(...).base_edge_logits`，**不**走 `sigmoid → clip → logit`，也**不**走 `_route_hypotheses` / `utility > 0` 预过滤。

---

## 3. Training config

```text
folds              = holdout_family1 | holdout_family2
validator          = training.source_transfer_cv.validate_fold_sources
train corpora      = WB74/V5A family-pure historical overlay (seed 20260813)
val payloads       = iteration_00_draw_01, iteration_00_draw_01_plus_common
                     (train family only)
candidates         = field candidates, chi2_gate=None, all adjacent pairs
enumeration        = truth-free contiguous 2/3/4 walks
loss               = LAI hinge (margin 1) + exact inclusion-gap hinge (margin 0)
optimizer          = Adam
lr                 = 3e-4
seed               = 20260905
epochs             = 20
patience           = 6
device             = cpu   # LCG torch 2.11 无 V100 CC 7.0 kernel
Arm C              = not implemented
early_stop         = patience 6；两 fold 均在 epoch 7 停下，恢复 epoch 1 最低 val
```

Fold 映射：

| fold | 训练 | 评估 |
|---|---|---|
| `holdout_family1` | family2 = 100047/100048 `00100_00149` | family1 = 100043/100044 |
| `holdout_family2` | family1 = 100043/100044 | family2 = 100047/100048 `00100_00149` |

每个 fold 固定保存：git SHA、resolved config、dataset manifest、checkpoint ancestry、evaluation JSON。

---

## 4. 文件修改列表

新增：

- `models/explicit_route_energy.py`
- `training/explicit_route_energy.py`
- `tests/test_explicit_route_energy.py`
- `scripts/train_eval_wb79_explicit_route_energy.py`
- `scripts/run_wb79_explicit_route_energy_condor.sh`
- `scripts/submit_wb79_explicit_route_energy_condor.py`
- `scripts/summarize_wb79_explicit_route_energy.py`
- `configs/research_review/wp79_explicit_route_energy.yaml`
- `workbook/2026-09-06_79_四站ExplicitRouteEnergy.md`

修改：

- `training/global_route_energy_loss.py` — 增加 `differentiable_inclusion_gap_hinge`
- `tests/test_global_route_energy_loss.py`
- `models/__init__.py` — 导出 `ExplicitRouteEnergyScorer`
- `configs/research_review/wp4_representation_study.yaml` — 仅注明 WB79 不解冻 A/B/C geometry-holdout study

未改：W64 checkpoint、V5A/WB75、`baselines/route_assignment.py`、`dataset_manifest_v2.protocol.json`、WB74–WB78 输出树。

Hermetic tests：`tests/test_explicit_route_energy.py` + `tests/test_global_route_energy_loss.py` + 既有 energy/truth-decoupling 合同测试。

---

## 5. Condor job

```text
worker      = scripts/run_wb79_explicit_route_energy_condor.sh
submitter   = scripts/submit_wb79_explicit_route_energy_condor.py
flavour     = tomorrow
device      = cpu
request     = 8 CPU, 32 GB  (Workbook-78 CPU 合同)
schedd      = bigbird24.cern.ch
output root = outputs/mc24_four_station_explicit_route_energy_v1/
```

GPU 轮次失败，不作为结果：

| cluster | 请求 | 结果 |
|---|---|---|
| 1109752 / 1109753 | 4 CPU / 32 GB → 11 CPU | 0 slot match；已 rm |
| 1109756 / 1109757 | 1 CPU / 3 GB，V100 `b9g47n1001` | return 1：LCG PyTorch 2.11 无 CC 7.0 kernel |

H100 空闲槽要求 `group_u_BE.ABP.gpu` / `group_u_ATS.u_gpu`。Arm B 是 width-64 MLP，正式训练改 CPU。

| fold | cluster | host | return | git SHA |
|---|---|---|---|---|
| holdout_family1 | 1109760 | `b9g03p6806.cern.ch` | 0 | `afc8eeb`（工作区 dirty，含未提交 WB79） |
| holdout_family2 | 1109761 | `b9p02p3945.cern.ch` | 0 | `afc8eeb`（同上） |

`1109760`：Usr 41 min，train family2 1145 events / val 480 / holdout 3360。  
`1109761`：Usr 77 min，train family1 2351 events / val 960 / holdout 1680。  
汇总：`outputs/mc24_four_station_explicit_route_energy_v1/summary.json`。

---

## 6. Baseline vs RouteEnergy-B

指标一律 `route_accounting_v2`。空分母保持 `None`。本表不是 production 晋升门。

| fold | holdout | Arm | complete eff. | complete fake | all-route purity | n events | n truth |
|---|---|---|---|---|---|---|---|
| holdout_family1 | family1 | A W64 | 0.842 (5718/6789) | 0.013 (77/5795) | 0.987 | 3360 | 6789 |
| holdout_family1 | family1 | B route-energy | 0.757 (5141/6789) | 0.155 (944/6085) | 0.840 | 3360 | 6789 |
| holdout_family2 | family2 | A W64 | 0.857 (2785/3248) | 0.020 (56/2841) | 0.980 | 1680 | 3248 |
| holdout_family2 | family2 | B route-energy | 0.836 (2714/3248) | 0.103 (312/3026) | 0.897 | 1680 | 3248 |

W64 六源泄漏必须和数字一起读。Arm B 只在互补 family 上训练。Arm A 只选 complete routes（0 fragment）；Arm B 有少量 fragment。两边 `fragmentation_rate = 0`。

读法：

1. 合同已跑通：同一 candidate / 2/3/4 enumeration / `raw_energy_v1` / exact solver / `route_accounting_v2`。
2. 第一个低容量 raw-physical energy scorer **没有**在 family holdout 上超过冻结 W64。
3. 这不能读成 “Physical Pair-Relative 失败”。WB74 仍在正确 train-range 内成立。
4. Arm B 的 val loss 从 epoch 1 起单调变差；用的是 epoch 1 checkpoint。不是 capacity sweep，也不是把 00350 当 holdout。

---

## 7. 明确停止的方向

- 重训 V5A / WB75 / W64
- architecture / loss / bound sweep
- Arm C
- domain adaptation
- 用 00350 做 holdout 或选模
- 打开 Final Blind / sealed test
- 15D relative WLS
- 把本实验读成 representation study PASS/FAIL

---

## 8. 下一允许步骤

A vs B 已落地。仍不授权：

1. Arm C；
2. 对 seed `271828` / `314159` 做 physical refit 后解冻 wp4 geometry-holdout study；
3. 用 00350 做 holdout / 选模。

当前默认维持暂停。
