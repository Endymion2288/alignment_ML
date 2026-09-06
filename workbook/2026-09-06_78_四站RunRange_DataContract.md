# Workbook 78 — Run-range Data Contract Audit

日期： 2026-09-06
分支： `4station`
任务： **只读 data-contract audit。** 不重训 V5A/WB75，不改 architecture / loss / B，不打开 Final Blind / sealed test。

---

## 0. 结论（先写）

```text
training_authorized = false
final_blind_eval_authorized = false
sealed_test_accessed = false
continue_to_v5a_frozen_head = false
continue_to_15d_relative_wls = false
```

**development `00350-00399` 不能直接当作 `00100-00149` 的交换测试。**

这不是因为 Physical Pair-Relative representation 在 train-range 内失败（WB74 仍然成立），而是因为 train 与 development 不是同一个 data contract。

HTCondor `cluster 1109749` 已完成（`b9p13p0517.cern.ch`，return 0，git `146e28b`）。源纯物理表见 `outputs/mc24_four_station_run_range_data_contract_v1/`。

| 层 | 预注册决策 | 落地数字 | 特征来源 |
|---|---|---|---|
| A 几何 payload | **supported as confound** | matched L2 = 0；`draw_00` L2 = 12.923 | 同名 `draw_*` 不是同一 payload |
| A 站平面 / `delta_z` | identity **quiet** | identity `delta_z` KS = 0，overlap = 1；`draw_00` `delta_z`/`z` KS = 1，overlap = 0 | WB76 的 `delta_z` KS 是几何银行混淆，不是 run-range 平面变化 |
| B 重建政策 | 政策相同；gate 因内容触发 | 两边 `s0013-r0022`；identity 最大 state-like KS = 0.223（`s3_log_sigma_tx`，p = 0.17） | 事件内容 / 协方差，不是 rec-tag 变更 |
| C 传播 | **supported on L23** | 100047 identity `e23_pull_tx` / `e23_residual_tx` KS = 0.333（n = 44/42，p = 0.012）；100048 identity `e23_residual_y` KS = 0.305（p = 0.023） | 同一 Acts 合同下，不同 tracklet 进入 L23 |
| D overlay | **not supported** | 配方完全相同，seed = 20260813 | 不能解释源纯物理文件漂移 |
| E 选择 | **not supported** | max \|Δ occupancy\| = 0.082 < 0.10 | 50 vs 49 事件不是主因 |
| W64 head | 机制，不是原始物理量 | identity L01/L23 overlap 0.60–0.68；`draw_00` L23 KS = 0.74/0.84，mean −6σ | 几何混淆放大 head；matched identity 上 head 已不完全在 train support |

因此 WB76 的 pooled-overlay `delta_z` KS≈0.43 **不能** 读成 “00100 vs 00350 的站平面几何变了”。它至少混入了：

1. 六源 train overlay vs 两源 development overlay；
2. 不同随机几何银行（同名 `draw_*`，L2 = 12.9，`delta_z` KS = 1）；
3. 标准化 / route-head 外推。

在 **matched identity** 上，`delta_z` 完全不动。剩下的是 L23 residual / covariance 以及冻结 W64 logit 的 support 不全。这足以禁止把 00350 当交换测试，但 **n 只有约 40 条 truth 边、约 35 张完整图**，不另开新物理机制。

**WB74 / WB75 / WB76 / WB77 结论不被推翻。** Representation study（WB77 Arm A/B/C）继续暂停，直到 Dataset contract v2 被接受。

---

## 1. Git / 冻结边界

```text
audit commit         = 146e28b  (Workbook 78 protocol + Condor worker)
this results commit  = (commit that adds these numbers)
W64 parent SHA256    = 0c3a28704cc01151fab7ac943e41338e859b5dde1502c0b10e2f12ada04e6236
origin/4station      = tracking may be stale; local history is the scientific source of truth
```

```text
retrained = false
architecture_changed = false
loss_changed = false
bound_changed = false
new_final_blind_content_accessed = false
sealed_test_accessed = false
continue_to_v5a_frozen_head = false
continue_to_15d_relative_wls = false
historical_artifacts_overwritten = false
```

禁止打开：

```text
mc24_100047_00800_00849
mc24_100048_00800_00849
mc24_100116_*
mc24_100117_*
```

`00350_00399` 只作为 **already-opened development diagnostic**。禁止当 geometry holdout，禁止用于选模型 / 选 OP。

---

## 2. 为什么 00350 不是 00100-00149 的交换测试

交换测试要求：**同一 source family、同一 run-range 合同、同一 geometry payload 银行、同一重建/传播/overlay 合同**，只换一个被测因素。

当前 development 同时换了至少三件事：

```text
run range:     00100-00149   →  00350-00399
geometry bank: seed 20260821 →  seed 20260824
source pool:   six-source overlay  →  two-source reserved-blind overlay
```

所以 WB75/WB76 的 development FAIL 回答的是：

> 在 **另一个 run-range + 另一套随机几何 + 另一源池** 上，已经打开的 development 诊断集表现如何？

而不是：

> 同一个 data contract 下，Physical Pair-Relative / W64 是否可迁移？

后者仍然由 WB74（train-range family holdout）支持。

---

## 3. Task 1 — 源纯 R_phys 漂移（Condor）

比较（物理文件，不是 pooled overlay）：

```text
100047_00100-00149  vs  100047_00350-00399
100048_00100-00149  vs  100048_00350-00399
```

先比 **matched geometry**：

- `iteration_00_reference`（identity）
- `iteration_00_hard_s3_ry`（同一 native：`s1_dx=0.30`, `s2_dy=-0.20`, `s3_ry=5.0`）

`iteration_00_draw_00` 只作为 **几何混淆检查**，不是 run-range 检验。

特征与来源：

| 特征 | origin |
|---|---|
| `delta_z_mm`, station `z_mm` | A 几何 / 站平面 |
| tracklet `x/y/tx/ty/chi2/n_hit`, `log_sigma_*` | B 重建输出 / 事件内容 |
| residual / pull / `log1p_chi2` / covariance logdet | C 传播候选 |
| overlay UID / recipe | D overlay |
| occupancy / accepted-event rate | E 选择 |
| `L01/L12/L23` | 冻结 W64 head（机制，不是原始物理量） |

每个特征输出：mean/std shift、KS、Wasserstein、quantile、train/development overlap。**不只报 KS 最大值。**

官方表：

```text
feature_shift_tables.json
layer_B_reconstruction.json
layer_C_propagation.json
layer_E_rphys.json
```

物理 corpus 规模（identity，Condor 与 content_audit 一致）：

| source | events | tracklets | W64 完整图 | 跳过不完整 |
|---|---:|---:|---:|---:|
| `100047_00100_00149` reference | 50 | 188 | 33 | 17 |
| `100047_00350_00399` reference | 49 | 179 | 38 | 11 |
| `100048_00100_00149` reference | 50 | 189 | 38 | 12 |
| `100048_00350_00399` reference | 50 | 184 | 33 | 17 |

这是 **源纯 refit 子集**，不是 WB76 的 pooled overlay。数字必须按这个 n 解释。

### 3.1 Matched identity — `delta_z` 与站平面

两边 identity `z` 与 `delta_z` 都是常数：e01 = 1907.55 mm，e12 = e23 = 1190 mm。KS = 0，Wasserstein = 0，overlap = 1。

**run-range 本身没有移动站平面。**

### 3.2 Matched identity — 定位到 L23 / 事件内容

100047 identity，truth 邻接边（origin 已标注）：

| 特征 | origin | n | KS | p | W | Δμ / σ_train | overlap |
|---|---|---:|---:|---:|---:|---:|---:|
| `e23_pull_tx` | C | 44/42 | 0.333 | 0.012 | 0.262 | −0.18 | 0.857 |
| `e23_residual_tx` | C | 44/42 | 0.333 | 0.012 | 0.0069 | −0.08 | 0.905 |
| `e23_combined_covariance_logdet` | C | 44/42 | 0.331 | 0.012 | 0.492 | +0.19 | 0.810 |
| `e12_residual_ty` | C | 43/42 | 0.274 | 0.059 | 0.00073 | +0.12 | 0.952 |
| `e12_log1p_chi2` | C | 43/42 | 0.219 | 0.225 | 0.775 | +0.33 | 0.738 |
| `e23_delta_z_mm` | A | 44/42 | 0.000 | 1 | 0 | 0 | 1.000 |
| `s0_log_sigma_y` | B | 45/44 | 0.350 | 0.006 | 0.087 | +0.20 | 0.909 |
| `s0_chi2` | B | 45/44 | 0.285 | 0.037 | 0.511 | −0.23 | 0.864 |
| `s3_x_mm` | B | 47/46 | 0.204 | 0.247 | 9.59 | +0.15 | 0.913 |

100048 identity：

| 特征 | origin | n | KS | p | W | Δμ / σ_train | overlap |
|---|---|---:|---:|---:|---:|---:|---:|
| `e23_residual_y_mm` | C | 46/44 | 0.305 | 0.023 | 0.333 | −0.21 | 0.886 |
| `e23_residual_ty` | C | 46/44 | 0.244 | 0.110 | 0.00061 | −0.11 | 0.864 |
| `s3_tx` | B | 48/44 | 0.216 | 0.198 | 0.013 | +0.62 | 0.818 |
| `e23_delta_z_mm` | A | 46/44 | 0.000 | 1 | 0 | 0 | 1.000 |

横移 `x/y` 的 Wasserstein 可以到 ~10 mm，但那是宽分布上的不同事件抽样（KS≈0.17–0.20，p 不显著，overlap 仍 ≥0.80）。**真正对准 L23 的是 residual / pull / covariance，不是 `delta_z`。**

`hard_s3_ry`（同一 native）重复同一图案：`delta_z` KS = 0；100047 `e23_residual_tx` KS = 0.311；100048 `e23_residual_y` KS = 0.305。

### 3.3 冻结 W64 head（只读，SHA 核验通过）

| pair | payload | L01 KS (Δμ/σ, ov) | L23 KS (Δμ/σ, ov) |
|---|---|---|---|
| 100047 | identity | 0.318 (−2.44σ, 0.605) | 0.207 (−0.57σ, 0.684) |
| 100048 | identity | 0.228 (−1.10σ, 0.697) | 0.212 (−1.59σ, 0.667) |
| 100048 | hard_s3_ry | 0.198 (−1.08σ, 0.727) | 0.297 (−2.92σ, 0.636) |
| 100047 | draw_00 | 0.321 (−1.73σ, 0.650) | **0.737 (−6.24σ, 0.250)** |
| 100048 | draw_00 | 0.220 (−1.37σ, 0.727) | **0.838 (−6.49σ, 0.152)** |

matched identity 上 W64 已经不完全落在 train support（overlap 0.60–0.70）。`draw_00` 把 L23 推到灾难区。这与 WB76 的机制一致：OOD 特征 → 负向 route correction → `U = L_edge + delta − 4`。

### 3.4 Name-matched `draw_00` 是几何混淆，不是 run-range 检验

`alignment_parameter_values` L2 = 12.923（两 DSID 相同，因为两边用的是各自银行里的同名点）。最大差在旋转：`s3_rz` −3.73 → +4.37 mrad。

S1/S2/S3 `z` 与所有 `delta_z` KS = 1、overlap = 0。偏移本身很小（S1 ~30 μm，S2 ~7 μm，S3 ~21 μm），但是 **两个不同常数**，所以 KS 饱和。这就是“只报最大 KS”会误导的地方：最大 KS 来自 A，不是来自 00100 vs 00350 的物理平面。

---

## 4. Task 2 — 沿数据链归因（A–E）

```text
ROOT metadata → tracklet reconstruction → ACTS propagation → overlay → R_phys / W64
```

### A. geometry payload — **supported as confound**

Artifact：`configs` 中的 `relative_sampling.seed`，以及物理 manifest 的 `alignment_parameter_values`。

- train Family-2：`20260821`
- development：`20260824`
- `iteration_00_reference`：全零，两边相同
- `iteration_00_hard_s3_ry`：同一显式 native，两边相同
- `iteration_00_draw_00` / `draw_01` / `*_plus_common`：同名不同数值

identity 站平面 z 两边相同，因此 **raw `delta_z` 不可能来自 00100 vs 00350 的平面几何变化**。WB76 的 `e12_delta_z_mm` KS 必须先用 payload mix / 标准化解释。

Condor 证实：`iteration_00_reference` / `hard_s3_ry` 参数 L2 = 0；`draw_00` L2 = 12.923。`layer_A_geometry.json`。

### B. reconstruction policy — **policy match; content may differ**

Artifact：xAOD 文件名 + `physical_scan_config.yaml` + `layer_B_reconstruction.json`。

```text
train 100047: .../100047-00100-00149-s0013-r0022-xAOD.root
dev   100047: .../100047-00350-00399-s0013-r0022-xAOD.root
train 100048: .../100048-00100-00149-s0013-r0022-xAOD.root
dev   100048: .../100048-00350-00399-s0013-r0022-xAOD.root
```

同一 `s0013-r0022`。scan policy 共享：`nevents=50`, `q_over_p_mode=0`, `min_truth_match_fraction=0.99`, `chi2_gate=25`, `refinement_iterations=1`, `four_station_v1`。sampling seed 不同（`20260821` vs `20260824`）只影响几何银行，不影响重建 tag。

预注册 gate 把 identity state-like max KS = 0.223 标成 supported。定位后，这个最大值是 `s3_log_sigma_tx`（p = 0.17），不是 `x/y/tx/ty` 的系统平移。运动学 KS 最大约 0.20–0.22，p 不显著。**B 的政策没有变；变的是另一段 run 的事件内容。**

### C. propagation / material / field — **same tool contract; L23 residual shifts**

Artifact：两边 `refit_chain`：

```text
persisted SCT_ClusterContainer -> SegmentFitRefit -> SegmentsRefit
  -> NtupleDumper -> FaserActsExtrapolationTool(mode 0)
```

`q_over_p_mode=0`。`layer_C_propagation.json`：identity / `hard_s3_ry` 上 `delta_z` KS = 0，但 L23 residual/pull/logdet KS ≈ 0.31–0.33（100047 p = 0.012；100048 `residual_y` p = 0.023）。

没有独立场/材料变更证据。读成：**同一传播器吃进不同 tracklet，残差差集中在 L23。** 这与 WB76 的 L23 / weak-margin 故事同方向，但现在是源纯、matched geometry。

### D. overlay generation — **recipe match; not the physical-file origin**

Artifact：`layer_D_overlay.json`。

```text
seed = 20260813
events_per_payload = 120
tracks_per_event = 3
missing_tracklet_probability = 0.1
hard_negative_mean_per_target_station = 0.5
scale_events_by_pool_sources = true
recipe_fields_match = true
```

UID 计数：train 六源各约 343–350；development 两源 343 / 350。这解释 overlay 池大小，**不能**解释源纯物理 ROOT 上的特征漂移。

### E. selection / filtering — **not primary**

同一 `nevents` / truth-match / chi2 gate。Occupancy：

```text
100047: train 0.90/1.00/0.92/0.94   vs  dev 0.90/0.92/0.90/0.94
100048: train 0.92/0.94/0.96/0.96   vs  dev 0.88/0.98/0.94/0.88
max |Δ occupancy| = 0.082 < 0.10
```

决策函数（预注册，不在本 run 上调参）在 `training/run_range_data_contract.decide_shift_sources`。完整决策：`shift_source_decision.json`。

---

## 5. Task 3 — Dataset contract v2

WB77 representation study **暂停执行**。协议文件：

- `configs/research_review/dataset_manifest_v2.protocol.json`（入 git）
- `outputs/mc24_four_station_run_range_data_contract_v1/dataset_manifest_v2.json`（Condor 实例化：git SHA / 时间戳）

必须固定：

```text
source family split
run-range split
geometry payload table + seed
event UID = source_id:run_id:event_id
overlay seed（历史 overlay = 20260813；新 corpus 必须另写)
reconstruction version = s0013-r0022
propagation = FaserActsExtrapolationTool mode 0
```

域：

| 域 | 内容 | 用途 |
|---|---|---|
| `train_domain` | 六源；Family2 = `00100_00149`；geometry seed `271828` | 未来训练 |
| `validation_domain` | `00350_00399` reserved blind | 已打开诊断；禁选模 |
| `heldout_geometry` | seed `314159` 新表 | 只生成表；**未授权** physical refit |
| `heldout_run_range` | `not_authorized` | 00350 不是它 |
| `prohibited_blind_assets` | `00800_00849`, `100116_*`, `100117_*` | 永不打开 |

---

## 6. Task 4 — 新 geometry holdout 表

**不使用** 已打开的 `00350_00399`。

按 WB77：

```text
train seed     = 271828   →  reference + hard_s3_ry + 2 random families + twins
held-out seed  = 314159   →  reference + 2 new random families + twins
                              （不含 hard_s3_ry：那是已知 train 几何）
```

相对盒：0.50 mm / 5 mrad；common：0.30 mm / 3 mrad。

单位测试与 Condor `geometry_holdout/disjointness.json` 都证明两表随机家族签名不相交；identity 允许共享。

```text
train points = 7
held-out points = 5
disjoint = true
train_seed = 271828
heldout_seed = 314159
```

Condor 已写入（不覆盖历史 payload bank，不做 refit / overlay）：

```text
outputs/mc24_four_station_run_range_data_contract_v1/geometry_holdout/train_payloads.json
outputs/mc24_four_station_run_range_data_contract_v1/geometry_holdout/heldout_payloads.json
outputs/mc24_four_station_run_range_data_contract_v1/geometry_holdout/disjointness.json
```

`physical_refit_authorized = false`。不进入 Final Blind。

---

## 7. Task 5 — 后续 Arm A/B/C 协议（只写，不训练）

固定（来自 WB77，不改）：

```text
same candidate generation
same truth-free route enumeration
same raw_energy_v1
same route_accounting_v2
same exact unit-capacity solver
```

只改 scorer：

| Arm | scorer | 现在 |
|---|---|---|
| A | 冻结 W64 canonical energy | 协议；W64 见过六源，必须披露 |
| B | raw physical route → 低容量 energy | 协议；不实现网络 |
| C | 可训练物理 route representation | 协议；不实现网络 |

`configs/research_review/wp4_representation_study.yaml` 现为：

```text
paused: true
paused_by_workbook: 78
geometry_held_out.status: tables_generated_refit_not_authorized
training_authorized: false
```

完成本 workbook 后，再决定进入 Arm B 还是 A/B/C study。**当前不训练任何模型。**

---

## 8. 工程 / Condor

新增：

- `training/run_range_data_contract.py`
- `scripts/audit_wb78_run_range_data_contract.py`
- `scripts/run_wb78_run_range_audit_condor.sh`
- `scripts/submit_wb78_run_range_audit_condor.py`
- `tests/test_run_range_data_contract.py`
- `configs/research_review/wp78_run_range_data_contract.yaml`
- `configs/research_review/dataset_manifest_v2.protocol.json`

修改（协议，不改历史 artifact）：

- `configs/research_review/wp4_representation_study.yaml`（暂停 + 指向新 geometry 表）

Hermetic tests：`tests/test_run_range_data_contract.py`（13 passed）。

HTCondor：

```text
cluster     = 1109749
schedd      = bigbird24.cern.ch
host        = b9p13p0517.cern.ch
return      = 0
utc         = 2026-09-06T13:25:51Z
git_sha     = 146e28b8badef09a6f339bb91b96763f91dce9a3
w64_sha256  = 0c3a28704cc01151fab7ac943e41338e859b5dde1502c0b10e2f12ada04e6236
flavour     = tomorrow
request     = 8 CPU, 32 GB
worker      = scripts/run_wb78_run_range_audit_condor.sh
```

每个 run 已保存 git SHA、resolved config、dataset manifest、environment、output JSON。未覆盖 WB74/WB75/WB76/WB77 artifact。未改 `baselines/route_assignment.py` 默认 legacy decoder。几何表 `disjoint = true`（train seed 271828，held-out seed 314159）。`physical_refit_authorized = false`。

---

## 9. 明确停止的方向

- 重训 V5A / WB75 / W64
- architecture / loss / B sweep
- domain adaptation
- 把 `00350_00399` 叫 geometry holdout
- 打开 Final Blind / sealed test
- 15D relative WLS
- 执行 Arm A/B/C 训练

---

## 10. 下一允许步骤

`shift_source_decision.json` 已读入。WB78 的科学问题已回答。**当前仍不训练。**

若用户明确授权，且新实验遵守 Dataset contract v2，才可以在下面两者中选一：

1. Arm B：低容量 Explicit Route Energy Model（同一 candidate / `raw_energy_v1` / `route_accounting_v2` / exact solver；**不用** 00350 做 holdout）；
2. Arm A/B/C representation study（仍 `training_authorized=false` 直到单独授权；必须先对 seed `271828` / `314159` 做 physical refit，而不是复用历史银行）。

禁止把 `00350_00399` 升级成 geometry holdout 或选模集。`heldout_run_range` 仍是 `not_authorized`。
