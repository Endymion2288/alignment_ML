# Workbook 73 — 四站 Route Representation 跨 Source 域偏移定位审计

日期：2026-09-05
分支：`4station`
性质：**只读机制诊断 / architecture-selection**。本 Workbook 不训练任何 production association model。
所有 diagnostic probe 均标记 `diagnostic_only=true`、`not_checkpoint_candidate=true`。

---

## 0. Git / local / remote state

```text
local HEAD      = a77812dfa015236e33a53eb6b2a0e7726584a6e5
origin/4station = cc8d22036be5f0381fa7ad9441477dae155d5a66
```

- 本地领先远端 **8 个 commit**（Workbook 71 + Workbook 72 全部工作）。
- 最近本地 commit：

```text
a77812d Workbook 72: V5A source-transfer CV evaluation results -- gate FAIL
e571a7e Workbook 72: single-pass solver eval + CPU submission for source-transfer
1ad9939 Workbook 72: execution-status log (corpora regenerated, 4 CV jobs submitted)
```

- Workbook 72 evaluation 实际使用 `git_commit = e571a7e72d04483fc172a1929f18d3df5ec3c103`。
- **push 状态**：因 SSH passphrase 非交互限制，本轮仍未 push。**未丢弃 / 未 reset / 未覆盖任何 Workbook 71/72 commit**，working tree 完整。本地 repository / outputs / workbook 是当前科学真值。
- 本 Workbook 新增脚本（待 commit）：`scripts/extract_route_representations.py`、`scripts/analyze_route_representation_domain.py`、`scripts/make_route_representation_decision.py`、`scripts/run_route_representation_extract_condor.sh`、`scripts/submit_route_representation_extract_condor.py`。

---

## 1. Workbook 72 gate failure 独立 replay（从 JSON，非 Markdown）

来源：`outputs/mc24_four_station_relative_route_v5a_source_transfer_v1/evaluation/source_transfer_summary.json`

| held-out fold | transfer 方向 | arm | C | D | catastrophic_truth_destruction | new_C |
|---|---|---|---|---|---|---|
| holdout_family1 | **family2 → family1（难）** | W64 | 6 | 74 | — | — |
| holdout_family1 | family2 → family1 | unbounded (control) | 203 | 144 | 184 | 197 |
| holdout_family1 | family2 → family1 | **bounded (V5A primary)** | **237** | **151** | **215** | **231** |
| holdout_family2 | **family1 → family2（易）** | W64 | 0 | 43 | — | — |
| holdout_family2 | family1 → family2 | unbounded (control) | 12 | 57 | 12 | 12 |
| holdout_family2 | family1 → family2 | bounded (V5A primary) | 2 | 54 | 2 | 2 |

冻结结论独立复核一致：

```text
gate_pass = false
stop_head_only_mainline = true
```

关键事实：

1. **fold asymmetry 确认**：`family2 → family1`（holdout_family1）catastrophic = 215，而 `family1 → family2`（holdout_family2）catastrophic = 2。难方向是 family2→family1。
2. **bounded 在难方向略差于 unbounded**（215 vs 184 catastrophic；231 vs 197 new_C）。证实 Workbook 72 结论：unbounded scale 是 amplifier，不是根因；bounding 到 `[-4,+4]` 消除了 O(10–100) correction，但没有解决 cross-source truth/fake boundary shift。
3. safety 通过（`|delta_bounded|<=4`、无 NaN/Inf），truth 在 `-4` 的饱和比例 ~0.0004（不是饱和问题）。

边界状态（沿用且本 Workbook 全程保持）：

```text
training_authorized = false
final_blind_eval_authorized = false
new_final_blind_content_accessed = false
sealed_test_accessed = false
continue_to_15d_relative_wls = false
```

---

## 2. Representation levels R0–R7 / R_phys 定义（以代码真实定义为准）

提取脚本：`scripts/extract_route_representations.py`。对**同一条 route candidate**，在单次 forward 中通过最小 forward hook 抽取以下层级。宽度来自 `extract/family1_extract_summary.json` 的 `widths`。

| level | 宽度 | 内容 | 来源 |
|---|---|---|---|
| **R0** | 33 | 三条相邻 edge 的 frozen 物理 pair-relative 观测量（每 edge 11 维） | `graph.score_edge_features`（residual/pull/chi2/cov/delta-z） |
| **R1** | 33 | R0 经 Workbook-64 `edge_standardizer` 标准化后的 edge 表示 | `_make_batch` standardized edge features |
| **R2** | 384(lat)+3(logit) | frozen W64 edge latent + edge logits `L01,L12,L23` | hook `edge_score` 前馈输入（latent）+ `score_node_states`（logits） |
| **R3** | 68 | 4 个 node 的 frozen W64 输入特征（每 node 17 维） | `graph.node_features` |
| ↳ R3_abs | 20 | node 绝对/全局分量 `x_mm,y_mm,tx,ty,z_mm`（idx 0–4） | R3 子集 |
| ↳ R3_qual | 48 | node 质量/协方差分量 `log_sigma*,chi2,n_hit,hit_layer_*`（idx 5–16） | R3 子集 |
| **R4** | 512 | frozen W64 node latent `s0,s1,s2,s3`（每 node 128） | hook backbone 最后一个 block 输出 |
| **R5** | 1075 | 进入 `route_encoder` 前的 Absolute route 输入（s0–s3 + edge context + pair embeddings + base edge logits） | hook `route_encoder` 前馈输入 |
| **R6** | 128 | `route_score` 最后一层 Linear 前的 route hidden | hook `route_score` 前馈输入 |
| **R7** | 4 | 输出分数 `raw_delta, bounded_delta, L_edge, L_corrected` | hook `route_score` 输出 + bound 映射 |
| **R_phys** | 36 | **诊断用** physical pair-relative route = `[edge01,edge12,edge23 物理观测量(33) + L01,L12,L23(3)]`，**不含 absolute node latent** | R0 + R2logits |

edge 11 维物理观测量（已确认物理语义）：
`residual_x_mm, residual_y_mm, residual_tx, residual_ty, pull_x, pull_y, pull_tx, pull_ty, log1p_chi2, combined_covariance_logdet, delta_z_mm`。

**固定 extractor 约定（关键）**：R5/R6/R7 依赖 route head。为避免「不同 family 用不同 head」的混淆，最终分析对 **family1 与 family2 都使用同一个固定 head = Workbook 72 fold1 primary（holdout_family1/primary）**。这样 R5/R6 的 cross-family probe 是「固定 extractor」的干净测试。family2 最初曾用 fold2-primary 抽取（cluster 1109212），已被 fold1-primary 版本（cluster 1109213）覆盖；R0–R4/R_phys 与 head 无关，两次抽取完全一致。

---

## 3. Source / family provenance 与语料

沿用 Workbook 72 的 provenance 证据（不从编号猜测）：

```text
Family 1 = {100043, 100044}
Family 2 = {100047, 100048}
```

复用 Workbook 72 已验证的 source-pure synthetic corpora（不用六源 pooled corpus，因其在 event 内混合 source family）。

| corpus | events | routes | truth | fake |
|---|---|---|---|---|
| family1 | 3360 | 405530 | 6789 | 398741 |
| family2 | 1680 | 192122 | 3248 | 188874 |

truth/fake share-class（沿用 Workbook 71 语义，按 `truth_particle_id` 与 `node_indices` 匹配）：

| class | 含义 | family1 | family2 |
|---|---|---|---|
| 0 A_truth | complete truth route | 6789 | 3248 |
| 1 B_fake_share3 | 共享 3 个 truth 端点 | 72087 | 34603 |
| 2 C_fake_share2 | 共享 2 个 truth 端点 | 246996 | 118273 |
| 3 D_fake_share1 | 共享 1 个 truth 端点 | 79159 | 35923 |
| 4 E_unrelated | 完全无关 fake | 499 | 75 |

存储（`extract/`）：`*_lowdim.npz`（全部 route：uid/payload/run/event/n0–n3/share_class/label + R0,R1,R2logits,R3,R7,R_phys）；`*_highdim.npz`（子样本：全部 truth + 12% fake，R2latent,R4,R5,R6），由 `uid` 关联。family1 highdim 子样本 54641 条，family2 25944 条。这些即 `route_representation_rows`。

---

## 4. 提取的 Condor provenance

| cluster | family | head | 结果 |
|---|---|---|---|
| 1109211 | family1 | fold1-primary | Normal termination, return 0 |
| 1109212 | family2 | fold2-primary（后被覆盖） | Normal termination, return 0 |
| 1109213 | family2 | **fold1-primary（最终使用）** | Normal termination, return 0 |

- schedd：`bigbird24.cern.ch`（经 `lxbatch/eossubmit`，支持 /eos 路径）。
- 资源：CPU（`request_cpus=4`, `request_memory=24000`, `+JobFlavour=workday`, `OpSysAndVer==AlmaLinux9`）。本任务为 forward extraction + 统计，**未请求 GPU**。
- 提取时 `git_commit = a77812d`，working tree dirty（含本 Workbook 新脚本，属预期）。
- return 0 ≠ scientific PASS；科学结论见下。

---

## 5. 第一类分析：直接 distribution shift（按 truth/fake class 分层）

对 R0–R7 计算 SMD / median / quantile / Wasserstein；高维 latent 用 MMD。完整 pooled 分布用于本审计。

**truth route（A_truth）的 family1 vs family2 边际分布偏移很小**：

| level | A_truth SMD max | A_truth MMD |
|---|---|---|
| R0 | 0.255 | — |
| R1 | 0.255 | — |
| R3 | 0.370 | — |
| R_phys | 0.255 | — |
| R2latent | — | 0.0026 |
| R4 | — | 0.0029 |
| R5 | — | 0.0034 |

（E_unrelated 的 SMD 较大 ~0.73，但该类 route 极少且不是 failure 主体。）

**结论：truth route 的边际分布在两个 family 间高度相似（SMD<0.4，MMD~0.003）。这不是一个 gross marginal distribution shift。** 问题必然是 conditional / boundary 层面，而非「两个 family 的 truth 长得完全不一样」。

---

## 6. 第二类分析：source-family predictability probe

对每个 representation 训练 `L2-regularized logistic regression`（`C=1.0`，`StandardScaler` 拟合于 train）预测 Family1 vs Family2，分 truth-only / fake-only。AUC 越高 = source 信息越强。

| level | truth-only AUC | fake-only AUC |
|---|---|---|
| R0 | 0.695 | 0.592 |
| R1 | 0.713 | 0.588 |
| R2 | 0.958 | 0.821 |
| R3 | 0.806 | 0.782 |
| R3_abs | 0.643 | 0.604 |
| R3_qual | 0.774 | 0.749 |
| **R_phys** | 0.710 | 0.596 |
| R4 | 0.851 | 0.867 |
| R5 | 0.949 | 0.912 |
| R6 | 0.723 | 0.552 |

**source 信息从 R0 起即存在（~0.7），随深度升高（R2 0.96、R4 0.85、R5 0.95）。** 注意 R6（route hidden）source predictability 反而降到 0.72——route encoder 抹掉了一部分 source 信息。

**关键：source predictability ≠ transfer failure。** R2 source AUC 高达 0.96，但 cross-family transfer 几乎完美（见下）。因此「能否预测 source」本身不是判据；判据是「source 信息是否破坏 truth/fake boundary 的可迁移性」。

---

## 7. 第三类分析：truth/fake cross-family transfer probe（本 Workbook 核心）

对每个 representation 用**同一个固定 logistic regression**（相同 preprocessing / regularization / sampling contract）做 `Train Family1→Test Family2` 与 `Train Family2→Test Family1` 的 truth vs fake 分类。

**Sampling contract（冻结，透明）**：每个 truth route 配 `1 truth + 1 share3 fake + 1 share2 fake + 1 share1/unrelated fake`（1:1:1:1），避免被 ~1:60 的 natural imbalance 淹没。`class_weight='balanced'`，`max_iter=2000`。诊断专用，不做 hyperparameter sweep。

| level | f1→f2 AUC | f2→f1 AUC | f1→f2 truth FNR | **f2→f1 truth FNR** | in-family AUC |
|---|---|---|---|---|---|
| R0 | 0.951 | 0.941 | 0.155 | 0.119 | 0.954 |
| R1 | 0.951 | 0.942 | 0.151 | 0.114 | 0.955 |
| R2 | 0.999 | 0.999 | 0.005 | 0.008 | 1.000 |
| R3 | 0.478 | 0.485 | 0.450 | 0.526 | 0.525 |
| R3_abs | 0.495 | 0.507 | 0.306 | 0.487 | 0.515 |
| R3_qual | 0.486 | 0.507 | 0.636 | 0.626 | 0.524 |
| **R_phys** | **0.999** | **0.999** | **0.000** | **0.002** | 1.000 |
| R4 | 0.637 | 0.626 | 0.468 | **0.391** | 0.661 |
| R5 | 0.998 | 0.997 | 0.030 | 0.048 | 0.999 |
| R6 | 0.999 | 0.999 | 0.000 | **0.036** | 0.999 |

读法（最重要的列是 **f2→f1 truth FNR**，对应 Workbook 72 的难方向 holdout_family1）：

1. **R_phys 是唯一 truth FNR≈0（0.002）的 representation。** 一个简单线性 probe 在 R_phys 上跨 family 几乎不杀任何 truth。
2. **R5/R6（当前 absolute route head 管线）的 f2→f1 truth FNR = 3.6–4.8%，与 V5A 实测 catastrophic rate ≈ 215/6789 = 3.2% 高度一致。** 即 route head 在 cross-family 时把 ~3–4% 的 truth 判成 fake——这正是 catastrophic truth destruction 的来源。
3. **R4（absolute node latent）是最差的可辨识层**（f2→f1 FNR 0.39，in-family AUC 仅 0.66）：node latent 本身就是弱且不可迁移的 truth/fake 判别量。
4. R3（node 输入特征）接近随机（AUC~0.5）：node 原始特征本身几乎不含 truth/fake 信息（信息在 edge / pair 层面）。
5. R0/R1（纯物理 edge）transfer 良好（FNR 0.11–0.15），但不如 R_phys/R2——加上 frozen edge logits `L01,L12,L23` 后达到近零 FNR。

**核心对比：R_phys（0.002）vs R6（0.036）—— 仅去掉 absolute node latent、改用物理 pair-relative edge 表示，cross-family truth FNR 降低约 22 倍。**

---

## 8. Fold asymmetry：为什么 family2 → family1 特别差

对每层计算 family1-truth 与 family2-truth 的相互支撑（nearest-neighbour / Mahalanobis 距离，标准化空间）：

| level | f1t→f2 nn_med | f2t→f1 nn_med | nn_asymmetry |
|---|---|---|---|
| R0 | 2.90 | 2.60 | +0.30 |
| R2 | 13.04 | 12.45 | +0.59 |
| R4 | 10.10 | 9.45 | +0.65 |
| R5 | 19.56 | 18.91 | +0.65 |
| R6 | 4.24 | 3.99 | +0.24 |
| **R_phys** | 3.21 | 3.08 | **+0.12** |

**解释**：

- 表示层面的支撑不对称**很小**（nn_asymmetry 0.12–0.65，两族 truth 分布高度重叠）。所以 family2→family1 难**不是**因为「family1 truth 落在 family2 训练分布之外」。
- 真正原因是 **boundary 位置误差**：在 family2 上学到的 truth/fake 边界，对 family1 中那 ~3% 物理上最难的 truth route（低 `L23` / 2→3 / S3-hard）校准错误，把它们判到 fake 一侧。这 ~3% 就是 catastrophic cohort。
- R_phys 的不对称最小（0.12）且 FNR≈0，说明在物理 pair-relative 空间里两族 truth 的 boundary 是对齐的。

---

## 9. Failure-cohort comparison（fold1 catastrophic truth cohort）

建立 Workbook 72 fold1 关键 cohort：`W64 viable（L_edge>4 的 truth）但 V5A bounded 后 U_truth<=0`（proxy：`L_edge>4 且 L_corrected<=4`）。

- proxy `n_doomed = 249`，`n_surviving = 6534`（对照 Workbook 72 solver 精确值 215；proxy 略多算入部分 D route，量级一致，验证通过）。

doomed vs surviving 的各层 |SMD|（max / mean）：

| level | SMD max | SMD mean |
|---|---|---|
| R0 | 1.040 | 0.232 |
| R1 | 1.040 | 0.235 |
| R2 | 1.282 | 0.379 |
| R3 | 0.729 | 0.222 |
| R4 | 0.938 | 0.371 |
| R5 | 2.000 | 0.432 |
| R6 | 2.907 | 1.298 |
| R_phys | 1.079 | 0.286 |

**结论：会被 route head 错杀的 truth route，在最早的 R0（物理 edge 观测量）就已经可区分（SMD max 1.04）。** 即 catastrophic cohort 是「物理上本来就难」的 route（2→3 / S3 / 低 L23），head 在 cross-family 时把这部分难 route 的 boundary 划错。这与 Workbook 71 的发现（development 上 `L23<0` 的 truth 收到大负 correction）一致。

---

## 10. Physical-relative representation audit（R_phys 作为 diagnostic control）

构造 `R_phys = [edge01, edge12, edge23 的物理 pair-relative 观测量(33) + L01,L12,L23(3)]`（36 维），**完全去掉 W64 absolute node latent**，只使用仓库中已存在且物理语义已确认的 pair-relative 量。未新推导 SE(3) feature，未猜 coordinate transform。

结果（见 §7）：**R_phys 的 cross-family truth FNR = 0.000–0.002，AUC = 0.999，是所有 representation 中 transfer 最好的**，远优于含 absolute node latent 的 R5（0.048）/ R6（0.036）/ R4（0.39）。

`physical_relative_probe_summary.json` 已记录。注意：R_phys 的低 FNR 主要由 frozen edge logits `L01,L12,L23`（W64 生产分数，本身跨 family 可迁移）贡献，物理 edge 观测量提供补充。

---

## 11. Implementation audit（truth label / route key 语义）

- share-class 语义直接复用 Workbook 71 的 `_route_share_classes` / `_share_class_name`（按 `truth_particle_id` 与 `node_indices` 匹配），未重新猜测。
- 提取的 route candidate `(n0,n1,n2,n3)` 与 truth label、share_class 在两个 corpus 上自洽（A_truth 数 = complete_truth_chains：family1 6789 = Workbook 72 的 6789，family2 3248）。
- 抽取的 `L_edge` / `L_corrected` 与 Workbook 72 evaluation 的 catastrophic 计数（215）通过 proxy（249）交叉验证一致。
- **未发现 route label / key semantic bug。** `implementation_semantic_bug = false`。

---

## 12. Primary localization（唯一主定位）

> **source-family 信息从 R0 起即进入表示（source-probe AUC~0.7），并随深度升高（R2 0.96、R4 0.85、R5 0.95）。但 source 信息本身不破坏迁移——R2 高 source 泄露却近乎完美迁移。真正破坏 truth/fake boundary 跨 family 可迁移性的，是 absolute node latent（R4）这一分量：它自身是弱且不可迁移的判别量（in-family AUC 0.66、cross-family truth FNR 0.39–0.47），当被 route head 消费（R5→R6）后，给 cross-family truth 边界引入 ~3.6–4.8% 的 false-negative 率，正好等于 V5A 实测的 catastrophic truth destruction（~3.2%）。去掉 absolute node latent、只用物理 pair-relative edge 表示 + frozen edge logits（R_phys），cross-family truth FNR 降到 ~0.2%。**

一句话：**问题不在「source 信息存在」，而在「route head 消费了 absolute node latent 这个 source-dependent 且弱判别的分量」。**

---

## 13. 按预注册 decision tree 选出的下一 architecture hypothesis

对照 Workbook 73 预注册的 decision rule：

- **Case 1 条件成立**：`R_phys / R0–R2` cross-family truth/fake transfer 良好（R_phys/R2 FNR≤0.008，R0/R1 FNR~0.12–0.15）；而 `R4` truth FN 明显恶化（0.39–0.47），且 `R5/R6` 的 truth FN（0.036–0.048）正好处于 catastrophic 量级。
- 非 Case 2：R0/R_phys 本身并非强 source-dependent 到不可迁移（R_phys FNR≈0）。
- 非 Case 3：R5/R6 并非「route encoder 后突然崩」（AUC 仍 0.999，是 boundary 位置误差而非信息崩塌）。
- 非 Case 4：并非「所有 probe 都好但 V4/V5A 失败」——R4 明显差，且 R5/R6 的 truth FNR 与 catastrophic 率吻合，说明 representation 选择（是否含 node latent）确实影响迁移。

**`decision_case = case_1`**

**下一主假设（冻结）：Physical Pair-Relative Route Encoder**

> route representation 不再消费 absolute node latent（s0..s3）；仅消费三个 adjacent 物理 pair-relative edge 表示（现有 residual/pull/chi2/cov/delta-z 观测量）+ frozen W64 edge logits `L01,L12,L23`。即 R_phys 表示。

这是最小、物理可解释的修正。**本 Workbook 不实现该网络、不训练、不 sweep。** 也未实现 domain-adversarial / gradient-reversal / SE(3)-equivariant Transformer——因为已存在一个已有的物理相对表示（R_phys）具有明显更好的 transfer，优先采用最小方案。

---

## 14. Outputs

`outputs/mc24_four_station_route_representation_domain_audit_v1/`：

```text
audit_contract.json
representation_manifest.json
extract/family1_lowdim.npz, family1_highdim.npz      (= route_representation_rows)
extract/family2_lowdim.npz, family2_highdim.npz
extract/family1_extract_summary.json, family2_extract_summary.json
extract/family2_extract_summary.fold2head.json       (被覆盖的 fold2-head 版本备份)
distribution_shift_summary.json
source_probe_summary.json
cross_family_truth_fake_probe.json
fold_asymmetry_summary.json
catastrophic_cohort_summary.json
physical_relative_probe_summary.json
decision.json
condor/  (submit files + logs)
```

---

## 15. 边界状态（全程保持）

```text
training_authorized = false
final_blind_eval_authorized = false
new_final_blind_content_accessed = false
sealed_test_accessed = false
continue_to_15d_relative_wls = false
```

- 未训练任何 production model（仅 diagnostic logistic probe，`diagnostic_only=true`、`not_checkpoint_candidate=true`）。
- 未做 B sweep / loss reweight / bounded relative V5B / full-six-source V5A / development model selection。
- 未打开 Final Blind（`00800_00849`）、未打开 sealed test、未开启 15D WLS。
- 历史 negative results 全部保留。

---

## 16. Done-when 核对

1. ✅ 本地 Workbook 71/72 commit chain 完整（local a77812d，8 unpushed，未丢弃）。
2. ✅ Workbook 72 gate failure 已从 JSON 独立 replay（§1）。
3. ✅ R0–R7/R_phys 已定义并提取（§2/§3/§4）。
4. ✅ truth/fake conditional source shift 已量化（§5）。
5. ✅ source-family predictability 已量化（§6）。
6. ✅ Family1→Family2 与 Family2→Family1 transfer 已逐 representation 比较（§7）。
7. ✅ fold asymmetry 已解释（§8）。
8. ✅ fold1 catastrophic truth cohort 已定位到 representation 层（§9，最早 R0 可区分）。
9. ✅ physical pair-relative route representation 已作为 diagnostic control 检验（§10）。
10. ✅ 给出唯一主 localization（§12：absolute node latent R4）。
11. ✅ 按 decision tree 选出唯一下一 hypothesis（§13：Case 1 Physical Pair-Relative Route Encoder）。
12. ✅ 未训练任何 production model。
13. ✅ Final Blind 未打开。
14. ✅ sealed test 未打开。
15. ✅ 15D WLS 未开启。
16. ✅ Workbook 73 完成。

**下一步（需用户授权后另开 Workbook 预注册）：实现并训练 Physical Pair-Relative Route Encoder（R_phys-based route head），仍冻结 W64 backbone，先在 train-side source-transfer CV 上验证 cross-family truth FNR 是否如本审计预测降到 ~0。本 Workbook 不启动该训练。**
