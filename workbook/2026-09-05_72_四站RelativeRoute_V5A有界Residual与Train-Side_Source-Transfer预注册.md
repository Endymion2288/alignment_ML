# Workbook 72：四站 RelativeRoute V5A 有界 Residual 与 Train-Side Source-Transfer 预注册

日期：2026-09-05
状态：**预注册（preregistration）。本条目在任何 V5A 训练之前写就并冻结。`training_authorized=false` 直至 Phase A audit 全部通过；`final_blind_eval_authorized=false`；`continue_to_15d_relative_wls=false`。**

本条目是 Workbook 71 的直接后续。它不重新解释 Workbook 70/71 的冻结结论，只在其之上预注册一条**单一假设**的可证伪实验。

---

## 0. 最新 git state 与 Workbook 71 落地确认

审计时刻真实本地状态：

```text
branch                 = 4station
HEAD                   = 2b2fb89c91705e29276c52295e238ff6f215e6d6
working tree           = clean（git status 无 dirty）
origin/4station        = cc8d22036be5f0381fa7ad9441477dae155d5a66
local ahead of origin  = 3 commits
```

本地领先 origin 的 3 个 commit（即 Workbook 71 工作，**完整保留、未丢弃、未 reset、未覆盖**）：

```text
8043888  Add Workbook 71 RelativeRoute V4 generalization failure audit
714671f  Add Workbook 71 analysis pass (route_rows -> audit verdicts)
2b2fb89  Workbook 71: RelativeRoute V4 train->development generalization failure audit
```

**push 状态**：本环境下唯一的 GitHub 凭证是带 passphrase 的 SSH key（`~/.ssh/id_rsa`），无 ssh-agent / gh / token，非交互无法完成 `git push`。用户已知情并选择 **defer**（本地 commit 安全，稍后由用户自行 push 或解锁 key）。本地工作树干净，3 个 commit 已落盘，无丢失风险。

Workbook 71 实际输出已独立复核（非仅信 Markdown）：

```text
outputs/mc24_four_station_relative_route_v4_generalization_audit_v1/
  decision.json:  case = B_generalization_failure
                  primary = [C_unbounded_correction_scale_collapse,
                             D_train_to_development_domain_shift,
                             E_relative_representation_transfer_failure]
                  train_C_arm0_arm1_arm2 = [0, 0, 0]
                  development_C_arm0_arm1_arm2 = [200, 327, 441]
                  implementation_semantic_bug = false
```

---

## 1. 冻结科学事实（禁止重新解释，除非发现可复现 implementation error）

### Workbook 70（development，`00350_00399`）

```text
Arm0 C/D = 200 / 170
Arm1 C/D = 327 / 123
Arm2 C/D = 441 / 135
真实 later-minus-earlier：Arm1 - Arm0 ΔC = +127；Arm2 - Arm1 ΔC = +114
Condor = 1108310（Normal termination return 0）
```

历史 `1108310` JSON 的 `arm1_minus_arm0_C` 符号键名错误已记录为 erratum；**禁止覆盖**该历史 artifact，科学计数以绝对 C/D 为准。

### Workbook 69（Head-Only 配对训练）

```text
Arm 1 (Absolute Control)  Condor = 1104860
Arm 2 (Relative Primary)  Condor = 1104861
两者 Normal termination return 0，30 unique epochs，last-completed-epoch checkpoint，
early_stopping=false，frozen W64 不变，edge logits 在冻结数值合同内不变。
frozen W64 parent = outputs/mc24_four_station_source_diversity_v1/checkpoint/route_aware_transformer_v2.pt
frozen W64 SHA256 = 0c3a28704cc01151fab7ac943e41338e859b5dde1502c0b10e2f12ada04e6236
trainable head params = 172,513；frozen backbone params = 614,947
seed = 20260822；AdamW lr=2e-4、weight_decay=1e-4、batch_size=32
```

### Workbook 71（Train→Development 泛化失败机制审计）

```text
failure_case = Case B generalization failure
TRAIN  Arm0/1/2 C = 0/0/0，D = 38/33/25   （objective 在 train 分布内改善，非 objective 失败）
DEV    Arm0/1/2 C = 200/327/441
truth delta_route_logit：
  TRAIN Arm1 mean +0.075；TRAIN Arm2 mean +0.160
  TRAIN fake Arm1 ≈ -130；TRAIN fake Arm2 ≈ -141
  DEV truth median ≈ 0，但极重负尾：Arm1 q01≈-33.85 / min≈-53；Arm2 q01≈-36.15 / min≈-54
Mechanism C 新增来源（exact decomposition）：
  Arm0->Arm1：selected->C = 72，D->C = 55
  Arm1->Arm2：selected->C = 78，D->C = 45（gross new C = 123，net ΔC = +114）
  development truth 且 L23<0 的 50 条：Arm1 delta mean≈-19.47，Arm2 delta mean≈-21.49
已排除：objective gradient direction bug / route-class imbalance 为主因 /
        route label/key semantic bug / gauge common-mode 为主因 / head 完全无辨识信息
当前主因排序：
  1. C_unbounded_correction_scale_collapse
  2. D_train_to_development_domain_shift
  3. E_relative_representation_transfer_failure（secondary）
```

---

## 2. 当前物理与数据边界（全程保持）

```text
continue_to_15d_relative_wls        = false
sealed_test_accessed                = false
new_final_blind_content_accessed    = false
training_authorized                 = false   （直至 Phase A 全过）
final_blind_eval_authorized         = false
```

已见过、现属 development（仅可用于历史 artifact replay，**本条目 model selection 禁止使用**）：

```text
mc24_100047_00350_00399
mc24_100048_00350_00399
```

Final Blind（**严格禁止打开**：ROOT open / metadata 内容检查 / event count / histogram / inference / branch scan 全部禁止）：

```text
mc24_100047_00800_00849
mc24_100048_00800_00849
```

Sealed test（`mc24_100116_*` / `mc24_100117_*`）继续永久关闭。

---

## 3. 本 Workbook 的唯一主假设

> Workbook 71 的 catastrophic development failure 需要**两个条件同时成立**：
>
> 1. route truth/fake decision boundary 跨 source/payload 迁移时发生**少量错位**；且
> 2. **unbounded correction** 把这些少量 classification mistake 放大成 O(10–100) 的 catastrophic negative route utility（`U_truth <= 0`）。
>
> 如果把 correction 严格限制在与 production solver utility 相同的 O(1) 尺度，则 source-transfer 错误**不应再**把原本正确的 truth route catastrophic 地压入 `U_truth <= 0`。

本条目**只测试这一条假设**。不同时更换 representation。

这是一条**可证伪**假设：若有界 head 在 held-out source 上仍系统性产生 `truth route -> fake-like correction` 或新增 C，则假设被否证（见 §14 失败分支）。

---

## 4. V5A 唯一 architecture / parameterization change

保持 Workbook-64 frozen backbone 与 edge scorer 不变。保持现有 **Absolute**-route head representation 不变。

原 V4（unbounded）：

```text
delta       = raw_delta
L_corrected = L_edge_W64 + delta
```

V5A（bounded）：

```text
B             = 4.0
delta_bounded = B * tanh(raw_delta / B)
L_corrected   = L_edge_W64 + delta_bounded
```

必须满足并在 Phase A 用单元测试验证：

```text
delta_bounded ∈ [-4, +4]                       （对任意有限 raw_delta）
delta(0)                       = 0             （zero-init 保持）
d delta_bounded / d raw_delta | raw_delta=0 = 1 （原点斜率保持 1）
```

数学注记：`d/dx [B·tanh(x/B)] = sech²(x/B)`，`sech²(0)=1`；`tanh(0)=0`。两条性质由 `tanh` 本身保证，非额外调节。

### B=4 的 solver-semantic 理由（非 development tuning）

production solver 的 complete-route utility（`baselines/route_assignment.py`，已核对代码）：

```text
U_complete = log_odds + len(route_stations) * unmatched_penalty
           = L_corrected + 4 * (-1)        （complete 4-station route，unmatched_penalty = -1）
           = L_corrected - 4
```

dustbin boundary（`U_complete = 0`，route 被丢弃的临界）在 **`L_corrected = 4`**。因此 **±4 正是 production utility 的 O(1) 物理决策尺度**：

- 一条 route 被选中（utility > 0）当且仅当 `L_corrected > 4`；
- `delta_bounded = ±4` 最多把 `L_corrected` 移动一个 dustbin-boundary 尺度——足以把 `L_corrected=0`（deep dustbin，`U=-4`）的 route 抬到边界，或把边界 route 压回 deep dustbin；
- 但**不可能**再出现 Workbook 71 中 `delta ≈ -130` 把 truth route 打到 `U ≈ -134` 的 catastrophic 放大。

**本实验只有一个预注册 B=4。禁止 sweep B；禁止 B=1/2/8；禁止 learnable B；禁止 per-route B。**

---

## 5. 第一阶段只用 Absolute representation

本 Workbook **禁止训练新的 Relative Arm（V5B）**。理由：Workbook 71 已发现 Relative Arm 为 secondary failure，且 Arm2 在 development 一致劣于 Arm1。若同时改变 `unbounded->bounded` 与 `absolute->relative`，将无法归因改善来源。

只允许两个 arm：

```text
Control ：V4A Absolute Unbounded   （route_correction_bound = null）
Primary ：V5A Absolute Bounded B=4 （route_correction_bound = 4.0）
```

除 bounded mapping 外，以下全部保持一致（以 Workbook 69 代码/合同为准）：

```text
model body / feature definitions / candidate graph / frozen W64 backbone /
edge scores / loss / loss weights / optimizer(AdamW) / LR(2e-4) /
weight_decay(1e-4) / epoch budget(30=8+22) / seed(20260822) / batch semantics(32) /
solver / OP / Platt(identity) / unmatched_penalty(-1) / tie-break / physical curriculum
```

---

## 6. 真正的 TRAIN-side source-transfer test（禁止随机 event split）

### 6.1 source family 的 provenance 证据（非编号猜测）

六个 authorized train sources 的 xAOD provenance（`configs/physical_curriculum_four_station_diversity_train_sources.yaml` + `physical_corpus_manifest.json`）：

| source | DSID | charge | run range | xAOD generator string |
| --- | --- | --- | --- | --- |
| `mc24_100043_00200_00299` | 100043 | μ− | 00200–00299 | `PG_mumi_fasernu_5mrad_flukaE` |
| `mc24_100043_00300_00399` | 100043 | μ− | 00300–00399 | 同上 |
| `mc24_100044_00200_00299` | 100044 | μ+ | 00200–00299 | `PG_mupl_fasernu_5mrad_flukaE` |
| `mc24_100044_00300_00399` | 100044 | μ+ | 00300–00399 | 同上 |
| `mc24_100047_00100_00149` | 100047 | μ− | 00100–00149 | `PG_mumi_fasernu_5mrad_flukaE` |
| `mc24_100048_00100_00149` | 100048 | μ+ | 00100–00149 | `PG_mupl_fasernu_5mrad_flukaE` |

证据链：

1. **run-range 是事件分区，不是物理差异**：同一 DSID 的两个 run range（如 `100043_00200_00299` 与 `100043_00300_00399`）generator string 完全相同、DSID 相同，仅事件区间不同。Workbook 63 明确写「`100047_00100`/`100048_00100` 提供与失败族**相同的 DSID、不同事件区间**」。因此同一 DSID 的不同 run-range 分区**共享物理**，必须归入同一 group。
2. **DSID pair 是独立物理生产**：Workbook 63 实测「**100047/100048 族**相对当前两条 train source（100043/100044）的**稳定 source characteristic**」（hard rate +0.565、winner rate +0.159、2→3 logit q05 −0.487 vs +1.978、S3 x 更宽）。manifest 注释明确「a **second DSID pair**」。因此 `{100043,100044}` 与 `{100047,100048}` 是两个**电荷对称的 DSID pair（物理 family）**。
3. **development 即 family 2**：development `100047_00350/100048_00350` 与 train family-2 源 `100047_00100/100048_00100` **同 DSID pair**（不同事件区间 + 不同 overlay geometry draw）。

**结论：六源 = 4 个 DSID = 2 个电荷对称 DSID-pair family。**

```text
Family 1（DSID pair 1，"current train"）：
    {100043_00200_00299, 100043_00300_00399, 100044_00200_00299, 100044_00300_00399}
Family 2（DSID pair 2，"失败族/second DSID pair"）：
    {100047_00100_00149, 100048_00100_00149}
```

### 6.2 为何是 2-fold 而非 6-fold / 3-fold / 4-fold

- **否 6-fold leave-one-source-out**：同一 DSID 的两个 run-range 分区共享物理（§6.1 证据 1）。holdout `100043_00300_00399` 而 train 含 `100043_00200_00299`，held-out 物理已在训练中 → 不是 source-transfer test（分布内泄漏）。
- **否 3-fold**：若按 run-range 把六源分成 3 个电荷对 `{00200-00299, 00300-00399, 00100-00149}`，其中前两对共享 DSID 100043/100044，**不是 3 个独立 pair**。provenance 不支持 3-fold。
- **否 4-fold leave-one-DSID-out**：holdout `100043`(μ−) 时 train 含 `100044`(μ+)，二者是同一物理过程的电荷对称对 → 只是 charge-level transfer，不是 process-level transfer，比 Workbook 71 的 pair-level 失败弱。
- **取 2-fold leave-one-family-out**：唯一让 held-out family 成为**真正未见物理生产**的方案。Fold 1（train family1 → test family2）**直接镜像** Workbook 71 的 train→development 失败方向（development 即 family-2 DSID）。

### 6.3 CV 方案（冻结）

```text
2-fold leave-one-DSID-pair-out：
  Fold 1：Head-train = Family 1（4 源）；Head-validation = Family 2（2 源）
  Fold 2：Head-train = Family 2（2 源）；Head-validation = Family 1（4 源）
```

每个 held-out family 内部再按 source 细分报告（Fold 1 holdout 分 `100047`/`100048`；Fold 2 holdout 分 4 个源），以满足「aggregate / worst-fold」与 per-source 粒度。

### 6.4 关键实现约束：pooled synthetic corpus 混合源 → 必须按 fold 重生成

现有 6 源 synthetic corpus 采用 `source_pooling = within_split_same_payload`：每个 synthetic multi-track event 由 `rng.choice(source_tracks, tracks_per_event)` 从**全部 6 源**的 pooled truth tracks 中抽取（`datasets/synthetic_overlay.py: write_synthetic_multitrack_root`）。因此**单个 event 混合多个 family 的 track**，无法在 event 级别做干净 source holdout。

故本 CV **按 fold 重生成 synthetic corpus**，限制 source pool：

```text
Family-1 corpus：仅以 Family 1 四源的 physical tracklets 为 pool 重生成（7 payload × 120 × 4 = 3360 events）
Family-2 corpus：仅以 Family 2 两源的 physical tracklets 为 pool 重生成（7 payload × 120 × 2 = 1680 events）
```

两个 corpus 复用于两个 fold（Family-1 corpus = Fold1 train = Fold2 holdout；Family-2 corpus = Fold1 holdout = Fold2 train）。重生成用 `scripts/materialize_pooled_curriculum_synthetics.py`，输入为 fold 受限的 physical corpus manifest；synthetic recipe（events_per_payload=120、tracks_per_event、fake rates、hard-negative chi2 band、seed=20260813、overlay_seed_scope、mode-0 propagation）与 Workbook 64 完全一致，**仅 source pool 不同**。42 个 per-source physical asset（6 源 × 7 payload 的 tracklets+propagations）已全部确认存在。

---

## 7. 重要声明：这是 head-level source-disjoint transfer audit

Workbook-64 backbone 本身已在全部六个 training sources 上训练过。因此本实验**只能称为**：

```text
head-level source-disjoint transfer audit
```

**不能称为** `fully source-unseen model validation`。本实验测试的是：

> bounded route correction head 是否能比 unbounded head 更稳定地跨 source 泛化。

backbone 见过所有源这一限制明确记录于此。

---

## 8. 每个 fold 都重新配对训练 Control 和 Primary

**禁止**拿 Workbook 69 的全六源 Arm1 checkpoint 当 source-holdout control（它已看过全部六源）。

每个 fold 分别从**相同 frozen W64 parent、相同 seed(20260822)、相同 zero-init** 训练：

```text
V4A-unbounded-control
V5A-bounded-primary
```

保持 paired contract（相同 batching / 相同数据顺序 / 相同 curriculum stage）。固定：

```text
30 epochs（8 nominal + 22 local_relative_and_gauge_control）
last epoch checkpoint
no early stopping
one fixed seed = 20260822
```

（以 Workbook 69 实际代码/合同为准，未猜测。）

---

## 9. 训练前 Phase A audit（`training_authorized=false` 直至全过）

训练前至少新增并通过以下单元测试：

1. `delta_bounded` 永远在 `[-4,+4]`（对任意有限 raw_delta，含极端值）。
2. zero-init 精确给出 `delta = 0`。
3. zero-init 时 `L_corrected == L_edge_W64`。
4. 原点导数 `d delta_bounded / d raw_delta |_{0} = 1`。
5. frozen W64 参数无梯度（`requires_grad=False` 且 backward 后 grad 为 None/0）。
6. frozen edge scores 不变（wrapper edge == W64 production edge，数值合同内）。
7. production solver 消费的是 bounded `L_corrected`（`route_logits = L_edge_W64 + delta_bounded`）。
8. save/load 保持 bound contract（artifact 重载后 bound 仍生效、delta 仍有界）。
9. source holdout guard：held-out family 不得进入训练（fold 训练源 ∩ held-out = ∅）。
10. development sources（`00350_00399`）禁止进入训练。
11. Final Blind（`00800_00849`）禁止。
12. sealed test 禁止。

并执行现有相关 regression tests。任一失败 → `training_authorized=false`，停止。

---

## 10. Primary validation metrics（不看 BCE/AUC，看 solver-level 物理结果）

对每个 held-out source/group（及 aggregate / worst-fold）报告：

```text
C / D / selected / efficiency / purity / fake
```

以及 transition matrix：

```text
selected -> C
D -> C
C -> selected
D -> selected
```

重点定义：

```text
catastrophic_truth_destruction =
    被 W64 / paired control 选中（selected）、
    但经 route correction 后 U_truth <= 0 的 route 数
```

C/D 语义沿用 `training/route_operating_audit.py: audit_event_truth_chains`（与 Workbook 65/70/71 同一语义）。

---

## 11. V5A source-transfer gate（Primary 必须同时满足）

### Safety（所有 held-out folds）

```text
|delta_bounded| <= 4；无 NaN；无 Inf；frozen W64 invariant 通过
```

### Mechanism C（最重要 guardrail）

原 full TRAIN 中 W64 C=0，因此：

```text
Primary 不得系统性引入新 C
```

每 fold 报告 `new_C_from_selected` 与 `new_C_from_D`，及 aggregate / worst-fold。目标：

```text
bounded << unbounded
```

尤其不得再出现 unbounded head 那种 O(100) catastrophic truth destruction。

### Mechanism D

V5A 不得通过简单放弃 fake suppression 来换取 C：

```text
D_primary <= D_control
```

或至少 aggregate selected / purity / fake 证明无明显 regression。若 C 改善而 fake/purity 大幅恶化 → 判失败。

### Correction distribution

分别对 truth / fake 报告：

```text
mean / median / q01 / q05 / q95 / q99
boundary saturation fraction：delta ≈ -4 与 delta ≈ +4 的比例
```

若大量 truth route 饱和在 `-4`：即使 C 暂未破，也必须标记 transfer risk。

---

## 12. 不得使用 development 选择模型

整个 source-transfer CV 阶段，禁止载入 `00350_00399` development 用于：

```text
checkpoint selection / B selection / architecture selection /
early stopping / hyperparameter selection / fold selection
```

它只能在训练前做历史 artifact replay，不进入本轮选择。

---

## 13. 若 source-transfer CV 失败

若 bounded absolute head 在 held-out train sources 上仍出现 `truth route -> fake-like correction` 或 `new C`：

```text
stop head-only mainline
```

不得继续：`bounded relative` / `learnable bound` / `loss reweighting` / `B sweep`。

下一步转向 `representation/domain-transfer architecture`，另开 Workbook。仍不打开 Final Blind。

---

## 14. 若 V5A source-transfer CV 通过

才允许进入下一阶段：

**Step 1**：用全部六个 authorized TRAIN sources 训练一个 full `V5A Absolute Bounded` checkpoint。仍固定 `B=4`、30 epochs、last epoch、single seed、无 development checkpoint selection。冻结 SHA。

**Step 2**：对已经见过的 `00350_00399` DEVELOPMENT 做一次**冻结 evaluation**（它是 development，不是 blind，可用）。但不根据结果改 B / 不调 OP / 不换 checkpoint / 不 early stop / 不重训同一 V5A。若失败：记录负结果，停止。若通过：冻结 candidate model。

---

## 15. Final Blind 仍不授权

即使 V5A train-side source transfer PASS 且 development PASS，也**不自行打开** `00800_00849`。完成 Workbook 后停止并报告用户，由用户决定是否授权 Final Blind。

---

## 16. Relative / gauge-aware 暂不做

本阶段禁止：RelativeRoute V5B、gauge-equivariant Transformer、SE(3)-equivariant network、new coordinate frame、learned alignment transform、end-to-end backbone finetuning。

原因：必须先回答「unbounded scale 是否是 catastrophic failure 的必要 amplifier」。只有 bounded Absolute control 做清楚后，才值得讨论 representation。

---

## 17. Execution policy

短测试（git/code audit、pytest、数 event smoke test、小 batch autograd）可交互 GPU。长训练与全量 fold evaluation 统一 HTCondor。Condor 必须记录：cluster id / schedd / hostname / GPU / git commit / git dirty / config SHA / checkpoint SHA / stdout / stderr / return code / artifact audit。**Condor return 0 ≠ scientific PASS。**

ML 环境：`source scripts/setup_environment.sh ml`（LCG_110_cuda, x86_64-centos*-gcc11-opt）。禁止静默 CPU fallback。

---

## 18. 输出与边界台账

计划输出目录：

```text
outputs/mc24_four_station_relative_route_v5a_source_transfer_v1/
  corpora/family1/...  corpora/family2/...        （按 fold 重生成的 synthetic corpus）
  fold1/{control,primary}/...  fold2/{control,primary}/...
  phaseA_audit.json
  per_fold_summary.json
  source_transfer_gate.json
  decision.json
```

全程保持：

```text
final_blind_content_accessed = false
sealed_test_accessed         = false
continue_to_15d_relative_wls = false
```

直至用户另行授权。

---

## 19. 本 Workbook 的 Done 判据

1. source grouping 的 provenance 证据已记录（§6）；
2. V5A bounded head 实现且 Phase A 12 项测试全过；
3. 现有 regression tests 通过；
4. Family-1 / Family-2 两个 fold corpus 重生成并通过 source-holdout guard；
5. 2 fold × (Control + Primary) 配对训练完成（Condor，配对合同）；
6. 每 fold solver-level C/D/selected/eff/purity/fake + transition + catastrophic_truth_destruction 已计算；
7. source-transfer gate 给出明确 PASS/FAIL；
8. 若 PASS：full-six-source V5A + 冻结 development eval；若 FAIL：stop head-only mainline；
9. Final Blind / sealed test 全程未打开；
10. 最终报告 §21 各项。

---

## 20. 执行状态台账（2026-09-04，训练已提交、评估待跑）

> 本节为执行日志，追加于预注册（§1–§19）之后，不改变任何预注册合同。

### 20.1 Git 状态

```text
branch        = 4station
HEAD          = 5a61d14 (Workbook 72 eval + Condor eval wrapper)
历史链         = cc8d220(remote) -> 8043888 -> 714671f -> 2b2fb89(WB71 doc) -> f90b2bd(WB72 scaffolding) -> 5a61d14
git_dirty     = false（提交时）
push          = 暂缓（用户已知晓；SSH publickey 在非交互环境不可用，本地 commits 安全）
```

### 20.2 Phase A 审计（训练前，全过）

- 新增 `tests/test_relative_route_v5a_bounded.py`：13 项单测全 PASS（delta_bounded ∈ [-4,4]、zero-init delta=0、zero-init L_corrected==L_edge_W64、零点导数=1、frozen W64 无梯度、frozen edge 不变、生产 solver 消费 bounded L_corrected、save/load 保持 bound、source-holdout guard、development/final-blind/sealed 禁止）。
- 现有 regression：`test_relative_route_v4.py` + `test_route_aware_transformer.py` + `test_gauge_consistent_route.py` + `test_route_operating_audit.py` 共 48 项 PASS（训练脚本改动后再跑 `test_relative_route_v5a_bounded.py`+`test_relative_route_v4.py` 37 项 PASS）。
- 结论：`training_authorized = true`（仅限本 CV）。

### 20.3 按 fold 重生成 source-pure 语料（已完成并验证）

现有 6 源 synthetic corpus 每个 event 由 `tracks_per_event=3` 从全部 6 源 pooled truth tracks 抽取（`source_pooling=within_split_same_payload`），单 event 混合多 family → 无法 event 级 holdout。故按 §6.4 重生成：

```text
corpora/family1/overlay_synthetic_v1   4 源(100043×2,100044×2)  480 ev/payload × 7 = 3360 events
corpora/family2/overlay_synthetic_v1   2 源(100047,100048)      240 ev/payload × 7 = 1680 events
```

- 生成 recipe 与 Workbook 64 完全一致（events_per_payload=120、tracks_per_event=3、missing=0.1、hard_negative 0.5/station chi2∈[1,1000]、random_easy 0.25/station、seed=20260813、overlay_seed_scope、synthetic_run_id_base=996000、mode-0 propagation），**仅 source pool 受限**。
- tracklet 级纯度已验证（`synthetic_tracklets.root` 的 namespaced `origin_run_id`）：family1 truth 仅落 namespace base {0,1,2,3}，family2 仅 {0,1}，均 PURE。
- 两个 corpus 复用于两个 fold（family1 corpus = Fold1 holdout = Fold2 train；family2 corpus = Fold1 train = Fold2 holdout）。

### 20.4 配对训练实现与提交

- `scripts/train_relative_route_v4_head_only.py` 新增可选 `--expected-train-sources`（缺省 = 历史六源行为，Workbook 69 复现性不变）；fold 训练时校验 constituents == fold 训练源（即 source-holdout guard，held-out family 缺席）。
- 训练脚本端到端 smoke test 通过（family2 corpus：2 源、1680 graphs、zero-init max|delta|=0、frozen W64 edge identity ~1e-7、L_corrected identity=0、30-epoch loop 正常进入）。
- 4 个配对训练 job 已提交 Condor（schedd=bigbird24，GPU，job_flavour=tomorrow）：

```text
1108927  holdout_family1 control    (train family2, unbounded)
1108928  holdout_family1 primary    (train family2, bounded B=4)
1108929  holdout_family2 control    (train family1, unbounded)
1108930  holdout_family2 primary    (train family1, bounded B=4)
```

每个 fold 内 control/primary 除 `route_correction_bound`（null vs 4.0）外全部相同（同一 frozen W64 parent、seed=20260822、zero-init、30 epochs、last-epoch、no early stopping）。

### 20.5 评估实现与 smoke test

- `scripts/evaluate_relative_route_v5a_source_transfer.py` 复用 Workbook 70/71 的 solver / C-D / transition 机制（`_evaluate_arm`、`_produce_truth_rows`、`_predict_route_details`、`cd_counts`、`frozen_packing_config`），按 held-out family corpus 计算 C/D/selected/efficiency/purity/fake、selected→C 与 D→C transition、catastrophic_truth_destruction、bounded-delta 分布与 ±B 饱和份额，并给出预注册 gate。
- 端到端 smoke test（以 Workbook 69 Arm1 unbounded checkpoint 作占位、2 events/payload）通过：输出结构完整；safety 检查正确识别出占位 unbounded head 的 fake delta 超出 [-4,4]（`safety_bounds_ok=false`），证明 bound 检查有效。
- 评估 Condor wrapper/submit 已就绪（`run_/submit_relative_route_v5a_source_transfer_eval_condor.*`），待 4 个训练 checkpoint 到位后提交。

### 20.6 当前边界状态

```text
training_authorized              = true（仅限本 CV 的 4 个配对 job）
final_blind_eval_authorized      = false
development_used_for_selection   = false（00350_00399 未进入任何选择）
new_final_blind_content_accessed = false
sealed_test_accessed             = false
continue_to_15d_relative_wls     = false
```

### 20.7 待办

1. 等 4 个训练 job 完成（Condor 队列当前拥挤，GPU job 排队中）；
2. 跑 source-transfer 评估（Condor），得每 fold gate；
3. gate PASS → full-six-source V5A + 冻结 development eval；FAIL → stop head-only mainline；
4. 最终报告 §21。
