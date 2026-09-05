# Workbook 74 — 四站 Physical Pair-Relative Route Encoder V1: train-side 2-fold source-transfer CV

日期： 2026-09-05/06
分支： `4station`
任务： 正式实现并验证 Workbook 73 的 next hypothesis —— **Physical Pair-Relative Route Encoder**。

---

## 0. 科学问题（唯一）

> 去掉 route head 对 absolute/global node latent `s0..s3` 的依赖，只使用已有、物理语义已验证的三条
> adjacent pair-relative observables 与冻结 Workbook-64 edge logits，是否能够消除 Workbook 72 中的
> 跨 source truth→fake boundary failure？

**结论（先写）：`source_transfer_gate = true`，hypothesis 被支持。**

---

## 1. local / origin git state

```text
local HEAD  = 5a86f3adbc14505e972ff7d14a1fa846854dbecf   (WB74 decision generator)
origin/4station = cc8d22036be5f0381fa7ad9441477dae155d5a66
local ahead of origin by 12 commits (WB71/72/73/74)
```

WB74 新增 commits（均在本地，未 push）：

```text
6f63dbb  Workbook 74: Physical Pair-Relative Route Encoder V1 (route_representation_mode)
acb0709  WB74 eval: git_dirty reflects tracked-file state
5a86f3a  WB74: decision/audit generator
```

`git push` 仍被 SSH passphrase 非交互阻塞（与 WB71/72/73 相同）。**未 reset 到远端，未丢弃任何本地历史。**
本地 repository + `/eos` outputs 是科学真值。所有 WB74 科学代码已 commit；`outputs/` 按仓库约定 gitignore，
作为本地真值保留在 `/eos`。

---

## 2. Workbook 73 artifact replay（不仅信 Markdown）

实际读取 `outputs/mc24_four_station_route_representation_domain_audit_v1/`：

```text
cross_family_truth_fake_probe.json
physical_relative_probe_summary.json
catastrophic_cohort_summary.json
decision.json
representation_manifest.json
```

复核关键数字（与 WB73 记录一致）：

```text
R_phys (= raw physical edge observables + frozen W64 production edge logits):
  AUC f1->f2 ≈ 0.999 ,  AUC f2->f1 ≈ 0.999
  truth FNR f1->f2 ≈ 0.000 ,  truth FNR f2->f1 ≈ 0.002
R4 (absolute node latent s0..s3):  cross-family truth FNR ≈ 0.39–0.47   <- 灾难
R5 :  f2->f1 truth FNR ≈ 0.048
R6 :  f2->f1 truth FNR ≈ 0.036
decision_case = case_1 , next_hypothesis = Physical Pair-Relative Route Encoder
```

Replay 一致 → 继续。本 Workbook 的 production encoder 实测 truth FNR proxy（见 §18）与 R_phys 诊断一致。

---

## 3. Architecture contract — Physical Pair-Relative Route Encoder V1

实现于 `models/route_transformer.py`，新增 `route_representation_mode: {"absolute", "physical_pair_relative"}`
（默认 `absolute`，**backward compatible**，WB69/72 checkpoint 加载行为不变）。

`physical_pair_relative` 模式下 route scorer **完全不消费**：

```text
s0, s1, s2, s3            (absolute/global node latent)
absolute x/y/z/tx/ty route context
route_query / route_node_key / route_edge_projection / route_pair_embedding   (这些模块根本不构建)
```

只允许消费 `R_phys`（36 维）：

```text
R_phys = [ edge01 physical observables (11),
           edge12 physical observables (11),
           edge23 physical observables (11),
           L01, L12, L23  (frozen Workbook-64 production edge logits) ]
```

**未声称 SE(3)-invariant / gauge-invariant / equivariant。** hypothesis 仅为：相比 absolute node latent，
现有 pair-relative physical observables 的 truth/fake boundary 在两个 source family 间更可迁移。

---

## 4. 精确 R_phys feature list（来自代码真实定义）

11 个 physical pair-relative edge observables（`EDGE_FEATURE_NAMES`，每条 adjacent edge）：

```text
0  residual_x_mm
1  residual_y_mm
2  residual_tx
3  residual_ty
4  pull_x
5  pull_y
6  pull_tx
7  pull_ty
8  log1p_chi2
9  combined_covariance_logdet
10 delta_z_mm
```

edge 顺序严格为 `ADJACENT_STATION_PAIRS = (0->1, 1->2, 2->3)`。
进入模型的 `score_edge_features` 是 frozen W64 standardizer 标准化后的物理 edge observables（R1），
与冻结 edge scorer 的输入完全一致、数值稳定；WB73 表明 R1≈R0 的迁移性。
`L01/L12/L23` 取 frozen W64 **production** edge logits（含 route_edge_correction），由
`RelativeRouteV4Inference` 传入 trainable head。总计 `3*11 + 3 = 36` 维。**未增加任何新 geometry quantity。**

---

## 5. 为什么删除 absolute node latent

WB73 直接证明：absolute/global node latent `s0..s3`（R4）的 cross-family truth FNR ≈ 0.39–0.47，
即它在 source 间承载了不可迁移的 truth/fake boundary；而 R_phys 的 cross-family truth FNR ≈ 0.000–0.002。
WB72 的 catastrophic truth destruction 正是 route head 依赖该 absolute latent 所致。因此 V1 将其完全移除。

---

## 6. Bounded correction（冻结 safety constraint）

```text
B = 4.0
raw_delta = route_head(R_phys)
delta = 4 * tanh(raw_delta / 4)        ∈ [-4, +4]
L_corrected = L_edge_W64 + delta
```

未做 B sweep / learnable B / unbounded primary。B=4 是 WB72 冻结的 safety constraint，不是实验变量。

---

## 7. Control / Primary（三臂）

```text
Arm 0: Frozen Workbook-64 edge-only baseline
Arm 1: Workbook-72 V5A bounded absolute-route control   (每 fold 重新训练)
Arm 2: Physical Pair-Relative Route Encoder bounded primary
```

Arm 1 与 Arm 2 共享：同一 W64 parent、同一 source corpus、同一 seed、optimizer、LR、weight decay、
epoch budget、batching、physical curriculum、loss、loss weights、bounded B=4、solver、OP、checkpoint selection。
**唯一 hypothesis-level 差异：`route_representation_mode`（absolute vs physical_pair_relative）。**
见 `paired_contract_audit.json`：`paired_contract_ok = true`，除 `route_representation_mode`（及由此导致的
`trainable_components`/`arm_name`/description）外两 config 无任何差异。

---

## 8. Trainable parameter counts（自然结构结果，未人工匹配）

```text
Control  (absolute)              : trainable = 172,513 | frozen = 614,947
Primary  (physical_pair_relative): trainable =  21,377 | frozen = 614,947
```

Primary 参数更少是 hypothesis 的自然结构结果（36 维 R_phys 输入 vs 1075 维 absolute 表示），
**未做 random lift / duplicate / zero-padding / dummy 参数**。相同 hidden width (128)、相同 route encoder
depth/activation/norm；仅第一层输入宽度改为 36。结论解释时已注明此点。

---

## 9. Phase A — implementation verification（全部通过才允许训练）

`tests/test_physical_pair_relative_route.py`：**19/19 PASS**。
`tests/test_relative_route_v5a_bounded.py`（regression）：**13/13 PASS**。
route/relative/solver/loss regression suite（route_aware_transformer + relative_route_v4 + route_assignment
+ gauge_consistent_route + dustbin_aware_route_margin）：**59/59 PASS**。

覆盖（Workbook 74 §14）：

* Representation: 无 node latent 模块；不读 s0..s3；输入恰为 3×11 physical + 3 logits；width=36；edge 顺序 01/12/23。
* Frozen physics: L01/L12/L23 来自 frozen W64 production scorer；candidate graph / fragment topology /
  unmatched penalty / solver 不变（head 只发 delta）。
* Bound: delta∈[-4,+4]；zero-init delta=0；zero-init L_corrected=L_edge_W64；save/load 保留 mode+B。
* Data guards: fold train/holdout disjoint；development / Final Blind / sealed test 均被拒绝。
* Numerical: CPU + GPU finite forward；backward 梯度只进 route_encoder/route_score；frozen W64 梯度泄漏=0。

`phaseA_audit.json: training_authorized = true`（任一失败则 false 并停止）。

---

## 10. Zero-init replay（optimizer step 之前）

`scripts/zero_init_replay_physical_pair_relative.py`，对固定 fold events 验证 Arm0 W64 == Arm2@init：

```text
family2 (fold1 train): 21 truth routes, 0 assignment mismatch, max|ΔL_corrected|=0.0, max|ΔU_complete|=2.66e-05
family1 (fold2 train): 35 truth routes, 0 assignment mismatch, max|ΔL_corrected|=0.0, max|ΔU_complete|=2.50e-05
```

selected routes、C/D class 完全一致；L_corrected 精确为 0；U_complete 的 ~2.7e-05 差异来自两条数值路径
（Arm0 edge-only packing utility vs Primary 注入 sigmoid(L_edge_W64)→solver_log_odds 的 logit-clip），
远在 O(1) 决策 margin 之下，符合 solver 数值合同。**PASS。**

---

## 11. Fold source manifests（复用 Workbook 72 已验证 source-pure corpora）

```text
Fold 1 (holdout_family1): train Family2 (1680 events) , holdout Family1 (3360 events)   # 最难方向
Fold 2 (holdout_family2): train Family1 (3360 events) , holdout Family2 (1680 events)
Family1 = {100043, 100044} , Family2 = {100047, 100048}
corpus root: outputs/mc24_four_station_relative_route_v5a_source_transfer_v1/corpora/{family1,family2}/...
```

未重新生成第三套 corpus。每 fold 的 Arm1/Arm2 从同一 frozen W64 parent + 同一 zero-init seed 重新训练。

---

## 12. Condor jobs

Training（HTCondor GPU，4 个 paired jobs；H100 MIG 2g.24gb slice）：

```text
cluster 1109471  holdout_family1 control   b9pgpun015   return 0
cluster 1109472  holdout_family1 primary   b9pgpun217   return 0
cluster 1109473  holdout_family2 control   b9pgpun501   return 0
cluster 1109474  holdout_family2 primary   b9pgpun216   return 0
```

每个 job 记录 cluster id / schedd / hostname / GPU / git commit / config / parent SHA（见各 `run_contract.json`
与 `checkpoint_freeze.json`）。Condor return 0 不是实验 PASS；PASS 由 §19 gate 决定。

Evaluation（HTCondor **CPU**，单 job 跑两 fold × 3 臂）：

```text
cluster 1109475  b9p20p9640   return 0
```

WB72 已证明完整 evaluation 在 CPU 可行（~25 min），故未占 GPU。

---

## 13. Checkpoints / SHA256

```text
W64 parent (frozen)                    : 0c3a28704cc01151fab7ac943e41338e859b5dde1502c0b10e2f12ada04e6236
fold1 control  checkpoint_last.pt SHA  : dea96a6e4122...  (trainable=172513, epochs=30)
fold1 primary  checkpoint_last.pt SHA  : 63ccee02a4fb...  (trainable=21377 , epochs=30)
fold2 control  checkpoint_last.pt SHA  : 57004ff1d183...  (trainable=172513, epochs=30)
fold2 primary  checkpoint_last.pt SHA  : 26ea49d5a7f9...  (trainable=21377 , epochs=30)
```

完整 SHA 见 `paired_contract_audit.json`。所有 arm 的 `frozen_backbone_sha256 == W64 parent`，
`frozen_parameter_hash_invariant = true`，`frozen_workbook64_replica_hash_invariant = true`，
post-training `max|edge_v4 - edge_w64| ≤ 7.6e-06`（冻结数值合同 < 1e-5）。

---

## 14. Fold 1 results（holdout Family1，最难方向 Family2→Family1）

| Arm | C | D | selected | efficiency | purity | fake |
|-----|----|----|----------|------------|--------|------|
| Arm0 W64 | 6 | 74 | 6709 | 0.9882 | 0.9823 | 2943 |
| Arm1 Control (bounded absolute) | **225** | 172 | 6392 | 0.9415 | 0.9828 | 3310 |
| Arm2 Primary (physical pair-relative) | **25** | 78 | 6686 | 0.9848 | **0.9853** | 2981 |

WB72 同方向 bounded absolute 曾 C=237/D=151/catastrophic=215；本 Workbook 重训的 Control 复现了该灾难
（C=225），而 Primary 把 C 压到 25。

## 15. Fold 2 results（holdout Family2，Family1→Family2）

| Arm | C | D | selected | efficiency | purity | fake |
|-----|----|----|----------|------------|--------|------|
| Arm0 W64 | 0 | 43 | 3205 | 0.9868 | 0.9742 | 1573 |
| Arm1 Control (bounded absolute) | 8 | 55 | 3185 | 0.9806 | 0.9743 | 1597 |
| Arm2 Primary (physical pair-relative) | **0** | 51 | 3197 | 0.9843 | 0.9744 | 1585 |

## 16. C/D transitions（truth-route 状态转移）

Fold 1 `W64 -> arm`（关键：`selected -> utility_nonpositive` = catastrophic）：

```text
W64 -> Control : selected->utility_nonpositive = 203 , selected->selected = 6384
W64 -> Primary : selected->utility_nonpositive =  11 , selected->selected = 6679
```

Fold 2 `W64 -> arm`：

```text
W64 -> Control : selected->utility_nonpositive = 8 , selected->selected = 3164
W64 -> Primary : selected->utility_nonpositive = 0 , selected->selected = 3190
```

## 17. Catastrophic truth destruction（与 WB72 同定义）

```text
Fold 1:  control = 203 ,  primary = 11     (W64_selected_truth = 6709)
Fold 2:  control = 8   ,  primary = 0      (W64_selected_truth = 3205)
new_C (from selected + from D):
Fold 1:  control = 219 ,  primary = 19
Fold 2:  control = 8   ,  primary = 0
```

## 18. Primary representation diagnostics（held-out truth routes）

truth FNR proxy（可行 W64 truth route 被 Primary 负 correction 压过 U=0 的比例）：

```text
Fold 1: 0.00164  (11 / 6709)     <- 与 WB73 R_phys 诊断 f2->f1 FNR≈0.002 一致
Fold 2: 0.00000  (0 / 3205)
```

delta 分布（bounded_delta，B=4）：

```text
Fold 1 truth : mean=+0.154 median=+0.180 frac_neg=0.137 frac_saturated=0.000 max_abs=2.38
Fold 1 fake  : mean=-3.95  frac_neg=0.998 frac_saturated_neg=0.938   (强抑制到 dustbin)
Fold 2 truth : mean=+0.883 median=+0.925 frac_neg=0.032 frac_saturated=0.000 max_abs=2.07
Fold 2 fake  : mean=-3.97  frac_neg=0.998 frac_saturated_neg=0.980
fraction truth delta < 0      : fold1 = 0.137 , fold2 = 0.032
fraction truth near -4 (≤-3.96): fold1 = 0.000 , fold2 = 0.000   <- 关键：truth 永不饱和到 -4
```

解读：Primary 对 truth route 给小的有界 correction（绝不接近 -4），对 fake route 强烈压到 -4。
truth 的 13.7%（fold1）/3.2%（fold2）拿到小负 delta，但**无一接近 -4**，因此 catastrophic 极低。
raw_delta / L_edge / L_corrected 分布见各 fold `evaluation.json` 的 `primary_representation_diagnostics`。

## 19. Gate A / B / C（机械判定）

**Gate A（相对 bounded absolute 明显改善，两 fold 均要求）** — 全 PASS：

```text
Fold 1: C 25<225 ✓  catastrophic 11<203 ✓  new_C 19<219 ✓  selected 6686>=6392 ✓
Fold 2: C 0<8    ✓  catastrophic 0<8    ✓  new_C 0<8    ✓  selected 3197>=3185 ✓
```

**Gate B（transfer safety，每 fold）** — 全 PASS：

```text
Fold 1: C/truth_count = 25/6789 = 0.00368 <= 0.01 ✓
        catastrophic/W64_selected_truth = 11/6709 = 0.00164 <= 0.01 ✓
        efficiency_primary 0.9848 >= efficiency_W64 0.9882 - 0.01 ✓ (drop=0.00339)
Fold 2: C/truth_count = 0/3248 = 0.0 <= 0.01 ✓
        catastrophic/W64_selected_truth = 0/3205 = 0.0 <= 0.01 ✓
        efficiency_primary 0.9843 >= efficiency_W64 0.9868 - 0.01 ✓ (drop=0.00246)
```

**Gate C（quality guardrail，每 fold）** — 全 PASS：

```text
Fold 1: purity_primary 0.9853 >= purity_W64 0.9823 - 0.01 ✓ (实际 purity 升高 0.0030)
        fake_rate_primary 0.3084 <= fake_rate_W64 0.3049 + 0.01 ✓ (increase=0.00346)
Fold 2: purity_primary 0.9744 >= purity_W64 0.9742 - 0.01 ✓ (升高 0.0002)
        fake_rate_primary 0.3315 <= fake_rate_W64 0.3292 + 0.01 ✓ (increase=0.00223)
```

（fake metric 统一用 fake_selected_routes / selected_routes 的 rate，未混用 count 与 rate。）

## 20. Final decision

```text
Fold1 PASS  AND  Fold2 PASS
=> source_transfer_gate = true
=> representation_transfer_hypothesis_supported = true
```

注意（§22）：本阶段证明的是 **representation transfer safety**，不是 “route head 已优于 W64 production”。
Primary 在 fold1 的 C=25 仍高于 W64 的 6，efficiency 与 W64 相当但未超越，D 未明显改善。
因此**不声称 association problem 已解决**。

## 21. Development access status

```text
development_used_for_training_or_evaluation = false
mc24_100047_00350_00399 / mc24_100048_00350_00399 未被本 Workbook 读取用于任何训练或评估。
```

## 22. Final Blind status

```text
final_blind_eval_authorized = false
new_final_blind_content_accessed = false
mc24_100047_00800_00849 / mc24_100048_00800_00849 未以任何形式访问（无 ROOT open / event count / metadata / feature / inference）。
```

## 23. sealed test

```text
sealed_test_accessed = false   (永久关闭)
```

## 24. 15D WLS state

```text
continue_to_15d_relative_wls = false
```

## 25. 是否授权 Workbook 75

```text
full_six_source_training_authorized = true_for_workbook75_physical_pair_relative
```

唯一授权的下一步：**Workbook 75 — Physical Pair-Relative Route Encoder 的 full six-source training**。
保持 `architecture transfer validation → full training → frozen development` 三阶段独立。
本任务**未**执行六源 full training、未触碰 development、未触碰 Final Blind、未触碰 sealed test。

---

## Outputs

```text
outputs/mc24_four_station_physical_pair_relative_route_v1_source_transfer/
  phaseA_audit.json
  paired_contract_audit.json
  decision.json
  training/holdout_family1/{control,primary}/   (checkpoint_last.pt + run_contract + checkpoint_freeze + calibration)
  training/holdout_family2/{control,primary}/
  training/condor/                               (4 个训练 job 的 sub/log/out/err)
  evaluation/holdout_family1/evaluation.json     (= fold1)
  evaluation/holdout_family2/evaluation.json     (= fold2)
  evaluation/source_transfer_summary.json
  evaluation/condor/                             (CPU eval job)
```

（fold1 ≡ holdout_family1，fold2 ≡ holdout_family2。）
