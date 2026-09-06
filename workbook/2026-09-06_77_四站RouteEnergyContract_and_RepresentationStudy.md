# Workbook 77 — RouteEnergy contract, fail-closed metrics, and representation-study protocol

日期： 2026-09-06
分支： `4station`
任务： **先修合同，不训练。** 停止 V5A / frozen-head 主线。为后续 W64 vs raw physical vs trainable route-energy 对照建立可失败关闭的实验基础。

---

## 0. 结论（先写）

```text
training_authorized = false
final_blind_eval_authorized = false
new_final_blind_content_accessed = false
sealed_test_accessed = false
continue_to_15d_relative_wls = false
continue_v5a_frozen_head = false
```

P0 合同已落地并可单测：

| 合同 | 版本 | 状态 |
|---|---|---|
| Route energy | `raw_energy_v1` | 已实现；legacy decoder 仅用于历史回放 |
| Evaluation | `route_accounting_v2` | 正确 fragment 不再算 complete fake |
| Gates | fail-closed | 缺 metric / `None` / NaN / 空 spec 永不 PASS |
| Enumeration | truth-free physical walk | 缺 truth 或置换 truth ID 不改候选 |
| Loss | exact inclusion gap + LAI hinge | 接口已立；未接入现有训练 |

**没有提交 Condor 训练，没有改历史 artifact，没有打开 Final Blind / sealed test。**

WB74 source-transfer PASS 与 WB75/76 development FAIL 都不被本 workbook 推翻。它们继续是历史结果，语义仍绑定当时的 legacy decoder 与当时的 fake 计数。

---

## 1. Git

```text
review snapshot      = e6ae90d   (4station_research_review_and_roadmap.md)
this workbook HEAD   = (commit that adds this file)
origin/4station      = tracking may be stale; local history is the scientific source of truth
```

本 workbook 之前的 P0 commits：

```text
42b1241  lock raw_energy_v1 RouteEnergy contract
183d5e5  split route accounting and fail-closed gates
2ef6439  keep route enumeration independent of truth IDs
4a26253  add exact inclusion-gap and loss-augmented hinge
```

---

## 2. 当前代码与计划差异

审查文档与用户优先级要求的是：先修 utility / metric / truth / structured loss，再做公平 representation study，再实现显式 route-energy 主模型，最后才谈 alignment 资格。对照 `HEAD=e6ae90d` 时的代码：

| 计划项 | 当时代码 | 本 workbook 之后 |
|---|---|---|
| 统一 RouteEnergy | `_route_hypotheses` 对 complete 做 sigmoid→clip→logit，对 fragment 做逐边 clip；零修正即可把 logit=8 的完整链打成 fragment（U≈9.82 vs 13） | **新** `models/route_energy.py`；历史路径未改默认行为 |
| 指标拆分 | `RouteMetrics.track_fake_rate` 用 `selected-truth_consistent`（正确）；V5A/WB74/75 脚本 `_metrics_from_routemetrics` 用 `selected-correct_complete`（把正确 fragment 当 fake） | **新** `evaluation/route_accounting.py`；历史脚本不回写 |
| Fail-closed gate | 缺 metric 可变成 Python `None` 比较后的假 PASS | **新** `evaluation/review_gates.py` |
| Truth 解耦 | `enumerate_complete_route_candidates` 无 truth 直接 raise | 物理枚举不再需要 truth；标签后贴 |
| Global set loss | `packing_route_competition_loss` 只比最强单 rival | **新** inclusion gap / LAI；旧 loss 保留给历史训练 |
| Representation study | 未设计；V5A/WB75 仍是 frozen-head | **协议已冻结**，`training_authorized=false` |
| Explicit route-energy 模型 | 不存在低容量 2/3/4 共享 scorer | **未实现**（P0 之后才允许） |
| Alignment 资格 | closure 被写成可用 alignment | **只降级陈述**；未跑 SE(3)/WLS 新实验 |

仍未改、也不应在本阶段改的部分：

- `baselines/route_assignment._route_hypotheses` 默认仍是 legacy decoder（历史可复现）。
- `datasets/synthetic_overlay.select_complete_truth_tracks` 仍是 truth-assisted 干净 pool（数据构造，不是 inference 枚举）。
- 历史 V5A / WB74 / WB75 evaluation JSON 的 `fake_selected_routes` 分子保持原样。
- Alignment backend 与已有 closure artifact 不覆盖。

---

## 3. 修改 / 新增文件

**新增**

- `models/route_energy.py`
- `evaluation/route_accounting.py`
- `evaluation/review_gates.py`
- `evaluation/route_counterfactuals.py`
- `training/global_route_energy_loss.py`
- `tests/test_route_energy_contract.py`
- `tests/test_route_accounting.py`
- `tests/test_review_gates.py`
- `tests/test_route_truth_decoupling.py`
- `tests/test_route_counterfactuals.py`
- `tests/test_global_route_energy_loss.py`
- `configs/research_review/wp1_energy_contract.yaml`
- `configs/research_review/wp4_representation_study.yaml`
- `workbook/2026-09-06_77_四站RouteEnergyContract_and_RepresentationStudy.md`

**修改**

- `models/__init__.py`（导出 energy contract）
- `training/route_aware_transformer.py`（物理枚举 / 标签分离）

---

## 4. P0 合同（已实现）

### 4.1 `raw_energy_v1`

```text
U(r) = sum_e energy_e + correction(r) + n_stations * unmatched_penalty
```

solver 直接最大化 `U`。禁止把 probability 当中间量。诊断概率是可选后处理，不进 packing。

零修正 identity（三边 logit=8，`unmatched=-1`）：

| 合同 | complete U | 3-station U | 选出 |
|---|---:|---:|---|
| `raw_energy_v1` | 20 | 13 | complete |
| `legacy_prob_clip_logit_v1` | ≈9.816 | 13 | fragment |

新实验调用 `assign_from_energy_table` 时，legacy 表必须显式 `allow_legacy=True`，否则 fail-closed。

### 4.2 `route_accounting_v2`

拆分后的正式分子：

- complete fake rate = fake complete / selected complete
- fragmentation = 有一致 fragment、没有正确 complete 的 truth chain
- unmatched truth chains / unmatched truth nodes
- clone / mixed conflict
- all-route purity / fake = truth-consistent / selected

`legacy_selected_minus_correct_complete` 只作为对照字段，禁止进 gate。

### 4.3 Truth / inference

`enumerate_physical_complete_route_chains` 只用 station 与邻接边。
`attach_route_candidate_labels` 才读 truth。缺 truth 时推理可执行；标签全假。
truth ID 的单射重编号不改变物理链、特征和 consistency 标签。

### 4.4 Global inclusion gap

两兼容 fake（U=6, 6）对 truth（U=10）：

```text
single-rival margin = 10 - 6 = +4     # 旧 packing loss 认为已分开
inclusion gap       = 10 - 12 = -2    # exact set packing 选 fake 集合
```

`loss_augmented_structured_hinge` 在同一 unit-capacity solver 上做 Hamming LAI。
**没有**把它接到现有 V2/V4/V5 训练循环。

---

## 5. 测试

环境：`source scripts/setup_environment.sh ml`，单线程 BLAS。

本 workbook 新增测试在提交前通过：

```text
tests/test_route_energy_contract.py
tests/test_route_accounting.py
tests/test_review_gates.py
tests/test_route_truth_decoupling.py
tests/test_route_counterfactuals.py
tests/test_global_route_energy_loss.py
tests/test_route_assignment.py
tests/test_route_aware_transformer.py
tests/test_structured_assignment.py
tests/test_physical_pair_relative_route.py
```

未声称重跑全库 460+ tests。已知历史失败 `test_registered_disk_contract_matches_workbook_45` 仍依赖外部 artifact，与本变更无关。

---

## 6. 公平 representation study（协议，未执行）

科学问题：

> 在 **同一候选、同一 exact solver、同一 `raw_energy_v1`、同一 `route_accounting_v2`、同一 source/geometry split** 下，association 失败来自 frozen latent 信息瓶颈，还是 objective/utility mismatch？

| Arm | 输入 | 是否训练 | 目的 |
|---|---|---|---|
| A | 冻结 W64 边 logit 的 canonical energy | 否 | production baseline |
| B | raw physical route 特征 → 低容量 energy scorer | 是（仅 scorer） | 测 raw 物理量是否够 |
| C | 同一物理输入上的可训练共享 route representation | 是（低容量 encoder + scorer） | 测是否需要学习表示 |

固定物理输入（Arm B/C）：propagation residual / pull / chi2 / covariance logdet / `delta_z` / tracklet state / route length / missing-station mask。**不是** V4 的 `[h0, h1-h0, ...]` latent 重参数化。

必须披露：W64 parent `0c3a2870…` 见过全部六源，Arm A **不是**全模型 unseen-family。Arm B/C 按 family 做 2-fold source-holdout 才是信息瓶颈对照。

几何 holdout 必须 **新生成 payload 表**（train seed `271828`，held-out seed `314159`）。禁止把已经打开的 `00350_00399` 叫 geometry holdout；它只是 development 诊断，禁止用于选模型或选 OP。`00800_00849` 与 sealed test 永不打开。

正式训练 / 批量评估必须 HTCondor。每个 run 保存 git SHA、resolved config、dataset manifest、checkpoint ancestry、完整 evaluation JSON。禁止覆盖历史 artifact。

协议文件：`configs/research_review/wp4_representation_study.yaml`。
`training_authorized = false`。本 workbook **不实现** Arm B/C 网络，**不提交** Condor。

---

## 7. Alignment 陈述降级（无新实验）

已有 route-selected / relative closure 只证明：

> given this association and these residual banks, linear response regression works.

它 **不能** 声称完整 alignment loop，也不能当 association 的已认证 downstream oracle。后续资格（未授权、未做）：

- 固定外场 JG 与场/材料一起变的坐标变换；
- 15D chart 的参考与近似阶；
- 独立噪声、robust outlier、更新几何后真实 refit、uncertainty coverage。

`continue_to_15d_relative_wls = false`。

---

## 8. 明确停止的方向

- V4 / V5A frozen-backbone Head-Only 调 loss / bound / 再训。
- 把 WB74 自动升级成新的 six-source production 候选。
- 在合同未锁定前比较模型精度。
- 打开 Final Blind / sealed test。

---

## 9. 下一允许步骤

只有在用户明确授权后才可以：

1. 实现低容量 Explicit Route Energy Model（Arm B 先于 Arm C），训练目标走 `raw_energy_v1` + LAI / inclusion gap；
2. 生成 geometry-holdout payload 表并做 HTCondor source-transfer；
3. 另开 alignment 资格 workbook。

当前 **不授权** 上述任何一项。
