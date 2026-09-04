# 条目 70 — RelativeRoute V4 Reserved-Blind Development 评估（Arm 0 / Arm 1 / Arm 2）

日期：2026-09-03（关闭审计 2026-09-04）  
分支：`4station`  
状态：**评估完成。作业 `1108310` Normal termination return 0。三臂 Workbook-64 门均失败；Arm 1/2 未优于 Arm 0。`continue_to_15d_relative_wls=false`。禁止用本条目数字改 OP / loss / architecture，禁止打开 `00800_00849`，禁止授权 15D WLS。**  
前置授权：Workbook 69 配对训练合同通过（`1104860`/`1104861` last-epoch checkpoint）  
物理闭环合同：`continue_to_15d_relative_wls = false`  
密封 Test：`sealed_test_accessed = false`  
新 Final Blind：`new_final_blind_content_accessed = false`（`00800_00849` 仍未打开）  
Development：本条目**已打开** reserved-blind overlay `mc24_100047_00350_00399` / `mc24_100048_00350_00399`，仅作冻结 checkpoint 的 development 评估。

---

## 1. 科学问题

在 **完全冻结** Workbook-64 边、identity Platt、packing `0.001 / -1.0`、candidate graph 与 unit-capacity solver 的前提下，问：

| 对比 | 问题 |
| --- | --- |
| Arm 1 − Arm 0 | 显式 complete-route correction head 能否相对 C/D 边-only 基线恢复 reserved-blind 上的完整轨效率 |
| Arm 2 − Arm 1 | Station-0-anchored relative representation 是否相对 absolute-route control 有额外增益 |

三臂同一 reserved-blind overlay、同一 OP、同一 last-epoch checkpoint。禁止用本条目结果选 checkpoint、改阈值或重训。

---

## 2. 冻结输入

| 臂 | 角色 | Checkpoint | SHA256 | `complete_route_scores` |
| --- | --- | --- | --- | --- |
| Arm 0 | Workbook-64 edge-only 基线 | `outputs/mc24_four_station_source_diversity_v1/checkpoint/route_aware_transformer_v2.pt` | `0c3a28704cc01151fab7ac943e41338e859b5dde1502c0b10e2f12ada04e6236` | **None** |
| Arm 1 | Absolute-route control | `.../absolute_control/checkpoint_last.pt` | `e8a6d6c6e26545de91d9ded59c7b4f9b7840de2e9c94b88d56e1857d6aebd3c8` | `σ(L_edge_W64 + delta)` replace |
| Arm 2 | RelativeRoute V4 primary | `.../relative_primary/checkpoint_last.pt` | `a52037ead07555ef937a7e02b65adc9526b509e765341e8af31992552f1fdd3a` | 同上 |

Overlay：`outputs/mc24_four_station_source_diversity_blind_v1/overlay_synthetic_v1/synthetic_corpus_manifest.json`  
Iteration：`outputs/mc24_four_station_source_diversity_blind_v1/iteration_manifest.json`（`reserved_blind_validation_only=true`）  
OP：identity Platt；阈值 `0.001`；unmatched `−1.0`；composition `replace`；tie-break `1.0e-9`。  
Gates：`configs/physical_four_station_diversity_training.yaml`（raw chain/edge ≥0.90；nominal purity ≥0.95、fake ≤0.05；vs-nominal efficiency drop ≤0.10、purity drop ≤0.05、fake increase ≤0.05）。  
C/D：`production_hypotheses` / `audit_event_truth_chains` 必须消费 packing 的 `L_corrected`；Arm 0 复放 Workbook 65 `C=200`、`D=170`。

禁止：

- 打开 `mc24_100047_00800_00849` / `mc24_100048_00800_00849`
- 打开 sealed test
- 重训、改 OP、改 loss、改 architecture、early stopping、多 seed
- 从本条目数字授权 15D WLS

---

## 3. 代码与提交

- 评估：`scripts/evaluate_relative_route_v4_development.py`
- 合同：`configs/relative_route_v4_head_only_development_eval.yaml`
- Condor worker：`scripts/run_relative_route_v4_development_eval_condor.sh`
- 提交：`python3 scripts/submit_relative_route_v4_development_eval_condor.py --submit`

一次 GPU 作业跑完三臂。flavour `tomorrow`，`request_gpus=1`，`request_disk=2000000`，`myschedd out` / `bigbird24`。不用交互式 `lxplus-gpu`。

### 3.1 提交记录

```bash
python3 scripts/submit_relative_route_v4_development_eval_condor.py --submit
```

| 作业 | Cluster | Schedd | flavour | 日志 |
| --- | ---: | --- | --- | --- |
| Arm 0/1/2 development eval | **1108310** | `bigbird24.cern.ch` | tomorrow | `outputs/mc24_four_station_relative_route_v4_head_only/condor_eval/logs/eval.1108310.*` |

提交时刻 stdout：`1 job(s) submitted to cluster 1108310.`  
本条目提交后不监控 Condor；作业完成后由用户通知再审计产物。

成功判定（完成后必须逐项核对）：

1. Condor return 0
2. Arm 0 checkpoint SHA 与 Workbook 64 一致；Arm 1/2 SHA 与上表一致
3. Arm 0 C/D 复放 `C=200`、`D=170`
4. 三臂都写出 payload 级 efficiency / purity / fake 与 Workbook 64 `assess()` 门
5. `decision.json` 中 `continue_to_15d_relative_wls=false`、`new_final_blind_content_accessed=false`、`test_data_accessed=false`
6. 输出目录 `outputs/mc24_four_station_relative_route_v4_head_only/development_eval/`

---

## 4. 有效作业 `1108310`

| 项 | 值 |
| --- | --- |
| Cluster | **1108310** |
| Schedd | `bigbird24.cern.ch` |
| 节点 / slot | `slot1_4@b9pgpun014.cern.ch` |
| GPU | NVIDIA H100 NVL `GPU-5b10c343` |
| TimeExecute | 146 s |
| 终止 | Normal termination **return 0** |
| stderr | 空 |
| 日志 | `outputs/mc24_four_station_relative_route_v4_head_only/condor_eval/logs/eval.1108310.*` |
| 产物 | `outputs/mc24_four_station_relative_route_v4_head_only/development_eval/` |

Worker stdout：`hostname b9pgpun014.cern.ch`；`date_utc 2026-09-03T17:34:05Z`；`git_commit 4d1484a656caf384c0b6f9197cf28d342bdb66f2`；作业开始时工作树脏（README、C/D 接线测试、本条目评估脚本尚未 commit）。结尾：`=== Workbook 70 development evaluation completed successfully ===`。

作业成功 ≠ 科学门通过。下面数字全部来自产物 JSON 的独立复核，不是 Condor return 0。

---

## 5. Checkpoint SHA 与密封合同（独立复核）

磁盘 SHA256 与 `decision.json` / 冻结表一致：

| 臂 | 文件 | SHA256 |
| --- | --- | --- |
| Arm 0 | `outputs/mc24_four_station_source_diversity_v1/checkpoint/route_aware_transformer_v2.pt` | `0c3a28704cc01151fab7ac943e41338e859b5dde1502c0b10e2f12ada04e6236` |
| Arm 1 | `.../absolute_control/checkpoint_last.pt` | `e8a6d6c6e26545de91d9ded59c7b4f9b7840de2e9c94b88d56e1857d6aebd3c8` |
| Arm 2 | `.../relative_primary/checkpoint_last.pt` | `a52037ead07555ef937a7e02b65adc9526b509e765341e8af31992552f1fdd3a` |

`complete_route_query_injected`：Arm 0 `false`，Arm 1/2 `true`。

Overlay / iteration 只含 reserved-blind 验证对：

- `mc24_100047_00350_00399`、`mc24_100048_00350_00399`
- `reserved_blind_validation_only=true`，`allowed_splits=['validation']`，`forbidden_splits=['test','train']`
- `materialized_splits=['validation']`，7 payloads，3378 条 complete truth chains
- overlay / physical corpus / 三臂 `*_evaluation.json` **不含** `00800_00849`
- `decision.json`：`development_accessed=true`，`new_final_blind_content_accessed=false`，`test_data_accessed=false`，`do_not_retune_after_seeing_development=true`

---

## 6. Arm 0 Workbook-65 C/D 复放

`decision.json` `arm0_workbook65_cd_replay.ok=true`。独立把七个 payload 的 C/D 相加，与 `cd_pooled` 一致：

| 臂 | A | B | C | D | selected | unselected | truth |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Arm 0 | 0 | 0 | **200** | **170** | 3008 | 370 | 3378 |
| Arm 1 | 0 | 0 | 327 | 123 | 2928 | 450 | 3378 |
| Arm 2 | 0 | 0 | 441 | 135 | 2802 | 576 | 3378 |

每条 payload 满足 `A=B=0`、`C+D=unselected`、`selected+unselected=complete_truth_chains`。

**C/D 差值符号勘误（产物冻结，不重跑）：** `1108310` 的 `paired_comparison.json` / `decision.json` 把 `arm1_minus_arm0_C` 写成了 **Arm0−Arm1**（−127），与同文件里 efficiency 的 later-minus-earlier 相反。科学计数以绝对 C/D 为准：

| 对比 | ΔC（later−earlier） | ΔD（later−earlier） |
| --- | ---: | ---: |
| Arm 1 − Arm 0 | **+127** | **−47** |
| Arm 2 − Arm 1 | **+114** | **+12** |

评估脚本已改为 later-minus-earlier；`1108310` JSON **不得覆盖**。

---

## 7. Payload 级 efficiency / purity / fake

| payload | Arm 0 eff | Arm 1 | Arm 2 | Arm1−Arm0 | Arm2−Arm1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `draw_00` | 0.81667 | 0.78125 | 0.71875 | −0.03542 | −0.06250 |
| `draw_00_plus_common` | 0.84663 | 0.81391 | 0.78119 | −0.03272 | −0.03272 |
| `draw_01` | 0.91116 | 0.89876 | 0.86570 | −0.01240 | −0.03306 |
| `draw_01_plus_common` | 0.90515 | 0.87629 | 0.85155 | −0.02887 | −0.02474 |
| `hard_s3_ry` | 0.92917 | 0.92292 | 0.88542 | −0.00625 | −0.03750 |
| `hard_s3_ry_plus_common` | 0.89167 | 0.85833 | 0.81667 | −0.03333 | −0.04167 |
| `reference` | 0.93333 | 0.91667 | 0.88750 | −0.01667 | −0.02917 |

七个 payload 上 Arm 1 全部低于 Arm 0，Arm 2 全部低于 Arm 1。Purity 多数略升、fake 多数略降，但效率与 C 同时变差：head 把更多 truth 链推进机制 C（`U_truth≤0`），而不是从 packing 竞争里救回整轨。

---

## 8. Workbook-64 三层门（独立复算 `assess()`）

门限来自 `configs/physical_four_station_diversity_training.yaml`：raw chain/edge ≥0.90；nominal purity ≥0.95、fake ≤0.05；vs-nominal efficiency drop ≤0.10、purity drop ≤0.05、fake increase ≤0.05；gauge twin |Δeff|≤0.05。

| 层 | Arm 0 | Arm 1 | Arm 2 |
| --- | --- | --- | --- |
| layer1 raw candidate | **true**（全部 payload chain/edge recall=1.0） | true | true |
| layer2 vs own nominal | **false** | **false** | **false** |
| layer3 gauge | true | **false** | **false** |
| `adjacent_23_and_s3` | false | false | false |
| `gate.passed` | **false** | **false** | **false** |
| `coverage_class` | `source_diversity_blind_failed_other` | `source_diversity_insufficient_for_gauge_transfer` | 同 Arm 1 |
| `continue_to_15d_relative_wls` | false | false | false |

layer2 失败原因（相对 **本臂** `iteration_00_reference` 的 efficiency drop > 0.10；purity/fake 未破门）：

| payload | Arm 0 Δeff | Arm 1 Δeff | Arm 2 Δeff |
| --- | ---: | ---: | ---: |
| `draw_00` | **0.11667** | **0.13542** | **0.16875** |
| `draw_00_plus_common` | 0.08671（过） | **0.10276** | **0.10631** |

2→3 恢复失败（drop > 0.10）：

- Arm 0：`draw_00_plus_common` drop=0.10129
- Arm 1：`draw_00` 0.11279；`draw_00_plus_common` 0.10952
- Arm 2：`draw_00` 0.12626；`draw_00_plus_common` 0.13121

layer3 gauge（twin |Δeff| 上限 0.05）：

- Arm 0：三族均过（`hard_s3_ry` 0.0375）
- Arm 1：`hard_s3_ry` **0.06458** 失败
- Arm 2：`draw_00` **0.06244**、`hard_s3_ry` **0.06875** 失败

`classify_blind_decision`：Arm 0 的 plus-common association 仍过，故 `failed_other`；Arm 1/2 的 plus-common 也破 efficiency 门，故 `source_diversity_insufficient_for_gauge_transfer`。两者的 `next` 都是 `stop_v2_mainline_and_discuss_architecture_level_relative_gauge_equivariant_representation`。这是 **预注册分类标签**，不是新实验授权。

---

## 9. 已确认 vs 仅声称

**已确认（磁盘 / JSON / Condor 日志 / 独立复算）：**

1. `1108310` return 0，H100 NVL，146 s
2. 三臂 checkpoint SHA 与冻结表一致
3. Arm 0 C/D 复放 C=200、D=170
4. 三臂都写出 payload 级 efficiency / purity / fake 与 `assess()` 门
5. 三臂 `workbook64_gates_passed=false`；layer1 过、layer2 不过
6. Arm 1 相对 Arm 0、Arm 2 相对 Arm 1：效率七项全负，C 上升
7. `continue_to_15d_relative_wls=false`；未打开 `00800_00849`；未打开 sealed test
8. overlay 源仅为 `00350_00399` validation 对

**仅声称 / 不得当作通过：**

- Condor “completed successfully” 不是科学 PASS
- 训练 loss 下降（Workbook 69）不能外推到 reserved-blind
- `arm1_minus_arm0_C=-127` 是 1108310 键名写反，真实 Arm1−Arm0 C 是 **+127**
- 不得把 `coverage_class` 的 `next` 字段当成已经授权下一条训练

---

## 10. 授权

```text
workbook70_development_eval = completed_gate_failure
continue_to_15d_relative_wls = false
sealed_test_accessed = false
new_final_blind_content_accessed = false
development_accessed = true          # 仅 00350_00399；禁止再当未见过的 blind 使用
do_not_retune_after_seeing_development = true
do_not_open_00800_00849 = true
training_authorized = false
final_blind_eval_authorized = false
```

本条目关闭后 **没有** 自动下一条。未知关联门未过，15D WLS 仍未授权。禁止：重训、改 OP / Platt / unmatched / packing、early stopping、多 seed、打开 `00800_00849`、把 development 数字当选模型依据。

若继续，需要用户显式授权的新条目，候选方向（讨论，不是执行）：

1. 停 V2/V4 head-only 主线，讨论架构级 relative / gauge-equivariant representation（预注册 `next`）
2. 调查 train-only C/D 目标为何在 reserved-blind 上放大机制 C（不改 OP，只审计）
3. 其它用户指定的最小诊断

当前：**Workbook 70 评估已完成且门失败；15D WLS 未授权。**
