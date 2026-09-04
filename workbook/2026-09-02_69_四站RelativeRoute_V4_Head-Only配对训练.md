# 条目 69 — RelativeRoute V4 Head-Only 配对训练（Arm 1 Control / Arm 2 Primary）

日期：2026-09-02  
分支：`4station`  
状态：**配对训练完成。有效作业 `1104860`/`1104861` 均 Normal termination return 0，last-epoch checkpoint 可读，配对合同通过。`9254670`/`9254671`/`1104565`/`1104567` 不计入。Workbook 70 development 评估已授权；本条目仍未打开 development / final blind / sealed test，也未授权 15D WLS。**  
前置授权：Workbook 68A（C/D 梯度）+ Workbook 68B（冻结 W64 生产边接线 + 正确保存接口）  
物理闭环合同：`continue_to_15d_relative_wls = false`  
密封 Test：`sealed_test_accessed = false`  
新 Final Blind：`new_final_blind_content_accessed = false`  
Development：`development_accessed = false`（本条目禁止打开 `00350_00399`）

---

## 1. 目标与假设

在 **完全冻结 Workbook-64 生产边** 的前提下，训练一个零初始化的 explicit complete-route correction head：

```text
L_corrected = L_edge_W64 + delta_route_logit
complete_route_scores = σ(L_corrected)
```

两臂唯一允许差别是 route representation：

| 臂 | 角色 | 表征 |
| --- | --- | --- |
| Arm 0 | 冻结基线（不训） | Workbook-64 edge-only |
| Arm 1 | Absolute-Route Control | `[s0,s1,s2,s3]` |
| Arm 2 | RelativeRoute V4 Primary | `[s0,s1-s0,s2-s0,s3-s0]` |

科学问题（本条目只训练，不做 development 判定）：

- 显式整轨修正头能否在 train-only C/D 目标上产生非零 `delta`；
- 两臂 trainable 参数量、冻结边、candidate graph 是否保持配对。

Workbook 70 才允许打开 development。禁止用本条目结果改 OP / loss / architecture。

---

## 2. 前置与禁止

允许：

- 同一冻结 W64 checkpoint `0c3a28704cc01151fab7ac943e41338e859b5dde1502c0b10e2f12ada04e6236`
- 同一六源 train overlay `outputs/mc24_four_station_source_diversity_train_v1/overlay_synthetic_v1/synthetic_corpus_manifest.json`
- 同一 15D `15d_gauge_then_left_se3` curriculum、AdamW `2e-4`、batch 32、seed `20260822`、30 epoch、last-epoch checkpoint
- 同一 solver-aware 权重（packing `0.07061055340401011`、dustbin `0.05`、gauge `1.0`）
- Condor GPU 提交；失败若是 EOS/CUDA/preemption 且科学合同未改，允许 exact resume

禁止：

- 多 seed、超参扫描、early stopping、development 选 checkpoint
- resume Condor `9254670` / `9254671` / `1104565` / `1104567`（均无 checkpoint）
- 打开 development / final blind / sealed test
- 改 candidate / solver / OP / 15D WLS

`9254670`/`9254671` 的 30 epoch 日志只证明旧接线曾跑过，**不**计入本条目。  
`1104565`/`1104567` 只证明 68B wrapper 在 GPU 上能加载六源并零初始化，**不**计入训练完成。

---

## 3. 实际 git / 环境

提交作业时：

- 分支：`4station`
- HEAD：`7234deb8347c4a87959a8f85b28083fe5c629cc8`
- working tree：**dirty**（68B 接线/保存修复尚未单独 commit）。Condor worker 从 EOS 项目树读当前文件；`environment.json` 将记录 `git_dirty=true` 与 `git status --porcelain`。
- 主机：`lxplus908.cern.ch`（只提交，不长训）
- CVMFS：`/cvmfs/sft.cern.ch/lcg/views/LCG_110_cuda/x86_64-el9-gcc13-opt/setup.sh`

短验证（交互节点）：

```text
pytest -q tests/test_relative_route_v4.py tests/test_route_aware_transformer.py
        tests/test_route_assignment.py tests/test_blind_failure_localization.py
# 44 passed in 25.76s
python scripts/audit_route_head_solver_gradients.py
# Problem C/D grad_norm(trainable_head)=1.293328, frozen leaks=0
```

---

## 4. 数据与配置

| 项 | 路径 / 值 |
| --- | --- |
| Arm 1 config | `configs/absolute_route_control_head_only_train.yaml` |
| Arm 2 config | `configs/relative_route_v4_head_only_train.yaml` |
| 六源 | `mc24_100043_00200_00299`, `mc24_100044_00300_00399`, `mc24_100043_00300_00399`, `mc24_100044_00200_00299`, `mc24_100047_00100_00149`, `mc24_100048_00100_00149` |
| 父 checkpoint | `outputs/mc24_four_station_source_diversity_v1/checkpoint/route_aware_transformer_v2.pt` |
| 输出 Arm 1 | `outputs/mc24_four_station_relative_route_v4_head_only/absolute_control/` |
| 输出 Arm 2 | `outputs/mc24_four_station_relative_route_v4_head_only/relative_primary/` |
| checkpoint schema | `faser-relative-route-v4-head-only-v1` |

---

## 5. Condor 提交

复用仓库既有 `lxbatch/eossubmit` + `myschedd out` + `condor_submit` 约定，脚本 `scripts/submit_relative_route_v4_head_only_condor.py`。

```bash
python3 scripts/submit_relative_route_v4_head_only_condor.py --submit
```

### 5.1 第一次提交：失败，不计入训练

| 臂 | Cluster | Schedd | 节点 / GPU | 结果 |
| --- | ---: | --- | --- | --- |
| Arm 1 | **1104565** | `bigbird24.cern.ch` | `b9pgpun204` H100L-2-24C MIG 2g.24gb，torch 2.11.0 / CUDA 12.5 | Normal termination return 1，epoch=0，无 checkpoint |
| Arm 2 | **1104567** | `bigbird24.cern.ch` | 同上 | 同上 |

日志：`outputs/mc24_four_station_relative_route_v4_head_only/condor/logs/arm{1,2}.110456{5,7}.*`。  
科学侧 init 已通过（7 samples / 6 sources / 5040 graphs / trainable=172513 / frozen=614947 / zero-init delta=0）。失败点是预训边身份 `max(|edge_v4-edge_w64|)`：Arm1=`9.5367431641e-07`、Arm2=`5.8114528656e-07`，当时阈值 `1e-12`。Allocated Disk 仅 2048 KB（`RequestDisk=3`）。详见 68B §7。**禁止 resume。**

### 5.2 第二次提交：当前有效作业

身份容差改为 `1.0e-5`；`L_corrected` 用同一次 forward 核对 `1e-12`；submit 文件显式 `request_disk = 2000000`。科学合同未改。提交命令：`python3 scripts/submit_relative_route_v4_head_only_condor.py --submit`。pytest 44 passed（15.57s）。

| 臂 | Cluster | Schedd | flavour | GPU / 资源 | 日志 |
| --- | ---: | --- | --- | --- | --- |
| Arm 1 | **1104860** | `bigbird24.cern.ch` | tomorrow | 1 GPU, 16 GB, 4 CPU, 2e6 KB disk | `outputs/mc24_four_station_relative_route_v4_head_only/condor/logs/arm1.1104860.*` |
| Arm 2 | **1104861** | `bigbird24.cern.ch` | tomorrow | 同上 | `outputs/mc24_four_station_relative_route_v4_head_only/condor/logs/arm2.1104861.*` |

提交时刻 `condor_q`：两臂 **Idle**。`getenv=True` 使 schedd 侧实际 `RequestCpus=6`、`RequestMemory=18000`（与 `1104565`/`1104567` 启动时相同，不是科学合同变更）。`request_disk=2000000` 已写入 submit 文件。`condor_q -better-analyze` 当时 0 willing GPU slot、44 drained；这与第一次提交排队后仍能上到 `b9pgpun204` 的情况相同，属于 GPU Hostgroup 池忙，不是 disk/requirements 写错。

旧 AFS 路径 `/afs/cern.ch/user/x/xcheng/condor_logs/relative_route_v4/` 只保留 `9254670`/`9254671`。

成功判定（完成后必须逐项核对，不能只看 return code）：

1. Condor normal termination return 0
2. `checkpoint_last.pt` 可读，schema `faser-relative-route-v4-head-only-v1`
3. `training_history.json` 恰好 30 epoch
4. `early_stopping=false`，`checkpoint_selection=last_completed_epoch_of_fixed_30_epoch_budget`
5. frozen W64 参数 hash 训练前后一致；post-training `max(|edge_v4-edge_w64|) <= 1.0e-5`（两次独立 GPU 前向的 float32 容差；同一次 forward 的 `L_corrected=L_edge+delta` 仍为 `1e-12`）
6. 两臂 trainable=172513、frozen=614947
7. `run_contract.json` 除 representation / arm 字段外一致
8. `development_validation_loaded=false`，`final_blind_loaded=false`，`test_events_loaded=false`

若需改 batch/LR/loss/architecture 才能继续：`training_contract_broken=true`，停止，另开 workbook。

---

## 6. 本条目尚未记录的量（作业完成后追加）

- checkpoint SHA256（两臂）
- frozen backbone hash
- train-only `delta_route_logit` 分布
- paired contract audit
- 是否授权 Workbook 70

当前：**授权 Workbook 70 打开 reserved-blind development 评估。不授权 15D WLS。不打开 `00800_00849`。**

---

## 7. 有效作业 `1104860` / `1104861`（计入训练）

身份容差 `1.0e-5`；`L_corrected` 同一次 forward 核对 `1e-12`；`request_disk = 2000000`。科学合同相对 68B 未改。提交命令：`python3 scripts/submit_relative_route_v4_head_only_condor.py --submit`。Schedd `bigbird24.cern.ch`。

| 臂 | Cluster | 节点 / GPU | Execute | 最后 epoch loss |
| --- | ---: | --- | ---: | ---: |
| Arm 1 Absolute Control | **1104860** | `b9g57n6407` A100-PCIE-40GB | 15842 s | 0.079256 |
| Arm 2 Relative Primary | **1104861** | `b9pgpun014` H100 NVL | 7526 s | 0.071784 |

两臂 Condor **Normal termination return 0**。日志：`outputs/mc24_four_station_relative_route_v4_head_only/condor/logs/arm{1,2}.110486{0,1}.*`。  
Init：7 samples / 6 sources / 5040 graphs / 92 frozen tensors / trainable=172513 / frozen=614947 / zero-init `delta=0`。  
Worker stdout HEAD 启动时为 `7234deb`；`environment.json` 记录 `git_commit=4d1484a`（作业期间 68B/69 接线已 commit）与 `git_dirty=true`（仅 `README.md` / `README_cn.md`）。训练代码本身在 `4d1484a`，含 `1e-5` 身份容差。

禁止 resume `9254670` / `9254671` / `1104565` / `1104567`。

---

## 8. Checkpoint 与配对合同（交叉验证）

Canonical freeze 文件是 `checkpoint_last.pt`（与 `relative_route_v4_head_only.pt` 张量相同，zip pickle 名不同导致 SHA 不同）。

| 项 | Arm 1 | Arm 2 |
| --- | --- | --- |
| `checkpoint_last.pt` SHA256 | `e8a6d6c6e26545de91d9ded59c7b4f9b7840de2e9c94b88d56e1857d6aebd3c8` | `a52037ead07555ef937a7e02b65adc9526b509e765341e8af31992552f1fdd3a` |
| schema | `faser-relative-route-v4-head-only-v1` | 同左 |
| epochs | 30 unique，1–30 | 同左 |
| `early_stopping` | false | false |
| `checkpoint_selection` | `last_completed_epoch_of_fixed_30_epoch_budget` | 同左 |
| trainable / frozen | 172513 / 614947 | 同左 |
| `use_relative_route_representation` | false | true |
| `complete_route_logit` | `L_edge_W64_plus_delta_route_logit` | 同左 |
| post-train `max(|edge_v4-edge_w64|)` | `3.814697265625e-06` ≤ `1e-5` | 同左 |
| trainable-replica frozen hash | `ccebe2affeb2db1e3d815fa3e98de100e2c1f4a9dbd7b530f168a3034bdc7348` | 同左（训练前后不变） |
| W64 replica hash | `917faa8309e5e324a1044827f39f504768e65786db7c5f3338937b24b290b6b3` | 同左（训练前后不变） |
| parent W64 SHA256 | `0c3a28704cc01151fab7ac943e41338e859b5dde1502c0b10e2f12ada04e6236` | 同左 |

`run_contract.json` 除 `arm` / `arm_name` / `config_file` / `config_sha256` 外一致。  
`development_validation_loaded=final_blind_loaded=test_events_loaded=false`。  
`continue_to_15d_relative_wls=false`。

Train-only `delta_route_logit` 非零（count=597781）：

| 臂 | mean | max | min |
| --- | ---: | ---: | ---: |
| Arm 1 | −128.63 | 0.642 | −1121.07 |
| Arm 2 | −138.70 | 0.699 | −1319.54 |

这只证明 head 在 train C/D 目标上移动，**不是** development 门。Packing_D 与 Dustbin_C 在 history 上数值相同（dustbin 不是最强对手时同一 rival）。

独立核对（本会话）：`frozen_workbook64` vs Arm0 maxabs=0（103 tensors）；trainable-frozen vs Arm0 maxabs=0（92）；Arm1 vs Arm2 frozen maxabs=0；两臂 head 不同 maxabs=0.379。

---

## 9. C/D 评估接线（本条目收尾，供 Workbook 70 使用）

生产 enumerator 现在把 `complete_route_scores` 送进 `_route_hypotheses`：

- `training/solver_hard_negative_audit.py` `production_hypotheses(..., complete_route_scores=None)`
- `training/route_operating_audit.py` `audit_event_truth_chains` / `assign_and_audit_payload`
- 4-station 竞争者若 `winner.complete_route_score` 已设置，`winner_physical = winner.utility - 1e-9*(n-1)`，禁止用相邻边重建
- truth packing utility：score map 缺 key 则 `RuntimeError`（“V4 score map and adjacent candidate graph disagree”）；candidate retained 时 `packing_utility = solver_log_odds(query_score) + 4 * unmatched_penalty`

片段拓扑仍只用冻结边。Arm 0 评估必须继续传 `complete_route_scores=None`。

---

## 10. 授权

```text
workbook69_paired_training = completed
training_authorized = false
development_eval_authorized = true_for_workbook70_reserved_blind_arm012
continue_to_15d_relative_wls = false
sealed_test_accessed = false
new_final_blind_content_accessed = false
development_accessed = false   # 本条目仍未打开 00350_00399；Workbook 70 才打开
resume_9254670_9254671_1104565_1104567 = false
```

Workbook 70 必须：

1. 冻结上述两臂 `checkpoint_last.pt` 与 W64 `0c3a287…`；
2. 同一 reserved-blind overlay `00350_00399` 上评估 Arm 0 / Arm 1 / Arm 2；
3. 冻结 identity Platt 与 packing `0.001 / -1.0`；Arm 0 不注入 route scores，Arm 1/2 注入 `L_corrected` replace；
4. 复放 Workbook 64 production gates 与 Workbook 65 C/D 计数；
5. 不重训、不改 OP / loss / architecture；不打开 `00800_00849`；不授权 15D WLS。
