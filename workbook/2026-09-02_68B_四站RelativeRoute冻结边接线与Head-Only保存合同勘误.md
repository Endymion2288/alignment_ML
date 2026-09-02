# 条目 68B — 四站 RelativeRoute 冻结边接线与 Head-Only 保存合同勘误

日期：2026-09-02  
分支：`4station`  
状态：**训练合同勘误完成。冻结 Workbook-64 生产边接线、additive `L_corrected` 与 checkpoint 保存接口已用代码/测试交叉验证。Workbook 69 两臂 head-only 训练重新授权，但必须用新接线从头训 30 epoch，禁止 resume `9254670`/`9254671`。**  
物理闭环合同：`continue_to_15d_relative_wls = false` 保持冻结  
密封 Test：永久封存，`sealed_test_accessed = false`  
新 Final Blind 内容访问：`new_final_blind_content_accessed = false`  
训练授权状态：`training_authorized = true_for_workbook69_head_only_primary_and_control_training`（仅授权 Workbook 69 按本条目冻结边合同重训；本条目不产生 checkpoint）

---

## 1. 审计结论：Workbook 65–69 哪些真实完成

不以 workbook 中的 PASS 文字为准，而以 git、代码、产物、Condor 日志交叉核对。

| 条目 | 文档声明 | 证据 | 判定 |
| --- | --- | --- | --- |
| 64 | 六源 diversity 训完，reserved-blind 失败 | checkpoint SHA256 `0c3a28704cc01151fab7ac943e41338e859b5dde1502c0b10e2f12ada04e6236` 文件存在且哈希一致；`checkpoint_freeze.json`、blind association、gate 数字齐全 | **真实完成（负结果保留）** |
| 65 | C+D 定位，A/B/实现 bug 排除 | `outputs/mc24_four_station_blind_failure_localization_v1/{decision,failure_summary,blind_failure_rows,nominal_vs_hard_alignment}.json*` 存在；`decision.json` 给出 C=200、D=170、A=B=E=0，`gate_replay_all_ok=true` | **真实完成** |
| 66 | 架构预注册，禁止训练 | 仅 workbook + 合同文本；无训练产物 | **真实完成（设计条目）** |
| 67 | RelativeRoute V4 实现 + pytest | `models/route_transformer.py`、`tests/test_relative_route_v4.py` 在 commit `73851f2` | **真实完成（实现条目）** |
| 68 | 三臂合同预注册，未训练 | YAML + dry-run 脚本存在；无 checkpoint | **真实完成（预注册）** |
| 68A | tie-break `1e-9`、solver-aware 梯度审计 | `_CONTINUATION_TIE_BREAK = 1.0e-9` 源码确认；`split_graph_route_logits` 把 `route_logits` 送进 packing/dustbin/gauge；`scripts/audit_route_head_solver_gradients.py` 与 Test 16/17 存在 | **梯度路由与勘误真实完成；但 additive 边接线未冻结（见下）** |
| 69 | Arm 1 / Arm 2 head-only 训练 | 无 workbook 69。Condor `9254670`/`9254671` 于 2026-09-01 提交，两臂都跑完 30 epoch，随后 `TypeError: RouteAwareTransformerArtifact.__init__() got an unexpected keyword argument 'station_path'`，exit 1。`absolute_control/` 与 `relative_primary/` 为空，无 checkpoint | **未完成。30 epoch 权重未落盘，禁止当完成，禁止 resume** |
| 70 | development evaluation | 无 workbook、无 evaluation 产物 | **未开始，且本条目不授权打开 development** |

当前工作区位于 `4station`，HEAD `7234deb`，相对 `origin/4station` ahead 1。本条目在该提交之后补勘误。

---

## 2. 68A 之后新发现的硬错误（必须勘误，不得现场 rescue）

### 2.1 Condor `9254670` / `9254671` 失败原因

| 臂 | Cluster | 执行节点 | GPU | 30 epoch | 保存 | return |
| --- | ---: | --- | --- | --- | --- | ---: |
| Arm 1 Absolute Control | 9254670 | `b9pgpun203.cern.ch` | NVIDIA H100L-2-24C MIG 2g.24gb | 是，最后 epoch total 0.0961 | 否 | 1 |
| Arm 2 Relative Primary | 9254671 | `b9pgpun204.cern.ch` | NVIDIA H100L-2-24C MIG 2g.24gb | 是，最后 epoch total 0.0819 | 否 | 1 |

日志：`/afs/cern.ch/user/x/xcheng/condor_logs/relative_route_v4/arm{1,2}.925467{0,1}.*`。  
失败栈在 `scripts/train_relative_route_v4_head_only.py` 用错误字段构造 `RouteAwareTransformerArtifact`（`station_path` / `feature_names`），真实 dataclass 需要 `node_feature_names`、`edge_feature_names`、`all_station_pairs`、`output_station_pairs`、`context_mode`、`training_summary`。这是实现 bug，不是 EOS/CUDA/preemption。**禁止 exact resume。** 那两次作业没有可读 checkpoint，不算训练完成。

两臂日志中的 train-only 数字（C/D loss 下降、frozen hash 相同、zero-init `delta=0`）只证明循环曾经跑过，不能替代落盘 artifact。

### 2.2 Additive 前向把可训练 `delta` 送进了冻结边

Workbook-64 生产边是

```text
edge_logits = base_edge_logits + route_edge_correction(base, mean(route_query), log1p(counts))
```

W64 checkpoint 的 `route_edge_correction` **不是零**：

- `route_edge_correction.0.weight` absmax ≈ 1.018，L2 ≈ 4.93
- `route_edge_correction.2.weight` absmax ≈ 0.343，L2 ≈ 0.646

历史 V2 用 route-query 输出驱动该校正。68/68A 把 `route_score` 改成 `delta_route_logit` 后，若仍把 `delta`（或 `L_edge+delta`）送进 `route_edge_correction`，生产边会随 head 训练而动，违反「冻结 Workbook-64 edge/backbone」。

同时 68A 的 `L_corrected = sum(base_edge_logits) + delta` 用的是 **uncorrected base**，不是生产 `edge_logits`。solver 若按合同消费 `complete_route_scores = σ(L_corrected)`，其 `L_edge` 必须是 **W64 生产边** 的三条相邻 log-odds 之和。

这不是科学合同重解释，是接线错误。68A 的 C/D 梯度「对某个 route logit 非零」仍然成立，但不能声称那个 logit 就是冻结生产 `L_edge_W64 + delta`。

### 2.3 默认旗标必须保持历史 V2 可加载

Workbook-64 `model_config` **没有** `use_additive_route_correction`。若默认改成 `True`，`load_route_aware_transformer_artifact` 会把冻结 checkpoint 解释成 additive 路径，生产边立即改变。默认必须保持 `False`。

---

## 3. 冻结边接线合同（取代 68A 的单模块 additive 前向）

```text
Arm 0  冻结 Workbook-64 前向（historical V2：route-query → route_edge_correction）
Arm 1/2 包装：
  frozen_w64.eval()  →  edge_logits_W64,  L_edge_W64 = sum_{k=01,12,23} edge_logits_W64
  trainable_head     →  delta_route_logit  （route_score 零初始化）
  L_corrected        =  L_edge_W64 + delta_route_logit
  complete_route_scores = σ(L_corrected)
  2/3 站 fragment    只用 edge_logits_W64，永不看 delta
```

允许的唯一实验差：Arm 1 `[s0,s1,s2,s3]`，Arm 2 `[s0,s1-s0,s2-s0,s3-s0]`。  
其余：同一 W64 checkpoint、同一六源 overlay、同一 15D curriculum、同一 AdamW/`2e-4`/`32`/`20260822`/30 epoch、同一 solver-aware 权重、同一 last-epoch 规则。

checkpoint schema：`faser-relative-route-v4-head-only-v1`，同时保存 `frozen_workbook64_state_dict` 与 trainable head，避免把 V4 误加载成 V2。

---

## 4. 代码与测试证据（本条目执行）

修改（最小必要）：

- `models/route_transformer.py`：历史默认 `use_additive_route_correction=False`；additive 路径不再把 `delta` 送进 `route_edge_correction`；新增 `RelativeRouteV4Inference`。
- `training/route_aware_transformer.py`：`save/load_relative_route_v4_head_only_artifact`。
- `scripts/train_relative_route_v4_head_only.py`：加载完整 W64 replica + 标准器；零初始化 `route_score`；训练前后核对边与 W64 逐元素一致；用正确 artifact 保存。
- `scripts/audit_route_head_solver_gradients.py` / `scripts/dry_run_relative_route_v4.py`：改走 wrapper。
- `scripts/submit_relative_route_v4_head_only_condor.py`：复用 workbook-59 Condor 约定（`module load lxbatch/eossubmit && myschedd out && condor_submit`），日志写入项目 `outputs/.../condor/logs/`。
- `tests/test_relative_route_v4.py`：Test 18–22。

验证（交互节点短测，未开 development / final blind / sealed test）：

```bash
source scripts/setup_environment.sh ml
pytest -q tests/test_relative_route_v4.py tests/test_route_aware_transformer.py \
  tests/test_route_assignment.py tests/test_blind_failure_localization.py
# 44 passed in 25.76s
python scripts/audit_route_head_solver_gradients.py
```

wrapper 上 Problem C/D 对 trainable head 的梯度范数均为 1.293328，复合目标 0.100827，冻结参数泄漏 0。历史默认 `use_additive_route_correction=False`。

68A 数字与负结果不删除，只在 68/68A 文末追加 erratum。

---

## 5. 数据与封存

- 训练源仍是 `configs/physical_curriculum_four_station_diversity_train_sources.yaml` 六源。
- development `mc24_100047_00350_00399` / `mc24_100048_00350_00399`：**Workbook 69 不得打开。**
- 新 final blind `00800_00849`：不打开 ROOT、不计数、不推理。
- sealed test：永久关闭。
- `continue_to_15d_relative_wls = false`。

---

## 6. 授权

```text
training_authorized = true_for_workbook69_head_only_primary_and_control_training
continue_to_15d_relative_wls = false
sealed_test_accessed = false
new_final_blind_content_accessed = false
resume_9254670_9254671 = false
```

Workbook 69 必须：

1. 用 Condor 分别提交 Arm 1 与 Arm 2，不用交互式 `lxplus-gpu` 长训；
2. 固定 30 epoch、last-epoch checkpoint、无 early stopping、无 seed/超参扫描、无 development 选点；
3. 训练成功不能只看 Condor return code，必须检查 checkpoint 可读、epoch=30、frozen W64 边哈希不变、两臂 frozen edge 与 Arm 0 一致。

---

## 7. Condor `1104565` / `1104567` 身份检查勘误（非科学合同变更）

Workbook 69 按本条目接线于 2026-09-02 15:26 提交 Arm 1=`1104565`、Arm 2=`1104567`（schedd `bigbird24`）。两臂都在 `b9pgpun204.cern.ch` 的 H100 MIG `2g.24gb` 上启动，科学侧 init **已通过**：

- 7 samples / 6 sources，5040 graphs
- 92 frozen tensors copied
- trainable=172,513 / frozen=614,947
- zero-init `max(|delta|)=0`
- W64 standardizer mean delta = 0

随后在训练循环前因 **两次独立 GPU 注意力前向的 float32 残差** 失败，return 1，**无 checkpoint**：

| 臂 | Cluster | `max(|edge_v4-edge_w64|)` | 当时阈值 | 结果 |
| --- | ---: | ---: | ---: | --- |
| Arm 1 | 1104565 | `9.5367431641e-07` | `1e-12` | 失败，epoch=0 |
| Arm 2 | 1104567 | `5.8114528656e-07` | `1e-12` | 失败，epoch=0 |

这不是边泄漏：`RelativeRouteV4Inference` 的 `edge_logits` 来自冻结 W64 replica 的独立 `eval()` 前向，与 trainable replica 的另一次注意力前向在 GPU float32 下不是逐 bit 相同。同一次 forward 上 `L_corrected = L_edge + delta` 仍应用 `1e-12`。边身份检查改为 `1.0e-5`，覆盖已观测 H100 MIG 残差（~`1e-6`），不放松“生产边等于冻结 W64 边”的科学声明。

另外，vanilla Condor 几乎不 transfer 输入，scratch 写 EOS，`RequestDisk=DiskUsage=3` 导致 Allocated Disk 仅 2048 KB。submit wrapper 现显式 `request_disk = 2000000`（KB）。

**禁止 resume `1104565`/`1104567`。** 这不是 EOS/CUDA/preemption；权重从未写入。Workbook 69 必须用当前源码从头重提。
