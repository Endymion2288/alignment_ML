# 条目 56 — 四站 V2 gauge-consistent route training

日期：2026-08-22  
分支：`4station`  
状态：transfer 闸已打开并失败。`continue_to_15d_relative_wls: false`。failure class = `association_domain_shift`；下一步 = `diagnose_packing_route_competition_objective`。objective / threshold / penalty / calibration 保持冻结。密封 test 未打开。  
密封 test：永久关闭。

## 本条目允许做什么

只做一个同架构 V2 的 targeted training-objective control。不增加
Transformer 层数、hidden size、feature，不改 candidate builder，不改
unit-capacity solver，不复活 V3 exact structured-margin。不在已经看过的
development validation 上选另一套 objective，也不再扫 unmatched penalty。

## 科学假设（预注册）

当前 V2 的 edge ranking 基本可用，但同一 `ΔT_ij` 在不同 common SE(3)
gauge 下存在 absolute score-scale drift；solver 中完整四站 truth route
会被 truth-consistent 2/3-station fragment 或 mixed route 竞争。因此保持
原 V2 edge BCE / route-query 主体，只加两个低容量辅助目标：

1. **gauge-twin consistency**：同一 source event、同一 origin edge/route、
   同一 `ΔT_ij` 的 chart/twin 配对，约束 raw adjacent logits 与 complete-route
   packing utility。无 provenance 对应的 fake 不强行一一相等。
2. **packing route-competition**：与生产相同的 adjacent-route 定义，要求
   truth complete 4-station packing utility 对最强 solver-feasible
   competitor 保持固定 margin。包含条目 55 已确认会抢路由的 3-station
   suffix/prefix、mixed route 与 endpoint-conflict route。不是 V3 全局
   loss-augmented assignment。

## 冻结的 operating convention（训练前）

代码审计确认：生产 packing 使用 clipped log-odds，而历史 validation Platt
会把训练 utility 与推理 utility 拉开。本条目在训练前冻结 identity Platt
（slope=1、intercept=0），使 `sigmoid(raw logit)` 就是推理分数。

| 量 | 冻结值 |
| --- | --- |
| score stream | raw sigmoid + identity Platt |
| `0→1 / 1→2 / 2→3` threshold | 0.001 |
| unmatched_penalty | −1.0 |
| dustbin utility | 0 |
| complete-route query 注入 packing | 否 |
| 第二套事后 Platt | 禁止 |

合同：`configs/physical_four_station_gauge_consistent_route_training.yaml`

## Loss 权重（不可在 validation 上 grid search）

算法预注册：在**第一次 optimizer step 之前**，对 train-only 做一次
no-grad 前向，用各辅助项均值把权重归一化到与 edge loss 同量级，再 clip
到 `[0.05, 20]`。算完立刻写入
`outputs/mc24_four_station_gauge_consistent_route_v1/checkpoint/aux_loss_weight_contract.json`。
之后禁止改。

| 项 | 规则 |
| --- | --- |
| 历史 V2 四项权重 | 保持 1.0 / 1.0 / 0.25 / 0.25 |
| packing margin | 1.0 |
| checkpoint | 固定 30 epoch 的最后一个；不用 development validation |
| 设备 | CUDA；不可用则失败 |

## Development validation 降级

从本条目起，下列源 `development_validation_only=true`：

- `mc24_100047_00050_00099`
- `mc24_100048_00050_00099`

可以保留作历史 mechanism comparison，**不能**用来宣称新 objective 通过
production association gate，也**不能**继续调 loss / margin / threshold /
penalty / calibration。

## Transfer-validation 源（最终闸）

审计脚本：`scripts/audit_four_station_transfer_sources.py`  
源合同：`configs/physical_curriculum_four_station_relative_transfer_sources.yaml`

| id | 电荷 | 角色 |
| --- | --- | --- |
| `mc24_100047_00300_00349` | μ− | transfer_validation |
| `mc24_100048_00300_00349` | μ+ | transfer_validation |

这两源未出现在条目 48–55 train / development validation、expanded
trainval、calibration-modes transfer、identifiability 列表或密封
`100116/100117`。curriculum 仍是 15D relative + left-SE(3) twin，新 seed
`20260822`，幅度 ≤0.5 mm / ≤5 mrad，含 nominal、两个 random relative、一个
hard S3/2→3，以及每个 configuration 的 gauge twin。不增加 FD probe，不扩大
物理问题。

打开 transfer 结果之后，禁止修改 objective、margin、loss weight、
threshold、penalty 或 calibration。

## 三层闸（只在新 transfer set 上）

1. raw candidate truth-chain recall ≥ 0.90。
2. association：nominal purity ≥ 0.95、fake ≤ 0.05；所有非 nominal 相对该
   bank nominal：efficiency drop ≤ 0.10、purity drop ≤ 0.05、fake increase
   ≤ 0.05。单独报告 0→1 / 1→2 / 2→3 与 S3 participation。
3. gauge：沿用条目 53 预注册 twin route-metric tolerance；并且相对同一
   transfer overlay 上的条目 54 retrained V2，origin-matched raw-logit 与
   truth-route utility 的 median |shift| 必须严格收缩且比值 ≤ 0.50。

若完整通过：冻结 checkpoint / identity calibration / 0.001/−1.0，第一次
打开已 truth-selected 的 15D route-selected `ΔT_ij` WLS。alignment capture
继续使用条目 49/50 已冻结的 S0/S3 chart、rank、condition 与 `ΔT_ij`
tolerance，绝不能重调。

若 association 仍失败但 gauge score drift 已消失：下一步诊断
route-competition objective。  
若 gauge twin 的 raw-logit drift 仍明显：这才是讨论
architecture-level relative-geometry representation 的证据。此前不增加
模型复杂度。

## 运行

```bash
source scripts/setup_environment.sh ml
bash scripts/run_four_station_gauge_consistent_training.sh audit-sources
bash scripts/run_four_station_gauge_consistent_training.sh prepare-transfer
bash scripts/run_four_station_gauge_consistent_training.sh submit-transfer
bash scripts/run_four_station_gauge_consistent_training.sh train
# Condor 完成后再 assemble-transfer → overlay-transfer → infer-transfer
# → score-scale → assess
```

训练复用现有
`outputs/mc24_four_station_relative_association_retrain_v1` 的 train overlay，
不新增 Athena production。Transfer bank 才新增物理生产。

## 运行中 provenance（不是闸）

- Transfer source audit：`outputs/mc24_four_station_gauge_consistent_route_v1/transfer_source_audit.json`（`ok` / unused vs 48–55）
- Transfer iteration：`outputs/mc24_four_station_relative_transfer_validation_v1/`，`transfer_validation_only: true`，`allowed_splits: [validation]`，源 `mc24_100047_00300_00349` + `mc24_100048_00300_00349`，7 点 × 2，seed `20260822`
- Condor：cluster `1000445`，2/2 正常结束（return 0），`schedd_mode: eossubmit`，`test_data_accessed: false`
- Transfer overlay：`outputs/mc24_four_station_relative_transfer_validation_v1/overlay_synthetic_v1/`，7 payload，validation-only，train/test 空
- 训练：lxplus902 的 T4 被其他用户占用。Condor GPU cluster `1000796` 因 7.5+ 槽全忙已 `condor_rm`。30 epoch 在 **lxplus901 Tesla T4** 上跑完。

## 结果

合同未改。数字只记录。15 维 WLS 未打开。

### 训练

- 节点：lxplus901 Tesla T4；seed `20260822`；最后第 30 epoch
- checkpoint：`outputs/mc24_four_station_gauge_consistent_route_v1/checkpoint/route_aware_transformer_v2.pt`
- sha256：`0e2ffe171e7cbbfd8ced426c9ce34759ffd6d0a2d6b26216df5814d48337cd80`
- 权重合同（optimizer step 之前写入）：`gauge_twin_consistency_weight = 20.0`（clip 上限；第一阶段 nominal-only 没有 twin，`gauge_twin_mean = 0`），`packing_route_competition_weight = 0.07061055340401011`
- identity Platt 与 0.001 / −1.0 未改
- 末 epoch packing-competition train loss ≈ 0.935；gauge-twin train loss ≈ 0.0022（第二阶段 720 pairs）

### Transfer 三层闸

决策：`outputs/mc24_four_station_gauge_consistent_route_v1/transfer_gate_decision.json`

| 层 | 结果 |
| --- | --- |
| 1 raw complete-chain recall | **过**：全部 7 payload = 1.0；adjacent truth-edge recall 均为 1.0 |
| 2 association | **不过**：7/7 payload `selected_routes = 0`，efficiency = 0，purity/fake = null，S3 endpoints = 0 |
| 3 score-scale vs 条目 54 | **过**：median \|Δ raw logit\| 0.0208 / 0.6302 = 0.033；median \|Δ truth-route utility\| 0.0461 / 1.3492 = 0.034 |
| 3 twin route-metric | **不过**：零选中轨道，purity/fake 无法定义 |

同一 overlay 上的条目 54 对照仍能选出轨道（nominal efficiency 0.776、purity 0.972、fake 0.103；非 nominal efficiency 0.654–0.745）。该对照在本 transfer bank 上自己也未过 vs-nominal association 稳定闸，但不是空选择。

本控制的 ranking 仍在（nominal ROC-AUC 0.973、AP 0.893），并且 `score_retained_complete_truth_chains = 483`（truth 边过了 0.001）。空选择因此不是 candidate graph 丢边，而是 **unit-capacity packing 相对 dustbin 0 一个 route 都没留下**。

### 冻结与下一步

- `continue_to_15d_relative_wls: false`
- 不改 threshold、unmatched penalty、calibration、margin、loss weight
- 不增加模型复杂度
- 下一步只诊断 packing-competition objective（以及它与 dustbin / clipped log-odds 的相对尺度），不是 architecture
