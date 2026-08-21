# 2026-08-20 (44) station ↔ IFT C_dx hierarchical closure：冻结 FASER hierarchical alignment V1

## 任务

条目 36 已冻结 station 5 个径迹约束 DoF + survey `dz`；条目 39/40 已冻结 IFT
唯一内部自由度 `C_dx=(dx_L0-dx_L2)/2`；条目 42/43 否定 `C_rx` 进入层级主线。
本条目验证这两层已经分别稳定闭合的自由度，能否在**同一真实 misalignment**
中按层级顺序工作，并据此冻结 **FASER hierarchical alignment V1**。

冻结不变：mode-0、V2 checkpoint/calibration/route policy、
`physical_edge_deduplicated`、station-level `dx/dy/rx/ry/rz`、`dz=5 mm`
survey prior、`C_dx` 1-D capture、sealed test。不重训模型、不调 χ²、不引入
`C_rx` / relative ry / layer dy/rz / layer1 / module。

禁止把 station 5-DoF 与 `C_dx` 放进同一个六参数 Newton solve。另一层不是
nuisance column。Remaining **不是** reference-linearized `proposed_next`（那会把
未浮动层级写成 0）。

主顺序：

```
station 5-DoF estimate
  → 写 station payload，C_dx 保持当前真值
  → /Tracker/Align → SegmentFitRefit → SegmentsRefit → NtupleDumper → Acts(mode 0)
  → frozen-V2 reassociate
  → 冻结新的 station geometry
  → 仅 C_dx 1-D estimate
  → 写 L0=+C_dx、L1=0、L2=-C_dx
  → 再次 real refit / reassociate
  → 必须真正生产第二步后的 detector geometry 并做最终 reassociation
```

反向 `C_dx → real refit → station 5-DoF` 是严格 control。train route-selected
决定下一步共享 payload；validation 只评价。

### 准入

若 `station→C_dx` 在 train 与 validation 上都使 station 五参数和 `C_dx` 同时进入
各自冻结 capture、最终 residual 显著下降、station payload 在 layer step 后保持
稳定，即使反向较差，也将该顺序冻结为物理层级一致的 canonical algorithm。
若两种顺序都收敛到一致 detector-element transforms，则作为更强的
hierarchy-independence 证据。通过后：

**FASER hierarchical alignment V1 = station 5 track-constrained DoF +
survey-constrained dz + IFT 1-D relative-dx internal deformation。**

随后结束当前自由度扩展阶段。下一阶段优先更大统计或真实数据 transfer，
而不是继续增加 layer/module 参数。

## 联合注入（少量真正 joint points）

原 10 train + 8 validation xAOD。seed **20260821**（不同于 5-DoF 的 20260819
与 contrast-2d 的 20260820）。station 五自由度在已验证 local regime 随机抽取；
`|C_dx|≤0.12 mm` 独立抽取；station `dz≡0`；`C_rx≡0`。

| 点 | 角色 |
| --- | --- |
| `iteration_00_reference` | 全零 / FD 锚 |
| 6 station 参数 × ± | 轴向 FD（含 survey `dz`） |
| `C_dx` × ± | 轴向 FD，步长 0.10 mm |
| `iteration_00_start` | 主联合起点，station severity 0.12 |
| `iteration_00_heldout_00` | 完全 held-out joint point，station severity 0.14 |

Jacobian 在零点线性化；观测打在联合注入 / remaining geometry 上。
17 点 × 18 源 = 306 次真实 refit。

采样值（seed 20260821）：

| 点 | station 5-DoF (dx, dy, rx, ry, rz) | `C_dx` / mm | severity |
| --- | --- | --- | --- |
| `iteration_00_start` | +0.430 mm, −0.101 mm, +4.56 mrad, −0.342 mrad, +1.66 mrad | **−0.0917** | 0.12 |
| `iteration_00_heldout_00` | −0.005 mm, +0.345 mm, +3.62 mrad, +5.05 mrad, −3.86 mrad | **+0.0710** | 0.14 |

两起点 `C_dx` 异号，station `dz≡0`，layer1 与 `C_rx` 全零。
`heldout_00` 不参与 FD、operating-point 或 remaining 选择。

## 生产状态

- bank：`outputs/mc24_ift_hierarchical_v1_iteration00_trainval_physical_v1/`
- 源配置：`configs/physical_curriculum_ift_hierarchical_v1_trainval.yaml`
- 模板：`configs/physical_refit_ift_hierarchical_v1_iteration.yaml`
- Condor **cluster 999304**（bigbird24，flavour `tomorrow`，6000 MB）：18/18 job、
  17/17 点完成，0 个 `failure.json`。两处截断 `tracklets.root` 已从完整
  `enhanced_tracklets.root` 本地 convert，**未**重提 cluster。
- 物理 corpus：`physical_corpus_manifest.json`（18 源 × 17 点）。
- synthetics：`outputs/mc24_ift_hierarchical_v1_iteration00_synthetic_trainval_v1/`
  （34 sample，train/validation × 17 payload）。overlay seed
  `alignment_iteration_shared_across_payloads`。
- 冻结 V2 只在 `iteration_00_reference` 上 reassociate：**已完成**
  （train 2823 routes / 2187 complete / 7606 edges，与 contrast-2d 参考点一致）。
- test 未打开。validation 不参与 operating-point / capture / remaining 选择。
- **不冻结 V1**：主顺序两层均未 capture；见下方准入判定。

### Truth-selected 泄漏①（Jacobian 在零点，观测在联合注入）

station 一步：train/validation × start/heldout **全部** `capture_success=false`。
`dx` 误差 / 注入 `C_dx` ≈ **−59**，`ry` 误差 / 注入 `C_dx` ≈ **−32**，跨点跨 split
稳定，说明 station 列在吸收真 `C_dx`，不是单源噪声。train start 残差降到
post/pre ≈ 0.11，但 remaining station severity 从 0.12 升到 1.09——参数被写错。
validation start 残差升到 post/pre ≈ 1.70。

`C_dx` 一步（station 保持注入真值，layer payload 稳定）：start 在 train 与
validation 上都 capture（remaining `C_dx` ≈ 0）；heldout（station `ry≈5 mrad`）
上 residual 不降，capture 失败。这支持主顺序必须先 station，但 station-first
的零点线性化已被 `C_dx` 严重偏置。最终准入仍以真实 remaining refit 为准。

### Route-selected 泄漏①（anchor = reference，V2 选路固定，synthetics 重测边）

station 一步：train/validation × start/heldout **全部** `capture_success=false`。
train start `dx` 误差 / 注入 `C_dx` ≈ **−59.2**（与 truth-selected 同一耦合），
remaining station severity 0.12 → **1.09**，`C_dx` 保持注入真值，写入 `dz=0`。
残差 post/pre ≈ 0.085，但参数被写错——不能把 residual 下降当成闭合。

`C_dx` 一步：station remaining 完全不动。train start engineering+statistical
通过，coverage 因本 fit σ 过小失败（误差 0.006 mm vs 3σ_fit=0.0013 mm）；
validation start/heldout capture 通过。train 选择 remaining，不根据
validation 改 payload。

CLI 打印曾因缺少 `station_transforms_unchanged` 在写完 JSON 后 KeyError；
artifact 完整。已在 `run_route_selected_multidof_update.py` 补上该键与
`.get()` 打印，**未**重跑这 8 个 update。

### Remaining 真实链（第一步后）

| 顺序 | remaining root | cluster | 状态 |
| --- | --- | --- | --- |
| 主顺序 station→C_dx 第一步后 | `.../station_then_cdx_remaining_after_station_v1/` | **999493** | 18×2 完成，0 `failure.json`，ROOT `load_events` 144/144 通过 |
| 反向 C_dx→station 第一步后 | `.../cdx_then_station_remaining_after_cdx_v1/` | **999494** | 同上 |

未重提 cluster。Jacobian 仍用 iteration-00 轴向 FD；anchor 仍是冻结
`iteration_00_reference` V2；target 只读 remaining synthetics
（各 4 sample，未在 remaining 上重训/重跑 V2）。

### 第二步（新物理 geometry，不是代数扣除）

**主顺序：在 remaining-after-station 上只浮 `C_dx`。** station leftover
`|dx|≈5.4 mm` 主导残差。truth 与 route-selected 都把 `C_dx` 收回约 **0**，
post/pre ≈ **1.00**，station payload **稳定**（max delta 0）。`C_dx` 未 capture。
train remaining 与已 refit 的第一步 remaining 只差 `ΔC_dx≈0.00013 mm`，
**不**再提交一份重复 36 次真实链。主顺序终点 geometry 就是 999493 已经生产的
remaining-after-station；route-selected 在该 candidate graph 上重测边即最终
reassociation。

**反向：在 remaining-after-cdx 上只浮 station 5-DoF。** `C_dx` leftover
≈0.006 mm 仍通过同一耦合偏置 `dx`（误差/`C_dx` 仍 ≈ **−59**）。train start
route-selected post/pre ≈ 0.46，severity 0.12→0.072，但 `dx` engineering 失败；
validation start post/pre ≈ **1.85**（残差变差）。station 未 capture。
这是**新** remaining（`dx≈0.36 mm`，`C_dx≈0.006 mm`）。第二步真实链
**cluster 999500** 已完成：18×2、0 `failure.json`、72 个 ROOT 深读通过。
synthetics 4 sample。在该终点上用冻结 reference V2 选路、只读 remaining
candidate graph 做 `C_dx` leftover lookup（不是再写一层 remaining）：
train/validation × start/heldout 全部 `capture_success=false`，回收 `C_dx≈0`，
remaining `C_dx` 仍为 0.006–0.008 mm，post/pre ≈ 0.98–1.00，station delta 0。
15 分钟 Condor 轮询已停止。

### 准入判定：**不冻结** hierarchical alignment V1

主顺序 station→`C_dx` 在 train 与 validation 上 **都没有** 让两层同时进入各自
冻结 capture。第一步把 station 写得更远；第二步在该 geometry 上测不到 `C_dx`。
残差下降出现在错误的参数方向上，不能当闭合。反向 control 第一步几乎收回
`C_dx`，第二步仍被 leftover `C_dx` 以同一 −59 耦合偏置 `dx`，validation 残差
不降。两种顺序也到不了一致 detector-element transforms。

不引入 `C_rx` / relative ry / layer dy/rz / layer1 / module 来补偿。不重调
V2、mode-0、route、χ²、capture。自由度扩展阶段在此停止。999500 完成后只做
反向终点 geometry 的审计与最终 reassociation 记录，不据此改判定。

条目 45 已用现有 Jacobian 做完泄漏算子 / 预算反演 / nuisance 投影终审：
`A_dx≈−59`、`A_ry≈−32` 是稳定几何；`dx` 工程 0.1 mm 要求 `|C_dx|≲1.7 µm`，
而已注册 `σ(C_dx)=6.91 µm`。投影不能救出可用的 station `dx`。**正式关闭
V1 rescue**：两层各自可测，但不可同时作为层级参数。见
`workbook/2026-08-21_45_station_C_dx_leakage_identifiability.md`。

回归：`tests/test_hierarchical_v1.py` 与 layer / contrast-2d / sequential /
5-DoF / route-selected layer / physical Jacobian 测试共 66 项通过。

代码：`alignment/hierarchical_v1.py`、`alignment/hierarchical_v1_sampling.py`、
`scripts/prepare_hierarchical_v1_iteration.py`、remaining collect/prepare、
`scripts/run_hierarchical_v1_truth_closure.py`。`ift_layer_hierarchy` plan
builder 仅在 `fit_basis: hierarchical_v1` 时允许 station + `C_dx` 与 survey
`dz`；无该标记时仍禁止混合。

## 银行完成后的分析顺序（不提前宣布闭合）

1. 完成审计 → `build_6dof_pilot_physical_corpus.py` → 冻结 V2 只在
   `iteration_00_reference` 上 reassociate；target/FD 用 synthetics +
   `anchor_selected_field_edge` + `physical_edge_deduplicated`。
2. **主顺序 station→C_dx**
   - truth + route-selected：`--only-parameters` 五个自由参数 + `ift_dz_mm`，
     `--prior-sigma ift_dz_mm:5.0`，capture =
     `outputs/mc24_ift_5dof_survey_dz_capture_criteria_train_v1/capture_criteria.json`。
     **不**把 `C_dx` 放进该 solve。量化泄漏①（dx/ry 是否吸收真 `C_dx`）。
   - train route-selected remaining：station leftover，`C_dx` 仍为注入真值，
     写入 `dz=0`。validation 只记录。
   - remaining bank 真实 refit（36 job）→ 再 1-D `C_dx`
     `--only-parameters C_dx`，capture =
     `outputs/mc24_ift_layer_contrast_2d_capture_criteria_train_v1/capture_criteria.json`
     （只评 `C_dx` 行）。station 固定。量化泄漏②。
   - 写第二 remaining layer payload → 再一次真实 refit + 最终 V2 reassociation。
3. **反向 control** `C_dx→station`：同样 freeze 规则，另写 remaining bank。
4. 仅当 station→C_dx 在 train **与** validation 上双层 capture、residual 下降、
   station payload 在 layer step 后稳定，才冻结 V1。

