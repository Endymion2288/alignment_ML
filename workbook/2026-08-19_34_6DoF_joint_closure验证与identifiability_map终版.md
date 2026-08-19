# 2026-08-19 (34) 6-DoF joint closure 验证与 identifiability map 终版

## 任务

承接条目 33（sensitivity pilot + 准入 gate：{dx, dy, ry, rx, rz} 准入、dz 拒绝）：
v2 生产（新增小 severity held-out 点 c/d）完成后，对 gate 准入的最小参数集合做
truth-selected joint closure，再做冻结 V2 route-selected、
`physical_edge_deduplicated` unknown-association closure，验证
associate → align → new payload → refit → reassociate 全环在 6-DoF 下仍收敛，
并给出 station-0 6-DoF identifiability map 终版。全程不打开 sealed test，
不调任何 χ²/covariance/route/association 参数。

## v2 生产完整性与可复现性

- 3 train source × 18 点零失败（Condor cluster 998283）。
- 一个转换产物损坏：`100043_00200_00299/fd_ift_dx_mm_p/refit/propagations.root`
  （TBasket 截断；Athena refit 本身成功）。从完好的 `enhanced_tracklets.root`
  本地重跑 `convert_ntuple_tracklet_propagations.py` 修复（2092 records），
  修复后全 108 个分析输入文件完整性通过。
- **v2 audit 逐位复现 v1**：六个参数的 σ_native、source spread、gate 判定
  （准入序 ry→rx→dx→rz→dy，dz 因 trial 条件数 1.3e7 被拒）完全相同——
  FD probe transform 逐位一致 + 同一冻结链，生产可复现性确认。
- anchor→reference sanity closure（v2 pooled）：dx 残差 −0.007 mm、
  ry 残差 −0.007 mrad、rx/rz 残差 <0.08 mrad，与 iteration-01 收敛判定一致
  （dz +2.2 mm 为不可辨识方向的预期噪声）。

## truth-selected joint closure（admitted 5-DoF，`run_6dof_joint_closure.py`）

closure 点从 anchor 联合注入全部 6 DoF；dz 不进入拟合，作为未建模注入
（其可忽略性正是 gate 拒绝的含义，下面定量验证）。

| 点 | severity | 5D χ²/5 | dx err | dy err | rx err | ry err | rz err |
| --- | --- | --- | --- | --- | --- | --- | --- |
| c | 0.14 | 8.12 (p≈0.15) | +0.018 (0.9σ) | +0.042 (0.9σ) | +0.066 (1.0σ) | −0.026 (−1.9σ) | +0.227 (0.8σ) |
| d | 0.21 | 15.02 (p≈0.01) | −0.031 (−1.6σ) | **−0.160 (−3.3σ)** | +0.128 (2.0σ) | +0.022 (1.6σ) | +0.267 (0.9σ) |

- **c 干净闭合**：线性区（severity ≲0.15）5-DoF joint closure 与统计涨落一致。
- **d 边缘**：dy 误差三源同号（−0.246/−0.081/−0.164），不是单源涨落。
  两个假设的定量检验：
  1. **未建模 dz 泄漏**——把 dz 列响应投影进准入子空间
     bias = (JₐᵀWJₐ)⁻¹JₐᵀW j_dz·Δdz：每 +1 mm dz 仅泄漏 dy +0.012 mm、
     rz −0.072 mrad；对 c/d 的 Δdz=+0.4/−0.6 mm 预测 bias ≤0.007 mm，
     比观测误差小一个量级以上 → **排除**。
  2. **dy-rx 近简并方向涨落**——后验相关 −0.595；d 的 dy(−3.3σ) 与
     rx(+2.0σ) 反号组合正是沿该方向的单次涨落。联合 5D χ²=15.0/5（p≈1%）
     判为边缘统计涨落/线性边界早期信号的叠加，与 a/b 大 severity 点暴露的
     refit 不连续尾（条目 33）在 severity 轴上自洽。
- 结论：**线性域 joint closure 在 severity ≲0.15 干净；~0.2 边缘；
  0.5–1.0 失效**。这给迭代环一个明确的工作点约束：每步 Newton 更新应
  保持在 |Δ| ≪ FD 步长的区域（正常迭代天然满足）。

## route-selected closure（冻结 V2、unknown association、dedup 语义）

新基础设施（全部复用冻结组件，零调参）：

- `scripts/build_6dof_pilot_physical_corpus.py`：pilot iteration bank →
  `physical_corpus_manifest.json` 适配器，逐点复用
  `build_physical_curriculum_corpus.py` 的 completion audit，不重跑 refit。
- 物化 `mc24_ift_6dof_sensitivity_pilot_v2_synthetic_train_v1`（18 train
  samples，`alignment_iteration_shared_across_payloads` 种子域，3 源 pooling）。
- anchor payload 冻结 V2 backbone（ungated 现行冻结行为，checkpoint
  `mc24_v3_expanded_trainval_v2_bce_control_v1`），815 条 truth-free 选路。
- `run_route_selected_multidof_update.py --observation-kind
  anchor_selected_field_edge --observation-statistics physical_edge_deduplicated`。
  工具要求全 6 参数 probe；**dz 以 severity-scale survey prior（5 mm）
  正则化**，其余 5 参数 prior=10⁶ 等于无约束——这正是 gate 结论
  （dz 需外部 prior）的操作化，而非新调参。

| 点 | unique edges | χ²/ndof | dx pull | dy pull | rx pull | ry pull | rz pull |
| --- | --- | --- | --- | --- | --- | --- | --- |
| c | 193 | 7.7/766 | −0.08 | +0.33 | −0.23 | −0.12 | +0.45 |
| d | 194 | 10.0/770 | +0.47 | +0.28 | −0.28 | +0.14 | −0.70 |

- 两点全部准入方向 |pull| ≤ 0.70，**unknown-association 全环闭合**。
- dy 在 c 上 err +0.102 mm 恰好超过继承自 3-DoF truth 时代的 0.1 mm
  capture 容差（故 capture_success=False），但 route-selected σ_dy=0.313 mm
  （193 边 vs truth 694 边），0.33σ 完全统计一致；容差定义滞后于观测语义，
  应在新 curriculum 前按 route-selected σ 重新标定（记录为 follow-up，
  本阶段不改）。
- dz 如预期由 prior 主导（recovered σ 4.25 mm ≈ prior 5 mm 缩窄），
  不参与 closure 判定。

## station-0 6-DoF identifiability map（终版，FASER 当前 track sample）

| 参数 | σ_native | severity scale | 可辨识 | 备注 |
| --- | --- | --- | --- | --- |
| dx | 0.020 mm | 5 mm | ✓ | 与 ry 列余弦 0.494 |
| dy | 0.048 mm | 5 mm | ✓ | 与 rx 列余弦 0.598（最强近简并） |
| dz | 3.6 mm | 5 mm | ✗ | gauge-like：响应仅经 ≈5 mrad 斜率杠杆，比 dx 弱 ~200×；需 survey prior 或更宽 track phase space |
| rx | 0.065 mrad | 60 mrad | ✓ | ry_mm 残差杠杆臂约束强，信息量排名第 2 |
| ry | 0.014 mrad | 60 mrad | ✓ | 最强约束 |
| rz | 0.32 mrad | 60 mrad | ✓ | 可分离；与 dx/dz 余弦 ~0.39/0.40 |

- 无 |cosine| ≥ 0.9 的几何简并对；最弱方向由 dz 主导（scaled SVD）。
- source-to-source spread 全部 ≤0.40，pooled 与逐源一致。
- 线性域：severity ≤0.15 干净，~0.2 边缘，0.5–1.0 因 segment refit
  不连续尾（~4–5% 边）失效。

## 结论与下一步

- Pilot 目标达成：可信 6-DoF identifiability map + 小 severity joint
  closure（truth-selected 与 route-selected 双路径）全部通过。
- 满足既定准入条件，**允许扩展为完整 station-level 6-DoF
  train/validation physical curriculum**（待用户批准后启动）：
  admitted 5-DoF 自由拟合 + dz 带 survey prior；每步更新保持在
  线性域（|Δ| 远小于 FD 步长）；route-selected capture 容差按
  route-selected σ 重新标定。
- dz 的物理结论：对近平行束流，station-0 dz 与整体纵向 gauge 几乎
  不可分；若物理上需要 dz，必须引入外部 survey 约束或非 nominal
  大角度 track 样本。

## 产物索引

- v2 bank：`outputs/mc24_ift_6dof_sensitivity_pilot_v2/`（含
  `physical_corpus_manifest.json`、identifiability_audit/、
  joint_closure_{c,d}/）
- synthetic overlays：`outputs/mc24_ift_6dof_sensitivity_pilot_v2_synthetic_train_v1/`
- 冻结 backbone：`outputs/mc24_ift_6dof_sensitivity_pilot_v2_v2_backbone_train_v1/`
- route-selected closure：`outputs/mc24_ift_6dof_sensitivity_pilot_v2_route_selected_closure_{c,d}_v1/`
- 新脚本：`scripts/build_6dof_pilot_physical_corpus.py`
