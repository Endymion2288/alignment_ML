# 项目审查与下一阶段计划

日期：2026-08-18
审查者：AI agent（只读审查；本轮未进行任何新的 physical production、训练、test-bank 生成
或新架构开发；未打开 sealed test 事件数据——只为 provenance 读取了历史 summary/contract
JSON）。

## 执行摘要

仓库的**核心管线健康且可信**：完整测试套件全绿（157/157，40.5 s），物理 refit/Acts
链真实且已验证，workbook 中的全部关键数值结论都与磁盘 artifact 精确一致，canonical
V1/V2/Ry/multi-DoF 入口全部正确执行 sealed-test 边界。问题集中在边缘：一个被提交进
git 的 Kerberos 凭证缓存（紧急）、过时的 README、仍可打开 sealed test 文件的 legacy
baseline 入口、`device="auto"` 背后的静默 CPU 回落，以及一个过期的 multi-DoF 语料
manifest。科学上，项目已经证明：物理链真实、单参数 dx/dy 与 Ry 可辨识且 truth-fixed
closure 成立、在大平移下首先失败的是 association 而非 candidate 丢失。项目**尚未**证明
multi-DoF 联合恢复、unknown-association alignment 或迭代闭环。下一步不是更多 ML，
而是聚合已经完成的 multi-source multi-DoF iteration-0 bank，检验 dx 在
100 events/source 下是否稳定。

## 环境

- LXPLUS GPU 节点；LCG view
  `/cvmfs/sft.cern.ch/lcg/views/LCG_110_cuda/x86_64-el9-gcc13-opt/setup.sh`
  （注意：实际 CVMFS bundle 是 `x86_64-el9-gcc13-opt`，不是旧说明中的 centos7-gcc11）。
- Python 3.13.11、PyTorch 2.11.0、CUDA 可用（Tesla T4）、uproot 5.7.1。
- `pytest -q`：**157 passed, 0 failed, 0 skipped, 40.47 s**（2026-08-18）。
- Git：`master` 分支，工作树干净，与 origin 同步。全部历史只有 **2 个 commit**
  （`268d091` 2026-08-12、`eea834c` 2026-08-17）；开发 provenance 在 `workbook/` 与
  `outputs/`，不在 git 历史。

## 仓库状态

| 区域 | 结论 |
| --- | --- |
| `alignment/` | Canonical。payload IO、物理有限差分 Jacobian、rotation closure、route-selected update。 |
| `baselines/route_assignment.py` | **Canonical route solver**（unit-capacity route packing；MILP + DP fast path）。无重复实现。 |
| `baselines/global_assignment.py` | Legacy 指派方法（greedy/Hungarian/dustbin/Sinkhorn）；`ScoreMatrix` 仍是共享基础设施。 |
| `baselines/multistation_assignment.py` | Legacy，已被 `route_assignment.py` 取代。 |
| `evaluation/pairwise_metrics.py` | **Canonical calibration**（Platt/temperature）。无重复实现。 |
| `models/`、`training/geometry_aware_transformer.py`、`training/route_aware_transformer.py` | Canonical V1/V2 栈。 |
| `training/structured_assignment.py` + V3 脚本 | Experimental，**已暂停**（negative result）。 |
| `training/station_pair_thresholds.py`、`scripts/materialize_curriculum_synthetics.py`、pre-refit alignment 脚本、合成 MC 时代脚本 | Legacy；保留作 provenance，不再扩展。 |
| `tests/` | 健康：157 个测试全过。 |
| `docs/` | 19/19 中英对齐全；抽查对数值同步。 |
| `README.md` / `README_cn.md` | **过时**（见下）。 |
| `xcheng.cc` | **紧急：Kerberos 凭证缓存被提交进 git。** |

## 数据清单

| 语料 | 状态 | 已核实事实 |
| --- | --- | --- |
| expanded dx/dy physical（`mc24_v3_expanded_trainval_physical_v1`） | COMPLETE | 18 source（10 train + 8 validation）；108/108 point accepted；magnitude {0, 0.1, 1, 5, 10, 50 mm}；train 994 / validation 796 source event；UID 交集 0；test 0；`q_over_p_mode=0`；无 `failure.json`；450 个引用路径全部存在。 |
| expanded dx/dy synthetic（`mc24_v3_expanded_trainval_synthetic_v1`） | COMPLETE | 12 sample = 每 split 6 个 payload group；UID 994/796，交集 0。 |
| IFT Ry physical bank（`mc24_ift_ry_expanded_trainval_physical_v1`） | COMPLETE | 234/234 point accepted；每 source 13 点（9 个纯 Ry：0、±10、±25、±40、±60 mrad；4 个 ±40 mrad⊕±1mm dx/dy 联合点）；condition axis 仅 `ift_ry_mrad`；994/796；交集 0。 |
| IFT Ry synthetic（`mc24_ift_ry_expanded_trainval_synthetic_v1`） | COMPLETE | 26 sample = 13 train + 13 validation。 |
| multi-DoF iteration-00 anchor bank（`mc24_multidof_ift_iteration00_anchor_trainval_physical_v1`） | 磁盘上 COMPLETE | 18 source × 8 point = 144/144 refit quartet；Condor cluster 991650 全部正常退出；无失败。**聚合 closure 与 pooled manifest 组装尚未运行。** |
| multi-DoF joint curriculum bank（`mc24_multidof_ift_joint_curriculum_trainval_physical_v1`） | 磁盘 COMPLETE，**manifest 过期** | 磁盘 234/234（cluster 991651）；`physical_corpus_manifest.json` 是提交时写入的版本，全部 point 标 `completed=false`；声明的 post-job 单进程 refresh 从未执行。**使用前必须刷新。** |
| 历史 sealed multidirection test（100116/100117，19 event） | SEALED，完好 | frozen contract 完整（SHA-256 固定的 checkpoint/calibration/operating point、validation-only 选择、`no_test_time_calibration=true`）；与 train/validation 严格 source 不相交。 |
| 早期小语料（`mc24_muon_2dfluka_curriculum_physical_v1`） | COMPLETE（历史） | 实际为 10 source / **97** event（59/19/19 train/val/test）；旧记录中的"99"是约数。 |

事件数脚注：6 个 source 为 99 event、2 个为 98（同一 source 的所有 point 一致；总数
994/796 处处相符）。这是输入 xAOD 的属性，不是 job 丢失。

## 实验清单

以下数字均在本次审查中从 artifact 重新读取，除特别标注外与 workbook 记录精确一致
（到所引精度）。

1. **物理链 smoke（station-3 +1 mm）**：恢复 [+1.0, 0.0] mm，27 个 truth-matched 对的
   最大误差 4.87e-13 mm。
   `outputs/mc24_muon_fasernu_5events_segment_refit_station3_dx1mm_closure/closure.json`
2. **物理 capture scan（25 点，9 个 magnitude × 3 方向，严格 0.01 mm 判据）**：
   0.1 mm 3/3 capture；1 mm 及以上 0/3；≥500 mm rank 塌陷。
   `outputs/mc24_muon_fasernu_physical_capture_scan_v1/capture_scan_summary.json`
3. **历史 sealed multidirection test**（冻结 MLP + unit-capacity route solver；只读记录，
   未重评估）：complete-track efficiency 在 0/0.1/1/5/10/50 mm 为
   0.773/0.797/0.820/0.553/0.264/0.0；raw truth-chain recall 处处 1.0。这正式确立了
   ≥5 mm 时"candidate graph 保留 truth、冻结 association 失败"。
   `outputs/mc24_muon_2dfluka_multidirection_test_route_level_v1/`
4. **V1 sealed test 对照**（记录）：所有 Transformer 变体在 5/10/50 mm 0/3 capture；
   geometry-aware 的 5/10 mm efficiency 优于 MLP，但 fake rate 0.065–0.070 超过 0.05
   门槛；no-context ablation 优于 full context。
   `outputs/geometry_aware_transformer_v1f_final_multidirection_test_v1/`
5. **expanded-corpus controls（validation-only，994/796 event）**：
   - MLP+route：nominal 0.9094/0.9724/fake 0.06286（fake 超门槛）；5/10/50 mm
     efficiency 0.0；candidate truth-chain recall 各 magnitude 全 1.0。
     `outputs/mc24_v3_expanded_trainval_mlp_route_validation_v1/`
   - V1 四个 control：5 mm 全部不过 gate；full-event context 未扩大 capture range；
     no-context 10 mm efficiency 0.5760 > full-context 0.2802。本次审查发现的修正：
     no-context 的 capture 集合是 {0, 1 mm}（0.1 mm purity 0.94568 < 0.95），不是
     workbook 表格暗示的 {0, 0.1, 1 mm}。
     `outputs/mc24_v3_expanded_trainval_v1_*/`
   - V2 BCE route-query：nominal 0.9163/0.9667/0.0443（过 gate），5 mm
     0.7559/0.9462/0.0725（purity+fake 失败）；capture {0, 0.1, 1 mm}；选定 context
     weight 0.05、unmatched penalty −0.5。**路径更正**：这些数字位于
     `outputs/mc24_v3_expanded_trainval_v2_direct_route_validation_v1`
     （流 `v2_direct_route_query`），不在 workbook 16 所写的
     `mc24_v3_expanded_trainval_v2_bce_control_v1`（后者是另一个 run：nominal
     0.7646/0.9798/0.0238）。
   - V3 exact structured-margin：nominal 0.7988/0.8639/0.2523；5 mm
     0.5450/0.8294/0.2638；10 mm 0.2188/0.5953/0.3218；六个 magnitude capture 全
     false；best epoch 2。soft control 更差（nominal 0.7335/0.8509/0.2779）。
     edge-only 对照 nominal efficiency 0.4460。**V3 是已确证的 negative result。**
     `outputs/mc24_v3_expanded_trainval_v3_route_validation_v1/`
6. **IFT Ry 物理研究**：
   - 状态响应：±60 mrad → Δx ∓111 mm、Δtx ±0.060——rotation 真实进入
     refit/propagation。
   - 从 nominal 的一次性线性化失败：±60 mrad 恢复为 +54.2246/−54.8091 mrad
     （误差 5.775/5.191 mrad）。
   - 局部物理迭代闭合：anchor ±50 mrad、probe ±45/±55、held-out ±60 恢复为
     **+60.068979 / −60.412497 mrad**（误差 0.069/0.413 mrad，rank 1，condition 1）。
     无 truth leakage：导数只用 probe refit；held-out 点从不进入拟合；station 1–3
     强制恒等（atol 1e-15）。设计说明：公共 truth-pair 集合跨 nominal/probe/observed
     四次评估取交集（pair 成员资格依赖 held-out refit；数值不进入）。
     `outputs/mc24_ift_ry_refinement_smoke_v1/local_step_to_p60/closure.json`、`.../local_step_to_m60/closure.json`
   - raw physical candidate complete truth-chain recall ≈ 0.95（validation：nominal
     0.9493、−60 mrad 0.9542、+60 mrad 0.9678）；0→1 是主要损失对。synthetic-overlay
     的 1.0 recall 使用"只选 complete truth track"的分母，不能与 raw coverage 混写。
     `outputs/mc24_ift_ry_expanded_trainval_physical_audit_v1/summary.json`
   - validation controls（nominal / |Ry|=60 mrad；全部 capture 5/5）：
     MLP+route 0.9489/0.9370 eff、0.9888/0.9878 purity、0.0312/0.0270 fake；
     V1 full 0.9651/0.9626、0.9819/0.9859、0.0295/0.0253；
     V1 no-context 0.9382/0.9429、0.9848/0.9870、0.0210/0.0192；
     V2 BCE 0.9355/0.9381、0.9797/0.9811、0.0404/0.0368。
     V2 same-checkpoint base-edge 对照失败（nominal fake 0.0628），说明 route query
     在此处是承重部件；V1 full-context efficiency 最高。
     `outputs/ift_ry_validation_control_assessment_v2/`
7. **multi-DoF（dx+dy+Ry）10-event joint anchor**：rank 3/3、scaled condition 511.79，
   但恢复增量 (+4.683 mm, +1.330 mm, −29.931 mrad) vs 目标 (−2.0, +1.5, −35.0)；
   dx leave-one-event-out 范围 −2.80…+9.55 mm；capture_success=false；candidate
   retention 1.0。一次性 closure（a/b）出现 dx 符号翻转。**10 event 下 multi-DoF
   closure 未达成。**
   `outputs/mc24_multidof_ift_iteration00_joint_a_local_step_v2/local_step.json`
8. **contract 扫描**：85 个 contract/summary/closure JSON；除历史 frozen-test 目录外，
   不存在任何 `test_events_loaded`/`test_opened`/`test_artifacts_opened=true`。

## 模型清单

| 模型 | Canonical config | 状态 |
| --- | --- | --- |
| Pairwise MLP + route | `configs/pairwise_mlp_route_ift_ry_validation.yaml`（Ry）；expanded 对照 `configs/pairwise_mlp_route_expanded_control.yaml` | Canonical baseline。Ry 上过 gate；dx/dy ≥5 mm 失败。 |
| V1 geometry-aware Transformer（full / no-context / no-chi2 / ordinary sparse） | `configs/geometry_aware_transformer_v1_ift_ry_validation.yaml`；expanded 对照 `configs/geometry_aware_transformer_v1_expanded*_control.yaml` | Canonical Transformer。Ry 上 efficiency 最高；dx/dy 上未显示 capture-range 扩大。 |
| V2 route-aware（BCE route-query） | `configs/geometry_aware_transformer_v2_ift_ry_validation.yaml`；`configs/geometry_aware_transformer_v2_expanded_bce_control.yaml` | Canonical route-aware 模型；alignment loop 的冻结 association backbone。历史（99-event）V2 选定 residual weight w=0；在 expanded/Ry 语料上 route query 有贡献（context weight 0.05）。 |
| V3 structured-margin | `configs/geometry_aware_transformer_v3.yaml` | **已暂停的 negative result。** 没有新假设不要重启。 |
| Chi2 / Hungarian / dustbin / Sinkhorn / multistation-flow baselines | 各 `physical_global_assignment_*.yaml` | Legacy。 |

## Alignment 清单

- payload 语义已对 Calypso 源码验证：每站
  `[dx_mm, dy_mm, dz_mm, rx_rad, ry_rad, rz_rad]`，全局 `T·Rz·Ry·Rx`；segment refit
  与 ACTS surface 都随 payload 变化。
- `q_over_p_mode=0` 在约 25 个检查点强制执行（语料构建、materialization、训练、评估、
  frozen-test、multi-DoF 入口）。两个 loader 级缺口：synthetic-manifest loader 不校验
  manifest 级 `q_over_p_mode`；`datasets/propagation_loader.py:127-131` 对缺失 q/p
  branch 静默填 0（legacy MC22 兼容），使"缺失"与"mode 0"不可区分。
- truth-fixed 单参数 closure：dx/dy 小偏移下达 5e-13 mm；Ry 经局部迭代在 ±60 mrad
  达 <1 mrad。
- multi-DoF（dx+dy+Ry）：满秩但 10 event 下统计不稳定；100 events/source 的
  multi-source iteration-0 bank 已完成、待聚合。
- field-aware global track fit：**接口缺失**——propagation tree 无 transport Jacobian
  （`supports_field_aware_global_track_fit=false`）；当前更新只能正确标注为
  route-consistency + WLS，不是独立 global likelihood。
- 迭代闭环（associate → fit → align → new payload → refit → re-associate）：组件齐备
  （`run_frozen_association_backbone.py --payload-id`、
  `run_route_selected_multidof_update.py`、带 `--update-json` 防护的 payload writer），
  但从未在真实 refit 上闭合。

## Test 边界审查

- Canonical 入口（V1/V2/V3 训练、refreeze、route evaluator、frozen backbone、Ry
  controls、multi-DoF 链）全部传 `allowed_splits=(train, validation)`；test 路径从不被
  解析。代码与 contract JSON 双重验证。
- 唯一被许可的 sealed-test 读取者是两个历史 frozen scan
  （`run_frozen_route_level_scan.py`、`run_frozen_geometry_aware_transformer_v1_scan.py`），
  它们通过 `_audit_sealed_samples` 校验 sealed source。
- **缺口（major）**：三个 legacy baseline 入口仍可打开 sealed test ROOT 文件：
  `scripts/run_curriculum_mlp_baseline.py:499+577`（无条件）、
  `scripts/run_station_pair_threshold_baseline.py:391+666-669`（`--evaluate-test`）、
  `scripts/run_global_assignment_mlp_baseline.py:118+1221-1223`（不加
  `--validation-only` 的默认模式）。loader 默认解析全部 split
  （`datasets/physical_curriculum.py:111-121`）；防护是逐调用点 opt-in。
  `scripts/audit_field_candidate_coverage.py` 接受 `--split test` 且无 seal 检查。
- 建议：加全局 seal guard（例如 loader 级拒绝解析 test 路径，除非显式传入
  `i_am_the_frozen_test_evaluation` 令牌），并修补三个 legacy 入口。

## 已完成工作

1. 真实物理链：`/Tracker/Align` payload → `SCT_ClusterContainer` → `SegmentFitRefit`
   → `SegmentsRefit` → `NtupleDumper` → mode-0 Acts——DONE 且已验证。
2. tracklet 导出契约 + ROOT schema（`faser-tracklets-v1`）——DONE。
3. dx/dy 物理扫描语料（expanded、source-disjoint、994/796）——DONE。
4. pairwise MLP + 全局 route 指派（unit-capacity packing）——DONE。
5. V1 Transformer + 机制诊断（local decoder 主导；over-smoothing；无 capture-range
   扩大）——DONE（negative/limits 已记录）。
6. V2 route-aware 目标（solver-consistent residual 形式）——DONE；expanded/Ry BCE
   route-query 是强对照。
7. V3 structured margin——DONE（negative result，已暂停）。
8. IFT Ry 物理 bank + candidate 审计 + validation controls + 局部迭代 closure——DONE。
9. multi-DoF 物理 bank（anchor + joint curriculum，仅 train/validation）——磁盘 DONE；
   聚合待运行。
10. 冻结 sealed multidirection test（历史、一次性）——DONE 且保持封存。

## 未完成工作

1. 已完成 iteration-0 bank 的 multi-DoF closure 聚合（最近的下一步）。
2. unknown-association alignment update（真实 refit 上的 truth-free route-selected
   update）。
3. 迭代 association/alignment 闭环。
4. raw candidate coverage 提升（0→1 损失；raw recall 约 95%）。
5. field-aware global track fit（需要 Calypso transport-Jacobian 导出或经验证的
   common-state Acts 重传播 API）。
6. Rx/Rz/dz 准入（需要逐 DoF 的有限差分 bank + rank/condition/closure 证据）。
7. layer/module 层级 alignment（预留；尚无 Calypso 映射）。
8. real-data 策略（workbook 03 中 data0 2022 IFT 重导出 provenance 检查失败；未解决）。
9. 最终独立 test 策略（新的 source-disjoint multi-DoF test bank——只能在方法论冻结后）。

## 已废弃工作

- coordinate-shift / residual-shift surrogate（已被物理 refit 链取代）。
- `baselines/multistation_assignment.py`、`training/station_pair_thresholds.py`、
  `scripts/materialize_curriculum_synthetics.py`、pre-refit alignment 脚本、合成 MC 时代
  脚本——保留作 provenance；不再扩展。
- V3 structured-margin 栈——已暂停的 negative result。
- 历史 99-event（实际 97）语料结论——在 workbook 注明处已被 expanded 语料取代。

## 已知 negative result

1. V3 exact structured-margin 与 soft-assignment control 在 expanded 语料的所有
   magnitude 上不过 gate（nominal fake rate 0.25–0.32）。
2. full-event Transformer context 不扩大 dx/dy capture range；no-context ablation 在
   10 mm 更好。
3. Ry 在 ±60 mrad 的大范围一次性线性化失败（5–6 mrad 误差）；必须局部迭代。
4. 10 event 的 multi-DoF 联合 closure 统计不稳定（dx 符号翻转）。
5. 历史 direct route-probability replacement 与 solver 不一致（从未选中 complete
   route）；residual 形式 `L = L_edge + w(L_route − L_edge)` 修复了它。

## 当前科学结论

**已确立：**
- 物理错位链端到端真实（canonical 路径无任何 surrogate）。
- 单站 dx/dy 在小偏移、truth-fixed association 下可辨识、可闭合到数值精度。
- IFT Ry 至少到 ±60 mrad 可辨识，局部物理迭代可恢复到 <1 mrad（truth-fixed，
  10 event）。
- 大平移（≥5 mm）下 physical candidate graph 保留 truth chain，而所有冻结
  association 模型不过 primary gate：**dx/dy 上首先失败的是 association，不是
  candidate 生成**。
- Ry 到 ±60 mrad，所有合理 association 模型在 validation 上过 gate：**Ry 不是
  association 瓶颈**，不能声称 Transformer 在 Ry 上扩大 capture range。
- V3 structured-margin 在当前规模是已确证的死路。

**未确立：**
- multi-DoF 联合可辨识性/恢复（秩满；10 event 统计失败）。
- unknown-association alignment 恢复。
- 完整闭环的迭代收敛。
- raw candidate coverage 充分性（约 95%，0→1 损失）。
- real data、6-DoF、layer/module alignment 的任何结论。
- 历史 sealed-test 数字只是单一冻结样本，不得外推。

## 当前瓶颈（排序）

1. **multi-DoF 统计稳定性/可辨识性**——10 event 下 dx 不稳定（condition ~512、符号
   翻转）。已完成的 100 events/source iteration-0 bank 是直接的检验。
2. **raw physical candidate coverage（约 95%，0→1 对）**——任何 unknown-association
   alignment 的上限。
3. **平移自由度上的 association capture range**——dx/dy 失效边界在 1–5 mm 之间；
   所有当前模型 ≥5 mm 失败。
4. **缺 field-aware global-fit 接口**——无 transport Jacobian 导出；当前更新只是
   route-consistency + WLS。

当前不是瓶颈：单 DoF 研究的数据量（994/796 在当前 gate 下足够）、route-solver 正确性、
calibration 基础设施。

## 风险清单

| # | 风险 | 严重度 | 行动 |
| --- | --- | --- | --- |
| 1 | `xcheng.cc` Kerberos 凭证缓存被提交进 git | **严重（安全）** | `git rm --cached xcheng.cc`，加入 `.gitignore`，视为已泄漏凭证处理（kdestroy/重新获取）。 |
| 2 | legacy baseline 脚本可打开 sealed test ROOT 文件 | Major | 加全局 seal guard；修补三个入口。 |
| 3 | `device="auto"` 静默 CPU 回落（2 个 helper + 约 15 个默认值） | Major | 训练/推理入口的 `auto` 在无 CUDA 时应响亮失败；单元测试保留显式 `cpu`。 |
| 4 | joint-curriculum `physical_corpus_manifest.json` 过期（把已完成 point 标为缺失） | Major | 任何使用前执行已声明的 post-job 单进程 refresh。 |
| 5 | README 过时（V3"生产中"、无 Ry/multi-DoF、3 对 doc 未入索引）；V3 doc 头部自相矛盾 | Minor | 更新 README 与 V3 doc 头部（中英）。 |
| 6 | loader 在两 key 并存时优先 legacy `magnitude_mm` | Minor | 翻转优先级或拒绝分歧。 |
| 7 | `propagation_loader.py` 静默填 0 缺失 q/p branch | Minor | 记录/标记该填充；legacy 路径保持显式。 |
| 8 | git 只有 2 个 commit；provenance 依赖 workbook/outputs | Minor | 今后按阶段提交。 |
| 9 | `faser_tracklet_alignment.egg-info/` 被跟踪 | Minor | 取消跟踪；`.gitignore` 加 `*.egg-info/`。 |
| 10 | workbook 16 把 V2 expanded-control 数字归属到错误输出目录 | Minor | 更正 provenance（本审查已记录正确路径）。 |

## 推荐的 canonical pipeline

```text
configs/physical_curriculum_v3_expanded_trainval.yaml        # source 列表（仅 train/val）
  -> scripts/build_physical_curriculum_corpus.py             # physical corpus manifest
  -> scripts/materialize_pooled_curriculum_synthetics.py     # synthetic overlay manifest
  -> 训练: train_geometry_aware_transformer_v1.py / train_route_aware_transformer_v2.py
       （validation-only calibration + operating point；forbidden_splits=[test]）
  -> 冻结推理: scripts/run_frozen_association_backbone.py --payload-id ...
  -> alignment 更新: scripts/run_route_selected_multidof_update.py
  -> 下一轮 payload: scripts/write_station_alignment_payload.py --update-json ...
  -> 物理 bank: prepare_multisource_multidof_iteration.py
       + submit_multisource_multidof_iteration_condor.py
       + run_multisource_refit_multidof_local_step.py（train 拟合，validation held-out）
```

Canonical artifact：solver `baselines/route_assignment.py`；calibration
`evaluation/pairwise_metrics.py`；route operating point
`training/transformer_route_selection.py`；闭环用 association backbone = 冻结 V2 BCE
route-query artifact。

## 下一实施计划（严格按依赖排序）

**Phase 1 — 审查收尾（本次审查的后续，均为小改）**
1. 从 git 移除 `xcheng.cc` 跟踪 + 更新 `.gitignore`（凭证模式、`*.egg-info/`）；
   取消 egg-info 跟踪。
2. 刷新过期的 joint-curriculum manifest（单进程、只读 refresh）。
3. 修补三个 legacy 入口使其拒绝 test split（或加 loader 级 seal guard）。
4. 让两个 device helper 的 `device="auto"` 在无 CUDA 时响亮失败；测试保留显式 `cpu`。
5. 修正 README（中英）与 V3 doc 状态头部；把三对缺失 doc 加入索引；更正 workbook 16
   的 V2-control 路径记录（指向
   `mc24_v3_expanded_trainval_v2_direct_route_validation_v1`）。

**Phase 2 — multi-DoF pilot（dx+dy+Ry，数据已产出）**
6. 在已完成的 iteration-0 anchor bank 上运行
   `run_multisource_refit_multidof_local_step.py`（train 定 update；validation 做
   held-out closure）。记录 rank、scaled condition number、参数 covariance/correlation、
   逐 source 与逐 station-pair 稳定性、candidate retention。
7. 若 dx 仍不稳定：先做诊断（逐 source Jacobian、event 数 scaling、anchor 选择），
   不推进任何新 production；closure 失败不得推进 payload。
8. 若 closure 通过：冻结 V2 逐物理点推理（`--payload-id`），然后
   `run_route_selected_multidof_update.py`（truth-free），再用 `--update-json` 写
   iteration-1 payload 并 refit——闭合第一次真实迭代。

**Phase 3 — multi-DoF 物理 curriculum**
9. 仅在 pilot 通过后：刷新 joint-curriculum manifest、审计
   （`audit_multidof_physical_corpus.py`）、materialize train/validation synthetic、
   运行 held-out joint closure（`run_multisource_multidof_physical_closure.py`）。
   错位保持随机联合采样（`configs/physical_curriculum_multidof_ift_trainval.yaml`
   已配置）。

**Phase 4 — multi-DoF 上的 association backbone 比较**
10. 用现有 validation-only 契约比较冻结 MLP+route、V1 full/no-context、V2 BCE。
    V3 保持 negative control。除非出现"candidate truth chain 保留但冻结 gate 失败"，
    否则不做新架构。

**Phase 5 — alignment 恢复**
11. 先做 truth-association 多参数 closure，再进入 unknown association。

**Phase 6 — 迭代闭环**
12. 闭合 associate → fit → align → payload → refit → re-associate；同时跟踪
    efficiency、purity、fake rate、chi2、unbiased residual、参数误差、normal matrix、
    condition number 与逐迭代收敛。

**Phase 7 — 更多 DoF**
13. Rx/Rz/dz 逐项准入，只凭有限差分 rank/condition/correlation 与 held-out closure
    证据。

**Phase 8 — 层级**
14. station → layer → module，仅在 Calypso 条件映射建立后。

**Phase 9 — 最终 test**
15. 仅在方法论冻结后：建立新的 source-disjoint multi-DoF 物理 test bank，一次性评估。
    历史 sealed test 保持不动。
