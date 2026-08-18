# 2026-08-18 审查收尾与 iteration-0 multi-DoF 聚合 closure

## 目标

按 `2026-08-18_19` 审查报告的执行方案推进：(1) 完成 Phase 1 审查收尾小修；(2) 在已完成的 iteration-0 anchor 物理库上运行多源聚合 local step（train 决定更新、validation 独立 held-out closure），作为 multi-DoF 主线的决策门。

## 执行与结果

### 1. Git 卫生（安全）

- `git rm --cached xcheng cc` 实际执行为 `git rm --cached xcheng.cc` 与 `git rm -r --cached faser_tracklet_alignment.egg-info`：两个路径从 git index 移除，磁盘文件保留（`xcheng.cc` 是在用 Kerberos 凭证，不能删）。
- `.gitignore` 新增 `*.egg-info/`、`xcheng.cc`、`krb5cc_*`；`git check-ignore` 确认两者均被忽略。
- 未提交 commit（留给用户决定）。

### 2. 三个 legacy 入口的封存 test 边界

- `scripts/run_global_assignment_mlp_baseline.py`：新增 `--allow-sealed-test`；默认（含 `--validation-only`）以 `allowed_splits=("train","validation")` 加载，test 路径不再解析；test 评估块、`test_opened`/`test_events_loaded`/`test_withheld_reason` 审计字段全部改由 `test_permitted` 控制。既有 Condor 封装 `run_ift_ry_validation_control_condor.sh` 已用 `--validation-only`，行为不变。
- `scripts/run_station_pair_threshold_baseline.py`：新增 `_load_manifest_for_scope(evaluate_test=...)`；默认不解析 test 路径，只有显式 `--evaluate-test` 才完整加载（该 flag 原本就是开 test ROOT 的闸门）。
- `scripts/run_curriculum_mlp_baseline.py`：test 评估贯穿全流程，无法局部切除；改为启动硬门——无 `--allow-sealed-test` 直接拒绝运行，报错指向两个获准的 frozen-test 读者。
- `tests/test_expanded_control_boundaries.py` 新增 4 个测试：station-pair 默认边界、station-pair 显式 evaluate_test 才解析 test、global-assignment 边界、curriculum-MLP 默认拒绝。`tests/test_geometry_aware_transformer.py` 新增 2 个 device 测试（见下）。

### 3. device="auto" 无 CUDA 响亮失败

- `training/geometry_aware_transformer.py::resolve_device` 与 `baselines/mlp_pair_classifier.py::_device`：`auto` 在无 CUDA 时改为 `ValueError`（原先静默回退 CPU）。显式 `cpu` 仍可用（单测与非模型工具）。所有训练/推理调用方均走这两个 helper，符合"训练/推理一律 GPU 否则失败"的既定策略。
- 全量测试：`163 passed`（含新增 6 个）。

### 4. 文档修正

- `README.md` / `README_cn.md`：V3 段落从"语料生产中、无结果"更正为"扩展语料（994/796）上完成训练验证、结构化目标未超过冻结 pairwise route 控制、作为负结果暂停"；新增 IFT Ry 与 multi-DoF 主线状态段落；文档索引补齐 4 对缺失条目（multidirection route scan、IFT Ry、multi-DoF loop、项目审查）。
- `docs/structured_assignment_v3.md` / `_cn.md`：状态头部更正为已完成评估的负结果（`v3_structured_route_utility` 全 magnitude `capture_successes=0`），指向证据目录。
- workbook 16 增加勘误：V2 control 数字（nominal 0.9163/0.9667/0.0443）实际来源目录是 `outputs/mc24_v3_expanded_trainval_v2_direct_route_validation_v1`（流 `v2_direct_route_query`），不是 `mc24_v3_expanded_trainval_v2_bce_control_v1`；数字本身无误。

### 5. joint-curriculum stale manifest 刷新

- `python scripts/build_physical_curriculum_corpus.py --config configs/physical_curriculum_multidof_ift_trainval.yaml --output-dir outputs/mc24_multidof_ift_joint_curriculum_trainval_physical_v1 --prepare-only`（只按磁盘状态重写 manifest，不跑生产）。
- 刷新前：18 source × 13 point 全部 `missing:payload_manifest,tracklets,propagations,content_audit`（提交时快照）。刷新后：**234/234 `accepted`**，每 source 13/13。

### 6. anchor 库单文件损坏的发现与修复

- 聚合首次运行在 uproot 读 basket 时 `IndexError`（exit code 被管道掩盖，实际失败）。逐文件扫描定位唯一坏文件：`mc24_100048_00050_00099/.../iteration_00_fd_ift_dy_mm_m/refit/tracklets.root`，仅 `cov_yy_mm2` 分支 basket 偏移表损坏，其余分支正常。
- 该点 `content_audit.json` 记录 366 tracklet / 98 event、协方差统计合理 → 生产时文件完好，损坏发生在存储层。
- 修复（纯本地、无需 Calypso）：坏文件改名 `tracklets.root.corrupt_20260818` 保留证据，旧 audit 存为 `content_audit.json.pre_repair_20260818`；由完好的 `enhanced_tracklets.root` 重跑 `convert_ntuple_tracklets.py --include-truth` 与 `audit_tracklets.py --require-mc-labels`。
- 验证：新 audit 与旧 audit **逐字段完全一致**（除 input 路径），`cov_yy_mm2` 可读 366 行。库恢复 144/144 完整。

### 7. iteration-0 聚合 closure（决策门）

命令：

```bash
python scripts/run_multisource_refit_multidof_local_step.py \
  --iteration-manifest outputs/mc24_multidof_ift_iteration00_anchor_trainval_physical_v1/iteration_manifest.json \
  --anchor-point iteration_00_anchor --target-point iteration_00_reference \
  --fit-split train --held-out-split validation \
  --capture-tolerance ift_dx_mm:0.1 --capture-tolerance ift_dy_mm:0.1 \
  --capture-tolerance ift_ry_mrad:1.0 --require-full-rank \
  --output-dir outputs/mc24_multidof_ift_iteration00_anchor_trainval_closure_v1
```

anchor 注入 `(dx, dy, ry) = (2.0 mm, -1.5 mm, 35.0 mrad)`，target 为 nominal；FD step `0.5/0.5/10`。train 10 source / 2296 观测，validation 8 source / 1756 观测。结果（`multisource_local_step.json`）：

| 参数 | 期望 Δ | train 恢复 Δ（误差） | validation 独立恢复 Δ（误差） | σ | 容差 | 判定 |
|---|---|---|---|---|---|---|
| ift_dx_mm | -2.0 | -2.141（-0.141） | -2.255（-0.255） | 0.022 | 0.1 | **失败** |
| ift_dy_mm | +1.5 | +1.614（+0.114） | +1.279（-0.221） | 0.022 | 0.1 | **失败** |
| ift_ry_mrad | -35.0 | -34.257（+0.743） | -35.055（-0.055） | 0.012 | 1.0 | 通过 |

- `capture_success=false`；rank 3 满秩；pooled 条件数 1880.8；dx-ry 参数相关 **0.850**（dx-dy、dy-ry 均约 -0.06）。
- **held-out 响应 chi2 下降 99.6% / 99.7%**（27.46M→2.15万、5.58M→7.73万）：线性响应模型的更新在独立数据上物理有效。
- 与 10-event joint-a smoke（v2：dx 误差 6.68 mm、符号错误）对比：100× 统计量后 dx 误差降到 0.14 mm 量级——**先前的 dx"不稳定"主要是统计涨落，不是结构性失败**。
- 逐源诊断（`source_fit_diagnostics.csv`，18/18 ok）：dx 逐源恢复 std 为 0.52（train）/ 0.54（validation）mm，而逐源统计 σ 仅 ~0.08-0.10 mm——**源间系统差是统计误差的 5-6 倍**；dy 逐源 std 0.32/0.37 mm；ry 0.46-1.14 mrad。train/validation 的逐源均值一致（dx -1.83/-1.80），但 pooled 偏差在两 split 间不同（-0.141 vs -0.255）→ 偏差含源依赖成分，不是常数偏移。
- 候选图参考指标：anchor 点 complete truth chain recall 0.44（train）/ 0.30（validation），nominal 0.50；anchor 处 Acts surface error 2470 次。这是已知的候选生成瓶颈，与本 closure 使用 truth pair 无关，但制约后续 ML 关联接入。

## 结论

1. Phase 1 审查收尾全部完成，全量测试 163 passed。
2. iteration-0 单步线性 closure **在冻结容差下失败**：dx/dy 存在 ~0.1-0.25 mm 的源依赖系统性偏差（6σ 量级），ry 在两 split 均精确恢复。
3. 失败模式已从"统计性崩溃"转为"系统性偏差"：响应非线性（35 mrad 大转动 anchor 处）经 dx-ry 0.85 相关泄漏进 dx，叠加逐源相空间差异。
4. 迭代闭环设计正为此而生：本步残差已收敛到 dx=-0.14、dy=+0.11、ry=+0.74（相对 anchor 缩小 10-15 倍），在残差点处重建 FD 探针的第二轮线性解大概率进入容差。但 iteration-1 payload 库是**新物理生产**（18 source × 8 point × 100 event 的 Calypso refit），按约定需用户明确批准后提交 Condor。

## 下一步（待用户决策）

- 选项 A：提交 iteration-1 anchor 库（anchor 移到 proposed next values），验证迭代收敛。
- 选项 B：先做纯只读归因（逐源 Jacobian 对比、缩小 FD step 的 toy 研究、ry 非线性量化），再决定生产。
- 选项 C：暂停 multi-DoF，回到关联主线（候选生成 recall 瓶颈）。

## 用户决策与 iteration-1 提交（2026-08-18 12:5x）

用户选择选项 A。已执行：

1. `prepare_multisource_multidof_iteration.py --source-config configs/physical_curriculum_v3_expanded_trainval.yaml --iteration-template configs/physical_refit_multidof_smoke_mc24_100043.yaml --output-root outputs/mc24_multidof_ift_iteration01_anchor_trainval_physical_v1 --iteration 1 --update-json outputs/mc24_multidof_ift_iteration00_anchor_trainval_closure_v1/multisource_local_step.json --allow-unverified-update --nevents 100`
   - `--allow-unverified-update` 是有记录的门控绕过：iteration-0 的 `capture_success=false`，但更新使 held-out chi2 下降 99.6%+、残差缩小 10-15 倍，属于标准迭代 alignment 工况；本记录即为所要求的显式文档。
   - iteration-1 anchor = `(-0.1408266434550427, +0.11440594271131221, +0.7426242467249153)`（fit split 冻结 proposal），reference 为 nominal，FD step 不变（0.5/0.5/10），8 点 × 18 source × 100 event。
2. `submit_multisource_multidof_iteration_condor.py ... --submit`：**Condor cluster `996698`**（eossubmit，18 job，每 source 一 job），`test_data_accessed=false`。
3. 等待 cluster 完成后：逐源完整性检查（144/144）、`assemble_multisource_multidof_iteration_manifest.py` 组装、重跑聚合 closure（anchor `iteration_01_anchor` / target `iteration_01_reference`）。若 iteration-1 在冻结容差下 capture=true，则按计划接入冻结 V2 route-selected update；若 dx/dy 仍有同量级偏差，则偏差与 anchor 半径无关的证据成立，转向选项 B 的归因。
