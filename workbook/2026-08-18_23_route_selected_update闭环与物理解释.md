# 2026-08-18 (23) route-selected update 闭环：association-driven 对齐首次在真实 refit 上闭合

## 背景

iteration-1 通过全部冻结容差（workbook 22），冻结判定 `proceed_to_route_selected_update`。用户批准范围：V2 路由、train 源（10 源 × 8 点）推理 + route-selected update。

## 过程中发现并修复的两个实现缺陷

1. **物化命名空间按 payload 偏移，溯源交集恒为空**（`materialize_pooled_curriculum_synthetics.py`）：
   `namespace_base = 9e9 + group_index*1e4 + source_index*100` 与 `synthetic_run_id = base + group_index`
   都含 payload 组索引，导致同一物理事件在不同 payload 的名字空间 id 不同，route-selected update 的
   溯源交集（`align_route_selected_observations`）恒为空。修复：`alignment_iteration_shared_across_payloads`
   scope 下两者都不再含 group_index（overlay 种子本来就共享）。回归测试加在 `test_pooled_synthetic_seed.py`。
   这是"迭代环从未在真实 refit 上闭合"（审查结论）的直接原因之一。

2. **逐 payload 独立选路 + 交集语义在 FD probe 下无公共路由**：anchor∩reference=1410 条边，但
   anchor∩±dx probe 只有 9/165 条，8 路交集为 0。路由选择随几何变化是物理事实，交集语义不可用。
   诊断证明 anchor 选中的边在每个 payload 的候选图中 97% 存在（候选图是全扇出、无 χ² 门），
   7 路共同可用率 88.9%。因此实现 **anchor-selected 模式**（标准对齐做法：固定在 anchor 的选路，
   在每个 payload 按 origin 溯源重测同一路由集合的 mode-0 ACTS 残差/协方差）：
   - 新读取器 `alignment/route_selected_update.py::read_anchor_selected_field_edge_observations`
     （按 origin→embedding 映射跨 synthetic event 配对；残差构造与 backbone 完全一致，复用
     `build_field_candidates`）；
   - `run_route_selected_multidof_update.py` 新 `--observation-kind anchor_selected_field_edge`
     （anchor 输入仍为 backbone 输出；target/probe 输入为物化 sample 目录，校验 payload_id）；
   - 单元测试 `tests/test_anchor_selected_update.py`（3 个：残差与 canonical 构造一致、缺失 origin 计数、
     跨 payload 交集）。全套 177 测试通过。

## 结果（`outputs/mc24_multidof_ift_iteration01_route_selected_update_v1/`）

- 公共观测 **2309 条边**（anchor station-0 关联边的 89.5%），rank 3/3，**条件数 80.3**（真值拟合为 5328）。
- **capture_success = True**，且余量两个数量级：

| 参数 | anchor | recovered delta | 误差 | 冻结容差 |
| --- | --- | --- | --- | --- |
| ift_dx_mm | −0.1408 | +0.1400 ± 0.0091 | **−0.0008 mm** | 0.1 |
| ift_dy_mm | +0.1144 | −0.1149 ± 0.0181 | **−0.0005 mm** | 0.1 |
| ift_ry_mrad | +0.7426 | −0.7429 ± 0.0244 | **−0.0003 mrad** | 1.0 |

- 参数相关矩阵接近对角（dx-Ry 相关 0.077，真值拟合为 0.57）——固定路由集合 + 大统计量消除了
  近简并方向的噪声放大。
- proposed next values ≈ (−0.001, −0.001, −0.0003) ≈ 0：anchor 已在目标上，**迭代收敛，无需 iteration-2**。

## 意义与备注

- 这是项目首次完成 **associate → fit → align → 新 payload → refit → 再 associate** 的完整闭环：
  冻结 V2 backbone 无真值选路 → 固定路由集合 → 跨 payload 重测 → Newton 步精确恢复剩余偏移。
- response χ²/ndof = 6.6/9233 偏低，与已诊断的 y/ty 协方差高估一致（协方差失校的遗留问题，
  不影响本步的点估计精度，但应在后续工作中修正 mode-0 传播协方差）。
- 本步为 train 源 closure（MC 目标已知）。validation 源的独立 route-selected 验证、以及
  `write_station_alignment_payload.py --update-json` 生成修正 payload，留待用户决定。
- 物化修复改变了 `alignment_iteration_shared_across_payloads` 的名字空间约定：iteration-0 若需重跑
  association 路径，须用修复后的物化重新生成 synthetic overlay（物理库本身不受影响）。
