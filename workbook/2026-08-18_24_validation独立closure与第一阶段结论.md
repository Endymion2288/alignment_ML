# 2026-08-18 (24) validation 独立 closure：第一个 unknown-association source-disjoint multi-DoF 迭代对齐闭合

## 任务边界（用户冻结）

不训练模型、不提交 iteration-2、不调 V2 checkpoint/calibration/route policy、不开 sealed test。
iteration-1 train 阶段确定的一切（V2 backbone、calibration、route threshold 0.001/pair、
unmatched penalty −1.0、anchor-selected observation 定义、FD convention ±0.5 mm/±10 mrad、
capture tolerance 0.1/0.1/1.0）原样冻结，只对 8 个 validation source 跑同样流程。

## 执行

1. 冻结 V2 backbone 推理：`run_frozen_association_backbone.py --split validation`，8 个 payload
   逐一独立输出（`outputs/mc24_multidof_ift_iteration01_v2_frozen_backbone_validation_v1/`），
   与 train 使用同一冻结 artifact（`mc24_v3_expanded_trainval_v2_bce_control_v1`，checkpoint
   sha256 与 direct-route validation 记录一致）。
2. validation anchor-selected route-selected update：
   `outputs/mc24_multidof_ift_iteration01_route_selected_update_validation_v1/`。
3. 并排比较工具 `scripts/compare_route_selected_updates.py`（新，含单元测试）：recovered delta、
   误差、rank、条件数、参数相关、公共路由比例、response χ²/ndof、逐 source WLS 重解
   （只读复用冻结观测的 derivative/response/covariance 数组，不重拟合）、anchor coverage。
   输出 `outputs/mc24_multidof_ift_iteration01_route_selected_comparison_v1/`（JSON/CSV/中文片段）。

## 结果（train vs validation 并排）

| 指标 | train | validation |
| --- | --- | --- |
| dx 误差（容差 0.1 mm） | −0.0008 (σ 0.009) | **−0.0945 (σ 0.018)** |
| dy 误差（容差 0.1 mm） | −0.0005 (σ 0.018) | +0.0575 (σ 0.021) |
| Ry 误差（容差 1.0 mrad） | −0.0003 (σ 0.024) | −0.0280 (σ 0.028) |
| rank / 条件数 | 3 / 80.3 | 3 / 85.5 |
| 公共路由边 / anchor 保留率 | 2309 / 0.895 | 1756 / 0.922 |
| response χ²/ndof | 0.0007 | 0.047 |
| dx-Ry 相关 | 0.077 | 0.267 |
| 逐源散布 dx/dy/Ry | 0.006/0.006/0.009 | 0.145/0.071/0.035 |
| anchor 0->1 效率 / 纯度 | 0.864 / 0.988 | 0.792 / 0.984 |
| 冻结判定 | 通过 | **通过** |

## 逐 source 分解（validation，目标 Δdx=+0.1408）

7/8 源恢复 dx ∈ [+0.129, +0.147]，与目标一致；**唯一离群源 `mc24_100047_00150_00199`
恢复 dx=−0.297、dy=+0.101、Ry=−0.842**（其余源 Ry 也在 −0.72…−0.75 正常）。该源把 pooled dx
从 ≈+0.14 拉到 +0.046，是 validation dx 误差 −0.0945（容差的 94.5%）的主要来源。
不做剔除或重加权（属于调参），仅记录事实；该源与 iteration-0 真值诊断中观察到的
source-dependent systematics 一致，留待 coverage/协方差瓶颈阶段处理。

## 冻结判定与结论

`validation_closure_confirmed`：validation 通过全部冻结容差。按用户预先冻结的规则，
正式记录为**项目第一个 unknown-association + source-disjoint + 真实 payload/refit/Acts +
multi-DoF 迭代对齐 closure**：

- 完整链路：冻结 V2 无真值选路 → anchor 固定路由集合 → 跨 payload origin 重测 →
  Newton/FD 更新 → 新 payload 物理 refit → 再 associate，两个 source-disjoint 划分均闭合。
- train 的近零误差部分来自自洽（anchor 即 train 拟合提案）；validation 是真正独立验证，
  其 dx 余量小（94.5% 容差）且由单一离群源主导——如实记录，不构成失败。

## 下一阶段（按用户指示，association 架构停止优化）

转向两个已知基础瓶颈：
1. **mode-0 传播协方差严重各向异性失校**（x/tx 低估 ~6-7×、y/ty 高估 ~180×，见 workbook 21）——
   它同时解释 χ² 门误杀真边与 dx/dy 残差偏差的噪声放大。
2. **0->1 raw candidate coverage**（anchor 处完整真链保留 0.40-0.44，χ² 门拒绝 38-45% 是主因）。

正式文档 `docs/global_alignment_multidof_loop.md`（EN）与 `_cn.md` 的"当前状态"已同步更新。
