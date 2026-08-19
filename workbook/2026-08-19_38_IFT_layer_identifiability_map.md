# 2026-08-19 (38) IFT layer identifiability map：线性子集、两种 gauge、最小可闭合集合

## 任务

条目 37 的 Condor 生产完成后，对 IFT station+layer 联合 Jacobian 做
identifiability 审计，给出 **gauge 定义 + 最小可闭合 layer 集合**。
不打开 sealed test，不调 mode-0 / V2 / route / χ²，不把 layer 条件写入
station payload，不启动完整 layer curriculum 或 module 层级。

## 生产完成

- Condor cluster **998508**（bigbird24，`testmatch`，3 source job）全部结束；
  3/3 source、43/43 点、0 个 `failure.json`。`test_data_accessed: false`，
  `q_over_p_mode: 0`。
- 一个转换产物损坏：`mc24_100044_00300_00399` 的
  `iteration_00_fd_ift_rx_mrad_m/refit/tracklets.root` 中 `chi2` 分支
  空 basket（Athena ntuple 与 `content_audit.json` 完好）。从
  `enhanced_tracklets.root` 重跑 `convert_ntuple_tracklets.py` 修复；
  `load_events` 恢复 99 事件。与条目 34 的 propagations 截断同类，不是物理失败。
- 以后这类 Condor 生产按约 **30 分钟** 查询一次，完成后立刻接着分析。

## 设计 held-out 不构成线性闭合检验

原 held-out（layer0/layer2 反向 dx ±0.12 mm、**dy ∓0.08 mm**、ry ±1.2 mrad）
把 **非线性、会掉径迹的 layer dy** 写进了观测。因此：

- 不能把该点上的 Newton 恢复失败解释成「layer dx/ry 不可辨识」；
- **现在不启动** truth-selected Athena 再传播，也 **不启动** 冻结 V2
  route-selected closure。下一步若要闭合，必须先换一个不含 dy/rz 的
  小 severity held-out。

## 无规范 20 参数审计是中毒样本

`outputs/mc24_ift_layer_identifiability_pilot_v1_audit/`（20 参数、43 点全局交）：

| 量 | 值 |
| --- | --- |
| 公共 truth 边 | **75**（0→1/0→2/0→3 = 36/23/16） |
| 无规范 rank | 20/20 |
| 条件数 | **3.9×10⁹** |
| 完整四站 route | 46 |

原因不是「20 参数形式上缺秩」，而是 **layer dy 的 FD 点毁掉交**。单源
`100043_00200_00299` 上，reference 有 234 条 IFT 边；layer0 dy ±0.2 mm 丢掉
约 90 条（39%）；layer1 dy 丢掉 60%。把全部 40 个 FD 点交在一起后，留下的
是「所有 layer-dy 踢动都还活着」的稀薄子集，WLS 被跳变残差主导。
两种 gauge 的准入集合因此 **不一致**，held-out 恢复把 station common 拉到
数毫米——这是中毒 Jacobian，不是物理结论。

## 逐参数线性与占用（真实链，IFT 边）

单源 `mc24_100043_00200_00299`，判据：占用损失 ≤10%，且
`rms((r₊−r₀)+(r₋−r₀))/rms(r₊−r₀) ≤ 0.20`（相对奇函数）。
表见 `outputs/mc24_ift_layer_identifiability_pilot_v1_audit/fd_linearity_mc24_100043_00200_00299.json`。

**线性（当前 FD 步长）**

- 全部 station 5-DoF（0.5 mm / 10 mrad）：station dx +0.5 mm 的 IFT 边
  median Δrₓ = **0.500 mm**，几乎完全线性。
- 三层 **dx, rx, ry**（0.2 mm / 2 mrad）。

**非线性 / 掉径迹（当前步长下不准入）**

- 全部 layer **dy**：占用损失 39–60%，相对奇函数 1.3–1.9。0.2 mm 已打乱
  IFT segment 拟合（精度在 strip y 上）。
- 全部 layer **rz**：相对奇函数 ~1，layer1 rz 还掉 21% 边。

补充：layer0/layer2 的 **dx 是线性的，但被立体角放大**——0.2 mm 平面平移
给出 median Δrₓ ≈ **9.7–9.9 mm**（约 50×）。这是重建算子，不是 station 刚体
平移。station dx 与 layer dx 的后验相关只有 0.13–0.41，**不是同一算子**，
禁止把 layer dx 倒进 station dx 槽。

## 线性 14 参数联合（station 5 + layer dx/rx/ry）

`outputs/mc24_ift_layer_identifiability_pilot_v1_audit_linear14/`
（去掉 layer dy/rz 的 FD 点后再交）：

| 量 | 值 |
| --- | --- |
| 公共 truth 边 | **690**（约 9× 中毒样本） |
| 无规范 rank / 条件数 | 14/14，**2.4×10⁵** |
| station dx ↔ station ry 相关 | **0.993** |
| layer0 dx ↔ layer2 dx 列余弦 | **−0.95** |

station dx 在与 station ry 同拟合时因杠杆臂简并不准入（trial 条件数
10⁴–10⁵）。这是 station 级已知结构，**不是** layer 新信息。layer 三层 dx
彼此高相关：可辨识的是 **一层相对变形**，不是三层独立平移。

## 冻结 station 后的 layer-only 9 参数（本条目的 map）

正确的层级：station 5-DoF 保持冻结；只浮动 IFT 层内部。
`outputs/mc24_ift_layer_identifiability_pilot_v1_audit_layer_only9/`：

| 量 | 值 |
| --- | --- |
| 参数 | 9（三层 × dx/rx/ry） |
| 公共边 | **692** |
| 无规范条件数 | **1.7×10⁴** |
| sum_to_zero 6 维 | 满秩，条件数 **6662**（低于 10⁴ 门） |
| reference_layer 6 维 | 满秩，条件数 **7327** |

### 两种 gauge 的准入（门：源间散布 ≤0.5，σ < severity，trial 满秩且条件数 ≤10⁴）

| 物理模式 | sum_to_zero（丢掉 layer 2） | reference_layer（固定 layer 0） |
| --- | --- | --- |
| 相对 dx | **layer0 dx** σ=7 μm；**layer1 dx** σ=10 μm | **layer2 dx** σ=6 μm；layer1 dx 源间不稳 |
| 相对 rx | **layer0 rx** σ=0.12 mrad；layer1 rx 源间不稳 | **layer2 rx** σ=0.15 mrad；layer1 rx 不稳 |
| 相对 ry | **layer0/1 ry** σ=0.87 / 0.74 mrad | **layer2 ry** σ=1.06 mrad；layer1 ry 不稳 |

两种 gauge **标签不同、物理相同**：外层（0 与 2）相对内层/参考层的
dx、rx，以及一个相对 ry。中间层（layer 1）的 rx（以及 reference_layer 下
的 dx/ry）源间散布 >0.5，**不准入**。

冻结 station 后，mixed 点的 station dx 读出保持 0（sum_to_zero）或只出现在
layer 加权均值里（reference_layer 的读出约定），**没有把 0.15 mm station
平移倒进 station 槽**——因为 station 根本没进本拟合。

held-out 仍含 dy，所以线性恢复会被污染：reference_layer 全 6 维对
**相对 dx** 已经接近真值（layer0/2：0.099 / −0.119 vs 注入 0.12 / −0.12），
ry/rx 则被 dy 泄漏拉偏。这只说明 dy 必须从闭合点拿走，不是相对 dx 失败。

## Identifiability map（冻结结论）

**可进入后续小 severity 闭合的最小集合（station 冻结 + 显式 gauge）**

1. **IFT 相对 dx**（一个 gauge 自由度：sum_to_zero 的 layer0/1，或
   reference_layer 的 layer2）。σ ~ 6–10 μm，远小于 0.2 mm FD 与 0.12 mm
   注入。外层立体放大提供信息；三层独立 dx **不可**同时自由。
2. **IFT 外层相对 rx**（layer0 或 layer2，视 gauge）。layer1 rx 源间不稳。
3. **IFT 外层相对 ry**，σ ~ 0.7–1.1 mrad，与 1.2 mrad 注入同量级，属
   勉强可辨识，闭合时需更小注入或更多统计。

**明确不准入**

- layer **dy**、layer **rz**（当前 0.2 mm / 2 mrad 下重建非线性 + 掉径迹）
- 三层独立平移/转动（layer0 dx 与 layer2 dx 近反相关）
- 把 layer 公共模再拟合进已闭合的 station 5-DoF（station dx↔ry 相关 0.993；
  条目 36 的 station 解保持冻结）
- 本轮设计的 internal/mixed held-out（含 layer dy）

默认比较约定继续用 **sum_to_zero**（station 承担 common mode；层改正和为零）。
两种 gauge 对「外层相对变形」的物理结论一致，只是丢掉哪一层不同。

## 下一步（尚未开工）

1. 新的 **线性 held-out**（建议 layer0/2 反向 dx ±0.12 mm，可选更小的
   相对 ry；**禁止 dy/rz**），3 source × 1 点即可，Jacobian 复用现有
   linear FD，不必重跑 40 个 probe。完成后才做 truth-selected 闭合。
2. 若仍想要 layer dy：必须先把 FD 步长降到重建连续区（远小于 0.2 mm）
   并重新做线性筛，而不是在当前步长上硬拟合。
3. 冻结 V2 route-selected 只在线性 held-out 闭合之后才启动。
4. 完整 10+8 layer curriculum 与 module 级：要等这张 map 上的相对 dx
   （及可选 rx/ry）在干净 held-out 上闭合。

## 产物

- 生产 bank：`outputs/mc24_ift_layer_identifiability_pilot_v1/`
- 中毒 20 参数审计：`outputs/mc24_ift_layer_identifiability_pilot_v1_audit/`
- 线性筛：`.../fd_linearity_mc24_100043_00200_00299.json`
- station+layer 14 参数：`outputs/mc24_ift_layer_identifiability_pilot_v1_audit_linear14/`
- **map 用** 冻结 station、layer-only 9 参数：
  `outputs/mc24_ift_layer_identifiability_pilot_v1_audit_layer_only9/`
