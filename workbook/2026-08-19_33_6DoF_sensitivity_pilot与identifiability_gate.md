# 2026-08-19 (33) Rx/Rz/dz sensitivity pilot：station-0 六参数 Jacobian、准入 gate 与 dz 不可辨识判定

## 任务

propagation/mode-3 分支正式关闭（条目 32）后返回 multi-DoF 主线第一阶段：
在**不**生产完整 6-DoF curriculum、**不**打开 sealed test、**不**调任何
χ²/covariance/route/association 参数的前提下，用少量 train source 在
nominal/近 nominal geometry 周围生成 ±Rx、±Rz、±dz 真实 central-FD probe，
与现有 dx/dy/Ry Jacobian 合并为 station-0 六参数 J=[dx,dy,dz,Rx,Ry,Rz]，
系统建立 sensitivity / rank / condition number / SVD / covariance /
source spread，并按可辨识性逐项准入。

## 设计与生产（全部沿用已验证链，零 C++ 改动）

- 基础设施确认：`/Tracker/Align` payload 六分量 `[dx,dy,dz,rx,ry,rz]`
  （mm + rad）由 `TrackerAlignDBTool` 按 `Translation·Rz·Ry·Rx` 应用；
  `alignment/physical_jacobian.py` 六分量映射齐全；closure 评估器按
  `finite_difference_for` 元数据自动发现 probe，与参数个数无关。
- 模板 `configs/physical_refit_6dof_sensitivity_pilot.yaml`：
  中心 = iteration-01 收敛 anchor（dx=-0.1408, dy=+0.1144 mm, Ry=+0.7426 mrad，
  新 DoF=0；canonical closure 残差 <10 μm / <1 μrad，即近 nominal 线性化点）。
  FD 步长：dx/dy 0.5 mm（沿用已验证扫描）；**dz 2.0 mm**（响应仅经 track
  斜率 ~t·dz ≈ 10 μm，纯平移严格线性，大步长安全）；**rx/rz 10 mrad**
  （沿用 Ry 已验证步长）。severity 约定不变：平移 5 mm、旋转 60 mrad。
- 每 source 18 点：reference + anchor + 12 FD + 4 个全 6-DoF held-out
  joint closure 点（a/b severity 0.54/0.95 作非线性边界测量；c/d
  severity 0.14/0.21 在 FD 包络内作线性 closure 验证）。
- 3 个 train source（mc24_100043_00200_00299、mc24_100043_00600_00699、
  mc24_100044_00300_00399，跨物理样本与事件区间），每点 100 事件，
  真实 `/Tracker/Align → SegmentFitRefit → SegmentsRefit → NtupleDumper →
  Acts(mode 0)` 链，Condor 生产零失败。
- 观测语义：truth-selected physical edge（min_truth_match 0.99，mode 0），
  逐 source 在全部扫描点间按 (run,event,tracklet) 对齐取交集；
  pooled 694 条 truth pair（670 条属完整 0→1→2→3 route）。
  无 association、无 overlay、无 test。

## 六参数 sensitivity（pooled，响应 RMS / native 单位）

| 参数 | rx_mm | ry_mm | rtx | rty | 加权信息范数 |
|---|---|---|---|---|---|
| dx (mm) | **1.002** | 0.092 | 8e-5 | 6e-5 | 61.6 |
| dy (mm) | 0.207 | **1.018** | 1.3e-4 | 9e-5 | 26.2 |
| dz (mm) | 0.020 | 0.005 | 1e-6 | 1e-6 | **0.31** |
| rx (mrad) | 0.073 | **1.549** | 5e-5 | 1.0e-3 | 19.3 |
| ry (mrad) | **1.558** | 0.082 | 1.0e-3 | 6e-5 | 83.7 |
| rz (mrad) | 0.056 | 0.077 | 1e-5 | 2e-5 | 3.65 |

逐 station pair：rx/ry 响应随 lever arm 增长（0→1: 0.033，0→2: 0.62，
0→3: 1.21 mm/mrad）；rz 近似 pair 无关（0.04–0.05 mm/mrad，in-plane
位置杠杆 ~|x|,|y|）；dz 各 pair 均 ~0.01 mm/mm（斜率耦合）。

物理读法：dx/dy 为 1:1 直接响应；ry/rx 经站间距杠杆（~1.5 m）产生
~1.6 mm/mrad 响应——**rx 意外地强**（信息量排名第 2，σ=65 μrad）；
rz 经 in-plane 位置响应可观（σ=0.32 mrad）；**dz 比 dx 弱 ~200 倍**。

## 近简并结构

响应列余弦（几何简并）与后验相关（|值|>0.3）：

| 耦合对 | 列余弦 | 后验相关 |
|---|---|---|
| **dy ~ rx** | +0.598 | -0.598 |
| dx ~ ry | +0.494 | -0.411 |
| dx ~ rz | +0.388 | <0.3 |
| dz ~ rz | +0.397 | -0.321 |
| ry ~ rz | +0.340 | <0.3 |

scaled normal matrix SVD（奇异值降序，参数组成 |系数|>0.15）：

| 奇异值 | 主导方向 |
|---|---|
| 2.53e7 | ry |
| 1.34e6 | rx |
| 7.75e4 | dx − 0.39·rz 混合 |
| 3.59e4 | rz − 0.39·dx 混合 |
| 1.09e4 | dy |
| **1.92** | **dz（纯方向，7 个数量级之下）** |

dy-rx 是最强新耦合（与预判一致）；dx-ry 耦合延续；rz 与 dx/dz/ry 广泛
弱耦合但均可分离；dz 是完全孤立的弱方向。

## 准入 gate

确定性阈值：source spread ≤0.5、数据 σ < severity、准入子块满秩且
条件数 ≤1e4。

| 参数 | 信息排名 | source spread | σ_data | 判定 |
|---|---|---|---|---|
| ry | 1 | 0.355 | 0.014 mrad | 准入 |
| rx | 2 | 0.032 | 0.065 mrad | 准入 |
| dx | 3 | 0.324 | 0.020 mm | 准入 |
| rz | 4 | 0.382 | 0.320 mrad | 准入 |
| dy | 5 | 0.080 | 0.048 mm | 准入 |
| **dz** | 6 | 0.401 | 3.6 mm | **拒绝（条件数）** |

6×6 scaled normal matrix 数值满秩（6/6）但条件数 1.3e7；dz 奇异值 1.92
vs 顶端 2.5e7。逐 source σ_dz = 7.2 / 4.9 / 10.9 mm——**FASER 现有
近平行 track sample（mrad 斜率）对单站 dz 无实际约束能力**：响应仅经
t·dz，属 gauge 型弱方向，需 survey prior 或更大角度相空间。

内置 sanity closure（anchor→reference，小残差 regime）：dx/dy/ry 精确
恢复（dx +0.134 vs 期望 +0.141；ry -0.749 vs -0.743），新 DoF 与 0 一致
（rx -0.016±0.065，rz -0.080±0.320）；**dz 回收 +2.2±3.6 mm**——
大 σ 再次确认其不可辨识。

## 大 severity closure 点（a/b）的负结果：线性 regime 边界

对准入 5 参数子集在 closure_a（severity 0.54）/ closure_b（severity 0.95）
上做 truth-selected 联合 closure，**线性模型失效**：

- post-fit 残差 RMS 1.6–2.1 mm（a）/ ~4 mm（b），χ²/ndof ≈ 3–11；
- 参数误差大且 source 间不一致（closure_b rz 误差 +7.3/+12.6/+21.2 mrad）；
- 逐边结构：median |残差|/|响应| 仅 6.2%（主体边仍近似线性），但
  ~4.5% 的边（31/691）偏离 >50%（max 3.5）——**segment refit 在大位移
  下再收敛的间断尾巴**污染 WLS 并拖垮弱方向（rz/dz 量级）参数回收。

结论：FD 线性化只对**小残差 regime**（alignment 迭代实际工作区，
post-iter1 残差 <10 μm 量级）成立；severity ≳0.5 的大注入超出线性
closure 适用范围。c/d 小 severity 点（0.14/0.21，FD 包络内）的联合
closure 验证在 v2 bank 生产完成后进行（条目 34）。

## 产出

- 生产：`outputs/mc24_ift_6dof_sensitivity_pilot_v1`（16 点×3 源）、
  `..._v2`（18 点×3 源，含 c/d；FD/anchor/reference 与 v1 同变换同链）。
- 审计：`identifiability_audit/identifiability_6dof.json`（pooled +
  逐 source + 完整 route 子集 + SVD + gate 全记录）。
- 大 severity closure：`joint_closure_admitted5/`（负结果证据）。
- 脚本：`scripts/audit_6dof_identifiability.py`、
  `scripts/run_6dof_joint_closure.py`；测试 `tests/test_6dof_identifiability.py`。
- commit：`4ca0043`（脚手架）、`6ae1b3a`（closure 驱动）。

## 暂定判定（待 c/d closure 确认）

1. **准入集合 = {dx, dy, ry, rx, rz}**；dz 拒绝并记录为 gauge 型弱方向
   （需 survey prior 或更丰富 track 相空间，当前不加入联合 closure 主线）。
2. dx-ry、dy-rx、rz 耦合均可分离（5 参数准入子块条件数 2.4e3），无
   rank 缺失；rx 灵敏度强（σ 65 μrad）是本次最显著的正面新结果。
3. 联合 closure 必须在 ≲0.2 severity 的线性 regime 内验证；大注入
   closure 的失效机制（refit 间断尾巴）已记录为 regime 边界。
