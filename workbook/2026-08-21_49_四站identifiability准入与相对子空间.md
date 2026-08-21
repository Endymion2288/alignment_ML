# 2026-08-21 (49) 四站 identifiability：生产完成、SVD 图与 15 维相对子空间准入

## 任务边界

本条目结束 Phase 1 identifiability：确认 Condor 物理生产真正完成，根据
truth-selected Jacobian 的 SVD/rank/condition/source spread 决定准入的
相对 alignment 子空间，并**预注册**下一轮 truth-selected relative
closure 的 capture 判据。不训练 Transformer，不打开密封 test，不生产
20 维 curriculum，不把 20 个自由参数直接当成可解。

线性 FD 恢复只作为 identifiability 诊断；正式 closure 评价留给预注册
之后的专用 driver。

## 环境与 provenance

- 分支：`4station`
- 节点：lxplus-gpu，Tesla T4；CVMFS
  `LCG_110_cuda/x86_64-el9-gcc13-opt/setup.sh`
- Calypso：已有 build，未重编
- 数据：MC24 train 源 `mc24_100043_00200_00299`（μ−）、
  `mc24_100044_00300_00399`（μ+），各 50 事件
- 物理链：真实 `/Tracker/Align` → SegmentFitRefit → SegmentsRefit →
  NtupleDumper → FaserActsExtrapolationTool(mode 0)
- 产物根：`outputs/mc24_four_station_identifiability_pilot_v1/`
- Condor cluster **1000360**（eossubmit / bigbird24 / tomorrow / 6000 MB）
- `test_data_accessed=false`；forbidden split 仍为 `test`

## 生产完成审计（不以 Condor 退出码为准）

| 源 | 计划点数 | 缺失 ROOT/payload | `failure.json` | `content_audit.json` |
| --- | ---: | ---: | ---: | ---: |
| `mc24_100043_00200_00299` | 51 | 0 | 0 | 51 |
| `mc24_100044_00300_00399` | 51 | 0 | 0 | 51 |

`identifiability_audit.json` 已写出；随后
`scripts/admit_four_station_relative_subspace.py` 写入
`admitted_subspace.json`。审计与准入都只读 train。

## 无约束图（FD 坐标 `unconstrained_full`）

Pooled truth pair：499。参数顺序为四站 × `(dx,dy,dz,rx,ry,rz)`。

| 图 | 维数 | rank | scaled condition | 注释 |
| --- | ---: | ---: | ---: | --- |
| 24（20 自由 + 4 survey dz） | 24 | 24 | **1.56×10⁹** | dz 列几乎不可观 |
| 20 自由（无 dz） | 20 | 20 | **8.55×10⁵** | 数值上满秩，但仍含 common mode |

24 维 SVD 分段：

- SV 0–5（~5×10⁶–5×10⁵）：相对 `rx`/`ry`（杠杆臂）
- SV 6–17（~6×10³–5×10²）：相对 `dx`/`dy`/`rz`
- SV 18–19（~12 与 ~6）：**global common `dy`、`dx`**
- SV 20–23（~1.7–3×10⁻³）：**survey `dz`**，数据 σ ≈ 43–49 mm

无约束 native σ 约 1 mm / 1 mrad，是 common-mode 膨胀：站间 dx–dx
相关 0.96–0.98，rx–rx 相关 ~0.99。这不是「每站绝对位置被径迹测到
1 mm」，而是整体平移几乎看不见。

20 维自由图的 condition 8.55×10⁵ 低于旧 IFT-only 6 维门限 10⁶，**不能**
因此准入。最后两个奇异值（common dx/dy）与最强相对转动模相差约 10⁶。

## 参考站 15 维图（求解坐标，不是物理约束）

丢掉一站的 5 个径迹约束坐标，得到 15 个相对自由参数。Jacobian 仍是
在全零 nominal 上的无约束 central-FD，只是求解时去掉列。

| 图 | pairs | rank | scaled condition | dx σ (mm) | dy σ (mm) | rx/ry σ (mrad) | rz σ (mrad) |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| S0 固定坐标 | 499 | 15/15 | 2.48×10⁴ | 0.18–0.29 | 0.10–0.16 | 0.11–0.16 | ~1.21–1.27 |
| S3 固定坐标 | 499 | 15/15 | 2.16×10⁴ | 0.23–0.28 | 0.15–0.17 | 0.10–0.17 | ~1.11–1.36 |

相对 `dz` 再加 3 列（S0 图）→ condition ~1.47×10⁸，σ_dz ~ 16–17 mm，
仍不是径迹测量。survey prior 5 mm 继续约束 dz，禁止把噪声 dz 写入
几何。

Source spread（仅 2 源、各 50 事件）：μ− 源的 dx/ry σ 明显差于 μ+
（例如 S0 图 `s1_dx` 0.85 mm vs 0.19 mm，相对 spread ~0.90）。dy/rx/rz
更稳定。pooled σ 比差的那一源紧，说明电荷合并有帮助。在把 σ 当作成
像分辨率冻结之前，应再加第三个 train 源（配置里已有
`mc24_100043_00600_00699`，尚未生产）。

## Gauge 语义（transform 不是标签）

Held-out payload 上：

- S0 / S3 `reference_station` 改写与 payload 的 `ΔT_ij` 一致
- 左乘 SE(3) common-mode `T_i' = G^{-1} T_i` 与 payload 一致（真 gauge）
- **加性**平均六矢量相减与 payload 的 `ΔT_ij` **不一致**（与先前
  layer rotation pivot 同类，不是 gauge）
- 把一站写成单位变换却不改写其他站，会改变 `ΔT_ij`，是不同物理约束

求解时的「丢掉 S0 的 5 列」是对**在单位变换处线性化的无约束 Jacobian**
做坐标选择。它**不等于**有限转动后做 `T_i' = T_ref^{-1} T_i` 再线性化。
本 bank 的两个 held-out 都写在 S0=identity 的坐标里，因此 S0 列删除的
线性恢复 χ²≈0，而 S3 列删除对同一注入会出现 ~2 mrad 的 `ΔT` `rz`
残差和 χ²~60。这不能解释成「S0 物理上正确」。比较只认 `ΔT_ij`。

## 准入决定

| 候选 | 决定 |
| --- | --- |
| 无约束 24 维（含 dz） | **拒绝**。condition ~10⁹，dz 数据 σ ~45 mm |
| 无约束 20 维自由 | **拒绝**。common dx/dy 奇异值 ~6–12；站间相关 ~0.96 |
| 15 维相对自由（`reference_station` 或事后左乘 SE(3) 改写） | **准入**，作为下一轮 truth-selected Newton/WLS 图 |
| `dz` | **survey prior 5 mm only**，不升为自由 Newton 坐标 |
| `rz` | 15 维中最弱（σ~1.2 mrad）。若相对 closure 失败，**首先丢掉 rz** 再试 12 维 |
| 冻结 V2 | 本阶段不评价。仅当 truth 边仍在而 score/route 因四站同时错位系统性退化时才重训 |
| 密封 test | 保持关闭 |

机器可读结果：`outputs/.../admitted_subspace.json`。
合同：`configs/physical_refit_four_station_relative_closure.yaml`。
旧 `run_refit_multidof_closure.py` 仍要求非空 reference，默认行为未改。
新 driver：`scripts/run_four_station_relative_closure.py`（opt-in，
在无约束 FD bank 上做求解期 15 维图）。

## 预注册 capture（打开正式 closure 评价之前）

只用于 train held-out，主判据是 gauge 不变量 `ΔT_ij`，不是未规范的
payload 六矢量。容差取 15 维 S0 图 pooled σ 最差站的约 3σ：

| 分量 | 容差 |
| --- | ---: |
| `dx` | 0.50 mm |
| `dy` | 0.45 mm |
| `dz` | 1.00 mm |
| `rx` | 0.50 mrad |
| `ry` | 0.50 mrad |
| `rz` | 4.0 mrad |

必须同时满足：

1. S0 图与 S3 图都对注入 `ΔT_ij` 过上述容差
2. 两套恢复 payload 彼此的 `ΔT_ij` 也在同一容差内
3. 左乘 SE(3) 改写保持恢复表
4. rank 15/15，scaled condition < 10⁶
5. 不读 test，不改 YAML 容差

`closure_relative_plus_common` 的 native `s1_dx` 对照 payload 0.55 mm
**不是**判据；S0 图应恢复相对 0.30 mm，并以 `ΔT_01` 为准。

## 下一轮 physical closure 计划

1. **本 bank、不再生产**：对
   `iteration_00_closure_relative` 与
   `iteration_00_closure_relative_plus_common` 跑
   `scripts/run_four_station_relative_closure.py`，按上面预注册判据
   评价。这是与历史 IFT multi-DoF 相同的 truth-fixed FD Jacobian WLS，
   **不是**新一轮 Athena 重传播，也不是未知关联。
2. 若 `ΔT_ij` 通过：在**同一物理 bank** 上接冻结 V2 association、
   现有 route solver 与 `physical_edge_deduplicated` 语义，做
   unknown-association closure。评价还要包括 route efficiency/purity/
   fake、candidate truth-chain recall、unique physical-edge count。
3. 若 15 维 `ΔT` 失败且残差集中在 `rz`：丢掉四站 `rz`，再试 12 维相对
   子空间，仍比较 `ΔT_ij`。
4. 不因 15 维线性诊断 χ²≈0 就宣称非线性 Newton 已闭合；若下一步需要
   有限转动迭代，必须在新的 current 附近重新 FD，而不是继续用 nominal
   线性化。
5. source spread 未冻结：第三 train 源只在需要把 σ 当作成像精度时再
   生产，不自动扩大 curriculum。
6. 仍然不重训 Transformer，除非出现「truth 边还在、冻结 V2 score/route
   因四站同时错位系统性变差」。

## 冻结判定

- 四站无约束 20D：**未准入**
- 15 维相对自由 + survey `dz` prior：**准入进入 truth-selected closure**
- 旧 IFT-only 默认路径：未静默改变
- 冻结 V2 / 密封 test：未打开、未重训
- capture YAML：已预注册，closure 评价开始后不得改

## 失败与限制

- 只有 2×50 事件；dx/ry 的单源 σ 差一个数量级，pooled 数不能当最终
  分辨率。
- Held-out 写在 S0=identity 图里，S0 列删除的线性恢复会看起来「过好」。
- 5 mrad 有限转动下，S3 列删除不是 SE(3) 规范坐标的线性化。
- 未评价 association，未做未知关联闭合。
