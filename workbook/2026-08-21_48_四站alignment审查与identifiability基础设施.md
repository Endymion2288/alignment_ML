# 2026-08-21 (48) 四站 alignment：旧假设审查、设计合同与 Phase 1 基础设施

## 任务边界

在新建 `4station` 分支上回答：IFT/S0、S1、S2、S3 **都存在真实 misalignment**
时，现有 source-disjoint 物理链能否恢复四站相对几何，以及 global gauge /
survey prior 起什么作用。本条目只做仓库审查、机器可读合同、单元测试与
identifiability 基础设施。**不训练 Transformer，不打开密封 test，不改旧主线
产物，不把 20 个自由度直接当成可解。**

确认：`git` 位于 `4station`，与 `master` 同 commit `a75800f` 分出，工作区干净。
本分支之后的 commit 不得回写旧 IFT-only 结果目录。

## 环境

- 节点：`lxplus909`，GPU `Tesla T4`
- CVMFS：优先 glob `LCG_110_cuda/**x86_64-centos*-gcc11-opt**` **不存在**。
  本节点实际只有
  `LCG_110_cuda/x86_64-el9-gcc13-opt` 与
  `LCG_110_cuda/x86_64-ubuntu2404-gcc13-opt`。
  沿用已验证 fallback `LCG_110_cuda/x86_64-el9-gcc13-opt/setup.sh`
- Calypso：`/eos/home-x/xcheng/FASER/calypso/run/setup.sh` 已在，不重 clone、
  不重编
- 数据：`/eos/experiment/faser/`
- 训练/推理若启动则 GPU-only；本阶段无模型

## 旧主线冻结结论（不得机械复制）

条目 36 冻结的是 **IFT/station-0** 的「5 个径迹约束 DoF + survey `dz`」，
S1–S3 被写成 identity 并称为 reference。条目 33/34 的 identifiability map
也只针对 station-0 六矢量。条目 46/47 的 calibration mode 仍是 IFT station
vs IFT-internal `C_dx`。这些结论对「四站同时错位」**不自动成立**。

## 修改清单（硬编码 station-0 / reference 假设）

| 位置 | 旧假设 | 本分支处理 |
| --- | --- | --- |
| `scripts/run_physical_refit_capture_scan.py` `_build_station_rigid_multidof_plan` | 强制非空 `reference_station_ids`，reference 必须为零，参数只能写在 movable 上 | **默认行为不变**。仅当 `alignment_formulation: four_station_v1` 时允许空 reference、四站都 movable |
| 同上，IFT layer / IFT Ry 计划 | 硬性 `reference=[1,2,3]`、`movable=[0]` | 不改。那是 IFT 内部层级，不是四站 station 问题 |
| `scripts/prepare_multidof_alignment_iteration.py` | 为零 current 再写一个 zero anchor 会与「唯一 nominal」冲突；历史用法是非零 IFT anchor | 四站 pilot 用专用 compiler，在 nominal 线性化，不走旧 iteration compiler |
| `alignment/five_dof_sampling.py` | 参数名 `ift_dx_mm`… 只描述 station 0 | 不改。新合同使用 `s{i}_{component}` |
| `alignment/calibration_modes.py` / hierarchical V1 | Station Mode = 只动 IFT | 不改旧 mode。四站是新 formulation |
| `scripts/run_refit_multidof_closure.py` | 要求非空 reference ∩ 空 movable | 本阶段不改；四站 audit 走新脚本 |
| `scripts/run_refit_ift_ry_*.py` | 只要 station-0 边 | 不改，属于 IFT-Ry 历史链 |
| `scripts/run_payload_*` / `alignment/closure.py` | `reference_station=0` 的坐标 surrogate | 不改；四站禁止 surrogate |
| `configs/physical_refit_6dof_sensitivity_pilot.yaml` 等 | `movable: [0]` | 不改旧配置。新模板 `physical_refit_four_station_identifiability_pilot.yaml` |
| 旧 `outputs/mc24_ift_*` | station-0-reference artifact | **不复用为四站证据**。本分支自建 manifest |
| 密封 test `100116/100117` | 永久禁止 | 新 source 合同 `forbidden_splits: [test]` |

Payload writer 本身已支持 `station:0..3` 独立六矢量；缺的是 plan / compiler /
gauge 合同，不是 Calypso 写入器。

## 设计合同（阶段 2）

- Schema：`faser-four-station-alignment-v1`（`alignment/four_station.py`）
- 自由参数：`s0..s3` × `{dx_mm,dy_mm,rx_mrad,ry_mrad,rz_mrad}`（20）
- Survey：`s{i}_dz_mm`，prior 5 mm，只做 FD 诊断
- 注入：每个点写齐四站 `[dx,dy,dz,rx,ry,rz]`，`T*Rz*Ry*Rx`，真实链
- FD 图：`gauge: unconstrained_full`（四站都扰动）
- 求解图（Jacobian 之后）：`reference_station` 与 `common_mode_constraint`
- 比较量：`ΔT_ij = T_i^{-1} T_j`。左乘公共 SE(3) 是真 gauge；加性公共平移
  也是；有限加性公共转动 **不是** 自动的 SE(3) gauge（已用单元测试钉死，
  与先前 layer rotation pivot 审计同类）；把一站清零却不改写其他站是不同
  物理约束。求解用的 common-mode 图是 `T_i' = G^{-1} T_i`，不是分量相减。
- 旧配置省略 `alignment_formulation` 时行为与历史逐位一致

文档：`docs/four_station_alignment.md` 与 `docs/four_station_alignment_cn.md`。

## 单元测试（阶段 3）

`tests/test_four_station.py`：

- 20 自由 + 4 survey schema
- SE(3) 往返、左乘 gauge 保持 `ΔT_ij`
- 加性平移是 gauge、有限加性转动不是
- S0/S3 reference 图与 common-mode 图在 `ΔT_ij` 上一致
- 「只把 S0 写成 0、不改写其他站」**不是** gauge
- 旧 multidof plan 仍拒绝移动 S1–S3
- 四站 plan 可独立写入四站，并拒绝把 S0 静默当成 reference
- compiler 在 nominal 线性化并对 s0..s3 全部分量出 FD
- `parse_transform` 接受 0–3 站独立六矢量

## 阶段 4 生产计划（提交前预注册，尚未跑）

| 项 | 值 |
| --- | --- |
| 目的 | 四站 FD smoke / identifiability，不训练模型 |
| source | train：`mc24_100043_00200_00299`、`mc24_100044_00300_00399`（μ−/μ+，已验证） |
| 事件 | 每源 50（smoke；需要 source spread 故用 2 源而非 1 源） |
| 点 / 源 | 1 nominal + 48 FD（24 参数 × ±）+ 2 held-out = **51** |
| 作业 | 每源 1 个 Condor 作业，共 2 |
| 内存 / flavour | 6000 MB / `tomorrow`（与条目 47 同量级；点×事件 ≈ 1.4× 条目 33） |
| 产物根 | `outputs/mc24_four_station_identifiability_pilot_v1/` |
| 禁止 | validation 选阈、密封 test、V2 重训、20D curriculum |

Held-out：`closure_relative`（S1 dx / S2 dy / S3 ry，S0=0 只是坐标图）与
`closure_relative_plus_common`（同一相对量 + 公共 dx）。只用于事后
`ΔT_ij` 比较，不在打开结果前改判据。

完成后必须检查：`failure.json`、ROOT 可读、content audit、manifest
completion。损坏的转换 ROOT 先核对 Athena `enhanced_tracklets.root`。

## 冻结判定

- 旧 IFT-only 默认路径：测试证明未静默改变
- 四站 20D：尚未准入，等待 SVD
- 冻结 V2：本阶段不评价 association
- 密封 test：未打开

## 下一步

1. 提交上述 2 源 × 51 点物理生产（阶段 4）
2. `audit_four_station_identifiability.py` 出 rank / SVD / 站对响应 /
   gauge 不变量
3. 只根据证据决定准入的相对子空间，再写 physical closure 计划（阶段 5）
