# 2026-08-19 (31) mode-3 matched retraining 联合判定：domain-shift 对照决定性成立，route purity/fake 存在张力

## 任务

条目 30 的 pilot 在 mode-3 数据上直接用冻结 mode-0 模型，route efficiency 崩塌
（0.74 → 0.12），无法区分"mode-3 本身有害"与"分布失配"。本条目执行**严格 matched
retraining 验证**：mode-3 candidate dataset 与 mode-0 共享全部 synthetic event
membership、overlay seed、candidate endpoint 与 physical payload point（逐比特一致，
见下文验证链），唯一差异 `q_over_p_mode=3`；4 个模型（MLP+route、V1 full-context、
V1 no-context、V2 BCE route-query）以相同 architecture/epoch budget/optimizer/
candidate graph/route solver 在 mode-3 数据上从头重训（禁止迁移 mode-0
checkpoint/calibration/threshold）；train 训练、validation 负责 early stopping/
calibration/threshold/unmatched penalty，sealed test 全程未触碰。不扩 Rx/Rz/dz、
不进 6-DoF。

## 数据同一性验证链（全部通过）

- **物理 bank**：v3 108/108、iter1 144/144 point 完成 Condor production；
  `verify_mode3_bank_identity.py` 判定双 bank 为 mode-0 bank 的逐比特超集
  （`verify_bank_v3.json` / `verify_bank_iter1.json` all_identical）。
- **overlay**：v3 12/12、iter1 16/16 sample 与原 overlay 逐比特一致
  （`verify_overlay_v3.json` / `verify_overlay_iter1.json`）。v3 复现需要
  `--legacy-seed-identity`（原 v3 overlay 用旧 magnitude-only seed 约定且按
  split 分两次 materialize，group_index 每 split 重启）；iter1 用现行约定即可复现。
  回归保护：`test_legacy_magnitude_identity_reproduces_v3_production_seed` 锁定
  v3 生产 seed = 876273998。
- **candidate coverage**：mode-0 与 mode-3 完全一致——eligible truth pairs 104450、
  propagation failed 449、overall truth-pair recall 0.99572（两侧相同）。

## 结果

### 1. Nominal primary 恢复（4 模型族，validation）——通过

| 模型族 | 指标 | mode-0 | mode-3 | Δ |
|---|---|---|---|---|
| MLP+route | AP / AUC / ECE | 0.8941 / 0.9778 / 0.1038 | 0.8842 / 0.9768 / 0.1064 | -0.010 / -0.001 / +0.003 |
| V1 full-context | AP / AUC / ECE | 0.8908 / 0.9498 / 0.0424 | 0.8865 / 0.9507 / 0.0440 | -0.004 / +0.001 / +0.002 |
| V1 no-context | AP / AUC / ECE | 0.9157 / 0.9718 / 0.0320 | 0.9094 / 0.9697 / 0.0358 | -0.006 / -0.002 / +0.004 |
| V2 BCE route-query | AP / AUC / ECE | 0.8706 / 0.9444 / 0.0552 | 0.8721 / 0.9463 / 0.0550 | +0.002 / +0.002 / -0.000 |

四个族全部恢复 nominal primary（|ΔAP| ≤ 0.01），candidate truth-chain recall 两侧相同。

### 2. Route metrics（canonical V2 backbone，iter1 anchor，validation）——效率大幅改善，purity/fake 轻度恶化

| arm | efficiency | purity | fake | chain recall | calAP |
|---|---|---|---|---|---|
| mode0-data + mode0-model | 0.7426 | 0.9669 | 0.0376 | 1.0 | 0.9620 |
| **mode3-data + mode3-model** | **0.9021** | 0.9499 | 0.0653 | 1.0 | 0.9562 |
| mode3-data + frozen mode0-model | 0.1157 | 0.9640 | 0.0404 | 1.0 | 0.9248 |

- efficiency **+16.0 pp**；purity **-1.7 pp**；fake **+2.8 pp**（同一 route solver、同一
  unmatched penalty -0.5）。
- V1 自身 validation route：full-context eff 0.8852 vs 0.9547（-7.0 pp，purity/fake 略优），
  no-context eff 0.8989 vs 0.9358（-3.7 pp）。MLP assoc eff 0.9615 vs 0.9578（+0.4 pp）。

### 3. Domain-shift control——决定性成立

冻结 mode-0 模型在 mode-3 数据上 efficiency 崩塌至 0.1157（Δ=-0.627），known-bad 边
median score 0.229；重训后恢复至 0.9021。**pilot 的效率崩塌来自训练分布失配，而非
mode-3 传播本身**。

### 4. Known bad 0→1 mismatch（run 996000，6 copies）——**更正：mode-3 下错配边更受偏爱（回退，非改善）**

**语义核查（2026-08-19 复审，只读）**：审计的六个 copy 就是条目 28 识别的那条坏边本身
（origin event 9000000300/12，tracklet 0@st0 → 1@st1，residual_x = -150.2 mm，
mode-0 下 chi2 = 12.78，mode-0/mode-3 重训双臂均 6/6 route-selected），并非它的
truth counterpart。审计输出中的 `is_truth: true` 只是 MC 标签（两端点同属
truth particle 10001，match fraction 1.0）——这正是该边的病理所在：一条
truth-标注但运动学上不可用的边（-150 mm 残差，条目 28 已因果验证：移除其来源源后
pooled dx 落到真值）。对该边，**score/rank 越高 = 坏边越受偏爱 = 回退**。

| arm | median calibrated score | median rank | rank-1 占比 | 6 copy route-selected |
|---|---|---|---|---|
| mode0-data + mode0-model | 0.8503 | 2.5 | 1/6 | 6/6 |
| mode3-data + mode3-model | **0.9173** | **1.0** | **4/6** | 6/6 |
| mode3-data + frozen mode0-model | 0.2288 | 3.0 | 0/6 | **0/6** |

正确解读：冻结 mode-0 模型在 mode-3 数据上把该边 chi2 从 12.8 推到 31.2（协方差收紧），
score 压到 0.08–0.40 并**全部拒选**；而重训后的 mode-3 模型反而把这条 truth-标注的
毒边学成了正例（score 0.90+、rank 1、6/6 选中）。**mode-3 重训在坏边处理上是回退，
不是改善。** 此前本节"实际改善"的结论作废。

### 5. Source-wise stability（8 源，backbone anchor）——更均匀但绝对水平略低

| arm | truth-consistent fraction min / max | spread |
|---|---|---|
| mode0-data + mode0-model | 0.9037 / 0.9631 | 0.059 |
| mode3-data + mode3-model | 0.8720 / 0.8957 | **0.024** |
| mode3-data + frozen mode0-model | 0.8955 / 0.9610 | 0.066 |

mode-3 源间离散度收窄 2.5 倍（source-dependent outlier 改善），但整体水平下移约 4 pp
（与 purity 下移一致：route solver 在 mode-3 下选出更多 complete route，单条平均
truth-consistency 略降）。

### 6. Multi-DoF closure（physical_edge_deduplicated，route-selected，station 0）——通过且更紧

| 参数 | mode-0 err (σ) | mode-3 err (σ) | mode-3 pull |
|---|---|---|---|
| ift_dx_mm | -0.0511 mm (0.0313) | **-0.0276 mm (0.0165)** | -1.67 |
| ift_dy_mm | +0.0396 mm (0.0402) | **+0.0076 mm (0.0299)** | +0.25 |
| ift_ry_mrad | -0.0159 mrad (0.0485) | **-0.0142 mrad (0.0125)** | -1.13 |

三参数全部 capture_success，绝对误差全面更小，σ 全面更紧（更物理的 covariance），
pull 均在 ±2 内。注意 normal-matrix condition number 1697 vs 108（mode-3 更病态，
后续 iteration 需观察）。

## 判定

预声明升级判据逐条核对：

1. validation 恢复 nominal primary —— **满足**（4 族全部 |ΔAP| ≤ 0.01，recall 相同）。
2. route metrics 至少不劣于 mode-0 —— **不完全满足**：canonical backbone efficiency
   +16 pp 大幅改善，但 purity -1.7 pp、fake +2.8 pp 轻度恶化；V1 自身 validation
   efficiency 亦下降 3.7–7.0 pp。
3. 更物理的 covariance/pull 与 multi-DoF closure —— **满足**（误差与 σ 全面更紧，
   pull 正常，capture 全过）。
4. 错配边/source outlier 实际改善 —— **不满足（更正后）**：bad-edge 语义核查表明
   mode-3 重训把 truth-标注的毒边学成正例（median rank 2.5→1.0 是回退）；源间
   spread 收窄 2.5× 是唯一留存的优势。
5. domain-shift control —— **决定性成立**。

**结论（2026-08-19 复审更正）：判据 1/3/5 成立，判据 2 存在实质张力（efficiency
大幅提升 vs purity/fake 轻度恶化），判据 4 不成立（bad-edge 回退）。不满足升级
门槛；mode-0 继续作为 canonical 与历史 control。**

条目 30 的替代解释"dummy q/p covariance 对 association 有正则化作用"获得新的支持
证据：mode-0 的病态 σ_y（数百 mm）恰好使这条 -150 mm 毒边的 chi2 落在 truth 的
50 分位（统计上不可识别），而 mode-3 收紧协方差后该边 chi2 升至 31.2——但重训把
它学成正例，抵消了协方差收紧的判别收益。domain-shift control 同时证明 pilot 崩塌
纯属分布失配。

## 后续（2026-08-19 复审决定）

执行 matched Pareto operating-point 复审（见条目 32）：validation-only、模型全冻结，
只扫描 route unmatched penalty 与 threshold grid，在等 efficiency / 等 purity/fake
下重新核对 canonical gate。

## 产物

- verdict 汇总：`outputs/mc24_mode3_matched_retraining_v1/mode3_matched_retraining_verdict.json`
- 重训模型：`outputs/mc24_v3_expanded_trainval_mode3_{mlp_control,v1_full_context_v1,v1_no_context_v1,v2_bce_control_v1}/`
- backbone 双臂：`outputs/mc24_mode3_matched_retraining_v1/backbone_mode3data_{mode3model,frozen_mode0model}_v2/iteration_01_anchor/`
- bad-edge 三臂审计、closure（`closure_mode3_dedup/`）、coverage 双审计、bank/overlay
  同一性验证 JSON 均在同目录。
- driver：`scripts/run_mode3_matched_retraining.sh`（stages: refresh-manifests →
  verify-bank → overlay → export → coverage → train-{mlp,v1-full,v1-nocontext,v2} →
  backbone → bad-edge → closure → verdict）。
