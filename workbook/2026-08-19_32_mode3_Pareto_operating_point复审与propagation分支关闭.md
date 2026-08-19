# 2026-08-19 (32) mode-3 vs mode-0 matched Pareto operating-point 复审：purity/fake 劣势稳定存在，propagation 分支关闭

## 任务

条目 31 的单点比较混合了两个变量：propagation mode 与 route operating point
（mode-0 冻结 penalty -1.0 vs mode-3 冻结 penalty -0.5）。本条目执行严格 matched
Pareto 复审：checkpoint、feature standardizer、calibration、candidate graph、V2 route
architecture 全冻结，只在 validation 上以 inference-time override 扫描 route
unmatched penalty（主轴）与冻结 per-pair threshold 的 scale（副轴），输出完整
frontier。不升级 canonical、不扩 Rx/Rz/dz/6-DoF、不训练新模型、不打开 sealed test。

前置更正（条目 31 已同步修订）：known bad 0→1 audit 的 score/rank 对应**坏边本身**
（residual_x -150 mm、truth-标注毒边），mode-3 下 score 0.8503→0.9173、median rank
2.5→1.0 是**回退**而非改善；verdict JSON 已加 semantics 注记防止误读。

## 方法

- 新 inference-time override（`run_frozen_association_backbone.py`）：
  `--unmatched-penalty-override` / `--threshold-scale-override`，逐点记录于输出
  `frozen_backbone.operating_point_override`；冻结 artifact 本身不变。
- 扫描：penalty ∈ {-2.5, -2.0, -1.5, -1.25, -1.0, -0.95, -0.9, -0.85, -0.75, -0.5,
  -0.35, -0.2, -0.1, 0.0} × 双臂（mode0-data+mode0-model on iter1 mode-0 overlay；
  mode3-data+mode3-model on iter1 mode-3 overlay），anchor payload，validation split。
  另加 threshold scale 10/100 探针（结果与冻结点完全相同——0.001 阈值在 100 倍内
  仍不构成约束，threshold 轴对本 frontier 退化）。
- 汇总：`scripts/evaluate_mode3_pareto_frontier.py` →
  `outputs/mc24_mode3_matched_retraining_v1/pareto/mode3_pareto_frontier.json`。
- 方向约定（实测确认）：penalty 越负越严格（route utility 须超过 |penalty|），
  efficiency 越低、purity 越高。

## Pareto frontier（anchor，validation）

| penalty | mode-0 eff / purity / fake | mode-3 eff / purity / fake |
|---|---|---|
| -1.50 | 0.1022 / 0.9947 / 0.0157 | 0.0573 / 1.0000 / 0.0114 |
| -1.25 | 0.4283 / 0.9851 / 0.0183 | 0.3688 / 0.9841 / 0.0243 |
| -1.00 | 0.7426 / 0.9669 / 0.0376 | 0.6955 / 0.9647 / 0.0407 |
| -0.95 | — | 0.7485 / 0.9624 / 0.0440 |
| -0.75 | 0.8875 / 0.9541 / 0.0508 | 0.8448 / 0.9559 / 0.0503 |
| -0.50 | 0.9243 / 0.9484 / 0.0584 | 0.9021 / 0.9499 / 0.0653 |
| -0.35 | 0.9275 / 0.9486 / 0.0686 | 0.9156 / 0.9469 / 0.0732 |
| -0.20 | 0.9319 / 0.9467 / 0.0785 | 0.9194 / 0.9429 / 0.0862 |
| 0.00 | 0.9292 / 0.9460 / 0.1001 | 0.9200 / 0.9461 / 0.1042 |

**同 penalty 下 mode-0 efficiency 全区间更高（+2~4.5 pp）**；条目 31 的
"+16 pp efficiency"完全来自 operating point 差异（-1.0 vs -0.5），不是 mode-3 的
真实增益。在 penalty -0.5 同点，mode-0 efficiency 更高（0.9243 vs 0.9021）、
fake 更低（0.0584 vs 0.0653），purity 基本持平（0.9484 vs 0.9499）。

### Matched comparison ①：等 efficiency（分段线性插值，含 -0.95/-0.9/-0.85 加密点）

| eff | mode-0 purity / fake | mode-3 purity / fake | Δpurity | Δfake |
|---|---|---|---|---|
| 0.70 | 0.9694 / 0.0350 | 0.9645 / 0.0409 | -0.49 pp | +0.59 pp |
| 0.7426（mode-0 冻结点） | 0.9669 / 0.0376 | 0.9627 / 0.0437 | -0.42 pp | +0.61 pp |
| 0.80 | 0.9618 / 0.0429 | 0.9575 / 0.0476 | -0.43 pp | +0.47 pp |
| 0.85 | 0.9574 / 0.0474 | 0.9554 / 0.0517 | -0.20 pp | +0.43 pp |
| 0.90 | 0.9521 / 0.0534 | 0.9501 / 0.0647 | -0.20 pp | +1.13 pp |

等 efficiency 下 mode-3 的 purity/fake 劣势**在全区间稳定存在**（purity -0.2~-0.5 pp，
fake +0.4~+1.1 pp）。missing-station recovery mode-3 略低（0.7426：0.9700 vs 0.9744）。
source-wise spread 在 eff ≤ 0.85 区间 mode-3 更小（0.7426：0.0352 vs 0.0595）——
源间更均匀是 mode-3 唯一留存的优势；0.90 处两者相当（0.0238 vs 0.0217）。

### Matched comparison ②：约束下最大 efficiency

| 约束 | mode-0 max eff | mode-3 max eff | Δ |
|---|---|---|---|
| fake ≤ 0.05 | **0.8786** | 0.8403 | -3.8 pp |
| purity ≥ 0.95 | **0.9139** | 0.9010 | -1.3 pp |

### 等效率 closure（physical_edge_deduplicated，station 0）

mode-0 @ eff 0.7426（冻结点，条目 27 产物）vs mode-3 @ eff 0.7485（penalty -0.95，
`closure_mode3_dedup_eff0743/`）：

| 参数 | mode-0 err (σ) | mode-3 err (σ) | mode-3 pull |
|---|---|---|---|
| ift_dx_mm | -0.0511 (0.0313) | **-0.0351 (0.0192)** | -1.83 |
| ift_dy_mm | +0.0396 (0.0402) | **+0.0071 (0.0305)** | +0.23 |
| ift_ry_mrad | **-0.0159 (0.0485)** | -0.0183 (0.0136) | -1.34 |

等效率下 mode-3 closure 的 σ 仍全面更紧（dx -39%、dy -24%、Ry -72%），dx/dy 误差
更小，Ry 误差略大但远在容差内，三参数 capture 全过、pull 均在 ±2 内。
**更物理的 covariance/pull 与更紧 closure 在等效率下依然成立。**

## Canonical gate 终审

预声明判据（条目 31 修订版 + 本 Pareto 复审）：

1. validation 恢复 nominal primary —— 成立（4 族 |ΔAP| ≤ 0.01）。
2. route metrics 至少不劣于 mode-0 —— **不成立**：等 efficiency / 等 purity/fake
   的 Pareto 比较中 purity/fake 劣势稳定存在，约束最大 efficiency 低 1.3–3.8 pp。
3. 更物理 covariance/pull、closure 更紧 —— 成立（等效率下仍成立）。
4. 错配边改善 —— **不成立**（bad-edge 回退：毒边被重训学成正例，rank 2.5→1.0）。
5. domain-shift control —— 决定性成立（pilot 崩塌 = 分布失配）。

**终审结论：维持 mode-0 为 canonical propagation mode。mode-3 保留为
physically-better-calibrated propagation branch（更合理 covariance/pull、更紧
closure、源间更均匀），但其 route-level purity/fake 劣势与 bad-edge 回退在
matched Pareto 上稳定存在，不满足升级门槛。本条 propagation 分支到此关闭，
返回 multi-DoF 主线，开始 Rx/Rz/dz sensitivity。**

机制注记：mode-0 的病态 σ_y 虽无物理意义，却使 -150 mm 毒边在 chi2 上不可识别
（truth 50 分位），配合 mode-0 训练分布形成了当前的 operating 平衡；mode-3 收紧
协方差后该边 chi2 升至 31.2，但重训把它学成正例，净效果是 route 层面并未获益。
"dummy q/p covariance 对当前 association 有正则化作用"作为明确负结果保留。

## 产物

- frontier 汇总：`outputs/mc24_mode3_matched_retraining_v1/pareto/mode3_pareto_frontier.json`
- 扫描点：`outputs/mc24_mode3_matched_retraining_v1/pareto/{mode0_data_mode0_model,mode3_data_mode3_model}/penalty_*/iteration_01_anchor/`
- 等效率 closure：`outputs/mc24_mode3_matched_retraining_v1/closure_mode3_dedup_eff0743/`
- 工具：`scripts/evaluate_mode3_pareto_frontier.py`；backbone override 参数
  （`--unmatched-penalty-override` / `--threshold-scale-override`）
