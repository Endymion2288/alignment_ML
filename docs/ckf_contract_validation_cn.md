# CKF 重建契约落地验证（Stage B / Task B7）

Workbook 103。WB102 已说明 transport validation 吃的是原始
`CKFTrackCollection`。本任务把 `eligible_for_transport_validation`
写成可复现谓词，并在同一套 WB98 dump、C0 / C1 / Q、闸门上重放
closure。

谓词独立于 closure。不用 χ²、q/p pull、残差或 truth。不调 C/Q，
不进入 Measurement Model V2 / alignment。

## 资格（B7.1）

一条 dump 行同时满足才算 eligible：

- ntuple long-track 等价（`longTracks > 0`）
- 站完备：四个不同的重建 tracklet 站（不用 `Track_InStation0`，
  它对几乎所有 long track 都是 0）
- segment 完备：`TrackSegments >= 4`
- 有限 5×5 输入协方差，对角为正
- 有限带符号 q/p，不是 dummy sentinel
- 有效 state surface（有限 z、站号 0–3、有限 5 维状态）

这些条目来自 WB102 契约文档，不是按 WB98 χ² 调出来的。

## 三个样本（B7.2）

| 样本 | dump 行 | 唯一 CKF | official pair | χ² 均值 | χ² 中位数 | q/p pull RMS |
| --- | --- | --- | --- | --- | --- | --- |
| A `raw_CKF` | 2680 | 895 | 2352 | 33.33 | 0.150 | 4.82 |
| B `contract_eligible` | 1989 | 663 | 1974 | 25.76 | 0.115 | 0.84 |
| C `reference_longTrack` | 2337 | 779 | 2094 | 24.64 | 0.116 | 1.15 |

互斥不合格原因：343 条不是 long track，348 条缺站。B 是 C 的子集：
116 条 long track 没有四个 tracklet 站。协方差本征值仍在同一 Cin
量级。残差只作诊断。

## Closure 重放（B7.3）

同一 WB98 dump，同一 C0 / C1 / Q，同一冻结闸门。只改输入范围。

official pair 事件 χ² 均值下降 22.7%（33.33 → 25.76）。这 **不是**
“filter 后 PASS”：

- `closure_under_contracted_input_scope = false`
- `covariance_model_fixed = false`
- construction MODEL1 每对仍失败；`100043/37` 让 (0,3) 均值保持 111
- validation χ² 均值降到 1.1–3.3，whitened-χ² 闸门在那里通过，但
  pencil 从约 5–11 变成约 11–21（更 overcover）
- 契约样本的每一对仍失败 pencil / generalized-eigenvalue 闸门

契约样本是物理一致的重建输入。它没有修好协方差模型。

## 剩余尾巴（B7.4）

WB99 冻结 1% 名单不重筛、不删除。契约之后仍剩 9 行 / 7 个身份。
唯一仍满足 `|q/p pull| ≥ 5` 的身份是 `100043/37`：

- long track，四站，四个 segment，21 个测量
- CKF χ²/ndof = 1.42
- reco 414 GeV vs truth 2.0 TeV，q/p pull ≈ −13.8
- transport χ² 在 (0,2) / (0,1) / (0,3) 为 759 / 2971 / 37120

这是 CKF 拟合 / 高动量 q/p，不是输入范围问题。

## 分类

| 情况 | 含义 | 本 run |
| --- | --- | --- |
| A | 契约后 closure 明显改善，主因是输入范围 | **主因**（均值降 22.7%） |
| B | 契约样本仍有少量 q/p 尾巴 | **次因**（`100043/37`） |
| C | 契约后仍整体失败 | 否（中位数 χ² 0.11） |

正式 run `sbb7_ckf_contract_validation_20260906T193419Z_6d906fda`。
下一步：Transport covariance V3，并用本契约作为输入范围。
不要用缩放 C/Q 去吞 event 37。
