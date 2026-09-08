# ACTS Transport Dump Helper（Stage B / Task B0）

Workbook 98。WB97 证明 Python 不能安全调用
`FaserActsExtrapolationTool::propagate`。本任务只在 `alignment_ML` 内增加独立
C++ helper。

不改 Calypso/FaserActs 源码，不在 PyAthena 里构造 Acts 对象，不进入
alignment / Measurement Model V2。

## 模型

- Model 0：关闭 MS/Eloss，`C0 = F Cin F^T`
- Model 1：打开 MS/Eloss，`C1 = C0 + Q_ACTS`（唯一通过模型）
- Model 2：Highland 仅诊断，不能替代 ACTS

`Cin` 是 WB96 的 CKF 5×5，导出到 `(x, y, tx, ty, q/p)`，`q/p` 带符号、单位
`1/MeV`。`F` 是无材料 propagate 的数值 Jacobian。`Q_ACTS := C1 - C0` 是 ACTS
的诚实增量，不是调出来的 scale。

## 失败分类

- 情况 A：`Q` 仍未物化 —— 继续修 transport export
- 情况 B：`Q` 已可见但 closure 失败 —— 这时才允许讨论物理 / MM V2
- PASS：`Q != 0`（不是人为 scale），construction/validation 通过冻结
  WB81/WB87 gate

WB98 正式 run `sbb0_acts_transport_dump_20260906T172303Z_9825e122` 是情况 B：
Q 已可见，closure 失败。未进入 Measurement Model V2。
