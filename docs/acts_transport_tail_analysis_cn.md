# ACTS 输运尾巴 / uncertainty 分析（Stage B / Task B4）

Workbook 100。WB99 已把 closure 失败判为 `high_chi2_tail_dominated`。
本任务只问：mean χ² 是少量错误 track state 造成的，还是物理
uncertainty 模型本身不够。

使用 WB99 冻结的 top 1% / 0.1% 名单。不重新筛选，不删除，不 clip，
不放大 C，不 rescale Q，不用 Highland 替代 ACTS Q，不进入
Measurement Model V2 / alignment。

## 分析内容

- **B4.1** 把冻结尾巴标成重建异常（A）、输运异常（B）或几何/材料（C）。
  原始残差、C、Q 保持不动。
- **B4.2** reco p 对 truth p、同一 event 跨 pair 重复、construction /
  validation 比例。事件全部保留。
- **B4.3** 官方全样本 pull；去掉尾巴的主体只作诊断，不作为删选结果。
- **B4.4** `Q = C1 − C0` 的特征值。只诊断，不调 Q。

## 失败分类

| 情况 | 含义 | 下一步 |
| --- | --- | --- |
| 1 | 尾巴主要是错误 track state | reconstruction / association 质量控制 |
| 2 | 尾巴在材料或入射角上富集 | ACTS material diagnosis |
| 3 | 真实 uncertainty 模型不匹配 | 重新评估 MM V2 **输入**模型 |

即使判为 3，B4 也不进入 V2。

错误动量门为 `|log10(p_reco/p_truth)| ≥ 1`（一个数量级），与 WB99
已报告的尺度一致。`|q/p|` bin 保持冻结。

正式 run `sbb4_acts_transport_tail_analysis_20260906T182102Z_a19d9ada`
为 **Case 1**：冻结 1% 的 18/24 条是重建异常（主要是 `|q/p pull|≥5`，
加上十年错误 p 与跨 pair 重复）。下一步是 reconstruction / association
质量控制，不是调协方差。
