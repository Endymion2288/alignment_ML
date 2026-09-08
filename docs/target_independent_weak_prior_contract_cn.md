# Target-independent 弱参数 prior 契约（Stage B / Task B14P）

Workbook 113。WB112 已经表明 5D LTO 状态不是 measurement-supported：
`x` / `ty` 是，`y`、`tx/φ`、`q/p` 是弱方向或 prior-dominated。

本任务只问：这些弱方向有没有真正独立于 target measurement、可用于
真实数据、并且 uncertainty semantics 明确的 prior？

**不**默认答案是“有”。**不**把 prior 写进 fitter 或 WB109 Cin。
**不**进入 B14M、B15、V4 C/D 或 Measurement Model V2。

## 问题

1. 为什么 `y`、`tx/φ`、`q/p` 需要 prior？
2. 哪些候选来源真正 target-independent？
3. 候选是事件级测量 prior，还是只是总体谱？
4. 如果 prior 存在，`L_meas × π_independent` 应该是什么？
5. 如果没有任何合法 prior，就停止强行修 5D Gaussian seed。

## 可接受条件

候选必须同时满足：

- 不使用被排除的 target measurements
- 可用于真实数据，不是 MC truth / particle-gun 能量
- 与幸存 LTO likelihood 在 measurement identity 上不重叠
- 独立于待估 geometry
- 有明确的 uncertainty semantics

禁止：full-track CKF q/p、WB107 Cin、truth residual、χ² closure、
`0.1/1/10` seed 尺度、任意高斯。

source-only IFT 可以约束局部 `x,y,tx,ty`，但**没有磁铁就不能测
`q/p`**。这些 hit 已经在 `L_surviving_measurements` 里，所以不是
独立 prior。

## 分类

- A：所有弱方向都有合法 prior → 写下 likelihood 契约，再重做 B14。
  `prior_contract_established` 不等于 `lto_cin_contract_established`。
- B：没有合法 prior，尤其 q/p 没有 → FAIL，下一步是 B14M
  （profile / marginalize 弱 nuisance）。不再修 5D Cin。
- C：只有部分弱方向有合法 prior
- D：看起来可用，但独立性或 bias 未建立
- E：混合

## 正式标记

正式 B14P run 落在 Case B：

```
decision = target_independent_prior_not_available
prior_contract_established = false
prior_introduced = false
lto_cin_contract_established = false
b15_authorized = false
b14m_authorized = true
```

`y`、`tx/φ`、`q/p` 都没有合法的 real-data prior。本 workbook
不进入 B14M。
