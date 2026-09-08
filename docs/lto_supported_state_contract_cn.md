# 物理上受 measurements 支持的 LTO track-state 契约（Stage B / Task B14S）

Workbook 112。WB111 已经表明：导出的 5×5 是 source 平面上的
Kalman fitted state，但协方差仍随 seed 变，并且少数灾难拟合
主导 y-pull RMS。

本任务问更基础的问题：去掉 target station 以后，剩余
measurements 实际支持哪些运动学自由度？

**不**寻找能过 covariance gate 的低维子空间。**不**引入 prior。
**不**进入 B14P、B15、V4 C/D 或 Measurement Model V2。

## 问题

1. 分别去掉 target 1/2/3 后，哪些幸存站约束 `x, y, tx, ty, q/p`？
2. 只改某一个 seed-variance 方向（`0.1 / 1 / 10`）时，哪些方向
   仍然跟着动？
3. 在同一 source-plane bound chart 上，相对无信息 seed，哪些方向
   真正获得了 information gain？
4. 哪些方向必须保持显式 weak nuisance，而不能伪装成
   seed-independent 的 5D posterior？
5. 是否存在不使用 target measurement 的独立物理 prior？本任务
   只盘点，不采用。
6. `100043/37` 与 `100048/86` 是否落在 prior-dominated 方向上？

## 类别

每个方向只由预注册的 seed-sensitivity / information-gain 阈值决定，
不用 truth、residual、χ² 或 Transport V3/V4：

```
measurement_dominated
weakly_measured
prior_dominated
unconstrained
fit_failed
```

`weakly_measured` 表示相对无信息 seed 有信息增益，但拟合宽度仍
随 seed 变。这是弱 nuisance，不是删除该参数的许可。

## 分类

- A：五个方向都由 measurements 决定 → 重新做 B14
- B：部分方向受支持，部分明显弱 / prior-dominated
  → 下一步是 B14P，不是 B15；不得删除 `q/p`
- C：关键运动学方向都得不到稳定 information gain
  → 停止把当前 LTO fitter 当传播似然种子
- D：主体契约可建，但灾难拟合机制仍无法解释
- E：混合

方向性 seed 尺度只作预注册证伪。生产尺度保持 1。`tx`/`ty` smoke
轴缩放的是 source 平面上的 bound `φ`/`θ`，是代理，不是精确派生斜率。

## 正式标记

正式 B14S run 落在 Case B：

```
decision = reduced_measurement_supported_state_with_weak_nuisance
lto_cin_contract_established = false
b15_authorized = false
b14p_authorized = true
prior_introduced = false
```

典型类别：`x` / `ty` 为 measurement-dominated；`y` / `q/p` 为
weakly_measured；`tx`/`φ` 为 prior-dominated。`q/p` 保持显式
nuisance。本 workbook 不引入 prior。
