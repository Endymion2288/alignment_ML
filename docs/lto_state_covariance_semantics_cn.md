# LTO 状态 / 协方差语义（Stage B / Task B14）

Workbook 110。WB109 已经物化了独立 leave-target-out 状态。
本任务只问：这些状态及其 5×5 协方差在 **propagation 之前**
是否具有正确的 uncertainty semantics。

比较对象是官方 full-track Cin（WB107）与新的 LTO Cin
（WB109/B13），使用冻结的 WB103 contracted identity 以及
construction/validation 划分。truth 只作诊断。

B14 **不**验证 transport covariance，也 **不**进入 V4 的 C/D
两臂。那是 Task B15。

## 比较内容

```
5×5 本征谱
条件数
q/p 不确定度
交叉相关
source-state 经验误差协方差
广义本征值
pencil
逐分量 pull
source / target 依赖
```

## 问题

1. WB105 中传播前 Cin overcoverage 是否在 LTO 下消失或明显减轻？
2. 官方 full-track smoothing / 共享测量泄漏是否确实是 shape
   mismatch 来源？
3. LTO 协方差是否仍然过宽或覆盖不足？
4. `100043/37` 的错误动量状态在 LTO refit 后是否仍在？

## 禁止

rescale、本征值裁剪、PSD 投影、经验标定、用 truth 调 fitter、
删除不漂亮事件、重定义 WB103 资格，以及 Measurement Model V2。

## 正式 token

只有 PASS 才允许：

```
decision = lto_state_covariance_semantics_established
lto_cin_contract_established = true
b15_authorized = true
```

即使 PASS 也必须保持：

```
transport_covariance_validated = false
measurement_model_v2_authorized = false
```
