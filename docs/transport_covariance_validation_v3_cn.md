# Transport Covariance Validation V3（Stage B / Task B8）

Workbook 104。WB103 已冻结 `eligible_for_transport_validation`。
本任务问：在该契约输入下，当前 ACTS 传播协方差能否描述
target-surface 预测误差。

正式输入是 WB103 契约 CKF 状态 + WB96 物理 5×5 + WB98 C0 / C1 / Q
dump。闸门保持冻结。不调 C/Q，不删 `100043/37`，不重开 raw CKF，
不进入 Measurement Model V2 / alignment。

## 正式输入（B8.1）

资格谓词与 WB103 相同，不用残差、χ²、pull、truth、event 37、
动量差：

- ntuple long-track 等价
- 四个不同的重建 tracklet 站
- `TrackSegments >= 4`
- 有限 5×5 Cin，对角为正
- 有限带符号 q/p，非 dummy
- 有效 state surface

| 量 | 计数 |
| --- | --- |
| 见到的 raw dump 行 | 2680 |
| 不合格剔除 | 691 |
| 契约行 | 1989 |
| official pair | 1974 |
| construction / validation 行 | 1116 / 873 |

raw CKF 不是正式输入。`100043/37` 保留。

## 正式 MODEL1 指标（B8.2）

同一 WB98 dump，同一 C0 / C1 / Q，同一 WB81/WB87 闸门。

事件级 official pair：χ²/ndof 均值 25.76，中位数 0.115，
q/p pull RMS 0.84。不能只用混合均值下结论。

| split | pair | N | χ² 均值 | χ² 中位 | pencil | λ(C_emp, C_prop) | 95% 覆盖 | χ² 闸门 | 形状 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| construction | (0,1) | 370 | 13.13 | 0.359 | 21.50 | 0.006–0.070 | 0.732 | 失败 | 失败 |
| construction | (0,2) | 370 | 8.24 | 0.261 | 20.99 | 0.018–0.155 | 0.781 | 失败 | 失败 |
| construction | (0,3) | 370 | 111.39 | 0.295 | 20.17 | 0.030–0.144 | 0.768 | 失败 | 失败 |
| validation | (0,1) | 288 | 3.34 | 0.044 | 12.67 | 0.007–0.111 | 0.920 | 通过 | 失败 |
| validation | (0,2) | 288 | 1.59 | 0.046 | 12.06 | 0.009–0.094 | 0.931 | 通过 | 失败 |
| validation | (0,3) | 288 | 1.10 | 0.048 | 10.85 | 0.027–0.181 | 0.931 | 通过 | 失败 |

主体层：中位数 χ² ≪ 1；validation 的 whitened-χ² 通过。
尾巴层：`100043/37` 占混合 χ² 总和的 80.3%。
形状层：每一对都失败 pencil / generalized eigenvalue。
C_prop 系统性宽于 C_emp。

source 级：validation 均值 0.99–2.96；construction 除
`mc24_100043_00400_00499` 外为 3.3–11.8。该 source 均值 181.6，
因为含 event 37，中位数仍是 0.33。

## 冻结诊断事例 100043/37（B8.3）

合法契约 long track：四站、四 segment、21 测量，
CKF χ²/ndof = 1.42，reco ≈ 414 GeV，truth 诊断 ≈ 2.0 TeV，
`|q/p pull| ≈ 13.8`。transport χ² 在 (0,1)/(0,2)/(0,3) 为
2971 / 760 / 37120。保留。

construction 上对 37 的 leave-one-out 只作诊断，不是 cut：

| pair | 正式 χ² 均值 | LOO χ² 均值 | LOO pencil |
| --- | --- | --- | --- |
| (0,1) | 13.13 | 5.11 | 21.84 |
| (0,2) | 8.24 | 6.20 | 21.39 |
| (0,3) | 111.39 | 11.10 | 20.53 |

去掉 37 会降低 construction 均值，尤其是 (0,3)。pencil 和
广义本征值仍失败。因此 V3 是主体 overcoverage 的 mixture，
同时被真实 CKF 拟合尾巴污染。不是“主体已校准、只被 37 挡住闸门”。

## 协方差形状（B8.4）

pencil 方向几乎是纯 `x`。λ(C_emp, C_prop) 的每个本征方向都
overwide；最小的 λ 落在 `tx` / `ty`。validation 的 95% 覆盖
0.92–0.93，同时均值 χ² 已在 1–3，说明 C_prop 过宽而不是过窄。
禁止标量 rescale。

## 物质 / Q 诊断（B8.5）

`Q = C1 − C0` 是两次可能重新线性化的传播之间的增量。
63.8% 的契约行 Q 非 PSD。这不使 V3 失败，也不做 PSD 投影。

Frobenius 中位数：C0 = 27.4，C1 = 18.3，Q = 5.7。
material-on 不是 material-off 的简单加性膨胀。Q 随低动量增大
（中位数 16.5 vs 高动量 2.1），并依赖轨迹。只作诊断。

## 分类（B8.6）

| 情况 | 含义 | 本 run |
| --- | --- | --- |
| A | construction 与 validation 都通过冻结闸门 | 否 |
| B | 主体校准成立，合法 CKF 尾巴挡住总体闸门 | **次因**（event 37，占 χ² 的 80%） |
| C | 主体事件上 generalized eigenvalue / pencil 仍失败 | **主因** |
| D | 样本不足以区分 | 否（N = 1974） |

正式 run `sbb8_transport_covariance_v3_20260906T195338Z_e6c68984`。

```
verdict = NOT_ESTABLISHED
decision = transport_covariance_shape_not_validated
next_step = transport_material_uncertainty_diagnosis
measurement_model_v2_authorized = false
measurement_model_v2_entered = false
```

不要为了过闸门再收紧重建契约。不要缩放 C/Q。
下一步允许的是：在同一契约输入上做 transport / material /
state-uncertainty 诊断。
