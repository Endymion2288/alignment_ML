# Workbook 112: Task B14S 物理上受 measurements 支持的 LTO track-state 契约

日期：2026-09-07
状态：**完成 / DIAGNOSED** —— 在排除 target station 后，逐方向确定剩余 measurements 实际支持哪些 track-state 自由度。未修 covariance。未从 `0.1× / 1× / 10×` 挑选生产 seed。未引入 prior。未删除 `q/p`。未删 `100043/37`。未进 B14P 执行，未进 B15，未进 V4 C/D，未进 Measurement Model V2。未改 WB96–WB111 结论。未覆盖 WB109 dumps。

**最终判定：`DIAGNOSED`**

- `decision = reduced_measurement_supported_state_with_weak_nuisance`
- `primary_case = reduced_measurement_supported_state_with_weak_nuisance`
- `active_mechanisms = B`
- `lto_cin_contract_established = false`
- `b15_authorized = false`
- `b14p_authorized = true`（只授权下一步契约，本 workbook **不**引入 prior）
- `prior_introduced = false`
- `transport_covariance_validated = false`
- `measurement_model_v2_authorized = false`
- `q_over_p_deleted = false`
- `focus_identity_retained = true`
- `seed_scale_selected_from_closure = false`

## 起始状态

HEAD：`32c9044a8543845aaa0767d8089ba6fd731f31a9`

WB109 PASS：`leave_target_out_state_materialization_established`  
Decision SHA：`af7ccd055c2ff3d34f92429e37be19a89ed90a9b0db11ec6770c161c79b6e933`

WB110 FAIL：`lto_input_covariance_shape_not_validated`  
Decision SHA：`eba85c1835e90f85a1ca0aec4ee896cc1b611214f9213c7602e160cbd291857b`

WB111 DIAGNOSED：`mixed_or_inconclusive`，`active_mechanisms = B + D`  
Decision SHA：`c9d35002671327dd401e1ca67bd17cb7dacfadf85efcb53e15af6d94dfc2a644`

冻结：

```
leave_target_out_state_materialization_established = true
lto_input_covariance_shape_not_validated
decision = mixed_or_inconclusive
active_mechanisms = B + D
lto_cin_contract_established = false
b15_authorized = false
transport_covariance_validated = false
measurement_model_v2_authorized = false
```

WB103 冻结 denominator 保持：raw 2680 / ineligible 691 / contracted 1989 / official pairs 1974。

本任务只问：target 被排除以后，这条轨迹实际上还能独立测出哪些运动学自由度。不追求 closure PASS，不寻找能过 covariance gate 的低维子空间。

## 1. 逐 target 的 state-information map

磁场只作为几何事实记录：spectrometer magnet 在 IFT/station 0 与 station 1 之间。**没有发明 field integral。**

每个 target 各 663 条 contracted 行。幸存站 hit 中位都是每站 6。`n_fit` 中位都是 14，`used` 中位都是 18。

| 排除 target | 幸存站 | 幸存 z [mm] | used 中位 | n_fit 中位 (min) | z-span 中位 [mm] | 是否穿过磁铁 | 0→第一下游站 [mm] |
| ---: | --- | --- | ---: | --- | ---: | --- | ---: |
| 1 | 0, 2, 3 | −1860.15, 1237.4, 2427.4 | 18 | 14 (2) | 4287.55 | 是 | 3097.55 |
| 2 | 0, 1, 3 | −1860.15, 47.4, 2427.4 | 18 | 14 (2) | 4287.55 | 是 | 1907.55 |
| 3 | 0, 1, 2 | −1860.15, 47.4, 1237.4 | 18 | 14 (2) | **3097.55** | 是 | 1907.55 |

分量来源（三个 target 共用，只换幸存站）：

| 分量 | 信息来源 | 几何退化 |
| --- | --- | --- |
| x | 幸存站上的 stereo / strip loc0 投影 | 一维 strip，不是二维像素 |
| y | 幸存站 stereo 组合 | 同上 |
| tx | 幸存站之间的 Δx/Δz lever arm | 去掉 target 3 时最大 Δz 从 4287.55 降到 3097.55 |
| ty | 幸存站之间的 Δy/Δz lever arm | 同上 |
| q/p | 必须穿过 0–1 之间的 spectrometer magnet，并保留弯曲 lever arm | source-only 会完全退化；当前三个 LTO 都不是 source-only |

去掉 target 1 并不取消磁铁：粒子仍要从 station 0 走到 station 2/3。去掉 target 3 缩短下游 lever，但不取消 0–1 磁铁段。三个 target 的 `n_fit≤2` 各 1 条，全部是 `100043/37`。

## 2. 方向性 seed-response Jacobian

预注册：五个方向 `x, y, tx, ty, q/p` 各自单独改 prior variance，尺度只有 `0.1 / 1 / 10`。禁止组合搜索，禁止按 truth / χ² 选最好。生产尺度保持 1。

`tx` / `ty` 在 helper 里缩放的是 source 平面 bound `φ` / `θ`，是代理，不是精确派生斜率。已写入 `seed_direction_bound_parameter` 与 `seed_direction_chart_note`。

B14S smoke：`outputs/leave_target_out_dump_v1/b14s_smoke/`，事件 `0, 1, 37`。135 行 = 3 事件 × 3 target × 5 方向 × 3 尺度，全部 `fit_success`。未覆盖 WB109 dumps（正式 dump 时间戳仍是 2026-09-07 00:15）。

有限差分：

```
∂ fitted_mean / ∂ log(seed variance_i) = (f(10) − f(0.1)) / (ln 10 − ln 0.1)
∂ fitted_σ     / ∂ log(seed variance_i)
```

典型事件（0/1，6 组）的 **自身方向** `own_σ` 相对 1× 的最大相对变化：

| 方向 | 中位 own_σ rel | >0.25 的组数 | 典型均值几乎不动 |
| --- | ---: | ---: | --- |
| x | 0.069 | 2/6 | 是 |
| y | 3.42 | 6/6 | 是 |
| tx (φ 代理) | 0.338 | 3/6 | 是 |
| ty (θ 代理) | 0.207 | 3/6 | 是 |
| q/p | **7.91** | 6/6 | 是 |

事件 0 / target 1 / 只改 q/p prior：

| 尺度 | n_fit | q/p [1/MeV] | σ(q/p) |
| ---: | ---: | --- | --- |
| 0.1 | 16 | −8.705e-7 | 1.93e-7 |
| 1 | 16 | −8.705e-7 | 3.65e-6 |
| 10 | 16 | −8.726e-7 | 3.09e-5 |

均值几乎不随 seed 变；`σ(q/p)` 跟着 q/p prior 走。这把 WB111 的各向同性尺度敏感性拆成了方向性结论：q/p 宽度是 prior-dominated / weakly measured，不是“挑一个尺度就能变成 measurement posterior”。

## 3. Posterior information gain

Seed 与 fitted 在同一 source 平面、同一 native bound MeV chart。解析生产 seed：

```
diag(1e4, 1e4, 2.5e-3, 2.5e-3, 1e-6)
```

`I_seed = C_seed^{-1}`，`I_fit = C_fit^{-1}`，`ΔI = I_fit − I_seed`。广义特征值来自 `I_fit v = λ I_seed v`。派生 chart 的 `seed_derived_covariance` 只在 B14S smoke 上存在；全样本官方比较保持 native bound MeV。无法可靠求逆则标 `unavailable`，不硬算。

1989 条 contracted 中 1988 条可比较。不可用的 1 条是 `100047/76` target 3：native 最小本征值 −2.09×10⁻²³，不是 SPD，标 `unavailable`。

典型 1979 条（去掉 9 条 seed-like / `n_fit≤2` / 预注册灾难 identity）对角增益比 `C_seed_ii / C_fit_ii`：

| native 分量 | 中位增益比 | p05 量级 | 含义 |
| --- | ---: | --- | --- |
| loc0 | 2.01×10⁶ | ≫4 | 相对无信息 seed，x 位置被压得很窄 |
| loc1 | 3.33×10⁴ | ≫4 | y 位置同样被压窄 |
| phi | **1.04** | ~0.8 | **几乎没有 data information** |
| theta | 2.42×10⁴ | >2 | 极角方向有增益 |
| q/p | 1.73×10⁶ | ≫4 | 相对无信息 seed，q/p 宽度被压窄 |

典型广义特征值中位：`1.38 / 5.47×10⁴ / 1.03×10⁵ / 2.31×10⁶ / 1.90×10⁷`。最小那个 λ≈1.38 对应几乎不被 measurements 更新的方向，与 phi 对角增益一致。

三个 target 的增益图案同类：都有一个 λ≈1 的弱方向，其余方向相对无信息 seed 增益很大。target 3 的 z-span 更短，但并没有单独变成“完全没有信息”。

关键区分（WB111 已经指出，这里定量确认）：

- 相对 **无信息 seed**，典型 q/p 有巨大 information gain；
- 相对 **方向性 seed 再缩放**，典型 q/p 宽度仍然强烈跟着 prior 走。

所以 q/p 不是 `unconstrained`（典型事件不是 gain≈1），也不是 `measurement_dominated`（seed 敏感性远超 0.25）。它是 **weakly_measured / explicit nuisance**。

## 4. 物理上支持的状态类别

预注册阈值，不用 truth / residual / χ² / Transport V3–V4 gate：

```
measurement_dominated: gain ≥ 4 且 own_σ rel ≤ 0.25
weakly_measured:       gain ≥ 4 且仍 seed-sensitive
prior_dominated:       gain ≤ 1.25
unconstrained:         n_fit ≤ 2 且 gain ≤ 1.25
```

典型方向分类：

| 方向 | 类别 | 依据 |
| --- | --- | --- |
| x | `measurement_dominated` | native loc0 增益巨大；方向性 own_σ rel 中位 0.069 |
| y | `weakly_measured` | loc1 增益巨大，但 6/6 组 own_σ 仍随 y-prior 变 |
| tx | `prior_dominated` | native φ 增益中位 1.04 ≤ 1.25 |
| ty | `measurement_dominated` | θ 增益巨大；own_σ rel 中位 0.207 ≤ 0.25 |
| q/p | `weakly_measured` | 相对无信息 seed 增益巨大，但 own_σ rel 中位 7.91 |

`five_d_state_physically_supported = false`。

正确表述：

```
q/p must remain an explicit nuisance /
externally constrained latent parameter
```

禁止的表述：`q/p = 0`，或把 q/p 固定成某个 seed。本 run 没有删除 q/p，也没有把它钉死。

ty 在 6 组 smoke 里有 3 组 own_σ rel > 0.25，属于边界。分类用的是预注册的 **中位** own_σ rel，不是事后改阈值。x 同样有 2/6 超过 0.25，中位仍低于阈值。不因此改判。

## 5. Target-independent prior 盘点（未采用）

`prior_introduced = false`。没有任何候选被授权为生产 prior。

| 候选 | 可用 | 原因 |
| --- | --- | --- |
| beam / production momentum | 否 | MC particle-gun 能量不是 real-data prior |
| source-only IFT | 否 | station 0 不穿过磁铁，q/p 无约束 |
| 其他 upstream 独立重建 | 否 | 除 tracker+磁铁外没有独立动量谱仪 |
| full-track CKF q/p | 否 | 含 target measurement，禁止 |
| WB107 official Cin | 否 | 全轨 KF front state，禁止 |
| 从 closure 或 0.1/1/10 反推宽度 | 否 | 禁止当作生产值 |

本任务只盘点。不引入 prior，不进入 B14P 实施。

## 6. 灾难 fit provenance（不做 cut）

全部事件保留。`rejection_cut_designed = false`。

正式 contracted 里 seed-scale `σ(q/p)=0.001` 的 9 行，q/p information gain 全部 ≈ 1：

| identity | target | used | n_fit | stations | χ² | q/p [1/MeV] | σ(q/p) | q/p gain |
| --- | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: |
| 100043/37 | 1 | 16 | **2** | 0,2,3 | 3.76e-7 | −2.416353439070118e-6 | 0.001 | 1.00 |
| 100043/37 | 2 | 16 | **2** | 0,1,3 | 3.76e-7 | −2.416353439070118e-6 | 0.001 | 1.00 |
| 100043/37 | 3 | 15 | **2** | 0,1,2 | 3.76e-7 | −2.416353439070118e-6 | 0.001 | 1.00 |
| 100047/19 | 1 | 18 | 4 | 0,2,3 | 0.54 | −8.27e-7 | 0.001 | 1.00 |
| 100048/31 | 1 | 18 | 4 | 0,2,3 | 0.005 | +2.59e-7 | 0.001 | 1.00 |
| **100048/86** | 1 | 17 | 4 | 0,2,3 | **37.6** | −1.183e-5 | 0.001 | 1.00 |
| **100048/86** | 2 | 16 | 4 | 0,1,3 | **37.6** | −1.183e-5 | 0.001 | 1.00 |
| **100048/86** | 3 | 17 | 4 | 0,1,2 | **37.6** | −1.183e-5 | 0.001 | 1.00 |
| 100048/42 | 1 | 16 | 4 | 0,2,3 | 1.02 | +8.83e-6 | 0.001 | 1.00 |

`qoverp_gain_near_zero_coincides_with_seedlike = true`。

WB109 schema 没有 filtered / smoothed / outlier / residual dimension，这些量标 `unavailable`，不推测。`n_fit` 仍按 ACTS MeasurementFlag 计数。

Frozen focus `100043/37` 的方向性 smoke：三个 target 的 q/p **完全等于** 共同 seed mean `−2.416353439070118e-6 /MeV`；只改 q/p prior 时 `σ(q/p) = 0.001 × √scale`（0.000316 / 0.001 / 0.00316）；`n_fit=2`。这继续证明至少对该类事件 q/p 没有被 LTO measurements 实质约束。未删除、未特殊处理。

`100048/86` 是 WB111 的 y-tail 主导事件：`n_fit=4` 仍不足以给 q/p 信息，三个 target 的 LTO q/p 完全相同，χ²=37.6。灾难 pull 与 prior-dominated / unconstrained 的 q/p 方向重合。机制已解释，因此 **Case D 不再作为独立未解机制**。

## 分类

| Case | token | 本 run |
| --- | --- | --- |
| A | `full_5d_lto_state_measurement_supported` | 否。φ 几乎无增益；y 与 q/p 仍 seed-sensitive |
| B | `reduced_measurement_supported_state_with_weak_nuisance` | **是。** x / ty 可由 measurements 支持；q/p、y 是弱方向；φ/tx 是 prior-dominated |
| C | `lto_track_fit_information_insufficient` | 否。不是所有关键方向都没有稳定增益 |
| D | `catastrophic_fit_mechanism_unresolved` | 否。9 条 seed-like 行的 q/p gain≈0，与 37 / 86 重合 |
| E | `mixed_or_inconclusive` | 否。单一主导机制是弱 nuisance，不是多根因并列 |

下一步不是 B15，也不是直接重做 B14。下一步是：

```
Task B14P — Target-Independent Weak-Parameter Prior Contract
```

只允许用独立于 target measurement 的物理信息，给弱方向建立 explicit prior，再写

```
LTO likelihood = measurement likelihood × independent prior
```

然后重新执行 B14 covariance semantics。本 workbook **没有** 做这一步。

未提交 9-source HTCondor 重 dump：WB109 全样本 + 方向性 smoke 已足够完成 A–E。新 helper schema 只写在 `outputs/leave_target_out_dump_v1/b14s_smoke/`。

## 工程记录

Config SHA：`665333f43df563838d4a4733892e8c8030f1c2fd8c62d6cab3070129fdabb88b`

当前 helper SHA：`f7051beb0578adb87f7718d8074405c3b34c6907ef9656043b3e5d2c7107ec78`  
（WB109 正式 dump 仍对应当时的 helper `46f30d1a…`。B14S 只新增默认关闭的 directional campaign；默认路径仍是单次 fit。）

WB111 decision SHA：`c9d35002671327dd401e1ca67bd17cb7dacfadf85efcb53e15af6d94dfc2a644`  
WB110 decision SHA：`eba85c1835e90f85a1ca0aec4ee896cc1b611214f9213c7602e160cbd291857b`  
WB109 decision SHA：`af7ccd055c2ff3d34f92429e37be19a89ed90a9b0db11ec6770c161c79b6e933`  
WB103 contract SHA：`e8c1f927e1ba03d418c2ed9b03084b198d09ba15196297a1a13cb5044ec48e3d`

Run ID：`sbb14s_lto_supported_state_20260907T073101Z_77373fd6`

| 产物 | SHA256 |
| --- | --- |
| `lto_supported_state_contract_decision.json` | `015087449a030232cfd088458b394ae82c409bae9bc26827fc2c944ec8c80dd8` |
| `lto_state_information_map.json` | `48c10a4545a86211b0d8e972becc6bc1394c79d1a13ca824398c201f3787079f` |
| `lto_directional_seed_sensitivity.json` | `b40a944ad5b754d3deeaf899a49e6809b9b6eb6d75b8f61ac729f5de91f0136b` |
| `lto_information_gain.json` | `d5297780a4742a47e9a4398a89e21e77244345efd6098087bb8ef69126ed1b3a` |
| `lto_supported_state_contract.json` | `c6250396c85024312c15f6162e056bb95c9b92c82832f54f2874c5f510330a0c` |
| `target_independent_prior_inventory.json` | `4407937a247b0e79464f74352d9b834960c498b352a68bd1b382786a1b1210ba` |
| `lto_catastrophic_fit_provenance.json` | `f0187a4b354ab6dc49155cf7e5ed3ec8a5f8e9ec98b48b0686e3f2c00e738849` |
| `inherited_stage.json` | `4b31d0d2176f07c1f67cd0c8a45ae0086eb9d2b127de891cf4eb8c031cedc40c` |
| `COMPLETE.json` | `87707f9a8ece5bdfccea62bdca46499082e9c1d51f1008542f0b1fef1a08da42` |

测试 8 passed。无 B15，无 V4 C/D，无 Measurement Model V2，无 alignment，无 ML，无 prior 引入。

Provenance 与 WB98 冻结一致：

- geometry `4e965f3631dd4ff6e04b141ac36efc49603a1dfc0625de5ea7558dd929486b15`
- field `60432de5864cfba361f91f47a694dc111bbb06dcec44434e9682548680460e9c`
- material_map `0df37bd621e6caac2e223fda12fec32a6f8c49685c7b831181b956e59996433c`
- conditions `d30733216d6eb392dbdbcf5a252fed01ff56a045ade59731ca8419febce7a63d`

## 下一步

保持冻结：

```
b15_authorized = false
lto_cin_contract_established = false
measurement_model_v2_authorized = false
transport_covariance_validated = false
prior_introduced = false
```

主线：

```
WB109: LTO state established
WB110: LTO covariance semantics FAIL
WB111 / B14R: mixed (B + D)
    ↓
WB112 / B14S: Case B
    5D LTO 不是 measurement-supported posterior
    x / ty 可由 measurements 支持
    y / q/p 是 explicit weak nuisance
    φ/tx 是 prior-dominated
        ↓
B14P: target-independent weak-parameter prior contract
        ↓
LTO likelihood = measurement likelihood × independent prior
        ↓
重新做 B14
        ↓
只有新 B14 PASS
        ↓
B15 V4 C/D
```

不得用 0.1× / 1× / 10× 里任何一个当作新的生产 seed。不得 rescale / clip / inflate LTO Cin。不得删除 `100043/37`。不得把 q/p 删掉或设成 0。
