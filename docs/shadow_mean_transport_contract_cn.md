# 生产 vs 影子 Mean 传输契约（Stage B / Task B14ZA）

Workbook 126。进线冻结是 **WB125**，不是 WB124。WB125 已经消掉
common-grid FD 的 accepted-step 分支噪声，并且 path length 与生产
对齐，但独立影子的名义 mean 仍复现不了生产 EigenStepper mean。
本任务**完全停止 derivative / Jacobian 认证**，只问：同一磁场、
同一表面能量损失、同一 supporting plane、同一名义 path，为什么
函数值还对不齐。

```
WB125:
common-grid 去掉 FD 分支噪声
但影子 mean ≠ 生产 mean
        ↓
WB126:
停止导数
先重建生产确定性 mean
先把函数本身算成同一个函数
        ↓
shadow_mean_contract_established
        ↓
下一本 workbook 才允许
用已认证的 mean shadow 做独立 common-grid FD
```

Required hops 冻结：

```
target 1: hit 6
target 2: hit 11
target 3: hit 11
```

禁止查看 Jacobian 符合度。禁止从 WB125 残留 FD 结果回调 20/10/5 mm
网格。允许一份**预先注册**的 mean-only 序列 `10, 5, 2.5, 1.25 mm`，
依据是 RKN4 阶数和 FASER 磁体网格尺度，不是 Jacobian。

## 源码支持的生产 mean

冻结 ACTS 32.0.2 / AthenaExternals 24.0.41：

- 场更新是 **RKN4 / Nyström**（`EigenStepper.ipp` +
  `GenericDefaultExtension.hpp`），不是经典 RK4。
- 每个 accepted step 之后显式 `normalize()` 方向。
- 场区间上 `q/p` 不变（`kQoP = 0`）。
- `MaterialInteractor` 在到达表面之后运行：`evaluateMaterialSlab`
  （起点 `PostUpdate`，终点 `PreUpdate`，中间 `FullUpdate`），再用
  `pathCorrection` 缩放 slab，然后
  `evaluatePointwiseMaterialInteraction`，最后 `updateState` **只改
  q/p**。
- 体积材料只记录，不进入 ACTS mean。

## 正式结果

```
decision = surface_energy_loss_semantics_mismatch
verdict  = FAIL
```

Shadow P（用生产 accepted-step 序列重放官方 RKN4）在**第一个表面
材料节点之前**与生产的位置、方向对齐。按物理事件对齐后，第一个
明显分叉永远是 `surface_material`：`Δpos ≈ 0`、`Δdir ≈ 0`，裂开的
是 `q/p`。

T1 第一块 slab 上，生产 `ΔE` 落在同一块（已经 path-corrected）slab、
同一个记录 `q/p` 上算出的 `computeEnergyLossMean` 和
`computeEnergyLossMode` **之间**。单位 internally 自洽（`q/p` 为
1/GeV，Eloss 为 GeV，厚度为 mm，muon 质量 `0.1057 GeV`）。后续有的
记录表面生产 `Δq/p = 0`，但 `computeEnergyLossMean` 非零。

方向 normalization 有源码支持，解释不了残差。mean-only
`10/5/2.5/1.25 mm` 序列平台化，不是积分分辨率问题。

冻结旗标保持：

```
control 0/1/37 PASS = true
mean_path_unchanged = true
focus_independent_reference_established = false
jacobian_contract_established = false
b14m_reopen_authorized = false
```
