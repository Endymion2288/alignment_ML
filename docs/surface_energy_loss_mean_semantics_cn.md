# ACTS 表面能损 Mean 语义契约（Stage B / Task B14ZB）

Workbook 127。冻结入口是 **WB126**，不是 WB125。WB126 已经证明：官方 accepted-step RKN4 在第一个表面材料节点之前与生产位置/方向对齐，而生产 `ΔE` 落在 `computeEnergyLossMean` 与 `computeEnergyLossMode` 之间。本任务**不重开导数**。唯一问题是：生产 ACTS 32.0.2 `MaterialInteractor` 在 surface material 节点究竟用哪个确定性能损量更新 mean `q/p`，以及哪些 recorded material surfaces 真正调用了 `updateState`。

```
WB126:
表面能损语义不匹配
        ↓
WB127:
复制生产真实的材料 mean 更新
先闭合逐节点 q/p，再闭合 mean 函数
        ↓
shadow_mean_contract_established
        ↓
下一本 workbook 才能：
把这个已认证 mean shadow 重新用于 common-grid independent FD
```

Required hops 冻结为：

```
target 1: hit 6
target 2: hit 11
target 3: hit 11
```

不评估 Jacobian。不拟合 Eloss 去追 endpoint。不从最终 `loc0` 反推材料损失。场传播继续复用 WB126 已闭合的 official accepted-step RKN4。

## 编译路径上的生产语义

钉死 ACTS 32.0.2 / AthenaExternals 24.0.41 `libActsCore.so`：

```
MaterialInteractor
  → evaluateMaterialSlab
  → evaluatePointwiseMaterialInteraction
  → PointwiseMaterialInteraction::updateState
```

`evaluatePointwiseMaterialInteraction` **不在**已安装头文件里。编译体 `0x2afce0` 调用的是 `Acts::computeEnergyLossBethe`，并把这个 float 写入 `Eloss`。它**不**调用 `computeEnergyLossMean` 或 `computeEnergyLossMode`。

其余编译恒等式只作离线对照：

```
Mean = Bethe + Radiative
Mode = 0.9 * Landau + 0.15 * Radiative
```

这就是 WB126 里生产 `ΔE` 落在 Mean 与 Mode 之间的原因：生产用的是电离平均 Bethe，不是组合 Mean，也不是 Landau 型 Mode。

`updateState` 换算（影子原样复制）：

```
nextE = hypot(mass, p) - Eloss * navDir
nextP = max(10 MeV, sqrt(nextE^2 - mass^2) if nextE > mass else 0)
q/p   = copysign(|q| / nextP, q/p)
```

## 门控与材料指针不是一一对应

`evaluateMaterialSlab` 会改写 stage：

```
start surface  → PostUpdate
target surface → PreUpdate
otherwise      → FullUpdate
```

然后 `ISurfaceMaterial::factor(direction, stage)` 缩放 slab。默认 `splitFactor = 1`。正向 + `PreUpdate` 的 factor 是 `0`，因此 **target** 表面不会调用 `updateState`。`MaterialInteractor` 只在 slab 有效时记录表面相互作用，而这正好就是 `updateState` 运行的条件。体积材料会被记录，但不会应用到 ACTS mean。

WB126 在每个 `hasSurfaceMaterial` actor 周期都发事件，再消费下一条 recorded slab。被门控的表面因此生产 `Δq/p = 0`，而影子对贴上去的同一块 slab 强制 Mean 会非零。这是 recorder / updater 配对错误，不是第二条 Eloss 公式。

## 正式结果

Official run：`sbb14zb_surface_eloss_semantics_20260909T161125Z_be4e054a`

```
decision = surface_energy_loss_semantics_established
verdict  = PASS
shadow_mean_contract_established = true
production_eloss_quantity = computeEnergyLossBethe
```

```
config SHA      6b89a65070355d0143561d3017dbd451870ab74a7cbcf5fac77e9e1bd1ba27fa
decision SHA    aa80751fb532cf36350a8cf78b25f9d74da5bc707b9739ba7b683ef294fe04c4
helper SHA      b95e2416296feb241addd75a01284fb60e310932706781168d3f9bd11e83869d
mean header SHA 80c21c62d17ce27fb306465b84102ad93f7a266e7ba368554f613728ef16f516
dump SHA        7a454ee298d6e9612afb6868b19a454086f199f7171b69d504df59205aa86537
```

Dump：`outputs/leave_target_out_dump_v1/b14zb_smoke/mc24_100048_00000_00049/ckf_leave_target_out_surface_eloss.jsonl`

T1 第一块 slab（WB126 烟枪）：

```
production ΔE = 0.460932 GeV
Bethe = evaluatePointwise Eloss = 0.460934 GeV
Mean  = 0.624323 GeV
Mode  = 0.338027 GeV
```

生产匹配 Bethe，不匹配 Mean/Mode。被门控的表面是 `FullUpdate` 真空 / 零厚度 bin：`updateState` 不运行，影子也不再强制 Mean。WB126 里生产 `Δq/p=0` 而影子 Mean 非零，是 recorder / updater 配对错误。

复制 Bethe + `updateState` 门控，并修掉影子对最后一步场步进的双计（共同的 −6.113 mm）之后：

```
T1/6:  Δloc0=-3.07e-7 mm, Δpath=0, Δqop_rel=1.60e-7, nodes 7/7
T2/11: Δloc0=+1.70e-8 mm, Δpath=0, Δqop_rel=3.19e-8, nodes 5/5
T3/11: Δloc0= 3.55e-14 mm, Δpath=0, Δqop_rel=1.14e-8, nodes 1/1
```

T1/6、T2/11、T3/11 的逐节点 `q/p` **和** 预注册 mean endpoint gates 全部通过，因此 `shadow_mean_contract_established=true`。即便如此：

```
focus_independent_reference_established = false
jacobian_contract_established = false
b14m_reopen_authorized = false
```

下一本 workbook 才能把这个已认证 mean shadow 重新用于 common-grid independent FD。本跑没有评估导数。