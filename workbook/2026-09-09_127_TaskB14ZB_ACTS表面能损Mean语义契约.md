# Workbook 127: Task B14ZB ACTS 表面能损 Mean 语义契约

日期：2026-09-09
状态：**完成 / PASS** —— 以 GitHub `master` `55cf982302a3c62c57b74f368d5e3ba7723fd33a` 为 checkpoint，以 **WB126** 为最新冻结。在**不改变** WB114 measurement likelihood、不改生产 `stepTolerance=1e-4`、不改变正式 `h,h/2,h/4,h/8` FD ladder、不增加 FD rung、不缩小 track-state FD 步长、不放宽冻结 5% gate、不回调 WB124 DOPRI5、不按 Jacobian 符合度挑选或改写影子网格、不重用已失败的 WB125 20/10/5 mm 配方、不把正式 sequential likelihood 换成 direct-from-source、不加 prior / ridge、不删 `100043/37` 或 `100048/86`、不用 truth q/p、不调 Q / Cin、不修改 `FieldGradientDefaultExtension` / field-gradient variational implementation / field map、**不评估也不读取任何 Jacobian**、不以修 5D Cin 为目标、**不为 endpoint 拟合 Eloss、不从最终 `loc0` 反推材料损失**的前提下，只从当前 AthenaExternals/ACTS 32.0.2 **实际编译源码和运行路径**确认生产 `MaterialInteractor` 用哪个 deterministic energy-loss quantity 更新 mean `q/p`，以及哪些 recorded material surfaces 真正调用了 `updateState`。然后只改 **shadow mean material semantics**，场传播继续复用 WB126 已闭合的 official accepted-step RKN replay。未重开 B14M，未做 restart invariance，未进 B15，未进 V4 C/D，未进 Measurement Model V2。未覆盖 WB109 / WB114–WB126 / `b14x_smoke` / `b14y_smoke` / `b14z_smoke` / `b14za_smoke` dumps。**未提交 1989 行 Condor**。禁止 analytic 自我认证。禁止把 target 2/3 已经收敛的 earlier hit 6 偷换成 required reference。禁止把真空 ODE 自动称为 production-map reference。

**最终判定：`PASS` / Case `surface_energy_loss_semantics_established`**

- `decision = surface_energy_loss_semantics_established`
- `primary_case = surface_energy_loss_semantics_established`
- `verdict = PASS`
- `smoke_gate_passed = true`
- `control_0_pass = true`
- `control_1_pass = true`
- `control_37_pass = true`
- `mean_path_unchanged = true`
- `target_exclusion_holds = true`（3 行，0 leaked）
- `shadow_mean_contract_established = true`
- `focus_independent_reference_established = false`
- `jacobian_contract_established = false`
- `b14m_reopen_authorized = false`
- `restart_invariance_authorized = false`
- `full_sample_authorized = false`
- `prior_introduced = false`
- `ridge_added = false`
- `statistical_model_unchanged = true`
- `five_percent_gate_unchanged = true`
- `derivative_not_evaluated = true`
- `jacobian_agreement_not_read = true`
- `production_eloss_quantity = computeEnergyLossBethe`
- `b15_authorized = false`
- `measurement_model_v2_authorized = false`
- `next_step = reuse_certified_mean_shadow_for_common_grid_independent_fd`

这是 **mean 函数闭合**，不是 Jacobian contract PASS，也不是 B14M PASS。T1/6、T2/11、T3/11 的逐节点 material `q/p` 和 WB125 预注册 mean endpoint gates **全部通过**，因此允许 `shadow_mean_contract_established=true`。后三个 `false` 旗标必须保持：下一本 workbook 才能把这个**已认证 mean shadow**重新用于 common-grid independent FD。本任务没有开任何 derivative。

不允许根据 API 名称猜 `Mean/Mode`。不允许为了 endpoint 好看直接拟合一个 Eloss。不允许从最终 `loc0` 反推 material loss。本轮结果是从生产 `MaterialInteractor → evaluateMaterialSlab → evaluatePointwiseMaterialInteraction → updateState` 前向复制得到的。

## 起始状态

HEAD：`55cf982302a3c62c57b74f368d5e3ba7723fd33a`

WB126 FAIL：`surface_energy_loss_semantics_mismatch`  
Official run：`sbb14za_shadow_mean_transport_20260909T151144Z_d24ca53b`  
Decision SHA：`f4d65c51ab90c04d318e935d116b419f22a06dc1e6006f4f821d6b150ed2a426`  
Config SHA：`297f8b47a0664bd130747403f36f72de508a3770d7018660cbfad110a02fd276`  
Helper SHA：`20f40424d7154cbcaa4fa3efc0dc7b1add2ec80be1d4a6d05ea463db5fd000f8`  
Dump SHA：`3930beaf31946fa493a50f14f82b62fc019dc9c2fbff2856cbe173d3922dead8`

```
decision = surface_energy_loss_semantics_mismatch
focus_independent_reference_established = false
jacobian_contract_established = false
b14m_reopen_authorized = false
control 0/1/37 PASS
mean_path_unchanged = true
next_step = keep_shadow_mean_transport_contract
```

WB126 已定位、本任务必须解释的事实：

```
Shadow P（官方 RKN4 on production accepted steps + computeEnergyLossMean）
在第一个 surface_material 之前 Δpos≈0、Δdir≈0；裂开的是 q/p。

T1 第一块 slab：
  slab_thickness = 266.505 mm
  path_correction ≈ 1.00011
  qop_before = -0.011841214
  production qop_after = -0.011906198     (ΔE ≈ 0.461 GeV)
  computeEnergyLossMean → Eloss = 0.624 GeV
  computeEnergyLossMode → Eloss = 0.338 GeV
  生产 ΔE 落在 Mean 与 Mode 之间

部分 recorded surfaces 生产 Δq/p=0，
而影子对贴上去的同一块 slab 强制 Mean 会非零。

三个 hop 共同 Δpath = -6.113 mm。
用户禁止在 material q/p 闭合之前把 path offset 当主因。
```

WB123 已冻结且本任务未改：

```
FieldGradientDefaultExtension.hpp SHA
ca5e4f0ef1a7a09edd1c24099e7ce689c4f0511e2d4afab3a160ebd2d6ff03d8
control 0/1/37 PASS
official mean Δloc0 = 0
```

冻结 likelihood 不得改写：

```
theta = (loc0, loc1, phi, theta, q/p)
alpha = (loc0, theta)
nu    = (loc1, phi, q/p)
chi2(θ) = Σ r_i(θ)^T R_i^{-1} r_i(θ)
R_i = (0.08 mm)² / 12
```

WB103 冻结 denominator 保持：raw 2680 / ineligible 691 / contracted 1989 / official pairs 1974。

预注册 mean endpoint gates（WB125，不回调）：

```
|Δloc0| ≤ 1e-3 mm
|Δpath| ≤ 1e-2 mm
|Δpos|  ≤ 1e-2 mm
|Δdir|  ≤ 1e-6
rel Δq/p ≤ 1e-4
```

逐节点 `q/p` 门：abs `1e-9` 或 rel `1e-6`（float Bethe vs double `updateState`）。

## 本任务做了什么

只跑 `100048/86`，required hops `1/6, 2/11, 3/11`。没有开 official supporting-plane Jacobian、没有开 field-gradient repair dump、没有开 WB124 DOPRI5、没有开 WB125 common-grid FD。`jacobian_validation` 在 mean-only 路径上被关掉。

1. 从 AthenaExternals 24.0.41 / ACTS 32.0.2 `libActsCore.so` 反汇编确认：`evaluatePointwiseMaterialInteraction` @ `0x2afce0` **只调用** `computeEnergyLossBethe`，把这个 float 写入 `Eloss`。它**不**调用 Mean 或 Mode。协方差路径上的 `computeEnergyLossLandauSigmaQOverP` 不改 mean `q/p`。
2. 编译恒等式（只作离线对照，不是生产 mean 更新）：
   ```
   Mean = Bethe + Radiative
   Mode = 0.9 * Landau + 0.15 * Radiative
   ```
   Mode 系数读自 `.rodata` `0x3784dc` / `0x3784e0`。
3. 逐层追踪 `MaterialInteractor → evaluateMaterialSlab → evaluatePointwiseMaterialInteraction → PointwiseMaterialInteraction::updateState`。`evaluateMaterialSlab` 改写 stage（start=`PostUpdate`，target=`PreUpdate`，否则 `FullUpdate`），再 `ISurfaceMaterial::factor`，再 `pathCorrection` 缩放厚度。`MaterialInteractor` **只在 slab 有效**（`material && thickness > 0`）时记录 **并且** 调用 `updateState`。体积材料只记录，不进 ACTS mean。
4. Dump 的生产 ledger 用 `MeanMaterialProbeActor`：构造 `PointwiseMaterialInteraction`，调用编译的 `evaluateMaterialSlab` + `evaluatePointwiseMaterialInteraction`，**不**自己调用 `updateState`。门控以探针的 `slabValid` / `updateStateCalled` 为准，不再用 WB126 的 `hasSurfaceMaterial || qopChanged` + 下一条 recorded slab 配对。
5. Shadow 材料更新改为 **Bethe + 同一套 `updateState` 门控 + 同一套 p/E→q/p 换算**。场传播仍是 WB126 official accepted-step RKN4。
6. 发现并修复影子 path 记账：`rkn4Step` 曾在 `Event ev = src` 之后把 `ev.sBefore = ev.sAfter`，于是 `sAfter = production_sAfter + h`，最后一步场步进（三个 hop 都是 ≈6.113 mm）被双计。删除该赋值后，caller 用运行 path 设 `sBefore`，`sAfter = sBefore + h`。这是 bookkeeping，不是物理 path offset。先闭合 material `q/p`，再处理这条 path。

## 正式产物

Official run：`sbb14zb_surface_eloss_semantics_20260909T161125Z_be4e054a`

```
config SHA      6b89a65070355d0143561d3017dbd451870ab74a7cbcf5fac77e9e1bd1ba27fa
decision SHA    aa80751fb532cf36350a8cf78b25f9d74da5bc707b9739ba7b683ef294fe04c4
helper SHA      b95e2416296feb241addd75a01284fb60e310932706781168d3f9bd11e83869d
mean header SHA 80c21c62d17ce27fb306465b84102ad93f7a266e7ba368554f613728ef16f516
dump SHA        7a454ee298d6e9612afb6868b19a454086f199f7171b69d504df59205aa86537
field-grad SHA  ca5e4f0ef1a7a09edd1c24099e7ce689c4f0511e2d4afab3a160ebd2d6ff03d8
```

Dump：`outputs/leave_target_out_dump_v1/b14zb_smoke/mc24_100048_00000_00049/ckf_leave_target_out_surface_eloss.jsonl`（3 行，无 Jacobian）

Path-fix 前的证据保留为：

```
ckf_leave_target_out_surface_eloss_pre_path_fix.jsonl
part_evt86_pre_path_fix.jsonl
```

当时三个 hop 已经 `Δloc0/pos/dir/qop` 低于 mean gate，只剩共同 `Δpath = -6.113 mm`，且最后生产场步长正好是 `6.113117 / 6.113357 / 6.113117` mm。

Artifacts：

```
production_mean_transition_ledger.json
shadow_mean_transition_ledger.json
surface_energy_loss_node_ledger.json
material_update_gating_contract.json
qop_energy_loss_unit_contract.json
required_segment_mean_failure_taxonomy.json
surface_energy_loss_mean_semantics_decision.json
inherited_stage.json
COMPLETE.json
```

`tests/test_surface_energy_loss_mean_semantics.py`：6 passed。

## 编译路径上的真实 Eloss

钉死 ACTS 32.0.2 / AthenaExternals 24.0.41 `libActsCore.so`：

```
MaterialInteractor
  → evaluateMaterialSlab
  → evaluatePointwiseMaterialInteraction   @ 0x2afce0
  → PointwiseMaterialInteraction::updateState
```

`evaluatePointwiseMaterialInteraction` **不在**已安装头文件里。编译体只调用 `Acts::computeEnergyLossBethe`，写入 `Eloss`。不能根据 `computeEnergyLossMean` / `computeEnergyLossMode` 的 API 名称猜测生产 mean。

`GenericDenseEnvironmentExtension` 的 Mean/Mode 开关属于 **dense volume stepper**，不是这条 surface `MaterialInteractor` 路径。

离线对照恒等式成立，并解释 WB126 的夹逼：

```
Mean = Bethe + Radiative
Mode = 0.9 * Landau + 0.15 * Radiative
```

厚 muon slab 上 Radiative 把 Mean 抬到 Bethe 之上，Landau 型 Mode 落在 Bethe 之下，所以生产 `ΔE`（Bethe）看起来“介于 Mean 与 Mode 之间”。这不是第三条未知公式，也不是 Vavilov 混合。

`updateState` 换算（影子原样复制，不是从 endpoint 反推）：

```
nextE = hypot(mass, p) - Eloss * navDir
nextP = max(10 MeV, sqrt(nextE^2 - mass^2) if nextE > mass else 0)
q/p   = copysign(|q| / nextP, q/p)
```

单位合同：

```
ACTS 内部 q/p = 1/GeV
Eloss 输入输出 = GeV
厚度 = mm
muon mass = 0.105658 GeV
updateState 最低动量 = 10 MeV
```

本事件 ~80 GeV muon，没有碰到 10 MeV cutoff。粒子假设全程 `muon` / `abs_pdg=13`。没有 particle-hypothesis 分支把 Mean/Mode 换进来。

## 门控：recorder 与 updater

`evaluateMaterialSlab` 先改写 stage：

```
start surface  → PostUpdate
target surface → PreUpdate
otherwise      → FullUpdate
```

然后 `ISurfaceMaterial::factor(direction, stage)` 缩放 slab。默认 `splitFactor=1` ⇒ 正向 `PreUpdate` factor = `1 - splitFactor = 0`，**target 表面不调用 `updateState`**。`MaterialInteractor` 只在 slab 有效时记录表面相互作用，而这正好就是 `updateState` 运行的条件。因此 **正式 recorder 与 updater 是同一谓词**；一个 `surfaceMaterial` 指针不够。

体积材料会被记录，但不应用到 ACTS mean。

本 hop 上观察到的门控节点**不是** target `PreUpdate`（那些根本不会进 MaterialInteractor 记录），而是 `FullUpdate` + `factor=1` + **零厚度 / vacuum bin**（`slab_valid=false`, `th=0`, `Δq/p=0`, Bethe=Mean=Mode=0）。

WB126 的“生产 `Δq/p=0` 而影子 Mean 非零”是 **ledger 配对错误**，不是第二条公式：当时每个 `hasSurfaceMaterial` actor 周期都发事件，再消费**下一条** recorded slab。被门控的表面没有 `updateState`，却被贴上后面一块真 slab 的 Mean。本轮探针按节点对齐后，`n_gated_where_mean_would_be_nonzero = 0`。

## T1 第一块 slab（WB126 烟枪，现已解释）

```
geometry_id     144115600661151744
surface_z       -1820.1075 mm
update_stage    FullUpdate
stage_factor    1.0
slab_valid      true
updateState     true
slab_thickness  266.505 mm
path_correction 1.000114
hypothesis      muon / pdg=13
qop_before      -0.011841214
p_before        84.4508 GeV
E_before        84.4509 GeV
production ΔE   0.460932 GeV
Bethe = Eval    0.460934 GeV
Mean            0.624323 GeV   (= Bethe + Radiative 0.163388)
Mode            0.338027 GeV   (= 0.9*Landau 0.348355 + 0.15*Radiative)
Landau          0.348355 GeV
Radiative       0.163388 GeV
production qop  -0.011906198
shadow qop      -0.011906199
node_qop_closed true
matches Bethe   true
matches Eval    true
matches Mean    false
matches Mode    false
```

生产匹配 Bethe 和 `evaluatePointwise` 的 `Eloss`，不匹配 Mean/Mode。残差是 float Bethe vs double `updateState`，rel ~ 2e-8，低于节点门。

## 逐节点 material 表（required hops）

### T1 / hit6 — 7 surfaces，5 次 `updateState`，2 次 gated

| # | geometry_id | z / mm | stage | upd | th / mm | ΔE / GeV | Bethe | Mean | Mode | node |
|---|-------------|--------|-------|-----|---------|----------|-------|------|------|------|
| 0 | 144115600661151744 | -1820.108 | FullUpdate | yes | 266.505 | 0.460932 | 0.460934 | 0.624323 | 0.338027 | yes |
| 1 | 576460889742377216 | -1758.899 | FullUpdate | no | 0 | 0 | 0 | 0 | 0 | yes |
| 2 | 720576077818233088 | -1630.899 | FullUpdate | no | 0 | 0 | 0 | 0 | 0 | yes |
| 3 | 1801439988655587328 | 7.358 | FullUpdate | yes | 14.931 | 0.007516 | 0.007521 | 0.007986 | 0.004349 | yes |
| 4 | 1801440126094540800 | 38.858 | FullUpdate | yes | 4.930 | 0.002627 | 0.002632 | 0.002952 | 0.001464 | yes |
| 5 | 1801440263533494272 | 70.358 | FullUpdate | yes | 235.785 | 0.280785 | 0.280786 | 0.365902 | 0.200328 | yes |
| 6 | 1945555176731443200 | 1197.358 | FullUpdate | yes | 29.831 | 0.032804 | 0.032805 | 0.042626 | 0.021314 | yes |

`n_match_computeEnergyLossBethe = 7`，`n_match_evaluatePointwise_Eloss = 7`。Mean/Mode 只在两个零厚度节点上“匹配”（三者都是 0）。

### T2 / hit11 — 5 surfaces，4 次 `updateState`，1 次 gated

| # | geometry_id | z / mm | stage | upd | th / mm | ΔE / GeV | Bethe | Mean | Mode | node |
|---|-------------|--------|-------|-----|---------|----------|-------|------|------|------|
| 0 | 1945555176731443200 | 1197.358 | FullUpdate | yes | 29.831 | 0.032804 | 0.032805 | 0.042626 | 0.021314 | yes |
| 1 | 1945555314170396672 | 1228.858 | FullUpdate | yes | 2.054 | 0.001201 | 0.001200 | 0.001331 | 0.000637 | yes |
| 2 | 1945555451609350144 | 1260.358 | FullUpdate | yes | 991.445 | 1.199668 | 1.199670 | 1.561950 | 0.907633 | yes |
| 3 | 2017612770500935936 | 1337.401 | FullUpdate | no | 0 | 0 | 0 | 0 | 0 | yes |
| 4 | 2089670364807299072 | 2387.358 | FullUpdate | yes | 4.571 | 0.002739 | 0.002740 | 0.003103 | 0.001529 | yes |

`n_match_Bethe = 5`，`n_match_evaluatePointwise = 5`。

### T3 / hit11 — 1 surface，1 次 `updateState`，0 次 gated

同一块 `1945555176731443200` / 29.831 mm / `ΔE=0.032804` / Bethe=0.032805 / Mean=0.042626 / Mode=0.021314。节点闭合。

没有 required hop 出现 `PreUpdate`/`PostUpdate` 记录：start/target 在 `splitFactor=1` 下 factor=0，正式 MaterialInteractor 既不记录也不更新。探针只列出带 `surfaceMaterial` 指针的 actor 周期。

## Path 记账（只在 material q/p 闭合之后处理）

Path-fix 前：`official_path == production_path`，`shadow_path = official_path + last_field_step`。三个 hop 的最后场步长正好是共同的 −6.113 mm。根因是 `rkn4Step` 把复制来的生产 `sAfter` 再加一次 `h`，不是物理 path offset，也不是终端 plane。

Path-fix 后：

```
T1/6:  official = production = shadow = 3026.023060574146 mm
T2/11: official = production = shadow = 2316.361515395455 mm
T3/11: official = production = shadow = 1126.221027383428 mm
Δpath = 0.0
first_divergence.found = false
```

不允许把这条 path 修复写成“为了 endpoint 好看而调 Eloss”。它只改影子运行 path 的加法，不改任何能量损失。

## Shadow P 残差（Bethe + 生产门控 + official accepted-step RKN4）

```
T1 / hit6:
Δloc0 = -3.07e-7 mm
Δpath = 0
Δpos  = 3.22e-7 mm
Δdir  = 4.25e-10
Δqop_rel = 1.60e-7
node_qop 7/7 closed
first_divergence = false

T2 / hit11:
Δloc0 = +1.70e-8 mm
Δpath = 0
Δpos  = 3.25e-8 mm
Δdir  = 6.77e-11
Δqop_rel = 3.19e-8
node_qop 5/5 closed
first_divergence = false

T3 / hit11:
Δloc0 = 3.55e-14 mm
Δpath = 0
Δpos  = 2.40e-13 mm
Δdir  = 2.67e-14
Δqop_rel = 1.14e-8
node_qop 1/1 closed
first_divergence = false
```

全部低于 WB125 预注册 mean gates。逐节点 `q/p` 也全部闭合。taxonomy 三个 hop 都是 `surface_energy_loss_semantics_established`。

对照 WB126（当时强制 Mean）：

```
T1 Δloc0 -0.01690 mm, Δqop_rel 3.10e-3, Δpath -6.113
T2 Δloc0 +0.00354 mm, Δqop_rel 4.54e-3, Δpath -6.113
T3 Δloc0  4.0e-10 mm, Δqop_rel 1.17e-4, Δpath -6.113
```

T3 当初 loc0 已经好看，挡住它的是 Mean 多出来的辐射项（rel `q/p` 刚好压过 `1e-4`）和双计 path。换成 Bethe 后两者一起消失。这不是从 T3 `loc0` 反推出来的。

## 合同条款（必须写死）

1. **生产 deterministic mean Eloss = `computeEnergyLossBethe`**。不是 `computeEnergyLossMean`，不是 `computeEnergyLossMode`，不是 Landau，不是 Vavilov 混合。Mean/Mode 只作离线对照。
2. **`updateState` 的充要条件**是 `evaluateMaterialSlab` 返回有效 slab。`surfaceMaterial` 指针、MaterialRecorder 事件、WB126 的 `hasSurfaceMaterial` actor 周期都不是充要条件。
3. **Stage gating**：start=`PostUpdate`，target=`PreUpdate`，中间=`FullUpdate`。默认 `splitFactor=1` 时正向 target 的 factor=0。本 hop 另外见到 FullUpdate + 零厚度 / vacuum bin。
4. **体积材料**记录但不改 ACTS mean `q/p`。
5. **p/E→q/p** 必须走 `updateState`：`hypot`、10 MeV floor、`copysign`。禁止从最终 `loc0` 或 hop endpoint 拟合 Eloss。
6. **场传播**继续复用 WB126 official accepted-step RKN4。本任务不改生产 stepper、不改 `stepTolerance`、不改 field map、不改 `FieldGradientDefaultExtension`。
7. **节点残差**来自编译 `float` Bethe 与 double 换算，不是第二条公式。门限 abs `1e-9` / rel `1e-6`。
8. **−6.113 mm** 是影子 path 双计最后一步，只在 material `q/p` 闭合之后处理。修复后 `Δpath=0`。

未出现、因此本事件不能当反例的项：momentum cutoff、particle-hypothesis 换公式、Bethe/Landau/Vavilov 在 mean `q/p` 上的运行时混合。协方差 Landau sigma 与 dense-volume Mean/Mode 开关都不在这条 mean 路径上。

## 分类

允许的六个 token 里，本跑是：

```
T1/6  = surface_energy_loss_semantics_established
T2/11 = surface_energy_loss_semantics_established
T3/11 = surface_energy_loss_semantics_established
primary_case = surface_energy_loss_semantics_established
```

没有 `surface_material_update_gating_mismatch`：门控节点生产 `Δq/p=0` 且影子也不更新。  
没有 `energy_loss_formula_or_particle_hypothesis_mismatch`：每个更新节点生产 = Bethe = evaluatePointwise。  
没有 `qop_update_conversion_mismatch`：同一 `updateState` 换算，节点 rel ≤ 1.6e-7。  
没有 `path_bookkeeping_mismatch_after_material_closure`：path 修复后 `Δpath=0`。  
没有 `mixed_or_inconclusive`。

## 冻结旗标

```
control 0 PASS = true
control 1 PASS = true
control 37 PASS = true
mean_path_unchanged = true
shadow_mean_contract_established = true
focus_independent_reference_established = false
jacobian_contract_established = false
b14m_reopen_authorized = false
```

后三个 `false` **即使本跑 mean PASS 也必须保持**。下一本 workbook 才能重新启用这个已认证 mean shadow 做 common-grid independent FD。本任务没有看任何 Jacobian 符合度，没有改冻结 5% gate，没有改正式 mean path。

## 下一任务

主线保持用户给定顺序，现已走完前两步：

```
WB126: 已定位 surface energy-loss semantics mismatch
  → WB127: 已复制生产 ACTS 的真实 material mean update
           逐节点 q/p 闭合，完整 mean 函数闭合
  → 下一本: 才能重新谈 derivative
           用已认证 mean shadow 做 common-grid independent FD
```

不要在下一本之前重开 B14M，不要提交 1989，不要把本跑误写成 Jacobian PASS。`next_step = reuse_certified_mean_shadow_for_common_grid_independent_fd` 是授权下一本开始 FD 的前提，不是本跑已经做了 FD。
