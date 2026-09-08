# Workbook 113: Task B14P Target-independent 弱参数 prior 可接受性审计

日期：2026-09-07
状态：**完成 / FAIL** —— 对 WB112 已识别的弱方向做 prior admissibility audit。未找到可用于真实数据、且与幸存 LTO measurements 独立的物理 prior。未引入 prior。未把 prior 加进 WB109 Cin。未删除或固定 `q/p`。未删 `100043/37`。未进 B14M 实施，未进 B15，未进 V4 C/D，未进 Measurement Model V2。未改 WB96–WB112 结论。未覆盖 WB109 dumps。

**最终判定：`FAIL`**

- `decision = target_independent_prior_not_available`
- `primary_case = target_independent_prior_not_available`
- `active_mechanisms = B`
- `prior_contract_established = false`
- `prior_introduced = false`
- `lto_cin_contract_established = false`
- `b15_authorized = false`
- `b14m_authorized = true`（只授权下一步，本 workbook **不**进入 B14M）
- `transport_covariance_validated = false`
- `measurement_model_v2_authorized = false`
- `q_over_p_deleted = false`
- `q_over_p_fixed = false`
- `focus_identity_retained = true`
- `do_not_force_5d_lto_covariance = true`

注意：`prior_contract_established != lto_cin_contract_established`。本 run 两者都是 false。

## 起始状态

HEAD：`32c9044a8543845aaa0767d8089ba6fd731f31a9`

WB109 PASS：`leave_target_out_state_materialization_established`  
Decision SHA：`af7ccd055c2ff3d34f92429e37be19a89ed90a9b0db11ec6770c161c79b6e933`

WB110 FAIL：`lto_input_covariance_shape_not_validated`  
Decision SHA：`eba85c1835e90f85a1ca0aec4ee896cc1b611214f9213c7602e160cbd291857b`

WB111 DIAGNOSED：`mixed_or_inconclusive`，`B + D`  
Decision SHA：`c9d35002671327dd401e1ca67bd17cb7dacfadf85efcb53e15af6d94dfc2a644`

WB112 DIAGNOSED：`reduced_measurement_supported_state_with_weak_nuisance`  
Decision SHA：`015087449a030232cfd088458b394ae82c409bae9bc26827fc2c944ec8c80dd8`

冻结方向分类（不得改写）：

```
x      = measurement_dominated
y      = weakly_measured
tx/φ   = prior_dominated
ty     = measurement_dominated
q/p    = weakly_measured / explicit nuisance
five_d_state_physically_supported = false
```

WB103 冻结 denominator 保持：raw 2680 / ineligible 691 / contracted 1989 / official pairs 1974。

本任务只问：这些弱方向有没有合法的 target-independent prior？不默认“有”。

## 1. 弱方向需要什么 prior

`x` / `ty` 已是 measurement-dominated，不额外加强 prior。

| 方向 | WB112 类 | 相对无信息 seed 的对角增益中位 | 方向性 seed 仍主导 σ | 为什么需要 prior | prior 会动均值？ | 必须 real-data 可获得 |
| --- | --- | ---: | --- | --- | --- | --- |
| y | weakly_measured | 3.33×10⁴ | 是 | 有 hits，但宽度仍跟 y-prior | 否（主要是 uncertainty） | 是 |
| tx / φ | prior_dominated | **1.04** | 是 | measurements 几乎不提供 φ 信息 | 否 | 是 |
| q/p | weakly_measured | 1.73×10⁶ | 是 | 相对无信息 seed 被压窄，但仍强烈跟 seed；传播关键 | 会 | 是 |

q/p 不得删除、不得设成 0、不得钉死成某个 seed。

## 2. Measurement identity 依赖图

不按探测器名称声称“独立”。用正式 WB109 LTO dump 的 `used_station_ids` 对 1989 条 contracted 行计数：

| 排除 target | N | 含 station 0 | target 漏进 used | 主导 used set | 未使用的其它 tracker 站 |
| ---: | ---: | ---: | ---: | --- | --- |
| 1 | 663 | 658 | **0** | {0,2,3} | 无 |
| 2 | 663 | 658 | **0** | {0,1,3} | 无 |
| 3 | 663 | 658 | **0** | {0,1,2} | 无 |

合计：1974/1989 行的 source IFT 已经在 `L_surviving_measurements` 里。15 行没用 station 0，但那也不能变成独立动量计。被排除的 target 是唯一不在该 LTO likelihood 里的 tracker 站；用它就破坏 leave-target-out。

系统图：

- station 0：磁铁前，不能测 q/p；对三个 target 通常已在 likelihood 中
- station 1/2/3：互为对方的幸存测量，不是外部 prior
- spectrometer magnet：在 0–1 之间，不是测量
- calorimeter / timing / veto：不在 LTO measurement set；MIP 缪子没有事件级动量
- 没有第二套独立 spectrometer
- full-track CKF：共享 target measurements，禁止

## 3. 候选 prior 逐项结论

11 个候选，**0 个 admissible**。`admissible_by_weak_direction = {y: false, tx: false, q/p: false}`。

### A. beam / production

| 候选 | 层级 | real-data | 结论 |
| --- | --- | --- | --- |
| MC particle-gun 能量 | population | 否 | 当前样本是 particle-gun；枪能量是 truth |
| LHC / FLUKA 谱 | population | 否 | 没有已注册的 real-data `p(q/p)`；用 FLUKA 是 simulation circularity；acceptance 仍依赖 geometry |
| 真实固定能量束 | event | 否 | FASER 碰撞/中微子缪子不是 test beam |

事件级 beam prior 不存在。总体谱即使将来要讨论，也不是某条 track 的独立测量，本任务不做 tuning。

### B. source-only IFT

可约束局部 `x,y,tx,ty`，**不能**独立测 q/p。1974/1989 行里这些 hit 已经在幸存 likelihood 中，再当 prior 是重复计数，不是独立信息。

### C. 独立 downstream

没有未使用的其它 tracker 站。量能器/时间/veto 不是动量谱仪。唯一谱仪就是 tracker+磁铁，也就是 LTO 测量系统本身。

### D. full-track CKF

冻结禁止：

```
full_track_CKF_qoverp = inadmissible
WB107_Cin = inadmissible
```

### E. empirical / closure

冻结禁止：truth residual、χ² closure、`0.1/1/10`、为过 gate 选宽度、任意高斯、当前 seed covariance。

## 4. Event-level vs population prior

二者没有混写。

```
n_admissible_event_level = 0
n_admissible_population_level = 0
```

如果将来有人想用 population `p(q/p)`，必须另外证明：来源、数据域、hyperparameters、会不会把 alignment 信号吸进 prior、以及是否独立于待估 geometry。本任务明确：这些都 **not_established**，并且 `could_bias_alignment_estimator = true`。

## 5. Likelihood 数学契约（未实例化）

形式上可以写：

```
L(θ) = L_surviving_measurements(θ) × π_independent(θ_weak)
θ = (x, y, tx, ty, q/p)
```

弱方向不得从 state 删除。chart 仍是 source 平面 native bound MeV。

本 run：

```
instantiated = false
admissible_prior_exists = false
prior_covariance_added_to_wb109_cin = false
```

没有把任何 prior 协方差加进 Cin，也没有宣称得到“新 covariance”。

## 6. Prior influence

没有 admissible candidate，因此 **不执行** 影响 smoke。禁止用任意高斯去“看看 posterior 会不会变”。

```
executed = false
reason = no_admissible_prior
```

## 7. Frozen catastrophic identities

`100043/37` 与 `100048/86` 全部保留。未降权、未特制 prior、未用 truth 替换。

| identity | target | n_fit | q/p gain | measurements 对 q/p 几乎无信息 |
| --- | ---: | ---: | ---: | --- |
| 100043/37 | 1/2/3 | 2 | 1.00 | 是 |
| 100048/86 | 1/2/3 | 4 | 1.00 | 是 |

如果这时强行塞一个 prior，posterior q/p 只会被 prior 接管，数值上看起来稳，实际是 `prior_dominated_latent_parameter`，不能写成 measured momentum。

## 分类

| Case | token | 本 run |
| --- | --- | --- |
| A | `target_independent_prior_contract_established` | 否 |
| B | `target_independent_prior_not_available` | **是。** 尤其 q/p 没有合法 independent prior |
| C | `prior_available_for_subset_only` | 否。y / tx 也没有独立于幸存 measurements 的 prior |
| D | `prior_bias_or_independence_not_established` | 否。不是“看起来可用但独立性未证”，而是明确不可用 |
| E | `mixed_or_inconclusive` | 否 |

下一步不是再找更好的 seed covariance，也不是重做 B14。下一步是：

```
Task B14M — Profiled / Marginalized Weak-Nuisance Measurement Likelihood
```

把 `q/p`、`y`、`tx` 留在 likelihood 里当 explicit nuisance，对它们 profile 或 marginalize，而不是假装已经有认证的 5×5 LTO Cin。B14M 仍须单独预注册，不是 Measurement Model V2。本 workbook **没有** 做 B14M。

未提交 HTCondor 重拟合：已有 dump + WB112 产物足够完成 A–E。

## 工程记录

Config SHA：`16f55a9249ee0a294b51ed604b055c906d31d57486bdf0051aed9deff3678f14`

WB112 decision SHA：`015087449a030232cfd088458b394ae82c409bae9bc26827fc2c944ec8c80dd8`  
WB111 decision SHA：`c9d35002671327dd401e1ca67bd17cb7dacfadf85efcb53e15af6d94dfc2a644`  
WB110 decision SHA：`eba85c1835e90f85a1ca0aec4ee896cc1b611214f9213c7602e160cbd291857b`  
WB109 decision SHA：`af7ccd055c2ff3d34f92429e37be19a89ed90a9b0db11ec6770c161c79b6e933`  
WB103 contract SHA：`e8c1f927e1ba03d418c2ed9b03084b198d09ba15196297a1a13cb5044ec48e3d`

Run ID：`sbb14p_target_independent_prior_20260907T123232Z_690debf3`

| 产物 | SHA256 |
| --- | --- |
| `target_independent_prior_decision.json` | `a01bb5490c346241f8282f0eb617a088b85aff535a4a72bfd2a44ff4b0931ecd` |
| `weak_parameter_prior_requirements.json` | `15852cfbe3bf142db03be20bffc3b5c33d86987004df794dc1f6e74409bb0c24` |
| `target_independent_prior_inventory_v2.json` | `79e17810e0d6913f22319bf669bb00aa3dfdc978c87dc2647515bae16d514c87` |
| `prior_level_semantics.json` | `f041d8b1151b41752f13924962f490a869292e151230ac558a148cab098d5b5c` |
| `lto_measurement_prior_likelihood_contract.json` | `03e97211d23be39d3ff38be528c70921c9c316f3287e2e6f81e506a4f2901060` |
| `prior_influence_falsification.json` | `5bb2e4786ae1cbdda0d97887a70330b3edb170a5a2e8bfe511e024d06483114a` |
| `focus_identity_prior_diagnostic.json` | `7630265a0ca88fd2a65cdacb6b9147e7f27d880e42460e89e86772323fc31ab1` |
| `inherited_stage.json` | `105f803a52c82eac75700cb36b084609ee4e47bb07da1a417332cca5eb743308` |
| `COMPLETE.json` | `0ec52948f387c81114269dabfd4ffa0c7185eebcf3687a9be2520b11edbbb9ee` |

测试 7 passed。无 B15，无 V4 C/D，无 Measurement Model V2，无 alignment，无 ML，无 prior 引入。

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
prior_introduced = false
measurement_model_v2_authorized = false
transport_covariance_validated = false
```

主线：

```
WB112: 5D state unsupported
    ↓
WB113 / B14P: Case B
    实验上不存在合法的 target-independent q/p prior
        ↓
停止把 q/p 塞进 5D Gaussian propagation seed
        ↓
B14M: measurement likelihood
      + explicit weak nuisance
      + profile / marginalize
        ↓
只有新的 uncertainty contract 被验证
        ↓
B15 V4 C/D
```

不得用 MC truth spectrum、full-track CKF、任意高斯或 seed covariance 继续“修 Cin”。
