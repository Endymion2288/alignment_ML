# Workbook 114: Task B14M Profiled 弱 nuisance 测量似然

日期：2026-09-07
状态：**完成 / FAIL** —— 从幸存 LTO measurements 建立 measurement-level likelihood，把弱参数留作显式 nuisance，并先做 **profile**。未引入 prior。未把 WB109 / WB107 Cin 当 likelihood。未删除或固定 `q/p`。未删 `100043/37` 或 `100048/86`。未做 marginalization。未进 B15，未进 V4 C/D，未进 Measurement Model V2。未改 WB96–WB113 结论。未覆盖 WB109 dumps。未提交全样本 HTCondor（helper 数值尚未稳定）。

**最终判定：`FAIL`**

- `decision = profile_likelihood_numerically_unstable`
- `primary_case = profile_likelihood_numerically_unstable`
- `active_mechanisms = D`
- `profiling_executed = true`
- `marginalization_executed = false`
- `marginalization_not_defined_without_prior = true`
- `prior_introduced = false`
- `prior_contract_established = false`
- `lto_cin_contract_established = false`
- `b15_authorized = false`
- `transport_covariance_validated = false`
- `measurement_model_v2_authorized = false`
- `do_not_force_5d_lto_covariance = true`
- `q_over_p_deleted = false`
- `q_over_p_fixed = false`
- `focus_identity_retained = true`
- `synthetic_profile_passed = true`
- `acts_profile_materialized = true`（仅 smoke）
- `target_exclusion_holds = true`

这仍然不是 Measurement Model V2。下一步是修 profile 数值实现，不是恢复 5D Cin，也不是进 B15。

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

WB113 FAIL：`target_independent_prior_not_available`  
Decision SHA：`a01bb5490c346241f8282f0eb617a088b85aff535a4a72bfd2a44ff4b0931ecd`  
`b14m_authorized = true`

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

本任务只问：surviving measurements 对 target-held-out prediction 能提供多少可认证信息。不再修 Cin。

## 1. Measurement likelihood 合同

状态用 source-plane native chart：

```
theta = (loc0, loc1, phi, theta, q/p)
```

Likelihood 来自幸存 hits，不是 WB109 fitted 5×5：

```
r_i = m_i - h_i(theta)
chi2 = sum_i r_i^T R_i^{-1} r_i
```

- `m_i`：LTO used 测量（target 站已排除）
- `R_i = (0.08 mm)² / 12`，与 CKF / LTO helper 相同
- `h_i`：FASERMagneticFieldWrapper + material（MS + eloss）的 ACTS propagate，再投到 wafer loc0

WB109 dump 对 1989 条 contracted 行：`n_target_leaked = 0`。

禁止项均未使用：truth residual、WB109 empirical cov、WB107 Cin、target 站测量、closure scale。

## 2. Supported / nuisance 划分

概念上冻结 WB112，但**没有截断状态，也没有把 nuisance 钉到 seed**：

```
alpha = (loc0, theta) ~ (x, ty)
nu    = (loc1, phi, q/p) ~ (y, tx, q/p)
chi2_prof(alpha) = min_nu chi2(alpha, nu)
```

耦合保留。`H_nn` 允许亏秩。

## 3. Profile 数值合同（synthetic）

预注册 `pinv_relative = 1e-8`。不是 alignment `rank_tolerance=0.01`。未加 ridge。

受控线性问题上：

| 方法 | 一致 |
| --- | --- |
| joint NLS | 是 |
| 显式 `min_nu` | 是 |
| Schur `H_aa - H_an H_nn^+ H_na` | 是 |

故意构造的亏秩 `H_nn`：rank 从 3 降到 2，被诊断出来，没有被 ridge 强行补满。

`synthetic_profile_passed = true`

## 4–7. ACTS smoke（login，不是全样本）

新 helper 开关默认关闭，不改 WB109 Kalman 路径。B14M smoke：

- `EnableProfileLikelihood=true`
- `ProfileOnly=true`
- 预注册 init：`nominal` / `loc1+1mm` / `phi+1e-3` / `q/p×1.1`
- 事件：`100043/0,1,37` 与 `100048/86`

WB109 official dump 未覆盖（size 仍为 1951622）。

| 样本 | 行数 | profile 成功 | 失败原因 |
| --- | ---: | ---: | --- |
| 100043 / 0,1 | 24 | 24 | — |
| 100043 / 37 | 12 | 0 | `measurement_propagation_failed`（step limit 10000） |
| 100048 / 86 | 12 | 0 | 11× propagate fail，1× jacobian fail |

成功行：target 排除成立；nominal / φ / q/p 扰动收敛到同一 χ² 和同一 `x_target`。`loc0` 变化 < 0.002 mm。

`loc1+1mm` **没有回到同一 χ²**（例如 event 0 / target 1：13.89 → 23.72），`x_target` 跟着偏约 1 mm。这不是“两个等价 measurement 极小”，而是 Gauss–Newton 没走完 profile。因此记：

```
optimizer_underconverged = true
```

而不是选择“最好 seed”，也还不能判 Case C。

Hessian 在成功行上典型 rank 3/5。null space 被保留；没有用 seed covariance 填，也没有把 pinv 零方向写成零 uncertainty。全样本 calibration 未执行（没有 1989 行 ACTS dump）。

未做 marginalization：没有合法 prior / measure。

## 8. Truth 诊断

Likelihood 构造未使用 truth。construction / validation 门未改。因为没有完整 profile dump，`profiled_prediction_calibration` 明确 `executed=false`。

## 9. 冻结灾难事例

### 100043/37

WB109：`n_fit=2`，q/p = seed。现在：16 个 surviving hits，但 source-plane → 各测量面的场感知 propagate 在 10000 step 处失败。**还不能回答 q/p profile 是否 flat**，因为 `chi2(theta)` 本身算不出来。未删除、未改 seed、未加 prior。

### 100048/86

WB109：`n_fit=4`，χ²=37.6，y-tail。现在：measurement-only `h_i(theta)` 从 official seed 出发同样 propagate 失败。先前的 y-tail **不能**只归因于 Kalman 5×5 实现；测量层 transport 目前也无法被数值求值。未删除。

## 10. Profiling vs marginalization

```
profiling = executed
marginalization = not executed
marginalization_not_defined_without_prior = true
```

## 最终决策为什么是 D 而不是 C 或 B

- A 不可能：全样本 calibration 不存在，uncertainty semantics 未建立。
- C 不成立：预测对 `loc1` init 的偏移伴随着**更大的 χ²**，说明还没到同一个 profile 极小。
- B 不成立：nuisance 虽像 flat，但 optimizer / propagate 尚未稳定到可以认证“observable 已识别”。
- D：synthetic 合同通过，但真实 ACTS profile 在 focus 事例上无法求值，并且 y 扰动 restart 欠收敛。先修数值，不加 ridge、不加 prior。

未提交 9-source HTCondor。不稳定 helper 的全样本 dump 不能代替数值修复。

## 官方产物

Config SHA：`de8876cd4b63c06f4e3fb7224f9eff7346cdb4e503e41b38b896c76ad3fb92b6`  
Helper SHA：`c83eba2038b30bd4d108e867be3e9445ee23987312d88101ab5bbb7a57da63bc`  
WB113 decision SHA：`a01bb5490c346241f8282f0eb617a088b85aff535a4a72bfd2a44ff4b0931ecd`  
WB112 decision SHA：`015087449a030232cfd088458b394ae82c409bae9bc26827fc2c944ec8c80dd8`  
WB111 decision SHA：`c9d35002671327dd401e1ca67bd17cb7dacfadf85efcb53e15af6d94dfc2a644`  
WB110 decision SHA：`eba85c1835e90f85a1ca0aec4ee896cc1b611214f9213c7602e160cbd291857b`  
WB109 decision SHA：`af7ccd055c2ff3d34f92429e37be19a89ed90a9b0db11ec6770c161c79b6e933`  
WB103 contract SHA：`e8c1f927e1ba03d418c2ed9b03084b198d09ba15196297a1a13cb5044ec48e3d`

Run ID：`sbb14m_profiled_likelihood_20260907T135010Z_f07d09cd`

Smoke：

- `outputs/leave_target_out_dump_v1/b14m_smoke/mc24_100043_00400_00499/ckf_leave_target_out_profile.jsonl` SHA `520f3beabed50cd80934da3e99b1b1ba4e6cc562d33b1de90279adc325c8d24e`
- `outputs/leave_target_out_dump_v1/b14m_smoke/mc24_100048_00000_00049/ckf_leave_target_out_profile.jsonl` SHA `ed01fec4fef18cdddb080bbbe950e129bef051ed86e7f433a8826da332c24bfa`

| 产物 | SHA256 |
| --- | --- |
| `profiled_measurement_likelihood_decision.json` | `82304f5af943f97adfe069e02c8bacdd7491cca72e6ecc58ea82cc470428a18f` |
| `measurement_likelihood_contract.json` | `b4ef14a010d026c434f8ed6356f38077d0ee48798d5e752f6346b4d6514b6069` |
| `profiled_state_partition.json` | `1dacbc46b3189cf5a32806ae86a22ad6c503625c9f8ac640e42c429da6c95006` |
| `profile_likelihood_numerical_validation.json` | `900606a8fbae82c316b2061c5bf0be11d024ac70b20ea40e08b5f4c55b6c1ff6` |
| `profile_nuisance_identifiability.json` | `bfa527e6400db333896328c588e444aff693466a40f5d19cc554601c869cec1d` |
| `profile_seed_invariance.json` | `dd8080523ed84cc85ba94af0f65e3ad9b5bb01a7cea236425988311b3afd73bb` |
| `profiled_target_prediction.json` | `bdefabe2b6e015d4db2cd23a1875e6ec8f167a2c76dff23c62bc732c8b625ed0` |
| `profiled_prediction_uncertainty_contract.json` | `06f07e305b0574160bf9eea2c51869628a8a98a9af201d1491a9bcf6ef3d85be` |
| `profiled_prediction_calibration.json` | `44e5fd16da8f7ab5f63a965ec0a2ccb10eeb203d0d7dbd31955e974ddc2c6b7c` |
| `focus_identity_profile_likelihood.json` | `18ac0ed0ef582fea2879b344cab8814eb8e12d8c152b5cbe2403d17d99f5ac2f` |
| `inherited_stage.json` | `92f66c6e024eb995c9ed399bb1a918eb8477b7274431a18e0aa378614a05cff5` |
| `COMPLETE.json` | `065021feccc0e31d8b076a7444765126e971f3da729e37957c201d3e505ffcca` |

测试 13 passed。无 B15，无 V4 C/D，无 Measurement Model V2，无 alignment，无 ML，无 prior，无 ridge，无 Cin 修补。

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
do_not_force_5d_lto_covariance = true
```

主线：

```
WB113: no legal independent prior
        ↓
WB114 / B14M: Case D
        measurement likelihood 合同已写下
        synthetic profile 正确
        ACTS profile 数值不稳定
        ↓
先修 propagate / Gauss–Newton
（不用 ridge、不用 prior、不删 37/86）
        ↓
再重做 seed-invariance 与 focus
        ↓
A / B / C 才有资格被判定
        ↓
只有 uncertainty contract 真正验证后
        ↓
再决定是否还需要 B15
```

不得用 MC truth spectrum、full-track CKF、任意高斯或 seed covariance 继续“修 Cin”。
