# 条目 63 — source-disjoint four-station training-diversity audit

日期：2026-08-24  
分支：`4station`  
状态：条目 62 的 weighting / reduction / OP 已冻结停止。本条目只做 **train-side + 新 source** 的 diversity audit，并按预注册规则决定是否新增 source-disjoint 训练源。不训练、不改条目 62 的 objective / 权重 / reduction / margin / OP、不打开 15D WLS、不打开密封 test。条目 56 transfer 从现在起只做 development diagnostic，不再承担下一模型的最终独立 gate。  
密封 test：永久关闭。

## 本条目允许做什么

只回答：现有两条 train source 是否覆盖了 transfer 中导致稳定 `draw_01 + common` truth-utility drop 的 track phase space / source characteristics。比较对象是现有 train、已经看过的 development / transfer，以及候选新增 non-sealed source。不改条目 59/62 的任何 loss、weight、reduction 或 operating point。

## 冻结 convention

合同：`configs/physical_four_station_training_diversity_feasibility.yaml`  
清单：`configs/physical_curriculum_four_station_diversity_sources.yaml`  
控制号：`source_disjoint_training_diversity_audit_v1`  
checkpoint：条目 62 `a46a35bd28eb294fe307590d4f12595f6d3bfaa8dc64aea0bf7418543605e1ef`

| 量 | 冻结值 |
| --- | --- |
| objective | 条目 62 整体：dustbin-aware + gauge + max reduction |
| Platt / threshold / unmatched_penalty | identity / 0.001 / −1.0 |
| `Δ` | `U_truth - max(U_fragment, 0)`，margin = 1 |
| 物理 curriculum | 条目 53：15D relative + left-SE(3) twin，`≤0.5 mm / ≤5 mrad` |
| 条目 56 transfer | development diagnostic only |
| 预留 blind | `mc24_100047_00350_00399`、`mc24_100048_00350_00399`（本条目不加载） |

## 决策规则（打开数字之前冻结）

failure 切片：transfer 上 `iteration_00_draw_01_plus_common` 的完整 truth route，且 unselected、`Δ < 1`，或相对 `draw_01` twin 发生 truth-utility drop。

- 该切片的 `x/y/tx/ty` 落入 train 5–95% 的平均比例 `< 0.90`，或 S2/S3 同样 `< 0.90`，或 occupancy / charge 不覆盖：判 **`source_phase_space_undercoverage`**，授权新增 source-disjoint 训练源。
- phase space 过，但 `draw_01+common` 上 transfer 相对 train 的 hard rate 高 `>0.10`、fragment-winner rate 高 `>0.05`、`Δ` 中位低 `>0.20`，或 2→3 logit 中位低 `>0.50`：判 **`source_characteristic_undercoverage`**，同样授权扩源。
- 两项都过：判 **`current_train_covers_failure_region`**，不再靠增加同类 muon PG 文件救模型；下一步讨论 architecture-level relative / gauge-equivariant representation。
- 无论哪一支，条目 62 objective / OP 保持冻结；新的最终 gate 只能是从未进入 48–62 的 reserved blind pair。

## 运行

```bash
source scripts/setup_environment.sh ml
bash scripts/run_four_station_training_diversity_audit.sh
```

单测：`tests/test_source_diversity_audit.py` 7 项在审计前通过。  
产物：`outputs/mc24_four_station_hard_aware_reduction_v1/training_diversity_audit_v1/`  
预留 blind 与 unused reserve 未被加载。密封 test 未打开。

## 结果

数字只记录。没有因为某个 payload 差一点而改 loss 或 OP。没有生产新的 Athena overlay。15 维 WLS 未打开。

### Overlay（冻结条目 62 checkpoint）

| 量 | train | development | transfer（diagnostic） |
| --- | ---: | ---: | ---: |
| complete truth | 3331 | 3234 | 3381 |
| `draw_01+common` | 478 | 462 | 483 |
| failure core | 178 | 395 | **405** |
| fragment winners | 16 | 668 | **490** |
| `draw_01+common` hard rate（`Δ<1`） | 0.234 | 0.736 | **0.799** |
| `draw_01+common` winner rate | 0.010 | 0.160 | **0.170** |
| `draw_01+common` `Δ` 中位 | 1.104 | 0.675 | **0.413** |
| `draw_01+common` 2→3 logit 中位 | 2.185 | 2.036 | 1.793 |
| 2→3 logit q05 | 1.978 | −0.401 | **−0.487** |
| aligned `draw_01` twins / utility drop | 183 / 93 | 339 / 180 | 269 / 217 |
| charge μ− / μ+ | 1567 / 1764 | 1656 / 1578 | 1472 / 1909 |
| 事件 occupancy 中位 | 3 | 3 | 3 |
| 完整 route multiplicity 中位 | 2 | 2 | 2 |

Train 与 transfer 的 occupancy、charge、multiplicity **不是**失败原因。失败切片的 charge 两边都有（μ− 150 / μ+ 255），train 也有两侧。occupancy 落入 train 5–95% 的平均比例是 **0.967**。

### Phase space：failure core 对 train 的覆盖

transfer failure core = 405 条。落入当前两条 train source 5–95% 的比例：

| 坐标 | S0 | S1 | S2 | S3 |
| --- | ---: | ---: | ---: | ---: |
| `x` | 0.884 | 0.869 | 0.869 | 0.854 |
| `y` | 0.832 | 0.872 | 0.911 | 0.872 |
| `tx` | 0.923 | 0.941 | 0.825 | **0.716** |
| `ty` | **0.595** | 0.743 | **0.595** | 0.827 |

kinematic 平均 **0.821 < 0.90**。S2/S3 平均 **0.809 < 0.90**。最弱的是 `ty` 与 S3 `tx`。当前两条 train source **没有**盖住导致 failure 的角度 / S3 尾巴。

### Source characteristics：同一 overlay recipe + 冻结条目 62

| 差（transfer − train，`draw_01+common`） | 值 | 冻结上限 |
| --- | ---: | ---: |
| hard rate | **+0.565** | 0.10 |
| winner rate | **+0.159** | 0.05 |
| `Δ` 中位（train − transfer） | **+0.691** | 0.20 |
| 2→3 logit 中位（train − transfer） | +0.391 | 0.50 |

2→3 logit 中位差刚过上限，但 transfer 的 2→3 logit **q05 = −0.487**，train 仍是 +1.978。development（`100047/100048` 的 00050–00099）已经重复同一图案：hard rate 0.736、winner 668、2→3 logit q05 −0.401。所以这不是某一个被烧掉的 transfer chunk 的偶然，而是 **100047/100048 族相对当前两条 train source 的稳定 source characteristic**。

### 候选新增 non-sealed identity bank

只读 V3 `mag_0` identity，不把它们做成下一模型的 gate。`100043_00300` 的 S2/S3 `ty` 上沿明显更宽（S3 `ty` q95 0.023 vs 当前 train 100043 的 0.007）。`100044_00200` 的 S2 `ty` 下沿到 −0.020。`100047_00100` / `100048_00100` 提供与失败族相同的 DSID、不同事件区间，且 S3 `x` 更宽。这些是不同的原始 xAOD，不是同一 source 的 synthetic 重复。

预留 blind `100047_00350` / `100048_00350` 与 unused reserve `00800` **没有**被本条目加载。

## 决策

| 冻结条件 | 本条目 |
| --- | --- |
| 是否改 OP / 权重 / reduction / 架构 | **否** |
| 条目 56 transfer 是否仍是最终独立 gate | **否**（只做 diagnostic） |
| phase space | **不足**（`ty` / S3 `tx`） |
| source characteristics | **不足**（hard / winner / `Δ`） |
| 失败归类 | **`source_phase_space_undercoverage`** |
| 是否授权新增 source-disjoint 训练源 | **是** |
| 冻结条目 62 objective 整体 | **是** |
| `continue_to_15d_relative_wls` | **false** |
| 打开 15D WLS | **否** |

授权后的训练源（保持现有两条，另加四个不同原始 xAOD）：

- 保留：`mc24_100043_00200_00299`（μ−）、`mc24_100044_00300_00399`（μ+）
- 新增：`mc24_100043_00300_00399`、`mc24_100044_00200_00299`、`mc24_100047_00100_00149`、`mc24_100048_00100_00149`

新的最终 gate 只能是 reserved blind：`mc24_100047_00350_00399` + `mc24_100048_00350_00399`。闸门仍是 nominal purity ≥ 0.95、fake ≤ 0.05、全部 non-nominal `Δeff≤0.10`、gauge-twin 与 2→3/S3 stability，不得重调。

下一步只允许：按条目 53 的 15D relative + left-SE(3) twin、`≤0.5 mm / ≤5 mrad`、真实 `/Tracker/Align -> refit -> Acts(mode 0)` 为新增源做 physical + overlay，然后用**整份冻结的条目 62 objective**训练一次。若该冻结模型在新 blind validation 上通过，才把此前问题定为 training-domain coverage，并第一次授权 route-selected 15D `ΔT_ij`。若仍失败且继续表现为相同 common-SE(3) 下的 truth-utility systematic drop，下一步才讨论 architecture-level relative / gauge-equivariant representation，而不是再调现有 V2 的 loss 或 OP。
