# 已认证 Mean 的共用网格独立导数契约（Stage B / Task B14ZC）

Workbook 128。冻结入口是 **WB127**。生产材料 mean 已经闭合：

```
computeEnergyLossBethe
  + evaluateMaterialSlab 门控
  + updateState p/E → q/p
```

场传播继续复用 official accepted-step RKN4。这些 mean 语义不再修改。
本任务只在 `100048/86` required hops `1/6, 2/11, 3/11` 上重新启用导数。

```
WB127 mean contract PASS
        ↓
WB128 用同一个已认证 mean 建独立 FD
        ↓
只有 C 自身收敛后才比较 A ≈ C（5%）
        ↓
jacobian_contract_established
        ↓
下一本才能重开 B14M smoke 和 restart invariance
```

认证顺序固定：

1. 每个 `+δ/−δ` 臂复用冻结的 nominal accepted-step / 材料表面 /
   Bethe 门控 / supporting-plane 分区。
2. certified-shadow FD ladder 自身必须在 `loc1/phi/q/p` 上收敛
   （`last-pair rel ≤ 5%` 且符号一致）。
3. 然后才比较 `C` 与 `A = WB123 repaired production tangent`。

不允许因为 A 稳定就跳过 C 自身收敛。不允许再引用 WB124 自适应
DOPRI5。不允许改 `h,h/2,h/4,h/8`。即使本跑 PASS 也不提交 1989。

## 正式结果

Official run：`sbb14zc_certified_mean_fd_20260909T165606Z_5bd09dfe`

```
decision = certified_mean_independent_reference_established
verdict  = PASS
shadow_mean_contract_established = true
focus_independent_reference_established = true
jacobian_contract_established = true
b14m_reopen_authorized = true
restart_invariance_authorized = false
full_sample_authorized = false
```

```
config SHA      91bbf89658082979ef92fb1ee56b8f32ee9133cfd091dfe5c01412425d942ab8
decision SHA    7f9d411c31d3eb94d63fb13e24f1ebd57b3095f0007039a5c3048704007141b1
helper SHA      0d58d4cfca59a8de2d798771b3d454757713348955411681854b0073404d32df
dump SHA        04eab2a75e84e46103c1edae75ed3cf014039a7780cf2919ebb5a3f9cc054631
```

Dump：`outputs/leave_target_out_dump_v1/b14zc_smoke/mc24_100048_00000_00049/ckf_leave_target_out_certified_mean_fd.jsonl`

每个 `+δ/−δ` 臂都复用了冻结的生产 accepted-step / Bethe 门控分区。C 自身在五列上都收敛（`last-pair ≪ 5%`，符号一致）。A 与 C 最差是 T1 `q/p` 1.20%，仍在冻结 5% gate 内。control `100043/0,1,37` 继续 PASS。官方 mean `Δloc0 = 0`。target leakage = 0。

下一本才能重开 B14M smoke 并重做 restart invariance。本跑没有提交 1989。
