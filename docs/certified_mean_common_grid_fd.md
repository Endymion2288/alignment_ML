# Certified-Mean Common-Grid Independent Derivative Contract (Stage B / Task B14ZC)

Workbook 128. Incoming freeze is **WB127**. The production material
mean is closed:

```
computeEnergyLossBethe
  + evaluateMaterialSlab gating
  + updateState p/E → q/p
```

Field replay stays official accepted-step RKN4. Those mean semantics
are not changed. This task re-enables derivatives only on
`100048/86` required hops `1/6, 2/11, 3/11`.

```
WB127 mean contract PASS
        ↓
WB128 independent FD of that same certified mean
        ↓
only if C converges, then A ≈ C at 5%
        ↓
jacobian_contract_established
        ↓
NEXT workbook only: reopen B14M smoke + restart invariance
```

Certification order is fixed:

1. Every `+δ/−δ` arm reuses the frozen nominal accepted-step /
   material-surface / Bethe-gating / supporting-plane partition.
2. The certified-shadow FD ladder itself must converge
   (`last-pair rel ≤ 5%` and sign-consistent) on `loc1/phi/q/p`.
3. Only then compare `C` to `A = WB123 repaired production tangent`.

Do not skip C self-convergence because A is stable. Do not reuse
WB124 adaptive DOPRI5. Do not retune `h,h/2,h/4,h/8`. Do not submit
1989 even if this book PASSes.

## Official result

Official run: `sbb14zc_certified_mean_fd_20260909T165606Z_5bd09dfe`

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

Dump: `outputs/leave_target_out_dump_v1/b14zc_smoke/mc24_100048_00000_00049/ckf_leave_target_out_certified_mean_fd.jsonl`

Every `+δ/−δ` arm reused the frozen production accepted-step /
Bethe-gating partition. C itself converged on all five columns
(`last-pair ≪ 5%`, sign-consistent). Worst `A` vs `C` is T1 `q/p`
1.20%, inside the frozen 5% gate. Controls `100043/0,1,37` remain
PASS. Official mean `Δloc0 = 0`. Target leakage = 0.

The next workbook may reopen B14M smoke and rerun restart
invariance. This run did not submit 1989.
