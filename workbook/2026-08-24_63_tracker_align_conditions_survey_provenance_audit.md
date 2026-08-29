# 2026-08-24 /Tracker/Align Conditions Dump & Survey-Provenance Audit V1

## 任务

条目 62 把 survey schema 准备好了，但还不知道 Calypso 现在实际读的 IFT
constants 是什么，以及它们能不能追溯到 2021 CERN survey / UNIGE
metrology。本阶段不训练、不 fit、不写 geometry，也不把 conditions
常数当成独立 survey。唯一目标是从真实 CVMFS/DBRelease dump
`OFLCOND-FASER-04/05/06` 的 `/Tracker/Align`，并做只读 provenance 审计。

## 答案

冻结 `existing_conditions_are_reconstruction_alignment_state_not_independent_survey`。

当前 TI12Data04 数据重建读 **CONDBR3** + `OFLCOND-FASER-06`。对 2024 run
14973，这解析到 `TRACKER-ALIGN-06` → `FASER-06_2024_Align.pool.root`。

1. **实际使用的 IFT constants（`existing_conditions_state`，不是 survey）**
   - `/Tracker/Align/Stations` station 0：`ry_cond = 0`，整个 6-DoF 是 identity。
   - 重建真正用到的 IFT 取向在 `/Tracker/Align/Planes`：L0/L1/L2 的
     `ry ≈ 75.36 / 74.16 / 74.39 mrad`，平均 **74.64 mrad**。
   - `dx_L0 / dx_L1 / dx_L2 ≈ 143.32 / 138.50 / 136.16 mm`，
     `C_dx_cond=(dx_L0-dx_L2)/2 ≈ 3.58 mm`。这个 `C_dx` 与
     `ry × |z_station0|` 同量级，是 global `/Planes` 存法的耦合，不是独立
     layer metrology。
   - `/Interface1/2/3` 有 8 个非零 local module transforms，x 为
     O(10–100 µm)，y 常到 0.2–0.9 mm。

2. **`-04 / -05 / -06` 怎么变**
   - 数据 CONDBR3：`-04 → TRACKER-ALIGN-04 → FASER-04_2022`（无 IFT
     planes / Interface members）；`-05` 按 run 切 2023/2024；`-06` 按
     run 切 2023/2024/2025。
   - `FASER-05_2023 == FASER-06_2023`，`FASER-05_2024 == FASER-06_2024`。
     2024 run 上两个 global tag 的 `/Tracker/Align` 是同一份 payload。
   - `FASER-06_2025` 的 IFT planes 回到 2023 那组，但 spectrometer
     module channels 已改。
   - **MC OFLP200 是旧的**：`-04/-05/-06` 都还指到 identity 的
     `FASER-02_Align.pool.root`。真实数据重建走 CONDBR3，不走这份 MC 映射。
   - 所有 dumped 文件的 Stations 通道都是 identity。官方 POOL 不是
     calypso-master 里空的 WriteAlignment 示例；DataHeader 生产者是
     `WriteAlignmentAlg.AlignDbTool`，但写入的是非零 constants。

3. **有没有足够 provenance 当独立 survey？没有。**
   - COOL tag description 为空，没有 author / input file / comment /
     release note。
   - 2021 IFT Survey & Metrology PPT 在 workspace / 用户目录 / CVMFS
     中未找到；未编造 PPT 上的四个 I/F 点。
   - 论文/公开幻灯片只作量级对照：CERN survey 角标度约 0.067 mrad
     （16 µm / 240 mm），conditions 里 ~75 mrad 大约大一千倍，不能因为
     “有个 ry” 就写成 survey。
   - Interface x 的 10–100 µm 可以记 `survey_origin_candidate`，但没有
     明确 provenance，不能升级为 `survey_derived`。Stations `ry=0` 记
     `survey_not_encoded_in_current_conditions`。非零 plane constants
     保持 `alignment_origin_unknown`。

报告：`outputs/tracker_align_conditions_survey_provenance_audit_v1/`。

下一步仍是拿到真正独立的 2021 CERN/UNIGE 表，按条目 62 schema 接入。
不要把这次 dump 出来的 `/Tracker/Align` 数字当作那条约束。
