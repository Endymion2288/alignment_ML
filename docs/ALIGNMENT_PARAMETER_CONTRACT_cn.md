# Alignment 参数契约

Stations payload 原生定义：

```text
/Tracker/Align/Stations
[dx, dy, dz, rx, ry, rz]   mm / rad
主动左乘  G = T * Rz * Ry * Rx
pivot = FASER 全局原点
```

即 `stations_global_origin_TRzRyRx`。报告里的 mrad 只在边界换算。

绝对 payload 是 `G` 的六分量。左复合增量是 `G_new = G_inc @ G_base`，不能把
Euler 角直接相加。

## 中心 pivot

```text
t_origin = t_center + (I − R) c
```

Jacobian / prior 协方差必须一起变换。改 pivot 不是新的物理信息。

## 旧 cluster-local Ry

`legacy_cluster_local_station_z_ry_opposite_sign`：Ry 正弦与 Stations 相反，
绕 `(0,0,station_z_mm)` 转，不是 FASER 原点。不能裸复制为 Stations `ry`。

## Survey 提取

`extract_alpha_beta_gamma` 是 Athena 辅助函数，不是混合角下 `T Rz Ry Rx`
的一般逆。Nov-22 仍只作 cross-check，本契约不改写那些数。
