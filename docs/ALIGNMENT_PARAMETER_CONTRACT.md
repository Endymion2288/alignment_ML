# Alignment Parameter Contract

Native Stations payload:

```text
/Tracker/Align/Stations
[dx, dy, dz, rx, ry, rz]   mm / rad
active left-multiply  G = T * Rz * Ry * Rx
pivot = FASER global origin
```

This is `stations_global_origin_TRzRyRx`.  Reports may show mrad only at the
boundary (`1000 mrad = 1 rad`).

Absolute payload is the six-vector of `G`.  A left-composed increment is
`G_new = G_inc @ G_base`.  Euler angles are not added.

## Center pivot

```text
t_origin = t_center + (I − R) c
```

Jacobian / prior covariance must be transported with this map.  Changing the
pivot does not create new physical information.

## Legacy cluster-local Ry

`legacy_cluster_local_station_z_ry_opposite_sign`:

- Ry matrix has the opposite sine sign from Stations `Ry`
- rotation is about `(0, 0, station_z_mm)`, not the FASER origin

Do not copy this chart into Stations `ry`.

## Survey extract

`extract_alpha_beta_gamma` is Athena's `TrackerAlignDBTool` helper.  It is not
a general inverse of `T Rz Ry Rx` for mixed angles.  Nov-22 ingest stays
cross-check only; this contract does not rewrite those numbers.
