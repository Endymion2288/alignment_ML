"""Machine-locked FaserFieldTable_v2 interpolator and unit/sign contract.

Coordinates are millimetres.  The Calypso field service returns kT.
This module converts to Tesla with the ACTS wrapper factor 1000 and
trilinear-interpolates exactly as BFieldMesh / BFieldCache.

The unofficial straight-line 1.13 T m, seed 0.55 T, and handwritten 0.3
are never used as a Jacobian.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

# PDG conversion: |q|=e, B in tesla, length in metres, p in GeV/c.
# This is c in GeV/(T m e), not CircleFit's handwritten 0.3.
K_GEV_PER_TM = 0.299792458
KT_TO_T = 1000.0
MM_TO_M = 0.001
GEV_TO_MEV = 1000.0
# FaserFieldCache.h: outside the map, 0.1 * CLHEP::gauss in the kT service.
# 1 gauss = 1e-4 T, so 0.1 gauss = 1e-5 T.
OUTSIDE_B_TESLA = 1.0e-5

PINNED_TABLE_CANDIDATES = (
    "/cvmfs/faser.cern.ch/repo/sw/software/22.0/faser/offline/ReleaseData/v20/MagneticFieldMaps/FaserFieldTable_v2.root",
    "/cvmfs/faser.cern.ch/repo/sw/database/DBRelease/current/GLOBAL-BField-Maps-03/FaserFieldTable_v2.root",
    "/cvmfs/faser.cern.ch/repo/sw/database/DBRelease/current/geomDB/FaserFieldTable_v2.root",
    "/cvmfs/faser.cern.ch/repo/sw/database/DBRelease/current/FaserFieldTable_v2.root",
)

S1_Z_MM = 47.4
S2_Z_MM = 1237.4
S3_Z_MM = 2427.4


class FaserFieldTableError(ValueError):
    """Raised when the field table cannot be locked."""


def find_pinned_table(candidates: Sequence[str] | None = None) -> Path:
    for raw in candidates or PINNED_TABLE_CANDIDATES:
        path = Path(raw)
        if path.is_file():
            return path
    raise FaserFieldTableError(
        "FaserFieldTable_v2.root not found in the pinned candidate paths"
    )


def unit_sign_source_contract() -> dict[str, Any]:
    return {
        "xyz_unit": "mm",
        "service_b_unit": "kT",
        "acts_b_unit": "tesla_after_1000_times_UnitConstants_T",
        "table_stored": "int16 * bscale -> kT",
        "tesla_from_stored": "B_T = 1000 * short * bscale",
        "interpolation": "trilinear_in_xyz_bin",
        "outside_map_tesla": OUTSIDE_B_TESLA,
        "dipole_scale": {
            "value": 1.0,
            "source": "FaserFieldCacheCondAlg.h UseDCS=false UseDipoScale=1.0",
            "cool_folder": "/GLOBAL/BField/Scales",
            "cool_reread": False,
            "kind": "geometry_field_derived_default",
            "empirical": False,
        },
        "k_gev_per_tm": K_GEV_PER_TM,
        "k_kind": "pdg_ec_in_GeV_per_T_m",
        "k_empirical": False,
        "forbidden_as_jacobian": (1.13, 0.55, 0.3),
        "bx_by_bz": "cartesian_lab_same_order_as_fieldx_fieldy_fieldz",
        "lorentz": "dp = q dl x B; dpy = q (Bx dz - Bz dx)",
        "yz_response": "I = int (Bx dz - Bz dx) with B in tesla and dl in metres",
        "bending_raw": "atan(ty12)-atan(ty23); plus_bending_is_plus_charge",
        "delta_theta_y": "theta_out-theta_in = -bending_raw",
        "qp_per_gev": "-bending_raw / (K_GEV_PER_TM * I)",
        "qp_per_mev": "qp_per_gev / 1000",
        "acts_q_over_p": "CircleFitTrackSeedTool.cxx:405 charge/momentum; charge=+-1",
        "circle_fit_charge": "cy<0 => charge +1; seed only, not a Jacobian",
        "sources": {
            "FaserFieldSvc.h": "xyz[3] is in mm, bxyz[3] is in kT",
            "FaserFieldCache.h": "xyz[3] is in mm, bxyz[3] is in kT",
            "BFieldCache.h": "m_scale is unit of m_field in kT; trilinear getB",
            "BFieldMesh.h": "trilinear corners; B[i] = m_scale * interpolant",
            "FaserFieldSvc.cxx": "commented ASCII readMap: m->mm and T->kT",
            "FaserBFieldData/data/README.txt": "ASCII source length mm, B tesla",
            "FASERMagneticFieldWrapper.h:65-99": "bfield *= 1000 * Acts::UnitConstants::T; kT -> T",
            "FaserFieldCacheCondAlg.h:59-79": "UseDCS false, UseDipoScale 1.0",
            "CircleFitTrackSeedTool.cxx:334-405": "seed 0.55/0.3 unofficial; q/p=charge/p",
        },
    }


class FaserFieldTable:
    """Trilinear interpolator over one BFieldMap zone, returning tesla."""

    def __init__(
        self,
        *,
        mesh_x: np.ndarray,
        mesh_y: np.ndarray,
        mesh_z: np.ndarray,
        field_tesla: np.ndarray,
        bscale: float,
        zone: Mapping[str, float],
        path: str,
        dipole_scale: float = 1.0,
    ):
        self.mesh_x = np.asarray(mesh_x, dtype=np.float64)
        self.mesh_y = np.asarray(mesh_y, dtype=np.float64)
        self.mesh_z = np.asarray(mesh_z, dtype=np.float64)
        self.field_tesla = np.asarray(field_tesla, dtype=np.float64)
        if self.field_tesla.shape != (self.mesh_x.size, self.mesh_y.size, self.mesh_z.size, 3):
            raise FaserFieldTableError("field_tesla shape must be (nx, ny, nz, 3)")
        self.bscale = float(bscale)
        self.zone = dict(zone)
        self.path = str(path)
        self.dipole_scale = float(dipole_scale)
        if self.dipole_scale != 1.0:
            self.field_tesla = self.field_tesla * self.dipole_scale

    @classmethod
    def from_uniform(
        cls,
        bx_tesla: float,
        by_tesla: float = 0.0,
        bz_tesla: float = 0.0,
        *,
        bounds_mm: tuple[float, float, float, float, float, float] = (-200.0, 200.0, -200.0, 200.0, -2000.0, 2600.0),
    ) -> "FaserFieldTable":
        xmin, xmax, ymin, ymax, zmin, zmax = bounds_mm
        mesh_x = np.linspace(xmin, xmax, 5)
        mesh_y = np.linspace(ymin, ymax, 5)
        mesh_z = np.linspace(zmin, zmax, 9)
        field = np.zeros((mesh_x.size, mesh_y.size, mesh_z.size, 3), dtype=np.float64)
        field[..., 0] = float(bx_tesla)
        field[..., 1] = float(by_tesla)
        field[..., 2] = float(bz_tesla)
        return cls(
            mesh_x=mesh_x,
            mesh_y=mesh_y,
            mesh_z=mesh_z,
            field_tesla=field,
            bscale=1.0e-7,
            zone={
                "id": 0,
                "xmin": xmin,
                "xmax": xmax,
                "ymin": ymin,
                "ymax": ymax,
                "zmin": zmin,
                "zmax": zmax,
            },
            path="uniform_synthetic",
        )

    @classmethod
    def from_root(cls, path: str | Path | None = None, *, dipole_scale: float = 1.0) -> "FaserFieldTable":
        table_path = Path(path) if path else find_pinned_table()
        try:
            import ROOT  # type: ignore
        except Exception as exc:
            raise FaserFieldTableError(f"ROOT is required to lock FaserFieldTable_v2: {exc}") from exc
        handle = ROOT.TFile.Open(str(table_path))
        if handle is None or handle.IsZombie():
            raise FaserFieldTableError(f"failed to open field table: {table_path}")
        try:
            tree = handle.Get("BFieldMap")
            if tree is None:
                raise FaserFieldTableError("BFieldMap tree absent")
            if int(tree.GetEntries()) < 1:
                raise FaserFieldTableError("BFieldMap has no zones")
            tree.GetEntry(0)
            nmeshx = int(tree.nmeshx)
            nmeshy = int(tree.nmeshy)
            nmeshz = int(tree.nmeshz)
            nfield = int(tree.nfield)
            if nfield != nmeshx * nmeshy * nmeshz:
                raise FaserFieldTableError("nfield is not nx*ny*nz")
            mesh_x = np.asarray(tree.meshx, dtype=np.float64)[:nmeshx].copy()
            mesh_y = np.asarray(tree.meshy, dtype=np.float64)[:nmeshy].copy()
            mesh_z = np.asarray(tree.meshz, dtype=np.float64)[:nmeshz].copy()
            stored = np.stack(
                (
                    np.asarray(tree.fieldx, dtype=np.float64)[:nfield],
                    np.asarray(tree.fieldy, dtype=np.float64)[:nfield],
                    np.asarray(tree.fieldz, dtype=np.float64)[:nfield],
                ),
                axis=1,
            )
            bscale = float(tree.bscale)
            tesla = (KT_TO_T * bscale) * stored.reshape(nmeshx, nmeshy, nmeshz, 3)
            zone = {
                "id": int(tree.id),
                "xmin": float(tree.xmin),
                "xmax": float(tree.xmax),
                "ymin": float(tree.ymin),
                "ymax": float(tree.ymax),
                "zmin": float(tree.zmin),
                "zmax": float(tree.zmax),
                "bscale": bscale,
                "nmeshx": nmeshx,
                "nmeshy": nmeshy,
                "nmeshz": nmeshz,
            }
        finally:
            handle.Close()
        return cls(
            mesh_x=mesh_x,
            mesh_y=mesh_y,
            mesh_z=mesh_z,
            field_tesla=tesla,
            bscale=bscale,
            zone=zone,
            path=str(table_path),
            dipole_scale=dipole_scale,
        )

    def contains(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        return (
            (x >= self.mesh_x[0])
            & (x <= self.mesh_x[-1])
            & (y >= self.mesh_y[0])
            & (y <= self.mesh_y[-1])
            & (z >= self.mesh_z[0])
            & (z <= self.mesh_z[-1])
        )

    def get_b_tesla(self, xyz_mm: np.ndarray) -> np.ndarray:
        points = np.asarray(xyz_mm, dtype=np.float64)
        if points.ndim == 1:
            points = points.reshape(1, 3)
        if points.shape[-1] != 3:
            raise FaserFieldTableError("xyz must be (N, 3) in mm")
        flat = points.reshape(-1, 3)
        out = np.full((flat.shape[0], 3), OUTSIDE_B_TESLA, dtype=np.float64)
        inside = self.contains(flat[:, 0], flat[:, 1], flat[:, 2])
        if not bool(np.any(inside)):
            return out.reshape(points.shape)
        sel = flat[inside]
        ix = np.clip(np.searchsorted(self.mesh_x, sel[:, 0], side="right") - 1, 0, self.mesh_x.size - 2)
        iy = np.clip(np.searchsorted(self.mesh_y, sel[:, 1], side="right") - 1, 0, self.mesh_y.size - 2)
        iz = np.clip(np.searchsorted(self.mesh_z, sel[:, 2], side="right") - 1, 0, self.mesh_z.size - 2)
        x0 = self.mesh_x[ix]
        x1 = self.mesh_x[ix + 1]
        y0 = self.mesh_y[iy]
        y1 = self.mesh_y[iy + 1]
        z0 = self.mesh_z[iz]
        z1 = self.mesh_z[iz + 1]
        fx = np.divide(sel[:, 0] - x0, x1 - x0, out=np.zeros_like(sel[:, 0]), where=(x1 != x0))
        fy = np.divide(sel[:, 1] - y0, y1 - y0, out=np.zeros_like(sel[:, 1]), where=(y1 != y0))
        fz = np.divide(sel[:, 2] - z0, z1 - z0, out=np.zeros_like(sel[:, 2]), where=(z1 != z0))
        gx = 1.0 - fx
        gy = 1.0 - fy
        gz = 1.0 - fz
        c000 = self.field_tesla[ix, iy, iz]
        c001 = self.field_tesla[ix, iy, iz + 1]
        c010 = self.field_tesla[ix, iy + 1, iz]
        c011 = self.field_tesla[ix, iy + 1, iz + 1]
        c100 = self.field_tesla[ix + 1, iy, iz]
        c101 = self.field_tesla[ix + 1, iy, iz + 1]
        c110 = self.field_tesla[ix + 1, iy + 1, iz]
        c111 = self.field_tesla[ix + 1, iy + 1, iz + 1]
        interpolated = (
            gx[:, None] * (
                gy[:, None] * (gz[:, None] * c000 + fz[:, None] * c001)
                + fy[:, None] * (gz[:, None] * c010 + fz[:, None] * c011)
            )
            + fx[:, None] * (
                gy[:, None] * (gz[:, None] * c100 + fz[:, None] * c101)
                + fy[:, None] * (gz[:, None] * c110 + fz[:, None] * c111)
            )
        )
        out[inside] = interpolated
        return out.reshape(points.shape)

    def integrate_yz_response(
        self,
        start_mm: Mapping[str, float] | Sequence[float],
        end_mm: Mapping[str, float] | Sequence[float],
        *,
        n_steps: int,
    ) -> float:
        values = self.integrate_yz_response_many(
            np.asarray([_xyz(start_mm)], dtype=np.float64),
            np.asarray([_xyz(end_mm)], dtype=np.float64),
            n_steps=n_steps,
        )
        return float(values[0])

    def integrate_yz_response_many(
        self,
        starts_mm: np.ndarray,
        ends_mm: np.ndarray,
        *,
        n_steps: int,
    ) -> np.ndarray:
        if n_steps < 1:
            raise FaserFieldTableError("n_steps must be >= 1")
        start = np.asarray(starts_mm, dtype=np.float64).reshape(-1, 3)
        end = np.asarray(ends_mm, dtype=np.float64).reshape(-1, 3)
        if start.shape != end.shape:
            raise FaserFieldTableError("start/end shapes must match")
        frac = np.linspace(0.0, 1.0, n_steps + 1)
        # (N, n_pts, 3)
        points = start[:, None, :] + (end - start)[:, None, :] * frac[None, :, None]
        field = self.get_b_tesla(points.reshape(-1, 3)).reshape(start.shape[0], n_steps + 1, 3)
        dx_m = (end[:, 0] - start[:, 0]) * MM_TO_M / n_steps
        dz_m = (end[:, 2] - start[:, 2]) * MM_TO_M / n_steps
        bx = field[:, :, 0]
        bz = field[:, :, 2]
        trap_bx = bx[:, 0] * 0.5 + bx[:, 1:-1].sum(axis=1) + bx[:, -1] * 0.5
        trap_bz = bz[:, 0] * 0.5 + bz[:, 1:-1].sum(axis=1) + bz[:, -1] * 0.5
        return trap_bx * dz_m - trap_bz * dx_m

    def path_integral_s1_s2_s3(
        self,
        c1: Mapping[str, float] | Sequence[float],
        c2: Mapping[str, float] | Sequence[float],
        c3: Mapping[str, float] | Sequence[float],
        *,
        n_steps: int,
    ) -> dict[str, float]:
        starts = np.asarray([_xyz(c1), _xyz(c2)], dtype=np.float64)
        ends = np.asarray([_xyz(c2), _xyz(c3)], dtype=np.float64)
        parts = self.integrate_yz_response_many(starts, ends, n_steps=n_steps)
        return {
            "I_s1_s2_tm": float(parts[0]),
            "I_s2_s3_tm": float(parts[1]),
            "I_yz_tm": float(parts[0] + parts[1]),
        }

    def probe_axis_integral(self, *, n_steps: int = 200) -> dict[str, Any]:
        start = {"x": 0.0, "y": 0.0, "z": S1_Z_MM}
        end = {"x": 0.0, "y": 0.0, "z": S3_Z_MM}
        value = self.integrate_yz_response(start, end, n_steps=n_steps)
        center = self.get_b_tesla(np.array([[0.0, 0.0, 0.5 * (S1_Z_MM + S3_Z_MM)]]))[0]
        magnet = self.get_b_tesla(np.array([[0.0, 0.0, 650.0]]))[0]
        return {
            "path": "x=y=0, z=S1->S3 piecewise-linear (single chord)",
            "official_jacobian": False,
            "I_yz_tm": float(value),
            "B_tesla_at_mid_s1_s3": [float(v) for v in center],
            "B_tesla_at_0_0_650": [float(v) for v in magnet],
            "n_steps": n_steps,
            "note": "consistency probe only; never invert a single I as the track Jacobian",
        }


def _xyz(point: Mapping[str, float] | Sequence[float]) -> np.ndarray:
    if isinstance(point, Mapping):
        return np.array(
            [float(point["x"]), float(point["y"]), float(point["z"])],
            dtype=np.float64,
        )
    return np.asarray(point, dtype=np.float64).reshape(3)


def qp_bending_proxy_per_mev(bending_raw: float, integral_yz_tm: float) -> float:
    """Geometry/field-derived q/p proxy in 1/MeV.  No empirical scale."""
    return float(-bending_raw / (K_GEV_PER_TM * integral_yz_tm) / GEV_TO_MEV)


def lock_field_unit_sign_contract(table: FaserFieldTable) -> dict[str, Any]:
    contract = unit_sign_source_contract()
    probe = table.probe_axis_integral()
    zone = table.zone
    contains_stations = (
        float(zone.get("zmin", 9e9)) <= S1_Z_MM
        and float(zone.get("zmax", -9e9)) >= S3_Z_MM
        and float(zone.get("xmin", 9e9)) <= 0.0 <= float(zone.get("xmax", -9e9))
        and float(zone.get("ymin", 9e9)) <= 0.0 <= float(zone.get("ymax", -9e9))
    )
    bx_center = float(probe["B_tesla_at_0_0_650"][0])
    i_probe = float(probe["I_yz_tm"])
    sources_ok = all(contract["sources"].values())
    units_ok = (
        contract["xyz_unit"] == "mm"
        and contract["service_b_unit"] == "kT"
        and contract["k_gev_per_tm"] == K_GEV_PER_TM
        and not contract["k_empirical"]
        and table.dipole_scale == 1.0
    )
    sign_ok = bx_center < 0.0 and i_probe < 0.0
    interpolation_ok = table.field_tesla.ndim == 4
    unofficial_unused = True
    supported = bool(
        sources_ok and units_ok and sign_ok and interpolation_ok and contains_stations and unofficial_unused
    )
    return {
        "kind": "field_unit_sign_contract",
        "status": "supported" if supported else "rejected",
        "table_path": table.path,
        "zone": zone,
        "dipole_scale": table.dipole_scale,
        "bscale": table.bscale,
        "contains_s1_s2_s3": contains_stations,
        "bx_tesla_at_magnet_sample": bx_center,
        "bx_sign_negative": bx_center < 0.0,
        "axis_probe": probe,
        "axis_probe_I_negative": i_probe < 0.0,
        "axis_probe_is_not_jacobian": True,
        "interpolation": "trilinear_BFieldMesh",
        "source_contract": contract,
        "unofficial_constants_used_as_jacobian": False,
        "supported": supported,
    }
