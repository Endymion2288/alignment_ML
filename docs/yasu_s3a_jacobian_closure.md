# Yasu-S3A: fixed-measurement curvature/geometry response Jacobian

Workbook 128.  Isolated from S2K/WB127 `qp_bending_proxy`.  Answers
whether a reasonable track-curvature \(q/p\) error can produce an IFT
residual response similar to \(R_y\), \(d_x\), or \(t_y\), **without**
requiring a trusted reconstructed momentum.

The only legal final claim is:

> Under a frozen and verified measurement/surface contract, a curvature
> error can or cannot produce a given IFT residual response, and is or
> is not similar to listed geometry/slope responses.

This does **not** establish a natural-data \(q/p\)-\(R_y\) weak mode.
High cosine similarity is not a weak mode.

## Isolation of S2K

Reviewer P0: `bending_raw` is a two-chord angle difference whose
response is the triangle-weighted kernel \(W(z)\), not unweighted
\(\int(B_x dz-B_z dx)\).  Uniform-field circular orbits give
\(\sim 0.5\times\) truth.  Do not times-two patch, fit a free scale, or
continue the S2K batch.

## Frozen contract

- Identity: `file_sha256 + source_id + run_id + event_id + skip_index + collection + track_index + measurement_digest`
- Collection: `CKFTrackCollectionWithoutIFT`
- IFT association: truth-SDO barcode equality on `SCT_ClusterContainer` station 0.  Never 4ST survival (P1-6 isolated)
- Residual: real IFT sensor loc0, \(r=\mathrm{loc0}_{cluster}-\mathrm{loc0}_{predicted}\)
- Geometry: runtime `SiDetectorElement` / ACTS wafer transform; pivot at the FASER origin; \(G=T R_z R_y R_x\) active left-multiply.  No handwritten station response.  No `/Tracker/Align` write
- Mean-response only.  Do not weight by the miscalibrated native 5×5
- Do not change fitter, seed, hits, geometry payload, or covariance scale
- Do not enter B14M / B15 / MM V2.  Do not touch held-out/sealed data.  Do not submit HTCondor

Twenty development tracks are frozen in
`configs/yasu_s3a_jacobian_closure_v1.yaml` before any Jacobian number
is inspected.  Coverage: complete three-station, WB119 sign-flip and
large-pull, S2-front dirty/missing-station.  The list must not be
replaced after seeing results.

## Jacobian

\[
J=\left[\partial r_{\rm IFT}/\partial(q/p),\;
\partial r_{\rm IFT}/\partial R_y,\;
\partial r_{\rm IFT}/\partial d_x,\;
\partial r_{\rm IFT}/\partial t_y\right]
\]

Central finite differences \(\pm\delta,\pm\delta/2,\pm\delta/4\) with
frozen amplitudes \(10^{-6}\,\mathrm{MeV}^{-1}\), \(10^{-3}\,\mathrm{rad}\),
\(0.10\,\mathrm{mm}\), \(10^{-4}\).  Report relative convergence of the
last two rungs; near-zero responses use a \(10^{-3}\,\mathrm{mm}\)
absolute residual tolerance.

**Fixed-state** intervention is dumped now: change one parameter, keep
the rest of the state fixed, same ACTS mean transport to the same IFT
surface.  **Profiled-track** intervention is recorded as deferred: this
stage does not change the fitter.

Independent controls: zero-field analytic line/rotated-plane
intersection; uniform-field analytic circular orbit; energy-loss
on/off; fixed surface-sequence derivative convergence.  Stop physical
interpretation immediately if surface identity, association, navigation
path, or finite differences do not close.

## Stage PASS

Numerical derivatives close, odd/even linear terms close, and the
stage can answer how large \(\delta(q/p)\) must be to reach a
Yasu-like residual and whether that layer/local-coordinate pattern is
nearly collinear with \(R_y\), \(d_x\), or \(t_y\).  If the curvature
response cannot mimic that residual in direction or required
amplitude, reject this mechanism and add no more momentum-proxy
workbooks.

Authorize a later Schur/profile reduced-normal-matrix falsification
**only** if the fixed-state response is significantly near-collinear.
Otherwise stop.

## Recorded dump (2026-09-10)

Local Athena, no HTCondor.  Artifact
`outputs/yasu_s3a_jacobian_closure_v1/yasu_s3a_jacobian_closure_20260910T193937Z_24be51ac`.

**Verdict: FAIL `yasu_s3a_jacobian_recorded`.**  `authorize_profiled_schur_next=false`.
`weak_mode_claimed=false`.  Config SHA
`d3d2e28f18943805e303ffb2676607bfc6e4d947c9ceebefad0c39c647cc671a`.
Git HEAD `b617614e7746ec400072347b7cf3f056a4b9a9a7`.  Dump SHA-256:
100043 `c24fa593…e4db`, 100048 `3c10c586…3290`.

Twenty frozen identities dumped.  Twelve Jacobian-eligible; eight
coverage-only dirty tracks do not decide PASS.  Three eligible tracks
broke the surface/navigation contract (`navigation_path_changed` on
100048 skip 1 and 9; `propagate_surface_failed` on skip 15).  Physical
interpretation of the whole stage therefore stops.

On the nine closed eligible tracks, mean-response columns in IFT loc0
are numerically closed.  \(\partial r/\partial(q/p)\) RMS is
\(2.6\times10^5\,\mathrm{mm\,MeV}\); \(\delta(q/p)\approx 3.7\times10^{-7}\,\mathrm{MeV}^{-1}\)
moves loc0 by \(0.1\,\mathrm{mm}\).  Pairwise cosine:
\(\cos(q/p,t_y)\approx 0.99994\), \(\lvert\cos(q/p,R_y)\rvert\lesssim 0.007\),
\(\lvert\cos(q/p,d_x)\rvert\lesssim 0.001\).  High cosine with \(t_y\) is
**not** a \(q/p\)-\(R_y\) weak mode.  Schur/profile is not authorized
until the three eligible surface failures are isolated without changing
the fitter, seed, or geometry payload.

## P1 blockers (recorded, not reopened)

P1-1 plurality matcher; P1-2 S1 truth momentum surface; P1-3
bound→curvilinear / KF seed surface; P1-4 CircleFit static map (pin
independently confirmed); P1-5 bending space-point fallback; P1-6 4ST
association (isolated here).

## How to run

```
scripts/build_ckf_yasu_s3a_jacobian_dump.sh
scripts/run_ckf_yasu_s3a_jacobian_dump.sh
scripts/audit_yasu_s3a_jacobian_closure.py
```

Until dumps exist, the audit records the frozen contract, focus
manifest, analytic controls, and P1 register.  Jacobian numbers must
come from the runtime dump.
