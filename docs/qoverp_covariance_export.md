# CKF q/p Covariance Export Contract (Stage B / Task A2)

Workbook 96.  This task exports the persisted `CKFTrackCollection` 5x5 from the
same reconstructed xAOD already used by WB87/95.  It does not refit SegmentFit,
does not write geometry, and does not enter Task B unless the contract passes.

Native state: `(loc1, loc2, phi, theta, q/p)` in Athena TrackParameters, q/p in
`1/MeV`.  Derived state: `(x, y, tx, ty, q/p)` using the frozen CurvilinearUVT
exporter Jacobian, with q/p left unchanged.

Allowed inputs: Calypso reconstruction chain, `CKFTrackCollection`, ACTS
TrackState already persisted on that collection.

Forbidden: truth q/p, dummy SegmentFit covariance, synthetic q/p injection,
new source campaigns, geometry writes, alignment, Measurement Model V2.

PASS requires both construction and validation dumps to materialize a full 5x5,
remain SPD and symmetric, differ from the SegmentFit dummy
(`q/p = 1e-5 /MeV`, `var = 5e-6 /MeV^2`), and export nonzero q/p cross terms.

FAIL keeps `faseracts_transport_covariance_not_validated` and does not open
Task B.  The project then stays on residual/DQ monitoring plus a reproducible
negative result.
