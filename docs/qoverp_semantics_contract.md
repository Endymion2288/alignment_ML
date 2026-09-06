# Physical q/p Uncertainty Semantics (Stage B / Task A)

Workbook 95.  This task inventories reconstruction-chain q/p sources already
exported on the frozen WB87 construction/validation split.  It is not
alignment, does not reopen T00–T12, and does not start a new source campaign.

A physical prior requires all three of:

- a signed q/p estimate
- an uncertainty
- correlations with the rest of the native Athena state

from the reconstruction chain.  Truth q/p is not a real-data solution.
SegmentFit dummy q/p is not physical.  Deleting the q/p column is not the
final scheme.  Covariance is not rescaled.

Native state: `(loc1, loc2, phi, theta, q/p)` with q/p in `1/MeV`.
