# Operating Protocol V1 reviewer-risk checklist

Anticipated misreadings of the frozen real-data result.  The scientific conclusion is that geometry must not be written from the present self-nulling residuals.

## empty_100_event_looks_like_v2_failure

**Risk.** A reviewer may read the empty 100-event selected graph as a V2 failure.

**Response.** The all-pairs candidate graph is already nonempty at 100 events.  Selected routes appear as soon as the same frozen protocol sees O(10^3)–O(10^4) events and recover at full segment.  Thresholds were not retuned.

## residual_drop_as_success

**Risk.** A reviewer may treat a residual or χ² decrease as proof that the geometry has been corrected.

**Response.** Every residual/χ² change in this package is labeled DQ observable.  The reduced-mode 14974→14973 transfer worsens χ² by ×16, so even the same estimator is not a transferable correction.

## implied_cdx_as_measurement

**Risk.** A reviewer may quote implied |C_dx| as a measured layer contrast.

**Response.** Implied |C_dx| is only a contamination diagnostic against the MC 1.5–1.7 µm isolation budget.  No real-data C_dx Mode is opened.

## fisher_identifiable_should_be_written

**Risk.** A reviewer may argue that identifiable {dy,rx,rz} should be written.

**Response.** Fisher identifiability is necessary, not sufficient.  The two independent calibration runs produce opposite-sign, hundreds-of-σ inconsistent updates.  A non-transferable correction is not a geometry write.

## not_writing_looks_unfinished

**Risk.** A reviewer may call geometry_write_allowed=false an incomplete result.

**Response.** The paper result is the negative: the observed topology cannot isolate a transferable station update from C_dx leakage.  Monitoring on seven later runs stays nominal at current geometry, so there is no empirical mandate to force a write.

## quieter_subset_could_rescue

**Risk.** A reviewer may ask for a quieter subset that would allow a write.

**Response.** Same-topology scaling leaves the six-DoF condition number unchanged.  dx/ry leakage cannot be rescued by more events or looser thresholds.  Searching a residual-selected subset would break the residual-blind occupancy contract.

## 14977_looks_like_anomaly

**Risk.** A reviewer may read 14977 as an alignment excursion.

**Response.** The frozen alarm table classifies 0<selected<10 as insufficient statistics, not an alignment anomaly.  14977 has two selected routes on a 1687-event remainder.

## monitoring_equals_alignment

**Risk.** A reviewer may treat stable dy/rx monitoring as a successful alignment.

**Response.** Stability at current official geometry supports DQ monitoring only.  It does not validate a new station payload and is not a C_dx Mode.
