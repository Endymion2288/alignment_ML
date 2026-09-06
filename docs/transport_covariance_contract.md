# T11 Transport Covariance Contract

Mathematical conversions for `[x, y, tx, ty]` and `q/p` (native per MeV) are
tested.  Production `C_prop` remains the frozen WB87 result
`faseracts_transport_covariance_not_validated`.

T11 does not rerun Stage B, does not submit 18-source jobs, does not rescale
covariance, and does not use truth q/p as a deployment seed.  T12 may not treat
this contract as established.
