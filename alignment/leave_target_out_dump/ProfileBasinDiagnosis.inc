// B14M-S diagnosis only.  Included inside the anonymous namespace of
// CkfLeaveTargetOutDumpAlg.cxx.  Does not change runOneProfileNumerics.

using Mat3 = Eigen::Matrix<double, 3, 3>;
using Vec3 = Eigen::Matrix<double, 3, 1>;

constexpr int kAlphaIdx[2] = {0, 3};
constexpr int kNuIdx[3] = {1, 2, 4};
constexpr std::array<double, 8> kFrozenDamps{
    1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125};

Mat3 symmetricPinv3(const Mat3& matrix, double relative, int* rank) {
  Eigen::SelfAdjointEigenSolver<Mat3> solver(0.5 * (matrix + matrix.transpose()));
  const auto values = solver.eigenvalues();
  const auto vectors = solver.eigenvectors();
  double scale = values.cwiseAbs().maxCoeff();
  if (!(scale > 0.0) || !std::isfinite(scale)) {
    scale = 1.0;
  }
  Mat3 inverse = Mat3::Zero();
  int kept = 0;
  for (int i = 0; i < 3; ++i) {
    if (values[i] > relative * scale) {
      inverse += (1.0 / values[i]) * (vectors.col(i) * vectors.col(i).transpose());
      ++kept;
    }
  }
  if (rank != nullptr) {
    *rank = kept;
  }
  return 0.5 * (inverse + inverse.transpose());
}

bool isFailureIdentity(int run, int event, int target) {
  return (run == 100043 && event == 37 && (target == 1 || target == 2)) ||
         (run == 100048 && event == 86 && target == 1);
}

bool isControlIdentity(int run, int event, int target) {
  return (run == 100043 && event == 0 && target == 1) ||
         (run == 100043 && event == 37 && target == 3) ||
         (run == 100048 && event == 86 && target == 2);
}

json jsonFieldOrNull(const json& rec, const char* key) {
  if (rec.contains(key)) {
    return rec[key];
  }
  return nullptr;
}

double jsonNumberOr(const json& rec, const char* key, double fallback) {
  if (!rec.contains(key) || rec[key].is_null() || !rec[key].is_number()) {
    return fallback;
  }
  return rec[key].get<double>();
}

int jsonIntOr(const json& rec, const char* key, int fallback) {
  if (!rec.contains(key) || rec[key].is_null() || !rec[key].is_number()) {
    return fallback;
  }
  return rec[key].get<int>();
}

std::string jsonStringOr(const json& rec, const char* key,
                         const std::string& fallback) {
  if (!rec.contains(key) || rec[key].is_null() || !rec[key].is_string()) {
    return fallback;
  }
  return rec[key].get<std::string>();
}

const std::vector<json>& basinSourceRows(const std::string& path) {
  static std::mutex mu;
  static std::string loaded;
  static std::vector<json> rows;
  std::lock_guard<std::mutex> lock(mu);
  if (loaded == path) {
    return rows;
  }
  rows.clear();
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    json rec = json::parse(line, nullptr, false);
    if (rec.is_discarded()) {
      continue;
    }
    if (jsonStringOr(rec, "row_role", "") == "profile_optimize") {
      rows.push_back(std::move(rec));
    }
  }
  loaded = path;
  return rows;
}

bool boundFromDump(const json& row, Acts::BoundVector& theta) {
  if (row.contains("profiled_native_bound") &&
      row["profiled_native_bound"].is_object()) {
    const auto& bound = row["profiled_native_bound"];
    theta[Acts::eBoundLoc0] = jsonNumberOr(bound, "loc0", 0.0);
    theta[Acts::eBoundLoc1] = jsonNumberOr(bound, "loc1", 0.0);
    theta[Acts::eBoundPhi] = jsonNumberOr(bound, "phi", 0.0);
    theta[Acts::eBoundTheta] = jsonNumberOr(bound, "theta", 0.0);
    if (bound.contains("q_over_p_per_gev") &&
        !bound["q_over_p_per_gev"].is_null()) {
      theta[Acts::eBoundQOverP] = bound["q_over_p_per_gev"].get<double>();
    } else {
      theta[Acts::eBoundQOverP] =
          jsonNumberOr(bound, "q_over_p_per_mev", 0.0) / 1_MeV;
    }
    return std::isfinite(theta[Acts::eBoundLoc0]) &&
           std::isfinite(theta[Acts::eBoundLoc1]) &&
           std::isfinite(theta[Acts::eBoundPhi]) &&
           std::isfinite(theta[Acts::eBoundTheta]) &&
           std::isfinite(theta[Acts::eBoundQOverP]);
  }
  if (row.contains("profiled_native_state") &&
      row["profiled_native_state"].is_array() &&
      row["profiled_native_state"].size() == 5) {
    theta[Acts::eBoundLoc0] = row["profiled_native_state"][0].get<double>();
    theta[Acts::eBoundLoc1] = row["profiled_native_state"][1].get<double>();
    theta[Acts::eBoundPhi] = row["profiled_native_state"][2].get<double>();
    theta[Acts::eBoundTheta] = row["profiled_native_state"][3].get<double>();
    theta[Acts::eBoundQOverP] =
        row["profiled_native_state"][4].get<double>() / 1_MeV;
    return true;
  }
  return false;
}

struct ScaledSystem {
  bool ok{false};
  double chi2{0.0};
  Chi2Eval eval;
  Eigen::VectorXd residual;
  Eigen::MatrixXd jacobian;
  Mat5 hessian = Mat5::Zero();
  Bound5 gradient = Bound5::Zero();
  Mat5 hz = Mat5::Zero();
  Bound5 gz = Bound5::Zero();
  Bound5 deltaZ = Bound5::Zero();
  Bound5 deltaNative = Bound5::Zero();
  int rank{0};
  std::string jacobianKind{"unset"};
};

ScaledSystem evaluateScaledSystem(
    const Propagator& propagator, const Acts::GeometryContext& geometryContext,
    const Acts::MagneticFieldContext& magFieldContext,
    const Acts::BoundTrackParameters& current, const std::vector<ProfileHit>& hits,
    int maxSteps, bool matchKalman, bool sequential, bool supportingPlane,
    double pinvRelative, bool useScaling,
    const GradientPropagator* gradientPropagator, bool fieldGradientRepair) {
  ScaledSystem out;
  out.eval = evaluateSequentialResiduals(
      propagator, geometryContext, magFieldContext, current, hits, maxSteps,
      matchKalman, sequential, supportingPlane);
  if (!out.eval.ok) {
    return out;
  }
  out.chi2 = out.eval.chi2;
  out.residual = out.eval.residual;
  out.jacobian.resize(static_cast<int>(hits.size()), 5);
  out.jacobian.setZero();
  bool jacOk = false;
  out.jacobianKind = "official_adaptive_central_fd";
  if (fieldGradientRepair && gradientPropagator != nullptr && supportingPlane &&
      sequential) {
    jacOk = repairedSourceLoc0Jacobian(
        *gradientPropagator, geometryContext, magFieldContext, current, hits,
        out.eval, maxSteps, matchKalman, out.jacobian);
    out.jacobianKind = "wb123_repaired_production_tangent";
  }
  if (!jacOk) {
    return out;
  }
  Mat5 h = Mat5::Zero();
  Bound5 g = Bound5::Zero();
  for (int i = 0; i < out.residual.size(); ++i) {
    const double w = 1.0 / hits[static_cast<std::size_t>(i)].variance;
    const Eigen::Matrix<double, 1, 5> ji = out.jacobian.row(i);
    h += w * (ji.transpose() * ji);
    g += w * ji.transpose() * out.residual[i];
  }
  out.hessian = 0.5 * (h + h.transpose());
  out.gradient = g;
  Bound5 scale = Bound5::Ones();
  if (useScaling) {
    for (int i = 0; i < 5; ++i) {
      scale[i] = kNumericScale[static_cast<std::size_t>(i)];
    }
  }
  for (int i = 0; i < 5; ++i) {
    out.gz[i] = g[i] * scale[i];
    for (int j = 0; j < 5; ++j) {
      out.hz(i, j) = out.hessian(i, j) * scale[i] * scale[j];
    }
  }
  out.deltaZ = symmetricPinv5(out.hz, pinvRelative, &out.rank) * out.gz;
  for (int i = 0; i < 5; ++i) {
    out.deltaNative[i] = out.deltaZ[i] * scale[i];
  }
  out.ok = true;
  return out;
}

json svdStationarityJson(const ScaledSystem& sys, double pinvRelative) {
  json out;
  if (!sys.ok) {
    out["ok"] = false;
    return out;
  }
  Eigen::JacobiSVD<Mat5> svd(sys.hz, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const auto singular = svd.singularValues();
  const Mat5 V = svd.matrixV();
  double smax = singular.maxCoeff();
  if (!(smax > 0.0) || !std::isfinite(smax)) {
    smax = 1.0;
  }
  json sigmas = json::array();
  Mat5 projR = Mat5::Zero();
  Mat5 projN = Mat5::Zero();
  int svdRank = 0;
  for (int i = 0; i < 5; ++i) {
    sigmas.push_back(singular[i]);
    if (singular[i] > pinvRelative * smax) {
      projR += V.col(i) * V.col(i).transpose();
      ++svdRank;
    } else {
      projN += V.col(i) * V.col(i).transpose();
    }
  }
  const Bound5 gR = projR * sys.gz;
  const Bound5 gN = projN * sys.gz;
  const Bound5 gChi2 = -2.0 * sys.gz;
  const Bound5 gChi2R = projR * gChi2;
  const Bound5 gChi2N = projN * gChi2;
  const double gTHpG = sys.gz.dot(sys.deltaZ);
  Eigen::Matrix<double, 2, 1> ga;
  ga[0] = sys.gradient[kAlphaIdx[0]];
  ga[1] = sys.gradient[kAlphaIdx[1]];
  Vec3 gn;
  gn[0] = sys.gradient[kNuIdx[0]];
  gn[1] = sys.gradient[kNuIdx[1]];
  gn[2] = sys.gradient[kNuIdx[2]];
  Eigen::Matrix<double, 2, 3> han;
  Mat3 hnn = Mat3::Zero();
  for (int a = 0; a < 2; ++a) {
    for (int n = 0; n < 3; ++n) {
      han(a, n) = sys.hessian(kAlphaIdx[a], kNuIdx[n]);
    }
  }
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      hnn(i, j) = sys.hessian(kNuIdx[i], kNuIdx[j]);
    }
  }
  int rankNn = 0;
  const Mat3 hnnP = symmetricPinv3(hnn, pinvRelative, &rankNn);
  const Eigen::Matrix<double, 2, 1> gProf = ga - han * hnnP * gn;
  Eigen::Matrix<double, 2, 1> gaZ;
  Vec3 gnZ;
  gaZ[0] = sys.gz[kAlphaIdx[0]];
  gaZ[1] = sys.gz[kAlphaIdx[1]];
  gnZ[0] = sys.gz[kNuIdx[0]];
  gnZ[1] = sys.gz[kNuIdx[1]];
  gnZ[2] = sys.gz[kNuIdx[2]];
  Eigen::Matrix<double, 2, 3> hanZ;
  Mat3 hnnZ = Mat3::Zero();
  for (int a = 0; a < 2; ++a) {
    for (int n = 0; n < 3; ++n) {
      hanZ(a, n) = sys.hz(kAlphaIdx[a], kNuIdx[n]);
    }
  }
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      hnnZ(i, j) = sys.hz(kNuIdx[i], kNuIdx[j]);
    }
  }
  int rankNnZ = 0;
  const Mat3 hnnZP = symmetricPinv3(hnnZ, pinvRelative, &rankNnZ);
  const Eigen::Matrix<double, 2, 1> gProfZ = gaZ - hanZ * hnnZP * gnZ;
  out = {
      {"ok", true},
      {"chi2", sys.chi2},
      {"rank_H", sys.rank},
      {"svd_rank_H", svdRank},
      {"singular_values", sigmas},
      {"pinv_relative", pinvRelative},
      {"jacobian_implementation", sys.jacobianKind},
      {"g_optimizer_z",
       json::array({sys.gz[0], sys.gz[1], sys.gz[2], sys.gz[3], sys.gz[4]})},
      {"g_chi2_z",
       json::array({gChi2[0], gChi2[1], gChi2[2], gChi2[3], gChi2[4]})},
      {"norm_g_optimizer_z", sys.gz.norm()},
      {"norm_g_R_optimizer_z", gR.norm()},
      {"norm_g_N_optimizer_z", gN.norm()},
      {"norm_g_chi2_z", gChi2.norm()},
      {"norm_g_R_chi2_z", gChi2R.norm()},
      {"norm_g_N_chi2_z", gChi2N.norm()},
      {"norm_Hp_g", sys.deltaZ.norm()},
      {"g_opt_T_Hp_g", gTHpG},
      {"g_chi2_T_neg_Hp_g", -2.0 * gTHpG},
      {"predicted_gn_decrease_chi2", -gTHpG},
      {"delta_z_optimizer",
       json::array({sys.deltaZ[0], sys.deltaZ[1], sys.deltaZ[2], sys.deltaZ[3],
                    sys.deltaZ[4]})},
      {"g_alpha_profiled_native", {gProf[0], gProf[1]}},
      {"norm_g_alpha_profiled_native", gProf.norm()},
      {"g_alpha_profiled_z", {gProfZ[0], gProfZ[1]}},
      {"norm_g_alpha_profiled_z", gProfZ.norm()},
      {"rank_H_nn", rankNn},
      {"rank_H_nn_z", rankNnZ},
      {"do_not_rewrite_schur", true},
  };
  return out;
}

json frozenLineSearchReplayJson(
    const Propagator& propagator, const Acts::GeometryContext& geometryContext,
    const Acts::MagneticFieldContext& magFieldContext,
    const std::shared_ptr<const Acts::Surface>& surface,
    const Acts::BoundVector& theta, const ScaledSystem& sys,
    const std::vector<ProfileHit>& hits, int maxSteps, bool matchKalman,
    bool sequential, bool supportingPlane) {
  json trials = json::array();
  bool anyStrictDecrease = false;
  bool anyExistingAccept = false;
  for (double damp : kFrozenDamps) {
    Acts::BoundVector trial = theta;
    bool trialBound = false;
    for (int ipar = 0; ipar < 5; ++ipar) {
      trial[ipar] += damp * sys.deltaNative[ipar];
    }
    json item = {{"lambda", damp}, {"predicted_decrease", -sys.gz.dot(sys.deltaZ) * damp},
                 {"armijo_not_used", true},
                 {"accept_condition", "chi2_new <= chi2_final"}};
    if (!applyPhysicalBounds(trial, trialBound)) {
      item["ok"] = false;
      item["transport_valid"] = false;
      item["status"] = "bound_rejected";
      trials.push_back(item);
      continue;
    }
    Chi2Eval trialEval = evaluateSequentialResiduals(
        propagator, geometryContext, magFieldContext,
        Acts::BoundTrackParameters(surface, trial, std::nullopt,
                                   Acts::ParticleHypothesis::muon()),
        hits, maxSteps, matchKalman, sequential, supportingPlane);
    item["ok"] = trialEval.ok;
    item["transport_valid"] = trialEval.ok;
    item["chi2_trial"] = trialEval.ok ? json(trialEval.chi2) : json(nullptr);
    item["chi2_final"] = sys.chi2;
    item["actual_decrease"] =
        trialEval.ok ? json(sys.chi2 - trialEval.chi2) : json(nullptr);
    item["hit_boundary"] = trialBound;
    item["surviving_predicted_measurement_loc0"] =
        trialEval.ok ? predictedMeasurementLoc0Json(trialEval) : json(nullptr);
    item["transport_branch_identity"] =
        trialEval.ok ? transportBranchIdentityJson(trialEval) : json(nullptr);
    const bool existingAccept = trialEval.ok && trialEval.chi2 <= sys.chi2;
    const bool strictDecrease = trialEval.ok && trialEval.chi2 < sys.chi2;
    item["existing_accept"] = existingAccept;
    item["strict_decrease"] = strictDecrease;
    item["status"] = trialEval.ok ? (strictDecrease ? "strict_decrease"
                                                    : (existingAccept ? "equal_chi2"
                                                                      : "rejected"))
                                  : "propagation_rejected";
    anyStrictDecrease = anyStrictDecrease || strictDecrease;
    anyExistingAccept = anyExistingAccept || existingAccept;
    trials.push_back(item);
  }
  return {{"chi2_final", sys.chi2},
          {"n_frozen_lambdas", static_cast<int>(kFrozenDamps.size())},
          {"line_search_floor_unchanged", true},
          {"any_strict_decrease", anyStrictDecrease},
          {"any_existing_accept", anyExistingAccept},
          {"trials", trials}};
}

struct NuisanceOptResult {
  bool ok{false};
  double chi2{0.0};
  Acts::BoundVector theta = Acts::BoundVector::Zero();
  Vec3 gradientNuZ = Vec3::Zero();
  int rankNn{0};
  int nIter{0};
  std::string termination{"unset"};
  Chi2Eval eval;
  json stationarity = nullptr;
};

NuisanceOptResult runNuisanceOnly(
    const Propagator& propagator, const Acts::GeometryContext& geometryContext,
    const Acts::MagneticFieldContext& magFieldContext,
    const std::shared_ptr<const Acts::Surface>& surface,
    const Acts::BoundVector& start, const std::vector<ProfileHit>& hits,
    int maxIterations, double pinvRelative, int maxSteps, bool matchKalman,
    bool sequential, bool useScaling, bool supportingPlane,
    const GradientPropagator* gradientPropagator, bool fieldGradientRepair) {
  NuisanceOptResult out;
  Acts::BoundVector theta = start;
  bool hitBoundary = false;
  if (!applyPhysicalBounds(theta, hitBoundary)) {
    out.termination = "nonfinite_initialization";
    return out;
  }
  const Acts::BoundTrackParameters startP(
      surface, theta, std::nullopt, Acts::ParticleHypothesis::muon());
  ScaledSystem sys = evaluateScaledSystem(
      propagator, geometryContext, magFieldContext, startP, hits, maxSteps,
      matchKalman, sequential, supportingPlane, pinvRelative, useScaling,
      gradientPropagator, fieldGradientRepair);
  if (!sys.ok) {
    out.termination = "propagation_blocked";
    return out;
  }
  std::string termination = "max_iterations";
  int nIter = 0;
  for (nIter = 0; nIter < maxIterations; ++nIter) {
    const Acts::BoundTrackParameters current(
        surface, theta, std::nullopt, Acts::ParticleHypothesis::muon());
    sys = evaluateScaledSystem(
        propagator, geometryContext, magFieldContext, current, hits, maxSteps,
        matchKalman, sequential, supportingPlane, pinvRelative, useScaling,
        gradientPropagator, fieldGradientRepair);
    if (!sys.ok) {
      termination = "propagation_blocked";
      break;
    }
    Vec3 gzN = Vec3::Zero();
    Mat3 hzN = Mat3::Zero();
    Bound5 scale = Bound5::Ones();
    if (useScaling) {
      for (int i = 0; i < 5; ++i) {
        scale[i] = kNumericScale[static_cast<std::size_t>(i)];
      }
    }
    for (int i = 0; i < 3; ++i) {
      gzN[i] = sys.gradient[kNuIdx[i]] * scale[kNuIdx[i]];
      for (int j = 0; j < 3; ++j) {
        hzN(i, j) = sys.hessian(kNuIdx[i], kNuIdx[j]) * scale[kNuIdx[i]] *
                    scale[kNuIdx[j]];
      }
    }
    int rankNn = 0;
    const Vec3 deltaZN = symmetricPinv3(hzN, pinvRelative, &rankNn) * gzN;
    Bound5 deltaNative = Bound5::Zero();
    for (int i = 0; i < 3; ++i) {
      deltaNative[kNuIdx[i]] = deltaZN[i] * scale[kNuIdx[i]];
    }
    const double gradNorm = gzN.norm();
    const double stepNorm = deltaZN.norm();
    bool accepted = false;
    Acts::BoundVector best = theta;
    double bestChi2 = sys.chi2;
    for (double damp : kFrozenDamps) {
      Acts::BoundVector trial = theta;
      bool trialBound = hitBoundary;
      for (int i = 0; i < 3; ++i) {
        trial[kNuIdx[i]] += damp * deltaNative[kNuIdx[i]];
      }
      trial[kAlphaIdx[0]] = start[kAlphaIdx[0]];
      trial[kAlphaIdx[1]] = start[kAlphaIdx[1]];
      if (!applyPhysicalBounds(trial, trialBound)) {
        continue;
      }
      Chi2Eval trialEval = evaluateSequentialResiduals(
          propagator, geometryContext, magFieldContext,
          Acts::BoundTrackParameters(surface, trial, std::nullopt,
                                     Acts::ParticleHypothesis::muon()),
          hits, maxSteps, matchKalman, sequential, supportingPlane);
      if (!trialEval.ok) {
        continue;
      }
      if (trialEval.chi2 <= bestChi2) {
        best = trial;
        bestChi2 = trialEval.chi2;
        accepted = true;
        hitBoundary = trialBound;
      }
    }
    const double relDec = (sys.chi2 - bestChi2) / std::max(sys.chi2, 1.0);
    if (!accepted) {
      termination = (gradNorm < 1.0e-6 || stepNorm < 1.0e-8) ? "converged"
                                                             : "flat_direction";
      out.gradientNuZ = gzN;
      out.rankNn = rankNn;
      break;
    }
    theta = best;
    if (relDec < 1.0e-8 && (gradNorm < 1.0e-6 || stepNorm < 1.0e-8)) {
      termination = (rankNn < 3) ? "flat_direction" : "converged";
      out.gradientNuZ = gzN;
      out.rankNn = rankNn;
      break;
    }
    out.gradientNuZ = gzN;
    out.rankNn = rankNn;
  }
  const Acts::BoundTrackParameters fitted(
      surface, theta, std::nullopt, Acts::ParticleHypothesis::muon());
  sys = evaluateScaledSystem(
      propagator, geometryContext, magFieldContext, fitted, hits, maxSteps,
      matchKalman, sequential, supportingPlane, pinvRelative, useScaling,
      gradientPropagator, fieldGradientRepair);
  out.ok = sys.ok;
  out.chi2 = sys.chi2;
  out.theta = theta;
  out.nIter = nIter;
  out.termination = termination;
  out.eval = sys.eval;
  out.stationarity = svdStationarityJson(sys, pinvRelative);
  return out;
}

json endpointAuditRow(
    const json& baseRow, const json& source, const std::string& role,
    const Propagator& propagator, const Acts::GeometryContext& geometryContext,
    const Acts::MagneticFieldContext& magFieldContext,
    const Acts::BoundTrackParameters& seed, const std::vector<ProfileHit>& hits,
    int maxSteps, bool matchKalman, bool sequential, bool useScaling,
    bool supportingPlane, double pinvRelative,
    const GradientPropagator* gradientPropagator, bool fieldGradientRepair) {
  json row = baseRow;
  row["row_kind"] = "basin_endpoint_audit";
  row["identity_role"] = role;
  row["do_not_change_optimizer"] = true;
  row["held_out_target_inaccessible"] = true;
  row["wb129_source_variant"] = jsonStringOr(source, "profile_init_variant", "");
  row["wb129_chi2_prof"] = jsonFieldOrNull(source, "chi2_prof");
  row["wb129_termination_reason"] = jsonFieldOrNull(source, "termination_reason");
  row["wb129_n_iterations"] = jsonFieldOrNull(source, "profile_n_iterations");
  Acts::BoundVector theta = Acts::BoundVector::Zero();
  if (!boundFromDump(source, theta)) {
    row["ok"] = false;
    row["fit_failure_reason"] = "endpoint_native_unreadable";
    return row;
  }
  auto surface = seed.referenceSurface().getSharedPtr();
  const Acts::BoundTrackParameters current(
      surface, theta, std::nullopt, Acts::ParticleHypothesis::muon());
  const ScaledSystem sys = evaluateScaledSystem(
      propagator, geometryContext, magFieldContext, current, hits, maxSteps,
      matchKalman, sequential, supportingPlane, pinvRelative, useScaling,
      gradientPropagator, fieldGradientRepair);
  row["ok"] = sys.ok;
  row["chi2"] = sys.ok ? json(sys.chi2) : json(nullptr);
  row["chi2_prof"] = row["chi2"];
  row["profiled_native_bound"] = nativeBoundJson(theta);
  row["stationarity"] = svdStationarityJson(sys, pinvRelative);
  row["flat_direction_termination_audit"] = {
      {"wb129_termination", jsonStringOr(source, "termination_reason", "")},
      {"rank_deficiency_is_not_successful_termination", true},
      {"supported_gradient_must_be_stationary", true},
  };
  if (sys.ok) {
    row["frozen_linesearch_replay"] = frozenLineSearchReplayJson(
        propagator, geometryContext, magFieldContext, surface, theta, sys, hits,
        maxSteps, matchKalman, sequential, supportingPlane);
    row["surviving_predicted_measurement_loc0"] =
        predictedMeasurementLoc0Json(sys.eval);
    row["transport_branch_identity"] = transportBranchIdentityJson(sys.eval);
    row["propagation_success"] = true;
  } else {
    row["frozen_linesearch_replay"] = nullptr;
    row["propagation_success"] = false;
  }
  row["predicted_target_loc0"] = nullptr;
  row["held_out_prediction_not_used_for_selection"] = true;
  return row;
}

json nuisanceResultJson(const json& baseRow, const std::string& kind,
                        const std::string& role, const NuisanceOptResult& result,
                        const Acts::BoundVector& alphaHeld,
                        const std::string& alphaFrom, const std::string& nuFrom,
                        double lambda, const std::string& direction) {
  json row = baseRow;
  row["row_kind"] = kind;
  row["identity_role"] = role;
  row["do_not_change_optimizer"] = true;
  row["held_out_target_inaccessible"] = true;
  row["predicted_target_loc0"] = nullptr;
  row["held_out_prediction_not_used_for_selection"] = true;
  row["alpha_held_fixed"] = true;
  row["alpha_from_restart"] = alphaFrom;
  row["nu_start_restart"] = nuFrom;
  row["lambda"] = lambda;
  row["continuation_direction"] = direction;
  row["ok"] = result.ok;
  row["chi2"] = result.ok ? json(result.chi2) : json(nullptr);
  row["chi2_prof"] = row["chi2"];
  row["termination_reason"] = result.termination;
  row["profile_n_iterations"] = result.nIter;
  row["rank_H_nn"] = result.rankNn;
  row["gradient_nu_z"] = {result.gradientNuZ[0], result.gradientNuZ[1],
                          result.gradientNuZ[2]};
  row["norm_gradient_nu_z"] = result.gradientNuZ.norm();
  row["alpha_held"] = {{"loc0", alphaHeld[kAlphaIdx[0]]},
                       {"theta", alphaHeld[kAlphaIdx[1]]}};
  row["profiled_native_bound"] = nativeBoundJson(result.theta);
  row["stationarity"] = result.stationarity;
  if (result.ok) {
    row["surviving_predicted_measurement_loc0"] =
        predictedMeasurementLoc0Json(result.eval);
    row["transport_branch_identity"] = transportBranchIdentityJson(result.eval);
    row["propagation_success"] = true;
  } else {
    row["propagation_success"] = false;
  }
  return row;
}

std::vector<json> runBasinDiagnosis(
    const json& baseRow, const Propagator& propagator,
    const Acts::GeometryContext& geometryContext,
    const Acts::MagneticFieldContext& magFieldContext,
    const Acts::BoundTrackParameters& seed, const std::vector<ProfileHit>& hits,
    int runId, int eventId, int targetStation, const std::string& sourceJsonl,
    bool runContinuation, int maxIterations, double pinvRelative, int maxSteps,
    bool matchKalman, bool sequential, bool useScaling, bool supportingPlane,
    const GradientPropagator* gradientPropagator, bool fieldGradientRepair) {
  std::vector<json> rows;
  const std::string role = isFailureIdentity(runId, eventId, targetStation)
                               ? "failure"
                               : (isControlIdentity(runId, eventId, targetStation)
                                      ? "negative_control"
                                      : "skip");
  if (role == "skip") {
    json skip = baseRow;
    skip["row_kind"] = "basin_identity_skip";
    skip["identity_role"] = "skip";
    skip["ok"] = true;
    rows.push_back(skip);
    return rows;
  }
  if (sourceJsonl.empty()) {
    json fail = baseRow;
    fail["row_kind"] = "basin_endpoint_audit";
    fail["ok"] = false;
    fail["fit_failure_reason"] = "basin_source_jsonl_empty";
    rows.push_back(fail);
    return rows;
  }
  std::vector<json> endpoints;
  for (const auto& rec : basinSourceRows(sourceJsonl)) {
    if (jsonIntOr(rec, "run_id", -1) == runId &&
        jsonIntOr(rec, "event_id", -1) == eventId &&
        jsonIntOr(rec, "target_station", -1) == targetStation) {
      endpoints.push_back(rec);
    }
  }
  if (endpoints.size() != 4) {
    json fail = baseRow;
    fail["row_kind"] = "basin_endpoint_audit";
    fail["ok"] = false;
    fail["fit_failure_reason"] = "expected_four_wb129_endpoints";
    fail["n_source_endpoints"] = static_cast<int>(endpoints.size());
    rows.push_back(fail);
    return rows;
  }
  std::sort(endpoints.begin(), endpoints.end(), [](const json& a, const json& b) {
    return jsonStringOr(a, "profile_init_variant", "") <
           jsonStringOr(b, "profile_init_variant", "");
  });
  for (const auto& source : endpoints) {
    rows.push_back(endpointAuditRow(
        baseRow, source, role, propagator, geometryContext, magFieldContext, seed,
        hits, maxSteps, matchKalman, sequential, useScaling, supportingPlane,
        pinvRelative, gradientPropagator, fieldGradientRepair));
  }
  if (role != "failure") {
    return rows;
  }
  auto surface = seed.referenceSurface().getSharedPtr();
  std::array<Acts::BoundVector, 4> finals{};
  std::array<std::string, 4> names{};
  std::array<double, 4> chi2s{};
  std::array<bool, 4> valid{};
  for (int i = 0; i < 4; ++i) {
    names[i] = jsonStringOr(endpoints[static_cast<std::size_t>(i)],
                            "profile_init_variant", "");
    valid[i] = boundFromDump(endpoints[static_cast<std::size_t>(i)], finals[i]);
    chi2s[i] = jsonNumberOr(endpoints[static_cast<std::size_t>(i)], "chi2_prof",
                            std::numeric_limits<double>::quiet_NaN());
    valid[i] = valid[i] && std::isfinite(chi2s[i]);
  }
  for (int ia = 0; ia < 4; ++ia) {
    if (!valid[ia]) {
      continue;
    }
    for (int jn = 0; jn < 4; ++jn) {
      if (!valid[jn]) {
        continue;
      }
      Acts::BoundVector start = finals[ia];
      start[kNuIdx[0]] = finals[jn][kNuIdx[0]];
      start[kNuIdx[1]] = finals[jn][kNuIdx[1]];
      start[kNuIdx[2]] = finals[jn][kNuIdx[2]];
      const NuisanceOptResult result = runNuisanceOnly(
          propagator, geometryContext, magFieldContext, surface, start, hits,
          maxIterations, pinvRelative, maxSteps, matchKalman, sequential,
          useScaling, supportingPlane, gradientPropagator, fieldGradientRepair);
      rows.push_back(nuisanceResultJson(baseRow, "basin_fixed_alpha_cross_start",
                                        role, result, start, names[ia], names[jn],
                                        0.0, "cross_start"));
    }
  }
  if (!runContinuation) {
    return rows;
  }
  int iHigh = -1;
  int iLow = -1;
  for (int i = 0; i < 4; ++i) {
    if (!valid[i]) {
      continue;
    }
    if (iHigh < 0 || chi2s[i] > chi2s[iHigh]) {
      iHigh = i;
    }
    if (iLow < 0 || chi2s[i] < chi2s[iLow]) {
      iLow = i;
    }
  }
  if (iHigh < 0 || iLow < 0 || iHigh == iLow) {
    json skip = baseRow;
    skip["row_kind"] = "basin_profile_continuation";
    skip["ok"] = false;
    skip["fit_failure_reason"] = "continuation_endpoints_unavailable";
    rows.push_back(skip);
    return rows;
  }
  const Acts::BoundVector alphaA = finals[iHigh];
  const Acts::BoundVector alphaB = finals[iLow];
  const auto runOneDirection = [&](const char* direction, int iStart) {
    Acts::BoundVector nu = finals[iStart];
    for (int k = 0; k <= 20; ++k) {
      const double lambda =
          (std::string(direction) == "A_to_B") ? 0.05 * k : 1.0 - 0.05 * k;
      Acts::BoundVector start = nu;
      start[kAlphaIdx[0]] =
          (1.0 - lambda) * alphaA[kAlphaIdx[0]] + lambda * alphaB[kAlphaIdx[0]];
      start[kAlphaIdx[1]] =
          (1.0 - lambda) * alphaA[kAlphaIdx[1]] + lambda * alphaB[kAlphaIdx[1]];
      const NuisanceOptResult result = runNuisanceOnly(
          propagator, geometryContext, magFieldContext, surface, start, hits,
          maxIterations, pinvRelative, maxSteps, matchKalman, sequential,
          useScaling, supportingPlane, gradientPropagator, fieldGradientRepair);
      json row = nuisanceResultJson(
          baseRow, "basin_profile_continuation", role, result, start,
          names[iHigh], names[iStart], lambda, direction);
      row["alpha_A_restart"] = names[iHigh];
      row["alpha_B_restart"] = names[iLow];
      row["chi2_A"] = chi2s[iHigh];
      row["chi2_B"] = chi2s[iLow];
      row["selection_used_held_out_target"] = false;
      rows.push_back(row);
      if (result.ok) {
        nu = result.theta;
      }
    }
  };
  runOneDirection("A_to_B", iHigh);
  runOneDirection("B_to_A", iLow);
  return rows;
}
