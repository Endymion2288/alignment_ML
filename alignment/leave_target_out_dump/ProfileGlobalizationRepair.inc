// B14M-T: range-space trust-region profile solver.
// Included inside the anonymous namespace after ProfileBasinDiagnosis.inc.
// Does not modify runOneProfileNumerics.  LM lambda is a globalization
// multiplier only: not prior, not ridge, not information.

struct TrustRegionContract {
  double deltaMax{10.0};
  double deltaMin{1.0e-8};
  double deltaCap0{1.0};
  double rhoAccept{0.10};
  double rhoExpand{0.75};
  double boundaryFrac{0.80};
  double shrink{0.25};
  double expand{2.0};
  int maxTrials{20};
  int maxOuter{50};
  int maxInner{50};
  double pinvRelative{1.0e-8};
  double gradientNormZ{1.0e-6};
  double stepNormZ{1.0e-8};
  double relativeChi2{1.0e-8};
};

constexpr TrustRegionContract kFrozenTR{};

template <int N>
struct RangeTRStep {
  Eigen::Matrix<double, N, 1> delta = Eigen::Matrix<double, N, 1>::Zero();
  Eigen::Matrix<double, N, 1> gR = Eigen::Matrix<double, N, 1>::Zero();
  double normGR{0.0};
  double normDelta{0.0};
  double predictedChi2Decrease{0.0};
  double lambdaLM{0.0};
  int rank{0};
  bool nullPresent{false};
  bool ok{false};
};

template <int N>
RangeTRStep<N> rangeSpaceTrustRegionStep(
    const Eigen::Matrix<double, N, N>& hessian,
    const Eigen::Matrix<double, N, 1>& gradient, double radius,
    double pinvRelative) {
  using Mat = Eigen::Matrix<double, N, N>;
  RangeTRStep<N> out;
  Eigen::JacobiSVD<Mat> svd(hessian, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const auto singular = svd.singularValues();
  const Mat V = svd.matrixV();
  double smax = singular.maxCoeff();
  if (!(smax > 0.0) || !std::isfinite(smax)) {
    smax = 1.0;
  }
  std::array<int, N> keep{};
  int rank = 0;
  for (int i = 0; i < N; ++i) {
    if (singular[i] > pinvRelative * smax) {
      keep[static_cast<std::size_t>(rank++)] = i;
    }
  }
  out.rank = rank;
  out.nullPresent = rank < N;
  Mat projR = Mat::Zero();
  for (int k = 0; k < rank; ++k) {
    projR += V.col(keep[static_cast<std::size_t>(k)]) *
             V.col(keep[static_cast<std::size_t>(k)]).transpose();
  }
  out.gR = projR * gradient;
  out.normGR = out.gR.norm();
  if (rank == 0 || !(radius > 0.0)) {
    out.ok = true;
    return out;
  }
  Eigen::Matrix<double, Eigen::Dynamic, 1> ghat(rank);
  Eigen::Matrix<double, Eigen::Dynamic, 1> sR(rank);
  Eigen::Matrix<double, N, Eigen::Dynamic> vR(N, rank);
  for (int k = 0; k < rank; ++k) {
    const int idx = keep[static_cast<std::size_t>(k)];
    vR.col(k) = V.col(idx);
    sR[k] = singular[idx];
    ghat[k] = V.col(idx).dot(gradient);
  }
  Eigen::Matrix<double, Eigen::Dynamic, 1> dhat(rank);
  for (int k = 0; k < rank; ++k) {
    dhat[k] = ghat[k] / sR[k];
  }
  double lambda = 0.0;
  if (dhat.norm() > radius) {
    lambda = 0.0;
    for (int it = 0; it < 60; ++it) {
      double f = 0.0;
      double fp = 0.0;
      for (int k = 0; k < rank; ++k) {
        const double den = sR[k] + lambda;
        const double t = ghat[k] / den;
        f += t * t;
        fp += -2.0 * ghat[k] * ghat[k] / (den * den * den);
      }
      f -= radius * radius;
      if (std::abs(f) <= 1.0e-14 * std::max(radius * radius, 1.0)) {
        break;
      }
      if (!(std::abs(fp) > 0.0)) {
        break;
      }
      double next = lambda - f / fp;
      if (!(next >= 0.0) || !std::isfinite(next)) {
        next = std::max(0.5 * lambda, 1.0e-16);
      }
      lambda = next;
    }
    for (int k = 0; k < rank; ++k) {
      dhat[k] = ghat[k] / (sR[k] + lambda);
    }
    if (dhat.norm() > radius && dhat.norm() > 0.0) {
      dhat *= radius / dhat.norm();
    }
  }
  out.lambdaLM = lambda;
  out.delta = vR * dhat;
  out.normDelta = out.delta.norm();
  const double gTd = gradient.dot(out.delta);
  const double dHd = out.delta.dot(hessian * out.delta);
  out.predictedChi2Decrease = 2.0 * gTd - dHd;
  out.ok = true;
  return out;
}

struct ProfileSolveResult {
  bool ok{false};
  bool validSolution{false};
  bool validStationary{false};
  bool innerStationary{false};
  bool outerStationary{false};
  double chi2{0.0};
  Acts::BoundVector theta = Acts::BoundVector::Zero();
  std::string termination{"unset"};
  std::string innerTermination{"unset"};
  std::string outerTermination{"unset"};
  int outerIters{0};
  int innerItersTotal{0};
  int acceptedTrials{0};
  int rejectedTrials{0};
  double finalDeltaInner{0.0};
  double finalDeltaOuter{0.0};
  double normGnR{0.0};
  double normGaProfZ{0.0};
  double normGR{0.0};
  int rankH{0};
  int rankHnn{0};
  bool nullPresent{false};
  Chi2Eval eval;
  json innerTrace = json::array();
  json outerTrace = json::array();
  json trialTrace = json::array();
};

Bound5 numericScale(bool useScaling) {
  Bound5 scale = Bound5::Ones();
  if (useScaling) {
    for (int i = 0; i < 5; ++i) {
      scale[i] = kNumericScale[static_cast<std::size_t>(i)];
    }
  }
  return scale;
}

void extractNuBlocks(const ScaledSystem& sys, bool useScaling, Mat3& hnnZ,
                     Vec3& gnZ) {
  const Bound5 scale = numericScale(useScaling);
  hnnZ = Mat3::Zero();
  gnZ = Vec3::Zero();
  for (int i = 0; i < 3; ++i) {
    gnZ[i] = sys.gradient[kNuIdx[i]] * scale[kNuIdx[i]];
    for (int j = 0; j < 3; ++j) {
      hnnZ(i, j) = sys.hessian(kNuIdx[i], kNuIdx[j]) * scale[kNuIdx[i]] *
                   scale[kNuIdx[j]];
    }
  }
}

void extractSchurAlpha(const ScaledSystem& sys, bool useScaling, double pinvRel,
                       Eigen::Matrix<double, 2, 2>& hSchur,
                       Eigen::Matrix<double, 2, 1>& gProf, int* rankNn) {
  const Bound5 scale = numericScale(useScaling);
  Eigen::Matrix<double, 2, 1> gaZ;
  Vec3 gnZ;
  Eigen::Matrix<double, 2, 3> hanZ;
  Mat3 hnnZ = Mat3::Zero();
  Eigen::Matrix<double, 2, 2> haaZ = Eigen::Matrix<double, 2, 2>::Zero();
  for (int a = 0; a < 2; ++a) {
    gaZ[a] = sys.gradient[kAlphaIdx[a]] * scale[kAlphaIdx[a]];
    for (int b = 0; b < 2; ++b) {
      haaZ(a, b) = sys.hessian(kAlphaIdx[a], kAlphaIdx[b]) * scale[kAlphaIdx[a]] *
                   scale[kAlphaIdx[b]];
    }
    for (int n = 0; n < 3; ++n) {
      hanZ(a, n) = sys.hessian(kAlphaIdx[a], kNuIdx[n]) * scale[kAlphaIdx[a]] *
                   scale[kNuIdx[n]];
    }
  }
  for (int i = 0; i < 3; ++i) {
    gnZ[i] = sys.gradient[kNuIdx[i]] * scale[kNuIdx[i]];
    for (int j = 0; j < 3; ++j) {
      hnnZ(i, j) = sys.hessian(kNuIdx[i], kNuIdx[j]) * scale[kNuIdx[i]] *
                   scale[kNuIdx[j]];
    }
  }
  const Mat3 hnnP = symmetricPinv3(hnnZ, pinvRel, rankNn);
  hSchur = haaZ - hanZ * hnnP * hanZ.transpose();
  hSchur = 0.5 * (hSchur + hSchur.transpose());
  gProf = gaZ - hanZ * hnnP * gnZ;
}

Acts::BoundVector applyNuDelta(const Acts::BoundVector& theta,
                               const Vec3& deltaZ, bool useScaling,
                               bool& hitBoundary) {
  const Bound5 scale = numericScale(useScaling);
  Acts::BoundVector trial = theta;
  for (int i = 0; i < 3; ++i) {
    trial[kNuIdx[i]] += deltaZ[i] * scale[kNuIdx[i]];
  }
  trial[kAlphaIdx[0]] = theta[kAlphaIdx[0]];
  trial[kAlphaIdx[1]] = theta[kAlphaIdx[1]];
  applyPhysicalBounds(trial, hitBoundary);
  trial[kAlphaIdx[0]] = theta[kAlphaIdx[0]];
  trial[kAlphaIdx[1]] = theta[kAlphaIdx[1]];
  return trial;
}

Acts::BoundVector applyAlphaDelta(const Acts::BoundVector& theta,
                                  const Eigen::Matrix<double, 2, 1>& deltaZ,
                                  bool useScaling, bool& hitBoundary) {
  const Bound5 scale = numericScale(useScaling);
  Acts::BoundVector trial = theta;
  for (int i = 0; i < 2; ++i) {
    trial[kAlphaIdx[i]] += deltaZ[i] * scale[kAlphaIdx[i]];
  }
  applyPhysicalBounds(trial, hitBoundary);
  return trial;
}

ProfileSolveResult profileNuisanceTR(
    const Propagator& propagator, const Acts::GeometryContext& geometryContext,
    const Acts::MagneticFieldContext& magFieldContext,
    const std::shared_ptr<const Acts::Surface>& surface,
    const Acts::BoundVector& start, const std::vector<ProfileHit>& hits,
    const TrustRegionContract& tr, int maxSteps, bool matchKalman,
    bool sequential, bool useScaling, bool supportingPlane,
    const GradientPropagator* gradientPropagator, bool fieldGradientRepair,
    json* trialSink) {
  ProfileSolveResult out;
  Acts::BoundVector theta = start;
  bool hitBoundary = false;
  if (!applyPhysicalBounds(theta, hitBoundary)) {
    out.termination = "nonfinite_initialization";
    out.innerTermination = out.termination;
    return out;
  }
  theta[kAlphaIdx[0]] = start[kAlphaIdx[0]];
  theta[kAlphaIdx[1]] = start[kAlphaIdx[1]];
  ScaledSystem sys = evaluateScaledSystem(
      propagator, geometryContext, magFieldContext,
      Acts::BoundTrackParameters(surface, theta, std::nullopt,
                                 Acts::ParticleHypothesis::muon()),
      hits, maxSteps, matchKalman, sequential, supportingPlane, tr.pinvRelative,
      useScaling, gradientPropagator, fieldGradientRepair);
  if (!sys.ok) {
    out.termination = "propagation_blocked";
    out.innerTermination = out.termination;
    return out;
  }
  Mat3 hnn = Mat3::Zero();
  Vec3 gn = Vec3::Zero();
  extractNuBlocks(sys, useScaling, hnn, gn);
  auto gnStep = rangeSpaceTrustRegionStep<3>(hnn, gn, tr.deltaMax, tr.pinvRelative);
  double radius = std::min(tr.deltaCap0, std::max(gnStep.normDelta, tr.deltaMin));
  std::string termination = "max_iterations";
  int nIter = 0;
  int accepted = 0;
  int rejected = 0;
  for (nIter = 0; nIter < tr.maxInner; ++nIter) {
    sys = evaluateScaledSystem(
        propagator, geometryContext, magFieldContext,
        Acts::BoundTrackParameters(surface, theta, std::nullopt,
                                   Acts::ParticleHypothesis::muon()),
        hits, maxSteps, matchKalman, sequential, supportingPlane, tr.pinvRelative,
        useScaling, gradientPropagator, fieldGradientRepair);
    if (!sys.ok) {
      termination = "propagation_blocked";
      break;
    }
    extractNuBlocks(sys, useScaling, hnn, gn);
    gnStep = rangeSpaceTrustRegionStep<3>(hnn, gn, radius, tr.pinvRelative);
    out.normGnR = gnStep.normGR;
    out.rankHnn = gnStep.rank;
    out.nullPresent = gnStep.nullPresent;
    if (nIter == 0) {
      radius = std::min(tr.deltaCap0, std::max(gnStep.normDelta, tr.deltaMin));
      gnStep = rangeSpaceTrustRegionStep<3>(hnn, gn, radius, tr.pinvRelative);
    }
    if (gnStep.normGR <= tr.gradientNormZ) {
      termination = "supported_stationary";
      out.innerStationary = true;
      break;
    }
    if (gnStep.normDelta <= tr.stepNormZ) {
      termination = (gnStep.normGR <= tr.gradientNormZ) ? "step_converged"
                                                        : "globalization_failure";
      out.innerStationary = gnStep.normGR <= tr.gradientNormZ;
      break;
    }
    bool acceptedStep = false;
    json lastTrial;
    for (int trial = 0; trial < tr.maxTrials; ++trial) {
      if (radius < tr.deltaMin) {
        termination = (gnStep.normGR > tr.gradientNormZ) ? "globalization_failure"
                                                         : "supported_stationary";
        out.innerStationary = gnStep.normGR <= tr.gradientNormZ;
        break;
      }
      auto step =
          rangeSpaceTrustRegionStep<3>(hnn, gn, radius, tr.pinvRelative);
      lastTrial = {{"scope", "inner_nu"},
                   {"iteration", nIter},
                   {"trial", trial},
                   {"Delta", radius},
                   {"lambda_lm_not_prior", step.lambdaLM},
                   {"predicted_decrease", step.predictedChi2Decrease},
                   {"norm_g_n_R", step.normGR},
                   {"norm_delta_R", step.normDelta}};
      if (!(step.predictedChi2Decrease > 0.0)) {
        radius = std::max(tr.deltaMin, radius * tr.shrink);
        ++rejected;
        lastTrial["status"] = "nonpositive_model_decrease";
        if (trialSink != nullptr) {
          trialSink->push_back(lastTrial);
        }
        continue;
      }
      bool trialBound = hitBoundary;
      const Acts::BoundVector cand =
          applyNuDelta(theta, step.delta, useScaling, trialBound);
      Chi2Eval trialEval = evaluateSequentialResiduals(
          propagator, geometryContext, magFieldContext,
          Acts::BoundTrackParameters(surface, cand, std::nullopt,
                                     Acts::ParticleHypothesis::muon()),
          hits, maxSteps, matchKalman, sequential, supportingPlane);
      lastTrial["transport_valid"] = trialEval.ok;
      if (!trialEval.ok) {
        radius = std::max(tr.deltaMin, radius * tr.shrink);
        ++rejected;
        lastTrial["status"] = "propagation_rejected";
        if (trialSink != nullptr) {
          trialSink->push_back(lastTrial);
        }
        continue;
      }
      const double actual = sys.chi2 - trialEval.chi2;
      const double rho = actual / step.predictedChi2Decrease;
      lastTrial["chi2_trial"] = trialEval.chi2;
      lastTrial["actual_decrease"] = actual;
      lastTrial["rho"] = rho;
      if (rho >= tr.rhoAccept) {
        theta = cand;
        hitBoundary = trialBound;
        acceptedStep = true;
        ++accepted;
        lastTrial["status"] = "accepted";
        if (rho >= tr.rhoExpand &&
            step.normDelta >= tr.boundaryFrac * radius) {
          radius = std::min(tr.deltaMax, radius * tr.expand);
        }
        if (trialSink != nullptr) {
          trialSink->push_back(lastTrial);
        }
        const double relDec = actual / std::max(sys.chi2, 1.0);
        if (relDec < tr.relativeChi2 && step.normGR <= tr.gradientNormZ) {
          termination = "supported_stationary";
          out.innerStationary = true;
        }
        break;
      }
      radius = std::max(tr.deltaMin, radius * tr.shrink);
      ++rejected;
      lastTrial["status"] = "rejected_rho";
      if (trialSink != nullptr) {
        trialSink->push_back(lastTrial);
      }
    }
    out.innerTrace.push_back({{"iteration", nIter},
                              {"chi2", sys.chi2},
                              {"norm_g_n_R", gnStep.normGR},
                              {"Delta", radius},
                              {"accepted", acceptedStep},
                              {"rank_H_nn", gnStep.rank}});
    if (out.innerStationary) {
      break;
    }
    if (!acceptedStep) {
      if (radius < tr.deltaMin && gnStep.normGR > tr.gradientNormZ) {
        termination = "globalization_failure";
      } else if (gnStep.normGR <= tr.gradientNormZ) {
        termination = "supported_stationary";
        out.innerStationary = true;
      } else {
        termination = "globalization_failure";
      }
      break;
    }
  }
  sys = evaluateScaledSystem(
      propagator, geometryContext, magFieldContext,
      Acts::BoundTrackParameters(surface, theta, std::nullopt,
                                 Acts::ParticleHypothesis::muon()),
      hits, maxSteps, matchKalman, sequential, supportingPlane, tr.pinvRelative,
      useScaling, gradientPropagator, fieldGradientRepair);
  extractNuBlocks(sys, useScaling, hnn, gn);
  gnStep = rangeSpaceTrustRegionStep<3>(hnn, gn, radius, tr.pinvRelative);
  out.ok = sys.ok;
  out.chi2 = sys.chi2;
  out.theta = theta;
  out.eval = sys.eval;
  out.innerItersTotal = nIter;
  out.acceptedTrials = accepted;
  out.rejectedTrials = rejected;
  out.finalDeltaInner = radius;
  out.normGnR = gnStep.normGR;
  out.rankHnn = gnStep.rank;
  out.innerTermination = termination;
  out.termination = termination;
  out.innerStationary = sys.ok && gnStep.normGR <= tr.gradientNormZ;
  out.validSolution = out.innerStationary && termination != "globalization_failure";
  return out;
}

ProfileSolveResult runExplicitProfileTR(
    const Propagator& propagator, const Acts::GeometryContext& geometryContext,
    const Acts::MagneticFieldContext& magFieldContext,
    const std::shared_ptr<const Acts::Surface>& surface,
    const Acts::BoundVector& start, const std::vector<ProfileHit>& hits,
    const TrustRegionContract& tr, int maxSteps, bool matchKalman,
    bool sequential, bool useScaling, bool supportingPlane,
    const GradientPropagator* gradientPropagator, bool fieldGradientRepair) {
  ProfileSolveResult out;
  json trials = json::array();
  ProfileSolveResult inner = profileNuisanceTR(
      propagator, geometryContext, magFieldContext, surface, start, hits, tr,
      maxSteps, matchKalman, sequential, useScaling, supportingPlane,
      gradientPropagator, fieldGradientRepair, &trials);
  out.innerTrace = inner.innerTrace;
  out.innerItersTotal += inner.innerItersTotal;
  out.acceptedTrials += inner.acceptedTrials;
  out.rejectedTrials += inner.rejectedTrials;
  out.finalDeltaInner = inner.finalDeltaInner;
  out.innerTermination = inner.innerTermination;
  if (!inner.ok || !inner.innerStationary) {
    out = inner;
    out.outerTermination = "inner_not_converged";
    out.validSolution = false;
    out.validStationary = false;
    out.trialTrace = trials;
    return out;
  }
  Acts::BoundVector theta = inner.theta;
  ScaledSystem sys = evaluateScaledSystem(
      propagator, geometryContext, magFieldContext,
      Acts::BoundTrackParameters(surface, theta, std::nullopt,
                                 Acts::ParticleHypothesis::muon()),
      hits, maxSteps, matchKalman, sequential, supportingPlane, tr.pinvRelative,
      useScaling, gradientPropagator, fieldGradientRepair);
  if (!sys.ok) {
    out.termination = "propagation_blocked";
    out.outerTermination = out.termination;
    return out;
  }
  Eigen::Matrix<double, 2, 2> hSchur;
  Eigen::Matrix<double, 2, 1> gProf;
  int rankNn = 0;
  extractSchurAlpha(sys, useScaling, tr.pinvRelative, hSchur, gProf, &rankNn);
  auto aStep =
      rangeSpaceTrustRegionStep<2>(hSchur, gProf, tr.deltaMax, tr.pinvRelative);
  double radius = std::min(tr.deltaCap0, std::max(aStep.normDelta, tr.deltaMin));
  std::string termination = "max_iterations";
  int nIter = 0;
  for (nIter = 0; nIter < tr.maxOuter; ++nIter) {
    sys = evaluateScaledSystem(
        propagator, geometryContext, magFieldContext,
        Acts::BoundTrackParameters(surface, theta, std::nullopt,
                                   Acts::ParticleHypothesis::muon()),
        hits, maxSteps, matchKalman, sequential, supportingPlane, tr.pinvRelative,
        useScaling, gradientPropagator, fieldGradientRepair);
    if (!sys.ok) {
      termination = "propagation_blocked";
      break;
    }
    extractSchurAlpha(sys, useScaling, tr.pinvRelative, hSchur, gProf, &rankNn);
    Mat3 hnn = Mat3::Zero();
    Vec3 gn = Vec3::Zero();
    extractNuBlocks(sys, useScaling, hnn, gn);
    auto nStep =
        rangeSpaceTrustRegionStep<3>(hnn, gn, tr.deltaMax, tr.pinvRelative);
    if (nStep.normGR > tr.gradientNormZ) {
      inner = profileNuisanceTR(
          propagator, geometryContext, magFieldContext, surface, theta, hits, tr,
          maxSteps, matchKalman, sequential, useScaling, supportingPlane,
          gradientPropagator, fieldGradientRepair, &trials);
      out.innerItersTotal += inner.innerItersTotal;
      out.acceptedTrials += inner.acceptedTrials;
      out.rejectedTrials += inner.rejectedTrials;
      if (!inner.innerStationary) {
        termination = "inner_nuisance_profile_not_converged";
        theta = inner.theta;
        break;
      }
      theta = inner.theta;
      sys = evaluateScaledSystem(
          propagator, geometryContext, magFieldContext,
          Acts::BoundTrackParameters(surface, theta, std::nullopt,
                                     Acts::ParticleHypothesis::muon()),
          hits, maxSteps, matchKalman, sequential, supportingPlane,
          tr.pinvRelative, useScaling, gradientPropagator, fieldGradientRepair);
      if (!sys.ok) {
        termination = "propagation_blocked";
        break;
      }
      extractSchurAlpha(sys, useScaling, tr.pinvRelative, hSchur, gProf, &rankNn);
    }
    aStep = rangeSpaceTrustRegionStep<2>(hSchur, gProf, radius, tr.pinvRelative);
    out.normGaProfZ = aStep.normGR;
    out.rankHnn = rankNn;
    if (nIter == 0) {
      radius = std::min(tr.deltaCap0, std::max(aStep.normDelta, tr.deltaMin));
      aStep = rangeSpaceTrustRegionStep<2>(hSchur, gProf, radius, tr.pinvRelative);
    }
    if (aStep.normGR <= tr.gradientNormZ) {
      termination = "supported_stationary";
      out.outerStationary = true;
      break;
    }
    if (aStep.normDelta <= tr.stepNormZ) {
      termination = (aStep.normGR <= tr.gradientNormZ) ? "step_converged"
                                                       : "globalization_failure";
      out.outerStationary = aStep.normGR <= tr.gradientNormZ;
      break;
    }
    bool acceptedStep = false;
    for (int trial = 0; trial < tr.maxTrials; ++trial) {
      if (radius < tr.deltaMin) {
        termination = (aStep.normGR > tr.gradientNormZ) ? "globalization_failure"
                                                        : "supported_stationary";
        out.outerStationary = aStep.normGR <= tr.gradientNormZ;
        break;
      }
      auto step =
          rangeSpaceTrustRegionStep<2>(hSchur, gProf, radius, tr.pinvRelative);
      json item = {{"scope", "outer_alpha"},
                   {"iteration", nIter},
                   {"trial", trial},
                   {"Delta", radius},
                   {"lambda_lm_not_prior", step.lambdaLM},
                   {"predicted_decrease", step.predictedChi2Decrease},
                   {"norm_g_alpha_prof_z", step.normGR},
                   {"norm_delta_R", step.normDelta},
                   {"reprofile_required", true}};
      if (!(step.predictedChi2Decrease > 0.0)) {
        radius = std::max(tr.deltaMin, radius * tr.shrink);
        ++out.rejectedTrials;
        item["status"] = "nonpositive_model_decrease";
        trials.push_back(item);
        continue;
      }
      bool trialBound = false;
      Acts::BoundVector cand =
          applyAlphaDelta(theta, step.delta, useScaling, trialBound);
      cand[kNuIdx[0]] = theta[kNuIdx[0]];
      cand[kNuIdx[1]] = theta[kNuIdx[1]];
      cand[kNuIdx[2]] = theta[kNuIdx[2]];
      ProfileSolveResult trialInner = profileNuisanceTR(
          propagator, geometryContext, magFieldContext, surface, cand, hits, tr,
          maxSteps, matchKalman, sequential, useScaling, supportingPlane,
          gradientPropagator, fieldGradientRepair, &trials);
      out.innerItersTotal += trialInner.innerItersTotal;
      item["inner_stationary"] = trialInner.innerStationary;
      item["inner_termination"] = trialInner.innerTermination;
      if (!trialInner.ok || !trialInner.innerStationary) {
        radius = std::max(tr.deltaMin, radius * tr.shrink);
        ++out.rejectedTrials;
        item["status"] = "inner_reprofile_failed";
        trials.push_back(item);
        continue;
      }
      const double actual = sys.chi2 - trialInner.chi2;
      const double rho = actual / step.predictedChi2Decrease;
      item["chi2_prof_trial"] = trialInner.chi2;
      item["actual_decrease"] = actual;
      item["rho"] = rho;
      if (rho >= tr.rhoAccept) {
        theta = trialInner.theta;
        acceptedStep = true;
        ++out.acceptedTrials;
        item["status"] = "accepted";
        if (rho >= tr.rhoExpand &&
            step.normDelta >= tr.boundaryFrac * radius) {
          radius = std::min(tr.deltaMax, radius * tr.expand);
        }
        trials.push_back(item);
        const double relDec = actual / std::max(sys.chi2, 1.0);
        if (relDec < tr.relativeChi2 && step.normGR <= tr.gradientNormZ) {
          termination = "supported_stationary";
          out.outerStationary = true;
        }
        break;
      }
      radius = std::max(tr.deltaMin, radius * tr.shrink);
      ++out.rejectedTrials;
      item["status"] = "rejected_rho";
      trials.push_back(item);
    }
    out.outerTrace.push_back({{"iteration", nIter},
                              {"chi2_prof", sys.chi2},
                              {"norm_g_alpha_prof_z", aStep.normGR},
                              {"Delta", radius},
                              {"accepted", acceptedStep}});
    if (out.outerStationary) {
      break;
    }
    if (!acceptedStep) {
      if (radius < tr.deltaMin && aStep.normGR > tr.gradientNormZ) {
        termination = "globalization_failure";
      } else if (aStep.normGR <= tr.gradientNormZ) {
        termination = "supported_stationary";
        out.outerStationary = true;
      } else {
        termination = "globalization_failure";
      }
      break;
    }
  }
  inner = profileNuisanceTR(
      propagator, geometryContext, magFieldContext, surface, theta, hits, tr,
      maxSteps, matchKalman, sequential, useScaling, supportingPlane,
      gradientPropagator, fieldGradientRepair, &trials);
  out.innerItersTotal += inner.innerItersTotal;
  theta = inner.theta;
  sys = evaluateScaledSystem(
      propagator, geometryContext, magFieldContext,
      Acts::BoundTrackParameters(surface, theta, std::nullopt,
                                 Acts::ParticleHypothesis::muon()),
      hits, maxSteps, matchKalman, sequential, supportingPlane, tr.pinvRelative,
      useScaling, gradientPropagator, fieldGradientRepair);
  extractSchurAlpha(sys, useScaling, tr.pinvRelative, hSchur, gProf, &rankNn);
  aStep = rangeSpaceTrustRegionStep<2>(hSchur, gProf, radius, tr.pinvRelative);
  Mat3 hnn = Mat3::Zero();
  Vec3 gn = Vec3::Zero();
  extractNuBlocks(sys, useScaling, hnn, gn);
  auto nStep =
      rangeSpaceTrustRegionStep<3>(hnn, gn, tr.deltaMax, tr.pinvRelative);
  auto jointStep =
      rangeSpaceTrustRegionStep<5>(sys.hz, sys.gz, radius, tr.pinvRelative);
  out.ok = sys.ok && inner.ok;
  out.chi2 = sys.chi2;
  out.theta = theta;
  out.eval = sys.eval;
  out.outerIters = nIter;
  out.finalDeltaOuter = radius;
  out.normGaProfZ = aStep.normGR;
  out.normGnR = nStep.normGR;
  out.normGR = jointStep.normGR;
  out.rankH = jointStep.rank;
  out.rankHnn = nStep.rank;
  out.nullPresent =
      jointStep.nullPresent || nStep.nullPresent || aStep.nullPresent;
  out.innerStationary = inner.innerStationary && nStep.normGR <= tr.gradientNormZ;
  out.outerStationary = aStep.normGR <= tr.gradientNormZ;
  out.validStationary = out.innerStationary && out.outerStationary;
  out.innerTermination = inner.innerTermination;
  out.outerTermination = termination;
  if (!out.innerStationary) {
    out.termination = "inner_nuisance_profile_not_converged";
  } else if (termination == "globalization_failure") {
    out.termination = "globalization_failure";
  } else if (out.validStationary) {
    out.termination = aStep.nullPresent ? "supported_stationary" : "supported_stationary";
  } else if (nIter >= tr.maxOuter && !out.validStationary) {
    out.termination = "max_iterations";
  } else {
    out.termination = termination;
  }
  out.validSolution = out.validStationary &&
                      out.termination != "globalization_failure" &&
                      out.termination != "inner_nuisance_profile_not_converged";
  out.trialTrace = trials;
  return out;
}

json profileGlobalizationRow(
    const json& baseRow, const ProfileSolveResult& result,
    const std::string& variant, double targetZ, const Propagator& propagator,
    const Acts::GeometryContext& geometryContext,
    const Acts::MagneticFieldContext& magFieldContext,
    const std::shared_ptr<const Acts::Surface>& surface, int maxSteps,
    bool matchKalman) {
  json row = baseRow;
  row["row_kind"] = "profile_optimize";
  row["row_role"] = "profile_optimize";
  row["transport_task"] = "B14MT";
  row["solver"] = "explicit_profile_range_space_trust_region";
  row["profile_init_variant"] = variant;
  row["restart_name"] = variant;
  row["likelihood_kind"] = "measurement_only_chi2";
  row["statistical_model_unchanged"] = true;
  row["prior_term_present"] = false;
  row["ridge_added"] = false;
  row["lm_lambda_is_not_prior"] = true;
  row["lm_lambda_is_not_ridge"] = true;
  row["lm_lambda_is_not_information"] = true;
  row["trust_region_is_algorithmic_globalization_only"] = true;
  row["do_not_take_min_of_four_restarts"] = true;
  row["held_out_target_inaccessible_during_fit"] = true;
  row["held_out_used_for_solver"] = false;
  row["ok"] = result.ok;
  row["fit_success"] = result.ok && result.eval.ok;
  row["profile_success"] = result.ok;
  row["valid_solution"] = result.validSolution;
  row["valid_stationary"] = result.validStationary;
  row["inner_nuisance_stationary"] = result.innerStationary;
  row["profile_alpha_stationary"] = result.outerStationary;
  row["termination_reason"] = result.termination;
  row["inner_termination"] = result.innerTermination;
  row["outer_termination"] = result.outerTermination;
  row["chi2"] = result.ok ? json(result.chi2) : json(nullptr);
  row["chi2_prof"] = row["chi2"];
  row["profile_n_iterations"] = result.outerIters;
  row["inner_iterations_total"] = result.innerItersTotal;
  row["trust_region_accepted_trials"] = result.acceptedTrials;
  row["trust_region_rejected_trials"] = result.rejectedTrials;
  row["final_Delta_inner"] = result.finalDeltaInner;
  row["final_Delta_outer"] = result.finalDeltaOuter;
  row["norm_g_n_R"] = result.normGnR;
  row["norm_g_alpha_prof_z"] = result.normGaProfZ;
  row["norm_g_R"] = result.normGR;
  row["norm_g_optimizer_z"] = result.normGR;
  row["hessian_rank"] = result.rankH;
  row["jacobian_implementation"] = "wb123_repaired_production_tangent";
  row["validated_jacobian_is_numerical_derivative"] = true;
  row["validated_jacobian_is_not_a_new_statistical_model"] = true;
  row["solver_is_not_joint_5d_stall"] = true;
  row["flat_direction_forbidden_as_success"] = true;
  row["rank_H_nn"] = result.rankHnn;
  row["null_direction_present"] = result.nullPresent;
  row["profiled_native_bound"] = nativeBoundJson(result.theta);
  row["optimizer_trace"] = result.outerTrace;
  row["inner_nuisance_trace"] = result.innerTrace;
  row["trust_region_trial_trace"] = result.trialTrace;
  row["flat_direction_is_not_a_successful_termination"] = true;
  if (result.ok && result.eval.ok) {
    row["surviving_predicted_measurement_loc0"] =
        predictedMeasurementLoc0Json(result.eval);
    row["transport_branch_identity"] = transportBranchIdentityJson(result.eval);
    row["propagation_success"] = true;
    const Acts::BoundTrackParameters fitted(
        surface, result.theta, std::nullopt, Acts::ParticleHypothesis::muon());
    double predictedTargetLoc0 = 0.0;
    const bool targetOk = predictSupportingPlaneLoc0(
        propagator, geometryContext, magFieldContext, fitted, targetZ, maxSteps,
        matchKalman, predictedTargetLoc0);
    row["predicted_target_loc0"] =
        targetOk ? json(predictedTargetLoc0) : json(nullptr);
    row["held_out_prediction_is_stage_b_diagnostic"] = true;
    row["held_out_prediction_not_used_for_selection"] = true;
    json residualJson = json::array();
    json weightJson = json::array();
    for (int i = 0; i < result.eval.residual.size(); ++i) {
      residualJson.push_back(result.eval.residual[i]);
    }
    row["residual_vector"] = residualJson;
  } else {
    row["propagation_success"] = false;
    row["predicted_target_loc0"] = nullptr;
    row["fit_failure_reason"] = result.termination;
  }
  return row;
}

std::vector<json> runProfileGlobalizationRepair(
    const json& baseRow, const Propagator& propagator,
    const Acts::GeometryContext& geometryContext,
    const Acts::MagneticFieldContext& magFieldContext,
    const Acts::BoundTrackParameters& seed, const std::vector<ProfileHit>& hits,
    double targetZ, int maxSteps, bool matchKalman, bool sequential,
    bool useScaling, bool supportingPlane,
    const GradientPropagator* gradientPropagator, bool fieldGradientRepair) {
  std::vector<json> rows;
  json eval = runOneProfileNumerics(
      baseRow, propagator, geometryContext, magFieldContext, seed, hits, targetZ,
      "nominal", "wb114_official_seed_mean", 50, kFrozenTR.pinvRelative, maxSteps,
      matchKalman, sequential, useScaling, true, false, supportingPlane, false, 0,
      false, false, 3, false, false, nullptr, false, false, false, false);
  eval["transport_task"] = "B14MT";
  eval["row_role"] = "evaluate_only_nominal_seed";
  eval["b14m_profile_globalization_repair"] = true;
  rows.push_back(eval);
  struct Variant {
    const char* name;
    int index;
    double add;
    double scale;
  };
  const std::array<Variant, 4> variants{{
      {"nominal", -1, 0.0, 1.0},
      {"loc1_plus_1mm", static_cast<int>(Acts::eBoundLoc1), 1.0, 1.0},
      {"phi_plus_1e-3", static_cast<int>(Acts::eBoundPhi), 1.0e-3, 1.0},
      {"qoverp_times_1p1", static_cast<int>(Acts::eBoundQOverP), 0.0, 1.1},
  }};
  auto surface = seed.referenceSurface().getSharedPtr();
  for (const auto& variant : variants) {
    Acts::BoundVector params = seed.parameters();
    if (variant.index >= 0) {
      params[variant.index] = params[variant.index] * variant.scale + variant.add;
    }
    const ProfileSolveResult result = runExplicitProfileTR(
        propagator, geometryContext, magFieldContext, surface, params, hits,
        kFrozenTR, maxSteps, matchKalman, sequential, useScaling, supportingPlane,
        gradientPropagator, fieldGradientRepair);
    json row = profileGlobalizationRow(
        baseRow, result, variant.name, targetZ, propagator, geometryContext,
        magFieldContext, surface, maxSteps, matchKalman);
    row["b14m_profile_globalization_repair"] = true;
    row["init_native"] = {params[Acts::eBoundLoc0], params[Acts::eBoundLoc1],
                          params[Acts::eBoundPhi], params[Acts::eBoundTheta],
                          params[Acts::eBoundQOverP] * 1_MeV};
    rows.push_back(row);
  }
  return rows;
}
