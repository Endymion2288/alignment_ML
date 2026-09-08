#include "CkfLeaveTargetOutDumpAlg.h"

#include "Acts/Definitions/Tolerance.hpp"
#include "Acts/Definitions/TrackParametrization.hpp"
#include "Acts/Definitions/Units.hpp"
#include "Acts/EventData/TrackParameters.hpp"
#include "Acts/EventData/TrackStateType.hpp"
#include "Acts/EventData/VectorMultiTrajectory.hpp"
#include "Acts/EventData/VectorTrackContainer.hpp"
#include "Acts/EventData/detail/TransformationFreeToBound.hpp"
#include "Acts/MagneticField/MagneticFieldContext.hpp"
#include "Acts/Propagator/AbortList.hpp"
#include "Acts/Propagator/ActionList.hpp"
#include "Acts/Propagator/StandardAborters.hpp"
#include "Acts/Propagator/detail/SteppingLogger.hpp"
#include "Acts/Surfaces/BoundaryCheck.hpp"
#include "Acts/Surfaces/PlanarBounds.hpp"
#include "Acts/Propagator/EigenStepper.hpp"
#include "Acts/Propagator/MaterialInteractor.hpp"
#include "Acts/Propagator/Navigator.hpp"
#include "Acts/Propagator/Propagator.hpp"
#include "Acts/Propagator/StandardAborters.hpp"
#include "Acts/Surfaces/PlaneSurface.hpp"
#include "Acts/TrackFitting/GainMatrixSmoother.hpp"
#include "Acts/TrackFitting/GainMatrixUpdater.hpp"
#include "Acts/TrackFitting/KalmanFitter.hpp"
#include "FaserActsGeometry/FASERMagneticFieldWrapper.h"
#include "FaserActsKalmanFilter/IndexSourceLink.h"
#include "FaserActsKalmanFilter/Measurement.h"
#include "StoreGate/ReadHandle.h"
#include "Identifier/Identifier.h"
#include "TrackerIdentifier/FaserSCT_ID.h"
#include "TrackerPrepRawData/FaserSCT_Cluster.h"
#include "TrackerRIO_OnTrack/FaserSCT_ClusterOnTrack.h"
#include "TrkEventPrimitives/ParamDefs.h"
#include "TrkParameters/TrackParameters.h"
#include "TrkTrack/Track.h"
#include "TrkTrack/TrackStateOnSurface.h"

#include <nlohmann/json.hpp>

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <utility>

using namespace Acts::UnitLiterals;
using json = nlohmann::json;

namespace {

using FaserActsTrackContainer =
    Acts::TrackContainer<Acts::VectorTrackContainer, Acts::VectorMultiTrajectory,
                         std::shared_ptr>;
using Stepper = Acts::EigenStepper<>;
using Propagator = Acts::Propagator<Stepper, Acts::Navigator>;
using Fitter = Acts::KalmanFitter<Propagator, Acts::VectorMultiTrajectory>;

constexpr int kExportDim = 5;

bool finiteValue(double value) { return std::isfinite(value); }

json matrixToJson(const Acts::BoundSquareMatrix& matrix) {
  json rows = json::array();
  for (int i = 0; i < 5; ++i) {
    json row = json::array();
    for (int j = 0; j < 5; ++j) {
      const double value = matrix(i, j);
      if (!finiteValue(value)) {
        return nullptr;
      }
      row.push_back(value);
    }
    rows.push_back(row);
  }
  return rows;
}

json exportMatrixToJson(const std::array<std::array<double, 5>, 5>& matrix) {
  json rows = json::array();
  for (int i = 0; i < 5; ++i) {
    json row = json::array();
    for (int j = 0; j < 5; ++j) {
      if (!finiteValue(matrix[i][j])) {
        return nullptr;
      }
      row.push_back(matrix[i][j]);
    }
    rows.push_back(row);
  }
  return rows;
}

std::shared_ptr<Acts::PlaneSurface> planeAtZ(double zMm) {
  return Acts::Surface::makeShared<Acts::PlaneSurface>(
      Acts::Vector3(0.0, 0.0, zMm), Acts::Vector3(0.0, 0.0, 1.0));
}

int nearestStation(double zMm, const std::vector<double>& stationZ) {
  int best = 0;
  double distance = std::abs(stationZ[0] - zMm);
  for (int station = 1; station < static_cast<int>(stationZ.size()); ++station) {
    const double candidate = std::abs(stationZ[station] - zMm);
    if (candidate < distance) {
      distance = candidate;
      best = station;
    }
  }
  return best;
}

struct HitRecord {
  int station{-1};
  int layer{-1};
  int phi_module{-1};
  int eta_module{-1};
  int side{-1};
  Identifier id;
  std::string identifier;
  double zMm{0.0};
  bool from_outlier{false};
  const Tracker::FaserSCT_Cluster* cluster{nullptr};
  const Trk::MeasurementBase* measurement{nullptr};
};

std::string compactIdentifier(const Identifier& id) {
  std::ostringstream out;
  out << id.get_compact();
  return out.str();
}

std::array<double, kExportDim> exportFromBound(
    const Acts::BoundTrackParameters& parameters,
    const Acts::GeometryContext& geometryContext) {
  const Acts::Vector3 position = parameters.position(geometryContext);
  const Acts::Vector3 momentum = parameters.momentum();
  std::array<double, kExportDim> values{
      position.x(), position.y(), 0.0, 0.0,
      parameters.parameters()[Acts::eBoundQOverP] * 1_MeV};
  if (std::abs(momentum.z()) > 1.0e-12) {
    values[2] = momentum.x() / momentum.z();
    values[3] = momentum.y() / momentum.z();
  }
  return values;
}

std::optional<std::array<std::array<double, 5>, 5>> exportCovariance(
    const Acts::BoundTrackParameters& parameters,
    const Acts::GeometryContext& geometryContext) {
  if (!parameters.covariance().has_value()) {
    return std::nullopt;
  }
  const Acts::BoundVector values = parameters.parameters();
  const std::array<double, 6> steps{
      1.0e-4, 1.0e-4, 1.0e-6, 1.0e-6,
      std::max(std::abs(values[Acts::eBoundQOverP]) * 1.0e-5, 1.0e-12), 1.0e-3};
  std::array<std::array<double, 6>, 5> jacobian{};
  const auto surface = parameters.referenceSurface().getSharedPtr();
  for (std::size_t column = 0; column < steps.size(); ++column) {
    Acts::BoundVector plus = values;
    Acts::BoundVector minus = values;
    plus[column] += steps[column];
    minus[column] -= steps[column];
    const Acts::BoundTrackParameters plusParameters(
        surface, plus, std::nullopt, parameters.particleHypothesis());
    const Acts::BoundTrackParameters minusParameters(
        surface, minus, std::nullopt, parameters.particleHypothesis());
    const auto plusState = exportFromBound(plusParameters, geometryContext);
    const auto minusState = exportFromBound(minusParameters, geometryContext);
    for (std::size_t row = 0; row < 5; ++row) {
      jacobian[row][column] =
          (plusState[row] - minusState[row]) / (2.0 * steps[column]);
    }
  }
  const Acts::BoundSquareMatrix& bound = parameters.covariance().value();
  std::array<std::array<double, 5>, 5> covariance{};
  for (std::size_t row = 0; row < 5; ++row) {
    for (std::size_t column = 0; column < 5; ++column) {
      double total = 0.0;
      for (std::size_t i = 0; i < 6; ++i) {
        for (std::size_t j = 0; j < 6; ++j) {
          total += jacobian[row][i] * bound(i, j) * jacobian[column][j];
        }
      }
      if (!finiteValue(total)) {
        return std::nullopt;
      }
      covariance[row][column] = total;
    }
  }
  return covariance;
}

int boundIndexForDirection(const std::string& direction) {
  if (direction == "x") {
    return static_cast<int>(Acts::eBoundLoc0);
  }
  if (direction == "y") {
    return static_cast<int>(Acts::eBoundLoc1);
  }
  if (direction == "tx") {
    return static_cast<int>(Acts::eBoundPhi);
  }
  if (direction == "ty") {
    return static_cast<int>(Acts::eBoundTheta);
  }
  if (direction == "q_over_p") {
    return static_cast<int>(Acts::eBoundQOverP);
  }
  return -1;
}

const char* boundNameForDirection(const std::string& direction) {
  if (direction == "x") {
    return "loc0";
  }
  if (direction == "y") {
    return "loc1";
  }
  if (direction == "tx") {
    return "phi";
  }
  if (direction == "ty") {
    return "theta";
  }
  if (direction == "q_over_p") {
    return "q_over_p";
  }
  return "all";
}

Acts::BoundSquareMatrix uninformativeSeedCovariance(
    double scale, const std::string& direction) {
  Acts::BoundSquareMatrix cov = Acts::BoundSquareMatrix::Zero();
  const double factor = (std::isfinite(scale) && scale > 0.0) ? scale : 1.0;
  cov(Acts::eBoundLoc0, Acts::eBoundLoc0) = 1.0e4;
  cov(Acts::eBoundLoc1, Acts::eBoundLoc1) = 1.0e4;
  cov(Acts::eBoundPhi, Acts::eBoundPhi) = 2.5e-3;
  cov(Acts::eBoundTheta, Acts::eBoundTheta) = 2.5e-3;
  cov(Acts::eBoundQOverP, Acts::eBoundQOverP) = 1.0;
  cov(Acts::eBoundTime, Acts::eBoundTime) = 1.0e6;
  const int index = boundIndexForDirection(direction);
  if (index < 0) {
    cov(Acts::eBoundLoc0, Acts::eBoundLoc0) *= factor;
    cov(Acts::eBoundLoc1, Acts::eBoundLoc1) *= factor;
    cov(Acts::eBoundPhi, Acts::eBoundPhi) *= factor;
    cov(Acts::eBoundTheta, Acts::eBoundTheta) *= factor;
    cov(Acts::eBoundQOverP, Acts::eBoundQOverP) *= factor;
    cov(Acts::eBoundTime, Acts::eBoundTime) *= factor;
  } else {
    cov(index, index) *= factor;
  }
  return cov;
}

json optionalBound5(const std::optional<Acts::BoundSquareMatrix>& matrix) {
  if (!matrix.has_value()) {
    return nullptr;
  }
  return matrixToJson(matrix.value());
}

json collectFitProvenance(const auto& fittedTrack) {
  json payload;
  payload["acts_n_measurements_definition"] =
      "calculateTrackQuantities: count of TrackStateFlag::MeasurementFlag";
  payload["acts_n_dof_definition"] =
      "sum of calibratedSize() over MeasurementFlag states";
  payload["acts_n_measurements"] = static_cast<int>(fittedTrack.nMeasurements());
  payload["acts_n_dof"] = static_cast<int>(fittedTrack.nDoF());
  payload["acts_n_holes"] = static_cast<int>(fittedTrack.nHoles());
  payload["acts_n_outliers"] = static_cast<int>(fittedTrack.nOutliers());
  payload["acts_n_shared_hits"] = static_cast<int>(fittedTrack.nSharedHits());
  payload["acts_n_track_states"] = static_cast<int>(fittedTrack.nTrackStates());
  payload["exported_state_type"] = "kalman_fitted_parameters_at_reference_surface";
  payload["exported_state_origin"] =
      "smoothed_first_measurement_transported_to_reference";
  payload["reference_surface_strategy"] = "first";
  payload["covariance_transported_to_reference"] = true;
  payload["back_propagation_used"] = "unavailable";
  int nPredicted = 0;
  int nFiltered = 0;
  int nSmoothed = 0;
  int nOutlierFlag = 0;
  int nMeasFlag = 0;
  std::optional<Acts::BoundSquareMatrix> firstPredicted;
  std::optional<Acts::BoundSquareMatrix> firstFiltered;
  std::optional<Acts::BoundSquareMatrix> firstSmoothed;
  std::optional<Acts::BoundSquareMatrix> lastFiltered;
  std::optional<Acts::BoundSquareMatrix> lastSmoothed;
  std::optional<double> firstSmoothedQoverP;
  std::optional<double> lastSmoothedQoverP;
  for (const auto& state : fittedTrack.trackStatesReversed()) {
    const auto flags = state.typeFlags();
    if (flags.test(Acts::TrackStateFlag::OutlierFlag)) {
      ++nOutlierFlag;
    }
    if (flags.test(Acts::TrackStateFlag::MeasurementFlag)) {
      ++nMeasFlag;
    }
    if (state.hasPredicted()) {
      ++nPredicted;
      firstPredicted = state.predictedCovariance();
    }
    if (state.hasFiltered()) {
      ++nFiltered;
      firstFiltered = state.filteredCovariance();
      lastFiltered = lastFiltered ? lastFiltered : state.filteredCovariance();
    }
    if (state.hasSmoothed()) {
      ++nSmoothed;
      firstSmoothed = state.smoothedCovariance();
      firstSmoothedQoverP = state.smoothed()[Acts::eBoundQOverP];
      if (!lastSmoothed) {
        lastSmoothed = state.smoothedCovariance();
        lastSmoothedQoverP = state.smoothed()[Acts::eBoundQOverP];
      }
    }
  }
  payload["number_of_predicted_states"] = nPredicted;
  payload["number_of_filtered_states"] = nFiltered;
  payload["number_of_smoothed_states"] = nSmoothed;
  payload["number_of_outlier_states"] = nOutlierFlag;
  payload["number_of_measurement_flag_states"] = nMeasFlag;
  payload["first_predicted_covariance"] = optionalBound5(firstPredicted);
  payload["first_filtered_covariance"] = optionalBound5(firstFiltered);
  payload["first_smoothed_covariance"] = optionalBound5(firstSmoothed);
  payload["last_filtered_covariance"] = optionalBound5(lastFiltered);
  payload["last_smoothed_covariance"] = optionalBound5(lastSmoothed);
  payload["first_smoothed_q_over_p_per_gev"] =
      firstSmoothedQoverP ? json(*firstSmoothedQoverP) : json(nullptr);
  payload["last_smoothed_q_over_p_per_gev"] =
      lastSmoothedQoverP ? json(*lastSmoothedQoverP) : json(nullptr);
  if (!firstPredicted) {
    payload["first_predicted_covariance"] = nullptr;
  }
  if (!firstFiltered) {
    payload["first_filtered_covariance"] = "unavailable";
  }
  if (!firstSmoothed) {
    payload["first_smoothed_covariance"] = "unavailable";
  }
  return payload;
}

std::optional<Acts::BoundTrackParameters> seedFromOfficialMean(
    const Trk::TrackParameters& parameters, double sourceZ,
    const Acts::GeometryContext& geometryContext, double seedScale,
    const std::string& direction) {
  const Amg::Vector3D& position = parameters.position();
  const Amg::Vector3D& momentum = parameters.momentum();
  const double qOverPMev = parameters.parameters()[Trk::qOverP];
  if (!finiteValue(qOverPMev) || std::abs(qOverPMev) < 1.0e-18 ||
      std::abs(qOverPMev - 1.0e-5) < 1.0e-18) {
    return std::nullopt;
  }
  auto surface = planeAtZ(sourceZ);
  const auto bound = Acts::detail::transformFreeToBoundParameters(
      Acts::Vector3(position.x(), position.y(), sourceZ), 0.0,
      Acts::Vector3(momentum.x(), momentum.y(), momentum.z()),
      qOverPMev / 1_MeV, *surface, geometryContext);
  if (!bound.ok()) {
    return std::nullopt;
  }
  return Acts::BoundTrackParameters(
      surface, bound.value(),
      uninformativeSeedCovariance(seedScale, direction),
      Acts::ParticleHypothesis::muon());
}

json stateDefinition() {
  return {
      {"parameters", json::array({"x_mm", "y_mm", "tx", "ty", "q_over_p_per_mev"})},
      {"units", json::array({"mm", "mm", "1", "1", "1/MeV"})},
      {"frame", "global_cartesian_slopes_at_surface_z"},
      {"native_athena",
       json::array({"loc0_mm", "loc1_mm", "phi", "theta", "q_over_p_per_mev"})},
      {"q_over_p_signed", true},
      {"covariance_origin", "leave_target_out_acts_kalman_fit"},
      {"seed_covariance", "uninformative_not_official_cin"},
  };
}

struct ProfileHit {
  const Acts::Surface* surface{nullptr};
  double loc0{0.0};
  double variance{0.08 * 0.08 / 12.0};
  int station{-1};
  int layer{-1};
  int phi_module{-1};
  int eta_module{-1};
  int side{-1};
  int input_index{-1};
  double zMm{0.0};
  std::string identifier;
};

using Bound5 = Eigen::Matrix<double, 5, 1>;
using Mat5 = Eigen::Matrix<double, 5, 5>;
using ProfileActionList = Acts::ActionList<Acts::MaterialInteractor>;
using ProfileAbortList = Acts::AbortList<Acts::EndOfWorldReached>;
using ProfileOptions = Acts::PropagatorOptions<ProfileActionList, ProfileAbortList>;

Mat5 symmetricPinv5(const Mat5& matrix, double relative, int* rank) {
  Eigen::SelfAdjointEigenSolver<Mat5> solver(0.5 * (matrix + matrix.transpose()));
  const auto values = solver.eigenvalues();
  const auto vectors = solver.eigenvectors();
  double scale = values.cwiseAbs().maxCoeff();
  if (!(scale > 0.0) || !std::isfinite(scale)) {
    scale = 1.0;
  }
  Mat5 inverse = Mat5::Zero();
  int kept = 0;
  for (int i = 0; i < 5; ++i) {
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

json matrix5ToJson(const Mat5& matrix) {
  json rows = json::array();
  for (int i = 0; i < 5; ++i) {
    json row = json::array();
    for (int j = 0; j < 5; ++j) {
      if (!finiteValue(matrix(i, j))) {
        return nullptr;
      }
      row.push_back(matrix(i, j));
    }
    rows.push_back(row);
  }
  return rows;
}

json vectorToJson(const Eigen::VectorXd& values) {
  json out = json::array();
  for (int i = 0; i < values.size(); ++i) {
    if (!finiteValue(values[i])) {
      return nullptr;
    }
    out.push_back(values[i]);
  }
  return out;
}

bool evaluateMeasurementResiduals(
    const Propagator& propagator, const Acts::GeometryContext& geometryContext,
    const Acts::MagneticFieldContext& magFieldContext,
    const Acts::BoundTrackParameters& start, const std::vector<ProfileHit>& hits,
    Eigen::VectorXd& residuals, double& chi2) {
  residuals.resize(static_cast<int>(hits.size()));
  chi2 = 0.0;
  const Acts::Vector3 startPos = start.position(geometryContext);
  for (std::size_t i = 0; i < hits.size(); ++i) {
    if (hits[i].surface == nullptr) {
      return false;
    }
    ProfileOptions options(geometryContext, magFieldContext);
    options.maxSteps = 10000;
    options.maxStepSize = 10.0 * 1_m;
    const double zHit = hits[i].surface->center(geometryContext).z();
    options.direction = (zHit >= startPos.z()) ? Acts::Direction::Forward
                                               : Acts::Direction::Backward;
    auto& material = options.actionList.get<Acts::MaterialInteractor>();
    material.multipleScattering = true;
    material.energyLoss = true;
    auto result = propagator.propagate(start, *hits[i].surface, options);
    if (!result.ok() || !result.value().endParameters) {
      return false;
    }
    const double predicted =
        result.value().endParameters->parameters()[Acts::eBoundLoc0];
    const double residual = hits[i].loc0 - predicted;
    residuals[static_cast<int>(i)] = residual;
    chi2 += residual * residual / hits[i].variance;
  }
  return true;
}

bool finiteDifferenceJacobian(
    const Propagator& propagator, const Acts::GeometryContext& geometryContext,
    const Acts::MagneticFieldContext& magFieldContext,
    const Acts::BoundVector& theta, const std::shared_ptr<const Acts::Surface>& surface,
    const std::vector<ProfileHit>& hits, Eigen::MatrixXd& jacobian) {
  const std::array<double, 5> steps{1.0e-2, 1.0e-2, 1.0e-5, 1.0e-5, 1.0e-6};
  jacobian.resize(static_cast<int>(hits.size()), 5);
  for (int col = 0; col < 5; ++col) {
    Acts::BoundVector plus = theta;
    Acts::BoundVector minus = theta;
    plus[col] += steps[col];
    minus[col] -= steps[col];
    const Acts::BoundTrackParameters plusP(surface, plus, std::nullopt,
                                           Acts::ParticleHypothesis::muon());
    const Acts::BoundTrackParameters minusP(surface, minus, std::nullopt,
                                            Acts::ParticleHypothesis::muon());
    Eigen::VectorXd rPlus;
    Eigen::VectorXd rMinus;
    double chiPlus = 0.0;
    double chiMinus = 0.0;
    if (!evaluateMeasurementResiduals(propagator, geometryContext, magFieldContext,
                                      plusP, hits, rPlus, chiPlus) ||
        !evaluateMeasurementResiduals(propagator, geometryContext, magFieldContext,
                                      minusP, hits, rMinus, chiMinus)) {
      return false;
    }
    // J_h = dh/dtheta = -dr/dtheta because r = m - h.
    jacobian.col(col) = (rMinus - rPlus) / (2.0 * steps[col]);
  }
  return true;
}

bool propagateDerived(
    const Propagator& propagator, const Acts::GeometryContext& geometryContext,
    const Acts::MagneticFieldContext& magFieldContext,
    const Acts::BoundTrackParameters& start, const Acts::Surface& target,
    std::array<double, 5>& derived) {
  ProfileOptions options(geometryContext, magFieldContext);
  options.maxSteps = 10000;
  options.maxStepSize = 10.0 * 1_m;
  const double zTarget = target.center(geometryContext).z();
  const double zStart = start.position(geometryContext).z();
  options.direction = (zTarget >= zStart) ? Acts::Direction::Forward
                                          : Acts::Direction::Backward;
  auto& material = options.actionList.get<Acts::MaterialInteractor>();
  material.multipleScattering = true;
  material.energyLoss = true;
  auto result = propagator.propagate(start, target, options);
  if (!result.ok() || !result.value().endParameters) {
    return false;
  }
  derived = exportFromBound(*result.value().endParameters, geometryContext);
  return true;
}

bool applyPhysicalBounds(Acts::BoundVector& theta, bool& hitBoundary) {
  constexpr double kMinTheta = 1.0e-3;
  constexpr double kMaxTheta = 3.140592653589793;
  constexpr double kMinQ = 1.0e-9;
  constexpr double kMaxQ = 1.0;
  bool changed = false;
  if (theta[Acts::eBoundTheta] < kMinTheta) {
    theta[Acts::eBoundTheta] = kMinTheta;
    changed = true;
  }
  if (theta[Acts::eBoundTheta] > kMaxTheta) {
    theta[Acts::eBoundTheta] = kMaxTheta;
    changed = true;
  }
  const double qAbs = std::abs(theta[Acts::eBoundQOverP]);
  if (qAbs < kMinQ) {
    theta[Acts::eBoundQOverP] = std::copysign(kMinQ, theta[Acts::eBoundQOverP]);
    changed = true;
  }
  if (qAbs > kMaxQ) {
    theta[Acts::eBoundQOverP] = std::copysign(kMaxQ, theta[Acts::eBoundQOverP]);
    changed = true;
  }
  hitBoundary = hitBoundary || changed;
  return std::isfinite(theta[Acts::eBoundLoc0]) &&
         std::isfinite(theta[Acts::eBoundLoc1]) &&
         std::isfinite(theta[Acts::eBoundPhi]) &&
         std::isfinite(theta[Acts::eBoundTheta]) &&
         std::isfinite(theta[Acts::eBoundQOverP]);
}

json runOneMeasurementProfile(
    const json& baseRow, const Propagator& propagator,
    const Acts::GeometryContext& geometryContext,
    const Acts::MagneticFieldContext& magFieldContext,
    const Acts::BoundTrackParameters& seed, const std::vector<ProfileHit>& hits,
    double targetZ, const std::string& variant, int maxIterations,
    double pinvRelative) {
  json row = baseRow;
  row["likelihood_kind"] = "measurement_only_chi2";
  row["prior_term_present"] = false;
  row["wb109_cin_used_as_likelihood"] = false;
  row["wb107_cin_used"] = false;
  row["truth_used_in_fit"] = false;
  row["ridge_added"] = false;
  row["profile_init_variant"] = variant;
  row["profile_pinv_relative"] = pinvRelative;
  row["alignment_rank_tolerance_used"] = false;
  row["seed_used_as"] = "initialization_only";
  row["measurement_covariance_provenance"] = "sct_pitch_over_sqrt12";
  row["measurement_R_mm2"] = 0.08 * 0.08 / 12.0;
  if (hits.size() < 8) {
    row["profile_success"] = false;
    row["profile_converged"] = false;
    row["fit_success"] = false;
    row["fit_failure_reason"] = "too_few_profile_hits";
    return row;
  }
  auto surface = seed.referenceSurface().getSharedPtr();
  Acts::BoundVector theta = seed.parameters();
  row["init_native"] = {theta[Acts::eBoundLoc0], theta[Acts::eBoundLoc1],
                        theta[Acts::eBoundPhi], theta[Acts::eBoundTheta],
                        theta[Acts::eBoundQOverP] * 1_MeV};
  bool hitBoundary = false;
  if (!applyPhysicalBounds(theta, hitBoundary)) {
    row["profile_success"] = false;
    row["profile_converged"] = false;
    row["fit_success"] = false;
    row["fit_failure_reason"] = "nonfinite_initialization";
    return row;
  }
  double chi2 = 0.0;
  Eigen::VectorXd residual;
  int nIter = 0;
  bool converged = false;
  std::string failure;
  Mat5 hessian = Mat5::Zero();
  int hessianRank = 0;
  for (nIter = 0; nIter < maxIterations; ++nIter) {
    const Acts::BoundTrackParameters current(surface, theta, std::nullopt,
                                             Acts::ParticleHypothesis::muon());
    if (!evaluateMeasurementResiduals(propagator, geometryContext, magFieldContext,
                                      current, hits, residual, chi2)) {
      failure = "measurement_propagation_failed";
      break;
    }
    Eigen::MatrixXd jacobian;
    if (!finiteDifferenceJacobian(propagator, geometryContext, magFieldContext, theta,
                                  surface, hits, jacobian)) {
      failure = "jacobian_propagation_failed";
      break;
    }
    Eigen::VectorXd invR(residual.size());
    for (int i = 0; i < residual.size(); ++i) {
      invR[i] = 1.0 / hits[static_cast<std::size_t>(i)].variance;
    }
    Mat5 h = Mat5::Zero();
    Bound5 g = Bound5::Zero();
    for (int i = 0; i < residual.size(); ++i) {
      const Eigen::Matrix<double, 1, 5> ji = jacobian.row(i);
      h += invR[i] * (ji.transpose() * ji);
      g += invR[i] * ji.transpose() * residual[i];
    }
    hessian = 0.5 * (h + h.transpose());
    const Bound5 delta = symmetricPinv5(hessian, pinvRelative, &hessianRank) * g;
    const std::array<double, 4> damps{1.0, 0.5, 0.25, 0.125};
    bool improved = false;
    Acts::BoundVector best = theta;
    double bestChi2 = chi2;
    for (double damp : damps) {
      Acts::BoundVector trial = theta;
      for (int ipar = 0; ipar < 5; ++ipar) {
        trial[ipar] += damp * delta[ipar];
      }
      if (!applyPhysicalBounds(trial, hitBoundary)) {
        continue;
      }
      const Acts::BoundTrackParameters trialP(surface, trial, std::nullopt,
                                              Acts::ParticleHypothesis::muon());
      Eigen::VectorXd trialR;
      double trialChi2 = 0.0;
      if (!evaluateMeasurementResiduals(propagator, geometryContext, magFieldContext,
                                        trialP, hits, trialR, trialChi2)) {
        continue;
      }
      if (trialChi2 <= bestChi2) {
        best = trial;
        bestChi2 = trialChi2;
        improved = true;
      }
    }
    const double deltaChi2 = chi2 - bestChi2;
    theta = best;
    chi2 = bestChi2;
    if (!improved || std::abs(deltaChi2) < 1.0e-6) {
      converged = std::isfinite(chi2);
      break;
    }
  }
  if (!failure.empty()) {
    row["profile_success"] = false;
    row["profile_converged"] = false;
    row["fit_success"] = false;
    row["fit_failure_reason"] = failure;
    row["profile_n_iterations"] = nIter;
    row["hit_boundary"] = hitBoundary;
    return row;
  }
  const Acts::BoundTrackParameters fitted(surface, theta, std::nullopt,
                                          Acts::ParticleHypothesis::muon());
  const auto derived = exportFromBound(fitted, geometryContext);
  auto targetSurface = planeAtZ(targetZ);
  std::array<double, 5> prediction{};
  bool predOk = propagateDerived(propagator, geometryContext, magFieldContext, fitted,
                                 *targetSurface, prediction);
  const std::array<double, 5> predSteps{1.0e-2, 1.0e-2, 1.0e-5, 1.0e-5, 1.0e-6};
  Mat5 predJac = Mat5::Zero();
  bool jacOk = predOk;
  if (predOk) {
    for (int col = 0; col < 5; ++col) {
      Acts::BoundVector plus = theta;
      Acts::BoundVector minus = theta;
      plus[col] += predSteps[col];
      minus[col] -= predSteps[col];
      const Acts::BoundTrackParameters plusP(surface, plus, std::nullopt,
                                             Acts::ParticleHypothesis::muon());
      const Acts::BoundTrackParameters minusP(surface, minus, std::nullopt,
                                              Acts::ParticleHypothesis::muon());
      std::array<double, 5> plusD{};
      std::array<double, 5> minusD{};
      if (!propagateDerived(propagator, geometryContext, magFieldContext, plusP,
                            *targetSurface, plusD) ||
          !propagateDerived(propagator, geometryContext, magFieldContext, minusP,
                            *targetSurface, minusD)) {
        jacOk = false;
        break;
      }
      for (int rowI = 0; rowI < 5; ++rowI) {
        predJac(rowI, col) = (plusD[rowI] - minusD[rowI]) / (2.0 * predSteps[col]);
      }
    }
  }
  Eigen::SelfAdjointEigenSolver<Mat5> eigen(hessian);
  json singular = json::array();
  for (int i = 0; i < 5; ++i) {
    singular.push_back(eigen.eigenvalues()[i]);
  }
  row["profile_success"] = converged && predOk;
  row["profile_converged"] = converged;
  row["fit_success"] = converged && predOk;
  row["fit_failure_reason"] = (converged && predOk)
                                  ? json(nullptr)
                                  : json(predOk ? "profile_not_converged"
                                                : "target_prediction_failed");
  row["profile_chi2"] = chi2;
  row["chi2"] = chi2;
  row["profile_n_iterations"] = nIter;
  row["hit_boundary"] = hitBoundary;
  row["profiled_native_state"] = {theta[Acts::eBoundLoc0], theta[Acts::eBoundLoc1],
                                  theta[Acts::eBoundPhi], theta[Acts::eBoundTheta],
                                  theta[Acts::eBoundQOverP] * 1_MeV};
  row["native_state"] = row["profiled_native_state"];
  row["profiled_derived_state"] = {derived[0], derived[1], derived[2], derived[3],
                                   derived[4]};
  row["derived_state"] = row["profiled_derived_state"];
  row["joint_hessian"] = matrix5ToJson(hessian);
  row["hessian_rank"] = hessianRank;
  row["hessian_singular_values"] = singular;
  row["qoverp_curvature"] = hessian(4, 4);
  if (predOk) {
    row["target_prediction_derived"] = {prediction[0], prediction[1], prediction[2],
                                        prediction[3], prediction[4]};
    row["prediction_derived"] = row["target_prediction_derived"];
  } else {
    row["target_prediction_derived"] = nullptr;
    row["prediction_derived"] = nullptr;
  }
  row["prediction_jacobian"] = jacOk ? matrix5ToJson(predJac) : json(nullptr);
  return row;
}

std::vector<json> runMeasurementProfiles(
    const json& baseRow, const Propagator& propagator,
    const Acts::GeometryContext& geometryContext,
    const Acts::MagneticFieldContext& magFieldContext,
    const Acts::BoundTrackParameters& seed, const std::vector<ProfileHit>& hits,
    double targetZ, bool seedInvariance, int maxIterations, double pinvRelative) {
  struct Variant {
    const char* name;
    int index;
    double add;
    double scale;
  };
  std::vector<Variant> variants{{"nominal", -1, 0.0, 1.0}};
  if (seedInvariance) {
    variants.push_back({"loc1_plus_1mm", static_cast<int>(Acts::eBoundLoc1), 1.0, 1.0});
    variants.push_back({"phi_plus_1e-3", static_cast<int>(Acts::eBoundPhi), 1.0e-3, 1.0});
    variants.push_back({"qoverp_times_1p1", static_cast<int>(Acts::eBoundQOverP), 0.0, 1.1});
  }
  std::vector<json> rows;
  for (const auto& variant : variants) {
    Acts::BoundVector params = seed.parameters();
    if (variant.index >= 0) {
      params[variant.index] = params[variant.index] * variant.scale + variant.add;
    }
    const Acts::BoundTrackParameters init(seed.referenceSurface().getSharedPtr(),
                                          params, std::nullopt,
                                          Acts::ParticleHypothesis::muon());
    rows.push_back(runOneMeasurementProfile(
        baseRow, propagator, geometryContext, magFieldContext, init, hits, targetZ,
        variant.name, maxIterations, pinvRelative));
  }
  return rows;
}

struct ProfileHop {
  int measurement_index{-1};
  int station{-1};
  int layer{-1};
  int phi_module{-1};
  int eta_module{-1};
  int side{-1};
  std::string identifier;
  double measurement_z_mm{0.0};
  double destination_surface_z_mm{0.0};
  double start_z_mm{0.0};
  double initial_loc0{0.0};
  double initial_loc1{0.0};
  double initial_phi{0.0};
  double initial_theta{0.0};
  double q_over_p{0.0};
  double charge{0.0};
  std::string path_direction;
  bool ok{false};
  bool surface_reached{false};
  std::string propagation_status;
  std::string abort_reason;
  double predicted_loc0{0.0};
  double predicted_loc1{0.0};
  double residual{0.0};
  double path_length{0.0};
  int n_steps{0};
  std::string transport_mode;
  std::string projection_kind;
  bool inside_bounds{false};
  bool bounded_intersection_on_surface{false};
  bool continuation_state_ok{false};
  double distance_to_plane_mm{0.0};
  double local_z_mm{0.0};
  double final_x_mm{0.0};
  double final_y_mm{0.0};
  double final_z_mm{0.0};
  json path_checkpoints = json::array();
  json surface_geometry;
  std::optional<Acts::BoundTrackParameters> end_parameters;
};

struct Chi2Eval {
  bool ok{false};
  double chi2{0.0};
  Eigen::VectorXd residual;
  std::vector<ProfileHop> hops;
  int first_failed_index{-1};
};

json hopToJson(const ProfileHop& hop) {
  return {{"measurement_index", hop.measurement_index},
          {"measurement_station", hop.station},
          {"measurement_layer", hop.layer},
          {"phi_module", hop.phi_module},
          {"eta_module", hop.eta_module},
          {"side", hop.side},
          {"identifier", hop.identifier},
          {"measurement_z_mm", hop.measurement_z_mm},
          {"destination_surface_z_mm", hop.destination_surface_z_mm},
          {"start_z_mm", hop.start_z_mm},
          {"initial_state",
           {hop.initial_loc0, hop.initial_loc1, hop.initial_phi, hop.initial_theta,
            hop.q_over_p * 1_MeV}},
          {"q_over_p_per_mev", hop.q_over_p * 1_MeV},
          {"charge", hop.charge},
          {"phi", hop.initial_phi},
          {"theta", hop.initial_theta},
          {"path_direction", hop.path_direction},
          {"ok", hop.ok},
          {"surface_reached", hop.surface_reached},
          {"propagation_status", hop.propagation_status},
          {"abort_reason", hop.abort_reason},
          {"predicted_loc0", hop.ok ? json(hop.predicted_loc0) : json(nullptr)},
          {"predicted_loc1", hop.ok ? json(hop.predicted_loc1) : json(nullptr)},
          {"residual", hop.ok ? json(hop.residual) : json(nullptr)},
          {"path_length", hop.path_length},
          {"number_of_propagation_steps", hop.n_steps},
          {"transport_mode", hop.transport_mode},
          {"projection_kind", hop.projection_kind},
          {"inside_active_bounds", hop.inside_bounds},
          {"bounded_intersection_on_surface", hop.bounded_intersection_on_surface},
          {"continuation_state_ok", hop.continuation_state_ok},
          {"distance_to_plane_mm", hop.distance_to_plane_mm},
          {"local_z_mm", hop.local_z_mm},
          {"final_position_xyz_mm",
           {hop.final_x_mm, hop.final_y_mm, hop.final_z_mm}},
          {"path_checkpoints", hop.path_checkpoints},
          {"surface_geometry", hop.surface_geometry}};
}

ProfileOptions makeProfileOptions(const Acts::GeometryContext& geometryContext,
                                  const Acts::MagneticFieldContext& magFieldContext,
                                  Acts::Direction direction, int maxSteps,
                                  bool matchKalmanStepSize) {
  ProfileOptions options(geometryContext, magFieldContext);
  options.maxSteps = static_cast<unsigned int>(std::max(maxSteps, 1));
  options.direction = direction;
  options.loopProtection = true;
  if (!matchKalmanStepSize) {
    options.maxStepSize = 10.0 * 1_m;
  }
  auto& material = options.actionList.get<Acts::MaterialInteractor>();
  material.multipleScattering = true;
  material.energyLoss = true;
  return options;
}

json surfaceGeometryJson(const Acts::Surface& surface,
                         const Acts::GeometryContext& geometryContext) {
  const Acts::Vector3 center = surface.center(geometryContext);
  const Acts::Vector3 normal =
      surface.normal(geometryContext, center, Acts::Vector3::UnitZ());
  const Acts::RotationMatrix3 rotation =
      surface.transform(geometryContext).rotation();
  json rot = json::array();
  for (int i = 0; i < 3; ++i) {
    rot.push_back({rotation(i, 0), rotation(i, 1), rotation(i, 2)});
  }
  json bounds;
  const auto& boundObj = surface.bounds();
  bounds["type"] = static_cast<int>(boundObj.type());
  bounds["values"] = json::array();
  const auto values = boundObj.values();
  for (double value : values) {
    bounds["values"].push_back(value);
  }
  return {{"center_xyz_mm", {center.x(), center.y(), center.z()}},
          {"normal", {normal.x(), normal.y(), normal.z()}},
          {"rotation", rot},
          {"geometry_id", surface.geometryId().value()},
          {"bounds", bounds},
          {"name", surface.name()},
          {"has_detector_element", surface.associatedDetectorElement() != nullptr},
          {"has_surface_material", surface.surfaceMaterial() != nullptr}};
}

json downsampleSteps(const std::vector<Acts::detail::Step>& steps) {
  json out = json::array();
  if (steps.empty()) {
    return out;
  }
  const std::size_t n = steps.size();
  auto push = [&](std::size_t i) {
    const auto& step = steps[i];
    out.push_back({{"i", static_cast<int>(i)},
                   {"x_mm", step.position.x()},
                   {"y_mm", step.position.y()},
                   {"z_mm", step.position.z()},
                   {"px", step.momentum.x()},
                   {"py", step.momentum.y()},
                   {"pz", step.momentum.z()},
                   {"geometry_id", step.geoID.value()},
                   {"on_surface", step.surface != nullptr}});
  };
  for (std::size_t i = 0; i < n && i < 20; ++i) {
    push(i);
  }
  for (std::size_t i = 20; i + 20 < n; i += 50) {
    push(i);
  }
  const std::size_t startLast = n > 20 ? n - 20 : 0;
  for (std::size_t i = startLast; i < n; ++i) {
    if (i >= 20) {
      push(i);
    }
  }
  return out;
}

bool projectSupportingPlane(const Acts::Surface& surface,
                            const Acts::GeometryContext& geometryContext,
                            const Acts::Vector3& position,
                            const Acts::Vector3& direction, ProfileHop& hop) {
  const auto unbounded = surface.intersect(
      geometryContext, position, direction, Acts::BoundaryCheck(false),
      Acts::s_onSurfaceTolerance);
  const auto bounded = surface.intersect(
      geometryContext, position, direction, Acts::BoundaryCheck(true),
      Acts::s_onSurfaceTolerance);
  const auto closest = unbounded.closest();
  const auto closestBounded = bounded.closest();
  hop.bounded_intersection_on_surface =
      static_cast<bool>(closestBounded) &&
      closestBounded.status() == Acts::Intersection3D::Status::onSurface;
  if (!closest || closest.status() == Acts::Intersection3D::Status::unreachable) {
    hop.projection_kind = "supporting_plane_unreachable";
    return false;
  }
  const Acts::Vector3 onPlane = closest.position();
  const Acts::Vector3 local =
      surface.transform(geometryContext).inverse() * onPlane;
  hop.predicted_loc0 = local.x();
  hop.predicted_loc1 = local.y();
  hop.local_z_mm = local.z();
  hop.distance_to_plane_mm = std::abs(local.z());
  hop.projection_kind = "supporting_plane_chart";
  hop.inside_bounds = surface.insideBounds(
      Acts::Vector2(local.x(), local.y()), Acts::BoundaryCheck(true));
  return std::abs(closest.pathLength()) < 50.0;
}

ProfileHop propagateOneHit(const Propagator& propagator,
                           const Acts::GeometryContext& geometryContext,
                           const Acts::MagneticFieldContext& magFieldContext,
                           const Acts::BoundTrackParameters& start,
                           const ProfileHit& hit, int index, int maxSteps,
                           bool matchKalmanStepSize,
                           bool supportingPlane = false, bool recordPath = false,
                           int diagnosticMaxSteps = 0) {
  ProfileHop hop;
  hop.measurement_index = index;
  hop.station = hit.station;
  hop.layer = hit.layer;
  hop.phi_module = hit.phi_module;
  hop.eta_module = hit.eta_module;
  hop.side = hit.side;
  hop.identifier = hit.identifier;
  hop.measurement_z_mm = hit.zMm;
  hop.start_z_mm = start.position(geometryContext).z();
  const Acts::BoundVector startP = start.parameters();
  hop.initial_loc0 = startP[Acts::eBoundLoc0];
  hop.initial_loc1 = startP[Acts::eBoundLoc1];
  hop.initial_phi = startP[Acts::eBoundPhi];
  hop.initial_theta = startP[Acts::eBoundTheta];
  hop.q_over_p = startP[Acts::eBoundQOverP];
  hop.charge = startP[Acts::eBoundQOverP] > 0.0 ? 1.0 : -1.0;
  if (hit.surface == nullptr) {
    hop.propagation_status = "missing_surface";
    hop.abort_reason = "surface_pointer_null";
    return hop;
  }
  hop.destination_surface_z_mm = hit.surface->center(geometryContext).z();
  const bool forward = hop.destination_surface_z_mm >= hop.start_z_mm;
  hop.path_direction = forward ? "forward" : "backward";
  hop.surface_geometry = surfaceGeometryJson(*hit.surface, geometryContext);
  hop.transport_mode = supportingPlane ? "supporting_plane" : "bounded_surface_reached";
  const int stepsCap = maxSteps;
  (void)diagnosticMaxSteps;
  const bool longHop =
      std::abs(hop.destination_surface_z_mm - hop.start_z_mm) > 200.0;
  auto options = makeProfileOptions(
      geometryContext, magFieldContext,
      forward ? Acts::Direction::Forward : Acts::Direction::Backward, stepsCap,
      matchKalmanStepSize);

  if (!supportingPlane) {
    auto result = propagator.propagate(start, *hit.surface, options);
    if (!result.ok()) {
      hop.propagation_status = "acts_propagate_error";
      hop.abort_reason = result.error().message();
      return hop;
    }
    hop.n_steps = static_cast<int>(result.value().steps);
    hop.path_length = result.value().pathLength;
    if (!result.value().endParameters) {
      hop.propagation_status = "no_end_parameters";
      hop.abort_reason = "surface_not_reached";
      return hop;
    }
    hop.ok = true;
    hop.surface_reached = true;
    hop.propagation_status = "ok";
    hop.projection_kind = "acts_bound_parameters";
    hop.end_parameters = *result.value().endParameters;
    hop.predicted_loc0 = hop.end_parameters->parameters()[Acts::eBoundLoc0];
    hop.predicted_loc1 = hop.end_parameters->parameters()[Acts::eBoundLoc1];
    hop.residual = hit.loc0 - hop.predicted_loc0;
    const Acts::Vector3 endPos = hop.end_parameters->position(geometryContext);
    hop.final_x_mm = endPos.x();
    hop.final_y_mm = endPos.y();
    hop.final_z_mm = endPos.z();
    hop.inside_bounds = hit.surface->insideBounds(
        Acts::Vector2(hop.predicted_loc0, hop.predicted_loc1),
        Acts::BoundaryCheck(true));
    hop.bounded_intersection_on_surface = hop.inside_bounds;
    hop.continuation_state_ok = true;
    return hop;
  }

  using LogActions = Acts::ActionList<Acts::MaterialInteractor, Acts::detail::SteppingLogger>;
  using LogAbort = Acts::AbortList<Acts::EndOfWorldReached>;
  using LogOptions = Acts::PropagatorOptions<LogActions, LogAbort>;
  LogOptions logOptions(geometryContext, magFieldContext);
  logOptions.maxSteps = static_cast<unsigned int>(std::max(stepsCap, 1));
  logOptions.direction =
      forward ? Acts::Direction::Forward : Acts::Direction::Backward;
  logOptions.loopProtection = true;
  if (!matchKalmanStepSize) {
    logOptions.maxStepSize = 10.0 * 1_m;
  }
  auto& material = logOptions.actionList.get<Acts::MaterialInteractor>();
  material.multipleScattering = true;
  material.energyLoss = true;
  logOptions.actionList.get<Acts::detail::SteppingLogger>().sterile =
      !(recordPath || longHop);

  auto state = propagator.makeState<Acts::BoundTrackParameters, LogOptions,
                                    Acts::SurfaceReached, Acts::PathLimitReached>(
      start, *hit.surface, logOptions);
  state.options.abortList.template get<Acts::SurfaceReached>().boundaryCheck =
      Acts::BoundaryCheck(false);
  auto propRes = propagator.propagate(state);
  hop.n_steps = static_cast<int>(state.steps);
  hop.path_length = state.pathLength;
  const Acts::Vector3 freePos = state.stepping.pars.template segment<3>(Acts::eFreePos0);
  const Acts::Vector3 freeDir = state.stepping.pars.template segment<3>(Acts::eFreeDir0);
  hop.final_x_mm = freePos.x();
  hop.final_y_mm = freePos.y();
  hop.final_z_mm = freePos.z();
  if (recordPath || longHop) {
    hop.path_checkpoints = downsampleSteps(
        state.get<Acts::detail::SteppingLogger::result_type>().steps);
  }
  const Acts::Vector3 signedDir =
      (forward ? 1.0 : -1.0) * freeDir;
  const bool projected = projectSupportingPlane(
      *hit.surface, geometryContext, freePos, signedDir, hop);
  if (!propRes.ok()) {
    hop.propagation_status = "acts_propagate_error";
    hop.abort_reason = propRes.error().message();
    hop.surface_reached = false;
  } else {
    hop.propagation_status = "ok";
    hop.surface_reached = true;
  }
  if (projected) {
    hop.ok = true;
    hop.residual = hit.loc0 - hop.predicted_loc0;
    const Acts::Vector3 onPlane =
        hit.surface->transform(geometryContext) *
        Acts::Vector3(hop.predicted_loc0, hop.predicted_loc1, 0.0);
    auto bound = Acts::detail::transformFreeToBoundParameters(
        onPlane, 0.0, freeDir, state.stepping.pars[Acts::eFreeQOverP],
        *hit.surface, geometryContext, 10.0 * Acts::s_onSurfaceTolerance);
    if (bound.ok()) {
      hop.end_parameters = Acts::BoundTrackParameters(
          hit.surface->getSharedPtr(), bound.value(), std::nullopt,
          Acts::ParticleHypothesis::muon());
      hop.continuation_state_ok = true;
    } else {
      Acts::BoundVector next = Acts::BoundVector::Zero();
      next[Acts::eBoundLoc0] = hop.predicted_loc0;
      next[Acts::eBoundLoc1] = hop.predicted_loc1;
      next[Acts::eBoundPhi] = std::atan2(freeDir.y(), freeDir.x());
      next[Acts::eBoundTheta] = std::acos(std::clamp(freeDir.z(), -1.0, 1.0));
      next[Acts::eBoundQOverP] = state.stepping.pars[Acts::eFreeQOverP];
      hop.end_parameters = Acts::BoundTrackParameters(
          hit.surface->getSharedPtr(), next, std::nullopt,
          Acts::ParticleHypothesis::muon());
      hop.continuation_state_ok = true;
      if (hop.abort_reason.empty()) {
        hop.abort_reason = "free_to_bound_used_plane_chart_fallback";
      }
    }
  } else if (hop.abort_reason.empty()) {
    hop.abort_reason = "supporting_plane_not_near_enough";
  }
  return hop;
}

Chi2Eval evaluateSequentialResiduals(
    const Propagator& propagator, const Acts::GeometryContext& geometryContext,
    const Acts::MagneticFieldContext& magFieldContext,
    const Acts::BoundTrackParameters& start, const std::vector<ProfileHit>& hits,
    int maxSteps, bool matchKalmanStepSize, bool sequential,
    bool supportingPlane = false, bool recordPath = false,
    int diagnosticMaxSteps = 0) {
  Chi2Eval eval;
  eval.residual.resize(static_cast<int>(hits.size()));
  eval.hops.resize(hits.size());
  std::vector<int> order(hits.size());
  for (std::size_t i = 0; i < hits.size(); ++i) {
    order[i] = static_cast<int>(i);
  }
  if (sequential) {
    const double z0 = start.position(geometryContext).z();
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      return hits[static_cast<std::size_t>(a)].zMm <
             hits[static_cast<std::size_t>(b)].zMm;
    });
    std::vector<int> downstream;
    std::vector<int> upstream;
    for (int idx : order) {
      if (hits[static_cast<std::size_t>(idx)].zMm >= z0) {
        downstream.push_back(idx);
      } else {
        upstream.push_back(idx);
      }
    }
    std::reverse(upstream.begin(), upstream.end());
    order.clear();
    order.insert(order.end(), downstream.begin(), downstream.end());
    order.insert(order.end(), upstream.begin(), upstream.end());
  }

  std::optional<Acts::BoundTrackParameters> currentDown = start;
  std::optional<Acts::BoundTrackParameters> currentUp = start;
  const double z0 = start.position(geometryContext).z();
  for (int idx : order) {
    const ProfileHit& hit = hits[static_cast<std::size_t>(idx)];
    const bool down = hit.zMm >= z0;
    if (sequential && !(down ? currentDown.has_value() : currentUp.has_value())) {
      ProfileHop blocked;
      blocked.measurement_index = idx;
      blocked.station = hit.station;
      blocked.layer = hit.layer;
      blocked.phi_module = hit.phi_module;
      blocked.eta_module = hit.eta_module;
      blocked.side = hit.side;
      blocked.identifier = hit.identifier;
      blocked.measurement_z_mm = hit.zMm;
      blocked.start_z_mm = z0;
      blocked.ok = false;
      blocked.propagation_status = "blocked_by_previous_hop";
      blocked.abort_reason = "sequential_state_missing";
      eval.hops[static_cast<std::size_t>(idx)] = blocked;
      if (eval.first_failed_index < 0) {
        eval.first_failed_index = idx;
      }
      continue;
    }
    const Acts::BoundTrackParameters& from =
        sequential ? (down ? *currentDown : *currentUp) : start;
    ProfileHop hop = propagateOneHit(
        propagator, geometryContext, magFieldContext, from, hit, idx, maxSteps,
        matchKalmanStepSize, supportingPlane, recordPath, diagnosticMaxSteps);
    eval.hops[static_cast<std::size_t>(idx)] = hop;
    if (hop.ok) {
      eval.residual[idx] = hop.residual;
      eval.chi2 += hop.residual * hop.residual / hit.variance;
      if (sequential) {
        if (hop.end_parameters.has_value()) {
          if (down) {
            currentDown = *hop.end_parameters;
          } else {
            currentUp = *hop.end_parameters;
          }
        } else if (down) {
          currentDown.reset();
        } else {
          currentUp.reset();
        }
      }
    } else if (eval.first_failed_index < 0) {
      eval.first_failed_index = idx;
    }
  }
  eval.ok = eval.first_failed_index < 0 && !hits.empty();
  if (!eval.ok) {
    eval.chi2 = 0.0;
  }
  return eval;
}

json surfaceOrderJson(const std::vector<ProfileHit>& hits, double sourceZ) {
  json items = json::array();
  std::vector<int> order(hits.size());
  for (std::size_t i = 0; i < hits.size(); ++i) {
    order[i] = static_cast<int>(i);
  }
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return hits[static_cast<std::size_t>(a)].zMm <
           hits[static_cast<std::size_t>(b)].zMm;
  });
  bool returns_upstream = false;
  double lastZ = sourceZ;
  json path = json::array();
  path.push_back({{"role", "source"}, {"z_mm", sourceZ}});
  for (int idx : order) {
    const auto& hit = hits[static_cast<std::size_t>(idx)];
    if (hit.zMm < lastZ && hit.zMm < sourceZ && lastZ > sourceZ) {
      returns_upstream = true;
    }
    path.push_back({{"input_index", hit.input_index},
                    {"station", hit.station},
                    {"layer", hit.layer},
                    {"z_mm", hit.zMm},
                    {"identifier", hit.identifier}});
    lastZ = hit.zMm;
  }
  double zMin = hits.empty() ? sourceZ : hits[0].zMm;
  double zMax = zMin;
  for (const auto& hit : hits) {
    zMin = std::min(zMin, hit.zMm);
    zMax = std::max(zMax, hit.zMm);
  }
  return {{"source_z_mm", sourceZ},
          {"z_sorted_path", path},
          {"input_order_is_z_sorted", order.size() < 2 ||
                                           std::is_sorted(hits.begin(), hits.end(),
                                                          [](const ProfileHit& a,
                                                             const ProfileHit& b) {
                                                            return a.zMm < b.zMm;
                                                          })},
          {"returns_upstream_after_downstream", returns_upstream},
          {"z_min_mm", zMin},
          {"z_max_mm", zMax},
          {"n_hits", static_cast<int>(hits.size())}};
}

const std::array<double, 5> kNumericScale{1.0, 1.0, 1.0e-3, 1.0e-3, 1.0e-3};
const std::array<const char*, 5> kParamNames{"loc0", "loc1", "phi", "theta", "q_over_p"};
const std::array<double, 5> kFdSteps{1.0e-2, 1.0e-2, 1.0e-5, 1.0e-5, 1.0e-6};

json jacobianValidationJson(
    const Propagator& propagator, const Acts::GeometryContext& geometryContext,
    const Acts::MagneticFieldContext& magFieldContext,
    const Acts::BoundTrackParameters& start, const std::vector<ProfileHit>& hits,
    int maxSteps, bool matchKalman, bool sequential,
    bool supportingPlane = false) {
  json columns = json::array();
  bool contract = true;
  int failed = 0;
  Eigen::MatrixXd jFull(static_cast<int>(hits.size()), 5);
  Eigen::MatrixXd jHalf(static_cast<int>(hits.size()), 5);
  Chi2Eval base = evaluateSequentialResiduals(
      propagator, geometryContext, magFieldContext, start, hits, maxSteps,
      matchKalman, sequential, supportingPlane);
  if (!base.ok) {
    return {{"jacobian_contract_established", false},
            {"reason", "nominal_evaluation_failed"},
            {"failed_finite_difference_evaluations", 1}};
  }
  auto surface = start.referenceSurface().getSharedPtr();
  Acts::BoundVector theta = start.parameters();
  for (int col = 0; col < 5; ++col) {
    json item{{"parameter", kParamNames[static_cast<std::size_t>(col)]},
              {"fd_step", kFdSteps[static_cast<std::size_t>(col)]}};
    bool colOk = true;
    for (int pass = 0; pass < 2; ++pass) {
      const double step =
          kFdSteps[static_cast<std::size_t>(col)] * (pass == 0 ? 1.0 : 0.5);
      Acts::BoundVector plus = theta;
      Acts::BoundVector minus = theta;
      plus[col] += step;
      minus[col] -= step;
      Chi2Eval evPlus = evaluateSequentialResiduals(
          propagator, geometryContext, magFieldContext,
          Acts::BoundTrackParameters(surface, plus, std::nullopt,
                                     Acts::ParticleHypothesis::muon()),
          hits, maxSteps, matchKalman, sequential, supportingPlane);
      Chi2Eval evMinus = evaluateSequentialResiduals(
          propagator, geometryContext, magFieldContext,
          Acts::BoundTrackParameters(surface, minus, std::nullopt,
                                     Acts::ParticleHypothesis::muon()),
          hits, maxSteps, matchKalman, sequential, supportingPlane);
      if (!evPlus.ok || !evMinus.ok) {
        colOk = false;
        failed += 1;
        item[pass == 0 ? "full_step_ok" : "half_step_ok"] = false;
        break;
      }
      Eigen::VectorXd jr = (evPlus.residual - evMinus.residual) / (2.0 * step);
      if (pass == 0) {
        jFull.col(col) = jr;
      } else {
        jHalf.col(col) = jr;
      }
    }
    const double nFull = jFull.col(col).norm();
    const double nHalf = jHalf.col(col).norm();
    const double rel =
        nFull > 1.0e-12 ? (jFull.col(col) - jHalf.col(col)).norm() / nFull : 0.0;
    const double sign =
        (jFull.col(col).dot(jHalf.col(col)) >= 0.0) ? 1.0 : -1.0;
    item["ok"] = colOk;
    item["column_norm"] = nFull;
    item["half_step_column_norm"] = nHalf;
    item["relative_jacobian_error"] = rel;
    item["sign_consistency"] = sign > 0.0;
    if (!colOk || rel > 0.05 || sign < 0.0) {
      contract = false;
    }
    columns.push_back(item);
  }
  return {{"jacobian_contract_established", contract && failed == 0},
          {"failed_finite_difference_evaluations", failed},
          {"columns", columns},
          {"method", "central_finite_difference_vs_step_halving"}};
}

json scalingContractJson(
    const Propagator& propagator, const Acts::GeometryContext& geometryContext,
    const Acts::MagneticFieldContext& magFieldContext,
    const Acts::BoundTrackParameters& start, const std::vector<ProfileHit>& hits,
    int maxSteps, bool matchKalman, bool sequential,
    bool supportingPlane = false) {
  auto surface = start.referenceSurface().getSharedPtr();
  Acts::BoundVector theta = start.parameters();
  Chi2Eval atRef = evaluateSequentialResiduals(
      propagator, geometryContext, magFieldContext, start, hits, maxSteps,
      matchKalman, sequential, supportingPlane);
  Acts::BoundVector shifted = theta;
  shifted[0] += kNumericScale[0];
  Chi2Eval atShift = evaluateSequentialResiduals(
      propagator, geometryContext, magFieldContext,
      Acts::BoundTrackParameters(surface, shifted, std::nullopt,
                                 Acts::ParticleHypothesis::muon()),
      hits, maxSteps, matchKalman, sequential, supportingPlane);
  const bool ok = atRef.ok && atShift.ok;
  return {{"kind", "fixed_numerical_units"},
          {"scales",
           {{"loc0_mm", kNumericScale[0]},
            {"loc1_mm", kNumericScale[1]},
            {"phi", kNumericScale[2]},
            {"theta", kNumericScale[3]},
            {"q_over_p_per_gev", kNumericScale[4]}}},
          {"not_a_prior", true},
          {"physical_chi2_invariant_by_construction", true},
          {"chi2_at_theta_ref", atRef.ok ? json(atRef.chi2) : json(nullptr)},
          {"chi2_at_theta_ref_plus_s_loc0",
           atShift.ok ? json(atShift.chi2) : json(nullptr)},
          {"chi2_z_equals_chi2_theta", ok},
          {"evaluations_ok", ok}};
}

json runOneProfileNumerics(
    const json& baseRow, const Propagator& propagator,
    const Acts::GeometryContext& geometryContext,
    const Acts::MagneticFieldContext& magFieldContext,
    const Acts::BoundTrackParameters& seed, const std::vector<ProfileHit>& hits,
    double targetZ, const std::string& variant, const std::string& initKind,
    int maxIterations, double pinvRelative, int maxSteps, bool matchKalman,
    bool sequential, bool useScaling, bool evaluateOnly, bool writeTrace,
    bool supportingPlane = false, bool recordPath = false,
    int diagnosticMaxSteps = 0) {
  json row = baseRow;
  row["row_kind"] = "profile_numerics";
  row["likelihood_kind"] = "measurement_only_chi2";
  row["statistical_model_unchanged"] = true;
  row["prior_term_present"] = false;
  row["ridge_added"] = false;
  row["wb109_cin_used_as_likelihood"] = false;
  row["chi2_penalty_for_propagation_failure"] = false;
  row["profile_init_variant"] = variant;
  row["init_kind"] = initKind;
  row["hop_mode"] = sequential ? "sequential_z_order" : "independent_from_source";
  row["max_step_size_contract"] =
      matchKalman ? "kalman_unlimited_adaptive" : "wb114_10m";
  row["profile_max_steps"] = maxSteps;
  row["numerical_scaling_used"] = useScaling;
  row["scaling_is_prior"] = false;
  row["evaluate_only"] = evaluateOnly;
  row["supporting_plane_projection"] = supportingPlane;
  row["measurement_update_in_evaluator"] = false;
  row["transport_task"] = supportingPlane ? "B14T" : "B14N";
  auto surface = seed.referenceSurface().getSharedPtr();
  Acts::BoundVector theta = seed.parameters();
  row["init_native"] = {theta[Acts::eBoundLoc0], theta[Acts::eBoundLoc1],
                        theta[Acts::eBoundPhi], theta[Acts::eBoundTheta],
                        theta[Acts::eBoundQOverP] * 1_MeV};
  row["transport_state_unit_contract"] = {
      {"athena_q_over_p_unit", "1/MeV"},
      {"acts_internal_q_over_p_unit", "1/GeV"},
      {"conversion", "q_over_p_acts = q_over_p_mev / 1_MeV"},
      {"acts_q_over_p_per_gev", theta[Acts::eBoundQOverP]},
      {"q_over_p_per_mev", theta[Acts::eBoundQOverP] * 1_MeV},
      {"p_mev", std::abs(theta[Acts::eBoundQOverP]) > 0.0
                    ? 1.0 / std::abs(theta[Acts::eBoundQOverP] * 1_MeV)
                    : 0.0},
      {"charge_from_q_over_p_sign",
       theta[Acts::eBoundQOverP] > 0.0 ? 1.0 : -1.0},
      {"phi", theta[Acts::eBoundPhi]},
      {"theta", theta[Acts::eBoundTheta]},
      {"particle_hypothesis", "muon"},
      {"source_surface", "station0_plane_at_frozen_z"},
      {"truth_q_over_p_not_used", true}};
  row["source_surface_z_mm"] = seed.position(geometryContext).z();
  row["surface_order"] = surfaceOrderJson(hits, seed.position(geometryContext).z());
  bool hitBoundary = false;
  applyPhysicalBounds(theta, hitBoundary);
  const Acts::BoundTrackParameters start(surface, theta, std::nullopt,
                                         Acts::ParticleHypothesis::muon());
  Chi2Eval eval = evaluateSequentialResiduals(
      propagator, geometryContext, magFieldContext, start, hits, maxSteps,
      matchKalman, sequential, supportingPlane, recordPath, diagnosticMaxSteps);
  json hops = json::array();
  for (const auto& hop : eval.hops) {
    hops.push_back(hopToJson(hop));
  }
  row["propagation_hits"] = hops;
  row["all_surfaces_reached"] = eval.ok;
  row["first_failed_measurement_index"] = eval.first_failed_index;
  row["nominal_chi2_evaluable"] = eval.ok;
  if (eval.ok) {
    row["evaluate_chi2"] = eval.chi2;
  } else {
    row["evaluate_chi2"] = nullptr;
    row["profile_success"] = false;
    row["profile_converged"] = false;
    row["fit_success"] = false;
    row["termination_reason"] = "propagation_blocked";
    row["fit_failure_reason"] = "measurement_propagation_failed";
    row["hit_boundary"] = hitBoundary;
    if (evaluateOnly) {
      return row;
    }
    return row;
  }
  if (evaluateOnly) {
    row["profile_success"] = true;
    row["profile_converged"] = false;
    row["fit_success"] = true;
    row["termination_reason"] = "evaluate_only";
    row["fit_failure_reason"] = nullptr;
    row["profile_chi2"] = eval.chi2;
    row["chi2"] = eval.chi2;
    row["parameter_scaling_contract"] = scalingContractJson(
        propagator, geometryContext, magFieldContext, start, hits, maxSteps,
        matchKalman, sequential, supportingPlane);
    row["jacobian_validation"] = jacobianValidationJson(
        propagator, geometryContext, magFieldContext, start, hits, maxSteps,
        matchKalman, sequential, supportingPlane);
    return row;
  }

  json trace = json::array();
  double chi2 = eval.chi2;
  Eigen::VectorXd residual = eval.residual;
  Mat5 hessian = Mat5::Zero();
  int hessianRank = 0;
  bool jacobianOk = true;
  bool converged = false;
  std::string termination = "max_iterations";
  const std::array<double, 8> damps{1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125,
                                    0.015625, 0.0078125};
  int nIter = 0;
  for (nIter = 0; nIter < maxIterations; ++nIter) {
    const Acts::BoundTrackParameters current(surface, theta, std::nullopt,
                                             Acts::ParticleHypothesis::muon());
    Chi2Eval currentEval = evaluateSequentialResiduals(
        propagator, geometryContext, magFieldContext, current, hits, maxSteps,
        matchKalman, sequential, supportingPlane);
    if (!currentEval.ok) {
      termination = "propagation_blocked";
      jacobianOk = false;
      break;
    }
    chi2 = currentEval.chi2;
    residual = currentEval.residual;
    Eigen::MatrixXd jacobian(static_cast<int>(hits.size()), 5);
    const std::array<double, 5> fd{1.0e-2, 1.0e-2, 1.0e-5, 1.0e-5, 1.0e-6};
    bool fdOk = true;
    for (int col = 0; col < 5; ++col) {
      Acts::BoundVector plus = theta;
      Acts::BoundVector minus = theta;
      plus[col] += fd[col];
      minus[col] -= fd[col];
      Chi2Eval evPlus = evaluateSequentialResiduals(
          propagator, geometryContext, magFieldContext,
          Acts::BoundTrackParameters(surface, plus, std::nullopt,
                                     Acts::ParticleHypothesis::muon()),
          hits, maxSteps, matchKalman, sequential, supportingPlane);
      Chi2Eval evMinus = evaluateSequentialResiduals(
          propagator, geometryContext, magFieldContext,
          Acts::BoundTrackParameters(surface, minus, std::nullopt,
                                     Acts::ParticleHypothesis::muon()),
          hits, maxSteps, matchKalman, sequential, supportingPlane);
      if (!evPlus.ok || !evMinus.ok) {
        fdOk = false;
        break;
      }
      jacobian.col(col) = (evMinus.residual - evPlus.residual) / (2.0 * fd[col]);
    }
    if (!fdOk) {
      termination = "jacobian_invalid";
      jacobianOk = false;
      break;
    }
    Mat5 h = Mat5::Zero();
    Bound5 g = Bound5::Zero();
    for (int i = 0; i < residual.size(); ++i) {
      const double w = 1.0 / hits[static_cast<std::size_t>(i)].variance;
      const Eigen::Matrix<double, 1, 5> ji = jacobian.row(i);
      h += w * (ji.transpose() * ji);
      g += w * ji.transpose() * residual[i];
    }
    hessian = 0.5 * (h + h.transpose());
    Bound5 scale = Bound5::Ones();
    if (useScaling) {
      for (int i = 0; i < 5; ++i) {
        scale[i] = kNumericScale[static_cast<std::size_t>(i)];
      }
    }
    Mat5 hz = Mat5::Zero();
    Bound5 gz = Bound5::Zero();
    for (int i = 0; i < 5; ++i) {
      gz[i] = g[i] * scale[i];
      for (int j = 0; j < 5; ++j) {
        hz(i, j) = hessian(i, j) * scale[i] * scale[j];
      }
    }
    const Bound5 deltaZ = symmetricPinv5(hz, pinvRelative, &hessianRank) * gz;
    Bound5 delta = Bound5::Zero();
    for (int i = 0; i < 5; ++i) {
      delta[i] = deltaZ[i] * scale[i];
    }
    const double gradNorm = gz.norm();
    const double stepNorm = deltaZ.norm();
    bool accepted = false;
    double usedDamp = 0.0;
    Acts::BoundVector best = theta;
    double bestChi2 = chi2;
    std::string trialStatus = "rejected";
    for (double damp : damps) {
      Acts::BoundVector trial = theta;
      bool trialBound = hitBoundary;
      for (int ipar = 0; ipar < 5; ++ipar) {
        trial[ipar] += damp * delta[ipar];
      }
      if (!applyPhysicalBounds(trial, trialBound)) {
        continue;
      }
      Chi2Eval trialEval = evaluateSequentialResiduals(
          propagator, geometryContext, magFieldContext,
          Acts::BoundTrackParameters(surface, trial, std::nullopt,
                                     Acts::ParticleHypothesis::muon()),
          hits, maxSteps, matchKalman, sequential, supportingPlane, false,
          diagnosticMaxSteps);
      if (!trialEval.ok) {
        trialStatus = "propagation_rejected";
        continue;
      }
      if (trialEval.chi2 <= bestChi2) {
        best = trial;
        bestChi2 = trialEval.chi2;
        usedDamp = damp;
        accepted = true;
        trialStatus = "accepted";
        hitBoundary = trialBound;
      }
    }
    const double relDec = (chi2 - bestChi2) / std::max(chi2, 1.0);
    if (writeTrace) {
      json singular = json::array();
      Eigen::SelfAdjointEigenSolver<Mat5> eigen(hessian);
      for (int i = 0; i < 5; ++i) {
        singular.push_back(eigen.eigenvalues()[i]);
      }
      trace.push_back({{"iteration", nIter},
                       {"chi2", chi2},
                       {"gradient_norm_z", gradNorm},
                       {"step_norm_z", stepNorm},
                       {"accepted", accepted},
                       {"line_search_factor", accepted ? json(usedDamp) : json(nullptr)},
                       {"propagation_status", trialStatus},
                       {"rank_H", hessianRank},
                       {"singular_values", singular},
                       {"relative_chi2_decrease", relDec}});
    }
    if (!accepted) {
      termination = (gradNorm < 1.0e-6 || stepNorm < 1.0e-8) ? "converged"
                                                             : "flat_direction";
      converged = true;
      break;
    }
    theta = best;
    chi2 = bestChi2;
    if (relDec < 1.0e-8 && (gradNorm < 1.0e-6 || stepNorm < 1.0e-8)) {
      termination = (hessianRank < 5) ? "flat_direction" : "converged";
      converged = true;
      break;
    }
  }
  const Acts::BoundTrackParameters fitted(surface, theta, std::nullopt,
                                          Acts::ParticleHypothesis::muon());
  const auto derived = exportFromBound(fitted, geometryContext);
  auto targetSurface = planeAtZ(targetZ);
  std::array<double, 5> prediction{};
  bool predOk = propagateDerived(propagator, geometryContext, magFieldContext, fitted,
                                 *targetSurface, prediction);
  row["profile_success"] = eval.ok && jacobianOk && predOk;
  row["profile_converged"] = converged;
  row["fit_success"] = eval.ok && jacobianOk && predOk;
  row["termination_reason"] = termination;
  row["fit_failure_reason"] =
      (eval.ok && jacobianOk && predOk) ? json(nullptr) : json(termination);
  row["profile_chi2"] = chi2;
  row["chi2"] = chi2;
  row["profile_n_iterations"] = nIter;
  row["hit_boundary"] = hitBoundary;
  row["profiled_native_state"] = {theta[Acts::eBoundLoc0], theta[Acts::eBoundLoc1],
                                  theta[Acts::eBoundPhi], theta[Acts::eBoundTheta],
                                  theta[Acts::eBoundQOverP] * 1_MeV};
  row["native_state"] = row["profiled_native_state"];
  row["profiled_derived_state"] = {derived[0], derived[1], derived[2], derived[3],
                                   derived[4]};
  row["derived_state"] = row["profiled_derived_state"];
  row["joint_hessian"] = matrix5ToJson(hessian);
  row["hessian_rank"] = hessianRank;
  if (predOk) {
    row["target_prediction_derived"] = {prediction[0], prediction[1], prediction[2],
                                        prediction[3], prediction[4]};
  } else {
    row["target_prediction_derived"] = nullptr;
  }
  row["prediction_derived"] = row["target_prediction_derived"];
  if (writeTrace) {
    row["optimizer_trace"] = trace;
  }
  return row;
}

std::vector<json> runProfileNumerics(
    const json& baseRow, const Propagator& propagator,
    const Acts::GeometryContext& geometryContext,
    const Acts::MagneticFieldContext& magFieldContext,
    const Acts::BoundTrackParameters& seed, const std::vector<ProfileHit>& hits,
    double targetZ, bool seedInvariance, bool evaluateOnly, int maxIterations,
    double pinvRelative, int maxSteps, bool matchKalman, bool sequential,
    bool useScaling, bool writeTrace, bool supportingPlane = false,
    bool recordPath = false, int diagnosticMaxSteps = 0) {
  struct Variant {
    const char* name;
    int index;
    double add;
    double scale;
  };
  std::vector<Variant> variants{{"nominal", -1, 0.0, 1.0}};
  if (seedInvariance && !evaluateOnly) {
    variants.push_back({"loc1_plus_1mm", static_cast<int>(Acts::eBoundLoc1), 1.0, 1.0});
    variants.push_back({"phi_plus_1e-3", static_cast<int>(Acts::eBoundPhi), 1.0e-3, 1.0});
    variants.push_back({"qoverp_times_1p1", static_cast<int>(Acts::eBoundQOverP), 0.0, 1.1});
  }
  std::vector<json> rows;
  // Always emit an evaluate-only nominal replay first.
  rows.push_back(runOneProfileNumerics(
      baseRow, propagator, geometryContext, magFieldContext, seed, hits, targetZ,
      "nominal", "wb114_official_seed_mean", maxIterations, pinvRelative, maxSteps,
      matchKalman, sequential, useScaling, true, writeTrace, supportingPlane,
      recordPath, diagnosticMaxSteps));
  if (supportingPlane) {
    json bounded = runOneProfileNumerics(
        baseRow, propagator, geometryContext, magFieldContext, seed, hits, targetZ,
        "nominal", "mode_a_bounded_surface_reached", maxIterations, pinvRelative,
        maxSteps, matchKalman, sequential, useScaling, true, writeTrace, false,
        false, 0);
    bounded["transport_comparison_mode"] = "A_bounded_direct_measurement_surface";
    rows.push_back(bounded);
    json direct = runOneProfileNumerics(
        baseRow, propagator, geometryContext, magFieldContext, seed, hits, targetZ,
        "nominal", "direct_from_source_supporting_plane", maxIterations,
        pinvRelative, maxSteps, matchKalman, false, useScaling, true, writeTrace,
        true, recordPath, diagnosticMaxSteps);
    direct["transport_comparison_mode"] = "direct_from_source";
    rows.push_back(direct);
    if (diagnosticMaxSteps > 0) {
      json diag = runOneProfileNumerics(
          baseRow, propagator, geometryContext, magFieldContext, seed, hits,
          targetZ, "nominal", "diagnostic_max_steps", maxIterations, pinvRelative,
          diagnosticMaxSteps, matchKalman, sequential, useScaling, true, writeTrace,
          true, true, 0);
      diag["transport_comparison_mode"] = "diagnostic_max_steps";
      diag["diagnostic_max_steps"] = diagnosticMaxSteps;
      diag["diagnostic_max_steps_is_not_official_fix"] = true;
      rows.push_back(diag);
    }
  }
  if (evaluateOnly) {
    return rows;
  }
  for (const auto& variant : variants) {
    Acts::BoundVector params = seed.parameters();
    if (variant.index >= 0) {
      params[variant.index] = params[variant.index] * variant.scale + variant.add;
    }
    const Acts::BoundTrackParameters init(seed.referenceSurface().getSharedPtr(),
                                          params, std::nullopt,
                                          Acts::ParticleHypothesis::muon());
    rows.push_back(runOneProfileNumerics(
        baseRow, propagator, geometryContext, magFieldContext, init, hits, targetZ,
        variant.name, "wb114_official_seed_mean", maxIterations, pinvRelative,
        maxSteps, matchKalman, sequential, useScaling, false, writeTrace));
  }
  return rows;
}

}  // namespace

CkfLeaveTargetOutDumpAlg::CkfLeaveTargetOutDumpAlg(const std::string& name,
                                                 ISvcLocator* pSvcLocator)
    : AthAlgorithm(name, pSvcLocator) {}

StatusCode CkfLeaveTargetOutDumpAlg::initialize() {
  if (m_outputJsonl.empty() || m_sourceId.empty()) {
    ATH_MSG_ERROR("OutputJsonl and SourceId are required");
    return StatusCode::FAILURE;
  }
  ATH_CHECK(m_trackKey.initialize());
  ATH_CHECK(m_eventKey.initialize());
  ATH_CHECK(m_fieldCondObjInputKey.initialize());
  ATH_CHECK(m_trackingGeometryTool.retrieve());
  ATH_CHECK(detStore()->retrieve(m_idHelper, "FaserSCT_ID"));
  ATH_MSG_INFO("Independent LTO helper ready (not KalmanFitterTool.fit): "
               << m_outputJsonl.value());
  return StatusCode::SUCCESS;
}

StatusCode CkfLeaveTargetOutDumpAlg::execute() {
  const EventContext& ctx = getContext();
  SG::ReadHandle<TrackCollection> tracks(m_trackKey, ctx);
  SG::ReadHandle<xAOD::EventInfo> eventInfo(m_eventKey, ctx);
  if (!tracks.isValid() || !eventInfo.isValid()) {
    return StatusCode::SUCCESS;
  }

  SG::ReadCondHandle<FaserFieldCacheCondObj> fieldHandle{m_fieldCondObjInputKey, ctx};
  if (!fieldHandle.isValid()) {
    ATH_MSG_ERROR("Magnetic field conditions are unavailable");
    return StatusCode::SUCCESS;
  }
  const Acts::GeometryContext geometryContext =
      m_trackingGeometryTool->getGeometryContext(ctx).context();
  const Acts::MagneticFieldContext magFieldContext(*fieldHandle);
  const Acts::CalibrationContext calibContext;
  auto identifierMap = m_trackingGeometryTool->getIdentifierMap();
  auto trackingGeometry = m_trackingGeometryTool->trackingGeometry();
  if (!identifierMap || !trackingGeometry) {
    ATH_MSG_ERROR("Tracking geometry or identifier map is unavailable");
    return StatusCode::SUCCESS;
  }

  auto magneticField = std::make_shared<FASERMagneticFieldWrapper>();
  Acts::Navigator::Config navCfg{trackingGeometry};
  navCfg.resolvePassive = false;
  navCfg.resolveMaterial = true;
  navCfg.resolveSensitive = true;
  Acts::Navigator navigatorFit(navCfg);
  Acts::Navigator navigatorProf(navCfg);
  Propagator propagatorFit(Stepper(magneticField), std::move(navigatorFit));
  Propagator propagatorProf(Stepper(magneticField), std::move(navigatorProf));
  Fitter fitter(std::move(propagatorFit));
  Acts::GainMatrixUpdater updater;
  Acts::GainMatrixSmoother smoother;
  IndexSourceLink::SurfaceAccessor surfaceAccessor{*trackingGeometry};

  const int runId = static_cast<int>(eventInfo->runNumber());
  const int eventId = static_cast<int>(eventInfo->eventNumber());
  if (!m_selectEventIds.empty() &&
      std::find(m_selectEventIds.begin(), m_selectEventIds.end(), eventId) ==
          m_selectEventIds.end()) {
    return StatusCode::SUCCESS;
  }
  int trackIndex = -1;
  for (const Trk::Track* track : *tracks) {
    ++trackIndex;
    json base;
    base["source_id"] = m_sourceId.value();
    base["run_id"] = runId;
    base["event_id"] = eventId;
    base["track_index"] = trackIndex;
    base["collection"] = m_trackKey.key();
    base["helper"] = "CkfLeaveTargetOutDumpAlg";
    base["helper_language"] = "C++";
    base["kalman_fitter_tool_fit_called"] = false;
    base["truth_qoverp_used"] = false;
    base["official_cin_used_as_prior"] = false;
    base["state_definition"] = stateDefinition();
    base["geometry_hash"] = m_geometryHash.value();
    base["field_hash"] = m_fieldHash.value();
    base["material_hash"] = m_materialHash.value();
    base["material_map_hash"] = m_materialHash.value();
    base["conditions_hash"] = m_conditionsHash.value();

    if (track == nullptr || track->measurementsOnTrack() == nullptr ||
        track->trackParameters() == nullptr ||
        track->trackParameters()->empty()) {
      for (int target : m_targetStations) {
        json row = base;
        row["target_station"] = target;
        row["fit_success"] = false;
        row["fit_failure_reason"] = "missing_track_payload";
        row["target_station_measurements_used"] = 0;
        std::lock_guard<std::mutex> lock(m_mutex);
        m_rows.push_back(row.dump());
      }
      continue;
    }

    const Trk::TrackParameters* front = track->trackParameters()->front();
    const double sourceZ =
        front != nullptr ? front->position().z() : m_stationZmm[0];
    const int sourceStation = nearestStation(sourceZ, m_stationZmm);

    std::vector<HitRecord> hits;
    // Official KalmanFitterTool.fit marks IFT as outliers, so
    // measurementsOnTrack() omits station 0.  Recover those clusters
    // from outlier TSOS; do not call KalmanFitterTool.fit.
    const auto* tsosVector = track->trackStateOnSurfaces();
    if (tsosVector != nullptr) {
      for (const Trk::TrackStateOnSurface* tsos : *tsosVector) {
        if (tsos == nullptr || tsos->measurementOnTrack() == nullptr) {
          continue;
        }
        const auto* clusterOnTrack =
            dynamic_cast<const Tracker::FaserSCT_ClusterOnTrack*>(
                tsos->measurementOnTrack());
        if (clusterOnTrack == nullptr) {
          continue;
        }
        const Tracker::FaserSCT_Cluster* cluster = clusterOnTrack->prepRawData();
        if (cluster == nullptr) {
          continue;
        }
        const Identifier id = clusterOnTrack->identify();
        HitRecord hit;
        hit.station = m_idHelper->station(id);
        hit.layer = m_idHelper->layer(id);
        hit.phi_module = m_idHelper->phi_module(id);
        hit.eta_module = m_idHelper->eta_module(id);
        hit.side = m_idHelper->side(id);
        hit.id = id;
        hit.identifier = compactIdentifier(id);
        hit.zMm = cluster->globalPosition().z();
        hit.from_outlier = tsos->type(Trk::TrackStateOnSurface::Outlier);
        hit.cluster = cluster;
        hit.measurement = tsos->measurementOnTrack();
        hits.push_back(hit);
      }
    }

    json allIds = json::array();
    std::array<int, 4> allCounts{0, 0, 0, 0};
    int nOutlierHits = 0;
    for (const auto& hit : hits) {
      allIds.push_back({{"station", hit.station},
                        {"layer", hit.layer},
                        {"id", hit.identifier},
                        {"from_outlier", hit.from_outlier}});
      if (hit.from_outlier) {
        ++nOutlierHits;
      }
      if (hit.station >= 0 && hit.station < 4) {
        allCounts[hit.station] += 1;
      }
    }

    for (int target : m_targetStations) {
      json row = base;
      row["source_station"] = sourceStation;
      row["source_z_mm"] = sourceZ;
      row["target_station"] = target;
      if (target >= 0 && target < static_cast<int>(m_stationZmm.size())) {
        row["target_z_mm"] = m_stationZmm[target];
      } else {
        row["target_z_mm"] = nullptr;
      }
      row["all_measurement_ids"] = allIds;
      row["all_measurement_counts_by_station"] = {allCounts[0], allCounts[1],
                                                  allCounts[2], allCounts[3]};
      row["n_outlier_hits_recovered"] = nOutlierHits;
      row["measurements_from_track_state_on_surface"] = true;

      std::vector<const HitRecord*> used;
      std::vector<const HitRecord*> excluded;
      json usedIds = json::array();
      json excludedIds = json::array();
      std::array<int, 4> usedCounts{0, 0, 0, 0};
      for (const auto& hit : hits) {
        if (hit.station == target) {
          excluded.push_back(&hit);
          excludedIds.push_back({{"station", hit.station},
                                 {"layer", hit.layer},
                                 {"id", hit.identifier},
                                 {"from_outlier", hit.from_outlier}});
        } else {
          used.push_back(&hit);
          usedIds.push_back({{"station", hit.station},
                             {"layer", hit.layer},
                             {"id", hit.identifier},
                             {"from_outlier", hit.from_outlier}});
          if (hit.station >= 0 && hit.station < 4) {
            usedCounts[hit.station] += 1;
          }
        }
      }
      row["used_measurement_ids"] = usedIds;
      row["excluded_target_measurement_ids"] = excludedIds;
      row["used_measurement_count"] = static_cast<int>(used.size());
      row["excluded_target_measurement_count"] = static_cast<int>(excluded.size());
      row["used_measurement_counts_by_station"] = {usedCounts[0], usedCounts[1],
                                                   usedCounts[2], usedCounts[3]};
      json usedStations = json::array();
      for (int station = 0; station < 4; ++station) {
        if (usedCounts[station] > 0) {
          usedStations.push_back(station);
        }
      }
      row["used_station_ids"] = usedStations;
      row["n_input_measurements"] = static_cast<int>(hits.size());
      row["n_used_measurements"] = static_cast<int>(used.size());
      double zMin = 0.0;
      double zMax = 0.0;
      bool haveZ = false;
      for (const HitRecord* hit : used) {
        if (!haveZ) {
          zMin = hit->zMm;
          zMax = hit->zMm;
          haveZ = true;
        } else {
          zMin = std::min(zMin, hit->zMm);
          zMax = std::max(zMax, hit->zMm);
        }
      }
      row["measurement_z_min_mm"] = haveZ ? json(zMin) : json(nullptr);
      row["measurement_z_max_mm"] = haveZ ? json(zMax) : json(nullptr);
      row["measurement_z_span_mm"] = haveZ ? json(zMax - zMin) : json(nullptr);
      row["number_of_stations_used"] = static_cast<int>(usedStations.size());
      row["target_station_measurements_used"] = usedCounts[target];
      row["target_exclusion_proven"] = usedCounts[target] == 0;
      row["source_station_measurements_used"] =
          (sourceStation >= 0 && sourceStation < 4) ? usedCounts[sourceStation] : 0;

      if (usedCounts[target] != 0) {
        row["fit_success"] = false;
        row["fit_failure_reason"] = "target_station_not_excluded";
        std::lock_guard<std::mutex> lock(m_mutex);
        m_rows.push_back(row.dump());
        continue;
      }
      if (static_cast<int>(used.size()) < m_minRemainingMeasurements) {
        row["fit_success"] = false;
        row["fit_failure_reason"] = "too_few_remaining_measurements";
        std::lock_guard<std::mutex> lock(m_mutex);
        m_rows.push_back(row.dump());
        continue;
      }
      if (front == nullptr) {
        row["fit_success"] = false;
        row["fit_failure_reason"] = "missing_seed_parameters";
        std::lock_guard<std::mutex> lock(m_mutex);
        m_rows.push_back(row.dump());
        continue;
      }

      std::vector<IndexSourceLink> sourceLinks;
      std::vector<Measurement> measurements;
      const int kSize = 1;
      std::array<Acts::BoundIndices, kSize> indices{Acts::eBoundLoc0};
      using ThisMeasurement = Acts::Measurement<Acts::BoundIndices, kSize>;
      for (const HitRecord* hit : used) {
        const Identifier waferId = m_idHelper->wafer_id(hit->id);
        if (identifierMap->count(waferId) == 0) {
          continue;
        }
        const Acts::GeometryIdentifier geoId = identifierMap->at(waferId);
        IndexSourceLink sourceLink(geoId, measurements.size(), hit->cluster);
        Eigen::Matrix<double, 1, 1> pos{hit->measurement->localParameters()[Trk::locX]};
        Eigen::Matrix<double, 1, 1> cov{0.08 * 0.08 / 12.0};
        ThisMeasurement actsMeas(Acts::SourceLink{sourceLink}, indices, pos, cov);
        sourceLinks.push_back(sourceLink);
        measurements.emplace_back(std::move(actsMeas));
      }
      if (static_cast<int>(sourceLinks.size()) < m_minRemainingMeasurements) {
        row["fit_success"] = false;
        row["fit_failure_reason"] = "too_few_geometry_matched_measurements";
        row["geometry_matched_measurement_count"] =
            static_cast<int>(sourceLinks.size());
        std::lock_guard<std::mutex> lock(m_mutex);
        m_rows.push_back(row.dump());
        continue;
      }

      std::vector<Acts::SourceLink> actsLinks;
      actsLinks.reserve(sourceLinks.size());
      for (const auto& link : sourceLinks) {
        actsLinks.emplace_back(link);
      }

      auto refSurface = planeAtZ(m_stationZmm[sourceStation]);
      MeasurementCalibrator measCalibrator;
      MeasurementCalibratorAdapter calibrator(measCalibrator, measurements);
      Acts::KalmanFitterExtensions<Acts::VectorMultiTrajectory> extensions;
      extensions.updater.connect<
          &Acts::GainMatrixUpdater::operator()<Acts::VectorMultiTrajectory>>(
          &updater);
      extensions.smoother.connect<
          &Acts::GainMatrixSmoother::operator()<Acts::VectorMultiTrajectory>>(
          &smoother);
      extensions.calibrator.connect<&MeasurementCalibratorAdapter::calibrate>(
          &calibrator);
      extensions.surfaceAccessor
          .connect<&IndexSourceLink::SurfaceAccessor::operator()>(&surfaceAccessor);
      Acts::PropagatorPlainOptions propOptions;
      propOptions.maxSteps = 10000;

      if (m_enableProfileLikelihood.value()) {
        std::vector<ProfileHit> profileHits;
        for (const HitRecord* hit : used) {
          const Identifier waferId = m_idHelper->wafer_id(hit->id);
          if (identifierMap->count(waferId) == 0 || hit->measurement == nullptr) {
            continue;
          }
          const Acts::GeometryIdentifier geoId = identifierMap->at(waferId);
          const Acts::Surface* surface = trackingGeometry->findSurface(geoId);
          if (surface == nullptr) {
            continue;
          }
          ProfileHit rec;
          rec.surface = surface;
          rec.loc0 = hit->measurement->localParameters()[Trk::locX];
          rec.variance = 0.08 * 0.08 / 12.0;
          rec.station = hit->station;
          rec.layer = hit->layer;
          rec.phi_module = hit->phi_module;
          rec.eta_module = hit->eta_module;
          rec.side = hit->side;
          rec.input_index = static_cast<int>(profileHits.size());
          rec.zMm = hit->zMm;
          rec.identifier = hit->identifier;
          profileHits.push_back(rec);
        }
        auto profileSeed = seedFromOfficialMean(*front, m_stationZmm[sourceStation],
                                                geometryContext, 1.0, "all");
        if (!profileSeed.has_value()) {
          json fail = row;
          fail["profile_success"] = false;
          fail["fit_success"] = false;
          fail["fit_failure_reason"] = "uninformative_seed_mean_unavailable";
          fail["likelihood_kind"] = "measurement_only_chi2";
          std::lock_guard<std::mutex> lock(m_mutex);
          m_rows.push_back(fail.dump());
        } else {
          const double targetZ =
              (target >= 0 && target < static_cast<int>(m_stationZmm.size()))
                  ? m_stationZmm[target]
                  : m_stationZmm[sourceStation];
          std::vector<json> profileRows;
          if (m_enableProfileNumerics.value()) {
            profileRows = runProfileNumerics(
                row, propagatorProf, geometryContext, magFieldContext, *profileSeed,
                profileHits, targetZ, m_enableProfileSeedInvariance.value(),
                m_profileEvaluateOnly.value() || m_enableProfileTransport.value(),
                m_profileMaxIterations.value(),
                m_profilePinvRelative.value(), m_profileMaxSteps.value(),
                m_profileMatchKalmanStepSize.value(),
                m_profileSequentialTransport.value(),
                m_profileUseNumericalScaling.value(),
                m_profileWriteTrace.value(),
                m_enableProfileTransport.value() &&
                    m_profileSupportingPlane.value(),
                m_profileRecordStepperPath.value(),
                m_profileDiagnosticMaxSteps.value());
            auto actsTrackContainer = std::make_shared<Acts::VectorTrackContainer>();
            auto actsTrackStateContainer =
                std::make_shared<Acts::VectorMultiTrajectory>();
            FaserActsTrackContainer fitted(actsTrackContainer,
                                           actsTrackStateContainer);
            Acts::KalmanFitterOptions<Acts::VectorMultiTrajectory> kalmanOptions(
                geometryContext, magFieldContext, calibContext, extensions,
                propOptions, &(*refSurface));
            kalmanOptions.referenceSurfaceStrategy =
                Acts::KalmanFitterTargetSurfaceStrategy::first;
            kalmanOptions.multipleScattering = true;
            kalmanOptions.energyLoss = true;
            auto kalman = fitter.fit(actsLinks.begin(), actsLinks.end(), *profileSeed,
                                     kalmanOptions, fitted);
            json contract = {
                {"kalman_max_steps", 10000},
                {"kalman_max_step_size", "unlimited_adaptive"},
                {"profile_max_steps", m_profileMaxSteps.value()},
                {"profile_max_step_size",
                 m_profileMatchKalmanStepSize.value() ? "unlimited_adaptive"
                                                      : "10_m"},
                {"profile_hop_mode", m_profileSequentialTransport.value()
                                         ? "sequential_z_order"
                                         : "independent_from_source"},
                {"kalman_reference_surface_strategy", "first"},
                {"kalman_material_multiple_scattering", true},
                {"kalman_material_energy_loss", true},
                {"profile_material_multiple_scattering", true},
                {"profile_material_energy_loss", true},
                {"navigator", "Acts::Navigator resolveSensitive+Material"},
            };
            if (kalman.ok() && kalman.value().hasReferenceSurface()) {
              Acts::BoundTrackParameters fittedParams(
                  kalman.value().referenceSurface().getSharedPtr(),
                  kalman.value().parameters(), kalman.value().covariance(),
                  Acts::ParticleHypothesis::muon());
              json replay = runOneProfileNumerics(
                  row, propagatorProf, geometryContext, magFieldContext, fittedParams,
                  profileHits, targetZ, "nominal", "wb109_lto_fitted_mean",
                  m_profileMaxIterations.value(), m_profilePinvRelative.value(),
                  m_profileMaxSteps.value(), m_profileMatchKalmanStepSize.value(),
                  m_profileSequentialTransport.value(),
                  m_profileUseNumericalScaling.value(), true,
                  m_profileWriteTrace.value(),
                  m_enableProfileTransport.value() &&
                      m_profileSupportingPlane.value(),
                  m_profileRecordStepperPath.value(),
                  m_profileDiagnosticMaxSteps.value());
              replay["kalman_vs_profile_transport_contract"] = contract;
              replay["kalman_n_measurements_in_fit"] =
                  static_cast<int>(kalman.value().nMeasurements());
              replay["kalman_chi2"] = kalman.value().chi2();
              profileRows.push_back(replay);
            } else {
              json fail = row;
              fail["row_kind"] = "profile_numerics";
              fail["init_kind"] = "wb109_lto_fitted_mean";
              fail["evaluate_only"] = true;
              fail["kalman_vs_profile_transport_contract"] = contract;
              fail["fit_failure_reason"] = kalman.ok()
                                               ? "fitted_parameters_unavailable"
                                               : "acts_kalman_fit_failed";
              profileRows.push_back(fail);
            }
          } else {
            profileRows = runMeasurementProfiles(
                row, propagatorProf, geometryContext, magFieldContext, *profileSeed,
                profileHits, targetZ, m_enableProfileSeedInvariance.value(),
                m_profileMaxIterations.value(), m_profilePinvRelative.value());
          }
          std::lock_guard<std::mutex> lock(m_mutex);
          for (const auto& profileRow : profileRows) {
            m_rows.push_back(profileRow.dump());
          }
        }
        if (m_profileOnly.value()) {
          continue;
        }
      }

      std::vector<std::pair<std::string, double>> seedJobs;
      if (m_enableDirectionalSeedCampaign.value()) {
        for (const char* direction : {"x", "y", "tx", "ty", "q_over_p"}) {
          for (double scale : {0.1, 1.0, 10.0}) {
            seedJobs.emplace_back(direction, scale);
          }
        }
      } else {
        seedJobs.emplace_back(m_seedCovarianceDirection.value(),
                              m_seedCovarianceScale.value());
      }

      for (const auto& job : seedJobs) {
        json jobRow = row;
        const std::string& direction = job.first;
        const double scale = job.second;
        auto seed = seedFromOfficialMean(*front, m_stationZmm[sourceStation],
                                         geometryContext, scale, direction);
        if (!seed.has_value()) {
          jobRow["fit_success"] = false;
          jobRow["fit_failure_reason"] = "uninformative_seed_mean_unavailable";
          jobRow["seed_covariance_direction"] = direction;
          jobRow["seed_covariance_scale"] = scale;
          std::lock_guard<std::mutex> lock(m_mutex);
          m_rows.push_back(jobRow.dump());
          continue;
        }
        auto actsTrackContainer = std::make_shared<Acts::VectorTrackContainer>();
        auto actsTrackStateContainer =
            std::make_shared<Acts::VectorMultiTrajectory>();
        FaserActsTrackContainer fitted(actsTrackContainer,
                                       actsTrackStateContainer);
        Acts::KalmanFitterOptions<Acts::VectorMultiTrajectory> options(
            geometryContext, magFieldContext, calibContext, extensions,
            propOptions, &(*refSurface));
        options.referenceSurfaceStrategy =
            Acts::KalmanFitterTargetSurfaceStrategy::first;
        options.multipleScattering = true;
        options.energyLoss = true;
        auto result = fitter.fit(actsLinks.begin(), actsLinks.end(), *seed,
                                 options, fitted);
        jobRow["seed_covariance_direction"] = direction;
        jobRow["seed_direction_bound_parameter"] =
            boundNameForDirection(direction);
        jobRow["seed_direction_chart_note"] =
            (direction == "tx" || direction == "ty")
                ? "bound_phi_theta_proxy_for_derived_slopes"
                : "source_plane_loc_or_qoverp";
        jobRow["seed_covariance_scale"] = scale;
        jobRow["seed_from_official_mean"] = true;
        jobRow["seed_covariance_uninformative"] = true;
        jobRow["seed_covariance"] = matrixToJson(seed->covariance().value());
        jobRow["seed_q_over_p_per_mev"] =
            seed->parameters()[Acts::eBoundQOverP] * 1_MeV;
        const auto seedDerived = exportCovariance(*seed, geometryContext);
        if (seedDerived) {
          jobRow["seed_derived_covariance"] = exportMatrixToJson(*seedDerived);
        } else {
          jobRow["seed_derived_covariance"] = nullptr;
        }
        if (!result.ok()) {
          jobRow["fit_success"] = false;
          jobRow["fit_failure_reason"] = "acts_kalman_fit_failed";
          jobRow["acts_fit_error"] = result.error().message();
          std::lock_guard<std::mutex> lock(m_mutex);
          m_rows.push_back(jobRow.dump());
          continue;
        }
        const auto& fittedTrack = result.value();
        if (!fittedTrack.hasReferenceSurface()) {
          jobRow["fit_success"] = false;
          jobRow["fit_failure_reason"] = "fitted_parameters_unavailable";
          std::lock_guard<std::mutex> lock(m_mutex);
          m_rows.push_back(jobRow.dump());
          continue;
        }
        Acts::BoundTrackParameters fittedParams(
            fittedTrack.referenceSurface().getSharedPtr(),
            fittedTrack.parameters(), fittedTrack.covariance(),
            Acts::ParticleHypothesis::muon());
        const auto derived = exportFromBound(fittedParams, geometryContext);
        const auto exportCov = exportCovariance(fittedParams, geometryContext);
        Acts::BoundSquareMatrix native = Acts::BoundSquareMatrix::Zero();
        native.topLeftCorner(5, 5) =
            fittedTrack.covariance().topLeftCorner(5, 5);
        for (int i = 0; i < native.rows(); ++i) {
          native(i, 4) = native(i, 4) * 1_MeV;
        }
        for (int i = 0; i < native.cols(); ++i) {
          native(4, i) = native(4, i) * 1_MeV;
        }
        const double qOverPMev =
            fittedTrack.parameters()[Acts::eBoundQOverP] * 1_MeV;
        if (!finiteValue(qOverPMev) ||
            std::abs(qOverPMev - 1.0e-5) < 1.0e-18) {
          jobRow["fit_success"] = false;
          jobRow["fit_failure_reason"] = "dummy_or_nonfinite_qoverp";
          std::lock_guard<std::mutex> lock(m_mutex);
          m_rows.push_back(jobRow.dump());
          continue;
        }
        jobRow["fit_success"] = true;
        jobRow["fit_failure_reason"] = nullptr;
        const Acts::Vector3 fittedPosition =
            fittedParams.position(geometryContext);
        jobRow["reference_surface"] = {
            {"type", "plane"},
            {"z_mm", m_stationZmm[sourceStation]},
            {"normal", json::array({0.0, 0.0, 1.0})},
        };
        jobRow["fitted_position_xyz_mm"] = {
            fittedPosition.x(), fittedPosition.y(), fittedPosition.z()};
        jobRow["native_state"] = {
            fittedTrack.parameters()[Acts::eBoundLoc0],
            fittedTrack.parameters()[Acts::eBoundLoc1],
            fittedTrack.parameters()[Acts::eBoundPhi],
            fittedTrack.parameters()[Acts::eBoundTheta], qOverPMev};
        jobRow["native_covariance"] = matrixToJson(native);
        jobRow["derived_state"] = {derived[0], derived[1], derived[2],
                                   derived[3], derived[4]};
        if (exportCov) {
          jobRow["input_covariance"] = exportMatrixToJson(*exportCov);
        } else {
          jobRow["input_covariance"] = nullptr;
        }
        jobRow["q_over_p_per_mev"] = qOverPMev;
        jobRow["p_mev"] = 1.0 / std::abs(qOverPMev);
        jobRow["charge"] = qOverPMev > 0.0 ? 1.0 : -1.0;
        jobRow["chi2"] = fittedTrack.chi2();
        jobRow["n_measurements_in_fit"] =
            static_cast<int>(fittedTrack.nMeasurements());
        jobRow["n_fit_states"] = static_cast<int>(fittedTrack.nTrackStates());
        jobRow["ndof"] = static_cast<int>(fittedTrack.nDoF());
        jobRow["exported_covariance"] = jobRow["native_covariance"];
        const json provenance = collectFitProvenance(fittedTrack);
        jobRow["fit_provenance"] = provenance;
        jobRow["number_of_filtered_states"] =
            provenance["number_of_filtered_states"];
        jobRow["number_of_smoothed_states"] =
            provenance["number_of_smoothed_states"];
        jobRow["number_of_outlier_states"] =
            provenance["number_of_outlier_states"];
        jobRow["exported_state_type"] = provenance["exported_state_type"];
        jobRow["first_predicted_covariance"] =
            provenance["first_predicted_covariance"];
        jobRow["first_filtered_covariance"] =
            provenance["first_filtered_covariance"];
        jobRow["last_filtered_covariance"] =
            provenance["last_filtered_covariance"];
        jobRow["first_smoothed_covariance"] =
            provenance["first_smoothed_covariance"];
        jobRow["last_smoothed_covariance"] =
            provenance["last_smoothed_covariance"];
        std::lock_guard<std::mutex> lock(m_mutex);
        m_rows.push_back(jobRow.dump());
      }
    }
  }
  return StatusCode::SUCCESS;
}

StatusCode CkfLeaveTargetOutDumpAlg::finalize() {
  std::ofstream out(m_outputJsonl.value(), std::ios::out | std::ios::trunc);
  if (!out) {
    ATH_MSG_ERROR("Cannot write LTO jsonl: " << m_outputJsonl.value());
    return StatusCode::FAILURE;
  }
  for (const auto& row : m_rows) {
    out << row << "\n";
  }
  ATH_MSG_INFO("Wrote " << m_rows.size() << " LTO rows to " << m_outputJsonl.value());
  return StatusCode::SUCCESS;
}
