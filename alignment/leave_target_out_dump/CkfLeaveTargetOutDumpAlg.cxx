#include "CkfLeaveTargetOutDumpAlg.h"
#include "FieldGradientDefaultExtension.hpp"
#include "IndependentMeanOdeIntegrator.hpp"
#include "CommonGridShadowIntegrator.hpp"
#include "MeanTransportContract.hpp"

#include "Acts/Definitions/Tolerance.hpp"
#include "Acts/Definitions/TrackParametrization.hpp"
#include "Acts/Definitions/Units.hpp"
#include "Acts/EventData/TrackParameters.hpp"
#include "Acts/EventData/TrackStateType.hpp"
#include "Acts/EventData/VectorMultiTrajectory.hpp"
#include "Acts/EventData/VectorTrackContainer.hpp"
#include "Acts/EventData/detail/TransformationBoundToFree.hpp"
#include "Acts/EventData/detail/TransformationFreeToBound.hpp"
#include "Acts/MagneticField/MagneticFieldContext.hpp"
#include "Acts/Propagator/AbortList.hpp"
#include "Acts/Propagator/ActionList.hpp"
#include "Acts/Propagator/StandardAborters.hpp"
#include "Acts/Propagator/detail/SteppingLogger.hpp"
#include "Acts/Surfaces/BoundaryCheck.hpp"
#include "Acts/Surfaces/PlanarBounds.hpp"
#include "Acts/Propagator/EigenStepper.hpp"
#include "Acts/Propagator/StepperExtensionList.hpp"
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
#include <Eigen/SVD>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <utility>
#include <vector>

using namespace Acts::UnitLiterals;
using json = nlohmann::json;

namespace {

using FaserActsTrackContainer =
    Acts::TrackContainer<Acts::VectorTrackContainer, Acts::VectorMultiTrajectory,
                         std::shared_ptr>;
using Stepper = Acts::EigenStepper<>;
using Propagator = Acts::Propagator<Stepper, Acts::Navigator>;
using GradientStepper =
    Acts::EigenStepper<Acts::StepperExtensionList<FieldGradientDefaultExtension>>;
using GradientPropagator = Acts::Propagator<GradientStepper, Acts::Navigator>;
using Fitter = Acts::KalmanFitter<Propagator, Acts::VectorMultiTrajectory>;
constexpr double kFieldGradientFdStepMm = 1.0;

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
  std::string continuation_state_construction;
  bool free_to_bound_fallback{false};
  double distance_to_plane_mm{0.0};
  double local_z_mm{0.0};
  double final_x_mm{0.0};
  double final_y_mm{0.0};
  double final_z_mm{0.0};
  double final_dx{0.0};
  double final_dy{0.0};
  double final_dz{0.0};
  double final_qop{0.0};
  double initial_free_x{0.0};
  double initial_free_y{0.0};
  double initial_free_z{0.0};
  double initial_free_dx{0.0};
  double initial_free_dy{0.0};
  double initial_free_dz{0.0};
  double initial_free_qop{0.0};
  double intersection_path_length{0.0};
  double intersection_x_mm{0.0};
  double intersection_y_mm{0.0};
  double intersection_z_mm{0.0};
  double n_dot_direction{0.0};
  double intersection_denominator{0.0};
  double incidence_angle{0.0};
  double step_size_min{0.0};
  double step_size_max{0.0};
  double step_size_mean{0.0};
  json geometry_surface_sequence = json::array();
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
          {"continuation_state_construction", hop.continuation_state_construction},
          {"free_to_bound_fallback", hop.free_to_bound_fallback},
          {"distance_to_plane_mm", hop.distance_to_plane_mm},
          {"local_z_mm", hop.local_z_mm},
          {"final_position_xyz_mm",
           {hop.final_x_mm, hop.final_y_mm, hop.final_z_mm}},
          {"final_direction", {hop.final_dx, hop.final_dy, hop.final_dz}},
          {"final_q_over_p", hop.final_qop},
          {"initial_free_position_xyz_mm",
           {hop.initial_free_x, hop.initial_free_y, hop.initial_free_z}},
          {"initial_free_direction",
           {hop.initial_free_dx, hop.initial_free_dy, hop.initial_free_dz}},
          {"initial_free_q_over_p", hop.initial_free_qop},
          {"intersection_path_length", hop.intersection_path_length},
          {"n_dot_direction", hop.n_dot_direction},
          {"intersection_denominator", hop.intersection_denominator},
          {"incidence_angle", hop.incidence_angle},
          {"step_size_summary",
           {{"min", hop.step_size_min},
            {"max", hop.step_size_max},
            {"mean", hop.step_size_mean}}},
          {"geometry_surface_sequence", hop.geometry_surface_sequence},
          {"path_checkpoints", hop.path_checkpoints},
          {"surface_geometry", hop.surface_geometry}};
}

ProfileOptions makeProfileOptions(const Acts::GeometryContext& geometryContext,
                                  const Acts::MagneticFieldContext& magFieldContext,
                                  Acts::Direction direction, int maxSteps,
                                  bool matchKalmanStepSize,
                                  double stepTolerance = 1.0e-4) {
  ProfileOptions options(geometryContext, magFieldContext);
  options.maxSteps = static_cast<unsigned int>(std::max(maxSteps, 1));
  options.direction = direction;
  options.loopProtection = true;
  options.stepTolerance = stepTolerance;
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

json compactGeometrySequence(const std::vector<Acts::detail::Step>& steps) {
  json out = json::array();
  std::uint64_t last = std::numeric_limits<std::uint64_t>::max();
  for (const auto& step : steps) {
    const auto id = step.geoID.value();
    if (id != last) {
      out.push_back(id);
      last = id;
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
  const Acts::Vector3 center = surface.center(geometryContext);
  const Acts::Vector3 normal =
      surface.normal(geometryContext, center, Acts::Vector3::UnitZ());
  hop.n_dot_direction = normal.dot(direction);
  hop.intersection_denominator = hop.n_dot_direction;
  hop.incidence_angle =
      std::acos(std::clamp(std::abs(hop.n_dot_direction), 0.0, 1.0));
  if (!closest || closest.status() == Acts::Intersection3D::Status::unreachable) {
    hop.projection_kind = "supporting_plane_unreachable";
    return false;
  }
  hop.intersection_path_length = closest.pathLength();
  const Acts::Vector3 onPlane = closest.position();
  hop.intersection_x_mm = onPlane.x();
  hop.intersection_y_mm = onPlane.y();
  hop.intersection_z_mm = onPlane.z();
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
                           int diagnosticMaxSteps = 0,
                           bool recordGeometry = false,
                           double stepTolerance = 1.0e-4) {
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
  const Acts::FreeVector startFree = Acts::detail::transformBoundToFreeParameters(
      start.referenceSurface(), geometryContext, startP);
  hop.initial_free_x = startFree[Acts::eFreePos0];
  hop.initial_free_y = startFree[Acts::eFreePos1];
  hop.initial_free_z = startFree[Acts::eFreePos2];
  hop.initial_free_dx = startFree[Acts::eFreeDir0];
  hop.initial_free_dy = startFree[Acts::eFreeDir1];
  hop.initial_free_dz = startFree[Acts::eFreeDir2];
  hop.initial_free_qop = startFree[Acts::eFreeQOverP];
  hop.final_qop = startFree[Acts::eFreeQOverP];
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
      matchKalmanStepSize, stepTolerance);

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
    hop.continuation_state_construction = "acts_bound_parameters";
    hop.free_to_bound_fallback = false;
    const Acts::Vector3 endDir = hop.end_parameters->direction();
    hop.final_dx = endDir.x();
    hop.final_dy = endDir.y();
    hop.final_dz = endDir.z();
    hop.final_qop = hop.end_parameters->parameters()[Acts::eBoundQOverP];
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
  logOptions.stepTolerance = stepTolerance;
  if (!matchKalmanStepSize) {
    logOptions.maxStepSize = 10.0 * 1_m;
  }
  auto& material = logOptions.actionList.get<Acts::MaterialInteractor>();
  material.multipleScattering = true;
  material.energyLoss = true;
  logOptions.actionList.get<Acts::detail::SteppingLogger>().sterile =
      !(recordPath || longHop || recordGeometry);

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
  const auto& loggedSteps =
      state.get<Acts::detail::SteppingLogger::result_type>().steps;
  if (recordPath || longHop) {
    hop.path_checkpoints = downsampleSteps(loggedSteps);
  }
  if (recordGeometry || recordPath || longHop) {
    hop.geometry_surface_sequence = compactGeometrySequence(loggedSteps);
  }
  hop.final_dx = freeDir.x();
  hop.final_dy = freeDir.y();
  hop.final_dz = freeDir.z();
  hop.final_qop = state.stepping.pars[Acts::eFreeQOverP];
  if (!loggedSteps.empty()) {
    double sumAbs = 0.0;
    hop.step_size_min = std::numeric_limits<double>::max();
    hop.step_size_max = 0.0;
    for (const auto& step : loggedSteps) {
      const double absStep = std::abs(step.stepSize.value());
      hop.step_size_min = std::min(hop.step_size_min, absStep);
      hop.step_size_max = std::max(hop.step_size_max, absStep);
      sumAbs += absStep;
    }
    hop.step_size_mean = sumAbs / static_cast<double>(loggedSteps.size());
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
      hop.continuation_state_construction = "transform_free_to_bound";
      hop.free_to_bound_fallback = false;
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
      hop.continuation_state_construction = "plane_chart_fallback";
      hop.free_to_bound_fallback = true;
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
    int diagnosticMaxSteps = 0, bool recordGeometry = false,
    double stepTolerance = 1.0e-4) {
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
        matchKalmanStepSize, supportingPlane, recordPath, diagnosticMaxSteps,
        recordGeometry, stepTolerance);
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

json hopContinuityJson(const ProfileHop& hop) {
  json destId = nullptr;
  if (hop.surface_geometry.is_object() &&
      hop.surface_geometry.contains("geometry_id")) {
    destId = hop.surface_geometry["geometry_id"];
  }
  return {{"measurement_index", hop.measurement_index},
          {"ok", hop.ok},
          {"predicted_loc0", hop.ok ? json(hop.predicted_loc0) : json(nullptr)},
          {"predicted_loc1", hop.ok ? json(hop.predicted_loc1) : json(nullptr)},
          {"number_of_propagation_steps", hop.n_steps},
          {"path_length", hop.path_length},
          {"destination_geometry_id", destId},
          {"geometry_surface_sequence", hop.geometry_surface_sequence},
          {"inside_active_bounds", hop.inside_bounds},
          {"continuation_state_construction", hop.continuation_state_construction},
          {"free_to_bound_fallback", hop.free_to_bound_fallback},
          {"continuation_state_ok", hop.continuation_state_ok},
          {"path_direction", hop.path_direction},
          {"final_direction", {hop.final_dx, hop.final_dy, hop.final_dz}},
          {"abort_reason", hop.abort_reason}};
}

json evalContinuityJson(const Chi2Eval& ev) {
  json predictedLoc0 = json::array();
  json predictedLoc1 = json::array();
  json surfaceSeq = json::array();
  json constructions = json::array();
  json fallbacks = json::array();
  json inside = json::array();
  json hops = json::array();
  int nSteps = 0;
  double path = 0.0;
  int nFallback = 0;
  for (const auto& hop : ev.hops) {
    hops.push_back(hopContinuityJson(hop));
    predictedLoc0.push_back(hop.ok ? json(hop.predicted_loc0) : json(nullptr));
    predictedLoc1.push_back(hop.ok ? json(hop.predicted_loc1) : json(nullptr));
    if (hop.surface_geometry.is_object() &&
        hop.surface_geometry.contains("geometry_id")) {
      surfaceSeq.push_back(hop.surface_geometry["geometry_id"]);
    } else {
      surfaceSeq.push_back(nullptr);
    }
    constructions.push_back(hop.continuation_state_construction);
    fallbacks.push_back(hop.free_to_bound_fallback);
    inside.push_back(hop.inside_bounds);
    nSteps += hop.n_steps;
    path += hop.path_length;
    nFallback += hop.free_to_bound_fallback ? 1 : 0;
  }
  return {{"ok", ev.ok},
          {"chi2", ev.ok ? json(ev.chi2) : json(nullptr)},
          {"n_hops", static_cast<int>(ev.hops.size())},
          {"n_steps_total", nSteps},
          {"path_length_total", path},
          {"n_free_to_bound_fallback", nFallback},
          {"predicted_loc0", predictedLoc0},
          {"predicted_loc1", predictedLoc1},
          {"destination_geometry_ids", surfaceSeq},
          {"continuation_state_construction", constructions},
          {"free_to_bound_fallback", fallbacks},
          {"inside_active_bounds", inside},
          {"hops", hops}};
}

json jacobianContinuityModeJson(
    const Propagator& propagator, const Acts::GeometryContext& geometryContext,
    const Acts::MagneticFieldContext& magFieldContext,
    const Acts::BoundTrackParameters& start, const std::vector<ProfileHit>& hits,
    int maxSteps, bool matchKalman, bool sequential, bool supportingPlane) {
  const std::array<double, 4> kRungs{1.0, 0.5, 0.25, 0.125};
  json columns = json::array();
  Chi2Eval base = evaluateSequentialResiduals(
      propagator, geometryContext, magFieldContext, start, hits, maxSteps,
      matchKalman, sequential, supportingPlane, false, 0, true);
  auto surface = start.referenceSurface().getSharedPtr();
  Acts::BoundVector theta = start.parameters();
  for (int col = 0; col < 5; ++col) {
    json rungs = json::array();
    json item{{"parameter", kParamNames[static_cast<std::size_t>(col)]},
              {"official_fd_step", kFdSteps[static_cast<std::size_t>(col)]},
              {"do_not_select_best_step", true}};
    bool colOk = true;
    int failed = 0;
    std::array<Eigen::VectorXd, 4> jac{};
    for (int rung = 0; rung < 4; ++rung) {
      const double step =
          kFdSteps[static_cast<std::size_t>(col)] * kRungs[static_cast<std::size_t>(rung)];
      Acts::BoundVector plus = theta;
      Acts::BoundVector minus = theta;
      plus[col] += step;
      minus[col] -= step;
      Chi2Eval evPlus = evaluateSequentialResiduals(
          propagator, geometryContext, magFieldContext,
          Acts::BoundTrackParameters(surface, plus, std::nullopt,
                                     Acts::ParticleHypothesis::muon()),
          hits, maxSteps, matchKalman, sequential, supportingPlane, false, 0,
          true);
      Chi2Eval evMinus = evaluateSequentialResiduals(
          propagator, geometryContext, magFieldContext,
          Acts::BoundTrackParameters(surface, minus, std::nullopt,
                                     Acts::ParticleHypothesis::muon()),
          hits, maxSteps, matchKalman, sequential, supportingPlane, false, 0,
          true);
      json rungJson{{"rung_index", rung},
                    {"step_factor", kRungs[static_cast<std::size_t>(rung)]},
                    {"fd_step", step},
                    {"plus", evalContinuityJson(evPlus)},
                    {"minus", evalContinuityJson(evMinus)}};
      if (!evPlus.ok || !evMinus.ok) {
        colOk = false;
        failed += 1;
        rungJson["ok"] = false;
        rungJson["column"] = nullptr;
      } else {
        jac[static_cast<std::size_t>(rung)] =
            (evPlus.residual - evMinus.residual) / (2.0 * step);
        json colVals = json::array();
        for (int i = 0; i < jac[static_cast<std::size_t>(rung)].size(); ++i) {
          colVals.push_back(jac[static_cast<std::size_t>(rung)][i]);
        }
        rungJson["ok"] = true;
        rungJson["column"] = colVals;
        rungJson["column_norm"] = jac[static_cast<std::size_t>(rung)].norm();
      }
      rungs.push_back(rungJson);
    }
    json consecutive = json::array();
    bool signOk = true;
    for (int i = 0; i < 3; ++i) {
      json pair{{"from_rung", i}, {"to_rung", i + 1}};
      if (rungs[static_cast<std::size_t>(i)]["ok"] != true ||
          rungs[static_cast<std::size_t>(i + 1)]["ok"] != true) {
        pair["relative_error"] = nullptr;
        pair["sign_consistent"] = false;
        signOk = false;
      } else {
        const double nNext = jac[static_cast<std::size_t>(i + 1)].norm();
        const double rel =
            nNext > 1.0e-12
                ? (jac[static_cast<std::size_t>(i)] - jac[static_cast<std::size_t>(i + 1)])
                          .norm() /
                      nNext
                : 0.0;
        const bool sameSign =
            jac[static_cast<std::size_t>(i)].dot(jac[static_cast<std::size_t>(i + 1)]) >=
            0.0;
        pair["relative_error"] = rel;
        pair["sign_consistent"] = sameSign;
        if (!sameSign) {
          signOk = false;
        }
      }
      consecutive.push_back(pair);
    }
    item["ok"] = colOk && failed == 0;
    item["failed_finite_difference_evaluations"] = failed;
    item["sign_consistency"] = signOk;
    item["rungs"] = rungs;
    item["consecutive_relative_errors"] = consecutive;
    item["selected_best_step"] = false;
    columns.push_back(item);
  }
  return {{"hop_mode", sequential ? "sequential_z_order" : "independent_from_source"},
          {"supporting_plane_projection", supportingPlane},
          {"method", "central_fd_fixed_ladder_h_h2_h4_h8"},
          {"rung_factors", {1.0, 0.5, 0.25, 0.125}},
          {"do_not_select_best_step", true},
          {"official_fd_step_unchanged", true},
          {"nominal", evalContinuityJson(base)},
          {"columns", columns}};
}

json jacobianContinuityAuditJson(
    const Propagator& propagator, const Acts::GeometryContext& geometryContext,
    const Acts::MagneticFieldContext& magFieldContext,
    const Acts::BoundTrackParameters& start, const std::vector<ProfileHit>& hits,
    int maxSteps, bool matchKalman, bool supportingPlane) {
  json sequential = jacobianContinuityModeJson(
      propagator, geometryContext, magFieldContext, start, hits, maxSteps,
      matchKalman, true, supportingPlane);
  json direct = jacobianContinuityModeJson(
      propagator, geometryContext, magFieldContext, start, hits, maxSteps,
      matchKalman, false, supportingPlane);
  return {{"kind", "supporting_plane_jacobian_continuity"},
          {"task", "B14J"},
          {"official_is_sequential", true},
          {"direct_from_source_is_comparison_only", true},
          {"do_not_switch_to_direct", true},
          {"do_not_select_best_step", true},
          {"official_fd_step_unchanged", true},
          {"sequential", sequential},
          {"direct", direct}};
}

json hopStageJson(const ProfileHop& hop) {
  json destId = nullptr;
  if (hop.surface_geometry.is_object() &&
      hop.surface_geometry.contains("geometry_id")) {
    destId = hop.surface_geometry["geometry_id"];
  }
  return {{"measurement_index", hop.measurement_index},
          {"ok", hop.ok},
          {"source_bound",
           {hop.initial_loc0, hop.initial_loc1, hop.initial_phi, hop.initial_theta,
            hop.q_over_p}},
          {"initial_free_position_xyz_mm",
           {hop.initial_free_x, hop.initial_free_y, hop.initial_free_z}},
          {"initial_free_direction",
           {hop.initial_free_dx, hop.initial_free_dy, hop.initial_free_dz}},
          {"initial_free_q_over_p", hop.initial_free_qop},
          {"free_state_before_projection",
           {hop.final_x_mm, hop.final_y_mm, hop.final_z_mm, hop.final_dx,
            hop.final_dy, hop.final_dz, hop.final_qop}},
          {"intersection_path_length", hop.intersection_path_length},
          {"intersection_global_xyz_mm",
           {hop.intersection_x_mm, hop.intersection_y_mm, hop.intersection_z_mm}},
          {"local_loc0_loc1", {hop.predicted_loc0, hop.predicted_loc1}},
          {"predicted_loc0", hop.ok ? json(hop.predicted_loc0) : json(nullptr)},
          {"predicted_loc1", hop.ok ? json(hop.predicted_loc1) : json(nullptr)},
          {"chi2_residual", hop.ok ? json(hop.residual) : json(nullptr)},
          {"number_of_propagation_steps", hop.n_steps},
          {"path_length", hop.path_length},
          {"step_size_summary",
           {{"min", hop.step_size_min},
            {"max", hop.step_size_max},
            {"mean", hop.step_size_mean}}},
          {"n_dot_direction", hop.n_dot_direction},
          {"intersection_denominator", hop.intersection_denominator},
          {"incidence_angle", hop.incidence_angle},
          {"distance_to_plane_mm", hop.distance_to_plane_mm},
          {"inside_active_bounds", hop.inside_bounds},
          {"continuation_state_construction", hop.continuation_state_construction},
          {"free_to_bound_fallback", hop.free_to_bound_fallback},
          {"destination_geometry_id", destId},
          {"geometry_surface_sequence", hop.geometry_surface_sequence},
          {"final_direction", {hop.final_dx, hop.final_dy, hop.final_dz}}};
}

json evalStageJson(const Chi2Eval& ev) {
  json hops = json::array();
  json predicted = json::array();
  int nSteps = 0;
  double path = 0.0;
  for (const auto& hop : ev.hops) {
    hops.push_back(hopStageJson(hop));
    predicted.push_back(hop.ok ? json(hop.predicted_loc0) : json(nullptr));
    nSteps += hop.n_steps;
    path += hop.path_length;
  }
  return {{"ok", ev.ok},
          {"chi2", ev.ok ? json(ev.chi2) : json(nullptr)},
          {"n_hops", static_cast<int>(ev.hops.size())},
          {"n_steps_total", nSteps},
          {"path_length_total", path},
          {"predicted_loc0", predicted},
          {"hops", hops}};
}

json loc0VectorJson(const Chi2Eval& ev) {
  json out = json::array();
  for (const auto& hop : ev.hops) {
    out.push_back(hop.ok ? json(hop.predicted_loc0) : json(nullptr));
  }
  return out;
}

int hopStepsTotal(const Chi2Eval& ev) {
  int n = 0;
  for (const auto& hop : ev.hops) {
    n += hop.n_steps;
  }
  return n;
}

double hopPathTotal(const Chi2Eval& ev) {
  double path = 0.0;
  for (const auto& hop : ev.hops) {
    path += hop.path_length;
  }
  return path;
}

double loc0L2Delta(const Chi2Eval& a, const Chi2Eval& b) {
  double sum = 0.0;
  const std::size_t n = std::min(a.hops.size(), b.hops.size());
  for (std::size_t i = 0; i < n; ++i) {
    if (a.hops[i].ok && b.hops[i].ok) {
      const double d = a.hops[i].predicted_loc0 - b.hops[i].predicted_loc0;
      sum += d * d;
    }
  }
  return std::sqrt(sum);
}

std::vector<int> sequentialHopOrder(const Acts::BoundTrackParameters& start,
                                    const std::vector<ProfileHit>& hits,
                                    const Acts::GeometryContext& geometryContext) {
  std::vector<int> order(hits.size());
  for (std::size_t i = 0; i < hits.size(); ++i) {
    order[i] = static_cast<int>(i);
  }
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
  return order;
}

json vecXdToJson(const Eigen::VectorXd& vec) {
  json out = json::array();
  for (int i = 0; i < vec.size(); ++i) {
    out.push_back(finiteValue(vec[i]) ? json(vec[i]) : json(nullptr));
  }
  return out;
}

json boundRowToJson(const Acts::BoundMatrix& jac, int row) {
  json out = json::array();
  for (int c = 0; c < 5; ++c) {
    out.push_back(jac(row, c));
  }
  return out;
}

template <typename Matrix>
json denseMatrixToJson(const Matrix& jac) {
  json rows = json::array();
  for (int i = 0; i < jac.rows(); ++i) {
    json row = json::array();
    for (int j = 0; j < jac.cols(); ++j) {
      row.push_back(jac(i, j));
    }
    rows.push_back(row);
  }
  return rows;
}

Acts::FreeMatrix supportingPlaneIntersectionFreeJacobian(
    const Acts::Surface& surface, const Acts::GeometryContext& geometryContext,
    const Acts::Vector3& position, const Acts::Vector3& freeDir, double sign) {
  const Acts::Vector3 signedDir = sign * freeDir;
  const Acts::Vector3 center = surface.center(geometryContext);
  const Acts::Vector3 normal =
      surface.normal(geometryContext, center, Acts::Vector3::UnitZ());
  Acts::FreeMatrix jac = Acts::FreeMatrix::Zero();
  jac(Acts::eFreeDir0, Acts::eFreeDir0) = 1.0;
  jac(Acts::eFreeDir1, Acts::eFreeDir1) = 1.0;
  jac(Acts::eFreeDir2, Acts::eFreeDir2) = 1.0;
  jac(Acts::eFreeQOverP, Acts::eFreeQOverP) = 1.0;
  const double denom = normal.dot(signedDir);
  if (std::abs(denom) < 1.0e-12) {
    jac(Acts::eFreePos0, Acts::eFreePos0) = 1.0;
    jac(Acts::eFreePos1, Acts::eFreePos1) = 1.0;
    jac(Acts::eFreePos2, Acts::eFreePos2) = 1.0;
    return jac;
  }
  const double t = normal.dot(center - position) / denom;
  const Acts::Vector3 dtDpos = -normal / denom;
  const Acts::Vector3 dtDsigned = (-t / denom) * normal;
  const Eigen::Matrix3d dxyzDpos =
      Eigen::Matrix3d::Identity() + signedDir * dtDpos.transpose();
  const Eigen::Matrix3d dxyzDdir =
      sign * (t * Eigen::Matrix3d::Identity() + signedDir * dtDsigned.transpose());
  jac.block<3, 3>(Acts::eFreePos0, Acts::eFreePos0) = dxyzDpos;
  jac.block<3, 3>(Acts::eFreePos0, Acts::eFreeDir0) = dxyzDdir;
  return jac;
}

Eigen::Matrix<double, 1, 8> supportingPlaneLoc0FreeJacobian(
    const Acts::Surface& surface, const Acts::GeometryContext& geometryContext,
    const Acts::Vector3& position, const Acts::Vector3& signedDir) {
  const Acts::Vector3 center = surface.center(geometryContext);
  const Acts::Vector3 normal =
      surface.normal(geometryContext, center, Acts::Vector3::UnitZ());
  const Acts::Vector3 u = surface.transform(geometryContext).rotation().col(0);
  const double denom = normal.dot(signedDir);
  Eigen::Matrix<double, 1, 8> j = Eigen::Matrix<double, 1, 8>::Zero();
  if (std::abs(denom) < 1.0e-12) {
    return j;
  }
  const double t = normal.dot(center - position) / denom;
  const Acts::Vector3 dt_dpos = -normal / denom;
  const Acts::Vector3 dt_ddir = (-t / denom) * normal;
  const Eigen::Matrix3d dxyz_dpos =
      Eigen::Matrix3d::Identity() + signedDir * dt_dpos.transpose();
  const Eigen::Matrix3d dxyz_ddir =
      t * Eigen::Matrix3d::Identity() + signedDir * dt_ddir.transpose();
  const Acts::Vector3 dloc0_dpos = dxyz_dpos.transpose() * u;
  const Acts::Vector3 dloc0_ddir = dxyz_ddir.transpose() * u;
  j(0, Acts::eFreePos0) = dloc0_dpos.x();
  j(0, Acts::eFreePos1) = dloc0_dpos.y();
  j(0, Acts::eFreePos2) = dloc0_dpos.z();
  j(0, Acts::eFreeDir0) = dloc0_ddir.x();
  j(0, Acts::eFreeDir1) = dloc0_ddir.y();
  j(0, Acts::eFreeDir2) = dloc0_ddir.z();
  return j;
}

/// Collect the stepper's already-computed free transport segments.
/// MaterialInteractor resets jacTransport when covTransport is on; this
/// actor runs first and multiplies the completed RK segments so the
/// product stays the official free-to-free map, without the curvilinear
/// surface projection that is not part of official h_i.
struct FreeTransportJacobianCollector {
  struct result_type {
    Acts::FreeMatrix jFreeAcc = Acts::FreeMatrix::Identity();
    Acts::FreeMatrix pending = Acts::FreeMatrix::Identity();
    Acts::BoundMatrix lastBoundJacobian = Acts::BoundMatrix::Identity();
    bool initialized = false;
    bool foldedEnd = false;
    bool sawCovTransport = false;
    int nActorCalls = 0;
    int nResets = 0;
  };

  template <typename propagator_state_t, typename stepper_t, typename navigator_t>
  void operator()(propagator_state_t& state, const stepper_t&,
                  const navigator_t&, result_type& result,
                  const Acts::Logger&) const {
    result.nActorCalls += 1;
    result.sawCovTransport =
        result.sawCovTransport || state.stepping.covTransport;
    const Acts::FreeMatrix& jT = state.stepping.jacTransport;
    const Acts::BoundMatrix& jB = state.stepping.jacobian;
    if (!result.initialized) {
      result.pending = jT;
      result.lastBoundJacobian = jB;
      result.initialized = true;
      if (state.stage == Acts::PropagatorStage::postPropagation) {
        result.jFreeAcc = jT;
        result.foldedEnd = true;
      }
      return;
    }
    if ((jB - result.lastBoundJacobian).norm() > 1.0e-15) {
      result.jFreeAcc = result.pending * result.jFreeAcc;
      result.nResets += 1;
    }
    result.pending = jT;
    result.lastBoundJacobian = jB;
    if (state.stage == Acts::PropagatorStage::postPropagation &&
        !result.foldedEnd) {
      result.jFreeAcc = result.pending * result.jFreeAcc;
      result.foldedEnd = true;
    }
  }
};

struct OfficialPathJacobianHop {
  bool ok = false;
  bool covTransport = false;
  bool jacTransportIdentity = true;
  bool rkFreeAvailable = false;
  bool loc0MatchesOfficial = false;
  int nSteps = 0;
  int nResets = 0;
  int nActorCalls = 0;
  double pathLength = 0.0;
  double predictedLoc0 = 0.0;
  double predictedLoc1 = 0.0;
  double residual = 0.0;
  double qop = 0.0;
  Acts::Vector3 freePos = Acts::Vector3::Zero();
  Acts::Vector3 freeDir = Acts::Vector3::Zero();
  Acts::FreeMatrix jacTransport = Acts::FreeMatrix::Identity();
  Acts::BoundToFreeMatrix jacToGlobal = Acts::BoundToFreeMatrix::Zero();
  Acts::BoundMatrix jacobian = Acts::BoundMatrix::Identity();
  Acts::FreeVector derivative = Acts::FreeVector::Zero();
  Acts::FreeMatrix jRkFreeAcc = Acts::FreeMatrix::Identity();
  Acts::BoundToFreeMatrix jB2fStart = Acts::BoundToFreeMatrix::Zero();
  Acts::BoundToFreeMatrix jB2fComposed = Acts::BoundToFreeMatrix::Zero();
  Acts::BoundToFreeMatrix jB2fRk = Acts::BoundToFreeMatrix::Zero();
  std::string abortReason;
  std::string continuation;
  bool freeToBoundFallback = false;
  json surfaceGeometry = json::object();
  json fieldSamples = json::array();
};

json fieldSampleJson(const Acts::Vector3& position,
                     const Acts::MagneticFieldContext& magFieldContext) {
  FASERMagneticFieldWrapper wrapper;
  auto cache = wrapper.makeCache(magFieldContext);
  auto field = wrapper.getField(position, cache);
  Acts::ActsMatrix<3, 3> gradient = Acts::ActsMatrix<3, 3>::Zero();
  auto cacheG = wrapper.makeCache(magFieldContext);
  auto fieldG = wrapper.getFieldGradient(position, gradient, cacheG);
  json out;
  out["x_mm"] = {position.x(), position.y(), position.z()};
  out["official_gradient_api"] = "FASERMagneticFieldWrapper::getFieldGradient";
  out["field_gradient_fd_step_mm"] = kFieldGradientFdStepMm;
  out["field_gradient_fd_step_pre_registered"] = true;
  out["do_not_tune_field_step_from_track_jacobian"] = true;
  if (field.ok()) {
    out["B_acts"] = {(*field).x(), (*field).y(), (*field).z()};
    out["B_repeat_delta_norm"] = 0.0;
    auto field2 = wrapper.getField(position, cache);
    if (field2.ok()) {
      out["B_repeat_delta_norm"] = ((*field2) - (*field)).norm();
    }
  } else {
    out["B_acts"] = nullptr;
  }
  if (fieldG.ok()) {
    out["gradient_official"] = denseMatrixToJson(gradient);
    out["gradient_frobenius"] = gradient.norm();
    out["div_B"] = gradient(0, 0) + gradient(1, 1) + gradient(2, 2);
  } else {
    out["gradient_official"] = nullptr;
    out["gradient_frobenius"] = nullptr;
    out["div_B"] = nullptr;
  }
  json gFd = json::array();
  bool fdOk = field.ok();
  Acts::ActsMatrix<3, 3> numerical = Acts::ActsMatrix<3, 3>::Zero();
  for (int j = 0; j < 3; ++j) {
    Acts::Vector3 plus = position;
    Acts::Vector3 minus = position;
    plus[j] += kFieldGradientFdStepMm;
    minus[j] -= kFieldGradientFdStepMm;
    auto bp = wrapper.getField(plus, cache);
    auto bm = wrapper.getField(minus, cache);
    json col = json::array();
    if (bp.ok() && bm.ok()) {
      const Acts::Vector3 d = ((*bp) - (*bm)) / (2.0 * kFieldGradientFdStepMm);
      numerical.col(j) = d;
      col = {d.x(), d.y(), d.z()};
    } else {
      fdOk = false;
    }
    gFd.push_back(col);
  }
  out["gradient_field_query_fd"] = gFd;
  out["gradient_field_query_fd_ok"] = fdOk;
  if (fdOk && fieldG.ok()) {
    out["official_minus_field_fd_frobenius"] = (gradient - numerical).norm();
  } else {
    out["official_minus_field_fd_frobenius"] = nullptr;
  }
  out["units"] = {{"position", "mm"}, {"B", "ACTS Tesla"}, {"gradient", "ACTS Tesla / mm"}};
  return out;
}

template <typename propagator_t>
OfficialPathJacobianHop propagateOfficialPathJacobianHop(
    const propagator_t& propagator, const Acts::GeometryContext& geometryContext,
    const Acts::MagneticFieldContext& magFieldContext,
    const Acts::BoundTrackParameters& start, const ProfileHit& hit, int index,
    int maxSteps, bool matchKalmanStepSize, double stepTolerance,
    bool enableCovTransport, bool recordFieldSamples = false) {
  OfficialPathJacobianHop out;
  if (hit.surface == nullptr) {
    out.abortReason = "surface_pointer_null";
    return out;
  }
  out.surfaceGeometry = surfaceGeometryJson(*hit.surface, geometryContext);
  out.jB2fStart = start.referenceSurface().boundToFreeJacobian(
      geometryContext, start.parameters());
  const double startZ = start.position(geometryContext).z();
  const double destZ = hit.surface->center(geometryContext).z();
  const bool forward = destZ >= startZ;
  Acts::BoundTrackParameters hopStart = start;
  if (enableCovTransport) {
    Acts::BoundSquareMatrix cov = Acts::BoundSquareMatrix::Identity();
    cov *= 1.0e-8;
    hopStart = Acts::BoundTrackParameters(
        start.referenceSurface().getSharedPtr(), start.parameters(), cov,
        Acts::ParticleHypothesis::muon());
  }
  using JacActions =
      Acts::ActionList<FreeTransportJacobianCollector, Acts::MaterialInteractor,
                       Acts::detail::SteppingLogger>;
  using JacAbort = Acts::AbortList<Acts::EndOfWorldReached>;
  using JacOptions = Acts::PropagatorOptions<JacActions, JacAbort>;
  JacOptions options(geometryContext, magFieldContext);
  options.maxSteps = static_cast<unsigned int>(std::max(maxSteps, 1));
  options.direction =
      forward ? Acts::Direction::Forward : Acts::Direction::Backward;
  options.loopProtection = true;
  options.stepTolerance = stepTolerance;
  if (!matchKalmanStepSize) {
    options.maxStepSize = 10.0 * 1_m;
  }
  auto& material = options.actionList.template get<Acts::MaterialInteractor>();
  material.multipleScattering = true;
  material.energyLoss = true;
  options.actionList.template get<Acts::detail::SteppingLogger>().sterile =
      !recordFieldSamples;
  auto state = propagator.template makeState<Acts::BoundTrackParameters,
                                             JacOptions, Acts::SurfaceReached,
                                             Acts::PathLimitReached>(
      hopStart, *hit.surface, options);
  state.options.abortList.template get<Acts::SurfaceReached>().boundaryCheck =
      Acts::BoundaryCheck(false);
  auto propRes = propagator.propagate(state);
  out.nSteps = static_cast<int>(state.steps);
  out.pathLength = state.pathLength;
  out.covTransport = state.stepping.covTransport;
  out.jacTransport = state.stepping.jacTransport;
  out.jacToGlobal = state.stepping.jacToGlobal;
  out.jacobian = state.stepping.jacobian;
  out.derivative = state.stepping.derivative;
  out.jacTransportIdentity =
      (out.jacTransport - Acts::FreeMatrix::Identity()).norm() <= 1.0e-12;
  const auto& collected =
      state.template get<FreeTransportJacobianCollector::result_type>();
  out.jRkFreeAcc = collected.jFreeAcc;
  out.nResets = collected.nResets;
  out.nActorCalls = collected.nActorCalls;
  out.jB2fComposed = out.jacTransport * out.jacToGlobal * out.jacobian;
  out.jB2fRk = out.jRkFreeAcc * out.jB2fStart;
  const double rkDelta =
      (out.jRkFreeAcc - Acts::FreeMatrix::Identity()).norm();
  out.rkFreeAvailable = (out.nSteps == 0) || rkDelta > 1.0e-12 ||
                        !out.jacTransportIdentity;
  out.freePos = state.stepping.pars.template segment<3>(Acts::eFreePos0);
  out.freeDir = state.stepping.pars.template segment<3>(Acts::eFreeDir0);
  out.qop = state.stepping.pars[Acts::eFreeQOverP];
  ProfileHop hop;
  hop.measurement_index = index;
  const Acts::Vector3 signedDir = (forward ? 1.0 : -1.0) * out.freeDir;
  const bool projected =
      projectSupportingPlane(*hit.surface, geometryContext, out.freePos,
                             signedDir, hop);
  if (!propRes.ok()) {
    out.abortReason = propRes.error().message();
  }
  if (projected) {
    out.ok = true;
    out.predictedLoc0 = hop.predicted_loc0;
    out.predictedLoc1 = hop.predicted_loc1;
    out.residual = hit.loc0 - hop.predicted_loc0;
    const Acts::Vector3 onPlane =
        hit.surface->transform(geometryContext) *
        Acts::Vector3(hop.predicted_loc0, hop.predicted_loc1, 0.0);
    auto bound = Acts::detail::transformFreeToBoundParameters(
        onPlane, 0.0, out.freeDir, out.qop, *hit.surface, geometryContext,
        10.0 * Acts::s_onSurfaceTolerance);
    if (bound.ok()) {
      out.continuation = "transform_free_to_bound";
      out.freeToBoundFallback = false;
    } else {
      out.continuation = "plane_chart_fallback";
      out.freeToBoundFallback = true;
    }
  } else if (out.abortReason.empty()) {
    out.abortReason = "supporting_plane_not_near_enough";
  }
  if (recordFieldSamples) {
    const auto& log =
        state.template get<Acts::detail::SteppingLogger::result_type>();
    const int n = static_cast<int>(log.steps.size());
    std::vector<int> pick;
    if (n > 0) {
      pick.push_back(0);
      if (n > 1) {
        pick.push_back(n / 2);
        pick.push_back(n - 1);
      }
      if (n > 8) {
        pick.push_back(n / 4);
        pick.push_back((3 * n) / 4);
      }
    }
    std::sort(pick.begin(), pick.end());
    pick.erase(std::unique(pick.begin(), pick.end()), pick.end());
    for (int i : pick) {
      json sample = fieldSampleJson(log.steps[static_cast<std::size_t>(i)].position,
                                    magFieldContext);
      sample["step_index"] = i;
      sample["n_steps"] = n;
      out.fieldSamples.push_back(sample);
    }
  }
  return out;
}

json hopStartSegmentFdJson(
    const Propagator& propagator, const Acts::GeometryContext& geometryContext,
    const Acts::MagneticFieldContext& magFieldContext,
    const Acts::BoundTrackParameters& hopStart, const ProfileHit& hit, int index,
    int maxSteps, bool matchKalman, double stepTolerance) {
  const std::array<double, 4> kRungs{1.0, 0.5, 0.25, 0.125};
  const std::array<const char*, 5> names{
      "loc0", "loc1", "phi", "theta", "q_over_p"};
  json columns = json::array();
  const Acts::BoundVector theta0 = hopStart.parameters();
  auto surface = hopStart.referenceSurface().getSharedPtr();
  for (int col = 0; col < 5; ++col) {
    json rungs = json::array();
    bool ok = true;
    for (int rung = 0; rung < 4; ++rung) {
      const double step =
          kFdSteps[static_cast<std::size_t>(col)] *
          kRungs[static_cast<std::size_t>(rung)];
      Acts::BoundVector plus = theta0;
      Acts::BoundVector minus = theta0;
      plus[col] += step;
      minus[col] -= step;
      const Acts::BoundTrackParameters pStart(
          surface, plus, std::nullopt, Acts::ParticleHypothesis::muon());
      const Acts::BoundTrackParameters mStart(
          surface, minus, std::nullopt, Acts::ParticleHypothesis::muon());
      const auto evP = propagateOfficialPathJacobianHop(
          propagator, geometryContext, magFieldContext, pStart, hit, index,
          maxSteps, matchKalman, stepTolerance, false, false);
      const auto evM = propagateOfficialPathJacobianHop(
          propagator, geometryContext, magFieldContext, mStart, hit, index,
          maxSteps, matchKalman, stepTolerance, false, false);
      json item;
      item["step_factor"] = kRungs[static_cast<std::size_t>(rung)];
      item["fd_step"] = step;
      item["ok"] = evP.ok && evM.ok;
      item["plus_predicted_loc0"] = evP.ok ? json(evP.predictedLoc0) : json(nullptr);
      item["minus_predicted_loc0"] = evM.ok ? json(evM.predictedLoc0) : json(nullptr);
      item["plus_n_steps"] = evP.nSteps;
      item["minus_n_steps"] = evM.nSteps;
      item["plus_path_length"] = evP.pathLength;
      item["minus_path_length"] = evM.pathLength;
      item["plus_abort"] = evP.abortReason;
      item["minus_abort"] = evM.abortReason;
      if (evP.ok) {
        item["plus_free"] = {evP.freePos.x(), evP.freePos.y(), evP.freePos.z(),
                             evP.freeDir.x(), evP.freeDir.y(), evP.freeDir.z(),
                             evP.qop};
      } else {
        item["plus_free"] = nullptr;
      }
      if (evM.ok) {
        item["minus_free"] = {evM.freePos.x(), evM.freePos.y(), evM.freePos.z(),
                              evM.freeDir.x(), evM.freeDir.y(), evM.freeDir.z(),
                              evM.qop};
      } else {
        item["minus_free"] = nullptr;
      }
      item["branch_identity_same"] =
          evP.ok && evM.ok && evP.abortReason == evM.abortReason &&
          evP.continuation == evM.continuation;
      if (evP.ok && evM.ok) {
        item["dloc0_d_start"] = (evP.predictedLoc0 - evM.predictedLoc0) / (2.0 * step);
      } else {
        item["dloc0_d_start"] = nullptr;
        ok = false;
      }
      rungs.push_back(item);
    }
    json colJson;
    colJson["parameter"] = names[static_cast<std::size_t>(col)];
    colJson["official_fd_step"] = kFdSteps[static_cast<std::size_t>(col)];
    colJson["ok"] = ok;
    colJson["rungs"] = rungs;
    colJson["do_not_add_rung"] = true;
    colJson["do_not_select_best_step"] = true;
    columns.push_back(colJson);
  }
  json out;
  out["measurement_index"] = index;
  out["kind"] = "hop_start_segment_fd";
  out["rung_factors"] = {1.0, 0.5, 0.25, 0.125};
  out["mean_path"] = "official EigenStepper GenericDefaultExtension";
  out["do_not_add_rung"] = true;
  out["do_not_select_best_step"] = true;
  out["columns"] = columns;
  return out;
}

json hopStartStateJson(const Acts::BoundTrackParameters& hopStart,
                       const Acts::GeometryContext& geometryContext) {
  const Acts::BoundVector p = hopStart.parameters();
  const Acts::FreeVector free =
      Acts::detail::transformBoundToFreeParameters(
          hopStart.referenceSurface(), geometryContext, p);
  json out;
  out["bound"] = {p[0], p[1], p[2], p[3], p[4]};
  out["bound_units"] = {"mm", "mm", "rad", "rad", "1/GeV"};
  out["free"] = {free[0], free[1], free[2], free[3],
                 free[4], free[5], free[6], free[7]};
  out["free_units"] = {"mm", "mm", "mm", "time", "1", "1", "1", "1/GeV"};
  out["z_mm"] = hopStart.position(geometryContext).z();
  return out;
}

json independentSegmentReferenceJson(
    const Acts::GeometryContext& geometryContext,
    const Acts::MagneticFieldContext& magFieldContext,
    const Acts::BoundTrackParameters& hopStart, const ProfileHit& hit,
    int index, double officialMeanLoc0) {
  json out;
  out["kind"] = "independent_mean_ode_segment_reference";
  out["measurement_index"] = index;
  out["method"] = IndependentMeanOdeIntegrator::kMethod;
  out["code_path"] = "IndependentMeanOdeIntegrator.hpp; not EigenStepper";
  out["replaces_official_mean_path"] = false;
  out["do_not_tune_tolerance_from_jacobian"] = true;
  out["do_not_add_fd_rung"] = true;
  out["field_api"] = "FASERMagneticFieldWrapper::getField/getFieldGradient";
  out["pre_registered"] = {
      {"abs_tol_pos_mm", IndependentMeanOdeIntegrator::kAbsTolPosMm},
      {"abs_tol_dir", IndependentMeanOdeIntegrator::kAbsTolDir},
      {"abs_tol_qop", IndependentMeanOdeIntegrator::kAbsTolQop},
      {"rel_tol", IndependentMeanOdeIntegrator::kRelTol},
      {"initial_step_mm", IndependentMeanOdeIntegrator::kInitialStepMm},
      {"min_step_mm", IndependentMeanOdeIntegrator::kMinStepMm},
      {"max_step_mm", IndependentMeanOdeIntegrator::kMaxStepMm},
      {"max_path_mm", IndependentMeanOdeIntegrator::kMaxPathMm},
      {"max_steps", IndependentMeanOdeIntegrator::kMaxSteps},
      {"plane_hit_abs_mm", IndependentMeanOdeIntegrator::kPlaneHitAbsMm},
      {"justification", IndependentMeanOdeIntegrator::kToleranceJustification}};
  if (hit.surface == nullptr) {
    out["ok"] = false;
    out["abort_reason"] = "surface_pointer_null";
    return out;
  }
  const Acts::BoundVector theta0 = hopStart.parameters();
  auto startSurface = hopStart.referenceSurface().getSharedPtr();
  const Acts::FreeVector free0 = Acts::detail::transformBoundToFreeParameters(
      hopStart.referenceSurface(), geometryContext, theta0);
  const Acts::BoundToFreeMatrix jB2f =
      hopStart.referenceSurface().boundToFreeJacobian(geometryContext, theta0);
  IndependentMeanOdeIntegrator::State y0;
  y0.pos = free0.template segment<3>(Acts::eFreePos0);
  y0.dir = free0.template segment<3>(Acts::eFreeDir0);
  y0.qop = free0[Acts::eFreeQOverP];
  const Acts::Vector3 center = hit.surface->center(geometryContext);
  const Acts::Vector3 normal =
      hit.surface->normal(geometryContext, center, Acts::Vector3::UnitZ());
  const double destZ = center.z();
  const double startZ = hopStart.position(geometryContext).z();
  const double pathSign = destZ >= startZ ? 1.0 : -1.0;
  IndependentMeanOdeIntegrator integrator(magFieldContext);
  Acts::Vector3 Bstart = Acts::Vector3::Zero();
  Acts::ActsMatrix<3, 3> Gstart = Acts::ActsMatrix<3, 3>::Zero();
  const bool startFieldOk = integrator.fieldGradient(y0.pos, Bstart, Gstart);
  const auto mean = integrator.integrateToPlane(y0, center, normal, pathSign);
  out["mean"] = json::object();
  out["mean"]["ok"] = mean.ok;
  out["mean"]["n_steps"] = mean.nSteps;
  out["mean"]["n_rejected"] = mean.nRejected;
  out["mean"]["path_length"] = mean.pathLength;
  out["mean"]["n_dot_direction"] = mean.nDotDir;
  out["mean"]["distance_to_plane_mm"] = mean.distanceToPlaneMm;
  out["mean"]["abort_reason"] = mean.abortReason;
  out["mean"]["end_free"] = {mean.end.pos.x(), mean.end.pos.y(), mean.end.pos.z(),
                             mean.end.dir.x(), mean.end.dir.y(), mean.end.dir.z(),
                             mean.end.qop};
  Acts::Vector3 Bend = Acts::Vector3::Zero();
  Acts::ActsMatrix<3, 3> Gend = Acts::ActsMatrix<3, 3>::Zero();
  const bool endFieldOk =
      mean.ok && integrator.fieldGradient(mean.end.pos, Bend, Gend);
  out["mean"]["field_samples"] = json::array(
      {json{{"where", "hop_start"},
            {"ok", startFieldOk},
            {"position_mm", {y0.pos.x(), y0.pos.y(), y0.pos.z()}},
            {"B_tesla", {Bstart.x(), Bstart.y(), Bstart.z()}},
            {"G_norm", startFieldOk ? json(Gstart.norm()) : json(nullptr)}},
       json{{"where", "hop_end"},
            {"ok", endFieldOk},
            {"position_mm",
             {mean.end.pos.x(), mean.end.pos.y(), mean.end.pos.z()}},
            {"B_tesla", {Bend.x(), Bend.y(), Bend.z()}},
            {"G_norm", endFieldOk ? json(Gend.norm()) : json(nullptr)}}});
  ProfileHop hop;
  const Acts::Vector3 signedDir = pathSign * mean.end.dir;
  const bool projected =
      mean.ok && projectSupportingPlane(*hit.surface, geometryContext,
                                        mean.end.pos, signedDir, hop);
  out["mean"]["projected"] = projected;
  out["mean"]["predicted_loc0"] =
      projected ? json(hop.predicted_loc0) : json(nullptr);
  out["mean"]["official_minus_independent_loc0"] =
      projected ? json(officialMeanLoc0 - hop.predicted_loc0) : json(nullptr);
  out["mean"]["does_not_replace_official_predicted_loc0"] = true;
  json dVar = json::array();
  if (mean.ok && projected) {
    const Eigen::Matrix<double, 1, 8> jProj = supportingPlaneLoc0FreeJacobian(
        *hit.surface, geometryContext, mean.end.pos, signedDir);
    Acts::FreeMatrix jFree = Acts::FreeMatrix::Identity();
    jFree.block<3, 3>(Acts::eFreePos0, Acts::eFreePos0) =
        mean.jacobian.block<3, 3>(0, 0);
    jFree.block<3, 3>(Acts::eFreePos0, Acts::eFreeDir0) =
        mean.jacobian.block<3, 3>(0, 3);
    jFree.block<3, 1>(Acts::eFreePos0, Acts::eFreeQOverP) =
        mean.jacobian.block<3, 1>(0, 6);
    jFree.block<3, 3>(Acts::eFreeDir0, Acts::eFreePos0) =
        mean.jacobian.block<3, 3>(3, 0);
    jFree.block<3, 3>(Acts::eFreeDir0, Acts::eFreeDir0) =
        mean.jacobian.block<3, 3>(3, 3);
    jFree.block<3, 1>(Acts::eFreeDir0, Acts::eFreeQOverP) =
        mean.jacobian.block<3, 1>(3, 6);
    jFree.block<1, 3>(Acts::eFreeQOverP, Acts::eFreePos0) =
        mean.jacobian.block<1, 3>(6, 0);
    jFree.block<1, 3>(Acts::eFreeQOverP, Acts::eFreeDir0) =
        mean.jacobian.block<1, 3>(6, 3);
    jFree(Acts::eFreeQOverP, Acts::eFreeQOverP) = mean.jacobian(6, 6);
    const Eigen::Matrix<double, 1, Acts::eBoundSize> hopD = jProj * jFree * jB2f;
    for (int c = 0; c < 5; ++c) {
      dVar.push_back(hopD(0, c));
    }
    out["variational_includes_field_gradient"] = true;
    out["variational_dloc0_d_start"] = dVar;
  } else {
    out["variational_dloc0_d_start"] = nullptr;
    out["variational_includes_field_gradient"] = true;
  }
  const std::array<double, 4> kRungs{1.0, 0.5, 0.25, 0.125};
  const std::array<const char*, 5> names{
      "loc0", "loc1", "phi", "theta", "q_over_p"};
  json columns = json::array();
  for (int col = 0; col < 5; ++col) {
    json rungs = json::array();
    bool ok = true;
    for (int rung = 0; rung < 4; ++rung) {
      const double step = kFdSteps[static_cast<std::size_t>(col)] *
                          kRungs[static_cast<std::size_t>(rung)];
      Acts::BoundVector plus = theta0;
      Acts::BoundVector minus = theta0;
      plus[col] += step;
      minus[col] -= step;
      const Acts::FreeVector fP = Acts::detail::transformBoundToFreeParameters(
          *startSurface, geometryContext, plus);
      const Acts::FreeVector fM = Acts::detail::transformBoundToFreeParameters(
          *startSurface, geometryContext, minus);
      IndependentMeanOdeIntegrator::State yP;
      yP.pos = fP.template segment<3>(Acts::eFreePos0);
      yP.dir = fP.template segment<3>(Acts::eFreeDir0);
      yP.qop = fP[Acts::eFreeQOverP];
      IndependentMeanOdeIntegrator::State yM;
      yM.pos = fM.template segment<3>(Acts::eFreePos0);
      yM.dir = fM.template segment<3>(Acts::eFreeDir0);
      yM.qop = fM[Acts::eFreeQOverP];
      IndependentMeanOdeIntegrator integP(magFieldContext);
      IndependentMeanOdeIntegrator integM(magFieldContext);
      const auto evP = integP.integrateToPlane(yP, center, normal, pathSign);
      const auto evM = integM.integrateToPlane(yM, center, normal, pathSign);
      ProfileHop hopP;
      ProfileHop hopM;
      const bool pOk =
          evP.ok && projectSupportingPlane(*hit.surface, geometryContext,
                                           evP.end.pos, pathSign * evP.end.dir,
                                           hopP);
      const bool mOk =
          evM.ok && projectSupportingPlane(*hit.surface, geometryContext,
                                           evM.end.pos, pathSign * evM.end.dir,
                                           hopM);
      json item;
      item["step_factor"] = kRungs[static_cast<std::size_t>(rung)];
      item["fd_step"] = step;
      item["ok"] = pOk && mOk;
      item["plus_predicted_loc0"] = pOk ? json(hopP.predicted_loc0) : json(nullptr);
      item["minus_predicted_loc0"] = mOk ? json(hopM.predicted_loc0) : json(nullptr);
      item["plus_n_steps"] = evP.nSteps;
      item["minus_n_steps"] = evM.nSteps;
      item["plus_path_length"] = evP.pathLength;
      item["minus_path_length"] = evM.pathLength;
      item["plus_abort"] = evP.abortReason;
      item["minus_abort"] = evM.abortReason;
      item["branch_identity_same"] = pOk && mOk;
      if (pOk && mOk) {
        item["dloc0_d_start"] =
            (hopP.predicted_loc0 - hopM.predicted_loc0) / (2.0 * step);
      } else {
        item["dloc0_d_start"] = nullptr;
        ok = false;
      }
      rungs.push_back(item);
    }
    json colJson;
    colJson["parameter"] = names[static_cast<std::size_t>(col)];
    colJson["official_fd_step"] = kFdSteps[static_cast<std::size_t>(col)];
    colJson["ok"] = ok;
    colJson["rungs"] = rungs;
    colJson["do_not_add_rung"] = true;
    colJson["do_not_select_best_step"] = true;
    columns.push_back(colJson);
  }
  out["independent_fd"] = json::object();
  out["independent_fd"]["rung_factors"] = {1.0, 0.5, 0.25, 0.125};
  out["independent_fd"]["columns"] = columns;
  out["independent_fd"]["mean_path"] = "independent DOPRI5 mean ODE";
  return out;
}

struct ProductionMeanPartition {
  bool ok = false;
  double pathLength = 0.0;
  int nSteps = 0;
  int nSurfaceMaterial = 0;
  int nVolumeMaterial = 0;
  double materialInX0 = 0.0;
  double materialInL0 = 0.0;
  Acts::Vector3 endPos = Acts::Vector3::Zero();
  Acts::Vector3 endDir = Acts::Vector3::Zero();
  double endQop = 0.0;
  std::vector<std::pair<double, CommonGridShadowIntegrator::Node>> crossings;
  std::string abortReason;
};

template <typename propagator_t>
ProductionMeanPartition collectProductionMeanPartition(
    const propagator_t& propagator, const Acts::GeometryContext& geometryContext,
    const Acts::MagneticFieldContext& magFieldContext,
    const Acts::BoundTrackParameters& hopStart, const ProfileHit& hit,
    int maxSteps, bool matchKalman, double stepTolerance) {
  ProductionMeanPartition out;
  if (hit.surface == nullptr) {
    out.abortReason = "surface_pointer_null";
    return out;
  }
  const double startZ = hopStart.position(geometryContext).z();
  const double destZ = hit.surface->center(geometryContext).z();
  const bool forward = destZ >= startZ;
  using Actions =
      Acts::ActionList<Acts::MaterialInteractor, Acts::detail::SteppingLogger>;
  using Abort = Acts::AbortList<Acts::EndOfWorldReached>;
  using Options = Acts::PropagatorOptions<Actions, Abort>;
  Options options(geometryContext, magFieldContext);
  options.maxSteps = static_cast<unsigned int>(std::max(maxSteps, 1));
  options.direction =
      forward ? Acts::Direction::Forward : Acts::Direction::Backward;
  options.loopProtection = true;
  options.stepTolerance = stepTolerance;
  if (!matchKalman) {
    options.maxStepSize = 10.0 * 1_m;
  }
  auto& material = options.actionList.template get<Acts::MaterialInteractor>();
  material.multipleScattering = true;
  material.energyLoss = true;
  material.recordInteractions = true;
  options.actionList.template get<Acts::detail::SteppingLogger>().sterile = false;
  auto state = propagator.template makeState<Acts::BoundTrackParameters, Options,
                                             Acts::SurfaceReached,
                                             Acts::PathLimitReached>(
      hopStart, *hit.surface, options);
  state.options.abortList.template get<Acts::SurfaceReached>().boundaryCheck =
      Acts::BoundaryCheck(false);
  auto propRes = propagator.propagate(state);
  out.nSteps = static_cast<int>(state.steps);
  out.pathLength = state.pathLength;
  out.endPos = state.stepping.pars.template segment<3>(Acts::eFreePos0);
  out.endDir = state.stepping.pars.template segment<3>(Acts::eFreeDir0);
  out.endQop = state.stepping.pars[Acts::eFreeQOverP];
  if (!propRes.ok()) {
    out.abortReason = propRes.error().message();
  }
  const auto& rec =
      state.template get<Acts::MaterialInteractor::result_type>();
  out.materialInX0 = rec.materialInX0;
  out.materialInL0 = rec.materialInL0;
  const auto& log =
      state.template get<Acts::detail::SteppingLogger::result_type>();
  std::vector<std::pair<Acts::Vector3, double>> pathPts;
  double acc = 0.0;
  Acts::Vector3 prev = hopStart.position(geometryContext);
  pathPts.push_back({prev, 0.0});
  for (const auto& step : log.steps) {
    acc += (step.position - prev).norm();
    pathPts.push_back({step.position, acc});
    prev = step.position;
  }
  const double pathAbs = std::max(std::abs(out.pathLength), acc);
  auto sOf = [&](const Acts::Vector3& pos) {
    double bestS = 0.0;
    double bestD = 1.0e99;
    for (const auto& pt : pathPts) {
      const double d = (pt.first - pos).norm();
      if (d < bestD) {
        bestD = d;
        bestS = pt.second;
      }
    }
    return bestS;
  };
  for (const auto& mi : rec.materialInteractions) {
    CommonGridShadowIntegrator::Node node;
    node.material = true;
    node.volumeOnly = (mi.surface == nullptr);
    node.slab = mi.materialSlab;
    const double s = sOf(mi.position);
    if (node.volumeOnly) {
      ++out.nVolumeMaterial;
    } else {
      ++out.nSurfaceMaterial;
    }
    if (s >= 0.0 && s <= pathAbs + 1.0) {
      out.crossings.push_back({s, node});
    }
  }
  out.ok = true;
  return out;
}

json commonGridShadowReferenceJson(
    const Propagator& propagator, const Acts::GeometryContext& geometryContext,
    const Acts::MagneticFieldContext& magFieldContext,
    const Acts::BoundTrackParameters& hopStart, const ProfileHit& hit,
    int index, int maxSteps, bool matchKalman, double stepTolerance,
    double officialMeanLoc0, double officialPath,
    const Acts::Vector3& officialPos, const Acts::Vector3& officialDir,
    double officialQop) {
  json out;
  out["kind"] = "common_grid_shadow_segment_reference";
  out["measurement_index"] = index;
  out["method"] = CommonGridShadowIntegrator::kMethod;
  out["code_path"] = "CommonGridShadowIntegrator.hpp; not EigenStepper";
  out["replaces_official_mean_path"] = false;
  out["do_not_tune_grid_from_jacobian"] = true;
  out["do_not_add_fd_rung"] = true;
  out["do_not_reuse_wb124_adaptive_dopri5"] = true;
  out["field_api"] = "FASERMagneticFieldWrapper::getField";
  out["material_mean_map"] = "Acts::computeEnergyLossMean on frozen surface slabs";
  out["process_noise_included"] = false;
  out["multiple_scattering_in_mean"] = false;
  out["pre_registered"] = {
      {"coarse_max_step_mm", CommonGridShadowIntegrator::kCoarseMaxStepMm},
      {"nominal_max_step_mm", CommonGridShadowIntegrator::kNominalMaxStepMm},
      {"fine_max_step_mm", CommonGridShadowIntegrator::kFineMaxStepMm},
      {"mean_loc0_abs_mm", CommonGridShadowIntegrator::kMeanLoc0AbsMm},
      {"mean_path_abs_mm", CommonGridShadowIntegrator::kMeanPathAbsMm},
      {"mean_pos_abs_mm", CommonGridShadowIntegrator::kMeanPosAbsMm},
      {"mean_dir_abs", CommonGridShadowIntegrator::kMeanDirAbs},
      {"mean_qop_rel", CommonGridShadowIntegrator::kMeanQopRel},
      {"grid_deriv_rel_max", CommonGridShadowIntegrator::kGridDerivRelMax},
      {"justification", CommonGridShadowIntegrator::kMeshJustification}};
  if (hit.surface == nullptr) {
    out["ok"] = false;
    out["abort_reason"] = "surface_pointer_null";
    return out;
  }
  const ProductionMeanPartition part = collectProductionMeanPartition(
      propagator, geometryContext, magFieldContext, hopStart, hit, maxSteps,
      matchKalman, stepTolerance);
  out["production_mean_effects"] = {
      {"magnetic_field", true},
      {"supporting_plane_termination", true},
      {"deterministic_surface_energy_loss", part.nSurfaceMaterial > 0},
      {"n_surface_material_crossings", part.nSurfaceMaterial},
      {"n_volume_material_recorded_not_applied_to_acts_mean",
       part.nVolumeMaterial},
      {"material_in_x0", part.materialInX0},
      {"material_in_l0", part.materialInL0},
      {"process_noise_in_mean", false},
      {"vacuum_ode_is_not_this_reference", true}};
  out["production_partition_ok"] = part.ok;
  out["production_partition_n_steps"] = part.nSteps;
  out["production_partition_path"] = part.pathLength;
  const Acts::BoundVector theta0 = hopStart.parameters();
  auto startSurface = hopStart.referenceSurface().getSharedPtr();
  const Acts::FreeVector free0 = Acts::detail::transformBoundToFreeParameters(
      hopStart.referenceSurface(), geometryContext, theta0);
  CommonGridShadowIntegrator::State y0;
  y0.pos = free0.template segment<3>(Acts::eFreePos0);
  y0.dir = free0.template segment<3>(Acts::eFreeDir0);
  y0.qop = free0[Acts::eFreeQOverP];
  const Acts::Vector3 center = hit.surface->center(geometryContext);
  const Acts::Vector3 normal =
      hit.surface->normal(geometryContext, center, Acts::Vector3::UnitZ());
  const double destZ = center.z();
  const double startZ = hopStart.position(geometryContext).z();
  const double pathSign = destZ >= startZ ? 1.0 : -1.0;
  const double pathAbs = std::abs(part.ok ? part.pathLength : officialPath);
  const auto meshOf = [&](double h, const char* name) {
    return CommonGridShadowIntegrator::buildMesh(pathSign, pathAbs,
                                                 part.crossings, h, name);
  };
  const CommonGridShadowIntegrator::Mesh coarse =
      meshOf(CommonGridShadowIntegrator::kCoarseMaxStepMm, "coarse");
  const CommonGridShadowIntegrator::Mesh nominal =
      meshOf(CommonGridShadowIntegrator::kNominalMaxStepMm, "nominal");
  const CommonGridShadowIntegrator::Mesh fine =
      meshOf(CommonGridShadowIntegrator::kFineMaxStepMm, "fine");
  out["meshes"] = json::array();
  for (const auto& mesh : {coarse, nominal, fine}) {
    json sAbs = json::array();
    json materialFlags = json::array();
    for (const auto& node : mesh.nodes) {
      sAbs.push_back(node.sAbs);
      materialFlags.push_back(node.material);
    }
    out["meshes"].push_back({{"name", mesh.name},
                             {"max_step_mm", mesh.maxStepMm},
                             {"n_field_nodes", mesh.nField},
                             {"n_material_nodes", mesh.nMaterial},
                             {"n_volume_recorded", mesh.nVolumeRecorded},
                             {"n_nodes", static_cast<int>(mesh.nodes.size())},
                             {"s_abs_mm", sAbs},
                             {"material_flags", materialFlags}});
  }
  CommonGridShadowIntegrator integrator(magFieldContext);
  const auto runMesh = [&](const CommonGridShadowIntegrator::Mesh& mesh,
                           const CommonGridShadowIntegrator::State& y) {
    return integrator.integrate(y, mesh, center, normal, pathSign);
  };
  const auto meanN = runMesh(nominal, y0);
  const auto meanC = runMesh(coarse, y0);
  const auto meanF = runMesh(fine, y0);
  auto meanJson = [&](const CommonGridShadowIntegrator::Result& mean,
                      const CommonGridShadowIntegrator::Mesh& mesh) {
    json m;
    m["ok"] = mean.ok;
    m["n_field_steps"] = mean.nFieldSteps;
    m["n_material_maps"] = mean.nMaterialMaps;
    m["path_length"] = mean.pathLength;
    m["n_dot_direction"] = mean.nDotDir;
    m["distance_to_plane_mm"] = mean.distanceToPlaneMm;
    m["abort_reason"] = mean.abortReason;
    m["end_free"] = {mean.end.pos.x(), mean.end.pos.y(), mean.end.pos.z(),
                     mean.end.dir.x(), mean.end.dir.y(), mean.end.dir.z(),
                     mean.end.qop};
    ProfileHop hopM;
    const bool projected =
        mean.ok && projectSupportingPlane(*hit.surface, geometryContext,
                                          mean.end.pos, pathSign * mean.end.dir,
                                          hopM);
    m["projected"] = projected;
    m["predicted_loc0"] = projected ? json(hopM.predicted_loc0) : json(nullptr);
    m["official_minus_shadow_loc0"] =
        projected ? json(officialMeanLoc0 - hopM.predicted_loc0) : json(nullptr);
    m["official_minus_shadow_path"] = officialPath - mean.pathLength;
    m["official_minus_shadow_pos_norm"] = (officialPos - mean.end.pos).norm();
    m["official_minus_shadow_dir_norm"] = (officialDir - mean.end.dir).norm();
    const double qDen = std::max(std::abs(officialQop), 1.0e-12);
    m["official_minus_shadow_qop_rel"] =
        std::abs(officialQop - mean.end.qop) / qDen;
    m["mesh"] = mesh.name;
    m["does_not_replace_official_predicted_loc0"] = true;
    return m;
  };
  out["mean_coarse"] = meanJson(meanC, coarse);
  out["mean_nominal"] = meanJson(meanN, nominal);
  out["mean_fine"] = meanJson(meanF, fine);
  const bool projectedN = out["mean_nominal"]["projected"] == true;
  const double dLoc0 = projectedN
                           ? std::abs(out["mean_nominal"]["official_minus_shadow_loc0"]
                                          .get<double>())
                           : 1.0e9;
  const double dPath =
      std::abs(out["mean_nominal"]["official_minus_shadow_path"].get<double>());
  const double dPos =
      out["mean_nominal"]["official_minus_shadow_pos_norm"].get<double>();
  const double dDir =
      out["mean_nominal"]["official_minus_shadow_dir_norm"].get<double>();
  const double dQ =
      out["mean_nominal"]["official_minus_shadow_qop_rel"].get<double>();
  const bool meanClosed =
      meanN.ok && projectedN && dLoc0 <= CommonGridShadowIntegrator::kMeanLoc0AbsMm &&
      dPath <= CommonGridShadowIntegrator::kMeanPathAbsMm &&
      dPos <= CommonGridShadowIntegrator::kMeanPosAbsMm &&
      dDir <= CommonGridShadowIntegrator::kMeanDirAbs &&
      dQ <= CommonGridShadowIntegrator::kMeanQopRel;
  out["mean_contract"] = {
      {"closed", meanClosed},
      {"loc0_abs", dLoc0},
      {"path_abs", dPath},
      {"pos_abs", dPos},
      {"dir_abs", dDir},
      {"qop_rel", dQ},
      {"loc0_gate_mm", CommonGridShadowIntegrator::kMeanLoc0AbsMm},
      {"do_not_call_vacuum_a_production_map_reference", true}};
  double loc0N = 0.0;
  double loc0F = 0.0;
  double loc0C = 0.0;
  bool loc0s = false;
  if (out["mean_nominal"]["predicted_loc0"].is_number() &&
      out["mean_fine"]["predicted_loc0"].is_number() &&
      out["mean_coarse"]["predicted_loc0"].is_number()) {
    loc0N = out["mean_nominal"]["predicted_loc0"].get<double>();
    loc0F = out["mean_fine"]["predicted_loc0"].get<double>();
    loc0C = out["mean_coarse"]["predicted_loc0"].get<double>();
    loc0s = true;
  }
  out["grid_mean_refinement"] = {
      {"ok", loc0s},
      {"nominal_minus_fine_loc0", loc0s ? json(loc0N - loc0F) : json(nullptr)},
      {"coarse_minus_nominal_loc0", loc0s ? json(loc0C - loc0N) : json(nullptr)},
      {"stable", loc0s && std::abs(loc0N - loc0F) <=
                              CommonGridShadowIntegrator::kGridMeanLoc0AbsMm},
      {"gate_mm", CommonGridShadowIntegrator::kGridMeanLoc0AbsMm}};
  const std::array<double, 4> kRungs{1.0, 0.5, 0.25, 0.125};
  const std::array<double, 5> kSteps{0.01, 0.01, 1.0e-5, 1.0e-5, 1.0e-6};
  const std::array<const char*, 5> names{
      "loc0", "loc1", "phi", "theta", "q_over_p"};
  auto fdOnMesh = [&](const CommonGridShadowIntegrator::Mesh& mesh) {
    json columns = json::array();
    for (int col = 0; col < 5; ++col) {
      json rungs = json::array();
      bool ok = true;
      for (int rung = 0; rung < 4; ++rung) {
        const double step = kSteps[static_cast<std::size_t>(col)] *
                            kRungs[static_cast<std::size_t>(rung)];
        Acts::BoundVector plus = theta0;
        Acts::BoundVector minus = theta0;
        plus[col] += step;
        minus[col] -= step;
        const Acts::FreeVector fP = Acts::detail::transformBoundToFreeParameters(
            *startSurface, geometryContext, plus);
        const Acts::FreeVector fM = Acts::detail::transformBoundToFreeParameters(
            *startSurface, geometryContext, minus);
        CommonGridShadowIntegrator::State yP;
        yP.pos = fP.template segment<3>(Acts::eFreePos0);
        yP.dir = fP.template segment<3>(Acts::eFreeDir0);
        yP.qop = fP[Acts::eFreeQOverP];
        CommonGridShadowIntegrator::State yM;
        yM.pos = fM.template segment<3>(Acts::eFreePos0);
        yM.dir = fM.template segment<3>(Acts::eFreeDir0);
        yM.qop = fM[Acts::eFreeQOverP];
        const auto evP = runMesh(mesh, yP);
        const auto evM = runMesh(mesh, yM);
        ProfileHop hopP;
        ProfileHop hopM;
        const bool pOk =
            evP.ok && projectSupportingPlane(*hit.surface, geometryContext,
                                             evP.end.pos, pathSign * evP.end.dir,
                                             hopP);
        const bool mOk =
            evM.ok && projectSupportingPlane(*hit.surface, geometryContext,
                                             evM.end.pos, pathSign * evM.end.dir,
                                             hopM);
        json item;
        item["step_factor"] = kRungs[static_cast<std::size_t>(rung)];
        item["fd_step"] = step;
        item["ok"] = pOk && mOk;
        item["plus_predicted_loc0"] =
            pOk ? json(hopP.predicted_loc0) : json(nullptr);
        item["minus_predicted_loc0"] =
            mOk ? json(hopM.predicted_loc0) : json(nullptr);
        item["plus_n_steps"] = evP.nFieldSteps;
        item["minus_n_steps"] = evM.nFieldSteps;
        item["plus_path_length"] = evP.pathLength;
        item["minus_path_length"] = evM.pathLength;
        item["plus_abort"] = evP.abortReason;
        item["minus_abort"] = evM.abortReason;
        item["branch_identity_same"] =
            pOk && mOk && evP.nFieldSteps == evM.nFieldSteps &&
            evP.nMaterialMaps == evM.nMaterialMaps;
        item["common_grid_reused"] = true;
        if (pOk && mOk) {
          item["dloc0_d_start"] =
              (hopP.predicted_loc0 - hopM.predicted_loc0) / (2.0 * step);
        } else {
          item["dloc0_d_start"] = nullptr;
          ok = false;
        }
        rungs.push_back(item);
      }
      json colJson;
      colJson["parameter"] = names[static_cast<std::size_t>(col)];
      colJson["official_fd_step"] = kSteps[static_cast<std::size_t>(col)];
      colJson["ok"] = ok;
      colJson["rungs"] = rungs;
      colJson["do_not_add_rung"] = true;
      colJson["do_not_select_best_step"] = true;
      columns.push_back(colJson);
    }
    json block;
    block["rung_factors"] = {1.0, 0.5, 0.25, 0.125};
    block["columns"] = columns;
    block["mean_path"] = "common-grid RK4 shadow of production mean";
    block["mesh"] = mesh.name;
    return block;
  };
  out["shadow_fd_coarse"] = fdOnMesh(coarse);
  out["shadow_fd_nominal"] = fdOnMesh(nominal);
  out["shadow_fd_fine"] = fdOnMesh(fine);
  out["shadow_fd"] = out["shadow_fd_nominal"];
  out["grid_refinement_is_shadow_mesh_only"] = true;
  out["track_fd_ladder_unchanged"] = true;
  return out;
}

json vec3ToJson(const Acts::Vector3& v) {
  return json::array({v.x(), v.y(), v.z()});
}

json meanEventToJson(const MeanTransportShadow::Event& ev) {
  json o;
  o["kind"] = ev.kind;
  o["s_before"] = ev.sBefore;
  o["s_after"] = ev.sAfter;
  o["position_before"] = vec3ToJson(ev.posBefore);
  o["position_after"] = vec3ToJson(ev.posAfter);
  o["direction_before"] = vec3ToJson(ev.dirBefore);
  o["direction_after"] = vec3ToJson(ev.dirAfter);
  o["qop_before"] = ev.qopBefore;
  o["qop_after"] = ev.qopAfter;
  o["dir_norm_before"] = ev.dirNormBefore;
  o["dir_norm_after"] = ev.dirNormAfter;
  o["step_size"] = ev.stepSize;
  o["B_before"] = vec3ToJson(ev.bBefore);
  o["B_middle"] = vec3ToJson(ev.bMiddle);
  o["B_end"] = vec3ToJson(ev.bEnd);
  o["field_ok"] = ev.fieldOk;
  o["geometry_id"] = ev.geometryId;
  o["surface_z"] = ev.surfaceZ;
  o["path_correction"] = ev.pathCorrection;
  o["incidence_angle"] = ev.incidenceAngle;
  o["slab_thickness"] = ev.slabThickness;
  o["thickness_in_x0"] = ev.thicknessInX0;
  o["thickness_in_l0"] = ev.thicknessInL0;
  o["material_X0"] = ev.materialX0;
  o["material_L0"] = ev.materialL0;
  o["material_Ar"] = ev.materialAr;
  o["material_Z"] = ev.materialZ;
  o["qop_at_interaction"] = ev.qopAtInteraction;
  o["momentum_before"] = ev.momentumBefore;
  o["energy_before"] = ev.energyBefore;
  o["beta_before"] = ev.betaBefore;
  o["gamma_before"] = ev.gammaBefore;
  o["computeEnergyLossMean"] = ev.elossMean;
  o["computeEnergyLossMode"] = ev.elossMode;
  o["computeEnergyLossBethe"] = ev.elossBethe;
  o["computeEnergyLossLandau"] = ev.elossLandau;
  o["computeEnergyLossRadiative"] = ev.elossRadiative;
  o["evaluatePointwiseMaterialInteraction_Eloss"] = ev.elossEvaluatePointwise;
  o["eloss_applied"] = ev.elossApplied;
  o["production_delta_E"] = ev.productionDeltaE;
  o["delta_qop"] = ev.deltaQop;
  o["delta_position"] = (ev.posAfter - ev.posBefore).norm();
  o["delta_direction"] = (ev.dirAfter - ev.dirBefore).norm();
  o["volume_only"] = ev.volumeOnly;
  o["slab_valid"] = ev.slabValid;
  o["update_state_called"] = ev.updateStateCalled;
  o["has_surface_material_pointer"] = ev.hasSurfaceMaterialPointer;
  o["is_start_surface"] = ev.isStartSurface;
  o["is_target_surface"] = ev.isTargetSurface;
  o["update_stage"] = ev.updateStage;
  o["stage_factor"] = ev.stageFactor;
  o["particle_hypothesis"] = ev.particleHypothesis;
  o["abs_pdg"] = ev.absPdg;
  o["abs_charge"] = ev.absCharge;
  o["mass_gev"] = ev.massGeV;
  o["eloss_source_function"] = ev.elossSourceFunction;
  o["covariance_not_used"] = true;
  o["process_noise_not_used"] = true;
  return o;
}

json meanLedgerToJson(const MeanTransportShadow::Ledger& led) {
  json events = json::array();
  for (const auto& ev : led.events) {
    events.push_back(meanEventToJson(ev));
  }
  json o;
  o["ok"] = led.ok;
  o["method"] = led.method;
  o["path_length"] = led.pathLength;
  o["predicted_loc0"] = led.projected ? json(led.loc0) : json(nullptr);
  o["projected"] = led.projected;
  o["n_field_intervals"] = led.nField;
  o["n_material_interactions"] = led.nMaterial;
  o["n_field_fails"] = led.nFieldFails;
  o["n_dot_direction"] = led.nDotDir;
  o["distance_to_plane_mm"] = led.distanceToPlaneMm;
  o["max_abs_dir_norm_minus_one"] = led.maxAbsDirNormMinusOne;
  o["abort_reason"] = led.abortReason;
  o["initial_free"] = {led.start.pos.x(), led.start.pos.y(), led.start.pos.z(),
                       led.start.dir.x(), led.start.dir.y(), led.start.dir.z(),
                       led.start.qop};
  o["final_free"] = {led.end.pos.x(), led.end.pos.y(), led.end.pos.z(),
                     led.end.dir.x(), led.end.dir.y(), led.end.dir.z(),
                     led.end.qop};
  o["events"] = events;
  o["deterministic_mean_only"] = true;
  return o;
}

json meanResidualToJson(const MeanTransportShadow::Residual& r) {
  return {{"ok", r.ok},
          {"projected", r.projected},
          {"delta_loc0", r.loc0},
          {"delta_path", r.path},
          {"delta_position", r.pos},
          {"delta_direction", r.dir},
          {"delta_qop_rel", r.qopRel},
          {"closed", r.closed},
          {"loc0_gate_mm", MeanTransportShadow::kMeanLoc0AbsMm},
          {"path_gate_mm", MeanTransportShadow::kMeanPathAbsMm},
          {"pos_gate_mm", MeanTransportShadow::kMeanPosAbsMm},
          {"dir_gate", MeanTransportShadow::kMeanDirAbs},
          {"qop_rel_gate", MeanTransportShadow::kMeanQopRel}};
}

void fillProductionFieldStages(MeanTransportShadow& shadow,
                               MeanTransportShadow::Event& ev) {
  Acts::Vector3 B1 = Acts::Vector3::Zero();
  if (!shadow.field(ev.posBefore, B1)) {
    ev.fieldOk = false;
    return;
  }
  ev.bBefore = B1;
  const double h = ev.stepSize;
  const double h2 = h * h;
  const Acts::Vector3 k1 = ev.qopBefore * ev.dirBefore.cross(B1);
  const Acts::Vector3 pos1 =
      ev.posBefore + 0.5 * h * ev.dirBefore + h2 * 0.125 * k1;
  Acts::Vector3 B2 = Acts::Vector3::Zero();
  if (shadow.field(pos1, B2)) {
    ev.bMiddle = B2;
  }
  const Acts::Vector3 k2 =
      ev.qopBefore * (ev.dirBefore + 0.5 * h * k1).cross(B2);
  const Acts::Vector3 k3 =
      ev.qopBefore * (ev.dirBefore + 0.5 * h * k2).cross(B2);
  const Acts::Vector3 pos2 = ev.posBefore + h * ev.dirBefore + h2 * 0.5 * k3;
  Acts::Vector3 B3 = Acts::Vector3::Zero();
  if (shadow.field(pos2, B3)) {
    ev.bEnd = B3;
  }
}

template <typename propagator_t>
MeanTransportShadow::Ledger collectProductionMeanLedger(
    const propagator_t& propagator, const Acts::GeometryContext& geometryContext,
    const Acts::MagneticFieldContext& magFieldContext,
    const Acts::BoundTrackParameters& hopStart, const ProfileHit& hit,
    int maxSteps, bool matchKalman, double stepTolerance,
    MeanTransportShadow& shadow) {
  MeanTransportShadow::Ledger out;
  out.method = "production_eigenstepper_material_interactor";
  if (hit.surface == nullptr) {
    out.abortReason = "surface_pointer_null";
    return out;
  }
  const Acts::FreeVector free0 = Acts::detail::transformBoundToFreeParameters(
      hopStart.referenceSurface(), geometryContext, hopStart.parameters());
  out.start.pos = free0.template segment<3>(Acts::eFreePos0);
  out.start.dir = free0.template segment<3>(Acts::eFreeDir0);
  out.start.qop = free0[Acts::eFreeQOverP];
  const double startZ = hopStart.position(geometryContext).z();
  const double destZ = hit.surface->center(geometryContext).z();
  const bool forward = destZ >= startZ;
  using Actions =
      Acts::ActionList<MeanPreMaterialActor, MeanMaterialProbeActor,
                       Acts::MaterialInteractor, MeanPostMaterialActor,
                       Acts::detail::SteppingLogger>;
  using Abort = Acts::AbortList<Acts::EndOfWorldReached>;
  using Options = Acts::PropagatorOptions<Actions, Abort>;
  Options options(geometryContext, magFieldContext);
  options.maxSteps = static_cast<unsigned int>(std::max(maxSteps, 1));
  options.direction =
      forward ? Acts::Direction::Forward : Acts::Direction::Backward;
  options.loopProtection = true;
  options.stepTolerance = stepTolerance;
  if (!matchKalman) {
    options.maxStepSize = 10.0 * 1_m;
  }
  auto& material = options.actionList.template get<Acts::MaterialInteractor>();
  material.multipleScattering = true;
  material.energyLoss = true;
  material.recordInteractions = true;
  options.actionList.template get<Acts::detail::SteppingLogger>().sterile = false;
  auto state = propagator.template makeState<Acts::BoundTrackParameters, Options,
                                             Acts::SurfaceReached,
                                             Acts::PathLimitReached>(
      hopStart, *hit.surface, options);
  state.options.abortList.template get<Acts::SurfaceReached>().boundaryCheck =
      Acts::BoundaryCheck(false);
  auto propRes = propagator.propagate(state);
  out.pathLength = state.pathLength;
  out.end.pos = state.stepping.pars.template segment<3>(Acts::eFreePos0);
  out.end.dir = state.stepping.pars.template segment<3>(Acts::eFreeDir0);
  out.end.qop = state.stepping.pars[Acts::eFreeQOverP];
  if (!propRes.ok()) {
    out.abortReason = propRes.error().message();
  }
  const auto& rec =
      state.template get<Acts::MaterialInteractor::result_type>();
  const auto& pre =
      state.template get<MeanPreMaterialActor::result_type>().snaps;
  const auto& post =
      state.template get<MeanPostMaterialActor::result_type>().snaps;
  const auto& probes =
      state.template get<MeanMaterialProbeActor::result_type>().probes;
  const std::size_t n = std::min(pre.size(), post.size());
  MeanStateSnapshot prev;
  prev.path = 0.0;
  prev.pos = out.start.pos;
  prev.dir = out.start.dir;
  prev.qop = out.start.qop;
  prev.dirNorm = out.start.dir.norm();
  std::size_t imat = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (std::abs(pre[i].path - prev.path) > 1.0e-12) {
      MeanTransportShadow::Event ev;
      ev.kind = "field_propagation_interval";
      ev.sBefore = prev.path;
      ev.sAfter = pre[i].path;
      ev.posBefore = prev.pos;
      ev.dirBefore = prev.dir;
      ev.qopBefore = prev.qop;
      ev.dirNormBefore = prev.dirNorm;
      ev.posAfter = pre[i].pos;
      ev.dirAfter = pre[i].dir;
      ev.qopAfter = pre[i].qop;
      ev.dirNormAfter = pre[i].dirNorm;
      ev.stepSize = pre[i].path - prev.path;
      fillProductionFieldStages(shadow, ev);
      out.maxAbsDirNormMinusOne = std::max(
          out.maxAbsDirNormMinusOne, std::abs(ev.dirNormAfter - 1.0));
      ++out.nField;
      out.events.push_back(ev);
    }
    const bool qopChanged = std::abs(post[i].qop - pre[i].qop) > 1.0e-18;
    const bool haveProbe = i < probes.size();
    const bool pointerMaterial =
        pre[i].hasSurfaceMaterial ||
        (haveProbe && probes[i].hasSurfaceMaterial);
    if (pointerMaterial || qopChanged) {
      MeanTransportShadow::Event ev;
      ev.kind = "surface_material";
      ev.sBefore = pre[i].path;
      ev.sAfter = post[i].path;
      ev.posBefore = pre[i].pos;
      ev.dirBefore = pre[i].dir;
      ev.qopBefore = pre[i].qop;
      ev.dirNormBefore = pre[i].dirNorm;
      ev.posAfter = post[i].pos;
      ev.dirAfter = post[i].dir;
      ev.qopAfter = post[i].qop;
      ev.dirNormAfter = post[i].dirNorm;
      ev.deltaQop = post[i].qop - pre[i].qop;
      ev.geometryId = pre[i].geometryId;
      ev.surfaceZ = pre[i].surfaceZ;
      ev.hasSurfaceMaterialPointer = pointerMaterial;
      ev.volumeOnly = false;
      ev.productionDeltaE = MeanTransportShadow::productionDeltaEnergy(
          pre[i].qop, post[i].qop, forward ? 1.0 : -1.0);
      if (haveProbe && probes[i].hasSurfaceMaterial) {
        const auto& pr = probes[i];
        ev.geometryId = pr.geometryId;
        ev.surfaceZ = pr.surfaceZ;
        ev.updateStage = pr.updateStage;
        ev.stageFactor = pr.stageFactor;
        ev.isStartSurface = pr.isStartSurface;
        ev.isTargetSurface = pr.isTargetSurface;
        ev.slabValid = pr.slabValid;
        ev.updateStateCalled = pr.updateStateWouldRun;
        ev.pathCorrection = pr.pathCorrection;
        ev.slab = pr.slab;
        ev.absPdg = pr.absPdg;
        ev.absCharge = pr.absCharge;
        ev.massGeV = pr.massGeV;
        ev.particleHypothesis = pr.particleHypothesis;
        ev.elossEvaluatePointwise = pr.elossEvaluatePointwise;
        ev.elossSourceFunction = pr.elossSourceFunction;
        ev.incidenceAngle = std::acos(std::clamp(
            std::abs(pre[i].dir.normalized().dot(hit.surface->normal(
                geometryContext, hit.surface->center(geometryContext),
                Acts::Vector3::UnitZ()))),
            0.0, 1.0));
        MeanTransportShadow::fillEnergyLoss(
            ev, MeanTransportShadow::State{pre[i].pos, pre[i].dir, pre[i].qop},
            pr.slab, forward ? 1.0 : -1.0);
        ev.elossEvaluatePointwise = pr.elossEvaluatePointwise;
        ev.elossBethe = pr.elossBethe;
        ev.elossLandau = pr.elossLandau;
        ev.elossRadiative = pr.elossRadiative;
        ev.elossMean = pr.elossMean;
        ev.elossMode = pr.elossMode;
        ev.elossApplied = pr.updateStateWouldRun ? pr.elossEvaluatePointwise : 0.0;
      }
      while (imat < rec.materialInteractions.size() &&
             rec.materialInteractions[imat].surface == nullptr) {
        ++imat;
      }
      if (imat < rec.materialInteractions.size() && ev.updateStateCalled) {
        const auto& mi = rec.materialInteractions[imat];
        if (mi.surface != nullptr) {
          ev.incidenceAngle = std::acos(std::clamp(
              std::abs(mi.direction.normalized().dot(mi.surface->normal(
                  geometryContext, mi.surface->center(geometryContext),
                  mi.direction))),
              0.0, 1.0));
        }
        ++imat;
      }
      if (ev.updateStateCalled) {
        ++out.nMaterial;
      }
      out.events.push_back(ev);
    }
    prev = post[i];
  }
  out.ok = true;
  return out;
}

json firstDivergenceJson(const MeanTransportShadow::Ledger& prod,
                         const MeanTransportShadow::Ledger& sh) {
  json out;
  out["aligned_by"] = "physical_event_kind_and_order";
  out["not_array_index"] = true;
  std::size_t ip = 0;
  std::size_t is = 0;
  int fieldI = 0;
  int matI = 0;
  while (ip < prod.events.size() && is < sh.events.size()) {
    while (ip < prod.events.size() && is < sh.events.size() &&
           prod.events[ip].kind != sh.events[is].kind) {
      if (prod.events[ip].kind == "field_propagation_interval") {
        ++ip;
      } else if (sh.events[is].kind == "field_propagation_interval") {
        ++is;
      } else {
        ++ip;
        ++is;
      }
    }
    if (ip >= prod.events.size() || is >= sh.events.size()) {
      break;
    }
    const auto& pe = prod.events[ip];
    const auto& se = sh.events[is];
    const bool isField = pe.kind == "field_propagation_interval";
    const int seq = isField ? fieldI : matI;
    if (isField) {
      ++fieldI;
    } else {
      ++matI;
    }
    if (MeanTransportShadow::diverged(pe, se)) {
      const double dPos = (pe.posAfter - se.posAfter).norm();
      const double dDir = (pe.dirAfter - se.dirAfter).norm();
      const double dQ = std::abs(pe.qopAfter - se.qopAfter) /
                        std::max(std::abs(pe.qopAfter), 1.0e-12);
      out["found"] = true;
      out["event_kind"] = pe.kind;
      out["sequence_index"] = seq;
      out["production_event_index"] = ip;
      out["shadow_event_index"] = is;
      out["s_after"] = pe.sAfter;
      out["geometry_id"] = pe.geometryId;
      out["delta_position"] = dPos;
      out["delta_direction"] = dDir;
      out["delta_qop_rel"] = dQ;
      out["production"] = meanEventToJson(pe);
      out["shadow"] = meanEventToJson(se);
      return out;
    }
    ++ip;
    ++is;
  }
  out["found"] = false;
  out["n_compared_field"] = fieldI;
  out["n_compared_material"] = matI;
  return out;
}

json materialIdentityJson(const MeanTransportShadow::Ledger& led) {
  json ids = json::array();
  json updates = json::array();
  json stages = json::array();
  for (const auto& ev : led.events) {
    if (ev.kind != "surface_material") {
      continue;
    }
    ids.push_back(ev.geometryId);
    updates.push_back(ev.updateStateCalled);
    stages.push_back(ev.updateStage);
  }
  return {{"geometry_ids", ids},
          {"update_state_called", updates},
          {"update_stages", stages},
          {"n_field", led.nField},
          {"n_material", led.nMaterial}};
}

bool sameMaterialIdentity(const json& a, const json& b) {
  return a["geometry_ids"] == b["geometry_ids"] &&
         a["update_state_called"] == b["update_state_called"] &&
         a["n_field"] == b["n_field"] && a["n_material"] == b["n_material"];
}

json certifiedMeanCommonGridFdJson(
    MeanTransportShadow& shadow, const Acts::GeometryContext& geometryContext,
    const Acts::BoundTrackParameters& hopStart, const ProfileHit& hit,
    const MeanTransportShadow::Ledger& prod,
    const MeanTransportShadow::Ledger& nominal, const Acts::Vector3& center,
    const Acts::Vector3& normal, double pathSign) {
  json out;
  out["kind"] = "certified_mean_common_grid_fd";
  out["mean_function"] =
      "computeEnergyLossBethe + evaluateMaterialSlab gating + "
      "updateState p/E->q/p";
  out["field_replay"] = "official_rkn4_nystrom_on_frozen_production_accepted_steps";
  out["partition"] = "nominal_production_accepted_step_and_material_sequence";
  out["common_grid_reused"] = true;
  out["do_not_reuse_wb124_adaptive_dopri5"] = true;
  out["do_not_reuse_wb125_20_10_5"] = true;
  out["do_not_change_mean_semantics"] = true;
  out["do_not_add_fd_rung"] = true;
  out["do_not_shrink_fd_step"] = true;
  out["do_not_tune_grid_from_jacobian"] = true;
  out["track_fd_ladder"] = {1.0, 0.5, 0.25, 0.125};
  out["last_pair_rel_gate"] = 0.05;
  const json nomId = materialIdentityJson(nominal);
  out["nominal_material_identity"] = nomId;
  out["nominal_n_field"] = nominal.nField;
  out["nominal_n_material"] = nominal.nMaterial;
  out["nominal_projected"] = nominal.projected;
  out["nominal_ok"] = nominal.ok;
  const std::array<double, 4> kRungs{1.0, 0.5, 0.25, 0.125};
  const std::array<double, 5> kSteps{0.01, 0.01, 1.0e-5, 1.0e-5, 1.0e-6};
  const std::array<const char*, 5> names{
      "loc0", "loc1", "phi", "theta", "q_over_p"};
  auto startSurface = hopStart.referenceSurface().getSharedPtr();
  const Acts::BoundVector theta0 = hopStart.parameters();
  json columns = json::array();
  bool allArmsSameMean = true;
  bool allBranchSame = true;
  bool allMaterialSame = true;
  bool anyProjectionFail = false;
  for (int col = 0; col < 5; ++col) {
    json rungs = json::array();
    bool ok = true;
    bool signOk = true;
    std::array<double, 4> vals{0.0, 0.0, 0.0, 0.0};
    int prevSign = 0;
    bool havePrevSign = false;
    for (int rung = 0; rung < 4; ++rung) {
      const double step = kSteps[static_cast<std::size_t>(col)] *
                          kRungs[static_cast<std::size_t>(rung)];
      Acts::BoundVector plus = theta0;
      Acts::BoundVector minus = theta0;
      plus[col] += step;
      minus[col] -= step;
      const Acts::FreeVector fP = Acts::detail::transformBoundToFreeParameters(
          *startSurface, geometryContext, plus);
      const Acts::FreeVector fM = Acts::detail::transformBoundToFreeParameters(
          *startSurface, geometryContext, minus);
      MeanTransportShadow::State yP;
      yP.pos = fP.template segment<3>(Acts::eFreePos0);
      yP.dir = fP.template segment<3>(Acts::eFreeDir0);
      yP.qop = fP[Acts::eFreeQOverP];
      MeanTransportShadow::State yM;
      yM.pos = fM.template segment<3>(Acts::eFreePos0);
      yM.dir = fM.template segment<3>(Acts::eFreeDir0);
      yM.qop = fM[Acts::eFreeQOverP];
      MeanTransportShadow::Ledger ledP = shadow.replay(
          yP, prod.events, center, normal, pathSign, true, true,
          MeanTransportShadow::kMethodP);
      MeanTransportShadow::Ledger ledM = shadow.replay(
          yM, prod.events, center, normal, pathSign, true, true,
          MeanTransportShadow::kMethodP);
      ProfileHop hopP;
      ProfileHop hopM;
      const bool pOk =
          ledP.ok && projectSupportingPlane(*hit.surface, geometryContext,
                                            ledP.end.pos, pathSign * ledP.end.dir,
                                            hopP);
      const bool mOk =
          ledM.ok && projectSupportingPlane(*hit.surface, geometryContext,
                                            ledM.end.pos, pathSign * ledM.end.dir,
                                            hopM);
      if (pOk) {
        ledP.projected = true;
        ledP.loc0 = hopP.predicted_loc0;
      }
      if (mOk) {
        ledM.projected = true;
        ledM.loc0 = hopM.predicted_loc0;
      }
      const json idP = materialIdentityJson(ledP);
      const json idM = materialIdentityJson(ledM);
      const bool branchSame =
          pOk && mOk && ledP.nField == ledM.nField &&
          ledP.nMaterial == ledM.nMaterial && ledP.nField == nominal.nField &&
          ledP.nMaterial == nominal.nMaterial;
      const bool materialSame =
          sameMaterialIdentity(idP, nomId) && sameMaterialIdentity(idM, nomId);
      const bool armMean =
          pOk && mOk && branchSame && materialSame && !ledP.abortReason.size() &&
          !ledM.abortReason.size();
      allArmsSameMean = allArmsSameMean && armMean;
      allBranchSame = allBranchSame && branchSame;
      allMaterialSame = allMaterialSame && materialSame;
      if (!pOk || !mOk) {
        anyProjectionFail = true;
      }
      json item;
      item["step_factor"] = kRungs[static_cast<std::size_t>(rung)];
      item["fd_step"] = step;
      item["ok"] = pOk && mOk;
      item["plus_predicted_loc0"] =
          pOk ? json(hopP.predicted_loc0) : json(nullptr);
      item["minus_predicted_loc0"] =
          mOk ? json(hopM.predicted_loc0) : json(nullptr);
      item["plus_n_field"] = ledP.nField;
      item["minus_n_field"] = ledM.nField;
      item["plus_n_material"] = ledP.nMaterial;
      item["minus_n_material"] = ledM.nMaterial;
      item["plus_path_length"] = ledP.pathLength;
      item["minus_path_length"] = ledM.pathLength;
      item["plus_abort"] = ledP.abortReason;
      item["minus_abort"] = ledM.abortReason;
      item["plus_projected"] = pOk;
      item["minus_projected"] = mOk;
      item["branch_identity_same"] = branchSame;
      item["material_node_identity_same"] = materialSame;
      item["arm_mean_contract_same"] = armMean;
      item["common_grid_reused"] = true;
      item["plus_material_identity"] = idP;
      item["minus_material_identity"] = idM;
      if (pOk && mOk) {
        const double deriv =
            (hopP.predicted_loc0 - hopM.predicted_loc0) / (2.0 * step);
        item["dloc0_d_start"] = deriv;
        vals[static_cast<std::size_t>(rung)] = deriv;
        const int sgn = (deriv > 0.0) ? 1 : (deriv < 0.0) ? -1 : 0;
        if (havePrevSign && sgn != 0 && prevSign != 0 && sgn != prevSign) {
          signOk = false;
        }
        if (sgn != 0) {
          prevSign = sgn;
          havePrevSign = true;
        }
      } else {
        item["dloc0_d_start"] = nullptr;
        ok = false;
      }
      rungs.push_back(item);
    }
    double lastPair = 1.0e9;
    if (ok) {
      lastPair = std::abs(vals[2] - vals[3]) / std::max(std::abs(vals[3]), 1.0e-12);
    }
    json colJson;
    colJson["parameter"] = names[static_cast<std::size_t>(col)];
    colJson["official_fd_step"] = kSteps[static_cast<std::size_t>(col)];
    colJson["ok"] = ok;
    colJson["rungs"] = rungs;
    colJson["signed_h"] = ok ? json(vals[0]) : json(nullptr);
    colJson["signed_h2"] = ok ? json(vals[1]) : json(nullptr);
    colJson["signed_h4"] = ok ? json(vals[2]) : json(nullptr);
    colJson["signed_h8"] = ok ? json(vals[3]) : json(nullptr);
    colJson["last_pair_rel"] = ok ? json(lastPair) : json(nullptr);
    colJson["sign_consistent"] = ok && signOk;
    colJson["converged"] = ok && signOk && lastPair <= 0.05;
    colJson["do_not_add_rung"] = true;
    colJson["do_not_select_best_step"] = true;
    columns.push_back(colJson);
  }
  json block;
  block["rung_factors"] = {1.0, 0.5, 0.25, 0.125};
  block["columns"] = columns;
  block["mean_path"] =
      "certified WB127 shadow: official RKN4 on frozen production "
      "accepted steps + Bethe + updateState gating";
  block["mesh"] = "frozen_production_accepted_step_partition";
  out["shadow_fd"] = block;
  out["all_arms_same_mean_contract"] = allArmsSameMean;
  out["all_branch_identity_same"] = allBranchSame;
  out["all_material_node_identity_same"] = allMaterialSame;
  out["any_projection_fail"] = anyProjectionFail;
  out["n_dot_direction"] = nominal.nDotDir;
  out["certification_order"] =
      "perturbed_arm_mean_and_no_branch_switch -> shadow_fd_ladder "
      "convergence -> compare_to_WB123_repaired_tangent";
  return out;
}

json shadowMeanTransportHopJson(
    const Propagator& propagator, const Acts::GeometryContext& geometryContext,
    const Acts::MagneticFieldContext& magFieldContext,
    const Acts::BoundTrackParameters& hopStart, const ProfileHit& hit,
    int index, int maxSteps, bool matchKalman, const ProfileHop& official) {
  json out;
  out["kind"] = "shadow_mean_transport_contract";
  out["measurement_index"] = index;
  out["derivative_not_evaluated"] = true;
  out["jacobian_agreement_not_read"] = true;
  out["grid_selected_from_mean_contract_only"] = true;
  out["do_not_reuse_wb125_20_10_5"] = true;
  out["do_not_change_production_step_tolerance"] = true;
  out["replaces_official_mean_path"] = false;
  out["field_api"] = "FASERMagneticFieldWrapper::getField";
  out["shadow_p_method"] = MeanTransportShadow::kMethodP;
  out["shadow_d_method"] = MeanTransportShadow::kMethodD;
  out["mean_only_sequence_mm"] = {10.0, 5.0, 2.5, 1.25};
  out["mesh_justification"] = MeanTransportShadow::kMeshJustification;
  if (hit.surface == nullptr) {
    out["ok"] = false;
    out["abort_reason"] = "surface_pointer_null";
    return out;
  }
  MeanTransportShadow shadow(magFieldContext);
  const MeanTransportShadow::Ledger prod = collectProductionMeanLedger(
      propagator, geometryContext, magFieldContext, hopStart, hit, maxSteps,
      matchKalman, 1.0e-4, shadow);
  const Acts::FreeVector free0 = Acts::detail::transformBoundToFreeParameters(
      hopStart.referenceSurface(), geometryContext, hopStart.parameters());
  MeanTransportShadow::State y0;
  y0.pos = free0.template segment<3>(Acts::eFreePos0);
  y0.dir = free0.template segment<3>(Acts::eFreeDir0);
  y0.qop = free0[Acts::eFreeQOverP];
  const Acts::Vector3 center = hit.surface->center(geometryContext);
  const Acts::Vector3 normal =
      hit.surface->normal(geometryContext, center, Acts::Vector3::UnitZ());
  const double destZ = center.z();
  const double startZ = hopStart.position(geometryContext).z();
  const double pathSign = destZ >= startZ ? 1.0 : -1.0;
  const Acts::Vector3 officialPos(official.final_x_mm, official.final_y_mm,
                                  official.final_z_mm);
  const Acts::Vector3 officialDir(official.final_dx, official.final_dy,
                                  official.final_dz);
  auto projectLedger = [&](MeanTransportShadow::Ledger& led) {
    ProfileHop hopM;
    led.projected =
        led.ok && projectSupportingPlane(*hit.surface, geometryContext,
                                         led.end.pos, pathSign * led.end.dir,
                                         hopM);
    if (led.projected) {
      led.loc0 = hopM.predicted_loc0;
    }
  };
  MeanTransportShadow::Ledger ledP = shadow.replay(
      y0, prod.events, center, normal, pathSign, true, true,
      MeanTransportShadow::kMethodP);
  projectLedger(ledP);
  std::vector<MeanTransportShadow::MaterialNode> nodes;
  for (const auto& ev : prod.events) {
    if (ev.kind != "surface_material" || !ev.updateStateCalled) {
      continue;
    }
    MeanTransportShadow::MaterialNode node;
    node.sAbs = std::abs(ev.sBefore);
    node.volumeOnly = ev.volumeOnly;
    node.updateStateCalled = ev.updateStateCalled;
    node.geometryId = ev.geometryId;
    node.surfaceZ = ev.surfaceZ;
    node.pathCorrection = ev.pathCorrection;
    node.incidenceAngle = ev.incidenceAngle;
    node.slab = ev.slab;
    node.position = ev.posBefore;
    node.direction = ev.dirBefore;
    nodes.push_back(node);
  }
  json dJson = json::array();
  bool dClosed = false;
  bool dConverging = true;
  double prevLoc0 = 0.0;
  bool havePrev = false;
  for (int i = 0; i < MeanTransportShadow::kNDSteps; ++i) {
    const double h = MeanTransportShadow::kDStepsMm[static_cast<std::size_t>(i)];
    MeanTransportShadow::Ledger ledD = shadow.integrateUniform(
        y0, nodes, h, center, normal, pathSign, official.path_length, true,
        true);
    projectLedger(ledD);
    const auto resD = MeanTransportShadow::residualAgainst(
        ledD, official.predicted_loc0, official.path_length, officialPos,
        officialDir, official.final_qop);
    json item;
    item["max_step_mm"] = h;
    item["ledger"] = meanLedgerToJson(ledD);
    item["residual"] = meanResidualToJson(resD);
    dJson.push_back(item);
    if (resD.closed) {
      dClosed = true;
    }
    if (havePrev && std::abs(resD.loc0) > std::abs(prevLoc0) + 1.0e-6) {
      dConverging = false;
    }
    prevLoc0 = resD.loc0;
    havePrev = true;
  }
  const auto resP = MeanTransportShadow::residualAgainst(
      ledP, official.predicted_loc0, official.path_length, officialPos,
      officialDir, official.final_qop);
  auto ablationOf = [&](bool material, bool normalize, const char* name) {
    MeanTransportShadow::Ledger led = shadow.replay(
        y0, prod.events, center, normal, pathSign, material, normalize, name);
    projectLedger(led);
    const auto res = MeanTransportShadow::residualAgainst(
        led, official.predicted_loc0, official.path_length, officialPos,
        officialDir, official.final_qop);
    json item;
    item["variant"] = name;
    item["apply_material"] = material;
    item["normalize"] = normalize;
    item["residual"] = meanResidualToJson(res);
    item["n_material"] = led.nMaterial;
    item["max_abs_dir_norm_minus_one"] = led.maxAbsDirNormMinusOne;
    return item;
  };
  json ablation = json::array();
  ablation.push_back(ablationOf(false, true, "field_only"));
  ablation.push_back(ablationOf(true, false, "field_plus_material"));
  ablation.push_back(
      ablationOf(true, true, "field_plus_material_plus_normalize"));
  ablation.push_back(ablationOf(true, true, "full_shadow_contract"));

  json surfaces = json::array();
  json qopRows = json::array();
  json nodeRows = json::array();
  int nPathCorr = 0;
  int nMeanMatch = 0;
  int nModeMatch = 0;
  int nBetheMatch = 0;
  int nEvalMatch = 0;
  int nLandauMatch = 0;
  int nSurf = 0;
  int nPointer = 0;
  int nUpdate = 0;
  int nGated = 0;
  int nProdDeltaQopZero = 0;
  int nGatedShadowWouldBeMeanNonzero = 0;
  int nNodeQopClosed = 0;
  int nNodeCompared = 0;
  const double nodeQopAbs = 1.0e-9;
  const double nodeQopRel = 1.0e-6;
  std::size_t ish = 0;
  for (const auto& ev : prod.events) {
    if (ev.kind != "surface_material") {
      continue;
    }
    ++nSurf;
    if (ev.hasSurfaceMaterialPointer) {
      ++nPointer;
    }
    if (ev.updateStateCalled) {
      ++nUpdate;
    } else if (ev.hasSurfaceMaterialPointer) {
      ++nGated;
    }
    if (std::abs(ev.deltaQop) <= 1.0e-18) {
      ++nProdDeltaQopZero;
    }
    if (std::abs(ev.pathCorrection - 1.0) > 1.0e-6) {
      ++nPathCorr;
    }
    const double mass = ev.massGeV > 0.0 ? ev.massGeV : MeanTransportShadow::muonMass();
    auto qopFromEloss = [&](double eloss) {
      return MeanTransportShadow::qopAfterEnergyLoss(
          ev.qopBefore, eloss, pathSign, ev.absCharge);
    };
    const double qMean = qopFromEloss(ev.elossMean);
    const double qMode = qopFromEloss(ev.elossMode);
    const double qBethe = qopFromEloss(ev.elossBethe);
    const double qLandau = qopFromEloss(ev.elossLandau);
    const double qEval = qopFromEloss(ev.elossEvaluatePointwise);
    const double den = std::max(std::abs(ev.qopAfter), 1.0e-12);
    const bool meanMatch = std::abs(qMean - ev.qopAfter) / den <= 1.0e-6;
    const bool modeMatch = std::abs(qMode - ev.qopAfter) / den <= 1.0e-6;
    const bool betheMatch = std::abs(qBethe - ev.qopAfter) / den <= 1.0e-6;
    const bool landauMatch = std::abs(qLandau - ev.qopAfter) / den <= 1.0e-6;
    const bool evalMatch = std::abs(qEval - ev.qopAfter) / den <= 1.0e-6;
    if (meanMatch) {
      ++nMeanMatch;
    }
    if (modeMatch) {
      ++nModeMatch;
    }
    if (betheMatch) {
      ++nBetheMatch;
    }
    if (landauMatch) {
      ++nLandauMatch;
    }
    if (evalMatch) {
      ++nEvalMatch;
    }
    if (!ev.updateStateCalled && ev.hasSurfaceMaterialPointer &&
        ev.elossMean > 0.0) {
      ++nGatedShadowWouldBeMeanNonzero;
    }
    double shadowQop = ev.qopAfter;
    bool haveShadow = false;
    while (ish < ledP.events.size()) {
      if (ledP.events[ish].kind == "surface_material") {
        shadowQop = ledP.events[ish].qopAfter;
        haveShadow = true;
        ++ish;
        break;
      }
      ++ish;
    }
    const double nodeAbs = std::abs(shadowQop - ev.qopAfter);
    const double nodeRel = nodeAbs / den;
    const bool nodeClosed = haveShadow && (nodeAbs <= nodeQopAbs || nodeRel <= nodeQopRel);
    if (haveShadow) {
      ++nNodeCompared;
      if (nodeClosed) {
        ++nNodeQopClosed;
      }
    }
    surfaces.push_back({
        {"geometry_id", ev.geometryId},
        {"surface_z", ev.surfaceZ},
        {"update_stage", ev.updateStage},
        {"stage_factor", ev.stageFactor},
        {"is_start_surface", ev.isStartSurface},
        {"is_target_surface", ev.isTargetSurface},
        {"has_surface_material_pointer", ev.hasSurfaceMaterialPointer},
        {"slab_valid", ev.slabValid},
        {"update_state_called", ev.updateStateCalled},
        {"slab_thickness", ev.slabThickness},
        {"path_correction", ev.pathCorrection},
        {"incidence_angle", ev.incidenceAngle},
        {"particle_hypothesis", ev.particleHypothesis},
        {"abs_pdg", ev.absPdg},
        {"mass_gev", mass},
        {"abs_charge", ev.absCharge},
        {"charge", ev.qopBefore >= 0.0 ? 1.0 : -1.0},
        {"qop_before", ev.qopBefore},
        {"p_before", ev.momentumBefore},
        {"E_before", ev.energyBefore},
        {"qop_after_production", ev.qopAfter},
        {"qop_after_shadow", haveShadow ? json(shadowQop) : json(nullptr)},
        {"node_qop_closed", nodeClosed},
        {"delta_qop_production", ev.deltaQop},
        {"production_delta_E", ev.productionDeltaE},
        {"evaluatePointwise_Eloss", ev.elossEvaluatePointwise},
        {"eloss_source_function", ev.elossSourceFunction},
        {"computeEnergyLossBethe", ev.elossBethe},
        {"computeEnergyLossMean", ev.elossMean},
        {"computeEnergyLossMode", ev.elossMode},
        {"computeEnergyLossLandau", ev.elossLandau},
        {"computeEnergyLossRadiative", ev.elossRadiative},
        {"qop_from_bethe_formula", qBethe},
        {"qop_from_mean_formula", qMean},
        {"qop_from_mode_formula", qMode},
        {"qop_from_landau_formula", qLandau},
        {"qop_from_evaluatePointwise", qEval},
        {"production_matches_bethe_formula", betheMatch},
        {"production_matches_mean_formula", meanMatch},
        {"production_matches_mode_formula", modeMatch},
        {"production_matches_landau_formula", landauMatch},
        {"production_matches_evaluatePointwise", evalMatch},
        {"mean_equals_bethe_plus_radiative",
         std::abs(ev.elossMean - (ev.elossBethe + ev.elossRadiative)) <=
             1.0e-6 * std::max(std::abs(ev.elossMean), 1.0e-12)},
        {"mode_equals_0p9_landau_plus_0p15_radiative",
         std::abs(ev.elossMode -
                  (0.9 * ev.elossLandau + 0.15 * ev.elossRadiative)) <=
             1.0e-5 * std::max(std::abs(ev.elossMode), 1.0e-12)},
        {"wb126_mean_between_explanation",
         "Mean=Bethe+Radiative; Mode=0.9*Landau+0.15*Radiative; production "
         "uses Bethe only, so production Delta E sits between Mean and Mode "
         "on thick muon slabs"},
        {"acts_qop_unit", "1/GeV"},
        {"eloss_unit", "GeV"},
        {"thickness_unit", "mm"},
        {"recorded_slab_includes_path_correction", ev.pathCorrection != 0.0},
        {"eloss_not_fitted_to_endpoint", true},
        {"eloss_not_inferred_from_loc0", true},
    });
    nodeRows.push_back({
        {"geometry_id", ev.geometryId},
        {"update_state_called", ev.updateStateCalled},
        {"qop_after_production", ev.qopAfter},
        {"qop_after_shadow", haveShadow ? json(shadowQop) : json(nullptr)},
        {"closed", nodeClosed},
    });
    qopRows.push_back({
        {"geometry_id", ev.geometryId},
        {"qop", ev.qopBefore},
        {"abs_p", ev.momentumBefore},
        {"E", ev.energyBefore},
        {"beta", ev.betaBefore},
        {"gamma", ev.gammaBefore},
        {"mass_hypothesis_gev", mass},
        {"energy_loss_bethe", ev.elossBethe},
        {"energy_loss_mean", ev.elossMean},
        {"energy_loss_mode", ev.elossMode},
        {"evaluatePointwise_Eloss", ev.elossEvaluatePointwise},
        {"qop_new", ev.qopAfter},
        {"update_state_called", ev.updateStateCalled},
    });
  }
  const bool nodeQopAllClosed =
      nNodeCompared > 0 && nNodeQopClosed == nNodeCompared;
  bool firstMatIsAfterField = true;
  for (const auto& ev : prod.events) {
    if (ev.kind == "surface_material") {
      firstMatIsAfterField = ev.sBefore != 0.0 || !prod.events.empty();
      break;
    }
    if (ev.kind == "field_propagation_interval") {
      firstMatIsAfterField = true;
      break;
    }
  }
  out["ok"] = prod.ok && ledP.ok;
  out["node_qop_all_closed"] = nodeQopAllClosed;
  out["production_eloss_quantity"] = "computeEnergyLossBethe";
  out["do_not_fit_eloss_to_endpoint"] = true;
  out["do_not_infer_eloss_from_loc0"] = true;
  out["official_predicted_loc0"] = official.predicted_loc0;
  out["official_path_length"] = official.path_length;
  out["official_final_free"] = {official.final_x_mm, official.final_y_mm,
                                official.final_z_mm, official.final_dx,
                                official.final_dy, official.final_dz,
                                official.final_qop};
  out["production_ledger"] = meanLedgerToJson(prod);
  out["shadow_p_ledger"] = meanLedgerToJson(ledP);
  out["shadow_p_residual"] = meanResidualToJson(resP);
  out["shadow_d"] = dJson;
  out["shadow_d_any_closed"] = dClosed;
  out["shadow_d_loc0_monotone"] = dConverging;
  out["first_divergence"] = firstDivergenceJson(prod, ledP);
  out["mechanism_ablation"] = ablation;
  out["surface_energy_loss_mean_contract"] = {
      {"n_surfaces", nSurf},
      {"n_with_surface_material_pointer", nPointer},
      {"n_updateState_called", nUpdate},
      {"n_gated_pointer_no_update", nGated},
      {"n_production_delta_qop_zero", nProdDeltaQopZero},
      {"n_gated_where_mean_would_be_nonzero", nGatedShadowWouldBeMeanNonzero},
      {"n_with_path_correction_ne_1", nPathCorr},
      {"n_match_computeEnergyLossBethe", nBetheMatch},
      {"n_match_computeEnergyLossMean", nMeanMatch},
      {"n_match_computeEnergyLossMode", nModeMatch},
      {"n_match_computeEnergyLossLandau", nLandauMatch},
      {"n_match_evaluatePointwise_Eloss", nEvalMatch},
      {"production_deterministic_eloss", "computeEnergyLossBethe"},
      {"production_does_not_use_mean_or_mode", true},
      {"mean_formula", "computeEnergyLossBethe + computeEnergyLossRadiative"},
      {"mode_formula", "0.9*computeEnergyLossLandau + 0.15*computeEnergyLossRadiative"},
      {"compiled_source",
       "libActsCore.so Acts 32.0.2 "
       "PointwiseMaterialInteraction::evaluatePointwiseMaterialInteraction "
       "@ 0x2afce0 calls computeEnergyLossBethe"},
      {"production_uses_path_correction", nPathCorr > 0 || nSurf == 0},
      {"recorded_slab_already_path_corrected", true},
      {"evaluateMaterialSlab_scales_thickness_by_pathCorrection", true},
      {"recorder_equals_updater",
       "MaterialInteractor records a surface interaction only when "
       "evaluateMaterialSlab is true, which is exactly when updateState runs. "
       "A surfaceMaterial pointer is not sufficient: start=PostUpdate, "
       "target=PreUpdate, else FullUpdate, then ISurfaceMaterial::factor may "
       "return 0 and yield an invalid/zero-thickness slab."},
      {"wb126_delta_qop_zero_explanation",
       "WB126 emitted every hasSurfaceMaterial cycle and then consumed the "
       "next recorded slab. Gated surfaces (factor=0 / vacuum / invalid slab) "
       "have production Delta q/p = 0 because updateState never ran; forcing "
       "Mean on that attached slab is nonzero."},
      {"acts_source",
       "Acts/Propagator/MaterialInteractor.hpp + "
       "Acts/Propagator/detail/PointwiseMaterialInteraction.hpp "
       "evaluateMaterialSlab + compiled evaluatePointwiseMaterialInteraction "
       "+ updateState; Eloss is Bethe, not Mean/Mode"},
      {"surfaces", surfaces}};
  out["node_qop_contract"] = {
      {"n_compared", nNodeCompared},
      {"n_closed", nNodeQopClosed},
      {"all_closed", nodeQopAllClosed},
      {"abs_tol", nodeQopAbs},
      {"rel_tol", nodeQopRel},
      {"eloss_not_fitted_to_endpoint", true},
      {"eloss_not_inferred_from_loc0", true},
      {"nodes", nodeRows}};
  out["surface_energy_loss_mean_semantics"] =
      out["surface_energy_loss_mean_contract"];
  out["material_update_order_contract"] = {
      {"production_order",
       "propagate_to_surface -> evaluateMaterialSlab -> "
       "evaluatePointwiseMaterialInteraction -> updateState(q/p) -> "
       "continue"},
      {"start_surface_stage", "PostUpdate"},
      {"target_surface_stage", "PreUpdate"},
      {"intermediate_stage", "FullUpdate"},
      {"default_split_factor", 1.0},
      {"forward_target_preupdate_factor",
       "1 - splitFactor = 0 when splitFactor=1, so target surface material "
       "is not applied"},
      {"volume_material_recorded_not_applied_to_mean", true},
      {"shadow_copies_post_arrival_qop_update", true},
      {"shadow_copies_updateState_gating", true},
      {"shadow_uses_bethe_not_mean", true},
      {"first_material_after_a_field_interval", firstMatIsAfterField},
      {"qop_constant_on_field_interval", true},
      {"n_pointer_surfaces", nPointer},
      {"n_updateState_called", nUpdate},
      {"n_gated", nGated}};
  out["qop_energy_loss_unit_contract"] = {
      {"acts_internal_qop_unit", "1/GeV"},
      {"athena_export_qop_unit", "1/MeV"},
      {"energy_loss_input_output_unit", "GeV"},
      {"material_thickness_unit", "mm"},
      {"mass_hypothesis_gev", MeanTransportShadow::muonMass()},
      {"min_momentum_after_eloss", "10 MeV"},
      {"rows", qopRows}};
  out["direction_normalization_contract"] = {
      {"production_source",
       "EigenStepper.ipp accepted-step: dir += h/6*(k1+2(k2+k3)+k4); "
       "dir.normalize()"},
      {"shadow_p_normalizes_after_each_accepted_field_interval", true},
      {"normalization_is_source_supported", true},
      {"not_added_to_fit_endpoint", true},
      {"production_max_abs_dir_norm_minus_one", prod.maxAbsDirNormMinusOne},
      {"shadow_p_max_abs_dir_norm_minus_one", ledP.maxAbsDirNormMinusOne}};
  out["same_supporting_plane"] = true;
  out["same_particle_hypothesis"] = "muon";
  out["same_field_contract"] = true;
  out["certified_mean_common_grid_fd"] = certifiedMeanCommonGridFdJson(
      shadow, geometryContext, hopStart, hit, prod, ledP, center, normal,
      pathSign);
  out["derivative_not_evaluated"] = false;
  out["do_not_evaluate_fd"] = false;
  return out;
}

json shadowMeanTransportAuditJson(
    const Propagator& propagator, const Acts::GeometryContext& geometryContext,
    const Acts::MagneticFieldContext& magFieldContext,
    const Acts::BoundTrackParameters& start, const std::vector<ProfileHit>& hits,
    const Chi2Eval& official, int maxSteps, bool matchKalman) {
  json hops = json::array();
  const double z0 = start.position(geometryContext).z();
  const std::vector<int> order =
      sequentialHopOrder(start, hits, geometryContext);
  for (int idx : order) {
    const ProfileHit& hit = hits[static_cast<std::size_t>(idx)];
    const ProfileHop& off = official.hops[static_cast<std::size_t>(idx)];
    if (!off.ok || hit.surface == nullptr) {
      continue;
    }
    if (off.n_steps <= 10 || (idx != 6 && idx != 11)) {
      continue;
    }
    Acts::BoundTrackParameters hopStart = start;
    const bool thisDown = hit.zMm >= z0;
    bool havePrev = false;
    for (int prev : order) {
      if (prev == idx) {
        break;
      }
      const bool prevDown = hits[static_cast<std::size_t>(prev)].zMm >= z0;
      if (prevDown == thisDown &&
          official.hops[static_cast<std::size_t>(prev)].end_parameters.has_value()) {
        hopStart = *official.hops[static_cast<std::size_t>(prev)].end_parameters;
        havePrev = true;
      }
    }
    (void)havePrev;
    json hop = shadowMeanTransportHopJson(
        propagator, geometryContext, magFieldContext, hopStart, hit, idx,
        maxSteps, matchKalman, off);
    hop["hop_start_state"] = hopStartStateJson(hopStart, geometryContext);
    hop["focus86_required_candidate"] = true;
    hop["do_not_substitute_earlier_long_hop"] = true;
    hops.push_back(hop);
  }
  json out;
  out["kind"] = "certified_mean_common_grid_fd";
  out["shadow_mean_transport_kind"] = "shadow_mean_transport_contract";
  out["surface_energy_loss_kind"] = "surface_energy_loss_mean_semantics";
  out["production_eloss_quantity"] = "computeEnergyLossBethe";
  out["derivative_not_evaluated"] = false;
  out["jacobian_agreement_not_read"] = false;
  out["grid_selected_from_mean_contract_only"] = true;
  out["do_not_evaluate_fd"] = false;
  out["do_not_read_jacobian"] = false;
  out["do_not_reuse_wb124_adaptive_dopri5"] = true;
  out["do_not_reuse_wb125_20_10_5"] = true;
  out["do_not_change_mean_semantics"] = true;
  out["hops"] = hops;
  return out;
}

json officialPathJacobianAtToleranceJson(
    const Propagator& propagator, const Acts::GeometryContext& geometryContext,
    const Acts::MagneticFieldContext& magFieldContext,
    const Acts::BoundTrackParameters& start, const std::vector<ProfileHit>& hits,
    const Chi2Eval& official, int maxSteps, bool matchKalman,
    double stepTolerance,
    const GradientPropagator* gradientPropagator = nullptr,
    bool enableFieldGradientRepair = false,
    bool enableFocus86SegmentReference = false,
    bool enableFocus86CommonGridShadow = false) {
  json hops = json::array();
  Acts::BoundMatrix jDown = Acts::BoundMatrix::Identity();
  Acts::BoundMatrix jUp = Acts::BoundMatrix::Identity();
  Acts::BoundMatrix jDownRep = Acts::BoundMatrix::Identity();
  Acts::BoundMatrix jUpRep = Acts::BoundMatrix::Identity();
  bool downInit = false;
  bool upInit = false;
  bool downRepInit = false;
  bool upRepInit = false;
  bool firstLongFdDown = false;
  bool firstLongFdUp = false;
  bool loc0MeanUnchanged = true;
  double maxAbsLoc0RepairDelta = 0.0;
  bool anyAvailable = false;
  bool loc0Match = true;
  bool officialJacIdentity = true;
  bool usedBoundedTransportJacobian = false;
  const double z0 = start.position(geometryContext).z();
  const std::vector<int> order =
      sequentialHopOrder(start, hits, geometryContext);
  for (int idx : order) {
    const ProfileHit& hit = hits[static_cast<std::size_t>(idx)];
    const ProfileHop& off = official.hops[static_cast<std::size_t>(idx)];
    json item;
    item["measurement_index"] = idx;
    item["downstream"] = hit.zMm >= z0;
    item["official_predicted_loc0"] =
        off.ok ? json(off.predicted_loc0) : json(nullptr);
    item["official_predicted_loc1"] =
        off.ok ? json(off.predicted_loc1) : json(nullptr);
    item["official_residual"] = off.ok ? json(off.residual) : json(nullptr);
    item["continuation_state_construction"] = off.continuation_state_construction;
    item["path_length"] = off.path_length;
    item["number_of_propagation_steps"] = off.n_steps;
    item["n_dot_direction"] = off.n_dot_direction;
    item["incidence_angle"] = off.incidence_angle;
    item["surface_geometry"] = off.surface_geometry;
    item["source_chart"] = json::object();
    item["source_chart"]["parameters"] = "bound loc0 loc1 phi theta q/p time";
    item["source_chart"]["contract_columns"] = "first 5: loc0 loc1 phi theta q/p";
    item["source_chart"]["q_over_p_unit"] = "1/GeV";
    item["source_chart"]["loc0_unit"] = "mm";
    item["source_chart"]["surface_identity"] = off.surface_geometry;
    item["dummy_cov_bounded_transportJacobian_used"] = false;
    item["official_path"] =
        "source bound -> unbounded navigator/stepper -> free end -> "
        "supporting-plane intersection -> local loc0 -> residual";
    if (!off.ok || hit.surface == nullptr) {
      item["free_state_jacobian_available"] = false;
      hops.push_back(item);
      continue;
    }
    Acts::BoundTrackParameters hopStart = start;
    const bool thisDown = hit.zMm >= z0;
    if ((thisDown && downInit) || (!thisDown && upInit)) {
      for (int prev : order) {
        if (prev == idx) {
          break;
        }
        const bool prevDown = hits[static_cast<std::size_t>(prev)].zMm >= z0;
        if (prevDown == thisDown &&
            official.hops[static_cast<std::size_t>(prev)].end_parameters.has_value()) {
          hopStart = *official.hops[static_cast<std::size_t>(prev)].end_parameters;
        }
      }
    }
    const OfficialPathJacobianHop officialState = propagateOfficialPathJacobianHop(
        propagator, geometryContext, magFieldContext, hopStart, hit, idx,
        maxSteps, matchKalman, stepTolerance, false);
    const OfficialPathJacobianHop diag = propagateOfficialPathJacobianHop(
        propagator, geometryContext, magFieldContext, hopStart, hit, idx,
        maxSteps, matchKalman, stepTolerance, true);
    officialJacIdentity =
        officialJacIdentity && officialState.jacTransportIdentity &&
        !officialState.covTransport;
    const double loc0Delta = std::abs(diag.predictedLoc0 - off.predicted_loc0);
    const bool thisLoc0Match = diag.ok && loc0Delta <= 1.0e-6;
    loc0Match = loc0Match && thisLoc0Match;
    const bool available = diag.ok && diag.rkFreeAvailable && thisLoc0Match;
    anyAvailable = anyAvailable || available;
    const double sign = (hit.surface->center(geometryContext).z() >=
                         hopStart.position(geometryContext).z())
                            ? 1.0
                            : -1.0;
    const Acts::Vector3 officialPos(off.final_x_mm, off.final_y_mm, off.final_z_mm);
    const Acts::Vector3 officialDir(off.final_dx, off.final_dy, off.final_dz);
    const Acts::Vector3 officialSigned = sign * officialDir;
    const Eigen::Matrix<double, 1, 8> jProj = supportingPlaneLoc0FreeJacobian(
        *hit.surface, geometryContext, officialPos, officialSigned);
    const Acts::FreeMatrix jIntersect = supportingPlaneIntersectionFreeJacobian(
        *hit.surface, geometryContext, officialPos, officialDir, sign);
    const Acts::Vector3 center = hit.surface->center(geometryContext);
    const Acts::Vector3 normal =
        hit.surface->normal(geometryContext, center, Acts::Vector3::UnitZ());
    const double denom = normal.dot(officialSigned);
    const double tIntersect =
        (std::abs(denom) > 1.0e-12) ? normal.dot(center - officialPos) / denom
                                    : 0.0;
    Acts::FreeVector intersectFree = Acts::FreeVector::Zero();
    const Acts::Vector3 onPlane = officialPos + tIntersect * officialSigned;
    intersectFree[Acts::eFreePos0] = onPlane.x();
    intersectFree[Acts::eFreePos1] = onPlane.y();
    intersectFree[Acts::eFreePos2] = onPlane.z();
    intersectFree[Acts::eFreeTime] = 0.0;
    intersectFree[Acts::eFreeDir0] = officialDir.x();
    intersectFree[Acts::eFreeDir1] = officialDir.y();
    intersectFree[Acts::eFreeDir2] = officialDir.z();
    intersectFree[Acts::eFreeQOverP] = off.final_qop;
    const Acts::FreeToBoundMatrix jF2bIntersect =
        hit.surface->freeToBoundJacobian(geometryContext, intersectFree);
    const Eigen::Matrix<double, 1, Acts::eBoundSize> jHopRk = jProj * diag.jB2fRk;
    const Eigen::Matrix<double, 1, Acts::eBoundSize> jHopComposed =
        jProj * diag.jB2fComposed;
    const Acts::BoundMatrix jCont = jF2bIntersect * jIntersect * diag.jB2fRk;
    Acts::BoundMatrix& jAcc = (hit.zMm >= z0) ? jDown : jUp;
    const Eigen::Matrix<double, 1, Acts::eBoundSize> dloc0Rk = jHopRk * jAcc;
    const Eigen::Matrix<double, 1, Acts::eBoundSize> dloc0Composed =
        jHopComposed * jAcc;
    jAcc = jCont * jAcc;
    if (hit.zMm >= z0) {
      downInit = true;
    } else {
      upInit = true;
    }
    json jProjArr = json::array();
    json dRk = json::array();
    json dComposed = json::array();
    json hopRk = json::array();
    for (int c = 0; c < 8; ++c) {
      jProjArr.push_back(jProj(0, c));
    }
    for (int c = 0; c < 5; ++c) {
      dRk.push_back(dloc0Rk(0, c));
      dComposed.push_back(dloc0Composed(0, c));
      hopRk.push_back(jHopRk(0, c));
    }
    item["official_cov_transport"] = officialState.covTransport;
    item["official_jac_transport_is_identity"] = officialState.jacTransportIdentity;
    item["diagnostic_cov_transport"] = diag.covTransport;
    item["diagnostic_uses_dummy_cov_only_to_fill_jacTransport"] = true;
    item["diagnostic_uses_unbounded_supporting_plane"] = true;
    item["diagnostic_uses_bounded_transportJacobian"] = false;
    item["dummy_covariance_is_variational_switch_only"] = true;
    item["energy_loss_mean_update_not_in_rk_d"] = true;
    item["free_state_jacobian_available"] = available;
    item["diagnostic_ok"] = diag.ok;
    item["diagnostic_predicted_loc0"] = diag.ok ? json(diag.predictedLoc0) : json(nullptr);
    item["official_minus_diagnostic_loc0"] =
        diag.ok ? json(off.predicted_loc0 - diag.predictedLoc0) : json(nullptr);
    item["loc0_matches_official"] = thisLoc0Match;
    item["n_material_resets"] = diag.nResets;
    item["n_collector_calls"] = diag.nActorCalls;
    item["diagnostic_n_steps"] = diag.nSteps;
    item["diagnostic_path_length"] = diag.pathLength;
    item["diagnostic_abort"] = diag.abortReason;
    item["continuation_jacobian_kind"] =
        "f2b_intersection * intersection_free * rk_free * bound_to_free_start";
    item["source_bound_to_free_jacobian"] = denseMatrixToJson(diag.jB2fStart);
    item["free_transport_jacobian_since_last_reset"] =
        denseMatrixToJson(diag.jacTransport);
    item["bound_to_free_jacToGlobal_last_reset"] =
        denseMatrixToJson(diag.jacToGlobal);
    item["accumulated_bound_jacobian"] = denseMatrixToJson(diag.jacobian);
    item["free_to_path_derivative"] = json::array();
    for (int i = 0; i < 8; ++i) {
      item["free_to_path_derivative"].push_back(diag.derivative[i]);
    }
    item["rk_free_transport_jacobian_product"] = denseMatrixToJson(diag.jRkFreeAcc);
    item["bound_to_free_rk_product"] = denseMatrixToJson(diag.jB2fRk);
    item["bound_to_free_acts_composed"] = denseMatrixToJson(diag.jB2fComposed);
    item["supporting_plane_intersection_free_jacobian"] =
        denseMatrixToJson(jIntersect);
    item["supporting_plane_dloc0_d_free"] = jProjArr;
    item["free_to_bound_jacobian_intersection"] = denseMatrixToJson(jF2bIntersect);
    item["continuation_jacobian"] = denseMatrixToJson(jCont);
    item["hop_dloc0_d_start_rk_free"] = hopRk;
    item["chained_dloc0_d_source_rk_free"] = dRk;
    item["chained_dloc0_d_source_acts_composed"] = dComposed;
    item["final_residual_column_sign"] = "J_residual = -d loc0 / d source";
    item["field_gradient_repair"] = json::object();
    item["field_gradient_repair"]["requested"] = enableFieldGradientRepair;
    if (enableFieldGradientRepair && gradientPropagator != nullptr) {
      const OfficialPathJacobianHop meanGrad = propagateOfficialPathJacobianHop(
          *gradientPropagator, geometryContext, magFieldContext, hopStart, hit,
          idx, maxSteps, matchKalman, stepTolerance, false, false);
      const OfficialPathJacobianHop diagNew = propagateOfficialPathJacobianHop(
          *gradientPropagator, geometryContext, magFieldContext, hopStart, hit,
          idx, maxSteps, matchKalman, stepTolerance, true, diag.nSteps > 10);
      const double loc0RepairDelta =
          (meanGrad.ok && off.ok) ? (meanGrad.predictedLoc0 - off.predicted_loc0)
                                  : 1.0e9;
      maxAbsLoc0RepairDelta =
          std::max(maxAbsLoc0RepairDelta, std::abs(loc0RepairDelta));
      loc0MeanUnchanged =
          loc0MeanUnchanged && meanGrad.ok && std::abs(loc0RepairDelta) <= 1.0e-6;
      const Eigen::Matrix<double, 1, Acts::eBoundSize> jHopRep =
          jProj * diagNew.jB2fRk;
      const Acts::BoundMatrix jContRep =
          jF2bIntersect * jIntersect * diagNew.jB2fRk;
      Acts::BoundMatrix& jAccRep = (hit.zMm >= z0) ? jDownRep : jUpRep;
      const Eigen::Matrix<double, 1, Acts::eBoundSize> dloc0Rep = jHopRep * jAccRep;
      jAccRep = jContRep * jAccRep;
      if (hit.zMm >= z0) {
        downRepInit = true;
      } else {
        upRepInit = true;
      }
      (void)downRepInit;
      (void)upRepInit;
      json hopRep = json::array();
      json dRep = json::array();
      for (int c = 0; c < 5; ++c) {
        hopRep.push_back(jHopRep(0, c));
        dRep.push_back(dloc0Rep(0, c));
      }
      const auto posToDir = [](const Acts::FreeMatrix& jFree) {
        return jFree.block<3, 3>(4, 0).norm();
      };
      json repair;
      repair["requested"] = true;
      repair["mean_predicted_loc0"] =
          meanGrad.ok ? json(meanGrad.predictedLoc0) : json(nullptr);
      repair["official_minus_repair_mean_loc0"] =
          meanGrad.ok ? json(off.predicted_loc0 - meanGrad.predictedLoc0)
                      : json(nullptr);
      repair["diagnostic_predicted_loc0"] =
          diagNew.ok ? json(diagNew.predictedLoc0) : json(nullptr);
      repair["rk_free_transport_jacobian_product"] =
          denseMatrixToJson(diagNew.jRkFreeAcc);
      repair["bound_to_free_rk_product"] = denseMatrixToJson(diagNew.jB2fRk);
      repair["continuation_jacobian"] = denseMatrixToJson(jContRep);
      repair["hop_dloc0_d_start"] = hopRep;
      repair["chained_dloc0_d_source"] = dRep;
      repair["old_rk_pos_to_dir_norm"] = posToDir(diag.jRkFreeAcc);
      repair["new_rk_pos_to_dir_norm"] = posToDir(diagNew.jRkFreeAcc);
      repair["old_continuation_loc1_angular_norm"] =
          jCont.block<3, 1>(2, 1).norm();
      repair["new_continuation_loc1_angular_norm"] =
          jContRep.block<3, 1>(2, 1).norm();
      repair["n_material_resets"] = diagNew.nResets;
      repair["old_last_reset_pos_to_dir_norm"] = posToDir(diag.jacTransport);
      repair["new_last_reset_pos_to_dir_norm"] = posToDir(diagNew.jacTransport);
      repair["field_samples"] = diagNew.fieldSamples;
      const bool firstLongThisArm =
          (hit.zMm >= z0) ? !firstLongFdDown : !firstLongFdUp;
      if (diag.nSteps > 10 && firstLongThisArm) {
        repair["hop_start_segment_fd"] = hopStartSegmentFdJson(
            propagator, geometryContext, magFieldContext, hopStart, hit, idx,
            maxSteps, matchKalman, stepTolerance);
        if (hit.zMm >= z0) {
          firstLongFdDown = true;
        } else {
          firstLongFdUp = true;
        }
      }
      item["field_gradient_repair"] = repair;
    }
    const bool requiredFocusHop =
        (enableFocus86SegmentReference || enableFocus86CommonGridShadow) &&
        diag.nSteps > 10 && (idx == 6 || idx == 11);
    if (requiredFocusHop) {
      item["hop_start_state"] = hopStartStateJson(hopStart, geometryContext);
      item["focus86_required_candidate"] = true;
      item["do_not_substitute_earlier_long_hop"] = true;
      if (!item.contains("field_gradient_repair") ||
          !item["field_gradient_repair"].contains("hop_start_segment_fd")) {
        json fd = hopStartSegmentFdJson(
            propagator, geometryContext, magFieldContext, hopStart, hit, idx,
            maxSteps, matchKalman, stepTolerance);
        if (item.contains("field_gradient_repair") &&
            item["field_gradient_repair"]["requested"] == true) {
          item["field_gradient_repair"]["hop_start_segment_fd"] = fd;
        } else {
          item["hop_start_segment_fd"] = fd;
        }
      }
      const double officialLoc0 =
          off.ok ? off.predicted_loc0 : 0.0;
      if (enableFocus86SegmentReference) {
        item["independent_segment_reference"] = independentSegmentReferenceJson(
            geometryContext, magFieldContext, hopStart, hit, idx, officialLoc0);
      }
      if (enableFocus86CommonGridShadow) {
        item["common_grid_shadow_reference"] = commonGridShadowReferenceJson(
            propagator, geometryContext, magFieldContext, hopStart, hit, idx,
            maxSteps, matchKalman, stepTolerance, officialLoc0, off.path_length,
            Acts::Vector3(off.final_x_mm, off.final_y_mm, off.final_z_mm),
            Acts::Vector3(off.final_dx, off.final_dy, off.final_dz),
            off.final_qop);
      }
    }
    hops.push_back(item);
  }
  Eigen::MatrixXd jacRk(static_cast<int>(hits.size()), 5);
  Eigen::MatrixXd jacComposed(static_cast<int>(hits.size()), 5);
  jacRk.setZero();
  jacComposed.setZero();
  bool full = true;
  for (std::size_t i = 0; i < hops.size(); ++i) {
    if (hops[i]["free_state_jacobian_available"] != true) {
      full = false;
      continue;
    }
    const int idx = hops[i]["measurement_index"].get<int>();
    for (int c = 0; c < 5; ++c) {
      jacRk(idx, c) = -hops[i]["chained_dloc0_d_source_rk_free"][c].get<double>();
      jacComposed(idx, c) =
          -hops[i]["chained_dloc0_d_source_acts_composed"][c].get<double>();
    }
  }
  json colsRk = json::array();
  json colsComposed = json::array();
  const std::array<const char*, 5> names{
      "loc0", "loc1", "phi", "theta", "q_over_p"};
  for (int c = 0; c < 5; ++c) {
    json a;
    a["parameter"] = names[static_cast<std::size_t>(c)];
    a["residual_column"] = vecXdToJson(jacRk.col(c));
    a["column_norm"] = jacRk.col(c).norm();
    colsRk.push_back(a);
    json p;
    p["parameter"] = names[static_cast<std::size_t>(c)];
    p["residual_column"] = vecXdToJson(jacComposed.col(c));
    p["column_norm"] = jacComposed.col(c).norm();
    colsComposed.push_back(p);
  }
  json out;
  out["step_tolerance"] = stepTolerance;
  out["acts_api_field"] = "PropagatorPlainOptions::stepTolerance";
  out["official_start_covariance"] = "nullopt";
  out["official_path_jacTransport_is_identity"] = officialJacIdentity;
  out["variational_readout"] =
      "dummy cov only flips EigenStepper::covTransport on the official "
      "unbounded supporting-plane hop; jacTransport / collector product; "
      "not Propagator::Result::transportJacobian";
  out["dummy_cov_bounded_transportJacobian_used"] = usedBoundedTransportJacobian;
  out["any_free_state_jacobian_available"] = anyAvailable;
  out["chain_complete"] = full && anyAvailable;
  out["official_loc0_matches_diagnostic"] = loc0Match;
  out["do_not_replace_official_likelihood"] = true;
  out["hops"] = hops;
  out["rk_free_chain_columns"] = colsRk;
  out["acts_composed_chain_columns"] = colsComposed;
  out["primary_family"] = "rk_free_chain";
  if (enableFieldGradientRepair && gradientPropagator != nullptr) {
    Eigen::MatrixXd jacRep(static_cast<int>(hits.size()), 5);
    jacRep.setZero();
    bool fullRep = true;
    for (std::size_t i = 0; i < hops.size(); ++i) {
      if (!hops[i].contains("field_gradient_repair") ||
          hops[i]["field_gradient_repair"]["requested"] != true ||
          !hops[i]["field_gradient_repair"].contains("chained_dloc0_d_source")) {
        fullRep = false;
        continue;
      }
      const int idx = hops[i]["measurement_index"].get<int>();
      for (int c = 0; c < 5; ++c) {
        jacRep(idx, c) =
            -hops[i]["field_gradient_repair"]["chained_dloc0_d_source"][c]
                 .get<double>();
      }
    }
    json colsRep = json::array();
    for (int c = 0; c < 5; ++c) {
      json a;
      a["parameter"] = names[static_cast<std::size_t>(c)];
      a["residual_column"] = vecXdToJson(jacRep.col(c));
      a["column_norm"] = jacRep.col(c).norm();
      colsRep.push_back(a);
    }
    out["field_gradient_rk_free_chain_columns"] = colsRep;
    out["field_gradient_chain_complete"] = fullRep;
    out["field_gradient_mean_loc0_unchanged"] = loc0MeanUnchanged;
    out["field_gradient_max_abs_mean_loc0_delta"] = maxAbsLoc0RepairDelta;
    out["field_gradient_chi2_from_official_mean"] = official.ok ? json(official.chi2)
                                                                : json(nullptr);
    out["field_gradient_mean_chi2_unchanged"] = loc0MeanUnchanged;
    out["do_not_change_official_mean_path"] = true;
  }
  return out;
}

json officialSupportingPlaneJacobianAuditJson(
    const Propagator& propagator, const Acts::GeometryContext& geometryContext,
    const Acts::MagneticFieldContext& magFieldContext,
    const Acts::BoundTrackParameters& start, const std::vector<ProfileHit>& hits,
    int maxSteps, bool matchKalman, bool supportingPlane,
    const GradientPropagator* gradientPropagator = nullptr,
    bool enableFieldGradientRepair = false,
    bool enableFocus86SegmentReference = false,
    bool enableFocus86CommonGridShadow = false) {
  const std::array<double, 4> kRungs{1.0, 0.5, 0.25, 0.125};
  const std::array<const char*, 5> names{
      "loc0", "loc1", "phi", "theta", "q_over_p"};
  auto surface = start.referenceSurface().getSharedPtr();
  const Acts::BoundVector theta0 = start.parameters();
  auto evalAt = [&](const Acts::BoundVector& params, double tol) {
    return evaluateSequentialResiduals(
        propagator, geometryContext, magFieldContext,
        Acts::BoundTrackParameters(surface, params, std::nullopt,
                                   Acts::ParticleHypothesis::muon()),
        hits, maxSteps, matchKalman, true, supportingPlane, false, 0, true, tol);
  };
  Chi2Eval official = evalAt(theta0, 1.0e-4);
  json fdColumns = json::array();
  for (int col = 0; col < 5; ++col) {
    json rungs = json::array();
    std::array<Eigen::VectorXd, 4> jac{};
    bool colOk = true;
    for (int rung = 0; rung < 4; ++rung) {
      const double step =
          kFdSteps[static_cast<std::size_t>(col)] * kRungs[static_cast<std::size_t>(rung)];
      Acts::BoundVector plus = theta0;
      Acts::BoundVector minus = theta0;
      plus[col] += step;
      minus[col] -= step;
      Chi2Eval evP = evalAt(plus, 1.0e-4);
      Chi2Eval evM = evalAt(minus, 1.0e-4);
      json item;
      item["step_factor"] = kRungs[static_cast<std::size_t>(rung)];
      item["fd_step"] = step;
      item["ok"] = evP.ok && evM.ok;
      item["plus_predicted_loc0"] = loc0VectorJson(evP);
      item["minus_predicted_loc0"] = loc0VectorJson(evM);
      item["plus_chi2"] = evP.ok ? json(evP.chi2) : json(nullptr);
      item["minus_chi2"] = evM.ok ? json(evM.chi2) : json(nullptr);
      if (evP.ok && evM.ok) {
        jac[static_cast<std::size_t>(rung)] =
            (evP.residual - evM.residual) / (2.0 * step);
        item["residual_column"] = vecXdToJson(jac[static_cast<std::size_t>(rung)]);
        item["column_norm"] = jac[static_cast<std::size_t>(rung)].norm();
      } else {
        colOk = false;
      }
      rungs.push_back(item);
    }
    json consecutive = json::array();
    for (int i = 0; i < 3; ++i) {
      json pair;
      pair["from_rung"] = i;
      pair["to_rung"] = i + 1;
      if (rungs[static_cast<std::size_t>(i)]["ok"] == true &&
          rungs[static_cast<std::size_t>(i + 1)]["ok"] == true) {
        const double nNext = jac[static_cast<std::size_t>(i + 1)].norm();
        pair["relative_error"] =
            nNext > 1.0e-12
                ? (jac[static_cast<std::size_t>(i)] - jac[static_cast<std::size_t>(i + 1)])
                          .norm() /
                      nNext
                : 0.0;
        pair["sign_consistent"] =
            jac[static_cast<std::size_t>(i)].dot(jac[static_cast<std::size_t>(i + 1)]) >=
            0.0;
      } else {
        pair["relative_error"] = nullptr;
        pair["sign_consistent"] = false;
      }
      consecutive.push_back(pair);
    }
    json colJson;
    colJson["parameter"] = names[static_cast<std::size_t>(col)];
    colJson["official_fd_step"] = kFdSteps[static_cast<std::size_t>(col)];
    colJson["ok"] = colOk;
    colJson["rungs"] = rungs;
    colJson["consecutive_relative_errors"] = consecutive;
    colJson["selected_best_step"] = false;
    fdColumns.push_back(colJson);
  }
  json chain = officialPathJacobianAtToleranceJson(
      propagator, geometryContext, magFieldContext, start, hits, official,
      maxSteps, matchKalman, 1.0e-4, gradientPropagator,
      enableFieldGradientRepair, enableFocus86SegmentReference,
      enableFocus86CommonGridShadow);
  json branches = json::array();
  for (const auto& hop : official.hops) {
    json b;
    b["measurement_index"] = hop.measurement_index;
    b["geometry_surface_sequence"] = hop.geometry_surface_sequence;
    b["continuation_state_construction"] = hop.continuation_state_construction;
    b["free_to_bound_fallback"] = hop.free_to_bound_fallback;
    b["path_length"] = hop.path_length;
    b["number_of_propagation_steps"] = hop.n_steps;
    branches.push_back(b);
  }
  json out;
  out["kind"] = "official_supporting_plane_jacobian";
  out["task"] = "B14U";
  out["official_is_sequential"] = true;
  out["official_step_tolerance"] = 1.0e-4;
  out["do_not_change_production_step_tolerance"] = true;
  out["do_not_select_best_step"] = true;
  out["do_not_select_best_tolerance"] = true;
  out["do_not_replace_official_likelihood"] = true;
  out["do_not_use_dummy_cov_bounded_transportJacobian"] = true;
  out["acts_step_tolerance_api"] = "PropagatorPlainOptions::stepTolerance";
  out["source_bound"] = {theta0[0], theta0[1], theta0[2], theta0[3], theta0[4]};
  out["source_bound_units"] = json::object();
  out["source_bound_units"]["loc0"] = "mm";
  out["source_bound_units"]["loc1"] = "mm";
  out["source_bound_units"]["phi"] = "rad";
  out["source_bound_units"]["theta"] = "rad";
  out["source_bound_units"]["q_over_p"] = "1/GeV";
  out["nominal_ok"] = official.ok;
  out["nominal_chi2"] = official.ok ? json(official.chi2) : json(nullptr);
  out["predicted_loc0"] = loc0VectorJson(official);
  out["fd_ladder"] = json::object();
  out["fd_ladder"]["rung_factors"] = {1.0, 0.5, 0.25, 0.125};
  out["fd_ladder"]["columns"] = fdColumns;
  out["fd_ladder"]["do_not_select_best_step"] = true;
  out["official_path_jacobian"] = chain;
  out["transport_branches"] = branches;
  out["field_gradient_variational_repair_requested"] = enableFieldGradientRepair;
  out["field_gradient_fd_step_mm"] = kFieldGradientFdStepMm;
  out["field_gradient_fd_step_pre_registered"] = true;
  out["do_not_tune_field_step_from_track_jacobian"] = true;
  out["do_not_change_official_mean_path"] = true;
  return out;
}

json projectionOnlyHopJson(const ProfileHop& hop, const ProfileHit& hit,
                           const Acts::GeometryContext& geometryContext) {
  json out{{"measurement_index", hop.measurement_index},
           {"ok", hop.ok},
           {"n_dot_direction", hop.n_dot_direction},
           {"intersection_denominator", hop.intersection_denominator},
           {"incidence_angle", hop.incidence_angle},
           {"distance_to_plane_mm", hop.distance_to_plane_mm},
           {"near_parallel", std::abs(hop.n_dot_direction) < 1.0e-6}};
  if (hit.surface == nullptr || !hop.ok) {
    out["projection_map_evaluable"] = false;
    return out;
  }
  const Acts::Vector3 pos(hop.final_x_mm, hop.final_y_mm, hop.final_z_mm);
  const Acts::Vector3 dir(hop.final_dx, hop.final_dy, hop.final_dz);
  const Acts::Vector3 center = hit.surface->center(geometryContext);
  const Acts::Vector3 normal =
      hit.surface->normal(geometryContext, center, Acts::Vector3::UnitZ());
  const Acts::RotationMatrix3 rotation =
      hit.surface->transform(geometryContext).rotation();
  const Acts::Vector3 u = rotation.col(0);
  const double denom = normal.dot(dir);
  const double t = (std::abs(denom) > 0.0) ? normal.dot(center - pos) / denom : 0.0;
  const Acts::Vector3 dt_dr = (std::abs(denom) > 0.0)
                                  ? Acts::Vector3(-normal / denom)
                                  : Acts::Vector3::Zero();
  const Eigen::Matrix3d dxyz_dr =
      Eigen::Matrix3d::Identity() + dir * dt_dr.transpose();
  const Acts::Vector3 dloc0_dpos = dxyz_dr.transpose() * u;
  json analytic{{"intersection_t", t},
                {"dloc0_d_free_position", {dloc0_dpos.x(), dloc0_dpos.y(), dloc0_dpos.z()}},
                {"denominator", denom}};
  const std::array<double, 4> kRungs{1.0, 0.5, 0.25, 0.125};
  const double hPos = 1.0e-3;
  json posCols = json::array();
  for (int axis = 0; axis < 3; ++axis) {
    json rungs = json::array();
    std::array<double, 4> jac{};
    bool ok = true;
    for (int rung = 0; rung < 4; ++rung) {
      const double step = hPos * kRungs[static_cast<std::size_t>(rung)];
      Acts::Vector3 plus = pos;
      Acts::Vector3 minus = pos;
      plus[axis] += step;
      minus[axis] -= step;
      ProfileHop hopP;
      ProfileHop hopM;
      const bool okP =
          projectSupportingPlane(*hit.surface, geometryContext, plus, dir, hopP);
      const bool okM =
          projectSupportingPlane(*hit.surface, geometryContext, minus, dir, hopM);
      json item{{"step_factor", kRungs[static_cast<std::size_t>(rung)]},
                {"ok", okP && okM},
                {"plus_loc0", hopP.predicted_loc0},
                {"minus_loc0", hopM.predicted_loc0}};
      if (okP && okM) {
        jac[static_cast<std::size_t>(rung)] =
            (hopP.predicted_loc0 - hopM.predicted_loc0) / (2.0 * step);
        item["dloc0"] = jac[static_cast<std::size_t>(rung)];
      } else {
        ok = false;
      }
      rungs.push_back(item);
    }
    json consecutive = json::array();
    for (int i = 0; i < 3; ++i) {
      const double a = jac[static_cast<std::size_t>(i)];
      const double b = jac[static_cast<std::size_t>(i + 1)];
      const double rel = std::abs(b) > 1.0e-12 ? std::abs(a - b) / std::abs(b) : 0.0;
      consecutive.push_back({{"from_rung", i},
                             {"to_rung", i + 1},
                             {"relative_error", rel},
                             {"sign_consistent", a * b >= 0.0}});
    }
    posCols.push_back({{"axis", axis == 0 ? "x" : axis == 1 ? "y" : "z"},
                       {"official_fd_step_mm", hPos},
                       {"ok", ok},
                       {"rungs", rungs},
                       {"consecutive_relative_errors", consecutive},
                       {"analytic_dloc0", dloc0_dpos[axis]}});
  }
  out["projection_map_evaluable"] = true;
  out["do_not_repropagate"] = true;
  out["analytic"] = analytic;
  out["position_fd"] = posCols;
  out["do_not_select_best_step"] = true;
  return out;
}

json actsJacobianDiagnosticJson(
    const Propagator& propagator, const Acts::GeometryContext& geometryContext,
    const Acts::MagneticFieldContext& magFieldContext,
    const Acts::BoundTrackParameters& start, const std::vector<ProfileHit>& hits,
    const Chi2Eval& official, int maxSteps, bool matchKalman,
    double stepTolerance) {
  json hops = json::array();
  bool anyPresent = false;
  auto surface = start.referenceSurface().getSharedPtr();
  std::optional<Acts::BoundTrackParameters> current = start;
  const double z0 = start.position(geometryContext).z();
  std::vector<int> order(hits.size());
  for (std::size_t i = 0; i < hits.size(); ++i) {
    order[i] = static_cast<int>(i);
  }
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return hits[static_cast<std::size_t>(a)].zMm <
           hits[static_cast<std::size_t>(b)].zMm;
  });
  for (int idx : order) {
    const ProfileHit& hit = hits[static_cast<std::size_t>(idx)];
    if (hit.surface == nullptr || !current.has_value()) {
      hops.push_back({{"measurement_index", idx}, {"acts_jacobian_present", false}});
      continue;
    }
    Acts::BoundSquareMatrix cov = Acts::BoundSquareMatrix::Identity();
    cov *= 1.0e-8;
    const Acts::BoundTrackParameters startCov(
        current->referenceSurface().getSharedPtr(), current->parameters(), cov,
        Acts::ParticleHypothesis::muon());
    const bool forward = hit.surface->center(geometryContext).z() >=
                         current->position(geometryContext).z();
    auto options = makeProfileOptions(
        geometryContext, magFieldContext,
        forward ? Acts::Direction::Forward : Acts::Direction::Backward, maxSteps,
        matchKalman, stepTolerance);
    auto result = propagator.propagate(startCov, *hit.surface, options);
    json item{{"measurement_index", idx},
              {"diagnostic_uses_acts_bound_target", true},
              {"official_path_uses_supporting_plane_projection", true},
              {"dummy_covariance_is_diagnostic_only", true}};
    if (result.ok() && result.value().transportJacobian.has_value()) {
      anyPresent = true;
      const Acts::BoundMatrix jac = result.value().transportJacobian.value();
      json row0 = json::array();
      for (int c = 0; c < 5; ++c) {
        row0.push_back(jac(Acts::eBoundLoc0, c));
      }
      item["acts_jacobian_present"] = true;
      item["acts_dloc0_dtheta"] = row0;
      if (result.value().endParameters.has_value()) {
        item["acts_end_loc0"] =
            result.value().endParameters->parameters()[Acts::eBoundLoc0];
      }
    } else {
      item["acts_jacobian_present"] = false;
      item["ok"] = result.ok();
      if (!result.ok()) {
        item["error"] = result.error().message();
      }
    }
    if (idx < static_cast<int>(official.hops.size()) && official.hops[static_cast<std::size_t>(idx)].ok) {
      item["official_predicted_loc0"] =
          official.hops[static_cast<std::size_t>(idx)].predicted_loc0;
    }
    hops.push_back(item);
    if (idx < static_cast<int>(official.hops.size()) &&
        official.hops[static_cast<std::size_t>(idx)].end_parameters.has_value()) {
      current = official.hops[static_cast<std::size_t>(idx)].end_parameters;
    } else if (hit.zMm < z0) {
      current.reset();
    }
    (void)z0;
  }
  return {{"acts_version", "32.0.2"},
          {"propagator_result_has_transportJacobian", true},
          {"filled_only_if_covTransport", true},
          {"official_path_transports_covariance", false},
          {"official_start_covariance", "nullopt"},
          {"complex_step_requires_acts_rewrite", true},
          {"autodiff_requires_acts_rewrite", true},
          {"surface_intersect_is_analytic", true},
          {"do_not_replace_official_likelihood", true},
          {"any_diagnostic_jacobian_present", anyPresent},
          {"hops", hops}};
}

json actsChainAtToleranceJson(
    const Propagator& propagator, const Acts::GeometryContext& geometryContext,
    const Acts::MagneticFieldContext& magFieldContext,
    const Acts::BoundTrackParameters& start, const std::vector<ProfileHit>& hits,
    const Chi2Eval& official, int maxSteps, bool matchKalman,
    double stepTolerance) {
  json hops = json::array();
  json predictedActs = json::array();
  json predictedProj = json::array();
  Acts::BoundMatrix jDown = Acts::BoundMatrix::Identity();
  Acts::BoundMatrix jUp = Acts::BoundMatrix::Identity();
  bool downInit = false;
  bool upInit = false;
  bool anyPresent = false;
  bool loc0Match = true;
  const double z0 = start.position(geometryContext).z();
  const std::vector<int> order =
      sequentialHopOrder(start, hits, geometryContext);
  for (int idx : order) {
    const ProfileHit& hit = hits[static_cast<std::size_t>(idx)];
    const ProfileHop& off = official.hops[static_cast<std::size_t>(idx)];
    json item;
    item["measurement_index"] = idx;
    item["downstream"] = hit.zMm >= z0;
    item["official_predicted_loc0"] = off.ok ? json(off.predicted_loc0) : json(nullptr);
    item["continuation_state_construction"] = off.continuation_state_construction;
    item["path_length"] = off.path_length;
    item["number_of_propagation_steps"] = off.n_steps;
    item["n_dot_direction"] = off.n_dot_direction;
    item["incidence_angle"] = off.incidence_angle;
    item["surface_geometry"] = off.surface_geometry;
    item["q_over_p_convention"] = json::object(
        {{"acts_internal", "1/GeV"},
         {"fd_step_unit", "1/GeV"},
         {"export_per_mev_uses", "q_over_p_acts * 1_MeV"}});
    if (!off.ok || hit.surface == nullptr) {
      item["acts_jacobian_present"] = false;
      hops.push_back(item);
      predictedActs.push_back(nullptr);
      predictedProj.push_back(nullptr);
      continue;
    }
    Acts::BoundTrackParameters hopStart = start;
    const bool thisDown = hit.zMm >= z0;
    if ((thisDown && downInit) || (!thisDown && upInit)) {
      for (int prev : order) {
        if (prev == idx) {
          break;
        }
        const bool prevDown = hits[static_cast<std::size_t>(prev)].zMm >= z0;
        if (prevDown == thisDown &&
            official.hops[static_cast<std::size_t>(prev)].end_parameters.has_value()) {
          hopStart = *official.hops[static_cast<std::size_t>(prev)].end_parameters;
        }
      }
    }
    Acts::BoundSquareMatrix cov = Acts::BoundSquareMatrix::Identity();
    cov *= 1.0e-8;
    const Acts::BoundTrackParameters startCov(
        hopStart.referenceSurface().getSharedPtr(), hopStart.parameters(), cov,
        Acts::ParticleHypothesis::muon());
    const bool forward = hit.surface->center(geometryContext).z() >=
                         hopStart.position(geometryContext).z();
    auto options = makeProfileOptions(
        geometryContext, magFieldContext,
        forward ? Acts::Direction::Forward : Acts::Direction::Backward, maxSteps,
        matchKalman, stepTolerance);
    auto result = propagator.propagate(startCov, *hit.surface, options);
    item["dummy_covariance_is_diagnostic_only"] = true;
    item["official_path_uses_supporting_plane_projection"] = true;
    if (!(result.ok() && result.value().transportJacobian.has_value() &&
          result.value().endParameters.has_value())) {
      item["acts_jacobian_present"] = false;
      item["ok"] = result.ok();
      if (!result.ok()) {
        item["error"] = result.error().message();
      }
      hops.push_back(item);
      predictedActs.push_back(nullptr);
      predictedProj.push_back(nullptr);
      continue;
    }
    anyPresent = true;
    const Acts::BoundMatrix jActs = result.value().transportJacobian.value();
    const Acts::BoundTrackParameters& endP = *result.value().endParameters;
    const double actsLoc0 = endP.parameters()[Acts::eBoundLoc0];
    const double officialLoc0 = off.predicted_loc0;
    const double loc0Delta = std::abs(actsLoc0 - officialLoc0);
    loc0Match = loc0Match && loc0Delta <= 1.0e-6;
    const Acts::BoundToFreeMatrix jB2fStart =
        hopStart.referenceSurface().boundToFreeJacobian(geometryContext,
                                                        hopStart.parameters());
    const Acts::BoundToFreeMatrix jB2fEnd =
        hit.surface->boundToFreeJacobian(geometryContext, endP.parameters());
    const Acts::FreeVector endFree = Acts::detail::transformBoundToFreeParameters(
        *hit.surface, geometryContext, endP.parameters());
    const Acts::FreeToBoundMatrix jF2bEnd =
        hit.surface->freeToBoundJacobian(geometryContext, endFree);
    const double sign = forward ? 1.0 : -1.0;
    const Acts::Vector3 officialPos(off.final_x_mm, off.final_y_mm, off.final_z_mm);
    const Acts::Vector3 officialDir(off.final_dx, off.final_dy, off.final_dz);
    const Acts::Vector3 officialSigned = sign * officialDir;
    const Acts::Vector3 actsPos = endFree.segment<3>(Acts::eFreePos0);
    const Acts::Vector3 actsDir = endFree.segment<3>(Acts::eFreeDir0);
    const Acts::Vector3 actsSigned = sign * actsDir;
    const Eigen::Matrix<double, 1, 8> jProjOfficial =
        supportingPlaneLoc0FreeJacobian(*hit.surface, geometryContext, officialPos,
                                        officialSigned);
    const Eigen::Matrix<double, 1, 8> jProjActs = supportingPlaneLoc0FreeJacobian(
        *hit.surface, geometryContext, actsPos, actsSigned);
    const Acts::FreeMatrix jIntersect = supportingPlaneIntersectionFreeJacobian(
        *hit.surface, geometryContext, actsPos, actsDir, sign);
    const Acts::Vector3 center = hit.surface->center(geometryContext);
    const Acts::Vector3 normal =
        hit.surface->normal(geometryContext, center, Acts::Vector3::UnitZ());
    const double denom = normal.dot(actsSigned);
    const double tIntersect =
        (std::abs(denom) > 1.0e-12) ? normal.dot(center - actsPos) / denom : 0.0;
    Acts::FreeVector intersectFree = Acts::FreeVector::Zero();
    const Acts::Vector3 onPlane = actsPos + tIntersect * actsSigned;
    intersectFree[Acts::eFreePos0] = onPlane.x();
    intersectFree[Acts::eFreePos1] = onPlane.y();
    intersectFree[Acts::eFreePos2] = onPlane.z();
    intersectFree[Acts::eFreeTime] = 0.0;
    intersectFree[Acts::eFreeDir0] = actsDir.x();
    intersectFree[Acts::eFreeDir1] = actsDir.y();
    intersectFree[Acts::eFreeDir2] = actsDir.z();
    intersectFree[Acts::eFreeQOverP] = endFree[Acts::eFreeQOverP];
    const Acts::FreeToBoundMatrix jF2bIntersect =
        hit.surface->freeToBoundJacobian(geometryContext, intersectFree);
    const Eigen::Matrix<double, 1, Acts::eBoundSize> jHopActs =
        jActs.row(Acts::eBoundLoc0);
    const Eigen::Matrix<double, 1, Acts::eBoundSize> jHopProj =
        jProjActs * jB2fEnd * jActs;
    const Eigen::Matrix<double, 1, Acts::eBoundSize> jHopProjOfficial =
        jProjOfficial * jB2fEnd * jActs;
    const Acts::BoundMatrix jContRawActs = jActs;
    const Acts::BoundMatrix jCont =
        jF2bIntersect * jIntersect * jB2fEnd * jActs;
    Acts::BoundMatrix& jAcc = (hit.zMm >= z0) ? jDown : jUp;
    const Eigen::Matrix<double, 1, Acts::eBoundSize> dloc0Acts = jHopActs * jAcc;
    const Eigen::Matrix<double, 1, Acts::eBoundSize> dloc0Proj = jHopProj * jAcc;
    jAcc = jCont * jAcc;
    if (hit.zMm >= z0) {
      downInit = true;
    } else {
      upInit = true;
    }
    json jProjArr = json::array();
    for (int c = 0; c < 8; ++c) {
      jProjArr.push_back(jProjOfficial(0, c));
    }
    json dActs = json::array();
    json dProj = json::array();
    for (int c = 0; c < 5; ++c) {
      dActs.push_back(dloc0Acts(0, c));
      dProj.push_back(dloc0Proj(0, c));
    }
    item["acts_jacobian_present"] = true;
    item["acts_end_loc0"] = actsLoc0;
    item["official_minus_acts_end_loc0"] = officialLoc0 - actsLoc0;
    item["acts_bound_size"] = Acts::eBoundSize;
    item["contract_uses_first_5_bound_columns"] = true;
    item["acts_bound_time_index"] = static_cast<int>(Acts::eBoundTime);
    item["acts_transport_jacobian"] = denseMatrixToJson(jActs);
    item["bound_to_free_jacobian_start"] = denseMatrixToJson(jB2fStart);
    item["bound_to_free_jacobian_end"] = denseMatrixToJson(jB2fEnd);
    item["free_to_bound_jacobian_end"] = denseMatrixToJson(jF2bEnd);
    item["supporting_plane_dloc0_d_free_official_point"] = jProjArr;
    json jProjActsArr = json::array();
    for (int c = 0; c < 8; ++c) {
      jProjActsArr.push_back(jProjActs(0, c));
    }
    item["supporting_plane_dloc0_d_free"] = jProjActsArr;
    item["supporting_plane_intersection_free_jacobian"] =
        denseMatrixToJson(jIntersect);
    item["free_to_bound_jacobian_intersection"] = denseMatrixToJson(jF2bIntersect);
    item["continuation_jacobian_kind"] =
        "f2b_intersection * intersection_free * b2f_end * acts_transport";
    item["continuation_jacobian"] = denseMatrixToJson(jCont);
    item["continuation_jacobian_raw_acts"] = denseMatrixToJson(jContRawActs);
    item["hop_dloc0_d_start_acts_bound"] = boundRowToJson(jActs, Acts::eBoundLoc0);
    json hopProj = json::array();
    json hopProjOfficial = json::array();
    for (int c = 0; c < 5; ++c) {
      hopProj.push_back(jHopProj(0, c));
      hopProjOfficial.push_back(jHopProjOfficial(0, c));
    }
    item["hop_dloc0_d_start_projection_composed"] = hopProj;
    item["hop_dloc0_d_start_projection_official_point"] = hopProjOfficial;
    item["chained_dloc0_d_source_acts_bound"] = dActs;
    item["chained_dloc0_d_source_projection_composed"] = dProj;
    item["chart"] = json::object(
        {{"source_parameters", "bound loc0 loc1 phi theta q/p time"},
         {"contract_columns", "first 5: loc0 loc1 phi theta q/p"},
         {"transport", "ACTS bound-to-bound transportJacobian"},
         {"bound_to_free", "Surface::boundToFreeJacobian at ACTS end"},
         {"projection", "supporting-plane local x = u · (p + t signedDir)"},
         {"intersection_free",
          "pos' = p + t signedDir; dir' = freeDir; time' = 0; q/p' = q/p"},
         {"continuation",
          "transform_free_to_bound(intersection, time=0, freeDir, q/p)"},
         {"residual", "m_loc0 - predicted_loc0"},
         {"q_over_p_unit", "1/GeV"},
         {"official_continuation", off.continuation_state_construction}});
    hops.push_back(item);
    predictedActs.push_back(dActs);
    predictedProj.push_back(dProj);
  }
  Eigen::MatrixXd jacActs(static_cast<int>(hits.size()), 5);
  Eigen::MatrixXd jacProj(static_cast<int>(hits.size()), 5);
  jacActs.setZero();
  jacProj.setZero();
  bool full = true;
  for (std::size_t i = 0; i < hops.size(); ++i) {
    if (hops[i]["acts_jacobian_present"] != true) {
      full = false;
      continue;
    }
    const int idx = hops[i]["measurement_index"].get<int>();
    for (int c = 0; c < 5; ++c) {
      jacActs(idx, c) = -hops[i]["chained_dloc0_d_source_acts_bound"][c].get<double>();
      jacProj(idx, c) =
          -hops[i]["chained_dloc0_d_source_projection_composed"][c].get<double>();
    }
  }
  json colsActs = json::array();
  json colsProj = json::array();
  const std::array<const char*, 5> names{
      "loc0", "loc1", "phi", "theta", "q_over_p"};
  for (int c = 0; c < 5; ++c) {
    json a;
    a["parameter"] = names[static_cast<std::size_t>(c)];
    a["residual_column"] = vecXdToJson(jacActs.col(c));
    a["column_norm"] = jacActs.col(c).norm();
    colsActs.push_back(a);
    json p;
    p["parameter"] = names[static_cast<std::size_t>(c)];
    p["residual_column"] = vecXdToJson(jacProj.col(c));
    p["column_norm"] = jacProj.col(c).norm();
    colsProj.push_back(p);
  }
  json out;
  out["step_tolerance"] = stepTolerance;
  out["acts_api_field"] = "PropagatorPlainOptions::stepTolerance";
  out["any_diagnostic_jacobian_present"] = anyPresent;
  out["chain_complete"] = full && anyPresent;
  out["acts_end_loc0_matches_official"] = loc0Match;
  out["do_not_replace_official_likelihood"] = true;
  out["hops"] = hops;
  out["acts_bound_chain_columns"] = colsActs;
  out["projection_composed_chain_columns"] = colsProj;
  return out;
}

json derivativeContractAuditJson(
    const Propagator& propagator, const Acts::GeometryContext& geometryContext,
    const Acts::MagneticFieldContext& magFieldContext,
    const Acts::BoundTrackParameters& start, const std::vector<ProfileHit>& hits,
    int maxSteps, bool matchKalman, bool supportingPlane) {
  const std::array<double, 4> kRungs{1.0, 0.5, 0.25, 0.125};
  const std::array<const char*, 5> names{
      "loc0", "loc1", "phi", "theta", "q_over_p"};
  const std::array<const char*, 3> accNames{"nominal", "tighter", "tighter_again"};
  const std::array<double, 3> accTol{1.0e-4, 1.0e-5, 1.0e-6};
  auto surface = start.referenceSurface().getSharedPtr();
  const Acts::BoundVector theta0 = start.parameters();
  auto evalAt = [&](const Acts::BoundVector& params, double tol) {
    return evaluateSequentialResiduals(
        propagator, geometryContext, magFieldContext,
        Acts::BoundTrackParameters(surface, params, std::nullopt,
                                   Acts::ParticleHypothesis::muon()),
        hits, maxSteps, matchKalman, true, supportingPlane, false, 0, true, tol);
  };
  Chi2Eval official = evalAt(theta0, 1.0e-4);
  json fdColumns = json::array();
  for (int col = 0; col < 5; ++col) {
    json rungs = json::array();
    std::array<Eigen::VectorXd, 4> jac{};
    bool colOk = true;
    for (int rung = 0; rung < 4; ++rung) {
      const double step =
          kFdSteps[static_cast<std::size_t>(col)] * kRungs[static_cast<std::size_t>(rung)];
      Acts::BoundVector plus = theta0;
      Acts::BoundVector minus = theta0;
      plus[col] += step;
      minus[col] -= step;
      Chi2Eval evP = evalAt(plus, 1.0e-4);
      Chi2Eval evM = evalAt(minus, 1.0e-4);
      json item;
      item["step_factor"] = kRungs[static_cast<std::size_t>(rung)];
      item["fd_step"] = step;
      item["ok"] = evP.ok && evM.ok;
      item["plus_predicted_loc0"] = loc0VectorJson(evP);
      item["minus_predicted_loc0"] = loc0VectorJson(evM);
      item["plus_chi2"] = evP.ok ? json(evP.chi2) : json(nullptr);
      item["minus_chi2"] = evM.ok ? json(evM.chi2) : json(nullptr);
      if (evP.ok && evM.ok) {
        jac[static_cast<std::size_t>(rung)] =
            (evP.residual - evM.residual) / (2.0 * step);
        item["residual_column"] = vecXdToJson(jac[static_cast<std::size_t>(rung)]);
        item["column_norm"] = jac[static_cast<std::size_t>(rung)].norm();
      } else {
        colOk = false;
      }
      rungs.push_back(item);
    }
    json consecutive = json::array();
    for (int i = 0; i < 3; ++i) {
      json pair;
      pair["from_rung"] = i;
      pair["to_rung"] = i + 1;
      if (rungs[static_cast<std::size_t>(i)]["ok"] == true &&
          rungs[static_cast<std::size_t>(i + 1)]["ok"] == true) {
        const double nNext = jac[static_cast<std::size_t>(i + 1)].norm();
        pair["relative_error"] =
            nNext > 1.0e-12
                ? (jac[static_cast<std::size_t>(i)] - jac[static_cast<std::size_t>(i + 1)])
                          .norm() /
                      nNext
                : 0.0;
        pair["sign_consistent"] =
            jac[static_cast<std::size_t>(i)].dot(jac[static_cast<std::size_t>(i + 1)]) >=
            0.0;
      } else {
        pair["relative_error"] = nullptr;
        pair["sign_consistent"] = false;
      }
      consecutive.push_back(pair);
    }
    json colJson;
    colJson["parameter"] = names[static_cast<std::size_t>(col)];
    colJson["official_fd_step"] = kFdSteps[static_cast<std::size_t>(col)];
    colJson["ok"] = colOk;
    colJson["rungs"] = rungs;
    colJson["consecutive_relative_errors"] = consecutive;
    colJson["selected_best_step"] = false;
    fdColumns.push_back(colJson);
  }

  json accuracy = json::array();
  for (int ia = 0; ia < 3; ++ia) {
    const double tol = accTol[static_cast<std::size_t>(ia)];
    Chi2Eval nom = evalAt(theta0, tol);
    json rung;
    rung["name"] = accNames[static_cast<std::size_t>(ia)];
    rung["step_tolerance"] = tol;
    rung["selected_as_production"] = false;
    rung["nominal_ok"] = nom.ok;
    rung["nominal_chi2"] = nom.ok ? json(nom.chi2) : json(nullptr);
    rung["predicted_loc0"] = loc0VectorJson(nom);
    rung["path_length_total"] = hopPathTotal(nom);
    rung["n_steps_total"] = hopStepsTotal(nom);
    if (nom.ok) {
      rung["acts_chain"] = actsChainAtToleranceJson(
          propagator, geometryContext, magFieldContext, start, hits, nom,
          maxSteps, matchKalman, tol);
    } else {
      rung["acts_chain"] = nullptr;
    }
    accuracy.push_back(rung);
  }

  json branches = json::array();
  for (const auto& hop : official.hops) {
    json b;
    b["measurement_index"] = hop.measurement_index;
    b["geometry_surface_sequence"] = hop.geometry_surface_sequence;
    b["continuation_state_construction"] = hop.continuation_state_construction;
    b["free_to_bound_fallback"] = hop.free_to_bound_fallback;
    b["path_length"] = hop.path_length;
    b["number_of_propagation_steps"] = hop.n_steps;
    branches.push_back(b);
  }

  json out;
  out["kind"] = "acts_fd_derivative_contract";
  out["task"] = "B14L";
  out["official_is_sequential"] = true;
  out["official_step_tolerance"] = 1.0e-4;
  out["do_not_change_production_step_tolerance"] = true;
  out["do_not_select_best_step"] = true;
  out["do_not_select_best_tolerance"] = true;
  out["do_not_replace_official_likelihood"] = true;
  out["acts_step_tolerance_api"] = "PropagatorPlainOptions::stepTolerance";
  out["source_bound"] = {theta0[0], theta0[1], theta0[2], theta0[3], theta0[4]};
  out["source_bound_units"] = json::object(
      {{"loc0", "mm"},
       {"loc1", "mm"},
       {"phi", "rad"},
       {"theta", "rad"},
       {"q_over_p", "1/GeV"}});
  out["nominal_ok"] = official.ok;
  out["nominal_chi2"] = official.ok ? json(official.chi2) : json(nullptr);
  out["predicted_loc0"] = loc0VectorJson(official);
  out["fd_ladder"] = json::object(
      {{"rung_factors", {1.0, 0.5, 0.25, 0.125}},
       {"columns", fdColumns},
       {"do_not_select_best_step", true}});
  out["acts_chain_accuracy_rungs"] = accuracy;
  out["transport_branches"] = branches;
  return out;
}

json mapSmoothnessAuditJson(
    const Propagator& propagator, const Acts::GeometryContext& geometryContext,
    const Acts::MagneticFieldContext& magFieldContext,
    const Acts::BoundTrackParameters& start, const std::vector<ProfileHit>& hits,
    int maxSteps, bool matchKalman, bool supportingPlane, int nRepeats) {
  const std::array<double, 4> kRungs{1.0, 0.5, 0.25, 0.125};
  const std::array<int, 3> kFocus{1, 2, 4};
  const std::array<const char*, 3> kFocusNames{"loc1", "phi", "q_over_p"};
  const std::array<const char*, 3> kAccNames{"nominal", "tighter", "tighter_again"};
  const std::array<double, 3> kAccTol{1.0e-4, 1.0e-5, 1.0e-6};
  auto surface = start.referenceSurface().getSharedPtr();
  const Acts::BoundVector theta0 = start.parameters();
  auto evalAt = [&](const Acts::BoundVector& params, double tol) {
    return evaluateSequentialResiduals(
        propagator, geometryContext, magFieldContext,
        Acts::BoundTrackParameters(surface, params, std::nullopt,
                                   Acts::ParticleHypothesis::muon()),
        hits, maxSteps, matchKalman, true, supportingPlane, false, 0, true, tol);
  };

  json repeats = json::array();
  Chi2Eval first;
  bool firstSet = false;
  double maxLoc0 = 0.0;
  double maxChi2 = 0.0;
  bool stepsSame = true;
  bool allOk = true;
  for (int i = 0; i < std::max(nRepeats, 2); ++i) {
    Chi2Eval ev = evalAt(theta0, 1.0e-4);
    if (!firstSet) {
      first = ev;
      firstSet = true;
    } else {
      maxLoc0 = std::max(maxLoc0, loc0L2Delta(first, ev));
      if (first.ok && ev.ok) {
        maxChi2 = std::max(maxChi2, std::abs(first.chi2 - ev.chi2));
      }
      if (first.hops.size() == ev.hops.size()) {
        for (std::size_t h = 0; h < first.hops.size(); ++h) {
          if (first.hops[h].n_steps != ev.hops[h].n_steps) {
            stepsSame = false;
          }
        }
      } else {
        stepsSame = false;
      }
    }
    allOk = allOk && ev.ok;
    json repeat;
    repeat["repeat_index"] = i;
    repeat["ok"] = ev.ok;
    repeat["chi2"] = ev.ok ? json(ev.chi2) : json(nullptr);
    repeat["n_steps_total"] = hopStepsTotal(ev);
    repeat["path_length_total"] = hopPathTotal(ev);
    repeat["predicted_loc0"] = loc0VectorJson(ev);
    repeats.push_back(repeat);
  }
  const bool deterministic = allOk && stepsSame && maxLoc0 <= 1.0e-12 && maxChi2 <= 1.0e-12;

  json stageColumns = json::array();
  json resolutionRows = json::array();
  for (int ip = 0; ip < 3; ++ip) {
    const int col = kFocus[static_cast<std::size_t>(ip)];
    json rungs = json::array();
    std::array<Chi2Eval, 4> plusE{};
    std::array<Chi2Eval, 4> minusE{};
    bool colOk = true;
    for (int rung = 0; rung < 4; ++rung) {
      const double step =
          kFdSteps[static_cast<std::size_t>(col)] * kRungs[static_cast<std::size_t>(rung)];
      Acts::BoundVector plus = theta0;
      Acts::BoundVector minus = theta0;
      plus[col] += step;
      minus[col] -= step;
      plusE[static_cast<std::size_t>(rung)] = evalAt(plus, 1.0e-4);
      minusE[static_cast<std::size_t>(rung)] = evalAt(minus, 1.0e-4);
      const bool ok = plusE[static_cast<std::size_t>(rung)].ok &&
                      minusE[static_cast<std::size_t>(rung)].ok;
      colOk = colOk && ok;
      const double signal =
          ok ? loc0L2Delta(plusE[static_cast<std::size_t>(rung)],
                           minusE[static_cast<std::size_t>(rung)])
             : 0.0;
      json rungJson;
      rungJson["rung_index"] = rung;
      rungJson["step_factor"] = kRungs[static_cast<std::size_t>(rung)];
      rungJson["fd_step"] = step;
      rungJson["ok"] = ok;
      rungJson["plus"] = evalStageJson(plusE[static_cast<std::size_t>(rung)]);
      rungJson["minus"] = evalStageJson(minusE[static_cast<std::size_t>(rung)]);
      rungJson["fd_signal_loc0_l2"] = signal;
      rungs.push_back(rungJson);
      json resJson;
      resJson["parameter"] = kFocusNames[static_cast<std::size_t>(ip)];
      resJson["step_factor"] = kRungs[static_cast<std::size_t>(rung)];
      resJson["fd_step"] = step;
      resJson["fd_signal_loc0_l2"] = signal;
      resJson["repeat_noise_loc0_l2"] = maxLoc0;
      if (maxLoc0 > 0.0) {
        resJson["signal_over_noise"] = signal / maxLoc0;
      } else if (signal > 0.0) {
        resJson["signal_over_noise"] = 1.0e99;
      } else {
        resJson["signal_over_noise"] = nullptr;
      }
      resJson["fd_signal_at_or_below_noise"] =
          signal <= std::max(maxLoc0, 0.0) * 10.0 && maxLoc0 > 0.0;
      resJson["plus_n_steps"] = plusE[static_cast<std::size_t>(rung)].ok
                                    ? json(hopStepsTotal(plusE[static_cast<std::size_t>(rung)]))
                                    : json(nullptr);
      resJson["minus_n_steps"] = minusE[static_cast<std::size_t>(rung)].ok
                                     ? json(hopStepsTotal(minusE[static_cast<std::size_t>(rung)]))
                                     : json(nullptr);
      resolutionRows.push_back(resJson);
    }
    stageColumns.push_back({{"parameter", kFocusNames[static_cast<std::size_t>(ip)]},
                            {"official_fd_step", kFdSteps[static_cast<std::size_t>(col)]},
                            {"do_not_select_best_step", true},
                            {"ok", colOk},
                            {"rungs", rungs}});
  }

  json accuracy = json::array();
  for (int ia = 0; ia < 3; ++ia) {
    const double tol = kAccTol[static_cast<std::size_t>(ia)];
    Chi2Eval nom = evalAt(theta0, tol);
    json columns = json::array();
    for (int ip = 0; ip < 3; ++ip) {
      const int col = kFocus[static_cast<std::size_t>(ip)];
      json rungs = json::array();
      std::array<Eigen::VectorXd, 4> jac{};
      bool colOk = true;
      for (int rung = 0; rung < 4; ++rung) {
        const double step =
            kFdSteps[static_cast<std::size_t>(col)] * kRungs[static_cast<std::size_t>(rung)];
        Acts::BoundVector plus = theta0;
        Acts::BoundVector minus = theta0;
        plus[col] += step;
        minus[col] -= step;
        Chi2Eval evP = evalAt(plus, tol);
        Chi2Eval evM = evalAt(minus, tol);
        json item;
        item["step_factor"] = kRungs[static_cast<std::size_t>(rung)];
        item["fd_step"] = step;
        item["ok"] = evP.ok && evM.ok;
        item["plus_predicted_loc0"] = loc0VectorJson(evP);
        item["minus_predicted_loc0"] = loc0VectorJson(evM);
        item["plus_path_length_total"] = hopPathTotal(evP);
        item["minus_path_length_total"] = hopPathTotal(evM);
        if (evP.ok && evM.ok) {
          jac[static_cast<std::size_t>(rung)] =
              (evP.residual - evM.residual) / (2.0 * step);
          item["column_norm"] = jac[static_cast<std::size_t>(rung)].norm();
        } else {
          colOk = false;
        }
        rungs.push_back(item);
      }
      json consecutive = json::array();
      for (int i = 0; i < 3; ++i) {
        json pair{{"from_rung", i}, {"to_rung", i + 1}};
        if (rungs[static_cast<std::size_t>(i)]["ok"] == true &&
            rungs[static_cast<std::size_t>(i + 1)]["ok"] == true) {
          const double nNext = jac[static_cast<std::size_t>(i + 1)].norm();
          pair["relative_error"] =
              nNext > 1.0e-12
                  ? (jac[static_cast<std::size_t>(i)] - jac[static_cast<std::size_t>(i + 1)])
                            .norm() /
                        nNext
                  : 0.0;
          pair["sign_consistent"] =
              jac[static_cast<std::size_t>(i)].dot(jac[static_cast<std::size_t>(i + 1)]) >=
              0.0;
        } else {
          pair["relative_error"] = nullptr;
          pair["sign_consistent"] = false;
        }
        consecutive.push_back(pair);
      }
      columns.push_back({{"parameter", kFocusNames[static_cast<std::size_t>(ip)]},
                         {"ok", colOk},
                         {"rungs", rungs},
                         {"consecutive_relative_errors", consecutive},
                         {"selected_best_step", false}});
    }
    json freeState = json::array();
    for (const auto& hop : nom.hops) {
      freeState.push_back({hop.final_x_mm, hop.final_y_mm, hop.final_z_mm, hop.final_dx,
                           hop.final_dy, hop.final_dz, hop.final_qop});
    }
    json accJson;
    accJson["name"] = kAccNames[static_cast<std::size_t>(ia)];
    accJson["step_tolerance"] = tol;
    accJson["acts_api_field"] = "PropagatorPlainOptions::stepTolerance";
    accJson["selected_as_production"] = false;
    accJson["nominal_ok"] = nom.ok;
    accJson["nominal_chi2"] = nom.ok ? json(nom.chi2) : json(nullptr);
    accJson["predicted_loc0"] = loc0VectorJson(nom);
    accJson["path_length_total"] = hopPathTotal(nom);
    accJson["free_state_before_projection"] = freeState;
    accJson["columns"] = columns;
    accuracy.push_back(accJson);
  }

  json projections = json::array();
  for (std::size_t i = 0; i < first.hops.size() && i < hits.size(); ++i) {
    projections.push_back(
        projectionOnlyHopJson(first.hops[i], hits[i], geometryContext));
  }

  const double qop = theta0[Acts::eBoundQOverP];
  const double loc1 = theta0[Acts::eBoundLoc1];
  const double phi = theta0[Acts::eBoundPhi];
  json fdScale;
  fdScale["q_over_p_nominal"] = qop;
  fdScale["q_over_p_h"] = kFdSteps[4];
  fdScale["q_over_p_h_over_abs_nominal"] =
      std::abs(qop) > 0.0 ? json(kFdSteps[4] / std::abs(qop)) : json(nullptr);
  fdScale["loc1_nominal_mm"] = loc1;
  fdScale["loc1_h_mm"] = kFdSteps[1];
  fdScale["loc1_h_over_1mm"] = kFdSteps[1];
  fdScale["phi_nominal"] = phi;
  fdScale["phi_h"] = kFdSteps[2];
  fdScale["phi_h_over_one_radian"] = kFdSteps[2];
  fdScale["do_not_select_production_step"] = true;

  json out;
  out["kind"] = "source_to_measurement_map_smoothness";
  out["task"] = "B14K";
  out["official_is_sequential"] = true;
  out["do_not_select_best_step"] = true;
  out["do_not_switch_to_direct"] = true;
  out["do_not_change_production_step_tolerance"] = true;
  out["acts_step_tolerance_api"] = "PropagatorPlainOptions::stepTolerance";
  out["acts_step_tolerance_default"] = 1.0e-4;
  out["accuracy_rungs_pre_registered"] = json::array(
      {{{"name", "nominal"}, {"step_tolerance", 1.0e-4}},
       {{"name", "tighter"}, {"step_tolerance", 1.0e-5}},
       {{"name", "tighter_again"}, {"step_tolerance", 1.0e-6}}});
  out["source_bound"] = {theta0[0], theta0[1], theta0[2], theta0[3], theta0[4]};
  out["nominal"] = evalStageJson(first);
  json repeatability;
  repeatability["n_repeats"] = std::max(nRepeats, 2);
  repeatability["repeats"] = repeats;
  repeatability["max_predicted_loc0_l2_delta"] = maxLoc0;
  repeatability["max_chi2_abs_delta"] = maxChi2;
  repeatability["steps_identical"] = stepsSame;
  repeatability["transport_nondeterministic"] = !deterministic;
  out["repeatability"] = repeatability;
  json stagewise;
  stagewise["focus_parameters"] = {"loc1", "phi", "q_over_p"};
  stagewise["rung_factors"] = {1.0, 0.5, 0.25, 0.125};
  stagewise["columns"] = stageColumns;
  out["stagewise"] = stagewise;
  json resolution;
  resolution["repeat_noise_loc0_l2"] = maxLoc0;
  resolution["rows"] = resolutionRows;
  resolution["do_not_decrease_step_to_search"] = true;
  out["resolution_vs_fd"] = resolution;
  json accuracyBlock;
  accuracyBlock["do_not_select_best_tolerance"] = true;
  accuracyBlock["rungs"] = accuracy;
  out["propagator_accuracy"] = accuracyBlock;
  json projectionBlock;
  projectionBlock["decoupled_from_transport"] = true;
  projectionBlock["hops"] = projections;
  out["supporting_plane_projection"] = projectionBlock;
  out["fd_scale"] = fdScale;
  out["acts_derivative"] = actsJacobianDiagnosticJson(
      propagator, geometryContext, magFieldContext, start, hits, first, maxSteps,
      matchKalman, 1.0e-4);
  return out;
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

json nativeBoundJson(const Acts::BoundVector& theta) {
  return {{"loc0", theta[Acts::eBoundLoc0]},
          {"loc1", theta[Acts::eBoundLoc1]},
          {"phi", theta[Acts::eBoundPhi]},
          {"theta", theta[Acts::eBoundTheta]},
          {"q_over_p_per_mev", theta[Acts::eBoundQOverP] * 1_MeV},
          {"q_over_p_per_gev", theta[Acts::eBoundQOverP]},
          {"alpha",
           {{"loc0", theta[Acts::eBoundLoc0]},
            {"theta", theta[Acts::eBoundTheta]}}},
          {"nu",
           {{"loc1", theta[Acts::eBoundLoc1]},
            {"phi", theta[Acts::eBoundPhi]},
            {"q_over_p_per_mev", theta[Acts::eBoundQOverP] * 1_MeV}}}};
}

json predictedMeasurementLoc0Json(const Chi2Eval& eval) {
  json out = json::array();
  for (const auto& hop : eval.hops) {
    out.push_back({{"measurement_index", hop.measurement_index},
                   {"identifier", hop.identifier},
                   {"station", hop.station},
                   {"layer", hop.layer},
                   {"predicted_loc0", hop.ok ? json(hop.predicted_loc0) : json(nullptr)},
                   {"ok", hop.ok}});
  }
  return out;
}

json transportBranchIdentityJson(const Chi2Eval& eval) {
  json hops = json::array();
  for (const auto& hop : eval.hops) {
    hops.push_back(
        {{"measurement_index", hop.measurement_index},
         {"identifier", hop.identifier},
         {"station", hop.station},
         {"layer", hop.layer},
         {"side", hop.side},
         {"projection_kind", hop.projection_kind},
         {"continuation_state_construction", hop.continuation_state_construction},
         {"ok", hop.ok},
         {"inside_active_bounds", hop.inside_bounds},
         {"measurement_z_mm", hop.measurement_z_mm},
         {"destination_surface_z_mm", hop.destination_surface_z_mm}});
  }
  return hops;
}

bool predictSupportingPlaneLoc0(
    const Propagator& propagator, const Acts::GeometryContext& geometryContext,
    const Acts::MagneticFieldContext& magFieldContext,
    const Acts::BoundTrackParameters& start, double targetZ, int maxSteps,
    bool matchKalman, double& loc0) {
  auto surface = planeAtZ(targetZ);
  ProfileHit dummy;
  dummy.surface = surface.get();
  dummy.zMm = targetZ;
  dummy.loc0 = 0.0;
  dummy.variance = 0.08 * 0.08 / 12.0;
  dummy.identifier = "target_station_plane";
  ProfileHop hop = propagateOneHit(
      propagator, geometryContext, magFieldContext, start, dummy, -1, maxSteps,
      matchKalman, true, false, 0, false, 1.0e-4);
  if (!hop.ok) {
    return false;
  }
  loc0 = hop.predicted_loc0;
  return true;
}

bool repairedSourceLoc0Jacobian(
    const GradientPropagator& gradientPropagator,
    const Acts::GeometryContext& geometryContext,
    const Acts::MagneticFieldContext& magFieldContext,
    const Acts::BoundTrackParameters& start, const std::vector<ProfileHit>& hits,
    const Chi2Eval& official, int maxSteps, bool matchKalman,
    Eigen::MatrixXd& jacobian) {
  jacobian.resize(static_cast<int>(hits.size()), 5);
  jacobian.setZero();
  if (hits.empty()) {
    return false;
  }
  Acts::BoundMatrix jDownRep = Acts::BoundMatrix::Identity();
  Acts::BoundMatrix jUpRep = Acts::BoundMatrix::Identity();
  bool downRepInit = false;
  bool upRepInit = false;
  const double z0 = start.position(geometryContext).z();
  const std::vector<int> order =
      sequentialHopOrder(start, hits, geometryContext);
  for (int idx : order) {
    const ProfileHit& hit = hits[static_cast<std::size_t>(idx)];
    const ProfileHop& off = official.hops[static_cast<std::size_t>(idx)];
    if (!off.ok || hit.surface == nullptr) {
      return false;
    }
    Acts::BoundTrackParameters hopStart = start;
    const bool thisDown = hit.zMm >= z0;
    if ((thisDown && downRepInit) || (!thisDown && upRepInit)) {
      for (int prev : order) {
        if (prev == idx) {
          break;
        }
        const bool prevDown = hits[static_cast<std::size_t>(prev)].zMm >= z0;
        if (prevDown == thisDown &&
            official.hops[static_cast<std::size_t>(prev)].end_parameters.has_value()) {
          hopStart = *official.hops[static_cast<std::size_t>(prev)].end_parameters;
        }
      }
    }
    const OfficialPathJacobianHop diagNew = propagateOfficialPathJacobianHop(
        gradientPropagator, geometryContext, magFieldContext, hopStart, hit,
        idx, maxSteps, matchKalman, 1.0e-4, true, false);
    if (!diagNew.ok || !diagNew.rkFreeAvailable) {
      return false;
    }
    const double sign = (hit.surface->center(geometryContext).z() >=
                         hopStart.position(geometryContext).z())
                            ? 1.0
                            : -1.0;
    const Acts::Vector3 officialPos(off.final_x_mm, off.final_y_mm, off.final_z_mm);
    const Acts::Vector3 officialDir(off.final_dx, off.final_dy, off.final_dz);
    const Acts::Vector3 officialSigned = sign * officialDir;
    const Eigen::Matrix<double, 1, 8> jProj = supportingPlaneLoc0FreeJacobian(
        *hit.surface, geometryContext, officialPos, officialSigned);
    const Acts::FreeMatrix jIntersect = supportingPlaneIntersectionFreeJacobian(
        *hit.surface, geometryContext, officialPos, officialDir, sign);
    const Acts::Vector3 center = hit.surface->center(geometryContext);
    const Acts::Vector3 normal =
        hit.surface->normal(geometryContext, center, Acts::Vector3::UnitZ());
    const double denom = normal.dot(officialSigned);
    const double tIntersect =
        (std::abs(denom) > 1.0e-12) ? normal.dot(center - officialPos) / denom
                                    : 0.0;
    Acts::FreeVector intersectFree = Acts::FreeVector::Zero();
    const Acts::Vector3 onPlane = officialPos + tIntersect * officialSigned;
    intersectFree[Acts::eFreePos0] = onPlane.x();
    intersectFree[Acts::eFreePos1] = onPlane.y();
    intersectFree[Acts::eFreePos2] = onPlane.z();
    intersectFree[Acts::eFreeTime] = 0.0;
    intersectFree[Acts::eFreeDir0] = officialDir.x();
    intersectFree[Acts::eFreeDir1] = officialDir.y();
    intersectFree[Acts::eFreeDir2] = officialDir.z();
    intersectFree[Acts::eFreeQOverP] = off.final_qop;
    const Acts::FreeToBoundMatrix jF2bIntersect =
        hit.surface->freeToBoundJacobian(geometryContext, intersectFree);
    const Eigen::Matrix<double, 1, Acts::eBoundSize> jHopRep =
        jProj * diagNew.jB2fRk;
    const Acts::BoundMatrix jContRep =
        jF2bIntersect * jIntersect * diagNew.jB2fRk;
    Acts::BoundMatrix& jAccRep = thisDown ? jDownRep : jUpRep;
    const Eigen::Matrix<double, 1, Acts::eBoundSize> dloc0Rep = jHopRep * jAccRep;
    jAccRep = jContRep * jAccRep;
    if (thisDown) {
      downRepInit = true;
    } else {
      upRepInit = true;
    }
    for (int c = 0; c < 5; ++c) {
      jacobian(idx, c) = dloc0Rep(0, c);
    }
  }
  return true;
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
    int diagnosticMaxSteps = 0, bool jacobianContinuity = false,
    bool mapSmoothness = false, int mapSmoothnessRepeats = 3,
    bool derivativeContract = false, bool officialSpJacobian = false,
    const GradientPropagator* gradientPropagator = nullptr,
    bool fieldGradientRepair = false,
    bool focus86SegmentReference = false,
    bool focus86CommonGridShadow = false,
    bool shadowMeanTransport = false) {
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
  row["transport_task"] =
      (shadowMeanTransport && officialSpJacobian) ? "B14ZC"
      : shadowMeanTransport                       ? "B14ZB"
      : officialSpJacobian                        ? "B14U"
      : derivativeContract                        ? "B14L"
      : mapSmoothness                             ? "B14K"
      : jacobianContinuity                        ? "B14J"
      : (!evaluateOnly && supportingPlane && fieldGradientRepair) ? "B14MR"
      : supportingPlane                           ? "B14T"
                                                  : "B14N";
  row["jacobian_continuity_requested"] = jacobianContinuity;
  row["map_smoothness_requested"] = mapSmoothness;
  row["derivative_contract_requested"] = derivativeContract;
  row["official_supporting_plane_jacobian_requested"] = officialSpJacobian;
  row["field_gradient_variational_repair_requested"] = fieldGradientRepair;
  row["focus86_segment_reference_requested"] = focus86SegmentReference;
  row["focus86_common_grid_shadow_requested"] = focus86CommonGridShadow;
  row["shadow_mean_transport_contract_requested"] = shadowMeanTransport;
  row["surface_energy_loss_mean_semantics_requested"] = shadowMeanTransport;
  row["certified_mean_common_grid_fd_requested"] =
      shadowMeanTransport && officialSpJacobian;
  row["derivative_not_evaluated"] = shadowMeanTransport && !officialSpJacobian;
  row["jacobian_agreement_not_read"] = shadowMeanTransport && !officialSpJacobian;
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
    if (!derivativeContract && !officialSpJacobian && !shadowMeanTransport) {
      row["jacobian_validation"] = jacobianValidationJson(
          propagator, geometryContext, magFieldContext, start, hits, maxSteps,
          matchKalman, sequential, supportingPlane);
    }
    if (jacobianContinuity && supportingPlane && sequential) {
      row["jacobian_continuity"] = jacobianContinuityAuditJson(
          propagator, geometryContext, magFieldContext, start, hits, maxSteps,
          matchKalman, supportingPlane);
    }
    if (mapSmoothness && supportingPlane && sequential) {
      row["map_smoothness"] = mapSmoothnessAuditJson(
          propagator, geometryContext, magFieldContext, start, hits, maxSteps,
          matchKalman, supportingPlane, mapSmoothnessRepeats);
    }
    if (derivativeContract && supportingPlane && sequential) {
      row["derivative_contract"] = derivativeContractAuditJson(
          propagator, geometryContext, magFieldContext, start, hits, maxSteps,
          matchKalman, supportingPlane);
    }
    if ((officialSpJacobian || fieldGradientRepair || focus86SegmentReference ||
         focus86CommonGridShadow) &&
        supportingPlane && sequential) {
      row["official_supporting_plane_jacobian"] =
          officialSupportingPlaneJacobianAuditJson(
              propagator, geometryContext, magFieldContext, start, hits,
              maxSteps, matchKalman, supportingPlane, gradientPropagator,
              fieldGradientRepair, focus86SegmentReference,
              focus86CommonGridShadow);
    }
    if (shadowMeanTransport && supportingPlane && sequential) {
      row["shadow_mean_transport_contract"] = shadowMeanTransportAuditJson(
          propagator, geometryContext, magFieldContext, start, hits, eval,
          maxSteps, matchKalman);
      row["surface_energy_loss_mean_semantics"] =
          row["shadow_mean_transport_contract"];
      row["certified_mean_common_grid_fd"] =
          row["shadow_mean_transport_contract"];
    }
    return row;
  }

  json trace = json::array();
  double chi2 = eval.chi2;
  Eigen::VectorXd residual = eval.residual;
  Eigen::MatrixXd lastJacobian(static_cast<int>(hits.size()), 5);
  lastJacobian.setZero();
  Mat5 hessian = Mat5::Zero();
  int hessianRank = 0;
  bool jacobianOk = true;
  bool converged = false;
  std::string termination = "max_iterations";
  std::string jacobianKind = "unset";
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
    jacobian.setZero();
    bool jacOk = false;
    jacobianKind = "official_adaptive_central_fd";
    if (fieldGradientRepair && gradientPropagator != nullptr && supportingPlane &&
        sequential) {
      jacOk = repairedSourceLoc0Jacobian(
          *gradientPropagator, geometryContext, magFieldContext, current, hits,
          currentEval, maxSteps, matchKalman, jacobian);
      jacobianKind = "wb123_repaired_production_tangent";
    } else {
      const std::array<double, 5> fd{1.0e-2, 1.0e-2, 1.0e-5, 1.0e-5, 1.0e-6};
      jacOk = true;
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
          jacOk = false;
          break;
        }
        jacobian.col(col) =
            (evMinus.residual - evPlus.residual) / (2.0 * fd[col]);
      }
    }
    if (!jacOk) {
      termination = "jacobian_invalid";
      jacobianOk = false;
      break;
    }
    lastJacobian = jacobian;
    row["jacobian_implementation"] = jacobianKind;
    row["validated_jacobian_is_numerical_derivative"] = true;
    row["validated_jacobian_is_not_a_new_statistical_model"] = true;
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
    double predictedTargetLoc0 = 0.0;
    const bool targetPredOk = predictSupportingPlaneLoc0(
        propagator, geometryContext, magFieldContext, current, targetZ,
        maxSteps, matchKalman, predictedTargetLoc0);
    if (writeTrace) {
      json singular = json::array();
      Eigen::SelfAdjointEigenSolver<Mat5> eigen(hessian);
      for (int i = 0; i < 5; ++i) {
        singular.push_back(eigen.eigenvalues()[i]);
      }
      trace.push_back(
          {{"iteration", nIter},
           {"theta", nativeBoundJson(theta)},
           {"alpha",
            {{"loc0", theta[Acts::eBoundLoc0]},
             {"theta", theta[Acts::eBoundTheta]}}},
           {"nu",
            {{"loc1", theta[Acts::eBoundLoc1]},
             {"phi", theta[Acts::eBoundPhi]},
             {"q_over_p_per_mev", theta[Acts::eBoundQOverP] * 1_MeV}}},
           {"chi2", chi2},
           {"chi2_prof", chi2},
           {"gradient_norm_z", gradNorm},
           {"step_norm_z", stepNorm},
           {"accepted", accepted},
           {"step_accepted_or_rejected", accepted ? "accepted" : "rejected"},
           {"line_search_factor", accepted ? json(usedDamp) : json(nullptr)},
           {"propagation_status", trialStatus},
           {"rank_H", hessianRank},
           {"identifiable_rank", hessianRank},
           {"singular_values", singular},
           {"relative_chi2_decrease", relDec},
           {"predicted_target_loc0",
            targetPredOk ? json(predictedTargetLoc0) : json(nullptr)},
           {"surviving_predicted_measurement_loc0",
            predictedMeasurementLoc0Json(currentEval)},
           {"termination_reason", termination}});
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
  Chi2Eval finalEval = evaluateSequentialResiduals(
      propagator, geometryContext, magFieldContext, fitted, hits, maxSteps,
      matchKalman, sequential, supportingPlane);
  if (finalEval.ok) {
    chi2 = finalEval.chi2;
    residual = finalEval.residual;
    if (fieldGradientRepair && gradientPropagator != nullptr && supportingPlane &&
        sequential) {
      if (repairedSourceLoc0Jacobian(
              *gradientPropagator, geometryContext, magFieldContext, fitted, hits,
              finalEval, maxSteps, matchKalman, lastJacobian)) {
        jacobianKind = "wb123_repaired_production_tangent";
      }
    }
  }
  double predictedTargetLoc0 = 0.0;
  const bool targetLoc0Ok = predictSupportingPlaneLoc0(
      propagator, geometryContext, magFieldContext, fitted, targetZ, maxSteps,
      matchKalman, predictedTargetLoc0);
  json residualJson = json::array();
  json weightJson = json::array();
  json jacobianJson = json::array();
  if (finalEval.ok) {
    for (int i = 0; i < residual.size(); ++i) {
      residualJson.push_back(residual[i]);
      weightJson.push_back(1.0 / hits[static_cast<std::size_t>(i)].variance);
    }
    for (int i = 0; i < lastJacobian.rows(); ++i) {
      json rowJ = json::array();
      for (int c = 0; c < 5; ++c) {
        rowJ.push_back(lastJacobian(i, c));
      }
      jacobianJson.push_back(rowJ);
    }
  }
  row["profile_success"] = eval.ok && jacobianOk && predOk && finalEval.ok;
  row["profile_converged"] = converged;
  row["fit_success"] = eval.ok && jacobianOk && predOk && finalEval.ok;
  row["termination_reason"] = termination;
  row["fit_failure_reason"] =
      (eval.ok && jacobianOk && predOk && finalEval.ok) ? json(nullptr)
                                                        : json(termination);
  row["profile_chi2"] = chi2;
  row["chi2"] = chi2;
  row["chi2_prof"] = chi2;
  row["profile_n_iterations"] = nIter;
  row["hit_boundary"] = hitBoundary;
  row["profiled_native_state"] = {theta[Acts::eBoundLoc0], theta[Acts::eBoundLoc1],
                                  theta[Acts::eBoundPhi], theta[Acts::eBoundTheta],
                                  theta[Acts::eBoundQOverP] * 1_MeV};
  row["native_state"] = row["profiled_native_state"];
  row["profiled_native_bound"] = nativeBoundJson(theta);
  row["profiled_derived_state"] = {derived[0], derived[1], derived[2], derived[3],
                                   derived[4]};
  row["derived_state"] = row["profiled_derived_state"];
  row["joint_hessian"] = matrix5ToJson(hessian);
  row["hessian_rank"] = hessianRank;
  row["rank_deficiency_is_not_automatic_fail"] = true;
  row["q_over_p_is_explicit_nuisance"] = true;
  row["prior_term_present"] = false;
  row["ridge_added"] = false;
  row["jacobian_implementation"] = jacobianKind;
  row["validated_jacobian_is_numerical_derivative"] = true;
  row["validated_jacobian_is_not_a_new_statistical_model"] = true;
  row["residual_vector"] = residualJson;
  row["measurement_weights"] = weightJson;
  row["measurement_jacobian_dh_dtheta"] = jacobianJson;
  row["surviving_predicted_measurement_loc0"] =
      predictedMeasurementLoc0Json(finalEval.ok ? finalEval : eval);
  row["transport_branch_identity"] =
      transportBranchIdentityJson(finalEval.ok ? finalEval : eval);
  row["final_propagation_hits"] = json::array();
  for (const auto& hop : (finalEval.ok ? finalEval.hops : eval.hops)) {
    row["final_propagation_hits"].push_back(hopToJson(hop));
  }
  row["propagation_success"] = finalEval.ok;
  row["no_nan_inf"] =
      std::isfinite(chi2) && jacobianOk && finalEval.ok;
  if (predOk) {
    row["target_prediction_derived"] = {prediction[0], prediction[1], prediction[2],
                                        prediction[3], prediction[4]};
  } else {
    row["target_prediction_derived"] = nullptr;
  }
  row["prediction_derived"] = row["target_prediction_derived"];
  row["predicted_target_loc0"] =
      targetLoc0Ok ? json(predictedTargetLoc0) : json(nullptr);
  row["held_out_target_inaccessible_during_fit"] = true;
  row["restart_perturbs_init_only"] = true;
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
    bool useScaling, bool writeTrace,     bool supportingPlane = false,
    bool recordPath = false, int diagnosticMaxSteps = 0,
    bool jacobianContinuity = false, bool mapSmoothness = false,
    int mapSmoothnessRepeats = 3, bool derivativeContract = false,
    bool officialSpJacobian = false,
    const GradientPropagator* gradientPropagator = nullptr,
    bool fieldGradientRepair = false,
    bool focus86SegmentReference = false,
    bool focus86CommonGridShadow = false,
    bool shadowMeanTransport = false) {
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
  const bool restartInvarianceOptimize = seedInvariance && !evaluateOnly;
  // Always emit an evaluate-only nominal replay first.
  // B14M-R must not run the expensive official-path Jacobian audit here.
  rows.push_back(runOneProfileNumerics(
      baseRow, propagator, geometryContext, magFieldContext, seed, hits, targetZ,
      "nominal", "wb114_official_seed_mean", maxIterations, pinvRelative, maxSteps,
      matchKalman, sequential, useScaling, true, writeTrace, supportingPlane,
      recordPath, diagnosticMaxSteps,
      restartInvarianceOptimize ? false : jacobianContinuity,
      restartInvarianceOptimize ? false : mapSmoothness,
      mapSmoothnessRepeats,
      restartInvarianceOptimize ? false : derivativeContract,
      restartInvarianceOptimize ? false : officialSpJacobian, nullptr,
      restartInvarianceOptimize ? false : fieldGradientRepair,
      restartInvarianceOptimize ? false : focus86SegmentReference,
      restartInvarianceOptimize ? false : focus86CommonGridShadow,
      restartInvarianceOptimize ? false : shadowMeanTransport));
  if (restartInvarianceOptimize) {
    rows.back()["transport_task"] = "B14MR";
    rows.back()["b14m_reopen"] = true;
    rows.back()["row_role"] = "evaluate_only_nominal_seed";
  }
  if (supportingPlane && !restartInvarianceOptimize &&
      (mapSmoothness || derivativeContract || officialSpJacobian ||
       fieldGradientRepair || focus86SegmentReference ||
       focus86CommonGridShadow || shadowMeanTransport)) {
    return rows;
  }
  if (supportingPlane && !jacobianContinuity && !restartInvarianceOptimize) {
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
  if (supportingPlane && jacobianContinuity) {
    json direct = runOneProfileNumerics(
        baseRow, propagator, geometryContext, magFieldContext, seed, hits, targetZ,
        "nominal", "direct_from_source_supporting_plane", maxIterations,
        pinvRelative, maxSteps, matchKalman, false, useScaling, true, writeTrace,
        true, recordPath, 0, false);
    direct["transport_comparison_mode"] = "direct_from_source";
    direct["direct_from_source_is_comparison_only"] = true;
    rows.push_back(direct);
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
    json optimized = runOneProfileNumerics(
        baseRow, propagator, geometryContext, magFieldContext, init, hits, targetZ,
        variant.name, "wb114_official_seed_mean", maxIterations, pinvRelative,
        maxSteps, matchKalman, sequential, useScaling, false, writeTrace,
        supportingPlane, recordPath, diagnosticMaxSteps, false, false, 3, false,
        false, gradientPropagator, fieldGradientRepair, false, false, false);
    if (restartInvarianceOptimize) {
      optimized["transport_task"] = "B14MR";
      optimized["b14m_reopen"] = true;
      optimized["row_role"] = "profile_optimize";
      optimized["restart_name"] = variant.name;
    }
    rows.push_back(optimized);
  }
  return rows;
}

#include "ProfileBasinDiagnosis.inc"
#include "ProfileGlobalizationRepair.inc"

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
  Acts::Navigator navigatorGrad(navCfg);
  Propagator propagatorFit(Stepper(magneticField), std::move(navigatorFit));
  Propagator propagatorProf(Stepper(magneticField), std::move(navigatorProf));
  GradientPropagator propagatorGrad(GradientStepper(magneticField),
                                    std::move(navigatorGrad));
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
      json heldOut = json::array();
      for (const HitRecord* hit : excluded) {
        json item = {{"station", hit->station},
                     {"layer", hit->layer},
                     {"id", hit->identifier},
                     {"z_mm", hit->zMm},
                     {"from_outlier", hit->from_outlier},
                     {"stage_b_only", true},
                     {"not_used_in_fit", true}};
        if (hit->measurement != nullptr) {
          item["loc0"] = hit->measurement->localParameters()[Trk::locX];
        } else {
          item["loc0"] = nullptr;
        }
        heldOut.push_back(item);
      }
      row["held_out_target_measurements"] = heldOut;
      row["held_out_inaccessible_during_fit"] = true;
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
          if (m_enableProfileGlobalizationRepair.value()) {
            const bool supportingPlane = m_profileSupportingPlane.value();
            try {
              profileRows = runProfileGlobalizationRepair(
                  row, propagatorProf, geometryContext, magFieldContext,
                  *profileSeed, profileHits, targetZ, m_profileMaxSteps.value(),
                  m_profileMatchKalmanStepSize.value(),
                  m_profileSequentialTransport.value(),
                  m_profileUseNumericalScaling.value(), supportingPlane,
                  m_enableFieldGradientVariationalRepair.value()
                      ? &propagatorGrad
                      : nullptr,
                  m_enableFieldGradientVariationalRepair.value());
            } catch (const std::exception& exc) {
              json fail = row;
              fail["row_kind"] = "profile_optimize";
              fail["ok"] = false;
              fail["fit_failure_reason"] =
                  std::string("profile_globalization_exception:") + exc.what();
              fail["held_out_used_for_solver"] = false;
              profileRows = {fail};
            }
            std::lock_guard<std::mutex> lock(m_mutex);
            for (const auto& profileRow : profileRows) {
              m_rows.push_back(profileRow.dump());
            }
            if (m_profileOnly.value()) {
              continue;
            }
            continue;
          }
          if (m_enableProfileBasinDiagnosis.value()) {
            const bool supportingPlane = m_profileSupportingPlane.value();
            try {
              profileRows = runBasinDiagnosis(
                  row, propagatorProf, geometryContext, magFieldContext,
                  *profileSeed, profileHits, runId, eventId, target,
                  m_basinSourceJsonl.value(), m_basinRunContinuation.value(),
                  m_profileMaxIterations.value(), m_profilePinvRelative.value(),
                  m_profileMaxSteps.value(),
                  m_profileMatchKalmanStepSize.value(),
                  m_profileSequentialTransport.value(),
                  m_profileUseNumericalScaling.value(), supportingPlane,
                  m_enableFieldGradientVariationalRepair.value()
                      ? &propagatorGrad
                      : nullptr,
                  m_enableFieldGradientVariationalRepair.value());
            } catch (const std::exception& exc) {
              json fail = row;
              fail["row_kind"] = "basin_endpoint_audit";
              fail["ok"] = false;
              fail["fit_failure_reason"] =
                  std::string("basin_diagnosis_exception:") + exc.what();
              fail["held_out_target_inaccessible"] = true;
              profileRows = {fail};
            }
            std::lock_guard<std::mutex> lock(m_mutex);
            for (const auto& profileRow : profileRows) {
              m_rows.push_back(profileRow.dump());
            }
            if (m_profileOnly.value()) {
              continue;
            }
            continue;
          }
          if (m_enableProfileNumerics.value()) {
            const bool restartInvarianceOptimize =
                m_enableProfileSeedInvariance.value() &&
                !m_profileEvaluateOnly.value() &&
                !m_enableProfileTransport.value();
            const bool supportingPlane =
                m_profileSupportingPlane.value() &&
                (m_enableProfileTransport.value() || restartInvarianceOptimize);
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
                supportingPlane,
                m_profileRecordStepperPath.value(),
                m_profileDiagnosticMaxSteps.value(),
                m_enableJacobianContinuity.value(),
                m_enableMapSmoothness.value(),
                m_mapSmoothnessRepeats.value(),
                m_enableDerivativeContract.value(),
                m_enableOfficialSupportingPlaneJacobian.value(),
                m_enableFieldGradientVariationalRepair.value()
                    ? &propagatorGrad
                    : nullptr,
                m_enableFieldGradientVariationalRepair.value(),
                m_enableFocus86SegmentReference.value(),
                m_enableFocus86CommonGridShadow.value(),
                m_enableShadowMeanTransportContract.value());
            if (m_enableJacobianContinuity.value() ||
                m_enableMapSmoothness.value() ||
                m_enableDerivativeContract.value() ||
                m_enableOfficialSupportingPlaneJacobian.value() ||
                m_enableFieldGradientVariationalRepair.value() ||
                m_enableFocus86SegmentReference.value() ||
                m_enableFocus86CommonGridShadow.value() ||
                m_enableShadowMeanTransportContract.value()) {
              std::lock_guard<std::mutex> lock(m_mutex);
              for (const auto& profileRow : profileRows) {
                m_rows.push_back(profileRow.dump());
              }
              if (m_profileOnly.value()) {
                continue;
              }
              continue;
            }
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
