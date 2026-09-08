#include "CkfActsTransportDumpAlg.h"

#include "Acts/Definitions/Units.hpp"
#include "Acts/EventData/TrackParameters.hpp"
#include "Acts/EventData/detail/TransformationFreeToBound.hpp"
#include "Acts/Surfaces/PlaneSurface.hpp"
#include "StoreGate/ReadHandle.h"
#include "TrkEventPrimitives/ParamDefs.h"
#include "TrkParameters/TrackParameters.h"
#include "TrkSurfaces/Surface.h"
#include "TrkTrack/Track.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <optional>

using namespace Acts::UnitLiterals;
using json = nlohmann::json;

namespace {

constexpr int kExportDim = 5;
constexpr int kBoundDim = 6;

bool finiteValue(double value) { return std::isfinite(value); }

bool finiteArray(const std::array<double, kExportDim>& values) {
  for (double value : values) {
    if (!finiteValue(value)) {
      return false;
    }
  }
  return true;
}

json matrixToJson(const std::array<std::array<double, kExportDim>, kExportDim>& matrix) {
  json rows = json::array();
  for (int i = 0; i < kExportDim; ++i) {
    json row = json::array();
    for (int j = 0; j < kExportDim; ++j) {
      if (!finiteValue(matrix[i][j])) {
        return nullptr;
      }
      row.push_back(matrix[i][j]);
    }
    rows.push_back(row);
  }
  return rows;
}

json matrix4ToJson(const std::array<std::array<double, 4>, 4>& matrix) {
  json rows = json::array();
  for (int i = 0; i < 4; ++i) {
    json row = json::array();
    for (int j = 0; j < 4; ++j) {
      if (!finiteValue(matrix[i][j])) {
        return nullptr;
      }
      row.push_back(matrix[i][j]);
    }
    rows.push_back(row);
  }
  return rows;
}

std::array<std::array<double, 4>, 4> topLeft4(
    const std::array<std::array<double, kExportDim>, kExportDim>& matrix) {
  std::array<std::array<double, 4>, 4> out{};
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      out[i][j] = matrix[i][j];
    }
  }
  return out;
}

std::array<std::array<double, kExportDim>, kExportDim> multiply5(
    const std::array<std::array<double, kExportDim>, kExportDim>& left,
    const std::array<std::array<double, kExportDim>, kExportDim>& right) {
  std::array<std::array<double, kExportDim>, kExportDim> out{};
  for (int i = 0; i < kExportDim; ++i) {
    for (int j = 0; j < kExportDim; ++j) {
      double total = 0.0;
      for (int k = 0; k < kExportDim; ++k) {
        total += left[i][k] * right[k][j];
      }
      out[i][j] = total;
    }
  }
  return out;
}

std::array<std::array<double, kExportDim>, kExportDim> transpose5(
    const std::array<std::array<double, kExportDim>, kExportDim>& matrix) {
  std::array<std::array<double, kExportDim>, kExportDim> out{};
  for (int i = 0; i < kExportDim; ++i) {
    for (int j = 0; j < kExportDim; ++j) {
      out[i][j] = matrix[j][i];
    }
  }
  return out;
}

std::array<std::array<double, kExportDim>, kExportDim> subtract5(
    const std::array<std::array<double, kExportDim>, kExportDim>& left,
    const std::array<std::array<double, kExportDim>, kExportDim>& right) {
  std::array<std::array<double, kExportDim>, kExportDim> out{};
  for (int i = 0; i < kExportDim; ++i) {
    for (int j = 0; j < kExportDim; ++j) {
      out[i][j] = left[i][j] - right[i][j];
    }
  }
  return out;
}

double frobenius(
    const std::array<std::array<double, kExportDim>, kExportDim>& matrix) {
  double total = 0.0;
  for (int i = 0; i < kExportDim; ++i) {
    for (int j = 0; j < kExportDim; ++j) {
      total += matrix[i][j] * matrix[i][j];
    }
  }
  return std::sqrt(total);
}

json nativeMatrixToJson(const AmgSymMatrix(5)& matrix) {
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

std::shared_ptr<Acts::PlaneSurface> planeAtZ(double zMm) {
  return Acts::Surface::makeShared<Acts::PlaneSurface>(
      Acts::Vector3(0.0, 0.0, zMm), Acts::Vector3(0.0, 0.0, 1.0));
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

std::optional<Acts::BoundTrackParameters> actsFromExport(
    const std::array<double, kExportDim>& exportState,
    double zMm,
    const Acts::GeometryContext& geometryContext) {
  if (!finiteArray(exportState) || std::abs(exportState[4]) < 1.0e-18) {
    return std::nullopt;
  }
  const double pMev = 1.0 / std::abs(exportState[4]);
  const double denom = std::sqrt(1.0 + exportState[2] * exportState[2] +
                                 exportState[3] * exportState[3]);
  if (!finiteValue(denom) || denom < 1.0e-12) {
    return std::nullopt;
  }
  const double pz = pMev / denom;
  const Acts::Vector3 position(exportState[0], exportState[1], zMm);
  const Acts::Vector3 momentum(exportState[2] * pz, exportState[3] * pz, pz);
  auto surface = planeAtZ(zMm);
  const auto bound = Acts::detail::transformFreeToBoundParameters(
      position, 0.0, momentum, exportState[4] / 1_MeV, *surface,
      geometryContext);
  if (!bound.ok()) {
    return std::nullopt;
  }
  return Acts::BoundTrackParameters(
      surface, bound.value(), std::nullopt, Acts::ParticleHypothesis::muon());
}

std::optional<Acts::BoundTrackParameters> actsFromTrk(
    const Trk::TrackParameters& parameters,
    const Acts::GeometryContext& geometryContext) {
  const Amg::Vector3D& position = parameters.position();
  const Amg::Vector3D& momentum = parameters.momentum();
  const AmgVector(5)& nativeParameters = parameters.parameters();
  const double nativeQOverP = nativeParameters[Trk::qOverP];
  if (!std::isfinite(position.x()) || !std::isfinite(position.y()) ||
      !std::isfinite(position.z()) || !std::isfinite(momentum.x()) ||
      !std::isfinite(momentum.y()) || !std::isfinite(momentum.z()) ||
      !std::isfinite(nativeQOverP) || std::abs(nativeQOverP) < 1.0e-18) {
    return std::nullopt;
  }

  auto surface = planeAtZ(position.z());
  auto bound = Acts::detail::transformFreeToBoundParameters(
      position, 0.0, momentum, nativeQOverP / 1_MeV, *surface, geometryContext);
  if (!bound.ok()) {
    return std::nullopt;
  }

  std::optional<Acts::BoundSquareMatrix> covariance = std::nullopt;
  const AmgSymMatrix(5)* nativeCovariance = parameters.covariance();
  if (nativeCovariance != nullptr) {
    const std::array<double, 5> values{
        nativeParameters[Trk::loc1], nativeParameters[Trk::loc2],
        nativeParameters[Trk::phi], nativeParameters[Trk::theta], nativeQOverP};
    const std::array<double, 5> steps{
        1.0e-4, 1.0e-4, 1.0e-6, 1.0e-6,
        std::max(std::abs(nativeQOverP) * 1.0e-5, 1.0e-12)};
    std::array<std::array<double, 5>, kBoundDim> jacobian{};
    const Trk::Surface& nativeSurface = parameters.associatedSurface();
    bool transformValid = true;
    for (std::size_t column = 0; column < values.size(); ++column) {
      std::array<double, 5> plus = values;
      std::array<double, 5> minus = values;
      plus[column] += steps[column];
      minus[column] -= steps[column];
      auto plusParameters = nativeSurface.createUniqueTrackParameters(
          plus[Trk::loc1], plus[Trk::loc2], plus[Trk::phi], plus[Trk::theta],
          plus[Trk::qOverP]);
      auto minusParameters = nativeSurface.createUniqueTrackParameters(
          minus[Trk::loc1], minus[Trk::loc2], minus[Trk::phi],
          minus[Trk::theta], minus[Trk::qOverP]);
      if (!plusParameters || !minusParameters) {
        transformValid = false;
        break;
      }
      const auto plusBound = Acts::detail::transformFreeToBoundParameters(
          plusParameters->position(), 0.0, plusParameters->momentum(),
          plus[Trk::qOverP] / 1_MeV, *surface, geometryContext);
      const auto minusBound = Acts::detail::transformFreeToBoundParameters(
          minusParameters->position(), 0.0, minusParameters->momentum(),
          minus[Trk::qOverP] / 1_MeV, *surface, geometryContext);
      if (!plusBound.ok() || !minusBound.ok()) {
        transformValid = false;
        break;
      }
      for (std::size_t row = 0; row < jacobian.size(); ++row) {
        double difference = plusBound.value()[row] - minusBound.value()[row];
        if (row == Acts::eBoundPhi) {
          difference = std::atan2(std::sin(difference), std::cos(difference));
        }
        jacobian[row][column] = difference / (2.0 * steps[column]);
      }
    }
    if (transformValid) {
      Acts::BoundSquareMatrix converted = Acts::BoundSquareMatrix::Zero();
      for (std::size_t row = 0; row < jacobian.size(); ++row) {
        for (std::size_t column = 0; column < jacobian.size(); ++column) {
          for (std::size_t nativeRow = 0; nativeRow < values.size(); ++nativeRow) {
            for (std::size_t nativeColumn = 0; nativeColumn < values.size();
                 ++nativeColumn) {
              converted(row, column) += jacobian[row][nativeRow] *
                  (*nativeCovariance)(nativeRow, nativeColumn) *
                  jacobian[column][nativeColumn];
            }
          }
        }
      }
      if (converted.allFinite()) {
        covariance = converted;
      }
    }
  }

  return Acts::BoundTrackParameters(
      surface, bound.value(), covariance, Acts::ParticleHypothesis::muon());
}

std::optional<std::array<std::array<double, kExportDim>, kExportDim>>
exportCovariance(const Acts::BoundTrackParameters& parameters,
                 const Acts::GeometryContext& geometryContext) {
  if (!parameters.covariance().has_value()) {
    return std::nullopt;
  }
  const Acts::BoundVector values = parameters.parameters();
  const std::array<double, kBoundDim> steps{
      1.0e-4, 1.0e-4, 1.0e-6, 1.0e-6,
      std::max(std::abs(values[Acts::eBoundQOverP]) * 1.0e-5, 1.0e-12), 1.0e-3};
  std::array<std::array<double, kBoundDim>, kExportDim> jacobian{};
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
    if (!finiteArray(plusState) || !finiteArray(minusState)) {
      return std::nullopt;
    }
    for (std::size_t row = 0; row < jacobian.size(); ++row) {
      jacobian[row][column] =
          (plusState[row] - minusState[row]) / (2.0 * steps[column]);
    }
  }

  const Acts::BoundSquareMatrix& boundCovariance = parameters.covariance().value();
  std::array<std::array<double, kExportDim>, kExportDim> covariance{};
  for (std::size_t row = 0; row < covariance.size(); ++row) {
    for (std::size_t column = 0; column < covariance.size(); ++column) {
      double total = 0.0;
      for (std::size_t boundRow = 0; boundRow < steps.size(); ++boundRow) {
        for (std::size_t boundColumn = 0; boundColumn < steps.size();
             ++boundColumn) {
          total += jacobian[row][boundRow] * boundCovariance(boundRow, boundColumn) *
                   jacobian[column][boundColumn];
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

std::optional<Acts::BoundTrackParameters> propagateToZ(
    const IFaserActsExtrapolationTool& tool,
    const EventContext& ctx,
    const Acts::BoundTrackParameters& start,
    double sourceZ,
    double targetZ) {
  const auto surface = planeAtZ(targetZ);
  const Acts::Direction direction =
      targetZ >= sourceZ ? Acts::Direction::Forward : Acts::Direction::Backward;
  return tool.propagate(ctx, start, *surface, direction);
}

json modelPayload(bool success,
                  const std::string& reason,
                  const std::array<double, kExportDim>* state,
                  const std::array<std::array<double, kExportDim>, kExportDim>* cov5) {
  json payload;
  payload["success"] = success;
  if (!success) {
    payload["reason"] = reason;
    return payload;
  }
  payload["state_xy_tx_ty"] = {(*state)[0], (*state)[1], (*state)[2], (*state)[3]};
  payload["state_xy_tx_ty_qoverp"] = {(*state)[0], (*state)[1], (*state)[2],
                                      (*state)[3], (*state)[4]};
  payload["covariance_5x5"] = matrixToJson(*cov5);
  payload["covariance_4x4"] = matrix4ToJson(topLeft4(*cov5));
  return payload;
}

json stateDefinition() {
  return {
      {"parameters", json::array({"x_mm", "y_mm", "tx", "ty", "q_over_p_per_mev"})},
      {"units", json::array({"mm", "mm", "1", "1", "1/MeV"})},
      {"frame", "global_cartesian_slopes_at_surface_z"},
      {"native_athena",
       json::array({"loc1_mm", "loc2_mm", "phi", "theta", "q_over_p_per_mev"})},
      {"acts_bound",
       json::array({"loc0", "loc1", "phi", "theta", "q_over_p_per_GeV", "time"})},
      {"q_over_p_signed", true},
      {"particle_hypothesis", "Acts::ParticleHypothesis::muon()"},
  };
}

}  // namespace

CkfActsTransportDumpAlg::CkfActsTransportDumpAlg(const std::string& name,
                                               ISvcLocator* pSvcLocator)
    : AthAlgorithm(name, pSvcLocator) {}

StatusCode CkfActsTransportDumpAlg::initialize() {
  if (m_outputJsonl.empty() || m_sourceId.empty()) {
    ATH_MSG_ERROR("OutputJsonl and SourceId are required");
    return StatusCode::FAILURE;
  }
  ATH_CHECK(m_trackKey.initialize());
  ATH_CHECK(m_eventKey.initialize());
  ATH_CHECK(m_toolNoNoise.retrieve());
  ATH_CHECK(m_toolWithNoise.retrieve());
  ATH_CHECK(m_trackingGeometryTool.retrieve());
  ATH_MSG_INFO("CKF ACTS transport dump helper ready: " << m_outputJsonl.value());
  return StatusCode::SUCCESS;
}

StatusCode CkfActsTransportDumpAlg::execute() {
  const EventContext& ctx = getContext();
  SG::ReadHandle<TrackCollection> tracks(m_trackKey, ctx);
  SG::ReadHandle<xAOD::EventInfo> eventInfo(m_eventKey, ctx);
  if (!tracks.isValid() || !eventInfo.isValid()) {
    return StatusCode::SUCCESS;
  }

  const Acts::GeometryContext geometryContext =
      m_trackingGeometryTool->getGeometryContext(ctx).context();
  const int runId = static_cast<int>(eventInfo->runNumber());
  const int eventId = static_cast<int>(eventInfo->eventNumber());

  int trackIndex = -1;
  for (const Trk::Track* track : *tracks) {
    ++trackIndex;
    if (track == nullptr || track->trackParameters() == nullptr ||
        track->trackParameters()->empty()) {
      continue;
    }
    const Trk::TrackParameters* params = track->trackParameters()->front();
    if (params == nullptr || params->covariance() == nullptr) {
      continue;
    }

    const Amg::Vector3D& position = params->position();
    const Amg::Vector3D& momentum = params->momentum();
    const AmgVector(5)& native = params->parameters();
    const double pz = momentum.z();
    if (std::abs(pz) < 1.0e-18) {
      continue;
    }
    const double pMev = std::sqrt(momentum.x() * momentum.x() +
                                  momentum.y() * momentum.y() +
                                  momentum.z() * momentum.z());
    const double sourceZ = position.z();
    int sourceStation = 0;
    double best = std::abs(m_stationZmm[0] - sourceZ);
    for (int station = 1; station < static_cast<int>(m_stationZmm.size());
         ++station) {
      const double distance = std::abs(m_stationZmm[station] - sourceZ);
      if (distance < best) {
        best = distance;
        sourceStation = station;
      }
    }

    const auto start = actsFromTrk(*params, geometryContext);
    const auto cin = start ? exportCovariance(*start, geometryContext) : std::nullopt;
    const auto startExport =
        start ? std::optional<std::array<double, kExportDim>>(
                    exportFromBound(*start, geometryContext))
              : std::nullopt;

    for (int target : m_targetStations) {
      if (target < 0 || target >= static_cast<int>(m_stationZmm.size())) {
        continue;
      }
      const double targetZ = m_stationZmm[target];
      if (targetZ <= sourceZ && sourceStation >= target) {
        continue;
      }

      json row;
      row["source_id"] = m_sourceId.value();
      row["run_id"] = runId;
      row["event_id"] = eventId;
      row["track_index"] = trackIndex;
      row["collection"] = m_trackKey.key();
      row["is_truth"] = false;
      row["helper"] = "CkfActsTransportDumpAlg";
      row["helper_language"] = "C++";
      row["state_definition"] = stateDefinition();
      row["native_state"] = {native[Trk::loc1], native[Trk::loc2], native[Trk::phi],
                             native[Trk::theta], native[Trk::qOverP]};
      row["native_covariance"] = nativeMatrixToJson(*params->covariance());
      row["q_over_p_per_mev"] = native[Trk::qOverP];
      row["p_mev"] = pMev;
      row["charge"] = params->charge();
      row["source_z_mm"] = sourceZ;
      row["source_station"] = sourceStation;
      row["target_station"] = target;
      row["target_z_mm"] = targetZ;
      row["derived_state"] = {position.x(), position.y(), momentum.x() / pz,
                              momentum.y() / pz, native[Trk::qOverP]};
      row["surface"] = {
          {"type", "plane"},
          {"origin_mm", json::array({0.0, 0.0, targetZ})},
          {"normal", json::array({0.0, 0.0, 1.0})},
          {"z_mm", targetZ},
      };
      row["geometry_hash"] = m_geometryHash.value();
      row["field_hash"] = m_fieldHash.value();
      row["material_hash"] = m_materialHash.value();
      row["material_map_hash"] = m_materialHash.value();
      row["conditions_hash"] = m_conditionsHash.value();
      row["process_noise_artificial_scale"] = false;

      if (!start || !cin || !startExport) {
        row["input_covariance"] = nullptr;
        row["transport_jacobian"] = nullptr;
        row["output_covariance_no_material"] = nullptr;
        row["output_covariance_with_material"] = nullptr;
        row["process_noise"] = nullptr;
        row["model0_no_process_noise"] = modelPayload(
            false, start ? "cin_unavailable" : "acts_start_unavailable", nullptr,
            nullptr);
        row["model1_acts_process_noise"] =
            modelPayload(false, "acts_start_unavailable", nullptr, nullptr);
        std::lock_guard<std::mutex> lock(m_mutex);
        m_rows.push_back(row.dump());
        continue;
      }

      row["input_covariance"] = matrixToJson(*cin);

      const auto out0 = propagateToZ(*m_toolNoNoise, ctx, *start, sourceZ, targetZ);
      const auto out1 = propagateToZ(*m_toolWithNoise, ctx, *start, sourceZ, targetZ);
      std::optional<std::array<double, kExportDim>> state0;
      std::optional<std::array<std::array<double, kExportDim>, kExportDim>> cov0;
      std::optional<std::array<double, kExportDim>> state1;
      std::optional<std::array<std::array<double, kExportDim>, kExportDim>> cov1;
      if (out0) {
        state0 = exportFromBound(*out0, geometryContext);
        cov0 = exportCovariance(*out0, geometryContext);
      }
      if (out1) {
        state1 = exportFromBound(*out1, geometryContext);
        cov1 = exportCovariance(*out1, geometryContext);
      }

      std::array<std::array<double, kExportDim>, kExportDim> jacobian{};
      bool jacobianOk = true;
      const std::array<double, kExportDim> steps{
          1.0e-3, 1.0e-3, 1.0e-6, 1.0e-6,
          std::max(std::abs((*startExport)[4]) * 1.0e-5, 1.0e-12)};
      for (int column = 0; column < kExportDim; ++column) {
        std::array<double, kExportDim> plus = *startExport;
        std::array<double, kExportDim> minus = *startExport;
        plus[column] += steps[column];
        minus[column] -= steps[column];
        const auto startPlus = actsFromExport(plus, sourceZ, geometryContext);
        const auto startMinus = actsFromExport(minus, sourceZ, geometryContext);
        if (!startPlus || !startMinus) {
          jacobianOk = false;
          break;
        }
        const auto outPlus =
            propagateToZ(*m_toolNoNoise, ctx, *startPlus, sourceZ, targetZ);
        const auto outMinus =
            propagateToZ(*m_toolNoNoise, ctx, *startMinus, sourceZ, targetZ);
        if (!outPlus || !outMinus) {
          jacobianOk = false;
          break;
        }
        const auto plusState = exportFromBound(*outPlus, geometryContext);
        const auto minusState = exportFromBound(*outMinus, geometryContext);
        if (!finiteArray(plusState) || !finiteArray(minusState)) {
          jacobianOk = false;
          break;
        }
        for (int rowIdx = 0; rowIdx < kExportDim; ++rowIdx) {
          jacobian[rowIdx][column] =
              (plusState[rowIdx] - minusState[rowIdx]) / (2.0 * steps[column]);
        }
      }

      if (jacobianOk) {
        row["transport_jacobian"] = matrixToJson(jacobian);
        const auto c0FromF =
            multiply5(multiply5(jacobian, *cin), transpose5(jacobian));
        row["output_covariance_no_material_from_jacobian"] = matrixToJson(c0FromF);
        if (cov0) {
          const auto residual = subtract5(*cov0, c0FromF);
          const double denom = frobenius(*cov0);
          if (denom > 0.0) {
            row["c0_jacobian_frobenius_rel"] = frobenius(residual) / denom;
          } else {
            row["c0_jacobian_frobenius_rel"] = nullptr;
          }
        }
      } else {
        row["transport_jacobian"] = nullptr;
      }

      if (state0 && cov0) {
        row["output_covariance_no_material"] = matrixToJson(*cov0);
        row["model0_no_process_noise"] =
            modelPayload(true, "", &*state0, &*cov0);
      } else {
        row["output_covariance_no_material"] = nullptr;
        row["model0_no_process_noise"] =
            modelPayload(false, out0 ? "export_covariance_failed" : "propagate0_failed",
                         nullptr, nullptr);
      }

      if (state1 && cov1) {
        row["output_covariance_with_material"] = matrixToJson(*cov1);
        row["model1_acts_process_noise"] =
            modelPayload(true, "", &*state1, &*cov1);
      } else {
        row["output_covariance_with_material"] = nullptr;
        row["model1_acts_process_noise"] =
            modelPayload(false, out1 ? "export_covariance_failed" : "propagate1_failed",
                         nullptr, nullptr);
      }

      if (cov0 && cov1) {
        const auto processNoise = subtract5(*cov1, *cov0);
        row["process_noise"] = matrixToJson(processNoise);
        row["q_acts"] = row["process_noise"];
        row["q_frobenius"] = frobenius(processNoise);
        row["q_max_abs"] = [&processNoise]() {
          double bestAbs = 0.0;
          for (int i = 0; i < kExportDim; ++i) {
            for (int j = 0; j < kExportDim; ++j) {
              bestAbs = std::max(bestAbs, std::abs(processNoise[i][j]));
            }
          }
          return bestAbs;
        }();
      } else {
        row["process_noise"] = nullptr;
        row["q_acts"] = nullptr;
      }

      std::lock_guard<std::mutex> lock(m_mutex);
      m_rows.push_back(row.dump());
    }
  }
  return StatusCode::SUCCESS;
}

StatusCode CkfActsTransportDumpAlg::finalize() {
  const std::string dest = m_outputJsonl.value();
  const std::string tmp = dest + ".tmp";
  {
    std::ofstream out(tmp);
    if (!out) {
      ATH_MSG_ERROR("Cannot write " << tmp);
      return StatusCode::FAILURE;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    for (const std::string& line : m_rows) {
      out << line << '\n';
    }
  }
  if (std::rename(tmp.c_str(), dest.c_str()) != 0) {
    ATH_MSG_ERROR("Cannot replace " << dest);
    return StatusCode::FAILURE;
  }
  ATH_MSG_INFO("Wrote " << m_rows.size() << " CKF ACTS transport rows to " << dest);
  return StatusCode::SUCCESS;
}
