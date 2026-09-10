#include "CkfThreeStToIftPredictionAlg.h"

#include "Acts/Definitions/Units.hpp"
#include "Acts/EventData/TrackParameters.hpp"
#include "Acts/EventData/detail/TransformationFreeToBound.hpp"
#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/Surfaces/BoundaryCheck.hpp"
#include "Acts/Surfaces/PlaneSurface.hpp"
#include "Identifier/Identifier.h"
#include "StoreGate/ReadHandle.h"
#include "TrackerIdentifier/FaserSCT_ID.h"
#include "TrackerPrepRawData/FaserSCT_Cluster.h"
#include "TrackerPrepRawData/FaserSCT_ClusterCollection.h"
#include "TrackerRIO_OnTrack/FaserSCT_ClusterOnTrack.h"
#include "TrkEventPrimitives/ParamDefs.h"
#include "TrkParameters/TrackParameters.h"
#include "TrkSurfaces/Surface.h"
#include "TrkTrack/Track.h"
#include "TrkTrack/TrackStateOnSurface.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <optional>
#include <set>
#include <sstream>

using namespace Acts::UnitLiterals;
using json = nlohmann::json;

namespace {

constexpr int kExportDim = 5;
constexpr int kBoundDim = 6;
constexpr int kIftStation = 0;

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

std::optional<Acts::BoundTrackParameters> propagateToSurface(
    const IFaserActsExtrapolationTool& tool,
    const EventContext& ctx,
    const Acts::BoundTrackParameters& start,
    const Acts::Surface& target,
    double sourceZ,
    double targetZ) {
  const Acts::Direction direction =
      targetZ >= sourceZ ? Acts::Direction::Forward : Acts::Direction::Backward;
  return tool.propagate(ctx, start, target, direction);
}

json modelPayload(bool success, const std::string& reason,
                  const std::array<double, kExportDim>* state,
                  const std::array<std::array<double, kExportDim>, kExportDim>* cov5) {
  json payload;
  payload["success"] = success;
  if (!success) {
    payload["reason"] = reason;
    payload["state_xy_tx_ty_qoverp"] = nullptr;
    payload["covariance_5x5"] = nullptr;
    return payload;
  }
  payload["reason"] = nullptr;
  payload["state_xy_tx_ty_qoverp"] = {(*state)[0], (*state)[1], (*state)[2],
                                      (*state)[3], (*state)[4]};
  payload["covariance_5x5"] = cov5 == nullptr ? json(nullptr) : matrixToJson(*cov5);
  return payload;
}

json stateDefinition() {
  return {
      {"native_athena",
       json::array({"loc1_mm", "loc2_mm", "phi", "theta", "q_over_p_per_mev"})},
      {"export_parameters",
       json::array({"x_mm", "y_mm", "tx", "ty", "q_over_p_per_mev"})},
      {"units", json::array({"mm", "mm", "1", "1", "1/MeV"})},
      {"frame", "global_cartesian_slopes_at_surface_z"},
      {"native_parameter_type", "Trk::CurvilinearParameters"},
      {"acts_bound",
       json::array({"loc0", "loc1", "phi", "theta", "q_over_p_per_GeV", "time"})},
      {"q_over_p_signed", true},
      {"particle_hypothesis", "Acts::ParticleHypothesis::muon()"},
      {"prediction_uses_ift_measurement", false},
  };
}

json residualDefinition() {
  return {
      {"kind", "independent_ift_cluster_local_loc0"},
      {"measurement", "FaserSCT_Cluster.localPosition Trk::locX"},
      {"prediction", "Acts BoundTrackParameters loc0 on the IFT wafer surface"},
      {"formula", "r = loc0_cluster - loc0_predicted"},
      {"unit", "mm"},
      {"source", "SCT_ClusterContainer"},
      {"station_decoder", "FaserSCT_ID"},
      {"prediction_uses_ift_measurement", false},
      {"inside_bounds_required_for_residual", false},
  };
}

struct ClusterHit {
  int station{-1};
  int layer{-1};
  int phi_module{-1};
  int eta_module{-1};
  int side{-1};
  std::string identifier;
  Identifier id;
  double loc0{0.0};
  double loc1{0.0};
  double global_x{0.0};
  double global_y{0.0};
  double global_z{0.0};
  bool reconstruction_associated{false};
};

const Tracker::FaserSCT_ClusterOnTrack* asClusterOnTrack(
    const Trk::MeasurementBase* meas) {
  return dynamic_cast<const Tracker::FaserSCT_ClusterOnTrack*>(meas);
}

void collectStations(const Trk::Track& track, const FaserSCT_ID& idHelper,
                     json& motStations, json& tsosStations, int& nMotIft,
                     int& nTsosIft, int& nOutlier) {
  motStations = json::array();
  tsosStations = json::array();
  nMotIft = 0;
  nTsosIft = 0;
  nOutlier = 0;
  if (track.measurementsOnTrack() != nullptr) {
    for (const Trk::MeasurementBase* meas : *track.measurementsOnTrack()) {
      const auto* cluster = asClusterOnTrack(meas);
      if (cluster == nullptr) {
        continue;
      }
      const int station = idHelper.station(cluster->identify());
      motStations.push_back(station);
      if (station == kIftStation) {
        ++nMotIft;
      }
    }
  }
  if (track.trackStateOnSurfaces() != nullptr) {
    for (const Trk::TrackStateOnSurface* tsos : *track.trackStateOnSurfaces()) {
      if (tsos == nullptr) {
        continue;
      }
      const auto* cluster = asClusterOnTrack(tsos->measurementOnTrack());
      if (cluster == nullptr) {
        continue;
      }
      const int station = idHelper.station(cluster->identify());
      tsosStations.push_back(station);
      if (station == kIftStation) {
        ++nTsosIft;
      }
      if (tsos->type(Trk::TrackStateOnSurface::Outlier)) {
        ++nOutlier;
      }
    }
  }
}

std::set<std::string> associatedIftIds(const TrackCollection* tracks,
                                       const FaserSCT_ID& idHelper) {
  std::set<std::string> ids;
  if (tracks == nullptr || tracks->size() != 1) {
    return ids;
  }
  const Trk::Track* track = tracks->front();
  if (track == nullptr || track->trackStateOnSurfaces() == nullptr) {
    return ids;
  }
  for (const Trk::TrackStateOnSurface* tsos : *track->trackStateOnSurfaces()) {
    if (tsos == nullptr || tsos->measurementOnTrack() == nullptr) {
      continue;
    }
    const auto* cluster = asClusterOnTrack(tsos->measurementOnTrack());
    if (cluster == nullptr) {
      continue;
    }
    const Identifier id = cluster->identify();
    if (idHelper.station(id) == kIftStation) {
      ids.insert(compactIdentifier(id));
    }
  }
  return ids;
}

bool writeLines(const std::string& dest, const std::vector<std::string>& rows) {
  if (dest.empty()) {
    return true;
  }
  std::ifstream existing(dest);
  if (existing.good()) {
    return false;
  }
  const std::string tmp = dest + ".tmp";
  {
    std::ofstream out(tmp);
    if (!out) {
      return false;
    }
    for (const std::string& line : rows) {
      out << line << '\n';
    }
  }
  return std::rename(tmp.c_str(), dest.c_str()) == 0;
}

}  // namespace

CkfThreeStToIftPredictionAlg::CkfThreeStToIftPredictionAlg(
    const std::string& name, ISvcLocator* pSvcLocator)
    : AthAlgorithm(name, pSvcLocator) {}

StatusCode CkfThreeStToIftPredictionAlg::initialize() {
  if (m_outputJsonl.empty() || m_sourceId.empty()) {
    ATH_MSG_ERROR("OutputJsonl and SourceId are required");
    return StatusCode::FAILURE;
  }
  if (!m_allowBackwardToIft) {
    ATH_MSG_ERROR("AllowBackwardToIft must be true; WB98 skip must not apply");
    return StatusCode::FAILURE;
  }
  if (m_targetStation != kIftStation) {
    ATH_MSG_ERROR("TargetStation must be IFT station 0");
    return StatusCode::FAILURE;
  }
  ATH_CHECK(m_trackKey.initialize());
  ATH_CHECK(m_fourStationKey.initialize());
  ATH_CHECK(m_clusterKey.initialize());
  ATH_CHECK(m_eventKey.initialize());
  ATH_CHECK(m_toolNoNoise.retrieve());
  ATH_CHECK(m_toolWithNoise.retrieve());
  ATH_CHECK(m_trackingGeometryTool.retrieve());
  ATH_CHECK(detStore()->retrieve(m_idHelper, "FaserSCT_ID"));
  ATH_MSG_INFO("3ST→IFT prediction helper ready: " << m_outputJsonl.value());
  return StatusCode::SUCCESS;
}

StatusCode CkfThreeStToIftPredictionAlg::execute() {
  const EventContext& ctx = getContext();
  SG::ReadHandle<TrackCollection> tracks(m_trackKey, ctx);
  SG::ReadHandle<xAOD::EventInfo> eventInfo(m_eventKey, ctx);
  if (!tracks.isValid() || !eventInfo.isValid()) {
    return StatusCode::SUCCESS;
  }

  const Acts::GeometryContext geometryContext =
      m_trackingGeometryTool->getGeometryContext(ctx).context();
  auto identifierMap = m_trackingGeometryTool->getIdentifierMap();
  auto trackingGeometry = m_trackingGeometryTool->trackingGeometry();
  const int runId = static_cast<int>(eventInfo->runNumber());
  const int eventId = static_cast<int>(eventInfo->eventNumber());
  const double targetZ = m_stationZmm[m_targetStation];

  std::vector<ClusterHit> iftClusters;
  bool clusterUnavailable = false;
  SG::ReadHandle<Tracker::FaserSCT_ClusterContainer> clusters(m_clusterKey, ctx);
  if (!clusters.isValid() || clusters.cptr() == nullptr) {
    clusterUnavailable = true;
  } else {
    for (const Tracker::FaserSCT_ClusterCollection* collection : *clusters) {
      if (collection == nullptr) {
        continue;
      }
      for (const Tracker::FaserSCT_Cluster* cluster : *collection) {
        if (cluster == nullptr) {
          continue;
        }
        const Identifier id = cluster->identify();
        if (m_idHelper->station(id) != kIftStation) {
          continue;
        }
        ClusterHit hit;
        hit.station = kIftStation;
        hit.layer = m_idHelper->layer(id);
        hit.phi_module = m_idHelper->phi_module(id);
        hit.eta_module = m_idHelper->eta_module(id);
        hit.side = m_idHelper->side(id);
        hit.id = id;
        hit.identifier = compactIdentifier(id);
        const Amg::Vector2D& local = cluster->localPosition();
        hit.loc0 = local[Trk::locX];
        hit.loc1 = local.rows() > 1 ? local[Trk::locY] : 0.0;
        const Amg::Vector3D& global = cluster->globalPosition();
        hit.global_x = global.x();
        hit.global_y = global.y();
        hit.global_z = global.z();
        iftClusters.push_back(hit);
      }
    }
  }

  const TrackCollection* fourStation = nullptr;
  SG::ReadHandle<TrackCollection> fourHandle(m_fourStationKey, ctx);
  if (fourHandle.isValid()) {
    fourStation = fourHandle.cptr();
  }
  const auto associated = associatedIftIds(fourStation, *m_idHelper);
  for (auto& hit : iftClusters) {
    hit.reconstruction_associated = associated.count(hit.identifier) > 0;
  }

  int nWithoutIft = 0;
  for (const Trk::Track* track : *tracks) {
    if (track != nullptr && track->trackParameters() != nullptr &&
        !track->trackParameters()->empty() &&
        track->trackParameters()->front() != nullptr) {
      ++nWithoutIft;
    }
  }

  json eventRow;
  eventRow["kind"] = "event";
  eventRow["source_id"] = m_sourceId.value();
  eventRow["run_id"] = runId;
  eventRow["event_id"] = eventId;
  eventRow["collection"] = m_trackKey.key();
  eventRow["independent_measurement_container"] = m_clusterKey.key();
  eventRow["n_without_ift_tracks"] = nWithoutIft;
  eventRow["n_four_station_tracks"] =
      fourStation == nullptr ? json(nullptr) : json(static_cast<int>(fourStation->size()));
  eventRow["n_ift_clusters"] = static_cast<int>(iftClusters.size());
  eventRow["cluster_stations_unavailable"] = clusterUnavailable;
  eventRow["station_decoded_by_fasersct_id"] = true;
  eventRow["selection_loss"] = nWithoutIft == 0;
  eventRow["selection_loss_reason"] =
      nWithoutIft == 0 ? json("no_without_ift_candidate") : json(nullptr);
  eventRow["prediction_uses_ift_measurement"] = false;
  eventRow["is_truth"] = false;
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_eventRows.push_back(eventRow.dump());
  }

  if (nWithoutIft == 0) {
    return StatusCode::SUCCESS;
  }

  int trackIndex = -1;
  for (const Trk::Track* track : *tracks) {
    ++trackIndex;
    json row;
    row["kind"] = "track";
    row["source_id"] = m_sourceId.value();
    row["run_id"] = runId;
    row["event_id"] = eventId;
    row["track_index"] = trackIndex;
    row["collection"] = m_trackKey.key();
    row["independent_measurement_container"] = m_clusterKey.key();
    row["helper"] = "CkfThreeStToIftPredictionAlg";
    row["helper_language"] = "C++";
    row["is_truth"] = false;
    row["prediction_uses_ift_measurement"] = false;
    row["official_prediction_model"] = "model1_acts_process_noise";
    row["propagation_direction"] = "backward";
    row["target_station"] = m_targetStation.value();
    row["target_z_mm"] = targetZ;
    row["state_definition"] = stateDefinition();
    row["residual_definition"] = residualDefinition();
    row["geometry_hash"] = m_geometryHash.value();
    row["field_hash"] = m_fieldHash.value();
    row["material_hash"] = m_materialHash.value();
    row["material_map_hash"] = m_materialHash.value();
    row["conditions_hash"] = m_conditionsHash.value();
    row["n_ift_clusters_in_event"] = static_cast<int>(iftClusters.size());
    row["cluster_stations_unavailable"] = clusterUnavailable;
    row["four_station_association"] =
        (fourStation != nullptr && fourStation->size() == 1)
            ? "unique_four_station_track"
            : "unmatched";
    row["station_decoded_by_fasersct_id"] = true;

    if (track == nullptr || track->trackParameters() == nullptr ||
        track->trackParameters()->empty()) {
      row["primary_failure_class"] = "missing_track_payload";
      row["has_covariance"] = false;
      std::lock_guard<std::mutex> lock(m_mutex);
      m_trackRows.push_back(row.dump());
      continue;
    }
    const Trk::TrackParameters* params = track->trackParameters()->front();
    if (params == nullptr) {
      row["primary_failure_class"] = "missing_track_payload";
      row["has_covariance"] = false;
      std::lock_guard<std::mutex> lock(m_mutex);
      m_trackRows.push_back(row.dump());
      continue;
    }

    json motStations;
    json tsosStations;
    int nMotIft = 0;
    int nTsosIft = 0;
    int nOutlier = 0;
    collectStations(*track, *m_idHelper, motStations, tsosStations, nMotIft,
                    nTsosIft, nOutlier);
    row["measurements_on_track_stations"] = motStations;
    row["tsos_stations"] = tsosStations;
    row["n_tracks_with_ift_measurements_on_track"] = nMotIft > 0 ? 1 : 0;
    row["n_ift_mot"] = nMotIft;
    row["n_ift_tsos"] = nTsosIft;
    row["n_outlier_hits"] = nOutlier;
    row["ift_leak"] = (nMotIft + nTsosIft) > 0;

    const Amg::Vector3D& position = params->position();
    const Amg::Vector3D& momentum = params->momentum();
    const AmgVector(5)& native = params->parameters();
    const double pz = momentum.z();
    const double pMev = std::sqrt(momentum.x() * momentum.x() +
                                  momentum.y() * momentum.y() +
                                  momentum.z() * momentum.z());
    const double sourceZ = position.z();
    row["native_state"] = {native[Trk::loc1], native[Trk::loc2], native[Trk::phi],
                           native[Trk::theta], native[Trk::qOverP]};
    row["native_covariance"] =
        params->covariance() == nullptr ? json(nullptr)
                                        : nativeMatrixToJson(*params->covariance());
    row["has_covariance"] = params->covariance() != nullptr;
    row["q_over_p_per_mev"] = native[Trk::qOverP];
    row["p_mev"] = pMev;
    row["charge"] = params->charge();
    row["source_x_mm"] = position.x();
    row["source_y_mm"] = position.y();
    row["source_z_mm"] = sourceZ;
    row["source_station"] = nearestStation(sourceZ, m_stationZmm);
    row["derived_state"] =
        std::abs(pz) < 1.0e-18
            ? json(nullptr)
            : json({position.x(), position.y(), momentum.x() / pz, momentum.y() / pz,
                    native[Trk::qOverP]});
    row["hop_class"] = std::abs(targetZ - sourceZ) > 200.0 ? "long_magnet_crossing"
                                                           : "local_surface";

    if (nMotIft + nTsosIft > 0) {
      row["primary_failure_class"] = "ift_measurement_leak";
      row["model0_no_process_noise"] =
          modelPayload(false, "ift_measurement_leak", nullptr, nullptr);
      row["model1_acts_process_noise"] =
          modelPayload(false, "ift_measurement_leak", nullptr, nullptr);
      row["residuals"] = json::array();
      std::lock_guard<std::mutex> lock(m_mutex);
      m_trackRows.push_back(row.dump());
      continue;
    }

    const auto start = actsFromTrk(*params, geometryContext);
    if (!start) {
      row["primary_failure_class"] = "acts_start_unavailable";
      row["model0_no_process_noise"] =
          modelPayload(false, "acts_start_unavailable", nullptr, nullptr);
      row["model1_acts_process_noise"] =
          modelPayload(false, "acts_start_unavailable", nullptr, nullptr);
      row["residuals"] = json::array();
      std::lock_guard<std::mutex> lock(m_mutex);
      m_trackRows.push_back(row.dump());
      continue;
    }

    auto plane = planeAtZ(targetZ);
    const auto out0 =
        propagateToSurface(*m_toolNoNoise, ctx, *start, *plane, sourceZ, targetZ);
    const auto out1 =
        propagateToSurface(*m_toolWithNoise, ctx, *start, *plane, sourceZ, targetZ);
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
    if (state0) {
      row["model0_no_process_noise"] =
          modelPayload(true, "", &*state0, cov0 ? &*cov0 : nullptr);
    } else {
      row["model0_no_process_noise"] = modelPayload(
          false, out0 ? "export_state_failed" : "propagate0_failed", nullptr,
          nullptr);
    }
    if (state1) {
      row["model1_acts_process_noise"] =
          modelPayload(true, "", &*state1, cov1 ? &*cov1 : nullptr);
    } else {
      row["model1_acts_process_noise"] = modelPayload(
          false, out1 ? "export_state_failed" : "propagate1_failed", nullptr,
          nullptr);
    }

    json residuals = json::array();
    int nResidualAvailable = 0;
    int nSurfaceMissing = 0;
    int nSurfacePropagateFailed = 0;
    if (!identifierMap || !trackingGeometry) {
      row["surface_lookup"] = "identifier_map_or_geometry_unavailable";
    } else {
      row["surface_lookup"] = "identifier_map";
    }
    for (const auto& hit : iftClusters) {
      json residual;
      residual["identifier"] = hit.identifier;
      residual["station"] = hit.station;
      residual["layer"] = hit.layer;
      residual["phi_module"] = hit.phi_module;
      residual["eta_module"] = hit.eta_module;
      residual["side"] = hit.side;
      residual["layer_code"] = 6 * hit.station + 2 * hit.layer + hit.side;
      residual["loc0_cluster_mm"] = hit.loc0;
      residual["loc1_cluster_mm"] = hit.loc1;
      residual["global_x_mm"] = hit.global_x;
      residual["global_y_mm"] = hit.global_y;
      residual["global_z_mm"] = hit.global_z;
      residual["reconstruction_associated"] = hit.reconstruction_associated;
      residual["prediction_uses_this_measurement"] = false;
      residual["hop_class"] = "long_magnet_crossing";

      if (!identifierMap || !trackingGeometry) {
        residual["propagate_success"] = false;
        residual["residual_available"] = false;
        residual["failure_class"] = "surface_not_in_identifier_map";
        residual["residual_loc0_mm"] = nullptr;
        ++nSurfaceMissing;
        residuals.push_back(residual);
        continue;
      }
      const Identifier waferId = m_idHelper->wafer_id(hit.id);
      if (identifierMap->count(waferId) == 0) {
        residual["propagate_success"] = false;
        residual["residual_available"] = false;
        residual["failure_class"] = "surface_not_in_identifier_map";
        residual["residual_loc0_mm"] = nullptr;
        ++nSurfaceMissing;
        residuals.push_back(residual);
        continue;
      }
      const Acts::GeometryIdentifier geoId = identifierMap->at(waferId);
      const Acts::Surface* surface = trackingGeometry->findSurface(geoId);
      if (surface == nullptr) {
        residual["propagate_success"] = false;
        residual["residual_available"] = false;
        residual["failure_class"] = "surface_pointer_null";
        residual["residual_loc0_mm"] = nullptr;
        ++nSurfaceMissing;
        residuals.push_back(residual);
        continue;
      }
      residual["surface_center_z_mm"] = surface->center(geometryContext).z();
      const auto predicted = propagateToSurface(
          *m_toolWithNoise, ctx, *start, *surface, sourceZ,
          surface->center(geometryContext).z());
      if (!predicted) {
        residual["propagate_success"] = false;
        residual["residual_available"] = false;
        residual["failure_class"] = "propagate_surface_failed";
        residual["residual_loc0_mm"] = nullptr;
        ++nSurfacePropagateFailed;
        residuals.push_back(residual);
        continue;
      }
      const double loc0Pred = predicted->parameters()[Acts::eBoundLoc0];
      const double loc1Pred = predicted->parameters()[Acts::eBoundLoc1];
      const Acts::Vector3 endPos = predicted->position(geometryContext);
      residual["propagate_success"] = true;
      residual["predicted_loc0_mm"] = loc0Pred;
      residual["predicted_loc1_mm"] = loc1Pred;
      residual["predicted_x_mm"] = endPos.x();
      residual["predicted_y_mm"] = endPos.y();
      residual["predicted_z_mm"] = endPos.z();
      residual["predicted_q_over_p_per_mev"] =
          predicted->parameters()[Acts::eBoundQOverP] * 1_MeV;
      residual["residual_loc0_mm"] = hit.loc0 - loc0Pred;
      residual["inside_bounds"] = surface->insideBounds(
          Acts::Vector2(loc0Pred, loc1Pred), Acts::BoundaryCheck(true));
      residual["residual_available"] = true;
      residual["failure_class"] = nullptr;
      ++nResidualAvailable;
      residuals.push_back(residual);
    }
    row["residuals"] = residuals;
    row["n_residual_available"] = nResidualAvailable;
    row["n_surface_missing"] = nSurfaceMissing;
    row["n_surface_propagate_failed"] = nSurfacePropagateFailed;

    if (row["ift_leak"].get<bool>()) {
      row["primary_failure_class"] = "ift_measurement_leak";
    } else if (!state1) {
      row["primary_failure_class"] = "acts_s1_to_ift_propagation_failed";
    } else if (iftClusters.empty()) {
      row["primary_failure_class"] = clusterUnavailable
                                         ? "independent_ift_measurement_absent"
                                         : "no_ift_cluster_in_event";
    } else if (nResidualAvailable == 0) {
      row["primary_failure_class"] = "ift_prediction_available_residual_unavailable";
    } else {
      row["primary_failure_class"] = nullptr;
    }

    std::lock_guard<std::mutex> lock(m_mutex);
    m_trackRows.push_back(row.dump());
  }
  return StatusCode::SUCCESS;
}

StatusCode CkfThreeStToIftPredictionAlg::finalize() {
  std::lock_guard<std::mutex> lock(m_mutex);
  if (!writeLines(m_outputJsonl.value(), m_trackRows)) {
    ATH_MSG_ERROR("Cannot write track jsonl (exists or I/O failed): "
                  << m_outputJsonl.value());
    return StatusCode::FAILURE;
  }
  if (!writeLines(m_eventJsonl.value(), m_eventRows)) {
    ATH_MSG_ERROR("Cannot write event jsonl (exists or I/O failed): "
                  << m_eventJsonl.value());
    return StatusCode::FAILURE;
  }
  ATH_MSG_INFO("Wrote " << m_trackRows.size() << " 3ST→IFT track rows and "
                        << m_eventRows.size() << " event rows");
  return StatusCode::SUCCESS;
}
