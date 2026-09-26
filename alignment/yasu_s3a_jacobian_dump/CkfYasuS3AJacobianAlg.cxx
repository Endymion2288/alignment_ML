#include "CkfYasuS3AJacobianAlg.h"

#include "EventPrimitives/EventPrimitives.h"
#include "GeoPrimitives/GeoPrimitives.h"

#include "Acts/Definitions/Units.hpp"
#include "Acts/EventData/TrackParameters.hpp"
#include "Acts/EventData/detail/TransformationFreeToBound.hpp"
#include "Acts/Geometry/GeometryIdentifier.hpp"
#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/Surfaces/BoundaryCheck.hpp"
#include "Acts/Surfaces/PlaneSurface.hpp"
#include "Identifier/Identifier.h"
#include "StoreGate/ReadHandle.h"
#include "TrackerIdentifier/FaserSCT_ID.h"
#include "TrackerPrepRawData/FaserSCT_Cluster.h"
#include "TrackerPrepRawData/FaserSCT_ClusterCollection.h"
#include "TrackerReadoutGeometry/SCT_DetectorManager.h"
#include "TrackerReadoutGeometry/SiDetectorElement.h"
#include "TrackerRIO_OnTrack/FaserSCT_ClusterOnTrack.h"
#include "TrkEventPrimitives/ParamDefs.h"
#include "TrkParameters/TrackParameters.h"
#include "TrkSurfaces/Surface.h"
#include "TrkTrack/Track.h"
#include "TrkTrack/TrackStateOnSurface.h"
#include "xAODTruth/TruthParticle.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <sstream>

using namespace Acts::UnitLiterals;
using json = nlohmann::json;

namespace {

constexpr int kIftStation = 0;
constexpr double kRungFactors[3] = {1.0, 0.5, 0.25};

bool finiteValue(double value) { return std::isfinite(value); }

json finiteOrNull(double value) {
  return finiteValue(value) ? json(value) : json(nullptr);
}

std::shared_ptr<Acts::PlaneSurface> planeAtZ(double zMm) {
  return Acts::Surface::makeShared<Acts::PlaneSurface>(
      Acts::Vector3(0.0, 0.0, zMm), Acts::Vector3(0.0, 0.0, 1.0));
}

std::string compactIdentifier(const Identifier& id) {
  std::ostringstream out;
  out << id.get_compact();
  return out.str();
}

json identifierList(const std::vector<std::string>& ids) {
  json out = json::array();
  for (const auto& item : ids) {
    out.push_back(item);
  }
  return out;
}

const Tracker::FaserSCT_ClusterOnTrack* asClusterOnTrack(
    const Trk::MeasurementBase* meas) {
  return dynamic_cast<const Tracker::FaserSCT_ClusterOnTrack*>(meas);
}

json residualDefinition() {
  return {
      {"kind", "truth_associated_ift_sensor_local_loc0"},
      {"measurement",
       "FaserSCT_Cluster.localPosition Trk::locX on the runtime SiDetectorElement"},
      {"prediction", "Acts BoundTrackParameters loc0 on the same IFT wafer surface"},
      {"formula", "r = loc0_cluster - loc0_predicted"},
      {"unit", "mm"},
      {"frame", "actual_sensor_local_measurement_axis"},
      {"not_yz_bending_geometry", true},
      {"inside_bounds_required_for_residual", false},
      {"association", "truth_sdo_barcode_on_ift_cluster"},
      {"four_station_association_forbidden", true},
  };
}

json stateDefinition() {
  return {
      {"native_athena",
       json::array({"loc1_mm", "loc2_mm", "phi", "theta", "q_over_p_per_mev"})},
      {"export_parameters",
       json::array({"x_mm", "y_mm", "tx", "ty", "q_over_p_per_mev"})},
      {"export_frame", "global_cartesian_slopes_at_source_surface"},
      {"q_over_p_native_unit", "per_MeV"},
      {"mean_response_only", true},
      {"native_5x5_not_a_jacobian_weight", true},
      {"perturbation_ty_is_export_ty", true},
      {"perturbation_qp_is_native_q_over_p", true},
  };
}

json geometryResponseDefinition() {
  return {
      {"pivot_mm", json::array({0.0, 0.0, 0.0})},
      {"composition", "G = T * Rz * Ry * Rx, active left-multiply"},
      {"convention", "stations_global_origin_TRzRyRx_active_left_multiply"},
      {"R_y_matrix", "[[c,0,s],[0,1,0],[-s,0,c]] with c=cos(Ry), s=sin(Ry)"},
      {"d_x", "global translation of the IFT station, millimetres"},
      {"applied_to",
       "runtime sensor transform read from Calypso, then left-composed"},
      {"handwritten_station_response", false},
      {"payload_write", false},
  };
}

Acts::Transform3 se3RyDx(double ryRad, double dxMm) {
  const double c = std::cos(ryRad);
  const double s = std::sin(ryRad);
  Acts::Transform3 transform = Acts::Transform3::Identity();
  transform.linear() << c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c;
  transform.translation() = Acts::Vector3(dxMm, 0.0, 0.0);
  return transform;
}

std::optional<Acts::BoundTrackParameters> actsFromTrk(
    const Trk::TrackParameters& parameters,
    const Acts::GeometryContext& geometryContext, double dqOverPPerMev,
    double dTy) {
  const Amg::Vector3D& position = parameters.position();
  Amg::Vector3D momentum = parameters.momentum();
  const AmgVector(5)& nativeParameters = parameters.parameters();
  const double nativeQOverP = nativeParameters[Trk::qOverP] + dqOverPPerMev;
  if (!std::isfinite(position.x()) || !std::isfinite(position.y()) ||
      !std::isfinite(position.z()) || !std::isfinite(momentum.x()) ||
      !std::isfinite(momentum.y()) || !std::isfinite(momentum.z()) ||
      !std::isfinite(nativeQOverP) || std::abs(nativeQOverP) < 1.0e-18) {
    return std::nullopt;
  }
  if (std::abs(dTy) > 0.0) {
    const double pz = momentum.z();
    if (std::abs(pz) < 1.0e-18) {
      return std::nullopt;
    }
    const double tx = momentum.x() / pz;
    const double ty = momentum.y() / pz + dTy;
    const double p = std::sqrt(momentum.x() * momentum.x() +
                               momentum.y() * momentum.y() + pz * pz);
    const double norm = std::sqrt(tx * tx + ty * ty + 1.0);
    if (norm < 1.0e-18) {
      return std::nullopt;
    }
    const double signPz = pz >= 0.0 ? 1.0 : -1.0;
    momentum = Amg::Vector3D(p * signPz * tx / norm, p * signPz * ty / norm,
                             p * signPz / norm);
  }

  auto surface = planeAtZ(position.z());
  auto bound = Acts::detail::transformFreeToBoundParameters(
      position, 0.0, momentum, nativeQOverP / 1_MeV, *surface, geometryContext);
  if (!bound.ok()) {
    return std::nullopt;
  }
  return Acts::BoundTrackParameters(surface, bound.value(), std::nullopt,
                                    Acts::ParticleHypothesis::muon());
}

std::optional<Acts::BoundTrackParameters> propagateToSurface(
    const IFaserActsExtrapolationTool& tool, const EventContext& ctx,
    const Acts::BoundTrackParameters& start, const Acts::Surface& target,
    double sourceZ, double targetZ) {
  const Acts::Direction direction =
      targetZ >= sourceZ ? Acts::Direction::Forward : Acts::Direction::Backward;
  return tool.propagate(ctx, start, target, direction);
}

struct ClusterHit {
  int station{-1};
  int layer{-1};
  int phi_module{-1};
  int eta_module{-1};
  int side{-1};
  std::string identifier;
  Identifier id;
  Identifier waferId;
  double loc0{0.0};
  double loc1{0.0};
  double global_x{0.0};
  double global_y{0.0};
  double global_z{0.0};
  bool truth_associated{false};
  std::vector<int> barcodes;
  Acts::Vector3 center{Acts::Vector3::Zero()};
  Acts::Vector3 u{Acts::Vector3::UnitX()};
  Acts::Vector3 v{Acts::Vector3::UnitY()};
  Acts::Vector3 n{Acts::Vector3::UnitZ()};
  Acts::Transform3 transform{Acts::Transform3::Identity()};
  Acts::GeometryIdentifier geoId{};
  const Acts::Surface* surface{nullptr};
  json runtime_geometry;
};

void collectStations(const Trk::Track& track, const FaserSCT_ID& idHelper,
                     json& motStations, json& tsosStations, int& nMot,
                     int& nMotIft, int& nTsosIft, int& nOutlier,
                     std::vector<std::string>& motIds) {
  motStations = json::array();
  tsosStations = json::array();
  nMot = 0;
  nMotIft = 0;
  nTsosIft = 0;
  nOutlier = 0;
  motIds.clear();
  if (track.measurementsOnTrack() != nullptr) {
    for (const Trk::MeasurementBase* meas : *track.measurementsOnTrack()) {
      const auto* cluster = asClusterOnTrack(meas);
      if (cluster == nullptr) {
        continue;
      }
      const Identifier id = cluster->identify();
      const int station = idHelper.station(id);
      motStations.push_back(station);
      motIds.push_back(compactIdentifier(id));
      ++nMot;
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

std::vector<int> barcodesOnCluster(const Tracker::FaserSCT_Cluster& cluster,
                                   const TrackerSimDataCollection* sdo) {
  std::vector<int> barcodes;
  if (sdo == nullptr) {
    return barcodes;
  }
  for (Identifier rdo : cluster.rdoList()) {
    if (sdo->count(rdo) == 0) {
      continue;
    }
    const auto& deposits = sdo->at(rdo).getdeposits();
    for (const TrackerSimData::Deposit& deposit : deposits) {
      const int barcode = deposit.first.barcode();
      if (std::find(barcodes.begin(), barcodes.end(), barcode) ==
          barcodes.end()) {
        barcodes.push_back(barcode);
      }
    }
  }
  return barcodes;
}

struct ParticleHitCount {
  int barcode{0};
  int hitCount{0};
};

void increaseHitCount(std::vector<ParticleHitCount>& counts, int barcode) {
  auto it = std::find_if(counts.begin(), counts.end(),
                         [barcode](const ParticleHitCount& row) {
                           return row.barcode == barcode;
                         });
  if (it != counts.end()) {
    it->hitCount += 1;
  } else {
    counts.push_back({barcode, 1});
  }
}

json pluralityFromTrack(const Trk::Track& track,
                        const TrackerSimDataCollection* sdo, int nMot) {
  json payload;
  payload["method"] = "truth_sdo_barcode";
  payload["plurality_is_not_majority"] = true;
  payload["four_station_forbidden"] = true;
  std::vector<ParticleHitCount> counts;
  if (track.measurementsOnTrack() != nullptr && sdo != nullptr) {
    for (const Trk::MeasurementBase* meas : *track.measurementsOnTrack()) {
      const auto* clusterOnTrack = asClusterOnTrack(meas);
      if (clusterOnTrack == nullptr || clusterOnTrack->prepRawData() == nullptr) {
        continue;
      }
      std::vector<int> barcodes;
      for (int barcode :
           barcodesOnCluster(*clusterOnTrack->prepRawData(), sdo)) {
        if (std::find(barcodes.begin(), barcodes.end(), barcode) ==
            barcodes.end()) {
          barcodes.push_back(barcode);
          increaseHitCount(counts, barcode);
        }
      }
    }
  }
  std::sort(counts.begin(), counts.end(),
            [](const ParticleHitCount& a, const ParticleHitCount& b) {
              return a.hitCount > b.hitCount;
            });
  payload["n_contributing_barcodes"] = static_cast<int>(counts.size());
  if (counts.empty()) {
    payload["barcode"] = nullptr;
    payload["hit_count"] = 0;
    payload["is_majority"] = false;
    payload["is_tie"] = false;
    payload["match_fraction_mot"] = nullptr;
    return payload;
  }
  const bool tie =
      counts.size() > 1 && counts[0].hitCount == counts[1].hitCount;
  const bool majority = nMot > 0 && (2 * counts[0].hitCount) > nMot && !tie;
  payload["barcode"] = counts[0].barcode;
  payload["hit_count"] = counts[0].hitCount;
  payload["is_majority"] = majority;
  payload["is_tie"] = tie;
  payload["match_fraction_mot"] =
      nMot > 0 ? json(static_cast<double>(counts[0].hitCount) /
                      static_cast<double>(nMot))
               : json(nullptr);
  json all = json::array();
  for (const auto& row : counts) {
    all.push_back({{"barcode", row.barcode}, {"hit_count", row.hitCount}});
  }
  payload["contributors"] = all;
  return payload;
}

json runtimeGeometry(const TrackerDD::SiDetectorElement& detEl) {
  json geo;
  const Amg::Vector3D& center = detEl.center();
  const Amg::Vector3D& phi = detEl.phiAxis();
  const Amg::Vector3D& eta = detEl.etaAxis();
  const Amg::Vector3D& normal = detEl.normal();
  geo["center_mm"] = {center.x(), center.y(), center.z()};
  geo["u"] = {phi.x(), phi.y(), phi.z()};
  geo["v"] = {eta.x(), eta.y(), eta.z()};
  geo["n"] = {normal.x(), normal.y(), normal.z()};
  geo["pivot_mm"] = {0.0, 0.0, 0.0};
  geo["source"] = "TrackerDD::SiDetectorElement runtime";
  geo["handwritten_station_response"] = false;
  const Amg::Transform3D& transform = detEl.transform();
  json matrix = json::array();
  for (int i = 0; i < 4; ++i) {
    json row = json::array();
    for (int j = 0; j < 4; ++j) {
      row.push_back(transform(i, j));
    }
    matrix.push_back(row);
  }
  geo["transform_4x4"] = matrix;
  return geo;
}

json residualRow(const ClusterHit& hit, bool success, const char* failure,
                 double loc0Pred, double loc1Pred, const Acts::Vector3* endPos,
                 uint64_t surfaceValue, bool insideBounds) {
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
  residual["truth_associated"] = hit.truth_associated;
  residual["sdo_barcodes"] = hit.barcodes;
  residual["prediction_uses_this_measurement"] = false;
  residual["four_station_association_used"] = false;
  residual["surface_geo_id"] = surfaceValue;
  residual["runtime_geometry"] = hit.runtime_geometry;
  residual["hop_class"] = "long_magnet_crossing";
  if (!success) {
    residual["propagate_success"] = false;
    residual["residual_available"] = false;
    residual["failure_class"] = failure;
    residual["residual_loc0_mm"] = nullptr;
    residual["predicted_loc0_mm"] = nullptr;
    return residual;
  }
  residual["propagate_success"] = true;
  residual["predicted_loc0_mm"] = loc0Pred;
  residual["predicted_loc1_mm"] = loc1Pred;
  if (endPos != nullptr) {
    residual["predicted_x_mm"] = (*endPos).x();
    residual["predicted_y_mm"] = (*endPos).y();
    residual["predicted_z_mm"] = (*endPos).z();
  }
  residual["residual_loc0_mm"] = hit.loc0 - loc0Pred;
  residual["inside_bounds"] = insideBounds;
  residual["residual_available"] = true;
  residual["failure_class"] = nullptr;
  return residual;
}

json evaluateResiduals(const std::vector<ClusterHit>& hits,
                       const IFaserActsExtrapolationTool& tool,
                       const EventContext& ctx,
                       const Acts::BoundTrackParameters& start,
                       const Acts::GeometryContext& geometryContext,
                       double sourceZ, double ryRad, double dxMm) {
  json residuals = json::array();
  const Acts::Transform3 increment = se3RyDx(ryRad, dxMm);
  const bool identityGeometry = std::abs(ryRad) < 1.0e-18 && std::abs(dxMm) < 1.0e-18;
  for (const auto& hit : hits) {
    if (hit.surface == nullptr) {
      residuals.push_back(residualRow(hit, false, "surface_pointer_null", 0.0,
                                      0.0, nullptr, 0, false));
      continue;
    }
    const auto* plane = dynamic_cast<const Acts::PlaneSurface*>(hit.surface);
    if (plane == nullptr) {
      residuals.push_back(residualRow(hit, false, "surface_not_plane", 0.0, 0.0,
                                      nullptr, hit.geoId.value(), false));
      continue;
    }
    std::shared_ptr<Acts::PlaneSurface> shifted;
    const Acts::Surface* target = hit.surface;
    if (!identityGeometry) {
      shifted = Acts::Surface::makeShared<Acts::PlaneSurface>(
          geometryContext, *plane, increment);
      target = shifted.get();
    }
    const double targetZ = target->center(geometryContext).z();
    const auto predicted =
        propagateToSurface(tool, ctx, start, *target, sourceZ, targetZ);
    if (!predicted) {
      residuals.push_back(residualRow(hit, false, "propagate_surface_failed",
                                      0.0, 0.0, nullptr, hit.geoId.value(),
                                      false));
      continue;
    }
    const double loc0Pred = predicted->parameters()[Acts::eBoundLoc0];
    const double loc1Pred = predicted->parameters()[Acts::eBoundLoc1];
    const Acts::Vector3 endPos = predicted->position(geometryContext);
    const bool inside = target->insideBounds(Acts::Vector2(loc0Pred, loc1Pred),
                                             Acts::BoundaryCheck(true));
    residuals.push_back(residualRow(hit, true, nullptr, loc0Pred, loc1Pred,
                                    &endPos, hit.geoId.value(), inside));
  }
  return residuals;
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

std::set<std::pair<int, int>> parseFocus(const std::string& text) {
  std::set<std::pair<int, int>> out;
  if (text.empty()) {
    return out;
  }
  std::stringstream stream(text);
  std::string token;
  while (std::getline(stream, token, ',')) {
    if (token.empty()) {
      continue;
    }
    const auto colon = token.find(':');
    if (colon == std::string::npos) {
      continue;
    }
    out.emplace(std::stoi(token.substr(0, colon)),
                std::stoi(token.substr(colon + 1)));
  }
  return out;
}

}  // namespace

CkfYasuS3AJacobianAlg::CkfYasuS3AJacobianAlg(const std::string& name,
                                             ISvcLocator* pSvcLocator)
    : AthAlgorithm(name, pSvcLocator) {}

StatusCode CkfYasuS3AJacobianAlg::initialize() {
  if (m_outputJsonl.empty() || m_sourceId.empty() || m_fileSha256.empty()) {
    ATH_MSG_ERROR("OutputJsonl, SourceId and FileSha256 are required");
    return StatusCode::FAILURE;
  }
  ATH_CHECK(m_trackKey.initialize());
  ATH_CHECK(m_clusterKey.initialize());
  ATH_CHECK(m_sdoKey.initialize());
  ATH_CHECK(m_eventKey.initialize());
  ATH_CHECK(m_toolNoNoise.retrieve());
  ATH_CHECK(m_toolWithNoise.retrieve());
  ATH_CHECK(m_toolElossOffMsOn.retrieve());
  ATH_CHECK(m_trackingGeometryTool.retrieve());
  ATH_CHECK(m_truthMatchingTool.retrieve());
  ATH_CHECK(detStore()->retrieve(m_idHelper, "FaserSCT_ID"));
  ATH_CHECK(detStore()->retrieve(m_detManager, "SCT"));
  m_focus = parseFocus(m_focusPairs.value());
  ATH_MSG_INFO("Yasu-S3A Jacobian dump ready: " << m_outputJsonl.value()
                                                << " n_focus=" << m_focus.size());
  return StatusCode::SUCCESS;
}

StatusCode CkfYasuS3AJacobianAlg::execute() {
  const EventContext& ctx = getContext();
  const int skipIndex = m_skipEvents.value() + m_eventsSeen;
  ++m_eventsSeen;

  SG::ReadHandle<TrackCollection> tracks(m_trackKey, ctx);
  SG::ReadHandle<xAOD::EventInfo> eventInfo(m_eventKey, ctx);
  if (!tracks.isValid() || !eventInfo.isValid()) {
    return StatusCode::SUCCESS;
  }
  const int runId = static_cast<int>(eventInfo->runNumber());
  const int eventId = static_cast<int>(eventInfo->eventNumber());
  const bool focusEvent =
      m_focus.empty() ||
      std::any_of(m_focus.begin(), m_focus.end(),
                  [skipIndex](const std::pair<int, int>& item) {
                    return item.first == skipIndex;
                  });
  if (!focusEvent) {
    return StatusCode::SUCCESS;
  }

  const Acts::GeometryContext geometryContext =
      m_trackingGeometryTool->getGeometryContext(ctx).context();
  auto identifierMap = m_trackingGeometryTool->getIdentifierMap();
  auto trackingGeometry = m_trackingGeometryTool->trackingGeometry();

  SG::ReadHandle<TrackerSimDataCollection> sdoHandle(m_sdoKey, ctx);
  const TrackerSimDataCollection* sdo =
      sdoHandle.isValid() ? sdoHandle.cptr() : nullptr;

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
        hit.waferId = m_idHelper->wafer_id(id);
        hit.identifier = compactIdentifier(id);
        const Amg::Vector2D& local = cluster->localPosition();
        hit.loc0 = local[Trk::locX];
        hit.loc1 = local.rows() > 1 ? local[Trk::locY] : 0.0;
        const Amg::Vector3D& global = cluster->globalPosition();
        hit.global_x = global.x();
        hit.global_y = global.y();
        hit.global_z = global.z();
        hit.barcodes = barcodesOnCluster(*cluster, sdo);
        const TrackerDD::SiDetectorElement* detEl = cluster->detectorElement();
        if (detEl == nullptr && m_detManager != nullptr) {
          detEl = m_detManager->getDetectorElement(id);
        }
        if (detEl != nullptr) {
          hit.center = Acts::Vector3(detEl->center().x(), detEl->center().y(),
                                     detEl->center().z());
          hit.u = Acts::Vector3(detEl->phiAxis().x(), detEl->phiAxis().y(),
                                detEl->phiAxis().z());
          hit.v = Acts::Vector3(detEl->etaAxis().x(), detEl->etaAxis().y(),
                                detEl->etaAxis().z());
          hit.n = Acts::Vector3(detEl->normal().x(), detEl->normal().y(),
                                detEl->normal().z());
          hit.transform = detEl->transform();
          hit.runtime_geometry = runtimeGeometry(*detEl);
        } else {
          hit.runtime_geometry = {{"source", "unavailable"}};
        }
        if (identifierMap && trackingGeometry &&
            identifierMap->count(hit.waferId) > 0) {
          hit.geoId = identifierMap->at(hit.waferId);
          hit.surface = trackingGeometry->findSurface(hit.geoId);
          if (hit.surface != nullptr) {
            hit.transform = hit.surface->transform(geometryContext);
            hit.center = hit.surface->center(geometryContext);
          }
        }
        iftClusters.push_back(hit);
      }
    }
  }

  json eventRow;
  eventRow["kind"] = "event";
  eventRow["source_id"] = m_sourceId.value();
  eventRow["file_sha256"] = m_fileSha256.value();
  eventRow["run_id"] = runId;
  eventRow["event_id"] = eventId;
  eventRow["skip_index"] = skipIndex;
  eventRow["collection"] = m_trackKey.key();
  eventRow["independent_measurement_container"] = m_clusterKey.key();
  eventRow["sdo_map"] = m_sdoKey.key();
  eventRow["sdo_map_present"] = sdo != nullptr;
  eventRow["n_ift_clusters"] = static_cast<int>(iftClusters.size());
  eventRow["cluster_stations_unavailable"] = clusterUnavailable;
  eventRow["four_station_association_used"] = false;
  eventRow["used_qp_bending_proxy"] = false;
  eventRow["geometry_write_allowed"] = false;
  eventRow["mc_channel_number"] = static_cast<int>(eventInfo->mcChannelNumber());
  eventRow["mc_event_number"] = static_cast<int>(eventInfo->mcEventNumber());
  eventRow["event_guid"] = nullptr;
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_eventRows.push_back(eventRow.dump());
  }

  int trackIndex = -1;
  for (const Trk::Track* track : *tracks) {
    ++trackIndex;
    if (!m_focus.empty() &&
        m_focus.count(std::make_pair(skipIndex, trackIndex)) == 0) {
      continue;
    }
    json row;
    row["kind"] = "track";
    row["helper"] = "CkfYasuS3AJacobianAlg";
    row["helper_language"] = "C++";
    row["task"] = "YASU-S3A";
    row["workbook"] = 128;
    row["source_id"] = m_sourceId.value();
    row["file_sha256"] = m_fileSha256.value();
    row["run_id"] = runId;
    row["event_id"] = eventId;
    row["skip_index"] = skipIndex;
    row["track_index"] = trackIndex;
    row["collection"] = m_trackKey.key();
    row["independent_measurement_container"] = m_clusterKey.key();
    row["four_station_association_used"] = false;
    row["used_qp_bending_proxy"] = false;
    row["native_5x5_weighted"] = false;
    row["mean_response_only"] = true;
    row["geometry_write_allowed"] = false;
    row["residual_definition"] = residualDefinition();
    row["state_definition"] = stateDefinition();
    row["geometry_response_definition"] = geometryResponseDefinition();
    row["geometry_hash"] = m_geometryHash.value();
    row["field_hash"] = m_fieldHash.value();
    row["material_hash"] = m_materialHash.value();
    row["conditions_hash"] = m_conditionsHash.value();
    row["iov_provenance"] = m_iovProvenance.value();
    row["finite_difference"] = {
        {"kind", "central"},
        {"rungs", json::array({"+/-delta", "+/-delta/2", "+/-delta/4"})},
        {"rung_factors", json::array({1.0, 0.5, 0.25})},
        {"deltas",
         {{"q_over_p", m_deltaQp.value()},
          {"R_y", m_deltaRy.value()},
          {"d_x", m_deltaDx.value()},
          {"t_y", m_deltaTy.value()}}},
    };

    if (track == nullptr || track->trackParameters() == nullptr ||
        track->trackParameters()->empty() ||
        track->trackParameters()->front() == nullptr) {
      row["primary_failure_class"] = "missing_track_payload";
      row["measurement_digest"] = "missing";
      std::lock_guard<std::mutex> lock(m_mutex);
      m_trackRows.push_back(row.dump());
      continue;
    }
    const Trk::TrackParameters* params = track->trackParameters()->front();

    json motStations;
    json tsosStations;
    int nMot = 0;
    int nMotIft = 0;
    int nTsosIft = 0;
    int nOutlier = 0;
    std::vector<std::string> motIds;
    collectStations(*track, *m_idHelper, motStations, tsosStations, nMot,
                    nMotIft, nTsosIft, nOutlier, motIds);
    row["measurements_on_track_stations"] = motStations;
    row["tsos_stations"] = tsosStations;
    row["n_mot"] = nMot;
    row["n_ift_mot"] = nMotIft;
    row["n_ift_tsos"] = nTsosIft;
    row["n_outlier_hits"] = nOutlier;
    row["ift_leak"] = (nMotIft + nTsosIft) > 0;

    json association = pluralityFromTrack(*track, sdo, nMot);
    auto [truthParticle, officialHitCount] =
        m_truthMatchingTool->getTruthParticle(track);
    association["official_tool_hit_count"] = officialHitCount;
    association["official_tool_is_plurality_not_majority"] = true;
    if (truthParticle != nullptr) {
      association["official_tool_barcode"] = truthParticle->barcode();
      row["truth_pdg"] = truthParticle->pdgId();
      row["truth_charge"] = finiteOrNull(truthParticle->charge());
    } else {
      association["official_tool_barcode"] = nullptr;
    }
    const int trackBarcode =
        association["barcode"].is_number() ? association["barcode"].get<int>()
                                           : -1;

    std::vector<ClusterHit> associated;
    std::vector<std::string> assocIds;
    for (auto hit : iftClusters) {
      hit.truth_associated =
          trackBarcode >= 0 &&
          std::find(hit.barcodes.begin(), hit.barcodes.end(), trackBarcode) !=
              hit.barcodes.end();
      if (hit.truth_associated) {
        associated.push_back(hit);
        assocIds.push_back(hit.identifier);
      }
    }
    association["n_truth_associated_ift_clusters"] =
        static_cast<int>(associated.size());
    association["n_ift_clusters_in_event"] = static_cast<int>(iftClusters.size());
    row["ift_association"] = association;

    std::vector<std::string> digestIds = motIds;
    digestIds.insert(digestIds.end(), assocIds.begin(), assocIds.end());
    std::sort(digestIds.begin(), digestIds.end());
    std::vector<std::string> assocSorted = assocIds;
    std::sort(assocSorted.begin(), assocSorted.end());
    row["mot_identifiers"] = identifierList(motIds);
    row["associated_ift_identifiers"] = identifierList(assocSorted);
    row["measurement_digest_inputs"] = identifierList(digestIds);
    row["measurement_digest"] = nullptr;
    row["association_digest"] = nullptr;
    row["digest_computed_in_python"] = true;

    const Amg::Vector3D& position = params->position();
    const Amg::Vector3D& momentum = params->momentum();
    const AmgVector(5)& native = params->parameters();
    const double pz = momentum.z();
    const double sourceZ = position.z();
    row["native_state"] = {native[Trk::loc1], native[Trk::loc2], native[Trk::phi],
                           native[Trk::theta], native[Trk::qOverP]};
    row["has_covariance"] = params->covariance() != nullptr;
    row["q_over_p_per_mev"] = native[Trk::qOverP];
    row["source_x_mm"] = position.x();
    row["source_y_mm"] = position.y();
    row["source_z_mm"] = sourceZ;
    row["derived_state"] =
        std::abs(pz) < 1.0e-18
            ? json(nullptr)
            : json({position.x(), position.y(), momentum.x() / pz,
                    momentum.y() / pz, native[Trk::qOverP]});

    if (nMotIft + nTsosIft > 0) {
      row["primary_failure_class"] = "ift_measurement_leak";
      row["nominal_residuals"] = json::array();
      std::lock_guard<std::mutex> lock(m_mutex);
      m_trackRows.push_back(row.dump());
      continue;
    }
    if (associated.empty()) {
      row["primary_failure_class"] = clusterUnavailable
                                         ? "independent_ift_measurement_absent"
                                         : "truth_sdo_ift_unmatched";
      row["nominal_residuals"] = json::array();
      std::lock_guard<std::mutex> lock(m_mutex);
      m_trackRows.push_back(row.dump());
      continue;
    }

    const auto startNominal =
        actsFromTrk(*params, geometryContext, 0.0, 0.0);
    if (!startNominal) {
      row["primary_failure_class"] = "acts_start_unavailable";
      row["nominal_residuals"] = json::array();
      std::lock_guard<std::mutex> lock(m_mutex);
      m_trackRows.push_back(row.dump());
      continue;
    }

    json nominal = evaluateResiduals(associated, *m_toolWithNoise, ctx,
                                     *startNominal, geometryContext, sourceZ,
                                     0.0, 0.0);
    json nominalElossOff = evaluateResiduals(
        associated, *m_toolElossOffMsOn, ctx, *startNominal, geometryContext,
        sourceZ, 0.0, 0.0);
    json nominalNoNoise = evaluateResiduals(
        associated, *m_toolNoNoise, ctx, *startNominal, geometryContext,
        sourceZ, 0.0, 0.0);
    row["nominal_residuals"] = nominal;
    row["nominal_residuals_eloss_off"] = nominalElossOff;
    row["nominal_residuals_no_noise"] = nominalNoNoise;

    double maxElossDelta = 0.0;
    bool elossComparable = nominal.size() == nominalElossOff.size();
    if (elossComparable) {
      for (std::size_t i = 0; i < nominal.size(); ++i) {
        if (nominal[i]["identifier"] != nominalElossOff[i]["identifier"] ||
            nominal[i]["surface_geo_id"] !=
                nominalElossOff[i]["surface_geo_id"]) {
          elossComparable = false;
          break;
        }
        if (nominal[i]["residual_available"] == true &&
            nominalElossOff[i]["residual_available"] == true) {
          maxElossDelta = std::max(
              maxElossDelta,
              std::abs(nominal[i]["residual_loc0_mm"].get<double>() -
                       nominalElossOff[i]["residual_loc0_mm"].get<double>()));
        }
      }
    }
    row["eloss_on_off_max_abs_delta_mm"] =
        elossComparable ? json(maxElossDelta) : json(nullptr);
    row["eloss_on_off_surface_identity_ok"] = elossComparable;

    auto fillParam = [&](double delta, bool qp, bool ry, bool dx, bool ty) {
      json block = {{"plus", json::array()}, {"minus", json::array()}};
      for (double factor : kRungFactors) {
        const double step = delta * factor;
        for (double sign : {1.0, -1.0}) {
          const double dqp = qp ? sign * step : 0.0;
          const double dry = ry ? sign * step : 0.0;
          const double ddx = dx ? sign * step : 0.0;
          const double dty = ty ? sign * step : 0.0;
          const auto start =
              actsFromTrk(*params, geometryContext, dqp, dty);
          json residuals;
          if (!start) {
            residuals = json::array();
            for (const auto& hit : associated) {
              residuals.push_back(residualRow(hit, false, "acts_start_unavailable",
                                              0.0, 0.0, nullptr, hit.geoId.value(),
                                              false));
            }
          } else {
            residuals = evaluateResiduals(associated, *m_toolWithNoise, ctx,
                                          *start, geometryContext, sourceZ, dry,
                                          ddx);
          }
          if (sign > 0.0) {
            block["plus"].push_back(residuals);
          } else {
            block["minus"].push_back(residuals);
          }
        }
      }
      return block;
    };

    json fixed;
    fixed["q_over_p"] = fillParam(m_deltaQp.value(), true, false, false, false);
    fixed["R_y"] = fillParam(m_deltaRy.value(), false, true, false, false);
    fixed["d_x"] = fillParam(m_deltaDx.value(), false, false, true, false);
    fixed["t_y"] = fillParam(m_deltaTy.value(), false, false, false, true);
    row["fixed_state_rungs"] = fixed;
    row["profiled_rungs"] = nullptr;
    row["profiled_status"] =
        m_runProfiled.value()
            ? "deferred_no_fitter_change_this_stage"
            : "not_requested";
    row["profiled_note"] =
        "S3A does not change the fitter; profiled-track intervention is a "
        "separate later dump that re-minimizes remaining track nuisances "
        "without rewriting seed/hit/geometry.";
    row["native_5x5_not_used_as_jacobian_weight"] = true;
    row["primary_failure_class"] = nullptr;

    std::lock_guard<std::mutex> lock(m_mutex);
    m_trackRows.push_back(row.dump());
  }
  return StatusCode::SUCCESS;
}

StatusCode CkfYasuS3AJacobianAlg::finalize() {
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
  ATH_MSG_INFO("Wrote " << m_trackRows.size() << " Yasu-S3A track rows and "
                        << m_eventRows.size() << " event rows");
  return StatusCode::SUCCESS;
}
