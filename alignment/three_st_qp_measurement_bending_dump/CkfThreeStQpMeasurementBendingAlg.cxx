#include "CkfThreeStQpMeasurementBendingAlg.h"

#include "CLHEP/Geometry/Point3D.h"
#include "StoreGate/ReadHandle.h"
#include "TrackerIdentifier/FaserSCT_ID.h"
#include "TrackerPrepRawData/FaserSCT_Cluster.h"
#include "TrackerReadoutGeometry/SCT_DetectorManager.h"
#include "TrackerReadoutGeometry/SiDetectorElement.h"
#include "TrackerRIO_OnTrack/FaserSCT_ClusterOnTrack.h"
#include "TrackerSimData/TrackerSimDataCollection.h"
#include "TrackerSpacePoint/FaserSCT_SpacePoint.h"
#include "TrackerSimEvent/FaserSiHitCollection.h"
#include "TrkEventPrimitives/FitQuality.h"
#include "TrkEventPrimitives/ParamDefs.h"
#include "TrkParameters/TrackParameters.h"
#include "TrkTrack/Track.h"
#include "TrkTrack/TrackStateOnSurface.h"
#include "xAODTruth/TruthParticle.h"
#include "xAODTruth/TruthParticleContainer.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <string>

using json = nlohmann::json;

namespace {

constexpr int kIftStation = 0;
constexpr int kOfficialTruthStation = 1;

json finiteOrNull(double value) {
  return std::isfinite(value) ? json(value) : json(nullptr);
}

json nativeStateToJson(const AmgVector(5)& values) {
  json row = json::array();
  for (int i = 0; i < 5; ++i) {
    if (!std::isfinite(values[i])) {
      return nullptr;
    }
    row.push_back(values[i]);
  }
  return row;
}

double magnitude(const HepGeom::Point3D<double>& vector) {
  return std::sqrt(vector.x() * vector.x() + vector.y() * vector.y() +
                   vector.z() * vector.z());
}

json stationVector(const HepGeom::Point3D<double>& vector) {
  json payload;
  const double px = vector.x();
  const double py = vector.y();
  const double pz = vector.z();
  const double p = magnitude(vector);
  payload["px"] = finiteOrNull(px);
  payload["py"] = finiteOrNull(py);
  payload["pz"] = finiteOrNull(pz);
  payload["p"] = finiteOrNull(p);
  if (std::isfinite(pz) && std::abs(pz) > 1.0e-12 && std::isfinite(px) &&
      std::isfinite(py)) {
    payload["tx"] = px / pz;
    payload["ty"] = py / pz;
  } else {
    payload["tx"] = nullptr;
    payload["ty"] = nullptr;
  }
  payload["available"] = std::isfinite(p) && p > 0.0;
  return payload;
}

const Tracker::FaserSCT_ClusterOnTrack* asClusterOnTrack(
    const Trk::MeasurementBase* meas) {
  return dynamic_cast<const Tracker::FaserSCT_ClusterOnTrack*>(meas);
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

struct StationAccum {
  double x{0.0};
  double y{0.0};
  double z{0.0};
  int n{0};
  int n_space_points{0};
};

bool finiteVec(const Amg::Vector3D& pos) {
  return std::isfinite(pos.x()) && std::isfinite(pos.y()) && std::isfinite(pos.z());
}

bool reconstructClusterPosition(
    const Tracker::FaserSCT_ClusterOnTrack& cluster,
    const TrackerDD::SCT_DetectorManager* detManager,
    const std::map<Identifier, const Tracker::FaserSCT_SpacePoint*>& idToSp,
    const std::map<unsigned int, const Tracker::FaserSCT_SpacePoint*>& hashToSp,
    Amg::Vector3D& out, std::string& source) {
  auto fromSp = [&](const Tracker::FaserSCT_SpacePoint* spacePoint) {
    if (spacePoint == nullptr) {
      return false;
    }
    out = spacePoint->globalPosition();
    if (!finiteVec(out)) {
      return false;
    }
    source = "space_point";
    return true;
  };
  const Identifier id = cluster.identify();
  const auto idIt = idToSp.find(id);
  if (idIt != idToSp.end() && fromSp(idIt->second)) {
    return true;
  }
  const unsigned int hash = static_cast<unsigned int>(cluster.idDE());
  const auto hashIt = hashToSp.find(hash);
  if (hashIt != hashToSp.end() && fromSp(hashIt->second)) {
    return true;
  }
  const Tracker::FaserSCT_Cluster* prd = cluster.prepRawData();
  if (prd != nullptr) {
    out = prd->globalPosition();
    if (finiteVec(out)) {
      source = "prd_cluster";
      return true;
    }
  }
  const TrackerDD::SiDetectorElement* detEl = cluster.detectorElement();
  if (detEl == nullptr && detManager != nullptr) {
    detEl = detManager->getDetectorElement(id);
  }
  if (detEl != nullptr) {
    double along = cluster.positionAlongStrip();
    if (!std::isfinite(along)) {
      along = 0.0;
    }
    const Amg::Vector2D lpos(cluster.localParameters().get(Trk::locX), along);
    out = detEl->surface(id).localToGlobal(lpos);
    if (finiteVec(out)) {
      source = "detel_localToGlobal";
      return true;
    }
  }
  out = cluster.globalPosition();
  if (finiteVec(out)) {
    source = "rot_globalPosition";
    return true;
  }
  return false;
}

void collectMeasurements(
    const Trk::Track& track, const FaserSCT_ID& idHelper,
    const TrackerDD::SCT_DetectorManager* detManager,
    const std::map<Identifier, const Tracker::FaserSCT_SpacePoint*>& idToSp,
    const std::map<unsigned int, const Tracker::FaserSCT_SpacePoint*>& hashToSp,
    json& motStations, json& motHits, json& centroids, int& nMot, int& nMotIft,
    int& nOutlier) {
  motStations = json::array();
  motHits = json::array();
  nMot = 0;
  nMotIft = 0;
  nOutlier = 0;
  std::array<StationAccum, 4> acc{};
  if (track.measurementsOnTrack() != nullptr) {
    for (const Trk::MeasurementBase* meas : *track.measurementsOnTrack()) {
      const auto* cluster = asClusterOnTrack(meas);
      if (cluster == nullptr) {
        continue;
      }
      const int station = idHelper.station(cluster->identify());
      motStations.push_back(station);
      ++nMot;
      if (station == kIftStation) {
        ++nMotIft;
      }
      Amg::Vector3D pos;
      std::string source;
      const bool ok = reconstructClusterPosition(*cluster, detManager, idToSp,
                                                 hashToSp, pos, source);
      json hit;
      hit["station"] = station;
      hit["identifier"] = cluster->identify().get_compact();
      hit["position_source"] = ok ? json(source) : json(nullptr);
      hit["x_mm"] = ok ? finiteOrNull(pos.x()) : json(nullptr);
      hit["y_mm"] = ok ? finiteOrNull(pos.y()) : json(nullptr);
      hit["z_mm"] = ok ? finiteOrNull(pos.z()) : json(nullptr);
      const Amg::MatrixX& loc = cluster->localCovariance();
      if (loc.rows() >= 1 && loc.cols() >= 1 && std::isfinite(loc(0, 0))) {
        hit["local_cov00"] = loc(0, 0);
      } else {
        hit["local_cov00"] = nullptr;
      }
      motHits.push_back(hit);
      if (ok && station >= 0 && station < 4) {
        acc[static_cast<size_t>(station)].x += pos.x();
        acc[static_cast<size_t>(station)].y += pos.y();
        acc[static_cast<size_t>(station)].z += pos.z();
        acc[static_cast<size_t>(station)].n += 1;
        if (source == "space_point") {
          acc[static_cast<size_t>(station)].n_space_points += 1;
        }
      }
    }
  }
  if (track.trackStateOnSurfaces() != nullptr) {
    for (const Trk::TrackStateOnSurface* tsos : *track.trackStateOnSurfaces()) {
      if (tsos != nullptr && tsos->type(Trk::TrackStateOnSurface::Outlier)) {
        ++nOutlier;
      }
    }
  }
  centroids = json::object();
  for (int station = 1; station <= 3; ++station) {
    const StationAccum& item = acc[static_cast<size_t>(station)];
    json row;
    row["n"] = item.n;
    row["n_space_point_hits"] = item.n_space_points;
    if (item.n > 0) {
      row["x_mm"] = item.x / static_cast<double>(item.n);
      row["y_mm"] = item.y / static_cast<double>(item.n);
      row["z_mm"] = item.z / static_cast<double>(item.n);
    } else {
      row["x_mm"] = nullptr;
      row["y_mm"] = nullptr;
      row["z_mm"] = nullptr;
    }
    centroids[std::to_string(station)] = row;
  }
}

}  // namespace

CkfThreeStQpMeasurementBendingAlg::CkfThreeStQpMeasurementBendingAlg(
    const std::string& name, ISvcLocator* pSvcLocator)
    : AthAlgorithm(name, pSvcLocator) {}

StatusCode CkfThreeStQpMeasurementBendingAlg::initialize() {
  if (m_outputJsonl.empty() || m_sourceId.empty()) {
    ATH_MSG_ERROR("OutputJsonl and SourceId are required");
    return StatusCode::FAILURE;
  }
  if (m_officialTruthStation != kOfficialTruthStation) {
    ATH_MSG_ERROR("OfficialTruthStation must remain 1 (S1 / WithoutIFT front)");
    return StatusCode::FAILURE;
  }
  ATH_CHECK(m_trackKey.initialize());
  ATH_CHECK(m_eventKey.initialize());
  ATH_CHECK(m_clusterKey.initialize());
  ATH_CHECK(m_spacePointKey.initialize());
  ATH_CHECK(m_truthMatchingTool.retrieve());
  ATH_CHECK(m_fiducialTool.retrieve());
  ATH_CHECK(detStore()->retrieve(m_idHelper, "FaserSCT_ID"));
  ATH_CHECK(detStore()->retrieve(m_detManager, "SCT"));
  ATH_MSG_INFO("3ST measurement-bending dump ready: " << m_outputJsonl.value());
  return StatusCode::SUCCESS;
}

StatusCode CkfThreeStQpMeasurementBendingAlg::execute() {
  const EventContext& ctx = getContext();
  SG::ReadHandle<TrackCollection> tracks(m_trackKey, ctx);
  SG::ReadHandle<xAOD::EventInfo> eventInfo(m_eventKey, ctx);
  if (!tracks.isValid() || !eventInfo.isValid()) {
    ATH_MSG_ERROR("TrackCollection or EventInfo is missing");
    return StatusCode::FAILURE;
  }
  const int skipIndex = m_skipEvents.value() + m_eventsSeen;
  ++m_eventsSeen;

  std::map<Identifier, const Tracker::FaserSCT_SpacePoint*> idToSp;
  std::map<unsigned int, const Tracker::FaserSCT_SpacePoint*> hashToSp;
  SG::ReadHandle<FaserSCT_SpacePointContainer> spacePoints(m_spacePointKey, ctx);
  SG::ReadHandle<Tracker::FaserSCT_ClusterContainer> clusters(m_clusterKey, ctx);
  if (spacePoints.isValid()) {
    for (const FaserSCT_SpacePointCollection* collection : *spacePoints) {
      if (collection == nullptr) {
        continue;
      }
      for (const Tracker::FaserSCT_SpacePoint* spacePoint : *collection) {
        if (spacePoint == nullptr) {
          continue;
        }
        hashToSp[static_cast<unsigned int>(spacePoint->elementIdList().first)] =
            spacePoint;
        hashToSp[static_cast<unsigned int>(spacePoint->elementIdList().second)] =
            spacePoint;
        if (spacePoint->cluster1() != nullptr) {
          idToSp[spacePoint->cluster1()->identify()] = spacePoint;
        }
        if (spacePoint->cluster2() != nullptr) {
          idToSp[spacePoint->cluster2()->identify()] = spacePoint;
        }
      }
    }
  }

  const bool sdoPresent =
      evtStore()->contains<TrackerSimDataCollection>("SCT_SDO_Map");
  const bool truthPresent =
      evtStore()->contains<xAOD::TruthParticleContainer>("TruthParticles");
  const bool hitsPresent = evtStore()->contains<FaserSiHitCollection>("SCT_Hits");

  json event;
  event["kind"] = "event";
  event["source_id"] = m_sourceId.value();
  event["campaign"] = m_campaign.value();
  event["run_id"] = static_cast<int>(eventInfo->runNumber());
  event["event_id"] = static_cast<int>(eventInfo->eventNumber());
  event["skip_index"] = skipIndex;
  event["collection"] = "CKFTrackCollectionWithoutIFT";
  event["n_without_ift_tracks"] = static_cast<int>(tracks->size());
  event["selection_loss"] = tracks->empty();
  event["sdo_map_present"] = sdoPresent;
  event["truth_particles_present"] = truthPresent;
  event["sct_hits_present"] = hitsPresent;
  event["space_point_container_present"] = spacePoints.isValid();
  event["cluster_container_present"] = clusters.isValid();
  event["n_space_points_mapped"] = static_cast<int>(hashToSp.size());
  event["bending_constructed_in_cpp"] = false;
  event["station_decoded_by_fasersct_id"] = true;
  event["official_truth_station"] = m_officialTruthStation.value();
  event["is_truth"] = false;
  event["geometry_hash"] = m_geometryHash.value();
  event["field_hash"] = m_fieldHash.value();
  event["material_hash"] = m_materialHash.value();
  event["conditions_hash"] = m_conditionsHash.value();
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_eventRows.push_back(event.dump());
  }

  int trackIndex = -1;
  for (const Trk::Track* track : *tracks) {
    ++trackIndex;
    json row;
    row["kind"] = "track";
    row["source_id"] = m_sourceId.value();
    row["campaign"] = m_campaign.value();
    row["run_id"] = static_cast<int>(eventInfo->runNumber());
    row["event_id"] = static_cast<int>(eventInfo->eventNumber());
    row["skip_index"] = skipIndex;
    row["track_index"] = trackIndex;
    row["collection"] = "CKFTrackCollectionWithoutIFT";
    row["is_truth"] = false;
    row["truth_used_as_fit_seed"] = false;
    row["truth_used_as_solution"] = false;
    row["truth_role"] = "calibration_reference_only";
    row["official_truth_station"] = m_officialTruthStation.value();
    row["station_decoded_by_fasersct_id"] = true;
    row["bending_constructed_in_cpp"] = false;
    row["bending_uses_fitted_q_over_p"] = false;
    row["centroid_source"] = "MOT_cluster_globalPosition_mean";
    row["geometry_hash"] = m_geometryHash.value();
    row["field_hash"] = m_fieldHash.value();
    row["material_hash"] = m_materialHash.value();
    row["conditions_hash"] = m_conditionsHash.value();

    if (track == nullptr) {
      row["has_parameters"] = false;
      row["truth_matched"] = false;
      row["truth_reference_available"] = false;
      std::lock_guard<std::mutex> lock(m_mutex);
      m_trackRows.push_back(row.dump());
      continue;
    }

    json motStations;
    json motHits;
    json centroids;
    int nMot = 0;
    int nMotIft = 0;
    int nOutlier = 0;
    collectMeasurements(*track, *m_idHelper, m_detManager, idToSp, hashToSp,
                        motStations, motHits, centroids, nMot, nMotIft,
                        nOutlier);
    row["measurements_on_track_stations"] = motStations;
    row["mot_hits"] = motHits;
    row["station_centroids"] = centroids;
    row["n_mot"] = nMot;
    row["n_ift_mot"] = nMotIft;
    row["n_outlier_hits"] = nOutlier;
    row["ift_leak"] = nMotIft > 0;

    if (track->trackParameters() == nullptr ||
        track->trackParameters()->empty() ||
        track->trackParameters()->front() == nullptr) {
      row["has_parameters"] = false;
      row["q_over_p_fit_per_mev"] = nullptr;
      row["sigma_q_over_p_per_mev"] = nullptr;
      row["x_mm"] = nullptr;
      row["y_mm"] = nullptr;
      row["z_mm"] = nullptr;
    } else {
      const Trk::TrackParameters* params = track->trackParameters()->front();
      const AmgVector(5)& native = params->parameters();
      const Amg::Vector3D& position = params->position();
      row["has_parameters"] = true;
      row["native_state"] = nativeStateToJson(native);
      row["q_over_p_fit_per_mev"] = finiteOrNull(native[Trk::qOverP]);
      row["fit_charge"] = finiteOrNull(params->charge());
      row["x_mm"] = finiteOrNull(position.x());
      row["y_mm"] = finiteOrNull(position.y());
      row["z_mm"] = finiteOrNull(position.z());
      row["front_xyz_is_not_a_bending_input"] = true;
      const AmgSymMatrix(5)* cov = params->covariance();
      if (cov == nullptr) {
        row["sigma_q_over_p_per_mev"] = nullptr;
      } else {
        const double variance = (*cov)(Trk::qOverP, Trk::qOverP);
        row["sigma_q_over_p_per_mev"] =
            (std::isfinite(variance) && variance >= 0.0)
                ? json(std::sqrt(variance))
                : json(nullptr);
      }
    }

    const Trk::FitQuality* quality = track->fitQuality();
    if (quality != nullptr) {
      row["chi2"] = finiteOrNull(quality->chiSquared());
      row["ndof"] = finiteOrNull(quality->numberDoF());
    } else {
      row["chi2"] = nullptr;
      row["ndof"] = nullptr;
    }

    auto [truthParticle, hitCount] = m_truthMatchingTool->getTruthParticle(track);
    row["truth_hit_count"] = hitCount;
    bool matched = truthParticle != nullptr;
    bool referenceAvailable = false;
    row["truth_matched"] = matched;
    if (!matched) {
      row["truth_barcode"] = nullptr;
      row["truth_pdg"] = nullptr;
      row["truth_charge"] = nullptr;
      row["q_over_p_truth_s1_per_mev"] = nullptr;
      row["p_truth_s1_mev"] = nullptr;
    } else {
      row["truth_barcode"] = truthParticle->barcode();
      row["truth_pdg"] = truthParticle->pdgId();
      row["truth_charge"] = finiteOrNull(truthParticle->charge());
      const auto momenta = m_fiducialTool->getTruthMomenta(truthParticle->barcode());
      json stations = json::array();
      for (int station = 0; station < 4; ++station) {
        json item = stationVector(momenta[station]);
        item["station"] = station;
        stations.push_back(item);
      }
      row["truth_station_momenta"] = stations;
      const json& s1 = stations[kOfficialTruthStation];
      row["p_truth_s1_mev"] = s1["p"];
      if (s1["available"].get<bool>() && s1["p"].is_number() &&
          std::isfinite(truthParticle->charge())) {
        const double pS1 = s1["p"].get<double>();
        row["q_over_p_truth_s1_per_mev"] = truthParticle->charge() / pS1;
        referenceAvailable = true;
      } else {
        row["q_over_p_truth_s1_per_mev"] = nullptr;
      }
    }
    row["truth_reference_available"] = referenceAvailable;

    std::lock_guard<std::mutex> lock(m_mutex);
    m_trackRows.push_back(row.dump());
  }
  return StatusCode::SUCCESS;
}

StatusCode CkfThreeStQpMeasurementBendingAlg::finalize() {
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
  ATH_MSG_INFO("Wrote " << m_trackRows.size()
                        << " measurement-bending track rows and "
                        << m_eventRows.size() << " event rows");
  return StatusCode::SUCCESS;
}
