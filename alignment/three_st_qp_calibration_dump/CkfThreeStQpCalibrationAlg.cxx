#include "CkfThreeStQpCalibrationAlg.h"

#include "CLHEP/Geometry/Point3D.h"
#include "StoreGate/ReadHandle.h"
#include "TrackerIdentifier/FaserSCT_ID.h"
#include "TrackerRIO_OnTrack/FaserSCT_ClusterOnTrack.h"
#include "TrackerSimData/TrackerSimDataCollection.h"
#include "TrackerSimEvent/FaserSiHitCollection.h"
#include "TrkEventPrimitives/ParamDefs.h"
#include "TrkParameters/TrackParameters.h"
#include "TrkTrack/Track.h"
#include "TrkTrack/TrackStateOnSurface.h"
#include "xAODTruth/TruthParticle.h"
#include "xAODTruth/TruthParticleContainer.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <limits>

using json = nlohmann::json;

namespace {

constexpr int kIftStation = 0;
constexpr int kOfficialTruthStation = 1;

json finiteOrNull(double value) {
  return std::isfinite(value) ? json(value) : json(nullptr);
}

json nativeMatrixToJson(const AmgSymMatrix(5)& matrix) {
  json rows = json::array();
  for (int i = 0; i < 5; ++i) {
    json row = json::array();
    for (int j = 0; j < 5; ++j) {
      const double value = matrix(i, j);
      if (!std::isfinite(value)) {
        return nullptr;
      }
      row.push_back(value);
    }
    rows.push_back(row);
  }
  return rows;
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

void collectStations(const Trk::Track& track, const FaserSCT_ID& idHelper,
                     json& motStations, json& tsosStations, int& nMot,
                     int& nMotIft, int& nTsosIft, int& nOutlier) {
  motStations = json::array();
  tsosStations = json::array();
  nMot = 0;
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
      ++nMot;
      if (station == kIftStation) {
        ++nMotIft;
      }
    }
  }
  if (track.trackStateOnSurfaces() != nullptr) {
    for (const Trk::TrackStateOnSurface* tsos : *track.trackStateOnSurfaces()) {
      if (tsos == nullptr || tsos->measurementOnTrack() == nullptr) {
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

CkfThreeStQpCalibrationAlg::CkfThreeStQpCalibrationAlg(
    const std::string& name, ISvcLocator* pSvcLocator)
    : AthAlgorithm(name, pSvcLocator) {}

StatusCode CkfThreeStQpCalibrationAlg::initialize() {
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
  ATH_CHECK(m_truthMatchingTool.retrieve());
  ATH_CHECK(m_fiducialTool.retrieve());
  ATH_CHECK(detStore()->retrieve(m_idHelper, "FaserSCT_ID"));
  ATH_MSG_INFO("3ST q/p calibration dump ready: " << m_outputJsonl.value());
  return StatusCode::SUCCESS;
}

StatusCode CkfThreeStQpCalibrationAlg::execute() {
  const EventContext& ctx = getContext();
  SG::ReadHandle<TrackCollection> tracks(m_trackKey, ctx);
  SG::ReadHandle<xAOD::EventInfo> eventInfo(m_eventKey, ctx);
  if (!tracks.isValid() || !eventInfo.isValid()) {
    ATH_MSG_ERROR("TrackCollection or EventInfo is missing");
    return StatusCode::FAILURE;
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
  event["collection"] = "CKFTrackCollectionWithoutIFT";
  event["n_without_ift_tracks"] = static_cast<int>(tracks->size());
  event["selection_loss"] = tracks->empty();
  event["sdo_map_present"] = sdoPresent;
  event["truth_particles_present"] = truthPresent;
  event["sct_hits_present"] = hitsPresent;
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
    row["track_index"] = trackIndex;
    row["collection"] = "CKFTrackCollectionWithoutIFT";
    row["is_truth"] = false;
    row["truth_used_as_fit_seed"] = false;
    row["truth_used_as_solution"] = false;
    row["truth_role"] = "calibration_reference_only";
    row["official_truth_station"] = m_officialTruthStation.value();
    row["station_decoded_by_fasersct_id"] = true;
    row["geometry_hash"] = m_geometryHash.value();
    row["field_hash"] = m_fieldHash.value();
    row["material_hash"] = m_materialHash.value();
    row["conditions_hash"] = m_conditionsHash.value();

    if (track == nullptr || track->trackParameters() == nullptr ||
        track->trackParameters()->empty() ||
        track->trackParameters()->front() == nullptr) {
      row["has_parameters"] = false;
      row["has_covariance"] = false;
      row["truth_matched"] = false;
      row["truth_reference_available"] = false;
      row["ift_leak"] = false;
      row["primary_failure_class"] = "fit_failure";
      std::lock_guard<std::mutex> lock(m_mutex);
      m_trackRows.push_back(row.dump());
      continue;
    }

    const Trk::TrackParameters* params = track->trackParameters()->front();
    const AmgVector(5)& native = params->parameters();
    const Amg::Vector3D& position = params->position();
    const Amg::Vector3D& momentum = params->momentum();
    const double pFit = momentum.mag();
    double tx = std::numeric_limits<double>::quiet_NaN();
    double ty = std::numeric_limits<double>::quiet_NaN();
    if (std::abs(momentum.z()) > 1.0e-12) {
      tx = momentum.x() / momentum.z();
      ty = momentum.y() / momentum.z();
    }
    json motStations;
    json tsosStations;
    int nMot = 0;
    int nMotIft = 0;
    int nTsosIft = 0;
    int nOutlier = 0;
    collectStations(*track, *m_idHelper, motStations, tsosStations, nMot, nMotIft,
                    nTsosIft, nOutlier);
    const bool leak = nMotIft > 0 || nTsosIft > 0;

    row["has_parameters"] = true;
    row["parameter_type"] = "Trk::CurvilinearParameters";
    row["native_state"] = nativeStateToJson(native);
    row["q_over_p_fit_per_mev"] = finiteOrNull(native[Trk::qOverP]);
    row["fit_charge"] = finiteOrNull(params->charge());
    row["p_fit_mev"] = finiteOrNull(pFit);
    row["x_mm"] = finiteOrNull(position.x());
    row["y_mm"] = finiteOrNull(position.y());
    row["z_mm"] = finiteOrNull(position.z());
    row["tx"] = finiteOrNull(tx);
    row["ty"] = finiteOrNull(ty);
    row["phi"] = finiteOrNull(native[Trk::phi]);
    row["theta"] = finiteOrNull(native[Trk::theta]);
    row["measurements_on_track_stations"] = motStations;
    row["tsos_stations"] = tsosStations;
    row["n_mot"] = nMot;
    row["n_ift_mot"] = nMotIft;
    row["n_ift_tsos"] = nTsosIft;
    row["n_outlier_hits"] = nOutlier;
    row["ift_leak"] = leak;

    const Trk::FitQuality* quality = track->fitQuality();
    if (quality != nullptr) {
      row["chi2"] = finiteOrNull(quality->chiSquared());
      row["ndof"] = finiteOrNull(quality->numberDoF());
    } else {
      row["chi2"] = nullptr;
      row["ndof"] = nullptr;
    }

    const AmgSymMatrix(5)* cov = params->covariance();
    if (cov == nullptr) {
      row["has_covariance"] = false;
      row["native_covariance"] = nullptr;
      row["sigma_q_over_p_per_mev"] = nullptr;
      row["qoverp_cross_terms"] = nullptr;
    } else {
      row["has_covariance"] = true;
      row["native_covariance"] = nativeMatrixToJson(*cov);
      const double variance = (*cov)(Trk::qOverP, Trk::qOverP);
      row["sigma_q_over_p_per_mev"] =
          (std::isfinite(variance) && variance >= 0.0)
              ? json(std::sqrt(variance))
              : json(nullptr);
      json cross = json::array();
      bool crossOk = true;
      for (int i = 0; i < 4; ++i) {
        const double value = (*cov)(Trk::qOverP, i);
        if (!std::isfinite(value)) {
          crossOk = false;
          break;
        }
        cross.push_back(value);
      }
      row["qoverp_cross_terms"] = crossOk ? cross : json(nullptr);
    }

    auto [truthParticle, hitCount] = m_truthMatchingTool->getTruthParticle(track);
    row["truth_hit_count"] = hitCount;
    if (nMot > 0 && hitCount >= 0) {
      row["match_fraction_mot"] =
          static_cast<double>(hitCount) / static_cast<double>(nMot);
    } else {
      row["match_fraction_mot"] = nullptr;
    }
    if (quality != nullptr && (quality->numberDoF() + 5.0) > 0.0 && hitCount >= 0) {
      row["match_fraction_ckf"] =
          static_cast<double>(hitCount) / (quality->numberDoF() + 5.0);
    } else {
      row["match_fraction_ckf"] = nullptr;
    }

    bool matched = truthParticle != nullptr;
    bool referenceAvailable = false;
    row["truth_matched"] = matched;
    if (!matched) {
      row["truth_barcode"] = nullptr;
      row["truth_pdg"] = nullptr;
      row["truth_charge"] = nullptr;
      row["q_over_p_truth_s1_per_mev"] = nullptr;
      row["p_truth_s1_mev"] = nullptr;
      row["tx_truth"] = nullptr;
      row["ty_truth"] = nullptr;
      row["q_over_p_truth_production_per_mev"] = nullptr;
      row["p_truth_production_mev"] = nullptr;
      row["truth_station_momenta"] = nullptr;
      row["is_fiducial"] = nullptr;
    } else {
      row["truth_barcode"] = truthParticle->barcode();
      row["truth_pdg"] = truthParticle->pdgId();
      row["truth_charge"] = finiteOrNull(truthParticle->charge());
      const double pProd = truthParticle->p4().P();
      row["p_truth_production_mev"] = finiteOrNull(pProd);
      if (std::isfinite(pProd) && pProd > 0.0 &&
          std::isfinite(truthParticle->charge())) {
        row["q_over_p_truth_production_per_mev"] = truthParticle->charge() / pProd;
      } else {
        row["q_over_p_truth_production_per_mev"] = nullptr;
      }
      row["truth_production_px"] = finiteOrNull(truthParticle->p4().Px());
      row["truth_production_py"] = finiteOrNull(truthParticle->p4().Py());
      row["truth_production_pz"] = finiteOrNull(truthParticle->p4().Pz());
      row["is_fiducial"] = m_fiducialTool->isFiducial(truthParticle->barcode());

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
      row["tx_truth"] = s1["tx"];
      row["ty_truth"] = s1["ty"];
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
    row["match_fraction_is_diagnostic_not_a_cut"] = true;

    if (leak) {
      row["primary_failure_class"] = "ift_measurement_leak";
    } else if (!row["has_covariance"].get<bool>() ||
               row["native_state"].is_null()) {
      row["primary_failure_class"] = "fit_failure";
    } else if (!matched) {
      row["primary_failure_class"] = "truth_match_unavailable";
    } else if (!referenceAvailable) {
      row["primary_failure_class"] = "truth_reference_unavailable";
    } else {
      row["primary_failure_class"] = nullptr;
    }

    std::lock_guard<std::mutex> lock(m_mutex);
    m_trackRows.push_back(row.dump());
  }
  return StatusCode::SUCCESS;
}

StatusCode CkfThreeStQpCalibrationAlg::finalize() {
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
  ATH_MSG_INFO("Wrote " << m_trackRows.size() << " 3ST q/p track rows and "
                        << m_eventRows.size() << " event rows");
  return StatusCode::SUCCESS;
}
