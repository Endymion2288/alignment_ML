#include "CkfThreeStQpRefitProvenanceAlg.h"

#include "CLHEP/Geometry/Point3D.h"
#include "StoreGate/ReadHandle.h"
#include "TrackerIdentifier/FaserSCT_ID.h"
#include "TrackerRIO_OnTrack/FaserSCT_ClusterOnTrack.h"
#include "TrkEventPrimitives/FitQuality.h"
#include "TrkEventPrimitives/FitQualityOnSurface.h"
#include "TrkEventPrimitives/ParamDefs.h"
#include "TrkParameters/TrackParameters.h"
#include "TrkTrack/Track.h"
#include "TrkTrack/TrackStateOnSurface.h"
#include "xAODTruth/TruthParticle.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>

using json = nlohmann::json;

namespace {

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

json snapshotParameters(const Trk::TrackParameters* params) {
  json out;
  if (params == nullptr) {
    out["has_parameters"] = false;
    return out;
  }
  const AmgVector(5)& native = params->parameters();
  const Amg::Vector3D& position = params->position();
  out["has_parameters"] = true;
  out["native_state"] = nativeStateToJson(native);
  out["q_over_p_per_mev"] = finiteOrNull(native[Trk::qOverP]);
  out["x_mm"] = finiteOrNull(position.x());
  out["y_mm"] = finiteOrNull(position.y());
  out["z_mm"] = finiteOrNull(position.z());
  const AmgSymMatrix(5)* cov = params->covariance();
  if (cov == nullptr) {
    out["has_covariance"] = false;
    out["native_covariance"] = nullptr;
    out["sigma_q_over_p_per_mev"] = nullptr;
  } else {
    out["has_covariance"] = true;
    out["native_covariance"] = nativeMatrixToJson(*cov);
    const double variance = (*cov)(Trk::qOverP, Trk::qOverP);
    out["sigma_q_over_p_per_mev"] =
        (std::isfinite(variance) && variance >= 0.0) ? json(std::sqrt(variance))
                                                     : json(nullptr);
  }
  return out;
}

}  // namespace

CkfThreeStQpRefitProvenanceAlg::CkfThreeStQpRefitProvenanceAlg(
    const std::string& name, ISvcLocator* pSvcLocator)
    : AthAlgorithm(name, pSvcLocator) {}

StatusCode CkfThreeStQpRefitProvenanceAlg::initialize() {
  if (m_outputJsonl.empty() || m_sourceId.empty()) {
    ATH_MSG_ERROR("OutputJsonl and SourceId are required");
    return StatusCode::FAILURE;
  }
  if (m_officialTruthStation != kOfficialTruthStation) {
    ATH_MSG_ERROR("OfficialTruthStation must remain 1");
    return StatusCode::FAILURE;
  }
  ATH_CHECK(m_trackKey.initialize());
  ATH_CHECK(m_eventKey.initialize());
  ATH_CHECK(m_truthMatchingTool.retrieve());
  ATH_CHECK(m_fiducialTool.retrieve());
  ATH_CHECK(detStore()->retrieve(m_idHelper, "FaserSCT_ID"));
  std::stringstream stream(m_focusPairs.value());
  std::string item;
  while (std::getline(stream, item, ',')) {
    if (item.empty()) {
      continue;
    }
    const auto colon = item.find(':');
    if (colon == std::string::npos) {
      continue;
    }
    m_focus.emplace(std::stoi(item.substr(0, colon)),
                    std::stoi(item.substr(colon + 1)));
  }
  if (m_focus.empty() || m_focus.size() > 20) {
    ATH_MSG_ERROR("FocusPairs must contain 1–20 event:track identities");
    return StatusCode::FAILURE;
  }
  ATH_MSG_INFO("3ST q/p refit-provenance dump ready, n_focus=" << m_focus.size());
  return StatusCode::SUCCESS;
}

StatusCode CkfThreeStQpRefitProvenanceAlg::execute() {
  const EventContext& ctx = getContext();
  SG::ReadHandle<TrackCollection> tracks(m_trackKey, ctx);
  SG::ReadHandle<xAOD::EventInfo> eventInfo(m_eventKey, ctx);
  if (!tracks.isValid() || !eventInfo.isValid()) {
    ATH_MSG_ERROR("TrackCollection or EventInfo is missing");
    return StatusCode::FAILURE;
  }
  const int eventId = static_cast<int>(eventInfo->eventNumber());
  const int runId = static_cast<int>(eventInfo->runNumber());
  const int skipIndex = m_skipEvents.value() + m_eventsSeen;
  ++m_eventsSeen;

  json event;
  event["kind"] = "event";
  event["source_id"] = m_sourceId.value();
  event["campaign"] = m_campaign.value();
  event["run_id"] = runId;
  event["event_id"] = eventId;
  event["skip_index"] = skipIndex;
  event["collection"] = "CKFTrackCollectionWithoutIFT";
  event["n_without_ift_tracks"] = static_cast<int>(tracks->size());
  event["truth_used_as_fit_seed"] = false;
  event["truth_used_as_solution"] = false;

  int written = 0;
  int trackIndex = -1;
  for (const Trk::Track* track : *tracks) {
    ++trackIndex;
    if (!m_focus.empty() && m_focus.count({eventId, trackIndex}) == 0) {
      continue;
    }
    json row;
    row["kind"] = "track";
    row["source_id"] = m_sourceId.value();
    row["campaign"] = m_campaign.value();
    row["run_id"] = runId;
    row["event_id"] = eventId;
    row["skip_index"] = skipIndex;
    row["track_index"] = trackIndex;
    row["collection"] = "CKFTrackCollectionWithoutIFT";
    row["truth_used_as_fit_seed"] = false;
    row["truth_used_as_solution"] = false;
    row["kf_refit_attempted"] = true;

    if (track == nullptr || track->trackParameters() == nullptr ||
        track->trackParameters()->empty() ||
        track->trackParameters()->front() == nullptr) {
      row["has_parameters"] = false;
      row["persisted_front_kind"] = "other";
      std::lock_guard<std::mutex> lock(m_mutex);
      m_trackRows.push_back(row.dump());
      ++written;
      continue;
    }

    const Trk::TrackParameters* params = track->trackParameters()->front();
    json persisted = snapshotParameters(params);
    row["has_parameters"] = persisted["has_parameters"];
    row["native_state"] = persisted["native_state"];
    row["q_over_p_fit_per_mev"] = persisted["q_over_p_per_mev"];
    row["x_mm"] = persisted["x_mm"];
    row["y_mm"] = persisted["y_mm"];
    row["z_mm"] = persisted["z_mm"];
    row["has_covariance"] = persisted["has_covariance"];
    row["native_covariance"] = persisted["native_covariance"];
    row["sigma_q_over_p_per_mev"] = persisted["sigma_q_over_p_per_mev"];

    bool frontHole = false;
    bool frontMeas = false;
    double frontChi2 = std::numeric_limits<double>::quiet_NaN();
    double frontNdof = std::numeric_limits<double>::quiet_NaN();
    json tsosList = json::array();
    int nMot = 0;
    int nTsos = 0;
    json motStations = json::array();
    if (track->trackStateOnSurfaces() != nullptr) {
      bool first = true;
      for (const Trk::TrackStateOnSurface* tsos : *track->trackStateOnSurfaces()) {
        if (tsos == nullptr) {
          continue;
        }
        ++nTsos;
        json item;
        item["is_hole"] = tsos->type(Trk::TrackStateOnSurface::Hole);
        item["is_measurement"] = tsos->type(Trk::TrackStateOnSurface::Measurement);
        item["is_outlier"] = tsos->type(Trk::TrackStateOnSurface::Outlier);
        const Trk::FitQualityOnSurface fq = tsos->fitQualityOnSurface();
        item["chi2"] = finiteOrNull(fq.chiSquared());
        item["ndof"] = finiteOrNull(fq.numberDoF());
        if (tsos->trackParameters() != nullptr) {
          item["z_mm"] = finiteOrNull(tsos->trackParameters()->position().z());
        }
        const auto* cluster = asClusterOnTrack(tsos->measurementOnTrack());
        if (cluster != nullptr) {
          const int station = m_idHelper->station(cluster->identify());
          item["station"] = station;
          if (tsos->type(Trk::TrackStateOnSurface::Measurement)) {
            motStations.push_back(station);
            ++nMot;
          }
        }
        tsosList.push_back(item);
        if (first) {
          frontHole = tsos->type(Trk::TrackStateOnSurface::Hole);
          frontMeas = tsos->type(Trk::TrackStateOnSurface::Measurement);
          frontChi2 = fq.chiSquared();
          frontNdof = fq.numberDoF();
          first = false;
        }
      }
    }
    row["n_mot"] = nMot;
    row["n_tsos"] = nTsos;
    row["measurements_on_track_stations"] = motStations;
    {
      std::set<int> have;
      for (const auto& station : motStations) {
        have.insert(station.get<int>());
      }
      const bool s1 = have.count(1) > 0;
      const bool s2 = have.count(2) > 0;
      const bool s3 = have.count(3) > 0;
      const int nStations = static_cast<int>(s1) + static_cast<int>(s2) + static_cast<int>(s3);
      if (nStations == 3) {
        row["missing_pattern"] = "complete";
      } else if (!s1 && nStations == 2) {
        row["missing_pattern"] = "missing_s1";
      } else if (nStations <= 1) {
        row["missing_pattern"] = "missing_two_or_more";
      } else if (!s2) {
        row["missing_pattern"] = "missing_s2";
      } else if (!s3) {
        row["missing_pattern"] = "missing_s3";
      } else {
        row["missing_pattern"] = "missing_other";
      }
    }
    row["front_surface_type"] = params->associatedSurface().name();
    row["front_is_hole"] = frontHole;
    row["front_is_measurement"] = frontMeas;
    row["front_fit_quality_chi2"] = finiteOrNull(frontChi2);
    row["front_fit_quality_ndof"] = finiteOrNull(frontNdof);
    row["tsos"] = tsosList;
    const bool ckfHole =
        frontHole && std::isfinite(frontChi2) && std::isfinite(frontNdof) &&
        std::abs(frontChi2 + 99.0) <= 1.0e-6 && std::abs(frontNdof) <= 1.0e-6;
    if (ckfHole) {
      row["persisted_front_kind"] = "ckf_target_hole";
      row["kf_refit_succeeded"] = false;
    } else if (frontMeas) {
      row["persisted_front_kind"] = "first_mot";
      row["kf_refit_succeeded"] = true;
    } else {
      row["persisted_front_kind"] = "other";
      row["kf_refit_succeeded"] = nullptr;
    }

    const Trk::FitQuality* quality = track->fitQuality();
    if (quality != nullptr) {
      row["chi2"] = finiteOrNull(quality->chiSquared());
      row["ndof"] = finiteOrNull(quality->numberDoF());
    }

    row["diagnostic_refit_attempted"] = false;
    row["diagnostic_refit_succeeded"] = nullptr;
    row["diagnostic_fallback_reason"] = nullptr;
    row["diagnostic_q_over_p_per_mev"] = nullptr;
    row["diagnostic_sigma_q_over_p_per_mev"] = nullptr;
    row["diagnostic_z_mm"] = nullptr;
    row["original_complementary_state_in_xaod"] = false;
    if (ckfHole) {
      row["persisted_state_role"] = "ckf_pre_refit";
    } else if (frontMeas) {
      row["persisted_state_role"] = "kf_post_refit_first_mot";
    } else {
      row["persisted_state_role"] = "unknown";
    }

    auto [truthParticle, hitCount] = m_truthMatchingTool->getTruthParticle(track);
    row["truth_matched"] = truthParticle != nullptr;
    row["truth_hit_count"] = hitCount;
    if (truthParticle != nullptr) {
      row["truth_charge"] = finiteOrNull(truthParticle->charge());
      const auto momenta = m_fiducialTool->getTruthMomenta(truthParticle->barcode());
      const HepGeom::Point3D<double>& s1 = momenta[kOfficialTruthStation];
      const double pS1 = std::sqrt(s1.x() * s1.x() + s1.y() * s1.y() + s1.z() * s1.z());
      if (std::isfinite(pS1) && pS1 > 0.0 && std::isfinite(truthParticle->charge())) {
        row["p_truth_s1_mev"] = pS1;
        row["q_over_p_truth_s1_per_mev"] = truthParticle->charge() / pS1;
        row["truth_reference_available"] = true;
        if (std::abs(s1.z()) > 1.0e-12) {
          row["tx_truth"] = s1.x() / s1.z();
          row["ty_truth"] = s1.y() / s1.z();
        }
      } else {
        row["truth_reference_available"] = false;
        row["q_over_p_truth_s1_per_mev"] = nullptr;
      }
    } else {
      row["truth_reference_available"] = false;
      row["q_over_p_truth_s1_per_mev"] = nullptr;
    }

    std::lock_guard<std::mutex> lock(m_mutex);
    m_trackRows.push_back(row.dump());
    ++written;
  }
  if (written > 0) {
    event["n_written_tracks"] = written;
    std::lock_guard<std::mutex> lock(m_mutex);
    m_eventRows.push_back(event.dump());
  }
  return StatusCode::SUCCESS;
}

StatusCode CkfThreeStQpRefitProvenanceAlg::finalize() {
  std::lock_guard<std::mutex> lock(m_mutex);
  if (!writeLines(m_outputJsonl.value(), m_trackRows)) {
    ATH_MSG_ERROR("Cannot write track jsonl: " << m_outputJsonl.value());
    return StatusCode::FAILURE;
  }
  if (!writeLines(m_eventJsonl.value(), m_eventRows)) {
    ATH_MSG_ERROR("Cannot write event jsonl: " << m_eventJsonl.value());
    return StatusCode::FAILURE;
  }
  ATH_MSG_INFO("Wrote " << m_trackRows.size() << " provenance tracks");
  return StatusCode::SUCCESS;
}
