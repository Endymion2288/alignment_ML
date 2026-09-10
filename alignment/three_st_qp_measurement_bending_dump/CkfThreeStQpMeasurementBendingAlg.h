#ifndef ALIGNMENT_ML_CKF_THREE_ST_QP_MEASUREMENT_BENDING_ALG_H
#define ALIGNMENT_ML_CKF_THREE_ST_QP_MEASUREMENT_BENDING_ALG_H

#include "AthenaBaseComps/AthAlgorithm.h"
#include "FaserActsKalmanFilter/IFiducialParticleTool.h"
#include "FaserActsKalmanFilter/ITrackTruthMatchingTool.h"
#include "Gaudi/Property.h"
#include "GaudiKernel/ToolHandle.h"
#include "StoreGate/ReadHandleKey.h"
#include "TrackerPrepRawData/FaserSCT_ClusterContainer.h"
#include "TrackerSpacePoint/FaserSCT_SpacePointContainer.h"
#include "TrkTrack/TrackCollection.h"
#include "xAODEventInfo/EventInfo.h"

#include <mutex>
#include <string>
#include <vector>

class FaserSCT_ID;
namespace TrackerDD {
class SCT_DetectorManager;
}

/// Inspect-only MOT centroid dump for Yasu-S2I.  Reads official
/// CKFTrackCollectionWithoutIFT.  Does not refit, does not write
/// /Tracker/Align, and does not construct bending_raw in C++.
/// Truth is a calibration reference only.
class CkfThreeStQpMeasurementBendingAlg : public AthAlgorithm {
 public:
  CkfThreeStQpMeasurementBendingAlg(const std::string& name,
                                    ISvcLocator* pSvcLocator);
  virtual ~CkfThreeStQpMeasurementBendingAlg() = default;

  StatusCode initialize() override;
  StatusCode execute() override;
  StatusCode finalize() override;

 private:
  SG::ReadHandleKey<TrackCollection> m_trackKey{
      this, "TrackCollection", "CKFTrackCollectionWithoutIFT",
      "Official 3-station CKF; comparison q/p only"};
  SG::ReadHandleKey<xAOD::EventInfo> m_eventKey{
      this, "EventInfoKey", "EventInfo", "Event identity"};
  SG::ReadHandleKey<Tracker::FaserSCT_ClusterContainer> m_clusterKey{
      this, "ClusterContainer", "SCT_ClusterContainer",
      "Resolves PRD ElementLinks; not a fit input"};
  SG::ReadHandleKey<FaserSCT_SpacePointContainer> m_spacePointKey{
      this, "SpacePointContainer", "SCT_SpacePointContainer",
      "CircleFit measurement geometry; not a fit input"};

  ToolHandle<ITrackTruthMatchingTool> m_truthMatchingTool{
      this, "TrackTruthMatchingTool", "TrackTruthMatchingTool",
      "Official majority-barcode matcher; not a fit or bending input"};
  ToolHandle<IFiducialParticleTool> m_fiducialTool{
      this, "FiducialParticleTool", "FiducialParticleTool",
      "Official per-station truth momentum; calibration reference only"};

  Gaudi::Property<std::string> m_outputJsonl{this, "OutputJsonl", ""};
  Gaudi::Property<std::string> m_eventJsonl{this, "EventJsonl", ""};
  Gaudi::Property<std::string> m_sourceId{this, "SourceId", ""};
  Gaudi::Property<std::string> m_campaign{this, "Campaign", "smoke"};
  Gaudi::Property<std::string> m_geometryHash{this, "GeometryHash", ""};
  Gaudi::Property<std::string> m_fieldHash{this, "FieldHash", ""};
  Gaudi::Property<std::string> m_materialHash{this, "MaterialHash", ""};
  Gaudi::Property<std::string> m_conditionsHash{this, "ConditionsHash", ""};
  Gaudi::Property<int> m_officialTruthStation{this, "OfficialTruthStation", 1};
  Gaudi::Property<int> m_skipEvents{this, "SkipEvents", 0,
                                    "Athena SkipEvents; skip_index offset"};

  const FaserSCT_ID* m_idHelper{nullptr};
  const TrackerDD::SCT_DetectorManager* m_detManager{nullptr};
  std::mutex m_mutex;
  std::vector<std::string> m_trackRows;
  std::vector<std::string> m_eventRows;
  int m_eventsSeen{0};
};

#endif
