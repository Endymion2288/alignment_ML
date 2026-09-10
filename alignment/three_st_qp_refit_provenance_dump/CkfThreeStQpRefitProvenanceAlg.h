#ifndef ALIGNMENT_ML_CKF_THREE_ST_QP_REFIT_PROVENANCE_ALG_H
#define ALIGNMENT_ML_CKF_THREE_ST_QP_REFIT_PROVENANCE_ALG_H

#include "AthenaBaseComps/AthAlgorithm.h"
#include "FaserActsKalmanFilter/IFiducialParticleTool.h"
#include "FaserActsKalmanFilter/ITrackTruthMatchingTool.h"
#include "Gaudi/Property.h"
#include "GaudiKernel/ToolHandle.h"
#include "StoreGate/ReadHandleKey.h"
#include "TrkTrack/TrackCollection.h"
#include "xAODEventInfo/EventInfo.h"

#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

class FaserSCT_ID;

/// ≤20-event CKF/KF-refit provenance dump.  Reads official
/// CKFTrackCollectionWithoutIFT and inspects persisted TSOS.  Does not
/// change the persisted collection, seed, geometry, or covariance
/// scale.  Truth is a calibration reference only.  Original
/// complementary CKF/KF state is not in the xAOD; this alg does not
/// run a second KalmanFitter.
class CkfThreeStQpRefitProvenanceAlg : public AthAlgorithm {
 public:
  CkfThreeStQpRefitProvenanceAlg(const std::string& name, ISvcLocator* pSvcLocator);
  virtual ~CkfThreeStQpRefitProvenanceAlg() = default;

  StatusCode initialize() override;
  StatusCode execute() override;
  StatusCode finalize() override;

 private:
  SG::ReadHandleKey<TrackCollection> m_trackKey{
      this, "TrackCollection", "CKFTrackCollectionWithoutIFT",
      "Official 3-station CKF"};
  SG::ReadHandleKey<xAOD::EventInfo> m_eventKey{
      this, "EventInfoKey", "EventInfo", "Event identity"};

  ToolHandle<ITrackTruthMatchingTool> m_truthMatchingTool{
      this, "TrackTruthMatchingTool", "TrackTruthMatchingTool"};
  ToolHandle<IFiducialParticleTool> m_fiducialTool{
      this, "FiducialParticleTool", "FiducialParticleTool"};

  Gaudi::Property<std::string> m_outputJsonl{this, "OutputJsonl", ""};
  Gaudi::Property<std::string> m_eventJsonl{this, "EventJsonl", ""};
  Gaudi::Property<std::string> m_sourceId{this, "SourceId", ""};
  Gaudi::Property<std::string> m_campaign{this, "Campaign", "smoke"};
  Gaudi::Property<std::string> m_focusPairs{this, "FocusPairs", "",
                                            "event:track,event:track"};
  Gaudi::Property<int> m_officialTruthStation{this, "OfficialTruthStation", 1};
  Gaudi::Property<int> m_skipEvents{this, "SkipEvents", 0,
                                    "Athena SkipEvents; skip_index offset"};

  const FaserSCT_ID* m_idHelper{nullptr};
  std::mutex m_mutex;
  std::vector<std::string> m_trackRows;
  std::vector<std::string> m_eventRows;
  std::set<std::pair<int, int>> m_focus;
  int m_eventsSeen{0};
};

#endif
