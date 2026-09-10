#ifndef ALIGNMENT_ML_CKF_THREE_ST_QP_CALIBRATION_ALG_H
#define ALIGNMENT_ML_CKF_THREE_ST_QP_CALIBRATION_ALG_H

#include "AthenaBaseComps/AthAlgorithm.h"
#include "FaserActsKalmanFilter/IFiducialParticleTool.h"
#include "FaserActsKalmanFilter/ITrackTruthMatchingTool.h"
#include "Gaudi/Property.h"
#include "GaudiKernel/ToolHandle.h"
#include "StoreGate/ReadHandleKey.h"
#include "TrkTrack/TrackCollection.h"
#include "xAODEventInfo/EventInfo.h"

#include <mutex>
#include <string>
#include <vector>

class FaserSCT_ID;

/// Truth-matched WithoutIFT q/p dump.  Lives only in this repository.
/// Reads official CKFTrackCollectionWithoutIFT.  Truth is a calibration
/// reference only and never replaces the fit.  Does not refit and does
/// not write /Tracker/Align.
class CkfThreeStQpCalibrationAlg : public AthAlgorithm {
 public:
  CkfThreeStQpCalibrationAlg(const std::string& name, ISvcLocator* pSvcLocator);
  virtual ~CkfThreeStQpCalibrationAlg() = default;

  StatusCode initialize() override;
  StatusCode execute() override;
  StatusCode finalize() override;

 private:
  SG::ReadHandleKey<TrackCollection> m_trackKey{
      this, "TrackCollection", "CKFTrackCollectionWithoutIFT",
      "Official 3-station CKF; q/p_fit comes only from here"};
  SG::ReadHandleKey<xAOD::EventInfo> m_eventKey{
      this, "EventInfoKey", "EventInfo", "Event identity"};

  ToolHandle<ITrackTruthMatchingTool> m_truthMatchingTool{
      this, "TrackTruthMatchingTool", "TrackTruthMatchingTool",
      "Official majority-barcode matcher; not a fit input"};
  ToolHandle<IFiducialParticleTool> m_fiducialTool{
      this, "FiducialParticleTool", "FiducialParticleTool",
      "Official per-station truth momentum; calibration reference only"};

  Gaudi::Property<std::string> m_outputJsonl{this, "OutputJsonl", "",
                                             "Track-level jsonl path"};
  Gaudi::Property<std::string> m_eventJsonl{this, "EventJsonl", "",
                                            "Event-level jsonl path"};
  Gaudi::Property<std::string> m_sourceId{this, "SourceId", "",
                                          "WB87 source id"};
  Gaudi::Property<std::string> m_campaign{this, "Campaign", "smoke",
                                          "smoke or batch; dumps must not overwrite"};
  Gaudi::Property<std::string> m_geometryHash{this, "GeometryHash", "", ""};
  Gaudi::Property<std::string> m_fieldHash{this, "FieldHash", "", ""};
  Gaudi::Property<std::string> m_materialHash{this, "MaterialHash", "", ""};
  Gaudi::Property<std::string> m_conditionsHash{this, "ConditionsHash", "", ""};
  Gaudi::Property<int> m_officialTruthStation{
      this, "OfficialTruthStation", 1,
      "S1: WithoutIFT front() station; do not change"};

  const FaserSCT_ID* m_idHelper{nullptr};
  std::mutex m_mutex;
  std::vector<std::string> m_trackRows;
  std::vector<std::string> m_eventRows;
};

#endif
