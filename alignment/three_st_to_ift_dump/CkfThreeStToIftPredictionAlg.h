#ifndef ALIGNMENT_ML_CKF_THREE_ST_TO_IFT_PREDICTION_ALG_H
#define ALIGNMENT_ML_CKF_THREE_ST_TO_IFT_PREDICTION_ALG_H

#include "EventPrimitives/EventPrimitives.h"
#include "GeoPrimitives/GeoPrimitives.h"

#include "AthenaBaseComps/AthAlgorithm.h"
#include "FaserActsGeometryInterfaces/IFaserActsExtrapolationTool.h"
#include "FaserActsGeometryInterfaces/IFaserActsTrackingGeometryTool.h"
#include "Gaudi/Property.h"
#include "GaudiKernel/ToolHandle.h"
#include "StoreGate/ReadHandleKey.h"
#include "TrackerPrepRawData/FaserSCT_ClusterContainer.h"
#include "TrkTrack/TrackCollection.h"
#include "xAODEventInfo/EventInfo.h"

#include <mutex>
#include <string>
#include <vector>

class FaserSCT_ID;

/// Independent 3ST→IFT prediction.  Lives only in this repository.
/// Reads official CKFTrackCollectionWithoutIFT.  IFT residuals come from
/// SCT_ClusterContainer.  Does not refit, does not write /Tracker/Align,
/// and does not replace q/p with truth.
class CkfThreeStToIftPredictionAlg : public AthAlgorithm {
 public:
  CkfThreeStToIftPredictionAlg(const std::string& name, ISvcLocator* pSvcLocator);
  virtual ~CkfThreeStToIftPredictionAlg() = default;

  StatusCode initialize() override;
  StatusCode execute() override;
  StatusCode finalize() override;

 private:
  SG::ReadHandleKey<TrackCollection> m_trackKey{
      this, "TrackCollection", "CKFTrackCollectionWithoutIFT",
      "Official 3-station CKF; Cin and the prediction start here"};
  SG::ReadHandleKey<TrackCollection> m_fourStationKey{
      this, "FourStationCollection", "CKFTrackCollection",
      "Contrast only: reconstruction-associated IFT identifiers"};
  SG::ReadHandleKey<Tracker::FaserSCT_ClusterContainer> m_clusterKey{
      this, "ClusterContainer", "SCT_ClusterContainer",
      "Independent IFT measurements; never enter the 3ST prediction"};
  SG::ReadHandleKey<xAOD::EventInfo> m_eventKey{
      this, "EventInfoKey", "EventInfo", "Event identity"};

  ToolHandle<IFaserActsExtrapolationTool> m_toolNoNoise{
      this, "ExtrapolationNoNoise", "",
      "FaserActsExtrapolationTool with MS/Eloss off"};
  ToolHandle<IFaserActsExtrapolationTool> m_toolWithNoise{
      this, "ExtrapolationWithNoise", "",
      "FaserActsExtrapolationTool with MS/Eloss on; official prediction"};
  ToolHandle<IFaserActsTrackingGeometryTool> m_trackingGeometryTool{
      this, "TrackingGeometryTool", "FaserActsTrackingGeometryTool",
      "Geometry context and identifier map"};

  Gaudi::Property<std::string> m_outputJsonl{this, "OutputJsonl", "",
                                             "Track-level jsonl path"};
  Gaudi::Property<std::string> m_eventJsonl{this, "EventJsonl", "",
                                            "Event-level jsonl path"};
  Gaudi::Property<std::string> m_sourceId{this, "SourceId", "",
                                          "WB87 source id"};
  Gaudi::Property<std::string> m_geometryHash{this, "GeometryHash", "", ""};
  Gaudi::Property<std::string> m_fieldHash{this, "FieldHash", "", ""};
  Gaudi::Property<std::string> m_materialHash{this, "MaterialHash", "", ""};
  Gaudi::Property<std::string> m_conditionsHash{this, "ConditionsHash", "", ""};
  Gaudi::Property<std::vector<double>> m_stationZmm{
      this, "StationZmm", {-1860.15, 47.4, 1237.4, 2427.4},
      "Fixed WB87/WB98 station z [mm]; not re-derived here"};
  Gaudi::Property<int> m_targetStation{this, "TargetStation", 0,
                                       "IFT station id"};
  Gaudi::Property<bool> m_allowBackwardToIft{
      this, "AllowBackwardToIft", true,
      "Required: WB98 skipped targetZ<=sourceZ; this chain must not"};

  const FaserSCT_ID* m_idHelper{nullptr};
  std::mutex m_mutex;
  std::vector<std::string> m_trackRows;
  std::vector<std::string> m_eventRows;
};

#endif
