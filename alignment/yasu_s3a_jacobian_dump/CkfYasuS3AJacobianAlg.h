#ifndef ALIGNMENT_ML_CKF_YASU_S3A_JACOBIAN_ALG_H
#define ALIGNMENT_ML_CKF_YASU_S3A_JACOBIAN_ALG_H

#include "EventPrimitives/EventPrimitives.h"
#include "GeoPrimitives/GeoPrimitives.h"

#include "AthenaBaseComps/AthAlgorithm.h"
#include "FaserActsGeometryInterfaces/IFaserActsExtrapolationTool.h"
#include "FaserActsGeometryInterfaces/IFaserActsTrackingGeometryTool.h"
#include "FaserActsKalmanFilter/ITrackTruthMatchingTool.h"
#include "Gaudi/Property.h"
#include "GaudiKernel/ToolHandle.h"
#include "StoreGate/ReadHandleKey.h"
#include "TrackerPrepRawData/FaserSCT_ClusterContainer.h"
#include "TrackerSimData/TrackerSimDataCollection.h"
#include "TrkTrack/TrackCollection.h"
#include "xAODEventInfo/EventInfo.h"

#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

class FaserSCT_ID;
namespace TrackerDD {
class SCT_DetectorManager;
}

/// Yasu-S3A mean-response Jacobian dump.  Frozen 20-track contract.
/// Truth-SDO IFT association.  Runtime sensor transforms.  Does not
/// write /Tracker/Align, does not use qp_bending_proxy, and does not
/// weight by native 5x5.
class CkfYasuS3AJacobianAlg : public AthAlgorithm {
 public:
  CkfYasuS3AJacobianAlg(const std::string& name, ISvcLocator* pSvcLocator);
  virtual ~CkfYasuS3AJacobianAlg() = default;

  StatusCode initialize() override;
  StatusCode execute() override;
  StatusCode finalize() override;

 private:
  SG::ReadHandleKey<TrackCollection> m_trackKey{
      this, "TrackCollection", "CKFTrackCollectionWithoutIFT",
      "Official 3-station CKF; prediction start"};
  SG::ReadHandleKey<Tracker::FaserSCT_ClusterContainer> m_clusterKey{
      this, "ClusterContainer", "SCT_ClusterContainer",
      "Independent IFT measurements"};
  SG::ReadHandleKey<TrackerSimDataCollection> m_sdoKey{
      this, "SDOMap", "SCT_SDO_Map", "Truth association source"};
  SG::ReadHandleKey<xAOD::EventInfo> m_eventKey{
      this, "EventInfoKey", "EventInfo", "Event identity"};

  ToolHandle<IFaserActsExtrapolationTool> m_toolNoNoise{
      this, "ExtrapolationNoNoise", ""};
  ToolHandle<IFaserActsExtrapolationTool> m_toolWithNoise{
      this, "ExtrapolationWithNoise", ""};
  ToolHandle<IFaserActsExtrapolationTool> m_toolElossOffMsOn{
      this, "ExtrapolationElossOff", "",
      "MS on, Eloss off; energy-loss control"};
  ToolHandle<IFaserActsTrackingGeometryTool> m_trackingGeometryTool{
      this, "TrackingGeometryTool", "FaserActsTrackingGeometryTool"};
  ToolHandle<ITrackTruthMatchingTool> m_truthMatchingTool{
      this, "TrackTruthMatchingTool", "TrackTruthMatchingTool"};

  Gaudi::Property<std::string> m_outputJsonl{this, "OutputJsonl", ""};
  Gaudi::Property<std::string> m_eventJsonl{this, "EventJsonl", ""};
  Gaudi::Property<std::string> m_sourceId{this, "SourceId", ""};
  Gaudi::Property<std::string> m_fileSha256{this, "FileSha256", ""};
  Gaudi::Property<std::string> m_focusPairs{this, "FocusPairs", ""};
  Gaudi::Property<std::string> m_geometryHash{this, "GeometryHash", ""};
  Gaudi::Property<std::string> m_fieldHash{this, "FieldHash", ""};
  Gaudi::Property<std::string> m_materialHash{this, "MaterialHash", ""};
  Gaudi::Property<std::string> m_conditionsHash{this, "ConditionsHash", ""};
  Gaudi::Property<std::string> m_iovProvenance{this, "IovProvenance", ""};
  Gaudi::Property<int> m_skipEvents{this, "SkipEvents", 0};
  Gaudi::Property<bool> m_runProfiled{this, "RunProfiled", true};
  Gaudi::Property<int> m_profileIterations{this, "ProfileIterations", 5};
  Gaudi::Property<double> m_deltaQp{this, "DeltaQOverPPerMeV", 1.0e-6};
  Gaudi::Property<double> m_deltaRy{this, "DeltaRyRad", 1.0e-3};
  Gaudi::Property<double> m_deltaDx{this, "DeltaDxMm", 0.10};
  Gaudi::Property<double> m_deltaTy{this, "DeltaTy", 1.0e-4};

  const FaserSCT_ID* m_idHelper{nullptr};
  const TrackerDD::SCT_DetectorManager* m_detManager{nullptr};
  std::mutex m_mutex;
  std::vector<std::string> m_trackRows;
  std::vector<std::string> m_eventRows;
  std::set<std::pair<int, int>> m_focus;
  int m_eventsSeen{0};
};

#endif
