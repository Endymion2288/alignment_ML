#ifndef ALIGNMENT_ML_CKF_LEAVE_TARGET_OUT_DUMP_ALG_H
#define ALIGNMENT_ML_CKF_LEAVE_TARGET_OUT_DUMP_ALG_H

#include "EventPrimitives/EventPrimitives.h"
#include "GeoPrimitives/GeoPrimitives.h"

#include "AthenaBaseComps/AthAlgorithm.h"
#include "FaserActsGeometryInterfaces/IFaserActsTrackingGeometryTool.h"
#include "Gaudi/Property.h"
#include "GaudiKernel/ToolHandle.h"
#include "MagFieldConditions/FaserFieldCacheCondObj.h"
#include "StoreGate/ReadCondHandleKey.h"
#include "StoreGate/ReadHandleKey.h"
#include "TrkTrack/TrackCollection.h"
#include "xAODEventInfo/EventInfo.h"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

class FaserSCT_ID;

/// Independent leave-target-out refit helper.  Lives only in alignment_ML.
/// Does not call KalmanFitterTool.fit, does not edit Calypso, does not
/// write /Tracker/Align, and does not seed q/p from truth.
class CkfLeaveTargetOutDumpAlg : public AthAlgorithm {
 public:
  CkfLeaveTargetOutDumpAlg(const std::string& name, ISvcLocator* pSvcLocator);
  virtual ~CkfLeaveTargetOutDumpAlg() = default;

  StatusCode initialize() override;
  StatusCode execute() override;
  StatusCode finalize() override;

 private:
  SG::ReadHandleKey<TrackCollection> m_trackKey{
      this, "TrackCollection", "CKFTrackCollection",
      "Persisted CKF tracks; measurements are read and then filtered"};
  SG::ReadHandleKey<xAOD::EventInfo> m_eventKey{
      this, "EventInfoKey", "EventInfo", "Event identity"};
  SG::ReadCondHandleKey<FaserFieldCacheCondObj> m_fieldCondObjInputKey{
      this, "FaserFieldCacheCondObj", "fieldCondObj",
      "Magnetic field conditions object"};
  ToolHandle<IFaserActsTrackingGeometryTool> m_trackingGeometryTool{
      this, "TrackingGeometryTool", "FaserActsTrackingGeometryTool",
      "Geometry context and identifier map"};

  Gaudi::Property<std::string> m_outputJsonl{this, "OutputJsonl", "",
                                             "Destination jsonl path"};
  Gaudi::Property<std::string> m_sourceId{this, "SourceId", "", "WB87 source id"};
  Gaudi::Property<std::string> m_geometryHash{this, "GeometryHash", "", ""};
  Gaudi::Property<std::string> m_fieldHash{this, "FieldHash", "", ""};
  Gaudi::Property<std::string> m_materialHash{this, "MaterialHash", "", ""};
  Gaudi::Property<std::string> m_conditionsHash{this, "ConditionsHash", "", ""};
  Gaudi::Property<std::vector<double>> m_stationZmm{
      this, "StationZmm", {-1860.15, 47.4, 1237.4, 2427.4},
      "Fixed WB87 station z [mm]"};
  Gaudi::Property<std::vector<int>> m_targetStations{
      this, "TargetStations", {1, 2, 3},
      "Exclude each of these stations in turn"};
  Gaudi::Property<int> m_minRemainingMeasurements{
      this, "MinRemainingMeasurements", 8,
      "Fail the LTO fit if fewer remaining measurements remain"};
  Gaudi::Property<double> m_seedCovarianceScale{
      this, "SeedCovarianceScale", 1.0,
      "Pre-registered uninformative-seed scale for B14R/B14S sensitivity only"};
  Gaudi::Property<std::string> m_seedCovarianceDirection{
      this, "SeedCovarianceDirection", "all",
      "Which seed-covariance direction to scale: all, x, y, tx, ty, q_over_p"};
  Gaudi::Property<bool> m_enableDirectionalSeedCampaign{
      this, "EnableDirectionalSeedCampaign", false,
      "B14S only: refit each direction at 0.1 / 1 / 10 in one event loop"};
  Gaudi::Property<bool> m_enableProfileLikelihood{
      this, "EnableProfileLikelihood", false,
      "B14M: evaluate measurement-only chi2 / profile; default off"};
  Gaudi::Property<bool> m_profileOnly{
      this, "ProfileOnly", false,
      "B14M: skip Kalman export and write profile rows only"};
  Gaudi::Property<bool> m_enableProfileSeedInvariance{
      this, "EnableProfileSeedInvariance", false,
      "B14M smoke: restart profile from pre-registered init variants"};
  Gaudi::Property<int> m_profileMaxIterations{
      this, "ProfileMaxIterations", 20, "B14M Gauss-Newton iteration cap"};
  Gaudi::Property<double> m_profilePinvRelative{
      this, "ProfilePinvRelative", 1.0e-8,
      "B14M pre-registered pseudoinverse relative tolerance; not 0.01"};
  Gaudi::Property<std::vector<int>> m_selectEventIds{
      this, "SelectEventIds", {},
      "If non-empty, only process these event numbers"};
  Gaudi::Property<bool> m_enableProfileNumerics{
      this, "EnableProfileNumerics", false,
      "B14N: sequential transport, scaling, backtracking, per-hit audit"};
  Gaudi::Property<bool> m_profileEvaluateOnly{
      this, "ProfileEvaluateOnly", false,
      "B14N: write per-hit chi2 evaluation without optimizing"};
  Gaudi::Property<bool> m_profileSequentialTransport{
      this, "ProfileSequentialTransport", true,
      "B14N: hop source to surfaces in physical z order"};
  Gaudi::Property<bool> m_profileMatchKalmanStepSize{
      this, "ProfileMatchKalmanStepSize", true,
      "B14N: do not impose the WB114 10 m maxStepSize"};
  Gaudi::Property<int> m_profileMaxSteps{
      this, "ProfileMaxSteps", 4000,
      "B14N per-hop step cap; not a statistical gate"};
  Gaudi::Property<bool> m_profileUseNumericalScaling{
      this, "ProfileUseNumericalScaling", true,
      "B14N: fixed unit scales only, not a prior"};
  Gaudi::Property<bool> m_profileWriteTrace{
      this, "ProfileWriteTrace", true,
      "B14N: record optimizer iterations"};
  Gaudi::Property<bool> m_enableProfileTransport{
      this, "EnableProfileTransport", false,
      "B14T: supporting-plane measurement transport; default off"};
  Gaudi::Property<bool> m_profileSupportingPlane{
      this, "ProfileSupportingPlane", true,
      "B14T: SurfaceReached uses infinite supporting plane, not active bounds"};
  Gaudi::Property<bool> m_profileRecordStepperPath{
      this, "ProfileRecordStepperPath", false,
      "B14T: record downsampled stepper checkpoints on long hops"};
  Gaudi::Property<int> m_profileDiagnosticMaxSteps{
      this, "ProfileDiagnosticMaxSteps", 0,
      "B14T diagnostic maxSteps overlay; 0 means unused"};

  const FaserSCT_ID* m_idHelper{nullptr};
  std::mutex m_mutex;
  std::vector<std::string> m_rows;
};

#endif
