#ifndef ALIGNMENT_ML_CKF_ACTS_TRANSPORT_DUMP_ALG_H
#define ALIGNMENT_ML_CKF_ACTS_TRANSPORT_DUMP_ALG_H

#include "EventPrimitives/EventPrimitives.h"
#include "GeoPrimitives/GeoPrimitives.h"

#include "AthenaBaseComps/AthAlgorithm.h"
#include "FaserActsGeometryInterfaces/IFaserActsExtrapolationTool.h"
#include "FaserActsGeometryInterfaces/IFaserActsTrackingGeometryTool.h"
#include "Gaudi/Property.h"
#include "GaudiKernel/ToolHandle.h"
#include "StoreGate/ReadHandleKey.h"
#include "TrkTrack/TrackCollection.h"
#include "xAODEventInfo/EventInfo.h"

#include <mutex>
#include <string>
#include <vector>

/// Independent ACTS transport dump.  Lives only in alignment_ML.
/// Does not edit Calypso/FaserActs source, does not write /Tracker/Align,
/// and does not replace CKF q/p with truth.
class CkfActsTransportDumpAlg : public AthAlgorithm {
 public:
  CkfActsTransportDumpAlg(const std::string& name, ISvcLocator* pSvcLocator);
  virtual ~CkfActsTransportDumpAlg() = default;

  StatusCode initialize() override;
  StatusCode execute() override;
  StatusCode finalize() override;

 private:
  SG::ReadHandleKey<TrackCollection> m_trackKey{
      this, "TrackCollection", "CKFTrackCollection",
      "Persisted CKF tracks; Cin comes from these 5x5 covariances"};
  SG::ReadHandleKey<xAOD::EventInfo> m_eventKey{
      this, "EventInfoKey", "EventInfo", "Event identity for truth join"};

  ToolHandle<IFaserActsExtrapolationTool> m_toolNoNoise{
      this, "ExtrapolationNoNoise", "",
      "FaserActsExtrapolationTool with MS/Eloss off"};
  ToolHandle<IFaserActsExtrapolationTool> m_toolWithNoise{
      this, "ExtrapolationWithNoise", "",
      "FaserActsExtrapolationTool with MS/Eloss on"};
  ToolHandle<IFaserActsTrackingGeometryTool> m_trackingGeometryTool{
      this, "TrackingGeometryTool", "FaserActsTrackingGeometryTool",
      "Geometry context for bound <-> export conversion"};

  Gaudi::Property<std::string> m_outputJsonl{this, "OutputJsonl", "",
                                             "Destination jsonl path"};
  Gaudi::Property<std::string> m_sourceId{this, "SourceId", "",
                                          "WB87 source id"};
  Gaudi::Property<std::string> m_geometryHash{this, "GeometryHash", "", ""};
  Gaudi::Property<std::string> m_fieldHash{this, "FieldHash", "", ""};
  Gaudi::Property<std::string> m_materialHash{this, "MaterialHash", "", ""};
  Gaudi::Property<std::string> m_conditionsHash{this, "ConditionsHash", "", ""};
  Gaudi::Property<std::vector<double>> m_stationZmm{
      this, "StationZmm", {-1860.15, 47.4, 1237.4, 2427.4},
      "Fixed WB87/WB97 station z [mm]"};
  Gaudi::Property<std::vector<int>> m_targetStations{
      this, "TargetStations", {1, 2, 3},
      "Propagate from the CKF front surface to these stations"};

  std::mutex m_mutex;
  std::vector<std::string> m_rows;
};

#endif
