#include "AthenaBaseComps/AthAlgorithm.h"
#include "GaudiKernel/ToolHandle.h"
#include "GaudiKernel/ThreadLocalContext.h"
#include "FaserActsGeometryInterfaces/IFaserActsTrackingGeometryTool.h"
#include "TrackerIdentifier/FaserSCT_ID.h"
#include "xAODEventInfo/EventInfo.h"
#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "Acts/Surfaces/BoundaryCheck.hpp"
#include <nlohmann/json.hpp>
#include <Eigen/Core>
#include <fstream>
#include <set>
#include <fcntl.h>
#include <unistd.h>

namespace WB130 {
using Json=nlohmann::json;
template<class D> Json pack(const Eigen::MatrixBase<D>& m){if(!m.allFinite())throw std::runtime_error("nonfinite");Json j=Json::array();for(int i=0;i<m.rows();++i)j.push_back(m[i]);return j;}
Acts::Vector3 vec3(const Json& j){Acts::Vector3 v;for(int i=0;i<3;++i)v[i]=j.at(i).is_array()?j.at(i).at(0).get<double>():j.at(i).get<double>();if(!v.allFinite())throw std::runtime_error("nonfinite vec3");return v;}
Acts::Transform3 transform(const Json& j){Acts::Transform3 t=Acts::Transform3::Identity();for(int i=0;i<4;++i)for(int k=0;k<4;++k)t.matrix()(i,k)=j.at(i).at(k).get<double>();return t;}
class SensorBoundsAudit final:public AthAlgorithm{
 public:SensorBoundsAudit(const std::string& n,ISvcLocator* s):AthAlgorithm(n,s){}
 StatusCode initialize()override{ATH_CHECK(m_geometry.retrieve());ATH_CHECK(detStore()->retrieve(m_id,"FaserSCT_ID"));try{std::ifstream f(m_fixture.value()),e(m_export.value()),c(m_curve.value());f>>m_f;e>>m_e;c>>m_c;if(!f||!e||!c)throw std::runtime_error("input read");}catch(const std::exception& x){ATH_MSG_ERROR(x.what());return StatusCode::FAILURE;}return StatusCode::SUCCESS;}
 StatusCode execute()override{try{audit();}catch(const std::exception& x){ATH_MSG_ERROR("WB130 fail closed: "<<x.what());return StatusCode::FAILURE;}return StatusCode::SUCCESS;}
 private:
 ToolHandle<IFaserActsTrackingGeometryTool> m_geometry{this,"TrackingGeometryTool",""};Gaudi::Property<std::string> m_fixture{this,"FixturePath",""},m_export{this,"ExportPath",""},m_curve{this,"CurvePath",""},m_output{this,"OutputPath",""};const FaserSCT_ID* m_id=nullptr;Json m_f,m_e,m_c;
 void audit(){const auto& ctx=Gaudi::Hive::currentContext();const xAOD::EventInfo* h=nullptr;if(evtStore()->retrieve(h,"EventInfo").isFailure()||!h||h->runNumber()!=m_f.at("actual_run").get<unsigned>()||h->eventNumber()!=m_f.at("actual_event").get<uint64_t>())throw std::runtime_error("event identity");if(m_e.at("identity")!=m_c.at("identity"))throw std::runtime_error("export/curve identity");const auto g=m_geometry->getGeometryContext(ctx).context();const auto tracking=m_geometry->trackingGeometry();const auto ids=m_geometry->getIdentifierMap();if(!tracking||!ids)throw std::runtime_error("geometry map");
  const auto& erows=m_e.at("rows");const auto& crows=m_c.at("rows");if(erows.size()!=crows.size())throw std::runtime_error("row count");Json rows=Json::array();std::set<unsigned long long> seen;
  for(size_t i=0;i<erows.size();++i){const auto& e=erows.at(i);const auto& c=crows.at(i);const auto id=e.at("cluster_id").get<unsigned long long>();if(id!=c.at("cluster_id").get<unsigned long long>()||!seen.insert(id).second)throw std::runtime_error("cluster order");Identifier wafer(static_cast<Identifier::value_type>(e.at("wafer_id").get<unsigned long long>()));if(ids->count(wafer)!=1)throw std::runtime_error("wafer missing");const auto* surface=tracking->findSurface(ids->at(wafer));if(!surface)throw std::runtime_error("surface missing");const auto frame=surface->transform(g);const auto expected=transform(e.at("sensor_transform"));if((frame.matrix()-expected.matrix()).cwiseAbs().maxCoeff()>1e-9)throw std::runtime_error("frame mismatch");const Acts::Vector3 pos=vec3(c.at("state").at("global_position_mm")),dir=vec3(c.at("state").at("global_direction"));const Acts::Vector3 local=frame.inverse()*pos;const bool inside=surface->insideBounds(local.head<2>(),Acts::BoundaryCheck(true));const bool on=surface->isOnSurface(g,pos,dir,Acts::BoundaryCheck(true));Json row{{"cluster_id",id},{"station",e.at("station")},{"wafer_id",e.at("wafer_id")},{"surface_type",static_cast<int>(surface->type())},{"bounds_type",static_cast<int>(surface->bounds().type())},{"bounds_values",surface->bounds().values()},{"endpoint_local_mm",pack(local)},{"endpoint_local_y_mm",local.y()},{"measured_local_position",e.at("local_position")},{"inside_bounds",inside},{"is_on_surface_with_bounds",on},{"frame_roundtrip_mm",(frame*local-pos).norm()},{"curve_residual_mm",c.at("local_residual_mm")}};rows.push_back(row);}
  Json out{{"schema","wb130_sensor_bounds_audit_v1"},{"identity",m_e.at("identity")},{"rows",rows},{"new_propagation_calls",0},{"new_reconstruction_calls",0},{"field_queries",0}};int fd=::open(m_output.value().c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);if(fd<0)throw std::runtime_error("exclusive output");::close(fd);std::ofstream o(m_output.value());o<<out.dump(2)<<std::endl;if(!o)throw std::runtime_error("write");}
};
}
DECLARE_COMPONENT(WB130::SensorBoundsAudit)
