#include "AthenaBaseComps/AthAlgorithm.h"
#include "GaudiKernel/ToolHandle.h"
#include "GaudiKernel/ThreadLocalContext.h"
#include "FaserActsGeometryInterfaces/IFaserActsExtrapolationTool.h"
#include "FaserActsGeometryInterfaces/IFaserActsTrackingGeometryTool.h"
#include "TrackerIdentifier/FaserSCT_ID.h"
#include "xAODEventInfo/EventInfo.h"
#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "Acts/Surfaces/PlaneSurface.hpp"
#include "Acts/Surfaces/BoundaryCheck.hpp"
#include "Acts/EventData/detail/TransformationFreeToBound.hpp"
#include "Acts/Definitions/Units.hpp"
#include <nlohmann/json.hpp>
#include <Eigen/Core>
#include <fstream>
#include <cmath>
#include <set>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

namespace WB133 {
using Json=nlohmann::json;
using V5=Eigen::Matrix<double,5,1>;

template<class D> Json pack(const Eigen::MatrixBase<D>& m) {
  if(!m.allFinite()) throw std::runtime_error("nonfinite matrix");
  Json j=Json::array();
  for(int i=0;i<m.rows();++i){Json r=Json::array();for(int k=0;k<m.cols();++k)r.push_back(m(i,k));j.push_back(r);}
  return j;
}
V5 vector5(const Json& j) {
  V5 v;
  for(int i=0;i<5;++i) v[i]=j.at(i).is_array()?j.at(i).at(0).get<double>():j.at(i).get<double>();
  if(!v.allFinite()) throw std::runtime_error("nonfinite seed");
  return v;
}
Acts::Transform3 transform(const Json& j) {
  Acts::Transform3 t=Acts::Transform3::Identity();
  for(int i=0;i<4;++i) for(int k=0;k<4;++k) t.matrix()(i,k)=j.at(i).at(k).get<double>();
  if(!t.matrix().allFinite() ||
     (t.linear().transpose()*t.linear()-Acts::RotationMatrix3::Identity()).cwiseAbs().maxCoeff()>1e-9 ||
     std::abs(t.linear().determinant()-1)>1e-9 ||
     (t.matrix().row(3)-Eigen::RowVector4d(0,0,0,1)).cwiseAbs().maxCoeff()!=0.)
    throw std::runtime_error("invalid sensor transform");
  return t;
}

Acts::Vector3 vector3(const Json& j) {
  Acts::Vector3 v;
  for(int i=0;i<3;++i)v[i]=j.at(i).get<double>();
  if(!v.allFinite())throw std::runtime_error("nonfinite vector");
  return v;
}
class AcceptedAnchorPrediction final: public AthAlgorithm {
 public:
  AcceptedAnchorPrediction(const std::string& n,ISvcLocator* s):AthAlgorithm(n,s){}
  StatusCode initialize() override {
    ATH_CHECK(m_tool.retrieve());
    try{std::ifstream f(m_fixture.value()),e(m_export.value()),r(m_request.value());f>>m_f;e>>m_e;r>>m_r;
      if(!f||!e||!r)throw std::runtime_error("input read");}
    catch(const std::exception& x){ATH_MSG_ERROR(x.what());return StatusCode::FAILURE;}
    return StatusCode::SUCCESS;
  }
  StatusCode execute() override {
    try{audit();}catch(const std::exception& x){ATH_MSG_ERROR("WB133 fail closed: "<<x.what());return StatusCode::FAILURE;}
    return StatusCode::SUCCESS;
  }
 private:
  ToolHandle<IFaserActsExtrapolationTool> m_tool{this,"ExtrapolationTool",""};
  Gaudi::Property<std::string> m_fixture{this,"FixturePath",""},m_export{this,"ExportPath",""},
    m_request{this,"RequestPath",""},m_output{this,"OutputPath",""};
  Json m_f,m_e,m_r;
  void audit(){
    const auto& ctx=Gaudi::Hive::currentContext();const xAOD::EventInfo* header=nullptr;
    if(evtStore()->retrieve(header,"EventInfo").isFailure()||!header||
      header->runNumber()!=m_f.at("actual_run").get<unsigned>()||header->eventNumber()!=m_f.at("actual_event").get<uint64_t>())
      throw std::runtime_error("event identity");
    if(m_e.at("schema")!="wb127_strip_measurement_export_v1"||m_r.at("schema")!="wb133_accepted_anchor_request_v1"||
      m_r.at("identity")!=m_e.at("identity")||m_e.at("identity").at("actual_run")!=m_f.at("actual_run")||
      m_e.at("identity").at("actual_event")!=m_f.at("actual_event"))throw std::runtime_error("request/export identity");
    const auto* geometry=m_tool->trackingGeometryTool();const auto g=geometry->getGeometryContext(ctx).context();
    const auto tracking=geometry->trackingGeometry();const auto ids=geometry->getIdentifierMap();
    if(!tracking||!ids)throw std::runtime_error("tracking geometry map missing");
    const auto& a=m_r.at("source");const auto& flags=a.at("type_flags");
    if(!flags.at(0).get<bool>()||flags.at(5).get<bool>()||flags.at(6).get<bool>())throw std::runtime_error("source not accepted Measurement");
    const Acts::Vector3 origin=vector3(a.at("position_mm")),direction=vector3(a.at("direction"));
    const double qop=a.at("qop_per_MeV").get<double>();
    if(std::abs(direction.norm()-1)>1e-12||!std::isfinite(qop)||qop==0.)throw std::runtime_error("source direction/qop");
    auto plane=Acts::Surface::makeShared<Acts::PlaneSurface>(origin,direction);
    auto bound=Acts::detail::transformFreeToBoundParameters(origin,0.,direction,qop/Acts::UnitConstants::MeV,*plane,g);
    if(!bound.ok())throw std::runtime_error("source free-to-bound failed");
    Acts::BoundTrackParameters source(plane,bound.value(),std::nullopt,Acts::ParticleHypothesis::muon());
    const auto position=source.position(g),actualDirection=source.direction();
    if((position-origin).norm()>1e-9||(actualDirection-direction).norm()>1e-12)throw std::runtime_error("source representation changed");
    Json rows=Json::array();std::set<uint64_t> seen;size_t calls=0,controls=0;
    for(const auto& input:m_e.at("rows")){
      const auto id=input.at("cluster_id").get<uint64_t>();if(!seen.insert(id).second)throw std::runtime_error("duplicate cluster");
      Identifier wafer(static_cast<Identifier::value_type>(input.at("wafer_id").get<uint64_t>()));
      if(ids->count(wafer)!=1)throw std::runtime_error("sensor geometry id missing");
      const auto* target=tracking->findSurface(ids->at(wafer));if(!target)throw std::runtime_error("sensor surface missing");
      const auto frame=target->transform(g),expected=transform(input.at("sensor_transform"));
      if((frame.matrix()-expected.matrix()).cwiseAbs().maxCoeff()>1e-9)throw std::runtime_error("sensor transform mismatch");
      const double denominator=frame.linear().col(2).dot(direction);
      if(std::abs(denominator)<1e-12)throw std::runtime_error("near-parallel target");
      const double path=(frame.translation()-origin).dot(frame.linear().col(2))/denominator;
      Json row{{"cluster_id",id},{"wafer_id",input.at("wafer_id")},{"station",input.at("station")},
        {"tangent_path_mm",path},{"call_id",nullptr}};
      Acts::Vector3 endpoint,endpointDirection;double endQop=qop;bool covariance=false;
      const bool anchor=id==a.at("cluster_id").get<uint64_t>();
      if(anchor){
        if(input.at("wafer_id")!=a.at("wafer_id"))throw std::runtime_error("source wafer");
        ++controls;row["kind"]="SOURCE_IDENTITY_CONTROL";row["propagation_direction"]="ZERO_PATH";
        endpoint=position;endpointDirection=actualDirection;
      }else{
        ++calls;row["kind"]="PROPAGATED";row["call_id"]=calls;row["propagation_direction"]=path>=0?"FORWARD":"BACKWARD";
        ATH_MSG_INFO("WB133_CALL_BEGIN id="<<calls<<" cluster="<<id);
        auto result=m_tool->propagate(ctx,source,*target,path>=0?Acts::Direction::Forward:Acts::Direction::Backward);
        ATH_MSG_INFO("WB133_CALL_END id="<<calls<<" success="<<(result?"true":"false"));
        if(!result){row["status"]="FAIL_OFFICIAL_NULL";row["failure_reason"]="UNKNOWN_OPTIONAL_ONLY";rows.push_back(row);continue;}
        endpoint=result->position(g);endpointDirection=result->direction();
        endQop=result->parameters()[Acts::eBoundQOverP]*Acts::UnitConstants::MeV;covariance=result->covariance().has_value();
      }
      const Acts::Vector3 local=frame.inverse()*endpoint;
      if(!endpoint.allFinite()||!endpointDirection.allFinite()||!std::isfinite(endQop)||!local.allFinite())throw std::runtime_error("nonfinite endpoint");
      const bool inside=target->insideBounds(local.head<2>(),Acts::BoundaryCheck(true));
      const bool onSurface=target->isOnSurface(g,endpoint,endpointDirection,Acts::BoundaryCheck(true));
      row["status"]="SUCCESS";row["global_position_mm"]=pack(endpoint);row["global_direction"]=pack(endpointDirection);
      row["local_position_mm"]=pack(local);row["qop_per_MeV"]=endQop;row["covariance_present"]=covariance;
      row["inside_bounds"]=inside;row["strict_is_on_surface_with_bounds"]=onSurface;
      row["plane_tolerance_mm"]=anchor?a.at("persistence_plane_tolerance_mm").get<double>():1e-5;
      row["frame_roundtrip_mm"]=(frame*local-endpoint).norm();row["predicted_loc0_mm"]=local.x();
      row["local_residual_mm"]=input.at("local_position").at(0).get<double>()-local.x();
      row["bounds_type"]=static_cast<int>(target->bounds().type());row["bounds_values"]=target->bounds().values();rows.push_back(row);
    }
    if(controls!=1||calls+controls!=m_e.at("rows").size())throw std::runtime_error("call/control count");
    Json out{{"schema","wb133_accepted_anchor_response_v1"},{"identity",m_e.at("identity")},{"source",a},
      {"source_position_roundtrip_mm",(position-origin).norm()},{"source_direction_roundtrip",(actualDirection-direction).norm()},
      {"source_actual_position_mm",pack(position)},{"source_actual_direction",pack(actualDirection)},
      {"source_actual_qop_per_MeV",source.parameters()[Acts::eBoundQOverP]*Acts::UnitConstants::MeV},
      {"rows",rows},{"official_calls",calls},{"zero_path_controls",controls},{"new_propagation_calls",calls},
      {"new_reconstruction_calls",0},{"material_source","None"},{"field_mode","FASER"},{"covariance_transport",false}};
    int fd=::open(m_output.value().c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);if(fd<0)throw std::runtime_error("exclusive output");::close(fd);
    std::ofstream output(m_output.value());output<<out.dump(2)<<std::endl;if(!output)throw std::runtime_error("output write");
  }
};
}
DECLARE_COMPONENT(WB133::AcceptedAnchorPrediction)
