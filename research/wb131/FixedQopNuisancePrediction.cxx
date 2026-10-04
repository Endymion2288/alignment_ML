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

namespace WB131 {
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

class FixedQopNuisancePrediction final: public AthAlgorithm {
 public:
  FixedQopNuisancePrediction(const std::string& n,ISvcLocator* s):AthAlgorithm(n,s){}
  StatusCode initialize() override {
    ATH_CHECK(m_tool.retrieve());
    ATH_CHECK(detStore()->retrieve(m_id,"FaserSCT_ID"));
    try {std::ifstream f(m_fixture.value()),e(m_export.value());std::ifstream r(m_request.value());f>>m_f;e>>m_e;r>>m_r;if(!f||!e||!r)throw std::runtime_error("input read");}
    catch(const std::exception& x){ATH_MSG_ERROR(x.what());return StatusCode::FAILURE;}
    return StatusCode::SUCCESS;
  }
  StatusCode execute() override {
    try {audit();}
    catch(const std::exception& x){ATH_MSG_ERROR("WB131 fail closed: "<<x.what());return StatusCode::FAILURE;}
    return StatusCode::SUCCESS;
  }
 private:
  ToolHandle<IFaserActsExtrapolationTool> m_tool{this,"ExtrapolationTool",""};
  Gaudi::Property<std::string> m_fixture{this,"FixturePath",""},m_export{this,"ExportPath",""},m_output{this,"OutputPath",""},m_request{this,"RequestPath",""};
  const FaserSCT_ID* m_id=nullptr;Json m_f,m_e,m_r;

  Json state(const Acts::BoundTrackParameters& p,const Acts::Transform3& frame,const Acts::GeometryContext& g) const {
    const Acts::Vector3 x=frame.inverse()*p.position(g),d=frame.linear().transpose()*p.direction();
    if(!x.allFinite()||!d.allFinite()||std::abs(x.z())>1e-5||std::abs(d.z())<1e-12) throw std::runtime_error("invalid target state");
    Eigen::Matrix<double,4,1> h;h<<x.x(),x.y(),d.x()/d.z(),d.y()/d.z();
    return Json{{"h",pack(h)}, {"local_position_mm",pack(x)}, {"global_position_mm",pack(p.position(g))},
      {"global_direction",pack(p.direction())}, {"qop_per_MeV",p.parameters()[Acts::eBoundQOverP]*Acts::UnitConstants::MeV},
      {"covariance_present",p.covariance().has_value()}, {"on_surface",true}};
  }
  Acts::BoundTrackParameters start(const V5& seed,double z,const Acts::GeometryContext& g) const {
    auto s=Acts::Surface::makeShared<Acts::PlaneSurface>(Acts::Vector3(0,0,z),Acts::Vector3(0,0,1));
    const Acts::Vector3 d=Acts::Vector3(seed[2],seed[3],1).normalized();
    auto b=Acts::detail::transformFreeToBoundParameters(Acts::Vector3(seed[0],seed[1],z),0.,d,
      seed[4]/Acts::UnitConstants::MeV,*s,g);
    if(!b.ok()) throw std::runtime_error("free-to-bound failed");
    return Acts::BoundTrackParameters(s,b.value(),std::nullopt,Acts::ParticleHypothesis::muon());
  }
  void audit() {
    const auto& ctx=Gaudi::Hive::currentContext(); const xAOD::EventInfo* h=nullptr;
    if(evtStore()->retrieve(h,"EventInfo").isFailure()||!h||h->runNumber()!=m_f.at("actual_run").get<unsigned>()||h->eventNumber()!=m_f.at("actual_event").get<uint64_t>())
      throw std::runtime_error("event identity");
    if(m_e.at("schema")!="wb127_strip_measurement_export_v1"||m_e.at("identity").at("actual_run")!=m_f.at("actual_run")||m_e.at("identity").at("actual_event")!=m_f.at("actual_event"))
      throw std::runtime_error("export identity");
    const auto* geometry=m_tool->trackingGeometryTool(); const auto g=geometry->getGeometryContext(ctx).context();
    const auto tracking=geometry->trackingGeometry(); const auto ids=geometry->getIdentifierMap();
    if(!tracking||!ids) throw std::runtime_error("tracking geometry map missing");
    const V5 nominal=vector5(m_e.at("p_seed"));
    const double z0=m_e.at("references").at(0).at("z_center_mm").get<double>();
    Json arms=Json::array(); size_t calls=0;
    for(const auto& arm:m_r.at("arms")) {
      const V5 seed=vector5(arm.at("seed"));
      if(seed[4]!=nominal[4]) throw std::runtime_error("qop changed");
      auto source=start(seed,z0,g); Json rows=Json::array(); std::set<unsigned long long> seen;
      for(const auto& input:m_e.at("rows")) {
        const auto id=input.at("cluster_id").get<unsigned long long>();
        if(!seen.insert(id).second) throw std::runtime_error("duplicate cluster");
        Identifier wafer(static_cast<Identifier::value_type>(input.at("wafer_id").get<unsigned long long>()));
        if(ids->count(wafer)!=1) throw std::runtime_error("sensor geometry id missing");
        const auto* target=tracking->findSurface(ids->at(wafer));
        if(!target) throw std::runtime_error("sensor surface missing");
        const Acts::Transform3 frame=target->transform(g),expected=transform(input.at("sensor_transform"));
        if((frame.matrix()-expected.matrix()).cwiseAbs().maxCoeff()>1e-9) throw std::runtime_error("sensor transform mismatch");
        const double distance=(frame.translation()-source.position(g)).dot(source.direction());
        ++calls;
        ATH_MSG_INFO("WB131_CALL_BEGIN id="<<calls<<" tag="<<arm.at("tag")<<" cluster="<<id);
        Json row{{"cluster_id",id},{"station",input.at("station")},{"wafer_id",input.at("wafer_id")},{"call_id",calls}};
        auto result=m_tool->propagate(ctx,source,*target,distance>=0?Acts::Direction::Forward:Acts::Direction::Backward);
        if(!result) {row["status"]="FAIL_OFFICIAL_NULL";row["failure_reason"]="UNKNOWN_OPTIONAL_ONLY";}
        else {
          row["status"]="SUCCESS"; row["state"]=state(*result,frame,g);
          const Acts::Vector3 local=frame.inverse()*result->position(g);
          row["inside_bounds"]=target->insideBounds(local.head<2>(),Acts::BoundaryCheck(true));
          row["is_on_surface_with_bounds"]=target->isOnSurface(g,result->position(g),result->direction(),Acts::BoundaryCheck(true));
          row["state"]["on_surface"]=row["is_on_surface_with_bounds"];
          row["frame_roundtrip_mm"]=(frame*local-result->position(g)).norm();
          row["predicted_loc0_mm"]=local.x();
          row["local_residual_mm"]=input.at("local_position").at(0).get<double>()-local.x();
          row["bounds_type"]=static_cast<int>(target->bounds().type());
          row["bounds_values"]=target->bounds().values();
        }
        ATH_MSG_INFO("WB131_CALL_END id="<<calls<<" success="<<(result?"true":"false"));
        rows.push_back(row);
      }
      arms.push_back(Json{{"tag",arm.at("tag")},{"seed",arm.at("seed")},{"rows",rows}});
    }
    if(calls!=m_r.at("arms").size()*m_e.at("rows").size()) throw std::runtime_error("call count");
    Json out{{"schema","wb131_fixed_qop_response_v1"},{"identity",m_e.at("identity")},{"z0_mm",z0},
      {"arms",arms},{"official_calls",calls},{"new_reconstruction_calls",0},{"new_propagation_calls",calls},
      {"material_source","None"},{"field_mode","FASER"},{"covariance_transport",false}};
    int fd=::open(m_output.value().c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);if(fd<0)throw std::runtime_error("exclusive output");::close(fd);
    std::ofstream o(m_output.value());o<<out.dump(2)<<std::endl;if(!o)throw std::runtime_error("output write");
  }
};
}
DECLARE_COMPONENT(WB131::FixedQopNuisancePrediction)
