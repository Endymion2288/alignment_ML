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

namespace WB129 {
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

class SensorSurfaceCurvePrediction final: public AthAlgorithm {
 public:
  SensorSurfaceCurvePrediction(const std::string& n,ISvcLocator* s):AthAlgorithm(n,s){}
  StatusCode initialize() override {
    ATH_CHECK(m_tool.retrieve());
    ATH_CHECK(detStore()->retrieve(m_id,"FaserSCT_ID"));
    try {std::ifstream f(m_fixture.value()),e(m_export.value());f>>m_f;e>>m_e;if(!f||!e)throw std::runtime_error("input read");}
    catch(const std::exception& x){ATH_MSG_ERROR(x.what());return StatusCode::FAILURE;}
    return StatusCode::SUCCESS;
  }
  StatusCode execute() override {
    try {audit();}
    catch(const std::exception& x){ATH_MSG_ERROR("WB129 fail closed: "<<x.what());return StatusCode::FAILURE;}
    return StatusCode::SUCCESS;
  }
 private:
  ToolHandle<IFaserActsExtrapolationTool> m_tool{this,"ExtrapolationTool",""};
  Gaudi::Property<std::string> m_fixture{this,"FixturePath",""},m_export{this,"ExportPath",""},m_output{this,"OutputPath",""};
  const FaserSCT_ID* m_id=nullptr;Json m_f,m_e;

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
    const V5 seed=vector5(m_e.at("p_seed"));
    const double z0=m_e.at("references").at(0).at("z_center_mm").get<double>();
    auto source=start(seed,z0,g);
    const Acts::Vector3 sourcePosition=source.position(g); Json rows=Json::array(); std::set<unsigned long long> seen;
    size_t calls=0;
    for(const auto& input:m_e.at("rows")) {
      const auto id=input.at("cluster_id").get<unsigned long long>(); if(!seen.insert(id).second) throw std::runtime_error("duplicate cluster");
      Identifier wafer(static_cast<Identifier::value_type>(input.at("wafer_id").get<unsigned long long>()));
      if(ids->count(wafer)!=1) throw std::runtime_error("sensor geometry id missing");
      const auto* target=tracking->findSurface(ids->at(wafer)); if(!target) throw std::runtime_error("sensor surface missing");
      const Acts::Transform3 frame=target->transform(g); const Acts::Transform3 expected=transform(input.at("sensor_transform"));
      if((frame.matrix()-expected.matrix()).cwiseAbs().maxCoeff()>1e-9) throw std::runtime_error("sensor transform mismatch");
      const double distance=(frame.translation()-sourcePosition).dot(source.direction());
      ++calls; ATH_MSG_INFO("WB129_CALL_BEGIN id="<<calls<<" cluster="<<id<<" station="<<input.at("station"));
      Json row{{"cluster_id",id},{"station",input.at("station")},{"wafer_id",input.at("wafer_id")},{"call_id",calls},
        {"measured_local_position",input.at("local_position")},{"measured_global_position",input.at("global_position")},{"sensor_transform",input.at("sensor_transform")}};
      auto result=m_tool->propagate(ctx,source,*target,distance>=0?Acts::Direction::Forward:Acts::Direction::Backward);
      if(!result) {row["status"]="FAIL_OFFICIAL_NULL";row["failure_reason"]="UNKNOWN_OPTIONAL_ONLY";}
      else {
        row["status"]="SUCCESS"; row["state"]=state(*result,frame,g);
        const auto predicted=row.at("state").at("local_position_mm").at(0).get<double>();
        const auto measured=input.at("local_position").at(0).get<double>();
        row["local_residual_mm"]=measured-predicted;
        if(!std::isfinite(measured-predicted)) throw std::runtime_error("nonfinite residual");
      }
      ATH_MSG_INFO("WB129_CALL_END id="<<calls<<" cluster="<<id<<" success="<<(result?"true":"false")); rows.push_back(row);
    }
    if(seen.size()!=m_e.at("rows").size()||calls!=m_e.at("rows").size()) throw std::runtime_error("row/call count");
    Json out{{"schema","wb129_sensor_surface_curve_response_v1"},{"identity",m_e.at("identity")},{"p_seed",m_e.at("p_seed")},
      {"z0_mm",z0},{"rows",rows},{"official_calls",calls},{"new_reconstruction_calls",0},{"new_propagation_calls",calls},
      {"material_source","None"},{"field_mode","FASER"},{"covariance_transport",false}};
    int fd=::open(m_output.value().c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);if(fd<0)throw std::runtime_error("exclusive output");::close(fd);
    std::ofstream o(m_output.value());o<<out.dump(2)<<std::endl;if(!o)throw std::runtime_error("output write");
  }
};
}
DECLARE_COMPONENT(WB129::SensorSurfaceCurvePrediction)
