#include "AthenaBaseComps/AthAlgorithm.h"
#include "GaudiKernel/ToolHandle.h"
#include "GaudiKernel/ThreadLocalContext.h"
#include "FaserActsGeometryInterfaces/IFaserActsExtrapolationTool.h"
#include "FaserActsGeometryInterfaces/IFaserActsTrackingGeometryTool.h"
#include "FaserActsGeometry/FASERMagneticFieldWrapper.h"
#include "TrackerIdentifier/FaserSCT_ID.h"
#include "Identifier/Identifier.h"
#include "xAODEventInfo/EventInfo.h"
#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/Geometry/GeometryIdentifier.hpp"
#include "Acts/Surfaces/PlaneSurface.hpp"
#include "Acts/EventData/TrackParameters.hpp"
#include "Acts/EventData/detail/TransformationFreeToBound.hpp"
#include "Acts/Definitions/Units.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <set>
#include <cmath>

namespace WB92 {
using Json = nlohmann::json;
using V4 = Eigen::Matrix<double,4,1>;
using V5 = Eigen::Matrix<double,5,1>;
using V6 = Eigen::Matrix<double,6,1>;

template<class Derived> Json encode(const Eigen::MatrixBase<Derived>& value) {
  if(!value.allFinite()) throw std::runtime_error("nonfinite scientific matrix");
  Json out=Json::array();
  for(int i=0;i<value.rows();++i) {
    Json row=Json::array();
    for(int j=0;j<value.cols();++j) row.push_back(value(i,j));
    out.push_back(row);
  }
  return out;
}
template<int N> Eigen::Matrix<double,N,1> vector(const Json& j) {
  Eigen::Matrix<double,N,1> v;
  for(int i=0;i<N;++i) v[i]=j.at(i).is_array()?j.at(i).at(0).get<double>():j.at(i).get<double>();
  if(!v.allFinite()) throw std::runtime_error("nonfinite fixture");
  return v;
}
Acts::Transform3 transform(const Json& j) {
  Acts::Transform3 t=Acts::Transform3::Identity();
  for(int i=0;i<4;++i) for(int k=0;k<4;++k) t.matrix()(i,k)=j.at(i).at(k).get<double>();
  return t;
}
Acts::Transform3 exp(const V6& d) {
  const Acts::Vector3 w=d.tail<3>();
  Eigen::Matrix3d W;
  W<<0,-w.z(),w.y(),w.z(),0,-w.x(),-w.y(),w.x(),0;
  const double a=w.norm();
  Eigen::Matrix3d R,V;
  if(a<1e-8) {
    R=Eigen::Matrix3d::Identity()+W+0.5*W*W;
    V=Eigen::Matrix3d::Identity()+0.5*W+(1./6.)*W*W;
  } else {
    R=Eigen::Matrix3d::Identity()+std::sin(a)/a*W+(1-std::cos(a))/(a*a)*W*W;
    V=Eigen::Matrix3d::Identity()+(1-std::cos(a))/(a*a)*W+(a-std::sin(a))/(a*a*a)*W*W;
  }
  Acts::Transform3 t=Acts::Transform3::Identity();
  t.linear()=R;t.translation()=V*d.head<3>();return t;
}

class CommonSeedAudit final: public AthAlgorithm {
 public:
  CommonSeedAudit(const std::string& name,ISvcLocator* svc):AthAlgorithm(name,svc) {}
  StatusCode initialize() override {
    ATH_CHECK(m_tool.retrieve());ATH_CHECK(detStore()->retrieve(m_helper,"FaserSCT_ID"));
    try {std::ifstream f(m_input.value());f>>m_fixture;if(!f)throw std::runtime_error("fixture read failed");}
    catch(const std::exception& e) {ATH_MSG_ERROR(e.what());return StatusCode::FAILURE;}
    return StatusCode::SUCCESS;
  }
  StatusCode execute() override {
    try {audit();} catch(const std::exception& e) {ATH_MSG_ERROR("WB92 fail-closed: "<<e.what());return StatusCode::FAILURE;}
    return StatusCode::SUCCESS;
  }
 private:
  ToolHandle<IFaserActsExtrapolationTool> m_tool{this,"ExtrapolationTool",""};
  Gaudi::Property<std::string> m_input{this,"FixturePath",""};
  Gaudi::Property<std::string> m_output{this,"AuditPath",""};
  const FaserSCT_ID* m_helper=nullptr;
  Json m_fixture;

  Acts::BoundTrackParameters bound(const V5& seed,double z,const Acts::GeometryContext& g) const {
    auto surface=Acts::Surface::makeShared<Acts::PlaneSurface>(Acts::Vector3(0,0,z),Acts::Vector3(0,0,1));
    const Acts::Vector3 direction=Acts::Vector3(seed[2],seed[3],1).normalized();
    auto b=Acts::detail::transformFreeToBoundParameters(Acts::Vector3(seed[0],seed[1],z),0.,direction,
      seed[4]/Acts::UnitConstants::MeV,*surface,g);
    if(!b.ok())throw std::runtime_error("free-to-bound failed");
    return Acts::BoundTrackParameters(surface,b.value(),std::nullopt,Acts::ParticleHypothesis::muon());
  }
  V4 prediction(const V5& seed,double z,const Acts::Transform3& frame,
                const EventContext& ctx,const Acts::GeometryContext& g) const {
    const auto start=bound(seed,z,g);
    Acts::Transform3 seedFrame=Acts::Transform3::Identity();seedFrame.translation().z()=z;
    // The common-state boundary is exact and never counted as a propagation.
    if((frame.matrix()-seedFrame.matrix()).cwiseAbs().maxCoeff()<1e-12)return seed.head<4>();
    auto target=Acts::Surface::makeShared<Acts::PlaneSurface>(frame);
    const double distance=(frame.translation()-start.position(g)).dot(start.direction());
    auto result=m_tool->propagate(ctx,start,*target,distance>=0?Acts::Direction::Forward:Acts::Direction::Backward);
    if(!result.has_value())throw std::runtime_error("official ACTS propagation returned no state");
    const auto local=frame.inverse()*result->position(g);
    const auto direction=frame.linear().transpose()*result->direction();
    if(std::abs(local.z())>1e-6 || std::abs(direction.z())<1e-12)
      throw std::runtime_error("target-plane intersection or local direction invalid");
    V4 out;out<<local.x(),local.y(),direction.x()/direction.z(),direction.y()/direction.z();
    if(!out.allFinite())throw std::runtime_error("nonfinite prediction");
    return out;
  }
  void audit() {
    const auto& ctx=Gaudi::Hive::currentContext();
    const xAOD::EventInfo* header=nullptr;
    if(evtStore()->retrieve(header,"EventInfo").isFailure())throw std::runtime_error("missing actual EventInfo");
    if(header->runNumber()!=m_fixture.at("actual_run").get<unsigned>() ||
       header->eventNumber()!=m_fixture.at("actual_event").get<unsigned long long>() ||
       header->runNumber()!=ctx.eventID().run_number() || header->eventNumber()!=ctx.eventID().event_number())
      throw std::runtime_error("actual source event header mismatch");
    const auto* geometry=m_tool->trackingGeometryTool();
    const auto g=geometry->getGeometryContext(ctx).context();
    const auto tracking=geometry->trackingGeometry();const auto ids=geometry->getIdentifierMap();
    FASERMagneticFieldWrapper field;auto cache=field.makeCache(m_tool->getMagneticFieldContext(ctx));
    const auto& p=m_fixture.at("protocol");
    V5 seed;seed.head<4>()=vector<4>(m_fixture.at("references").at(0).at("fixed_z_state"));
    seed[4]=m_fixture.at("references").at(0).at("q_over_p_per_MeV");
    const double z=m_fixture.at("references").at(0).at("z_state_mm");
    const V5 xs=vector<5>(p.at("seed_steps"));const V6 gs=vector<6>(p.at("geometry_steps"));
    auto start=bound(seed,z,g);
    V5 back;const auto pos=start.position(g);const auto dir=start.direction();
    back<<pos.x(),pos.y(),dir.x()/dir.z(),dir.y()/dir.z(),start.parameters()[Acts::eBoundQOverP]*Acts::UnitConstants::MeV;
    Json out={{"input_xaod",m_fixture.at("input_xaod")},{"ordinal",m_fixture.at("ordinal")},
      {"actual_run",header->runNumber()},{"actual_event",header->eventNumber()},
      {"seed",encode(seed)},{"seed_roundtrip",encode(back)},
      {"acts_q_over_p",start.parameters()[Acts::eBoundQOverP]},
      {"acts_MeV_unit",Acts::UnitConstants::MeV},{"acts_T_unit",Acts::UnitConstants::T},
      {"seed_z_mm",z},{"variant",m_fixture.at("variant")},{"targets",Json::array()}};
    for(const auto& ref:m_fixture.at("references")) {
      const int station=ref.at("station");
      Json sensors=Json::array();Acts::Transform3 A=Acts::Transform3::Identity();
      for(const auto& cluster:ref.at("clusters")) {
        Identifier strip(static_cast<Identifier::value_type>(cluster.get<unsigned long long>()));
        const Identifier wafer=m_helper->wafer_id(strip);
        if(m_helper->station(wafer)!=station || ids->count(wafer)!=1)throw std::runtime_error("wafer station/map mismatch");
        const auto* surface=tracking->findSurface(ids->at(wafer));
        if(!surface)throw std::runtime_error("missing actual ACTS sensitive surface");
        const auto current=surface->transform(g);
        Acts::Transform3 delta=Acts::Transform3::Identity();
        if(ref.contains("baseline_sensors")) {
          const auto& baseline=ref.at("baseline_sensors").at(sensors.size());
          if(baseline.at("strip")!=cluster || baseline.at("wafer")!=wafer.get_compact() ||
             baseline.at("geometry_id")!=ids->at(wafer).value())throw std::runtime_error("frozen surface identity mismatch");
          delta=current*transform(baseline.at("transform")).inverse();
        }
        if(sensors.empty())A=delta;
        sensors.push_back({{"strip",cluster},{"wafer",wafer.get_compact()},
          {"geometry_id",ids->at(wafer).value()},{"transform",encode(current.matrix())},{"delta",encode(delta.matrix())}});
      }
      if(sensors.empty())throw std::runtime_error("no sensor provenance");
      Acts::Transform3 base=Acts::Transform3::Identity();base.translation().z()=ref.at("z_state_mm");
      const Acts::Transform3 frame=A*base;
      auto h=[&](const V5& x,const Acts::Transform3& t){return prediction(x,z,t,ctx,g);};
      const V4 nominal=h(seed,frame);const auto y=vector<4>(ref.at("fixed_z_state"));
      const Acts::Vector3 globalPosition=frame*Acts::Vector3(nominal[0],nominal[1],0);
      const Acts::Vector3 globalDirection=frame.linear()*Acts::Vector3(nominal[2],nominal[3],1).normalized();
      Json target={{"station",station},{"sensors",sensors},{"frame",encode(frame.matrix())},
        {"y",encode(y)},{"h",encode(nominal)},{"repeat_h",encode(h(seed,frame))},
        {"global_position",encode(globalPosition)},{"global_direction",encode(globalDirection)},
        {"boundary_not_propagated",station==0},{"xi",Json::array()},{"theta_plane",Json::array()}};
      Json samples=Json::array();
      for(double fraction:{0.,0.5,1.}) {
        const Acts::Vector3 point=(1-fraction)*pos+fraction*globalPosition;
        auto b=field.getField(point,cache);if(!b.ok())throw std::runtime_error("field sampling failed");
        samples.push_back({{"position_mm",encode(point)},{"field_T",encode(b.value()/Acts::UnitConstants::T)}});
      }
      target["field_samples"]=samples;
      if(m_fixture.at("variant")=="baseline") {
        Eigen::Matrix<double,4,5> Jx;
        for(int k=0;k<5;++k) {
          V5 delta=V5::Zero();delta[k]=xs[k];
          const V4 plus=h(seed+delta,frame),minus=h(seed-delta,frame);
          const V4 hp=h(seed+0.5*delta,frame),hm=h(seed-0.5*delta,frame);
          Jx.col(k)=(hp-hm)/xs[k];
          target["xi"].push_back({{"plus",encode(plus)},{"minus",encode(minus)},
            {"half_plus",encode(hp)},{"half_minus",encode(hm)}});
        }
        V5 xd=xs*0.25;for(int k=1;k<5;k+=2)xd[k]*=-1;
        target["xi_direction"]={{"delta",encode(xd)},{"linear_effect",encode(Jx*xd)},
          {"full",encode(h(seed+xd,frame))},{"half",encode(h(seed+0.5*xd,frame))}};
        if(station!=0) {
          Eigen::Matrix<double,4,6> Jg;
          for(int k=0;k<6;++k) {
            V6 delta=V6::Zero();delta[k]=gs[k];
            const V4 plus=h(seed,exp(delta)*frame),minus=h(seed,exp(-delta)*frame);
            const V4 hp=h(seed,exp(0.5*delta)*frame),hm=h(seed,exp(-0.5*delta)*frame);
            Jg.col(k)=(hp-hm)/gs[k];
            target["theta_plane"].push_back({{"plus",encode(plus)},{"minus",encode(minus)},
              {"half_plus",encode(hp)},{"half_minus",encode(hm)}});
          }
          V6 gd=gs*0.25;for(int k=1;k<6;k+=2)gd[k]*=-1;
          target["theta_direction"]={{"delta",encode(gd)},{"linear_effect",encode(Jg*gd)},
            {"full",encode(h(seed,exp(gd)*frame))},{"half",encode(h(seed,exp(0.5*gd)*frame))}};
        }
      }
      out["targets"].push_back(target);
    }
    std::ifstream maps("/proc/self/maps");std::string line;std::set<std::string> libraries;
    while(std::getline(maps,line)) {
      const auto slash=line.find('/');if(slash!=std::string::npos && line.find(".so",slash)!=std::string::npos)
        libraries.insert(line.substr(slash));
    }
    out["loaded_libraries"]=libraries;
    const int fd=::open(m_output.value().c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);
    if(fd<0)throw std::runtime_error("exclusive result creation failed");::close(fd);
    std::ofstream result(m_output.value());result<<out.dump(2)<<std::endl;
    if(!result)throw std::runtime_error("result write failed");
  }
};
}
DECLARE_COMPONENT(WB92::CommonSeedAudit)
