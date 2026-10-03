#include "AthenaBaseComps/AthAlgorithm.h"
#include "GaudiKernel/ToolHandle.h"
#include "GaudiKernel/ThreadLocalContext.h"
#include "StoreGate/ReadCondHandle.h"
#include "FaserActsGeometryInterfaces/IFaserActsExtrapolationTool.h"
#include "FaserActsGeometryInterfaces/IFaserActsTrackingGeometryTool.h"
#include "FaserActsGeometry/FaserActsGeometryContext.h"
#include "FaserActsGeometry/FASERMagneticFieldWrapper.h"
#include "MagFieldConditions/FaserFieldMapCondObj.h"
#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/Geometry/TrackingVolume.hpp"
#include "Acts/Geometry/VolumeBounds.hpp"
#include "Acts/Propagator/Propagator.hpp"
#include "Acts/Propagator/EigenStepper.hpp"
#include "Acts/Propagator/VoidNavigator.hpp"
#include "Acts/Propagator/MaterialInteractor.hpp"
#include "Acts/Surfaces/PlaneSurface.hpp"
#include "Acts/EventData/detail/TransformationFreeToBound.hpp"
#include "Acts/Utilities/Helpers.hpp"
#include "xAODEventInfo/EventInfo.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <set>
#include <sstream>
#include <fcntl.h>
#include <unistd.h>

namespace WB111 {
using Json=nlohmann::json;
template<class D> Json encode(const Eigen::MatrixBase<D>& v) {
  if(!v.allFinite())throw std::runtime_error("nonfinite diagnostic value");
  Json j=Json::array();for(int i=0;i<v.rows();++i){Json r=Json::array();
    for(int k=0;k<v.cols();++k)r.push_back(v(i,k));j.push_back(r);}return j;
}
Acts::Vector3 vector(const Json& j) {
  Acts::Vector3 v;for(int i=0;i<3;++i)v[i]=j.at(i).at(0).get<double>();
  if(!v.allFinite())throw std::runtime_error("nonfinite input");return v;
}
Acts::Transform3 transform(const Json& j) {
  Acts::Transform3 t=Acts::Transform3::Identity();
  for(int i=0;i<4;++i)for(int k=0;k<4;++k)t.matrix()(i,k)=j.at(i).at(k);return t;
}
Json freeState(const Acts::Vector3& x,const Acts::Vector3& u,double q,double time) {
  return {{"position_mm",encode(x)},{"direction",encode(u)},{"qop_acts",q},{"time_acts",time}};
}
Json state(const Acts::BoundTrackParameters& s,const Acts::GeometryContext& g,const Acts::Transform3& t) {
  const Acts::Vector3 x=t.inverse()*s.position(g),u=t.linear().transpose()*s.direction();
  if(u.z()<=0)throw std::runtime_error("nonforward state");
  Eigen::Matrix<double,4,1> h;h<<x.x(),x.y(),u.x()/u.z(),u.y()/u.z();
  auto j=freeState(s.position(g),s.direction(),s.parameters()[Acts::eBoundQOverP],s.time());
  j["local"]=encode(x);j["h"]=encode(h);j["covariance_present"]=s.covariance().has_value();return j;
}
Json options(const Acts::PropagatorPlainOptions& o) {
  return {{"stepTolerance",o.stepTolerance},{"surfaceTolerance",o.surfaceTolerance},{"maxSteps",o.maxSteps},
    {"maxRungeKuttaStepTrials",o.maxRungeKuttaStepTrials},{"stepSizeCutOff",o.stepSizeCutOff},
    {"maxStepSize_mm",o.maxStepSize},{"pathLimit",o.pathLimit},{"loopProtection",o.loopProtection},
    {"loopFraction",o.loopFraction},{"forward",o.direction==Acts::Direction::Forward}};
}
class Field final:public Acts::MagneticFieldProvider {
 public:
  Cache makeCache(const Acts::MagneticFieldContext& c)const override{return m_official.makeCache(c);}
  Acts::Result<Acts::Vector3> getField(const Acts::Vector3& x,Cache& c)const override {
    auto b=m_official.getField(x,c);
    if(!b.ok())throw std::runtime_error("official field query failed");
    raw.push_back(Json{{"position_mm",encode(x)},{"field_acts",encode(*b)},
      {"field_T",encode(*b/Acts::UnitConstants::T)}});return b;
  }
  Acts::Result<Acts::Vector3> getFieldGradient(const Acts::Vector3& x,Acts::ActsMatrix<3,3>& a,Cache& c)const override {
    ++gradients;return m_official.getFieldGradient(x,a,c);
  }
  mutable Json raw=Json::array();mutable size_t gradients=0;
 private:FASERMagneticFieldWrapper m_official;
};
struct Trace {Json steps=Json::array(),last=nullptr;size_t rejected=0;};
struct Action {
  struct result_type{};Trace* trace=nullptr;const Field* field=nullptr;
  template<class S,class P,class N>void operator()(S& s,const P& p,const N&,result_type&,const Acts::Logger&)const {
    trace->last=freeState(p.position(s.stepping),p.direction(s.stepping),p.qOverP(s.stepping),p.time(s.stepping));
    trace->last["path_mm"]=s.stepping.pathAccumulated;
    if(s.stage!=Acts::PropagatorStage::postStep)return;
    Json row=trace->last;row["query_end"]=field->raw.size();
    row["rejected_trials"]=s.stepping.stepSize.nStepTrials;trace->rejected+=s.stepping.stepSize.nStepTrials;
    trace->steps.push_back(row);
  }
};
class NavigationReachability final:public AthAlgorithm {
 public:
  NavigationReachability(const std::string& n,ISvcLocator* s):AthAlgorithm(n,s){}
  StatusCode initialize()override {
    ATH_CHECK(m_tool.retrieve());ATH_CHECK(m_mapKey.initialize());ATH_CHECK(m_cacheKey.initialize());
    try{std::ifstream f(m_fixturePath.value()),c(m_controlPath.value());f>>m_fixture;c>>m_control;
      if(!f||!c)throw std::runtime_error("input read");}
    catch(const std::exception& e){ATH_MSG_ERROR(e.what());return StatusCode::FAILURE;}return StatusCode::SUCCESS;
  }
  StatusCode execute()override {
    try{audit();}catch(const std::exception& e){ATH_MSG_ERROR("WB111 fail-closed: "<<e.what());return StatusCode::FAILURE;}
    return StatusCode::SUCCESS;
  }
 private:
  ToolHandle<IFaserActsExtrapolationTool> m_tool{this,"ExtrapolationTool",""};
  SG::ReadCondHandleKey<FaserFieldMapCondObj> m_mapKey{this,"MapKey","fieldMapCondObj","actual map"};
  SG::ReadCondHandleKey<FaserFieldCacheCondObj> m_cacheKey{this,"CacheKey","fieldCondObj","actual cache"};
  Gaudi::Property<std::string> m_fixturePath{this,"FixturePath",""},m_controlPath{this,"ControlPath",""},m_output{this,"AuditPath",""};
  Json m_fixture,m_control;
  Acts::BoundTrackParameters bound(const Acts::Vector3& x,const Acts::Vector3& u,double q,double time,const Acts::GeometryContext& g) {
    auto s=Acts::Surface::makeShared<Acts::PlaneSurface>(Acts::Vector3(0,0,x.z()),Acts::Vector3(0,0,1));
    auto b=Acts::detail::transformFreeToBoundParameters(x,time,u,q,*s,g);
    if(!b.ok())throw std::runtime_error("free to bound");
    return Acts::BoundTrackParameters(s,b.value(),std::nullopt,Acts::ParticleHypothesis::muon());
  }
  void audit() {
    const auto& ctx=Gaudi::Hive::currentContext();const xAOD::EventInfo* h=nullptr;
    if(evtStore()->retrieve(h,"EventInfo").isFailure()||!h||h->runNumber()!=m_fixture.at("actual_run").get<unsigned>()||
      h->eventNumber()!=m_fixture.at("actual_event").get<unsigned long long>()||h->runNumber()!=ctx.eventID().run_number()||
      h->eventNumber()!=ctx.eventID().event_number())throw std::runtime_error("event identity");
    SG::ReadCondHandle<FaserFieldMapCondObj> map(m_mapKey,ctx);SG::ReadCondHandle<FaserFieldCacheCondObj> cache(m_cacheKey,ctx);
    if(!map.isValid()||!cache.isValid()||!map->fieldMap())throw std::runtime_error("missing field conditions");
    const auto* geometry=m_tool->trackingGeometryTool();const auto g=geometry->getGeometryContext(ctx).context();
    const auto m=m_tool->getMagneticFieldContext(ctx);
    if(m.get<const FaserFieldCacheCondObj*>()!=cache.cptr())throw std::runtime_error("field context/cache mismatch");
    const auto& s=m_control.at("seed");const double seedZ=m_control.at("seed_z_mm");
    const Acts::Vector3 pos(s.at(0).at(0),s.at(1).at(0),seedZ);
    const Acts::Vector3 u=Acts::Vector3(s.at(2).at(0),s.at(3).at(0),1).normalized();
    const auto start=bound(pos,u,s.at(4).at(0).get<double>()/Acts::UnitConstants::MeV,0.,g);
    const auto& terminal=m_control.at("terminal");
    const auto continuation=bound(vector(terminal.at("position")),vector(terminal.at("direction")),
      terminal.at("qop_acts"),terminal.at("time"),g);
    Json official=Json::array();
    for(const auto& t:m_control.at("targets")) {
      const auto frame=transform(t.at("frame"));auto plane=Acts::Surface::makeShared<Acts::PlaneSurface>(frame);
      const double distance=(frame.translation()-start.position(g)).dot(start.direction());
      auto r=m_tool->propagate(ctx,start,*plane,distance>=0?Acts::Direction::Forward:Acts::Direction::Backward);
      Json row{{"station",t.at("station")},{"historical_call_id",t.at("call_id")},{"target_frame",t.at("frame")},
        {"has_value",r.has_value()},{"start",state(start,g,start.referenceSurface().transform(g))}};
      if(r)row["state"]=state(*r,g,frame);
      if(r.has_value()!=t.at("official_has_value").get<bool>() ||
        (r && row.at("state").at("h")!=t.at("official_h")))throw std::runtime_error("official guard differs from saved nominal");
      official.push_back(row);
    }
    auto field=std::make_shared<Field>();
    using Engine=Acts::Propagator<Acts::EigenStepper<>,Acts::VoidNavigator>;
    Engine engine{Acts::EigenStepper<>(field),Acts::VoidNavigator{},Acts::getDefaultLogger("WB111Void",Acts::Logging::ERROR)};
    Json diagnostic=Json::array();
    auto call=[&](const Acts::BoundTrackParameters& input,const Json& target,const std::string& arm) {
      field->raw=Json::array();field->gradients=0;Trace trace;
      using Options=Acts::PropagatorOptions<Acts::ActionList<Action>,Acts::AbortList<>>;
      Options o(g,m);o.maxSteps=m_fixture.at("protocol").at("max_steps");
      o.maxStepSize=m_fixture.at("protocol").at("max_step_size_m").get<double>()*Acts::UnitConstants::m;
      const auto frame=transform(target.at("frame"));
      const double distance=(frame.translation()-input.position(g)).dot(input.direction());
      o.direction=distance>=0?Acts::Direction::Forward:Acts::Direction::Backward;
      o.loopProtection=Acts::VectorHelpers::perp(input.momentum())<m_control.at("pt_loopers_MeV").get<double>()*Acts::UnitConstants::MeV;
      auto& action=o.actionList.get<Action>();action.trace=&trace;action.field=field.get();
      auto plane=Acts::Surface::makeShared<Acts::PlaneSurface>(frame);
      auto r=engine.propagate(input,*plane,o);
      Json row{{"arm",arm},{"station",target.at("station")},{"target_frame",target.at("frame")},
        {"start",state(input,g,input.referenceSurface().transform(g))},{"options",options(o)},
        {"navigator","Acts::VoidNavigator"},{"user_aborters",Json::array()},{"material_actor",false},
        {"last_free",trace.last},{"accepted_trace",trace.steps},{"accepted_steps",trace.steps.size()},
        {"rejected_trials",trace.rejected},{"field_queries",field->raw},{"field_query_count",field->raw.size()},
        {"field_gradient_count",field->gradients},{"status","ERROR"}};
      if(!r.ok()){row["error_category"]=r.error().category().name();row["error_value"]=r.error().value();row["error_message"]=r.error().message();}
      else if(!r->endParameters)row["error_message"]="no endParameters";
      else {row["status"]="SUCCESS";row["state"]=state(*r->endParameters,g,frame);
        row["path_mm"]=r->pathLength;row["propagator_steps_counter"]=r->steps;}
      diagnostic.push_back(row);
    };
    for(const auto& t:m_control.at("targets"))call(start,t,"from_seed");
    call(continuation,m_control.at("targets").at(2),"from_terminal");
    EventIDRange mr,cr;if(!map.range(mr)||!cache.range(cr))throw std::runtime_error("missing field IOV");
    std::ostringstream mi,ci;mi<<mr;ci<<cr;
    const auto* world=geometry->trackingGeometry()->highestTrackingVolume();
    if(!world)throw std::runtime_error("missing World");
    Json worldMetadata{{"name",world->volumeName()},{"geometry_id",world->geometryId().value()},
      {"transform",encode(world->transform().matrix())},{"bounds_type",static_cast<int>(world->volumeBounds().type())},
      {"bounds_values",world->volumeBounds().values()}};
    std::set<std::string> libs;std::ifstream maps("/proc/self/maps");std::string line;
    while(std::getline(maps,line)){auto slash=line.find('/');
      if(slash!=std::string::npos&&line.find(".so",slash)!=std::string::npos)libs.insert(line.substr(slash));}
    Json result{{"schema","wb111_reachability_runtime_v1"},{"control_used",m_control},
      {"identity",Json{{"actual_run",h->runNumber()},{"actual_event",h->eventNumber()},
        {"input_xaod",m_fixture.at("input_xaod")},{"ordinal",m_fixture.at("ordinal")}}},
      {"world",worldMetadata},{"official",official},{"diagnostic",diagnostic},
      {"field_conditions",Json{{"map_key",m_mapKey.key()},{"cache_key",m_cacheKey.key()},
        {"map_IOV",mi.str()},{"cache_IOV",ci.str()},{"scale",cache->dipoleFieldScaleFactor()},
        {"official_context_is_actual_cache",true}}},
      {"acts_MeV_unit",Acts::UnitConstants::MeV},{"acts_T_unit",Acts::UnitConstants::T},
      {"official_calls",official.size()},{"diagnostic_calls",diagnostic.size()},{"loaded_libraries",libs}};
    int fd=::open(m_output.value().c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);
    if(fd<0)throw std::runtime_error("exclusive output");::close(fd);
    std::ofstream out(m_output.value());out<<result.dump(2)<<std::endl;
    if(!out)throw std::runtime_error("output write");
  }
};
}
DECLARE_COMPONENT(WB111::NavigationReachability)
