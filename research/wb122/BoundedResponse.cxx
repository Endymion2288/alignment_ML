#include "Trace.h"
#include "AthenaBaseComps/AthAlgorithm.h"
#include "GaudiKernel/ToolHandle.h"
#include "GaudiKernel/ThreadLocalContext.h"
#include "StoreGate/ReadCondHandle.h"
#include "FaserActsGeometryInterfaces/IFaserActsExtrapolationTool.h"
#include "FaserActsGeometryInterfaces/IFaserActsTrackingGeometryTool.h"
#include "FaserActsGeometry/FaserActsGeometryContext.h"
#include "MagFieldConditions/FaserFieldMapCondObj.h"
#include "MagFieldConditions/FaserFieldCacheCondObj.h"
#include "TrackerIdentifier/FaserSCT_ID.h"
#include "xAODEventInfo/EventInfo.h"
#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/Geometry/TrackingVolume.hpp"
#include "Acts/Geometry/VolumeBounds.hpp"
#include "Acts/Surfaces/PlaneSurface.hpp"
#include "Acts/EventData/detail/TransformationFreeToBound.hpp"
#include "Acts/Definitions/Units.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <set>
#include <sstream>
#include <limits>
#include <cmath>
#include <fcntl.h>
#include <unistd.h>

namespace WB122 {
using Json=nlohmann::json;
using V5=Eigen::Matrix<double,5,1>;
template<class D> Json encode(const Eigen::MatrixBase<D>& v) {
  if(!v.allFinite())throw std::runtime_error("nonfinite matrix");
  Json j=Json::array();for(int i=0;i<v.rows();++i){Json r=Json::array();
    for(int k=0;k<v.cols();++k)r.push_back(v(i,k));j.push_back(r);}return j;
}
V5 vector(const Json& j) {
  V5 v;for(int i=0;i<5;++i)v[i]=j.at(i).is_array()?j.at(i).at(0).get<double>():j.at(i).get<double>();
  if(!v.allFinite())throw std::runtime_error("nonfinite seed");return v;
}
Acts::Transform3 transform(const Json& j) {
  Acts::Transform3 t=Acts::Transform3::Identity();
  for(int i=0;i<4;++i)for(int k=0;k<4;++k)t.matrix()(i,k)=j.at(i).at(k);return t;
}
void writeNew(const std::string& path,const Json& data) {
  int fd=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);
  if(fd<0)throw std::runtime_error("exclusive output");::close(fd);
  std::ofstream f(path);f<<data.dump(2)<<std::endl;if(!f)throw std::runtime_error("output write");
}
class BoundedResponse final:public AthAlgorithm {
 public:
  BoundedResponse(const std::string& n,ISvcLocator* s):AthAlgorithm(n,s){}
  StatusCode initialize()override {
    ATH_CHECK(m_diagnostic.retrieve());
    ATH_CHECK(m_tool.retrieve());ATH_CHECK(m_mapKey.initialize());ATH_CHECK(m_cacheKey.initialize());
    ATH_CHECK(detStore()->retrieve(m_helper,"FaserSCT_ID"));
    try {std::ifstream f(m_fixturePath.value()),c(m_controlPath.value());f>>m_fixture;c>>m_control;
      if(!f||!c)throw std::runtime_error("input read");
      int fd=::open((m_output.value()+".calls.ndjson").c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);
      if(fd<0)throw std::runtime_error("exclusive calls");::close(fd);
      m_calls.open(m_output.value()+".calls.ndjson",std::ios::app);
      int ofd=::open((m_output.value()+".observer.ndjson").c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);
      if(ofd<0)throw std::runtime_error("exclusive observer stream");::close(ofd);
      m_observer.open(m_output.value()+".observer.ndjson",std::ios::app);
    }catch(const std::exception& e){ATH_MSG_ERROR(e.what());return StatusCode::FAILURE;}
    return StatusCode::SUCCESS;
  }
  StatusCode execute()override {
    try {audit();}catch(const std::exception& e){record(Json{{"record","terminal"},{"status","FAIL_CLOSED"},{"error",e.what()}});
      ATH_MSG_ERROR("WB122 fail-closed: "<<e.what());return StatusCode::FAILURE;}
    return StatusCode::SUCCESS;
  }
 private:
  ToolHandle<IFaserActsExtrapolationTool> m_diagnostic{this,"DiagnosticTool",""};
  ToolHandle<IFaserActsExtrapolationTool> m_tool{this,"ExtrapolationTool",""};
  SG::ReadCondHandleKey<FaserFieldMapCondObj> m_mapKey{this,"MapKey","fieldMapCondObj","actual map"};
  SG::ReadCondHandleKey<FaserFieldCacheCondObj> m_cacheKey{this,"CacheKey","fieldCondObj","actual cache"};
  Gaudi::Property<std::string> m_fixturePath{this,"FixturePath",""},m_controlPath{this,"ControlPath",""},m_output{this,"AuditPath",""};
  const FaserSCT_ID* m_helper=nullptr;
  Json m_fixture,m_control;std::ofstream m_calls,m_observer;size_t m_observerCount=0;
  void record(const Json& j){m_calls<<j.dump()<<std::endl;if(!m_calls)throw std::runtime_error("trace write");}
  Acts::BoundTrackParameters bound(const V5& seed,double z,const Acts::GeometryContext& g)const {
    auto surface=Acts::Surface::makeShared<Acts::PlaneSurface>(Acts::Vector3(0,0,z),Acts::Vector3(0,0,1));
    const Acts::Vector3 direction=Acts::Vector3(seed[2],seed[3],1).normalized();
    auto b=Acts::detail::transformFreeToBoundParameters(Acts::Vector3(seed[0],seed[1],z),0.,direction,
      seed[4]/Acts::UnitConstants::MeV,*surface,g);
    if(!b.ok())throw std::runtime_error("free-to-bound failed");
    return Acts::BoundTrackParameters(surface,b.value(),std::nullopt,Acts::ParticleHypothesis::muon());
  }
  Json state(const Acts::BoundTrackParameters& s,const Acts::GeometryContext& g,const Acts::Transform3& frame)const {
    Acts::Vector3 x=frame.inverse()*s.position(g),u=frame.linear().transpose()*s.direction();
    if(std::abs(x.z())>1e-6||std::abs(u.z())<1e-12)throw std::runtime_error("target-plane/local-direction invalid");
    Eigen::Matrix<double,4,1> h;h<<x.x(),x.y(),u.x()/u.z(),u.y()/u.z();
    return Json{{"h",encode(h)},{"local_position_mm",encode(x)},{"global_position_mm",encode(s.position(g))},
      {"global_direction",encode(s.direction())},{"qop_acts",s.parameters()[Acts::eBoundQOverP]},
      {"time_acts",s.time()},{"covariance_present",s.covariance().has_value()}};
  }
  void audit() {
    const auto& ctx=Gaudi::Hive::currentContext();const xAOD::EventInfo* h=nullptr;
    if(evtStore()->retrieve(h,"EventInfo").isFailure()||!h||h->runNumber()!=m_fixture.at("actual_run").get<unsigned>()||
      h->eventNumber()!=m_fixture.at("actual_event").get<unsigned long long>()||h->runNumber()!=ctx.eventID().run_number()||
      h->eventNumber()!=ctx.eventID().event_number())throw std::runtime_error("event identity");
    if(m_fixture.at("index")!=12||m_fixture.at("ordinal")!=2268||h->runNumber()!=100044||h->eventNumber()!=2268)
      throw std::runtime_error("single seen allowlist");
    SG::ReadCondHandle<FaserFieldMapCondObj> map(m_mapKey,ctx);SG::ReadCondHandle<FaserFieldCacheCondObj> cache(m_cacheKey,ctx);
    if(!map.isValid()||!cache.isValid()||!map->fieldMap())throw std::runtime_error("missing field conditions");
    const auto* geo=m_tool->trackingGeometryTool();const auto g=geo->getGeometryContext(ctx).context();
    if(m_tool->getMagneticFieldContext(ctx).get<const FaserFieldCacheCondObj*>()!=cache.cptr())throw std::runtime_error("field context mismatch");
    const auto tracking=geo->trackingGeometry();const auto ids=geo->getIdentifierMap();Json sensors=Json::array();
    for(const auto& ref:m_fixture.at("references"))for(const auto& cluster:ref.at("clusters")) {
      Identifier strip(static_cast<Identifier::value_type>(cluster.get<unsigned long long>()));const Identifier wafer=m_helper->wafer_id(strip);
      if(m_helper->station(wafer)!=ref.at("station").get<int>()||ids->count(wafer)!=1)throw std::runtime_error("sensor identity");
      const auto* s=tracking->findSurface(ids->at(wafer));if(!s)throw std::runtime_error("actual sensor surface missing");
      sensors.push_back(Json{{"station",ref.at("station")},{"strip",cluster},{"wafer",wafer.get_compact()},
        {"geometry_id",ids->at(wafer).value()},{"transform",encode(s->transform(g).matrix())}});
    }
    if(WB122Trace::conditionMap(*cache.cptr())!=map->fieldMap())
      throw std::runtime_error("actual field cache/map conditions mismatch");
    const V5 nominal=vector(m_control.at("seed"));const double z=m_control.at("seed_z_mm");
    Acts::Transform3 startframe=Acts::Transform3::Identity();startframe.translation().z()=z;
    Json rows=Json::array();size_t count=0;
    if(m_control.at("samples").size()!=9||m_control.at("targets").size()!=2)throw std::runtime_error("matrix shape");
    for(const auto& sample:m_control.at("samples")) {
      const V5 seed=vector(sample.at("seed"));const double factor=sample.at("offset_multiple");
      if(seed[4]<=0||(seed.head<4>()-nominal.head<4>()).cwiseAbs().maxCoeff()!=0)throw std::runtime_error("qop-only positive seed");
      for(size_t ti=0;ti<2;++ti) {
        const auto& target=m_control.at("targets").at(ti);if(target.at("station").get<int>()!=static_cast<int>(ti)+1)throw std::runtime_error("target order");
        auto start=bound(seed,z,g);const auto frame=transform(target.at("frame"));
        auto plane=Acts::Surface::makeShared<Acts::PlaneSurface>(frame);
        const double distance=(frame.translation()-start.position(g)).dot(start.direction());
        const auto direction=distance>=0?Acts::Direction::Forward:Acts::Direction::Backward;
        for(const std::string arm:{"official","observer_disabled","observer_enabled"}) {
          // Each production interface receives a fresh independently constructed bound seed.
          auto fresh=bound(seed,z,g);
          Json input{{"call_id",++count},{"arm",arm},{"sample_id",sample.at("sample_id")},
            {"step_multiple",sample.at("step_multiple")},{"sign",sample.at("sign")},{"factor",factor},
            {"station",target.at("station")},{"seed",encode(seed)},{"seed_z_mm",z},
            {"frame",target.at("frame")},{"navigation_direction",direction==Acts::Direction::Forward?1:-1},
            {"role","POSITIVE_QOP_DIAGNOSTIC"},{"start_state",state(fresh,g,startframe)}};
          record(Json{{"record","before_official"},{"input",input}});
          Json row{{"input",input}};
          Json observations=Json::array();
          WB122Trace::reset();WB122Trace::sink={};WB122Trace::requested=plane.get();WB122Trace::expectedMap=map->fieldMap();
          if(arm=="observer_enabled")WB122Trace::sink=[&](const Json& data){
            if(m_observerCount>=3600000)throw std::runtime_error("whole observer stream budget");
            Json copy=data;copy["call_id"]=count;copy["global_record_index"]=m_observerCount++;
            m_observer<<copy.dump()<<std::endl;if(!m_observer)throw std::runtime_error("observer stream write");
            observations.push_back(copy);
          };
          auto propagateArm=[&](){return arm=="official"?m_tool->propagate(ctx,fresh,*plane,direction):m_diagnostic->propagate(ctx,fresh,*plane,direction);};
          struct Cleanup {~Cleanup(){WB122Trace::sink={};WB122Trace::requested=nullptr;WB122Trace::expectedMap=nullptr;}} cleanup;
          auto result=propagateArm();
          WB122Trace::sink={};WB122Trace::requested=nullptr;WB122Trace::expectedMap=nullptr;
          row["has_value"]=result.has_value();
          if(result){row["state"]=state(*result,g,frame);row["bound_parameters"]=encode(result->parameters());}
          else row["error_detail"]="UNKNOWN_OPTIONAL_ONLY; consult Athena log";
          row["observations"]=observations;row["observer_record_count"]=observations.size();
          record(Json{{"record","after_official"},{"row",row}});rows.push_back(row);
          if(arm=="official"&&factor==0.&&(!result||row.at("state").at("h")!=target.at("official_h")))
            throw std::runtime_error("nominal official executor guard");
        }
      }
    }
    if(count!=54)throw std::runtime_error("call bound");
    EventIDRange mr,cr;if(!map.range(mr)||!cache.range(cr))throw std::runtime_error("field IOV missing");
    std::ostringstream mi,ci;mi<<mr;ci<<cr;const auto* world=tracking->highestTrackingVolume();
    if(!world)throw std::runtime_error("World missing");
    std::set<std::string> libs;std::ifstream maps("/proc/self/maps");std::string line;
    while(std::getline(maps,line)){auto slash=line.find('/');if(slash!=std::string::npos&&line.find(".so",slash)!=std::string::npos)libs.insert(line.substr(slash));}
    Json out{{"schema","wb122_complete_rk_runtime_v1"},{"control_used",m_control},{"rows",rows},{"official_calls",count},
      {"identity",Json{{"actual_run",h->runNumber()},{"actual_event",h->eventNumber()},
        {"input_xaod",m_fixture.at("input_xaod")},{"ordinal",m_fixture.at("ordinal")}}},
      {"sensors",sensors},{"acts_MeV_unit",Acts::UnitConstants::MeV},{"acts_T_unit",Acts::UnitConstants::T},
      {"world",Json{{"name",world->volumeName()},{"geometry_id",world->geometryId().value()},
        {"transform",encode(world->transform().matrix())},{"bounds_type",static_cast<int>(world->volumeBounds().type())},{"bounds_values",world->volumeBounds().values()}}},
      {"field_conditions",Json{{"map_key",m_mapKey.key()},{"cache_key",m_cacheKey.key()},
        {"map_IOV",mi.str()},{"cache_IOV",ci.str()},{"scale",cache->dipoleFieldScaleFactor()},{"official_context_is_actual_cache",true}}},
      {"loaded_libraries",libs},{"observer_records",m_observerCount}};
    m_observer<<Json{{"record","terminal"},{"status","COMPLETED"},{"observer_records",m_observerCount},{"propagation_calls",count}}.dump()<<std::endl;
    if(!m_observer)throw std::runtime_error("observer terminal write");
    writeNew(m_output.value(),out);record(Json{{"record","terminal"},{"status","COMPLETED"},{"official_calls",count}});
  }
};
}
DECLARE_COMPONENT(WB122::BoundedResponse)
