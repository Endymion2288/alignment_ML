#include "AthenaBaseComps/AthAlgorithm.h"
#include "GaudiKernel/ToolHandle.h"
#include "GaudiKernel/ThreadLocalContext.h"
#include "FaserActsGeometryInterfaces/IFaserActsExtrapolationTool.h"
#include "FaserActsGeometryInterfaces/IFaserActsTrackingGeometryTool.h"
#include "FaserActsGeometry/FaserActsGeometryContext.h"
#include "TrackerIdentifier/FaserSCT_ID.h"
#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/Surfaces/PlaneSurface.hpp"
#include "Acts/EventData/detail/TransformationFreeToBound.hpp"
#include "Acts/Definitions/Units.hpp"
#include "xAODEventInfo/EventInfo.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <cmath>
#include <set>
#include <limits>
#include <optional>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

namespace WB125 {
using Json=nlohmann::json;
using V5=Eigen::Matrix<double,5,1>;
template<class D> Json pack(const Eigen::MatrixBase<D>& m) {
  if(!m.allFinite())throw std::runtime_error("nonfinite output");
  Json j=Json::array();for(int i=0;i<m.rows();++i){Json r=Json::array();
    for(int k=0;k<m.cols();++k)r.push_back(m(i,k));j.push_back(r);}return j;
}
template<int N> Eigen::Matrix<double,N,1> vec(const Json& j) {
  Eigen::Matrix<double,N,1> v;for(int i=0;i<N;++i)v[i]=j.at(i).is_array()?j.at(i).at(0).get<double>():j.at(i).get<double>();
  if(!v.allFinite())throw std::runtime_error("nonfinite input");return v;
}
Acts::Transform3 pose(const Json& j) {
  Acts::Transform3 t=Acts::Transform3::Identity();
  for(int i=0;i<4;++i)for(int k=0;k<4;++k)t.matrix()(i,k)=j.at(i).at(k).get<double>();
  if(!t.matrix().allFinite() || (t.linear().transpose()*t.linear()-Acts::RotationMatrix3::Identity()).cwiseAbs().maxCoeff()>1e-9
     || std::abs(t.linear().determinant()-1)>1e-9
     || (t.matrix().row(3)-Eigen::RowVector4d(0,0,0,1)).cwiseAbs().maxCoeff()!=0.)throw std::runtime_error("invalid source surface rotation");return t;
}
class PhysicalSeedAudit final:public AthAlgorithm {
 public:
  PhysicalSeedAudit(const std::string& n,ISvcLocator* s):AthAlgorithm(n,s){}
  StatusCode initialize() override {
    ATH_CHECK(m_tool.retrieve());ATH_CHECK(detStore()->retrieve(m_id,"FaserSCT_ID"));
    try{std::ifstream f(m_input.value());f>>m_f;if(!f)throw std::runtime_error("fixture read");}
    catch(const std::exception& e){ATH_MSG_ERROR(e.what());return StatusCode::FAILURE;}return StatusCode::SUCCESS;
  }
  StatusCode execute() override {
    try{audit();}catch(const std::exception& e){ATH_MSG_ERROR("WB125 fail closed: "<<e.what());return StatusCode::FAILURE;}
    return StatusCode::SUCCESS;
  }
 private:
  ToolHandle<IFaserActsExtrapolationTool> m_tool{this,"ExtrapolationTool",""};
  Gaudi::Property<std::string> m_input{this,"FixturePath",""},m_provenance{this,"ProvenancePath",""},m_output{this,"AuditPath",""};
  const FaserSCT_ID* m_id=nullptr;Json m_f,rows=Json::array();int calls=0;
  Acts::BoundTrackParameters bind(const Acts::Vector3& x,const Acts::Vector3& d,double q,
      const Acts::Transform3& frame,const Acts::GeometryContext& g)const {
    auto s=Acts::Surface::makeShared<Acts::PlaneSurface>(frame);
    auto b=Acts::detail::transformFreeToBoundParameters(x,0.,d,q/Acts::UnitConstants::MeV,*s,g);
    if(!b.ok())throw std::runtime_error("source free-to-bound failed");
    return Acts::BoundTrackParameters(s,b.value(),std::nullopt,Acts::ParticleHypothesis::muon());
  }
  Acts::BoundTrackParameters seedState(const V5& s,double z,const Acts::GeometryContext& g)const {
    // Preserve WB92's exact start-surface constructor and arithmetic for D.
    auto surface=Acts::Surface::makeShared<Acts::PlaneSurface>(Acts::Vector3(0,0,z),Acts::Vector3(0,0,1));
    const Acts::Vector3 direction=Acts::Vector3(s[2],s[3],1).normalized();
    auto b=Acts::detail::transformFreeToBoundParameters(Acts::Vector3(s[0],s[1],z),0.,direction,s[4]/Acts::UnitConstants::MeV,*surface,g);
    if(!b.ok())throw std::runtime_error("seed free-to-bound failed");
    return Acts::BoundTrackParameters(surface,b.value(),std::nullopt,Acts::ParticleHypothesis::muon());
  }
  Json describe(const Acts::BoundTrackParameters& p,const Acts::Transform3& frame,const Acts::GeometryContext& g)const {
    const Acts::Vector3 x=frame.inverse()*p.position(g),d=frame.linear().transpose()*p.direction();
    if(!x.allFinite() || !d.allFinite() || d.z()<=0)throw std::runtime_error("invalid forward state");
    Eigen::Matrix<double,4,1> h;h<<x.x(),x.y(),d.x()/d.z(),d.y()/d.z();
    auto surface=Acts::Surface::makeShared<Acts::PlaneSurface>(frame);
    auto intersection=surface->intersect(g,p.position(g),p.direction(),Acts::BoundaryCheck(true),1e-4);
    return Json{{"h",pack(h)},{"global_position_mm",pack(p.position(g))},{"direction",pack(p.direction())},
      {"local",pack(x)},{"qop_per_MeV",p.parameters()[Acts::eBoundQOverP]*Acts::UnitConstants::MeV},
      {"bound",pack(p.parameters())},{"on_surface",intersection.closest().status()==Acts::Intersection3D::Status::onSurface},
      {"covariance_present",p.covariance().has_value()}};
  }
  std::optional<const Acts::BoundTrackParameters> propagate(const Acts::BoundTrackParameters& start,
      const Acts::Transform3& frame,const EventContext& ctx,const std::string& label,int station) {
    const double distance=(frame.translation()-start.position(m_tool->trackingGeometryTool()->getGeometryContext(ctx).context())).dot(start.direction());
    auto target=Acts::Surface::makeShared<Acts::PlaneSurface>(frame);
    ++calls;ATH_MSG_INFO("WB125_CALL_BEGIN id="<<calls<<" label="<<label<<" station="<<station);
    auto result=m_tool->propagate(ctx,start,*target,distance>=0?Acts::Direction::Forward:Acts::Direction::Backward);
    ATH_MSG_INFO("WB125_CALL_END id="<<calls<<" success="<<result.has_value());return result;
  }
  void audit() {
    const auto& ctx=Gaudi::Hive::currentContext();const xAOD::EventInfo* header=nullptr;
    if(evtStore()->retrieve(header,"EventInfo").isFailure())throw std::runtime_error("header retrieve");
    if(!header || header->runNumber()!=m_f.at("actual_run").get<unsigned>() || header->eventNumber()!=m_f.at("actual_event").get<uint64_t>())throw std::runtime_error("header mismatch");
    Json provenance;std::ifstream pf(m_provenance.value());pf>>provenance;if(!pf)throw std::runtime_error("provenance missing");
    const auto* geometry=m_tool->trackingGeometryTool();const auto g=geometry->getGeometryContext(ctx).context();
    const auto tracking=geometry->trackingGeometry();const auto ids=geometry->getIdentifierMap();
    std::vector<Acts::Transform3> frames;Json targets=Json::array();
    for(const auto& r:m_f.at("references")) {
      const int station=r.at("station");if(station!=int(frames.size()))throw std::runtime_error("station order");
      Acts::Transform3 frame=Acts::Transform3::Identity();frame.translation().z()=r.at("z_state_mm");frames.push_back(frame);
      Json sensors=Json::array();for(const auto& id:r.at("clusters")) {
        Identifier strip(static_cast<Identifier::value_type>(id.get<uint64_t>()));const auto wafer=m_id->wafer_id(strip);
        if(m_id->station(wafer)!=station || ids->count(wafer)!=1)throw std::runtime_error("sensor identity");
        const auto* s=tracking->findSurface(ids->at(wafer));if(!s)throw std::runtime_error("surface missing");
        sensors.push_back(Json{{"strip",id},{"wafer",wafer.get_compact()},{"geometry_id",ids->at(wafer).value()},{"transform",pack(s->transform(g).matrix())}});
      }
      targets.push_back(Json{{"station",station},{"frame",pack(frame.matrix())},{"y",r.at("fixed_z_state")},{"sensors",sensors}});
    }
    const double z=frames.at(0).translation().z();V5 dummy;dummy.head<4>()=vec<4>(m_f.at("references").at(0).at("fixed_z_state"));dummy[4]=1e-5;
    if(m_f.at("references").at(0).at("q_over_p_per_MeV").get<double>()!=dummy[4])throw std::runtime_error("dummy changed");
    Json selection{{"status","UNKNOWN"},{"reason","SOURCE_UNAVAILABLE"}},bridge=nullptr;std::optional<V5> physical;
    try {
      const auto& collections=provenance.at("persisted_tracks");
      if(collections.size()!=1 || collections.at(0).at("key")!="CKFTrackCollection" || collections.at(0).at("container_size")!=1
          || collections.at(0).at("matching_tracks").size()!=1)throw std::runtime_error("CKFTrackCollection must contain exactly one track");
      const auto& states=collections.at(0).at("matching_tracks").at(0).at("parameters");
      if(states.empty())throw std::runtime_error("no source parameters");
      size_t selected=0;double nearest=std::numeric_limits<double>::infinity();
      for(size_t i=0;i<states.size();++i) {
        if(states.at(i).at("persistent_index").get<size_t>()!=i)throw std::runtime_error("persistent order");
        const auto x=vec<3>(states.at(i).at("global_position_native"));const double delta=std::abs(x.z()-z);
        if(delta<nearest){nearest=delta;selected=i;}
      }
      selection={{"status","SELECTED"},{"selected_index",selected},{"abs_delta_z_mm",nearest}};
      const auto& s=states.at(selected);const auto x=vec<3>(s.at("global_position_native")),p=vec<3>(s.at("global_momentum_native"));
      const double charge=s.at("charge_e"),q=charge/p.norm();const auto native=vec<5>(s.at("native_parameters"));
      if(s.at("position_unit")!="mm" || s.at("momentum_unit")!="MeV" || s.at("qop_unit")!="MeV^-1")throw std::runtime_error("unit contract");
      if(p.z()<=0 || std::abs(charge)!=1 || !std::isfinite(q) || q==0)throw std::runtime_error("charge/momentum contract");
      const double eps=8.*std::numeric_limits<float>::epsilon();
      const double phi=std::atan2(p.y(),p.x()),theta=std::atan2(p.head<2>().norm(),p.z());
      if(std::abs(native[4]-q)>eps*std::max(std::abs(native[4]),std::abs(q))
         || std::abs(std::atan2(std::sin(native[2]-phi),std::cos(native[2]-phi)))>eps*std::max(1.,std::abs(phi))
         || std::abs(native[3]-theta)>eps*std::max(1.,std::abs(theta)))throw std::runtime_error("native/global mismatch");
      if(s.at("surface_type").get<int>()!=4)throw std::runtime_error("unsupported non-plane source surface");
      const auto sourceFrame=pose(s.at("surface_transform"));const Acts::Vector3 sourceLocal=sourceFrame.inverse()*x;
      if(std::abs(sourceLocal.z())>1e-6 || (sourceLocal.head<2>()-native.head<2>()).cwiseAbs().maxCoeff()>1e-6)throw std::runtime_error("source surface/native position mismatch");
      auto start=bind(x,p.normalized(),q,sourceFrame,g);
      if((start.position(g)-x).cwiseAbs().maxCoeff()>1e-6 || (start.direction()-p.normalized()).cwiseAbs().maxCoeff()>1e-9)throw std::runtime_error("source ACTS roundtrip");
      selection["source_qop_per_MeV"]=q;selection["source_state_roundtrip"]=describe(start,sourceFrame,g);
      auto result=propagate(start,frames.at(0),ctx,"bridge",0);
      if(!result.has_value()){bridge={{"status","FAIL_OFFICIAL_NULL"},{"failed_state","UNKNOWN_NOT_EXPOSED"}};throw std::runtime_error("official bridge returned null");}
      bridge=describe(*result,frames.at(0),g);bridge["status"]="SUCCESS";
      if(!bridge.at("on_surface").get<bool>() || bridge.at("covariance_present").get<bool>())throw std::runtime_error("bridge target/C contract");
      V5 v;v.head<4>()=vec<4>(bridge.at("h"));v[4]=bridge.at("qop_per_MeV");physical=v;selection["status"]="KNOWN";selection["reason"]=nullptr;
    }catch(const std::exception& e){selection["status"]="UNKNOWN";selection["reason"]=e.what();}
    Json seeds{{"D",pack(dummy)},{"M",nullptr},{"P",nullptr}};
    if(physical){V5 mixed=dummy;mixed[4]=(*physical)[4];seeds["M"]=pack(mixed);seeds["P"]=pack(*physical);}
    const int index=m_f.at("index");
    auto run=[&](const std::string& arm,const V5& seed,int station,const std::string& kind,double multiplier) {
      Json row{{"arm",arm},{"station",station},{"kind",kind},{"qop_multiplier",multiplier},{"seed",pack(seed)},{"propagation_call",nullptr}};
      try{
        auto start=seedState(seed,z,g);
        if(station==0){row["state"]=describe(start,frames.at(0),g);row["status"]="SUCCESS";}
        else{auto result=propagate(start,frames.at(station),ctx,arm+":"+kind,station);row["propagation_call"]=calls;
          if(result){row["state"]=describe(*result,frames.at(station),g);row["status"]="SUCCESS";}
          else{row["state"]=nullptr;row["status"]="FAIL_OFFICIAL_NULL";row["failed_state"]="UNKNOWN_NOT_EXPOSED";}}
      }catch(const std::exception& e){row["status"]="UNKNOWN_ADAPTER";row["error"]=e.what();}
      rows.push_back(row);
    };
    for(const std::string arm:{"D","M","P"}) {
      if(seeds.at(arm).is_null()){for(int station=0;station<4;++station)rows.push_back(Json{{"arm",arm},{"station",station},{"kind","nominal"},{"status","UNKNOWN_SEED"},{"state",nullptr}});continue;}
      const auto seed=vec<5>(seeds.at(arm));for(int station=0;station<4;++station)run(arm,seed,station,"nominal",0.);
    }
    if(index==12 || index==20) {
      for(const std::string arm:{"D","M","P"})if(!seeds.at(arm).is_null())for(int station:{2,3})run(arm,vec<5>(seeds.at(arm)),station,"repeat",0.);
      if(physical)for(int station:{2,3})for(double multiplier:{-1.,-.5,.5,1.}){V5 perturbed=*physical;perturbed[4]+=multiplier*1e-8;run("P",perturbed,station,"fd",multiplier);}
    }
    if(calls>((index==12 || index==20)?24:10))throw std::runtime_error("call budget");
    std::set<std::string> libraries;std::ifstream maps("/proc/self/maps");std::string line;
    while(std::getline(maps,line)){auto s=line.find('/');if(s!=std::string::npos&&line.find(".so",s)!=std::string::npos)libraries.insert(line.substr(s));}
    Json result{{"schema","wb125_physical_seed_response_v1"},{"identity",provenance.at("identity")},{"index",index},
      {"selection",selection},{"bridge",bridge},{"seeds",seeds},{"targets",targets},{"responses",rows},{"propagation_calls",calls},
      {"new_reconstruction_calls",0},{"covariance_transport",false},{"material",false},{"loaded_libraries",libraries},
      {"failure_observability","OFFICIAL_LOG_ONLY; free/currentVolume/abort UNKNOWN_NOT_EXPOSED"}};
    const int fd=::open(m_output.value().c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);if(fd<0)throw std::runtime_error("exclusive output");::close(fd);
    std::ofstream out(m_output.value());out<<result.dump(2)<<std::endl;if(!out)throw std::runtime_error("output write");
  }
};
}
DECLARE_COMPONENT(WB125::PhysicalSeedAudit)
