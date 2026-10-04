#include "AthenaBaseComps/AthAlgorithm.h"
#include "xAODEventInfo/EventInfo.h"
#include "TrackerPrepRawData/FaserSCT_ClusterContainer.h"
#include "TrackerIdentifier/FaserSCT_ID.h"
#include "TrackerReadoutGeometry/SiDetectorElement.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <cmath>
#include <map>
#include <set>
#include <fcntl.h>
#include <unistd.h>
namespace WB127 {
using Json=nlohmann::json;
double finite(double x){if(!std::isfinite(x))throw std::runtime_error("nonfinite");return x;}
template<class V> Json vec(const V& v){Json j=Json::array();for(int i=0;i<v.size();++i)j.push_back(finite(v[i]));return j;}
Json mat(const Amg::Transform3D& t){Json j=Json::array();for(int i=0;i<4;++i){Json r=Json::array();for(int k=0;k<4;++k)r.push_back(finite(t.matrix()(i,k)));j.push_back(r);}return j;}
class StripMeasurementExport final:public AthAlgorithm {
 public: StripMeasurementExport(const std::string& n,ISvcLocator* s):AthAlgorithm(n,s){}
 StatusCode initialize() override {ATH_CHECK(detStore()->retrieve(m_id,"FaserSCT_ID"));try{std::ifstream f(m_fixture.value());f>>m_f;std::ifstream s(m_seed.value());s>>m_seedJson;}catch(...){return StatusCode::FAILURE;}return StatusCode::SUCCESS;}
 StatusCode execute() override {try{audit();}catch(const std::exception& e){ATH_MSG_ERROR("WB127 export: "<<e.what());return StatusCode::FAILURE;}return StatusCode::SUCCESS;}
 private:
 Gaudi::Property<std::string> m_fixture{this,"FixturePath",""},m_seed{this,"SeedPath",""},m_output{this,"OutputPath",""};
 const FaserSCT_ID* m_id=nullptr;Json m_f,m_seedJson;
 void audit(){const xAOD::EventInfo* h=nullptr;ATH_CHECK(evtStore()->retrieve(h,"EventInfo"));if(h->runNumber()!=m_f.at("actual_run")||h->eventNumber()!=m_f.at("actual_event"))throw std::runtime_error("event identity");
  const Tracker::FaserSCT_ClusterContainer* cc=nullptr;ATH_CHECK(evtStore()->retrieve(cc,"SCT_ClusterContainer"));
  std::map<uint64_t,int> allow;for(const auto& r:m_f.at("references"))for(const auto& id:r.at("clusters")){if(!allow.emplace(id.get<uint64_t>(),r.at("station").get<int>()).second)throw std::runtime_error("duplicate allowlist");}
  std::map<uint64_t,const Tracker::FaserSCT_Cluster*> found;for(const auto* c:*cc)for(const auto* p:*c){const auto id=p->identify().get_compact();if(allow.count(id)){if(!found.emplace(id,p).second)throw std::runtime_error("duplicate cluster");if(!p->detectorElement()||m_id->station(p->identify())!=allow.at(id))throw std::runtime_error("cluster station");}}
  if(found.size()!=allow.size())throw std::runtime_error("allowlist missing");
  Json rows=Json::array();Json refs=Json::array();
  for(const auto& r:m_f.at("references")){const int station=r.at("station");const double zc=r.at("z_center_mm");Json ids=Json::array();
   for(const auto& idj:r.at("clusters")){const uint64_t id=idj;const auto* c=found.at(id);const auto* e=c->detectorElement();const Identifier wafer=e->identify();const auto& tr=e->transform();
    if((tr.linear().transpose()*tr.linear()-Amg::Matrix3D::Identity()).cwiseAbs().maxCoeff()>1e-9||std::abs(tr.linear().determinant()-1)>1e-9)throw std::runtime_error("non-rigid frame");
    const auto lp=c->localPosition();const auto gp=c->globalPosition();const auto cov=c->localCovariance();if(cov.rows()<1||cov.cols()<1||!(cov(0,0)>0)||!std::isfinite(cov(0,0)))throw std::runtime_error("invalid covariance");
    double alpha=std::abs(std::asin(e->sinStereo()));int eta=m_id->eta_module(wafer),phi=m_id->phi_module(wafer),mi=(((eta+1)/2+phi)%2==1?phi:phi+4);double sa=0,ca=0;switch(mi){case 0:case 2:case 1:case 3:sa=-std::sin(alpha);ca=std::cos(alpha);break;case 4:case 6:case 5:case 7:sa=std::sin(alpha);ca=std::cos(alpha);break;default:throw std::runtime_error("module");}if(m_id->side(wafer)>0)sa=-sa;
    const double z=gp.z()-zc,u=gp.y()*ca+gp.x()*sa;Json rdo=Json::array();for(const auto& rid:c->rdoList())rdo.push_back(rid.get_compact());
    rows.push_back(Json{{"cluster_id",id},{"station",station},{"wafer_id",wafer.get_compact()},{"local_position",vec(lp)},{"global_position",vec(gp)},{"local_covariance",Json{{"xx",finite(cov(0,0))}}},{"sensor_transform",mat(tr)},{"rdo_ids",rdo},{"sin_alpha",sa},{"cos_alpha",ca},{"z_relative_center",z},{"u",u},{"sigma_sq",finite(cov(0,0))}});ids.push_back(id);}
   refs.push_back(Json{{"station",station},{"z_center_mm",zc},{"cluster_ids",ids}});}
  const auto seed=m_seedJson.at("P");Json out{{"schema","wb127_strip_measurement_export_v1"},{"identity",Json{{"actual_run",m_f.at("actual_run")},{"actual_event",m_f.at("actual_event")},{"ordinal",m_f.at("ordinal")},{"input_xaod",m_f.at("input_xaod")}}},{"rows",rows},{"references",refs},{"p_seed",seed},{"new_reconstruction_calls",1},{"new_propagation_calls",0}};
  int fd=::open(m_output.value().c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);if(fd<0)throw std::runtime_error("exclusive output");::close(fd);std::ofstream o(m_output.value());o<<out.dump(2)<<std::endl;
 }
};}
DECLARE_COMPONENT(WB127::StripMeasurementExport)
