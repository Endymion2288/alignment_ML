#include "AthenaBaseComps/AthAlgorithm.h"
#include "xAODEventInfo/EventInfo.h"
#include "TrackerPrepRawData/FaserSCT_ClusterContainer.h"
#include "TrackerIdentifier/FaserSCT_ID.h"
#include "TrackerReadoutGeometry/SiDetectorElement.h"
#include "TrackerRIO_OnTrack/FaserSCT_ClusterOnTrack.h"
#include "TrkTrack/TrackCollection.h"
#include "TrkTrack/TrackStateOnSurface.h"
#include "TrkParameters/TrackParameters.h"
#include "TrkSurfaces/Surface.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <map>
#include <set>
#include <cmath>
#include <limits>
#include <fcntl.h>
#include <unistd.h>

namespace WB132 {
using Json=nlohmann::json;
double finite(double x){if(!std::isfinite(x))throw std::runtime_error("nonfinite EDM value");return x;}
template<class V> Json vec(const V& v){Json j=Json::array();for(int i=0;i<v.size();++i)j.push_back(finite(v[i]));return j;}
template<class M> Json mat(const M& m){Json j=Json::array();for(int i=0;i<m.rows();++i){Json r=Json::array();for(int k=0;k<m.cols();++k)r.push_back(finite(m(i,k)));j.push_back(r);}return j;}
Json parameter(const Trk::TrackParameters& p,size_t index){
  const auto& frame=p.associatedSurface().transform();
  Json j{{"persistent_index",index},{"surface_type",static_cast<int>(p.associatedSurface().type())},
    {"surface_transform",mat(frame.matrix())},{"position_unit","mm"},{"momentum_unit","MeV"},{"qop_unit","MeV^-1"},
    {"native_parameters",vec(p.parameters())},{"global_position_native",vec(p.position())},
    {"global_momentum_native",vec(p.momentum())},{"charge_e",finite(p.charge())},
    {"parameter_type",static_cast<int>(p.type())},{"covariance",nullptr}};
  if(p.covariance())j["covariance"]=mat(*p.covariance());
  const Amg::Vector3D local=frame.inverse()*p.position();
  j["position_in_parameter_frame_mm"]=vec(local);
  j["parameter_frame_roundtrip_mm"]=finite((frame*local-p.position()).norm());
  return j;
}

class NativeStateAudit final:public AthAlgorithm {
 public:
  NativeStateAudit(const std::string& n,ISvcLocator* s):AthAlgorithm(n,s){}
  StatusCode initialize()override{
    ATH_CHECK(detStore()->retrieve(m_id,"FaserSCT_ID"));
    try{std::ifstream f(m_fixture.value()),e(m_export.value()),p(m_parent.value());f>>m_f;e>>m_e;p>>m_p;
      if(!f||!e||!p)throw std::runtime_error("input read");}
    catch(const std::exception& e){ATH_MSG_ERROR(e.what());return StatusCode::FAILURE;}
    return StatusCode::SUCCESS;
  }
  StatusCode execute()override{
    try{audit();}catch(const std::exception& e){ATH_MSG_ERROR("WB132 native audit: "<<e.what());return StatusCode::FAILURE;}
    return StatusCode::SUCCESS;
  }
 private:
  Gaudi::Property<std::string> m_fixture{this,"FixturePath",""},m_export{this,"ExportPath",""},
    m_parent{this,"ParentProvenancePath",""},m_output{this,"OutputPath",""};
  const FaserSCT_ID* m_id=nullptr;Json m_f,m_e,m_p;
  void audit(){
    const xAOD::EventInfo* header=nullptr;
    if(evtStore()->retrieve(header,"EventInfo").isFailure()||!header||
       header->runNumber()!=m_f.at("actual_run").get<unsigned>()||header->eventNumber()!=m_f.at("actual_event").get<uint64_t>())
      throw std::runtime_error("event identity");
    const Tracker::FaserSCT_ClusterContainer* cc=nullptr;
    if(evtStore()->retrieve(cc,"SCT_ClusterContainer").isFailure()||!cc)throw std::runtime_error("clusters missing");
    std::set<uint64_t> allowed;
    for(const auto& row:m_e.at("rows"))if(!allowed.insert(row.at("cluster_id").get<uint64_t>()).second)throw std::runtime_error("duplicate allowlist");
    std::map<uint64_t,const Tracker::FaserSCT_Cluster*> found;
    for(const auto* collection:*cc)for(const auto* c:*collection)if(allowed.count(c->identify().get_compact()))
      if(!found.emplace(c->identify().get_compact(),c).second)throw std::runtime_error("duplicate PRD");
    if(found.size()!=allowed.size())throw std::runtime_error("selected PRD missing");
    const auto& oldContainer=m_p.at("persisted_tracks").at(0);
    if(oldContainer.at("key")!="CKFTrackCollection")throw std::runtime_error("container selection");
    const TrackCollection* tracks=nullptr;
    if(evtStore()->retrieve(tracks,"CKFTrackCollection").isFailure()||!tracks||tracks->size()!=oldContainer.at("container_size").get<size_t>())
      throw std::runtime_error("track container changed");
    const auto& old=oldContainer.at("matching_tracks").at(0);
    const size_t selected=old.at("track_index");
    if(selected!=0||selected>=tracks->size()||!tracks->at(selected))throw std::runtime_error("track selection");
    const Trk::Track* track=tracks->at(selected);
    auto parameters=track->trackParameters();
    if(!parameters)throw std::runtime_error("flat parameters missing");
    std::map<const Trk::TrackParameters*,size_t> indices;Json flat=Json::array();size_t index=0;
    for(const auto* p:*parameters){if(!p||!indices.emplace(p,index).second)throw std::runtime_error("null/duplicate parameter");flat.push_back(parameter(*p,index++));}
    Json states=Json::array();const auto* tsos=track->trackStateOnSurfaces();
    if(!tsos)throw std::runtime_error("TSOS missing");
    size_t stateIndex=0;
    for(const auto* s:*tsos){
      if(!s)throw std::runtime_error("null TSOS");
      const auto* p=s->trackParameters();const auto* measurement=s->measurementOnTrack();
      const auto* rot=dynamic_cast<const Tracker::FaserSCT_ClusterOnTrack*>(measurement);
      Json flags=Json::array();for(int k=0;k<Trk::TrackStateOnSurface::NumberOfTrackStateOnSurfaceTypes;++k)
        flags.push_back(s->type(static_cast<Trk::TrackStateOnSurface::TrackStateOnSurfaceType>(k)));
      const auto& quality=s->fitQualityOnSurface();
      Json row{{"tsos_index",stateIndex++},{"type_flags",flags},{"type_mask",s->types().to_ulong()},{"type_description",s->dumpType()},
        {"parameter_index",p?Json(indices.at(p)):Json(nullptr)},{"measurement_present",measurement!=nullptr},
        {"sct_rot_present",rot!=nullptr},{"material_effects_present",s->materialEffectsOnTrack()!=nullptr},
        {"fit_quality_present",static_cast<bool>(quality)},{"fit_chi2",finite(quality.chiSquared())},
        {"fit_dof",finite(quality.doubleNumberDoF())},{"measurement",nullptr}};
      if(rot){
        const auto* prd=rot->prepRawData();const auto id=prd?prd->identify():rot->identify();
        const auto& lp=rot->localParameters();
        Json m{{"cluster_id",id.get_compact()},{"rot_identifier",rot->identify().get_compact()},
          {"wafer_id",m_id->wafer_id(id).get_compact()},{"station",m_id->station(id)},
          {"prd_link_resolved",prd!=nullptr},{"prd_link_valid",rot->prepRawDataLink().isValid()},
          {"selected_prd",allowed.count(id.get_compact())!=0},{"selected_prd_pointer_equal",false},
          {"local_parameter_key",lp.parameterKey()},{"rot_dimension",lp.dimension()},
          {"rot_loc1",lp.contains(Trk::loc1)?Json(finite(lp[Trk::loc1])):Json(nullptr)},
          {"rot_loc2",lp.contains(Trk::loc2)?Json(finite(lp[Trk::loc2])):Json(nullptr)},
          {"rot_covariance",mat(rot->localCovariance())},{"sensor_transform",nullptr},
          {"native_prediction",nullptr}};
        if(prd&&allowed.count(id.get_compact()))m["selected_prd_pointer_equal"]=prd==found.at(id.get_compact());
        if(prd&&rot->detectorElement()){
          const auto& surface=rot->associatedSurface();const auto& frame=surface.transform();
          m["sensor_transform"]=mat(frame.matrix());m["measurement_surface_type"]=static_cast<int>(surface.type());
          m["prd_local_position"]=vec(prd->localPosition());m["prd_local_covariance"]=mat(prd->localCovariance());
          m["prd_global_position"]=vec(prd->globalPosition());
          Json rdos=Json::array();for(const auto& rid:prd->rdoList())rdos.push_back(rid.get_compact());m["rdo_ids"]=rdos;
          if(p){
            const Amg::Vector3D local=frame.inverse()*p->position();
            const double tolerance=std::max(1e-6,4.*std::numeric_limits<float>::epsilon()*
              std::max({1.,p->position().cwiseAbs().maxCoeff(),frame.translation().cwiseAbs().maxCoeff()}));
            m["native_prediction"]={{"sensor_local_position_mm",vec(local)},{"loc0_mm",finite(local.x())},
              {"plane_distance_mm",finite(local.z())},{"persistence_plane_tolerance_mm",tolerance},
              {"on_plane_with_persistence_tolerance",std::abs(local.z())<=tolerance},
              {"strict_is_on_surface",surface.isOnSurface(p->position(),true,1e-6,1e-6)},
              {"inside_bounds",surface.insideBounds(local.head<2>(),0.,0.)},
              {"sensor_frame_roundtrip_mm",finite((frame*local-p->position()).norm())},
              {"prd_loc0_residual_mm",finite(prd->localPosition()[0]-local.x())},
              {"rot_loc1_residual_mm",lp.contains(Trk::loc1)?Json(finite(lp[Trk::loc1]-local.x())):Json(nullptr)}};
          }
        }
        row["measurement"]=m;
      }
      states.push_back(row);
    }
    Json raw=Json::array();for(const auto& original:m_e.at("rows")){
      const uint64_t id=original.at("cluster_id");const auto* c=found.at(id);if(!c->detectorElement())throw std::runtime_error("PRD detector element");
      Json rdos=Json::array();for(const auto& rid:c->rdoList())rdos.push_back(rid.get_compact());
      raw.push_back(Json{{"cluster_id",id},{"wafer_id",m_id->wafer_id(c->identify()).get_compact()},{"station",m_id->station(c->identify())},
        {"local_position",vec(c->localPosition())},{"sigma_sq",finite(c->localCovariance()(0,0))},
        {"sensor_transform",mat(c->detectorElement()->transform().matrix())},{"rdo_ids",rdos}});
    }
    const auto& info=track->info();const auto* fq=track->fitQuality();
    Json result{{"schema","wb132_native_state_export_v1"},{"identity",m_e.at("identity")},{"flat_parameters",flat},
      {"track_key","CKFTrackCollection"},{"track_index",selected},{"container_size",tracks->size()},
      {"states",states},{"selected_prds",raw},{"track_info",Json{{"fitter",static_cast<int>(info.trackFitter())},
        {"particle_hypothesis",static_cast<int>(info.particleHypothesis())},{"properties",info.properties().to_string()},
        {"pattern_recognition",info.patternRecognition().to_string()},{"dump",info.dumpInfo()}}},
      {"global_fit_quality",fq?Json{{"chi2",finite(fq->chiSquared())},{"dof",finite(fq->doubleNumberDoF())}}:Json(nullptr)},
      {"new_reconstruction_calls",0},{"new_propagation_calls",0},{"algorithm_field_queries",0},{"truth_access",false}};
    const int fd=::open(m_output.value().c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);if(fd<0)throw std::runtime_error("exclusive output");::close(fd);
    std::ofstream out(m_output.value());out<<result.dump(2)<<std::endl;if(!out)throw std::runtime_error("write");
  }
};
}
DECLARE_COMPONENT(WB132::NativeStateAudit)
