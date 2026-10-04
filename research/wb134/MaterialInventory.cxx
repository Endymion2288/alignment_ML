#include "AthenaBaseComps/AthAlgorithm.h"
#include "GaudiKernel/ToolHandle.h"
#include "GaudiKernel/ThreadLocalContext.h"
#include "FaserActsGeometryInterfaces/IFaserActsTrackingGeometryTool.h"
#include "xAODEventInfo/EventInfo.h"
#include "Identifier/Identifier.h"
#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/Geometry/TrackingVolume.hpp"
#include "Acts/Geometry/Layer.hpp"
#include "Acts/Geometry/ApproachDescriptor.hpp"
#include "Acts/Surfaces/SurfaceArray.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "Acts/Material/ProtoSurfaceMaterial.hpp"
#include "Acts/Material/ProtoVolumeMaterial.hpp"
#include "Acts/Material/HomogeneousSurfaceMaterial.hpp"
#include "Acts/Material/BinnedSurfaceMaterial.hpp"
#include "Acts/Material/MaterialSlab.hpp"
#include "Acts/Material/Interactions.hpp"
#include "Acts/Definitions/Units.hpp"
#include "Acts/Definitions/PdgParticle.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <map>
#include <set>
#include <cmath>
#include <fcntl.h>
#include <unistd.h>
namespace WB134 {
using Json=nlohmann::json;
template<class M> Json matrix(const M& m){Json a=Json::array();for(int i=0;i<m.rows();++i){Json r=Json::array();for(int k=0;k<m.cols();++k){if(!std::isfinite(m(i,k)))throw std::runtime_error("nonfinite geometry");r.push_back(m(i,k));}a.push_back(r);}return a;}
Json slab(const Acts::MaterialSlab& s){return Json{{"valid",static_cast<bool>(s)},{"thickness_mm",s.thickness()},
  {"thickness_in_X0",s.thicknessInX0()},{"thickness_in_L0",s.thicknessInL0()}};}
class MaterialInventory final:public AthAlgorithm {
 public:
  MaterialInventory(const std::string& n,ISvcLocator* s):AthAlgorithm(n,s){}
  StatusCode initialize()override{
    ATH_CHECK(m_geometry.retrieve());
    try{std::ifstream f(m_fixture.value()),e(m_export.value()),r(m_request.value());f>>m_f;e>>m_e;r>>m_r;
      if(!f||!e||!r)throw std::runtime_error("input read");}
    catch(const std::exception& x){ATH_MSG_ERROR(x.what());return StatusCode::FAILURE;}
    return StatusCode::SUCCESS;
  }
  StatusCode execute()override{try{audit();}catch(const std::exception& x){ATH_MSG_ERROR(x.what());return StatusCode::FAILURE;}return StatusCode::SUCCESS;}
 private:
  ToolHandle<IFaserActsTrackingGeometryTool> m_geometry{this,"TrackingGeometryTool",""};
  Gaudi::Property<std::string> m_fixture{this,"FixturePath",""},m_export{this,"ExportPath",""},m_request{this,"RequestPath",""},m_output{this,"OutputPath",""};
  Json m_f,m_e,m_r,surfaces=Json::array(),volumes=Json::array();std::map<const Acts::Surface*,size_t> sindex;
  std::map<const Acts::TrackingVolume*,size_t> vindex;Acts::GeometryContext g;
  size_t surface(const Acts::Surface* s,const std::string& role,size_t owner){
    auto found=sindex.find(s);if(found!=sindex.end()){surfaces[found->second]["memberships"].push_back(Json{{"role",role},{"volume_index",owner}});return found->second;}
    const auto index=surfaces.size();sindex[s]=index;const auto* m=s->surfaceMaterial();std::string kind="NONE";
    if(m){kind=dynamic_cast<const Acts::ProtoSurfaceMaterial*>(m)?"PROTO_VACUUM":
      dynamic_cast<const Acts::HomogeneousSurfaceMaterial*>(m)?"HOMOGENEOUS":
      dynamic_cast<const Acts::BinnedSurfaceMaterial*>(m)?"BINNED":"OTHER_NONPROTO_UNKNOWN";}
    Json sample=nullptr;if(m)sample=slab(m->materialSlab(s->center(g)));
    surfaces.push_back(Json{{"index",index},{"geometry_id",s->geometryId().value()},{"surface_type",static_cast<int>(s->type())},
      {"transform",matrix(s->transform(g).matrix())},{"material_kind",kind},{"center_sample",sample},
      {"memberships",Json::array({Json{{"role",role},{"volume_index",owner}}})}});return index;
  }
  size_t volume(const Acts::TrackingVolume* v){
    auto found=vindex.find(v);if(found!=vindex.end())return found->second;
    const auto index=volumes.size();vindex[v]=index;const auto* m=v->volumeMaterial();
    volumes.push_back(Json{{"index",index},{"name",v->volumeName()},{"geometry_id",v->geometryId().value()},
      {"material_kind",m?(dynamic_cast<const Acts::ProtoVolumeMaterial*>(m)?"PROTO":"NONPROTO_UNKNOWN"):"NONE"},
      {"children",Json::array()},{"surface_indices",Json::array()}});
    std::set<size_t> local;
    for(const auto& b:v->boundarySurfaces())local.insert(surface(&b->surfaceRepresentation(),"BOUNDARY",index));
    if(v->confinedLayers())for(const auto& layer:v->confinedLayers()->arrayObjects()){
      local.insert(surface(&layer->surfaceRepresentation(),"LAYER_REPRESENTATION",index));
      if(layer->surfaceArray())for(const auto* s:layer->surfaceArray()->surfaces())local.insert(surface(s,"SENSITIVE_ARRAY",index));
      if(layer->approachDescriptor())for(const auto* s:layer->approachDescriptor()->containedSurfaces())local.insert(surface(s,"APPROACH",index));
    }
    std::set<size_t> child;
    if(v->confinedVolumes())for(const auto& c:v->confinedVolumes()->arrayObjects())child.insert(volume(c.get()));
    for(const auto& c:v->denseVolumes())child.insert(volume(c.get()));
    for(auto i:local)volumes[index]["surface_indices"].push_back(i);
    for(auto i:child)volumes[index]["children"].push_back(i);
    return index;
  }
  void audit(){
    const auto& ctx=Gaudi::Hive::currentContext();const xAOD::EventInfo* h=nullptr;
    if(evtStore()->retrieve(h,"EventInfo").isFailure()||!h||h->runNumber()!=m_f.at("actual_run").get<unsigned>()||h->eventNumber()!=m_f.at("actual_event").get<uint64_t>())throw std::runtime_error("event identity");
    if(m_r.at("identity")!=m_e.at("identity")||m_e.at("identity").at("actual_run")!=m_f.at("actual_run")||m_e.at("identity").at("actual_event")!=m_f.at("actual_event"))throw std::runtime_error("input identity");
    g=m_geometry->getGeometryContext(ctx).context();const auto tracking=m_geometry->trackingGeometry();const auto ids=m_geometry->getIdentifierMap();
    if(!tracking||!ids)throw std::runtime_error("geometry map missing");
    const auto root=volume(tracking->highestTrackingVolume());Json targets=Json::array();std::set<uint64_t> seen;
    std::set<const Acts::Surface*> sensitive;
    tracking->visitSurfaces([&](const Acts::Surface* s){if(!sindex.count(s))throw std::runtime_error("sensitive traversal coverage");sensitive.insert(s);});
    for(const auto& row:m_e.at("rows")){
      const auto id=row.at("cluster_id").get<uint64_t>();if(!seen.insert(id).second)throw std::runtime_error("duplicate target");
      Identifier wafer(static_cast<Identifier::value_type>(row.at("wafer_id").get<uint64_t>()));
      if(ids->count(wafer)!=1)throw std::runtime_error("wafer missing");
      const auto* s=tracking->findSurface(ids->at(wafer));
      if(!s||!sindex.count(s))throw std::runtime_error("target absent from recursive inventory");
      const auto t=s->transform(g).matrix();double error=0;for(int i=0;i<4;++i)for(int k=0;k<4;++k)error=std::max(error,std::abs(t(i,k)-row.at("sensor_transform").at(i).at(k).get<double>()));
      if(error>1e-9)throw std::runtime_error("target frame changed");
      targets.push_back(Json{{"cluster_id",id},{"wafer_id",row.at("wafer_id")},{"station",row.at("station")},{"surface_index",sindex.at(s)},
        {"geometry_id",s->geometryId().value()},{"frame_max_error",error}});
    }
    // Synthetic control only: never assigned to any surface or volume.
    const auto silicon=Acts::Material::fromMolarDensity(93.7,465.2,28.0855,14.,(0.002329/28.0855)*Acts::UnitConstants::mol/Acts::UnitConstants::mm3);
    const Acts::MaterialSlab thin(silicon,0.3),vacuum;
    const float theta=Acts::computeMultipleScatteringTheta0(thin,Acts::PdgParticle::eMuon,105.6583755*Acts::UnitConstants::MeV,0.01,1.);
    if(!std::isfinite(theta)||theta<=0)throw std::runtime_error("thin slab control");
    Json controls{{"synthetic_only",true},{"thin_slab",slab(thin)},{"vacuum",slab(vacuum)},
      {"momentum_GeV",100.},{"mass_MeV",105.6583755},{"theta0_rad",theta},{"lever_mm",1000.},{"position_sigma_mm",1000.*theta}};
    Json output{{"schema","wb134_material_inventory_v1"},{"identity",m_e.at("identity")},{"source",m_r.at("source")},
      {"root_volume_index",root},{"volumes",volumes},{"surfaces",surfaces},{"targets",targets},{"controls",controls},
      {"sensitive_visit_count",sensitive.size()},{"identifier_map_size",ids->size()},
      {"material_source","None"},{"map_loaded",false},{"path_integral",nullptr},{"process_noise_covariance",nullptr},
      {"new_propagation_calls",0},{"new_reconstruction_calls",0},{"algorithm_field_queries",0},{"truth_access",false}};
    int fd=::open(m_output.value().c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);if(fd<0)throw std::runtime_error("exclusive output");::close(fd);
    std::ofstream out(m_output.value());out<<output.dump(2)<<std::endl;if(!out)throw std::runtime_error("output write");
  }
};
}
DECLARE_COMPONENT(WB134::MaterialInventory)
