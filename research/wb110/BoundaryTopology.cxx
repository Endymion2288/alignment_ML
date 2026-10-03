#include "AthenaBaseComps/AthAlgorithm.h"
#include "GaudiKernel/ToolHandle.h"
#include "GaudiKernel/ThreadLocalContext.h"
#include "FaserActsGeometryInterfaces/IFaserActsTrackingGeometryTool.h"
#include "FaserActsGeometry/FaserActsGeometryContext.h"
#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/Geometry/TrackingVolume.hpp"
#include "Acts/Geometry/VolumeBounds.hpp"
#include "Acts/Geometry/BoundarySurfaceT.hpp"
#include "xAODEventInfo/EventInfo.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <set>
#include <functional>
#include <fcntl.h>
#include <unistd.h>

namespace WB110 {
using Json=nlohmann::json;
template<class D> Json encode(const Eigen::MatrixBase<D>& v) {
  if(!v.allFinite())throw std::runtime_error("nonfinite geometry value");
  Json j=Json::array();
  for(int i=0;i<v.rows();++i){Json r=Json::array();for(int k=0;k<v.cols();++k)r.push_back(v(i,k));j.push_back(r);}
  return j;
}
Acts::Vector3 vector(const Json& j) {
  Acts::Vector3 v;for(int i=0;i<3;++i)v[i]=j.at(i).at(0).get<double>();
  if(!v.allFinite())throw std::runtime_error("nonfinite probe");return v;
}
Json volume(const Acts::TrackingVolume* v) {
  if(!v)return nullptr;
  return Json{{"name",v->volumeName()},{"geometry_id",v->geometryId().value()},
    {"transform",encode(v->transform().matrix())},{"bounds_type",static_cast<int>(v->volumeBounds().type())},
    {"bounds_values",v->volumeBounds().values()}};
}
Json surface(const Acts::Surface& s,const Acts::GeometryContext& g) {
  return Json{{"geometry_id",s.geometryId().value()},{"type",static_cast<int>(s.type())},
    {"frame",encode(s.transform(g).matrix())},{"bounds_type",static_cast<int>(s.bounds().type())},
    {"bounds_values",s.bounds().values()}};
}
class BoundaryTopology final:public AthAlgorithm {
 public:
  BoundaryTopology(const std::string& name,ISvcLocator* svc):AthAlgorithm(name,svc){}
  StatusCode initialize() override {
    ATH_CHECK(m_geometry.retrieve());
    try {
      std::ifstream f(m_fixturePath.value());f>>m_fixture;if(!f)throw std::runtime_error("fixture read");
      std::ifstream p(m_probePath.value());p>>m_probe;if(!p)throw std::runtime_error("probe read");
    }catch(const std::exception& e){ATH_MSG_ERROR(e.what());return StatusCode::FAILURE;}
    return StatusCode::SUCCESS;
  }
  StatusCode execute() override {
    try{audit();}catch(const std::exception& e){ATH_MSG_ERROR("WB110 fail-closed: "<<e.what());return StatusCode::FAILURE;}
    return StatusCode::SUCCESS;
  }
 private:
  ToolHandle<IFaserActsTrackingGeometryTool> m_geometry{this,"TrackingGeometryTool",""};
  Gaudi::Property<std::string> m_fixturePath{this,"FixturePath",""};
  Gaudi::Property<std::string> m_probePath{this,"ProbePath",""};
  Gaudi::Property<std::string> m_output{this,"AuditPath",""};
  Json m_fixture,m_probe;
  void audit() {
    const auto& ctx=Gaudi::Hive::currentContext();
    const xAOD::EventInfo* header=nullptr;
    if(evtStore()->retrieve(header,"EventInfo").isFailure() || header==nullptr)throw std::runtime_error("missing event header");
    if(header->runNumber()!=m_fixture.at("actual_run").get<unsigned>() ||
       header->eventNumber()!=m_fixture.at("actual_event").get<unsigned>())
      throw std::runtime_error("event identity changed");
    auto geometry=m_geometry->trackingGeometry();
    const auto g=m_geometry->getGeometryContext(ctx).context();
    const auto* world=geometry->highestTrackingVolume();
    if(!world)throw std::runtime_error("missing world");
    const auto pos=vector(m_probe.at("position"));const auto dir=vector(m_probe.at("direction"));
    const auto id=m_probe.at("current_surface").at("geometry_id").get<uint64_t>();
    Json nodes=Json::array(),edges=Json::array(),matches=Json::array();
    std::set<const Acts::TrackingVolume*> visited;
    size_t attachmentQueries=0;
    std::function<void(const Acts::TrackingVolume*)> visit;
    visit=[&](const Acts::TrackingVolume* v) {
      if(!v || !visited.insert(v).second)return;
      Json node=volume(v);Json boundaries=Json::array();
      for(const auto& boundary:v->boundarySurfaces()) {
        if(!boundary)throw std::runtime_error("null boundary surface");
        const auto& s=boundary->surfaceRepresentation();
        Json sf=surface(s,g);boundaries.push_back(sf);
        if(s.geometryId().value()==id) {
          // Pure topology queries at the original stored point/directions.
          const auto* forward=boundary->attachedVolume(g,pos,dir);
          const auto* reverse=boundary->attachedVolume(g,pos,-dir);
          attachmentQueries+=2;
          matches.push_back(Json{{"owner_geometry_id",v->geometryId().value()},
            {"surface",sf},{"forward_attachment",volume(forward)},{"reverse_attachment",volume(reverse)}});
        }
      }
      node["boundaries"]=boundaries;nodes.push_back(node);
      auto children=v->confinedVolumes();
      if(children)for(const auto& child:children->arrayObjects()) {
        if(!child)throw std::runtime_error("null confined volume");
        edges.push_back(Json{{"parent",v->geometryId().value()},{"child",child->geometryId().value()},{"kind","confined"}});visit(child.get());
      }
      for(const auto& child:v->denseVolumes()) {
        if(!child)throw std::runtime_error("null dense volume");
        edges.push_back(Json{{"parent",v->geometryId().value()},{"child",child->geometryId().value()},{"kind","dense"}});visit(child.get());
      }
    };
    visit(world);
    std::set<std::string> libraries;std::ifstream loaded("/proc/self/maps");std::string line;
    while(std::getline(loaded,line)){auto slash=line.find('/');
      if(slash!=std::string::npos && line.find(".so",slash)!=std::string::npos)libraries.insert(line.substr(slash));}
    Json result{{"schema","wb110_runtime_topology_v1"},{"probe_used",m_probe},
      {"identity",Json{{"actual_run",header->runNumber()},{"actual_event",header->eventNumber()},
        {"input_xaod",m_fixture.at("input_xaod")},{"ordinal",m_fixture.at("ordinal")}}},
      {"world",volume(world)},{"world_inside_probe",world->inside(pos,0.)},
      {"volumes",nodes},{"edges",edges},{"matches",matches},{"attachment_queries",attachmentQueries},
      {"new_propagation_calls",0},{"algorithm_field_queries",0},{"loaded_libraries",libraries}};
    const int fd=::open(m_output.value().c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);
    if(fd<0)throw std::runtime_error("exclusive output creation failed");::close(fd);
    std::ofstream out(m_output.value());out<<result.dump(2)<<std::endl;
    if(!out)throw std::runtime_error("topology write failed");
  }
};
}
DECLARE_COMPONENT(WB110::BoundaryTopology)
