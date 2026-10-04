// Saved-vector target verification only: no Stepper/Propagator/field/event API.
#include "Acts/Surfaces/PlaneSurface.hpp"
#include "Acts/Geometry/GeometryContext.hpp"
#include "Acts/Definitions/Tolerance.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <cmath>
#include <fcntl.h>
#include <unistd.h>
using Json=nlohmann::json;
Acts::Vector3 vector(const Json& data){
  if(data.size()!=3)throw std::runtime_error("vector shape");
  Acts::Vector3 v;for(int j=0;j<3;++j)v[j]=data.at(j).at(0).get<double>();
  if(!v.allFinite())throw std::runtime_error("nonfinite vector");return v;
}
Json encode(const Acts::Vector3& v){
  if(!v.allFinite())throw std::runtime_error("nonfinite result");
  return Json::array({Json::array({v.x()}),Json::array({v.y()}),Json::array({v.z()})});
}
int main(int argc,char** argv){
 try{
  if(argc!=3)throw std::runtime_error("request/result paths required");
  std::ifstream input(argv[1]);Json request;input>>request;if(!input)throw std::runtime_error("input read");
  Acts::GeometryContext ctx;Json rows=Json::array();
  for(const auto& row:request.at("rows")){
    const auto& f=row.at("frame");if(f.size()!=4)throw std::runtime_error("frame shape");
    Acts::Transform3 frame=Acts::Transform3::Identity();
    for(int j=0;j<4;++j){if(f.at(j).size()!=4)throw std::runtime_error("frame row");for(int k=0;k<4;++k)frame.matrix()(j,k)=f.at(j).at(k).get<double>();}
    if(!frame.matrix().allFinite() || (frame.linear().transpose()*frame.linear()-Acts::RotationMatrix3::Identity()).cwiseAbs().maxCoeff()>1e-8 || std::abs(frame.linear().determinant()-1.)>1e-8 || (frame.matrix().row(3)-Eigen::RowVector4d(0,0,0,1)).cwiseAbs().maxCoeff()!=0.)throw std::runtime_error("invalid rigid frame");
    const Acts::Vector3 p=vector(row.at("position_mm")),u=vector(row.at("direction"));
    if(std::abs(u.norm()-1.)>1e-8)throw std::runtime_error("nonunit direction");
    const int nav=row.at("navigation_direction").get<int>();if(nav!=1&&nav!=-1)throw std::runtime_error("navigation sign");
    auto plane=Acts::Surface::makeShared<Acts::PlaneSurface>(frame);
    const auto intersections=plane->intersect(ctx,p,nav*u,Acts::BoundaryCheck(true),Acts::s_onSurfaceTolerance);
    const auto closest=intersections.closest();
    if(!std::isfinite(closest.pathLength()))throw std::runtime_error("nonfinite/parallel intersection");
    rows.push_back({{"input",row},{"on_surface",closest.status()==Acts::Intersection3D::Status::onSurface},
      {"status_value",static_cast<int>(closest.status())},{"path_to_plane_mm",closest.pathLength()},
      {"intersection_mm",encode(closest.position())},{"local_position_mm",encode(frame.inverse()*p)},
      {"surface_tolerance_mm",Acts::s_onSurfaceTolerance}});
  }
  Json result{{"schema","wb123_installed_plane_contract_v1"},{"rows",rows},{"propagation_calls",0},{"event_access",false},{"field_access",false}};
  const int fd=::open(argv[2],O_WRONLY|O_CREAT|O_EXCL,0600);if(fd<0)throw std::runtime_error("exclusive output");::close(fd);
  std::ofstream output(argv[2]);output<<result.dump(2)<<std::endl;if(!output)throw std::runtime_error("write failure");return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
