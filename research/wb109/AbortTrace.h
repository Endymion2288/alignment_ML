// Passive wrapper: delegate to the installed ACTS aborter exactly once.
#pragma once
#include "BoundTrace.h"
#include "Acts/Propagator/StandardAborters.hpp"
#include "Acts/Propagator/Propagator.hpp"
#include "Acts/Geometry/TrackingVolume.hpp"
#include "Acts/Geometry/BoundarySurfaceT.hpp"

namespace WB109Trace {
using WB107Trace::Json;
using WB107Trace::encode;
inline std::string stageName(Acts::PropagatorStage stage) {
  switch(stage) {
    case Acts::PropagatorStage::prePropagation:return "prePropagation";
    case Acts::PropagatorStage::postPropagation:return "postPropagation";
    case Acts::PropagatorStage::preStep:return "preStep";
    case Acts::PropagatorStage::postStep:return "postStep";
    default:return "invalid";
  }
}
inline Json volume(const Acts::TrackingVolume* v) {
  if(!v)return nullptr;
  return Json{{"name",v->volumeName()},{"geometry_id",v->geometryId().value()},
    {"transform",encode(v->transform().matrix())},{"bounds_type",static_cast<int>(v->volumeBounds().type())},
    {"bounds_values",v->volumeBounds().values()}};
}
inline Json surface(const Acts::Surface* p,const Acts::GeometryContext& g) {
  if(!p)return nullptr;
  return Json{{"geometry_id",p->geometryId().value()},{"type",static_cast<int>(p->type())},
    {"frame",encode(p->transform(g).matrix())},{"bounds_type",static_cast<int>(p->bounds().type())},
    {"bounds_values",p->bounds().values()}};
}
struct EndOfWorld {
  template<class State,class Stepper,class Navigator>
  bool operator()(State& s,const Stepper& stepper,const Navigator& navigator,const Acts::Logger& logger) const {
    const auto& nav=s.navigation;
    const auto& g=s.geoContext;
    const auto* current=navigator.currentSurface(nav);
    Json row{{"record","end_world_check"},{"stage",stageName(s.stage)},
      {"step_index",s.steps},{"position",encode(stepper.position(s.stepping))},
      {"direction",encode(stepper.direction(s.stepping))},{"path_mm",s.stepping.pathAccumulated},
      {"current_volume_null_before",navigator.currentVolume(nav)==nullptr},
      {"current_volume_name",navigator.currentVolume(nav)?Json(navigator.currentVolume(nav)->volumeName()):Json(nullptr)},
      {"end_of_world_before",navigator.endOfWorldReached(nav)},
      {"target_reached_before",navigator.targetReached(nav)},
      {"navigation_break_before",navigator.navigationBreak(nav)},
      {"current_surface_is_target",current!=nullptr && current==WB107Trace::requested},
      {"navigator_target_matches_requested",navigator.targetSurface(nav)==WB107Trace::requested},
      {"options",Json{{"maxSteps",s.options.maxSteps},{"maxStepSize_mm",s.options.maxStepSize},
        {"surfaceTolerance_mm",s.options.surfaceTolerance},{"stepTolerance",s.options.stepTolerance},
        {"pathLimit",s.options.pathLimit},{"loopProtection",s.options.loopProtection},
        {"forward",s.options.direction==Acts::Direction::Forward}}}};
    // Only the original call below changes propagation state.
    const bool result=Acts::EndOfWorldReached{}(s,stepper,navigator,logger);
    row["returned"]=result;
    row["target_reached_after"]=navigator.targetReached(nav);
    row["current_volume_null_after"]=navigator.currentVolume(nav)==nullptr;
    row["navigation_break_after"]=navigator.navigationBreak(nav);
    if(result) {
      row["current_surface"]=surface(current,g);
      row["target_surface"]=surface(WB107Trace::requested,g);
      row["world"]=volume(nav.worldVolume);
      row["target_volume"]=volume(nav.targetVolume);
      row["navigation_boundary_valid"]=nav.navBoundaryIndex<nav.navBoundaries.size();
      row["navigation_boundary_surface"]=row["navigation_boundary_valid"].get<bool>()
        ?surface(&nav.navBoundaries.at(nav.navBoundaryIndex).second->surfaceRepresentation(),g):Json(nullptr);
      Json boundaries=Json::array();
      if(nav.worldVolume) {
        for(const auto& b:nav.worldVolume->boundarySurfaces()) {
          const auto& p=b->surfaceRepresentation();
          Json item=surface(&p,g);item["current_surface_pointer_match"]=&p==current;
          boundaries.push_back(item);
        }
      }
      row["world_boundary_surfaces"]=boundaries;
    }
    WB107Trace::emit(row);
    return result;
  }
};
}
