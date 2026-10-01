#pragma once
#include "Acts/Propagator/Propagator.hpp"
#include "Acts/Propagator/EigenStepper.hpp"
#include "Acts/Propagator/Navigator.hpp"
#include "Acts/Propagator/VoidNavigator.hpp"
#include "Acts/Propagator/MaterialInteractor.hpp"
#include "Acts/Propagator/StandardAborters.hpp"
#include "Acts/MagneticField/ConstantBField.hpp"
#include "Acts/Utilities/Helpers.hpp"
#include "FaserActsGeometry/FASERMagneticFieldWrapper.h"
#include "MagFieldElements/FaserFieldMap.h"
#include <nlohmann/json.hpp>
#include <map>
namespace WB96Observation {
using Json=nlohmann::json;
template<class T> Json vec(const T& v) {
  if(!v.allFinite())throw std::runtime_error("nonfinite observed vector");
  Json a=Json::array();for(int i=0;i<v.size();++i)a.push_back(v[i]);return a;
}
inline Json matrix(const Acts::Transform3& t) {
  Json a=Json::array();for(int i=0;i<4;++i){Json row=Json::array();for(int j=0;j<4;++j)row.push_back(t.matrix()(i,j));a.push_back(row);}return a;
}
class Field final:public Acts::MagneticFieldProvider {
 public:
  explicit Field(const MagField::FaserFieldMap* map):m_map(map) {}
  Cache makeCache(const Acts::MagneticFieldContext& c) const override {return m_official.makeCache(c);}
  Acts::Result<Acts::Vector3> getField(const Acts::Vector3& x,Cache& cache) const override {
    auto b=m_official.getField(x,cache);if(!b.ok())return b;
    if(!x.allFinite() || !b->allFinite())throw std::runtime_error("nonfinite field query");
    const auto* zone=m_map->findBFieldZone(x.x(),x.y(),x.z());++queries;if(zone)++inside;else ++outside;
    if(record)raw.push_back({{"position_mm",vec(x)},{"field_T",vec(*b/Acts::UnitConstants::T)},{"zone_id",zone?zone->id():-1}});
    return b;
  }
  Acts::Result<Acts::Vector3> getFieldGradient(const Acts::Vector3& x,Acts::ActsMatrix<3,3>& g,Cache& c) const override {
    ++gradients;return m_official.getFieldGradient(x,g,c);
  }
  void reset(bool save)const{record=save;queries=inside=outside=gradients=0;raw=Json::array();}
  mutable bool record=false;
  mutable size_t queries=0,inside=0,outside=0,gradients=0;
  mutable Json raw=Json::array();
 private:
  FASERMagneticFieldWrapper m_official;
  const MagField::FaserFieldMap* m_map;
};
struct Trace {
  bool save=false;size_t accepted=0,rejected=0,lastQuery=0;double path=0,maxError=0;
  Json steps=Json::array(),sensitive=Json::array(),lastFree;
};
struct Action {
  struct result_type {};
  Trace* trace=nullptr;
  const Field* field=nullptr;
  std::map<uint64_t,Json>* surfaces=nullptr;
  template<class State,class Stepper,class Navigator>
  void operator()(State& s,const Stepper& stepper,const Navigator& navigator,result_type&,const Acts::Logger&)const {
    const auto pos=stepper.position(s.stepping),dir=stepper.direction(s.stepping);
    trace->lastFree={{"position_mm",vec(pos)},{"direction",vec(dir)},
      {"q_over_p_Acts",stepper.qOverP(s.stepping)},{"time_Acts",stepper.time(s.stepping)},
      {"target_reached",navigator.targetReached(s.navigation)},{"navigation_break",navigator.navigationBreak(s.navigation)}};
    const auto* surf=navigator.currentSurface(s.navigation);
    uint64_t id=surf?surf->geometryId().value():0;
    if(surf && id && !surfaces->count(id))(*surfaces)[id]={{"geometry_id",id},{"surface_type",static_cast<int>(surf->type())},
      {"transform",matrix(surf->transform(s.options.geoContext.get()))},{"has_material",surf->surfaceMaterial()!=nullptr}};
    if(id&0x000000000fffff00ULL)if(trace->sensitive.empty() || trace->sensitive.back()!=id)trace->sensitive.push_back(id);
    if(s.stage!=Acts::PropagatorStage::postStep)return;
    const double h=s.stepping.pathAccumulated-trace->path;trace->path=s.stepping.pathAccumulated;
    const auto& k=s.stepping.stepData;
    const double err=std::max(h*h*((k.k1-k.k2-k.k3+k.k4).template lpNorm<1>()+
      std::abs(k.kQoP[0]-k.kQoP[1]-k.kQoP[2]+k.kQoP[3])),1e-20);
    if(!std::isfinite(err) || s.stepping.stepSize.nStepTrials==std::numeric_limits<size_t>::max())throw std::runtime_error("invalid accepted step observation");
    ++trace->accepted;trace->rejected+=s.stepping.stepSize.nStepTrials;trace->maxError=std::max(trace->maxError,err);
    if(trace->save)trace->steps.push_back({{"position_mm",vec(pos)},{"direction",vec(dir)},
      {"h_mm",h},{"rejected_trials",s.stepping.stepSize.nStepTrials},{"accepted_error_estimate",err},
      {"error_over_tolerance",err/s.options.stepTolerance},{"next_accuracy_mm",s.stepping.stepSize.accuracy()},
      {"constraints_after_post_step",s.stepping.stepSize.toString()},{"geometry_id",id},
      {"query_begin",trace->lastQuery},{"query_end",field->queries}});
    trace->lastQuery=field->queries;
  }
};
inline Json options(const Acts::PropagatorPlainOptions& o) {
  return {{"stepTolerance",o.stepTolerance},{"surfaceTolerance",o.surfaceTolerance},{"maxSteps",o.maxSteps},
    {"maxRungeKuttaStepTrials",o.maxRungeKuttaStepTrials},{"stepSizeCutOff",o.stepSizeCutOff},
    {"maxStepSize_mm",o.maxStepSize},{"pathLimit",o.pathLimit},{"loopProtection",o.loopProtection},
    {"loopFraction",o.loopFraction},{"forward",o.direction==Acts::Direction::Forward}};
}
}
