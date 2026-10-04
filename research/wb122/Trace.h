#pragma once
#include "Acts/Propagator/EigenStepper.hpp"
#include "Acts/Propagator/detail/SteppingLogger.hpp"
#include "FaserActsGeometry/FASERMagneticFieldWrapper.h"
#include "MagFieldElements/BFieldCache.h"
#include "MagFieldConditions/FaserFieldCacheCondObj.h"
#include <nlohmann/json.hpp>
#include <functional>
#include <limits>
#include <cmath>

namespace WB122Trace {
using Json=nlohmann::json;
// Explicit-instantiation access exception [temp.explicit]. No class or layout change.
template<class Tag,typename Tag::type Member> struct ReadMember {
  friend typename Tag::type member(Tag) {return Member;}
};
#define WB122_READ_TAG(tag,cls,typ,name) struct tag {using type=typ cls::*;friend type member(tag);}; template struct ReadMember<tag,&cls::name>;
WB122_READ_TAG(Cell,MagField::FaserFieldCache,BFieldCache,m_cache3d)
WB122_READ_TAG(Map,MagField::FaserFieldCache,const MagField::FaserFieldMap*,m_fieldMap)
WB122_READ_TAG(Scale,MagField::FaserFieldCache,double,m_scale)
WB122_READ_TAG(ScaleUse,MagField::FaserFieldCache,double,m_scaleToUse)
WB122_READ_TAG(ConditionMap,FaserFieldCacheCondObj,const MagField::FaserFieldMap*,m_fieldMap)
WB122_READ_TAG(Xmin,BFieldCache,double,m_xmin)
WB122_READ_TAG(Xmax,BFieldCache,double,m_xmax)
WB122_READ_TAG(Ymin,BFieldCache,double,m_ymin)
WB122_READ_TAG(Ymax,BFieldCache,double,m_ymax)
WB122_READ_TAG(Zmin,BFieldCache,double,m_zmin)
WB122_READ_TAG(Zmax,BFieldCache,double,m_zmax)
WB122_READ_TAG(Invx,BFieldCache,float,m_invx)
WB122_READ_TAG(Invy,BFieldCache,float,m_invy)
WB122_READ_TAG(Invz,BFieldCache,float,m_invz)
WB122_READ_TAG(Bscale,BFieldCache,float,m_scale)
struct Corners {using type=float (BFieldCache::*)[3][8];friend type member(Corners);};
template struct ReadMember<Corners,&BFieldCache::m_field>;
#undef WB122_READ_TAG
inline const MagField::FaserFieldMap* conditionMap(const FaserFieldCacheCondObj& c){return c.*member(ConditionMap{});}
inline thread_local std::function<void(const Json&)> sink;
inline thread_local const Acts::Surface* requested=nullptr;
inline thread_local const MagField::FaserFieldMap* expectedMap=nullptr;
inline thread_local size_t stepIndex=0,trialIndex=0,queryIndex=0,records=0;
inline thread_local std::string queryPhase="OUTSIDE_STEP";
inline void reset(){stepIndex=trialIndex=queryIndex=records=0;queryPhase="OUTSIDE_STEP";}
inline void requireFinite(const Json& j){
  if(j.is_number_float() && !std::isfinite(j.get<double>()))throw std::runtime_error("WB122 nonfinite observed scalar");
  if(j.is_structured())for(const auto& v:j)requireFinite(v);
}
inline void emit(Json row) {
  if(!sink)return;
  if(records>=200000)throw std::runtime_error("WB122 observer record budget");
  requireFinite(row);row["observer_record_index"]=records++;sink(row);
}
template<class D> Json encode(const Eigen::MatrixBase<D>& v) {
  if(!v.allFinite())throw std::runtime_error("WB122 nonfinite observed vector");
  Json j=Json::array();for(int i=0;i<v.rows();++i){Json r=Json::array();for(int k=0;k<v.cols();++k)r.push_back(v(i,k));j.push_back(r);}return j;
}
inline Json finiteOrUnset(double x){if(std::abs(x)==std::numeric_limits<double>::max())return nullptr;
  if(!std::isfinite(x))throw std::runtime_error("WB122 nonfinite constraint");return x;}
inline Json constraints(const Acts::ConstrainedStep& c){return {{"actor_mm",finiteOrUnset(c.value(Acts::ConstrainedStep::actor))},
  {"aborter_mm",finiteOrUnset(c.value(Acts::ConstrainedStep::aborter))},{"user_mm",finiteOrUnset(c.value(Acts::ConstrainedStep::user))},
  {"accuracy_mm",finiteOrUnset(c.accuracy())},{"effective_mm",finiteOrUnset(c.value())}};}
inline Json cacheState(const MagField::FaserFieldCache& c,const Acts::Vector3& p) {
  const auto& cell=c.*member(Cell{});
  const double xmin=cell.*member(Xmin{}),xmax=cell.*member(Xmax{}),ymin=cell.*member(Ymin{}),ymax=cell.*member(Ymax{}),
               zmin=cell.*member(Zmin{}),zmax=cell.*member(Zmax{});
  const bool valid=zmin<=zmax && xmin<xmax && ymin<ymax;
  Json result{{"valid_cell",valid},{"ranges_mm",Json::array({Json::array({xmin,xmax}),Json::array({ymin,ymax}),Json::array({zmin,zmax})})},
    {"contains_query",cell.inside(p.x(),p.y(),p.z())},{"field_scale",c.*member(Scale{})},
    {"scale_to_use",c.*member(ScaleUse{})},{"actual_condition_map_match",c.*member(Map{})==expectedMap},
    {"float_inverse_widths",nullptr},{"float_corner_fields",nullptr},{"float_bscale_kT",nullptr}};
  if(valid){
    result["float_inverse_widths"]=Json::array({cell.*member(Invx{}),cell.*member(Invy{}),cell.*member(Invz{})});
    result["float_bscale_kT"]=cell.*member(Bscale{});Json fields=Json::array();
    const auto& raw=cell.*member(Corners{});
    for(int j=0;j<3;++j){Json values=Json::array();for(int i=0;i<8;++i){if(!std::isfinite(raw[j][i]))throw std::runtime_error("WB122 nonfinite cache");values.push_back(raw[j][i]);}fields.push_back(values);}
    result["float_corner_fields"]=fields;
  }
  return result;
}
class Stepper:public Acts::EigenStepper<> {
 public:
  using Base=Acts::EigenStepper<>;using Base::Base;
  void resetState(State& s,const Acts::BoundVector& p,const Acts::BoundSquareMatrix& c,const Acts::Surface& t,double h=std::numeric_limits<double>::max())const{Base::resetState(s,p,c,t,h);}
  Acts::Result<Acts::Vector3> getField(State& s,const Acts::Vector3& p)const {
    if(!sink)return Base::getField(s,p);
    auto& c=s.fieldCache.as<FASERMagneticFieldWrapper::Cache>().fieldCache;
    const Json before=cacheState(c,p);auto value=Base::getField(s,p);const Json after=cacheState(c,p);
    Json row{{"record","field_query"},{"step_index",stepIndex-1},{"trial_index",queryPhase=="FIRST_SHARED"?Json(nullptr):Json(trialIndex-1)},
      {"query_index",queryIndex++},{"query_phase",queryPhase},{"position_mm",encode(p)},
      {"cache_before",before},{"cache_after",after},{"cache_hit_before",before.at("contains_query")},
      {"cache_refilled",!before.at("contains_query").get<bool>() && after.at("contains_query").get<bool>()},
      {"outside_map_fallback",!after.at("contains_query").get<bool>()},{"ok",value.ok()}};
    if(value.ok()){row["field_native"]=encode(*value);row["field_T"]=encode(*value/Acts::UnitConstants::T);}
    else {row["error_category"]=value.error().category().name();row["error_value"]=value.error().value();}
    emit(row);return value;
  }
  Acts::Vector3 position(const State& s)const{return Base::position(s);}
  Acts::Vector3 direction(const State& s)const{return Base::direction(s);}
  double qOverP(const State& s)const{return Base::qOverP(s);}
  double absoluteMomentum(const State& s)const{return Base::absoluteMomentum(s);}
  Acts::Vector3 momentum(const State& s)const{return Base::momentum(s);}
  double charge(const State& s)const{return Base::charge(s);}
  const Acts::ParticleHypothesis& particleHypothesis(const State& s)const{return Base::particleHypothesis(s);}
  double time(const State& s)const{return Base::time(s);}
  double overstepLimit(const State& s)const{return Base::overstepLimit(s);}
  CurvilinearState curvilinearState(State& s,bool c=true)const{return Base::curvilinearState(s,c);}
  void transportCovarianceToCurvilinear(State& s)const{Base::transportCovarianceToCurvilinear(s);}
  void transportCovarianceToBound(State& s,const Acts::Surface& t,const Acts::FreeToBoundCorrection& c=Acts::FreeToBoundCorrection(false))const{Base::transportCovarianceToBound(s,t,c);}
  Acts::Intersection3D::Status updateSurfaceStatus(State& s,const Acts::Surface& t,std::uint8_t i,Acts::Direction d,const Acts::BoundaryCheck& b,Acts::ActsScalar tol=Acts::s_onSurfaceTolerance,const Acts::Logger& l=Acts::getDummyLogger())const{return Base::updateSurfaceStatus(s,t,i,d,b,tol,l);}
  using Base::updateStepSize;
  void updateStepSize(State& s,double h,Acts::ConstrainedStep::Type t,bool r=true)const{Base::updateStepSize(s,h,t,r);}
  double getStepSize(const State& s,Acts::ConstrainedStep::Type t)const{return Base::getStepSize(s,t);}
  void releaseStepSize(State& s,Acts::ConstrainedStep::Type t)const{Base::releaseStepSize(s,t);}
  std::string outputStepSize(const State& s)const{return Base::outputStepSize(s);}
  void update(State& s,const Acts::FreeVector& f,const Acts::BoundVector& b,const Acts::BoundSquareMatrix& c,const Acts::Surface& t)const{Base::update(s,f,b,c,t);}
  void update(State& s,const Acts::Vector3& p,const Acts::Vector3& d,double q,double t)const{Base::update(s,p,d,q,t);}
  void setIdentityJacobian(State& s)const{Base::setIdentityJacobian(s);}
  template<class S,class N> Acts::Result<double> step(S& s,const N& n)const;
  Acts::Result<BoundState> boundState(State& s,const Acts::Surface& t,bool transport=true,const Acts::FreeToBoundCorrection& correction=Acts::FreeToBoundCorrection(false))const {
    if(!sink)return Base::boundState(s,t,transport,correction);
    emit({{"record","bound_before"},{"requested_target",&t==requested},{"frame",encode(t.transform(s.geoContext).matrix())},
      {"position_mm",encode(Base::position(s))},{"direction",encode(Base::direction(s))},{"qop_acts",Base::qOverP(s)},
      {"time_acts",Base::time(s)},{"path_mm",s.pathAccumulated},{"cov_transport",s.covTransport}});
    auto result=Base::boundState(s,t,transport,correction);
    Json row{{"record","bound_after"},{"ok",result.ok()}};
    if(result.ok()){const auto& bound=std::get<0>(result.value());row["parameters"]=encode(bound.parameters());row["position_mm"]=encode(bound.position(s.geoContext));row["direction"]=encode(bound.direction());}
    else{row["error_value"]=result.error().value();row["error_category"]=result.error().category().name();}
    emit(row);return result;
  }
};
struct Action {
  struct result_type{};
  template<class S,class T,class N> void operator()(S& s,const T& stepper,const N& nav,result_type&,const Acts::Logger&)const {
    if(!sink)return;
    const auto* surface=nav.currentSurface(s.navigation);const auto* volume=nav.currentVolume(s.navigation);
    const Acts::Vector3 momentum=stepper.momentum(s.stepping)/Acts::UnitConstants::MeV;
    std::string stage=s.stage==Acts::PropagatorStage::prePropagation?"prePropagation":s.stage==Acts::PropagatorStage::postStep?"postStep":s.stage==Acts::PropagatorStage::postPropagation?"postPropagation":"OTHER";
    Json counter=s.stepping.stepSize.nStepTrials==std::numeric_limits<size_t>::max()?Json(nullptr):Json(s.stepping.stepSize.nStepTrials);
    emit({{"record","action"},{"stage",stage},{"accepted_step_count",stepIndex},{"position_mm",encode(stepper.position(s.stepping))},
      {"direction",encode(stepper.direction(s.stepping))},{"momentum_MeV",encode(momentum)},
      {"logger_direction",encode(momentum.normalized())},{"qop_acts",stepper.qOverP(s.stepping)},{"time_acts",stepper.time(s.stepping)},
      {"path_mm",s.stepping.pathAccumulated},{"target_reached",nav.targetReached(s.navigation)},
      {"navigation_break",nav.navigationBreak(s.navigation)},{"geometry_id",surface?surface->geometryId().value():(volume?volume->geometryId().value():0)},
      {"surface_present",surface!=nullptr},{"surface_geometry_id",surface?Json(surface->geometryId().value()):Json(nullptr)},
      {"navigation_direction",s.options.direction.sign()},{"rk_rejections",counter},{"constraints",constraints(s.stepping.stepSize)}});
  }
};
}
#include "ObservedStep.inc"
