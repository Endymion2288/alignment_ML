// Passive diagnostic extension of the installed ACTS EigenStepper.
#pragma once
#include "Acts/Propagator/EigenStepper.hpp"
#include "Acts/Definitions/Tolerance.hpp"
#include "Acts/Definitions/Units.hpp"
#include <nlohmann/json.hpp>
#include <functional>
#include <stdexcept>

namespace WB107Trace {
using Json=nlohmann::json;
inline thread_local std::function<void(const Json&)> sink;
inline thread_local const Acts::Surface* requested=nullptr;
inline void emit(const Json& row) {
  if(!sink)throw std::runtime_error("WB107 missing diagnostic sink");
  sink(row);
}
template<class D> Json encode(const Eigen::MatrixBase<D>& v) {
  if(!v.allFinite())throw std::runtime_error("WB107 nonfinite state");
  Json j=Json::array();
  for(int i=0;i<v.rows();++i){Json r=Json::array();for(int k=0;k<v.cols();++k)r.push_back(v(i,k));j.push_back(r);}
  return j;
}
class Stepper: public Acts::EigenStepper<> {
 public:
  using Base=Acts::EigenStepper<>;
  using Base::Base;
  // ACTS's member-pointer concept checks require methods declared on this
  // class, rather than inherited member pointers. Every adapter delegates
  // unchanged to Base; no integration, field, or navigation rule is replaced.
  void resetState(State& s,const Acts::BoundVector& p,const Acts::BoundSquareMatrix& c,
      const Acts::Surface& t,const double h=std::numeric_limits<double>::max()) const {Base::resetState(s,p,c,t,h);}
  Acts::Result<Acts::Vector3> getField(State& s,const Acts::Vector3& p) const {return Base::getField(s,p);}
  Acts::Vector3 position(const State& s) const {return Base::position(s);}
  Acts::Vector3 direction(const State& s) const {return Base::direction(s);}
  double qOverP(const State& s) const {return Base::qOverP(s);}
  double absoluteMomentum(const State& s) const {return Base::absoluteMomentum(s);}
  Acts::Vector3 momentum(const State& s) const {return Base::momentum(s);}
  double charge(const State& s) const {return Base::charge(s);}
  double time(const State& s) const {return Base::time(s);}
  double overstepLimit(const State& s) const {return Base::overstepLimit(s);}
  CurvilinearState curvilinearState(State& s,bool c=true) const {return Base::curvilinearState(s,c);}
  void transportCovarianceToCurvilinear(State& s) const {Base::transportCovarianceToCurvilinear(s);}
  void transportCovarianceToBound(State& s,const Acts::Surface& t,
      const Acts::FreeToBoundCorrection& c=Acts::FreeToBoundCorrection(false)) const {Base::transportCovarianceToBound(s,t,c);}
  Acts::Intersection3D::Status updateSurfaceStatus(State& s,const Acts::Surface& t,std::uint8_t i,
      Acts::Direction d,const Acts::BoundaryCheck& b,Acts::ActsScalar tol=Acts::s_onSurfaceTolerance,
      const Acts::Logger& l=Acts::getDummyLogger()) const {return Base::updateSurfaceStatus(s,t,i,d,b,tol,l);}
  using Base::updateStepSize;
  void updateStepSize(State& s,double h,Acts::ConstrainedStep::Type t,bool r=true) const {Base::updateStepSize(s,h,t,r);}
  double getStepSize(const State& s,Acts::ConstrainedStep::Type t) const {return Base::getStepSize(s,t);}
  void releaseStepSize(State& s,Acts::ConstrainedStep::Type t) const {Base::releaseStepSize(s,t);}
  std::string outputStepSize(const State& s) const {return Base::outputStepSize(s);}
  void update(State& s,const Acts::FreeVector& f,const Acts::BoundVector& b,
      const Acts::BoundSquareMatrix& c,const Acts::Surface& t) const {Base::update(s,f,b,c,t);}
  void update(State& s,const Acts::Vector3& p,const Acts::Vector3& d,double q,double t) const {Base::update(s,p,d,q,t);}
  template<class S,class N> Acts::Result<double> step(S& s,const N& n) const {return Base::step(s,n);}
  Acts::Result<BoundState> boundState(State& state,const Acts::Surface& surface,
      bool transportCov=true,const Acts::FreeToBoundCorrection& correction=Acts::FreeToBoundCorrection(false)) const {
    const auto& g=state.geoContext;
    emit(Json{{"record","bound_before"},{"requested_target",&surface==requested},
      {"surface_geometry_id",surface.geometryId().value()},
      {"surface_frame",encode(surface.transform(g).matrix())},
      {"position",encode(Base::position(state))},{"direction",encode(Base::direction(state))},
      {"qop_acts",Base::qOverP(state)},{"MeV_acts",Acts::UnitConstants::MeV},{"time",Base::time(state)},
      {"path_mm",state.pathAccumulated},{"cov_transport",state.covTransport},
      {"transport_cov_argument",transportCov},{"on_surface_tolerance_mm",Acts::s_onSurfaceTolerance}});
    auto result=Base::boundState(state,surface,transportCov,correction);
    Json row{{"record","bound_after"},{"ok",result.ok()}};
    if(!result.ok()) {
      row["error_category"]=result.error().category().name();
      row["error_value"]=result.error().value();row["error_message"]=result.error().message();
    }
    emit(row);
    return result;
  }
};
}
