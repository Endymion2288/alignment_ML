#pragma once
#include "Acts/Propagator/EigenStepper.hpp"
#include "Envelope.h"
#include <functional>
namespace WB101 {
using V=Acts::Vector3;using J=nlohmann::json;
inline V jsonVector(const J& row){
  if(!row.is_array()||row.size()!=3)throw std::runtime_error("WB101 vector shape");
  V out;for(int k=0;k<3;++k){const J& x=row.at(k);
    if(x.is_array()){if(x.size()!=1)throw std::runtime_error("WB101 column vector shape");out[k]=x.at(0).get<double>();}
    else out[k]=x.get<double>();}
  if(!out.allFinite())throw std::runtime_error("WB101 nonfinite vector");return out;
}
struct Control {
  double threshold=0,allowance=1e-11;
  bool record=false;
  size_t trials=0,directionRejected=0,positionRejected=0,accepted=0,nodeQueries=0;
  double maxAcceptedBudget=0;
  size_t lastQuery=0;
  J rows=J::array();
  std::function<J(const V&,const V&,double,double,const V&)> envelope;
  std::function<size_t()> queryCount;
  std::function<void(const J&)> sink;
  void reset(double tau,bool save){threshold=tau;record=save;trials=directionRejected=positionRejected=accepted=nodeQueries=0;maxAcceptedBudget=0;lastQuery=0;rows=J::array();}
  template<class State> bool judge(State& state,const V& p,const V& u,double h,double positionError,bool& directionReject){
    ++trials;directionReject=false;
    const bool position=positionError<=state.options.stepTolerance;
    double E=-1;J result;
    if(position){const auto& sd=state.stepping.stepData;
      V full=u;full+=h/6.*(sd.k1+2.*(sd.k2+sd.k3)+sd.k4);full.normalize();
      result=envelope(p,u,h,state.stepping.pars[Acts::eFreeQOverP],full);E=result.at("E");
      if(result.at("slope").at("gate")!="PASS")throw std::runtime_error("nonforward direction envelope");
      directionReject=E+allowance>threshold;
    }
    const bool ok=position&&!directionReject;
    if(!position)++positionRejected;if(directionReject)++directionRejected;if(ok){++accepted;maxAcceptedBudget=std::max(maxAcceptedBudget,E+allowance);}
    const size_t endQuery=queryCount?queryCount():0;
    if(record||sink){J row={{"trial",trials-1},{"h_mm",h},{"position_error_mm",positionError},{"position_pass",position},
      {"direction_rejected",directionReject},{"accepted",ok},{"start_position_mm",WB100::array(p)},{"start_direction",WB100::array(u)},
      {"q_over_p_Acts",state.stepping.pars[Acts::eFreeQOverP]},{"start_time_Acts",state.stepping.pars[Acts::eFreeTime]},
      {"start_path_mm",state.stepping.pathAccumulated},{"query_begin",lastQuery},{"query_end",endQuery},
      {"B_first",WB100::array(state.stepping.stepData.B_first)},{"B_middle",WB100::array(state.stepping.stepData.B_middle)},
      {"B_last",WB100::array(state.stepping.stepData.B_last)},
      {"k1",WB100::array(state.stepping.stepData.k1)},{"k2",WB100::array(state.stepping.stepData.k2)},
      {"k3",WB100::array(state.stepping.stepData.k3)},{"k4",WB100::array(state.stepping.stepData.k4)},
      {"envelope",position?result:J(nullptr)}};
      if(sink)sink(row);else rows.push_back(row);}
    lastQuery=endQuery;
    return ok;
  }
  J summary()const{return {{"threshold",threshold},{"allowance",allowance},{"trials",trials},{"direction_rejected",directionRejected},
    {"position_rejected",positionRejected},{"accepted",accepted},{"node_queries",nodeQueries},{"max_accepted_budget",maxAcceptedBudget}};}
};
class DirectionStepper:public Acts::EigenStepper<> {
 public:
  using Base=Acts::EigenStepper<>;
  DirectionStepper(std::shared_ptr<const Acts::MagneticFieldProvider> field,Control* control):Acts::EigenStepper<>(std::move(field)),m_control(control){}
  // ACTS32 checks member-pointer ownership, so inherited methods need forwarders.
  void resetState(State& s,const Acts::BoundVector& p,const Acts::BoundSquareMatrix& c,const Acts::Surface& f,double h=std::numeric_limits<double>::max())const{Base::resetState(s,p,c,f,h);}
  Acts::Result<Acts::Vector3> getField(State& s,const V& p)const{return Base::getField(s,p);}
  V position(const State& s)const{return Base::position(s);}
  V direction(const State& s)const{return Base::direction(s);}
  double qOverP(const State& s)const{return Base::qOverP(s);}
  double absoluteMomentum(const State& s)const{return Base::absoluteMomentum(s);}
  V momentum(const State& s)const{return Base::momentum(s);}
  double charge(const State& s)const{return Base::charge(s);}
  const Acts::ParticleHypothesis& particleHypothesis(const State& s)const{return Base::particleHypothesis(s);}
  double time(const State& s)const{return Base::time(s);}
  double overstepLimit(const State& s)const{return Base::overstepLimit(s);}
  Acts::Intersection3D::Status updateSurfaceStatus(State& s,const Acts::Surface& f,std::uint8_t i,Acts::Direction d,const Acts::BoundaryCheck& b,Acts::ActsScalar t=Acts::s_onSurfaceTolerance,const Acts::Logger& l=Acts::getDummyLogger())const{return Base::updateSurfaceStatus(s,f,i,d,b,t,l);}
  template<class I> void updateStepSize(State& s,const I& i,Acts::Direction d,bool r=true)const{Base::updateStepSize(s,i,d,r);}
  void updateStepSize(State& s,double h,Acts::ConstrainedStep::Type t,bool r=true)const{Base::updateStepSize(s,h,t,r);}
  double getStepSize(const State& s,Acts::ConstrainedStep::Type t)const{return Base::getStepSize(s,t);}
  void releaseStepSize(State& s,Acts::ConstrainedStep::Type t)const{Base::releaseStepSize(s,t);}
  std::string outputStepSize(const State& s)const{return Base::outputStepSize(s);}
  Acts::Result<BoundState> boundState(State& s,const Acts::Surface& f,bool t=true,const Acts::FreeToBoundCorrection& c=Acts::FreeToBoundCorrection(false))const{return Base::boundState(s,f,t,c);}
  CurvilinearState curvilinearState(State& s,bool t=true)const{return Base::curvilinearState(s,t);}
  void update(State& s,const Acts::FreeVector& p,const Acts::BoundVector& b,const Covariance& c,const Acts::Surface& f)const{Base::update(s,p,b,c,f);}
  void update(State& s,const V& p,const V& d,double q,double t)const{Base::update(s,p,d,q,t);}
  void transportCovarianceToCurvilinear(State& s)const{Base::transportCovarianceToCurvilinear(s);}
  void transportCovarianceToBound(State& s,const Acts::Surface& f,const Acts::FreeToBoundCorrection& c=Acts::FreeToBoundCorrection(false))const{Base::transportCovarianceToBound(s,f,c);}
  void setIdentityJacobian(State& s)const{Base::setIdentityJacobian(s);}
  template<class propagator_state_t,class navigator_t>
  Acts::Result<double> step(propagator_state_t& state,const navigator_t& navigator)const;
 private:
  Control* m_control;
};
}
#include "DirectionStep.inc"
