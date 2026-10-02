#include "Acceptance.h"
#include "Acts/Propagator/Propagator.hpp"
#include "Acts/Propagator/VoidNavigator.hpp"
#include "Acts/MagneticField/ConstantBField.hpp"
#include "Acts/Surfaces/PlaneSurface.hpp"
#include "Acts/EventData/TrackParameters.hpp"
#include "Acts/EventData/detail/TransformationFreeToBound.hpp"
#include <iostream>
using V=Acts::Vector3;using J=nlohmann::json;
int main(){try{
  auto check=[](bool ok,const char* msg){if(!ok)throw std::runtime_error(msg);};
  Acts::GeometryContext g;Acts::MagneticFieldContext m;
  auto surface=Acts::Surface::makeShared<Acts::PlaneSurface>(V::Zero(),V(0,0,1));
  const auto pars=Acts::detail::transformFreeToBoundParameters(V::Zero(),0.,V(0,0,1),.01,*surface,g);
  Acts::BoundTrackParameters start(surface,*pars,std::nullopt,Acts::ParticleHypothesis::muon());
  auto target=Acts::Surface::makeShared<Acts::PlaneSurface>(V(0,0,10),V(0,0,1));
  double worst=0;size_t rejected=0;
  for(bool zero:{true,false}){
    const V b=zero?V::Zero().eval():V(0,Acts::UnitConstants::T,0);
    auto field=std::make_shared<Acts::ConstantBField>(b);WB101::Control control;
    using Engine=Acts::Propagator<WB101::DirectionStepper,Acts::VoidNavigator>;
    Engine engine{WB101::DirectionStepper(field,&control),Acts::VoidNavigator{}};
    Acts::PropagatorOptions<> options(g,m);options.maxStepSize=10;options.maxSteps=10000;options.loopProtection=false;
    std::array<std::vector<double>,3> mesh;WB100::Bounds bounds{b.norm(),0,0,V::Constant(-INFINITY),V::Constant(INFINITY),true};
    control.envelope=[&](const V& p,const V& u,double h,double q,const V& full){return WB100::candidate(p,u,h,q,full,mesh,bounds,[&](const V&){return b;});};
    control.reset(0,false);const auto disabled=engine.propagate(start,*target,options);check(disabled.ok()&&disabled->endParameters.has_value(),"disabled analytic propagation");
    const V exact(-std::sin(std::asin(.01*b[1]*10)),0,std::cos(std::asin(.01*b[1]*10)));
    worst=std::max(worst,(disabled->endParameters->direction()-exact).cwiseAbs().maxCoeff());
    control.reset(1e-10,false);const auto enabled=engine.propagate(start,*target,options);check(enabled.ok()&&enabled->endParameters.has_value(),"enabled analytic propagation");
    worst=std::max(worst,(enabled->endParameters->direction()-exact).cwiseAbs().maxCoeff());
    check(control.maxAcceptedBudget<=1e-10&&control.accepted>0,"analytic acceptance budget");
    // Stress the retry path without using a scientific field to select a bound.
    control.envelope=[](const V&,const V&,double h,double,const V&){return J{{"E",std::abs(h)*1e-8},{"slope",{{"gate","PASS"}}}};};
    control.reset(1e-9,true);const auto stress=engine.propagate(start,*target,options);check(stress.ok()&&stress->endParameters.has_value(),"retry stress propagation");
    check(control.directionRejected>0,"retry stress must reject");rejected+=control.directionRejected;
    const double expectedTime=stress->pathLength*std::hypot(1.,start.particleHypothesis().mass()*.01);
    check(std::abs(stress->endParameters->time()-expectedTime)<1e-10,"rejected trials polluted time");
    check(stress->endParameters->parameters()[Acts::eBoundQOverP]==.01,"rejected trials polluted q/p");
    const J* previous=nullptr;
    for(const auto& row:control.rows){if(previous&&!previous->at("accepted").get<bool>()){
      check(row.at("start_position_mm")==previous->at("start_position_mm")&&row.at("start_direction")==previous->at("start_direction")&&
            row.at("start_time_Acts")==previous->at("start_time_Acts")&&row.at("start_path_mm")==previous->at("start_path_mm"),"rejected trial polluted state");
      if(previous->at("direction_rejected").get<bool>())check(row.at("h_mm").get<double>()==.5*previous->at("h_mm").get<double>(),"direction retry not half");}
      previous=&row;}
  }
  check(worst<1e-12,"analytic direction precision");
  bool nan=false;try{WB100::candidate(V::Zero(),V(0,0,1),1,NAN,V(0,0,1),{},WB100::Bounds{0,0,0,V::Zero(),V::Ones(),false},[](const V&){return V::Zero().eval();});}catch(...){nan=true;}
  check(nan,"NaN guard");
  std::cout<<J{{"gate","PASS"},{"analytic_direction_max",worst},{"forced_rejected_trials",rejected},{"retry_state_time_path_qop","PASS"},{"nan_rejected",nan}}.dump(2)<<std::endl;
  return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
