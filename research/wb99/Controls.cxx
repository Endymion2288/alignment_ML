// Independent six-state RK4 calibration; synthetic fields never enter Calypso.
#include "DirectionDoubling.h"
#include "Acts/Definitions/Units.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <limits>
#include <fcntl.h>
#include <unistd.h>
using namespace WB99;
using S=Eigen::Matrix<double,6,1>;
struct Field {
  std::string kind;double h,phase,pitch,peak;
  double value(double z)const{
    if(kind=="zero")return 0;if(kind=="constant")return 1;
    if(kind=="domain")return z>=phase*h?1:0;
    if(kind=="kink")return std::clamp((z-phase*h)/pitch,0.,1.);
    return std::max(0.,1.-std::abs(z-peak*pitch)/pitch);
  }
  std::vector<double> cuts()const{
    if(kind=="domain")return {phase*h};if(kind=="kink")return {phase*h,phase*h+pitch};
    if(kind=="triangle")return {(peak-1)*pitch,peak*pitch,(peak+1)*pitch};return {};
  }
  std::pair<double,double> line(double z)const{
    if(kind=="zero")return {0,0};if(kind=="constant")return {1,0};
    if(kind=="domain")return {value(z),0};
    double slope=0;
    if(kind=="kink" && z>phase*h && z<phase*h+pitch)slope=1/pitch;
    if(kind=="triangle" && z>(peak-1)*pitch && z<peak*pitch)slope=1/pitch;
    if(kind=="triangle" && z>peak*pitch && z<(peak+1)*pitch)slope=-1/pitch;
    return {value(z)-slope*z,slope};
  }
  V get(const V& p)const{return V(0,value(p[2])*Acts::UnitConstants::T,0);}
};
S rk4(const S& x,double ds,double q,std::pair<double,double> coeff){
  auto rhs=[&](const S& y)->S{S out;out.head<3>()=y.tail<3>();const V b(0,(coeff.first+coeff.second*y[2])*Acts::UnitConstants::T,0);
    out.tail<3>()=q*V(y.tail<3>()).cross(b);return out;};
  const S a=rhs(x),b=rhs(x+.5*ds*a),c=rhs(x+.5*ds*b),d=rhs(x+ds*c);return x+ds*(a+2*b+2*c+d)/6.;
}
V reference(const Field& f,double q,double ds,double epsilon){
  S x=S::Zero();x[5]=1;double arc=0;auto cuts=f.cuts();std::sort(cuts.begin(),cuts.end());size_t index=0;
  while(index<cuts.size() && cuts[index]<=0)++index;
  size_t count=0;
  while(arc<f.h){if(++count>1000000)throw std::runtime_error("control reference iteration budget");
    while(index<cuts.size() && cuts[index]<=x[2]+epsilon*.01)++index;
    const double next=index<cuts.size()?cuts[index]:std::numeric_limits<double>::infinity();
    auto line=f.line(x[2]+epsilon);double dt=std::min(ds,f.h-arc);S trial=rk4(x,dt,q,line);
    if(trial[2]>next){double lo=0,hi=dt;for(int k=0;k<55;++k){double mid=.5*(lo+hi);if(rk4(x,mid,q,line)[2]>next)hi=mid;else lo=mid;}
      dt=.5*(lo+hi);trial=rk4(x,dt,q,line);trial[2]=next;++index;}
    if(!(dt>0) || !trial.allFinite())throw std::runtime_error("control reference stalled");
    x=trial;arc+=dt;if(f.h-arc<1e-13)arc=f.h;
  }
  return V(x.tail<3>()).normalized();
}
int main(int argc,char** argv){try{
  if(argc!=3)throw std::runtime_error("controls protocol result");std::ifstream in(argv[1]);J p;in>>p;if(!in)throw std::runtime_error("control protocol");
  J rows=J::array();size_t unknown=0,miss=0,resolved=0;double worstExact=0,zeroMax=0;
  auto run=[&](Field f,double q){const V start=V::Zero(),u(0,0,1);
    const auto full=step(start,u,f.h,q,[&](const V& x){return f.get(x);});
    const auto half1=step(start,u,f.h/2,q,[&](const V& x){return f.get(x);});
    const auto half2=step(half1.p,half1.u,f.h/2,q,[&](const V& x){return f.get(x);});
    std::vector<V> refs;J ladder=J::array();
    for(double ds:p.at("control_reference_ds_mm")){const V ref=reference(f,q,ds,p.at("control_boundary_epsilon_mm"));refs.push_back(ref);ladder.push_back({{"ds_mm",ds},{"direction",array(ref)}});}
    const double coarse=(refs[1]-refs[0]).cwiseAbs().maxCoeff(),fine=(refs[2]-refs[1]).cwiseAbs().maxCoeff();
    const bool valid=fine<=p.at("control_reference_direction_tolerance").get<double>()&&(fine<=p.at("control_reference_contraction").get<double>()*coarse||std::max(fine,coarse)<=p.at("control_reference_roundoff").get<double>());
    double exact=-1;
    if(f.kind=="constant"||f.kind=="zero"||f.kind=="domain"){
      // Straight zero-field arc reaches z=phase*h before the constant-By arc.
      const double active=f.kind=="constant"?f.h:(f.kind=="zero"?0:std::max(0.,f.h-std::max(0.,f.phase*f.h)));
      const double angle=q*Acts::UnitConstants::T*active;const V analytic(-std::sin(angle),0,std::cos(angle));
      exact=(analytic-refs.back()).cwiseAbs().maxCoeff();worstExact=std::max(worstExact,exact);
    }
    const double uncertainty=std::max(p.at("absolute_direction_floor").get<double>(),fine),defect=(full.u-refs.back()).cwiseAbs().maxCoeff(),difference=(full.u-half2.u).cwiseAbs().maxCoeff();
    const double budget=p.at("calibration_factor").get<double>()*difference+p.at("uncertainty_factor").get<double>()*uncertainty;
    const bool isResolved=valid&&defect>p.at("resolved_factor").get<double>()*uncertainty,isMiss=isResolved&&defect>budget;
    unknown+=!valid;resolved+=isResolved;miss+=isMiss;if(f.kind=="zero")zeroMax=std::max(zeroMax,defect);
    rows.push_back({{"kind",f.kind},{"h_mm",f.h},{"phase",f.phase},{"peak_index",f.peak},{"q_over_p_Acts",q},
      {"full",save(full)},{"half1",save(half1)},{"half2",save(half2)},{"reference_ladder",ladder},{"coarse_difference",coarse},{"fine_difference",fine},
      {"reference_gate",valid?"PASS":"UNKNOWN"},{"analytic_difference",exact},{"direction_defect",array(full.u-refs.back())},{"defect_max",defect},{"difference_max",difference},
      {"budget",budget},{"resolved",isResolved},{"false_negative",isMiss},{"near_zero_estimate",difference<=p.at("absolute_direction_floor").get<double>()}});
  };
  for(double q:p.at("control_q_over_p_Acts")){
    for(double h:p.at("smooth_control_h_mm"))for(const std::string kind:{"zero","constant"})run({kind,h,0,p.at("mesh_pitch_mm"),0},q);
    for(double h:p.at("control_h_mm")){
      for(double phase:p.at("control_phases"))for(const std::string kind:{"domain","kink"})run({kind,h,phase,p.at("mesh_pitch_mm"),0},q);
      for(double peak:p.at("mesh_triangle_peak_index"))run({"triangle",h,0,p.at("mesh_pitch_mm"),peak},q);
    }
  }
  bool nan=false;try{step(V::Zero(),V(0,0,1),1,.01,[](const V&){return V::Constant(std::numeric_limits<double>::quiet_NaN());});}catch(...){nan=true;}
  const Field smooth{"constant",10,0,5,0};auto right=step(V::Zero(),V(0,0,1),10,.01,[&](const V& x){return smooth.get(x);});
  auto wrongsign=step(V::Zero(),V(0,0,1),10,-.01,[&](const V& x){return smooth.get(x);});
  auto wrongunit=step(V::Zero(),V(0,0,1),10,10,[&](const V& x){return smooth.get(x);});
  const double sign=(right.u-wrongsign.u).norm(),unit=(right.u-wrongunit.u).norm();
  J result={{"schema","wb99_controls_v1"},{"rows",rows},{"reference_unknown",unknown},{"resolved",resolved},{"false_negatives",miss},
    {"analytic_reference_max",worstExact},{"zero_max",zeroMax},{"nan_rejected",nan},{"wrong_sign_difference",sign},{"wrong_unit_difference",unit},
    {"execution_contract",nan&&worstExact<=1e-12&&zeroMax<=1e-12&&sign>1e-8&&unit>1e-8?"PASS":"FAIL"},
    {"calibration",miss?"NOT_SUPPORTED":(unknown?"UNKNOWN":"SUPPORTED_BUT_LIMITED")},{"qualification","NOT_EVALUATED"}};
  const int fd=::open(argv[2],O_CREAT|O_EXCL|O_WRONLY,0600);if(fd<0)throw std::runtime_error("exclusive control output");::close(fd);
  std::ofstream out(argv[2]);if(!out)throw std::runtime_error("control output");out<<result.dump(2)<<std::endl;
  std::cout<<"rows="<<rows.size()<<" misses="<<miss<<" unknown="<<unknown<<" analytic="<<worstExact<<std::endl;return result.at("execution_contract")=="PASS"?0:2;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
