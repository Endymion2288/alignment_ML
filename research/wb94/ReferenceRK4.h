#pragma once
// Independent fixed-z classical RK4; no ACTS stepper or navigator code.
#include <array>
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace WB94Reference {
using State=std::array<double,5>; // x(mm), y(mm), tx, ty, time(ACTS units)
using Field=std::array<double,3>; // field in ACTS native units
inline void finite(const State& s) {
  for(double x:s)if(!std::isfinite(x))throw std::runtime_error("nonfinite RK state");
}
template<class GetField> State rhs(const State& s,double z,double qop,double mass,GetField&& field) {
  finite(s);const auto b=field(s[0],s[1],z);
  for(double x:b)if(!std::isfinite(x))throw std::runtime_error("nonfinite RK field");
  const double uz=1./std::sqrt(1+s[2]*s[2]+s[3]*s[3]);
  if(uz<1e-6)throw std::runtime_error("fixed-z reference not monotonic");
  const double ux=s[2]*uz,uy=s[3]*uz;
  const double cx=uy*b[2]-uz*b[1],cy=uz*b[0]-ux*b[2],cz=ux*b[1]-uy*b[0];
  return {s[2],s[3],qop*(cx*uz-ux*cz)/(uz*uz*uz),
    qop*(cy*uz-uy*cz)/(uz*uz*uz),std::hypot(1.,mass*qop)/uz};
}
template<class GetField> State integrate(State s,double from,double to,double dz,double qop,double mass,GetField&& field) {
  if(!(dz>0) || !std::isfinite(dz) || !std::isfinite(from) || !std::isfinite(to) || to<from || !std::isfinite(qop))
    throw std::runtime_error("invalid RK domain/step");
  double z=from;
  while(z<to) {
    const double h=std::min(dz,to-z);if(z+h==z)throw std::runtime_error("RK step underflow");
    auto shifted=[&](const State& k,double a){State v=s;for(int i=0;i<5;++i)v[i]+=a*k[i];return v;};
    const State k1=rhs(s,z,qop,mass,field),k2=rhs(shifted(k1,h/2),z+h/2,qop,mass,field);
    const State k3=rhs(shifted(k2,h/2),z+h/2,qop,mass,field),k4=rhs(shifted(k3,h),z+h,qop,mass,field);
    for(int i=0;i<5;++i)s[i]+=h*(k1[i]+2*k2[i]+2*k3[i]+k4[i])/6.;
    finite(s);z=z+h;
  }
  return s;
}
inline std::array<double,3> controls(double tesla) {
  auto zero=[](double,double,double){return Field{0,0,0};};
  const State straight=integrate(State{47,74,.01,.002,0},0,1000,.5,.01,0,zero);
  double worst=std::max(std::abs(straight[0]-57),std::abs(straight[1]-76));
  auto by=[&](double,double,double){return Field{0,tesla,0};};
  auto exact=[&](double z){const double w=.01*tesla,c=std::sqrt(1-w*w*z*z);
    return State{-w*z*z/(1+c),0,-w*z/c,0,std::asin(w*z)/w};};
  for(double dz:{.5,.25,.125}) {
    const auto result=integrate(State{0,0,0,0,0},0,1000,dz,.01,0,by),truth=exact(1000);
    for(int i=0;i<4;++i)worst=std::max(worst,std::abs(result[i]-truth[i])/(i<2?1.:.001));
  }
  const auto truth=exact(10);
  const auto wrongSign=integrate(State{0,0,0,0,0},0,10,.125,-.01,0,by);
  const auto wrongUnit=integrate(State{0,0,0,0,0},0,10,.125,10.,0,by);
  return {worst,std::abs(wrongSign[0]-truth[0]),std::abs(wrongUnit[0]-truth[0])};
}
}
