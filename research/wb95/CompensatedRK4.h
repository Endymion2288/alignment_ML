#pragma once
#include "ReferenceRK4.h"
// Same classical fixed-z RK4; compensated accumulation in the two matched mesh modes.
namespace WB95Reference {
template<class GetField> WB94Reference::State integrate(WB94Reference::State s,double from,double to,double dz,
    double qop,double mass,GetField&& field) {
  if(!(dz>0) || !std::isfinite(dz) || !std::isfinite(from) || !std::isfinite(to) || to<from || !std::isfinite(qop))
    throw std::runtime_error("invalid compensated RK domain/step");
  WB94Reference::State correction{};double z=from;
  const size_t count=static_cast<size_t>(std::ceil((to-from)/dz));
  for(size_t n=0;n<count;++n) {
    const double next=n+1==count?to:std::min(to,from+(n+1)*dz),h=next-z;
    if(h<=0)throw std::runtime_error("compensated RK step underflow");
    auto shifted=[&](const WB94Reference::State& k,double a){auto v=s;for(int i=0;i<5;++i)v[i]+=a*k[i];return v;};
    const auto k1=WB94Reference::rhs(s,z,qop,mass,field);
    const auto k2=WB94Reference::rhs(shifted(k1,h/2),z+h/2,qop,mass,field);
    const auto k3=WB94Reference::rhs(shifted(k2,h/2),z+h/2,qop,mass,field);
    const auto k4=WB94Reference::rhs(shifted(k3,h),next,qop,mass,field);
    for(int i=0;i<5;++i) {
      const double increment=h*(k1[i]+2*k2[i]+2*k3[i]+k4[i])/6.-correction[i];
      const double updated=s[i]+increment;correction[i]=(updated-s[i])-increment;s[i]=updated;
    }
    WB94Reference::finite(s);z=next;
  }
  return s;
}
}
