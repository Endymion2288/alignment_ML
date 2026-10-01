#pragma once
#include "Acts/Definitions/Algebra.hpp"
#include <nlohmann/json.hpp>
#include <cmath>
#include <functional>
#include <stdexcept>
namespace WB99 {
using V=Acts::Vector3;
using J=nlohmann::json;
template<class T> J array(const T& v){if(!v.allFinite())throw std::runtime_error("nonfinite WB99 vector");
  J a=J::array();for(int k=0;k<v.size();++k)a.push_back(v[k]);return a;}
struct Step {V p,u;J queries;};
template<class Get> Step step(const V& p,const V& u,double h,double q,Get get){
  if(!p.allFinite()||!u.allFinite()||!std::isfinite(h)||!std::isfinite(q)||h<=0)throw std::runtime_error("invalid WB99 step");
  J queries=J::array();auto field=[&](const V& x){V b=get(x);if(!b.allFinite())throw std::runtime_error("nonfinite WB99 field");
    queries.push_back({{"position_mm",array(x)},{"field_native",array(b)}});return b;};
  const V b0=field(p),k1=q*u.cross(b0),pm=p+.5*h*u+.125*h*h*k1,bm=field(pm);
  const V k2=q*(u+.5*h*k1).cross(bm),k3=q*(u+.5*h*k2).cross(bm),pe=p+h*u+.5*h*h*k3,b1=field(pe),k4=q*(u+h*k3).cross(b1);
  return {p+h*u+h*h*(k1+k2+k3)/6.,(u+h*(k1+2.*k2+2.*k3+k4)/6.).normalized(),queries};
}
inline J save(const Step& s){return {{"position_mm",array(s.p)},{"direction",array(s.u)},{"queries",s.queries}};}
}
