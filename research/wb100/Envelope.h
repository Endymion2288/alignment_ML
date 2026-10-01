#pragma once
#include "../wb99/DirectionDoubling.h"
#include <algorithm>
#include <limits>
namespace WB100 {
using WB99::V; using WB99::J; using WB99::array;
struct Bounds {
  double inside, outside, L;
  V lo,hi;
  bool collinear=false;
  double M()const{return std::max(inside,outside);}
};
inline double inf(const V& x){return x.cwiseAbs().maxCoeff();}
inline V rotate(const V& u,const V& w){
  const double t=w.norm(),t2=t*t;
  const double sinc=t<1e-4?1-t2/6+t2*t2/120:std::sin(t)/t;
  const double cosc=t<1e-4?.5-t2/24+t2*t2/720:2*std::pow(std::sin(t/2)/t,2);
  return u+sinc*w.cross(u)+cosc*w.cross(w.cross(u));
}
inline void root(std::vector<double>& cuts,double a,double b,double c,double h){
  auto add=[&](double x){if(std::isfinite(x)&&x>0&&x<h)cuts.push_back(x);};
  if(a==0){if(b!=0)add(-c/b);return;}
  const long double aa=a,bb=b,cc=c,disc=bb*bb-4*aa*cc;
  if(disc<0)return;
  const long double t=-.5L*(bb+std::copysign(std::sqrt(disc),bb));
  if(t==0){add(double(-bb/(2*aa)));return;}
  add(double(t/aa));add(double(cc/t));
}
struct Tube {double ambiguous=0;J intervals=J::array();bool outside=true;};
inline Tube tube(const V& p,const V& u,double h,double k,const Bounds& b){
  std::vector<double> cuts{0,h};
  for(int axis=0;axis<3;++axis)for(double face:{b.lo[axis],b.hi[axis]})
    if(std::isfinite(face))for(double sign:{-1.,1.})root(cuts,sign*.5*k,u[axis],p[axis]-face,h);
  std::sort(cuts.begin(),cuts.end());cuts.erase(std::unique(cuts.begin(),cuts.end()),cuts.end());
  Tube out;
  for(size_t i=1;i<cuts.size();++i){const double s=cuts[i-1]+.5*(cuts[i]-cuts[i-1]),r=.5*k*s*s;const V c=p+s*u;
    bool in=true,ex=false;
    for(int axis=0;axis<3;++axis){in &= c[axis]>=b.lo[axis]+r && c[axis]<=b.hi[axis]-r;
      ex |= c[axis]<b.lo[axis]-r || c[axis]>b.hi[axis]+r;}
    out.outside &= ex;
    if(!in&&!ex){out.ambiguous+=cuts[i]-cuts[i-1];out.intervals.push_back({cuts[i-1],cuts[i]});}
  }
  return out;
}
inline std::vector<double> faces(const V& p,const V& u,double h,const std::array<std::vector<double>,3>& mesh,J& identities){
  std::vector<double> cuts{0,h};
  for(int axis=0;axis<3;++axis){if(u[axis]==0)continue;const double end=p[axis]+h*u[axis];
    auto it=std::lower_bound(mesh[axis].begin(),mesh[axis].end(),std::min(p[axis],end));
    for(;it!=mesh[axis].end()&&*it<=std::max(p[axis],end);++it){const double s=(*it-p[axis])/u[axis];
      if(s>0&&s<h){cuts.push_back(s);identities.push_back({{"axis",axis},{"face_mm",*it},{"s_mm",s}});}}
  }
  std::sort(cuts.begin(),cuts.end());cuts.erase(std::unique(cuts.begin(),cuts.end()),cuts.end());return cuts;
}
template<class Get> J candidate(const V& p,const V& u,double h,double q,const V& full,
  const std::array<std::vector<double>,3>& mesh,const Bounds& b,Get get){
  if(!p.allFinite()||!u.allFinite()||!full.allFinite()||!std::isfinite(h)||h<=0||!std::isfinite(q)||
    !std::isfinite(b.M())||!std::isfinite(b.L)||b.L<0||b.M()<0||std::abs(u.norm()-1)>1e-12)
    throw std::runtime_error("invalid envelope input");
  J identities=J::array();const auto cuts=faces(p,u,h,mesh,identities);V integral=V::Zero();double length=0;
  for(size_t i=1;i<cuts.size();++i){const double delta=cuts[i]-cuts[i-1],mid=cuts[i-1]+delta/2,off=delta/(2*std::sqrt(3.));
    const V a=get(p+(mid-off)*u),c=get(p+(mid+off)*u);
    if(!a.allFinite()||!c.allFinite())throw std::runtime_error("nonfinite field");
    integral+=delta*.5*(a+c);length+=delta;
  }
  const double k=std::abs(q)*b.M(),angle=k*h;const Tube t=tube(p,u,h,k,b);
  const V v=rotate(u,-q*integral);
  const double comm=(b.collinear||t.outside)?0:angle*angle*std::exp(angle);
  const double lips= t.outside?0:std::abs(q)*b.L*k*h*h*h/6;
  const double jump=t.outside?0:std::abs(q)*(b.inside+b.outside)*t.ambiguous;
  const double E=inf(full-v)+comm+lips+jump,C=inf(full-u)+angle;
  if(!std::isfinite(E)||std::abs(length-h)>1e-12*std::max(1.,h))throw std::runtime_error("envelope numerical contract");
  const double z=std::min({u[2]-angle,full[2],v[2]});
  J slope={{"gate","UNKNOWN"}};
  if(z>1e-6){const double transverse=std::max({std::abs(full[0]),std::abs(full[1]),std::abs(v[0]),std::abs(v[1]),std::abs(u[0])+angle,std::abs(u[1])+angle});
    const double G=1/z+transverse/(z*z);
    slope={{"gate","PASS"},{"G",G},{"E",(full.head<2>()/full[2]-v.head<2>()/v[2]).cwiseAbs().maxCoeff()+G*(comm+lips+jump)}};
  }
  return {{"cuts_mm",cuts},{"face_identities",identities},{"segment_length_sum_mm",length},{"integral_native_mm",array(integral)},
    {"predictor_direction",array(v)},{"kappa_per_mm",k},{"ambiguous_intervals_mm",t.intervals},{"ambiguous_length_mm",t.ambiguous},
    {"proved_constant_outside",t.outside},{"Rcomm",comm},{"Rbend_lipschitz",lips},{"Rbend_jump",jump},
    {"E",E},{"baseline_C",C},{"information_ratio",C>0?J(E/C):J(nullptr)},{"slope",slope}};
}
}
