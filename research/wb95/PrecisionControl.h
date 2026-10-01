#pragma once
#include <array>
#include <cmath>
#include <stdexcept>
namespace WB95Math {
using V=std::array<double,3>;
using Corners=std::array<V,8>;
inline V trilinear(const Corners& nodes,const V& fraction,double scale) {
  V out{0,0,0};
  for(int k=0;k<3;++k)if(!std::isfinite(fraction[k]) || fraction[k]<-1e-10 || fraction[k]>1+1e-10)
    throw std::runtime_error("invalid cell fraction");
  if(!std::isfinite(scale))throw std::runtime_error("nonfinite field scale");
  for(int corner=0;corner<8;++corner) {
    double w=1.;for(int axis=0;axis<3;++axis)w*=corner&(1<<(2-axis))?fraction[axis]:1-fraction[axis];
    for(int component=0;component<3;++component) {
      if(!std::isfinite(nodes[corner][component]))throw std::runtime_error("nonfinite node");
      out[component]+=w*nodes[corner][component]*scale;
    }
  }
  return out;
}
}
