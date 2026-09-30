#pragma once
#include "CovarianceContract.h"
#include <ostream>
#include <iomanip>

namespace WB91 {
template<class M> void jsonMatrix(std::ostream& out,const M& m) {
  out << '[';
  for(int i=0;i<m.rows();++i) {
    if(i) out << ',';
    out << '[';
    for(int j=0;j<m.cols();++j) {if(j) out << ',';out << m(i,j);}
    out << ']';
  }
  out << ']';
}
inline void dumpState(std::ostream& out,unsigned run,unsigned long long event,int station,
                      const std::string& clusters,int stateIndex,const V4& fit,const M4& cov,
                      const M4& normal,double zCenter,const Trk::TrackParameters& p) {
  const double dz=p.position().z()-zCenter;
  const auto a=nativeJacobian(fit,dz,p);
  const auto b=exportJacobian(p);
  const auto fd=exportFD(p);
  out << std::setprecision(17) << "{\"run\":" << run << ",\"event\":" << event
      << ",\"station\":" << station << ",\"clusters\":" << clusters
      << ",\"state_index\":" << stateIndex << ",\"z_center_mm\":" << zCenter
      << ",\"z_state_mm\":" << p.position().z() << ",\"raw_fit\":";
  jsonMatrix(out,fit);out << ",\"raw_covariance\":";jsonMatrix(out,cov);
  out << ",\"raw_normal\":";jsonMatrix(out,normal);
  out << ",\"native_covariance\":";jsonMatrix(out,*p.covariance());
  out << ",\"native_parameters\":";jsonMatrix(out,p.parameters());
  out << ",\"surface_transform\":";jsonMatrix(out,p.associatedSurface().transform().matrix());
  out << ",\"raw_to_native_jacobian\":";jsonMatrix(out,a);
  out << ",\"native_to_fixed_z_jacobian\":";jsonMatrix(out,b);
  out << ",\"native_to_fixed_z_fd\":";jsonMatrix(out,fd);
  out << ",\"fixed_z_state\":";jsonMatrix(out,fixedZState(p,p.position().z()));
  out << ",\"fixed_z_covariance\":";jsonMatrix(out,b*(*p.covariance())*b.transpose());
  out << ",\"input_event_header_verified\":true}\n";
  if(!out) throw std::runtime_error("WB91 audit write failed");
}
}
