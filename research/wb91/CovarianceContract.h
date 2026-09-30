// WB91 diagnostic contract; does not change the production Calypso package.
#pragma once
#include "TrkParameters/TrackParameters.h"
#include "TrkSurfaces/Surface.h"
#include <cmath>
#include <stdexcept>

namespace WB91 {
using V4 = Eigen::Matrix<double,4,1>;
using M4 = Eigen::Matrix<double,4,4>;
using M5 = Eigen::Matrix<double,5,5>;
using J54 = Eigen::Matrix<double,5,4>;
using J45 = Eigen::Matrix<double,4,5>;

inline J54 nativeJacobian(const V4& fit, double dz,
                         const Trk::TrackParameters& nominal) {
  const double tx=fit[2], ty=fit[3], r2=tx*tx+ty*ty, r=std::sqrt(r2);
  if (!(r2>0.) || !fit.allFinite() || !std::isfinite(dz))
    throw std::runtime_error("WB91 undefined slope chart");
  J54 j=J54::Zero();
  const auto basis=nominal.associatedSurface().transform().linear();
  for (int a=0;a<2;++a) {
    j(a,0)=basis(0,a); j(a,1)=basis(1,a);
    j(a,2)=dz*basis(0,a); j(a,3)=dz*basis(1,a);
  }
  j(2,2)=-ty/r2; j(2,3)=tx/r2;
  j(3,2)=tx/(r*(1.+r2)); j(3,3)=ty/(r*(1.+r2));
  return j;
}

inline M5 nativeCovariance(const V4& fit, const M4& covariance, double dz,
                           const Trk::TrackParameters& nominal) {
  const auto j=nativeJacobian(fit,dz,nominal);
  M5 c=j*covariance*j.transpose();
  // Preserve the existing dummy momentum and variance, in MeV units.
  c(4,4)=50000.*nominal.parameters()[4]*nominal.parameters()[4];
  return c;
}

inline V4 fixedZState(const Trk::TrackParameters& p, double zRef) {
  const auto& m=p.momentum();
  if (!m.allFinite() || m.z()==0.) throw std::runtime_error("WB91 invalid pz");
  const double tx=m.x()/m.z(), ty=m.y()/m.z();
  const auto& x=p.position();
  V4 out; out << x.x()+(zRef-x.z())*tx, x.y()+(zRef-x.z())*ty, tx, ty;
  return out;
}

inline J45 exportJacobian(const Trk::TrackParameters& p) {
  // Linearized at the nominal position, with the reference z held fixed.
  const auto f=fixedZState(p,p.position().z());
  const auto b=p.associatedSurface().transform().linear();
  const double phi=p.parameters()[2],theta=p.parameters()[3];
  J45 j=J45::Zero();
  for (int a=0;a<2;++a) {
    j(0,a)=b(0,a)-f[2]*b(2,a);
    j(1,a)=b(1,a)-f[3]*b(2,a);
  }
  j(2,2)=-std::sin(phi)*std::tan(theta);
  j(3,2)= std::cos(phi)*std::tan(theta);
  j(2,3)=std::cos(phi)/(std::cos(theta)*std::cos(theta));
  j(3,3)=std::sin(phi)/(std::cos(theta)*std::cos(theta));
  return j;
}

inline J45 exportFD(const Trk::TrackParameters& p, double fraction=1.) {
  const double steps[5]={1.e-3,1.e-3,1.e-7,1.e-7,1.e-9};
  J45 j;
  for(int a=0;a<5;++a) {
    auto plus=p.parameters().eval(),minus=plus;
    const double h=steps[a]*fraction;
    plus[a]+=h; minus[a]-=h;
    const auto& s=p.associatedSurface();
    auto pp=s.createUniqueTrackParameters(plus[0],plus[1],plus[2],plus[3],plus[4]);
    auto pm=s.createUniqueTrackParameters(minus[0],minus[1],minus[2],minus[3],minus[4]);
    if(!pp || !pm) throw std::runtime_error("WB91 surface factory failed");
    j.col(a)=(fixedZState(*pp,p.position().z())-fixedZState(*pm,p.position().z()))/(2.*h);
  }
  return j;
}
}
