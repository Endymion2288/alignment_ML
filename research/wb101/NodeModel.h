#pragma once
#include "Envelope.h"
#include "MagFieldElements/FaserFieldMap.h"
#include <fstream>
namespace WB101 {
struct NodeModel {
  using V=Acts::Vector3;using J=nlohmann::json;
  const BFieldZone* zone;
  double scale;
  std::array<std::vector<double>,3> mesh;
  WB100::Bounds bounds;
  J evidence;
  NodeModel(const MagField::FaserFieldMap* map,double conditionScale,const J& old,const J& savedBounds,const std::string& binary){
    const auto lo=old.at("conditions").at("min_mm"),hi=old.at("conditions").at("max_mm");
    zone=map->findBFieldZone(0,0,.5*(lo[2].get<double>()+hi[2].get<double>()));
    if(!zone||zone->id()!=old.at("conditions").at("zone_id")||conditionScale!=old.at("conditions").at("scale")||zone->bscale()!=old.at("conditions").at("bscale_kT"))throw std::runtime_error("node conditions identity");
    scale=conditionScale*zone->bscale()*1000*Acts::UnitConstants::T;
    for(int k=0;k<3;++k){for(unsigned i=0;i<zone->nmesh(k);++i)mesh[k].push_back(zone->mesh(k,i));
      if(J(mesh[k])!=old.at("mesh_mm").at(k)||zone->min(k)!=lo[k]||zone->max(k)!=hi[k])throw std::runtime_error("actual mesh/domain changed");}
    if(zone->nfield()!=old.at("node_count"))throw std::runtime_error("node count changed");
    std::ifstream input(binary,std::ios::binary);if(!input)throw std::runtime_error("frozen node file");
    double M=0;V L=V::Zero();const size_t ny=mesh[1].size(),nz=mesh[2].size();
    auto node=[&](size_t i)->V{return V(zone->field(i)[0],zone->field(i)[1],zone->field(i)[2])*scale;};
    for(size_t x=0;x<mesh[0].size();++x)for(size_t y=0;y<ny;++y)for(size_t z=0;z<nz;++z){
      const size_t i=(x*ny+y)*nz+z;for(int k=0;k<3;++k){unsigned char bytes[2];input.read(reinterpret_cast<char*>(bytes),2);
        if(!input)throw std::runtime_error("node binary truncated");unsigned raw=bytes[0]+256u*bytes[1];const int n=raw<32768?int(raw):int(raw)-65536;
        if(n!=zone->field(i)[k])throw std::runtime_error("actual node payload changed");}
      const V v=node(i);M=std::max(M,v.norm());for(int k=0;k<3;++k){const size_t ix=k==0?x:(k==1?y:z);
        if(ix+1>=mesh[k].size())continue;const size_t stride=k==0?ny*nz:(k==1?nz:1);
        L[k]=std::max(L[k],(node(i+stride)-v).norm()/(mesh[k][ix+1]-mesh[k][ix]));}}
    char extra;if(input.get(extra))throw std::runtime_error("node binary extra bytes");
    bounds={M,V::Constant(1e-5*Acts::UnitConstants::T).norm(),L.norm(),V(mesh[0].front(),mesh[1].front(),mesh[2].front()),V(mesh[0].back(),mesh[1].back(),mesh[2].back()),false};
    if(std::abs(M-savedBounds.at("M_inside_native").get<double>())>1e-16||std::abs(L.norm()-savedBounds.at("L_native_per_mm").get<double>())>1e-16)throw std::runtime_error("global bounds changed");
    double probe=0;size_t probes=0;for(const std::string key:{"probes","domain_controls"})for(const auto& row:old.at(key)){
      V p,b;for(int k=0;k<3;++k){p[k]=row.at("position_mm").at(k);b[k]=row.at("double_T").at(k);}
      probe=std::max(probe,(get(p)/Acts::UnitConstants::T-b).cwiseAbs().maxCoeff());++probes;}
    if(probe>1e-12)throw std::runtime_error("node probes mismatch");
    evidence=J({{"nodes_compared",zone->nfield()},{"M_inside_native",M},{"L_native_per_mm",L.norm()},{"L_axes_native_per_mm",WB100::array(L)},
      {"probe_count",probes},{"probe_max_T",probe},{"actual_collinear",false},{"official_field_replaced",false}});
  }
  V get(const V& p)const{
    if(!p.allFinite())throw std::runtime_error("nonfinite node position");
    for(int k=0;k<3;++k)if(p[k]<bounds.lo[k]||p[k]>bounds.hi[k])return V::Constant(1e-5*Acts::UnitConstants::T);
    std::array<int,3> c;V frac;for(int k=0;k<3;++k){c[k]=std::max(0,std::min(int(mesh[k].size())-2,int(std::lower_bound(mesh[k].begin(),mesh[k].end(),p[k])-mesh[k].begin())-1));
      frac[k]=(p[k]-mesh[k][c[k]])/(mesh[k][c[k]+1]-mesh[k][c[k]]);}
    V b=V::Zero();for(int corner=0;corner<8;++corner){auto ix=c;double weight=1;
      for(int k=0;k<3;++k){bool high=corner&(1<<(2-k));weight*=high?frac[k]:1-frac[k];ix[k]+=high;}
      const auto i=(ix[0]*mesh[1].size()+ix[1])*mesh[2].size()+ix[2];
      for(int k=0;k<3;++k)b[k]+=weight*zone->field(i)[k]*scale;}
    return b;
  }
};
}
