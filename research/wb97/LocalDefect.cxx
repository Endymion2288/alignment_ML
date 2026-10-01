// Saved-trace diagnostic only. No Calypso event execution or production update.
#include "Acts/EventData/TrackParameters.hpp"
#include "Acts/EventData/detail/TransformationFreeToBound.hpp"
#include "Acts/Surfaces/PlaneSurface.hpp"
#include "Acts/Definitions/Units.hpp"
#include "CompensatedRK4.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <set>
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
using J=nlohmann::json;
using V=Acts::Vector3;
using S=WB94Reference::State;
J read(const std::string& p){std::ifstream f(p);J j;f>>j;if(!f)throw std::runtime_error("read "+p);return j;}
template<int N> Eigen::Matrix<double,N,1> vec(const J& j){Eigen::Matrix<double,N,1> v;
  for(int k=0;k<N;++k)v[k]=j.at(k).is_array()?j.at(k).at(0).get<double>():j.at(k).get<double>();
  if(!v.allFinite())throw std::runtime_error("nonfinite input");return v;}
template<class T> J encode(const T& v){J a=J::array();for(int k=0;k<v.size();++k){if(!std::isfinite(v[k]))throw std::runtime_error("nonfinite output");a.push_back(v[k]);}return a;}
void exclusive(const std::string& p){int fd=::open(p.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);if(fd<0)throw std::runtime_error("exclusive output "+p);::close(fd);}
struct Footprint {
  std::set<bool> domain;std::array<std::set<int>,3> cells;
  void add(const Footprint& f){domain.insert(f.domain.begin(),f.domain.end());for(int k=0;k<3;++k)cells[k].insert(f.cells[k].begin(),f.cells[k].end());}
  bool crossing()const{if(domain.size()>1)return true;for(const auto& c:cells)if(c.size()>1)return true;return false;}
  J json()const{J axes=J::array();for(int k=0;k<3;++k)axes.push_back(cells[k].size()>1);
    return {{"class",domain.size()>1?"domain":(crossing()?"mesh":(domain.count(true)?"interior":"outside"))},
      {"axis_changes",axes},{"domain",domain},{"cells",cells}};}
};
struct Field {
  std::array<std::vector<double>,3> mesh;std::vector<int16_t> nodes;double scale=1e-4*Acts::UnitConstants::T;
  bool constant=false;V value=V::Zero();
  bool inside(const V& p)const{for(int k=0;k<3;++k)if(p[k]<mesh[k].front()||p[k]>mesh[k].back())return false;return true;}
  std::array<int,3> cell(const V& p)const{std::array<int,3> c;for(int k=0;k<3;++k)c[k]=std::max(0,std::min(int(mesh[k].size())-2,
    int(std::lower_bound(mesh[k].begin(),mesh[k].end(),p[k])-mesh[k].begin())-1));return c;}
  void observe(const V& p,Footprint& f)const{if(constant)return;const bool in=inside(p);f.domain.insert(in);if(in){const auto c=cell(p);for(int k=0;k<3;++k)f.cells[k].insert(c[k]);}}
  V get(const V& p)const{
    if(!p.allFinite())throw std::runtime_error("nonfinite field position");if(constant)return value;
    if(!inside(p))return V::Constant(1e-5*Acts::UnitConstants::T);
    const auto c=cell(p);V frac;for(int k=0;k<3;++k)frac[k]=(p[k]-mesh[k][c[k]])/(mesh[k][c[k]+1]-mesh[k][c[k]]);
    V out=V::Zero();for(int corner=0;corner<8;++corner){double w=1;std::array<int,3> ix=c;
      for(int k=0;k<3;++k){bool high=corner&(1<<(2-k));w*=high?frac[k]:1-frac[k];ix[k]+=high;}
      const size_t offset=3*((ix[0]*mesh[1].size()+ix[1])*mesh[2].size()+ix[2]);
      for(int k=0;k<3;++k)out[k]+=w*nodes.at(offset+k)*scale;
    }return out;
  }
};
struct Rkn {V p,u,middle,last;double estimate;};
Rkn rkn(const V& p,const V& u,double h,double q,const V& b0,const V& bm,const V& b1){
  if(!p.allFinite()||!u.allFinite()||!b0.allFinite()||!bm.allFinite()||!b1.allFinite()||!std::isfinite(h)||!std::isfinite(q))throw std::runtime_error("nonfinite RKN");
  const V k1=q*u.cross(b0),k2=q*(u+.5*h*k1).cross(bm),k3=q*(u+.5*h*k2).cross(bm),k4=q*(u+h*k3).cross(b1);
  return {p+h*u+h*h/6.*(k1+k2+k3),(u+h/6.*(k1+2.*(k2+k3)+k4)).normalized(),
    p+.5*h*u+h*h*.125*k1,p+h*u+h*h*.5*k3,std::max(h*h*(k1-k2-k3+k4).lpNorm<1>(),1e-20)};
}
struct Ref {V p,u;double arc_residual;Footprint footprint;};
std::pair<S,Footprint> integrate(const Field& f,const V& p,const V& u,double q,double z,double dz,double eps){
  S s{p[0],p[1],u[0]/u[2],u[1]/u[2],0};double current=p[2];Footprint fp;
  std::vector<double> cuts;if(!f.constant)for(double edge:f.mesh[2])if(edge>current && edge<z)cuts.push_back(edge);cuts.push_back(z);
  for(double to:cuts){const double mid=.5*(current+to);
    auto get=[&](double x,double y,double zz){if(!f.constant)for(double edge:{f.mesh[2].front(),f.mesh[2].back()})
      if(std::abs(zz-edge)<eps/2)zz=edge+(mid<edge?-eps:eps);
      V v(x,y,zz);f.observe(v,fp);const V b=f.get(v);return WB94Reference::Field{b[0],b[1],b[2]};};
    s=WB95Reference::integrate(s,current,to,dz,q,0.,get);current=to;
  }return {s,fp};
}
Ref reference(const Field& f,const V& p,const V& u,double q,double h,double endz,double dz,const J& protocol){
  if(!(u[2]>protocol.at("minimum_forward_uz").get<double>()) || !(endz-p[2]>protocol.at("minimum_forward_dz_mm").get<double>()))throw std::runtime_error("tiny/nonforward step");
  double z=endz;S s{};Footprint fp;double residual=0;
  for(int n=0;n<protocol.at("arc_root_iterations").get<int>();++n){
    if(!(z>p[2]))throw std::runtime_error("nonforward root");
    auto pair=integrate(f,p,u,q,z,dz,protocol.at("boundary_side_epsilon_mm"));s=pair.first;fp=pair.second;
    residual=s[4]-h;if(std::abs(residual)<=protocol.at("arc_root_residual_mm").get<double>())break;
    z-=residual/std::sqrt(1+s[2]*s[2]+s[3]*s[3]);
  }
  if(std::abs(residual)>protocol.at("arc_root_residual_mm").get<double>())throw std::runtime_error("arc root not closed");
  Ref result{V(s[0],s[1],z),V(s[2],s[3],1).normalized(),residual,fp};return result;
}
double scaled(const Ref& a,const Ref& b,double uscale){return std::max((a.p-b.p).cwiseAbs().maxCoeff(),(a.u-b.u).cwiseAbs().maxCoeff()/uscale);}
Acts::BoundTrackParameters initial(const J& raw,const J& setting,const J& target,const J& call,const std::string& path){
  Acts::GeometryContext g;double z=raw.at("seed_z_mm");
  if(path=="fixed_start"){
    z=vec<3>(target.at("fixed_start").at("position_mm"))[2];
    auto plane=Acts::Surface::makeShared<Acts::PlaneSurface>(V(0,0,z),V(0,0,1));
    return Acts::BoundTrackParameters(plane,vec<6>(target.at("fixed_start").at("bound_parameters")),std::nullopt,Acts::ParticleHypothesis::muon());
  }
  const auto seed=vec<5>(path=="entry"?raw.at("seed"):call.at("seed"));
  auto plane=Acts::Surface::makeShared<Acts::PlaneSurface>(V(0,0,z),V(0,0,1));const V u=V(seed[2],seed[3],1).normalized();
  auto pars=Acts::detail::transformFreeToBoundParameters(V(seed[0],seed[1],z),0.,u,seed[4]/Acts::UnitConstants::MeV,*plane,g);
  if(!pars.ok())throw std::runtime_error("initial free-to-bound failed");return Acts::BoundTrackParameters(plane,*pars,std::nullopt,Acts::ParticleHypothesis::muon());
}
J controls(const J& protocol){
  J rows=J::array();double calibration=1,refworst=0,zeroworst=0;bool nan=false;double wrongsign=0,wrongunit=0;
  for(double q:{.01,.0100025,.01000125})for(double h:protocol.at("smooth_control_h_mm"))for(bool zero:{true,false}){
    const V b=zero?V::Zero().eval():V(0,Acts::UnitConstants::T,0);const double w=q*b[1];
    const V p=V::Zero(),u(0,0,1);const auto got=rkn(p,u,h,q,b,b,b);
    const V xp=zero?V(0,0,h):V(-2*std::pow(std::sin(w*h/2),2)/w,0,std::sin(w*h)/w);
    const V xu=zero?u:V(-std::sin(w*h),0,std::cos(w*h));
    const double defect=(got.p-xp).lpNorm<1>();calibration=std::max(calibration,defect/got.estimate);
    Field f;f.constant=true;f.value=b;double rw=0;for(double dz:protocol.at("reference_dz_mm")){
      const auto ref=reference(f,p,u,q,h,got.p[2],dz,protocol);rw=std::max(rw,std::max((ref.p-xp).cwiseAbs().maxCoeff(),(ref.u-xu).cwiseAbs().maxCoeff()/protocol.at("direction_scale").get<double>()));}
    refworst=std::max(refworst,rw);if(zero)zeroworst=std::max(zeroworst,defect);
    rows.push_back({{"q_over_p_Acts",q},{"h_mm",h},{"zero",zero},{"position_L1_mm",defect},{"direction_max",(got.u-xu).cwiseAbs().maxCoeff()},
      {"estimate_mm",got.estimate},{"reference_scaled_error",rw}});
  }
  try{rkn(V::Constant(std::numeric_limits<double>::quiet_NaN()),V(0,0,1),1,.01,V::Zero(),V::Zero(),V::Zero());}catch(...){nan=true;}
  const V b(0,Acts::UnitConstants::T,0),u(0,0,1);const auto right=rkn(V::Zero(),u,10,.01,b,b,b);
  wrongsign=(right.p-rkn(V::Zero(),u,10,-.01,b,b,b).p).norm();wrongunit=(right.p-rkn(V::Zero(),u,10,10,b,b,b).p).norm();
  bool tiny=false;try{Field f;f.constant=true;reference(f,V::Zero(),u,.01,1e-12,1e-12,.125,protocol);}catch(...){tiny=true;}
  Footprint fi;fi.domain.insert(true);for(auto& c:fi.cells)c.insert(0);const bool interior=!fi.crossing();fi.cells[0].insert(1);const bool mesh=fi.crossing();fi.domain.insert(false);const bool domain=fi.json()["class"]=="domain";
  return {{"rows",rows},{"calibration_C",calibration},{"reference_scaled_error",refworst},{"zero_position_error_mm",zeroworst},
    {"nan_rejected",nan},{"tiny_rejected",tiny},{"classification_test",interior&&mesh&&domain},{"wrong_sign_mm",wrongsign},{"wrong_unit_mm",wrongunit},
    {"gate",refworst<=1e-8&&zeroworst<=1e-12&&nan&&tiny&&interior&&mesh&&domain&&wrongsign>1e-8&&wrongunit>1e-8?"PASS":"FAIL"}};
}
int main(int argc,char** argv){try{
  if(argc==3 && std::string(argv[1])=="--self-test"){std::cout<<controls(read(argv[2])).dump(2)<<std::endl;return 0;}
  if(argc!=6)throw std::runtime_error("usage: local protocol wb96raw wb95raw nodes output.ndjson");
  const J protocol=read(argv[1]),raw=read(argv[2]),old=read(argv[3]);
  if(raw.at("actual_run")!=protocol.at("actual_run")||raw.at("actual_event")!=protocol.at("actual_event"))throw std::runtime_error("pilot identity mismatch");
  Field field;for(int k=0;k<3;++k)field.mesh[k]=old.at("mesh_mm").at(k).get<std::vector<double>>();
  std::ifstream nodes(argv[4],std::ios::binary);std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(nodes)),{});
  if(bytes.size()!=6*81*81*861)throw std::runtime_error("node count");field.nodes.resize(bytes.size()/2);
  for(size_t k=0;k<field.nodes.size();++k){const unsigned v=bytes[2*k]+256u*bytes[2*k+1];field.nodes[k]=v<32768?int(v):int(v)-65536;}
  field.scale=old.at("conditions").at("bscale_kT").get<double>()*old.at("conditions").at("scale").get<double>()*1000*Acts::UnitConstants::T;
  double probeMax=0;size_t probes=0;for(const std::string key:{"probes","domain_controls"})for(const auto& probe:old.at(key)){
    const V p=vec<3>(probe.at("position_mm"));probeMax=std::max(probeMax,(field.get(p)/Acts::UnitConstants::T-vec<3>(probe.at("double_T"))).cwiseAbs().maxCoeff());++probes;}
  const J control=controls(protocol);exclusive(argv[5]);std::ofstream stream(argv[5]);
  stream<<J{{"record","controls"},{"controls",control},{"probe_count",probes},{"probe_max_T",probeMax}}.dump()<<'\n';
  if(control.at("gate")!="PASS"||probeMax>protocol.at("field_probe_T_tolerance").get<double>())throw std::runtime_error("control/probe contract failed");
  size_t traces=0,total=0;Acts::GeometryContext g;
  auto analyze=[&](const J& setting,const J& target,const J& call,const std::string& path,int station,int sample){
    if(call.at("accepted_trace").empty())return;++traces;
    const auto start=initial(raw,setting,target,call,path);V p=start.position(g),u=start.direction();const double q=start.parameters()[Acts::eBoundQOverP];
    size_t i=0;for(const auto& step:call.at("accepted_trace")){
      J row={{"record","step"},{"trace",traces-1},{"index",i},{"tolerance",setting.at("tolerance")},{"cap_m",setting.at("cap_m")},
        {"path",path},{"station",station},{"sample",sample},{"start_position_mm",encode(p)},{"start_direction",encode(u)},
        {"q_over_p_Acts",q},{"saved_step",step},{"closure_gate","UNKNOWN"},{"reference_gate","UNKNOWN"}};
      const V endp=vec<3>(step.at("position_mm")),endu=vec<3>(step.at("direction"));
      try{
        const size_t begin=step.at("query_begin"),end=step.at("query_end"),startup=i==0?int(call.at("options").at("loopProtection").get<bool>()):0;
        const size_t expected=3+2*step.at("rejected_trials").get<size_t>()+startup;
        if(end-begin!=expected)throw std::runtime_error("query/trial indexing");
        const J queries=J::array({call.at("field_queries").at(begin+startup),call.at("field_queries").at(end-2),call.at("field_queries").at(end-1)});
        row["accepted_queries"]=queries;
        if(startup && call.at("field_queries").at(0)!=call.at("field_queries").at(1))throw std::runtime_error("startup identity");
        const double h=step.at("h_mm"),estimate=step.at("accepted_error_estimate");
        const auto calc=rkn(p,u,h,q,vec<3>(queries[0]["field_T"])*Acts::UnitConstants::T,vec<3>(queries[1]["field_T"])*Acts::UnitConstants::T,vec<3>(queries[2]["field_T"])*Acts::UnitConstants::T);
        const V closure=calc.p-endp;const double cp=closure.cwiseAbs().maxCoeff(),cu=(calc.u-endu).cwiseAbs().maxCoeff();
        const double stage=std::max({(p-vec<3>(queries[0]["position_mm"])).cwiseAbs().maxCoeff(),
          (calc.middle-vec<3>(queries[1]["position_mm"])).cwiseAbs().maxCoeff(),(calc.last-vec<3>(queries[2]["position_mm"])).cwiseAbs().maxCoeff()});
        row["closure"]={{"position_max_mm",cp},{"position_L1_mm",closure.lpNorm<1>()},{"direction_max",cu},{"stage_max_mm",stage},{"estimate_difference_mm",std::abs(calc.estimate-estimate)}};
        if(cp>protocol["closure_position_mm"].get<double>()||stage>protocol["closure_position_mm"].get<double>()||cu>protocol["closure_direction"].get<double>()||
          std::abs(calc.estimate-estimate)>std::max(protocol["closure_estimate_absolute_mm"].get<double>(),protocol["closure_estimate_relative"].get<double>()*estimate))throw std::runtime_error("source-defined closure failed");
        row["closure_gate"]="PASS";std::vector<Ref> refs;J ladder=J::array();Footprint footprint;
        field.observe(p,footprint);field.observe(endp,footprint);for(const auto& query:queries)field.observe(vec<3>(query["position_mm"]),footprint);
        for(double dz:protocol.at("reference_dz_mm")){
          auto ref=reference(field,p,u,q,h,endp[2],dz,protocol);footprint.add(ref.footprint);
          ladder.push_back({{"dz_mm",dz},{"position_mm",encode(ref.p)},{"direction",encode(ref.u)},{"arc_residual_mm",ref.arc_residual}});refs.push_back(ref);}
        const double coarse=scaled(refs[0],refs[1],protocol["direction_scale"]),fine=scaled(refs[1],refs[2],protocol["direction_scale"]);
        const bool pass=fine<=protocol["reference_scaled_tolerance"].get<double>() && (fine<=protocol["reference_contraction_ratio"].get<double>()*coarse ||
          (coarse<=protocol["reference_roundoff_scaled"].get<double>()&&fine<=protocol["reference_roundoff_scaled"].get<double>()));
        row["reference_ladder"]=ladder;row["reference_gate"]=pass?"PASS":"UNKNOWN";row["reference_coarse_scaled"]=coarse;row["reference_fine_scaled"]=fine;
        const V dp=endp-refs[2].p,du=endu-refs[2].u;const double uncertainty=(refs[2].p-refs[1].p).lpNorm<1>();
        const double threshold=std::max({protocol["significance_factor"].get<double>()*uncertainty,
          protocol["significance_factor"].get<double>()*closure.lpNorm<1>(),protocol["significance_factor"].get<double>()*control["calibration_C"].get<double>()*estimate,
          protocol["significance_absolute_position_mm"].get<double>()});
        row["position_defect_mm"]=encode(dp);row["position_L1_mm"]=dp.lpNorm<1>();row["direction_defect"]=encode(du);
        row["direction_max"]=du.cwiseAbs().maxCoeff();row["reference_position_uncertainty_L1_mm"]=uncertainty;
        row["significance_threshold_mm"]=threshold;row["significant_underestimate"]=pass&&dp.lpNorm<1>()>threshold;row["classification"]=footprint.json();
      }catch(const std::exception& e){row["unknown_reason"]=e.what();}
      stream<<row.dump()<<'\n';if(!stream)throw std::runtime_error("step write failed");++total;++i;p=endp;u=endu;
    }
    stream.flush();std::cout<<"trace "<<traces<<" steps "<<i<<std::endl;
  };
  for(const auto& setting:raw.at("settings")){
    analyze(setting,J::object(),setting.at("entry_nominal"),"entry",0,0);
    for(const auto& target:setting.at("targets")){
      for(int sample:protocol.at("trace_sample_indices"))analyze(setting,target,target.at("samples").at(sample),"direct",target.at("station"),sample);
      analyze(setting,target,target.at("fixed_reference_start_nominal"),"fixed_start",target.at("station"),0);
    }
  }
  stream<<J{{"record","terminal"},{"traces",traces},{"steps",total},{"qualification","NOT_EVALUATED"}}.dump()<<'\n';
  return traces==protocol.at("expected_traces").get<size_t>()?0:2;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
