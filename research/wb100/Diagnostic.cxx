// No Athena, wrapper, reconstruction, or mutable production component.
#define main wb97_immutable_unused_main
#include "../wb97/LocalDefect.cxx"
#undef main
#include "Envelope.h"
#include "ControlReference.inc"
using WB100::Bounds;
using WB100::inf;
J evaluate(const J& e,const V& full,const V& ref,const V& prev,double U,const J& p,bool valid){
  const double d=inf(full-ref),budget=e.at("E").get<double>()+p.at("uncertainty_factor").get<double>()*U;
  const bool resolved=valid&&d>p.at("resolved_factor").get<double>()*U;
  const double z=std::min({full[2],ref[2],prev[2]});
  J slope={{"gate","UNKNOWN"}};
  if(z>p.at("minimum_forward_uz").get<double>()&&e["slope"]["gate"]=="PASS"){
    const double upper=std::max({full.head<2>().cwiseAbs().maxCoeff(),ref.head<2>().cwiseAbs().maxCoeff(),prev.head<2>().cwiseAbs().maxCoeff()});
    const double G=1/z+upper/(z*z),ds=(full.head<2>()/full[2]-ref.head<2>()/ref[2]).cwiseAbs().maxCoeff();
    const double sb=e["slope"]["E"].get<double>()+p["uncertainty_factor"].get<double>()*U*G;
    const bool sr=valid&&ds>p["resolved_factor"].get<double>()*U*G;
    slope={{"gate",valid?"PASS":"UNKNOWN"},{"defect",ds},{"uncertainty",U*G},{"budget",sb},{"resolved",sr},{"false_negative",sr&&ds>sb}};
  }
  return {{"gate",valid?"PASS":"UNKNOWN"},{"defect",d},{"uncertainty",U},{"budget",budget},{"resolved",resolved},{"false_negative",resolved&&d>budget},{"slope",slope}};
}
struct Synthetic {
  ControlReference::Field f;
  std::array<std::vector<double>,3> mesh;
  Bounds bounds;
  Synthetic(ControlReference::Field ff):f(ff){
    mesh[2]=f.cuts();std::sort(mesh[2].begin(),mesh[2].end());
    const double infinity=std::numeric_limits<double>::infinity(),T=Acts::UnitConstants::T;
    bounds={f.kind=="zero"?0:T,0,(f.kind=="triangle"||f.kind=="kink")?T/f.pitch:0,V::Constant(-infinity),V::Constant(infinity),true};
    if(f.kind=="domain")bounds.lo[2]=f.phase*f.h;
  }
  J predict(double q,const V& full)const{return WB100::candidate(V::Zero(),V(0,0,1),f.h,q,full,mesh,bounds,[&](const V& x){return f.get(x);});}
};
J selftest(){
  auto check=[](bool ok,const char* s){if(!ok)throw std::runtime_error(s);};
  const V u(0,0,1);double worst=0;
  for(const std::string kind:{"zero","constant","domain","kink","triangle"}){
    Synthetic f({kind,100,.1,5,2});const J e=f.predict(.01,u);
    const double exact=kind=="zero"?0:(kind=="constant"?100:(kind=="domain"?90:(kind=="kink"?87.5:5)));
    worst=std::max(worst,std::abs(e["integral_native_mm"][1].get<double>()-exact*Acts::UnitConstants::T));
  }
  check(worst<1e-15,"analytic chord integral");
  const V rot=WB100::rotate(u,V(0,-.01*Acts::UnitConstants::T*10,0));
  check(rot[0]<0&&inf(rot-V(-std::sin(.01*Acts::UnitConstants::T*10),0,std::cos(.01*Acts::UnitConstants::T*10)))<1e-15,"rotation sign");
  const double sign=(rot-WB100::rotate(u,V(0,.01*Acts::UnitConstants::T*10,0))).norm();
  const double unit=(rot-WB100::rotate(u,V(0,-10*Acts::UnitConstants::T*10,0))).norm();
  check(sign>1e-8&&unit>1e-8,"sign unit negative controls");
  // Multi-axis chord across trilinear xyz; Gauss integrates the cubic exactly.
  std::array<std::vector<double>,3> mesh{{{0,1,2},{0,1,2},{0,1,2}}};
  Bounds b{10,0,10,V::Zero(),V::Constant(2),false};const V diagonal=V::Ones().normalized();
  const auto poly=WB100::candidate(V::Zero(),diagonal,1.5*std::sqrt(3.),0,diagonal,mesh,b,[](const V& x){return V(x[0]*x[1]*x[2],0,0);});
  check(std::abs(poly["integral_native_mm"][0].get<double>()-std::pow(1.5,4)*std::sqrt(3.)/4)<1e-13&&poly["face_identities"].size()==3,"xyz cubic and simultaneous faces");
  Synthetic triangle({"triangle",100,0,5,2});
  const double segmented=triangle.predict(.01,u)["integral_native_mm"][1];
  double unsplit=0;for(double s:{50-50/std::sqrt(3.),50+50/std::sqrt(3.)})unsplit+=50*triangle.f.get(V(0,0,s))[1];
  check(segmented>0&&unsplit==0,"omitted face negative control");
  const auto aliasFull=WB99::step(V::Zero(),u,100,.01,[&](const V& x){return triangle.f.get(x);});
  const J aliasEnvelope=triangle.predict(.01,aliasFull.u);
  const V aliasRef=ControlReference::reference(triangle.f,.01,.025,1e-9);
  check(inf(aliasFull.u-aliasRef)>1e-8&&aliasEnvelope["E"].get<double>()>=inf(aliasFull.u-aliasRef),"independent triangular alias covered");
  for(const std::string kind:{"triangle","domain","kink"}){
    const double h=.78125;Synthetic small({kind,h,10/h,5,2});
    check(small.predict(.01,u)["E"].get<double>()<=1e-12,"fixed shrink returns to floor");
  }
  bool nan=false;try{triangle.predict(std::numeric_limits<double>::quiet_NaN(),u);}catch(...){nan=true;}check(nan,"NaN rejection");
  check(triangle.predict(.01,V(0,0,-1))["slope"]["gate"]=="UNKNOWN","nonforward unknown");
  std::vector<double> roots;WB100::root(roots,1,-2,1,2);check(roots.size()==2&&roots[0]==1&&roots[1]==1,"tangent roots");
  Bounds half{1,0,0,V(-INFINITY,-INFINITY,10),V::Constant(INFINITY),true};
  const auto t=WB100::tube(V::Zero(),u,20,0,half);check(t.ambiguous==0&&!t.outside,"zero radius halfspace");
  // A negative-going axis must encounter the same faces in s order.
  J identities=J::array();const auto negative=WB100::faces(V(1.5,1.5,1.5),-diagonal,std::sqrt(3.),mesh,identities);
  check(identities.size()==3&&negative.size()==3,"negative axis faces");
  return {{"gate","PASS"},{"analytic_integral_max_native_mm",worst},{"wrong_sign_difference",sign},{"wrong_unit_difference",unit},
    {"omitted_face_integral_difference_native_mm",segmented-unsplit},{"xyz_cubic","PASS"},{"nan_rejected",nan},{"nonforward","UNKNOWN as required"}};
}
void controlRun(const J& p,const J& old,const std::string& path){
  J rows=J::array();
  auto add=[&](const ControlReference::Field& f,double q,const J* saved){
    const V u(0,0,1);V full;
    if(saved)full=vec<3>(saved->at("full").at("direction"));
    else full=WB99::step(V::Zero(),u,f.h,q,[&](const V& x){return f.get(x);}).u;
    Synthetic synthetic(f);const J envelope=synthetic.predict(q,full); // No reference is read above.
    std::vector<V> refs;J ladder=J::array();
    if(saved)for(const auto& r:saved->at("reference_ladder")){refs.push_back(vec<3>(r.at("direction")));ladder.push_back(r);}
    else for(double ds:p.at("control_reference_ds_mm")){
      const V ref=ControlReference::reference(f,q,ds,p.at("control_boundary_epsilon_mm"));refs.push_back(ref);ladder.push_back({{"ds_mm",ds},{"direction",encode(ref)}});}
    const double coarse=inf(refs[1]-refs[0]),fine=inf(refs[2]-refs[1]);
    const bool valid=fine<=p["control_reference_direction_tolerance"].get<double>()&&(fine<=p["control_reference_contraction"].get<double>()*coarse||std::max(fine,coarse)<=p["control_reference_roundoff"].get<double>());
    const double U=std::max(p["absolute_direction_floor"].get<double>(),fine);
    rows.push_back({{"kind",f.kind},{"h_mm",f.h},{"phase",f.phase},{"peak_index",f.peak},{"q_over_p_Acts",q},{"origin",saved?"WB99":"fixed_shrink"},
      {"full_direction",encode(full)},{"reference_ladder",ladder},{"coarse_difference",coarse},{"fine_difference",fine},{"envelope",envelope},
      {"evaluation",evaluate(envelope,full,refs[2],refs[1],U,p,valid)},
      {"WB99_alias",saved&&saved->at("false_negative").get<bool>()&&saved->at("near_zero_estimate").get<bool>()}});
  };
  for(const auto& c:old.at("rows"))add({c.at("kind"),c.at("h_mm"),c.at("phase"),p.at("mesh_pitch_mm"),c.at("peak_index")},c.at("q_over_p_Acts"),&c);
  for(double q:p.at("shrink_q_over_p_Acts"))for(const std::string kind:p.at("shrink_kinds"))for(double h:p.at("shrink_h_mm"))
    add({kind,h,p["shrink_boundary_z_mm"].get<double>()/h,p.at("mesh_pitch_mm"),p.at("shrink_peak_index")},q,nullptr);
  exclusive(path);std::ofstream out(path);out<<J{{"schema","wb100_controls_v1"},{"rows",rows}}.dump(2)<<std::endl;
}
int main(int argc,char** argv){try{
  if(argc==2&&std::string(argv[1])=="--self-test"){std::cout<<selftest().dump(2)<<std::endl;return 0;}
  if(argc!=8&&argc!=9)throw std::runtime_error("protocol oldfield nodes wb99raw wb97steps oldcontrols outputdir [smoke]");
  const J p=read(argv[1]),old=read(argv[2]);const std::string outdir=argv[7];const bool smoke=argc==9;
  Field f;for(int k=0;k<3;++k)f.mesh[k]=old.at("mesh_mm").at(k).get<std::vector<double>>();
  std::ifstream nodefile(argv[3],std::ios::binary);std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(nodefile)),{});
  if(bytes.size()!=6*81*81*861)throw std::runtime_error("node count");f.nodes.resize(bytes.size()/2);
  for(size_t i=0;i<f.nodes.size();++i){unsigned n=bytes[2*i]+256u*bytes[2*i+1];f.nodes[i]=n<32768?int(n):int(n)-65536;}
  f.scale=old["conditions"]["bscale_kT"].get<double>()*old["conditions"]["scale"].get<double>()*1000*Acts::UnitConstants::T;
  double M=0;V L=V::Zero();const size_t ny=f.mesh[1].size(),nz=f.mesh[2].size();
  auto node=[&](size_t i)->V{return V(f.nodes[3*i],f.nodes[3*i+1],f.nodes[3*i+2])*f.scale;};
  for(size_t x=0;x<f.mesh[0].size();++x)for(size_t y=0;y<ny;++y)for(size_t z=0;z<nz;++z){
    const size_t i=(x*ny+y)*nz+z;const V v=node(i);M=std::max(M,v.norm());
    for(int k=0;k<3;++k){size_t ix=k==0?x:(k==1?y:z);if(ix+1>=f.mesh[k].size())continue;
      const size_t stride=k==0?ny*nz:(k==1?nz:1);L[k]=std::max(L[k],(node(i+stride)-v).norm()/(f.mesh[k][ix+1]-f.mesh[k][ix]));}}
  Bounds b{M,V::Constant(1e-5*Acts::UnitConstants::T).norm(),L.norm(),V(f.mesh[0].front(),f.mesh[1].front(),f.mesh[2].front()),V(f.mesh[0].back(),f.mesh[1].back(),f.mesh[2].back()),false};
  double probe=0;size_t probes=0;for(const std::string key:{"probes","domain_controls"})for(const auto& row:old.at(key)){
    probe=std::max(probe,inf(f.get(vec<3>(row.at("position_mm")))/Acts::UnitConstants::T-vec<3>(row.at("double_T"))));++probes;}
  if(probe>p["field_probe_T_tolerance"].get<double>())throw std::runtime_error("node evaluator probes");
  exclusive(outdir+"/bounds.json");std::ofstream bout(outdir+"/bounds.json");bout<<J{{"M_inside_native",M},{"M_outside_native",b.outside},{"L_axes_native_per_mm",encode(L)},{"L_native_per_mm",b.L},{"probe_max_T",probe},{"probes",probes},{"actual_collinear",false},{"selftest",selftest()}}.dump(2)<<std::endl;bout.close();
  std::ifstream raw(argv[4]),referencefile(argv[5]);std::string line,refline;std::getline(referencefile,refline);
  if(J::parse(refline).at("record")!="controls")throw std::runtime_error("prior header");
  exclusive(outdir+"/metrics.ndjson");std::ofstream metrics(outdir+"/metrics.ndjson");size_t n=0;J fidelity,terminal;std::map<int,int> traces;
  while(std::getline(raw,line)){const J row=J::parse(line);
    if(row.at("record")=="fidelity"){if(!fidelity.is_null())throw std::runtime_error("duplicate header");fidelity=row;if(fidelity.at("gate")!="PASS")throw std::runtime_error("prior fidelity");continue;}
    if(row.at("record")=="terminal"){terminal=row;break;}
    if(row.at("record")!="step")throw std::runtime_error("raw record");
    const V start=vec<3>(row.at("start_position_mm")),u=vec<3>(row.at("start_direction")),full=vec<3>(row.at("full").at("direction"));
    const double h=row.at("h_mm"),q=row.at("q_over_p_Acts");
    const J envelope=WB100::candidate(start,u,h,q,full,f.mesh,b,[&](const V& x){return f.get(x);});
    // Independent reference read occurs strictly AFTER candidate calculation.
    if(!std::getline(referencefile,refline))throw std::runtime_error("missing reference");const J refrow=J::parse(refline);
    for(const std::string key:{"trace","index","tolerance","cap_m","path","station","sample","q_over_p_Acts"})if(row.at(key)!=refrow.at(key))throw std::runtime_error("row identity "+key);
    if(h!=refrow.at("saved_step").at("h_mm").get<double>()||inf(start-vec<3>(refrow.at("start_position_mm")))>p["closure_position_mm"].get<double>()||inf(u-vec<3>(refrow.at("start_direction")))>p["closure_direction"].get<double>())throw std::runtime_error("arc/initial identity");
    const int trace=row.at("trace"),index=row.at("index");if(index!=traces[trace]++)throw std::runtime_error("index order");
    const auto& ladder=refrow.at("reference_ladder");const V ref=vec<3>(ladder[2].at("direction")),prev=vec<3>(ladder[1].at("direction"));
    const double closure=inf(full-vec<3>(refrow.at("saved_step").at("direction")));
    bool valid=refrow.at("closure_gate")=="PASS"&&refrow.at("reference_gate")=="PASS"&&closure<=p["closure_direction"].get<double>()&&row.at("full_position_closure_mm").get<double>()<=p["closure_position_mm"].get<double>();
    double arc=0;for(const auto& r:ladder)arc=std::max(arc,std::abs(r.at("arc_residual_mm").get<double>()));
    const double U=std::max({p["absolute_direction_floor"].get<double>(),inf(ref-prev),closure,envelope["kappa_per_mm"].get<double>()*arc});
    J result;for(const std::string key:{"trace","index","tolerance","cap_m","path","station","sample","h_mm"})result[key]=row.at(key);
    result["start_position_mm"]=encode(start);result["start_direction"]=encode(u);result["full_direction"]=encode(full);result["q_over_p_Acts"]=q;
    result["classification"]=refrow.at("classification").at("class");result["envelope"]=envelope;result["evaluation"]=evaluate(envelope,full,ref,prev,U,p,valid);
    metrics<<result.dump()<<'\n';++n;if(smoke)break;
  }
  metrics.close();
  if(!smoke){if(terminal.is_null()||terminal.at("steps")!=n||n!=p.at("expected_steps").get<size_t>()||traces.size()!=p.at("expected_traces").get<size_t>()||std::getline(raw,line))throw std::runtime_error("actual incomplete");
    if(!std::getline(referencefile,refline)||J::parse(refline).at("record")!="terminal"||std::getline(referencefile,refline))throw std::runtime_error("prior terminal");
    controlRun(p,read(argv[6]),outdir+"/controls.json");}
  exclusive(outdir+"/terminal.json");std::ofstream final(outdir+"/terminal.json");final<<J{{"steps",n},{"traces",traces},{"fidelity",fidelity},{"prior_terminal",terminal},{"new_wrapper_queries",0},{"smoke",smoke}}.dump(2)<<std::endl;
  return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
