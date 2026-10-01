// New diagnostic; immutable WB97 provides actual ACTS initial conversion and scalar reference.
#define main wb97_unused_main
#include "LocalDefect.cxx"
#undef main
#include <unsupported/Eigen/AutoDiff>
namespace WB98 {
using X=Eigen::Matrix<double,4,1>;using M=Eigen::Matrix<double,4,4>;using Y=Eigen::Matrix<double,20,1>;
using AD=Eigen::AutoDiffScalar<Eigen::Matrix<double,4,1>>;
X scales(){X d;d<<1,1,.001,.001;return d;}
double norm(const X& x){return (x.array()/scales().array()).abs().maxCoeff();}
double jnorm(const M& a){return (scales().cwiseInverse().asDiagonal()*a*scales().asDiagonal()).cwiseAbs().maxCoeff();}
J matrix(const M& m){J rows=J::array();for(int i=0;i<4;++i)rows.push_back(encode(m.row(i)));return rows;}
X chart(const V& p,const V& u){if(u[2]<=1e-6)throw std::runtime_error("nonforward chart");X x;x<<p[0],p[1],u[0]/u[2],u[1]/u[2];return x;}
struct Flow{X x;M a;};
struct Derivative{X value;M gradient;};
Derivative rhs(const Field& field,const X& x,double z,double q,double margin){
  const V position(x[0],x[1],z);V b=field.get(position);Eigen::Matrix<double,3,2> grad=Eigen::Matrix<double,3,2>::Zero();
  if(!field.constant && field.inside(position)){
    for(int k=0;k<2;++k)if(position[k]-field.mesh[k].front()<=margin || field.mesh[k].back()-position[k]<=margin)throw std::runtime_error("transverse domain derivative unresolved");
    const auto c=field.cell(position);V f;for(int k=0;k<3;++k)f[k]=(position[k]-field.mesh[k][c[k]])/(field.mesh[k][c[k]+1]-field.mesh[k][c[k]]);
    for(int corner=0;corner<8;++corner){std::array<int,3> ix=c;for(int k=0;k<3;++k)ix[k]+=bool(corner&(1<<(2-k)));
      const size_t offset=3*((ix[0]*field.mesh[1].size()+ix[1])*field.mesh[2].size()+ix[2]);
      for(int axis=0;axis<2;++axis){double w=(corner&(1<<(2-axis))?1.:-1.)/(field.mesh[axis][c[axis]+1]-field.mesh[axis][c[axis]]);
        for(int k=0;k<3;++k)if(k!=axis)w*=corner&(1<<(2-k))?f[k]:1-f[k];
        for(int component=0;component<3;++component)grad(component,axis)+=w*field.nodes.at(offset+component)*field.scale;
      }
    }
  }
  std::array<AD,4> t;for(int k=0;k<4;++k){t[k].value()=x[k];t[k].derivatives()=X::Unit(k);}
  std::array<AD,3> ba;for(int k=0;k<3;++k){ba[k].value()=b[k];ba[k].derivatives()=X::Zero();ba[k].derivatives()[0]=grad(k,0);ba[k].derivatives()[1]=grad(k,1);}
  const AD root=sqrt(AD(1.)+t[2]*t[2]+t[3]*t[3]);
  std::array<AD,4> result={t[2],t[3],q*root*(t[3]*ba[2]-(AD(1.)+t[2]*t[2])*ba[1]+t[2]*t[3]*ba[0]),
    q*root*((AD(1.)+t[3]*t[3])*ba[0]-t[2]*ba[2]-t[2]*t[3]*ba[1])};
  Derivative out;for(int k=0;k<4;++k){out.value[k]=result[k].value();out.gradient.row(k)=result[k].derivatives().transpose();}
  if(!out.value.allFinite() || !out.gradient.allFinite())throw std::runtime_error("nonfinite variational RHS");return out;
}
Flow flow(const Field& field,const X& initial,double from,double to,double dz,double q,const J& p){
  if(!initial.allFinite()||!std::isfinite(q)||!(to-from>p.at("minimum_forward_dz_mm").get<double>()))throw std::runtime_error("tiny/nonforward flow");
  Y state=Y::Zero();state.head<4>()=initial;Eigen::Map<M>(state.data()+4)=M::Identity();
  std::vector<double> cuts;if(!field.constant)for(double edge:field.mesh[2])if(edge>from&&edge<to)cuts.push_back(edge);cuts.push_back(to);
  double current=from;
  for(double stop:cuts){const double segment=current,mid=.5*(current+stop);Y correction=Y::Zero();
    auto evaluate=[&](const Y& y,double z)->Y{
      if(!field.constant)for(double edge:{field.mesh[2].front(),field.mesh[2].back()})if(std::abs(z-edge)<p.at("boundary_side_epsilon_mm").get<double>()/2)z=edge+(mid<edge?-1:1)*p.at("boundary_side_epsilon_mm").get<double>();
      const auto d=rhs(field,y.head<4>(),z,q,p.at("transverse_domain_margin_mm"));Y out;out.head<4>()=d.value;
      Eigen::Map<M>(out.data()+4)=d.gradient*Eigen::Map<const M>(y.data()+4);return out;};
    const size_t count=std::ceil((stop-segment)/dz);
    for(size_t i=0;i<count;++i){const double next=i+1==count?stop:std::min(stop,segment+(i+1)*dz),h=next-current;
      if(!(h>0))throw std::runtime_error("variational step underflow");
      const Y k1=evaluate(state,current),k2=evaluate(state+.5*h*k1,current+.5*h),k3=evaluate(state+.5*h*k2,current+.5*h),k4=evaluate(state+h*k3,next);
      const Y increment=h/6.*(k1+2*k2+2*k3+k4)-correction,updated=state+increment;
      correction=(updated-state)-increment;state=updated;current=next;
    }
  }
  if(!state.allFinite())throw std::runtime_error("nonfinite flow");return {state.head<4>(),Eigen::Map<const M>(state.data()+4)};
}
X scalar(const Field& field,const X& x,double from,double to,double dz,double q,const J& p){
  const auto pair=integrate(field,V(x[0],x[1],from),V(x[2],x[3],1).normalized(),q,to,dz,p.at("boundary_side_epsilon_mm"));
  X result;for(int k=0;k<4;++k)result[k]=pair.first[k];return result;
}
bool converged(double coarse,double fine,const J& p){return fine<=p.at("reference_scaled_tolerance").get<double>()&&
  (fine<=p.at("reference_contraction_ratio").get<double>()*coarse || std::max(coarse,fine)<=p.at("reference_roundoff_scaled").get<double>());}
J fd_check(const Field& field,const X& x,double from,double to,double q,const M& a,const J& p){
  J rows=J::array();std::vector<M> matrices;double center=0;
  for(double multiplier:p.at("jacobian_fd_multipliers")){
    M finite;for(int k=0;k<4;++k){const double d=multiplier*p.at("jacobian_fd_steps").at(k).get<double>();X shift=X::Zero();shift[k]=d;
      finite.col(k)=(scalar(field,x+shift,from,to,p.at("reference_dz_mm").back(),q,p)-scalar(field,x-shift,from,to,p.at("reference_dz_mm").back(),q,p))/(2*d);}
    const double difference=jnorm(finite-a);rows.push_back({{"multiplier",multiplier},{"matrix",matrix(finite)},{"versus_variational_scaled",difference}});matrices.push_back(finite);center=std::max(center,difference);
  }
  const double fdDifference=jnorm(matrices[1]-matrices[0]);return {{"rows",rows},{"fine_coarse_scaled",fdDifference},
    {"gate",std::max(center,fdDifference)<=p.at("jacobian_scaled_tolerance").get<double>()?"PASS":"UNKNOWN"}};
}
Ref plane_reference(const Field& field,const V& startp,const V& startu,double q,const Acts::Transform3& frame,double dz,const J& protocol){
  const V n=frame.linear().col(2),origin=frame.translation();double z=origin[2];S s{};Footprint fp;double residual=0;
  for(int iter=0;iter<protocol.at("arc_root_iterations").get<int>();++iter){
    if(z<=startp[2])throw std::runtime_error("nonforward plane root");auto pair=integrate(field,startp,startu,q,z,dz,protocol.at("boundary_side_epsilon_mm"));s=pair.first;fp=pair.second;
    residual=n.dot(V(s[0],s[1],z)-origin);if(std::abs(residual)<=protocol.at("plane_root_residual_mm").get<double>())break;
    const double derivative=n.dot(V(s[2],s[3],1));if(std::abs(derivative)<1e-6)throw std::runtime_error("unstable plane root");z-=residual/derivative;
  }
  if(std::abs(residual)>protocol.at("plane_root_residual_mm").get<double>())throw std::runtime_error("plane root unclosed");return {V(s[0],s[1],z),V(s[2],s[3],1).normalized(),residual,fp};
}
J controls98(const J& p){
  J rows=J::array();double worst=0,jworst=0;bool nan=false,tiny=false,domain=false;
  for(bool zero:{true,false})for(double length:{100.,1000.})for(double cap:{10.,100.}){
    Field f;f.constant=true;f.value=zero?V::Zero().eval():V(0,Acts::UnitConstants::T,0);const double q=.01,w=q*f.value[1];
    V xp=V::Zero(),u(0,0,1);X e=X::Zero(),pos=X::Zero(),ang=X::Zero();double from=0;
    for(int iteration=0;iteration<int(length/cap);++iteration){const double h=cap;const auto step=rkn(xp,u,h,q,f.value,f.value,f.value);
      const X initial=chart(xp,u),official=chart(step.p,step.u);const auto ref=flow(f,initial,from,step.p[2],.03125,q,p);
      const X delta=official-ref.x;X dpos=delta,dang=delta;dpos.tail<2>().setZero();dang.head<2>().setZero();
      e=ref.a*e+delta;pos=ref.a*pos+dpos;ang=ref.a*ang+dang;xp=step.p;u=step.u;from=xp[2];
    }
    const double c=std::sqrt(1-w*w*from*from);X exact=X::Zero();if(!zero){exact[0]=-w*from*from/(1+c);exact[2]=-w*from/c;}
    const X actual=chart(xp,u)-exact;const double error=norm(e-actual);worst=std::max(worst,error);
    const auto fd=fd_check(f,X::Zero(),0,100,q,flow(f,X::Zero(),0,100,.03125,q,p).a,p);
    for(const auto& row:fd.at("rows"))jworst=std::max(jworst,row.at("versus_variational_scaled").get<double>());
    rows.push_back({{"zero",zero},{"requested_arc_length_mm",length},{"actual_endpoint_z_mm",from},{"cap_arc_mm",cap},{"closure_scaled",error},{"split_closure_scaled",norm(pos+ang-e)},{"fd",fd}});
  }
  // Analytic zero-field injected-vector recurrence independent of source replay.
  M a=M::Identity();a(0,2)=100;a(1,3)=100;X ep=X::Zero(),eu=X::Zero(),sum=X::Zero();
  for(int i=0;i<10;++i){X dp=X::Zero(),du=X::Zero();dp[0]=1e-6;du[3]=1e-9;ep=a*ep+dp;eu=a*eu+du;sum=a*sum+dp+du;}
  X expected;expected<<1e-5,4.5e-6,0,1e-8;const double injected=norm(sum-expected);worst=std::max(worst,injected);
  Field f;f.constant=true;try{flow(f,X::Constant(std::numeric_limits<double>::quiet_NaN()),0,100,.03125,.01,p);}catch(...){nan=true;}
  try{flow(f,X::Zero(),0,1e-12,.03125,.01,p);}catch(...){tiny=true;}
  f.constant=false;f.mesh={std::vector<double>{-200,200},std::vector<double>{-200,200},std::vector<double>{0,100}};f.nodes.resize(24,0);
  X edge=X::Zero();edge[0]=200;try{rhs(f,edge,50,.01,.01);}catch(...){domain=true;}
  Field by;by.constant=true;by.value=V(0,Acts::UnitConstants::T,0);
  const X truth=scalar(by,X::Zero(),0,10,.03125,.01,p);const double wrongsign=norm(scalar(by,X::Zero(),0,10,.03125,-.01,p)-truth),wrongunit=norm(scalar(by,X::Zero(),0,10,.03125,10.,p)-truth);
  return {{"rows",rows},{"max_closure_scaled",worst},{"max_fd_scaled",jworst},{"injected_zero_scaled",injected},{"injected_split_scaled",norm(ep+eu-sum)},
    {"nan_rejected",nan},{"tiny_rejected",tiny},{"domain_derivative_rejected",domain},{"wrong_sign_scaled",wrongsign},{"wrong_unit_scaled",wrongunit},
    {"gate",worst<=p.at("controls_max_scaled_error").get<double>()&&jworst<=p.at("jacobian_scaled_tolerance").get<double>()&&nan&&tiny&&domain&&wrongsign>1e-8&&wrongunit>1e-8?"PASS":"FAIL"}};
}
}

int main(int argc,char** argv){try{
  using namespace WB98;
  if(argc==3&&std::string(argv[1])=="--self-test"){std::cout<<controls98(read(argv[2])).dump(2)<<std::endl;return 0;}
  if(argc!=7)throw std::runtime_error("usage: transport protocol wb96raw wb95raw nodes steps.ndjson traces.json");
  const J p=read(argv[1]),raw=read(argv[2]),old=read(argv[3]);Field field;
  for(int k=0;k<3;++k)field.mesh[k]=old.at("mesh_mm").at(k).get<std::vector<double>>();
  std::ifstream input(argv[4],std::ios::binary);std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(input)),{});
  if(bytes.size()!=6*81*81*861)throw std::runtime_error("node count");field.nodes.resize(bytes.size()/2);
  for(size_t k=0;k<field.nodes.size();++k){unsigned value=bytes[2*k]+256u*bytes[2*k+1];field.nodes[k]=value<32768?int(value):int(value)-65536;}
  field.scale=old.at("conditions").at("bscale_kT").get<double>()*old.at("conditions").at("scale").get<double>()*1000*Acts::UnitConstants::T;
  const J control=controls98(p);if(control.at("gate")!="PASS")throw std::runtime_error("controls98 failed");
  double probeMax=0;size_t probeCount=0;for(const std::string key:{"probes","domain_controls"})for(const auto& query:old.at(key)){
    probeMax=std::max(probeMax,(field.get(vec<3>(query.at("position_mm")))/Acts::UnitConstants::T-vec<3>(query.at("double_T"))).cwiseAbs().maxCoeff());++probeCount;}
  if(probeMax>p.at("field_probe_T_tolerance").get<double>())throw std::runtime_error("field probe failed");
  exclusive(argv[5]);std::ofstream steps(argv[5]);J results={{"controls",control},{"field_probe_count",probeCount},{"field_probe_max_T",probeMax},{"traces",J::array()}};
  size_t trace=0,total=0;Acts::GeometryContext g;
  auto analyze=[&](const J& setting,const J& target,const J& call,const std::string& path,int station,int sample){
    const auto start=initial(raw,setting,target,call,path);V sp=start.position(g),su=start.direction(),position=sp,direction=su;
    const X initialX=chart(sp,su);const double q=start.parameters()[Acts::eBoundQOverP];const size_t n=call.at("accepted_trace").size();
    std::array<X,3> prediction{X::Zero(),X::Zero(),X::Zero()},positions=prediction,angles=prediction,arithmetic=prediction;
    std::array<bool,3> valid{true,true,true};size_t referenceUnknown=0,jacUnknown=0,fdUnknown=0;double independentWorst=0,samePoint=0;
    J result={{"trace",trace},{"tolerance",setting.at("tolerance")},{"cap_m",setting.at("cap_m")},{"path",path},{"station",station},{"sample",sample},
      {"source_steps",n},{"initial_position_mm",encode(sp)},{"initial_direction",encode(su)},{"q_over_p_Acts",q}};
    for(const auto& query:call.at("field_queries"))samePoint=std::max(samePoint,(field.get(vec<3>(query.at("position_mm")))/Acts::UnitConstants::T-vec<3>(query.at("field_T"))).cwiseAbs().maxCoeff());
    size_t index=0;for(const auto& step:call.at("accepted_trace")){
      const V endp=vec<3>(step.at("position_mm")),endu=vec<3>(step.at("direction"));const X actual=chart(endp,endu),x=chart(position,direction);
      J row={{"trace",trace},{"index",index},{"start_X",encode(x)},{"end_X",encode(actual)},{"from_z_mm",position[2]},{"to_z_mm",endp[2]},
        {"h_mm",step.at("h_mm")},{"reference_gate","UNKNOWN"},{"jacobian_gate","UNKNOWN"},{"fd_gate","NOT_SELECTED"},{"ladders",J::array()}};
      try{
        const size_t begin=step.at("query_begin").get<size_t>()+size_t(index==0&&call.at("options").at("loopProtection").get<bool>()),end=step.at("query_end");
        std::array<J,3> queries{call.at("field_queries").at(begin),call.at("field_queries").at(end-2),call.at("field_queries").at(end-1)};
        std::array<V,3> b;for(int k=0;k<3;++k)b[k]=field.get(vec<3>(queries[k].at("position_mm")));
        const auto doubleRkn=rkn(position,direction,step.at("h_mm"),q,b[0],b[1],b[2]);
        const auto savedFieldRkn=rkn(position,direction,step.at("h_mm"),q,
          vec<3>(queries[0].at("field_T"))*Acts::UnitConstants::T,vec<3>(queries[1].at("field_T"))*Acts::UnitConstants::T,vec<3>(queries[2].at("field_T"))*Acts::UnitConstants::T);
        const double sourceClosure=norm(actual-chart(savedFieldRkn.p,savedFieldRkn.u));row["source_rkn_closure_scaled"]=sourceClosure;
        if(sourceClosure>1e-9)throw std::runtime_error("saved source RKN closure failed");
        const X fieldDelta=chart(savedFieldRkn.p,savedFieldRkn.u)-chart(doubleRkn.p,doubleRkn.u);
        row["fixed_stage_arithmetic_delta"]=encode(fieldDelta);std::vector<Flow> flows;
        for(int level=0;level<3;++level){const double dz=p.at("reference_dz_mm").at(level);const auto ref=flow(field,x,position[2],endp[2],dz,q,p);flows.push_back(ref);
          const X delta=actual-ref.x;X dpos=delta,dang=delta;dpos.tail<2>().setZero();dang.head<2>().setZero();
          prediction[level]=ref.a*prediction[level]+delta;positions[level]=ref.a*positions[level]+dpos;angles[level]=ref.a*angles[level]+dang;arithmetic[level]=ref.a*arithmetic[level]+fieldDelta;
          row["ladders"].push_back({{"dz_mm",dz},{"reference_X",encode(ref.x)},{"A",matrix(ref.a)},{"delta",encode(delta)},
            {"prediction",encode(prediction[level])},{"position_contribution",encode(positions[level])},{"slope_contribution",encode(angles[level])},{"fixed_stage_arithmetic_contribution",encode(arithmetic[level])}});
        }
        const double coarse=norm(flows[1].x-flows[0].x),fine=norm(flows[2].x-flows[1].x),jacFine=jnorm(flows[2].a-flows[1].a),jacCoarse=jnorm(flows[1].a-flows[0].a);
        const bool refPass=converged(coarse,fine,p),jacPass=std::max(jacFine,jacCoarse)<=p.at("jacobian_scaled_tolerance").get<double>();
        row["reference_coarse_scaled"]=coarse;row["reference_fine_scaled"]=fine;row["jacobian_coarse_scaled"]=jacCoarse;row["jacobian_fine_scaled"]=jacFine;
        row["reference_gate"]=refPass?"PASS":"UNKNOWN";row["jacobian_gate"]=jacPass?"PASS":"UNKNOWN";
        if(index==0 || index==n/2 || index+1==n){const J fd=fd_check(field,x,position[2],endp[2],q,flows[2].a,p);row["fd"]=fd;row["fd_gate"]=fd.at("gate");
          const double difference=norm(flows[2].x-scalar(field,x,position[2],endp[2],p.at("reference_dz_mm").back(),q,p));independentWorst=std::max(independentWorst,difference);
          row["independent_scalar_difference_scaled"]=difference;if(difference>p.at("reference_scaled_tolerance").get<double>())row["fd_gate"]="UNKNOWN";}
      }catch(const std::exception& e){row["unknown_reason"]=e.what();valid={false,false,false};}
      referenceUnknown+=row["reference_gate"]!="PASS";jacUnknown+=row["jacobian_gate"]!="PASS";fdUnknown+=row["fd_gate"]=="UNKNOWN";
      steps<<row.dump()<<'\n';if(!steps)throw std::runtime_error("step output failed");++total;++index;position=endp;direction=endu;
    }
    result["reference_unknown"]=referenceUnknown;result["jacobian_unknown"]=jacUnknown;result["fd_unknown"]=fdUnknown;result["independent_scalar_max_scaled"]=independentWorst;
    result["saved_field_queries"]=call.at("field_queries").size();result["same_point_float_double_max_T"]=samePoint;
    result["endpoint_ladders"]=J::array();result["global_gate"]="UNKNOWN";
    try{
      std::vector<X> global;std::vector<Ref> arc,planes;Acts::Transform3 frame=Acts::Transform3::Identity();
      if(path=="entry")frame.translation().z()=raw.at("conditions").at("min_mm").at(2);
      else for(int i=0;i<4;++i)for(int k=0;k<4;++k)frame.matrix()(i,k)=target.at("frame").at(i).at(k);
      const X actual=chart(position,direction);const X bound=vec<4>(call.at("state").at("h"));
      for(int level=0;level<3;++level){double dz=p.at("reference_dz_mm").at(level);global.push_back(scalar(field,initialX,sp[2],position[2],dz,q,p));
        arc.push_back(reference(field,sp,su,q,call.at("path_length_mm"),position[2],dz,p));planes.push_back(plane_reference(field,sp,su,q,frame,dz,p));
        const X defect=actual-global.back(),residual=prediction[level]-defect;
        result["endpoint_ladders"].push_back({{"dz_mm",dz},{"global_same_z_X",encode(global.back())},{"actual_same_z_defect",encode(defect)},
          {"prediction",encode(prediction[level])},{"position_contribution",encode(positions[level])},{"slope_contribution",encode(angles[level])},
          {"fixed_stage_arithmetic_contribution",encode(arithmetic[level])},{"closure_residual",encode(residual)},
          {"global_same_arc_position_mm",encode(arc.back().p)},{"global_same_arc_direction",encode(arc.back().u)},{"global_arc_root_residual_mm",arc.back().arc_residual},
          {"global_same_arc_X",encode(chart(arc.back().p,arc.back().u))},{"plane_reference_X",encode(chart(planes.back().p,planes.back().u))},
          {"plane_reference_position_mm",encode(planes.back().p)},{"plane_root_residual_mm",planes.back().arc_residual},
          {"bound_minus_plane_reference",encode(bound-chart(planes.back().p,planes.back().u))}});
      }
      const double coarse=norm(global[1]-global[0]),fine=norm(global[2]-global[1]);
      const bool globalPass=converged(coarse,fine,p)&&converged(scaled(arc[0],arc[1],p.at("direction_scale")),scaled(arc[1],arc[2],p.at("direction_scale")),p)
        &&converged(scaled(planes[0],planes[1],p.at("direction_scale")),scaled(planes[1],planes[2],p.at("direction_scale")),p);
      result["global_coarse_scaled"]=coarse;result["global_fine_scaled"]=fine;result["transport_fine_scaled"]=norm(prediction[2]-prediction[1]);
      result["global_gate"]=globalPass?"PASS":"UNKNOWN";
      const double budget=std::max({p.at("closure_budget_floor_scaled").get<double>(),p.at("closure_budget_factor").get<double>()*fine,
        p.at("closure_budget_factor").get<double>()*norm(prediction[2]-prediction[1]),p.at("closure_budget_factor").get<double>()*control.at("max_closure_scaled").get<double>()});
      result["closure_budget_scaled"]=budget;result["closure_scaled"]=norm(prediction[2]-(actual-global[2]));
      result["closure_gate"]=globalPass && !referenceUnknown && !jacUnknown && !fdUnknown && valid[2]?
        (result.at("closure_scaled").get<double>()<=budget?"PASS":"FAIL"):"UNKNOWN";
      result["free_to_bound_position_delta_mm"]=encode(vec<3>(call.at("state").at("position_mm"))-position);
      result["same_arc_minus_same_z_X"]=encode(chart(arc[2].p,arc[2].u)-global[2]);
      result["plane_minus_same_z_X"]=encode(chart(planes[2].p,planes[2].u)-global[2]);
    }catch(const std::exception& e){result["unknown_reason"]=e.what();result["closure_gate"]="UNKNOWN";}
    results["traces"].push_back(result);steps.flush();std::cout<<"trace "<<++trace<<" steps "<<n<<" closure "<<result.at("closure_gate")<<std::endl;
  };
  for(const auto& setting:raw.at("settings")){
    analyze(setting,J::object(),setting.at("entry_nominal"),"entry",0,0);
    for(const auto& target:setting.at("targets")){
      for(int sample:p.at("trace_sample_indices"))analyze(setting,target,target.at("samples").at(sample),"direct",target.at("station"),sample);
      analyze(setting,target,target.at("fixed_reference_start_nominal"),"fixed_start",target.at("station"),0);
    }
  }
  results["steps"]=total;results["qualification"]="NOT_EVALUATED";
  exclusive(argv[6]);std::ofstream summary(argv[6]);summary<<results.dump(2)<<std::endl;
  return total==p.at("expected_steps").get<size_t>()&&trace==p.at("expected_traces").get<size_t>()?0:2;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
