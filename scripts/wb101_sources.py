"""Exact versioned source transformations into an isolated component only."""
from pathlib import Path
from alignment.wb90_measurement_contract import ROOT
from wb92_contract import ACTS

WB96=ROOT/'outputs/mc24_four_station_wb96_acts_tolerance_v2'

def replace_one(source,old,new):
    if source.count(old)!=1:raise ValueError('source marker count '+old[:90])
    return source.replace(old,new)

def step_source():
    s=(ACTS/'include/Acts/Propagator/EigenStepper.ipp').read_text()
    start=s.index('template <typename E, typename A>\ntemplate <typename propagator_state_t, typename navigator_t>\nActs::Result<double> Acts::EigenStepper<E, A>::step(')
    end=s.index('\ntemplate <typename E, typename A>\nvoid Acts::EigenStepper<E, A>::setIdentityJacobian',start)
    s=s[start:end]
    s=replace_one(s,'template <typename E, typename A>\n','')
    s=replace_one(s,'Acts::EigenStepper<E, A>::step','WB101::DirectionStepper::step')
    s=replace_one(s,'  using namespace UnitLiterals;','''  using namespace Acts;
  if(!m_control || m_control->threshold==0) return Acts::EigenStepper<>::step(state,navigator);
  if(state.stepping.covTransport) throw std::runtime_error("WB101 covariance not authorized");
  bool directionReject=false;''')
    s=replace_one(s,'return success(error_estimate <= state.options.stepTolerance);',
                  'return success(m_control->judge(state,pos,dir,h,error_estimate,directionReject));')
    s=replace_one(s,'h *= stepSizeScaling;','h *= directionReject ? 0.5 : stepSizeScaling;')
    license=(ACTS/'include/Acts/Propagator/EigenStepper.ipp').read_text().split('#include')[0]
    return license+'// Isolated copy: disabled delegates to original, enabled adds pre-finalize acceptance.\n'+s

def component_source():
    s=(WB96/'isolated_source/WB96Diagnostic/ToleranceNavigationAudit.cxx').read_text()
    s=s.replace('namespace WB96 {','namespace WB101 {').replace('WB96::ToleranceNavigationAudit','WB101::DirectionAcceptanceAudit').replace('ToleranceNavigationAudit','DirectionAcceptanceAudit')
    s='#include "Acceptance.h"\n#include "NodeModel.h"\n'+s
    marker='    using Engine=Acts::Propagator<Acts::EigenStepper<>,Acts::Navigator>;'
    s=replace_one(s,marker,'''    Json oldField,bounds;std::ifstream oldInput(m_fixture.at("wb101_field_source").get<std::string>()),boundsInput(m_fixture.at("wb101_bounds_source").get<std::string>());
    oldInput>>oldField;boundsInput>>bounds;if(!oldInput||!boundsInput)throw std::runtime_error("WB101 field prerequisites");
    WB101::NodeModel nodes(mh->fieldMap(),ch->dipoleFieldScaleFactor(),oldField,bounds,m_fixture.at("wb101_nodes_source"));
    size_t identityRejected=0;
    for(const std::string key:{"scale","bscale_kT","mesh","node_count","node_file"}){
      Json wrong=oldField;double actualScale=ch->dipoleFieldScaleFactor();std::string path=m_fixture.at("wb101_nodes_source");
      if(key=="scale")actualScale+=1;
      if(key=="bscale_kT")wrong["conditions"]["bscale_kT"]=oldField.at("conditions").at("bscale_kT").get<double>()*2;
      if(key=="mesh")wrong["mesh_mm"][0][0]=wrong["mesh_mm"][0][0].get<double>()+1;
      if(key=="node_count")wrong["node_count"]=wrong["node_count"].get<size_t>()+1;
      if(key=="node_file")path="/dev/null";
      bool caught=false;try{WB101::NodeModel reject(mh->fieldMap(),actualScale,wrong,bounds,path);}catch(const std::runtime_error&){caught=true;}
      if(!caught)throw std::runtime_error("WB101 identity negative control failed");++identityRejected;
    }
    nodes.evidence["identity_negative_controls_rejected"]=identityRejected;
    WB101::Control control;control.allowance=m_fixture.at("wb101_protocol").at("direction_uncertainty_allowance");
    control.envelope=[&](const Acts::Vector3& pos,const Acts::Vector3& u,double h,double q,const Acts::Vector3& full){
      return WB100::candidate(pos,u,h,q,full,nodes.mesh,nodes.bounds,[&](const Acts::Vector3& x){++control.nodeQueries;return nodes.get(x);});};
    control.queryCount=[&](){return field->queries;};
    auto outputStream=[&](const std::string& suffix){const std::string path=m_output.value()+suffix;
      const int fd=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);if(fd<0)throw std::runtime_error("exclusive WB101 stream");::close(fd);return std::ofstream(path);};
    auto trials=outputStream(".trials.ndjson"),traceStream=outputStream(".traces.ndjson");
    size_t callId=0;double directionThreshold=0;
    using Engine=Acts::Propagator<WB101::DirectionStepper,Acts::Navigator>;''')
    s=replace_one(s,'Engine engine{Acts::EigenStepper<>(field),','Engine engine{WB101::DirectionStepper(field,&control),')
    s=replace_one(s,'      {"math_control",mathControls(g,mctx)},','      {"math_control",mathControls(g,mctx)},{"wb101_node_evidence",nodes.evidence},{"wb101_muon_mass_Acts",Acts::ParticleHypothesis::muon().mass()},')
    s=replace_one(s,'    const auto ids=m_tools[0]->trackingGeometryTool()->getIdentifierMap();',
                  '    if(out.at("conditions")!=oldField.at("conditions"))throw std::runtime_error("WB101 actual condition/IOV identity");\n    const auto ids=m_tools[0]->trackingGeometryTool()->getIdentifierMap();')
    s=replace_one(s,'      field->reset(save);WB96Observation::Trace trace;trace.save=save;', '''      const size_t currentCall=callId++;
      field->reset(save);WB96Observation::Trace trace;trace.save=save;
      control.reset(directionThreshold,false);
      control.sink=[&](const Json& input){Json row=input;row["call_id"]=currentCall;row["threshold"]=directionThreshold;
        trials<<row.dump()<<'\\n';if(!trials)throw std::runtime_error("WB101 trial stream");};''')
    s=replace_one(s,'{"field_counts",{{"total",field->queries}', '{"call_id",currentCall},{"direction_control",control.summary()},\n        {"start_state",state(start,g)},{"target_frame",WB96Observation::matrix(frame)},\n        {"field_counts",{{"total",field->queries}')
    marker='      if(!result.ok()){row["error_code"]'
    s=replace_one(s,marker,'''      if(save){traceStream<<Json{{"call_id",currentCall},{"threshold",directionThreshold},{"start_state",row.at("start_state")},
        {"field_queries",field->raw},{"accepted_trace",trace.steps},{"field_counts",row.at("field_counts")},{"direction_control",control.summary()}}.dump()<<'\\n';
        if(!traceStream)throw std::runtime_error("WB101 trace stream");}
      row.erase("field_queries");row.erase("accepted_trace");
      if(!result.ok()){row["error_code"]''')
    s=replace_one(s,'    for(const auto& toleranceValue:p.at("step_tolerances"))for(size_t ti=0;ti<m_tools.size();++ti) {',
                  '    for(const auto& tau:m_fixture.at("wb101_protocol").at("arms"))for(size_t ti=0;ti<m_tools.size();++ti) {\n      directionThreshold=tau;')
    s=replace_one(s,'      const double tolerance=toleranceValue,cap=', '      const double tolerance=m_fixture.at("wb101_protocol").at("step_tolerance"),cap=')
    s=replace_one(s,'const bool baseline=tolerance==p.at("step_tolerances").at(0);', 'const bool baseline=directionThreshold==0;')
    s=replace_one(s,'Json setting={{"tolerance",tolerance},{"cap_m",cap}', 'Json setting={{"direction_threshold",directionThreshold},{"tolerance",tolerance},{"cap_m",cap}')
    s=replace_one(s,'      out["settings"].push_back(setting);', '''      out["settings"].push_back(setting);
      ATH_MSG_INFO("WB101 completed tau="<<directionThreshold<<" cap="<<cap<<" calls="<<callId);''')
    s=replace_one(s,'    out["observed_surfaces"]=', '''    trials<<Json{{"record","terminal"},{"calls",callId}}.dump()<<'\\n';trials.close();traceStream.close();
    if(!trials||!traceStream)throw std::runtime_error("WB101 stream completion");
    out["wb101_calls"]=callId;
    out["observed_surfaces"]=''')
    s=replace_one(s,'        guard(setting.at("entry_nominal"),&saved.at("entry"));', '''        size_t guardIndex=0;const auto& frozenGuards=m_fixture.at("wb101_baseline_guards").at(ti);
        auto structure=[&](const Json& row){const auto& expected=frozenGuards.at(guardIndex++);
          for(const std::string key:{"accepted_steps","rejected_trials","field_counts","sensitive_sequence","options"})
            if(row.at(key)!=expected.at(key))throw std::runtime_error("WB101 disabled structural mismatch before enabled calls");};
        structure(setting.at("entry_nominal"));
        for(const auto& t:setting.at("targets")){for(const auto& r:t.at("samples"))structure(r);structure(t.at("fixed_reference_start_nominal"));}
        if(guardIndex!=79)throw std::runtime_error("WB101 disabled guard population");
        guard(setting.at("entry_nominal"),&saved.at("entry"));''')
    return s

def files():
    top=(WB96/'isolated_source/CMakeLists.txt').read_text().replace('WB96','WB101')
    cmake=(WB96/'isolated_source/WB96Diagnostic/CMakeLists.txt').read_text().replace('WB96','WB101').replace('ToleranceNavigationAudit','DirectionAcceptanceAudit')
    result={'CMakeLists.txt':top,'WB101Diagnostic/CMakeLists.txt':cmake,'WB101Diagnostic/DirectionAcceptanceAudit.cxx':component_source(),
            'WB101Diagnostic/DirectionStep.inc':step_source()}
    for path in ('research/wb101/Acceptance.h','research/wb101/NodeModel.h','research/wb100/Envelope.h','research/wb99/DirectionDoubling.h',
                 'research/wb96/ObservedPropagation.h','research/wb94/ReferenceRK4.h','research/wb95/PrecisionControl.h','research/wb95/CompensatedRK4.h'):
        content=(ROOT/path).read_text()
        if path.endswith('Envelope.h'):content=content.replace('../wb99/DirectionDoubling.h','DirectionDoubling.h')
        result['WB101Diagnostic/'+Path(path).name]=content
    return result
