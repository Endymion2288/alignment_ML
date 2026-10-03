"""Exact official-tool copy plus passive boundState instrumentation."""
from pathlib import Path
from wb101_sources import replace_one
from wb106_contract import ROOT,PARENT

OFFICIAL=ROOT.parent/'calypso/Tracking/Acts/FaserActsGeometry/src'

def tool_sources():
    files={}
    for ext in ('h','cxx'):
        s=(OFFICIAL/('FaserActsExtrapolationTool.'+ext)).read_text()
        s=s.replace('FASERACTSGEOMETRY_ACTSEXTRAPOLATIONTOOL_H','WB107_EXTRAPOLATION_TOOL_H')
        s=s.replace('FaserActsExtrapolationTool','WB107ExtrapolationTool')
        s=s.replace('ActsExtrapolationDetail','WB107ExtrapolationDetail')
        if ext=='cxx':
            s=replace_one(s,'#include "WB107ExtrapolationTool.h"','#include "WB107ExtrapolationTool.h"\n#include "BoundTrace.h"')
            s=s.replace('Acts::EigenStepper<>','WB107Trace::Stepper')
            s=replace_one(s,'''      auto result = propagator.propagate(startParameters, target, options);
      if (!result.ok()) {
        ATH_MSG_ERROR("Got error during propagation: " << result.error()''', '''      auto result = propagator.propagate(startParameters, target, options);
      WB107Trace::Json receipt{{"record","propagator_result"},{"ok",result.ok()}};
      if(!result.ok()) {
        receipt["error_category"]=result.error().category().name();
        receipt["error_value"]=result.error().value();receipt["error_message"]=result.error().message();
      }
      WB107Trace::emit(receipt);
      if (!result.ok()) {
        ATH_MSG_ERROR("Got error during propagation: " << result.error()''')
            s+='\nDECLARE_COMPONENT(WB107ExtrapolationTool)\n'
        files['WB107ExtrapolationTool.'+ext]=s
    return files

def generated_source():
    from wb106_contract import generated_source as parent_source,OUT as wb106_out
    original=parent_source()
    if original!=(wb106_out/'isolated_source/WB106Diagnostic/CommonSeedAudit.cxx').read_text():
        raise ValueError('WB106 generated source changed')
    s=original.replace('WB106','WB107')
    s='#include "BoundTrace.h"\n'+s
    s=replace_one(s,'    ATH_CHECK(m_tool.retrieve());','    ATH_CHECK(m_diagnostic.retrieve());\n    ATH_CHECK(m_tool.retrieve());')
    s=replace_one(s,'  ToolHandle<IFaserActsExtrapolationTool> m_tool',
      '  ToolHandle<IFaserActsExtrapolationTool> m_diagnostic{this,"DiagnosticTool",""};\n  ToolHandle<IFaserActsExtrapolationTool> m_tool')
    s=replace_one(s,'    const size_t cid=m_call++;',
      '    if(m_call>124)throw std::runtime_error("WB107 call budget exceeded");\n    const size_t cid=m_call++;')
    s=replace_one(s,'    if(!result.has_value())throw std::runtime_error("official ACTS propagation returned no state");', '''    WB107Trace::requested=target.get();
    WB107Trace::sink=[&](const Json& row){Json copy=row;copy["call_id"]=cid;record(copy);};
    record(Json{{"record","before_diagnostic"},{"call_id",cid}});
    auto diagnostic=m_diagnostic->propagate(ctx,start,*target,distance>=0?Acts::Direction::Forward:Acts::Direction::Backward);
    WB107Trace::sink={};WB107Trace::requested=nullptr;
    Json comparison{{"record","comparison"},{"call_id",cid},{"official_has_value",result.has_value()},
      {"diagnostic_has_value",diagnostic.has_value()}};
    if(result && diagnostic) {
      comparison["official_parameters"]=encode(result->parameters());
      comparison["diagnostic_parameters"]=encode(diagnostic->parameters());
      comparison["official_position"]=encode(result->position(g));
      comparison["diagnostic_position"]=encode(diagnostic->position(g));
      comparison["official_direction"]=encode(result->direction());
      comparison["diagnostic_direction"]=encode(diagnostic->direction());
    }
    record(comparison);
    if(cid==124 && result)throw std::runtime_error("WB107 historical failure not reproduced");
    if(result.has_value()!=diagnostic.has_value())throw std::runtime_error("WB107 diagnostic presence changed");
    if(result && (!(result->parameters().array()==diagnostic->parameters().array()).all() ||
       !(result->position(g).array()==diagnostic->position(g).array()).all() ||
       !(result->direction().array()==diagnostic->direction().array()).all()))
      throw std::runtime_error("WB107 diagnostic changed numerical result");
    if(!result)throw std::runtime_error("official ACTS propagation returned no state");''')
    return s

def files():
    cm=(PARENT/'isolated_source/WB92Diagnostic/CMakeLists.txt').read_text().replace('WB92','WB107')
    cm=replace_one(cm,'CommonSeedAudit.cxx LINK_LIBRARIES','CommonSeedAudit.cxx WB107ExtrapolationTool.cxx LINK_LIBRARIES')
    cm=replace_one(cm,'ActsCore PRIVATE_LINK_LIBRARIES','ActsCore ActsInteropLib PRIVATE_LINK_LIBRARIES')
    d={'CMakeLists.txt':(PARENT/'isolated_source/CMakeLists.txt').read_text().replace('WB92','WB107'),
       'WB107Diagnostic/CMakeLists.txt':cm,'WB107Diagnostic/CommonSeedAudit.cxx':generated_source(),
       'WB107Diagnostic/BoundTrace.h':(ROOT/'research/wb107/BoundTrace.h').read_text()}
    d.update({'WB107Diagnostic/'+k:v for k,v in tool_sources().items()})
    return d

def athena_source():
    s=(ROOT/'scripts/wb92_athena.py').read_text().replace('CompFactory.WB92.CommonSeedAudit','CompFactory.WB107.CommonSeedAudit')
    s=replace_one(s,'acc.addPublicTool(tool)', '''acc.addPublicTool(tool)
diagnostic = CompFactory.WB107ExtrapolationTool('WB107BoundDiagnostic', TrackingGeometryTool=geometry,
    FieldMode='FASER', MaxSteps=10000, MaxStepSize=10., InteractionMultiScatering=False,
    InteractionEloss=False, InteractionRecord=False)
acc.addPublicTool(diagnostic)''')
    return replace_one(s,'ExtrapolationTool=tool,','ExtrapolationTool=tool, DiagnosticTool=diagnostic,')
