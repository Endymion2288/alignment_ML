"""Reversible passive inserts in exact installed RKN body and isolated official tool."""
import hashlib
from alignment.wb90_measurement_contract import ROOT
from wb92_contract import ACTS, EXTERNAL
from wb101_sources import replace_one
from wb120_sources import files as previous_files, athena_source as previous_athena


def original_step():
    source=(ACTS/'include/Acts/Propagator/EigenStepper.ipp').read_text()
    start=source.index('template <typename E, typename A>\ntemplate <typename propagator_state_t, typename navigator_t>\nActs::Result<double> Acts::EigenStepper<E, A>::step(')
    end=source.index('\ntemplate <typename E, typename A>\nvoid Acts::EigenStepper<E, A>::setIdentityJacobian',start)
    return source[start:end]


def step_source_and_proof():
    original=original_step(); source=original; edits=[]
    def edit(old,new):
        nonlocal source
        source=replace_one(source,old,new);edits.append((old,new))
    edit('template <typename E, typename A>\n','')
    edit('Acts::EigenStepper<E, A>::step','WB122Trace::Stepper::step')
    edit('  using namespace UnitLiterals;', '''  using namespace Acts;
  using namespace UnitLiterals;
  if(!sink)return Base::step(state,navigator);
  if(state.stepping.covTransport)throw std::runtime_error("WB122 covariance not authorized");
  ++stepIndex;trialIndex=0;
  emit({{"record","step_begin"},{"step_index",stepIndex-1},
    {"position_mm",encode(position(state.stepping))},{"direction",encode(direction(state.stepping))},
    {"qop_acts",qOverP(state.stepping)},{"time_acts",time(state.stepping)},
    {"path_mm",state.stepping.pathAccumulated},{"constraints",constraints(state.stepping.stepSize)},
    {"step_tolerance",state.options.stepTolerance},{"max_step_mm",state.options.maxStepSize},
    {"max_steps",state.options.maxSteps},{"max_trials",state.options.maxRungeKuttaStepTrials},
    {"nav_direction",state.options.direction.sign()}});''')
    edit('  auto fieldRes = getField(state.stepping, pos);','  queryPhase="FIRST_SHARED";\n  auto fieldRes = getField(state.stepping, pos);')
    edit('    // helpers because bool and std::error_code are ambiguous','    ++trialIndex;\n    // helpers because bool and std::error_code are ambiguous')
    edit('    auto field = getField(state.stepping, pos1);','    queryPhase="MIDDLE";\n    auto field = getField(state.stepping, pos1);')
    edit('    field = getField(state.stepping, pos2);','    queryPhase="LAST";\n    field = getField(state.stepping, pos2);')
    edit('    return success(error_estimate <= state.options.stepTolerance);', '''    emit({{"record","trial"},{"step_index",stepIndex-1},{"trial_index",trialIndex-1},{"h_mm",h},
      {"start_position_mm",encode(pos)},{"start_direction",encode(dir)},{"pos1_mm",encode(pos1)},{"pos2_mm",encode(pos2)},
      {"start_path_mm",state.stepping.pathAccumulated},{"qop_acts",qOverP(state.stepping)},{"time_acts",time(state.stepping)},
      {"B_first_native",encode(sd.B_first)},{"B_middle_native",encode(sd.B_middle)},{"B_last_native",encode(sd.B_last)},
      {"k1",encode(sd.k1)},{"k2",encode(sd.k2)},{"k3",encode(sd.k3)},{"k4",encode(sd.k4)},
      {"kQoP",Json::array({sd.kQoP[0],sd.kQoP[1],sd.kQoP[2],sd.kQoP[3]})},
      {"error_estimate",error_estimate},{"step_tolerance",state.options.stepTolerance},
      {"accepted",error_estimate<=state.options.stepTolerance}});
    return success(error_estimate <= state.options.stepTolerance);''')
    edit('  return h;', '''  emit({{"record","step_end"},{"step_index",stepIndex-1},{"accepted_h_mm",h},
    {"rejections",state.stepping.stepSize.nStepTrials},{"trial_count",trialIndex},
    {"position_mm",encode(position(state.stepping))},{"direction",encode(direction(state.stepping))},
    {"qop_acts",qOverP(state.stepping)},{"time_acts",time(state.stepping)},
    {"path_mm",state.stepping.pathAccumulated},{"next_accuracy_mm",state.stepping.stepSize.accuracy()},
    {"constraints",constraints(state.stepping.stepSize)}});
  return h;''')
    restored=source
    for old,new in reversed(edits):
        if not new: # First template line was removed.
            restored=old+restored
        else:
            restored=replace_one(restored,new,old)
    if restored!=original:raise ValueError('exact RKN arithmetic source restoration')
    sha=lambda s:hashlib.sha256(s.encode()).hexdigest()
    return source,{'original_step_sha256':sha(original),'observed_step_sha256':sha(source),'restored_original_exact':True,
                   'edits':[{'original':old,'replacement':new} for old,new in edits]}


def tool_sources():
    result={}
    for ext in ('h','cxx'):
        s=(EXTERNAL/'Tracking/Acts/FaserActsGeometry/src'/('FaserActsExtrapolationTool.'+ext)).read_text()
        s=s.replace('FASERACTSGEOMETRY_ACTSEXTRAPOLATIONTOOL_H','WB122_EXTRAPOLATION_TOOL_H')
        # Replace the component class token only; preserve official interface type.
        import re
        s=re.sub(r'\bFaserActsExtrapolationTool\b','WB122ExtrapolationTool',s)
        s=s.replace('ActsExtrapolationDetail','WB122ExtrapolationDetail')
        if ext=='h':s=replace_one(s,'#include "Acts/Utilities/Logger.hpp"','#include "Acts/Utilities/Logger.hpp"\n#include "Trace.h"')
        else:
            s=s.replace('Acts::EigenStepper<>','WB122Trace::Stepper')
            old='using ActionList = Acts::ActionList<Acts::MaterialInteractor>;'
            if s.count(old)!=2:raise ValueError('exact two plain ActionLists')
            s=s.replace(old,'using ActionList = Acts::ActionList<WB122Trace::Action, Acts::MaterialInteractor>;')
            marker='''      auto result = propagator.propagate(startParameters, target, options);
      if (!result.ok()) {
        ATH_MSG_ERROR("Got error during propagation: " << result.error()'''
            s=replace_one(s,marker,'''      auto result = propagator.propagate(startParameters, target, options);
      WB122Trace::Json receipt{{"record","propagator_result"},{"ok",result.ok()}};
      if(result.ok()){receipt["steps"]=result->steps;receipt["path_mm"]=result->pathLength;}
      else{receipt["error_category"]=result.error().category().name();receipt["error_value"]=result.error().value();}
      WB122Trace::emit(receipt);
      if (!result.ok()) {
        ATH_MSG_ERROR("Got error during propagation: " << result.error()''')
            s+='\nDECLARE_COMPONENT(WB122ExtrapolationTool)\n'
        result['WB122ExtrapolationTool.'+ext]=s
    return result


def files():
    old=previous_files(); cm=old['WB120Diagnostic/CMakeLists.txt'].replace('WB120','WB122')
    cm=replace_one(cm,'BoundedResponse.cxx LINK_LIBRARIES','BoundedResponse.cxx WB122ExtrapolationTool.cxx LINK_LIBRARIES')
    cm=replace_one(cm,'ActsCore MagFieldConditions','ActsCore ActsInteropLib MagFieldConditions')
    source,proof=step_source_and_proof()
    result={'CMakeLists.txt':old['CMakeLists.txt'].replace('WB120','WB122'),
       'WB122Diagnostic/CMakeLists.txt':cm,'WB122Diagnostic/BoundedResponse.cxx':(ROOT/'research/wb122/BoundedResponse.cxx').read_text(),
       'WB122Diagnostic/Trace.h':(ROOT/'research/wb122/Trace.h').read_text(),'WB122Diagnostic/ObservedStep.inc':source}
    result.update({'WB122Diagnostic/'+k:v for k,v in tool_sources().items()})
    return result


def athena_source():
    s=previous_athena().replace('WB120','WB122')
    s=replace_one(s,'acc.addPublicTool(tool)', '''acc.addPublicTool(tool)
diagnostic = CompFactory.WB122ExtrapolationTool('WB122PassiveTrialTool', TrackingGeometryTool=geometry,
    FieldMode='FASER', MaxSteps=10000, MaxStepSize=10., InteractionMultiScatering=False,
    InteractionEloss=False, InteractionRecord=False)
acc.addPublicTool(diagnostic)''')
    s=replace_one(s,'ExtrapolationTool=tool,','ExtrapolationTool=tool, DiagnosticTool=diagnostic,')
    return replace_one(s,"'max_official_calls': 36","'max_official_calls': 54")
