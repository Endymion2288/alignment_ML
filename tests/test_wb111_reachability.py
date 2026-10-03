"""Synthetic controls and frozen seen receipts; never ROOT or propagation."""
import copy,json,sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from wb111_contract import control,BASE,WORKBOOK
from wb111_sources import files,athena_source
from audit_wb111_reachability import validate

def state(pos,u,q=.01,time=0.,frame=None):
    pos=np.asarray(pos);u=np.asarray(u);t=np.eye(4) if frame is None else np.asarray(frame)
    local=(np.linalg.inv(t)@np.r_[pos,1.])[:3];d=t[:3,:3].T@u
    return {'position_mm':pos[:,None].tolist(),'direction':u[:,None].tolist(),'qop_acts':q,'time_acts':time,
      'covariance_present':False,'local':local[:,None].tolist(),'h':np.r_[local[:2],d[:2]/d[2]][:,None].tolist()}

def sample():
    c=control();f=json.loads((BASE/'fixture.json').read_text());terminal=c['terminal']
    start=state(np.r_[np.asarray(c['seed']).reshape(5)[:2],c['seed_z_mm']],np.asarray(c['targets'][0]['before_official']['start_direction']).reshape(3))
    resumed=state(np.asarray(terminal['position']).reshape(3),np.asarray(terminal['direction']).reshape(3),terminal['qop_acts'],terminal['time'])
    official=[];diag=[]
    for i,t in enumerate(c['targets']):
        o={'station':t['station'],'historical_call_id':t['call_id'],'target_frame':t['frame'],
          'has_value':t['official_has_value'],'start':copy.deepcopy(start)}
        if t['official_has_value']:
            h=np.asarray(t['official_h']).reshape(4);u=np.r_[h[2:],1.];u=u/np.linalg.norm(u)
            o['state']=state([*h[:2],t['frame'][2][3]],u,time=2000.,frame=t['frame']);o['state']['h']=t['official_h']
        official.append(o)
    for i in range(4):
        t=c['targets'][i] if i<3 else c['targets'][2];s=start if i<3 else resumed
        endpoint=state([260.,42.,t['frame'][2][3]],[0.,0.,1.],time=s['time_acts']+100.,frame=t['frame'])
        last={k:copy.deepcopy(endpoint[k]) for k in ('position_mm','direction','qop_acts','time_acts')}
        last['path_mm']=100.;step=copy.deepcopy(last);step.update(query_end=1,rejected_trials=0)
        opts=c['baseline_options'].copy();opts['surfaceTolerance']=opts.pop('surfaceTolerance_mm')
        opts.update(maxRungeKuttaStepTrials=10000,stepSizeCutOff=0.,loopFraction=.5)
        diag.append({'arm':'from_seed' if i<3 else 'from_terminal','station':t['station'],'target_frame':t['frame'],
          'start':copy.deepcopy(s),'options':opts,'navigator':'Acts::VoidNavigator','user_aborters':[],'material_actor':False,
          'last_free':last,'accepted_trace':[step],'accepted_steps':1,'rejected_trials':0,
          'field_queries':[{'position_mm':s['position_mm'],'field_acts':[[0.],[.000299792458],[0.]],'field_T':[[0.],[1.],[0.]]}],
          'field_query_count':1,'field_gradient_count':0,'status':'SUCCESS','state':endpoint,'path_mm':100.,'propagator_steps_counter':1})
    r={'schema':'wb111_reachability_runtime_v1','control_used':copy.deepcopy(c),
      'identity':{k:f[k] for k in ('actual_run','actual_event','input_xaod','ordinal')},
      'official_calls':3,'diagnostic_calls':4,'world':copy.deepcopy(c['topology']['world']),
      'acts_MeV_unit':.001,'acts_T_unit':.000299792458,
      'field_conditions':{'map_key':'fieldMapCondObj','cache_key':'fieldCondObj','map_IOV':'synthetic','cache_IOV':'synthetic',
        'scale':1.,'official_context_is_actual_cache':True},'official':official,'diagnostic':diag}
    return r,c,f

def test_reachable_is_a_diagnostic_claim_and_preserves_measurement_disagreement():
    r,c,f=sample();s=validate(r,c,f)
    assert s['classification']=='TARGET_REACHABLE_WITHOUT_GEOMETRY_NAVIGATION'
    assert s['accuracy_and_sensitive_acceptance']==s['physical_seed_and_association_validity']=='UNKNOWN'
    assert abs(s['rows'][2]['measurement_difference'][0])>100.
    assert s['rows'][2]['saved_topology_containment']['World']['inside'] is True
    assert s['rows'][2]['saved_topology_containment']['ShortDipole_2']['inside'] is False
    assert s['qualification']=='NOT_EVALUATED'

@pytest.mark.parametrize('mutation',['event','source','control','world','seed','qop','time','frame','MeV','T','cache',
  'official_output','official_start','missing_call','option','field_unit','missing_query','gradient','missing_step',
  'free_target','bound_target','h','NaN','navigation','material','reject_count','returned_qop','path'])
def test_changed_identity_or_false_success_rejected(mutation):
    r,c,f=sample();d=r['diagnostic'][0]
    if mutation=='event':r['identity']['actual_event']+=1
    elif mutation=='source':r['identity']['input_xaod']='another.root'
    elif mutation=='control':r['control_used']['seed'][0][0]+=1.
    elif mutation=='world':r['world']['bounds_values'][0]+=1.
    elif mutation=='seed':d['start']['position_mm'][0][0]+=1.
    elif mutation=='qop':d['start']['qop_acts']/=1000.
    elif mutation=='time':r['diagnostic'][3]['start']['time_acts']=0.
    elif mutation=='frame':d['target_frame'][2][3]+=1.
    elif mutation=='MeV':r['acts_MeV_unit']=1.
    elif mutation=='T':r['acts_T_unit']=1.
    elif mutation=='cache':r['field_conditions']['official_context_is_actual_cache']=False
    elif mutation=='official_output':r['official'][0]['state']['h'][0][0]+=1.
    elif mutation=='official_start':r['official'][0]['start']['position_mm'][0][0]+=1e-10
    elif mutation=='missing_call':r['diagnostic'].pop()
    elif mutation=='option':d['options']['stepTolerance']/=10.
    elif mutation=='field_unit':d['field_queries'][0]['field_T'][1][0]*=1000.
    elif mutation=='missing_query':d['field_queries']=[]
    elif mutation=='gradient':d['field_gradient_count']=1
    elif mutation=='missing_step':d['accepted_trace']=[]
    elif mutation=='free_target':d['last_free']['position_mm'][2][0]+=1.
    elif mutation=='bound_target':d['state']['position_mm'][2][0]+=1.
    elif mutation=='h':d['state']['h'][2][0]+=1.
    elif mutation=='NaN':d['field_queries'][0]['position_mm'][0][0]=float('nan')
    elif mutation=='navigation':d['navigator']='Acts::Navigator'
    elif mutation=='material':d['material_actor']=True
    elif mutation=='reject_count':d['rejected_trials']=1
    elif mutation=='returned_qop':d['state']['qop_acts']/=1000.
    elif mutation=='path':d['path_mm']+=1.
    with pytest.raises(ValueError):validate(r,c,f)

@pytest.mark.parametrize('index',range(4))
def test_diagnostic_error_is_reported_without_changing_historical_failure(index):
    r,c,f=sample();d=r['diagnostic'][index];d['status']='ERROR';d['error_message']='synthetic propagation error';d.pop('state')
    s=validate(r,c,f);assert s['classification']=='NOT_DEMONSTRATED'
    assert s['official_guard']=='PASS' and s['rows'][index]['status']=='ERROR'

def test_source_and_runner_use_frozen_field_and_one_seen_event():
    s=files()['WB111Diagnostic/NavigationReachability.cxx'];runner=athena_source()
    assert 'm_official.getField(x,c)' in s and 'ConstantBField' not in s
    assert 'Acts::EigenStepper<>,Acts::VoidNavigator' in s and 'Acts::AbortList<>' in s
    assert 'ActionList<Acts::MaterialInteractor' not in s
    assert s.index('Json official=')<s.index('auto field=std::make_shared<Field>()')
    assert 'call(continuation' in s and 'for(const auto& t:m_control.at("targets"))call(start' in s
    assert 'glueTrackingVolume' not in s and 'attachVolume(' not in s
    assert "flags.Exec.SkipEvents = fixture['ordinal']; flags.Exec.MaxEvents = 1" in runner
    assert WORKBOOK.name.startswith('2026-10-03_111_')
    compile(runner,'wb111_athena.py','exec')
