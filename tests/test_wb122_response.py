"""Synthetic complete trial/cache/target trace and fail-closed negative controls."""
import copy,sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_wb122_response import analyze,trace_check,projection,cache_validate,ORDER,ARMS

def column(x):return [[v] for v in x]
def synthetic():
    seed=column([0.,0.,0.,0.,1e-5]);samples=[];targets=[]
    for sid,(m,s) in enumerate(ORDER):samples.append({'sample_id':sid,'step_multiple':m,'sign':s,'offset_multiple':m*s,'seed':[0.,0.,0.,0.,1e-5+m*s*1e-8]})
    for station in (1,2):
        frame=np.eye(4);frame[2,3]=station;targets.append({'station':station,'frame':frame.tolist(),'official_h':column([0.,0.,0.,0.])})
    control={'seed':seed,'seed_z_mm':0.,'targets':targets,'output_scales':[1.,1.,1.,1.],'qop_step_per_MeV':1e-8,'samples':samples,'max_official_calls':54,'calls_per_arm':18,'observer_budget_per_call':200000,'observer_budget_total':3600000}
    fixture={'index':12,'ordinal':2268,'actual_run':100044,'actual_event':2268,'input_xaod':'synthetic_seen_fixture'}
    r={'schema':'wb122_complete_rk_runtime_v1','control_used':control,'official_calls':54,'rows':[],'identity':{k:fixture[k] for k in ('actual_run','actual_event','input_xaod','ordinal')},'world':{},'field_conditions':{},'sensors':[],'acts_MeV_unit':.001,'acts_T_unit':.000299792458,'observer_records':0}
    old={k:copy.deepcopy(r[k]) for k in ('identity','world','field_conditions','sensors','acts_MeV_unit','acts_T_unit','control_used')};old['rows']=[]
    for sample in samples:
      sid=sample['sample_id'];qop=sample['seed'][4]/.001
      for target in targets:
        st=target['station'];h=float(st);p0=column([0.,0.,0.]);p1=column([0.,0.,h]);u=column([0.,0.,1.]);zero=column([0.,0.,0.]);params=column([0.,0.,0.,0.,qop,h])
        start={'h':column([0.,0.,0.,0.]),'global_position_mm':p0,'global_direction':u,'time_acts':0.,'qop_acts':qop,'covariance_present':False}
        state={**start,'global_position_mm':p1,'local_position_mm':column([0.,0.,0.]),'time_acts':h}
        inp={'sample_id':sid,'station':st,'step_multiple':sample['step_multiple'],'sign':sample['sign'],'factor':sample['offset_multiple'],'seed':column(sample['seed']),'seed_z_mm':0.,'frame':target['frame'],'navigation_direction':1,'role':'POSITIVE_QOP_DIAGNOSTIC','start_state':start}
        constraints={'actor_mm':h,'aborter_mm':1000.,'user_mm':10000.,'accuracy_mm':None,'effective_mm':h}
        initial={'record':'action','stage':'prePropagation','accepted_step_count':0,'position_mm':p0,'direction':u,'logger_direction':u,'momentum_MeV':column([0.,0.,1/sample['seed'][4]]),'qop_acts':qop,'time_acts':0.,'path_mm':0.,'target_reached':False,'navigation_break':False,'geometry_id':0,'surface_present':False,'surface_geometry_id':None,'navigation_direction':1,'rk_rejections':None,'constraints':constraints}
        begin={'record':'step_begin','step_index':0,'position_mm':p0,'direction':u,'qop_acts':qop,'time_acts':0.,'path_mm':0.,'constraints':constraints,'step_tolerance':1e-4,'max_step_mm':10000.,'max_steps':10000,'max_trials':10000,'nav_direction':1}
        c={'valid_cell':True,'ranges_mm':[[-100.,100.]]*3,'contains_query':True,'field_scale':1.,'scale_to_use':1.,'actual_condition_map_match':True,'float_inverse_widths':[float(np.float32(.005))]*3,'float_corner_fields':[[0.]*8]*3,'float_bscale_kT':1.}
        qs=[]
        for qi,(phase,pos) in enumerate(zip(('FIRST_SHARED','MIDDLE','LAST'),(p0,column([0.,0.,h*.5]),p1))):qs.append({'record':'field_query','step_index':0,'trial_index':None if qi==0 else 0,'query_index':qi,'query_phase':phase,'position_mm':pos,'cache_before':c,'cache_after':c,'cache_hit_before':True,'cache_refilled':False,'outside_map_fallback':False,'ok':True,'field_native':zero,'field_T':zero})
        trial={'record':'trial','step_index':0,'trial_index':0,'h_mm':h,'start_position_mm':p0,'start_direction':u,'pos1_mm':qs[1]['position_mm'],'pos2_mm':p1,'start_path_mm':0.,'qop_acts':qop,'time_acts':0.,'B_first_native':zero,'B_middle_native':zero,'B_last_native':zero,'k1':zero,'k2':zero,'k3':zero,'k4':zero,'kQoP':[0.]*4,'error_estimate':1e-20,'step_tolerance':1e-4,'accepted':True}
        end={'record':'step_end','step_index':0,'accepted_h_mm':h,'rejections':0,'trial_count':1,'position_mm':p1,'direction':u,'qop_acts':qop,'time_acts':h,'path_mm':h,'next_accuracy_mm':h*4.,'constraints':constraints}
        post={**initial,'stage':'postStep','accepted_step_count':1,'position_mm':p1,'time_acts':h,'path_mm':h,'target_reached':False,'rk_rejections':0}
        bb={'record':'bound_before','requested_target':True,'frame':target['frame'],'position_mm':p1,'direction':u,'qop_acts':qop,'time_acts':h,'path_mm':h,'cov_transport':False}
        ba={'record':'bound_after','ok':True,'parameters':params,'position_mm':p1,'direction':u};result={'record':'propagator_result','ok':True,'steps':0,'path_mm':h}
        template=[initial,begin,*qs,trial,end,post,{**post,'stage':'postPropagation','target_reached':True},bb,ba,result]
        for arm in ARMS:
          call=len(r['rows'])+1;obs=copy.deepcopy(template) if arm=='observer_enabled' else []
          for j,v in enumerate(obs):v.update(observer_record_index=j,call_id=call,global_record_index=r['observer_records']+j)
          r['observer_records']+=len(obs);r['rows'].append({'input':{**copy.deepcopy(inp),'arm':arm,'call_id':call},'has_value':True,'state':copy.deepcopy(state),'bound_parameters':copy.deepcopy(params),'observations':obs,'observer_record_count':len(obs)})
        old['rows'].extend([{'input':copy.deepcopy(inp),'state':copy.deepcopy(state)},{'snapshots':[projection(initial,0),projection(post,1)]}])
    return r,control,fixture,old

def test_full_matrix_complete_fidelity():
    data=synthetic();saved=copy.deepcopy(data);s=analyze(*data)
    assert data==saved and s['hypothesis']=='SUPPORTED_BUT_LIMITED_OBSERVATION_FIDELITY'
    assert len(s['profiles'])==18 and len(s['comparisons'])==36 and s['qualification']=='NOT_EVALUATED'
    assert s['profiles'][0]['public_snapshots']==2 and s['profiles'][0]['accepted_steps']==1

@pytest.mark.parametrize('bad',['missing_query','missing_end','duplicate_trial','acceptance','k_sign','position','qop','map','scale','cell','inverse','unit','counter','time','final_frame','bound_parameters','ordering','sentinel','cache_branch','nonfinite','budget'])
def test_complete_trace_negative_controls(bad):
    r,c,f,h=synthetic();row=r['rows'][2];obs=row['observations']
    if bad=='missing_query':obs.pop(3)
    elif bad=='missing_end':obs.pop(6)
    elif bad=='duplicate_trial':obs.insert(6,copy.deepcopy(obs[5]))
    elif bad=='acceptance':obs[5]['accepted']=False
    elif bad=='k_sign':obs[5]['k1'][0][0]=.01
    elif bad=='position':obs[5]['pos1_mm'][0][0]=1.
    elif bad=='qop':obs[5]['qop_acts']=1.
    elif bad=='map':obs[2]['cache_before']['actual_condition_map_match']=False
    elif bad=='scale':obs[2]['cache_before']['field_scale']=2.
    elif bad=='cell':obs[2]['cache_before']['contains_query']=False
    elif bad=='inverse':obs[2]['cache_before']['float_inverse_widths'][0]=1.
    elif bad=='unit':obs[2]['field_T'][0][0]=1.
    elif bad=='counter':obs[6]['rejections']=1
    elif bad=='time':obs[6]['time_acts']=0.
    elif bad=='final_frame':obs[9]['requested_target']=False
    elif bad=='bound_parameters':obs[10]['parameters'][0][0]=10.
    elif bad=='ordering':obs[3],obs[4]=obs[4],obs[3]
    elif bad=='sentinel':obs[0]['rk_rejections']=0
    elif bad=='cache_branch':obs[2]['cache_refilled']=True
    elif bad=='nonfinite':obs[5]['error_estimate']=float('nan')
    elif bad=='budget':obs[1]['max_trials']=99
    with pytest.raises(ValueError):trace_check(obs,row,h['rows'][1]['snapshots'],r['acts_T_unit'])

@pytest.mark.parametrize('bad',['endpoint','public_projection','coverage'])
def test_fidelity_differences_are_unknown_not_pass_or_physics_fail(bad):
    r,c,f,h=synthetic()
    if bad=='endpoint':r['rows'][1]['state']['time_acts']+=1.
    elif bad=='public_projection':h['rows'][1]['snapshots'][0]['geometry_id']=9
    else:r['rows'][2]['observations'].pop(5);r['rows'][2]['observer_record_count']-=1;r['observer_records']-=1
    assert analyze(r,c,f,h)['hypothesis']=='UNKNOWN_FIDELITY_OR_COVERAGE'

def test_invalid_cache_never_reads_uninitialized_and_closed_boundary():
    c={'valid_cell':False,'ranges_mm':[[0.,0.],[0.,0.],[0.,-1.]],'contains_query':False,'field_scale':1.,'scale_to_use':1.,'actual_condition_map_match':True,'float_inverse_widths':None,'float_corner_fields':None,'float_bscale_kT':None}
    assert cache_validate(c,np.zeros(3)) is False
    c['float_corner_fields']=[[0.]*8]*3
    with pytest.raises(ValueError):cache_validate(c,np.zeros(3))

def test_generated_source_is_exactly_reversible_and_official_interface_preserved():
    from wb122_sources import files,step_source_and_proof
    source,proof=step_source_and_proof();assert proof['restored_original_exact']
    result=files();assert 'IFaserActsExtrapolationTool' in result['WB122Diagnostic/WB122ExtrapolationTool.h']
    assert result['WB122Diagnostic/WB122ExtrapolationTool.cxx'].count('Acts::ActionList<WB122Trace::Action, Acts::MaterialInteractor>')==2
    assert 'if(!sink)return Base::step(state,navigator);' in source

def rejection_trace():
    r,c,f,old=synthetic();row=r['rows'][2];obs=row['observations'];initial,begin=obs[:2];trial=copy.deepcopy(obs[5]);end=copy.deepcopy(obs[6]);post=copy.deepcopy(obs[7]);final=copy.deepcopy(obs[8]);bb=copy.deepcopy(obs[9]);ba=copy.deepcopy(obs[10]);receipt=copy.deepcopy(obs[11]);q0=copy.deepcopy(obs[2])
    u=np.array([0.,0.,1.]);p=np.zeros(3);qop=begin['qop_acts'];T=r['acts_T_unit'];B=np.array([.1,0.,0.]);h=500.;new=[initial,begin]
    begin['constraints']['actor_mm']=begin['constraints']['effective_mm']=h
    cell={'valid_cell':True,'ranges_mm':[[-1000.,1000.]]*3,'contains_query':True,'field_scale':1.,'scale_to_use':1.,'actual_condition_map_match':True,'float_inverse_widths':[float(np.float32(.0005))]*3,'float_corner_fields':[[float(np.float32(.1/(1000*T)))]*8,[0.]*8,[0.]*8],'float_bscale_kT':1.}
    q0.update(cache_before=cell,cache_after=cell,field_native=column(B),field_T=column(B/T));new.append(q0);ts=[];qi=1
    while True:
        k1=qop*np.cross(u,B);k2=qop*np.cross(u+h*.5*k1,B);k3=qop*np.cross(u+h*.5*k2,B);k4=qop*np.cross(u+h*k3,B)
        p1=p+h*.5*u+h*h*.125*k1;p2=p+h*u+h*h*.5*k3;e=max(h*h*np.sum(np.abs(k1-k2-k3+k4)),1e-20)
        t=copy.deepcopy(trial);t.update(trial_index=len(ts),h_mm=h,pos1_mm=column(p1),pos2_mm=column(p2),error_estimate=float(e),accepted=bool(e<=1e-4),B_first_native=column(B),B_middle_native=column(B),B_last_native=column(B))
        for j,k in enumerate((k1,k2,k3,k4)):t['k'+str(j+1)]=column(k)
        for phase,position in (('MIDDLE',p1),('LAST',p2)):
            q=copy.deepcopy(q0);q.update(query_phase=phase,trial_index=len(ts),query_index=qi,position_mm=column(position));qi+=1;new.append(q)
        new.append(t);ts.append(t)
        if e<=1e-4:break
        h*=float(np.clip(np.sqrt(np.sqrt(np.float32(1e-4/abs(2*e)))),np.float32(.25),np.float32(4.)))
    pend=p+h*u+h*h/6.*(k1+k2+k3);uend=u+h/6.*(k1+2*(k2+k3)+k4);uend/=np.linalg.norm(uend)
    end.update(accepted_h_mm=float(h),trial_count=len(ts),rejections=len(ts)-1,position_mm=column(pend),direction=column(uend),path_mm=float(h),time_acts=float(h))
    for a in (post,final):a.update(position_mm=column(pend),direction=column(uend),logger_direction=column(uend),momentum_MeV=column(uend/row['input']['seed'][4][0]),rk_rejections=len(ts)-1,path_mm=float(h),time_acts=float(h))
    bb.update(position_mm=column(pend),direction=column(uend),path_mm=float(h),time_acts=float(h));ba.update(position_mm=column(pend),direction=column(uend));receipt['path_mm']=float(h)
    row['state'].update(global_position_mm=column(pend),global_direction=column(uend),time_acts=float(h))
    new.extend([end,post,final,bb,ba,receipt])
    for i,x in enumerate(new):x.update(observer_record_index=i,call_id=row['input']['call_id'])
    snaps=[projection(initial,0),projection(post,1)]
    return new,row,snaps,T

def test_rejected_trials_share_first_field_and_close_actual_counter():
    obs,row,snaps,T=rejection_trace();s=trace_check(obs,row,snaps,T)
    assert s['rejections']>0 and s['field_queries']==1+2*s['trials'] and s['public_projection_exact']

@pytest.mark.parametrize('bad',['lost_rejected_query','reject_scaling','trial_index','false_acceptance'])
def test_rejected_trial_negative_controls(bad):
    obs,row,snaps,T=rejection_trace()
    if bad=='lost_rejected_query':obs.pop(3)
    elif bad=='reject_scaling':[x for x in obs if x['record']=='trial'][1]['h_mm']*=2.
    elif bad=='trial_index':[x for x in obs if x['record']=='trial'][1]['trial_index']=99
    else:[x for x in obs if x['record']=='trial'][0]['accepted']=True
    with pytest.raises(ValueError):trace_check(obs,row,snaps,T)
