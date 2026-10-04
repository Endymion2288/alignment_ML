#!/usr/bin/env python3
"""Complete saved passive RK/cache trace verification; no ROOT or physical calls."""
import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import read_public
from audit_wb117_response import check,array
from wb121_recovery import load_analyzer
legacy=load_analyzer()
ORDER=legacy.ORDER
ARMS=('official','observer_disabled','observer_enabled')
BUDGET=1e-8

def finite_tree(value):
    if isinstance(value,dict):
        for v in value.values():finite_tree(v)
    elif isinstance(value,list):
        for v in value:finite_tree(v)
    elif isinstance(value,float):check(np.isfinite(value),'nonfinite scalar')

def near(a,b,label):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    check(a.shape==b.shape and np.all(np.isfinite(a)) and np.all(np.isfinite(b)),label+' shape/finite')
    delta=float(np.max(np.abs(a-b)/np.maximum(1.,np.maximum(np.abs(a),np.abs(b)))))
    check(delta<=BUDGET,label+' independent arithmetic')
    return delta

def vector(r,key):return array(r[key],(3,1)).reshape(3)

def projection(action,i):
    return {'snapshot_index':i,'global_position_mm':action['position_mm'],'global_direction':action['logger_direction'],
      'momentum_MeV':action['momentum_MeV'],'geometry_id':action['geometry_id'],'surface_present':action['surface_present'],
      'surface_geometry_id':action['surface_geometry_id'],'navigation_direction':action['navigation_direction'],
      'rk_counter_unset':action['rk_rejections'] is None,
      'rk_rejections_before_previous_accepted_step':action['rk_rejections'],'step_constraints':action['constraints']}

def cache_validate(c,p):
    check(c['actual_condition_map_match'] is True and c['field_scale']==1. and c['scale_to_use']==1.,'actual cache map/scale')
    ranges=array(c['ranges_mm'],(3,2));valid=ranges[2,0]<=ranges[2,1] and ranges[0,0]<ranges[0,1] and ranges[1,0]<ranges[1,1]
    check(type(c['valid_cell']) is bool and c['valid_cell']==valid,'cache valid ranges')
    inside=bool(np.all(p>=ranges[:,0]) and np.all(p<=ranges[:,1]));check(c['contains_query'] is inside,'actual closed cell containment')
    if valid:
        inv=array(c['float_inverse_widths'],(3,));corners=array(c['float_corner_fields'],(3,8))
        check(np.array_equal(inv,np.asarray(1./(ranges[:,1]-ranges[:,0]),dtype=np.float32).astype(float)),'actual float inverse width')
        check(type(c['float_bscale_kT']) in (int,float) and np.isfinite(c['float_bscale_kT']),'cache scale finite')
    else:
        check(all(c[k] is None for k in ('float_inverse_widths','float_corner_fields','float_bscale_kT')),'invalid cell unread float guards')
    return inside

def query_validate(q,T):
    check(q['ok'] is True,'field query error; incomplete coverage')
    p=vector(q,'position_mm');before=cache_validate(q['cache_before'],p);after=cache_validate(q['cache_after'],p)
    check(q['cache_hit_before'] is before and q['cache_refilled'] is (not before and after) and q['outside_map_fallback'] is (not after),'cache branch flags')
    near(vector(q,'field_native')/T,vector(q,'field_T'),'ACTS tesla conversion')
    if before:check(q['cache_before']==q['cache_after'],'hit must not mutate cache')
    if after:
        c=q['cache_after'];ranges=np.asarray(c['ranges_mm']);inv=np.asarray(c['float_inverse_widths'],dtype=np.float32)
        f=np.asarray((p-ranges[:,0])*inv,dtype=np.float32);g=np.asarray(1.-f,dtype=np.float32)
        fields=np.asarray(c['float_corner_fields'],dtype=np.float32);fx,fy,fz=f;gx,gy,gz=g
        kt=np.float32(c['float_bscale_kT'])*(gx*(gy*(gz*fields[:,0]+fz*fields[:,1])+fy*(gz*fields[:,2]+fz*fields[:,3]))+fx*(gy*(gz*fields[:,4]+fz*fields[:,5])+fy*(gz*fields[:,6]+fz*fields[:,7])))
        near(vector(q,'field_native'),kt.astype(float)*1000.*T,'actual float cache interpolation')
    if not after:near(vector(q,'field_T'),np.full(3,1e-5),'actual 0.1 gauss fallback')

def trace_check(obs,row,old_snaps,T):
    finite_tree(obs);check(0<len(obs)<=200000,'trace budget')
    check([x['observer_record_index'] for x in obs]==list(range(len(obs))),'observer local indices')
    check(all(x['call_id']==row['input']['call_id'] for x in obs),'trace call identity')
    bytype={}
    for x in obs:bytype.setdefault(x['record'],[]).append(x)
    check(set(bytype)=={'action','step_begin','field_query','trial','step_end','bound_before','bound_after','propagator_result'},'complete record type set')
    begins=bytype['step_begin'];ends=bytype['step_end'];actions=bytype['action'];queries=bytype['field_query'];trials=bytype['trial']
    n=len(begins);check(0<n<=10000 and len(ends)==n,'complete accepted step coverage')
    check([x['step_index'] for x in begins]==[x['step_index'] for x in ends]==list(range(n)),'ordered step indices')
    check([x['query_index'] for x in queries]==list(range(len(queries))),'all field query indices')
    check(actions[0]['stage']=='prePropagation' and actions[0]['accepted_step_count']==0 and actions[0]['rk_rejections'] is None,'initial action')
    check([x['accepted_step_count'] for x in actions if x['stage']=='postStep']==list(range(1,n+1)),'all postStep including target')
    check(actions[-1]['stage']=='postPropagation' and actions[-1]['target_reached'] is True,'target final action')
    check(all(x['stage'] in ('prePropagation','postStep','postPropagation') for x in actions),'action stages')
    projected=[projection(x,i) for i,x in enumerate(a for a in actions if not a['target_reached'])]
    for i,x in enumerate(projected):legacy.validate_snapshot(x,i,row['input']['navigation_direction'])
    public_exact=projected==old_snaps
    maxerr=0.;path=0.;previous=None;cursor=0;step_profiles=[]
    # Exact serialization order, including all trials and target-last-step; no ordinal nearest matching.
    check(obs[0]==actions[0],'first record action');cursor=1
    for i,(begin,end) in enumerate(zip(begins,ends)):
        check(obs[cursor]==begin,'ordered step_begin');cursor+=1
        check(begin['max_steps']==10000 and begin['max_trials']==10000 and begin['step_tolerance']==1e-4 and begin['max_step_mm']==10000.,'unchanged ACTS options')
        check(begin['qop_acts']==row['input']['start_state']['qop_acts'],'constant q/p')
        legacy.validate_snapshot(projection({**actions[0],'constraints':begin['constraints']},0),0,row['input']['navigation_direction'])
        if previous is not None:
            for key in ('position_mm','direction','qop_acts','time_acts','path_mm'):check(begin[key]==previous[key],'adjacent free state '+key)
        else:
            check(begin['path_mm']==0. and begin['time_acts']==0.,'initial path/time')
            check(begin['position_mm']==row['input']['start_state']['global_position_mm'],'initial free position')
            check(begin['direction']==row['input']['start_state']['global_direction'],'initial free direction')
        qs=[x for x in queries if x['step_index']==i];ts=[x for x in trials if x['step_index']==i]
        check(0<len(ts)<=10002 and len(qs)==1+2*len(ts),'complete shared first/two queries per trial')
        check([t['trial_index'] for t in ts]==list(range(len(ts))),'trial indices')
        check(qs[0]['query_phase']=='FIRST_SHARED' and qs[0]['trial_index'] is None and qs[0]['position_mm']==begin['position_mm'],'first shared query')
        check(obs[cursor]==qs[0],'ordered first query');cursor+=1
        p=vector(begin,'position_mm');u=vector(begin,'direction');qop=begin['qop_acts'];last_h=None
        for j,t in enumerate(ts):
            middle,last=qs[1+2*j:3+2*j]
            check(middle['query_phase']=='MIDDLE' and last['query_phase']=='LAST' and middle['trial_index']==last['trial_index']==j,'actual trial queries')
            check(obs[cursor:cursor+3]==[middle,last,t],'ordered two queries/full trial');cursor+=3
            check(t['start_position_mm']==begin['position_mm'] and t['start_direction']==begin['direction'] and t['qop_acts']==qop and t['time_acts']==begin['time_acts'] and t['start_path_mm']==begin['path_mm'],'trial same free start')
            check(t['B_first_native']==qs[0]['field_native'] and t['B_middle_native']==middle['field_native'] and t['B_last_native']==last['field_native'],'actual stage field identity')
            check(t['pos1_mm']==middle['position_mm'] and t['pos2_mm']==last['position_mm'],'actual stage query position')
            h=t['h_mm'];k=[vector(t,'k'+str(z)) for z in (1,2,3,4)]
            b=[vector(t,'B_'+z+'_native') for z in ('first','middle','last')]
            for observed,expected in ((k[0],qop*np.cross(u,b[0])),(k[1],qop*np.cross(u+h*.5*k[0],b[1])),(k[2],qop*np.cross(u+h*.5*k[1],b[1])),(k[3],qop*np.cross(u+h*k[2],b[2])),(vector(t,'pos1_mm'),p+h*.5*u+h*h*.125*k[0]),(vector(t,'pos2_mm'),p+h*u+h*h*.5*k[2])):
                maxerr=max(maxerr,near(observed,expected,'RKN stages'))
            check(t['kQoP']==[0.,0.,0.,0.],'default extension no material kQoP')
            estimate=max(h*h*(np.sum(np.abs(k[0]-k[1]-k[2]+k[3]))+abs(t['kQoP'][0]-t['kQoP'][1]-t['kQoP'][2]+t['kQoP'][3])),1e-20)
            maxerr=max(maxerr,near(t['error_estimate'],estimate,'RKN error estimate'))
            check(t['step_tolerance']==1e-4 and t['accepted'] is (t['error_estimate']<=1e-4) and t['accepted'] is (j==len(ts)-1),'original acceptance decision')
            if j==0:check(h==begin['constraints']['effective_mm']*begin['nav_direction'],'actual initial trial h')
            else:
                scale=float(np.clip(np.sqrt(np.sqrt(np.float32(1e-4/abs(2.*ts[j-1]['error_estimate'])))),np.float32(.25),np.float32(4.)))
                maxerr=max(maxerr,near(h,last_h*scale,'default rejected float scaling'))
            last_h=h
        check(obs[cursor]==end,'ordered step_end');cursor+=1
        check(end['accepted_h_mm']==last_h and end['rejections']==len(ts)-1 and end['trial_count']==len(ts),'complete trial/counter closure')
        maxerr=max(maxerr,near(vector(end,'position_mm'),p+h*u+h*h/6.*(k[0]+k[1]+k[2]),'accepted position'))
        uend=u+h/6.*(k[0]+2.*(k[1]+k[2])+k[3]);uend/=np.linalg.norm(uend)
        maxerr=max(maxerr,near(vector(end,'direction'),uend,'accepted direction'))
        check(end['qop_acts']==qop and end['time_acts']>begin['time_acts'],'accepted qop/time')
        path+=h;maxerr=max(maxerr,near(end['path_mm'],path,'actual accepted arc sum'))
        previous=end
        post=[a for a in actions if a['stage']=='postStep' and a['accepted_step_count']==i+1];check(len(post)==1,'postStep unique')
        check(obs[cursor]==post[0],'ordered postStep');cursor+=1
        for key in ('position_mm','direction','qop_acts','time_acts','path_mm'):check(post[0][key]==end[key],'postStep same free state '+key)
        check(post[0]['rk_rejections']==end['rejections'],'public counter closure')
        step_profiles.append({'step_index':i,'accepted_h_mm':h,'start_path_mm':begin['path_mm'],'end_path_mm':end['path_mm'],
          'trials':len(ts),'rejections':len(ts)-1,'error_estimates':[t['error_estimate'] for t in ts],
          'cache_branch_sequence':[[q['query_phase'],q['trial_index'],q['cache_hit_before'],q['cache_refilled'],q['outside_map_fallback'],q['cache_after']['ranges_mm']] for q in qs]})
    check(obs[cursor]==actions[-1],'postPropagation order');cursor+=1
    check(len(bytype['bound_before'])==len(bytype['bound_after'])==len(bytype['propagator_result'])==1,'unique target/result records')
    bb=bytype['bound_before'][0];ba=bytype['bound_after'][0];result=bytype['propagator_result'][0]
    check(obs[cursor:]==[bb,ba,result],'complete target bound/result order')
    check(bb['requested_target'] is True and bb['cov_transport'] is False and bb['frame']==row['input']['frame'],'actual target frame/no covariance')
    for key in ('position_mm','direction','qop_acts','time_acts','path_mm'):check(bb[key]==ends[-1][key],'target before exact free state '+key)
    check(ba['ok'] is True and result['ok'] is True and result['steps']==n-1 and result['path_mm']==ends[-1]['path_mm'],'actual returned propagation receipt')
    check(ba['parameters']==row['bound_parameters'] and ba['position_mm']==row['state']['global_position_mm'] and ba['direction']==row['state']['global_direction'],'actual bound return exact')
    for q in queries:query_validate(q,T)
    finalpost=[a for a in actions if a['stage']=='postStep'][-1]
    return {'call_id':row['input']['call_id'],'sample_id':row['input']['sample_id'],'station':row['input']['station'],
      'accepted_steps':n,'trials':len(trials),'field_queries':len(queries),'rejections':len(trials)-n,
      'cache_refills':sum(q['cache_refilled'] for q in queries),'cache_hits':sum(q['cache_hit_before'] for q in queries),
      'outside_fallbacks':sum(q['outside_map_fallback'] for q in queries),'public_snapshots':len(projected),'public_projection_exact':public_exact,
      'max_normalized_arithmetic_difference':maxerr,'target_last_poststep_reached_flag':finalpost['target_reached'],'propagator_zero_based_final_step_index':result['steps'],'target_last_step':ends[-1], 'target_bound_before':bb,'target_bound_after':ba,
      'step_profiles':step_profiles,'complete_order_and_counters':'PASS'}

def analyze(runtime,control,fixture,historical):
    finite_tree(runtime);check(runtime['schema']=='wb122_complete_rk_runtime_v1' and runtime['control_used']==control,'runtime schema/control')
    check((fixture['index'],fixture['ordinal'],fixture['actual_run'],fixture['actual_event'])==(12,2268,100044,2268),'single seen event')
    check(runtime['identity']==historical['identity']=={k:fixture[k] for k in ('actual_run','actual_event','input_xaod','ordinal')},'event identity')
    for key in ('world','sensors','field_conditions','acts_T_unit','acts_MeV_unit'):check(runtime[key]==historical[key],'historical '+key)
    for key in ('seed','seed_z_mm','targets','output_scales','qop_step_per_MeV','samples'):check(control[key]==historical['control_used'][key],'original controls '+key)
    check(control['max_official_calls']==54 and control['calls_per_arm']==18 and control['observer_budget_per_call']==200000 and control['observer_budget_total']==3600000,'frozen budget')
    check(runtime['official_calls']==len(runtime['rows'])==54,'full arm matrix')
    responses=[];profiles=[];unknown=[]
    for sid,(multiple,sign) in enumerate(ORDER):
        seed=np.asarray(control['samples'][sid]['seed']);expected=array(control['seed'],(5,1)).reshape(5).copy();expected[4]+=multiple*sign*1e-8
        check(np.array_equal(seed,expected) and expected[4]>0 and control['samples'][sid]['sample_id']==sid,'original seeds')
        for ti,target in enumerate(control['targets']):
            check(target['station']==ti+1,'station order');triplet=runtime['rows'][6*sid+3*ti:6*sid+3*ti+3]
            oldplain=historical['rows'][4*sid+2*ti];oldtrace=historical['rows'][4*sid+2*ti+1]
            for ai,row in enumerate(triplet):
                inp=row['input'];call=6*sid+3*ti+ai+1
                check((inp['call_id'],inp['arm'],inp['sample_id'],inp['station'],inp['step_multiple'],inp['sign'])==(call,ARMS[ai],sid,ti+1,multiple,sign),'call arm ordering')
                for key in ('seed','seed_z_mm','frame','navigation_direction','role','factor','start_state'):check(inp[key]==oldplain['input'][key],'historical input '+key)
                check(type(row['has_value']) is bool and row['observer_record_count']==len(row['observations']),'optional/record count')
                if ai!=2:check(row['observations']==[],'disabled/official no instrumentation')
            faithful=all(r['has_value'] for r in triplet) and all(triplet[0].get('state')==r.get('state') and triplet[0].get('bound_parameters')==r.get('bound_parameters') for r in triplet[1:])
            h_exact=triplet[0].get('state',{}).get('h')==oldplain['state']['h']
            responses.append({'sample_id':sid,'station':ti+1,'three_arm_state_exact':faithful,'historical_h_exact':h_exact})
            if not faithful or not h_exact:unknown.append({'sample_id':sid,'station':ti+1,'reason':'ENDPOINT_FIDELITY'})
            try:
                profile=trace_check(triplet[2]['observations'],triplet[2],oldtrace['snapshots'],runtime['acts_T_unit']);profiles.append(profile)
                if not profile['public_projection_exact']:unknown.append({'sample_id':sid,'station':ti+1,'reason':'PUBLIC_PROJECTION_DIFFERENCE'})
            except (ValueError,KeyError,IndexError,TypeError) as e:unknown.append({'sample_id':sid,'station':ti+1,'reason':'COVERAGE_OR_ARITHMETIC','detail':str(e)})
    check(sum(r['observer_record_count'] for r in runtime['rows'])==runtime['observer_records']<=3600000,'whole observer count')
    mapped={(p['station'],*ORDER[p['sample_id']]):p for p in profiles};pairs=[]
    for st in (1,2):
        choices=[('versus_nominal',(0.,0),v) for v in ORDER[1:]]+[('plus_minus',(m,-1),(m,1)) for m in (.25,.5,1.,2.)]+[('versus_full_step',(1.,s),(m,s)) for s in (1,-1) for m in (.25,.5,2.)]
        for role,a,b in choices:
            item={'station':st,'role':role,'reference_step_sign':list(a),'compared_step_sign':list(b),'status':'UNKNOWN_MISSING_COVERAGE','alignment':'ORDINAL_STEPS_NOT_EQUAL_ARC'}
            if (st,*a) in mapped and (st,*b) in mapped:
                pa=mapped[(st,*a)];pb=mapped[(st,*b)];sa=pa['step_profiles'];sb=pb['step_profiles']
                def divergence(key):return legacy.first_difference([x[key] for x in sa],[x[key] for x in sb])
                idx=divergence('rejections');cacheidx=divergence('cache_branch_sequence')
                item.update(status='DIAGNOSTIC_ORDINAL_COMPARISON',step_count_difference=pb['accepted_steps']-pa['accepted_steps'],first_rejection_difference_step=idx,first_cache_branch_difference_step=cacheidx,
                  first_error_estimate_difference_step=divergence('error_estimates'),first_accepted_h_difference_step=divergence('accepted_h_mm'),
                  rejection_context=None if idx is None else {'reference':sa[idx] if idx<len(sa) else None,'compared':sb[idx] if idx<len(sb) else None})
            pairs.append(item)
    return {'schema':'wb122_complete_rk_summary_v1','integrity_gate':'PASS','classification':'DIAGNOSTIC_ONLY_COMPLETE_RK_CACHE_TRACE',
      'hypothesis':'UNKNOWN_FIDELITY_OR_COVERAGE' if unknown else 'SUPPORTED_BUT_LIMITED_OBSERVATION_FIDELITY','unknowns':unknown,
      'propagation_calls':54,'calls_per_arm':18,'observer_records':runtime['observer_records'],'responses':responses,'profiles':profiles,'comparisons':pairs,
      'causal_attribution':'UNVERIFIED','physical_derivative_accuracy':'UNVERIFIED','qualification':'NOT_EVALUATED','physics_screening':'NOT_EVALUATED','final_oracle':'NOT_EVALUATED',
      'held_out_access':False,'truth_access':False,'production_changed':False,'new_reconstruction_calls':0}

def audit(out):
    runtime=read_public(out/'event/response.json');calls=[json.loads(x) for x in (out/'event/response.json.calls.ndjson').read_text().splitlines()]
    check(len(calls)==109 and calls[-1]=={'record':'terminal','status':'COMPLETED','official_calls':54},'complete call stream')
    for i,row in enumerate(runtime['rows']):check(calls[2*i]=={'record':'before_official','input':row['input']} and calls[2*i+1]=={'record':'after_official','row':row},'exact stream row')
    observer=[json.loads(x) for x in (out/'event/response.json.observer.ndjson').read_text().splitlines()]
    rows=[x for row in runtime['rows'] for x in row['observations']]
    check(observer[:-1]==rows and observer[-1]=={'record':'terminal','status':'COMPLETED','observer_records':len(rows),'propagation_calls':54},'observer stream/runtime closure')
    check([x['global_record_index'] for x in rows]==list(range(len(rows))),'whole observer indices')
    return analyze(runtime,read_public(out/'control.json'),read_public(out/'fixture.json'),read_public(out/'historical_runtime.json'))
if __name__=='__main__':
    print(json.dumps(audit(Path(sys.argv[1])),sort_keys=True,allow_nan=False))
