"""Independent saved-trial arithmetic and stream-completeness audit."""
import json
from pathlib import Path
import numpy as np
from alignment.wb90_measurement_contract import read_public
from alignment.wb99_direction_step_doubling import node_field,ACTS_T
from alignment.wb93_transport_error import v

def require(ok,message):
    if not ok:raise ValueError(message)

def retry(previous,row):
    if previous is None or previous['accepted']:return
    for k in ('start_position_mm','start_direction','start_time_Acts','start_path_mm','q_over_p_Acts'):
        require(previous[k]==row[k],'rejected trial changed '+k)
    if previous['direction_rejected']:require(row['h_mm']==.5*previous['h_mm'],'direction retry not half')

def endpoint(row):
    u=np.array(row['start_direction']);h=row['h_mm'];k=[np.array(row[x]) for x in ('k1','k2','k3','k4')]
    direction=u+h/6*(k[0]+2*(k[1]+k[2])+k[3]);direction/=np.linalg.norm(direction)
    position=np.array(row['start_position_mm'])+h*u+h*h/6*(k[0]+k[1]+k[2])
    return position,direction

def audit(out,raw,old,p):
    calls={};sampled={};historical={}
    for setting in raw['settings']:
        previous=next(x for x in old['settings'] if x['tolerance']==p['step_tolerance'] and x['cap_m']==setting['cap_m'])
        pairs=[(setting['entry_nominal'],previous['entry_nominal'],True)]
        for t,o in zip(setting['targets'],previous['targets']):
            pairs.extend((a,b,i in p['trace_sample_indices']) for i,(a,b) in enumerate(zip(t['samples'],o['samples'])))
            pairs.append((t['fixed_reference_start_nominal'],o['fixed_reference_start_nominal'],True))
        for a,b,save in pairs:
            calls[a['call_id']]=a
            if save:sampled[a['call_id']]=a
            if setting['direction_threshold']==0:historical[a['call_id']]=b
    fixture=read_public(out/'fixture.json');field=read_public(Path(fixture['wb101_field_source']));bounds=read_public(Path(fixture['wb101_bounds_source']))
    nodes=np.fromfile(fixture['wb101_nodes_source'],dtype='<i2').reshape(tuple(len(x) for x in field['mesh_mm'])+(3,))
    mesh=[np.array(x) for x in field['mesh_mm']];lo=np.array(field['conditions']['min_mm']);hi=np.array(field['conditions']['max_mm'])
    M=max(bounds['M_inside_native'],bounds['M_outside_native']);L=bounds['L_native_per_mm']
    stats={};previous=None;last=-1;terminal=False;batch=[];max_error=0.;total=0
    def flush():
        nonlocal max_error
        points=[];weights=[];sizes=[]
        for row,full in batch:
            e=row['envelope'];u=np.array(row['start_direction']);start=np.array(row['start_position_mm']);h=row['h_mm']
            faces=[]
            for k in range(3):
                if u[k]==0:continue
                a,b=sorted((start[k],start[k]+h*u[k]));begin=np.searchsorted(mesh[k],a);end=np.searchsorted(mesh[k],b,side='right')
                faces.extend((k,float(f),float((f-start[k])/u[k])) for f in mesh[k][begin:end] if 0<(f-start[k])/u[k]<h)
            got=[(a['axis'],a['face_mm'],a['s_mm']) for a in e['face_identities']]
            require(len(faces)==len(got) and all(a[:2]==b[:2] and abs(a[2]-b[2])<1e-12 for a,b in zip(faces,got)),'mesh faces')
            cuts=sorted(set([0.,h]+[a[2] for a in faces]));require(np.allclose(cuts,e['cuts_mm'],rtol=0,atol=1e-12),'mesh cuts')
            cuts=np.array(cuts);d=np.diff(cuts);mid=cuts[:-1]+d/2;off=d/(2*np.sqrt(3));s=np.column_stack((mid-off,mid+off)).reshape(-1)
            points.extend(start+s[:,None]*u);weights.extend(np.repeat(d/2,2));sizes.append(len(s))
        fields=node_field(np.array(points),field,nodes);offset=0
        for (row,full),n in zip(batch,sizes):
            integral=np.sum(fields[offset:offset+n]*np.array(weights[offset:offset+n])[:,None],axis=0);offset+=n
            e=row['envelope'];u=np.array(row['start_direction']);q=row['q_over_p_Acts'];h=row['h_mm'];w=-q*integral;t=np.linalg.norm(w)
            predictor=u+np.sinc(t/np.pi)*np.cross(w,u)+.5*np.sinc(t/(2*np.pi))**2*np.cross(w,np.cross(w,u))
            k=abs(q)*M;angle=k*h;start=np.array(row['start_position_mm']);end=start+h*u;radius=k*h*h/2
            lower=np.minimum(start,end)-radius;upper=np.maximum(start,end)+radius
            wholeInside=bool(np.all(lower>=lo) and np.all(upper<=hi));wholeOutside=bool(np.any(upper<lo) or np.any(lower>hi))
            amb=0.;outside=wholeOutside
            # Sufficient certificates avoid polynomial work for ordinary interior arcs.
            if not wholeInside and not wholeOutside:
                cuts=[0.,h]
                for axis in range(3):
                    for face in (lo[axis],hi[axis]):
                        for sign in (-1,1):
                            c=np.trim_zeros([sign*k/2,u[axis],start[axis]-face],'f')
                            cuts.extend(float(x.real) for x in np.roots(c) if abs(x.imag)<1e-14 and 0<x.real<h)
                cuts=sorted(set(cuts));outside=True
                for a,b in zip(cuts,cuts[1:]):
                    s=a+(b-a)/2;center=start+s*u;r=k*s*s/2
                    inside=np.all(center>=lo+r) and np.all(center<=hi-r);ex=np.any(center<lo-r) or np.any(center>hi+r)
                    outside=outside and ex
                    if not inside and not ex:amb+=b-a
            require(bool(outside)==e['proved_constant_outside'] and abs(amb-e['ambiguous_length_mm'])<=1e-10*max(1,h),'tube partition')
            comm=0. if outside else angle**2*np.exp(angle)
            lips=0. if outside else abs(q)*L*k*h**3/6
            jump=0. if outside else abs(q)*(bounds['M_inside_native']+bounds['M_outside_native'])*amb
            E=float(np.max(np.abs(full-predictor))+comm+lips+jump)
            errors=[np.max(np.abs(integral-e['integral_native_mm'])),np.max(np.abs(predictor-e['predictor_direction'])),abs(E-e['E'])]
            max_error=max(max_error,*map(float,errors));require(max(errors)<1e-12*max(1,E),'independent envelope arithmetic')
            for key,value in (('Rcomm',comm),('Rbend_lipschitz',lips),('Rbend_jump',jump)):
                require(abs(e[key]-value)<1e-12*max(1,value),'envelope component '+key)
        batch.clear()
    with (out/'event/acts.json.trials.ndjson').open() as stream:
        for line in stream:
            row=json.loads(line)
            if row.get('record')=='terminal':
                require(not terminal and row['calls']==len(calls),'trial terminal');terminal=True;continue
            require(not terminal,'rows after terminal');cid=row['call_id'];call=calls[cid];tau=call['direction_control']['threshold']
            require(tau>0 and row['threshold']==tau and cid>=last,'trial call identity/order')
            if cid!=last:previous=None
            st=stats.setdefault(cid,{'trials':0,'accepted':0,'position_rejected':0,'direction_rejected':0,'node_queries':0,'query_end':0})
            require(row['trial']==st['trials'] and row['query_begin']==st['query_end'],'trial sequence/query gap')
            retry(previous,row)
            if previous is None:
                require(np.max(np.abs(np.array(row['start_position_mm'])-v(call['start_state']['position_mm'],3)))<1e-9 and np.max(np.abs(np.array(row['start_direction'])-v(call['start_state']['direction'],3)))<1e-12,'initial state')
                require(row['start_time_Acts']==call['start_state']['time_Acts'] and row['start_path_mm']==0,'initial time/path')
            elif previous['accepted']:
                pp,uu=endpoint(previous)
                require(np.max(np.abs(pp-row['start_position_mm']))<1e-9 and np.max(np.abs(uu-row['start_direction']))<1e-12,'accepted state continuity')
                require(abs(row['start_path_mm']-previous['start_path_mm']-previous['h_mm'])<1e-9,'accepted path continuity')
                # Exact mass exported by the same ACTS ParticleHypothesis.
                mass=raw['wb101_muon_mass_Acts']
                expectedTime=previous['start_time_Acts']+previous['h_mm']*np.hypot(1,mass*previous['q_over_p_Acts'])
                require(abs(row['start_time_Acts']-expectedTime)<1e-9,'accepted time continuity')
            require(abs(row['q_over_p_Acts']*0.001-call['start_state']['q_over_p_per_MeV'])<1e-18,'constant qop')
            delta=row['query_end']-row['query_begin']
            expected=2+(int(call['options']['loopProtection'])+1 if previous is None else int(previous['accepted']))
            require(delta==expected,'trial official query interval')
            u=np.array(row['start_direction']);q=row['q_over_p_Acts'];h=row['h_mm']
            bf,bm,bl=(np.array(row[k]) for k in ('B_first','B_middle','B_last'))
            k1=q*np.cross(u,bf);k2=q*np.cross(u+h*k1/2,bm);k3=q*np.cross(u+h*k2/2,bm);k4=q*np.cross(u+h*k3,bl)
            for key,val in zip(('k1','k2','k3','k4'),(k1,k2,k3,k4)):require(np.max(np.abs(val-row[key]))<1e-15,'RKN stage '+key)
            # Use exported stages to retain the official arithmetic at a threshold boundary.
            kk=[np.array(row[k]) for k in ('k1','k2','k3','k4')]
            err=max(h*h*np.sum(np.abs(kk[0]-kk[1]-kk[2]+kk[3])),1e-20)
            require(abs(err-row['position_error_mm'])<1e-12*max(1,err),'position estimator')
            position=row['position_error_mm']<=p['step_tolerance'];require(position==row['position_pass'],'original position criterion')
            direction=position and row['envelope']['E']+p['direction_uncertainty_allowance']>tau
            require(direction==row['direction_rejected'] and row['accepted']==(position and not direction),'acceptance decision')
            if position:
                full=u+h*(kk[0]+2*(kk[1]+kk[2])+kk[3])/6;full/=np.linalg.norm(full)
                batch.append((row,full));st['node_queries']+=2*(len(row['envelope']['cuts_mm'])-1)
                if len(batch)>=128:flush()
            else:require(row['envelope'] is None,'position-rejected envelope evaluation')
            st['trials']+=1;st['accepted']+=row['accepted'];st['position_rejected']+=not position;st['direction_rejected']+=direction;st['query_end']=row['query_end']
            previous=row;last=cid;total+=1
    if batch:flush()
    require(terminal,'missing trial terminal')
    for cid,call in calls.items():
        cc=call['direction_control']
        if not cc['threshold']:continue
        st=stats.get(cid,{k:0 for k in ('trials','accepted','position_rejected','direction_rejected','node_queries','query_end')})
        for key in ('trials','accepted','position_rejected','direction_rejected','node_queries'):require(st[key]==cc[key],'trial summary '+key)
        require(st['query_end']<=call['field_counts']['total'],'trial queries exceed full call')
        if call['status']=='PASS':require(st['query_end']==call['field_counts']['total'],'successful unrecorded query')
    traces=0;seen=set()
    trialStream=(out/'event/acts.json.trials.ndjson').open()
    def trialRows():
        for line in trialStream:
            r=json.loads(line)
            if r.get('record')!='terminal':yield r
    trialIterator=iter(trialRows());cursor=next(trialIterator,None)
    with (out/'event/acts.json.traces.ndjson').open() as stream:
        for line in stream:
            row=json.loads(line);cid=row['call_id'];require(cid in sampled and cid not in seen,'sample trace identity');seen.add(cid);call=calls[cid]
            require(len(row['field_queries'])==call['field_counts']['total'] and len(row['accepted_trace'])==call['accepted_steps'],'trace completeness')
            if cid in historical:
                require(row['field_queries']==historical[cid]['field_queries'] and row['accepted_trace']==historical[cid]['accepted_trace'],'disabled complete historical trace')
            while cursor is not None and cursor['call_id']<cid:cursor=next(trialIterator,None)
            stepIndex=0
            while cursor is not None and cursor['call_id']==cid:
                qe=cursor['query_end'];queries=row['field_queries']
                for key,index in (('B_middle',qe-2),('B_last',qe-1)):
                    require(np.max(np.abs(np.array(queries[index]['field_T'])*ACTS_T-cursor[key]))<1e-15,'trial/official field query')
                u=np.array(cursor['start_direction']);h=cursor['h_mm'];start=np.array(cursor['start_position_mm'])
                mid=start+h*.5*u+h*h*.125*np.array(cursor['k1']);lastpos=start+h*u+h*h*.5*np.array(cursor['k3'])
                require(np.max(np.abs(mid-queries[qe-2]['position_mm']))<1e-9 and np.max(np.abs(lastpos-queries[qe-1]['position_mm']))<1e-9,'official RKN query coordinates')
                if cursor['accepted']:
                    step=row['accepted_trace'][stepIndex];pp,uu=endpoint(cursor)
                    require(np.max(np.abs(pp-step['position_mm']))<1e-9 and np.max(np.abs(uu-step['direction']))<1e-12 and step['query_end']==qe,'accepted/observer closure')
                    stepIndex+=1
                cursor=next(trialIterator,None)
            if cid not in historical:require(stepIndex==len(row['accepted_trace']),'accepted trial trace coverage')
            lastq=0
            for step in row['accepted_trace']:
                require(step['query_begin']==lastq and lastq<=step['query_end']<=len(row['field_queries']),'accepted trace query mapping');lastq=step['query_end']
            for query in row['field_queries']:
                x=np.array(query['position_mm']);b=np.array(query['field_T']);inside=np.all(x>=lo) and np.all(x<=hi)
                require(query['zone_id']==(field['conditions']['zone_id'] if inside else -1),'official query domain')
                if not inside:require(np.max(np.abs(b-1e-5))<=p['field_probe_T_tolerance'],'official fallback')
            traces+=1
    require(seen==set(sampled),'missing sampled traces')
    trialStream.close()
    return {'gate':'PASS','all_enabled_trials':total,'enabled_calls':len(stats),'sampled_traces':traces,'independent_envelope_max_difference':max_error,
            'rejection_state_check':'PASS','disabled_complete_trace_fidelity':'PASS','qualification':'NOT_EVALUATED','note':'Saved numerical audit, not a new trajectory local-defect reference or physical field oracle'}
