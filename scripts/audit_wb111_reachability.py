"""Independent same-input reachability checks; no accuracy/acceptance oracle."""
import json,re,xml.etree.ElementTree as ET
import numpy as np
from audit_wb106_trace import check,digest
from audit_wb109_abort import frame,vector

def free_state(s):
    p=vector(s['position_mm']);u=vector(s['direction'])
    check(abs(np.linalg.norm(u)-1.)<=1e-9 and u[2]>0,'direction convention')
    check(np.isfinite([s['qop_acts'],s['time_acts']]).all(),'qop/time finite')
    return p,u

def validate(r,c,f):
    check(r['schema']=='wb111_reachability_runtime_v1','runtime schema')
    check(r['control_used']==c,'control identity')
    check(r['identity']=={k:f[k] for k in ('actual_run','actual_event','input_xaod','ordinal')},'event/source')
    check(r['official_calls']==3 and r['diagnostic_calls']==4 and len(r['official'])==3 and len(r['diagnostic'])==4,'call accounting')
    check(r['world']==c['topology']['world'],'World identity')
    check(r['acts_MeV_unit']==.001 and r['acts_T_unit']==.000299792458,'ACTS units')
    cond=r['field_conditions'];check(cond['official_context_is_actual_cache'] is True,'field/cache identity')
    check((cond['map_key'],cond['cache_key'])==('fieldMapCondObj','fieldCondObj') and
      bool(cond['map_IOV']) and bool(cond['cache_IOV']) and np.isfinite(cond['scale']),'field conditions')
    seed=np.asarray(c['seed'],dtype=float).reshape(5)
    seedpos=np.r_[seed[:2],c['seed_z_mm']];seedq=seed[4]/r['acts_MeV_unit']
    seeddir=vector(c['targets'][0]['before_official']['start_direction'])
    seedtime=0.
    terminal=c['terminal'];tp=vector(terminal['position']);tu=vector(terminal['direction'])
    targets=c['targets'];check([t['station'] for t in targets]==[1,2,3],'target order')
    def input_check(s,arm):
        p,u=free_state(s)
        expected=(seedpos,seeddir,seedq,seedtime) if arm=='from_seed' else (tp,tu,terminal['qop_acts'],terminal['time'])
        check(np.max(np.abs(p-expected[0]))<=1e-9 and np.max(np.abs(u-expected[1]))<=1e-9 and
          s['qop_acts']==expected[2] and s['time_acts']==expected[3],'start roundtrip/units/time')
        check(s['covariance_present'] is False,'input covariance unauthorized')
    def bound_check(s,t,q):
        p,u=free_state(s);a=frame(t);local=(np.linalg.inv(a)@np.r_[p,1.])[:3];direction=a[:3,:3].T@u
        check(abs(local[2])<=1e-6 and direction[2]>0,'bound target contract')
        h=np.r_[local[:2],direction[:2]/direction[2]]
        check(np.max(np.abs(vector(s['local'])-local))<=1e-9 and
          np.max(np.abs(np.asarray(s['h']).reshape(4)-h))<=1e-9,'returned local/h calculation')
        check(s['qop_acts']==q and s['covariance_present'] is False,'returned qop/covariance')
        return h,p
    for i,o in enumerate(r['official']):
        t=targets[i]
        check(isinstance(o['has_value'],bool),'official outcome boolean')
        check((o['station'],o['historical_call_id'],o['target_frame'],o['has_value'])==
          (t['station'],t['call_id'],t['frame'],t['official_has_value']),'official reproduction identity')
        input_check(o['start'],'from_seed')
        check(np.array_equal(vector(o['start']['position_mm']),vector(t['before_official']['start_position'])) and
          np.array_equal(vector(o['start']['direction']),vector(t['before_official']['start_direction'])),'official exact start')
        if o['has_value']:
            bound_check(o['state'],t['frame'],seedq);check(o['state']['h']==t['official_h'],'official exact output')
        else:check('state' not in o,'official nullopt with state')
    rows=[];scales=np.asarray(f['protocol']['output_scales'])
    ids=[c['topology']['world']['geometry_id'],c['topology']['matches'][0]['owner_geometry_id'],
      c['topology']['probe_used']['target_volume']['geometry_id']]
    volumes={v['geometry_id']:v for v in c['topology']['volumes']}
    for i,d in enumerate(r['diagnostic']):
        t=targets[i] if i<3 else targets[2];arm='from_seed' if i<3 else 'from_terminal'
        check((d['arm'],d['station'],d['target_frame'])==(arm,t['station'],t['frame']),'diagnostic order/frame')
        check(d['navigator']=='Acts::VoidNavigator' and d['user_aborters']==[] and d['material_actor'] is False,'intervention identity')
        input_check(d['start'],arm);p,u=free_state(d['start']);q=d['start']['qop_acts']
        o=d['options'];expected=c['baseline_options'].copy();expected['surfaceTolerance']=expected.pop('surfaceTolerance_mm')
        expected.update(maxRungeKuttaStepTrials=10000,stepSizeCutOff=0.,loopFraction=.5)
        expected['loopProtection']=bool(np.linalg.norm(u[:2])/abs(q)<c['pt_loopers_MeV']*r['acts_MeV_unit'])
        expected['forward']=bool(np.dot(frame(t['frame'])[:3,3]-p,u)>=0)
        check(o==expected,'actual integration options')
        queries=d['field_queries'];check(d['field_query_count']==len(queries) and d['field_gradient_count']==0,'field accounting')
        for query in queries:
            vector(query['position_mm']);b=vector(query['field_acts']);bt=vector(query['field_T'])
            check(np.allclose(b,bt*r['acts_T_unit'],rtol=1e-14,atol=0.),'field native/T conversion')
        steps=d['accepted_trace'];check(d['accepted_steps']==len(steps) and
          d['rejected_trials']==sum(s['rejected_trials'] for s in steps),'step/rejection accounting')
        prev=0.;lastquery=0
        for s in steps:
            free_state(s);check(s['qop_acts']==q,'step qop changed')
            check(np.isfinite(s['path_mm']) and s['path_mm']>=prev and
              lastquery<=s['query_end']<=len(queries) and isinstance(s['rejected_trials'],int) and s['rejected_trials']>=0,'step path/query/reject')
            prev=s['path_mm'];lastquery=s['query_end']
        if d['last_free'] is not None:free_state(d['last_free'])
        check(d['status'] in ('SUCCESS','ERROR'),'diagnostic outcome')
        row={'arm':arm,'station':t['station'],'status':d['status'],'field_query_count':len(queries),
          'accepted_steps':len(steps),'rejected_trials':d['rejected_trials']}
        if d['status']=='ERROR':
            check(bool(d.get('error_message')) and 'state' not in d,'missing/contradictory diagnostic error')
            row['error_message']=d['error_message'];rows.append(row);continue
        check(d['last_free'] is not None and bool(steps) and bool(queries),'successful diagnostic trace missing')
        hp,end=bound_check(d['state'],t['frame'],q);lp,lu=free_state(d['last_free'])
        check(d['last_free']['qop_acts']==q and d['state']['time_acts']>=d['start']['time_acts'],'terminal qop/time')
        local=(np.linalg.inv(frame(t['frame']))@np.r_[lp,1.])[:3]
        check(abs(local[2])<=o['surfaceTolerance'],'free target not reached')
        check(np.max(np.abs(lp-end))<=o['surfaceTolerance'] and
          np.max(np.abs(lu-vector(d['state']['direction'])))<=1e-9,'free/bound consistency')
        check(np.isfinite(d['path_mm']) and abs(d['last_free']['path_mm']-d['path_mm'])<=1e-9,'path receipt')
        check(lastquery==len(queries),'unaccounted final field queries')
        y=np.asarray(f['references'][t['station']]['fixed_z_state']).reshape(4)
        row.update(h=hp.tolist(),measurement_difference=(hp-y).tolist(),
          measurement_scaled_difference=((hp-y)/scales).tolist(),free_target_distance_mm=float(local[2]),saved_topology_containment={})
        for vid in ids:
            v=volumes[vid];check(v['bounds_type']==1,'containment cuboid')
            margin=np.asarray(v['bounds_values'])-np.abs((np.linalg.inv(frame(v['transform']))@np.r_[end,1.])[:3])
            row['saved_topology_containment'][v['name']]={'inside':bool(np.all(margin>=0)),'margins_mm':margin.tolist()}
        if arm=='from_seed' and t['official_has_value']:
            row['void_minus_official_h']=(hp-np.asarray(t['official_h']).reshape(4)).tolist()
        rows.append(row)
    supported=all(row['status']=='SUCCESS' for row in rows)
    difference=(np.asarray(rows[3]['h'])-np.asarray(rows[2]['h'])).tolist() if rows[2]['status']==rows[3]['status']=='SUCCESS' else None
    return {'classification':'TARGET_REACHABLE_WITHOUT_GEOMETRY_NAVIGATION' if supported else 'NOT_DEMONSTRATED',
      'official_guard':'PASS','official_calls':3,'diagnostic_calls':4,'rows':rows,'continuation_minus_full_station3_h':difference,
      'accuracy_and_sensitive_acceptance':'UNKNOWN','physical_seed_and_association_validity':'UNKNOWN',
      'qualification':'NOT_EVALUATED','physics_screening':'NOT_EVALUATED','final_oracle':'NOT_EVALUATED','held_out_access':False}

def audit(out):
    from wb111_contract import BASE,verify
    frozen=verify(out);event=out/'event'
    fixture=json.loads((out/'fixture.json').read_text());control=json.loads((out/'control.json').read_text())
    result=json.loads((event/'reachability.json').read_text());summary=validate(result,control,fixture)
    for n in ('fixture.json','control.json'):check(digest(event/n)==digest(out/n),'event input copy')
    raw=json.loads((out/'raw_identity.json').read_text());previous=json.loads((BASE/'raw_identity.json').read_text())
    check(raw['path']==fixture['input_xaod'] and raw['sha256']==previous['sha256'] and
      {'bytes':raw['bytes'],'mtime_ns':raw['mtime_ns']}==fixture['source_stat'],'raw identity receipt')
    expected=json.loads((out/'generation_expectation.json').read_text())
    for n,h in expected['files'].items():check(digest(out/'isolated_source'/n)==h,'generated source')
    check(digest(event/'athena.py')==expected['athena_sha256'],'runner')
    binary=json.loads((out/'binary_manifest.json').read_text())
    check(binary['binary'] in result['loaded_libraries'] and digest(binary['binary'])==binary['sha256'],'binary')
    libs=[p for p in result['loaded_libraries'] if any(k in p for k in ('FaserActs','ActsCore','MagField'))]
    check(bool(libs),'scientific libraries absent')
    for p in libs:check(p in frozen['hashes'] and digest(p)==frozen['hashes'][p],'scientific library')
    for p in (out/'identity_payload').iterdir():
        if p.is_file():check(digest(p)==digest(BASE/'identity_payload'/p.name),'copied payload')
    for pfn in ET.parse(out/'identity_payload/PoolFileCatalog.xml').findall('.//pfn'):
        p=pfn.attrib['name'];check(digest(p)==frozen['hashes'][p],'actual POOL PFN')
    manifest=json.loads((event/'athena_manifest.json').read_text())
    check(manifest['sqlite_sha256']==digest(out/'identity_payload/tracker_alignment.sqlite') and
      manifest['pool_catalog_sha256']==digest(out/'identity_payload/PoolFileCatalog.xml'),'actual payload manifest')
    check((manifest['geometry'],manifest['global_tag'],manifest['field_mode'],manifest['diagnostic_navigator'],
      manifest['max_official_calls'],manifest['max_diagnostic_calls'])==('FASERNU-04','OFLCOND-FASER-06','FASER','Acts::VoidNavigator',3,4),'backend manifest')
    log=(event/'athena.log').read_text()
    for required in ('start processing event #2268, run #100044',str(out/'identity_payload/tracker_alignment.sqlite'),
      'Acts TrackingGeometry construction completed','Using FASER magnetic field service','SurfaceError:1. Returning empty parameters'):
        check(required in log,'runtime conditions/header/official error receipt')
    maps=re.findall(r'Initialized the field map from\s+([^\n]+)',log);check(bool(maps),'actual field map absent')
    for p in maps:
        p=p.strip();check(p in frozen['hashes'] and digest(p)==frozen['hashes'][p],'actual field map')
    check(json.loads((out/'athena_exit.json').read_text())['exit_code']==0,'Athena execution')
    return {'schema':'wb111_reachability_summary_v1','integrity_gate':'PASS','identity':result['identity'],**summary}
