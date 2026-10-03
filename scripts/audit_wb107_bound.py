"""Independent saved-call and bound-conversion audit; never executes physics."""
import json
import re
from pathlib import Path
import numpy as np
from audit_wb106_trace import validate_trace,check,digest

EXTRA={'before_diagnostic','bound_before','bound_after','propagator_result','comparison'}

def validate(rows,baseline,fixture):
    official=[r for r in rows if r['record'] not in EXTRA]
    validate_trace(official,fixture)
    # Library inventory differs because WB107 is a distinct component.
    clean=lambda seq:[{k:v for k,v in r.items() if k!='loaded_libraries'} for r in seq]
    check(clean(official)==clean(baseline),'official trace changed from WB106')
    calls=[r['call_id'] for r in official if r['record']=='before_official']
    check(len(calls)==101 and calls[-1]==124,'frozen WB106 call prefix')
    extras=[r for r in rows if r['record'] in EXTRA]
    check(all(r.get('call_id') in calls for r in extras),'foreign diagnostic call')
    final=None
    for cid in calls:
        rs=[r for r in rows if r.get('call_id')==cid]
        ds=[r for r in rs if r['record'] in EXTRA]
        cmp=[r for r in ds if r['record']=='comparison']
        prop=[r for r in ds if r['record']=='propagator_result']
        check(len(cmp)==len(prop)==1,'comparison/result multiplicity')
        c,p=cmp[0],prop[0]
        check(ds[0]['record']=='before_diagnostic' and ds[-1] is c,'diagnostic order')
        check(rs.index(ds[0])==rs.index(next(r for r in rs if r['record']=='after_official'))+1,'official precedes diagnostic')
        check(c['official_has_value']==c['diagnostic_has_value']==p['ok'],'presence mismatch')
        b=[r for r in ds if r['record']=='bound_before'];a=[r for r in ds if r['record']=='bound_after']
        check(len(b)==len(a)<=1,'bound receipt multiplicity')
        if b:
            check([r['record'] for r in ds]==['before_diagnostic','bound_before','bound_after','propagator_result','comparison'],'bound order')
            frame=np.asarray(next(r for r in rs if r['record']=='before_official')['target_frame'])
            check(b[0]['requested_target'] and b[0]['surface_geometry_id']==0,'target identity')
            check(np.array_equal(b[0]['surface_frame'],frame),'target frame')
            for k,shape in (('position',(3,1)),('direction',(3,1)),('surface_frame',(4,4))):
                v=np.asarray(b[0][k]);check(v.shape==shape and np.isfinite(v).all(),'nonfinite/shape '+k)
            check(all(np.isfinite(b[0][k]) for k in ('qop_acts','time','path_mm','on_surface_tolerance_mm')),'nonfinite scalar')
            check(b[0]['cov_transport'] is False and b[0]['transport_cov_argument'] is True,'covariance contract')
            check(b[0]['on_surface_tolerance_mm']==1e-4,'ACTS tolerance identity')
            seed=np.asarray(next(r for r in rs if r['record']=='input')['seed']).reshape(5)
            check(b[0]['MeV_acts']==1e-3 and b[0]['qop_acts']==seed[4]/b[0]['MeV_acts'],'qop unit/conservation')
            check(a[0]['ok']==p['ok'],'bound/propagator status mismatch')
        if c['official_has_value']:
            check(bool(b),'missing successful bound receipt')
            for k,shape in (('parameters',(6,1)),('position',(3,1)),('direction',(3,1))):
                x=np.asarray(c['official_'+k]);y=np.asarray(c['diagnostic_'+k])
                check(x.shape==y.shape==shape and np.isfinite(x).all() and np.isfinite(y).all(),'comparison shape/nonfinite')
                check(np.array_equal(x,y),'diagnostic numerical change '+k)
            # Bind the comparison to the original successful output.
            o=next(r for r in rs if r['record']=='output')
            frame=np.asarray(next(r for r in rs if r['record']=='before_official')['target_frame'])
            local_dir=frame[:3,:3].T@np.asarray(c['official_direction']).reshape(3)
            check(c['official_position']==o['global_position'] and np.allclose(local_dir,np.asarray(o['direction']).reshape(3),rtol=0,atol=1e-15),'comparison detached from official output')
        else:
            check(cid==124,'wrong failure call')
            check(p.get('error_category')=='SurfaceError' and p.get('error_value')==1,'failure category/value')
            if b:
                check(a[0].get('error_category')==p['error_category'] and a[0].get('error_value')==p['error_value'],'bound error mismatch')
                pos=np.asarray(b[0]['position']).reshape(3)
                local=(np.linalg.inv(frame)@np.r_[pos,1.])[:3]
                final={'classification':'FINAL_TARGET_BOUND_CONVERSION','call_id':cid,
                  'position_mm':pos.tolist(),'target_local_mm':local.tolist(),
                  'absolute_plane_distance_mm':abs(float(local[2])),
                  'on_surface_tolerance_mm':b[0]['on_surface_tolerance_mm'],
                  'error_category':p['error_category'],'error_value':p['error_value'],
                  'deeper_navigation_or_field_mechanism':'UNKNOWN'}
            else:
                check([r['record'] for r in ds]==['before_diagnostic','propagator_result','comparison'],'unresolved stage ordering')
                final={'classification':'OTHER_OR_UNRESOLVED_STAGE','call_id':cid,'deeper_navigation_or_field_mechanism':'UNKNOWN'}
    check(final is not None,'no reproduced failure')
    return final

def audit(out):
    from wb107_contract import BASE,verify
    baseline=[json.loads(x) for x in (BASE/'event/acts.json.calls.ndjson').read_text().splitlines()]
    rows=[json.loads(x) for x in (out/'event/acts.json.calls.ndjson').read_text().splitlines()]
    fixture=json.loads((out/'fixture.json').read_text())
    final=validate(rows,baseline,fixture)
    filtered=[r for r in rows if r['record'] not in EXTRA]
    frozen=verify(out)
    check(digest(out/'event/fixture.json')==digest(out/'fixture.json'),'event fixture')
    event=next(r for r in rows if r['record']=='event')
    binary=json.loads((out/'binary_manifest.json').read_text())
    check(binary['binary'] in event['loaded_libraries'] and digest(binary['binary'])==binary['sha256'],'diagnostic binary')
    for name,h in json.loads((out/'generated_source_manifest.json').read_text()).items():
        check(digest(out/'isolated_source'/name)==h,'generated source')
    official=[p for p in event['loaded_libraries'] if any(k in p for k in ('FaserActs','ActsCore','MagField'))]
    check(bool(official),'official libraries missing')
    for path in official:
        check(path in frozen['hashes'] and digest(path)==frozen['hashes'][path],'official library identity')
    log=(out/'event/athena.log').read_text()
    for required in ('start processing event #2268, run #100044','Using FASER magnetic field service',
       str(out/'identity_payload/tracker_alignment.sqlite'),'SurfaceError:1. Returning empty parameters.'):
        check(required in log,'log receipt '+required)
    maps=re.findall(r'Initialized the field map from\s+([^\n]+)',log)
    check(bool(maps),'field map missing')
    for path in maps:
        path=path.strip();check(path in frozen['hashes'] and digest(path)==frozen['hashes'][path],'field map identity')
    for path in (out/'identity_payload').iterdir():
        if path.is_file():check(digest(path)==digest(BASE/'identity_payload'/path.name),'copied payload')
    manifest=json.loads((out/'event/athena_manifest.json').read_text())
    check(digest(out/'event/athena.py')==json.loads((out/'generation_expectation.json').read_text())['athena_sha256'],'runner identity')
    check(manifest['sqlite_sha256']==digest(out/'identity_payload/tracker_alignment.sqlite'),'actual sqlite')
    check(json.loads((out/'athena_exit.json').read_text())['exit_code']==1,'expected Athena fail-closed')
    check(filtered[-1]['status']=='FAIL_CLOSED' and filtered[-1]['error']=='official ACTS propagation returned no state','terminal semantics')
    return {'schema':'wb107_bound_attribution_v1','integrity_gate':'PASS','official_trace_identical':True,
      'successful_comparisons':100,'official_calls':101,'diagnostic_calls':101,'population':1,
      'identity':{k:fixture[k] for k in ('input_xaod','ordinal','actual_run','actual_event')},
      'held_out_access':False,'qualification':'NOT_EVALUATED',**final}
