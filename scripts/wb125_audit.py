#!/usr/bin/env python3
"""Independent saved-response checks for the frozen six-event seed intervention."""
import math
import numpy as np
from audit_wb113_provenance import classify

SCALES=np.array([1.,1.,.001,.001])
EPS=8*np.finfo(np.float32).eps

def check(condition,message):
    if not condition: raise ValueError(message)

def array(value,size):
    a=np.asarray(value,dtype=float).reshape(-1)
    check(a.size==size and np.isfinite(a).all(),'shape/nonfinite')
    return a

def choose_state(provenance,fixture):
    """Select by z BEFORE checking momentum, never try the second-best state."""
    tracks=provenance['persisted_tracks']
    check(len(tracks)==1 and tracks[0]['key']=='CKFTrackCollection','primary collection')
    check(tracks[0]['container_size']==1 and len(tracks[0]['matching_tracks'])==1,'exactly one track')
    states=tracks[0]['matching_tracks'][0]['parameters']
    check(bool(states),'no states')
    z=float(fixture['references'][0]['z_state_mm'])
    distances=[]
    for i,s in enumerate(states):
        check(s['persistent_index']==i,'persistent order')
        distances.append(abs(array(s['global_position_native'],3)[2]-z))
    i=min(range(len(states)),key=lambda k:(distances[k],k));s=states[i]
    x=array(s['global_position_native'],3);p=array(s['global_momentum_native'],3)
    native=array(s['native_parameters'],5);q=float(s['charge_e'])
    check(math.isfinite(q) and abs(q)==1 and p[2]>0,'charge/forward momentum')
    check((s['position_unit'],s['momentum_unit'],s['qop_unit'])==('mm','MeV','MeV^-1'),'source units')
    qop=q/np.linalg.norm(p)
    check(abs(native[4]-qop)<=EPS*max(abs(native[4]),abs(qop)),'native/global qop')
    phi=math.atan2(p[1],p[0]);theta=math.atan2(np.linalg.norm(p[:2]),p[2])
    dphi=math.atan2(math.sin(native[2]-phi),math.cos(native[2]-phi))
    check(abs(dphi)<=EPS*max(1.,abs(phi)) and abs(native[3]-theta)<=EPS*max(1.,abs(theta)),'native/global angles')
    check(s['surface_type']==4,'source surface must be Trk plane')
    frame=np.asarray(s['surface_transform'],dtype=float)
    check(frame.shape==(4,4) and np.isfinite(frame).all(),'surface transform')
    check(np.max(abs(frame[3]-[0,0,0,1]))==0,'homogeneous row')
    rotation=frame[:3,:3]
    check(np.max(abs(rotation.T@rotation-np.eye(3)))<=1e-9 and abs(np.linalg.det(rotation)-1)<=1e-9,'rigid source frame')
    local=rotation.T@(x-frame[:3,3])
    check(abs(local[2])<=1e-6 and np.max(abs(local[:2]-native[:2]))<=1e-6,'native/source position')
    return {'selected_index':i,'source_qop_per_MeV':qop,'abs_delta_z_mm':distances[i]}

def expected_cells(index):
    cells=[(a,'nominal',s,0.) for a in ('D','M','P') for s in range(4)]
    if index in (12,20):
        cells += [(a,'repeat',s,0.) for a in ('D','M','P') for s in (2,3)]
        cells += [('P','fd',s,m) for s in (2,3) for m in (-1.,-.5,.5,1.)]
    return cells

def audit_event(fixture,provenance,response,old=None):
    identity={k:fixture[k] for k in ('input_xaod','ordinal','actual_run','actual_event')}
    check(provenance['identity']==identity and response['identity']==identity,'identity')
    check(response['index']==fixture['index'],'index')
    association=classify(fixture,provenance)
    track=provenance['persisted_tracks'][0]
    memberships=track['matching_tracks'][0]['cluster_membership'] if track['matching_tracks'] else []
    resolved={m['cluster_id'] for m in memberships if m['prd_link_resolved']}
    overlap={str(r['station']):len(set(r['clusters']) & resolved) for r in fixture['references']}
    downstream_membership=all(set(r['clusters'])<=resolved for r in fixture['references'] if r['station']>0)
    selected=None;selection_error=None
    try:selected=choose_state(provenance,fixture)
    except (ValueError,KeyError,TypeError) as e:selection_error=str(e)
    actual=response['selection'];known=actual['status']=='KNOWN'
    if actual.get('selected_index') is not None and selected:
        check(actual['selected_index']==selected['selected_index'],'independent state selection')
    if known:
        check(selected is not None,'C++ accepted invalid source')
        check(abs(actual['source_qop_per_MeV']-selected['source_qop_per_MeV'])<=1e-18,'source qop recompute')
        bridge=response['bridge'];check(bridge['status']=='SUCCESS' and bridge['on_surface'] and not bridge['covariance_present'],'bridge')
        source_q=selected['source_qop_per_MeV']
        check(abs(bridge['qop_per_MeV']-source_q)<=4*np.finfo(float).eps*abs(source_q),'no-material bridge qop conservation')
        physical=array(response['seeds']['P'],5);mixed=array(response['seeds']['M'],5)
        check(np.array_equal(physical[:4],array(bridge['h'],4)) and physical[4]==bridge['qop_per_MeV'],'P bridge definition')
    dummy=array(response['seeds']['D'],5)
    check(np.array_equal(dummy[:4],array(fixture['references'][0]['fixed_z_state'],4)) and dummy[4]==1e-5,'D freeze')
    if known:check(np.array_equal(mixed[:4],dummy[:4]) and mixed[4]==physical[4],'M single qop intervention')
    wanted=set(expected_cells(fixture['index']));rows={};call_ids=[];metrics=[]
    for row in response['responses']:
        key=(row['arm'],row['kind'],row['station'],float(row.get('qop_multiplier',0)))
        check(key in wanted and key not in rows,'matrix unknown/duplicate cell');rows[key]=row
        if row.get('propagation_call') is not None:call_ids.append(row['propagation_call'])
        if row['status']=='SUCCESS':
            state=row['state'];check(state['on_surface'] and not state['covariance_present'],'target/bound/C')
            h=array(state['h'],4);y=array(fixture['references'][row['station']]['fixed_z_state'],4)
            local=array(state['local'],3);check(abs(local[2])<=1e-6,'original target coordinate budget')
            seed=array(row['seed'],5);expected=array(response['seeds'][row['arm']],5)
            if row['kind']=='fd':expected[4]+=row['qop_multiplier']*1e-8
            check(np.array_equal(seed,expected),'response seed/FD freeze')
            residual=y-h;metrics.append({'cell':list(key),'h':h.tolist(),'residual':residual.tolist(),'R':float(np.linalg.norm(residual/SCALES))})
    missing=[list(k) for k in sorted(wanted-set(rows))]
    check(not missing or not known,'missing known-seed matrix cells')
    check(len(call_ids)==len(set(call_ids)),'duplicate call index')
    cap=24 if fixture['index'] in (12,20) else 10
    check(response['propagation_calls']<=cap,'event call cap')
    repeat_checks=[]
    for a in ('D','M','P'):
        for s in (2,3):
            n=rows.get((a,'nominal',s,0.));r=rows.get((a,'repeat',s,0.))
            if r:
                exact=r['status']==n['status'] and r.get('state')==n.get('state')
                repeat_checks.append({'arm':a,'station':s,'exact':exact})
                check(exact,'identical repeat not exact')
    fidelity=[]
    if old:
        check(all(np.array_equal(np.asarray(t['frame']),np.asarray(response['targets'][t['station']]['frame'])) for t in old['targets']),'old target frames')
        for target in old['targets']:
            s=target['station'];r=rows[('D','nominal',s,0.)]
            exact=r['status']=='SUCCESS' and np.array_equal(array(r['state']['h'],4),array(target['h'],4))
            check(exact,'D old h not exact');fidelity.append({'station':s,'exact':exact})
            for a,b in zip(target['sensors'],response['targets'][s]['sensors']):
                check(all(a[k]==b[k] for k in ('strip','wafer','geometry_id','transform')),'sensor identity/frame drift')
    elif fixture['index']==12:
        failure=rows[('D','nominal',3,0.)]['status']=='FAIL_OFFICIAL_NULL'
        check(failure,'historical index12 D failure not reproduced');fidelity=[{'station':3,'failure_reproduced':True}]
    comparisons=[]
    bycell={tuple(m['cell']):m for m in metrics}
    for s in range(4):
        d=bycell.get(('D','nominal',s,0.))
        for a in ('M','P'):
            m=bycell.get((a,'nominal',s,0.))
            ratio=m['R']/d['R'] if m and d and d['R']!=0 else None
            comparisons.append({'arm':a,'station':s,'R_ratio_to_D':ratio,'tenfold_reduction':ratio<=.1 if ratio is not None else None,
                'h_minus_D':(np.array(m['h'])-d['h']).tolist() if m and d else None})
    fd=[]
    for s in (2,3):
        qs=[rows.get(('P','fd',s,m)) for m in (-1.,-.5,.5,1.)]
        if all(r and r['status']=='SUCCESS' for r in qs):
            hs=[array(r['state']['h'],4) for r in qs]
            full=(hs[3]-hs[0])/(2e-8);half=(hs[2]-hs[1])/1e-8
            effect=(half-full)*1e-8
            fd.append({'station':s,'J_full':full.tolist(),'J_half':half.tolist(),'step_scaled_difference':effect.tolist(),'scaled_difference':(effect/SCALES).tolist()})
    availability={a:[rows[(a,'nominal',s,0.)]['status'] for s in range(4)] for a in ('D','M','P')}
    physical_ok=known and all(s=='SUCCESS' for s in availability['P'])
    charges=provenance['related_truth_particles']
    observed_barcodes={b for c in association['cluster_links'] for b in c['barcodes']}
    charge_status='UNKNOWN_PARTICLE_JOIN'
    if known and len(observed_barcodes)==1 and len(charges)==1 and charges[0]['barcode'] in observed_barcodes and association['xaod_barcode_join_unique'].get(charges[0]['barcode'],False):
        native_charge=provenance['persisted_tracks'][0]['matching_tracks'][0]['parameters'][selected['selected_index']]['charge_e']
        charge_status='MATCH_RELATED_TRUTH_CHARGE' if native_charge==charges[0]['charge_e'] else 'CONTRADICTED_RELATED_TRUTH_CHARGE'
    complete=association['association']=='SUPPORTED_PERSISTED_COMMON_PARTICLE' and downstream_membership and charge_status=='MATCH_RELATED_TRUTH_CHARGE'
    h_status='NOT_EVALUATED_NONPRIMARY_EVENT'
    if fixture['index']==12:
        removed=known and availability['P'][3]=='SUCCESS'
        h_status=('SUPPORTED_CONDITIONAL_INPUT_EFFECT' if not complete else 'SUPPORTED_BUT_LIMITED_SEED_INPUT_EFFECT') if removed else ('NOT_SUPPORTED' if known and availability['P'][3]=='FAIL_OFFICIAL_NULL' else 'UNKNOWN')
    return {'schema':'wb125_event_audit_v1','identity':identity,'index':fixture['index'],'integrity':'PASS',
        'association':association['association'],'cluster_links':association['cluster_links'],'membership_overlap_by_station':overlap,
        'downstream_membership_complete':downstream_membership,'source_selection':selected,'source_selection_error':selection_error,
        'source_charge_consistency':charge_status,
        'seed_status':actual,'availability':availability,'conditional':not complete,'physical_fixture_supported':physical_ok and complete,
        'hypothesis':h_status,'baseline_fidelity':fidelity,'metrics':metrics,'comparisons':comparisons,'fd_diagnostics':fd,
        'repeat_checks':repeat_checks,'missing_cells':missing,'propagation_calls':response['propagation_calls'],
        'qualification':'NOT_EVALUATED','covariance_calibration':'NOT_EVALUATED','historical_conditions':'UNKNOWN'}
